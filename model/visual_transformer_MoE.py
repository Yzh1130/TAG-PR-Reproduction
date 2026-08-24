import torch
import torch.nn.functional as F
from torch import nn
from .base_transformer import Transformer, LayerNorm, Transformer_MoE
from typing import Tuple, Union


class VisualTransformer(nn.Module):
    def __init__(self, config, input_resolution: Union[int, Tuple[int, int]], patch_size: int, width: int,
                 layers: int, heads: int, embed_dim: int,
                 checkpoint: bool, dropout: float = 0, emb_dropout: float = 0):
        super().__init__()
        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // patch_size + 1
        self.num_y = (input_resolution[0] - patch_size) // patch_size + 1
        if input_resolution[0] % patch_size or input_resolution[1] % patch_size:
            raise ValueError(
                "input_resolution height and width must be divisible by patch_size"
            )
        self.position_grid_size = (self.num_y, self.num_x)
        self.needs_positional_interpolation = self.position_grid_size != (14, 14)

        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.view_token = nn.Parameter(scale * torch.randn(width))
        position_capacity = 14 * 14
        self.positional_embedding = nn.Parameter(scale * torch.randn(position_capacity + 1, width))
        if self.needs_positional_interpolation:
            self.register_buffer(
                'positional_embedding_grid_size',
                torch.tensor(self.position_grid_size, dtype=torch.long),
            )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_MoE(config, width, layers, heads, checkpoint=checkpoint, dropout=dropout,
                                           emb_dropout=emb_dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.initialize_parameters()

        self.expert_blocks = config.expert_blocks

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if hasattr(block, 'experts') and len(block.experts) > 0:
                for expert in block.experts:
                    nn.init.normal_(expert.c_fc.weight, std=fc_std)
                    nn.init.normal_(expert.c_proj.weight, std=fc_std)
            else:
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:
            for layer in [self.conv1]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self

    @property
    def active_positional_embedding(self):
        active_length = self.num_x * self.num_y + 1
        if self.positional_embedding.shape[0] < active_length:
            raise RuntimeError(
                "positional_embedding has not been prepared for input_resolution"
            )
        return self.positional_embedding[:active_length]

    def ensure_positional_embedding_capacity(self, position_count):
        if self.positional_embedding.shape[0] == position_count:
            return
        self.positional_embedding = nn.Parameter(
            torch.empty(
                position_count,
                self.positional_embedding.shape[1],
                device=self.positional_embedding.device,
                dtype=self.positional_embedding.dtype,
            )
        )

    def interpolate_positional_embedding(self, positional_embedding, source_grid):
        source_height, source_width = source_grid
        source_patch_count = source_height * source_width
        target_patch_count = self.num_y * self.num_x
        if positional_embedding.shape[0] < source_patch_count + 1:
            raise ValueError(
                "positional_embedding is shorter than its declared source grid"
            )
        cls_position = positional_embedding[:1]
        patch_positions = positional_embedding[1:source_patch_count + 1]
        patch_positions = patch_positions.reshape(
            1,
            source_height,
            source_width,
            positional_embedding.shape[-1],
        ).permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions.float(),
            size=self.position_grid_size,
            mode='bicubic',
            align_corners=False,
        ).to(positional_embedding.dtype)
        patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
            target_patch_count,
            positional_embedding.shape[-1],
        )
        active_positions = torch.cat([cls_position, patch_positions], dim=0)

        position_capacity = max(14 * 14, target_patch_count) + 1
        packed_positions = positional_embedding.new_zeros(
            position_capacity,
            positional_embedding.shape[-1],
        )
        copy_count = min(position_capacity, positional_embedding.shape[0])
        packed_positions[:copy_count] = positional_embedding[:copy_count]
        packed_positions[:target_patch_count + 1] = active_positions
        self.ensure_positional_embedding_capacity(position_capacity)
        print(
            "Resized visual positional embedding with bicubic interpolation: "
            f"source_grid={source_height}x{source_width}, "
            f"target_grid={self.num_y}x{self.num_x}, "
            f"active_shape={tuple(active_positions.shape)}"
        )
        return packed_positions

    def forward(self, x: torch.Tensor, cam_id=None, return_dense=False, return_feature=False, training=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if self.needs_positional_interpolation:
            x = x + self.active_positional_embedding.to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)
        # The learned view token is appended after positional encoding and has
        # no positional-embedding entry of its own.
        x = torch.cat([x, self.view_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)], dim=1)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, losses, feaAttn = self.transformer(x, cam_id=cam_id, training=training)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        dense_feat = x

        if self.proj is not None:
            dense_feat = x @ self.proj
            x = dense_feat[:, 0, :]

        if training:
            if return_dense:
                return x, dense_feat, losses, feaAttn
            if return_feature:
                return dense_feat, losses, feaAttn
        else:
            if return_dense:
                return x, dense_feat, feaAttn
            if return_feature:
                return dense_feat, feaAttn
        return x, feaAttn


def visual_transformer(config):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    kwargs = {
        'config': config,
        'layers': vision_layers,
        'heads': vision_heads,
        'input_resolution': config.experiment.input_resolution,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': config.experiment.grad_checkpoint,
        'embed_dim': config.model.embed_dim,
    }

    model = VisualTransformer(**kwargs)
    return model
