import random
import math
import re
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy
from dataclasses import dataclass
from pathlib import Path

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer_MoE import visual_transformer
from .text_transformer import TextTransformer
from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict

from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer


@dataclass(frozen=True)
class ClipArchitecture:
    patch_size: int
    vision_width: int
    vision_layers: int
    vision_heads: int
    embed_dim: int
    text_width: int
    text_heads: int
    text_layers: int
    vocab_size: int
    source_context_length: int
    pretrained_grid_size: tuple


def _required_tensor(state_dict, *names):
    for name in names:
        value = state_dict.get(name)
        if torch.is_tensor(value):
            return value, name
    raise KeyError(
        "Checkpoint is missing required tensor; expected one of: "
        + ", ".join(names)
    )


def _continuous_layer_count(state_dict, prefix):
    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d+)\.attn\.in_proj_weight$"
    )
    indices = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := pattern.match(key)) is not None
    )
    if not indices:
        raise KeyError(f"Checkpoint has no attention blocks under {prefix}")
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Checkpoint block indices under {prefix} are not continuous: "
            f"{indices}"
        )
    return len(indices)


def infer_clip_architecture(state_dict):
    """Infer the CLIP ViT/text structure from checkpoint tensor shapes."""
    conv1, _ = _required_tensor(state_dict, 'visual.conv1.weight')
    if conv1.ndim != 4 or conv1.shape[1] != 3:
        raise ValueError(
            "Only CLIP ViT checkpoints with visual.conv1.weight [width,3,p,p] "
            "are supported"
        )
    patch_height, patch_width = conv1.shape[-2:]
    if patch_height != patch_width:
        raise ValueError("CLIP ViT patch embedding must use square patches")
    patch_size = int(patch_height)
    vision_width = int(conv1.shape[0])
    vision_layers = _continuous_layer_count(
        state_dict,
        'visual.transformer.resblocks.',
    )

    visual_projection, _ = _required_tensor(state_dict, 'visual.proj')
    if visual_projection.ndim != 2 or visual_projection.shape[0] != vision_width:
        raise ValueError(
            "visual.proj shape is inconsistent with visual.conv1 width"
        )
    embed_dim = int(visual_projection.shape[1])

    visual_position, _ = _required_tensor(
        state_dict,
        'visual.positional_embedding',
    )
    if visual_position.ndim != 2 or visual_position.shape[1] != vision_width:
        raise ValueError(
            "visual positional embedding width is inconsistent with visual.conv1"
        )
    position_grid = state_dict.get('visual.positional_embedding_grid_size')
    if position_grid is None:
        source_patch_count = int(visual_position.shape[0] - 1)
        source_grid_side = math.isqrt(source_patch_count)
        if source_grid_side * source_grid_side != source_patch_count:
            raise ValueError(
                "Cannot infer a square source grid from visual positional embedding"
            )
        pretrained_grid_size = (source_grid_side, source_grid_side)
    else:
        pretrained_grid_size = tuple(int(value) for value in position_grid.tolist())
        if len(pretrained_grid_size) != 2 or min(pretrained_grid_size) < 1:
            raise ValueError(
                "visual.positional_embedding_grid_size must contain two positive values"
            )
        active_position_count = (
            pretrained_grid_size[0] * pretrained_grid_size[1] + 1
        )
        if visual_position.shape[0] < active_position_count:
            raise ValueError(
                "visual positional embedding is shorter than its declared grid"
            )

    token_embedding, token_key = _required_tensor(
        state_dict,
        'token_embedding.weight',
        'encode_text.token_embedding.weight',
    )
    if token_embedding.ndim != 2:
        raise ValueError("token_embedding.weight must be two-dimensional")
    vocab_size, text_width = (int(value) for value in token_embedding.shape)
    text_prefix = (
        'encode_text.' if token_key.startswith('encode_text.') else ''
    )
    text_layers = _continuous_layer_count(
        state_dict,
        text_prefix + 'transformer.resblocks.',
    )
    text_projection, _ = _required_tensor(
        state_dict,
        text_prefix + 'text_projection',
    )
    if text_projection.shape != (text_width, embed_dim):
        raise ValueError(
            "text_projection shape is inconsistent with inferred text width "
            "and shared embed_dim"
        )
    text_position, _ = _required_tensor(
        state_dict,
        text_prefix + 'positional_embedding',
    )
    if text_position.ndim != 2 or text_position.shape[1] != text_width:
        raise ValueError(
            "text positional embedding width is inconsistent with token embedding"
        )
    source_context_length = int(text_position.shape[0])

    if vision_width % 64 or text_width % 64:
        raise ValueError(
            "OpenAI CLIP ViT attention widths must be divisible by 64"
        )
    vision_heads = vision_width // 64
    text_heads = text_width // 64

    for prefix, layers, width in (
        ('visual.transformer.resblocks.', vision_layers, vision_width),
        (text_prefix + 'transformer.resblocks.', text_layers, text_width),
    ):
        expected_shape = (3 * width, width)
        for block_index in range(layers):
            key = f'{prefix}{block_index}.attn.in_proj_weight'
            if tuple(state_dict[key].shape) != expected_shape:
                raise ValueError(
                    f"Checkpoint tensor {key} has shape "
                    f"{tuple(state_dict[key].shape)}, expected {expected_shape}"
                )

    return ClipArchitecture(
        patch_size=patch_size,
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_heads=vision_heads,
        embed_dim=embed_dim,
        text_width=text_width,
        text_heads=text_heads,
        text_layers=text_layers,
        vocab_size=vocab_size,
        source_context_length=source_context_length,
        pretrained_grid_size=pretrained_grid_size,
    )


def _checkpoint_model_state(config, checkpoint_path=None):
    if checkpoint_path is not None:
        resolved_path = Path(checkpoint_path).expanduser().resolve()
        payload = torch.load(
            resolved_path,
            map_location='cpu',
            weights_only=False,
        )
        source_description = str(resolved_path)
    elif config.model.ckpt_type == 'original_clip':
        resolved_path = Path(config.model.checkpoint).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"CLIP checkpoint does not exist: {resolved_path}"
            )
        with resolved_path.open('rb') as opened_file:
            scripted_model = torch.jit.load(opened_file, map_location='cpu')
        state_dict = scripted_model.state_dict()
        del scripted_model
        return state_dict, str(resolved_path)
    elif config.model.ckpt_type == 'saved':
        resolved_path = (
            Path(config.model.saved_path).expanduser().resolve()
            / 'checkpoint_best.pth'
        )
        payload = torch.load(
            resolved_path,
            map_location='cpu',
            weights_only=False,
        )
        source_description = str(resolved_path)
    else:
        raise KeyError(f"Unknown ckpt_type: {config.model.ckpt_type}")

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Checkpoint {source_description} must contain a state dict"
        )
    if isinstance(payload.get('model'), dict):
        state_dict = payload['model']
    elif isinstance(payload.get('state_dict'), dict):
        state_dict = payload['state_dict']
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state_dict = payload
    else:
        raise RuntimeError(
            f"Checkpoint {source_description} has no model/state_dict mapping"
        )
    if state_dict and all(key.startswith('module.') for key in state_dict):
        state_dict = OrderedDict(
            (key[len('module.'):], value) for key, value in state_dict.items()
        )
    return state_dict, source_description


def _mapped_expert_blocks(config, vision_layers):
    source_blocks = list(
        config.model.get('expert_blocks_source', config.expert_blocks)
    )
    if config.model.expert_blocks_auto and source_blocks and vision_layers != 12:
        # Scale the window's relative starting depth, then preserve its
        # configured offsets.  This maps [6, 7, 8] at 12 layers to the
        # requested contiguous [12, 13, 14] window at 24 layers.
        scale = vision_layers / 12.0
        mapped_start = int(round(source_blocks[0] * scale))
        mapped_blocks = [
            mapped_start + block_index - source_blocks[0]
            for block_index in source_blocks
        ]
    else:
        mapped_blocks = source_blocks
    if mapped_blocks != sorted(set(mapped_blocks)):
        raise ValueError(
            f"expert_blocks must be strictly increasing and unique: {mapped_blocks}"
        )
    if any(index < 0 or index >= vision_layers for index in mapped_blocks):
        raise ValueError(
            f"expert_blocks {mapped_blocks} are outside vision layer range "
            f"[0, {vision_layers - 1}]"
        )
    return source_blocks, mapped_blocks


def apply_clip_architecture(config, architecture):
    if config.model.backbone not in ('ViT-B/16', 'ViT-L/14'):
        raise ValueError(f"Unsupported CLIP backbone: {config.model.backbone}")
    expected_patch_size = int(config.model.backbone.rsplit('/', 1)[1])
    if architecture.patch_size != expected_patch_size:
        raise ValueError(
            f"Configured backbone {config.model.backbone} expects patch size "
            f"{expected_patch_size}, but checkpoint uses {architecture.patch_size}"
        )
    source_blocks, mapped_blocks = _mapped_expert_blocks(
        config,
        architecture.vision_layers,
    )
    config.model.expert_blocks_source = source_blocks
    config.expert_blocks = mapped_blocks
    for key in (
        'patch_size',
        'vision_width',
        'vision_layers',
        'vision_heads',
        'embed_dim',
        'text_width',
        'text_heads',
        'text_layers',
        'vocab_size',
        'source_context_length',
    ):
        config.model[key] = getattr(architecture, key)
    config.model.pretrained_grid_size = list(
        architecture.pretrained_grid_size
    )
    return architecture


def configure_backbone_from_checkpoint(config, checkpoint_path=None):
    state_dict, source_description = _checkpoint_model_state(
        config,
        checkpoint_path,
    )
    architecture = infer_clip_architecture(state_dict)
    apply_clip_architecture(config, architecture)
    config.model.architecture_source = source_description
    return architecture


def architecture_from_config(config):
    return ClipArchitecture(
        patch_size=config.model.patch_size,
        vision_width=config.model.vision_width,
        vision_layers=config.model.vision_layers,
        vision_heads=config.model.vision_heads,
        embed_dim=config.model.embed_dim,
        text_width=config.model.text_width,
        text_heads=config.model.text_heads,
        text_layers=config.model.text_layers,
        vocab_size=config.model.vocab_size,
        source_context_length=config.model.source_context_length,
        pretrained_grid_size=tuple(config.model.pretrained_grid_size),
    )


class CLIP(nn.Module):
    def __init__(self, config, image_encode, text_encode, num_classes=11003, eps=1e-2):
        super().__init__()
        self.visual = image_encode
        self.encode_text = text_encode
        self.embed_dim = config.model.embed_dim

        self.use_gather = config.model.use_gather
        self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        self.config = config
        self.eps = eps

        if config.experiment.view_prompt:
            if text_encode.context_length < 3:
                raise ValueError(
                    "view prompt requires text_length >= 3 for SOS and EOS"
                )
            prompt_width = text_encode.token_embedding.embedding_dim
            self.view_prompt_embed = nn.Parameter(torch.empty(2, prompt_width))
            nn.init.normal_(self.view_prompt_embed, std=0.02)

        if config.experiment.nrw_mode == 'view_norm':
            self.register_buffer('nrw_view_mu', torch.zeros(2))
            self.register_buffer('nrw_view_var', torch.ones(2))
        elif config.experiment.nrw_mode == 'global_norm':
            self.register_buffer('nrw_global_mu', torch.zeros(()))
            self.register_buffer('nrw_global_var', torch.ones(()))
        elif config.experiment.nrw_mode not in ('off', 'fixed'):
            raise ValueError(
                f"Unsupported experiment.nrw_mode: {config.experiment.nrw_mode}"
            )

        self.visual_emb_layer = VisualEmbeddingLayer(
            input_dim=self.embed_dim,
            ratio=0.3,
        )
        self.textual_emb_layer = TexualEmbeddingLayer(
            input_dim=self.embed_dim,
            ratio=0.3,
        )

        if config.experiment.id:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def forward(self, input, alpha, training=False, epoch=0):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']
        cam_id = input['cam_id']

        if self.config.experiment.view_prompt:
            text_tokens = tokenize(
                texts,
                context_length=self.config.experiment.text_length - 1,
            ).to(self.config.device)
        else:
            text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings, losses, feaAttn_i = self.encode_image(
            images, cam_id=cam_id, return_dense=True, training=training
        )
        if self.config.experiment.view_prompt:
            text_features, text_seq_embeddings, feaAttn_t = self.encode_text_with_view_prompt(
                text_tokens,
                cam_id,
                return_dense=True,
                training=training,
            )
            text_tokens_for_tse = self.view_prompt_token_ids(text_tokens)
        else:
            text_features, text_seq_embeddings, feaAttn_t = self.encode_text(
                text_tokens, return_dense=True, training=training
            )
            text_tokens_for_tse = text_tokens
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)
        image_features_norm_gathered = self.all_gather(image_features_norm)
        text_features_norm_gathered = self.all_gather(text_features_norm)
        view_ratio = self.config.experiment.view_ratio
        ortho_ratio = self.config.experiment.ortho_ratio
        if view_ratio > 0:
            view_loss = losses.get('view_loss', 0)
            ret['view_loss'] = view_loss if view_ratio == 1.0 else view_loss * view_ratio
        ret['view_acc'] = losses.get('view_acc', 0)
        if ortho_ratio > 0:
            ortho_loss = losses.get('ortho_loss', 0)
            ret['ortho_loss'] = ortho_loss if ortho_ratio == 1.0 else ortho_loss * ortho_ratio

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        idx = ids.view(-1, 1)
        gathered_ids = self.all_gather(ids)
        idx_all = gathered_ids.view(1, -1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        dhn_margin = self.config.experiment.dhn_margin
        dhn_hard_mask = None
        if dhn_margin > 0:
            domain_ids = input['domain_id'].to(self.config.device)
            gathered_domain_ids = self.all_gather(domain_ids)
            dhn_hard_mask = self.build_dhn_hard_mask(
                ids,
                gathered_ids,
                domain_ids,
                gathered_domain_ids,
            )
            ret['dhn_hard_ratio'] = (
                dhn_hard_mask.float().mean().detach()
            )

        image_features_s_norm = image_features_norm.detach()
        text_features_s_norm = text_features_norm.detach()
        image_features_s_norm_gathered = image_features_norm_gathered.detach()
        text_features_s_norm_gathered = text_features_norm_gathered.detach()

        nrw_mode = self.config.experiment.nrw_mode
        if nrw_mode == 'off':
            nitc_loss = self.calc_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                              text_features_s_norm, image_features_norm_gathered,
                                              text_features_norm_gathered, image_features_s_norm_gathered,
                                              text_features_s_norm_gathered, sim_targets, alpha, logit_scale,
                                              dhn_hard_mask, dhn_margin)
        else:
            nrw_weights, nrw_metrics = self.compute_nrw_weights(
                image_features_norm,
                text_features_norm,
                cam_id,
                epoch,
            )
            ret.update(nrw_metrics)
            nitc_loss = self.calc_contrastive_weighted(
                image_features_norm,
                text_features_norm,
                image_features_s_norm,
                text_features_s_norm,
                image_features_norm_gathered,
                text_features_norm_gathered,
                image_features_s_norm_gathered,
                text_features_s_norm_gathered,
                sim_targets,
                alpha,
                logit_scale,
                nrw_weights,
                dhn_hard_mask,
                dhn_margin,
            )

        i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
        t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens_for_tse, text_seq_embeddings), dim=-1)
        i_tse_f_norm_gathered = self.all_gather(i_tse_f)
        t_tse_f_norm_gathered = self.all_gather(t_tse_f)
        i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
        t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
        if nrw_mode == 'off':
            nitc_tse_loss = self.calc_contrastive(i_tse_f, t_tse_f, i_tse_f.detach(), t_tse_f.detach(),
                                                  i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                                  i_tse_f_s_gathered, t_tse_f_s_gathered, sim_targets, alpha, logit_scale,
                                                  dhn_hard_mask, dhn_margin)
        else:
            nitc_tse_loss = self.calc_contrastive_weighted(
                i_tse_f,
                t_tse_f,
                i_tse_f.detach(),
                t_tse_f.detach(),
                i_tse_f_norm_gathered,
                t_tse_f_norm_gathered,
                i_tse_f_s_gathered,
                t_tse_f_s_gathered,
                sim_targets,
                alpha,
                logit_scale,
                nrw_weights,
                dhn_hard_mask,
                dhn_margin,
            )

        ret['itc_loss'] = (nitc_loss + nitc_tse_loss) * self.config.experiment.nitc_ratio

        sdm_ratio = self.config.experiment.sdm_ratio
        if sdm_ratio > 0:
            sdm_global_loss = self.compute_sdm(
                image_features_norm,
                text_features_norm,
                image_features_norm_gathered,
                text_features_norm_gathered,
                sim_targets,
                logit_scale,
            )
            sdm_tse_loss = self.compute_sdm(
                i_tse_f,
                t_tse_f,
                i_tse_f_norm_gathered,
                t_tse_f_norm_gathered,
                sim_targets,
                logit_scale,
            )
            ret['sdm_loss'] = (
                sdm_global_loss + sdm_tse_loss
            ) * sdm_ratio

        tal_ratio = self.config.experiment.tal_ratio
        if tal_ratio > 0:
            tal_global_loss = self.compute_tal(
                image_features_norm,
                text_features_norm,
                image_features_norm_gathered,
                text_features_norm_gathered,
                ids,
                gathered_ids,
                self.config.experiment.tal_margin,
                self.config.experiment.tal_tau,
            )
            tal_tse_loss = self.compute_tal(
                i_tse_f,
                t_tse_f,
                i_tse_f_norm_gathered,
                t_tse_f_norm_gathered,
                ids,
                gathered_ids,
                self.config.experiment.tal_margin,
                self.config.experiment.tal_tau,
            )
            ret['tal_loss'] = (
                tal_global_loss + tal_tse_loss
            ) * tal_ratio

        if self.config.experiment.ritc:
            ritc_loss = self.compute_ritc(image_features_norm, text_features_norm, image_features_norm_gathered,
                                          text_features_norm_gathered, logit_scale, sim_targets)
            ritc_tse_loss = self.compute_ritc(i_tse_f, t_tse_f, i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                              logit_scale, sim_targets)
            ret['ritc_loss'] = (ritc_loss + ritc_tse_loss) * self.config.experiment.ritc_ratio

        if self.config.experiment.id:
            image_logits = self.classifier(image_features)
            text_logits = self.classifier(text_features)
            id_loss = (F.cross_entropy(image_logits, ids) + F.cross_entropy(text_logits, ids)) / 2
            ret['id_loss'] = id_loss * self.config.experiment.id_ratio

        bridge_cv_ratio = self.config.experiment.bridge_cv_ratio
        bridge_tg_ratio = self.config.experiment.bridge_tg_ratio
        if self.config.experiment.use_bridge and (
            bridge_cv_ratio > 0 or bridge_tg_ratio > 0
        ):
            ret.update(
                self.compute_bridge_losses(
                    input,
                    image_features_norm,
                    text_features_norm,
                    ids,
                    logit_scale,
                    bridge_cv_ratio,
                    bridge_tg_ratio,
                    training,
                )
            )

        return ret

    @torch.no_grad()
    def compute_nrw_weights(
        self,
        image_features_norm,
        text_features_norm,
        cam_id,
        epoch,
    ):
        mode = self.config.experiment.nrw_mode
        cam_id = cam_id.to(image_features_norm.device, dtype=torch.long)
        if mode == 'fixed':
            weights = torch.ones(
                cam_id.shape,
                dtype=torch.float32,
                device=cam_id.device,
            )
            weights[cam_id == 0] = self.config.experiment.nrw_fixed_aerial
            if epoch < self.config.experiment.nrw_warmup_epochs:
                weights.fill_(1.0)
            weights = weights.detach()
            zero = weights.new_zeros(())
            return weights, {
                'nrw_w_mean_aerial': self._masked_mean(
                    weights,
                    cam_id == 0,
                ),
                'nrw_w_mean_ground': self._masked_mean(
                    weights,
                    cam_id == 1,
                ),
                'nrw_mu_aerial': zero,
                'nrw_mu_ground': zero.clone(),
            }

        similarities = (
            image_features_norm.detach().float()
            * text_features_norm.detach().float()
        ).sum(dim=1)
        ema = self.config.experiment.nrw_ema

        if mode == 'view_norm':
            counts = torch.stack([
                (cam_id == view_id).sum() for view_id in (0, 1)
            ]).float()
            sums = torch.stack([
                similarities[cam_id == view_id].sum()
                for view_id in (0, 1)
            ])
            squared_sums = torch.stack([
                similarities[cam_id == view_id].square().sum()
                for view_id in (0, 1)
            ])
            counts, batch_mu, batch_var = self._nrw_batch_moments(
                counts,
                sums,
                squared_sums,
            )
            present = counts > 0
            self.nrw_view_mu.copy_(torch.where(
                present,
                ema * self.nrw_view_mu + (1.0 - ema) * batch_mu,
                self.nrw_view_mu,
            ))
            self.nrw_view_var.copy_(torch.where(
                present,
                ema * self.nrw_view_var + (1.0 - ema) * batch_var,
                self.nrw_view_var,
            ))
            selected_mu = self.nrw_view_mu[cam_id]
            selected_sigma = self.nrw_view_var[cam_id].clamp_min(0).sqrt()
            confidence = (similarities - selected_mu) / (
                selected_sigma + 1e-6
            )
            weights = self._nrw_sigmoid_weights(confidence)
            mu_aerial = self.nrw_view_mu[0].detach().clone()
            mu_ground = self.nrw_view_mu[1].detach().clone()
        elif mode == 'global_norm':
            counts, batch_mu, batch_var = self._nrw_batch_moments(
                similarities.new_tensor([similarities.numel()]),
                similarities.sum().view(1),
                similarities.square().sum().view(1),
            )
            if counts[0] > 0:
                self.nrw_global_mu.mul_(ema).add_(
                    batch_mu[0],
                    alpha=1.0 - ema,
                )
                self.nrw_global_var.mul_(ema).add_(
                    batch_var[0],
                    alpha=1.0 - ema,
                )
            sigma = self.nrw_global_var.clamp_min(0).sqrt()
            confidence = (similarities - self.nrw_global_mu) / (sigma + 1e-6)
            weights = self._nrw_sigmoid_weights(confidence)
            mu_aerial = self.nrw_global_mu.detach().clone()
            mu_ground = self.nrw_global_mu.detach().clone()
        else:
            raise ValueError(f"Unsupported experiment.nrw_mode: {mode}")

        if epoch < self.config.experiment.nrw_warmup_epochs:
            weights = torch.ones_like(weights)
        weights = weights.detach()
        aerial_mask = cam_id == 0
        ground_mask = cam_id == 1
        metrics = {
            'nrw_w_mean_aerial': self._masked_mean(weights, aerial_mask),
            'nrw_w_mean_ground': self._masked_mean(weights, ground_mask),
            'nrw_mu_aerial': mu_aerial,
            'nrw_mu_ground': mu_ground,
        }
        return weights, metrics

    @staticmethod
    def _nrw_batch_moments(counts, sums, squared_sums):
        statistics = torch.cat((counts, sums, squared_sums))
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.all_reduce(statistics)
        group_count = counts.numel()
        counts = statistics[:group_count]
        sums = statistics[group_count:2 * group_count]
        squared_sums = statistics[2 * group_count:]
        safe_counts = counts.clamp_min(1.0)
        means = sums / safe_counts
        variances = (squared_sums / safe_counts - means.square()).clamp_min(0)
        return counts, means, variances

    def _nrw_sigmoid_weights(self, confidence):
        min_weight = self.config.experiment.nrw_min_weight
        alpha = self.config.experiment.nrw_alpha
        return min_weight + (1.0 - min_weight) * torch.sigmoid(
            alpha * confidence
        )

    @staticmethod
    def _masked_mean(values, mask):
        if torch.any(mask):
            return values[mask].mean().detach()
        return values.new_zeros(())

    def compute_bridge_losses(
        self,
        input,
        image_features_norm,
        text_features_norm,
        ids,
        logit_scale,
        bridge_cv_ratio,
        bridge_tg_ratio,
        training,
    ):
        ret = {}
        has_bridge = input['has_bridge'].bool()
        feature_mask = has_bridge.to(image_features_norm.device)
        zero_loss = image_features_norm.sum() * 0.0

        if not torch.any(has_bridge):
            if bridge_cv_ratio > 0:
                ret['bridge_cv_loss'] = zero_loss
            if bridge_tg_ratio > 0:
                ret['bridge_tg_loss'] = zero_loss
            return ret

        bridge_images = input['bridge_image'][has_bridge].to(self.config.device)
        bridge_cam_id = torch.ones(
            bridge_images.shape[0],
            dtype=torch.long,
            device=bridge_images.device,
        )
        bridge_features = self.encode_bridge_image(
            bridge_images,
            bridge_cam_id,
            training,
        )
        bridge_features_norm = F.normalize(bridge_features, dim=-1)

        aerial_features_sub = image_features_norm[feature_mask]
        text_features_sub = text_features_norm[feature_mask]
        ids_sub = ids[feature_mask]
        pos = (ids_sub.view(-1, 1) == ids_sub.view(1, -1)).float()
        targets = pos / pos.sum(1, keepdim=True)

        if bridge_cv_ratio > 0:
            bridge_cv_loss = self.bidirectional_soft_target_loss(
                aerial_features_sub,
                bridge_features_norm,
                targets,
                logit_scale,
            )
            ret['bridge_cv_loss'] = bridge_cv_loss * bridge_cv_ratio
        if bridge_tg_ratio > 0:
            bridge_tg_loss = self.bidirectional_soft_target_loss(
                text_features_sub,
                bridge_features_norm,
                targets,
                logit_scale,
            )
            ret['bridge_tg_loss'] = bridge_tg_loss * bridge_tg_ratio
        return ret

    def encode_bridge_image(self, image, cam_id, training):
        bridge_features, _ = self.visual(
            image.type(self.dtype),
            cam_id=cam_id,
            training=training,
            return_dense=False,
        )
        return bridge_features

    @staticmethod
    def bidirectional_soft_target_loss(
        features_a,
        features_b,
        targets,
        logit_scale,
    ):
        logits_a_to_b = logit_scale * features_a @ features_b.t()
        logits_b_to_a = logits_a_to_b.t()
        loss_a_to_b = -torch.sum(
            F.log_softmax(logits_a_to_b, dim=1) * targets,
            dim=1,
        ).mean()
        loss_b_to_a = -torch.sum(
            F.log_softmax(logits_b_to_a, dim=1) * targets.t(),
            dim=1,
        ).mean()
        return (loss_a_to_b + loss_b_to_a) / 2

    def compute_ritc(self, img_feats, txt_feats, img_feats_gathered, txt_feats_gathered, logit_scale, sim_targets):
        logits_per_img = logit_scale * img_feats @ txt_feats_gathered.t()
        logits_per_txt = logit_scale * txt_feats @ img_feats_gathered.t()
        img_log = F.log_softmax(logits_per_img, dim=1)
        txt_log = F.log_softmax(logits_per_txt, dim=1)
        target_log = (sim_targets + self.eps).log()
        kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
        kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
        ritc_loss = 0.5 * (kl_img + kl_txt)
        return ritc_loss

    def compute_sdm(self, image_features, text_features, image_features_gathered, text_features_gathered,
                    labels, logit_scale, epsilon=1e-8):
        """
        Similarity Distribution Matching
        """

        t2i_cosine_theta = text_features @ image_features_gathered.t()
        i2t_cosine_theta = image_features @ text_features_gathered.t()

        text_proj_image = logit_scale * t2i_cosine_theta
        image_proj_text = logit_scale * i2t_cosine_theta

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels + epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels + epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
        return loss

    def compute_tal(
        self,
        image_features,
        text_features,
        image_features_gathered,
        text_features_gathered,
        ids,
        gathered_ids,
        margin,
        tau,
    ):
        """RDE triplet alignment loss with ID-aware positive sets."""
        if tau <= 0:
            raise ValueError("experiment.tal_tau must be positive")

        positive_mask = ids.view(-1, 1).eq(gathered_ids.view(1, -1))
        # Keep the temperature-scaled upper bound out of mixed precision.
        with torch.autocast(device_type=image_features.device.type, enabled=False):
            image_to_text = (
                image_features.float() @ text_features_gathered.float().t()
            )
            text_to_image = (
                text_features.float() @ image_features_gathered.float().t()
            )
            image_to_text_loss = self._compute_tal_direction(
                image_to_text,
                positive_mask,
                margin,
                tau,
            )
            text_to_image_loss = self._compute_tal_direction(
                text_to_image,
                positive_mask,
                margin,
                tau,
            )
        return (image_to_text_loss + text_to_image_loss) / 2

    @staticmethod
    def _compute_tal_direction(similarities, positive_mask, margin, tau):
        similarities = similarities.float()
        if similarities.shape != positive_mask.shape:
            raise ValueError("TAL similarity and ID-mask shapes must match")
        if torch.any(~positive_mask.any(dim=1)):
            raise ValueError("every TAL anchor must have at least one positive")

        negative_mask = ~positive_mask
        valid_rows = negative_mask.any(dim=1)
        if not torch.any(valid_rows):
            return similarities.sum() * 0.0

        similarities = similarities[valid_rows]
        positive_mask = positive_mask[valid_rows]
        negative_mask = negative_mask[valid_rows]

        positive_logits = (similarities / tau).masked_fill(
            ~positive_mask,
            float('-inf'),
        )
        positive_weights = F.softmax(positive_logits, dim=1).detach()
        positive_similarity = (positive_weights * similarities).sum(dim=1)

        negative_logits = (similarities / tau).masked_fill(
            ~negative_mask,
            float('-inf'),
        )
        negative_upper_bound = tau * torch.logsumexp(
            negative_logits,
            dim=1,
        )
        return F.relu(
            margin - positive_similarity + negative_upper_bound
        ).mean()

    @staticmethod
    def build_dhn_hard_mask(
        local_ids,
        gathered_ids,
        local_domain_ids,
        gathered_domain_ids,
    ):
        same_domain = local_domain_ids.view(-1, 1).eq(
            gathered_domain_ids.view(1, -1)
        )
        same_id = local_ids.view(-1, 1).eq(gathered_ids.view(1, -1))
        return (same_domain & ~same_id).detach()

    @staticmethod
    def apply_dhn_margin(
        cosine_similarities,
        hard_mask,
        dhn_margin,
        logit_scale,
    ):
        hard_mask = hard_mask.detach()
        if cosine_similarities.shape != hard_mask.shape:
            raise ValueError(
                "DHN similarity and hard-negative mask shapes must match"
            )
        adjusted_cosine = cosine_similarities + (
            hard_mask.to(cosine_similarities.dtype) * dhn_margin
        )
        return logit_scale * adjusted_cosine

    # input features are normed
    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale,
                         dhn_hard_mask=None, dhn_margin=0.0):
        with torch.no_grad():
            sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
        if dhn_hard_mask is None:
            sim_i2t = logit_scale * image_features @ text_features_gathered.t()
            sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        else:
            sim_i2t = self.apply_dhn_margin(
                image_features @ text_features_gathered.t(),
                dhn_hard_mask,
                dhn_margin,
                logit_scale,
            )
            sim_t2i = self.apply_dhn_margin(
                text_features @ image_features_gathered.t(),
                dhn_hard_mask,
                dhn_margin,
                logit_scale,
            )
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

    def calc_contrastive_weighted(
        self,
        image_features,
        text_features,
        image_features_s,
        text_features_s,
        image_features_gathered,
        text_features_gathered,
        image_features_s_gathered,
        text_features_s_gathered,
        sim_targets,
        alpha,
        logit_scale,
        sample_weights,
        dhn_hard_mask=None,
        dhn_margin=0.0,
    ):
        with torch.no_grad():
            sim_i2t_s = (
                logit_scale
                * image_features_s
                @ text_features_s_gathered.t()
            )
            sim_t2i_s = (
                logit_scale
                * text_features_s
                @ image_features_s_gathered.t()
            )
            sim_i2t_targets = (
                alpha * F.softmax(sim_i2t_s, dim=1)
                + (1 - alpha) * sim_targets
            )
            sim_t2i_targets = (
                alpha * F.softmax(sim_t2i_s, dim=1)
                + (1 - alpha) * sim_targets
            )
        if dhn_hard_mask is None:
            sim_i2t = logit_scale * image_features @ text_features_gathered.t()
            sim_t2i = logit_scale * text_features @ image_features_gathered.t()
        else:
            sim_i2t = self.apply_dhn_margin(
                image_features @ text_features_gathered.t(),
                dhn_hard_mask,
                dhn_margin,
                logit_scale,
            )
            sim_t2i = self.apply_dhn_margin(
                text_features @ image_features_gathered.t(),
                dhn_hard_mask,
                dhn_margin,
                logit_scale,
            )
        loss_i2t_per_sample = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
            dim=1,
        )
        loss_t2i_per_sample = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,
            dim=1,
        )
        per_sample_loss = (loss_i2t_per_sample + loss_t2i_per_sample) / 2
        sample_weights = sample_weights.to(
            dtype=per_sample_loss.dtype,
        ).detach()
        return (
            sample_weights * per_sample_loss
        ).sum() / (sample_weights.sum() + 1e-6)

    def compute_simclr_loss(self, logits_a, logits_b, logits_a_gathered, logits_b_gathered, labels, temperature):
        sim_aa = logits_a @ logits_a_gathered.t() / temperature
        sim_ab = logits_a @ logits_b_gathered.t() / temperature
        sim_ba = logits_b @ logits_a_gathered.t() / temperature
        sim_bb = logits_b @ logits_b_gathered.t() / temperature
        masks = torch.where(F.one_hot(labels, logits_a_gathered.size(0)) == 0, 0, float('-inf'))
        sim_aa += masks
        sim_bb += masks
        sim_a = torch.cat([sim_ab, sim_aa], 1)
        sim_b = torch.cat([sim_ba, sim_bb], 1)
        loss_a = F.cross_entropy(sim_a, labels)
        loss_b = F.cross_entropy(sim_b, labels)
        return (loss_a + loss_b) * 0.5

    def _build_mlp(self, in_dim=512, mlp_dim=512, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim)
        )

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image, cam_id=None, training=False, return_dense=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), cam_id=cam_id, training=training, return_dense=return_dense)
            return output
        output = self.visual(image.type(self.dtype), training=training)
        return output

    def view_prompt_token_ids(self, text_tokens):
        prompted_tokens = torch.zeros(
            text_tokens.shape[0],
            text_tokens.shape[1] + 1,
            dtype=text_tokens.dtype,
            device=text_tokens.device,
        )
        prompted_tokens[:, 0] = text_tokens[:, 0]
        prompted_tokens[:, 1] = 1  # non-padding placeholder for the learned prompt
        prompted_tokens[:, 2:] = text_tokens[:, 1:]
        return prompted_tokens

    def encode_text_with_view_prompt(
        self,
        text_tokens,
        cam_id,
        return_dense=False,
        training=False,
    ):
        del training
        encoder = self.encode_text
        if text_tokens.shape[1] + 1 != encoder.context_length:
            raise ValueError(
                "view-prompt input must reserve one position after the SOS token"
            )
        cam_id = cam_id.to(text_tokens.device, dtype=torch.long)
        if torch.any((cam_id < 0) | (cam_id > 1)):
            raise ValueError("view prompt requires cam_id values in {0, 1}")

        token_embeddings = encoder.token_embedding(text_tokens).type(encoder.dtype)
        prompt_embeddings = self.view_prompt_embed[cam_id].to(encoder.dtype).unsqueeze(1)
        x = torch.cat(
            [token_embeddings[:, :1], prompt_embeddings, token_embeddings[:, 1:]],
            dim=1,
        )
        if encoder.positional_embedding_flag:
            x = x + encoder.positional_embedding.type(encoder.dtype)
        x = x.permute(1, 0, 2)
        x, fea_attn = encoder.transformer(x)
        x = x.permute(1, 0, 2)
        x = encoder.ln_final(x).type(encoder.dtype)
        x = x @ encoder.text_projection

        eos_positions = text_tokens.argmax(dim=-1) + 1
        global_features = x[torch.arange(x.shape[0], device=x.device), eos_positions]
        if return_dense:
            return global_features, x, fea_attn
        return global_features, fea_attn

    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def load_state_dict(self, state_dict, *args, **kwargs):
        position_key = 'visual.positional_embedding'
        grid_key = 'visual.positional_embedding_grid_size'
        target_grid = tuple(self.visual.position_grid_size)
        if position_key not in state_dict:
            if not self.config.experiment.disable_moe:
                return super().load_state_dict(state_dict, *args, **kwargs)
            source_grid_tensor = None
            source_grid = target_grid
            resize_position = False
            remove_grid_marker = False
            position_capacity_mismatch = False
        else:
            source_grid_tensor = state_dict.get(grid_key)
            if source_grid_tensor is None:
                source_patch_count = state_dict[position_key].shape[0] - 1
                source_grid_width = int(source_patch_count ** 0.5)
                source_grid = (source_grid_width, source_grid_width)
                if source_grid_width * source_grid_width != source_patch_count:
                    raise ValueError(
                        "Cannot infer a square source grid from visual positional embedding"
                    )
            else:
                source_grid = tuple(int(value) for value in source_grid_tensor.tolist())

            resize_position = source_grid != target_grid
            remove_grid_marker = (
                source_grid_tensor is not None
                and not self.visual.needs_positional_interpolation
            )
            position_capacity_mismatch = (
                state_dict[position_key].shape
                != self.visual.positional_embedding.shape
            )
        if (not resize_position and not remove_grid_marker
                and not position_capacity_mismatch
                and not self.config.experiment.disable_moe):
            return super().load_state_dict(state_dict, *args, **kwargs)

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata

        if resize_position:
            state_dict[position_key] = self.visual.interpolate_positional_embedding(
                state_dict[position_key],
                source_grid,
            )
        elif position_key in state_dict:
            self.visual.ensure_positional_embedding_capacity(
                state_dict[position_key].shape[0]
            )
        if self.visual.needs_positional_interpolation:
            state_dict[grid_key] = self.visual.positional_embedding_grid_size
        else:
            state_dict.pop(grid_key, None)

        if self.config.experiment.disable_moe:
            # The original-CLIP loader expands each configured MLP into identical
            # expert copies. Collapse expert 0 back to the standard ViT MLP used by
            # this ablation, and discard all MoE-only parameters.
            for block_index in self.config.expert_blocks:
                prefix = f'visual.transformer.resblocks.{block_index}.'
                expert_zero_prefix = prefix + 'experts.0.'
                moe_prefixes = (
                    prefix + 'experts.',
                    prefix + 'router.',
                    prefix + 'img_router.',
                    prefix + 'fea_router.',
                )
                for key in list(state_dict):
                    if key.startswith(expert_zero_prefix):
                        mlp_key = prefix + 'mlp.' + key[len(expert_zero_prefix):]
                        state_dict.setdefault(mlp_key, state_dict[key])
                    if key.startswith(moe_prefixes):
                        del state_dict[key]
        return super().load_state_dict(state_dict, *args, **kwargs)


def clip_vitb(config, num_classes=11003):
    if 'vision_width' not in config.model:
        configure_backbone_from_checkpoint(config)
    architecture = architecture_from_config(config)
    image_encode = visual_transformer(config)
    text_encode = TextTransformer(
        config,
        context_length=config.experiment.text_length,
        transformer_width=architecture.text_width,
        transformer_heads=architecture.text_heads,
        transformer_layers=architecture.text_layers,
        positional_embedding_flag=True,
        checkpoint=False,
        embed_dim=architecture.embed_dim,
    )
    if text_encode.token_embedding.num_embeddings != architecture.vocab_size:
        text_encode.token_embedding = nn.Embedding(
            architecture.vocab_size,
            architecture.text_width,
        )
        nn.init.normal_(text_encode.token_embedding.weight, std=0.02)
    text_encode.transformer.checkpoint = config.experiment.grad_checkpoint
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model
