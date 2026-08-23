import random
import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import copy

from misc import utils
from misc.utils import is_using_distributed
from text_utils.tokenizer import tokenize
from .visual_transformer_MoE import visual_transformer
from .text_transformer import text_transformers
from .base_transformer import Transformer, LayerNorm, QuickGELU

from .shared_modules import AllGather
from collections import OrderedDict

from .CrossEmbeddingLayer_tse import VisualEmbeddingLayer, TexualEmbeddingLayer


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

        self.visual_emb_layer = VisualEmbeddingLayer(ratio=0.3)
        self.textual_emb_layer = TexualEmbeddingLayer(ratio=0.3)

        if config.experiment.id:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def forward(self, input, alpha, training=False, epoch=0):
        ret = dict()

        images = input['image'].to(self.config.device)
        texts = input['caption']
        cam_id = input['cam_id']

        text_tokens = tokenize(texts, context_length=self.config.experiment.text_length).to(self.config.device)
        ids = input['id'].to(self.config.device)

        image_features, image_seq_embeddings, losses, feaAttn_i = self.encode_image(
            images, cam_id=cam_id, return_dense=True, training=training
        )
        text_features, text_seq_embeddings, feaAttn_t = self.encode_text(
            text_tokens, return_dense=True, training=training
        )
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

        image_features_s_norm = image_features_norm.detach()
        text_features_s_norm = text_features_norm.detach()
        image_features_s_norm_gathered = image_features_norm_gathered.detach()
        text_features_s_norm_gathered = text_features_norm_gathered.detach()

        nrw_mode = self.config.experiment.nrw_mode
        if nrw_mode == 'off':
            nitc_loss = self.calc_contrastive(image_features_norm, text_features_norm, image_features_s_norm,
                                              text_features_s_norm, image_features_norm_gathered,
                                              text_features_norm_gathered, image_features_s_norm_gathered,
                                              text_features_s_norm_gathered, sim_targets, alpha, logit_scale)
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
            )

        i_tse_f = F.normalize(self.visual_emb_layer(feaAttn_i, image_seq_embeddings), dim=-1)
        t_tse_f = F.normalize(self.textual_emb_layer(feaAttn_t, text_tokens, text_seq_embeddings), dim=-1)
        i_tse_f_norm_gathered = self.all_gather(i_tse_f)
        t_tse_f_norm_gathered = self.all_gather(t_tse_f)
        i_tse_f_s_gathered = i_tse_f_norm_gathered.detach()
        t_tse_f_s_gathered = t_tse_f_norm_gathered.detach()
        if nrw_mode == 'off':
            nitc_tse_loss = self.calc_contrastive(i_tse_f, t_tse_f, i_tse_f.detach(), t_tse_f.detach(),
                                                  i_tse_f_norm_gathered, t_tse_f_norm_gathered,
                                                  i_tse_f_s_gathered, t_tse_f_s_gathered, sim_targets, alpha, logit_scale)
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
            )

        ret['itc_loss'] = (nitc_loss + nitc_tse_loss) * self.config.experiment.nitc_ratio


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

    # input features are normed
    def calc_contrastive(self, image_features, text_features, image_features_s, text_features_s,
                         image_features_gathered, text_features_gathered, image_features_s_gathered,
                         text_features_s_gathered,
                         sim_targets, alpha, logit_scale):
        with torch.no_grad():
            sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets  # soft + hard
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
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
        sim_i2t = logit_scale * image_features @ text_features_gathered.t()
        sim_t2i = logit_scale * text_features @ image_features_gathered.t()
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

    def all_gather(self, input):
        if not self.use_gather or not is_using_distributed():
            return input
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def load_state_dict(self, state_dict, *args, **kwargs):
        if not self.config.experiment.disable_moe:
            return super().load_state_dict(state_dict, *args, **kwargs)

        # The original-CLIP loader expands each configured MLP into identical
        # expert copies. Collapse expert 0 back to the standard ViT MLP used by
        # this ablation, and discard all MoE-only parameters.
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            state_dict._metadata = metadata
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
    image_encode = visual_transformer(config)
    text_encode = text_transformers(config)
    text_encode.transformer.checkpoint = config.experiment.grad_checkpoint
    model = CLIP(config, image_encode, text_encode, num_classes, config.experiment.ritc_eps)
    return model
