"""Unmodified baseline candidate loss functions.

Source commit: 8bf2616403145f5a0e64747ddf9914e0a9e0fcdf
Source file: model/tbps_model_MoE.py (CLIP methods)
Functions below are copied verbatim, including original reductions and masks.
"""

import torch
import torch.nn.functional as F


class ReferenceLosses:
    def __init__(self, eps=1e-2):
        self.eps = eps

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
