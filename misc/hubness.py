"""Label-free hubness diagnostics and CSLS score correction."""

import time

import torch
import torch.nn.functional as F


CSLS_QUERY_BLOCK_SIZE = 1024


def _validate_similarities(similarities, k):
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [num_query, num_gallery]")
    num_query, num_gallery = similarities.shape
    if num_query == 0 or num_gallery == 0:
        raise ValueError("similarities must contain queries and gallery samples")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("csls_k must be an integer")
    if not 1 <= k <= num_query:
        raise ValueError(f"csls_k must be in [1, {num_query}]")
    if not similarities.is_floating_point():
        raise TypeError("similarities must use a floating-point dtype")


@torch.no_grad()
def column_topk_mean(
    similarities,
    k,
    query_block_size=CSLS_QUERY_BLOCK_SIZE,
):
    """Return each gallery column's mean over its top-k query scores."""
    _validate_similarities(similarities, k)
    if query_block_size <= 0:
        raise ValueError("query_block_size must be positive")

    num_query = similarities.shape[0]
    running_topk = None
    for start in range(0, num_query, query_block_size):
        end = min(start + query_block_size, num_query)
        # A value discarded from a block's top-k cannot enter the global
        # top-k, so merging only block winners is exact and bounds workspace.
        block_topk = torch.topk(
            similarities[start:end],
            min(k, end - start),
            dim=0,
            largest=True,
            sorted=True,
        ).values
        if running_topk is None:
            running_topk = block_topk
            continue
        candidates = torch.cat((running_topk, block_topk), dim=0)
        running_topk = torch.topk(
            candidates,
            min(k, candidates.shape[0]),
            dim=0,
            largest=True,
            sorted=True,
        ).values
    return running_topk.mean(dim=0)


@torch.no_grad()
def csls_correct(
    similarities,
    k,
    query_block_size=CSLS_QUERY_BLOCK_SIZE,
    inplace=False,
):
    """Apply S'[query, gallery] = 2*S - r(g), optionally in place."""
    penalties = column_topk_mean(similarities, k, query_block_size)
    corrected = similarities if inplace else similarities.clone()
    corrected.mul_(2.0).sub_(penalties.unsqueeze(0))
    return corrected, penalties


@torch.no_grad()
def top1_hubness(similarities, top_n=10):
    """Return the largest top-1 selection counts and number of used galleries."""
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [num_query, num_gallery]")
    num_query, num_gallery = similarities.shape
    if num_query == 0 or num_gallery == 0:
        raise ValueError("similarities must contain queries and gallery samples")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    top1_indices = similarities.argmax(dim=1)
    counts = torch.bincount(top1_indices, minlength=num_gallery)
    largest = torch.topk(
        counts,
        min(top_n, num_gallery),
        largest=True,
        sorted=True,
    ).values
    return largest.cpu().tolist(), int((counts > 0).sum().item())


def _log_hubness_summary(logger, before, after):
    before_counts, before_unique = before
    after_counts, after_unique = after
    logger.info("HUBNESS SUMMARY")
    logger.info(
        "  before top-1 counts: %s",
        "/".join(str(value) for value in before_counts),
    )
    logger.info(
        "  after top-1 counts: %s",
        "/".join(str(value) for value in after_counts),
    )
    logger.info("  before unique top-1 gallery images: %s", before_unique)
    logger.info("  after unique top-1 gallery images: %s", after_unique)
    logger.info("END HUBNESS SUMMARY")


@torch.no_grad()
def evaluate_csls_scores(similarities, dataset, k, logger):
    """Evaluate an already fused query-gallery matrix after CSLS."""
    from misc.eval import metric_eval

    before_hubness = top1_hubness(similarities)
    if similarities.is_cuda:
        torch.cuda.synchronize(similarities.device)
    started = time.time()
    logger.info(
        "CSLS BLOCKS query=%s",
        CSLS_QUERY_BLOCK_SIZE,
    )
    try:
        corrected, _ = csls_correct(
            similarities,
            k,
            query_block_size=CSLS_QUERY_BLOCK_SIZE,
            inplace=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during CSLS correction; lower "
            "--test_batch_size, free GPU memory, or reduce "
            "CSLS_QUERY_BLOCK_SIZE in misc/hubness.py"
        ) from exc
    if corrected.is_cuda:
        torch.cuda.synchronize(corrected.device)
    logger.info("CSLS correction completed in %.3f seconds", time.time() - started)

    after_hubness = top1_hubness(corrected)
    _log_hubness_summary(logger, before_hubness, after_hubness)
    return metric_eval(
        corrected,
        dataset.img2person,
        dataset.txt2person,
    )


@torch.no_grad()
def test_tse_with_csls(model, data_loader, config, logger):
    """Evaluate raw fused global/TSE scores and their CSLS correction."""
    from misc.eval import metric_eval
    from text_utils.tokenizer import tokenize

    if not torch.cuda.is_available() or not str(config.device).startswith("cuda"):
        raise RuntimeError("CSLS evaluation requires CUDA; CPU fallback is disabled")

    model.eval()
    dataset = data_loader.dataset
    texts = dataset.text
    text_batch_size = 256
    max_length = config.experiment.text_length
    device = config.device

    if config.experiment.view_prompt:
        text_features_by_view = {}
        text_tse_features_by_view = {}
        for view in (0, 1):
            text_features = []
            text_tse_features = []
            for start in range(0, len(texts), text_batch_size):
                batch_texts = texts[start:min(len(texts), start + text_batch_size)]
                tokens = tokenize(
                    batch_texts,
                    context_length=max_length - 1,
                ).to(device)
                cam_ids = torch.full(
                    (tokens.shape[0],),
                    view,
                    dtype=torch.long,
                    device=device,
                )
                text_feature, text_dense, text_attention = (
                    model.encode_text_with_view_prompt(
                        tokens,
                        cam_ids,
                        return_dense=True,
                    )
                )
                prompted_tokens = model.view_prompt_token_ids(tokens)
                text_features.append(F.normalize(text_feature, dim=-1))
                text_tse_features.append(
                    F.normalize(
                        model.textual_emb_layer(
                            text_attention,
                            prompted_tokens,
                            text_dense,
                            training=False,
                        ),
                        dim=-1,
                    )
                )
            text_features_by_view[view] = torch.cat(text_features, dim=0)
            text_tse_features_by_view[view] = torch.cat(
                text_tse_features,
                dim=0,
            )
    else:
        text_features = []
        text_tse_features = []
        for start in range(0, len(texts), text_batch_size):
            batch_texts = texts[start:min(len(texts), start + text_batch_size)]
            tokens = tokenize(batch_texts, context_length=max_length).to(device)
            text_feature, text_dense, text_attention = model.encode_text(
                tokens,
                return_dense=True,
            )
            text_features.append(F.normalize(text_feature, dim=-1))
            text_tse_features.append(
                F.normalize(
                    model.textual_emb_layer(
                        text_attention,
                        tokens,
                        text_dense,
                        training=False,
                    ),
                    dim=-1,
                )
            )
        text_features = torch.cat(text_features, dim=0)
        text_tse_features = torch.cat(text_tse_features, dim=0)

    image_features = []
    image_tse_features = []
    for images in data_loader:
        images = images.to(device)
        image_feature, image_dense, image_attention = model.encode_image(
            images,
            return_dense=True,
        )
        image_features.append(F.normalize(image_feature, dim=-1))
        image_tse_features.append(
            F.normalize(
                model.visual_emb_layer(
                    image_attention,
                    image_dense,
                    training=False,
                ),
                dim=-1,
            )
        )
    image_features = torch.cat(image_features, dim=0)
    image_tse_features = torch.cat(image_tse_features, dim=0)

    if config.experiment.view_prompt:
        if not hasattr(dataset, "img_cam_ids"):
            raise RuntimeError("view-prompt CSLS evaluation requires img_cam_ids")
        gallery_cam_ids = dataset.img_cam_ids.to(device)
        if gallery_cam_ids.shape[0] != image_features.shape[0]:
            raise RuntimeError("gallery cam_id count does not match gallery images")
        if torch.any((gallery_cam_ids < 0) | (gallery_cam_ids > 1)):
            raise ValueError("view prompt requires cam_id values in {0, 1}")
        similarities = image_features.new_empty(
            len(texts),
            image_features.shape[0],
        )
        for view in (0, 1):
            image_mask = gallery_cam_ids == view
            if not image_mask.any():
                continue
            similarities[:, image_mask] = (
                text_features_by_view[view] @ image_features[image_mask].t()
                + text_tse_features_by_view[view]
                @ image_tse_features[image_mask].t()
            ) / 2
    else:
        similarities = (
            text_features @ image_features.t()
            + text_tse_features @ image_tse_features.t()
        ) / 2

    raw_result = metric_eval(
        similarities,
        dataset.img2person,
        dataset.txt2person,
    )
    csls_result = evaluate_csls_scores(
        similarities,
        dataset,
        config.experiment.csls_k,
        logger,
    )
    return raw_result, csls_result
