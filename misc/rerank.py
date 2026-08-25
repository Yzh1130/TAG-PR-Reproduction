"""Memory-bounded k-reciprocal re-ranking implemented with PyTorch."""

import torch


RANK_BLOCK_SIZE = 1024
ENCODING_BLOCK_SIZE = 256
QUERY_EXPANSION_BLOCK_SIZE = 64
JACCARD_QUERY_BLOCK_SIZE = 8


@torch.no_grad()
def self_cosine_distance(features, block_size=RANK_BLOCK_SIZE):
    """Return a symmetric cosine distance matrix in [0, 1]."""
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional tensor")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    features = features.float()
    count = features.shape[0]
    distances = torch.empty(
        (count, count),
        dtype=torch.float32,
        device=features.device,
    )
    with torch.autocast(device_type=features.device.type, enabled=False):
        transposed = features.t()
        for start in range(0, count, block_size):
            end = min(start + block_size, count)
            similarities = features[start:end] @ transposed
            distances[start:end] = (1.0 - similarities).mul_(0.5).clamp_(0.0, 1.0)
        for start in range(0, count, block_size):
            end = min(start + block_size, count)
            symmetric = (
                distances[start:end] + distances[:, start:end].t()
            ).mul_(0.5)
            distances[start:end] = symmetric
            distances[:, start:end] = symmetric.t()
    distances.diagonal().zero_()
    return distances


def _validate_distances(query_gallery, query_query, gallery_gallery, k1, k2, lambda_value):
    if query_gallery.ndim != 2:
        raise ValueError("query-gallery distance must be two-dimensional")
    query_count, gallery_count = query_gallery.shape
    if query_query.shape != (query_count, query_count):
        raise ValueError("query-query distance shape does not match query count")
    if gallery_gallery.shape != (gallery_count, gallery_count):
        raise ValueError("gallery-gallery distance shape does not match gallery count")
    if not (
        query_gallery.device == query_query.device == gallery_gallery.device
    ):
        raise ValueError("all re-ranking distances must use the same device")

    all_count = query_count + gallery_count
    if not 1 <= k1 < all_count:
        raise ValueError(f"rerank_k1 must be in [1, {all_count - 1}]")
    if not 1 <= k2 <= k1 + 1:
        raise ValueError("rerank_k2 must be in [1, rerank_k1 + 1]")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("rerank_lambda must be in [0, 1]")
    return query_count, gallery_count, all_count


def _combined_rows(
    query_gallery,
    query_query,
    gallery_gallery,
    start,
    end,
):
    """Read rows from [QQ, QG; QG.T, GG] without materializing it."""
    query_count = query_gallery.shape[0]
    if end <= query_count:
        return torch.cat(
            [query_query[start:end], query_gallery[start:end]],
            dim=1,
        )
    if start >= query_count:
        gallery_start = start - query_count
        gallery_end = end - query_count
        return torch.cat(
            [
                query_gallery[:, gallery_start:gallery_end].t(),
                gallery_gallery[gallery_start:gallery_end],
            ],
            dim=1,
        )
    return torch.cat(
        [
            _combined_rows(
                query_gallery,
                query_query,
                gallery_gallery,
                start,
                query_count,
            ),
            _combined_rows(
                query_gallery,
                query_query,
                gallery_gallery,
                query_count,
                end,
            ),
        ],
        dim=0,
    )


def _initial_rank(
    query_gallery,
    query_query,
    gallery_gallery,
    k1,
    block_size,
):
    all_count = query_query.shape[0] + gallery_gallery.shape[0]
    rank = torch.empty(
        (all_count, k1 + 1),
        dtype=torch.long,
        device=query_gallery.device,
    )
    for start in range(0, all_count, block_size):
        end = min(start + block_size, all_count)
        distances = _combined_rows(
            query_gallery,
            query_query,
            gallery_gallery,
            start,
            end,
        )
        nearest = torch.topk(
            distances,
            k=k1 + 1,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        row_ids = torch.arange(start, end, device=query_gallery.device)[:, None]
        positions = torch.arange(k1 + 1, device=query_gallery.device).expand(
            end - start,
            -1,
        )
        non_self_order = positions.masked_fill(nearest.eq(row_ids), k1 + 1).argsort(
            dim=1
        )
        nearest_non_self = nearest.gather(1, non_self_order[:, :k1])
        rank[start:end] = torch.cat([row_ids, nearest_non_self], dim=1)
    return rank


def _reciprocal_masks(initial_rank, k1):
    all_count = initial_rank.shape[0]
    row_ids = torch.arange(all_count, device=initial_rank.device)[:, None, None]

    forward = initial_rank[:, :k1 + 1]
    backward = initial_rank[forward, :k1 + 1]
    reciprocal_mask = backward.eq(row_ids).any(dim=-1)
    del backward

    half_count = int(round(k1 / 2.0)) + 1
    half_forward = initial_rank[:, :half_count]
    half_backward = initial_rank[half_forward, :half_count]
    half_reciprocal_mask = half_backward.eq(row_ids).any(dim=-1)
    del half_backward
    return forward, reciprocal_mask, half_forward, half_reciprocal_mask


def _build_encoding(
    query_gallery,
    query_query,
    gallery_gallery,
    initial_rank,
    k1,
    block_size,
):
    all_count = initial_rank.shape[0]
    forward, reciprocal_mask, half_forward, half_reciprocal_mask = (
        _reciprocal_masks(initial_rank, k1)
    )
    encoding = torch.zeros(
        (all_count, all_count),
        dtype=torch.float32,
        device=query_gallery.device,
    )

    for start in range(0, all_count, block_size):
        end = min(start + block_size, all_count)
        anchor_reciprocal = forward[start:end]
        anchor_mask = reciprocal_mask[start:end]
        candidate_half = half_forward[anchor_reciprocal]
        candidate_half_mask = half_reciprocal_mask[anchor_reciprocal]

        overlap = (
            candidate_half.unsqueeze(-1).eq(
                anchor_reciprocal[:, None, None, :]
            )
            & anchor_mask[:, None, None, :]
        ).any(dim=-1)
        overlap_count = (overlap & candidate_half_mask).sum(dim=-1)
        candidate_count = candidate_half_mask.sum(dim=-1)
        include_candidate = (
            overlap_count * 3 > candidate_count * 2
        ) & anchor_mask

        expanded_indices = torch.cat(
            [
                anchor_reciprocal,
                candidate_half.flatten(start_dim=1),
            ],
            dim=1,
        )
        expanded_mask = torch.cat(
            [
                anchor_mask,
                (include_candidate.unsqueeze(-1) & candidate_half_mask).flatten(
                    start_dim=1
                ),
            ],
            dim=1,
        )

        membership_count = torch.zeros(
            (end - start, all_count),
            dtype=torch.int32,
            device=query_gallery.device,
        )
        membership_count.scatter_add_(
            1,
            expanded_indices,
            expanded_mask.to(torch.int32),
        )
        membership = membership_count.ne(0)
        del membership_count

        distances = _combined_rows(
            query_gallery,
            query_query,
            gallery_gallery,
            start,
            end,
        )
        weights = torch.exp(-distances).mul_(membership)
        encoding[start:end] = weights / weights.sum(dim=1, keepdim=True).clamp_min_(
            1.0e-12
        )

    return encoding


def _local_query_expansion(encoding, initial_rank, k2, block_size):
    if k2 == 1:
        return encoding
    expanded = torch.empty_like(encoding)
    for start in range(0, encoding.shape[0], block_size):
        end = min(start + block_size, encoding.shape[0])
        neighbor_rows = initial_rank[start:end, :k2].reshape(-1)
        expanded[start:end] = encoding.index_select(0, neighbor_rows).view(
            end - start,
            k2,
            encoding.shape[1],
        ).mean(dim=1)
    return expanded


def _query_gallery_jaccard(
    encoding,
    query_count,
    gallery_count,
    block_size,
):
    gallery_encoding = encoding[query_count:]
    distances = torch.empty(
        (query_count, gallery_count),
        dtype=torch.float32,
        device=encoding.device,
    )
    for start in range(0, query_count, block_size):
        end = min(start + block_size, query_count)
        query_encoding = encoding[start:end]
        nonzero_count = query_encoding.gt(0).sum(dim=1)
        max_nonzero = int(nonzero_count.max().item())
        indices = torch.topk(
            query_encoding,
            k=max_nonzero,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        query_values = query_encoding.gather(1, indices)
        gallery_values = gallery_encoding[:, indices]
        intersection = torch.minimum(
            gallery_values,
            query_values.unsqueeze(0),
        ).sum(dim=-1).t()
        distances[start:end] = 1.0 - intersection / (
            2.0 - intersection
        ).clamp_min_(1.0e-12)
    return distances.clamp_(0.0, 1.0)


@torch.no_grad()
def k_reciprocal_rerank(
    query_gallery,
    query_query,
    gallery_gallery,
    k1=20,
    k2=6,
    lambda_value=0.3,
    rank_block_size=RANK_BLOCK_SIZE,
    encoding_block_size=ENCODING_BLOCK_SIZE,
    query_expansion_block_size=QUERY_EXPANSION_BLOCK_SIZE,
    jaccard_query_block_size=JACCARD_QUERY_BLOCK_SIZE,
):
    """Apply Zhong et al. k-reciprocal encoding to three distance blocks."""
    query_count, gallery_count, _ = _validate_distances(
        query_gallery,
        query_query,
        gallery_gallery,
        k1,
        k2,
        lambda_value,
    )
    if lambda_value == 1.0:
        return query_gallery.clone()

    query_gallery = query_gallery.float()
    query_query = query_query.float()
    gallery_gallery = gallery_gallery.float()
    try:
        initial_rank = _initial_rank(
            query_gallery,
            query_query,
            gallery_gallery,
            k1,
            rank_block_size,
        )
        encoding = _build_encoding(
            query_gallery,
            query_query,
            gallery_gallery,
            initial_rank,
            k1,
            encoding_block_size,
        )
        expanded_encoding = _local_query_expansion(
            encoding,
            initial_rank,
            k2,
            query_expansion_block_size,
        )
        if expanded_encoding is not encoding:
            del encoding
        jaccard = _query_gallery_jaccard(
            expanded_encoding,
            query_count,
            gallery_count,
            jaccard_query_block_size,
        )
        return (
            query_gallery * lambda_value
            + jaccard * (1.0 - lambda_value)
        ).clamp_(0.0)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during k-reciprocal re-ranking; lower "
            "--test_batch_size for feature extraction, free model/EMA memory, "
            "or reduce the internal re-ranking block sizes"
        ) from exc
