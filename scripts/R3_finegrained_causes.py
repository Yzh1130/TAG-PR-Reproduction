"""Read-only associations of source-image size, textual attributes and retrieval errors.

Image sizes are original PIL dimensions, not resized model inputs. No output
is a causal estimate. Metrics reuse the unchanged S11 NumPy evaluator.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

try:
    from ._analysis_audit import sha256_16
    from ._offline_common import compact_metrics, evaluate, load_annotations, load_similarity
except ImportError:
    from _analysis_audit import sha256_16
    from _offline_common import compact_metrics, evaluate, load_annotations, load_similarity


ATTRIBUTE_VOCABULARY = {
    "colors": ["black", "white", "red", "green", "blue", "yellow", "brown", "grey", "gray",
               "orange", "pink", "purple", "beige", "khaki", "navy", "maroon", "silver", "gold"],
    "clothing": ["shirt", "shirts", "tshirt", "tshirts", "jacket", "jackets", "coat", "coats",
                 "pants", "trousers", "shorts", "jeans", "skirt", "dress", "sweater", "hoodie",
                 "suit", "uniform", "vest", "blouse", "shoes", "sneakers", "boots", "sandals",
                 "hat", "cap", "backpack", "bag", "scarf", "sleeve", "sleeves", "striped",
                 "stripes", "plaid", "checkered", "floral", "long", "short"],
}


def attribute_tokens(text):
    words = set(re.findall(r"[a-z]+", text.lower().replace("t-shirt", "tshirt")))
    return words & set(ATTRIBUTE_VOCABULARY["colors"] + ATTRIBUTE_VOCABULARY["clothing"])


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def load_records(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        records = json.load(handle)
    # The shared loader checks IDs/views/domains, uniqueness and caption order.
    return records


def image_path(root, relative):
    root = Path(root).expanduser().resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"image is missing or outside --image-root: {path}")
    return path


def read_image_sizes(ann, root):
    sizes, manifest = [], []
    for index, relative in enumerate(ann.paths):
        path = image_path(root, relative)
        with Image.open(path) as image:
            width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"nonpositive image size: {path}")
        sizes.append((width, height))
        manifest.append({"gallery_index": index, "path": str(path), "bytes": path.stat().st_size,
                         "sha256_16": sha256_16(path), "width": width, "height": height})
    sizes = np.asarray(sizes, dtype=np.int64)
    return sizes, sizes[:, 0] * sizes[:, 1], manifest


def cluster_mean(values, ids, repeats, seed):
    """Query-weighted identity-cluster percentile CI; no IID-query substitution."""
    values, ids = np.asarray(values, dtype=np.float64), np.asarray(ids)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("cluster_mean needs aligned finite 1-D values and identities")
    if repeats < 1 or seed < 0:
        raise ValueError("bootstrap repeats must be positive and seed nonnegative")
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_queries": 0, "n_identities": 0,
                "seed": int(seed), "bootstrap": int(repeats)}
    _, inverse = np.unique(ids, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse)
    rng, estimates = np.random.default_rng(seed), np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 64):
        end = min(repeats, start + 64)
        draw = rng.integers(0, len(counts), (end - start, len(counts)))
        estimates[start:end] = sums[draw].sum(1) / counts[draw].sum(1)
    return {"mean": float(values.mean()), "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
            "n_queries": len(values), "n_identities": len(counts), "seed": int(seed),
            "bootstrap": int(repeats), "unit": "identity cluster, all selected member queries retained",
            "estimator": "query-weighted mean, not unweighted cluster means"}


def attribute_comparison(records, ann, top1, correct, seed):
    """Max over each candidate image's captions; random same-domain different-ID image."""
    texts = [caption for row in records for caption in row["captions"]]
    if len(texts) != len(ann.query_ids) or len(records) != len(ann.gallery_ids):
        raise ValueError("caption expansion does not match shared annotation order")
    query_sets = [attribute_tokens(text) for text in texts]
    gallery_sets = [[attribute_tokens(text) for text in row["captions"]] for row in records]
    errors = np.flatnonzero(~np.asarray(correct, dtype=bool))
    rng = np.random.default_rng(seed)
    negative = np.full(len(errors), -1, np.int64)
    wrong_overlap, random_overlap = np.empty(len(errors)), np.full(len(errors), np.nan)
    for i, query in enumerate(errors):
        wrong_overlap[i] = max(jaccard(query_sets[query], item) for item in gallery_sets[top1[query]])
        candidates = np.flatnonzero((ann.gallery_domains == ann.query_domains[query]) &
                                   (ann.gallery_ids != ann.query_ids[query]))
        if len(candidates):
            negative[i] = int(candidates[rng.integers(len(candidates))])
            random_overlap[i] = max(jaccard(query_sets[query], item) for item in gallery_sets[negative[i]])
    return {"texts": texts, "error_queries": errors, "error_jaccard": wrong_overlap,
            "random_jaccard": random_overlap, "random_gallery_indices": negative}


def rank_margin_data(sim, ann, raw):
    errors = np.flatnonzero(~raw["details"]["correct"])
    margins, truth_rank, truth_best = [], [], []
    for query in errors:
        row = np.asarray(sim[query], dtype=np.float32)
        positives = np.flatnonzero(ann.gallery_ids == ann.query_ids[query])
        best = int(positives[np.argmax(row[positives])])
        rank = 1 + int((row > row[best]).sum()) + int(((row == row[best]) &
                                                       (np.arange(len(row)) < best)).sum())
        predicted = int(raw["details"]["top1"][query])
        margins.append(float(row[predicted] - row[best]))
        truth_rank.append(rank)
        truth_best.append(best)
    return errors, np.asarray(margins), np.asarray(truth_rank, np.int64), np.asarray(truth_best, np.int64)


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "quantiles": {}}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)}}


def analyze(sim, ann, records, areas, sizes, seed=2026, bootstrap=10000):
    areas, sizes = np.asarray(areas), np.asarray(sizes)
    if areas.shape != (len(ann.gallery_ids),) or sizes.shape != (len(ann.gallery_ids), 2):
        raise ValueError("original image dimensions must align with every gallery image")
    raw = evaluate(sim, ann)
    queries = raw["details"]["query_indices"]
    if not np.array_equal(queries, np.arange(len(ann.query_ids))):
        raise AssertionError("full query order changed or no-positive queries were dropped")
    top1, correct = raw["details"]["top1"], raw["details"]["correct"]
    same_source_view_error = (~correct & (ann.query_domains == ann.gallery_domains[top1]) &
                              (ann.query_views == ann.gallery_views[top1]))
    strata = {}
    for domain in np.unique(ann.gallery_domains):
        gallery = np.flatnonzero(ann.gallery_domains == domain)
        boundaries = np.quantile(areas[gallery], [.25, .5, .75])
        gallery_bins = np.searchsorted(boundaries, areas[gallery], side="right")
        groups = []
        for quartile in range(4):
            images = gallery[gallery_bins == quartile]
            selected = np.flatnonzero(np.isin(ann.query_source_indices, images))
            metrics = evaluate(sim, ann, queries=selected)
            errors = selected[~correct[selected]]
            groups.append({"quartile": quartile + 1, "n_source_images": len(images),
                           "n_queries": len(selected), "metrics": compact_metrics(metrics),
                           "R1_identity_bootstrap_proportion": cluster_mean(correct[selected], ann.query_ids[selected], bootstrap, seed),
                           "same_source_same_view_error_share_among_errors": cluster_mean(
                               same_source_view_error[errors], ann.query_ids[errors], bootstrap, seed),
                           "same_source_same_view_error_share_among_all_queries": cluster_mean(
                               same_source_view_error[selected], ann.query_ids[selected], bootstrap, seed)})
        if sum(group["n_queries"] for group in groups) != int((ann.query_domains == domain).sum()):
            raise AssertionError("source quartile partition does not conserve queries")
        strata[str(domain)] = {"area_boundaries_pixels2": boundaries.tolist(), "groups": groups}
    attributes = attribute_comparison(records, ann, top1, correct, seed)
    errors, margin, rank, best = rank_margin_data(sim, ann, raw)
    valid = attributes["random_gallery_indices"] >= 0
    attr_ids = ann.query_ids[attributes["error_queries"]]
    attribute_stats = {
        "error_top1": cluster_mean(attributes["error_jaccard"], attr_ids, bootstrap, seed),
        "error_top1_paired_subset": cluster_mean(attributes["error_jaccard"][valid], attr_ids[valid], bootstrap, seed),
        "random_same_source_negative": cluster_mean(attributes["random_jaccard"][valid], attr_ids[valid], bootstrap, seed),
        "paired_error_minus_random": cluster_mean((attributes["error_jaccard"] - attributes["random_jaccard"])[valid],
                                                   attr_ids[valid], bootstrap, seed),
        "queries_with_no_same_source_negative": int((~valid).sum()),
        "per_error": [{"query_index": int(q), "error_gallery_index": int(top1[q]),
                       "error_jaccard": float(attributes["error_jaccard"][i]),
                       "random_gallery_index": int(attributes["random_gallery_indices"][i]),
                       "random_jaccard": float(attributes["random_jaccard"][i]) if valid[i] else None}
                      for i, q in enumerate(attributes["error_queries"])]}
    return {
        "tool": "R3_finegrained_causes", "status": "complete",
        "interpretation": "descriptive associations only; no causal claims",
        "raw": compact_metrics(raw), "annotation": ann.summary,
        "image_size": {"units": "original PIL width/height in pixels; area in pixels squared, not model input 224x224",
                       "quartile_basis": "gallery source-image areas within each source; equal-area ties assigned together using searchsorted(side=right); bins need not be equal-sized",
                       "source_quartiles": strata, "dimensions": sizes.tolist(), "areas": areas.tolist()},
        "attribute_vocabulary": ATTRIBUTE_VOCABULARY,
        "attribute_protocol": "case-insensitive regex English tokens; t-shirt becomes tshirt; exact vocabulary matching; empty-empty Jaccard=0; max over all image captions; uniform same-source different-identity image, one draw per error in query order",
        "attribute_overlap": attribute_stats,
        "similarity_margin": {"definition": "s(query,error top1) - max(s(query,correct-identity image)); raw full gallery, no source exclusion",
                              "distribution": distribution(margin),
                              "identity_bootstrap_mean": cluster_mean(margin, ann.query_ids[errors], bootstrap, seed)},
        "truth_rank": {"tie_rule": "descending score, ascending original gallery index (shared evaluator)",
                       "distribution": distribution(rank),
                       "histogram": {str(int(r)): int(n) for r, n in zip(*np.unique(rank, return_counts=True))}},
        "per_error_margin_rank": [{"query_index": int(q), "best_correct_gallery_index": int(b),
                                   "margin": float(m), "best_correct_rank": int(r)}
                                  for q, b, m, r in zip(errors, best, margin, rank)],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sim", "anno", "image-root", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser


def main(argv=None):
    parser, args = build_parser(), None
    args = parser.parse_args(argv)
    if args.seed < 0 or args.bootstrap < 1:
        parser.error("--seed must be nonnegative and --bootstrap positive")
    try:
        try:
            from .R1_joint_enrichment import write_audited_report
        except ImportError:
            from R1_joint_enrichment import write_audited_report
        ann, records = load_annotations(args.anno), load_records(args.anno)
        sim = load_similarity(args.sim, ann)
        sizes, areas, manifest = read_image_sizes(ann, args.image_root)
        report = analyze(sim, ann, records, areas, sizes, args.seed, args.bootstrap)
        report["image_input_manifest"] = manifest
        inputs = {"sim": args.sim, "anno": args.anno}
        report["image_manifest_provenance_note"] = "Each input image sha256_16 and bytes are explicitly recorded in image_input_manifest."
        path = write_audited_report(args, "R3_finegrained_causes", report, inputs,
                                   {"same_source_negative_draw": args.seed, "identity_bootstrap": args.seed}, argv)
        print("R3 descriptive association analysis complete:", path)
        return 0
    except (OSError, ValueError, AssertionError, KeyError) as exc:
        if "write_audited_report" in locals():
            try:
                write_audited_report(args, "R3_finegrained_causes",
                                     {"status": "stopped", "error": str(exc)},
                                     {"sim": args.sim, "anno": args.anno},
                                     {"same_source_negative_draw": args.seed, "identity_bootstrap": args.seed}, argv)
            except (OSError, ValueError) as report_error:
                print(f"Stopped-report write unavailable: {report_error}", file=sys.stderr)
        parser.exit(2, f"R3 input/self-check error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
