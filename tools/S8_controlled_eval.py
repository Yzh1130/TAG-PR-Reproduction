"""Offline dual-view retrieval with explicit text-source image exclusion."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._analysis_common import (
    column_topk_mean, eligible_queries, evaluate, load_annotations, load_similarity,
    make_mock_annotation, make_mock_similarity, print_table, write_json,
)


EXPECTED_R1 = 63.63306


def self_check_gate(scores, ann, sim_path=None):
    """No query selection, view filtering, source exclusion, or CSLS here."""
    result = evaluate(scores, ann)
    if result["n_query_excluded_no_positive"]:
        raise ValueError("full-gallery self-check unexpectedly removed queries without positives")
    expected_fp32 = float(np.float32(EXPECTED_R1))
    aligned = result["R1"] == expected_fp32
    gate = {"metrics": result, "expected_R1_display": f"{EXPECTED_R1:.5f}",
            "expected_R1_fp32": expected_fp32, "aligned_to_reference": aligned,
            "matrix_dtype": str(scores.dtype), "postprocessing": "none"}
    print(f"FULL SELF-CHECK R@1={result['R1']!r} ({result['R1']:.5f}%) "
          f"R@5={result['R5']:.5f} R@10={result['R10']:.5f} mAP={result['mAP']:.5f}")
    if scores.dtype == np.float32 and not aligned:
        raise ValueError(
            f"fp32 self-check failed: R@1={result['R1']!r}, expected {expected_fp32!r} "
            f"(online display {EXPECTED_R1:.5f}%); check checkpoint, annotation order, "
            "raw vs postprocessed matrix, and tied scores. Controlled results were not produced."
        )
    if scores.dtype == np.float16:
        print("WARNING: float16 quantization may change ranking ties; strict fp32 parity "
              "requires a --dump_sim_dtype float32 export.", file=sys.stderr)
    if sim_path is not None:
        meta_path = Path(sim_path).with_name("sim_meta.json")
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as handle:
                meta = json.load(handle)
            if (meta.get("n_query"), meta.get("n_gallery")) != scores.shape:
                raise ValueError("sibling sim_meta.json shape does not match sim_qg")
            online = meta.get("raw_metrics", {})
            gate["online_metadata_metrics"] = online
            gate["online_metric_differences"] = {
                local: result[local] - float(online[remote])
                for local, remote in (("R1", "R@1"), ("R5", "R@5"),
                                      ("R10", "R@10"), ("mAP", "mAP")) if remote in online
            }
            if scores.dtype == np.float32 and "R@1" in online and result["R1"] != online["R@1"]:
                raise ValueError("fp32 R@1 differs from sim_meta.json; verify matrix/snapshot pairing")
    return gate


def controlled_protocols(scores, ann, csls_k=None):
    dual = np.isin(ann.query_ids, ann.dual_ids)
    output = {}
    for query_view, query_name in ((0, "A"), (1, "G")):
        q = np.flatnonzero(dual & (ann.query_views == query_view))
        for gallery_view, gallery_name in ((0, "A"), (1, "G")):
            g = np.flatnonzero(ann.gallery_views == gallery_view)
            name = f"{query_name}->{gallery_name}"
            raw = evaluate(scores, ann, q, g, exclude_source=True)
            protocol = {"raw": raw}
            if csls_k is not None:
                eligible, _ = eligible_queries(ann, q, g, exclude_source=True)
                if not len(eligible):
                    protocol["csls"] = dict(raw)
                    protocol["csls_note"] = "no eligible queries; no penalty estimated"
                else:
                    penalties = column_topk_mean(
                        scores, csls_k, eligible, g, source_indices=ann.query_source_indices,
                    )
                    protocol["csls"] = evaluate(scores, ann, q, g, exclude_source=True,
                                                 penalties=penalties)
                    protocol["csls_penalty_stats"] = {
                        "mean": float(penalties.mean()), "std": float(penalties.std()),
                        "min": float(penalties.min()), "max": float(penalties.max()),
                    }
            output[name] = protocol
    return output


def display(protocols):
    rows = []
    for name, protocol in protocols.items():
        for setting in ("raw", "csls"):
            if setting not in protocol:
                continue
            result = protocol[setting]
            metric = lambda key: "N/A" if result[key] is None else f"{result[key]:.5f}"
            rows.append([name, setting, *[metric(k) for k in ("R1", "R5", "R10", "mAP")],
                         result["n_query"], result["n_query_excluded_no_positive"],
                         result["gallery_size"],
                         f"{result['gallery_size_min']}-{result['gallery_size_max']}"])
    print_table(["Protocol", "Scores", "R@1", "R@5", "R@10", "mAP", "Queries",
                 "No-positive", "Gallery", "After exclusion"], rows)


def self_test():
    ann = make_mock_annotation()
    scores = make_mock_similarity(ann)
    scores[0, 6] = 4.0  # a ground-domain hard distractor for an aerial query
    full = evaluate(scores, ann)
    assert scores.shape == (20, 10) and full["R1"] == 95.0
    protocols = controlled_protocols(scores, ann, csls_k=2)
    for name, values in protocols.items():
        qview, gview = (0 if letter == "A" else 1 for letter in name.split("->"))
        q = np.flatnonzero(np.isin(ann.query_ids, ann.dual_ids) & (ann.query_views == qview))
        g = np.flatnonzero(ann.gallery_views == gview)
        eligible, removed = eligible_queries(ann, q, g, True)
        for mode in ("raw", "csls"):
            penalty = np.zeros(len(g), dtype=np.float32)
            if mode == "csls":
                for j, gallery in enumerate(g):
                    candidates = [scores[query, gallery] for query in eligible
                                  if ann.query_source_indices[query] != gallery]
                    penalty[j] = np.mean(sorted(candidates, reverse=True)[:2], dtype=np.float32)
                actual = column_topk_mean(scores, 2, eligible, g,
                                          source_indices=ann.query_source_indices, block_size=3)
                np.testing.assert_array_equal(actual, penalty)
            hits, aps = {1: 0, 5: 0, 10: 0}, []
            for query in eligible:
                candidates = [int(gallery) for gallery in g
                              if gallery != ann.query_source_indices[query]]
                def score(gallery):
                    value = scores[query, gallery]
                    return value if mode == "raw" else 2 * value - penalty[np.flatnonzero(g == gallery)[0]]
                ranked = sorted(candidates, key=lambda gallery: (-score(gallery), gallery))
                positives = [ann.gallery_ids[index] == ann.query_ids[query] for index in ranked]
                ranks = [i + 1 for i, match in enumerate(positives) if match]
                aps.append(sum((j + 1) / rank for j, rank in enumerate(ranks)) / len(ranks))
                for k in hits:
                    hits[k] += any(positives[:k])
            result = values[mode]
            for k, hit in hits.items():
                expected = float(np.float32(np.float32(hit) * np.float32(100)) / np.float32(len(eligible)))
                assert result[f"R{k}"] == expected
            assert abs(result["mAP"] - np.mean(aps) * 100) < 1e-10
            assert result["n_query_excluded_no_positive"] == removed
    all_excluded = evaluate(scores, ann, exclude_source=True)
    assert all_excluded["n_query_excluded_no_positive"] == 4  # two singleton identities × two captions
    print("S8 MOCK shape=(20, 10); raw full R@1=95.0; oracle metric/mask error=0")
    print("S8 MOCK source exclusion drops 4 singleton queries; all CSLS penalties match manual top-k")
    display(protocols)
    print("S8 SELF-TEST PASSED")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim_qg", help="raw sim_qg.npy (float32 recommended for exact self-check)")
    parser.add_argument("--ann", help="TAG-PEDES test_reid.json")
    parser.add_argument("--csls_k", type=int, nargs="?", const=10, default=None,
                        help="also evaluate per-protocol CSLS (omitted: disabled; flag alone: k=10)")
    parser.add_argument("--out", help="result JSON path")
    parser.add_argument("--self_test", action="store_true", help="run synthetic 20-query/10-gallery checks")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        if not all((args.sim_qg, args.ann, args.out)):
            parser.error("--sim_qg, --ann and --out are required unless --self_test is used")
        if args.csls_k is not None and args.csls_k < 1:
            parser.error("--csls_k must be positive")
        ann = load_annotations(args.ann)
        scores = load_similarity(args.sim_qg, ann)
        gate = self_check_gate(scores, ann, args.sim_qg)
        protocols = controlled_protocols(scores, ann, args.csls_k)
        result = {"tool": "S8_controlled_eval", "annotation": ann.summary,
                  "sim_qg": str(Path(args.sim_qg).resolve()), "ann": str(Path(args.ann).resolve()),
                  "self_check": gate, "csls_k": args.csls_k,
                  "protocol": "dual-view-identity queries; all target-view gallery images; exclude source image",
                  "csls_calibration": "eligible protocol queries only; source-image pairs also excluded before top-k; "
                                      "columns with fewer than k eligible queries use all available queries",
                  "protocols": protocols}
        display(protocols)
        write_json(args.out, result)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
