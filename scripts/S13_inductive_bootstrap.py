"""S13: paired bootstrap of fixed-bank CSLS gains on archived retrieval scores.

NumPy-only, read-only analysis. Calibration penalties are estimated once from
each complete bank, then held fixed while queries/images/identities are sampled.
An S12 Gram-reconstructed training bank is an offline inductive simulation:
its coordinate alignment was fitted using test queries, NOT a fully
query-independent end-to-end inductive pipeline.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from ._offline_common import (
        compact_metrics, evaluate, json_ready, load_annotations, load_matrix,
        load_similarity, print_table, provenance, topk_values, write_report,
    )
    from ._s11_descriptive import paired_test
except ImportError:
    from _offline_common import (
        compact_metrics, evaluate, json_ready, load_annotations, load_matrix,
        load_similarity, print_table, provenance, topk_values, write_report,
    )
    from _s11_descriptive import paired_test


CONFIGURATIONS = (
    ("a", "train_full", 0.5, 30),
    ("b", "train_full", 0.25, 10),
    ("c", "test_full", 0.5, 10),
)
REGIME_WARNING = (
    "The supplied S12 reconstructed training bank used a W fitted on full test "
    "queries. Training-bank gains here are archived-matrix offline simulations, "
    "not evidence of a fully query-independent inductive pipeline."
)
FRESH_BANK_NOTE = (
    "The bank filename contains '_fresh': classified by the E3 filename "
    "convention as a fresh-forward training bank, without a test-fitted W. "
    "This filename classification does not verify the extraction provenance; "
    "retain the corresponding E3 JSON as evidence."
)


def _parse_extra_config(value):
    """Parse an optional bank:alpha:k without changing fixed a/b/c settings."""
    if not isinstance(value, str) or len(value.split(":")) != 3:
        raise ValueError("extra config must be BANK:ALPHA:K, e.g. train_full:0.5:10")
    bank_name, alpha, k = value.split(":")
    if bank_name not in ("train_full", "test_full"):
        raise ValueError("extra config BANK must be train_full or test_full")
    try:
        alpha, k = float(alpha), int(k)
    except ValueError as exc:
        raise ValueError("extra config ALPHA must be a number and K a positive integer") from exc
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("extra config ALPHA must be finite and nonnegative")
    if k < 1:
        raise ValueError("extra config K must be a positive integer")
    return bank_name, alpha, k


def _extra_config_arg(value):
    try:
        _parse_extra_config(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _config_label(index):
    label = ""
    while index >= 0:
        index, digit = divmod(index, 26)
        label = chr(ord("a") + digit) + label
        index -= 1
    return label


def _extra_config_provenance(args, argv):
    """New optional-config audit only; legacy invocations retain their schema.

    SHA-256 requires reading the complete inputs. This additional IO and the
    git dependency are deliberately incurred only by explicit --extra-config.
    """
    try:
        from ._analysis_audit import git_head, sha256_16
    except ImportError:
        from _analysis_audit import git_head, sha256_16
    return {
        "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
        "seeds": {"bootstrap": int(args.seed)},
        "git_HEAD": git_head(),
        "inputs": {name: {"path": str(Path(value).expanduser().resolve()),
                          "sha256_16": sha256_16(Path(value).expanduser())}
                   for name, value in (("sim", args.sim), ("bank", args.bank), ("anno", args.anno))},
        "scope": "present only when --extra-config is explicitly provided; all bootstrap comparisons use the recorded seed",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", "--sim-qg", dest="sim", required=True,
                        help="read-only archived query-gallery .npy (float16/float32)")
    parser.add_argument("--bank", "--sim-bank-g", dest="bank", required=True,
                        help="read-only S12 full training-text bank-gallery .npy")
    parser.add_argument("--anno", "--ann", dest="anno", required=True,
                        help="test_reid.json; image order then caption order defines queries")
    parser.add_argument("--out", required=True,
                        help="new output directory, outside all input archive directories")
    parser.add_argument("--tag", default="fk32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="paired percentile bootstrap repeats (default 10000)")
    parser.add_argument("--score-block-size", type=int, default=256,
                        help="bank rows per exact streaming column top-k block")
    parser.add_argument("--extra-config", action="append", type=_extra_config_arg,
                        default=argparse.SUPPRESS,
                        metavar="BANK:ALPHA:K",
                        help="append d/e/... after unchanged a/b/c; repeatable; BANK is train_full or test_full (e.g. train_full:0.5:10)")
    parser.add_argument("--summary-table", nargs="?", const="auto", default=argparse.SUPPRESS,
                        metavar="JSON", help="optionally write a separate four-metric/identity-CI table; omitted path: new file under --out; unchanged a/b/c retained")
    return parser


def _finite_matrix(matrix, name, columns=None, block_size=256):
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or
            min(matrix.shape) < 1 or not np.issubdtype(matrix.dtype, np.floating)):
        raise ValueError(f"{name}: expected a nonempty floating matrix")
    if columns is not None and matrix.shape[1] != columns:
        raise ValueError(f"{name}: expected {columns} gallery columns, got {matrix.shape[1]}")
    minimum, maximum = float("inf"), float("-inf")
    for start in range(0, len(matrix), block_size):
        block = matrix[start:start + block_size]
        if not np.isfinite(block).all():
            raise ValueError(f"{name}: NaN/infinity near row {start}")
        minimum, maximum = min(minimum, float(block.min())), max(maximum, float(block.max()))
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "min": minimum, "max": maximum, "clipped": False,
            "cosine_bounds_enforced": False}


def _penalty_summary(penalty, n_bank, k):
    return {"n_bank": int(n_bank), "k": int(k), "shape": list(penalty.shape),
            "dtype": str(penalty.dtype), "mean": float(penalty.mean(dtype=np.float64)),
            "std": float(penalty.std(dtype=np.float64)), "min": float(penalty.min()),
            "max": float(penalty.max()),
            "sha256_native_array_bytes": hashlib.sha256(penalty.tobytes()).hexdigest(),
            "estimated_once_from_complete_bank": True, "fixed_during_bootstrap": True}


def _paired_results(ann, first, second, repeats, seed):
    queries = np.asarray(first["details"]["query_indices"], dtype=np.int64)
    if not np.array_equal(queries, second["details"]["query_indices"]):
        raise ValueError("paired comparisons require identical ordered evaluation queries")
    results = {}
    for unit, groups in (("query", None),
                         ("image", ann.query_source_indices[queries]),
                         ("identity", ann.query_ids[queries])):
        result = paired_test(first["details"]["correct"], second["details"]["correct"],
                             groups=groups, repeats=repeats, seed=seed)
        result["resampling_unit"] = unit
        results[unit] = result
    return results


def run_analysis(sim, bank, ann, bootstrap=10000, seed=2026, block_size=256, progress=print,
                 extra_configs=None, fresh_bank=False):
    """Fix all column penalties first; bootstrap only paired binary outcomes."""
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError("score block size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if sim.shape != (len(ann.query_ids), len(ann.gallery_ids)):
        raise ValueError("query-gallery shape must match annotation query/gallery order")
    sim_info = _finite_matrix(sim, "query-gallery", block_size=block_size)
    bank_info = _finite_matrix(bank, "training bank-gallery", sim.shape[1], block_size)
    if len(bank) < 30 or len(sim) < 10:
        raise ValueError("fixed S13 configurations require >=30 training bank rows and >=10 test queries")
    extras = tuple((_config_label(len(CONFIGURATIONS) + index), *_parse_extra_config(value))
                   for index, value in enumerate(extra_configs or ()))
    for label, bank_name, _, k in extras:
        n_bank = len(bank) if bank_name == "train_full" else len(sim)
        if k > n_bank:
            raise ValueError(f"extra config {label}: k={k} exceeds {bank_name} rows={n_bank}")
    warnings = [] if fresh_bank else [REGIME_WARNING]
    if bank_info["min"] < -1.0001 or bank_info["max"] > 1.0001:
        warnings.append(("Fresh bank" if fresh_bank else "Mapped S12 bank") +
                        " scores extend beyond [-1,1]; finite values are used unchanged, without clipping or renormalization.")
    if progress:
        if fresh_bank:
            progress("S13 INFO: " + FRESH_BANK_NOTE)
        for warning in warnings:
            progress("S13 WARN: " + warning)
        progress("S13 fixing train top-30/top-10 and test top-10 penalties from complete banks; no bootstrap recalibration.")
    started = time.perf_counter()
    train_top = topk_values(bank, 30, block_size=block_size)
    test_top = topk_values(sim, 10, block_size=block_size)
    penalties = {
        "a": train_top.mean(axis=0, dtype=np.float32),
        "b": train_top[:10].mean(axis=0, dtype=np.float32),
        "c": test_top.mean(axis=0, dtype=np.float32),
    }
    extra_penalties = {("train_full", 30): penalties["a"],
                       ("train_full", 10): penalties["b"],
                       ("test_full", 10): penalties["c"]}
    for label, bank_name, _, k in extras:
        key = (bank_name, k)
        if key not in extra_penalties:
            top = train_top if bank_name == "train_full" else test_top
            if k > len(top):
                values = bank if bank_name == "train_full" else sim
                top = topk_values(values, k, block_size=block_size)
            extra_penalties[key] = top[:k].mean(axis=0, dtype=np.float32)
        penalties[label] = extra_penalties[key]
    for penalty in penalties.values():
        penalty.flags.writeable = False
    penalty_elapsed = time.perf_counter() - started
    raw = evaluate(sim, ann)
    if raw["n_query"] != len(ann.query_ids) or raw["skipped_no_positive"]:
        raise ValueError("S13 requires the full annotated query set, without source exclusions")
    comparisons, corrected = {}, {}
    for label, bank_name, alpha, k in CONFIGURATIONS + extras:
        if progress:
            progress(f"S13 {label}: {bank_name}, alpha={alpha}, k={k}; evaluating once then paired bootstrap x{bootstrap}.")
        result = evaluate(sim, ann, penalty=penalties[label], alpha=alpha)
        corrected[label] = result
        comparisons[label] = {
            "bank": bank_name, "alpha": alpha, "k": k,
            "regime": ("transductive: complete test-query bank" if bank_name == "test_full" else
                       "E3 fresh-forward training-bank inductive pipeline (filename convention; provenance not verified)" if fresh_bank else
                       "S12 reconstructed training-bank offline simulation (test-calibrated W)"),
            "metrics": compact_metrics(result),
            "observed_delta_R1_pp": float(100.0 * (np.asarray(result["details"]["correct"], dtype=np.int64)
                                                    - np.asarray(raw["details"]["correct"], dtype=np.int64)).mean()),
            "penalty": _penalty_summary(penalties[label], len(sim) if bank_name == "test_full" else len(bank), k),
            "bootstrap": _paired_results(ann, raw, result, bootstrap, seed),
        }
    difference = _paired_results(ann, corrected["c"], corrected["a"], bootstrap, seed)
    for value in difference.values():
        value["delta_definition"] = "[R1(a)-R1(raw)] - [R1(c)-R1(raw)] = R1(a)-R1(c), percentage points"
    return {
        "annotation": ann.summary, "query_gallery": sim_info, "training_bank_gallery": bank_info,
        "raw": compact_metrics(raw), "configurations": comparisons,
        "gain_difference_a_minus_c": {"definition": "delta(a vs raw) - delta(c vs raw); raw cancels query by query",
                                      "bootstrap": difference},
        "bootstrap_protocol": {
            "repeats": int(bootstrap), "seed": int(seed), "units": ["query", "image", "identity"],
            "implementation": "_s11_descriptive.paired_test",
            "pairing": "Each comparison uses identical ordered queries, group labels and seed. Consequently every resampling draw is identical within a unit; a-c is computed directly as correct_a-correct_c, not by subtracting independent intervals.",
            "clusters": "Uniformly sample whole clusters with replacement; all captions retained with multiplicity; query-weighted accuracy, not a mean of cluster accuracies.",
            "calibration": "Column top-k means computed once from each complete bank, frozen before all bootstrap calls; no resampling or refitting of banks.",
            "conditional_inference": "Intervals condition on the supplied matrices and fixed bank penalties; they exclude feature extraction, routing, alignment, and bank estimation uncertainty.",
            "p_method": "Two-sided percentile-bootstrap sign tails with +1 Monte Carlo correction; not a null-distribution bootstrap significance test. Query-level exact McNemar is additionally reported by the reused helper.",
        },
        "formula": "corrected s = s - alpha * mean(topk(bank[:,g])); alpha=.5 is ranking-equivalent to 2*s-r(g)",
        "source_image_exclusion": False, "labels_used_for_bank_penalty": False,
        "pure_query_independent_inductive_pipeline": bool(fresh_bank),
        "training_bank_provenance_note": FRESH_BANK_NOTE if fresh_bank else REGIME_WARNING,
        "warnings": warnings,
        **({"training_bank_provenance_classification": {
            "source": "bank filename contains '_fresh' (case-insensitive)",
            "verified": False, "scope": "train_full configurations only; test_full remains transductive",
        }} if fresh_bank else {}),
        "penalty_elapsed_sec": penalty_elapsed, "score_block_size": int(block_size),
    }


def print_results(report):
    baseline = report["raw"]
    print("S13 RAW " + " ".join(f"{key}={baseline[key]:.8f}" for key in ("R1", "R5", "R10", "mAP")))
    rows = []
    for label, result in report["configurations"].items():
        for unit, stats in result["bootstrap"].items():
            lower, upper = stats["ci95_percentile_pp"]
            rows.append((label, result["bank"], f"{result['alpha']:g}", str(result["k"]), unit,
                         f"{result['metrics']['R1']:.8f}", f"{stats['observed_delta_pp']:+.6f}",
                         f"{stats['bootstrap_mean_delta_pp']:+.6f}",
                         f"[{lower:+.6f}, {upper:+.6f}]", f"{stats['p_two_sided']:.8g}"))
    print_table(("config", "bank", "alpha", "k", "unit", "R1", "delta_pp", "boot_mean_pp", "CI95_pp", "p_sign_tail"), rows)
    rows = []
    for unit, stats in report["gain_difference_a_minus_c"]["bootstrap"].items():
        lower, upper = stats["ci95_percentile_pp"]
        rows.append(("gain(a)-gain(c)", unit, f"{stats['observed_delta_pp']:+.6f}",
                     f"{stats['bootstrap_mean_delta_pp']:+.6f}",
                     f"[{lower:+.6f}, {upper:+.6f}]", f"{stats['p_two_sided']:.8g}"))
    print_table(("comparison", "unit", "delta_pp", "boot_mean_pp", "CI95_pp", "p_sign_tail"), rows)
    print("S13 p: two-sided bootstrap sign-tail probability (+1 correction), NOT a null-bootstrap p-value.")


def build_summary_table(sim, bank, ann, bootstrap=10000, seed=2026, block_size=256):
    """One model/matrix, fixed paired settings; no selection from test results."""
    raw = evaluate(sim, ann)
    settings = (("inductive_default", "train_full", 0.5, 10),
                ("inductive_075_30", "train_full", 0.75, 30),
                ("transductive_k10", "test_full", 0.5, 10))
    train_top = topk_values(bank, 30, block_size=block_size)
    test_top = topk_values(sim, 10, block_size=block_size)
    fields = ("R1", "R5", "R10", "mAP")
    rows = [{"setting": "uncorrected", "bank": None, "alpha": None, "k": None,
             "metrics": {key: raw[key] for key in fields}, "delta_R1_pp": 0.0,
             "identity_bootstrap": None}]
    for label, bank_name, alpha, k in settings:
        top = train_top if bank_name == "train_full" else test_top
        penalty = top[:k].mean(axis=0, dtype=np.float32)
        result = evaluate(sim, ann, penalty=penalty, alpha=alpha)
        if not np.array_equal(raw["details"]["query_indices"], result["details"]["query_indices"]):
            raise ValueError("summary-table requires the identical ordered query set")
        stats = paired_test(raw["details"]["correct"], result["details"]["correct"],
                            groups=ann.query_ids[raw["details"]["query_indices"]], repeats=bootstrap, seed=seed)
        rows.append({"setting": label, "bank": bank_name, "alpha": alpha, "k": k,
                     "metrics": {key: result[key] for key in fields},
                     "delta_R1_pp": stats["observed_delta_pp"], "identity_bootstrap": stats})
    return {"status": "complete", "rows": rows, "n_query": len(ann.query_ids),
            "n_gallery": len(ann.gallery_ids), "formula": "S - alpha*mean(top-k(bank[:,g]))",
            "calibration_frozen_before_bootstrap": True, "source_image_exclusion": False,
            "selection_from_test_metrics": False,
            "interval_scope": "paired whole-identity percentile CI; conditional on each supplied matrix and fixed bank",
            "a_b_c_legacy_configurations_changed": False}


class _Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def write(self, text):
        self.first.write(text)
        self.second.write(text)

    def flush(self):
        self.first.flush()
        self.second.flush()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap < 1 or args.score_block_size < 1 or args.seed < 0:
        parser.error("--bootstrap/--score-block-size must be positive; --seed must be nonnegative")
    output = Path(args.out).expanduser().resolve()
    input_paths = [Path(value).expanduser().resolve() for value in (args.sim, args.bank, args.anno)]
    if any(output == path.parent or path.parent in output.parents for path in input_paths):
        parser.error("--out must not be inside a read-only input archive directory")
    if "_archive_evidence" in {part.lower() for part in output.parts}:
        parser.error("_archive_evidence is read-only; choose a new output directory")
    report = {"tool": "S13", "tag": args.tag, "arguments": vars(args),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    log, started, failure = io.StringIO(), time.perf_counter(), None
    sim = bank = None
    with redirect_stdout(_Tee(sys.stdout, log)):
        try:
            ann = load_annotations(args.anno)
            sim = load_similarity(args.sim, ann)
            bank = load_matrix(args.bank)
            report["inputs"] = {"sim": provenance(args.sim), "bank": provenance(args.bank),
                                "annotation_path": str(input_paths[2]), "matrix_mode": "read-only memory map"}
            if hasattr(args, "extra_config"):
                report["extra_config_provenance"] = _extra_config_provenance(args, argv)
            report.update(run_analysis(sim, bank, ann, args.bootstrap, args.seed, args.score_block_size,
                                       extra_configs=getattr(args, "extra_config", None),
                                       fresh_bank="_fresh" in Path(args.bank).name.lower()))
            report["status"] = "complete_with_warnings" if report["warnings"] else "complete"
            print_results(report)
            if hasattr(args, "summary_table"):
                try:
                    from .R1_joint_enrichment import write_audited_report
                except ImportError:
                    from R1_joint_enrichment import write_audited_report
                table = build_summary_table(sim, bank, ann, args.bootstrap, args.seed, args.score_block_size)
                table.update(model_tag=args.tag, fresh_bank_filename="_fresh" in Path(args.bank).name.lower(),
                             training_bank_provenance_note=report["training_bank_provenance_note"])
                summary_args = argparse.Namespace(**vars(args))
                safe_tag = "".join(c if c.isalnum() or c in "_.-" else "_" for c in args.tag)
                summary_args.out = (str(output / f"S13_{safe_tag}_summary_table.json")
                                    if args.summary_table == "auto" else args.summary_table)
                paths = write_audited_report(summary_args, "S13_summary_table", table,
                                            dict(sim=args.sim, bank=args.bank, anno=args.anno),
                                            {"bootstrap": args.seed}, argv)
                report["summary_table_file"] = str(paths[0])
                print_table(("summary setting", "R1", "R5", "R10", "mAP", "delta_R1_pp", "identity_CI95_pp"),
                            [[row["setting"], *[row["metrics"][key] for key in ("R1", "R5", "R10", "mAP")],
                              row["delta_R1_pp"], row["identity_bootstrap"]["ci95_percentile_pp"]
                              if row["identity_bootstrap"] else None] for row in table["rows"]])
        except Exception as exc:
            failure = exc
            report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
            print("S13 STOPPED: " + report["error"])
        finally:
            for matrix in (sim, bank):
                if isinstance(matrix, np.memmap):
                    matrix._mmap.close()
        report["elapsed_sec"] = time.perf_counter() - started
        print(f"S13 SUMMARY status={report['status']} elapsed_sec={report['elapsed_sec']:.3f}")
    try:
        paths = write_report(output, "S13_" + args.tag, json_ready(report), log.getvalue())
    except (OSError, ValueError) as exc:
        parser.error(f"cannot write new S13 report: {exc}")
    print(f"JSON: {paths[0]}\nLOG: {paths[1]}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
