import os, sys, json, glob, time, numpy as np
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7b"; S7A = "/root/autodl-tmp/_e_ext/s7a"

ALPHA, K, SEEDS, D_LEVELS, NULL_DRAWS = 0.5, 10, [2026, 2027, 2028, 2029, 2030], [250, 500, 1000, 2000], 300
NAME = {"cuhkpedes": "CUHK-PEDES", "icfgpedes": "ICFG-PEDES", "aeri": "AERI-PEDES", "tbapr": "TBAPR-merged"}

from pathlib import Path
try:
    from .S7_aggregate_seeds import (
        common_parser, parse_common, select_runs, result_directory, output_name,
        save_npy, save_npz, write_json, provenance, run_inputs, peak_rss_gb,
        logged_rss, load_s7a_artifact,
    )
except ImportError:
    from S7_aggregate_seeds import (
        common_parser, parse_common, select_runs, result_directory, output_name,
        save_npy, save_npz, write_json, provenance, run_inputs, peak_rss_gb,
        logged_rss, load_s7a_artifact,
    )


def log(*a): print(*a, flush=True)

def skew(x):
    x = np.asarray(x, np.float64); s = x.std(); return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0

def gini(c):
    c = np.sort(np.asarray(c, np.float64)); n = len(c); return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum())) if c.sum() else 0.0

def r_cols(Ssub):
    # r(g) = mean of the K largest query similarities per column; top block SORTED before summation so the
    # result is bitwise independent of memory layout / partition internals (definition unchanged).
    top = np.sort(np.partition(Ssub, Ssub.shape[0] - K, axis=0)[-K:], axis=0)
    return top.astype(np.float64).mean(0).astype(np.float32)

def eval_r1(sc, qpid, gpid):
    top1 = sc.argmax(1); return (gpid[top1] == qpid).mean() * 100, top1

def null_skew(nq, ng, rng): return np.mean([skew(np.bincount(rng.integers(0, ng, nq * K), minlength=ng)) for _ in range(NULL_DRAWS)])

def null_gini(nq, ng, rng): return np.mean([gini(np.bincount(rng.integers(0, ng, nq), minlength=ng)) for _ in range(NULL_DRAWS)])

def load_remap(m, originals):
    tbl = m["pid_remap_table"]; tbl = json.load(open(tbl)) if isinstance(tbl, str) else tbl
    if "evaluation_original_to_contiguous" in tbl: tbl = tbl["evaluation_original_to_contiguous"]
    return set(int(tbl[str(o)]) for o in originals)

def main(argv=None):
    global ROOT, OUT, t0, COH, SELECTED_RUNS
    parser = common_parser("S7b_distractor", "S7b_distractor: original external offline calculations, explicit runs")
    parser.add_argument("--cohort-json", required=True, help="Locked cohort_splits_seed2026.json")
    parser.add_argument("--s7a-json", required=True, help="Matching tag/seed S7a summary, penalties resolved by manifest")
    args = parse_common(parser, argv)
    ROOT = args.root
    SELECTED_RUNS = select_runs(args)
    COH = json.load(open(args.cohort_json))
    input_paths = [args.cohort_json, args.s7a_json] + [path for d, _ in SELECTED_RUNS for path in run_inputs(d, ("meta.json", "sim_qg.npy"))]
    OUT = result_directory(args)
    t0 = time.time()
    R = {}; ASSERT = None
    for d, m in SELECTED_RUNS:
        mp = d + "/meta.json"
        ds = m["dataset"]; coh = COH["datasets"][NAME[ds]]
        S = np.load(d + "/sim_qg.npy", mmap_mode="r")
        qpid = np.asarray(m["query_pid"]); gpid = np.asarray(m["gallery_pid"])
        A = load_remap(m, coh["cohort_A"])
        Qm = np.isin(qpid, list(A)); Pm = np.isin(gpid, list(A)); Poolm = ~Pm
        assert Qm.sum() == coh["nq"], f"{ds}: cohort nq mismatch {Qm.sum()}/{coh['nq']}"
        if ds == "tbapr":   # official gallery is row-level (117 duplicate rows); cohort file counted unique images
            log("  [tbapr] row-level gallery: |P| rows=%d (unique imgs %d)  |Pool| rows=%d (unique imgs %d)" % (Pm.sum(), coh["positive_union"], Poolm.sum(), coh["pool"]))
        else:
            assert Pm.sum() == coh["positive_union"] and Poolm.sum() == coh["pool"], \
                f"{ds}: cohort mismatch P {Pm.sum()}/{coh['positive_union']} pool {Poolm.sum()}/{coh['pool']}"
        log("\n" + "=" * 78 + "\n### %s  K=%d nq=%d |P|=%d |Pool|=%d" % (ds, coh["K"], Qm.sum(), Pm.sum(), Poolm.sum()))
        Sq = np.asarray(S[Qm]); qp = qpid[Qm]
        r_cohQ = r_cols(Sq)                                            # pre-registered: r over cohort queries Q
        r_fullQ, r_full_path = load_s7a_artifact(args.s7a_json, d, ds, "transductive")            # post-hoc: r over ALL official queries (symmetric)
        r_ind, r_ind_path = load_s7a_artifact(args.s7a_json, d, ds, "inductive")                 # inductive bank r (query-independent)
        input_paths.extend([r_full_path, r_ind_path])
        if isinstance(m["pid_remap_table"], str): input_paths.append(m["pid_remap_table"])
        Pidx = np.where(Pm)[0]; Poolidx = np.where(Poolm)[0]
        levels = [D for D in D_LEVELS if D <= len(Poolidx)] + ["full"]
        rows = []
        for D in levels:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                sel = Poolidx if D == "full" else rng.choice(Poolidx, size=D, replace=False)
                cols = np.concatenate([Pidx, sel]); sc = Sq[:, cols]; gp = gpid[cols]; ng = len(cols)
                if ASSERT is None and D == 1000:
                    r_re = r_cols(sc); eq = bool(np.array_equal(r_cohQ[cols], r_re)); md = float(np.abs(r_cohQ[cols] - r_re).max())
                    log("  [Lock-A assertion] r sliced vs recomputed on Gallery(D=1000,seed=%d): array_equal=%s max|diff|=%.3g" % (seed, eq, md))
                    ASSERT = {"dataset": ds, "seed": seed, "array_equal": eq, "max_abs_diff": md}
                    if not eq: raise SystemExit("EQUIVALENCE ASSERTION FAILED")
                b_r1, t1b = eval_r1(sc, qp, gp)
                tq_r1, _ = eval_r1(sc - np.float32(ALPHA) * r_cohQ[cols][None, :], qp, gp)
                tf_r1, _ = eval_r1(sc - np.float32(ALPHA) * r_fullQ[cols][None, :], qp, gp)
                in_r1, _ = eval_r1(sc - np.float32(ALPHA) * r_ind[cols][None, :], qp, gp)
                nk = np.zeros(ng, np.int64); np.add.at(nk, np.argpartition(-sc, K, axis=1)[:, :K].ravel(), 1)
                c1 = np.bincount(t1b, minlength=ng)
                rows.append({"D": int(len(sel)), "level": str(D), "seed": seed, "nq": int(len(qp)), "ng": ng, "pos_union": int(len(Pidx)),
                             "pos_distractor_ratio": round(len(Pidx) / len(sel), 4),
                             "base_r1": round(b_r1, 4), "trans_cohortQ_r1": round(tq_r1, 4), "trans_fullQ_r1": round(tf_r1, 4), "ind_r1": round(in_r1, 4),
                             "trans_cohortQ_delta": round(tq_r1 - b_r1, 4), "trans_fullQ_delta": round(tf_r1 - b_r1, 4), "ind_delta": round(in_r1 - b_r1, 4),
                             "skew_N10": skew(nk), "excess_skew_N10": skew(nk) - null_skew(len(qp), ng, np.random.default_rng(2026)),
                             "gini_top1": gini(c1), "excess_gini": gini(c1) - null_gini(len(qp), ng, np.random.default_rng(2026)),
                             "distinct_top1": float((c1 > 0).sum() / ng)})
            sub = [r for r in rows if r["level"] == str(D)]; mu = lambda k: np.mean([r[k] for r in sub]); sd = lambda k: np.std([r[k] for r in sub])
            log("  D=%-5s ng=%-6d base=%6.2f±%.2f | transΔ cohortQ=%+.2f±%.2f  fullQ=%+.2f±%.2f | indΔ=%+.2f±%.2f | ex_skew=%.3f ex_gini=%.4f" % (
                str(D), int(mu("ng")), mu("base_r1"), sd("base_r1"), mu("trans_cohortQ_delta"), sd("trans_cohortQ_delta"),
                mu("trans_fullQ_delta"), sd("trans_fullQ_delta"), mu("ind_delta"), sd("ind_delta"), mu("excess_skew_N10"), mu("excess_gini")))
        x = np.log10([r["D"] for r in rows]); hx = np.array([r["excess_skew_N10"] for r in rows]); rng = np.random.default_rng(2026)
        rank = lambda v: np.argsort(np.argsort(v))
        fits = {}
        for key in ("trans_cohortQ_delta", "trans_fullQ_delta", "ind_delta"):
            y = np.array([r[key] for r in rows]); slope = np.polyfit(x, y, 1)[0]
            bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
            fits[key] = {"slope_vs_log10D": float(slope), "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         "spearman_vs_D": float(np.corrcoef(rank(x), rank(y))[0, 1]), "spearman_vs_excess_skew": float(np.corrcoef(rank(hx), rank(y))[0, 1])}
            log("  %-20s slope=%+.3f CI=[%+.3f,%+.3f]  ρ(Δ,D)=%+.3f  ρ(Δ,ex_skew)=%+.3f" % (key, slope, fits[key]["slope_ci95"][0], fits[key]["slope_ci95"][1],
                fits[key]["spearman_vs_D"], fits[key]["spearman_vs_excess_skew"]))
        log("  hub vs D: ρ(excess_skew, D)=%+.3f" % np.corrcoef(rank(x), rank(hx))[0, 1])
        R[ds] = {"cohort": {k: v for k, v in coh.items() if k != "cohort_A"}, "rows": rows, "fits": fits,
                 "spearman_excess_skew_vs_D": float(np.corrcoef(rank(x), rank(hx))[0, 1])}
    R["_meta"] = {"alpha": ALPHA, "k": K, "seeds": SEEDS, "null_draws": NULL_DRAWS, "equivalence_assertion": ASSERT,
                  "variants": {"trans_cohortQ": "PRE-REGISTERED (Lock-A): r(g) over cohort query set Q",
                               "trans_fullQ": "POST-HOC: r(g) over all official test queries (every gallery image has legitimate queries; symmetric)",
                               "ind": "inductive, training-text bank"},
                  "peak_rss_gb": peak_rss_gb(), "elapsed_sec": round(time.time() - t0, 1)}
    R["_meta"]["provenance"] = provenance(args, input_paths, {"gallery": SEEDS, "null": 2026, "slope_bootstrap": 2026, "run_seeds": {m["dataset"]: int(m["seed"]) for _, m in SELECTED_RUNS}})
    result_path = output_name(OUT, "S7b_distractor", args.tag, ".json")
    write_json(result_path, R)
    log("\n=== S7b SUMMARY: slope of Δ vs log10(D) within dataset ===")
    log("  %-10s %28s %28s %28s %10s" % ("dataset", "trans cohortQ (pre-reg)", "trans fullQ (post-hoc)", "inductive", "ρ(hub,D)"))
    for ds in [k for k in R if not k.startswith("_")]:
        f = R[ds]["fits"]; fmt = lambda k: "%+.3f [%+.3f,%+.3f]" % (f[k]["slope_vs_log10D"], *f[k]["slope_ci95"])
        log("  %-10s %28s %28s %28s %+10.3f" % (ds, fmt("trans_cohortQ_delta"), fmt("trans_fullQ_delta"), fmt("ind_delta"), R[ds]["spearman_excess_skew_vs_D"]))
    log("peak RSS %.2f GB, elapsed %ss" % (logged_rss(R["_meta"]["peak_rss_gb"]), R["_meta"]["elapsed_sec"]))
    log("wrote", result_path)
    return R


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"S7b_distractor: {error}") from error
