import os, json, glob, time, numpy as np
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7c"
ALPHAS = [0.25, 0.5, 0.75, 1.0]; KS = [5, 10, 16, 20, 30]; KMAX = 30; NBOOT, SEED, CHUNK = 10000, 2026, 2048

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

def latest(ds, split):
    return SELECTED_BY_KEY[(ds, split)]

def topk_block(row_iter, ng):                      # sorted top-KMAX values per column
    top = np.full((KMAX, ng), -np.inf, np.float32)
    for blk in row_iter:
        comb = np.concatenate([top, blk], 0); top = np.partition(comb, comb.shape[0] - KMAX, axis=0)[-KMAX:]
    return np.sort(top, axis=0)

def r_from_block(top, k): return top[-k:].astype(np.float64).mean(0).astype(np.float32)

def trans_block(S):
    nq, ng = S.shape; return topk_block((np.asarray(S[a:min(a + CHUNK, nq)]) for a in range(0, nq, CHUNK)), ng)

def ind_block(d, ng):
    bg = np.load(d + "/feats_bank_text_global.npy", mmap_mode="r"); bl = np.load(d + "/feats_bank_text_local.npy", mmap_mode="r")
    ig = np.load(d + "/feats_img_global.npy"); il = np.load(d + "/feats_img_local.npy")
    def it():
        for a in range(0, bg.shape[0], 4096):
            b = min(a + 4096, bg.shape[0]); yield ((np.asarray(bg[a:b]) @ ig.T + np.asarray(bl[a:b]) @ il.T) / 2).astype(np.float32)
    return topk_block(it(), ng)

def r1_of(S, r, alpha, qpid, gpid):
    nq = S.shape[0]; correct = np.empty(nq, bool)
    for a in range(0, nq, CHUNK):
        b = min(a + CHUNK, nq); sc = np.asarray(S[a:b], dtype=np.float32)
        if r is not None: sc = sc - np.float32(alpha) * r[None, :]
        correct[a:b] = gpid[sc.argmax(1)] == qpid[a:b]
    return correct

def boot(cb, cn, clusters):
    uniq, inv = np.unique(clusters, return_inverse=True); n = np.bincount(inv).astype(float)
    sb = np.bincount(inv, weights=cb.astype(float)); sn = np.bincount(inv, weights=cn.astype(float))
    rng = np.random.default_rng(SEED); m = len(uniq); dl = np.empty(NBOOT)
    for i in range(NBOOT): idx = rng.integers(0, m, m); dl[i] = 100 * (sn[idx].sum() - sb[idx].sum()) / n[idx].sum()
    lo, hi = np.percentile(dl, [2.5, 97.5]); return [round(float(lo), 3), round(float(hi), 3)]

def grid(S, block, qpid, gpid):
    base = r1_of(S, None, 0, qpid, gpid); out = {}
    for k in KS:
        r = r_from_block(block, k)
        for a in ALPHAS: out[(a, k)] = 100 * r1_of(S, r, a, qpid, gpid).mean() - 100 * base.mean()
    return base, out

def pick(g):  # max val R@1 gain; ties -> smaller k, then alpha closest to 0.5 (pre-specified here, before seeing test)
    return sorted(g.items(), key=lambda kv: (-kv[1], kv[0][1], abs(kv[0][0] - 0.5)))[0][0]

def main(argv=None):
    global ROOT, OUT, t0, SELECTED_BY_KEY
    parser = common_parser("S7c_valselect", "S7c_valselect: original external offline calculations, explicit runs")
    # CUHK val and all four explicit test runs are required.
    args = parse_common(parser, argv)
    ROOT = args.root
    required = {("cuhkpedes", "val")} | {(ds, "test") for ds in ("cuhkpedes", "icfgpedes", "aeri", "tbapr")}
    selected = select_runs(args, required)
    SELECTED_BY_KEY = {(m["dataset"], m["split"]): (d, m) for d, m in selected}
    input_paths = [path for d, _ in selected for path in run_inputs(d, ("meta.json", "sim_qg.npy", "feats_bank_text_global.npy", "feats_bank_text_local.npy", "feats_img_global.npy", "feats_img_local.npy"))]
    OUT = result_directory(args)
    t0 = time.time()
    R = {"grid": {"alphas": ALPHAS, "ks": KS}, "tie_break": "max val ΔR@1; then smaller k; then alpha nearest 0.5"}
    # ---------------- CUHK: validation-only selection ----------------
    dv, mv = latest("cuhkpedes", "val"); dt, mt = latest("cuhkpedes", "test")
    log("CUHK val run :", os.path.basename(dv)); log("CUHK test run:", os.path.basename(dt))
    Sv = np.load(dv + "/sim_qg.npy", mmap_mode="r"); qv = np.asarray(mv["query_pid"]); gv = np.asarray(mv["gallery_pid"])
    St = np.load(dt + "/sim_qg.npy", mmap_mode="r"); qt = np.asarray(mt["query_pid"]); gt = np.asarray(mt["gallery_pid"])
    log("  val nq=%d ng=%d | test nq=%d ng=%d" % (*Sv.shape, *St.shape))
    sel = {}
    for mode in ("transductive", "inductive"):
        bv = trans_block(Sv) if mode == "transductive" else ind_block(dv, Sv.shape[1])
        base_v, gv_ = grid(Sv, bv, qv, gv)
        best = pick(gv_)
        log("\n  [%s] VAL grid ΔR@1 (rows α, cols k=%s):" % (mode, KS))
        for a in ALPHAS: log("    α=%.2f  " % a + "  ".join("%+.2f" % gv_[(a, k)] for k in KS))
        log("  [%s] val baseline R@1=%.2f  selected (α,k)=%s  val Δ=%+.2f  | literature default (0.5,10) val Δ=%+.2f"
            % (mode, 100 * base_v.mean(), best, gv_[best], gv_[(0.5, 10)]))
        # apply ONCE to test
        bt = trans_block(St) if mode == "transductive" else ind_block(dt, St.shape[1])
        base_t = r1_of(St, None, 0, qt, gt)
        c_sel = r1_of(St, r_from_block(bt, best[1]), best[0], qt, gt)
        c_def = r1_of(St, r_from_block(bt, 10), 0.5, qt, gt)
        d_sel, d_def = 100 * (c_sel.mean() - base_t.mean()), 100 * (c_def.mean() - base_t.mean())
        log("  [%s] TEST (evaluated once): selected %s Δ=%+.2f idCI=%s | default (0.5,10) Δ=%+.2f idCI=%s"
            % (mode, best, d_sel, boot(base_t, c_sel, qt), d_def, boot(base_t, c_def, qt)))
        sel[mode] = {"val_grid": {f"{a},{k}": round(v, 4) for (a, k), v in gv_.items()}, "val_baseline_r1": float(100 * base_v.mean()),
                     "selected": list(best), "val_delta_selected": round(gv_[best], 4), "val_delta_default": round(gv_[(0.5, 10)], 4),
                     "test_delta_selected": round(d_sel, 4), "test_ci_selected": boot(base_t, c_sel, qt),
                     "test_delta_default": round(d_def, 4), "test_ci_default": boot(base_t, c_def, qt)}
    R["cuhk_validation_selection"] = sel
    # ---------------- post-hoc TEST grids, all datasets (SENSITIVITY ONLY) ----------------
    log("\n=== POST-HOC TEST GRIDS (sensitivity upper bound only; NOT results) ===")
    R["posthoc_test_grid"] = {}
    for ds in ("cuhkpedes", "icfgpedes", "aeri", "tbapr"):
        d, m = latest(ds, "test"); S = np.load(d + "/sim_qg.npy", mmap_mode="r"); q = np.asarray(m["query_pid"]); g = np.asarray(m["gallery_pid"])
        ent = {}
        for mode in ("transductive", "inductive"):
            blk = trans_block(S) if mode == "transductive" else ind_block(d, S.shape[1])
            _, gg = grid(S, blk, q, g); best = pick(gg)
            log("  %-10s %-12s best test (α,k)=%s Δ=%+.2f | default Δ=%+.2f | #(α,k) with Δ>0: %d/%d | min Δ=%+.2f"
                % (ds, mode, best, gg[best], gg[(0.5, 10)], sum(v > 0 for v in gg.values()), len(gg), min(gg.values())))
            ent[mode] = {"grid": {f"{a},{k}": round(v, 4) for (a, k), v in gg.items()}, "best": list(best), "best_delta": round(gg[best], 4),
                         "default_delta": round(gg[(0.5, 10)], 4), "n_positive": int(sum(v > 0 for v in gg.values()))}
        R["posthoc_test_grid"][ds] = ent
    R["_meta"] = {"peak_rss_gb": peak_rss_gb(), "elapsed_sec": round(time.time() - t0, 1)}
    R["_meta"]["provenance"] = provenance(args, input_paths, {"bootstrap": SEED, "run_seeds": {m["dataset"] + ":" + m["split"]: int(m["seed"]) for _, m in selected}})
    result_path = output_name(OUT, "S7c_valselect", args.tag, ".json")
    write_json(result_path, R)
    log("\npeak RSS %.2f GB, elapsed %ss; wrote %s" % (logged_rss(R["_meta"]["peak_rss_gb"]), R["_meta"]["elapsed_sec"], result_path))
    return R


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"S7c_valselect: {error}") from error
