import os, sys, json, glob, hashlib, subprocess, time, numpy as np
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7a"
ALPHA, K, NBOOT, SEED, CHUNK, BANK_CHUNK = 0.5, 10, 10000, 2026, 2048, 4096

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


def sha16(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

def log(*a): print(*a, flush=True)

def run_dirs():
    return SELECTED_RUNS

def running_topk_mean(row_iter, ng):
    top = np.full((K, ng), -np.inf, np.float32)
    for blk in row_iter:
        comb = np.concatenate([top, blk], 0)
        top = np.partition(comb, comb.shape[0] - K, axis=0)[-K:]
    return top.mean(0).astype(np.float32)

def r_transductive(S):
    nq, ng = S.shape
    return running_topk_mean((np.asarray(S[a:min(a + CHUNK, nq)]) for a in range(0, nq, CHUNK)), ng)

def r_inductive(d, ng):
    bg = np.load(d + "/feats_bank_text_global.npy", mmap_mode="r"); bl = np.load(d + "/feats_bank_text_local.npy", mmap_mode="r")
    ig = np.load(d + "/feats_img_global.npy"); il = np.load(d + "/feats_img_local.npy")
    def it():
        for a in range(0, bg.shape[0], BANK_CHUNK):
            b = min(a + BANK_CHUNK, bg.shape[0])
            yield ((np.asarray(bg[a:b]) @ ig.T + np.asarray(bl[a:b]) @ il.T) / 2).astype(np.float32)
    return running_topk_mean(it(), ng), int(bg.shape[0])

def evaluate(S, r, qpid, gpid, tag):
    nq, ng = S.shape
    first = np.empty(nq, np.int32); ap = np.empty(nq, np.float64); top1 = np.empty(nq, np.int64)
    ranks = np.arange(1, ng + 1, dtype=np.float64)[None, :]
    for a in range(0, nq, CHUNK):
        b = min(a + CHUNK, nq); sc = np.asarray(S[a:b], dtype=np.float32)
        if r is not None: sc = sc - np.float32(ALPHA) * r[None, :]
        order = np.argsort(-sc, axis=1, kind="stable"); top1[a:b] = order[:, 0]
        match = gpid[order] == qpid[a:b, None]
        first[a:b] = match.argmax(1)
        cum = np.cumsum(match, 1, dtype=np.float64); npos = match.sum(1)
        ap[a:b] = ((cum / ranks) * match).sum(1) / np.maximum(npos, 1)
        if (a // CHUNK) % 4 == 0: log("    [%s] rows %d/%d" % (tag, b, nq))
    return first, ap, top1

def metrics(first, ap):
    return {"r1": float(100 * (first < 1).mean()), "r5": float(100 * (first < 5).mean()),
            "r10": float(100 * (first < 10).mean()), "mAP": float(100 * ap.mean())}

def hub(top1, ng):
    c = np.bincount(top1, minlength=ng)
    return {"distinct_top1_ratio": float((c > 0).sum() / ng), "max_top1_count": int(c.max())}

def boot(c_base, c_new, clusters):
    uniq, inv = np.unique(clusters, return_inverse=True)
    n = np.bincount(inv).astype(np.float64); sb = np.bincount(inv, weights=c_base); sn = np.bincount(inv, weights=c_new)
    rng = np.random.default_rng(SEED); m = len(uniq); dl = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = rng.integers(0, m, m); dl[i] = 100 * (sn[idx].sum() - sb[idx].sum()) / n[idx].sum()
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"ci95": [round(float(lo), 3), round(float(hi), 3)], "n_clusters": int(m), "excludes_zero": bool(lo > 0 or hi < 0)}

def main(argv=None):
    global ROOT, OUT, t0, SELECTED_RUNS
    parser = common_parser("S7a_csls", "S7a_csls: original external offline calculations, explicit runs")
    # Separate invocations/tags are required when a dataset has multiple seeds.
    args = parse_common(parser, argv)
    ROOT = args.root
    SELECTED_RUNS = select_runs(args)
    input_paths = [path for d, _ in SELECTED_RUNS for path in run_inputs(d, ("meta.json", "sim_qg.npy", "feats_bank_text_global.npy", "feats_bank_text_local.npy", "feats_img_global.npy", "feats_img_local.npy"))]
    OUT = result_directory(args)
    t0 = time.time()
    summary = {"alpha": ALPHA, "k": K, "nboot": NBOOT, "seed": SEED, "argv": sys.argv,
               "git_head": provenance(args, [], {})["git_HEAD"],
               "datasets": {}}
    for d, m in run_dirs():
        ds = m["dataset"]; log("\n" + "=" * 78 + "\n### %s  (%s)" % (ds, os.path.basename(d)))
        S = np.load(d + "/sim_qg.npy", mmap_mode="r"); nq, ng = S.shape
        qpid = np.asarray(m["query_pid"]); gpid = np.asarray(m["gallery_pid"])
        res = {"run": os.path.basename(d), "run_dir": str(Path(d).resolve()), "seed": int(m["seed"]), "nq": nq, "ng": ng, "artifacts": {},
               "inputs_sha16": {"sim_qg": sha16(d + "/sim_qg.npy"), "meta": sha16(d + "/meta.json")}}
        f0, ap0, t10 = evaluate(S, None, qpid, gpid, "base"); mb = metrics(f0, ap0); res["baseline"] = mb
        on = m.get("metrics", {}); diff = {k: round(mb[k] - float(on.get(k, np.nan)), 4) for k in mb if k in on}
        res["baseline_vs_online_diff"] = diff; res["baseline_hub"] = hub(t10, ng)
        log("  baseline offline: %s" % {k: round(v, 3) for k, v in mb.items()})
        log("  baseline online : %s" % {k: round(float(v), 3) for k, v in on.items()})
        log("  diff            : %s   %s" % (diff, "OK" if all(abs(v) < 0.02 for v in diff.values()) else "!! MISMATCH"))
        per_query = {"first_rank_base": f0, "ap_base": ap0, "top1_base": t10}
        for mode in ("transductive", "inductive"):
            if mode == "transductive": r = r_transductive(S); bank_n = nq
            else: r, bank_n = r_inductive(d, ng)
            f1, ap1, t11 = evaluate(S, r, qpid, gpid, mode[:5]); mc = metrics(f1, ap1)
            cb, cn = (f0 < 1).astype(np.float64), (f1 < 1).astype(np.float64)
            entry = {"metrics": mc, "delta": {k: round(mc[k] - mb[k], 3) for k in mc}, "bank_size": bank_n,
                     "boot_identity": boot(cb, cn, qpid), "boot_query": boot(cb, cn, np.arange(nq)), "hub_after": hub(t11, ng)}
            res[mode] = entry
            per_query["first_rank_" + mode] = f1; per_query["ap_" + mode] = ap1; per_query["top1_" + mode] = t11
            artifact = output_name(OUT, f"{ds}_r_{mode}", args.tag, ".npy")
            save_npy(artifact, r)
            res["artifacts"]["r_" + mode] = artifact
            log("  %-12s r1=%.2f (Δ%+.2f) r5=%.2f r10=%.2f mAP=%.2f | idCI %s%s | qCI %s | distinct %.3f→%.3f maxT1 %d→%d | bank=%d"
                % (mode, mc["r1"], entry["delta"]["r1"], mc["r5"], mc["r10"], mc["mAP"],
                   entry["boot_identity"]["ci95"], "*" if entry["boot_identity"]["excludes_zero"] else "",
                   entry["boot_query"]["ci95"], res["baseline_hub"]["distinct_top1_ratio"], entry["hub_after"]["distinct_top1_ratio"],
                   res["baseline_hub"]["max_top1_count"], entry["hub_after"]["max_top1_count"], bank_n))
        artifact = output_name(OUT, f"{ds}_per_query", args.tag, ".npz")
        save_npz(artifact, qpid=qpid, gpid=gpid, **per_query)
        res["artifacts"]["per_query"] = artifact
        summary["datasets"][ds] = res
    summary["peak_rss_gb"] = peak_rss_gb()
    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["provenance"] = provenance(args, input_paths, {"bootstrap": SEED, "run_seeds": {m["dataset"]: int(m["seed"]) for _, m in SELECTED_RUNS}})
    summary_path = output_name(OUT, "S7a_summary", args.tag, ".json")
    write_json(summary_path, summary)
    log("\n=== S7a SUMMARY (transductive ΔR@1, ordered) ===")
    for ds, r in sorted(summary["datasets"].items(), key=lambda kv: -kv[1]["transductive"]["delta"]["r1"]):
        log("  %-10s base=%6.2f  trans Δ=%+.2f %s  ind Δ=%+.2f %s"
            % (ds, r["baseline"]["r1"], r["transductive"]["delta"]["r1"], r["transductive"]["boot_identity"]["ci95"],
               r["inductive"]["delta"]["r1"], r["inductive"]["boot_identity"]["ci95"]))
    rss_status = ("UNAVAILABLE on this platform" if summary["peak_rss_gb"] is None else
                  "OK" if summary["peak_rss_gb"] < 8 else "!! EXCEEDS 8GB LIMIT")
    log("peak RSS %.2f GB  (%s)  elapsed %ss" % (logged_rss(summary["peak_rss_gb"]), rss_status, summary["elapsed_sec"]))
    log("wrote", summary_path)
    return summary


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        raise SystemExit(f"S7a_csls: {error}") from error
