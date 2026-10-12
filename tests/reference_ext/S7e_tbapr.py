import os, json, glob, re, resource, time, numpy as np
from collections import defaultdict
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7e"; DS = "/root/autodl-tmp/datasets/TBAGPR"
ALPHA, K, NBOOT, SEED = 0.5, 10, 10000, 2026
t0 = time.time()
def log(*a): print(*a, flush=True)
d = sorted([x for x in glob.glob(ROOT + "/tbapr_*") if os.path.exists(x + "/checkpoint_last.pth")])[-1]
m = json.load(open(d + "/meta.json")); log("run:", os.path.basename(d))
S = np.load(d + "/sim_qg.npy"); qpid = np.asarray(m["query_pid"]); gpid = np.asarray(m["gallery_pid"])
qsub = np.asarray(m["query_source_subset"]); gsub = np.asarray(m["gallery_source_subset"]); gpaths = m["gallery_paths"]
r_ind_full = np.load("/root/autodl-tmp/_e_ext/s7a/tbapr_r_inductive.npy")
# query texts in official merged order (AG then G2), same order the adapter used
ag = json.load(open(DS + "/AG-ReID/agtext_test.json")); g2 = json.load(open(DS + "/G2APS-AG/g2aps_test.json"))
texts = list(ag["captions"]) + list(g2["captions"]); assert len(texts) == S.shape[0] == 3200
norm = lambda t: re.sub(r"\s+", " ", t.strip().lower())
# duplicate / conflicting gallery paths
by = defaultdict(list)
for i, p in enumerate(gpaths): by[p].append(i)
dup_rows = set(i for ix in by.values() if len(ix) > 1 for i in ix[1:])                       # non-first duplicates
conflict_rows = set(i for ix in by.values() if len(set(gpid[j] for j in ix)) > 1 for i in ix)  # all rows of conflicting paths
log("gallery rows=%d unique paths=%d duplicate(non-first) rows=%d rows in conflicting paths=%d" % (len(gpaths), len(by), len(dup_rows), len(conflict_rows)))

def r_trans(Ssub): top = np.sort(np.partition(Ssub, Ssub.shape[0] - K, axis=0)[-K:], axis=0); return top.astype(np.float64).mean(0).astype(np.float32)
def r1(sc, qp, gp): return gp[sc.argmax(1)] == qp
def boot(cb, cn, cl):
    uniq, inv = np.unique(cl, return_inverse=True); n = np.bincount(inv).astype(float); sb = np.bincount(inv, weights=cb.astype(float)); sn = np.bincount(inv, weights=cn.astype(float))
    rng = np.random.default_rng(SEED); mm = len(uniq); dl = np.array([100 * (sn[i].sum() - sb[i].sum()) / n[i].sum() for i in (rng.integers(0, mm, mm) for _ in range(NBOOT))])
    return [round(float(np.percentile(dl, 2.5)), 3), round(float(np.percentile(dl, 97.5)), 3)]
def variant(name, qmask, gmask):
    Ssub = S[np.ix_(qmask, gmask)]; qp = qpid[qmask]; gp = gpid[gmask]
    cb = r1(Ssub, qp, gp); ct = r1(Ssub - np.float32(ALPHA) * r_trans(Ssub)[None, :], qp, gp); ci = r1(Ssub - np.float32(ALPHA) * r_ind_full[gmask][None, :], qp, gp)
    row = {"nq": int(qmask.sum()), "ng": int(gmask.sum()), "n_ids": int(len(set(qp.tolist()))), "base_r1": round(100 * cb.mean(), 3),
           "trans_delta": round(100 * (ct.mean() - cb.mean()), 3), "trans_ci": boot(cb, ct, qp),
           "ind_delta": round(100 * (ci.mean() - cb.mean()), 3), "ind_ci": boot(cb, ci, qp)}
    log("  %-38s nq=%-5d ng=%-5d ids=%-4d base=%6.2f | transΔ=%+.2f %s%s | indΔ=%+.2f %s%s" % (name, row["nq"], row["ng"], row["n_ids"], row["base_r1"],
        row["trans_delta"], row["trans_ci"], "*" if row["trans_ci"][0] > 0 or row["trans_ci"][1] < 0 else "", row["ind_delta"], row["ind_ci"], "*" if row["ind_ci"][0] > 0 or row["ind_ci"][1] < 0 else ""))
    return row
R = {"run": os.path.basename(d), "variants": {}}
allq = np.ones(len(qpid), bool); allg = np.ones(len(gpid), bool)
log("\n=== TBAPR variants (α=%.2f k=%d; transductive r recomputed within each variant's queries×gallery; inductive r sliced) ===" % (ALPHA, K))
R["variants"]["official_merged"] = variant("official merged (MAIN)", allq, allg)
R["variants"]["AG-ReID_only"] = variant("AG-ReID only", qsub == "AG-ReID", gsub == "AG-ReID")
R["variants"]["G2APS_only"] = variant("G2APS-AG only", qsub == "G2APS-AG", gsub == "G2APS-AG")
# unique text: keep first occurrence of each normalised caption (within merged), and G2-only version
seen = set(); uniq_q = np.zeros(len(texts), bool)
for i, t in enumerate(texts):
    if norm(t) not in seen: seen.add(norm(t)); uniq_q[i] = True
R["variants"]["merged_unique_text"] = variant("merged, unique text (first kept)", uniq_q, allg)
R["variants"]["G2APS_unique_text"] = variant("G2APS only, unique text", uniq_q & (qsub == "G2APS-AG"), gsub == "G2APS-AG")
# gallery dedup
keep_first = np.array([i not in dup_rows for i in range(len(gpid))]); drop_conf = np.array([i not in conflict_rows for i in range(len(gpid))])
R["variants"]["merged_gallery_keepfirst"] = variant("merged, gallery dedup: keep first row", allq, keep_first)
R["variants"]["merged_gallery_dropconflict"] = variant("merged, gallery: drop conflicting rows", allq, drop_conf)
R["variants"]["merged_unique_text_and_dropconflict"] = variant("merged, unique text + drop conflicts", uniq_q, drop_conf)
# duplicate-text stats
cnt = defaultdict(set)
for i, t in enumerate(texts): cnt[norm(t)].add(int(qpid[i]))
R["duplicate_text_stats"] = {"queries": len(texts), "unique_texts": len(cnt), "texts_shared_by_multiple_identities": sum(len(v) > 1 for v in cnt.values())}
R["gallery_dup_stats"] = {"rows": len(gpaths), "unique_paths": len(by), "nonfirst_duplicate_rows": len(dup_rows), "rows_in_conflicting_paths": len(conflict_rows)}
R["_meta"] = {"alpha": ALPHA, "k": K, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "elapsed_sec": round(time.time() - t0, 1)}
json.dump(R, open(f"{OUT}/S7e_tbapr.json", "w"), indent=2)
log("\nduplicate text: %s | gallery: %s" % (R["duplicate_text_stats"], R["gallery_dup_stats"]))
log("peak RSS %.2f GB, elapsed %ss" % (R["_meta"]["peak_rss_gb"], R["_meta"]["elapsed_sec"]))
