import os, json, glob, resource, time, numpy as np
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7d"
ALPHA, K, NBOOT, SEED, CHUNK, NULL = 0.5, 10, 10000, 2026, 2048, 300
t0 = time.time()
def log(*a): print(*a, flush=True)
d = sorted([x for x in glob.glob(ROOT + "/aeri_*") if os.path.exists(x + "/checkpoint_last.pth")])[-1]
m = json.load(open(d + "/meta.json")); log("run:", os.path.basename(d))
tg, tl = np.load(d + "/feats_text_global.npy"), np.load(d + "/feats_text_local.npy")
ag, al = np.load(d + "/feats_img_global.npy"), np.load(d + "/feats_img_local.npy")
eg, el = np.load(d + "/feats_extra_img_global.npy"), np.load(d + "/feats_extra_img_local.npy")
em = json.load(open(d + "/extra_meta.json")); log("extra_meta keys:", list(em)[:10])
qpid = np.asarray(m["query_pid"]); apid = np.asarray(m["gallery_pid"])
gpid_key = next(k for k in em if "pid" in k.lower()); gpid_g = np.asarray(em[gpid_key])
log("queries=%d aerial=%d ground=%d | ground pids in query set: %.3f" % (len(qpid), len(apid), len(gpid_g), np.isin(gpid_g, qpid).mean()))
bg = np.load(d + "/feats_bank_text_global.npy", mmap_mode="r"); bl = np.load(d + "/feats_bank_text_local.npy", mmap_mode="r")

def sims(ig, il): return ((tg @ ig.T + tl @ il.T) / 2).astype(np.float32)
def r_trans(S): top = np.sort(np.partition(S, S.shape[0] - K, axis=0)[-K:], axis=0); return top.astype(np.float64).mean(0).astype(np.float32)
def r_ind(ig, il):
    top = np.full((K, ig.shape[0]), -np.inf, np.float32)
    for a in range(0, bg.shape[0], 4096):
        b = min(a + 4096, bg.shape[0]); blk = ((np.asarray(bg[a:b]) @ ig.T + np.asarray(bl[a:b]) @ il.T) / 2).astype(np.float32)
        comb = np.concatenate([top, blk], 0); top = np.partition(comb, comb.shape[0] - K, axis=0)[-K:]
    return np.sort(top, axis=0).astype(np.float64).mean(0).astype(np.float32)
def evaluate(S, gpid, r=None):
    nq, ng = S.shape; first = np.empty(nq, np.int32); ap = np.empty(nq); top1 = np.empty(nq, np.int64); ranks = np.arange(1, ng + 1)[None, :]
    for a in range(0, nq, CHUNK):
        b = min(a + CHUNK, nq); sc = S[a:b] - (np.float32(ALPHA) * r[None, :] if r is not None else 0)
        order = np.argsort(-sc, axis=1, kind="stable"); top1[a:b] = order[:, 0]; match = gpid[order] == qpid[a:b, None]
        first[a:b] = match.argmax(1); cum = np.cumsum(match, 1); ap[a:b] = ((cum / ranks) * match).sum(1) / np.maximum(match.sum(1), 1)
    return first, ap, top1
def metrics(f, ap): return {"r1": round(100 * (f < 1).mean(), 3), "r5": round(100 * (f < 5).mean(), 3), "r10": round(100 * (f < 10).mean(), 3), "mAP": round(100 * ap.mean(), 3)}
def skew(x): x = np.asarray(x, float); s = x.std(); return float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0
def ex_skew(S, top1):
    nq, ng = S.shape; nk = np.zeros(ng, np.int64); np.add.at(nk, np.argpartition(-S, K, axis=1)[:, :K].ravel(), 1)
    rng = np.random.default_rng(2026); null = np.mean([skew(np.bincount(rng.integers(0, ng, nq * K), minlength=ng)) for _ in range(NULL)])
    c = np.bincount(top1, minlength=ng); return round(skew(nk) - null, 4), round(float((c > 0).sum() / ng), 4), int(c.max())
def boot(cb, cn):
    uniq, inv = np.unique(qpid, return_inverse=True); n = np.bincount(inv).astype(float); sb = np.bincount(inv, weights=cb.astype(float)); sn = np.bincount(inv, weights=cn.astype(float))
    rng = np.random.default_rng(SEED); m_ = len(uniq); dl = np.array([100 * (sn[i].sum() - sb[i].sum()) / n[i].sum() for i in (rng.integers(0, m_, m_) for _ in range(NBOOT))])
    return [round(float(np.percentile(dl, 2.5)), 3), round(float(np.percentile(dl, 97.5)), 3)]

def protocol(name, ig, il, gpid, view):
    S = sims(ig, il); f0, ap0, t0_ = evaluate(S, gpid); mb = metrics(f0, ap0)
    rt = r_trans(S); f1, ap1, t1 = evaluate(S, gpid, rt); mt = metrics(f1, ap1)
    ri = r_ind(ig, il); f2, ap2, _ = evaluate(S, gpid, ri); mi = metrics(f2, ap2)
    es, dist, mx = ex_skew(S, t0_)
    wrong = f0 >= 1; ground_share_gallery = float((view == "ground").mean())
    top1_ground_all = float((view[t0_] == "ground").mean()); top1_ground_wrong = float((view[t0_[wrong]] == "ground").mean()) if wrong.any() else float("nan")
    top1_ground_right = float((view[t0_[~wrong]] == "ground").mean()) if (~wrong).any() else float("nan")
    log("  %-28s ng=%-5d base %s" % (name, S.shape[1], mb))
    log("  %-28s trans Δr1=%+.2f idCI=%s | ind Δr1=%+.2f idCI=%s | ex_skew=%.3f distinct=%.3f maxT1=%d" % (
        "", mt["r1"] - mb["r1"], boot(f0 < 1, f1 < 1), mi["r1"] - mb["r1"], boot(f0 < 1, f2 < 1), es, dist, mx))
    if 0 < ground_share_gallery < 1:
        log("  %-28s view: ground share of gallery=%.3f | of top-1 (all)=%.3f | of top-1 among CORRECT=%.3f | among WRONG=%.3f" % (
            "", ground_share_gallery, top1_ground_all, top1_ground_right, top1_ground_wrong))
    return {"ng": int(S.shape[1]), "baseline": mb, "transductive": mt, "trans_delta_r1": round(mt["r1"] - mb["r1"], 3), "trans_ci": boot(f0 < 1, f1 < 1),
            "inductive": mi, "ind_delta_r1": round(mi["r1"] - mb["r1"], 3), "ind_ci": boot(f0 < 1, f2 < 1),
            "excess_skew_N10": es, "distinct_top1": dist, "max_top1": mx, "ground_share_gallery": ground_share_gallery,
            "top1_ground_share_all": top1_ground_all, "top1_ground_share_correct": top1_ground_right, "top1_ground_share_wrong": top1_ground_wrong}

R = {"run": os.path.basename(d), "alpha": ALPHA, "k": K, "protocols": {}}
vA = np.array(["aerial"] * len(apid)); vG = np.array(["ground"] * len(gpid_g))
log("\n=== AERI protocols (text queries = 6141, identity-level; positives = same identity) ===")
R["protocols"]["T2A_official"] = protocol("T→A official (aerial 5525)", ag, al, apid, vA)
R["protocols"]["T2G_ground_only"] = protocol("T→G ground-only (4220)", eg, el, gpid_g, vG)
R["protocols"]["T2M_mixed"] = protocol("T→A+G mixed (9745)", np.concatenate([ag, eg]), np.concatenate([al, el]), np.concatenate([apid, gpid_g]), np.concatenate([vA, vG]))
# size-controlled aerial (4220) x 5 seeds, to compare with ground-only fairly
sub = []
for s in range(2026, 2031):
    idx = np.random.default_rng(s).choice(len(apid), 4220, replace=False)
    sub.append(protocol("T→A size-controlled (4220) s%d" % s, ag[idx], al[idx], apid[idx], vA[idx]))
R["protocols"]["T2A_size_controlled_4220"] = {"seeds": list(range(2026, 2031)), "runs": sub,
    "mean_r1": round(float(np.mean([x["baseline"]["r1"] for x in sub])), 3), "sd_r1": round(float(np.std([x["baseline"]["r1"] for x in sub])), 3),
    "mean_trans_delta": round(float(np.mean([x["trans_delta_r1"] for x in sub])), 3), "mean_ind_delta": round(float(np.mean([x["ind_delta_r1"] for x in sub])), 3)}
log("\n=== SUMMARY ===")
for k in ("T2A_official", "T2G_ground_only", "T2M_mixed"):
    p = R["protocols"][k]; log("  %-18s ng=%-5d R@1=%6.2f mAP=%6.2f | transΔ=%+.2f %s | indΔ=%+.2f %s | ex_skew=%.3f" % (
        k, p["ng"], p["baseline"]["r1"], p["baseline"]["mAP"], p["trans_delta_r1"], p["trans_ci"], p["ind_delta_r1"], p["ind_ci"], p["excess_skew_N10"]))
sc = R["protocols"]["T2A_size_controlled_4220"]; log("  %-18s ng=4220  R@1=%6.2f±%.2f | transΔ=%+.2f | indΔ=%+.2f  (5 seeds)" % ("T2A_ctrl_4220", sc["mean_r1"], sc["sd_r1"], sc["mean_trans_delta"], sc["mean_ind_delta"]))
pm = R["protocols"]["T2M_mixed"]
log("  mixed-gallery view analysis: ground = %.1f%% of gallery, %.1f%% of all top-1, %.1f%% of CORRECT top-1, %.1f%% of WRONG top-1" % (
    100 * pm["ground_share_gallery"], 100 * pm["top1_ground_share_all"], 100 * pm["top1_ground_share_correct"], 100 * pm["top1_ground_share_wrong"]))
R["_meta"] = {"peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "elapsed_sec": round(time.time() - t0, 1),
              "caveat": "AERI test captions are identity-level; which view they were annotated from is NOT confirmed. Neither direction is labelled 'cross-view'."}
json.dump(R, open(f"{OUT}/S7d_aeri.json", "w"), indent=2); log("peak RSS %.2f GB, elapsed %ss" % (R["_meta"]["peak_rss_gb"], R["_meta"]["elapsed_sec"]))
