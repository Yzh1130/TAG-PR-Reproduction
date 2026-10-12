import os, json, glob, numpy as np
from collections import Counter, defaultdict
ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"; OUT = "/root/autodl-tmp/_e_ext/s7a"
K = 10; CHUNK = 2048
def log(*a): print(*a, flush=True)
def run_dirs():
    o = []
    for d in sorted(glob.glob(ROOT + "/*")):
        if not os.path.exists(d + "/meta.json"): continue
        m = json.load(open(d + "/meta.json"))
        if m.get("epoch") == 10 and m["split"] == "test": o.append((d, m))
    return o
R = {}
for d, m in run_dirs():
    ds = m["dataset"]; log("\n" + "=" * 78 + "\n### " + ds)
    S = np.load(d + "/sim_qg.npy", mmap_mode="r"); nq, ng = S.shape
    qpid = np.asarray(m["query_pid"]); gpid = np.asarray(m["gallery_pid"])
    pq = np.load(f"{OUT}/{ds}_per_query.npz")
    cb = pq["first_rank_base"] < 1; ct = pq["first_rank_transductive"] < 1; ci = pq["first_rank_inductive"] < 1
    t1b, t1t = pq["top1_base"], pq["top1_transductive"]
    # ---- 1. duplicate gallery paths & pid consistency ----
    paths = m.get("gallery_paths") or []
    dup_info = {}
    if paths:
        idx_by = defaultdict(list)
        for i, p in enumerate(paths): idx_by[p].append(i)
        dups = {p: ix for p, ix in idx_by.items() if len(ix) > 1}
        conflict = {p: sorted(set(int(gpid[i]) for i in ix)) for p, ix in dups.items() if len(set(gpid[i] for i in ix)) > 1}
        dup_info = {"dup_paths": len(dups), "dup_rows": sum(len(v) for v in dups.values()) - len(dups), "conflicting_pid_paths": len(conflict)}
        log("  gallery duplicates: %s  conflict examples: %s" % (dup_info, list(conflict.items())[:3]))
    # ---- 2. offline vs online top-1 (baseline) ----
    oj = d + "/output.json"
    if os.path.exists(oj):
        o = json.load(open(oj))
        log("  output.json type=%s len=%s first=%s" % (type(o).__name__, len(o) if hasattr(o, "__len__") else "?",
            str(o[0] if isinstance(o, list) else list(o.items())[0])[:200]))
    # ---- 3. flips ----
    def flips(c0, c1, t0, t1, name):
        lost = (c0 & ~c1); gained = (~c0 & c1)
        cnt_b = np.bincount(t0, minlength=ng)
        same_id_after = (gpid[t1[lost]] == gpid[t0[lost]]).mean() if lost.any() else float("nan")
        hub_of_lost = cnt_b[t0[lost]].mean() if lost.any() else float("nan")
        hub_of_gained_old = cnt_b[t0[gained]].mean() if gained.any() else float("nan")
        log("  %-12s lost=%d gained=%d net=%+d | lost: old-top1 hubcount=%.1f, new top1 same identity as old=%.2f | gained: old-top1 hubcount=%.1f"
            % (name, lost.sum(), gained.sum(), gained.sum() - lost.sum(), hub_of_lost, same_id_after, hub_of_gained_old))
        return {"lost": int(lost.sum()), "gained": int(gained.sum()), "lost_old_top1_hubcount": float(hub_of_lost),
                "lost_new_top1_same_identity": float(same_id_after), "gained_old_top1_hubcount": float(hub_of_gained_old)}
    fl_t = flips(cb, ct, t1b, t1t, "transductive"); fl_i = flips(cb, ci, t1b, pq["top1_inductive"], "inductive")
    # ---- 4. hub purity & neighbourhood legitimacy ----
    cnt = np.bincount(t1b, minlength=ng)
    attracted_correct = np.zeros(ng); np.add.at(attracted_correct, t1b, (gpid[t1b] == qpid).astype(float))
    hubs = cnt >= 5
    purity_hubs = attracted_correct[hubs].sum() / cnt[hubs].sum() if hubs.any() else float("nan")
    purity_all = attracted_correct.sum() / cnt.sum()
    # legit share of each gallery item's top-K query neighbourhood (the queries that define r(g))
    top = np.full((K, ng), -np.inf, np.float32); topi = np.zeros((K, ng), np.int64)
    for a in range(0, nq, CHUNK):
        b = min(a + CHUNK, nq); blk = np.asarray(S[a:b])
        comb = np.concatenate([top, blk], 0); combi = np.concatenate([topi, np.broadcast_to(np.arange(a, b)[:, None], blk.shape)], 0)
        part = np.argpartition(comb, comb.shape[0] - K, axis=0)[-K:]
        top = np.take_along_axis(comb, part, 0); topi = np.take_along_axis(combi, part, 0)
    legit = (qpid[topi] == gpid[None, :]).mean(0)          # per gallery item: fraction of its k-NN queries that are positives
    r = top.mean(0)
    imgs_per_id = ng / len(set(gpid.tolist())); q_per_id = nq / len(set(qpid.tolist()))
    corr_r_legit = float(np.corrcoef(r, legit)[0, 1])
    log("  structure: imgs/identity=%.2f queries/identity=%.2f nq/ng=%.2f" % (imgs_per_id, q_per_id, nq / ng))
    log("  hub purity: all top1=%.3f | hubs(count>=5, n=%d)=%.3f" % (purity_all, hubs.sum(), purity_hubs))
    log("  kNN legitimacy: mean legit share of r(g) neighbourhood=%.3f | corr(r(g), legit share)=%+.3f" % (legit.mean(), corr_r_legit))
    log("  r(g) by legitimacy: legit>=0.5 -> mean r=%.4f (n=%d) | legit<0.5 -> mean r=%.4f (n=%d)"
        % (r[legit >= .5].mean(), (legit >= .5).sum(), r[legit < .5].mean(), (legit < .5).sum()))
    R[ds] = {"dup": dup_info, "flip_trans": fl_t, "flip_ind": fl_i, "imgs_per_identity": imgs_per_id, "queries_per_identity": q_per_id,
             "hub_purity_all": float(purity_all), "hub_purity_hubs": float(purity_hubs), "n_hubs_ge5": int(hubs.sum()),
             "knn_legit_share_mean": float(legit.mean()), "corr_r_legit": corr_r_legit,
             "trans_delta_r1": float(100 * (ct.mean() - cb.mean())), "ind_delta_r1": float(100 * (ci.mean() - cb.mean()))}
json.dump(R, open(f"{OUT}/S7diag.json", "w"), indent=2)
log("\n=== cross-dataset view ===")
log("  %-10s %8s %8s %9s %9s %9s %9s" % ("dataset", "transΔ", "indΔ", "img/id", "purity", "legitNN", "corr"))
for ds, v in sorted(R.items(), key=lambda kv: -kv[1]["trans_delta_r1"]):
    log("  %-10s %+8.2f %+8.2f %9.2f %9.3f %9.3f %+9.3f" % (ds, v["trans_delta_r1"], v["ind_delta_r1"], v["imgs_per_identity"],
        v["hub_purity_all"], v["knn_legit_share_mean"], v["corr_r_legit"]))
log("wrote", f"{OUT}/S7diag.json")
