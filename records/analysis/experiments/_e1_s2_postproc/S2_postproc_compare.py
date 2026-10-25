#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2  推理后处理方法离线对照实验
================================

输入：S1（--dump_sim）导出的 sim_qg.npy / sim_gg.npy
输出：五种后处理方法在 R@1 / R@5 / R@10 / mAP 上的对照，
      以及 CSLS 相对无校正的配对 bootstrap 显著性检验。

全程 CPU、numpy，不占 GPU。整套跑完约 3~15 分钟（取决于是否开 k-reciprocal）。

用法
----
python S2_postproc_compare.py \
    --sim_qg /root/autodl-tmp/experiments/_dump/xxx_l14_dumpsim/sim_qg.npy \
    --sim_gg /root/autodl-tmp/experiments/_dump/xxx_l14_dumpsim/sim_gg.npy \
    --ann    /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
    --out    /root/autodl-tmp/experiments/postproc_compare.json

方法说明
--------
none        无后处理，直接按相似度排序
csls        跨域相似度局部缩放，s' = 2s(t,g) - r(g)，r(g) 取该图库列上查询相似度的前 k 均值
zscore      图库列标准化，s' = (s - mu_g) / sigma_g，最朴素的 hubness 抑制基线
invsoftmax  逆 softmax，s' = beta*s - logsumexp_t(beta*s(.,g))，Smith 等 2017
alphaqe     alpha 查询扩展，用 top-k 图库邻居的相似度加权扩展查询表示，需 sim_gg
krecip      k-reciprocal 重排序的图库端编码变体，需 sim_gg

关于 krecip 的诚实说明
----------------------
标准 k-reciprocal（Zhong 等 2017）要求查询与图库处于同一模态，
以便检验"互为近邻"。在文本-图像跨模态设定下，图库图像的近邻集合中
不可能出现文本查询，互近邻条件恒不成立，Jaccard 距离退化为常数——
这正是本项目此前观察到的增益为 0.00 的原因，属于任务设定导致的
结构性退化，而非实现缺陷。

本脚本实现的是一个可用的变体：仅在图库侧（同模态）构建 k-reciprocal
编码 V_g，再把查询的初始排序通过 V_g 扩展为查询侧编码 V_q，最后计算
Jaccard 距离。这是该方法在跨模态设定下的合理适配，可作为对照。
论文中应如实说明这一改动。
"""

import argparse
import json
import time

import numpy as np

# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_labels(ann_path):
    """从 test_reid.json 还原图库与查询的身份标签，顺序必须与评估流程一致。"""
    anns = json.load(open(ann_path))
    g_pid = np.array([r["id"] for r in anns], dtype=np.int64)
    g_view = np.array([r["cam_id"] for r in anns], dtype=np.int64)
    q_pid, q_view = [], []
    for r in anns:
        for _ in r["captions"]:
            q_pid.append(r["id"])
            q_view.append(r["cam_id"])
    return np.array(q_pid, dtype=np.int64), g_pid, np.array(q_view), g_view


def evaluate(score_fn, q_pid, g_pid, n_q, chunk=2048):
    """
    通用评估。score_fn(a, b) 返回 [a, b) 区间查询的 float32 打分矩阵。
    返回 (指标字典, 每查询 top-1 命中布尔数组, 每查询 AP 数组)。
    """
    hit1 = np.zeros(n_q, dtype=bool)
    hit5 = np.zeros(n_q, dtype=bool)
    hit10 = np.zeros(n_q, dtype=bool)
    aps = np.zeros(n_q, dtype=np.float64)

    for a in range(0, n_q, chunk):
        b = min(a + chunk, n_q)
        s = score_fn(a, b)                      # (m, n_g) float32
        order = np.argsort(-s, axis=1)          # 降序
        rel = g_pid[order] == q_pid[a:b, None]  # (m, n_g) bool

        hit1[a:b] = rel[:, 0]
        hit5[a:b] = rel[:, :5].any(axis=1)
        hit10[a:b] = rel[:, :10].any(axis=1)

        # AP：命中位置上的精确率均值
        csum = np.cumsum(rel, axis=1)
        ranks = np.arange(1, rel.shape[1] + 1, dtype=np.float64)
        prec = csum / ranks
        n_rel = rel.sum(axis=1)
        n_rel_safe = np.where(n_rel == 0, 1, n_rel)
        aps[a:b] = (prec * rel).sum(axis=1) / n_rel_safe

        del s, order, rel, csum, prec

    m = {
        "R@1": 100.0 * hit1.mean(),
        "R@5": 100.0 * hit5.mean(),
        "R@10": 100.0 * hit10.mean(),
        "mAP": 100.0 * aps.mean(),
    }
    return m, hit1, aps


# --------------------------------------------------------------------------- #
# 各后处理方法的列统计量
# --------------------------------------------------------------------------- #


def gallery_topk_mean(sim, k, col_chunk=512):
    """r(g)：每个图库列上，全部查询相似度的前 k 均值。CSLS 的校正项。"""
    n_g = sim.shape[1]
    r = np.empty(n_g, dtype=np.float32)
    for a in range(0, n_g, col_chunk):
        b = min(a + col_chunk, n_g)
        col = np.asarray(sim[:, a:b], dtype=np.float32)
        part = np.partition(col, -k, axis=0)[-k:, :]
        r[a:b] = part.mean(axis=0)
    return r


def gallery_moments(sim, col_chunk=512):
    n_g = sim.shape[1]
    mu = np.empty(n_g, dtype=np.float32)
    sd = np.empty(n_g, dtype=np.float32)
    for a in range(0, n_g, col_chunk):
        b = min(a + col_chunk, n_g)
        col = np.asarray(sim[:, a:b], dtype=np.float32)
        mu[a:b] = col.mean(axis=0)
        sd[a:b] = col.std(axis=0) + 1e-8
    return mu, sd


def gallery_logsumexp(sim, beta, col_chunk=512):
    n_g = sim.shape[1]
    lse = np.empty(n_g, dtype=np.float32)
    for a in range(0, n_g, col_chunk):
        b = min(a + col_chunk, n_g)
        col = np.asarray(sim[:, a:b], dtype=np.float32) * beta
        mx = col.max(axis=0)
        lse[a:b] = mx + np.log(np.exp(col - mx).sum(axis=0))
    return lse


# --------------------------------------------------------------------------- #
# k-reciprocal（图库端编码变体）
# --------------------------------------------------------------------------- #


def build_gallery_V(sim_gg, k1, lam_gg=0.3):
    """图库侧 k-reciprocal 编码，行 L1 归一化，返回 (n_g, n_g) float32。"""
    n_g = sim_gg.shape[0]
    S = np.asarray(sim_gg, dtype=np.float32).copy()
    np.fill_diagonal(S, np.inf)                 # 自身必为最近邻
    idx = np.argsort(-S, axis=1)[:, : k1 + 1]

    nb = [set(idx[i].tolist()) for i in range(n_g)]
    V = np.zeros((n_g, n_g), dtype=np.float32)
    np.fill_diagonal(S, -np.inf)

    for i in range(n_g):
        # 互为近邻的子集
        cand = [j for j in idx[i] if i in nb[j]]
        if not cand:
            cand = [i]
        # 局部扩展（Zhong 等的 half-k 扩展）
        expanded = set(cand)
        half = max(1, int(round(k1 / 2)))
        for j in cand:
            sub = [t for t in idx[j][: half + 1] if j in nb[t]]
            if len(set(sub) & set(cand)) > 2.0 / 3.0 * len(sub):
                expanded |= set(sub)
        e = np.array(sorted(expanded), dtype=np.int64)
        sv = S[i, e].copy()
        sv[~np.isfinite(sv)] = 1.0
        sv -= sv.max()
        w = np.exp(sv / max(lam_gg, 1e-6))
        V[i, e] = w / w.sum()
    return V


def krecip_scores(sim_q_block, V_g, k2):
    """
    查询侧：取初始排序前 k2 个图库项，用其 V_g 行的均值作为 V_q，
    再与所有图库项计算 Jaccard 相似度（1 - Jaccard 距离）。
    """
    top = np.argsort(-sim_q_block, axis=1)[:, :k2]
    V_q = V_g[top].mean(axis=1)                            # (m, n_g)
    inter = V_q @ V_g.T                                     # 近似 sum(min)
    su = V_q.sum(axis=1)[:, None] + V_g.sum(axis=1)[None, :]
    return inter / (su - inter + 1e-12)


# --------------------------------------------------------------------------- #
# 显著性检验
# --------------------------------------------------------------------------- #


def paired_bootstrap(hit_a, hit_b, n_boot=10000, seed=0):
    """对查询做配对 bootstrap，返回 (ΔR@1 均值, 2.5% 分位, 97.5% 分位, p 值)。"""
    rng = np.random.default_rng(seed)
    d = hit_b.astype(np.float64) - hit_a.astype(np.float64)
    n = d.shape[0]
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    boots *= 100.0
    obs = 100.0 * d.mean()
    p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
    return obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5), p


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_qg", required=True)
    ap.add_argument("--sim_gg", default=None, help="不给则跳过 alphaqe 与 krecip")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="postproc_compare.json")
    ap.add_argument("--csls_k", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--is_beta", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
    ap.add_argument("--qe_k", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--qe_alpha", type=float, default=3.0)
    ap.add_argument("--kr_k1", type=int, default=20)
    ap.add_argument("--kr_k2", type=int, default=6)
    ap.add_argument("--kr_lambda", type=float, default=0.3)
    ap.add_argument("--with_kr", action="store_true")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    log("载入相似度矩阵（mmap，不整体入内存）")
    sim = np.load(args.sim_qg, mmap_mode="r")
    n_q, n_g = sim.shape
    log(f"sim_qg: {sim.shape} dtype={sim.dtype}")

    q_pid, g_pid, q_view, g_view = load_labels(args.ann)
    assert len(q_pid) == n_q, f"查询数不符：标签 {len(q_pid)} vs 矩阵 {n_q}"
    assert len(g_pid) == n_g, f"图库数不符：标签 {len(g_pid)} vs 矩阵 {n_g}"
    log(f"标签对齐通过：{n_q} 查询 / {n_g} 图库 / {len(set(g_pid.tolist()))} 身份")

    results = {}

    # ---- 基线 ----
    log("评估 none")
    base_fn = lambda a, b: np.asarray(sim[a:b], dtype=np.float32)
    m_none, hit_none, ap_none = evaluate(base_fn, q_pid, g_pid, n_q, args.chunk)
    results["none"] = m_none
    log(f"  none  {m_none}")

    # ---- CSLS ----
    for k in args.csls_k:
        log(f"评估 csls k={k}")
        r = gallery_topk_mean(sim, k)
        fn = lambda a, b, r=r: 2.0 * np.asarray(sim[a:b], dtype=np.float32) - r
        m, hit, _ = evaluate(fn, q_pid, g_pid, n_q, args.chunk)
        results[f"csls_k{k}"] = m
        log(f"  csls_k{k}  {m}")
        if k == 10:
            obs, lo, hi, p = paired_bootstrap(hit_none, hit, args.n_boot)
            results["bootstrap_csls_k10_vs_none"] = {
                "delta_R@1": obs, "ci95_low": lo, "ci95_high": hi, "p_two_sided": p,
                "n_boot": args.n_boot,
            }
            log(f"  bootstrap ΔR@1 = {obs:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4g}")
            results["r_g_mean_transductive"] = float(r.mean())

    # ---- 图库列标准化 ----
    log("评估 zscore")
    mu, sd = gallery_moments(sim)
    fn = lambda a, b: (np.asarray(sim[a:b], dtype=np.float32) - mu) / sd
    m, _, _ = evaluate(fn, q_pid, g_pid, n_q, args.chunk)
    results["zscore"] = m
    log(f"  zscore  {m}")

    # ---- 逆 softmax ----
    for beta in args.is_beta:
        log(f"评估 invsoftmax beta={beta}")
        lse = gallery_logsumexp(sim, beta)
        fn = lambda a, b, beta=beta, lse=lse: beta * np.asarray(sim[a:b], dtype=np.float32) - lse
        m, _, _ = evaluate(fn, q_pid, g_pid, n_q, args.chunk)
        results[f"invsoftmax_b{beta}"] = m
        log(f"  invsoftmax_b{beta}  {m}")

    # ---- 需要 sim_gg 的方法 ----
    if args.sim_gg:
        log("载入 sim_gg")
        sgg = np.asarray(np.load(args.sim_gg), dtype=np.float32)
        results["sim_stats"] = {
            "sim_qg_mean": float(np.asarray(sim[::37], dtype=np.float32).mean()),
            "sim_gg_mean": float(sgg.mean()),
            "sim_gg_offdiag_mean": float((sgg.sum() - np.trace(sgg)) / (n_g * n_g - n_g)),
        }
        log(f"  跨模态与同模态相似度均值：{results['sim_stats']}")

        # 归纳式 CSLS（论文 5.3 节的失效边界，此处复算以便写进对照表）
        for k in [5, 10, 20]:
            S = sgg.copy()
            np.fill_diagonal(S, -np.inf)
            r_ind = np.partition(S, -k, axis=0)[-k:, :].mean(axis=0)
            fn = lambda a, b, r=r_ind: 2.0 * np.asarray(sim[a:b], dtype=np.float32) - r
            m, _, _ = evaluate(fn, q_pid, g_pid, n_q, args.chunk)
            results[f"csls_inductive_k{k}"] = m
            log(f"  csls_inductive_k{k}  {m}  r_ind_mean={r_ind.mean():.4f}")
            if k == 10:
                results["r_g_mean_inductive"] = float(r_ind.mean())

        # alpha-QE
        for k in args.qe_k:
            log(f"评估 alphaqe k={k}")

            def fn(a, b, k=k):
                s = np.asarray(sim[a:b], dtype=np.float32)
                top = np.argsort(-s, axis=1)[:, :k]
                w = np.take_along_axis(s, top, axis=1) ** args.qe_alpha
                w /= w.sum(axis=1, keepdims=True) + 1e-12
                agg = np.einsum("mk,mkg->mg", w, sgg[top])
                return s + agg

            m, _, _ = evaluate(fn, q_pid, g_pid, n_q, args.chunk // 4)
            results[f"alphaqe_k{k}"] = m
            log(f"  alphaqe_k{k}  {m}")

        # k-reciprocal 图库端编码变体
        if args.with_kr:
            log(f"构建图库 k-reciprocal 编码 k1={args.kr_k1}（较慢，约几分钟）")
            V_g = build_gallery_V(sgg, args.kr_k1, args.kr_lambda)
            log("评估 krecip")

            def fn_kr(a, b):
                s = np.asarray(sim[a:b], dtype=np.float32)
                j = krecip_scores(s, V_g, args.kr_k2)
                return 0.5 * s + 0.5 * j     # 与原始相似度等权融合

            m, _, _ = evaluate(fn_kr, q_pid, g_pid, n_q, 512)
            results[f"krecip_k1{args.kr_k1}_k2{args.kr_k2}"] = m
            log(f"  krecip  {m}")

    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    log(f"结果已写入 {args.out}")

    # ---- 打印可直接贴进论文的表 ----
    print("\n" + "=" * 78)
    print(f"{'方法':<28}{'R@1':>10}{'R@5':>10}{'R@10':>10}{'mAP':>10}")
    print("-" * 78)
    for name, v in results.items():
        if isinstance(v, dict) and "R@1" in v:
            print(f"{name:<28}{v['R@1']:>10.2f}{v['R@5']:>10.2f}"
                  f"{v['R@10']:>10.2f}{v['mAP']:>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
