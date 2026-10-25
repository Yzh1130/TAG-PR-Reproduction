#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only stratified analysis of alpha query expansion (paper Table 8, alphaqe_k2) on model 1.
Reproduces scripts/S2_postproc_compare.py exactly (alpha=3, score = s + weighted mean of sim_gg rows of top-k),
then splits queries by whether the baseline top-1 is correct. CPU / numpy only. Writes one JSON to /tmp."""
import json, sys, time
import numpy as np

SIM_QG = '/root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy'
SIM_GG = '/root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_gg.npy'
ANN = '/root/autodl-tmp/datasets/TAG-PEDES/test_reid.json'
OUT = '/tmp/tagpr_q3_alphaqe_%s.json' % time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
QE_ALPHA = 3.0
CHUNK = 512   # == S2 default chunk 2048 // 4 used for alphaqe


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def load_labels(ann_path):
    anns = json.load(open(ann_path))
    g_pid = np.array([r['id'] for r in anns], dtype=np.int64)
    q_pid = np.array([r['id'] for r in anns for _ in r['captions']], dtype=np.int64)
    return q_pid, g_pid


def per_query(score_fn, q_pid, g_pid, n_q, chunk):
    """Return dict of per-query arrays: hit1, hit5, hit10, ap, rank1 (1-based rank of first positive), nrel."""
    hit1 = np.zeros(n_q, bool); hit5 = np.zeros(n_q, bool); hit10 = np.zeros(n_q, bool)
    ap = np.zeros(n_q, np.float64); rank1 = np.zeros(n_q, np.int64); nrel = np.zeros(n_q, np.int64)
    top10 = np.zeros((n_q, 10), np.int64)
    for a in range(0, n_q, chunk):
        b = min(a + chunk, n_q)
        s = score_fn(a, b)
        order = np.argsort(-s, axis=1)
        rel = g_pid[order] == q_pid[a:b, None]
        hit1[a:b] = rel[:, 0]; hit5[a:b] = rel[:, :5].any(1); hit10[a:b] = rel[:, :10].any(1)
        csum = np.cumsum(rel, axis=1)
        ranks = np.arange(1, rel.shape[1] + 1, dtype=np.float64)
        n_rel = rel.sum(1)
        ap[a:b] = (csum / ranks * rel).sum(1) / np.where(n_rel == 0, 1, n_rel)
        nrel[a:b] = n_rel
        rank1[a:b] = rel.argmax(1) + 1
        top10[a:b] = order[:, :10]
        del s, order, rel, csum
    return dict(hit1=hit1, hit5=hit5, hit10=hit10, ap=ap, rank1=rank1, nrel=nrel, top10=top10)


def metrics(d, mask=None):
    if mask is None:
        mask = np.ones(d['hit1'].shape[0], bool)
    n = int(mask.sum())
    return {'n': n, 'R@1': float(100.0 * d['hit1'][mask].mean()), 'R@5': float(100.0 * d['hit5'][mask].mean()),
            'R@10': float(100.0 * d['hit10'][mask].mean()), 'mAP': float(100.0 * d['ap'][mask].mean()),
            'median_rank_first_pos': float(np.median(d['rank1'][mask]))}


def main():
    log('loading')
    sim = np.load(SIM_QG, mmap_mode='r')
    n_q, n_g = sim.shape
    q_pid, g_pid = load_labels(ANN)
    assert len(q_pid) == n_q and len(g_pid) == n_g
    sgg = np.asarray(np.load(SIM_GG), dtype=np.float32)
    log('sim_qg %s  sim_gg %s  queries %d gallery %d ids %d' % (sim.shape, sgg.shape, n_q, n_g, len(set(g_pid.tolist()))))

    base_fn = lambda a, b: np.asarray(sim[a:b], dtype=np.float32)

    def qe_fn_factory(k, with_s=True):
        def fn(a, b):
            s = np.asarray(sim[a:b], dtype=np.float32)
            top = np.argsort(-s, axis=1)[:, :k]
            w = np.take_along_axis(s, top, axis=1) ** QE_ALPHA
            w /= w.sum(axis=1, keepdims=True) + 1e-12
            agg = np.einsum('mk,mkg->mg', w, sgg[top])
            return (s + agg) if with_s else agg
        return fn

    log('baseline')
    B = per_query(base_fn, q_pid, g_pid, n_q, CHUNK)
    log('alphaqe k=2')
    Q2 = per_query(qe_fn_factory(2), q_pid, g_pid, n_q, CHUNK)
    log('agg-only k=2 (image-image term alone, no text term)')
    A2 = per_query(qe_fn_factory(2, with_s=False), q_pid, g_pid, n_q, CHUNK)
    log('alphaqe k=3')
    Q3 = per_query(qe_fn_factory(3), q_pid, g_pid, n_q, CHUNK)

    out = {'inputs': {'sim_qg': SIM_QG, 'sim_gg': SIM_GG, 'ann': ANN, 'qe_alpha': QE_ALPHA, 'chunk': CHUNK},
           'overall': {'none': metrics(B), 'alphaqe_k2': metrics(Q2), 'agg_only_k2': metrics(A2), 'alphaqe_k3': metrics(Q3)}}
    log('overall none      %s' % out['overall']['none'])
    log('overall alphaqe2  %s' % out['overall']['alphaqe_k2'])
    log('overall agg-only2 %s' % out['overall']['agg_only_k2'])
    log('overall alphaqe3  %s' % out['overall']['alphaqe_k3'])
    ref = {'R@1': 63.60882210373243, 'R@5': 73.50339311682016, 'R@10': 77.43577314590402, 'mAP': 55.239096385845656}
    maxdiff = max(abs(out['overall']['alphaqe_k2'][m] - ref[m]) for m in ref)
    out['reproduction_check'] = {'reference_postproc_fk32_json': ref, 'max_abs_diff': maxdiff, 'pass': bool(maxdiff < 1e-6)}
    log('reproduction check vs postproc_fk32.json alphaqe_k2: max abs diff = %.3e  %s' % (maxdiff, 'PASS' if maxdiff < 1e-6 else 'FAIL'))

    # strata by baseline top-1 correctness and by baseline rank of first positive
    r = B['rank1']
    strata = {
        'C_top1_correct': B['hit1'],
        'W_top1_wrong': ~B['hit1'],
        'W_rank2to5': (~B['hit1']) & (r <= 5),
        'W_rank6to10': (r >= 6) & (r <= 10),
        'W_rank_gt10': r > 10,
    }
    out['strata'] = {}
    for name, m in strata.items():
        out['strata'][name] = {'none': metrics(B, m), 'alphaqe_k2': metrics(Q2, m), 'agg_only_k2': metrics(A2, m), 'alphaqe_k3': metrics(Q3, m)}
    # contributions of each stratum to the overall change (percentage points of the overall metric)
    contrib = {}
    for name in ('C_top1_correct', 'W_top1_wrong'):
        m = strata[name]
        contrib[name] = {}
        for met, key in (('R@1', 'hit1'), ('R@5', 'hit5'), ('R@10', 'hit10'), ('mAP', 'ap')):
            d = Q2[key][m].astype(np.float64) - B[key][m].astype(np.float64)
            contrib[name][met] = 100.0 * d.sum() / n_q
    out['contribution_to_overall_delta_pp'] = contrib
    # transitions
    W = ~B['hit1']; C = B['hit1']
    out['transitions'] = {
        'C_lost_hit1': int((C & ~Q2['hit1']).sum()),
        'W_gained_hit1': int((W & Q2['hit1']).sum()),
        'W_lost_hit5': int((W & B['hit5'] & ~Q2['hit5']).sum()),
        'W_gained_hit5': int((W & ~B['hit5'] & Q2['hit5']).sum()),
        'W_lost_hit10': int((W & B['hit10'] & ~Q2['hit10']).sum()),
        'W_gained_hit10': int((W & ~B['hit10'] & Q2['hit10']).sum()),
        'C_lost_hit5': int((C & B['hit5'] & ~Q2['hit5']).sum()),
        'C_lost_hit10': int((C & B['hit10'] & ~Q2['hit10']).sum()),
        'all_lost_hit5': int((B['hit5'] & ~Q2['hit5']).sum()),
        'all_gained_hit5': int((~B['hit5'] & Q2['hit5']).sum()),
        'all_lost_hit10': int((B['hit10'] & ~Q2['hit10']).sum()),
        'all_gained_hit10': int((~B['hit10'] & Q2['hit10']).sum()),
    }
    # how much the ranking is governed by the image-image term: top-10 overlap between (s+agg) and agg-only rankings
    inter = np.array([len(set(Q2['top10'][i]) & set(A2['top10'][i])) for i in range(n_q)])
    out['top10_overlap_qe_vs_aggonly'] = {'mean_overlap_of_10': float(inter.mean()), 'frac_identical_top10_sets': float((inter == 10).mean())}
    # positives per query and AP change in C by number of positives
    out['positives_per_query'] = {'mean': float(B['nrel'].mean()), 'median': float(np.median(B['nrel'])),
                                  'frac_nrel_ge2': float((B['nrel'] >= 2).mean())}
    # second-ranked image correctness (k=2 anchors): both anchors correct / one / none, by stratum
    def anchor_stats(mask):
        t = B['top10'][mask][:, :2]
        ok = g_pid[t] == q_pid[mask][:, None]
        return {'n': int(mask.sum()), 'both_anchors_correct': float(ok.all(1).mean()), 'first_only': float((ok[:, 0] & ~ok[:, 1]).mean()),
                'second_only': float((~ok[:, 0] & ok[:, 1]).mean()), 'none': float((~ok).all(1).mean())}
    out['anchor_correctness'] = {'all': anchor_stats(np.ones(n_q, bool)), 'C': anchor_stats(C), 'W': anchor_stats(W)}
    # std of the two score terms (scale mismatch) on a subsample
    sub = np.asarray(sim[::37], dtype=np.float32)
    out['scale'] = {'sim_qg_std': float(sub.std()), 'sim_gg_std': float(sgg.std()), 'sim_qg_row_std_mean': float(sub.std(axis=1).mean())}

    json.dump(out, open(OUT, 'w'), indent=1, ensure_ascii=False, default=float)
    log('written ' + OUT)
    print()
    print('%-16s %6s %8s %8s %8s %8s | %8s %8s %8s %8s' % ('stratum', 'n', 'R@1', 'R@5', 'R@10', 'mAP', 'qe R@1', 'qe R@5', 'qe R@10', 'qe mAP'))
    for name in ('all',) + tuple(strata.keys()):
        a = out['overall']['none'] if name == 'all' else out['strata'][name]['none']
        b = out['overall']['alphaqe_k2'] if name == 'all' else out['strata'][name]['alphaqe_k2']
        print('%-16s %6d %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f %8.2f' % (name, a['n'], a['R@1'], a['R@5'], a['R@10'], a['mAP'], b['R@1'], b['R@5'], b['R@10'], b['mAP']))
    print('contribution to overall delta (pp):', json.dumps(contrib))
    print('transitions:', json.dumps(out['transitions']))
    print('top10 overlap qe vs agg-only:', json.dumps(out['top10_overlap_qe_vs_aggonly']))
    print('anchor correctness:', json.dumps(out['anchor_correctness']))
    print('agg-only overall:', json.dumps(out['overall']['agg_only_k2']))
    print('alphaqe_k3 overall:', json.dumps(out['overall']['alphaqe_k3']))
    print('positives per query:', json.dumps(out['positives_per_query']), ' scale:', json.dumps(out['scale']))


if __name__ == '__main__':
    main()
