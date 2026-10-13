#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表5零正例与抽样规则核查（CPU，只读归档，不改任何原始结果）。

输入：R1_table5_clusters.json（含每次抽样的保留索引）、test_reid.json（标注）。
做的事：
  1. 打印归档里被截断的 policy 字符串、日志、sim_meta.json；
  2. 从标注重建图库 pid/视角 与 查询→源图像 映射（自动识别字段并用已知计数校验）；
  3. 对 A/G 两侧的 original 与 10 次 authoritative_trials，按保留索引逐查询重算合法正样本数
     （同视角排除查询源图像），统计零正例、与归档 has_legal_positive 逐条对照；
  4. 核查抽样规则：子集是否只含目标视角、是否保留全部查询源图像、每身份保留的同视角/跨视角
     图像数下限、被删图像中“共同查询身份的图像”与“其他图像”的保留比例（判断是否均匀）；
  5. 由 correct_by_requested_query 重算每次抽样 R@1 及 10 次标准差（ddof=0/1）并打印 metrics；
  6. 打印仓库中 sample_gallery / common_query_sets / eligible_queries 的源码；
  7. 结果写入同目录 verify_zero_positive_recount.json（新文件）。
"""
import ast, collections, glob, hashlib, json, os, subprocess, sys
import numpy as np

ALL = sorted(glob.glob('/root/autodl-tmp/experiments/roundB_*/R1_table5_*/R1_table5_clusters.json'))
latest = {}
for j in ALL:
    latest[os.path.basename(os.path.dirname(j))] = j      # 同名目录取最新 roundB
print('== found table5 archives:', latest)


def run(J):
  d = json.load(open(J))
  OUT = os.path.join(os.path.dirname(J), 'verify_zero_positive_recount.json')
  print('\n\n######## JSON', J)
  print('== git_HEAD in archive:', d['provenance'].get('git_HEAD'), '| created', d['provenance'].get('created_utc'))

  print('\n== full policy strings')
  for k, v in d['zero_positive_original_policy'].items():
      print(f'  [{k}] {v}')
  for k, v in d['policies'].items():
      print(f'  [policies.{k}] {v}')
  print('  [positive_retention_warning]', d.get('positive_retention_warning'))
  print('  [definition]', d.get('definition'))

  logp = J[:-5] + '.log'
  if os.path.exists(logp):
      print('\n== R1_table5_clusters.log'); print(open(logp).read())

  simp = d['provenance']['arguments']['sim']
  metap = os.path.join(os.path.dirname(simp), 'sim_meta.json')
  print('== sim_meta.json', metap)
  print(open(metap).read() if os.path.exists(metap) else '  MISSING')

  s11p = d['provenance']['arguments'].get('s11_json')
  if s11p and os.path.exists(s11p):
      s11 = json.load(open(s11p))
      print('\n== S11 json', s11p, '| top-level keys:', list(s11.keys())[:40] if isinstance(s11, dict) else type(s11).__name__)

      def walk(o, path='', depth=0):
          if depth > 4:
              return
          if isinstance(o, dict):
              for k, v in o.items():
                  kp = f'{path}.{k}' if path else k
                  if isinstance(v, str) and any(t in (k + v).lower() for t in ('sampl', 'retain', 'preserve', 'policy', 'rule', 'uniform', 'stratif')):
                      print(f'   {kp} = {v[:300]}')
                  elif isinstance(v, (dict, list)):
                      walk(v, kp, depth + 1)
          elif isinstance(o, list) and o and isinstance(o[0], dict):
              walk(o[0], path + '[0]', depth + 1)
      walk(s11)

  # ---------------------------------------------------------------- annotation
  A = d['provenance']['arguments']['anno']
  sha = hashlib.sha256(open(A, 'rb').read()).hexdigest()[:16]
  print('\n== annotation', A, 'sha256_16', sha, '(archive says', d['provenance']['inputs']['annotation']['sha256_16'], ')')
  anno = json.load(open(A))
  ann = d['annotation']
  N_IMG, N_A, N_G, N_Q, N_ID = ann['n_gallery'], ann['aerial'], ann['ground'], ann['n_query'], ann['identities']


  def find_list(o):
      if isinstance(o, list):
          return o
      if isinstance(o, dict):
          for k in ('test', 'annotations', 'data', 'images', 'items'):
              if isinstance(o.get(k), list):
                  return o[k]
          for v in o.values():
              if isinstance(v, list) and v and isinstance(v[0], dict):
                  return v
      return None


  items = find_list(anno)
  print('   top-level type', type(anno).__name__, '| n_items', len(items) if items else None)
  if items is None:
      print('!! 无法识别标注结构，请把上面 top-level 信息发我'); return
  print('   item keys:', sorted(items[0].keys()))
  print('   example item:', {k: str(v)[:60] for k, v in items[0].items()})
  if len(items) != N_IMG:
      for sk in ('split', 'subset', 'set'):
          if sk in items[0]:
              cand = [it for it in items if str(it[sk]).lower() == 'test']
              print(f'   filtered by {sk}==test ->', len(cand))
              if len(cand) == N_IMG:
                  items = cand
                  break
  if len(items) != N_IMG:
      print(f'!! 图像数 {len(items)} != {N_IMG}，请把 item keys / example 发我'); return


  def nunique(k):
      try:
          return len(set(str(it[k]) for it in items))
      except Exception:
          return None


  pid_key = None
  for k in ('id', 'pid', 'person_id', 'identity', 'label', 'pid_label') + tuple(items[0].keys()):
      if k in items[0] and nunique(k) == N_ID:
          pid_key = k
          break
  view_key, view_map = None, None
  for k in items[0]:
      try:
          c = collections.Counter(str(it[k]) for it in items)
      except Exception:
          continue
      if len(c) == 2 and sorted(c.values()) == sorted([N_A, N_G]):
          view_key = k
          view_map = {v: (0 if n == N_A else 1) for v, n in c.items()}   # 0=aerial, 1=ground
          break
  if view_key is None:
      for k in items[0]:
          if not isinstance(items[0][k], str):
              continue
          for tok in ('aerial', 'drone', 'uav', '/a/', '_a_', 'air'):
              flags = [tok in it[k].lower() for it in items]
              if sum(flags) == N_A:
                  view_key, view_map = k, ('token', tok)
                  break
          if view_key:
              break
  cap_key = None
  for k in items[0]:
      if isinstance(items[0][k], list) and sum(len(it[k]) for it in items) == N_Q:
          cap_key = k
          break
  print(f'   detected pid_key={pid_key!r} view_key={view_key!r} view_map={view_map} cap_key={cap_key!r}')
  if None in (pid_key, view_key, cap_key):
      print('!! 字段自动识别失败，请把 item keys / example 发我'); return

  gal_pid = np.array([int(it[pid_key]) for it in items])
  if isinstance(view_map, dict):
      gal_view = np.array([view_map[str(it[view_key])] for it in items])
  else:
      gal_view = np.array([0 if view_map[1] in it[view_key].lower() else 1 for it in items])
  q_src = np.array([i for i, it in enumerate(items) for _ in it[cap_key]])
  print(f'   gallery: {len(gal_pid)} images, aerial={int((gal_view==0).sum())} ground={int((gal_view==1).sum())}, identities={len(set(gal_pid.tolist()))}, queries={len(q_src)}')
  dual = sum(1 for p in set(gal_pid.tolist()) if (gal_view[gal_pid == p] == 0).any() and (gal_view[gal_pid == p] == 1).any())
  print(f'   dual-view identities={dual} (archive {ann["dual_view_identities"]})')
  n_per = collections.Counter(len(it[cap_key]) for it in items)
  print('   captions per image distribution:', dict(n_per))

  # ---------------------------------------------------------------- recount
  view_of = {'A': 0, 'G': 1}
  result = {'json': J, 'json_sha256_16': hashlib.sha256(open(J, 'rb').read()).hexdigest()[:16],
            'annotation': A, 'annotation_sha256_16': sha, 'pid_key': pid_key, 'view_key': view_key,
            'cap_key': cap_key, 'groups': {}}


  def count_pos(gidx, view, qpid, src, exclude_src):
      gidx = np.asarray(gidx)
      sub_pid, sub_view = gal_pid[gidx], gal_view[gidx]
      wrong_view = int((sub_view != view).sum())
      inset = np.zeros(N_IMG, bool); inset[gidx] = True
      pc = collections.Counter(sub_pid.tolist())
      pos = np.array([pc.get(p, 0) - (1 if (exclude_src and inset[s]) else 0) for p, s in zip(qpid, src)])
      return pos, inset[src], pc, wrong_view, len(set(gidx.tolist())) == len(gidx)


  def retention(sub, full_idx, common_pids):
      sub = set(np.asarray(sub).tolist())
      rel = [i for i in full_idx if gal_pid[i] in common_pids]
      irr = [i for i in full_idx if gal_pid[i] not in common_pids]
      return dict(full=len(full_idx), kept=len(sub),
                  common_identity_images=len(rel), common_identity_images_kept=sum(i in sub for i in rel),
                  other_images=len(irr), other_images_kept=sum(i in sub for i in irr),
                  not_subset_of_full=len(sub - set(full_idx)))


  for grp in ('A', 'G'):
      g = d['groups'][grp]
      sv, cv = view_of[grp], 1 - view_of[grp]
      qidx = np.array(g['query_indices'])
      src = q_src[qidx]
      qpid = gal_pid[src]
      common_pids = set(qpid.tolist())
      full_same = np.where(gal_view == sv)[0]
      full_cross = np.where(gal_view == cv)[0]
      print(f'\n== group {grp}: n_queries={len(qidx)} (archive {g["n_queries"]}), identities={len(common_pids)} (archive {g["n_identities"]}), '
            f'all query sources in view {sv}: {bool((gal_view[src]==sv).all())}, identity set equals archive: {common_pids==set(g["identity_ids"])}')
      # 共同查询集定义核查：在完整图库下 同视角(排源)≥1 且 跨视角≥1 的查询
      pc_s_full = collections.Counter(gal_pid[full_same].tolist()); pc_c_full = collections.Counter(gal_pid[full_cross].tolist())
      legal_all = np.array([(pc_s_full.get(gal_pid[s], 0) - 1 >= 1) and (pc_c_full.get(gal_pid[s], 0) >= 1) for s in q_src])
      legal_all &= (gal_view[q_src] == sv)
      print(f'   full-gallery legal queries (same-view excl. source >=1 and cross-view >=1, source view={sv}): {int(legal_all.sum())}; equals archive query set: {set(np.where(legal_all)[0].tolist())==set(qidx.tolist())}')
      gres = {'n_queries': int(len(qidx)), 'n_identities': len(common_pids), 'trials': []}

      def analyse(label, blk, seed=None):
          ps, src_in, pc_s, wv_s, uniq_s = count_pos(blk['same']['gallery_indices'], sv, qpid, src, True)
          pcr, _, pc_c, wv_c, uniq_c = count_pos(blk['cross']['gallery_indices'], cv, qpid, src, False)
          hs = np.array(blk['same']['has_legal_positive_by_requested_query'], bool)
          hc = np.array(blk['cross']['has_legal_positive_by_requested_query'], bool)
          ids = sorted(common_pids)
          same_per_id = np.array([pc_s.get(p, 0) for p in ids]); cross_per_id = np.array([pc_c.get(p, 0) for p in ids])
          same_full_per_id = np.array([pc_s_full.get(p, 0) for p in ids]); cross_full_per_id = np.array([pc_c_full.get(p, 0) for p in ids])
          r_same = retention(blk['same']['gallery_indices'], full_same, common_pids)
          r_cross = retention(blk['cross']['gallery_indices'], full_cross, common_pids)
          cs = np.array(blk['same']['correct_by_requested_query'], bool); cc = np.array(blk['cross']['correct_by_requested_query'], bool)
          rec = dict(label=label, seed=seed,
                     n_same=len(blk['same']['gallery_indices']), n_cross=len(blk['cross']['gallery_indices']),
                     wrong_view_in_same=wv_s, wrong_view_in_cross=wv_c, unique_same=uniq_s, unique_cross=uniq_c,
                     zero_pos_same=int((ps == 0).sum()), zero_pos_cross=int((pcr == 0).sum()),
                     archive_zero_same=blk['same']['zero_positive_queries'], archive_zero_cross=blk['cross']['zero_positive_queries'],
                     agree_has_legal_same=bool(((ps > 0) == hs).all()), agree_has_legal_cross=bool(((pcr > 0) == hc).all()),
                     all_query_sources_retained=bool(src_in.all()), sources_retained=f'{int(src_in.sum())}/{len(src_in)}',
                     min_same_imgs_per_identity=int(same_per_id.min()), min_cross_imgs_per_identity=int(cross_per_id.min()),
                     identities_losing_same_imgs=int((same_per_id < same_full_per_id).sum()),
                     identities_losing_cross_imgs=int((cross_per_id < cross_full_per_id).sum()),
                     same_imgs_of_common_ids_kept=f'{int(same_per_id.sum())}/{int(same_full_per_id.sum())}',
                     cross_imgs_of_common_ids_kept=f'{int(cross_per_id.sum())}/{int(cross_full_per_id.sum())}',
                     min_legal_pos_same=int(ps.min()), min_legal_pos_cross=int(pcr.min()),
                     retention_same=r_same, retention_cross=r_cross,
                     r1_same_recount=float(cs.mean() * 100), r1_cross_recount=float(cc.mean() * 100),
                     delta_recount_pp=float((cc.astype(float) - cs.astype(float)).mean() * 100),
                     metrics_same={k: (round(v, 4) if isinstance(v, float) else v) for k, v in blk['same']['metrics'].items()},
                     metrics_cross={k: (round(v, 4) if isinstance(v, float) else v) for k, v in blk['cross']['metrics'].items()})
          rs, rc = rec['retention_same'], rec['retention_cross']
          print(f'  [{label}] same n={rec["n_same"]} cross n={rec["n_cross"]} | zero-pos same/cross={rec["zero_pos_same"]}/{rec["zero_pos_cross"]} '
                f'(archive {rec["archive_zero_same"]}/{rec["archive_zero_cross"]}) agree={rec["agree_has_legal_same"]}/{rec["agree_has_legal_cross"]} '
                f'| sources kept {rec["sources_retained"]} | min imgs/id same={rec["min_same_imgs_per_identity"]} cross={rec["min_cross_imgs_per_identity"]} '
                f'| ids losing imgs same/cross={rec["identities_losing_same_imgs"]}/{rec["identities_losing_cross_imgs"]} '
                f'| common-id imgs kept same={rec["same_imgs_of_common_ids_kept"]} cross={rec["cross_imgs_of_common_ids_kept"]} '
                f'| other imgs kept same={rs["other_images_kept"]}/{rs["other_images"]} cross={rc["other_images_kept"]}/{rc["other_images"]} '
                f'| wrong-view={wv_s}/{wv_c} unique={uniq_s}/{uniq_c} notsubset={rs["not_subset_of_full"]}/{rc["not_subset_of_full"]} '
                f'| R@1 same={rec["r1_same_recount"]:.2f} cross={rec["r1_cross_recount"]:.2f} d={rec["delta_recount_pp"]:.2f}')
          if label == 'original' or label.endswith('2026'):
              print(f'      metrics same={rec["metrics_same"]}\n      metrics cross={rec["metrics_cross"]}')
          return rec

      gres['original'] = analyse('original', g['original'])
      r1s, r1c, dl = [], [], []
      for t in g['authoritative_trials']:
          rec = analyse(f'trial seed {t["seed"]}', t, t['seed'])
          rec['index_source'] = t.get('index_source')
          gres['trials'].append(rec)
          r1s.append(rec['r1_same_recount']); r1c.append(rec['r1_cross_recount']); dl.append(rec['delta_recount_pp'])
      for name, arr in (('same R@1', r1s), ('cross R@1', r1c), ('delta pp', dl)):
          arr = np.array(arr)
          print(f'   10-draw {name}: mean={arr.mean():.3f} std(ddof=0)={arr.std(ddof=0):.3f} std(ddof=1)={arr.std(ddof=1):.3f} range=[{arr.min():.3f}, {arr.max():.3f}]')
          gres[f'draw_{name.replace(" ", "_").replace("@","")}'] = dict(mean=float(arr.mean()), std0=float(arr.std(ddof=0)), std1=float(arr.std(ddof=1)), values=[float(x) for x in arr])
      zp = g['zero_positive_policies']['a_fixed_misses']
      print(f'   archive a_fixed_misses: mean={zp["mean_of_per_draw_paired_deltas_pp"]:.4f} std={zp["std_of_per_draw_paired_deltas_pp"]:.4f} (ddof={zp["draw_variation_std_ddof"]})')
      ts = zp.get('two_stage_identity_and_gallery_bootstrap', {}); ib = zp.get('identity_bootstrap_conditional_on_fixed_draws', {})
      print('   two_stage CI:', {k: v for k, v in ts.items() if 'ci' in k or 'delta' in k or 'undefined' in k})
      print('   identity-only CI:', {k: v for k, v in ib.items() if 'ci' in k or 'delta' in k or 'undefined' in k})
      print('   table5_reference:', g.get('table5_reference'))
      gres['archive_a_fixed_misses'] = {k: v for k, v in zp.items() if k != 'per_draw'}
      result['groups'][grp] = gres

  json.dump(result, open(OUT, 'w'), indent=1, ensure_ascii=False)
  print('\n== written', OUT)



for name, J in latest.items():
    run(J)

# ---------------------------------------------------------------- sampler source in repo
print('\n== sampler source (repo cwd =', os.getcwd(), ')')
for fn in ('sample_gallery', 'common_query_sets', 'eligible_queries'):
    r = subprocess.run(['grep', '-rln', '--include=*.py', f'def {fn}', '.'], capture_output=True, text=True).stdout.split()
    print(f'-- def {fn}: {r or "NOT FOUND"}')
    for f in r[:2]:
        try:
            src = open(f, encoding='utf-8').read()
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == fn:
                    seg = ast.get_source_segment(src, node)
                    lines = seg.splitlines()
                    print(f'   {f}:{node.lineno}  ({len(lines)} lines)')
                    print('\n'.join('   | ' + l for l in lines[:90]))
                    if len(lines) > 90:
                        print('   | ... (truncated)')
        except Exception as e:
            print('   ERR', f, e)
