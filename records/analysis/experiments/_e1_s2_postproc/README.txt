Source files copied without modification on 20261025T145025Z (cp -p keeps mtimes):
  /root/autodl-tmp/scripts/S2_postproc_compare.py
  /root/autodl-tmp/experiments/S2_FK32.log
  /root/autodl-tmp/experiments/postproc_fk32.json
Purpose: offline comparison of inference-time post-processing on ViT-L/14 model 1 (paper Table 8),
including the alpha query expansion row (alpha=3, k=2) analysed in supplement Section 5.19 / Table B19.
Inputs used by the script: /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy, sim_gg.npy and test_reid.json.
SHA-256 of the copies:
bc3ecb38c1ac33c05690df7bd566fd82f1091f5530e023c5bc56f3e908ead536  S2_postproc_compare.py
bb6976a3ab98501fc0292e295d6a5556fb3324251d2af0f5f36afde149669fd6  S2_FK32.log
c2ee347d87e85986e528fdd40e16b452f0f28e87708ef0e9a823c41c4d507d38  postproc_fk32.json

Scope and status notes (added 2026-10-26, second revision of the manuscript)
---------------------------------------------------------------------------
1. This directory keeps the ORIGINAL offline comparison exactly as it was run on 2026-08-30
   (script, console log, JSON). The three files above are byte-for-byte copies and were not
   re-run, edited or re-rounded for the manuscript. The notes below only explain how they
   relate to the final analysis; they do not change any number.

2. Reciprocal-neighbour explanation in the script docstring (the paragraph on krecip whose
   heading translates as "honest note on krecip"). The docstring asserts that reciprocal
   neighbours cannot exist between text queries and gallery images and that the 0.00 gain of
   k-reciprocal is a structural degeneration. This was the early working explanation and was
   later superseded by direct measurement on the same similarity matrix: S11 part C
   (reciprocal_statistics in scripts/_s11_descriptive.py, run by scripts/S11_offline_stats.py;
   the S11 report is the file listed as inputs.authoritative_S11 in
   records/analysis/experiments/_e1_s14/S14.json) found that a large share of top-k candidate
   pairs do satisfy the reciprocal condition (44.8% at k=10 as reported in the manuscript),
   i.e. the reciprocal structure is present, while the direct k-reciprocal implementation
   changed no ranking and the gallery-side encoding variant lowered R@1. The final manuscript
   (Section 4.5.1) is based on that measurement. The docstring is kept verbatim as a
   historical record; it is not a method statement or a conclusion of the final manuscript.

3. Bootstrap unit. paired_bootstrap() in S2_postproc_compare.py resamples individual QUERIES
   (n = 16 504, seed 0). Its output, bootstrap_csls_k10_vs_none in postproc_fk32.json
   (delta R@1 = +1.036 pp, 95% CI [+0.703, +1.375]), is a historical query-level interval.
   The interval reported in the manuscript for the transductive CSLS gain of model 1 is the
   identity-clustered paired bootstrap produced by scripts/S13_inductive_bootstrap.py
   (configuration c, "transductive: complete test-query bank", alpha = 0.5, k = 10; whole
   identity clusters resampled, 2 000 clusters, 10 000 repeats, seed 2026), archived at
   records/analysis/experiments/_e1_s13/S13_fk32_20260831T122510_659688Z.json under
   configurations.c.bootstrap.identity (observed delta +1.0301 pp, 95% CI [+0.6498, +1.4087],
   reported as [+0.65, +1.41]). The two intervals use different resampling units and are not
   interchangeable; neither is a data error. The point estimates differ by one query
   (+1.036 vs +1.030 pp) only because the two scripts resolve the tied top-1 scores of the
   uncorrected ranking differently (this script: numpy argsort order, R@1 63.6270; S13: score
   descending then gallery index ascending, R@1 63.6331); the CSLS k=10 R@1 (64.6631) is the
   same in both. The Table 8 rows taken from this directory's JSON (none, column z-score,
   inverse softmax, transductive CSLS, alpha query expansion and the k-reciprocal variant) are
   unchanged; the +1.03 gain of model 1 and its identity-clustered interval use the S13 record.
