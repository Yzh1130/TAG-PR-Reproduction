q3_alphaqe_strata.py: stratified analysis of alpha query expansion (paper Table 8 row alphaqe k=2)
on ViT-L/14 model 1, reproducing S2_postproc_compare.py exactly and splitting queries by baseline top-1 correctness.
Used for supplement Section 5.19 / Table B19 and paper Section 4.5.3 (second revision).
Inputs: /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy, sim_gg.npy, /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json (read-only).
Original run: 2026-10-23, output tagpr_q3_alphaqe_20261023T081442Z.json (copied here if still present).
Rerun on 20261025T145025Z: log q3_alphaqe_rerun_20261025T145025Z.log, output tagpr_q3_alphaqe_20261025T145025Z.json.
SHA-256 of files in this directory:
41a7f2902aab5c8cf713835fc1e1cc1c2f2335a5c5fd862815a8cb679feeecb4  q3_alphaqe_rerun_20261025T145025Z.log
a6e3b6d30872e7e0c740342854a78163d4b024f7a3dc10da1a8df75cb1bae559  q3_alphaqe_strata.py
ae11dfd724a9c3fe1e3226639c63befebad472cf50c71f9eb1eb369a3bf95b0c  tagpr_q3_alphaqe_20261023T081442Z.json
ae11dfd724a9c3fe1e3226639c63befebad472cf50c71f9eb1eb369a3bf95b0c  tagpr_q3_alphaqe_20261025T145025Z.json
