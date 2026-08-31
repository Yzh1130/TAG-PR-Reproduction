# S13 paired inference and E3 fresh-forward confirmation

This extension only adds files. It does not change S11/S12, their helpers,
training, evaluation, model, configuration, or data loading. NumPy/stdlib serve
S13; E3 uses the repository's existing Torch environment. No dependency is added.

## Files

- `S13_inductive_bootstrap.py`: three fixed correction configurations and
  query/image/identity paired bootstrap, including the paired gain difference.
- `E3_fresh_forward.py`: separate `--fresh-forward` entrypoint, fresh matrices
  and the unchanged S12 352-row scan.
- `test_S13_bootstrap.py`, `test_E3_fresh_forward.py`: synthetic reference and
  orchestration tests; these are not the real fk32 experiment's results.

Run either entrypoint with `--help`. If scripts are copied outside the checkout,
keep `_offline_common.py`, `_s11_descriptive.py`, `_s12_features.py` and
`S12_inductive_bank.py` beside them, with the existing S12 helper files unchanged.
E3 accepts `--repo` for the model checkout. Outputs are timestamp/tag JSON and
logs under `--out`, never inside either input matrix directory or
`_archive_evidence`. Existing files are not overwritten.

## S13: fixed-penalty paired bootstrap

The ranking score is the unchanged S12 convention `s - alpha*r(g)`;
`alpha=0.5` has the same ranking as `2*s-r(g)`.

| Configuration | Full calibration bank | alpha | k |
|---|---|---:|---:|
| a | training texts | 0.5 | 30 |
| b | training texts | 0.25 | 10 |
| c | test queries (transductive control) | 0.5 | 10 |

Each column penalty is computed once using the entire specified bank. Then the
original/corrected per-query predictions are frozen. Bootstrap never re-encodes,
re-estimates penalties, resamples the bank, or changes the gallery. All methods
evaluate the same full ordered query set, with no source-image exclusion.

The existing `_s11_descriptive.paired_test` performs 10000 resamples (default
seed2026), separately using queries, whole source-image clusters, or whole
identity clusters. Cluster replicates keep all member queries and their
multiplicities, using query-weighted accuracy rather than an unweighted average
of per-cluster accuracies. Outputs include observed and bootstrap-mean DeltaR1,
percentile95 CI, and two-sided **bootstrap sign-tail p** with the existing +1
Monte Carlo correction. This is not a null-centred bootstrap-test p; query-level
results also retain the helper's exact McNemar result.

For the gain difference, `(R1_a-R1_raw) - (R1_c-R1_raw) = R1_a-R1_c`.
S13 therefore calls the same paired helper with `first=c, second=a` and the same
seed/group labels. Each replicate uses exactly the same resampling indices as
the two gains. It does **not** subtract independently bootstrapped CIs.

When the input training bank came from S12 Gram reconstruction, it inherits
S12's complete-test-query fitted W. These are **offline-simulation gain CIs**,
not proof of a query-independent end-to-end pipeline. S13 does not infer that
provenance from the matrix values. The requested configurations are held fixed;
the intervals/p values do not adjust for earlier selection from a larger grid
or for multiple comparisons.

## E3: independent fresh-forward entrypoint

The explicit `--fresh-forward` flag selects this new script's only execution
mode. It does not add or modify a flag in the existing S12 script.

1. Load the supplied checkpoint through the existing strict loader, including
   the configured EMA choice. Set Python/NumPy/Torch/CUDA seed **once**, before
   model initialization, via the existing loader. Default `--seed` is **0**.
2. Extract gallery images **once** with the existing eval dataset/transform,
   then extract test texts once, then training texts once, in the same process.
   Defaults: image batch32, text batch256, `num_workers=2`.
3. Check that both text extractions leave Python/NumPy/Torch RNG states and
   model buffers unchanged. Any change stops explicitly, without reseeding or
   retrying. Thus a change in test-query count/content cannot advance random
   state used by the subsequent training-text bank.
4. Use the **same one-pass gallery features** to compute all three float32
   score matrices, in `--score-block-size` row blocks (default256):

   ```text
   sim_qg_fresh     = .5*(T_test_global G_global.T + T_test_TSE G_TSE.T)
   sim_bank_g_fresh = .5*(T_train_global G_global.T + T_train_TSE G_TSE.T)
   sim_gg_fresh     = .5*(G_global G_global.T + G_TSE G_TSE.T)
   ```

   Each branch is independently normalized by the existing extractor. Unequal
   global/TSE widths are supported without reshaping or truncation (L/14:
   768/1024). View prompts also remain supported: encode both text-view variants
   and select by each gallery column's view for both train and test text scores.
   Gallery self terms remain in gg, matching S12's DualIS convention.
5. Only after writing the fresh matrices, read the archive to report its
   R1/R5/R10/mAP and the fresh-minus-archived differences. There is **no parity
   comparison or metric threshold**, no archived gg/metadata/cache dependency,
   no W and no seed search. Even a large metric difference is a reported result,
   not a reason to select a different routing realization.
6. Pass the three **fresh** matrices to the unchanged `run_corrections` defaults:
   11 banks x 32 configurations = **352 rows**. No archived matrix participates
   in this scan.

JSON explicitly records `pure_query_independent_inductive_pipeline=true` and
its scope: **train_* calibration paths only**. The included `test_full` rows
remain labelled transductive controls. Train-text, train-subset and gg penalties
have no test-query input; labels are only used when evaluating metrics.

Also recorded: seed, checkpoint/config/EMA provenance, actual feature shapes,
image/text batch sizes, worker count, extraction order and each extraction time,
score-generation time, scan time, device/Torch/TF32 settings, CUDA peak memory
when applicable, output paths and all archive/fresh metrics. A fixed seed and
one shared gallery freeze this run's Gumbel realization; they do not promise
identical scores after changing batch size, hardware, libraries or kernels.

Outputs are prefixed with `E3_<tag>_<timestamp>_`:
`sim_qg_fresh.npy`, `sim_bank_g_fresh.npy`, `sim_gg_fresh.npy`. At the supplied
sizes their combined matrix payload is about **1.99 GiB** (float32; headers and
JSON/logs extra). They are CPU disk-backed arrays; score computation reuses the
existing FP32 GPU/CPU block implementation. No full corrected matrix copies are
needed for the correction scan.

## AutoDL commands

Adjust the checkpoint/config/repository paths to the actual training machine.
These commands have **not** been run remotely by the local Windows task.

```bash
source /root/autodl-tmp/envs/tagpr5090/bin/activate
REPO=/root/autodl-tmp/TAG-PR-Reproduction
SCRIPTS="$REPO/scripts"
DUMP=/root/autodl-tmp/experiments/_dump/20260830_dump_fk32
ANN=/root/autodl-tmp/datasets/TAG-PEDES/test_reid.json
OUT=/root/autodl-tmp/experiments/_e1
# Use the exact existing S12 output, not a wildcard that may match multiple runs.
BANK=/root/autodl-tmp/experiments/_e1/S12_fk32_REPLACE_WITH_TIMESTAMP_sim_bank_g.npy

python "$SCRIPTS/S13_inductive_bootstrap.py" \
  --sim "$DUMP/sim_qg.npy" --bank "$BANK" --anno "$ANN" \
  --out /root/autodl-tmp/experiments/_e1_s13 \
  --tag fk32_s13 --bootstrap 10000 --seed 2026

python "$SCRIPTS/E3_fresh_forward.py" --fresh-forward \
  --sim "$DUMP/sim_qg.npy" --anno "$ANN" \
  --resume /root/autodl-tmp/experiments/20260827_l14_final_keep \
  --config /root/autodl-tmp/cfg_l14.yaml --backbone ViT-L/14 \
  --repo "$REPO" --seed 0 --image-batch-size 32 --text-batch-size 256 \
  --num-workers 2 --score-block-size 256 \
  --out "$OUT" --tag fk32_seed0_fresh
```

## Local verification

Combined regression run: **42 tests in 12.655s, all passed** (E3 11 + S13 6 +
unchanged S11/S12 regressions 25). Both new entrypoint `--help` commands and
compilation passed; `git diff --exit-code` confirmed no tracked file edits
before staging these additions.

```bash
python -m unittest scripts.test_S13_bootstrap scripts.test_E3_fresh_forward -v
python -m unittest scripts.test_S11_S12 scripts.test_S12_reconstruction -v
```

On this Windows Anaconda installation, the independent NumPy-GEMM/Torch-CPU
reference tests encounter duplicate Intel OpenMP runtimes. A safe **test-process
only** setting runs MKL sequentially; it neither changes repository files nor
enables `KMP_DUPLICATE_LIB_OK`:

```bash
python -c "import os; os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'; import unittest; s=unittest.defaultTestLoader.loadTestsFromNames(['scripts.test_E3_fresh_forward','scripts.test_S13_bootstrap','scripts.test_S11_S12','scripts.test_S12_reconstruction']); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())"
```

S13 200x100 query / 300x100 bank mock, seed2026, 10000 replicates:

| Comparison | R1 | Query-bootstrap mean delta (pp) | Percentile95 CI (pp) |
|---|---:|---:|---|
| raw | 25.5 | - | - |
| a vs raw | 36.5 | 11.00115 | [6.5, 15.5] |
| b vs raw | 32.5 | 6.98655 | [3.5, 10.5] |
| c vs raw | 36.5 | 10.96940 | [6.5, 15.5] |
| gain(a) - gain(c) | - | 0.03175 | [-3, 3] |

All 12 query/image/identity statistics match an independent shared-draw
reference within 1e-13 pp; `a-c` is verified replicate by replicate. Additional
20x10 full-sort/metric/exclusion references pass. Calibration is called exactly
twice (train top30 including top10, test top10), not inside any bootstrap.

E3 actual Torch-CPU scores vs independent NumPy-GEMM reference, 20x10 qg,
30x10 bank, widths4/6:

| Text mode | qg max abs error | bank-g max abs error | gg max abs error |
|---|---:|---:|---:|
| no view prompt | 5.96046448e-8 | 5.96046448e-8 | 5.96046448e-8 |
| view prompt | 2.98023224e-8 | 4.47034836e-8 | 5.96046448e-8 |

The mock integration reuses one gallery extraction, calls the real score writer
and metric functions, and runs all352 correction rows (mock subset sizes are
100/200/250 instead of real1000/5000/16504; all methods/grid/11banks retained).
Changing test text and archived scores leaves training-bank and gg matrices
**bitwise equal, max error0**. Deliberate text RNG/buffer mutation is rejected;
archive metric differences never trigger parity/retry. Inputs remain unchanged.

The real fk32 CI values and E3 retrieval gains require the archived matrices,
S12 training bank and checkpoint on AutoDL. Synthetic tests verify statistical
pairing, score construction and data dependencies, not real-paper effect sizes.
