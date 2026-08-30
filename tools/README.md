# Standalone TAG-PEDES analyses

These entry points do not modify or call the training/evaluation runner. S8,
S9 and S10 need only Python and NumPy and perform no GPU computation. The probe
imports the repository's training dependencies only when run with a checkpoint.
All commands below run from the repository root.

## Inputs and ordering

- `sim_qg.npy` must be the **raw, merged, pre-CSLS/pre-rerank** matrix exported
  by `main.py --eval_only --resume ... --dump_sim`.
- Use `--dump_sim_dtype float32` for exact R@1 validation. Float16 can introduce
  ranking ties. Offline ties are resolved by gallery index order, which is not
  guaranteed to match PyTorch's unspecified ordering of exact ties.
- Annotation records define gallery column order. Captions expanded in record
  order define query row order. No records are sorted or deduplicated.
- Normal offline runs assert **16,504 queries; 8,252 images; 4,498 aerial;
  3,754 ground; 2,000 identities; 800 dual-view identities**. Missing or
  contradictory camera information is an error, not a ground-view default.
- Explicit `cam_id` takes precedence and is cross-checked with recognizable
  paths. Without it: Market/Market-1501 is ground; PRAI-1581 and UAV-Human are
  aerial; G2APS `D<number>_`/`S<number>_` means aerial/ground; AG-ReID.v2 filename
  `C0F...`/`C3F...` means aerial/ground. Explicit aerial/ground subdirectories
  are also recognized. Unknown mixed-source camera patterns fail closed.
- NumPy memory maps, 128-row metric blocks and 256-row top-k blocks avoid full
  ranking copies. Float16 scores are promoted to float32 before arithmetic.
- R@1/R@5/R@10 use the online evaluator's float32 percentage arithmetic; mAP
  uses float64 precision accumulation and can differ from online float32 by
  rounding. All reported retrieval metrics are percentages.

## S8: controlled views

```bash
python tools/S8_controlled_eval.py \
  --sim_qg /path/to/experiment/sim_qg.npy \
  --ann /path/to/TAG-PEDES/test_reid.json \
  --csls_k 10 --out /path/to/analysis/S8_controlled.json
```

The unfiltered self-check must first reproduce the stated checkpoint's fp32
R@1 `63.633060455322266` (displayed as `63.63306%`, 10,502 / 16,504 hits).
Another checkpoint is deliberately rejected by this fixed research protocol.
If a sibling `sim_meta.json` exists, its original metrics are also compared and
differences recorded. Float16 input prints a quantization warning and records
whether it matches the reference, but cannot certify strict fp32 equivalence.

Controlled queries belong to the 800 dual-view identities. Each target gallery
contains **all** images of the target view, not only those 800 identities. The
query's source image is excluded, not the entire identity. Queries with zero
remaining positives are dropped and counted. JSON includes input/eligible query
counts, base gallery size and min/max gallery size after source exclusion.

`--csls_k` is optional (omitting it disables the CSLS comparison; using the flag
without a value selects 10). Each protocol estimates a fresh column penalty
using only its eligible query subset and target gallery. Source-image pairs are
also excluded from that top-k estimation. If an excluded column has fewer than
k eligible queries, it uses all available queries; an empty column is an error.

## S9: error enrichment

```bash
python tools/S9_error_stats.py \
  --pred /path/to/experiment/output.json \
  --ann /path/to/TAG-PEDES/test_reid.json \
  --out /path/to/analysis/S9_error_stats.json
```

Use the intended **snapshot**: the online evaluator may write postprocessed
predictions to `output.json` when CSLS/rerank is enabled. S9 validates every
stored match flag against the supplied annotation, but cannot infer which
checkpoint/postprocessor produced a snapshot.

For each error query, the conditional prior excludes all gallery images of its
true identity. Enrichment uses `P(same-source | error, query-view)` divided by
the mean conditional prior **over those same error queries**. Bootstrap samples
the indicator/prior pair together, at query (caption) level, 10,000 times with
seed 2026, separately for aerial/ground. CIs are 2.5/97.5 percentile intervals.
Queries sharing an image/identity are not clustered; this follows the specified
query-level bootstrap protocol. Empty/undefined groups use JSON `null`.

## S10: cross-fitting

```bash
python tools/S10_csls_crossfit.py \
  --sim_qg /path/to/experiment/sim_qg.npy \
  --ann /path/to/TAG-PEDES/test_reid.json \
  --k 10 --out /path/to/analysis/S10_crossfit.json
```

`numpy.default_rng(2026).permutation` makes the fixed half split. Every scan
samples without replacement **only from the initial calibration half**, never
from the evaluation half. This ensures all three `n_cal=8252` trials exactly
equal the crossfit setting. Smaller sizes use seeds 2026/2027/2028; mean and
population standard deviation (`ddof=0`) are reported. The complete gallery and
source image are retained in this protocol. No labels enter the CSLS penalty.
The split is at caption level as requested, not image/identity level; related
captions may fall in opposite halves. Split indices are saved in JSON.

## Orthogonality gradient probe

```bash
python tools/probe_ortho.py \
  --resume /path/to/experiment/checkpoint_best.pth \
  --config config/config.local.yaml --num_batches 20 \
  --out /path/to/analysis/probe_ortho.json
```

Use the checkpoint's training config with local paths and matching experimental
switches. Launch plain `python`, not multi-rank `torchrun`. The standalone probe
initializes a world-size-1 process group to preserve the repository's sampler
and all_gather behavior. The original training DataLoader uses `num_workers=2`.
`--batch_size` and `--device` are optional probe-only overrides. No optimizer is
constructed and no checkpoint is written; original `model` weights are loaded,
not EMA. Only trusted checkpoints should be loaded (`weights_only=False`).

The default `--epoch 1` is zero-based, using steady-state soft labels; specify
the intended epoch for warmup-dependent NRW behavior. The existing checkpoints
do not store an epoch. CUDA autocast matches the training forward. On the same
forward graph, the probe backpropagates weighted `ret['ortho_loss']`, clears
gradients, then independently backpropagates the exact training sum of all
returned loss terms. Both backwards use the same fixed `--loss_scale` (default
65536, the training GradScaler initial scale) to reduce FP16 gradient underflow.
Reported gradient norms are divided by that scale. A non-finite norm fails
clearly and requests a lower scale; there is no silent numerical fallback.
No optimizer step is taken. Ephemeral training buffers may update in the probe
process, but neither saved weights nor the training process are affected.

Hooks observe each enabled MoE block's actual post-`ln_2` CLS/view tokens, once
per original main-image forward. Bridge forwards and checkpoint recomputations
are not double-counted. Cosine statistics are aggregated over layer × sample.
The actual loss is `100 * mean(max(abs(cos) - 0.001, 0))`, then `ortho_ratio`.
Thus the requested `abs(cos) >= alpha` fraction is the **active side**, not an
upper saturation. The true zero-clamped `abs(cos) <= alpha` fraction is also
reported. All-disabled MoE/orthogonal-loss configurations fail with a clear error.

## Self-tests

```bash
python tools/S8_controlled_eval.py --self_test
python tools/S9_error_stats.py --self_test
python tools/S10_csls_crossfit.py --self_test
python tools/probe_ortho.py --self_test
python -m unittest tools.test_analysis_tools -v
```

The offline fixtures use 20 queries / 10 gallery images. Synthetic annotations
relax dataset-size anchors only through the self-test path. The probe's CPU
mock needs torch but no checkpoint; it tests the two-backward procedure,
checkpoint recomputation accounting and parameter immutability. Synthetic
results must not be reported as measured checkpoint performance.
