# S11 / S12: immutable-archive analyses

These are **new standalone tools**. No existing training, evaluation, dataset,
model, configuration or `tools/` file is changed. Similarity inputs are opened
read-only; output beneath `_archive_evidence` or beneath a similarity-input
directory is refused. Existing feature caches are verified, never overwritten.
No third-party dependency is added: S11 and the correction engine use NumPy and
the standard library; real S12 extraction uses the repository's existing torch,
torchvision, tokenizer and model dependencies. No faiss, SciPy or sklearn.

## Files and entry points

- `S11_offline_stats.py`: A–G statistics, CPU only; no model imports.
- `S12_inductive_bank.py`: strict feature stage followed by 352 correction rows.
- `_offline_common.py`: annotation/order validation, bounded-memory metrics,
  column top-k, safe output and reporting helpers.
- `_s11_descriptive.py`: Wilson intervals, reciprocal pairs, hubness, paired
  query/image/identity bootstrap and exact McNemar.
- `_s11_groups.py`: group-disjoint crossfit, whole-ID scans, held-out k selection.
- `_s12_features.py`: original text/image evaluation representations, strict
  checkpoint loading, reproducibility and full archived-matrix parity guards.
- `_s12_reconstruct.py`: opt-in float64 Gram factorization, text-coordinate
  least squares and blockwise residuals; no Torch or image encoder calls.
- `test_S11_S12.py`: independent CPU mock and injected torch-CPU feature tests.
- `test_S12_reconstruction.py`: numerical/reference, CLI-path, WARN/STOP,
  immutable-input and reconstruction-output tests.

Run `python scripts/S11_offline_stats.py --help` and
`python scripts/S12_inductive_bank.py --help` for all arguments.
When copying tools to `/root/autodl-tmp/scripts/`, copy **all nine Python files**
together. S11 does not need the repository on `PYTHONPATH`; S12 accepts `--repo`
when these files are outside the checkout.

Production runs print readable tables and create timestamp/tag JSON + logs in
`--out`. Failures after startup are recorded as well. S12 also saves
`train_text_feats.npz` and a timestamped `*_sim_bank_g.npy` only under `--out`.
Its `--self-test` mode needs no assets/GPU and prints its test summary only.

## Important protocol decisions

### No fixed accuracy gate; counts are derived

No expected R@1 is hardcoded. Annotation records determine gallery columns;
captions expanded in record order determine query rows. Camera labels are
cross-checked against recognized path rules. ID/source conflicts, ambiguous
views, duplicate image paths, malformed matrices and nonfinite values fail.

S11 derives both common-query sets by intersecting the queries with at least
one remaining positive in **each** target view, after excluding the source
image. The optional `--expected-common A_COUNT G_COUNT` is an explicit input
count check, not a hardcoded dataset/performance assertion. The real commands
below supply the requested `4472 3938`; another correctly shaped dataset can
run without any expected count gate.

### A: the requested downsampling is impossible in one direction

For the supplied dataset, aerial gallery size is 4498 and ground size is 3754.
A→A has **4497** available candidates per query after excluding its source.
It is impossible to *downsample* A→G's 3754 images to 4497.

The default `--gallery-control literal` therefore reports
`partial_protocol_infeasible` for that controlled comparison, retaining the
original four protocols, the feasible G-side control, and all B–G results.
It never duplicates images or silently substitutes another protocol.

**Only after agreement to the alternative**, use
`--gallery-control common-min`. For each source view it matches both protocols
to `min(same-view gallery size - 1, cross-view gallery size)` eligible candidates
per query. This is 3754 for A queries and 3753 for G queries. Same-view sampled
galleries retain all common-query source images and at least two images per
required identity; excluding the source then removes exactly one candidate.
Cross-view galleries retain at least one positive per required identity. Fill
remaining slots randomly without replacement, independently for each recorded
seed 2026–2035. This is **positive-preserving stratified sampling**, not uniform
sampling over all possible subsets. If mandatory images exceed the budget,
fail instead of violating the guarantee. JSON includes sampled gallery indices.

For each protocol report all four metrics, query count and actual candidate
count range. Ten-draw summaries use population std (`ddof=0`). Paired bootstrap
and McNemar compare cross-minus-same on identical ordered queries, separately
for each draw; draws are not pooled as independent additional queries.

### B–E: statistical definitions

- B: four error cells partition top-1 errors; denominators and Wilson95
  intervals are included. Four conditional probabilities are
  `P(same/cross view | same/cross source, error)`. All/aerial/ground strata are
  separate. Raw and CSLS versions are both reported.
- C: neighbors cross modalities: query-row top-k and gallery-column top-k.
  Report absolute reciprocal counts, both candidate-pair denominators and a
  full per-query count histogram. No query-query relation is invented.
- D: `N1[g]` includes every gallery item, including zeros. Skew is
  `mean((N1-mean(N1))**3) / std(N1)**3` (population moments). This is the
  [QB-Norm appendix H](https://arxiv.org/html/2112.12777v2#A8) definition,
  applied to the user-requested **N1**, not the paper's usual N10. Constant
  distributions have undefined skew (`null`). Head sizes use `ceil(0.01*G)`
  and `ceil(0.05*G)`. Errors-only heads and skew use that method's own errors.
- E: estimate full-query k=10 penalties **once**, freeze predictions, then
  perform 10,000 paired resamples at query, source-image and identity levels.
  Cluster bootstrap samples whole clusters uniformly and uses
  `sum(correct-count differences) / sum(query counts)`, not an unweighted
  average of cluster accuracies. Report bootstrap mean delta (percentage
  points), percentile95 interval and two-sided bootstrap sign-tail probability
  with finite-replicate correction. This p is labelled as a sign-tail summary,
  **not** a null-centred permutation-test p. Exact query-level McNemar is also
  reported, with its lack of cluster correction explicitly noted.

### F–G: leakage and bank definitions

F retains random-query halves and adds image/identity 2-fold splits. Whole-group
splits use decreasing group-size greedy balancing, with seeded tie order
(GroupKFold-style without a version-dependent sklearn dependency). Per-fold
query/image/identity overlaps and counts are reported. Image grouping may still
share identities; identity grouping must share neither images nor identities.

Calibration-scan budgets 100/500/1000/5000/full denote **query-count targets**.
Sample complete identities from the first identity calibration fold until the
target is reached/exceeded. Report actual identities/images/query counts and
overshoot, for seeds 2026/27/28. Never break an identity to achieve an exact
caption count. Evaluation remains the second identity fold. The full endpoint
must exactly reproduce that fold's crossfit R@1. Too-large budgets are explicitly
reported as unavailable for a small mock, not replaced with evaluation queries.

G shuffles IDs into halves, seeds 2026–2030. Select k using H1's bank and H1's
labels (smallest k breaks equal R@1). Lock k, estimate its penalty on **unlabelled
H2**, and evaluate the locked k once on H2. Swap halves. No alternative k is
scored on held-out labels. Report both a fixed-k10 **same H2-bank** control and
the requested fixed-k10 full-query-bank reference. The latter delta includes
bank changes, so is not a pure tuning-effect estimate. Also report the stricter
supplementary variant that freezes H1's chosen penalty on H2 (no H2 queries in
calibration); keep it separate from the primary result.

## S12 fidelity and formulas

The loader inspects and prints candidate train annotation schemas and counts
before selecting exactly one. Use `--train-anno` if ambiguous. Reuse existing
config resolution, checkpoint architecture inference and strict resume loading,
including the config's EMA choice. Record which state (`model`/`model_ema`)
was actually loaded. Seed once from `config.misc.seed`, unless `--seed` is
explicit; no reseeding between the two train-text passes and no RNG search.

Use existing `pre_caption(max_words)`, tokenizer/context length, `encode_text`
and `textual_emb_layer(..., training=False)`, normalizing global/TSE separately.
Both view-prompt configurations are supported by the default direct path. Compare the two consecutive
float32 extractions bitwise, including signed zero; print mismatching positions
and stop on a discrepancy. This stage does not call the image encoder.

Image caches may use `image_feat/image_tse_feat`, `image_feats/image_tse_feats`
or `global_features/tse_features` in a recognized `.npz`. Otherwise extract the
gallery once using the existing eval dataset and validation transform. Different
global/TSE embedding widths are supported. In both cases recompute the complete
test qg matrix in blocks and require max absolute error **strictly below
`--parity-tol` (default 1e-5)**. The separate Gram branch below has an explicitly
different acceptance rule; the default direct guard is not relaxed.
Check shape, finite values, and full row/element coverage. A mismatch produces
JSON/log diagnostics and stops **before** bank corrections; stochastic image
MoE routing, wrong config, weights or batching are possible causes, not assumed
proof of one cause. No existing matrix is regenerated or replaced.

Match `--image-batch-size` to the archived evaluation's actual image loader batch
size (default 32); text default is the original 256. The same merge is used:
`0.5*(text_global @ image_global.T + text_TSE @ image_TSE.T)`. For view prompts,
encode each view separately and select the gallery column's corresponding view.

The 11 banks are full test, full train and train subsets 1000/5000/16504 for
seeds 2026/27/28. Each bank produces 32 rows:

| Method | Ranking score | Grid |
|---|---|---|
| CSLS / NNN | `s - alpha * mean(topk(bank[:,g]))` | alpha .25/.5/.75/1; k 5/10/16/20/30 |
| IS | `s - LSE(beta*bank[:,g])/beta` | beta 5/10/20/50 |
| DIS | IS only if the query's original top1 is top1 of at least one bank query; otherwise raw | same beta |
| DualIS | `s - .5*(LSE(beta*query_bank[:,g])/beta + LSE(beta*gallery_bank[:,g])/beta)` | same beta |

CSLS `2s-r` has the same ranking as alpha=.5. DIS follows the
[official QB-Norm implementation](https://github.com/ioanacroi/qb-norm/blob/master/dynamic_inverted_softmax.py).
DualIS is the rank-equivalent log of the product in
[DBNorm Eq.1](https://aclanthology.org/2023.emnlp-main.652.pdf), with equal
temperatures. The requested test gallery is a fixed offline gallery bank;
this is explicitly labelled as an adaptation of the original train/validation
gallery bank. Its diagonal is retained (unlike gallery-CSLS self exclusion).
An archive declaring incompatible qg/gg scoring is rejected. Test identities
are used only for metrics, never for normalizers or hub activation.

IS reductions use stable streaming float64 log-sum-exp; float64 penalties stay
float64 during ranking. Raw and CSLS on float32 dumps stay float32. Metric
ties always favor ascending original gallery index, which can differ from an
unspecified GPU tie order. R@k percentage arithmetic matches the online float32
convention; mAP accumulates float64. Exact positive-rank counting avoids sorting
every negative when there are few positives; tests compare against full sorting,
including ties, source exclusion, and a full-sort fallback for many positives.

## S12 opt-in recovery from an archived gallery Gram matrix

Use this branch when the dump contains `sim_qg.npy` / `sim_gg.npy` but no gallery
features, and stochastic image routing prevents an identical fresh extraction.
It does **not** rerun the image encoder or read `output.json` as features.
Existing training/evaluation/model/data code is unchanged.

| Argument | Default | Meaning |
|---|---|---|
| `--reconstruct-gallery-from-gg` | off | Explicitly select this recovery protocol |
| `--gg-rank` | `1536` | Positive integer, or `auto` to retain eigenvalues `> lambda_max * 1e-8` |
| `--parity-tol` | `1e-5` | Numerical residual warning threshold in recovery; strict stop threshold in direct mode |
| `--score-block-size` | `256` | Rows per score/residual block and float32 output write |

The existing FP32 text encoder is reused unchanged. **All subsequent recovery
linear algebra is float64**, including branch concatenation, symmetrization,
eigendecomposition, pseudoinverses, coordinate fitting and score products. Only
final score blocks are converted to float32 and written immediately to memmaps.

1. Decompose `(sim_gg + sim_gg.T)/2`. Clip every negative eigenvalue to zero and
   retain the requested largest components: `G = U_r * sqrt(lambda_r)`.
   Report residuals against the **original**, not symmetrized, archive.
2. Solve `T = sim_qg @ G @ pinv(G.T @ G, rcond=1e-10)` in query blocks.
3. Extract both train and test texts **twice consecutively**, preserving the
   strict bitwise checks (including signed zero) and never calling the image
   encoder. Concatenate `[global/sqrt(2), TSE/sqrt(2)]` without renormalization.
4. Fit `W = pinv(T_test_raw, rcond=1e-10) @ T`. This is unconstrained least
   squares, **not orthogonal Procrustes**. A single SVD per pseudoinverse supplies
   both the pinv and its rank diagnostics; tests compare to `numpy.linalg.pinv`.
5. Write/evaluate all of `T_test_raw @ W @ G.T`. Only after its gate passes,
   write the training bank `(T_train_raw @ W) @ G.T` and run the existing scan.

The repository's L/14 text branches actually have widths **768 global + 1024
TSE = 1792**, not 1536. Widths are read from the extracted arrays; no columns are
truncated or reshaped to force the assumption. `--gg-rank 1536` controls only the
recovered coordinate rank. `auto` can select a different rank. A rank larger than
the number of gallery items is rejected rather than silently reduced.

This single-map branch explicitly rejects `view_prompt=true`: one query then
has two gallery-view-conditioned text vectors, which the specified formula
cannot represent. The original direct path still supports view prompts.

### Acceptance and JSON audit

The timestamped JSON contains `gallery_reconstruction` with:

- `a_gram_spectrum`: original symmetry max/relative error, negative eigenvalue
  count/max magnitude, selected rank, and retained nonnegative trace-energy
  ratio. The squared-eigenvalue energy ratio is also reported separately.
- `b_gram_residual`: `G @ G.T - archived_gg` max/relative Frobenius error.
- `c_coordinate_residual`: `T @ G.T - archived_qg` max/relative error;
  `coordinate_solver` records the explicit pinv cutoff/rank.
- `d_procrustes_residual`: `T_test_raw @ W - T` max/relative error. This field
  uses the request's name; `mapping_solver` explicitly identifies the actual
  unconstrained least-squares formula and records its rank/cutoff.
- `e_end_to_end`: full `T_test_raw @ W @ G.T - archived_qg` errors **before and
  after float32 conversion**, both full-set R1/R5/R10/mAP values, and their
  differences in percentage points. The gate evaluates the saved float32 file.
- `gate`: PASS/WARN/STOP, exact correct-count/R1 difference, and individual
  correct-to-incorrect / incorrect-to-correct / changed-top1 counts.

The hard gate is **absolute aggregate R1 difference <= 0.02 percentage points**:
for 16504 queries, a net three-query difference passes (0.0181774115366 pp),
four fails (0.0242365487155 pp). This is a *net correct-count* limit, not a limit
on individual prediction flips. Equal R1 does not prove identical rankings.
If (b), (c), or either version of (e) exceeds `--parity-tol` but the R1 gate
passes, print **WARN**, retain the residuals in JSON, and continue. If the R1
gate fails, save diagnostics/logs and exit nonzero **before creating a training
bank or correction results**. The reconstructed test-score file remains an
audit artifact under `--out`; no archive is overwritten.

Missing `sim_meta.json` is allowed **only in the opt-in branch**, with an
explicit stdout/JSON provenance warning. The supplied checkpoint/config is used,
and all residuals are measured; absence of metadata is not treated as verified
checkpoint provenance. `output.json` remains untouched. Existing metadata that
declares inconsistent global/TSE scoring or a different checkpoint still fails.

Every degradation is explicit: negative clipping, truncation, rank-deficient
text mapping and numerical gate warnings appear in JSON/stdout. Fitted training
scores can exceed [-1,1] because `W` is unconstrained. They are finite-checked,
reported with min/max/out-of-range count, and **not clipped or renormalized**.
The default direct branch retains its original cosine-range validation.

**Scientific scope:** `W` uses the complete archived test qg and freshly
extracted test text. The resulting train-text bank can be evaluated offline,
but this recovery-and-calibration pipeline is **not fully query-independent
inductive calibration**. Both JSON bank regimes and stdout mark this dependency.
Passing a test-query R1 gate does not certify mapping generalization to training
texts or an exact orthogonal recovery.

## Memory and timing

No dense corrected full-matrix copies or full ranking matrices are retained.
Metric blocks are 64 queries; top-k/score blocks are 256 rows; reciprocal
selection uses 128-row/column chunks. S12 needs roughly 1.23 GiB on disk for the
39908×8252 float32 bank matrix, plus about 273 MiB for train global(768)+TSE(1024)
features per extraction (twice transiently; double these for view prompts).
Image/test features and model weights add to this. Inputs use read-only memmaps.
S12 records extraction time, correction-scan time and CUDA allocated peak.
All 352 exact metric evaluations may take longer than the suggested five minutes;
no real-GPU runtime or peak is claimed from a mock.

Recovery runs EVD/SVD on the CPU after releasing the model. With 8252 gallery
items, each dense float64 Gram/eigenvector matrix is about 520 MiB; LAPACK needs
additional workspace. At rank1536, `G` is about 97 MiB and test coordinates about
193 MiB. Raw float64 train text at width1792 is about 546 MiB. A 256x8252 float64
score block is about 16.1 MiB. Full qg/bank output arrays are disk-backed float32
(about 520 MiB / 1.23 GiB); a full float64 bank is never created. These are array
size estimates, not measured real-run peaks. EVD/SVD and all 352 metric passes
may take substantial CPU time depending on BLAS and thread configuration;
reconstruction elapsed time is recorded separately.

## Commands on AutoDL

The current Windows task has no AutoDL SSH host/port/credentials, so remote copy
and execution have **not** been performed. Set `REPO` to the actual checkout.
After syncing the new `scripts/` files, the standalone commands are:

```bash
source /root/autodl-tmp/envs/tagpr5090/bin/activate
REPO=/root/autodl-tmp/TAG-PR-Reproduction  # adjust if the checkout lives elsewhere
mkdir -p /root/autodl-tmp/scripts /root/autodl-tmp/experiments/_e1
# Copy the new S11/S12 .py files and their new helpers from "$REPO/scripts/"
# to /root/autodl-tmp/scripts/; do not overwrite unrelated pre-existing tools.

for tag in fk32 b16f s1f s2f; do
  python /root/autodl-tmp/scripts/S11_offline_stats.py \
    --sim "/root/autodl-tmp/experiments/_dump/20260830_dump_${tag}/sim_qg.npy" \
    --anno /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
    --out /root/autodl-tmp/experiments/_e1 --tag "$tag" \
    --expected-common 4472 3938
done
# literal reports the infeasible A->G size control. After agreeing the alternative,
# add --gallery-control common-min to each command for fully matched galleries.

python /root/autodl-tmp/scripts/S12_inductive_bank.py \
  --sim /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy \
  --sim-gg /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_gg.npy \
  --anno /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
  --resume /root/autodl-tmp/experiments/20260827_l14_final_keep \
  --config /root/autodl-tmp/cfg_l14.yaml --backbone ViT-L/14 \
  --repo "$REPO" --image-batch-size 32 --num-workers 2 \
  --out /root/autodl-tmp/experiments/_e1 --tag fk32
```

For the feature-less 8/30 archive, replace the last command with this **explicit
recovery** run (no `--gallery-features` or image re-extraction):

```bash
python /root/autodl-tmp/scripts/S12_inductive_bank.py \
  --sim /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy \
  --sim-gg /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_gg.npy \
  --anno /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
  --resume /root/autodl-tmp/experiments/20260827_l14_final_keep \
  --config /root/autodl-tmp/cfg_l14.yaml --backbone ViT-L/14 \
  --repo "$REPO" --reconstruct-gallery-from-gg --gg-rank 1536 \
  --parity-tol 1e-5 --score-block-size 256 \
  --out /root/autodl-tmp/experiments/_e1 --tag fk32_gg1536
```

Use `--gg-rank auto --tag fk32_ggauto` for the explicit spectral-threshold
alternative. Sync `_s12_reconstruct.py` together with the updated S12 entrypoint;
copying only the entrypoint is insufficient. No original evidence file is edited.

## Local verification (mock, not paper results)

```bash
python -m unittest scripts.test_S11_S12 -v
python -m unittest scripts.test_S11_S12 scripts.test_S12_reconstruction -v
python scripts/S12_inductive_bank.py --self-test
```

Local run: `Ran 11 tests in 18.009s — OK`; both CLI help commands and
compilation of the original seven Python files also passed.

Recovery extension, combined regression run: **25 tests in 30.515s — OK**.
The 200x100 qg / 300x100 bank recovery fixture (seed2026, unequal global/TSE
widths 4/6) produced:

```text
Gram reconstruction max abs error            4.254019492222483e-08
test-coordinate max abs error                 3.477014309805071e-08
text-mapping max abs error                    1.042380803895071e-08
end-to-end float64 max abs / relative Frob     3.5371710449538796e-08 / 2.658947498201436e-08
saved float32 qg max abs / relative Frob       5.960464477539063e-08 / 2.09506032303678e-08
training bank vs known true scores max abs    3.583942465024137e-08
T / W vs literal numpy pinv max abs           4.440892098500626e-16 / 1.1102230246251565e-16
archived = rebuilt R1/R5/R10/mAP               1.0 / 9.0 / 20.5 / 5.70332580986575
all four metric deltas                       0
```

The +0.001 constant-offset fixture has saved-qg max error
`0.00100004673004`, R1 delta0 and correctly continues with **WARN**. An inconsistent
archive fixture gives R1 delta -99 pp and correctly **STOPs**, saves all a–e
diagnostics and creates no bank/correction. Exact 16504-query gate tests accept
three and reject four net changed correct queries. The CLI integration tests
cover absent metadata, no image calls, two passes for each text set, preserved
default direct behavior, read-only input hashes and float32 outputs. These are
mock tests, **not the 8/30 experiment's actual residuals or retrieval metrics**.

A separate comparison loaded the original S12 source from git commit
`52b3aa94a68af2eb9bfc1cd6b520d31179562780` and ran the same 11-bank/352-row mock
scan through both versions with reconstruction disabled: all result rows were
exactly equal, maximum metric error **0**. An out-of-range mapped-bank check
confirmed that direct mode still rejects a constant1.2 bank, while explicit
recovery processes it without clipping (all eight reduced-grid rows finite).

200×100 S11 mock (seed2026), A–G all exercised including the explicit common-min
alternative and literal infeasibility reporting:

```text
raw R1=25.5; CSLS R1=36.5
common A=48; common G=32
reciprocal pairs k5/10/20/50 = 455 / 982 / 1997 / 5000
query bootstrap mean Delta=10.9694 pp; CI=[6.5,15.5]
image-cluster bootstrap mean Delta=10.9683 pp; CI=[6.5,15.5]
identity-cluster bootstrap mean Delta=10.95495 pp; CI=[5.5,17.0]
all bootstrap repeats=10000; input SHA256 unchanged
```

200×100 S12 mock, 300×100 training bank:

```text
11 banks; 352 correction rows; all four method families
top-k direct-reference maximum error=0
IS maximum direct-reference error=4.440892098500626e-16
DIS applied/bypassed queries=2/198
independent full-matrix formula/metric comparisons: PASS
torch CPU consecutive text feature error=0; image encoder calls=0
prompted/unprompted extraction, correct qg: PASS (max error=0)
truncated text, NaN, 0.01 corruption, signed-zero mismatch: rejected
existing identical feature cache: verified, not overwritten
```

Read-only checks with the **real local annotations**, not real similarities:

```text
test: queries=16504, gallery=8252, aerial=4498, ground=3754,
      identities=2000, dual-view identities=800
derived common query sets: A=4472, G=3938
training candidate: train_reid.json, records=19954, captions=39908, ids=4840
training fields: cam_id, captions, file_path, id, split
identity split: each fold 8252 queries / 4126 images / 1000 identities
```

The four archived similarity matrices and ViT-L/14 checkpoint reside on AutoDL,
not in this Windows workspace. Their S11 results, real train-text determinism,
image-routing parity, all real S12 metrics, GPU time/memory and remote sync
remain to be verified there. A failed strict parity guard is a stopped experiment,
not a license to replace the evidence matrix.
