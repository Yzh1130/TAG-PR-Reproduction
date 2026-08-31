# S14 / S15 protocol corrections and optional S13 configuration

These are standalone analysis entrypoints. No training, model, loss, dataset,
online evaluation, S11, S12 or E3 file is modified. S13 alone has the requested
optional configuration and fresh-bank description changes. **Do not commit or
push this delivery until the owner approves it.** No dependencies are added.

## Authoritative evidence, not inferred experiment paths

The supplied evidence consists of:

- `fk32_20260831T044856_037170Z.json` (S11);
- `S9_fk.json`;
- `E3_fk32_20260831T130234425770Z.json`.

On the local Windows machine these were found in the supplied
`C:\Users\12823\Desktop\AutoDL\for_codex_0831.tgz`, not an already expanded
`_for_codex` directory. Only these three members were extracted to a new
temporary directory for read-only inspection. Neither the archive nor any
evidence file was changed.

The two existing READMEs (`README_S11_S12.md`, `README_S13_E3.md`) and the
authoritative split, sampling seeds, S9 definition and E3 provenance were read
before implementation. The scripts take explicit JSON paths and do not glob
for a convenient experiment. Matrix paths for S15 come exclusively from
`E3.fresh_files`; test annotations come from `E3.test_annotation.path`.

S9's exact definition is:

```text
P(predicted source equals true source | top-1 error, query view) / mean(same-source gallery fraction after removing ALL true-identity images | the same top-1 error queries and query view)
```

Thus its prior is averaged over **the same error queries**, not over every
query in the view stratum. A different definition stops S14 rather than being
silently reinterpreted.

## Outputs and audit trail

S14 requires a dedicated output root named `_e1_s14`, S15 `_e1_s15`.
Do not place either inside `_archive_evidence`, `_dump` or another `_e1*`
directory. Their existing output root may be reused: all JSON/log files use
exclusive creation; a collision adds a UTC timestamp and unique suffix.
Feature arrays live in a newly created exclusive run subdirectory.

Each new entrypoint records argv, parsed arguments, all random seeds,
`git_HEAD`, SHA256 first 16 hex characters and byte sizes for every input it
opens, exact-check outcomes, and a readable log. S15 also hashes every proxy
gallery image. Input paths are **not** opened automatically by the audit
constructor. Runtime failures leave a `status=stopped` report and nonzero exit
code; prior evidence and completed outputs are never overwritten. Malformed
CLI/output destinations fail before a run is created.

## S13: add the literature-default training-bank CI

`--extra-config BANK:ALPHA:K` is repeatable. For example,
`--extra-config train_full:0.5:10` appends **d** after unchanged a/b/c; later
options append e/f/... . Existing configurations and the a-minus-c paired
comparison do not change. Each extra receives query/image/identity paired
bootstrap using the existing helper and fixed, full-bank penalties.

Only the **bank basename**, case-insensitively, is checked for `_fresh`.
Such a name suppresses the inappropriate S12 test-fitted-W warning, labels
training-bank results as E3 fresh, and explicitly says the filename convention
is not provenance verification. Test-bank configuration c is still a
transductive control. A directory containing `_fresh` does not trigger this.
The original nonfresh/default invocation preserves its existing report format.
Explicit extra-config runs additionally include input-hash/git/argv audit
metadata; keep the matching E3 JSON as extraction evidence.

All corrections retain the S11/S12 score convention `S - alpha*r(g)`.
`alpha=0.5` gives the same ranking as `2*S-r(g)`.

## S14 protocols

### A: restore first, then change the gallery

The S11 JSON did not save F's identity lists. Restoration calls the **same**
`_s11_groups.balanced_group_halves(ann.query_ids, seed=F.split_seed)`;
fold i uses half i for calibration and half 1-i for evaluation. There is no
new split, seed search or tolerance fallback. For both folds, all of these
must equal the supplied S11 JSON exactly:

1. calibration/evaluation identity, image, text and query counts;
2. old full-gallery uncorrected R1/R5/R10/mAP;
3. old full-gallery crossfit-CSLS R1.

Only then is `identity_folds_seed2026.json` written (timestamped on collision),
including both identity lists, query/image indices and passed checks. The
script also gates full-gallery raw metrics against S11.raw. No expected
accuracy is hardcoded anywhere in the implementation.

The **new** A protocol uses all evaluation-fold images and excludes every
calibration-fold image from the gallery. Raw, crossfit and transductive control
use identical evaluation queries and gallery, alpha .5 and k10. A retains
the source image; the source-image exclusion belongs to controlled protocol D
and proxy validation S15. The calibration source-image and identity overlap
with the evaluation gallery must be empty.

Calibration scans draw individual descriptions without replacement at
100/500/1000/5000/full, with the three authoritative `seeds.scans` values.
All descriptions stay in the calibration fold. Each full endpoint must exactly
reproduce crossfit metrics. Reports contain all four metric means and
population standard deviations (`ddof=0`). Budgets larger than a mock fold
are explicitly unavailable, never replaced by invented results.

### B / C: identity-clustered error statistics

B recomputes S9's error-only, view-stratified point estimates and requires exact
agreement for P(same-source | error), mean conditional prior and enrichment.
Identity bootstrap draws from **all identities**, including clusters with zero
errors, keeping every member query and multiplicity. Each replicate then forms
the requested view/error conditional ratios. These are ratios of query-weighted
sums, not unweighted averages of per-identity ratios. Query-level bootstrap over
the corresponding error queries is reported alongside it.

C matches the four raw error counts at the actual schema path
`B_error_decomposition.raw.strata.all`, checks their conservation, and reports
identity-clustered percentile CIs for the four proportions and both conditional
cross-view probabilities. Undefined zero-denominator bootstrap replicates are
counted explicitly; that interval is null, with no silent redraw or removal.

### D: paired differences, no McNemar

The authoritative A/G common-query arrays are reused after exact validation.
The supplied S11 actually saved each trial's same/cross gallery indices: these
take priority and are checked against `sample_gallery` replay with
`seeds.gallery` and its original RNG consumption order. If a future equivalent
JSON lacks indices, the replay is used. Every original/drawn protocol R1 must
match the S11 JSON before paired CIs are computed.

Whole-identity paired bootstrap estimates cross-view minus same-view R1 on
each side, once for original galleries and once for each of ten common-min
draws. The report gives the mean and range of CI endpoints across draws; that
summary is **not** presented as a pooled CI. No McNemar test is computed.

## S15: training-only lock, then one frozen test phase

1. Read only training inputs. Sort unique IDs with `np.unique`, apply
   `default_rng(split_seed).permutation`, assign the first `val_identities`
   to V and all remaining IDs to B. Save both lists and image membership.
2. Import E3's existing model loader, gallery/text extractors, score writer and
   RNG/buffer guards. The classifier uses the **full** training identity count.
   Seed once, extract V images once, then V descriptions, then B descriptions.
   Text extraction must leave RNG states and model buffers unchanged. No
   reseeding/retry is used to conceal a failure. Default image batch32/workers2.
3. Verify image/caption partition conservation and V/B disjointness. Score
   proxy queries against V images; exclude each query's source image through
   the shared S11 evaluator. Queries without another positive are counted and
   excluded identically for all methods.
4. Evaluate raw and the prescribed 20 combinations. Only proxy R1 selects the
   winner; ties prefer smaller k, then smaller alpha as a deterministic final
   tie-break. Save the runner-up. Write `parameters_lock.json` **before the
   first read or hash of the E3 JSON or any test file**.
5. Read E3 provenance and reuse its saved fresh qg/full-training-bank arrays.
   Check annotation counts, matrix shapes, training-text fingerprint and
   checkpoint/config provenance. Historical checkpoint byte equality cannot
   be proved from E3's path-only provenance; this limitation is recorded.
   No test feature extraction or matrix reconstruction occurs.
6. Make one final block traversal of test qg for the frozen selected setting,
   the literature default (.5,10), and raw integrity metrics. If selected equals
   default it is evaluated only once. Raw four metrics must equal E3's
   `fresh_metrics` exactly. Test results cannot feed selection; no test scan,
   second split, alternate seed or additional tuning is performed.

JSON explicitly records: **代理验证图像为训练图像，模型在训练期已见**.
This is not an unseen-identity validation of the trained model: only the
post-processing parameter selection is isolated from the test set.

## Reproducible AutoDL commands

Run from the actual E3 repository `/root/autodl-tmp/TAG-PR` in the same activated
training environment. Copy the **three supplied JSONs unchanged** to a separate
read-only staging directory, here `/root/autodl-tmp/authority_s14_s15/_for_codex`.
That is an explicit staging location for these commands, not an inferred
original location. If they are already staged elsewhere, change only `AUTH`.
Do not use wildcards to select matrices or overwrite old evidence.

```bash
cd /root/autodl-tmp/TAG-PR
AUTH=/root/autodl-tmp/authority_s14_s15/_for_codex
S11_JSON="$AUTH/fk32_20260831T044856_037170Z.json"
S9_JSON="$AUTH/S9_fk.json"
E3_JSON="$AUTH/E3_fk32_20260831T130234425770Z.json"

python scripts/S14_identity_stats.py \
  --sim /root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy \
  --anno /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
  --s11-json "$S11_JSON" --s9-json "$S9_JSON" \
  --out /root/autodl-tmp/experiments/_e1_s14 \
  --seed 2026 --bootstrap 10000

python scripts/S15_proxy_validation.py \
  --train-anno /root/autodl-tmp/datasets/TAG-PEDES/train_reid.json \
  --config /root/autodl-tmp/cfg_l14.yaml --backbone ViT-L/14 \
  --resume /root/autodl-tmp/experiments/20260827_l14_final_keep/checkpoint_best.pth \
  --repo /root/autodl-tmp/TAG-PR \
  --out /root/autodl-tmp/experiments/_e1_s15 \
  --seed 0 --val-identities 1000 --split-seed 2026 --device cuda:0 \
  --image-batch-size 32 --num-workers 2 --text-batch-size 256 \
  --score-block-size 256 --e3-json "$E3_JSON"

# S13 literature-default identity-cluster CI on the matching E3 fresh pair.
# These paths are copied from the supplied E3.fresh_files, not re-extracted.
python scripts/S13_inductive_bootstrap.py \
  --sim /root/autodl-tmp/experiments/_e1_e3/E3_fk32_20260831T130234425770Z_sim_qg_fresh.npy \
  --bank /root/autodl-tmp/experiments/_e1_e3/E3_fk32_20260831T130234425770Z_sim_bank_g_fresh.npy \
  --anno /root/autodl-tmp/datasets/TAG-PEDES/test_reid.json \
  --out /root/autodl-tmp/experiments/_e1_s14 \
  --tag e3_fresh_literature_default --seed 2026 --bootstrap 10000 \
  --extra-config train_full:0.5:10
```

For the S12 offline-simulation main table, run the same S13 command with the
exact archived qg and the exact matching S12 bank filename. The supplied E3
JSON does not establish that S12 bank's filename, so no guessed S12 path is
provided here. The W warning remains appropriate for that nonfresh bank.

The existing E3 gg path is recorded in provenance but need not be opened:
training-bank CSLS uses only qg and bank-gallery scores.

## Local verification (mock, not new fk32 results)

```bash
python scripts/S13_inductive_bootstrap.py --help
python scripts/S14_identity_stats.py --help
python scripts/S15_proxy_validation.py --help
python -m unittest scripts.test_analysis_audit \
  scripts.test_S13_bootstrap scripts.test_S13_extra_config scripts.test_S13_fresh_bank \
  scripts.test_S14_identity_stats scripts.test_S15_proxy_validation -v
# Unmodified S11/S12/E3 regression suites:
python -m unittest scripts.test_S11_S12 scripts.test_S12_reconstruction \
  scripts.test_E3_fresh_forward -v
```

The tests use disposable 200-query x 100-gallery synthetic fixtures. They
exercise the exact replay gates, failed gates, source exclusion, all proxy
grid entries, rank selection ties, default deduplication, one test phase,
pre-lock test-access guards, RNG consumption rejection, and non-overwrite
failure reports. All mock outputs are labelled mock/synthetic. Real local
annotations were separately read: both restored fold count dictionaries
matched the supplied S11. The real A/G common-query arrays (4472/3938) and
both sides' ten saved paired-gallery draws also matched original-seed replay
exactly. These read-only checks wrote no real fold trace or analysis results.
The AutoDL matrices/checkpoint are **not** locally
available, so neither real metric-parity gates nor the GPU proxy run has been
claimed as passed. Those run through the commands above and stop on mismatch.

Integration run: **39 tests passed in 85.765 s**, including 10000-resample
positive bootstrap paths. The separate unmodified S11/S12/E3 regression run
passed **36 tests in 49.413 s**: **75 distinct tests passed** in total.
After widening S14's exception logging, its 12 S14/audit tests were rerun and
passed in 1.550 s. All three `--help` commands, compilation and
`git diff --check` passed. On Windows, if NumPy/Torch load competing OpenMP
runtimes, set `MKL_THREADING_LAYER=SEQUENTIAL` for the test process only; no
training configuration or global environment setting was changed.

Selected actual **mock-only** output:

```text
S13 default full report vs HEAD exact (timing excluded); max abs error=0
S13 extra d=train_full:0.5:10, identity bootstrap x10000:
  observed_delta_pp=9.5; bootstrap_mean_delta_pp=9.46315
  CI95=[4.0,15.0]; p=0.00039996000399960006
S14 shape=200x100; A/B/C/D and all exact-reference gates PASS
S14 two fold raw/crossfit/transductive R1: 53/50/54 and 45/61/61
S14 four error cell counts=25/9/59/56; sum=149
S14 D:20 saved-index paired draws replay EXACT; McNemar not called
S15 proxy_shape=[200,100]; V/B=50/50; G_V/Q_V/T_B=100/200/200
S15 grid_rows=20; gallery_passes=1; final_phase_calls=1
S15 mock selected={alpha:0.25,k:5}; frozen/shared-evaluator error_pp=0
```

These numbers describe synthetic fixtures, **not** a new TAG-PEDES result,
confidence interval or recommended hyperparameter.
