# External-dataset adapters and evaluation dumps

This document describes the external validation path for CUHK-PEDES,
ICFG-PEDES, AERI-PEDES, and TBAPR/TBAGPR.  It is deliberately separate from
the existing TAG-PEDES path: `main.py` dispatches here only when an external
configuration explicitly supplies `dataset_name`.  Running `main.py` without a
new configuration still selects `config/config.yaml` and does not import or
execute the external runtime.

All dataset paths below are relative to the configured
`external.data_root`, whose server default is `/root/autodl-tmp/datasets`.
Adapters parse labels and paths only; image decoding, transforms, training, and
feature extraction are centralized in `datasets/external/runtime.py`.

Implementation-audit note: this checkout and its reachable Git history do not
contain files named `cfg_l14.yaml` or `cfg_b16.yaml`.  The audit therefore used
the tracked `config/config.yaml`, current model/evaluation implementation, and
the protocol values fixed in this task; no untracked server configuration was
guessed or copied.

## Adapter contracts

### CUHK-PEDES (`cfg_cuhk.yaml`)

- Annotation: `CUHK-PEDES/reid_raw.json`.
- Image path: `CUHK-PEDES/imgs/` plus the annotation's `file_path`.
- The JSON `split` field is authoritative for `train`, `val`, and `test`.
- Training expands every caption of every train record into one independent
  image-text sample: 68,126 samples and 11,003 train identities.  Sorted
  original identities are remapped contiguously to `0..11002`.
- The test protocol has 6,156 text queries and 3,074 unique gallery images.
  The validation protocol is also exposed and has 6,158 queries, 3,078 unique
  gallery images, and 1,000 identities.  A gallery image is positive whenever
  its identity equals the query identity.

### ICFG-PEDES (`cfg_icfg.yaml`)

- Annotation: `ICFG-PEDES/ICFG-PEDES.json`.
- Image path: `ICFG-PEDES/imgs/` plus `file_path`.
- Only the JSON `split` field partitions the data.  The `train/` and `test/`
  components inside paths are inherited MSMT17 directory names and are never
  interpreted as ICFG-PEDES splits.
- Training contains 34,674 samples and 3,102 identities, remapped to
  `0..3101`.  Test contains 19,848 queries and 19,848 gallery images.
- The path
  `test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg` deliberately occurs in
  two train rows, once with identity 1 and once with identity 5.  The adapter
  asserts that this is the one and only duplicate image annotation, prints an
  `ICFG DUPLICATE SUMMARY`, and preserves both rows as independent samples.
  It never deduplicates, deletes, or merges them.

### AERI-PEDES (`cfg_aeri.yaml`)

- Train annotation: `AERI-PEDES/train_caption.json`.  Each train `aerial_img`
  already includes the `AERI-PEDES/` component and is joined directly to the
  data root.  The official W/O Ground setting creates exactly one
  `(aerial_img, caption, pid)` sample per row: 76,007 samples and 3,522
  identities.  `ground_img` is schema-checked but is never resolved, loaded,
  sampled, or otherwise used in training.
- Test annotation: `AERI-PEDES/test_caption.json`.  Test image paths receive
  the `AERI-PEDES/` prefix because this annotation stores paths differently
  from the train file.
- The official test protocol expands 6,141 captions and uses the union of
  5,525 aerial `file_paths` as gallery.  Every image with the same identity is
  positive.  Because captions are stored per identity rather than per source
  image, `query_source_gallery_idx` is `-1`.
- The 4,220 unique `ground_file_paths` are disjoint from the aerial gallery.
  They are not part of official retrieval, but their global/local features and
  labels are written as the optional extra-image dump for later mixed-gallery
  analysis.

### TBAPR / TBAGPR (`cfg_tbapr.yaml`)

This adapter follows the official AEA-FIRM merge rules rather than devising a
new union protocol.

- Train annotations are `TBAGPR/AG-ReID/agtext_train.json` (8,154 rows) and
  `TBAGPR/G2APS-AG/g2aps_train.json` (23,186 rows).  The two components of
  `img` are joined with `/` below the respective
  `TBAGPR/AG-ReID/` or `TBAGPR/G2APS-AG/` prefix.
- G2APS train identities are offset by the AG-ReID train identity count (199),
  and G2APS `image_id` values are offset by the AG-ReID train row count
  (8,154), before a final contiguous train-class remap.  This produces 31,340
  samples and 1,180 classes (`0..1179`).  Train `pair_img` is never used.
- At test time, G2APS `image_pids` and `caption_pids` are offset by the
  AG-ReID test identity count (189).  Direct concatenation yields 3,200 text
  queries, 3,200 gallery rows, and 529 identities.
- The official gallery is row based, not unique-image based.  In particular,
  the 2,051 G2APS rows contain only 1,934 unique paths, yet every row remains
  rankable and `query_source_gallery_idx` follows row alignment.
- Test `pair_img_paths` are excluded from the official gallery.  Their features
  are exported separately with `view="pair"`; official rows use `view="img"`.
  Both query and gallery metadata also retain the `AG-ReID` or `G2APS-AG`
  source-subset label.

Each adapter writes separate contiguous PID mappings for the train classifier
and the selected evaluation split.  The complete mappings are preserved in
`pid_remap_table.json`; original numeric labels are never assumed to be dense.

## Training protocol

The four root configurations differ only in `dataset_name`, annotation paths,
published train sample count, and classifier class count.  Their shared setup
is:

- CLIP ViT-B/16 at 224 x 224; `CLIP_B16_CHECKPOINT` supplies the original CLIP
  `.pt` path at runtime;
- HR-MoE disabled (`disable_moe: true`). `view_ratio: 1.0` and
  `ortho_ratio: 1.0` retain the paper's internal ViT-B/16-without-HR-MoE
  configuration verbatim; these ratios have no effect when MoE is disabled
  because no expert/view-routing block exists and the view token is removed
  before the standard ViT blocks;
- global batch size 320 (the runtime requires exact divisibility by DDP world
  size), 10 epochs, two data workers, and maximum caption words/context length
  77;
- AdamW, learning rate `1e-4`, weight decay `0.02`, and betas `(0.9, 0.98)`.

There is no per-epoch external validation and no best-R@1 selection.  Training
runs all ten epochs, evaluates once after epoch 10, and saves only
`checkpoint_last.pth` (model, optimizer, config, and final epoch).  An
`--eval_only --resume` run loads a saved checkpoint and performs one dump on
the requested split.  CUHK validation can be selected with
`--external_split val`.

## Immutable evaluation dump contract

Every normal external evaluation creates a fresh, non-overwriting directory:

```text
experiments_ext/{dataset}_{backbone}_{split}_seed{seed}_{UTC timestamp}/
```

Output roots beneath `_archive_evidence/`, `_dump/`, or any `_e1*` directory
are rejected.  The following files are required:

- `feats_text_global.npy`: normalized `float32 [nq, D_global]` text global features;
- `feats_text_local.npy`: normalized `float32 [nq, D_local]` text TSE features;
- `feats_img_global.npy`: normalized `float32 [ng, D_global]` gallery global features;
- `feats_img_local.npy`: normalized `float32 [ng, D_local]` gallery local features;
- `sim_qg.npy`: `float32 [nq, ng]` final similarity matrix;
- `meta.json`, `output.json`, and `pid_remap_table.json`;
- `config_used.yaml`, `git_commit.txt`, `run_command.txt`, and
  `environment_summary.txt`.

The external score is computed through
`misc.eval.compose_global_local_similarity`, a side-effect-free helper placed
next to the existing TAG-PEDES expression and implementing that expression
exactly.  The legacy TAG-PEDES call site itself is deliberately left
untouched, so its default execution path cannot change:

```text
sim_qg = 0.5 * (text_global @ image_global.T)
       + 0.5 * (text_local  @ image_local.T)
```

All four inputs are L2-normalized before persistence.  Scoring is blockwise,
then the saved four feature arrays are loaded again and used to recompute every
score.  `np.array_equal` must pass for every block; any one-bit discrepancy
aborts the dump before `meta.json` is committed.  This strict check is recorded
under `strict_score_recomputation`.

`output.json` contains the top 50 gallery row indices, one-based ranks, and
identity-match flags for every query.  `meta.json` records dataset, split,
protocol, backbone, seed, epoch, `nq`, `ng`, feature dimensions `D_global` and
`D_local`, remapped
query/gallery PID lists, source-gallery row indices,
`gallery_identity_image_index` (remapped PID to ordered global gallery rows),
optional view/source labels, ordered gallery paths, a digest of ordered query texts, the PID-map
path, argv, Git HEAD, checkpoint, annotation paths and SHA-256 prefixes, the
score definition, parity result, metrics, and optional-file inventory.

By default every adapter also emits the complete training-caption bank as
`feats_bank_text_global.npy` and `feats_bank_text_local.npy` (68,126 CUHK;
34,674 ICFG; 76,007 AERI; 31,340 TBAPR rows).  AERI ground images and TBAPR pair
images additionally produce `feats_extra_img_global.npy`,
`feats_extra_img_local.npy`, and `extra_meta.json`.  CUHK and ICFG have no
extra-image protocol, so no extra-image files are created for them.
Bank and extra-image global features use `D_global`; their local features use
`D_local`.  The two branch dimensions are validated independently and need not
be equal (the current ViT-B/16 path produces 512 and 1024 respectively).
For AERI, `extra_meta.json` preserves the annotation-order ground paths together
with their remapped `pid` and `view="ground"` labels.

## Engineering-only smoke mode

`--smoke N` trains exactly `N` optimizer steps; spelling `--smoke` without an
integer selects 200 steps.  Before/while running, it verifies configured train
sample count, classifier width, a stable 2,000-path sample with resolution
ratio exactly 1.0, PID range, finite loss, and evaluation feature/PID row
alignment.  It prints each train batch's image shape, PID range, finite-loss
flag, peak allocated GPU memory, and load time; it also reports the fraction of
captions exceeding the CLIP context and checks the first eight evaluation
batches.

Smoke mode is explicitly result-blinded: it never constructs rankings and
must not compute or print R@1/R@5/R@10/mAP, accuracy, top-1 distributions, or
any other retrieval result.  `smoke_engineering.json` recursively permits only
boolean and integer leaves and contains engineering checks/counts only.  The
runtime rejects result-like field names before writing it.

`--smoke_save_checkpoint` is opt-in and valid only together with `--smoke`.
After the requested optimizer steps it writes `checkpoint_smoke.pth` in that
run directory using exactly the same `model` / `optimizer` / `config` / `epoch`
schema as `checkpoint_last.pth`; it can be passed directly to
`--eval_only --resume`. Without the flag, smoke behavior and output files are
unchanged.

External evaluation normally prints R@1/R@5/R@10/mAP and also preserves them
under `meta.json.metrics`. `--quiet_metrics` leaves all files unchanged but
replaces the terminal metric line with `EVALUATION COMPLETE` and the strict
`array_equal` feature-to-similarity self-check result.

## Reproducible server commands

Set the checkpoint explicitly; the configuration intentionally does not guess
where a server stores CLIP weights:

```bash
cd /path/to/TAG-PR-Reproduction
export CLIP_B16_CHECKPOINT=/absolute/path/to/ViT-B-16.pt
```

The four runtime configurations already use
`/root/autodl-tmp/datasets`. `TAGPR_EXTERNAL_DATA_ROOT` is only a test-suite
override when real annotations are stored elsewhere.

CPU annotation/contract tests (these inspect every real adapter path):

```bash
python -m unittest \
  tests.test_cuhkpedes_ext tests.test_icfgpedes_ext \
  tests.test_aeri_ext tests.test_tbapr_ext -v
python -m unittest tests.test_external_configs -v
```

Engineering smoke runs (single GPU shown; omit `200` to use the same default):

```bash
torchrun --standalone --nproc_per_node=1 main.py --config cfg_cuhk.yaml --smoke 200
torchrun --standalone --nproc_per_node=1 main.py --config cfg_icfg.yaml --smoke 200
torchrun --standalone --nproc_per_node=1 main.py --config cfg_aeri.yaml --smoke 200
torchrun --standalone --nproc_per_node=1 main.py --config cfg_tbapr.yaml --smoke 200
torchrun --standalone --nproc_per_node=1 main.py --config cfg_tbapr.yaml \
  --smoke 200 --smoke_save_checkpoint
```

Full ten-epoch runs:

```bash
torchrun --standalone --nproc_per_node=1 main.py --config cfg_cuhk.yaml
torchrun --standalone --nproc_per_node=1 main.py --config cfg_icfg.yaml
torchrun --standalone --nproc_per_node=1 main.py --config cfg_aeri.yaml
torchrun --standalone --nproc_per_node=1 main.py --config cfg_tbapr.yaml
```

Direct evaluation and CUHK validation examples:

```bash
python main.py --config cfg_aeri.yaml --eval_only \
  --resume /absolute/path/to/checkpoint_last.pth
python main.py --config cfg_cuhk.yaml --external_split val --eval_only \
  --resume /absolute/path/to/checkpoint_last.pth
```

Numerical regression of the untouched TAG-PEDES path (run once on each Git
state with the same environment, checkpoint, and config, then compare the four
full-precision metrics and `sim_qg.npy` file digest):

```bash
TAGPEDES_CHECKPOINT=/absolute/path/to/checkpoint_best.pth \
TAGPEDES_CONFIG=config/config.local.yaml \
TAGPEDES_OUTPUT_ROOT=/root/autodl-tmp/tagpedes_regression_before \
bash scripts/regression_tagpedes.sh
```

Optional environment variables are `TAGPEDES_NPROC` (default `1`) and
`TAGPEDES_PYTHON` (default `python`).  The harness explicitly disables CSLS and
k-reciprocal re-ranking, requests a float32 dump, and hashes the exact persisted
`sim_qg.npy` bytes.

When the four real datasets are unavailable, the same adapter test modules run
their temporary-directory mock fixtures and mark only real-data tests skipped:

```bash
python -m unittest discover -s tests -p 'test_*_ext.py' -v
```

The mocks cover split selection, path prefixes, caption expansion, PID remaps,
query/gallery row alignment, ICFG's conflicting duplicate, AERI's W/O Ground
and disjoint extras, and TBAPR's offsets and repeated gallery rows.  They assert
only schema, shapes, counts, sets, and conservation invariants; no mock test
hard-codes or fabricates a retrieval accuracy.
