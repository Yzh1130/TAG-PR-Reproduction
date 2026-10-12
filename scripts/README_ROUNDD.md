# Round D: archived table rendering and Figure 5

These tools read saved results only. They do not run training, alter metrics,
or write into original experiment directories. Mock tests are not real results.

## Table renderer

`scripts/revision/render_revision_tables.py` has eleven independent subcommands.
Each requires explicit input paths and `--out NEW_OUTPUT_DIRECTORY`.
`--decimals` defaults to 2. Signed differences use `+` / Unicode `−`, recorded
intervals use `[lower, upper]`, and missing fields use `n/a` with reasons.
Each Markdown file ends with input paths, SHA-256 prefixes, and script git HEAD.
`render_manifest.json` additionally records argv, timestamps, and file hashes.

The renderer produces one Markdown file per invocation and accumulates the
files in `all_tables.md`. Existing individual tables are never overwritten;
repeated names get a UTC timestamp. Only the renderer's own manifest-verified
collection is updated. Manual changes to its tables/combined file cause an
error, rather than being overwritten. Use a new output directory to start a
different collection. Inputs are never changed.

```powershell
$py = 'E:\Study\Anaconda\python.exe' # replace with your Python
$data = 'C:\Users\12823\Desktop\TAG-PR-revision-data'
$out = "$data\tables_out" # new analysis output, not an input archive directory

& $py scripts/revision/render_revision_tables.py table5 --json "$data\R1_table5_l14\R1_table5_clusters.json" --label 'ViT-L/14' --out $out
& $py scripts/revision/render_revision_tables.py table5 --json "$data\R1_table5_b16\R1_table5_clusters.json" --label 'ViT-B/16' --out $out
& $py scripts/revision/render_revision_tables.py table3 --json "$data\R1_joint_l14\R1_joint_enrichment.json" --label 'ViT-L/14' --out $out
& $py scripts/revision/render_revision_tables.py table3 --json "$data\R1_joint_b16\R1_joint_enrichment.json" --label 'ViT-B/16' --out $out

$ext = "$data\roundB_ext_20261012T101926Z"
& $py scripts/revision/render_revision_tables.py table11 --aggregate "$ext\ext_seed_summary\S7_aggregate_seeds_seed012\S7_aggregate_seeds_seed012.json" --out $out
& $py scripts/revision/render_revision_tables.py table12 --s7b "$ext\ext_s7b\S7b_distractor_seed0\S7b_distractor_seed0.json" --s7b "$ext\ext_s7b\S7b_distractor_seed1\S7b_distractor_seed1.json" --s7b "$ext\ext_s7b\S7b_distractor_seed2\S7b_distractor_seed2.json" --out $out
& $py scripts/revision/render_revision_tables.py pairs --csv "$data\collector\pairs_last_e2_last.csv" --csv "$data\collector\pairs_last_e2_seed0_hist.csv" --metric R1 --out $out
& $py scripts/revision/render_revision_tables.py table-r3 --json "$data\R3\R3_finegrained_causes.json" --out $out
& $py scripts/revision/render_revision_tables.py runs --json "$data\collector\runs_all_runs.json" --include e2_ --include e3_ --out $out
```

S13 has no supplied real input in Round D. After running S13 with
`--summary-table`, explicitly supply its **summary** JSON (not the legacy a/b/c
bootstrap JSON):

```text
python scripts/revision/render_revision_tables.py s13 --summary "ViT-L/14=ABSOLUTE_L14_SUMMARY_JSON" --summary "ViT-B/16=ABSOLUTE_B16_SUMMARY_JSON" --out NEW_OUTPUT_DIRECTORY
```

### Round E: additional archived-result views

Use explicit `label=PATH` inputs; labels identify models, not guessed filenames.
The S7d view permits repeating a model label for distinct recorded training
seeds. All existing subcommands retain their original inputs and behavior.

```text
python scripts/revision/render_revision_tables.py table-e5 --json "ViT-L/14=ABSOLUTE_L14_R2_ROUTER_EVAL_JSON" --json "ViT-B/16=ABSOLUTE_B16_R2_ROUTER_EVAL_JSON" --out NEW_OUTPUT_DIRECTORY
python scripts/revision/render_revision_tables.py table-e6 --root "ViT-L/14=ABSOLUTE_L14_E6_RUN_ROOT" --root "ViT-B/16=ABSOLUTE_B16_E6_RUN_ROOT" --out NEW_OUTPUT_DIRECTORY
python scripts/revision/render_revision_tables.py table-s7d --s7d-json "ViT-B/16=ABSOLUTE_S7D_SEED0_JSON" --s7d-json "ViT-B/16=ABSOLUTE_S7D_SEED1_JSON" --s7d-json "ViT-B/16=ABSOLUTE_S7D_SEED2_JSON" --out NEW_OUTPUT_DIRECTORY
python scripts/revision/render_revision_tables.py s13 --compare --summary "ViT-L/14=ABSOLUTE_L14_SUMMARY_JSON" --summary "ViT-B/16=ABSOLUTE_B16_SUMMARY_JSON" --out NEW_OUTPUT_DIRECTORY
```

- `table-e5` reads `R2_router_eval.py`'s saved `blocks`, `block_order`,
  `main_report_block`, `n_images`, `n_gumbel` and `cam_source_table`. Each router
  block is identified explicitly; classification proportions are displayed as
  percentages. The producer does not serialize a pure-source assertion Boolean:
  the table explicitly labels its check as a recheck of saved cam/source counts,
  not as a recorded Boolean. Source-specific accuracy is a separate table.
- `table-e6 --root` must name the **single run root** printed by
  `e6_router_randomness.sh`, containing `router_randomness.json`. That summary
  records each run's `evaluation_json` path, metrics, seed, differences from
  argmax, and the maximum pairwise differences observed among these N Gumbel
  repetitions. The renderer reads this summary, not checkpoints or guessed log
  filenames; saved original run paths are audit references, not reopened inputs.
  It does not call the observed difference a bound on other seeds.
- `table-s7d` reads `protocols` and `paired_gallery_differences` from
  `scripts/ext/S7d_aeri.py`. Controlled-gallery rows are five-draw means/stds,
  **not five training seeds**. Only descriptive mean/std reductions of complete
  stored draws are allowed. No interval endpoints are averaged. A pooled CI for
  a controlled mean correction gain is not saved by this producer and remains
  `n/a`; the separate paired-gallery CI conditional on five draws is preserved.
- `s13 --compare` uses the same four validated summary settings as the existing
  `s13` command, with just R@1 and ΔR@1 [identity CI] per model. Without the flag,
  the original full four-metric summary view is retained.

### Stored-field conventions

- Table 5 uses six main rows plus identity/zero-positive accounting and rules
  a/b/c. Direction mean/std is reduced only from all ten stored
  `authoritative_trials`, with population std (`ddof=0`). Missing trials stay
  `n/a`. `table5_reference` displayed differences and float64 per-query
  bootstrap differences/intervals are separate columns, not silently equated.
- Table 3 reads nested `estimate` / `ci95_percentile` records. Global counts
  come from `counts`; conditional results and pure-view applicability follow
  the producer. No count is inferred from a share.
- Table 11 reads archived per-seed records and archived `summary` statistics.
  Baseline sign counts are inapplicable; gain signs are preserved.
- Table 12 reads training seeds only from
  `_meta.provenance.seeds.run_seeds[dataset]`. Gallery seeds are not training
  seeds. Dataset-level `spearman_excess_skew_vs_D` and fit-level
  `spearman_vs_excess_skew` are distinct, clearly named columns.
- Pairs accepts repeated `--csv`, `--metric {R1,R5,R10,mAP}` (default R1).
  Merge keys include pair name, last/best checkpoint scope, and seed. Exact
  duplicate rows are deduplicated; conflicting rows stop. Last/best are never
  pooled. Descriptive summaries reduce stored paired values only; absent Δ is
  not computed from B−A. `unpaired` remains `n/a`.
- R3 R@1 is stored in percent; its identity CI and SSSV shares are proportions
  and are converted to percent. Attribute Jaccard values remain unscaled.
  Empty Market quartiles and null endpoints remain `n/a`.
- Runs reads collector JSON directly, without rescanning experiments or
  loading checkpoints. Repeated `--include` uses the collector prefix rule.
- S13 reads four fixed `--summary-table` rows, with explicit model labels;
  configuration values must agree with the recorded row. The baseline has no
  gain interval. No interval is inferred.

## Figure 5

The existing figure tool consumes three S7d files and the saved aggregate.
All point values come from inputs; no fitted trend is plotted. Controlled
gallery means and paired identity intervals retain their saved conditional
scope. Differences between aggregate precision and S7d displayed precision
are printed as warnings, and the plot uses S7d values.

```powershell
& $py scripts/revision/plot_fig5_aeri.py --s7d-json "$ext\ext_s7d\S7d_aeri_seed0\S7d_aeri_seed0.json" --s7d-json "$ext\ext_s7d\S7d_aeri_seed1\S7d_aeri_seed1.json" --s7d-json "$ext\ext_s7d\S7d_aeri_seed2\S7d_aeri_seed2.json" --aggregate-json "$ext\ext_seed_summary\S7_aggregate_seeds_seed012\S7_aggregate_seeds_seed012.json" --font C:\Windows\Fonts\simhei.ttf --out "$data\fig5_out" --tag roundD
```

Font properties are applied before layout. ASCII negative tick labels are
scoped to figure layout/export to avoid SimHei's missing Unicode-minus glyph;
global matplotlib settings are restored. Missing font falls back to English
labels with a warning. Outputs are PNG (300 dpi), PDF, point CSV, JSON and log.
Existing figure output directories are preserved with timestamped new runs.

## Tests

```text
python -m unittest tests.test_roundd_fig5 tests.test_roundd_tag_tables tests.test_roundd_ext_tables tests.test_roundd_collection_tables tests.test_roundd_renderer_cli -v
python -m unittest tests.test_rounde_renderer -v
python -m unittest discover -s tests -v
```

Renderer tests use handwritten archived-schema mock values to verify exact
formatting, signs, population statistics, missing handling and conflicts.
Figure tests reuse the existing 200-query × 100-gallery S7d mock pathway.
No mock output is presented as a measurement from the real data.
