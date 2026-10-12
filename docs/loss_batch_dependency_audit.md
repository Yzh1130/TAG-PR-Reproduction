# Round A: loss, batch dependency, and final-weight provenance audit

Audit baseline: `8bf2616403145f5a0e64747ddf9914e0a9e0fcdf` on
`infra/exp-management`. All source line numbers below refer to that commit,
not to the subsequent Round A implementation. This report records source
semantics; it does not claim a completed training or CUDA/DDP experiment.

## 1. Actual training entry and effective defaults

TAG-PEDES uses `main.run` and `model.tbps_model_MoE.CLIP.forward`.
`main.py:1724-1729` executes the forward under CUDA autocast and sums every
returned value whose key contains `"loss"`. Metrics without that substring
are excluded from the objective. External datasets call the same model through
`datasets/external/runtime.py:534-545`, where the same key-based loss sum is
used. A metric registration alone does not establish a training loss.

The checked-in defaults in `config/config.yaml:11-51,64-73` are:

| Setting | Default | Participation |
| --- | --- | --- |
| `nitc_ratio` | `1.0` | NITC global + TSE, always constructed |
| `ritc`, `ritc_ratio`, `ritc_eps` | `true`, `1.0`, `0.01` | RITC global + TSE |
| `id`, `id_ratio` | `true`, `0.5` | image/text ID classification |
| `view_ratio`, `ortho_ratio`, `disable_moe` | `1.0`, `1.0`, `false` | MoE view/orthogonal supervision |
| `sdm_ratio`, `tal_ratio` | `0.0`, `0.0` | optional, inactive by default |
| `tal_margin`, `tal_tau` | `0.1`, `0.02` | TAL settings if activated |
| `use_bridge`, `bridge_cv_ratio`, `bridge_tg_ratio` | `false`, `0.0`, `0.0` | optional, inactive by default |
| `nrw_mode`, `nrw_ema`, `nrw_warmup_epochs` | `off`, `0.99`, `3` | optional NITC sample weights |
| `dhn_margin` | `0.0` | optional NITC candidate-logit adjustment |
| `use_gather`, `softlabel_ratio` | `true`, `0.5` | distributed candidates / teacher mixture |

`main.py:368-568` applies explicit CLI overrides. Consequently these defaults
do not prove the effective parameters of an archived experiment; its
`config_used.yaml` and command must be checked. External config
`cfg_cuhk.yaml:9-44` (and its three peers) keeps NITC/RITC/ID weights
`1.0/1.0/0.5` but sets `disable_moe=true`; its view/ortho ratios remain `1.0`.

`balance_loss` and `router_z_loss` occur in the meter dictionary
(`main.py:1595-1611,1748-1749`) but `CLIP.forward:388-605` does not return
those keys. Likewise TSE losses are combined into `itc_loss`/`ritc_loss`,
not returned as `itc_tse_loss`/`ritc_tse_loss`. `compute_simclr_loss`
(`model/tbps_model_MoE.py:1072-1086`) is defined but not called by this forward.
These are not additional active objective terms in the audited entry.

## 2. Shared feature, target, and parameter chain

`model/tbps_model_MoE.py:404-423` obtains image/text global and dense
features, L2-normalizes global features, and gathers normalized features.
`495-500` obtains and normalizes TSE features and gathers them. Denote local
anchor count by `B` and gathered candidate count by `N`. Their pair matrices
have shape `[B,N]`; in a single non-distributed process `N=B`.

The exact ID target construction (`437-441`) is:

```python
idx = ids.view(-1, 1)
gathered_ids = self.all_gather(ids)
idx_all = gathered_ids.view(1, -1)
pos_idx = torch.eq(idx, idx_all).float()
sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
```

Every same-ID candidate is positive, including duplicate pairs/captions.
Each row distributes unit mass equally across all same-ID candidates. There
is no one-hot assumption and no epsilon in this normalization. In a self
batch each anchor has its own positive. A restricted block must rebuild this
target from the IDs in that block; slicing a previously generated target
would preserve the wrong positive counts until explicitly renormalized and
would not rebuild the teacher distribution.

The learnable common temperature is `CLIP.logit_scale`:
`342-352` initializes its log value to `log(1/0.07)`; `434-435` computes
`exp()` and clamps its `.data` to at most `100`. Thus the audited code uses
the exponential scale with that exact clamp operation. No batch estimate
of temperature is present. Teacher features are current-forward features
detached at `458-461` and `499-500`; there is no momentum teacher, queue, or
cross-step teacher bank in NITC.

Gradients can reach the visual/text backbone (including `visual.proj` and
`encode_text.text_projection`), shared `logit_scale`, and, on TSE routes,
`visual_emb_layer`/`textual_emb_layer` projections and MLPs. The TSE modules
are constructed at `374-381`; their layers and token pooling are in
`model/CrossEmbeddingLayer_tse.py:38-124`. Local feature dimension defaults
to `1024` (`59,96`) while global dimension is the backbone `embed_dim`.
The TSE MLP includes `BatchNorm1d` over flattened sample-token features
(`47,50-54`), so producing features in separate forwards is not generally
equivalent to splitting losses after one full forward.

## 3. Per-loss dependency audit

### 3.1 NITC, global and TSE

Calls: `model/tbps_model_MoE.py:463-524`. Original unweighted function:
`973-1002`; weighted variant: `1004-1070`.

Inputs are normalized image/text local anchors, gathered image/text
candidates, their detached teacher copies, row-normalized ID targets,
`alpha`, and `logit_scale`; the weighted variant additionally receives
per-anchor detached weights. Optional DHN additionally receives a detached
ID/domain-derived candidate mask and scalar margin.

Within `torch.no_grad()` the teacher independently computes both directions:

```python
sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
sim_i2t_targets = alpha * F.softmax(sim_i2t_s, dim=1) + (1 - alpha) * sim_targets
sim_t2i_targets = alpha * F.softmax(sim_t2i_s, dim=1) + (1 - alpha) * sim_targets
```

The teacher softmax is over candidates (`dim=1`, `[B,N]`). There is no
epsilon smoothing. Each direction is soft-target cross entropy, with a mean
over local anchors, then the two directions are averaged. The two feature
routes are **summed**, then multiplied by `experiment.nitc_ratio` (`524`).

`alpha` is supplied by the training loop, not estimated from similarities.
`main.py:1719-1722` ramps `softlabel_ratio * min(1,i/len(train_loader))`
during epoch index `0`; later epochs use `softlabel_ratio` directly. Default
maximum is `0.5`. The first update uses `alpha=0`; the last update of epoch 0
does not quite reach `0.5` when the epoch has a finite positive iteration
count. External runtime uses the same formula at `748-752`.

DHN (`443-456,942-970`) sets `same_domain & ~same_id`, detaches it, adds
`margin*mask` **before** multiplying by logit scale in both student
directions. Teacher logits are not margin-adjusted. Mask and teacher
probabilities are block-reconstructible from domain/ID/features.

NRW is a sample-weight mechanism, not the ID/teacher target generator:
`608-741` computes detached positive-pair similarities and weights. `fixed`
is per-sample; `view_norm`/`global_norm` update EMA means/variances, with
distributed moment reduction at `720-734`. Warmup returns weights of one
while updating statistics (`706-708`). The original weighted NITC mean is
`sum(w*per_sample_loss)/(sum(w)+1e-6)` (`1064-1070`). For D1 the original
full-forward NRW computation must happen once; its already computed
per-sample weights are sliced, and EMA must not be updated again per block.
This preserves its weight input while rebuilding the candidate-dependent
targets in the original weighted function.

### 3.2 RITC, global and TSE

Calls: `574-579`; original function: `837-846`. Inputs are local normalized
image/text anchors, gathered opposite-modality candidates, scale, and the
same ID target matrix. Log probabilities are over candidate columns.

```python
target_log = (sim_targets + self.eps).log()
kl_img = F.kl_div(target_log, img_log, log_target=True, reduction='batchmean')
kl_txt = F.kl_div(target_log, txt_log, log_target=True, reduction='batchmean')
```

PyTorch's first argument is the supplied target log and its second argument
is model log probability. The exact implementation therefore accumulates
`P_model * (log P_model - log(target+eps))`, averaged over local anchor count;
do not silently reverse its arguments. `eps=0.01` comes from
`experiment.ritc_eps` via `clip_vitb:1271`. Adding epsilon does not
renormalize each target row. Directions are averaged; global and TSE terms
are summed and multiplied by `ritc_ratio`. No teacher mixture or epoch
warmup is inside RITC.

### 3.3 SDM, global and TSE (optional)

Calls: `526-546`; function: `848-866`. Inputs/candidates and ID target are
the same global and TSE sets as NITC. For both directions it computes
`P_model*(log P_model-log(sim_targets+1e-8))`, sums candidate columns, and
averages anchors. Directions are **summed**, not divided by two. Global
and TSE losses are summed and multiplied by `sdm_ratio`; zero skips both
calls. No teacher/warmup. Its targets can be regenerated per block from IDs.

### 3.4 TAL, global and TSE (optional)

Calls: `548-572`; functions: `868-940`. Same-ID positions in `[B,N]` are
positives and all other positions negatives. Positive weights are detached
`softmax(similarity/tau)` within the positive set (`923-928`); negatives use
`tau*logsumexp(negative_similarity/tau)` (`930-937`). Row losses are
`relu(margin - positive_similarity + negative_upper_bound)`.

Rows with no negatives are excluded from the original mean; if the whole
block has no negatives the function returns differentiable zero
(`914-921`). Both directions are averaged (`904`); global and TSE are
summed and multiplied by `tal_ratio`. It uses cosine similarities without
the shared logit scale, positive `tau` (default `0.02`), and margin `0.1`.
Its calculation explicitly disables autocast and uses float32 (`884-891`).
The existing internal positive/negative mask belongs to the original TAL
formula; it is not a full-batch candidate-restriction mask. Block invocation
must regenerate masks and positive softmax from that block's IDs/features.

### 3.5 ID classification

`581-585` sends **unnormalized** image/text global features through one
shared trainable `nn.Linear(embed_dim,num_classes)` (`383-386`). Each
direction uses `F.cross_entropy(logits,ids)` (default anchor mean), their
average is multiplied by `id_ratio=0.5`. It uses all identity head classes,
not batch candidates; duplicate IDs do not alter a target distribution.
No epsilon, teacher, or warmup. D1 must not split/recompute this loss.

### 3.6 View classification and orthogonality

Per MoE block: `model/base_transformer.py:109-128`. The router takes the
last view-token feature after that block's attention and `ln_2`:
`view_logits=self.img_router(x[:,-1,:])`. When `training` and `cam_id` is
provided and view computation enabled, CE uses the two cam classes, with
default sample mean. Trainable input backbone and `img_router` are involved;
no candidate IDs or teacher/epsilon/warmup.

Orthogonality uses samplewise absolute cosine between normalized CLS and
view-token features at that block:

```python
cos_sim = torch.cosine_similarity(F.normalize(x[:, 0, :], dim=-1), F.normalize(x[:, -1, :]), dim=-1).abs()
ortho_loss = (cos_sim - 0.001).clamp(min=0.0).mean() * 100
```

Each loss is averaged across active MoE blocks by `289-293`, then multiplied
by its experiment ratio in `CLIP.forward:424-432` (a ratio of exactly `1.0`
does not add a multiply operation). They have no batch candidate dependence.
`disable_moe` clears expert blocks at `225-239` and removes the view token
at `250-251`; no view/ortho values are produced, and the forward's fallback
is zero. Ratios of zero disable computation at block construction
(`229-236`) and omit the corresponding ret key. D1 leaves both untouched.

### 3.7 Bridge CV/TG (optional local subset contrastive losses)

`587-603,749-835`. Only when `use_bridge` and a positive ratio are set,
rows with `has_bridge` are encoded once (`760-783`). They use `cam_id=1`,
normalized bridge global features, corresponding primary image/text global
features, and local subset IDs. The bridge feature is not gathered.

`788-789` rebuilds same-ID uniform row targets on that subset. Both CV and
TG use bidirectional soft-target CE, average the two directions, then apply
their own ratio. No teacher, epsilon, or warmup. Zero bridges return a
differentiable zero (`763-769`). `encode_bridge_image:809-816` takes only
image features from the visual return; branch auxiliary view/ortho losses
are not accumulated.

For D1's continuous full-batch blocks, the bridge rows must retain their
**original main-batch block membership**; do not compact the full bridge
subset first and create different block boundaries. Rebuild the subset's
targets per original block and aggregate by that block's bridge-anchor
count divided by total bridge-anchor count. Empty subsets contribute zero.
The original bridge forward/zero-ratio branch remains unchanged. This
choice limits candidates while preserving one bridge forward and original
sample selection; it is an explicitly defined new enabled-mode protocol.

## 4. Candidate/gather and block feasibility

`CLIP.all_gather:1162-1167` returns input unchanged when `use_gather` is false
or `is_using_distributed()` is false. **The latter helper is a historical
constant that returns `True`** (`misc/utils.py:21-22`); it does not check
process-group initialization. `main.py:2021` calls
`init_distributed_mode`, which unconditionally reads `RANK/WORLD_SIZE/
LOCAL_RANK` and initializes the process group (`misc/utils.py:55-62`).
Therefore the actual TAG single-process path uses a one-rank initialized
group (e.g. torchrun); with `use_gather=true`, `AllGather.apply` still runs
collectives and flattens rank/sample axes, but candidate count is unchanged.
Ordinary `python main.py` without those environment variables is not a
supported baseline training launch. External runtime explicitly sets
`use_gather=false` if no process group exists (`121-124`), allowing an
ordinary single-process external run. Fixed-feature CPU tests may bypass
gather, but must not characterize that as the real one-rank TAG launch.
Custom autograd gather details are separately recorded in
`docs/ddp_vs_single_gpu_semantics.md`.

| Objective/input | Can original block loss regenerate candidate targets? | Required treatment |
| --- | --- | --- |
| NITC global/TSE | yes | rebuild ID target and detached teacher softmax inside each block |
| RITC global/TSE | yes | rebuild ID target, retain exact epsilon and KL argument order |
| SDM global/TSE | yes | rebuild ID target; retain sum of directions and both route weights |
| TAL global/TSE | yes | pass block IDs; original positive weights/negative upper bound regenerate |
| Bridge CV/TG | yes | preserve main-row block membership; rebuild local bridge ID target |
| NRW | weights are not targets | update original full-batch statistics once, then slice detached weights |
| DHN | yes | rebuild detached same-domain/different-ID mask per block |
| ID/view/ortho | no batch candidate target | preserve original full-forward loss |

No audited candidate loss needs a queue, momentum teacher, or
candidate-external teacher distribution. NRW statistics are full-batch
state, but are upstream weight inputs rather than contrastive target
generation; preserving that original computation once avoids a target
reconstruction blocker. There is therefore no identified D0 condition
requiring suspension of D1 for the audited fixed-feature protocol.

`sum(n_block/N * L_block)` is the required independent-block definition.
For NRW's original weighted mean, this is not generally identical to one
full-batch weighted mean because each block's denominator is
`sum(w_block)+1e-6`; that observation does not authorize changing the
required aggregation or loss weights. For TAL, a block with no negative
rows uses its original zero/valid-row semantics; block counts for the outer
aggregation still follow the specified anchor-count rule.

## 5. Training schedule, optimizer, precision, and data batching

* LR: `misc/build.py:100-120` builds one NumPy array of length
  `schedule.epoch * schedule.niter_per_ep`. Warmup length is
  `epoch_warmup*niter_per_ep`; `np.linspace(lr_start,lr,warmup_iters)`
  precedes a cosine ending toward `lr_end`. Defaults are 10 epochs, one
  warmup epoch, `lr=1e-4`, start `1e-6`, end `5e-6`
  (`config/config.yaml:53-62`). It is indexed **per optimizer iteration**
  via a monotonically increasing `it` (`main.py:1698-1722,1781`), although
  total length/warmup are configured in epochs.
* Optimizer: AdamW, `misc/build.py:123-145`; decay excluded for rank<2,
  bias/ln/bn parameters. TSE parameter names receive LR ratio ×10;
  classifier/cross/mlm_head receive `schedule.ratio_factor` (default `1`).
  Default decay `.02`, betas `[.9,.98]`, eps `1e-8` are optimizer epsilon,
  distinct from RITC/SDM smoothing.
* AMP: CUDA autocast at `main.py:1724`, CUDA GradScaler at `1703`,
  scaled backward/step/update at `1764-1771`; no gradient clipping call
  exists in the audited loop. Gradients are cleared after each update
  (`1779-1780`); no gradient accumulation is implemented there. TAL has
  the explicit float32 island described above. TSE's training forwards
  also explicitly cast selected token features to half
  (`CrossEmbeddingLayer_tse.py:86-89,116-118`); changing forward batching
  can change BatchNorm statistics and stochastic draws independently of
  candidate loss restriction.
* TAG training loader: `misc/data.py:138-159`, default
  `DistributedSampler(train_dataset)` if distributed; otherwise shuffle
  true. `drop_last=True`, batch=`config.data.batch_size`,
  workers=`config.data.num_workers`. Each epoch calls `set_epoch(epoch)`
  in distributed mode (`main.py:1707-1708`). With the constant distributed
  helper above, the actual TAG launch uses DistributedSampler even at
  world size one.
* Iterations per epoch are `len(train_loader)`, not a hardcoded number
  (`main.py:1698`). In one process this is floor(dataset length/batch);
  with DistributedSampler it is floor(sampler length/local batch), and
  sampler padding is a separate issue (see DDP audit). TAG config
  `data.batch_size` is used directly per rank; it is not divided by world
  size in `build_pedes_data`.
* External loaders: `runtime.py:284-324`, same train drop-last/default
  DistributedSampler pattern, but `_local_batch_size:227-243` divides
  locked `global_batch_size` by world size. `_train_full:730-770` uses
  full length 10×loader length, the same alpha ramp, autocast helper
  `526-532`, GradScaler, one update/batch, and no gradient clipping.

## 6. What tests can and cannot establish

| Can be proven by specified CPU loss tests | Not established; required additional conditions |
| --- | --- |
| At fixed identical input features, IDs, alpha, scale, weights and domains, enabled D1 equals independent calls of baseline functions on contiguous blocks, including a tail block. | Equivalence to gradient accumulation needs identical per-sample forward features/gradients, BatchNorm behavior, RNG consumption, AMP scale/overflow behavior, per-loss normalization, sample sets/order, alpha and LR schedule. Independent forwards alter TSE BatchNorm and dropout/Gumbel draws. |
| Off and full-size block calls preserve the original function loss and related gradients (subject to stated floating order tolerances in executed tests). | Equivalence to 4×80 without gather needs matching global sample sets and block/rank boundaries, sampler padding/drop-last, rank RNG and augmentations, DDP gradient reduction, no-gather candidate semantics and correct anchor weighting, plus optimizer/AMP behavior. |
| Repeated IDs within/across blocks, teacher mixtures, epsilon, all enabled candidate objectives and relevant upstream trainable-feature paths can be checked for finite values and matching reference gradients. | CPU fixed-feature equality does not prove CUDA kernel precision, TF32 behavior, physical model forward equivalence, full-training trajectory equality, nor validated multiprocess operation. |

The new D1 multiprocess path must be explicitly gated as unverified. A
matching forward loss alone is insufficient evidence of gradient or update
equivalence; the custom gather backward and DDP reduction require separate
measurement.

## 7. D0-3: final checkpoint and dump provenance

### External datasets

`datasets/external/runtime.py:730-770` trains exactly 10 epochs without
intermediate evaluation, returns the optimizer, and leaves the live model
at the last update. `run_external:1074-1083` immediately writes
`checkpoint_last.pth` with `epoch=10`. Its schema (`579-589`) is exactly
`model`, `optimizer`, `config`, `epoch`, with exclusive file creation.

Immediately afterward `1084-1100` calls `_dump_evaluation` on the in-memory
`model.module` (or model). There is **no re-load of checkpoint_last** in this
training-to-dump path. Passing `resume_path=checkpoint_path` there records
the source checkpoint name; it does not perform a load. `_dump_evaluation`
`802-829` extracts text/image features, writes the four feature arrays,
generates/verifies similarity, then computes official metrics/output.
Thus model parameters at dump are the saved last-epoch parameters; eval
feature extraction can still consume the official stochastic Gumbel routing
draws. Features/similarities are the particular evaluation realization,
not claimed deterministic across separate reruns.

The separate external `eval_only` path loads resume weights in
`_load_model:514-522`, reads its epoch at `1042-1048`, and calls the same dump
at `1049-1063`. That is different from training's direct in-memory dump.
External frozen E1/E2 artifacts can provide a verifiable final-model source
when the archived last checkpoint, config, epoch and feature/similarity
files are present and their provenance is retained. This is a conditional
source conclusion, not a claim that every server artifact was inspected.

### TAG-PEDES

`main.py:1713-1781` trains the live model for one epoch. Rank zero then calls
`evaluate_model(model_without_ddp,...)` at `1802-1817`; no best-checkpoint
load occurs at this epoch boundary. If `use_ema=true`, the context temporarily
swaps in that epoch's EMA parameters and restores the live state afterward
(`ModelEMA.average_parameters:872-880`); otherwise it uses the just-trained
live parameters. `1846-1859` appends exactly that `eval_result` to
`metrics_by_epoch.csv`; `experiment.py:231-261` writes and flushes/fsyncs the
row. Therefore a per-epoch CSV row is not reevaluation of historical best
weights, although an EMA-enabled row represents EMA rather than raw live
weights.

The only TAG training checkpoint save in the baseline is inside
`if best_rank_1 < rank_1` (`1868-1882`), writing `checkpoint_best.pth` with
`model`, `optimizer`, `config`, and optional `model_ema`; it has **no epoch
key**. The baseline contains no TAG `checkpoint_last.pth` save path and no
`save_last` flag. `final_results.txt` records best metrics/completed epochs
(`1913-1922`, `experiment.py:263-304`), not a last-model weight digest.

Consequently a TAG last CSV row is evidence of a last-epoch evaluation
result; baseline artifacts alone do not guarantee recoverable/verifiable
last-epoch raw weights unless the best epoch happens to be the last and
that fact is independently established. External and TAG final-weight
availability must not be conflated. No frozen tree/artifact was modified
to create missing evidence.
