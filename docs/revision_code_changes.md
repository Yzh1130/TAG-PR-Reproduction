# 返修代码变更：Git 记录审计

本文件只核对 Git 记录，不运行训练、checkpoint 评测或结果重算。
基线为 `8bf2616403145f5a0e64747ddf9914e0a9e0fcdf`；累计差异端点为
`37c0a083b22436ad8ae14ce0ba5b559ae6572075`；编写时历史锚点为
`ce8222093ea56e260f27c559cac8b96a56747139`。
本文自身的文档提交也属于最终 `HEAD`，其 SHA 不写入自身内容以避免自引用；
最终完整日志用 `git log --oneline 8bf2616..HEAD` 读取，并在交付报告中原样列出。
下文行号均指 `37c0a08` 中的文件，而不是任意服务器工作副本。

## 1. 全部提交的三类归组

归组依据是实际修改路径，不仅是 commit message；含训练文件的混合提交归入第一类。
证据命令：`git log --oneline --name-only 8bf2616..HEAD`。

### 1.1 训练代码改动

```text
05cd179 revision(roundA): audit loss/DDP semantics; add ITC blocks, eval router controls and verification gates
```

该提交修改 `main.py`、`model/base_transformer.py`、`model/tbps_model_MoE.py`、
`config/config.yaml`，同时新增审计文档、独立核查脚本和测试。
证据：`git show --stat 05cd179`。

### 1.2 锁定文件与结果文件

此区间没有此类提交。`8622dc2` 虽以 `archive(ext)` 命名，实际提交的是
`scripts/ext/S7*.py`，不是锁定文件或实验结果。
证据：`git log --oneline --name-only 8bf2616..HEAD` 的全部路径；
`git log --oneline 8bf2616..HEAD -- locks experiments _archive_evidence _dump`
输出为空；`git show --name-only 8622dc2` 只列六个脚本。

### 1.3 分析与渲染脚本、对应测试和文档

以下为上述历史锚点之前该类的全部提交，按 Git 的逆时间顺序原文列出：

```text
ce82220 docs(revision): register unchanged CPU Table 5 sampler verifier
3c9e350 fix(fig5): widen paper seed-mean lines behind markers
cc3e645 feat(fig3): add Chinese labels with glyph-safe English fallback
8cfe423 feat(pairs): merge explicit aliases with per-seed provenance
a1b9fe2 feat(renderer): add router, gallery and compact S13 tables
70b9786 feat(fig5): add three-panel paper layout
5a79d61 feat(revision): render archived result tables with provenance
32af475 fix(fig5): apply Chinese font before layout and preserve text style
3debe7a docs(roundC): clarify absolute run paths and revision analysis commands
f7e82a9 revision(roundC): plot AERI gallery and hubness comparisons
5099075 revision(roundC): parameterize inductive cases and render selected panels
1394849 revision(roundC): collect archived runs and seed-paired summaries
37c0a08 revision(roundB): add offline enrichment, clustered CIs and seed analyses
8622dc2 archive(ext): import S7a-e analysis scripts as run on 2026-09-01 (unmodified)
a7f06f0 fix(revision): allow matched initial AMP calibration skips
```

证据：`git log --oneline 8bf2616..ce82220`，排除已单列的 `05cd179`；
`git show --name-only <上述各 SHA>`。最终 HEAD 另含本文件的纯文档提交，
归入本类，证据为 `git log -1 --oneline -- docs/revision_code_changes.md`
及该提交的 `git show --stat`。

## 2. 37c0a08 的归属与累计改动文件

“训练代码提交 37c0a08”不是该提交本身的准确描述：`git show --stat 37c0a08`
显示它是 26 文件的 Round B 提交，未修改 main/model/config。
用户指定的 `git diff 8bf2616 37c0a08 --stat` 是两个端点之间的累计差异，
包含祖先 `05cd179` 的训练代码改动以及 `a7f06f0`、`8622dc2`、`37c0a08`。
证据：`git log --oneline 8bf2616..37c0a08`；以下为指定 stat 的实际输出：

```text
 .gitignore                               |   1 +
 config/config.yaml                       |   6 +
 docs/ddp_vs_single_gpu_semantics.md      | 245 +++++++++++++
 docs/loss_batch_dependency_audit.md      | 399 ++++++++++++++++++++
 main.py                                  | 242 +++++++++++-
 model/base_transformer.py                |  15 +-
 model/tbps_model_MoE.py                  | 269 +++++++++++++-
 scripts/E3_fresh_forward.py              |  71 +++-
 scripts/R1_joint_enrichment.py           | 276 ++++++++++++++
 scripts/R1_table5_clusters.py            | 415 +++++++++++++++++++++
 scripts/R2_hard_case_pool.py             | 297 +++++++++++++++
 scripts/R2_make_figure.py                | 117 ++++++
 scripts/R2_router_eval.py                | 375 +++++++++++++++++++
 scripts/R3_finegrained_causes.py         | 258 +++++++++++++
 scripts/README_REVISION.md               | 196 ++++++++++
 scripts/README_ROUNDB.md                 | 381 +++++++++++++++++++
 scripts/S13_inductive_bootstrap.py       |  53 +++
 scripts/ext/S7_aggregate_seeds.py        | 265 ++++++++++++++
 scripts/ext/S7a_csls.py                  | 147 ++++++++
 scripts/ext/S7b_distractor.py            | 146 ++++++++
 scripts/ext/S7c_valselect.py             | 136 +++++++
 scripts/ext/S7d_aeri.py                  | 175 +++++++++
 scripts/ext/S7diag.py                    |  81 +++++
 scripts/ext/S7e_tbapr.py                 | 106 ++++++
 scripts/revision/ab_eval_gate.sh         | 131 +++++++
 scripts/revision/compare_eval_outputs.py | 491 +++++++++++++++++++++++++
 scripts/revision/e6_router_randomness.sh | 116 ++++++
 scripts/revision/short_train_check.sh    |  60 +++
 tests/reference_ext/S7a_csls.py          | 110 ++++++
 tests/reference_ext/S7b_distractor.py    | 102 ++++++
 tests/reference_ext/S7c_valselect.py     |  95 +++++
 tests/reference_ext/S7d_aeri.py          |  86 +++++
 tests/reference_ext/S7e_tbapr.py         |  63 ++++
 tests/reference_losses.py                | 546 +++++++++++++++++++++++++++
 tests/test_itc_block.py                  | 503 +++++++++++++++++++++++++
 tests/test_roundb_bundle.py              | 369 +++++++++++++++++++
 tests/test_roundb_causes_cases.py        | 227 ++++++++++++
 tests/test_roundb_ext.py                 | 340 +++++++++++++++++
 tests/test_roundb_joint.py               | 122 +++++++++
 tests/test_roundb_table5.py              | 217 +++++++++++
 tests/test_router_eval_mode.py           | 607 +++++++++++++++++++++++++++++++
 41 files changed, 8812 insertions(+), 45 deletions(-)
```

### 2.1 逐文件说明（覆盖以上全部 41 个路径）

下表证据缩写 `D PATH` 的完整命令为 `git diff 8bf2616 37c0a08 -- PATH`；
对新增文件也可用 `git show 37c0a08:PATH` 读取原文。

| 文件 | Git 中的实际改动 | 对应证据命令 |
| --- | --- | --- |
| `.gitignore` | 新增 `/roundA_logs/` 忽略规则 | `D .gitignore` |
| `config/config.yaml` | 新增六个 experiment 默认键；已有值未改 | `D config/config.yaml` |
| `docs/ddp_vs_single_gpu_semantics.md` | 新增进程/gather/分母/RNG 审计及 Round B TSE BN 只读核查 | `D docs/ddp_vs_single_gpu_semantics.md` |
| `docs/loss_batch_dependency_audit.md` | 新增各损失的批依赖、日程及权重/转储溯源审计 | `D docs/loss_batch_dependency_audit.md` |
| `main.py` | 六 CLI、配置传递及合法性检查、外部 runtime 的非默认开关拒绝、推理种子/归档、block 多进程 gate、debug 更新计数/停止、可选末轮保存；详见第3节 | `D main.py` |
| `model/base_transformer.py` | 添加 eval_router_mode 属性/传递及仅 eval argmax 路由分支 | `D model/base_transformer.py` |
| `model/tbps_model_MoE.py` | 添加主对比项及桥接项的条件分块路径，默认 else 保留原调用 | `D model/tbps_model_MoE.py` |
| `scripts/E3_fresh_forward.py` | 归档矩阵比较改为可选；增加路由/种子、图像输入和调用的 provenance；未改 main 入口 | `D scripts/E3_fresh_forward.py` |
| `scripts/R1_joint_enrichment.py` | 新增同一错误查询条件下的来源×视角联合富集、聚类区间和审计输出 | `D scripts/R1_joint_enrichment.py` |
| `scripts/R1_table5_clusters.py` | 新增表5抽样复原、共同查询/零正例策略、身份及两阶段区间 | `D scripts/R1_table5_clusters.py` |
| `scripts/R2_hard_case_pool.py` | 新增带种子的困难案例池、归档矩阵核查及 contact sheet | `D scripts/R2_hard_case_pool.py` |
| `scripts/R2_make_figure.py` | 新增从显式人工 selection JSON 读取已保存案例并制图 | `D scripts/R2_make_figure.py` |
| `scripts/R2_router_eval.py` | 新增独立图库路由分类报告、argmax/Gumbel采样、来源断言及 logits 导出 | `D scripts/R2_router_eval.py` |
| `scripts/R3_finegrained_causes.py` | 新增面积、文本属性、相似度裕度/真值秩的离线关联分析 | `D scripts/R3_finegrained_causes.py` |
| `scripts/README_REVISION.md` | 新增 Round A 开关、核查命令、适用范围及已有日志比较说明 | `D scripts/README_REVISION.md` |
| `scripts/README_ROUNDB.md` | 新增 Round B 分析入口、协议、输入输出与限制说明 | `D scripts/README_ROUNDB.md` |
| `scripts/S13_inductive_bootstrap.py` | 增加可选 summary-table 四设置输出，不替换旧 a/b/c 配置 | `D scripts/S13_inductive_bootstrap.py` |
| `scripts/ext/S7_aggregate_seeds.py` | 新增显式多种子汇总及 S7 共用 I/O、路径选择和 provenance 函数 | `D scripts/ext/S7_aggregate_seeds.py` |
| `scripts/ext/S7a_csls.py` | 引入外部 CSLS 分析，增加显式运行选择、独占输出、penalty 产物和溯源 | `D scripts/ext/S7a_csls.py` |
| `scripts/ext/S7b_distractor.py` | 引入干扰图库分析，增加 cohort/S7a 输入、逐运行溯源和趋势描述统计 | `D scripts/ext/S7b_distractor.py` |
| `scripts/ext/S7c_valselect.py` | 引入验证集参数扫描与选定参数的测试对照，增加显式 I/O/溯源 | `D scripts/ext/S7c_valselect.py` |
| `scripts/ext/S7d_aeri.py` | 引入四类 AERI 图库分析，使用逐查询正确性数组计算并保存聚合指标及相对 official 的配对身份区间 | `D scripts/ext/S7d_aeri.py` |
| `scripts/ext/S7diag.py` | 引入独立外部运行文件/metadata 诊断脚本 | `D scripts/ext/S7diag.py` |
| `scripts/ext/S7e_tbapr.py` | 引入 TBAPR 子协议分析，增加数据根/S7a penalty 输入及运行溯源 | `D scripts/ext/S7e_tbapr.py` |
| `scripts/revision/ab_eval_gate.sh` | 新增两个 detached worktree 的默认评测 A/B 核查启动器 | `D scripts/revision/ab_eval_gate.sh` |
| `scripts/revision/compare_eval_outputs.py` | 新增既有输出/日志比较器；接受严格匹配的首次更新前 AMP 校准跳过 | `D scripts/revision/compare_eval_outputs.py` |
| `scripts/revision/e6_router_randomness.sh` | 新增 argmax/Gumbel 多评测种子启动器及保存指标差值汇总 | `D scripts/revision/e6_router_randomness.sh` |
| `scripts/revision/short_train_check.sh` | 新增正式日程下 block0/block80 短程核查启动器 | `D scripts/revision/short_train_check.sh` |
| `tests/reference_ext/S7a_csls.py` | 保存 S7a 引入版本的测试参考源码 | `D tests/reference_ext/S7a_csls.py` |
| `tests/reference_ext/S7b_distractor.py` | 保存 S7b 引入版本的测试参考源码 | `D tests/reference_ext/S7b_distractor.py` |
| `tests/reference_ext/S7c_valselect.py` | 保存 S7c 引入版本的测试参考源码 | `D tests/reference_ext/S7c_valselect.py` |
| `tests/reference_ext/S7d_aeri.py` | 保存 S7d 引入版本的测试参考源码 | `D tests/reference_ext/S7d_aeri.py` |
| `tests/reference_ext/S7e_tbapr.py` | 保存 S7e 引入版本的测试参考源码 | `D tests/reference_ext/S7e_tbapr.py` |
| `tests/reference_losses.py` | 新增冻结基线损失/默认 forward 的测试参考实现 | `D tests/reference_losses.py` |
| `tests/test_itc_block.py` | 新增对比分块、默认 forward、桥接、NRW 的 CPU mock 测试 | `D tests/test_itc_block.py` |
| `tests/test_roundb_bundle.py` | 新增 Round B 接口、只读库行为及脚本组合测试 | `D tests/test_roundb_bundle.py` |
| `tests/test_roundb_causes_cases.py` | 新增成因统计、困难案例及制图 mock 测试 | `D tests/test_roundb_causes_cases.py` |
| `tests/test_roundb_ext.py` | 新增外部 S7 参考一致性、显式 I/O 与多种子汇总测试 | `D tests/test_roundb_ext.py` |
| `tests/test_roundb_joint.py` | 新增联合富集定义、条件先验及区间 mock 测试 | `D tests/test_roundb_joint.py` |
| `tests/test_roundb_table5.py` | 新增共同查询、抽样复原、零正例策略及聚类区间 mock 测试 | `D tests/test_roundb_table5.py` |
| `tests/test_router_eval_mode.py` | 新增默认/argmax 路由 RNG、eval_seed、debug/末轮保存及路由报告测试 | `D tests/test_router_eval_mode.py` |

## 3. 补充材料句子的逐项核对

待核对原句：“仅新增 --itc_block_size / --eval_router_mode / --eval_seed / --save_last
四个命令行开关，训练循环、损失与评测口径未改动”。结论：不能原样成立。

### 3.1 主入口确实新增六项，而不是四项

| 参数 | argparse 默认 | 缺省配置生效值 |
| --- | --- | --- |
| `--itc_block_size` | `None` | `0` |
| `--allow_unverified_multiproc` | `False` | `False` |
| `--eval_router_mode` | `None` | `gumbel` |
| `--eval_seed` | `None` | `None` |
| `--save_last` | `False` | `False` |
| `--debug_max_iters` | `None` | `0` |

证据：`git diff 8bf2616 37c0a08 -- main.py config/config.yaml`；
`main.py:375-386` 的六条 `parser.add_argument`，`config/config.yaml:12-17`：

```yaml
  itc_block_size: 0
  allow_unverified_multiproc: false
  eval_router_mode: 'gumbel'
  eval_seed: null
  save_last: false
  debug_max_iters: 0
```

### 3.2 训练循环源码被改动，默认数值干预分支关闭

debug 会观测未缩放梯度、识别成功更新并提前停止；这些是新增循环控制，
不能按源码字面称“训练循环未改动”。默认 `debug_max_iters=0` 时不执行它们。
证据：`git diff 8bf2616 37c0a08 -- main.py` 的训练循环 hunk；
`main.py:1927-1934`、`1952-1960`：

```python
scaler.scale(loss).backward()
if debug_max_iters:
    # Debug-only observation of unscaled gradients; the formal
    # scheduler and alpha ramp above are never rebuilt from N.
    scaler.unscale_(optimizer)
    debug_scale_before = scaler.get_scale()
    debug_trace = _debug_iteration_trace(
        config, optimizer, ret, loss, debug_updates + 1, it, epoch, alpha)
```

```python
if debug_max_iters:
    if scaler.get_scale() >= debug_scale_before:
        debug_updates += 1
        logger.info("DEBUG ITER %s", json.dumps(debug_trace, sort_keys=True))
    else:
        logger.info("DEBUG SKIPPED overflow at iteration=%s", it)
    if debug_updates >= debug_max_iters:
        debug_stopped = True
        break
```

原 cosine scheduler、optimizer、GradScaler 初始化，LR/alpha 日程、autocast、
总损失求和、step/update、EMA 更新、zero_grad 和 iteration 递增未改。
DDP 的 `find_unused_parameters=True` 及数据加载/采样实现未改；新多进程 gate
在 block=0 时先短路。这里有新增 Python 状态/判断，不是训练入口源码零差异。
证据：同一 diff 的上下文；`git diff 8bf2616 37c0a08 -- misc/build.py misc/data.py misc/caption_dataset.py misc/utils.py`
输出为空；`main.py:1592-1596`：

```python
def validate_itc_multiprocess(config):
    if (config.experiment.itc_block_size > 0 and get_world_size() > 1
            and not config.experiment.allow_unverified_multiproc):
        raise ValueError("ITC blocks: 多进程路径未验证; explicitly pass "
                         "--allow_unverified_multiproc only for verification")
```

主入口另新增负 block/debug 值、路由模式的合法性检查；外部数据集 runtime
遇到任一非默认 Round A 开关即拒绝，不在外部路径实现这些开关。
证据：同一 `main.py` diff，`main.py:2147-2150`、`2187-2192`。

### 3.3 损失调用路径确有新增条件分支

block>0 为 NITC、RITC、SDM、TAL 的 global/TSE，以及 bridge CV/TG 选择新增 helper。
当 `0 < block_size < candidate_count` 时，主对比项实际分割候选，块内重建身份软目标、
教师分布/DHN mask，按锚点数汇总；bridge 的实际分块边界是主 batch 的 `batch_count`。
ID/view/ortho 不分块。原损失公式保留，不等于实际分块时损失值/候选集合不变。
若 `block_size >= candidate_count`（bridge 为 `>= batch_count`），helper 回退到完整候选
和原目标，不分割候选；主 helper 仍可能重建 DHN mask，不能称为没有额外 tensor 操作。
证据：`git diff 8bf2616 37c0a08 -- model/tbps_model_MoE.py`；
`main.py` 并未把这些更改限制为“仅 CLI 参数”。
`tbps_model_MoE.py:765-771`、`1031-1034` 的完整候选回退；
`tbps_model_MoE.py:781-791`、`805-806` 的新 helper：

```python
for start in range(0, candidate_count, block_size):
    stop = min(start + block_size, candidate_count)
    anchor_mask = (anchor_positions >= start) & (anchor_positions < stop)
    count = int(anchor_mask.sum().item())
    if count == 0:
        continue
    block_ids = ids[anchor_mask]
    block_candidate_ids = candidate_ids[start:stop]
    positives = block_ids.view(-1, 1).eq(
        block_candidate_ids.view(1, -1),
    ).float()
```

```python
losses.append(loss * (count / anchor_count))
```

默认 block=0 时，forward 的新增 `if itc_block_size > 0` 均走 else，
继续调用原 calc_contrastive/weighted、compute_ritc/sdm/tal 和 bridge 双向损失。
只增加标量开关读取/条件判断，未发现默认分支新增 tensor 运算、RNG 调用或 collective。
证据：上述 diff 的默认 else；`tbps_model_MoE.py:443`、`467-565`、`571-657`、`993-1021`。
此结论是静态路径核查，本轮未做真实模型训练或 checkpoint 数值回归，
不能由此宣称新的真实数据逐位一致实测。

### 3.4 推理路由新增可选分支，默认 Gumbel 保留

默认 gumbel 保留原一次采样；训练态也仍走 Gumbel。
只有模块在 eval 且显式 argmax 时才取消 Gumbel 采样，改用 one-hot argmax。
证据：`git diff 8bf2616 37c0a08 -- model/base_transformer.py`；`:134-138`：

```python
if not self.training and self.eval_router_mode == 'argmax':
    group_probs = F.one_hot(view_logits.argmax(-1), num_classes=view_logits.shape[-1]).to(view_logits.dtype)
else:
    group_probs = F.gumbel_softmax(view_logits, hard=not training)
group_id = torch.argmax(group_probs, dim=1)  # [B]
```

显式 eval_only+eval_seed 才走模型加载→重置 RNG→建 loader 的新顺序；
默认 None 保留原 loader→模型/权重加载顺序，训练不新增该重置。
证据：`git diff 8bf2616 37c0a08 -- main.py`；`main.py:1709-1720` 条件
`if eval_only and config.experiment.eval_seed is not None`，以及 `1576-1585`：

```python
if not eval_only or seed is None:
    return False
```

### 3.5 在线评测指标实现与口径未改，但日志/配置/归档有变化

`misc/eval.py`、evaluate_model、log_evaluation_results 的指标实现和原默认调用未改。
证据：`git diff 8bf2616 37c0a08 -- misc/eval.py` 输出为空；
`git diff 8bf2616 37c0a08 -- main.py` 没有这两个函数定义的修改 hunk。

默认配置也补入六新键，OVERRIDE SUMMARY 追加六项；默认 eval_only 追加
EVAL RNG 说明。因此 config_used.yaml、日志文本及 checkpoint 的 config 段
不是原文件内容不变；不能宣称 checkpoint 整体字节/哈希不变。
证据：`main.py:396-398`、`771-773`、`1781-1785`，同一累计 diff：

```python
for key, default_value in REVISION_CONFIG_DEFAULTS.items():
    if key not in config.experiment:
        config.experiment[key] = default_value
```

```python
for key, default_value in REVISION_CONFIG_DEFAULTS.items():
    logger.info("experiment.%s effective: %r", key,
                getattr(config.experiment, key, default_value))
```

默认最佳模型仍以原始 R@1 创新高判断，文件名仍为 checkpoint_best.pth；
save_last=False 不写末轮文件，debug=0 不写 debug 文件；显式 save_last 才新增
checkpoint_last.pth（model/optimizer/config/epoch，可另含 model_ema）。
显式 eval_seed 或非默认路由可创建独立评测目录和 evaluation_revision.json。
证据：`main.py:2055-2081`、`610-624`、`1804-1811` 的上述累计 diff。
debug/save_last 开启时还向 final_results.txt 追加更新计数或末轮路径与 SHA-256；
默认二者关闭时 helper 直接返回。新增显式 `run_status="completed"` 与归档函数的原默认值相同。
证据：`main.py:1633-1647`、`2114-2121`，以及 `experiment.py:271`。
归档模块自身未改：`git diff 8bf2616 37c0a08 -- experiment.py` 输出为空；
真实模块是根 experiment.py，并非不存在的 utils/experiment.py。

### 3.6 应明确披露的差异

1. 提交身份：训练代码来自 05cd179，而非 37c0a08 本身。
2. 开关数量：main 新增六项，原句漏了 allow_unverified_multiproc 与 debug_max_iters。
3. 源码范围：入口、训练循环控制、模型损失调用/分块 helper、路由及保存逻辑被修改，
   不止 CLI 参数；真正执行候选分割的 block 设置会改变目标/候选集合，较大正 block 可回退完整集合。
4. 默认范围：默认训练数值运算走原分支，在线指标定义未改；默认配置/日志/checkpoint.config
   内容增加，不能概括为所有行为或产物字节不变。
5. 验证边界：本轮是 Git 静态审计，没有新增真实训练或评测逐位回归。

这些结论分别由第2节 stat、第3.1-3.5节的 Git diff 与条件分支证明；
本次只如实记录，不为匹配原句删改代码。

## 4. 37c0a08 之后至最终 HEAD 的独立工具范围

范围是 `37c0a08..HEAD`（不含 37c0a08、含本文提交）；截至历史锚点，
依次是 1394849、5099075、f7e82a9、3debe7a、32af475、5a79d61、70b9786、
a1b9fe2、8cfe423、cc3e645、3c9e350、ce82220，最终另加本文纯文档提交。
实际内容为 collector、R2 参数/选例制图、图5绘图、表格 renderer/别名、
核查脚本及配套测试/README。证据：`git log --oneline --name-only 37c0a08..HEAD`。

这段范围未修改正式训练入口、模型、数据加载或在线指标实现，也没有被这些路径
自动导入/调用。证据（输出均为空）：

```bash
git log --oneline 37c0a08..HEAD -- main.py model misc datasets config utils experiment.py
git diff --stat 37c0a08 HEAD -- main.py model misc datasets config utils experiment.py
git grep -n -E 'collect_revision_runs|plot_fig5_aeri|render_revision_tables|R2_hard_case_pool|R2_make_figure|verify_table5_sampler' HEAD -- main.py model misc datasets config experiment.py
```

“不参与评测指标计算”须限定为“不改变或进入 main 的正式在线指标实现”，
不能扩大成“这些独立工具完全不计算指标”：R2_hard_case_pool 会在离线矩阵上
调用 evaluate；verify_table5_sampler 会从保存的正确性数组重算 R@1。
证据：`git show HEAD:scripts/R2_hard_case_pool.py` 的 analyze 中
`raw = evaluate(sim, ann)` / `inductive = evaluate(...)`；
`git show ce82220:scripts/revision/verify_table5_sampler.py` 的
`r1_same_recount=float(cs.mean() * 100)`。

同样，37c0a08 及之前的“分析”类也不全是纯渲染：E3 独立提取新特征并计算指标，
e6 脚本会显式启动 main.py --eval_only。它们不是默认训练流程的隐式组成部分。
证据：`git show 37c0a08:scripts/E3_fresh_forward.py` 的 `_run_real` 与 evaluate 调用；
`git show 37c0a08:scripts/revision/e6_router_randomness.sh` 的
`python -m torch.distributed.run ... main.py --eval_only`。

本文件只新增文档；最终核对可用 `git show --stat HEAD` 以及以上空 diff/log 命令。
