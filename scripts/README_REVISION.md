# Round A 返修核查入口

源码审计基线为 `8bf2616403145f5a0e64747ddf9914e0a9e0fcdf`。先读两份只读审计：

- [损失及批依赖](../docs/loss_batch_dependency_audit.md)：真实启用项、目标生成、日程、末轮权重/转储来源。
- [DDP 与单卡语义](../docs/ddp_vs_single_gpu_semantics.md)：可微 gather、各项分母、TSE BatchNorm、采样补齐与随机源。

本文的 GPU 命令是**待训练机执行**的核查，不表示已通过服务器测试。CPU 单元测试与 comparator mock 只证明它们覆盖的行为，不证明完整 TAG-PEDES 数字或未运行的多进程梯度等价。

## 新增开关

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--itc_block_size B` | `0` | 在同一次前向、同一次优化器更新内，仅限制批内对比候选；按 gathered 顺序连续分块，重新生成块内身份软目标和教师分布，尾块按真实锚点数加权 |
| `--allow_unverified_multiproc` | 关闭 | B>0 且真实 world-size>1 时默认报“多进程路径未验证”；本参数只是后续验证的显式放行，不代表已经验证 |
| `--eval_router_mode {gumbel,argmax}` | `gumbel` | 只控制推理图像级路由；argmax 使用 one-hot(argmax)，训练仍使用原 Gumbel 路径 |
| `--eval_seed S` | `None` | 仅明确传入且 `--eval_only` 时，在模型加载完成、eval loader 构造之前设置 Python/NumPy/torch CPU/CUDA seed；训练及训练中的逐轮评测不重设 seed |
| `--save_last` | 关闭 | 保留已有最佳权重逻辑，额外保存正式末轮 `checkpoint_last.pth`；在 `final_results.txt` 记录路径和 sha256 |
| `--debug_max_iters N` | `0` | 完成 N 次成功优化更新后正常结束；使用正式配置生成的日程，归档标记 `debug_stopped`，调试权重不会使用正式最佳权重名称 |

块限制覆盖主 NITC 全局/TSE、RITC 全局/TSE，以及启用时的 SDM/TAL 候选。ID/view/ortho 不受影响；原权重不修改。没有给全批 logits 加负无穷掩码，也没有裁剪全批目标再归一。

桥接项启用时仍只做一次有桥图像编码，不 gather；按对应主样本原始 index 的块归属限制候选，在每个有桥子块内重建 CV/TG 身份目标并调用原函数，以有桥子锚点数汇总。不把 bridge 的紧凑子集序号当作主样本块编号，不引入零填充图像的编码。

NRW 开启时，统计与权重在原完整前向中计算一次，然后各块取对应行的权重。块内仍调用原加权函数，本地 `sum(w*loss)/(sum(w)+1e-6)`；块间按锚点数汇总。这是指定块调用的行为，不应描述为与全批加权平均或不同 rank 加权平均天然相等。TSE 的 BN 同样保留原完整前向统计；拆成多个模型前向/梯度累积会改变 BN 的统计对象，不能从单进程损失测试推断等价。

TAG-PEDES 原入口的 `is_using_distributed()` 恒为 True，故下面用一个 `torch.distributed.run` 进程运行单卡检查；即使 world-size=1，仍保持官方进程组/gather 路径。未修复或绕过这一原有初始化约定。

## 默认路径 A/B 数值闸门（待训练机执行）

同一容器安装一次依赖，准备带绝对 `anno_dir` / `image_dir` 的既有配置、同一 trusted checkpoint。配置须关闭 CSLS/rerank，保留其 `misc.seed`、batch_size 和 worker 数。`NEW_COMMIT` 必须是包含本轮代码的已提交 SHA：

```bash
export PYTHON=python
export REVISION_OUT_ROOT=/root/autodl-tmp/roundA_checks
bash scripts/revision/ab_eval_gate.sh \
  8bf2616403145f5a0e64747ddf9914e0a9e0fcdf \
  "$NEW_COMMIT" "$CHECKPOINT" "$CONFIG" 'ViT-L/14'
```

脚本创建带时间戳及随机后缀的新目录，保留两个 detached worktree 和所有日志/转储，不删除或覆盖既有实验。两个命令均使用默认 Gumbel、不传 eval_seed；使用相同启动配置 seed。不为匹配历史数字切换路由、换种子或重试。

临时 wrapper 只添加检查，不改变训练/评测函数：实际 test loader 必须是 `SequentialSampler`，实际图像编码时 `model.training` 必须为 False，图像编码参数不能进入训练路径。记录两边实际 batch/worker/标注 sha256 和 seed 供比较。它不调用任何 RNG seed 函数。

各边开启已有 `--dump_sim --dump_sim_dtype float32`。比较程序检查：`sim_meta.json` 中四项原始指标的完整数值；`output.json` 每个查询的 prediction/match 等全部数值数组（包括保存的排序/索引）；fp32 `sim_qg.npy` 全矩阵值，逐块报告最大绝对差和首个差异位置。其接受条件是数值完全相同；不比较时间戳、路径或 Git SHA 等溯源字段的文件字节。若不一致，退出失败并保留证据，不迁就结果。

可以对两个既有新 run 目录直接执行比较：

```bash
python scripts/revision/compare_eval_outputs.py eval \
  --old-dir "$OLD_RUN" --new-dir "$NEW_RUN" \
  --block-size 256 --out "$NEW_COMPARISON_JSON"
```

`--out` 必须不存在；重复核查使用新的文件名。浮点转储必须是 fp32，不接受先转换 fp16 再宣称完整数值一致。

## 两种候选大小的短程训练核查（待训练机执行）

配置保留正式总轮数、warmup、batch、全部损失权重，例子只把停止点设为三次成功更新：

```bash
export REVISION_OUT_ROOT=/root/autodl-tmp/roundA_checks
bash scripts/revision/short_train_check.sh "$CONFIG" 0 3
```

脚本依次运行 `--itc_block_size 0` 和 `80`，同 seed、同配置；不把正式 epoch 数改为调试步数。每步 `DEBUG ITER` 的 JSON 记录成功更新序号、日程 iteration、epoch、所有参数组 LR、软标签 alpha、正式每轮迭代数、块大小、损失权重、全部返回损失、总损失、未缩放梯度范数与 finite 状态。

比较器要求两边恰好完成 N 次更新，损失/梯度有限；LR、alpha、iteration/epoch、每轮迭代数和损失权重逐步相同。两种候选集合的损失与梯度允许不同，不作为等价目标。再检查两边 `final_results.txt` 中 `run_status: debug_stopped` 与指定 N，且不存在未标记的 `checkpoint_best.pth`。启动器正常退出才会继续比较。

仅允许**首次成功优化更新之前**的 GradScaler 初始校准跳过，且两边跳过次数、逐条迭代号必须完全一致。比较器统一采用从零开始的 attempted iteration `it`：`DEBUG SKIPPED` 在 `it += 1` 后打印，故 `it = logged_iteration - 1`；`DEBUG ITER` 在递增前采集，原 `it` 不变。初始跳过必须从原日志 `iteration=1` 连续开始，随后成功更新的 `it` 也须连续；编号约定、两边逐条跳过记录、首次成功更新位置都写入比较 JSON。

现有日志未记录 loss scale，逐条写入 `"scale_not_logged"`，不从默认值或假定减半规律推算。若输入日志显式带有 `scale` / `scale_before` / `scale_after` 字段，保存并逐条比较；单边缺失或数值不同不通过。**首次成功更新之后出现任何 `DEBUG SKIPPED`（包括最后一条成功更新之后）仍失败**。原有 N 次成功更新、LR、alpha、iteration/epoch、每轮迭代数、损失权重逐步一致及成功更新的 finite/数值有限检查均保持；通过不表示此前被跳过的梯度有限，也不证明未记录的 scale 相同。只重新比较已有日志，不改训练代码、GradScaler 参数或日志格式，不重跑训练。

```bash
python scripts/revision/compare_eval_outputs.py train \
  --old-log "$BLOCK0_LOG" --new-log "$BLOCK80_LOG" \
  --expected-updates 3 --out "$NEW_COMPARISON_JSON"
```

## 推理路由/种子与测试图库路由器证据

受控抽样例子：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 main.py \
  --config "$CONFIG" --backbone 'ViT-L/14' --eval_only --resume "$CHECKPOINT" \
  --eval_router_mode gumbel --eval_seed 0 \
  --output_dir "$NEW_OUTPUT_ROOT" --exp_name roundA_gumbel_seed0
```

确定性推理例子把 router_mode 改为 argmax，并选择另一个新实验名。不要把这些显式受控参数加到默认 A/B 闸门中。

显式提供 eval_seed 或非默认 router_mode，而未给 output_dir/exp_name 时，入口会创建带 `eval_revision_` 时间后缀的新评测实验目录，归档最终配置与评测参数；不会回写既有 checkpoint 同级归档。手工扫描仍建议给每次运行指定新的实验名。

测试图库路由器脚本（先用 `--help` 确认入口；输入是测试标注，标签只用于分类报告）：

```bash
python scripts/R2_router_eval.py \
  --config "$CONFIG" --backbone 'ViT-L/14' --resume "$CHECKPOINT" \
  --anno "$TEST_ANNOTATION" --image-root "$IMAGE_ROOT" --split gallery \
  --batch-size 32 --num-workers 2 --n-gumbel 20 --seed 0 \
  --out "$NEW_ROUTER_JSON" --device cuda:0 --save-logits
```

每个 MoE block 记录路由器 logits，以第一个 MoE block 作主报告；输出 cam_id×来源交叉表、确定性 argmax 指标、固定种子的 Gumbel 多次采样均值/逐图翻转、margin 分位数与溯源。纯视角来源 cam_id 不一致会报错，不硬编码任何期望准确率。

## 本机 CPU 自检与界限

```bash
python -c "import main"
python main.py --help
python -m unittest tests.test_itc_block tests.test_router_eval_mode -v
python scripts/revision/compare_eval_outputs.py --help
python scripts/revision/compare_eval_outputs.py mock
bash scripts/revision/ab_eval_gate.sh --help
bash scripts/revision/short_train_check.sh --help
```

comparator mock 使用 seed 2026 的 `200×100` 合成矩阵，验证完全相同值通过、一个 ULP 差异失败和三步合成日志的日程检查；另覆盖相同初始校准跳过通过、次数/编号/scale 证据不同失败、成功后任意跳过失败，以及校准后原有日程/finite 检查。mock 输出明确标记 CPU mock，不冒充真实训练机运行。shell 的 `--help` 和语法检查不运行 GPU。

实际 F6 的 A/B 闸门与短程训练日志/比较 JSON 必须由训练机运行后归档。多进程 B>0 尚未验证；显式放行只是未来实验入口。完整评测路由种子的复现条件还包括依赖/GPU/AMP/TF32、相同 checkpoint/config/数据顺序和 worker 设置，详见 DDP 审计；源码检查本身不证明这些运行条件已全部满足。

## 本轮实际本机测试（2026-10-12）

环境：Windows、Python 3.12、PyTorch `2.5.1+cu121`、RTX 4060 Laptop GPU。
CUDA 可用，但没有执行完整 TAG-PEDES CUDA/checkpoint 评测或训练；现有
`einops` 依赖缺失。F2/F3 运行生产源码 AST，仅避开无关导入；不替换损失或路由公式。

```text
python -c "import main"                         import main OK
python main.py --help                           exit 0; six Round A parameters present
python -m unittest tests.test_itc_block -v       9 tests OK
python -m unittest tests.test_router_eval_mode -v 16 tests OK
python -m unittest discover -s tests -p 'test_*.py'
  Ran 71 tests; OK (skipped=6)
tools existing suite: Ran 12 tests; OK
scripts existing suite: Ran 75 tests; OK
```

`scripts/` 是 namespace package，原有测试使用相对导入，不能直接以
`unittest discover -s scripts` 导入为顶层模块。在本机 Anaconda 环境中还遇到
OpenMP DLL 重复加载错误。没有使用 `KMP_DUPLICATE_LIB_OK` 绕过；最终采用
进程内的 MKL sequential threading 后测试全部通过（不修改仓库或训练配置）：

```powershell
$env:MKL_THREADING_LAYER = 'SEQUENTIAL'
python -c "import torch; import unittest; from pathlib import Path; names=['scripts.'+p.stem for p in sorted(Path('scripts').glob('test_*.py'))]; result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); raise SystemExit(not result.wasSuccessful())"
```

F2 实测：N/B=`320/0,320/320,320/80,330/80,79/80` ×
NITC/加权 NITC/RITC/SDM/TAL，共 25 个 global+TSE 组合，loss 及全部参与
梯度最大绝对误差均为 `0`。bridge N=330/M=83，B=`0,80,330` 同为 `0`。
默认实际 forward 对冻结基线原 forward（NRW off/fixed）loss/梯度 `torch.equal=True`。
NRW view_norm/global_norm buffers 只更新一次，与未分块全批更新逐位一致。
显式比较参数为 `logit_scale`、image/text projection 的 weight/bias、visual/textual
TSE toy upstream layer 的 weight/bias，共 9 个；TAL 不使用 scale，两边对应梯度均为 None。
这不是全模型/多 GPU 梯度等价测试。

```text
F3 default: training=False/True Gumbel calls=1; inputs/output/RNG exact to 8bf2616
F3 argmax: onehot exact, dtype/shape preserved, calls=0, RNG unchanged
F3 training gumbel/argmax attributes: outputs/RNG exact
F4 eval_only seed=917: two toy scores/results exact; model load -> seed -> loader
F4 default vs 8bf2616 seed calls: Python=1 NumPy=1 torch=1 CUDA=1; scores/results exact
F4 training eval_seed: Python/NumPy/torch states unchanged; no extra seed calls
F4b actual CPU main loop: exit=0, updates=3, formal schedule[:3] exact
  base LR: 1e-6, 1.5142857142857144e-5, 2.9285714285714288e-5
  niter_per_ep=8; archive debug_stopped; no partial-epoch evaluation/formal best save
D4 CPU: checkpoint contains model/optimizer/config/epoch/model_ema; final SHA256 recorded
F5 MOCK: 200x100, two routers, 21 complete forwards, 14 original Gumbel calls/draw
  run_real mock gallery/train paths: head rows=11, hashes/source/provenance/errors PASS
F6 comparator CPU mock: PASS; ULP/prediction/metric/LR/nonfinite differences rejected
  matched initial calibration skips allowed; mismatched calibration or later skips rejected
F6 real A/B gate and short GPU training: NOT RUN; waiting for training machine
```

默认训练没有新增种子重置：原有 `set_seed(config)` 位置保持；新增四种 seed
调用仅位于 `set_evaluation_seed`，其调用受 `eval_only and eval_seed is not None`
限制。F4 同时执行冻结基线与当前入口验证调用次数，并检查训练 RNG 状态。

## 已有训练机日志的重新比较（2026-10-12）

本次只读取提供的 `roundA_logs/block0.log` 与 `block80.log`，没有重跑训练。
两边均为六次初始 `DEBUG SKIPPED iteration=1..6`，未记录 scale；首个成功
更新均为 `update=1, it=6`（第七次 attempted iteration）。此后各三十次
`DEBUG ITER` 均 finite，无再次跳过；所有参数组 LR、alpha、iteration/epoch、
每轮迭代数和损失权重逐步完全相同。比较器的新规则只接受这一匹配的校准前缀。
日志与新的比较 JSON 保存在被忽略的 `roundA_logs/`，不提交原始运行日志。

```bash
python scripts/revision/compare_eval_outputs.py train \
  --old-log roundA_logs/block0.log --new-log roundA_logs/block80.log \
  --expected-updates 30 --out roundA_logs/comparison_calibration_NEW_TIMESTAMP.json
```

`NEW_TIMESTAMP` 替换为本次时间戳；输出文件必须不存在，禁止覆盖旧证据。
