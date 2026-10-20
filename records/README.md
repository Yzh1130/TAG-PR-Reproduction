# 返修阶段实验记录（Revision records）

本目录是论文《文本引导空地行人检索：瓶颈诊断、训练协议分析与hubness校正》返修阶段的完整实验记录，内容直接复制自训练服务器（AutoDL，单张 RTX 5090），复制时未改动任何数值；唯一的改动是 `configs/` 下配置文件中的绝对路径被替换为占位符（见 `configs/SOURCES.txt`，其中记录了原文件的 SHA-256）。`MANIFEST.sha256` 列出本目录全部文件的 SHA-256。

训练代码版本、分析脚本版本与本记录的发布版本是三个不同的 commit。各运行实际使用的训练代码版本以其 `git_commit.txt` 为准（投稿版 8bf2616，返修补充运行 37c0a08，更早的历史运行见各自文件）；分析与渲染脚本为 b3752d2；本目录由其后的发布 commit 加入，不改变任何历史结果的署名。

## 一、论文与补充材料中引用路径的对应位置

| 文档表述 | 本仓库位置 |
|---|---|
| 补充材料第1节：训练配置、运行命令、逐轮与末轮指标、分析脚本及事前锁定文件按实验目录保存在该仓库中 | `records/runs/experiments/<运行目录>/` 与 `records/runs/experiments_ext/<运行目录>/`，每个目录含 `config_used.yaml`（有效配置）、`run_command.txt`、`git_commit.txt`、`environment_summary.txt`、`metrics_by_epoch.csv`、`final_results.txt`、`train.log`；外部数据集运行另含 `meta.json`。分析脚本见 `scripts/`，锁定文件见 `locks/` |
| 补充材料第1节与附表B17注：检查点信息与 SHA-256 校验和见仓库 checksums.txt | `records/checksums.txt`（全部 48 个已保存权重文件的完整 SHA-256，路径与 `records/runs/` 的目录名一致；权重文件本身不随仓库公开） |
| 补充材料第4节最小复现命令中的 `--config cfg_l14.yaml` | `records/configs/cfg_l14.yaml`。该文件是 `config/config.yaml` 的服务器副本，差异为 `max_words: 77`、`num_workers: 2`、`checkpoint` 指向 CLIP ViT-L/14 权重与数据路径；骨干、批大小、轮数与梯度检查点由命令行参数覆盖。`cfg_l14_8bf2616.yaml` 与其内容相同（同一 SHA-256，均为 2026-10-12 的文件）；2026 年 8 月历史运行的有效配置以各运行目录的 `config_used.yaml` 为准。其余 `cfg_*.yaml` 为各次运行 `run_command.txt` 所引用的配置文件 |
| 表A2“依赖：见仓库 requirements 文件” | `records/environment/requirements_tagpr5090.txt`（服务器环境 `pip freeze` 原样输出）与 `environment_versions.txt`。各运行目录的 `environment_summary.txt` 记录 Python 3.10.12、torch 2.7.1+cu128、CUDA 12.8、NVIDIA GeForce RTX 5090。仓库根目录的 `requirements.txt` 为上游 TAG-CLIP 的依赖文件，保持不变 |
| 附表B4注：其余模型的混淆矩阵见仓库 final_*/e5_*/R2_router_eval.json | `records/analysis/roundC_out/final_20261012T150714Z/e5_*/R2_router_eval.json` 与 `records/analysis/roundC_out/l14s3_20261013T013136Z/e5_l14_keep_s3/R2_router_eval.json` |
| 附表B5（评测期路由随机性） | `records/analysis/roundC_out/final_20261012T150714Z/e6_*/*/router_randomness.json` 与 `roundC_out/l14s3_20261013T013136Z/e6_l14_keep_s3/*/router_randomness.json` |
| 表11、附表B13（外部数据集三种子） | `records/analysis/experiments/roundB_ext_20261012T101926Z/ext_s7a/S7a_csls_seed{0,1,2}/` 与 `ext_seed_summary/S7_aggregate_seeds_seed012/S7_aggregate_seeds_seed012.json` |
| 表12、附表B14（受控干扰项规模实验） | `records/analysis/experiments/roundB_ext_20261012T101926Z/ext_s7b/S7b_distractor_seed{0,1,2}/` |
| 附表B15注：逐抽样值及全部配对差区间见仓库 roundB_ext_*/ext_s7d/ | `records/analysis/experiments/roundB_ext_20261012T101926Z/ext_s7d/` |
| 附表B16注：各变体的身份 bootstrap 区间见仓库 roundB_ext_*/ext_s7e/ | `records/analysis/experiments/roundB_ext_20261012T101926Z/ext_s7e/` |
| 4.6 节 CUHK-PEDES 官方验证集选参 | `records/analysis/experiments/roundB_ext_20261012T101926Z/ext_s7c/` 与 `results_ext/S7c_valselect.json` |
| 表3、附表B8（富集分析） | `records/analysis/experiments/roundB_20261012T075841Z/R1_joint_l14/`、`R1_joint_b16/` |
| 表5、附表B9（双视角身份受控评测） | `records/analysis/experiments/roundB_20261012T075841Z/R1_table5_l14/`、`R1_table5_b16/` |
| 4.3.5 节、附表B10、B11（细粒度混淆的关联证据） | `records/analysis/experiments/roundB_20261012T075841Z/R3/` |
| 图3、附表B18注：selected_cases_fig3_final_en.json 与候选池 | `records/analysis/roundC_out/fig3_final/R2_manual_figure_*/selected_cases_fig3_final_en.json`；候选池记录在 `records/analysis/experiments/roundB_20261012T075841Z/R2/` |
| 附表B6、B7、B17 的运行信息汇总 | `records/analysis/roundC_out/roundC_20261012T134247Z/collector/`（`runs_all_runs.json`、`pairs_*.csv`，由 `scripts/revision/collect_revision_runs.py` 只读生成） |
| 表8、表10、4.5 节（后处理对照、交叉拟合、代理验证选参） | `records/analysis/experiments/_e1_e3*/`、`_e1_s13*/`、`_e1_s14/`、`_e1_s15/` 的 JSON 与日志，以及 `records/runs/experiments/_csls*/`、`_rr/`、`_dump/` 下的评测运行记录 |
| 2026-09-01 的 seed 0 外部分析原始输出 | `results_ext/`（已在仓库中）与 `records/analysis/_e_ext/`（同一输出加运行日志） |
| 事前锁定文件 | `locks/`（仓库）与 `records/analysis/_locks/`（服务器副本，SHA-256 相同） |

## 二、来源

- 运行记录来自服务器目录 `/root/autodl-tmp/experiments/` 与 `/root/autodl-tmp/TAG-PR/experiments_ext/`，由训练入口 `experiment.py` 在运行时写出；本目录只复制了小体积文件。
- 分析归档来自 `/root/autodl-tmp/experiments/roundB_*`、`/root/autodl-tmp/roundC_out/`、`/root/autodl-tmp/experiments/_e1_*`、`/root/autodl-tmp/_e_ext/`，由 `scripts/R1_*.py`、`scripts/R2_*.py`、`scripts/R3_*.py`、`scripts/ext/S7*.py`、`scripts/S11_*`～`S15_*` 与 `scripts/revision/` 下的脚本生成。
- `git_commit.txt` 中的 `WORKTREE: DIRTY` 表示运行时检出目录含未跟踪文件（运行输出目录与本地配置），各文件内列出了具体条目。
- `checksums.txt` 由 `sha256sum` 对服务器上的权重文件直接计算。

## 三、不随仓库公开的内容

- 模型权重（`checkpoint_best.pth`、`checkpoint_last.pth`）：仅提供 `checksums.txt` 中的 SHA-256。
- 相似度矩阵、特征文件与查询库矩阵（`*.npy`）、逐查询排序文件（`output.json`）：体积过大，未随仓库公开。补充材料所述“可由保存的相似度矩阵直接复核”指作者保存的这些文件，需要时可联系作者获取。
- 原始数据集：按各数据集官方许可获取，划分未作改动。

## 四、复现说明

- 环境：`pip install -r records/environment/requirements_tagpr5090.txt`（Python 3.10.12、PyTorch 2.7.1+cu128）。
- 配置文件中的 `/path/to/clip/` 与 `/path/to/TAG-PEDES` 需替换为本地的 CLIP 权重目录与 TAG-PEDES 数据目录。
- 最小复现命令见补充材料第4节；其余运行的命令见各运行目录的 `run_command.txt`，有效配置见 `config_used.yaml`。

## 五、外部验证的种子说明

Lock-A（`locks/lockA_v1.0.json`）以 seed 0 为主，Lock-B 的方向预测只针对 seed 0。seed 1 与 seed 2 为返修阶段按审稿意见增加的训练重复（2026-10-12，投稿版代码 8bf2616），沿用 Lock-A 锁定的参数、查询组划分与分析规则，各运行的 `config_used.yaml` 与 `run_command.txt` 可核对。锁定文件、标签与其 SHA-256 未作任何改动。
