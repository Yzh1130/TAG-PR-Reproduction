# Round B：离线返修分析与路由重复评测

基线：`infra/exp-management`，`8622dc2540fb28e20c4c344204ef75603c8f3b3b`。
开始前已 fetch、ff-only pull，基线工作区为空。没有修改模型、损失、训练、
数据加载、指标实现或 Round A 开关；没有运行正式训练。
本机不可见 AutoDL 的矩阵、checkpoint 或原始图像，以下真机分析均为
**待训练机执行**，不提供猜测的准确率、增益或区间。

## 1. 公共契约与命令准备

每个入口都支持 `--help`。新增 JSON 保存 argv、随机种子、输入文件的
SHA-256 前 16 位、字节数、git HEAD。原始图像/字体的哈希也记录在图像
manifest 或 inputs 中。旧文件只读；输出独占创建，重名加 UTC 时间戳。
请只使用新的输出根，勿指向 `_dump/`、`_archive_evidence/` 或旧证据目录。
S11/S13 对账不通过时停止，不换种子、不宽松匹配、不重划集合。

离线 TAG 脚本复用 `_offline_common.py` 的来源/视角解析、排序与指标：
分数降序，同分按原始图库索引升序；query 顺序是标注逐图展开 captions。
CSLS 排序用 `s - alpha*r(g)`，等价于 `2*s-r(g)` 的 alpha=0.5 形式。
这里不重新实现或改变已有指标，也不将标签用于估计校正项。

在 AutoDL 仓库根执行以下 Bash 准备。带 `:?` 的变量必须手工指定为
已有 JSON 的**完整路径**，不会自动挑选最新文件：

```bash
cd /root/autodl-tmp/TAG-PR
ANN=/root/autodl-tmp/datasets/TAG-PEDES/test_reid.json
TRAIN_ANN=/root/autodl-tmp/datasets/TAG-PEDES/train_reid.json
IMAGE_ROOT=/root/autodl-tmp/datasets/TAG-PEDES
OUT=/root/autodl-tmp/experiments/roundB_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
L14_SIM=/root/autodl-tmp/experiments/_dump/20260830_dump_fk32/sim_qg.npy
B16_SIM=/root/autodl-tmp/experiments/_dump/20260830_dump_b16f/sim_qg.npy
: "${S11_L14:?设为 _e1/s11/fk32_*.json 的完整文件名}"
: "${S11_B16:?设为 _e1/s11/b16f_*.json 的完整文件名}"
```

S9 的授权 staging 路径与 `README_S14_S15.md` 相同：
`/root/autodl-tmp/authority_s14_s15/_for_codex/S9_fk.json`；本轮 D1/D2
的门槛实际读取匹配的 S11 JSON，不另假设 S9 内容。

## 2. D1 联合富集：`R1_joint_enrichment.py`

```bash
python scripts/R1_joint_enrichment.py --sim "$L14_SIM" --anno "$ANN" \
  --s11-json "$S11_L14" --out "$OUT/R1_joint_l14" --seed 2026 --bootstrap 10000
python scripts/R1_joint_enrichment.py --sim "$B16_SIM" --anno "$ANN" \
  --s11-json "$S11_B16" --out "$OUT/R1_joint_b16" --seed 2026 --bootstrap 10000
```

先逐位核对 `B_error_decomposition.raw.strata.all.cells` 的四格 count。
对同一错误查询集，负样本排除真值身份**全部**图像，四格先验逐查询
等权平均；输出 observed/expected share、绝对超额、比值与身份簇 CI。
期望份额 <1% 时抑制比值，仅保留份额、超额及标注。
同时报告总体、空中、地面、来源分层与混合来源的视角边际。
纯视角来源的同源跨视角格标为不适用，不将结构性零解释为富集。

条件先验明确定义为
`mean(pi_same_source_same_view)/mean(pi_same_source_any_view)`，
即在相同错误查询混合分布下的条件概率；不是逐查询条件比值的平均。
bootstrap 从全体查询身份均匀抽簇，含零错误身份，每次同时重组分子与
分母。无定义 replicate 不重抽：无条件 CI 为 null，记录无定义次数与
明确标为 descriptive 的 valid-only CI。

## 3. D2 表5：`R1_table5_clusters.py`

```bash
python scripts/R1_table5_clusters.py --sim "$L14_SIM" --anno "$ANN" \
  --s11-json "$S11_L14" --out "$OUT/R1_table5_l14" --seed 2026 --bootstrap 10000
python scripts/R1_table5_clusters.py --sim "$B16_SIM" --anno "$ANN" \
  --s11-json "$S11_B16" --out "$OUT/R1_table5_b16" --seed 2026 --bootstrap 10000
```

共同查询、图库规模、种子以 S11 的 `A.common_query_sets`、
`A.gallery_control`、`seeds.gallery` 为准。优先读保存索引；未保存时
复用原 sampler 重放。原始四协议及 10 抽样的四指标均与 S11 逐位对账。
表5参考值由 JSON 计算，**不硬编码论文中的期望数值**。
报告 n_A/n_G/并集、每身份查询分布、逐抽样失去正样本的查询/身份数。

三种零正样本口径并列，不裁决主口径：

1. a 固定共同查询集，零正样本计未命中。
2. b 每次抽样两协议共同有效查询集，逐抽样报告分母。固定 10 次抽样的
   逐查询估计只平均该查询共同有效的抽样，全部无效才剔除；同时报告
   逐抽样比率的均值，二者分母不同，不混称同一估计量。
3. c 每身份目标视角至少保留两图，再均匀填至相同规模，同视角源图还需
   保留以执行共同排除。该规则提高正样本保留率；不足两图或预算不够
   则显式 infeasible，绝不复制图像或放松规则。

原 S11 代码：共同查询集先满足合法正样本条件，原 sampler 本身也是
positive-preserving；shared evaluator 剔除 zero-positive 查询。
脚本如实报告这些事实，**不假定原表5发生了分母污染**。
区间为保留身份所有查询与重复次数的、查询加权的配对区间：原图库；
条件于 10 次固定图库抽样的逐查询平均差；身份/图库抽样两阶段敏感性。
报告逐抽样差的标准差和范围，不计算 McNemar。
点估计推断采用 float64 比率，S11 展示 R1 用 float32 百分数，可能出现
末位算术差；JSON 分别保存 exact metric gate 与口径差异说明。

## 4. D3 外部 S7：显式目录、多种子汇总

五个入口都接受 `--root --run-dir [dataset[:split]=]PATH`（可重复）、
`--tag --out`。一个调用/tag 对应一个训练种子；同 dataset/split 多目录
时拒绝歧义，不使用 `sorted(glob)[-1]`。输出为
`{out}/{script}_{tag}[_{UTC}]/`，其中每个产物文件名含 tag。
S7a summary 为 `S7a_summary_{tag}.json`。
S7b/e 用 `--s7a-json` 验证原 run 路径与 meta/sim 哈希，再定位 penalty。
五个入口共用 `S7_aggregate_seeds.select_runs/_run_meta`：
**`--run-dir` 的 PATH 若为相对路径，则相对当前工作目录解析，而不是
相对 `--root`**。`--root` 仅用于未给 `--run-dir` 时的目录扫描。
本轮只修正文档，不修改入口解析；下列所有 run 变量都显式拼成绝对路径。

以下为 seed0/1/2 完整循环。每个变量应是具有 epoch=10 转储的**实际**
目录（目录包含真实时间戳），不得填推测的时间戳或通配符。
`*_DIR` 环境变量只填已有目录的 basename（不带路径）；固定的绝对根
加上该 basename 形成实际输入路径：

```bash
EXT_ROOT=/root/autodl-tmp/TAG-PR/experiments_ext
CUHK0="$EXT_ROOT/${CUHK0_DIR:?填已有 cuhkpedes test seed0 目录名}"
CUHK1="$EXT_ROOT/${CUHK1_DIR:?填已有 cuhkpedes test seed1 目录名}"
CUHK2="$EXT_ROOT/${CUHK2_DIR:?填已有 cuhkpedes test seed2 目录名}"
ICFG0="$EXT_ROOT/${ICFG0_DIR:?填已有 icfgpedes test seed0 目录名}"
ICFG1="$EXT_ROOT/${ICFG1_DIR:?填已有 icfgpedes test seed1 目录名}"
ICFG2="$EXT_ROOT/${ICFG2_DIR:?填已有 icfgpedes test seed2 目录名}"
AERI0="$EXT_ROOT/${AERI0_DIR:?填已有 aeri test seed0 目录名}"
AERI1="$EXT_ROOT/${AERI1_DIR:?填已有 aeri test seed1 目录名}"
AERI2="$EXT_ROOT/${AERI2_DIR:?填已有 aeri test seed2 目录名}"
TBAPR0="$EXT_ROOT/${TBAPR0_DIR:?填已有 tbapr test seed0 目录名}"
TBAPR1="$EXT_ROOT/${TBAPR1_DIR:?填已有 tbapr test seed1 目录名}"
TBAPR2="$EXT_ROOT/${TBAPR2_DIR:?填已有 tbapr test seed2 目录名}"
CUHK_RUNS=("$CUHK0" "$CUHK1" "$CUHK2")
ICFG_RUNS=("$ICFG0" "$ICFG1" "$ICFG2")
AERI_RUNS=("$AERI0" "$AERI1" "$AERI2")
TBAPR_RUNS=("$TBAPR0" "$TBAPR1" "$TBAPR2")
# 仓库中已有锁定的 cohort；只读，不重建或修改。
COHORT="$PWD/locks/cohort_splits_seed2026.json"
for seed in 0 1 2; do
  tag="seed${seed}"
  python scripts/ext/S7a_csls.py --root "$EXT_ROOT" \
    --run-dir "cuhkpedes=${CUHK_RUNS[$seed]}" --run-dir "icfgpedes=${ICFG_RUNS[$seed]}" \
    --run-dir "aeri=${AERI_RUNS[$seed]}" --run-dir "tbapr=${TBAPR_RUNS[$seed]}" \
    --tag "$tag" --out "$OUT/ext_s7a"
  # OUT 是新的唯一目录。若重复运行发生后缀，使用 stdout 所打印的 summary 路径。
  summary="$OUT/ext_s7a/S7a_csls_$tag/S7a_summary_$tag.json"
  python scripts/ext/S7b_distractor.py --root "$EXT_ROOT" \
    --run-dir "cuhkpedes=${CUHK_RUNS[$seed]}" --run-dir "icfgpedes=${ICFG_RUNS[$seed]}" \
    --run-dir "aeri=${AERI_RUNS[$seed]}" --run-dir "tbapr=${TBAPR_RUNS[$seed]}" \
    --cohort-json "$COHORT" --s7a-json "$summary" --tag "$tag" --out "$OUT/ext_s7b"
  python scripts/ext/S7d_aeri.py --root "$EXT_ROOT" --run-dir "aeri=${AERI_RUNS[$seed]}" \
    --tag "$tag" --out "$OUT/ext_s7d"
  python scripts/ext/S7e_tbapr.py --root "$EXT_ROOT" --run-dir "tbapr=${TBAPR_RUNS[$seed]}" \
    --dataset-root /root/autodl-tmp/datasets/TBAGPR --s7a-json "$summary" \
    --tag "$tag" --out "$OUT/ext_s7e"
done
python scripts/ext/S7_aggregate_seeds.py \
  --s7a-json "$OUT/ext_s7a/S7a_csls_seed0/S7a_summary_seed0.json" \
  --s7a-json "$OUT/ext_s7a/S7a_csls_seed1/S7a_summary_seed1.json" \
  --s7a-json "$OUT/ext_s7a/S7a_csls_seed2/S7a_summary_seed2.json" \
  --tag seed012 --out "$OUT/ext_seed_summary"
```

只有 CUHK seed0 val 转储在任务中明确提供；不猜测 seed1/2 的 val：

```bash
CUHK_VAL0="$EXT_ROOT/${CUHK_VAL0_DIR:?填已有 cuhkpedes val seed0 目录名}"
python scripts/ext/S7c_valselect.py --root "$EXT_ROOT" \
  --run-dir "cuhkpedes:val=$CUHK_VAL0" --run-dir "cuhkpedes:test=$CUHK0" \
  --run-dir "icfgpedes:test=$ICFG0" --run-dir "aeri:test=$AERI0" --run-dir "tbapr:test=$TBAPR0" \
  --tag seed0 --out "$OUT/ext_s7c"
```

S7d 新增 `paired_gallery_differences`：ground/mixed/每次规模受控 aerial
相对 official 的 baseline R1 配对差，以及 transductive/inductive
CSLS **增益之差**，共享查询/身份重采样区间。原有 protocols 字段不变。
aggregate 给出逐种子 baseline R1/mAP、两类增益和身份 CI、种子均值、
population std(ddof=0)、范围、符号计数；不自动宣称“显著”。

原版数学一致性证据：`tests/reference_ext/` 五份文件逐字来自基线，
`test_01_math_functions_verbatim` 比较原 41 个函数（含日志/哈希函数；S7d protocol
只扣除两行新缓存）。其余原函数原文不变；差异为入口封装、显式 run
选择、命名、provenance、独占输出、Windows resource 不可用诊断，
以及明确要求的 S7d 新配对分析。可用
`git diff 8622dc2 -- scripts/ext/S7{a_csls,b_distractor,c_valselect,d_aeri,e_tbapr}.py`
审阅。S7diag.py 完全未改。

## 5. D4 成套后处理与模型组标签

E3 的 `--sim` 可省略：仅跳过归档对账，JSON 中 archive metrics/delta
为 null、`archive_comparison_skipped=true`；新鲜特征、矩阵与 352 行
扫描不变。`--tag` 写入文件名。provenance 记录 checkpoint/config/
标注的 SHA16/bytes、路由模式、初始提取 seed、git HEAD。
E3 使用原 `load_model` 初始 seed；**并不调用 main 的 post-load
eval-only seed reset**，故 `eval_seed=null`、
`eval_seed_reset_applied=false`，配置中的 eval_seed 单独记录，不能
误当成实际评测 reset seed。
图库图像逐文件 SHA16/bytes 另列在 provenance.gallery_image_inputs；
mock 替换提取器、不存在原图时明确 available=false，不伪造图像哈希。

S13 默认 a/b/c 及原 JSON 不变；只有 `--summary-table [JSON路径]` 才
额外输出四行统一表：无校正、train_full(.5,10)、train_full(.75,30)、
test_full(.5,10)。各行四指标、配对 ΔR1、身份 CI；是固定方案比较，
不根据测试集择优。原表和额外表分开独占保存。

L/14 主要性能组 `l14_keep_s1`、`l14_keep_s2` 直接读已有 E3 JSON 的
`fresh_files`，不能从目录名称猜测矩阵配对：

```bash
: "${E3_JSON_S1:?设为 _e1_e3_s1/E3_*.json 的完整文件名}"
: "${E3_JSON_S2:?设为 _e1_e3_s2/E3_*.json 的完整文件名}"
for seed in 1 2; do
  name="E3_JSON_S${seed}"; evidence="${!name}"
  mapfile -t matrices < <(python - "$evidence" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f: p=json.load(f)
print(p['fresh_files']['sim_qg']); print(p['fresh_files']['sim_bank_g'])
PY
  )
  python scripts/S13_inductive_bootstrap.py --sim "${matrices[0]}" --bank "${matrices[1]}" \
    --anno "$ANN" --out "$OUT/l14_keep_s$seed" --tag "l14_keep_s$seed" \
    --seed 2026 --bootstrap 10000 --summary-table
done
```

B/16 主要性能组先提取新训练末轮权重，不需归档 parity：

```bash
: "${B16_CONFIG:?设为服务器实际 cfg_b16.yaml 完整路径}"
: "${C0_CKPT_SEED0:?设为 <日期>_e3_c0_b0_seed0/checkpoint_last.pth 完整路径}"
: "${C0_CKPT_SEED1:?设为 <日期>_e3_c0_b0_seed1/checkpoint_last.pth 完整路径}"
: "${C0_CKPT_SEED2:?设为 <日期>_e3_c0_b0_seed2/checkpoint_last.pth 完整路径}"
for seed in 0 1 2; do
  name="C0_CKPT_SEED${seed}"; checkpoint="${!name}"
  python scripts/E3_fresh_forward.py --fresh-forward --config "$B16_CONFIG" \
    --backbone ViT-B/16 --resume "$checkpoint" --anno "$ANN" --train-anno "$TRAIN_ANN" \
    --image-root "$IMAGE_ROOT" --seed "$seed" --device cuda:0 --image-batch-size 32 \
    --text-batch-size 256 --score-block-size 256 --num-workers 2 \
    --tag "e3_c0_b0_seed$seed" --out "$OUT/e3_c0_b0_seed$seed"
done
# 从上一步 stdout 取各个 E3 JSON 的精确路径，再逐一执行：
: "${E3_C0_JSON0:?设为 seed0 本次 E3 JSON}"
: "${E3_C0_JSON1:?设为 seed1 本次 E3 JSON}"
: "${E3_C0_JSON2:?设为 seed2 本次 E3 JSON}"
for seed in 0 1 2; do
  name="E3_C0_JSON${seed}"; evidence="${!name}"
  mapfile -t matrices < <(python - "$evidence" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f: p=json.load(f)
print(p['fresh_files']['sim_qg']); print(p['fresh_files']['sim_bank_g'])
PY
  )
  python scripts/S13_inductive_bootstrap.py --sim "${matrices[0]}" --bank "${matrices[1]}" \
    --anno "$ANN" --out "$OUT/c0_summary_seed$seed" --tag "e3_c0_b0_seed$seed" \
    --seed 2026 --bootstrap 10000 --summary-table
done
```

历史模型1（L/14，第8轮）必须单独标组，不并入末轮/keep 组。
要分析 8/30 归档矩阵用同次 S12 reconstructed bank（离线模拟）：

```bash
: "${HISTORICAL_BANK:?设为同 fk32 归档配对的 S12_*_sim_bank_g.npy 完整路径}"
python scripts/S13_inductive_bootstrap.py --sim "$L14_SIM" --bank "$HISTORICAL_BANK" \
  --anno "$ANN" --out "$OUT/historical_l14_model1_epoch8" --tag historical_l14_model1_epoch8 \
  --seed 2026 --bootstrap 10000 --summary-table
```

若用该历史权重的 E3 fresh bank，则 qg 也必须换成**同一次** E3 的
fresh qg，不能混用 8/30 的随机路由矩阵。S12 重建 bank 的 W 来自测试
查询拟合，只能称离线模拟；E3 的 query-independent guards 另行记录。

## 6. D5 成因关联与 D6 困难案例

```bash
python scripts/R3_finegrained_causes.py --sim "$L14_SIM" --anno "$ANN" \
  --image-root "$IMAGE_ROOT" --out "$OUT/R3" --seed 2026 --bootstrap 10000
: "${S13_JSON:?设为使用 L14_SIM 与 HISTORICAL_BANK 的既有 S13 JSON}"
python scripts/R2_hard_case_pool.py --sim "$L14_SIM" --bank "$HISTORICAL_BANK" \
  --anno "$ANN" --image-root "$IMAGE_ROOT" --s13-json "$S13_JSON" \
  --out "$OUT/R2" --seed 2026 --cases-per-pool 30 --sheet-page-size 6
: "${SELECT_JSON:?设为人工挑选的 selection JSON 完整路径}"
python scripts/R2_make_figure.py --select "$SELECT_JSON" --out "$OUT/R2_figure"
```

R3 从原始 PIL 图像读取宽高面积，不用模型输入224×224替代；来源内
四分位的面积 ties 放同箱，空箱显式记录。颜色/服饰词表完整写 JSON，
Jaccard 取错误图所有描述的最大值（无截词），对照均匀抽同源异身份图。
无同源负样本单独计数，不造对照。裕度是错误首位分数减最佳正样本
分数，真值秩沿用原 stable tie 规则。仅输出关联，不作因果断言。

R2 七池：面积最低10%错误、属性重合最高10%错误、关键词遮挡待确认、
转导/训练库校正各自成功与失败。保留阈值 ties，因此候选比例不必
恰好10%。各池 seed 为2026加固定offset，最多取30，空池不虚造案例。
转导结果与 S13 c 及 raw R1 逐位对账。contact sheet 显示完整查询文字、
前后前三名、最佳真值与命中标记。遮挡需人工确认。
人工 selection 示例（query_index 必须实际在保存的 pool 中）：

```json
{
  "pool_json": "/absolute/new/R2_hard_case_pool.json",
  "items": [{"pool": "transductive_success", "query_index": 123}],
  "title": "Manually selected cases"
}
```

无合适中文字体时生成无文字 PNG 与完整 layout JSON，文字由论文端
合成；可用 `--font /absolute/font.ttf` 显式选择。图中校正条件被记录，
训练库不自动等同于纯归纳（保留 S12/E3 provenance）。

## 7. D7 路由重复评测（GPU、只评测）与 D8

```bash
: "${E6_CONFIG:?设为待核查的真实配置文件}"
: "${E6_CHECKPOINT:?设为该配置真实 checkpoint}"
PYTHON=python bash scripts/revision/e6_router_randomness.sh \
  "$E6_CONFIG" ViT-L/14 "$E6_CHECKPOINT" "$OUT/e6" 10
```

使用1进程 torchrun + `main.py --eval_only`。argmax seed0一次，Gumbel
eval_seed0..9各一次；关闭 CSLS/rerank，每次独立 exp_name/new run。
汇总读取实际 `evaluation_revision.json.original`，router/seed逐项校验，
读取每次 `config_used.yaml.misc.seed`，保存 config/checkpoint/结果证据。
给各次四指标、相对argmax差与“n次重复中观测到的最大差异”，不声称
这是所有随机种子的上界。没有运行训练，也不修改模型路由实现。
本机只测 shell help/语法与 mock 汇总；GPU重复评测**待训练机执行**。

D8：原作者 `CrossEmbeddingLayer_tse.py:45-54` 使用普通
`nn.BatchNorm1d`，`[B,N,D]` reshape 为 `[B*N,D]` 后统计；
`main.py:1822-1827` 直接DDP，没有SyncBatchNorm转换/当前BN统计跨rank合并。
所以 **4卡×80时每卡BN统计只覆盖本卡样本，单卡320时覆盖全批**
（具体是其选中token行）。DDP buffer广播不是batch统计同步。
详见 `../docs/ddp_vs_single_gpu_semantics.md#d8-round-b原作者-tse-batchnorm-的只读核查`。
只读结论，不推测性能影响。

## 8. 实测、原版限制与待执行清单

Windows / Python3.12 / torch2.5.1+cu121。所有新分析使用200×100
mock；全局/局部特征宽度8/16；E3训练bank mock为300×100。
mock的bootstrap次数明确写测试参数，S13 legacy/summary专项用10000次。
外部mock仅为可运行尺寸降低原版固定基数/扫描预算/bootstrap次数，
reference与新入口使用同一mock参数；**生产常量保持原值**。

```bash
MKL_THREADING_LAYER=SEQUENTIAL python -m unittest \
  tests.test_roundb_joint tests.test_roundb_table5 tests.test_roundb_ext \
  tests.test_roundb_causes_cases tests.test_roundb_bundle -v
```

本机实际执行结果（2026-10-12）：

```text
Round B 五组集成：Ran 38 tests in 19.828s; OK
既有 scripts suite：Ran 75 tests in 28.936s; OK
全 tests suite（含本轮38项）：Ran 109 tests in 48.182s; OK (skipped=6)
既有 tools suite：Ran 12 tests in 14.186s; OK
13个 Python 入口：--help / syntax 13/13 PASS
e6：GitBash -n / --help PASS；未执行GPU eval命令

R1 joint MOCK 200x100: explicit-negative prior max error=0
TABLE5 MOCK: exact S11 checks=355; zero-positive a/b/c denominators=48/42/48
S7a MOCK: D_global=8 D_local=16; reference JSON + penalties array_equal=True
S7b/c/d/e MOCK: all original calculations exactly equal
E3 MOCK: omitted --sim archive reads=0; 352 real scan rows
S13 MOCK: complete legacy a/b/c report EXACT (timing excluded), bootstrap=10000
```

既有 tests suite 包含其原有 CPU toy 训练循环单元测试（3步），以及 CPU
smoke mock；不是数据集训练或正式短程实验。本轮没有执行训练命令。

覆盖：D1守恒与独立CI；D2三口径、失正样本、saved/replayed索引门槛与
1ULP停止；S7五原版逐位一致/三种子aggregate/manifest与碰撞；
R3原尺寸与属性/无错误/空箱；R2七池/坏parity停止/人工面板/字体fallback；
E3省略与显式sim时所有fresh矩阵和352行逐位相同；S13无flag与基线
a/b/c完整结果相同（仅排除墙钟计时），summary四指标与身份CI；
e6 GitBash语法/help、三个由200×100矩阵实算的评测汇总与异常停止。

原版S7保留的一处限制：S7a/S7d metric 使用 `match.argmax(1)`，没有
正样本的行可能被误计为首位命中。S7d随机 aerial 子图库可能产生这类行。
S7e 用 `gp[argmax] == qp`，失去正样本时计未命中，不存在上述假命中。
本轮要求其计算原文不变，因此没有修复 S7a/S7d 的该限制，
不要将受影响子协议结果当成已修正的 zero-positive 口径。
S7外部统计本轮只做参数/I/O与明确授权新增分析，不自动升级为TAG
shared evaluator。不同原版脚本的top-k归约精度也原样保留。

待训练机执行：全部真实S11/S13 exact gates；真实表5三口径及区间；
真实联合先验与富集；外部seed0/1/2所有S7及汇总；B16末轮E3提取/352行
与各模型组summary；真实R3/案例人工挑选；e6 GPU路由重复评测。
本地mock不替代这些门槛，不报告任何未知实测数值。

## 9. Round C：结果收集、显式面板与图5（只读离线）

本节脚本不运行训练、不修改任何 run 或原有评测路径。真实实验结果均
**待训练机执行**；只指定已有输入目录/JSON，输出放入新的分析目录。

### 9.1 archived runs 与逐 seed 配对

```bash
: "${RUN_ROOT:?设为已有 TAG-PEDES 实验根目录的绝对路径}"
: "${ROUND_C_OUT:?设为新的分析输出目录的绝对路径}"
python scripts/revision/collect_revision_runs.py --root "$RUN_ROOT" \
  --include e3_ --pair 'e3_c0_b0|e3_c1_b80' --by seed --metric last \
  --tag e3_last --out "$ROUND_C_OUT/collector"

# 历史基线前缀由用户指定，不用目录日期或模型名字推断基线。
: "${HIST_B16_BS320_PREFIX:?填历史 B/16 bs320 基线的真实前缀}"
: "${E2_GROUP2_PREFIX:?填第2组 E2 的真实前缀}"
: "${E2_GROUP3_PREFIX:?填第3组 E2 的真实前缀}"
: "${E2_GROUP4_PREFIX:?填第4组 E2 的真实前缀}"
python scripts/revision/collect_revision_runs.py --root "$RUN_ROOT" \
  --include "$HIST_B16_BS320_PREFIX" --include e2_ \
  --pair "$HIST_B16_BS320_PREFIX|e2_nrw_view_norm" \
  --pair "$HIST_B16_BS320_PREFIX|$E2_GROUP2_PREFIX" \
  --pair "$HIST_B16_BS320_PREFIX|$E2_GROUP3_PREFIX" \
  --pair "$HIST_B16_BS320_PREFIX|$E2_GROUP4_PREFIX" \
  --by seed --metric last --tag e2_last --out "$ROUND_C_OUT/collector"

# best 是独立口径，独立 tag/表，不与 last 混列。
python scripts/revision/collect_revision_runs.py --root "$RUN_ROOT" \
  --include e3_ --pair 'e3_c0_b0|e3_c1_b80' --metric best \
  --tag e3_best --out "$ROUND_C_OUT/collector"
```

`--root`、`--include`、`--pair` 可重复；include 默认不过滤，pair 默认
不启用；`--by=seed`、`--metric=last`。前缀匹配目录名、去掉 `YYYYMMDD_`
的目录名及归档 exp_name，不从目录名提取 seed。输出 `runs_TAG.csv/json`
与有配对时的 `pairs_last_TAG.csv`（或 `pairs_best_TAG.csv`）。碰撞增加
UTC 后缀，不覆盖。每个 seed 两侧都必须恰好一个目录，缺失、重复、
两前缀命中同一目录或指标缺失均标 unpaired，不选择“最新”目录。
统计仅为 A/B 及 B−A 的均值、population std(ddof=0)、范围和差值符号
计数，不做检验或可加分解。

解析证据为本轮基线源码中的实际写入格式：

- `main.py:1962-1968` 的 `Epoch[k] Iteration[i/N]` 是按 print_period
  抽样的进度行；`1981-1987` 的 `Epoch k done.` 是完成证据。N 记为
  measured_loader_iterations；不能用最后一条打印的 i 冒充完整轮长度。
- `main.py:1370-1377` 原始 `Acc@1 ... Acc@5 ... Acc@10 ... mAP ...`
  只有5位小数；CSLS/RERANK 前缀行不是原始指标。优先读
  `metrics_by_epoch.csv` 的 epoch/R1/R5/R10/mAP 原始列（`1432-1490`）。
- `experiment.py:213-231` 归档 config_used.yaml/git_commit.txt/
  run_command.txt/environment_summary.txt；命令文件 `116-141` 保存
  command/sys.argv/config_source，git 文件 `104-115` 保存运行时 HEAD。
- `experiment.py:263-303` 的 final_results.txt 有 best_epoch、四指标、
  completed_epochs/run_status；`main.py:2058-2081` 保存 best/last checkpoint。
  collector 仅流式哈希 checkpoint，不反序列化。日志文件为 train.log
  （`experiment.py:65-79`），兼容历史 train_log.txt 及其他 `.log`。

缺失字段一律 null 并列入 missing_fields；不使用当前 config 默认值、
当前 git HEAD 或配置里的理论 epoch 数补训练证据。config_used 的 SHA16
来自归档文件；原始 config_source 的 SHA16 若当时未记录则 null，不能
用现在同路径的 YAML 冒充训练时文件。老日志可能没有 argv/exp_name/
seed/git/config hash/开关值；只留5位指标或没有完整迭代证据时也如实标注。

### 9.2 R2 归纳条件与显式 Figure 3 面板

```bash
: "${SIM_QG:?设为已有 sim_qg.npy 的绝对路径}"
: "${BANK_G:?设为已有 sim_bank_g_fresh.npy 的绝对路径}"
: "${ANN:?设为真实 test_reid.json 的绝对路径}"
: "${IMAGE_ROOT:?设为真实图像根目录的绝对路径}"
: "${S13_JSON:?设为同一矩阵对应 S13 JSON 的绝对路径}"
python scripts/R2_hard_case_pool.py --sim "$SIM_QG" --bank "$BANK_G" \
  --anno "$ANN" --image-root "$IMAGE_ROOT" --s13-json "$S13_JSON" \
  --inductive-alpha 0.5 --inductive-k 10 \
  --bank-label 'E3_fresh_forward e3_c0_b0_seed0' --out "$ROUND_C_OUT/r2_pool"

: "${R2_TRANS_JSON:?填保存所选查询行的 R2 JSON 绝对路径}"
: "${R2_IND_JSON:?填归纳条件 R2 JSON 绝对路径}"
python scripts/R2_make_figure.py --pool-json "trans=$R2_TRANS_JSON" \
  --pool-json "ind=$R2_IND_JSON" --panel 'a:446@trans,1663@trans' \
  --panel 'b:1421@trans' --panel 'c:1048@trans' --panel 'd:511@trans' \
  --panel 'e:227@ind' --tag fig3 --out "$ROUND_C_OUT/fig3"
```

hard_case 的新增默认值仍为 alpha=.5/k=10/bank-label=None，原 parity gate
不变。bank 路径、bytes、SHA16、alpha/k/label 都进 JSON；未给 label
时沿用旧 provenance note。用户自由文本 label 不是查询独立性的认证。

`--panel` 与 `--pool-json LABEL=PATH` 可重复；`--tag` 默认 selected。
也支持 `--select POOL_JSON --panel 'a:446,1663'`，或旧 selection JSON。
无 --panel 时旧 --select 的渲染与输出保持原样。若同 q 在一个 JSON
内有不同条件，显式用 `227@ind/inductive_success`；@label 仅指定 JSON，
不能隐式决定该 JSON 内的校正条件。缺少保存行（即使属于 candidate
但不在已保存抽样 cases 中）立即报错，不重算或静默跳过。

新面板按查询源图、before top-3、after top-3、best positive 共8格；
绿框命中、红框未命中，行首 `(a) q=...`。图内仅英文标签，无查询文本。
新目录输出 PNG300dpi/PDF/layout JSON/selected_cases_TAG.json；查询文本
只在 JSON。逐行保存所属池、池seed、保存的抽样规模、是否位于该池
selected_query_indices、条件、源 JSON 路径/bytes/SHA16；旧 JSON 缺 seed
或抽样列表时为 null，不补猜。全条件相同的多个池成员关系全部保留。

### 9.3 Figure 5：只画 S7d 已记录的结果

```bash
: "${S7D_SEED0_JSON:?填 seed0 S7d JSON 的绝对路径}"
: "${S7D_SEED1_JSON:?填 seed1 S7d JSON 的绝对路径}"
: "${S7D_SEED2_JSON:?填 seed2 S7d JSON 的绝对路径}"
: "${S7_AGG_JSON:?填同三训练种子 aggregate JSON 的绝对路径}"
python scripts/revision/plot_fig5_aeri.py \
  --s7d-json "$S7D_SEED0_JSON" --s7d-json "$S7D_SEED1_JSON" \
  --s7d-json "$S7D_SEED2_JSON" --aggregate-json "$S7_AGG_JSON" \
  --tag fig5 --out "$ROUND_C_OUT/fig5" --font /absolute/path/to/chinese.ttf
```

恰好三个 --s7d-json，按 `_meta.provenance.seeds.run_seed` 验证训练seed，
不从文件名猜。--out/--tag 必填，--font 默认 None；缺失或无效字体
打印 WARNING 后回退英文。输出新目录里的主图 PNG300dpi/PDF、存在
paired_gallery_differences 时的另一版配对差/身份CI图、逐点 CSV、
绘图 JSON 与日志。所有数值来自 protocols/paired_gallery_differences；
aggregate 只核对种子与记录 official 值对照，舍入差只警告，不替换值。

左图各训练seed的点与三seed均值横线；右图为 hub excess skewness_N10
与校正 ΔR1 的描述性散点，不拟合。规模受控项使用 S7d 保存的 mean
字段，hub 为其5次图库抽样偏度均值；5次抽样不当作5个训练seed。
配对CI不混合、不平均端点；缺失的CI保持空值；controlled CI 保留
conditional_on_five_draws 语义。CSV记录来源字段，便于逐点追溯。

### 9.4 Round C 自检

```bash
MKL_THREADING_LAYER=SEQUENTIAL python -m unittest \
  tests.test_roundc_collector tests.test_roundc_r2 tests.test_roundc_fig5 -v
MKL_THREADING_LAYER=SEQUENTIAL python -m unittest discover -s tests -v
```

collector 合成目录树包括历史与新结构、缺失字段、重复seed与手算
配对统计；R2/fig5 用200×100 mock（含不同校正条件、缺行停止、
旧JSON兼容、无字体回退、PNG/PDF与输入只读检查）。此处不是正式
训练或真实论文结果。

2026-10-12 本机实测：collector 14、R2 6、fig5 5 项全部通过；完整
tests suite `Ran 134 tests in 38.717s; OK (skipped=6)`，即128项通过、
6项跳过。既有 scripts suite 75项通过（15.079s），tools suite 12项
通过（7.772s）；4个本轮 Python 入口真实 `--help` 与语法检查通过。
PNG/PDF mock 已做视觉检查。跳过项为4个外部真实标注测试、可选
einops 测试与 Linux-only shell 测试，不涉及新增25项。

既有 scripts 测试使用包名调用（此目录为 namespace package），例如
`python -m unittest scripts.test_S11_S12 scripts.test_E3_fresh_forward`；
不能直接 `discover -s scripts`，否则相对 import 没有 package 上下文。
全量 tests 中原有 CPU toy loop/smoke mock 不等于正式训练；本轮未
执行任何正式训练、服务器操作或真实数据评测。
