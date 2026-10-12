# DDP 与单进程训练的语义审计（Round A D0-2）

审计基线：`8bf2616403145f5a0e64747ddf9914e0a9e0fcdf`。以下仓库行号均指该提交，不指后续实现的移动行号。本文只描述原有代码；没有运行多进程训练，也没有用前向损失相同推断梯度等价。

## 1. 进程初始化和“单进程”的实际含义

`misc/utils.py:21-22` 的 `is_using_distributed()` 无条件返回 `True`；`misc/utils.py:55-62` 因而读取 `RANK/WORLD_SIZE/LOCAL_RANK` 并初始化进程组。`main.py:2021` 调用该初始化；`main.py:1665-1670` 包装 DDP。这里的常规 TAG-PEDES 单卡路径实质是 `WORLD_SIZE=1` 的已初始化进程组，而不是无进程组的普通 Python 运行。

```python
# misc/utils.py:21-22
def is_using_distributed():
    return True

# misc/utils.py:55-62
def init_distributed_mode(config):
    if is_using_distributed():
        config.distributed.rank = int(os.environ['RANK'])
        config.distributed.world_size = int(os.environ['WORLD_SIZE'])
        config.distributed.local_rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend=config.distributed.backend,
                                             init_method=config.distributed.url)
        used_for_printing(get_rank() == 0)
```

`config/config.yaml:83-85` 配置 `backend: nccl` 与 `url: env://`。`misc/utils.py:64-69` 使用全局 rank 构造 `cuda:{rank}`。多节点的全局 rank 与本机 GPU 编号是否对应不能由这段代码保证，必须单独核实；本文不改变该路径。

因此新候选分块功能的多进程防护应读取实际已初始化的 `get_world_size()` (`misc/utils.py:33-36`)，不能用总为 True 的 `is_using_distributed()` 判断 W>1。基线比对脚本应以 `torch.distributed.run --nproc_per_node=1` 启动，以保持原单卡进程组/gather 路径；这些是后续实现的约束，本文未修改该遗留初始化行为。

外部数据集入口有独立行为：`datasets/external/runtime.py:121-124` 在未初始化分布式进程组时主动把 `config.model.use_gather` 设为 `False`，不能把它的普通单进程行为直接归为 TAG-PEDES 入口的行为。

## 2. all_gather 的前向与反向

这是自定义 `torch.autograd.Function`，不是 `@no_grad` 特征收集，也不是 `torch.distributed.nn` 包装。`model/shared_modules.py:8-32`：

```python
class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = int(os.environ['RANK'])
        ctx.world_size = int(os.environ['WORLD_SIZE'])
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        dist.all_gather(y, tensor.contiguous())
        y = torch.cat(y, 0).view(-1, *tensor.size())
        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        dist.all_reduce(in_grad.contiguous())
        # split
        return in_grad[ctx.rank]
```

`dist.all_reduce` 未显式传 `op`，采用 PyTorch 默认 SUM。每个进程产生的“对全体候选特征的梯度”先跨进程求和，再取拥有该特征的 rank 的切片。远端候选因此会获得来自其他 rank 锚点损失的梯度：**远端可训练特征并未被 detach**。这是一层特征梯度归约；后面的 DDP 参数梯度归约是另一层，不能把二者当作重复的无意义操作。

`model/tbps_model_MoE.py:1162-1167` 在 `use_gather=False` 时直接返回输入；开关为真时调用上述函数后展开为 `[world_size*B, ...]`。`WORLD_SIZE=1` 时仍执行 gather；已有进程组中的通信值不改变，但不能因此认为任意无进程组调用也合法。

收集对象和梯度状态如下：

| 对象 | 基线位置 | 梯度语义 |
| --- | --- | --- |
| 归一化全局图像/文本特征 | `model/tbps_model_MoE.py:420-423` | 可微 gather；候选梯度跨 rank 求和 |
| 身份 id | `model/tbps_model_MoE.py:437-441` | 整数张量，无 autograd 梯度；用于全局候选软目标 |
| domain id，DHN 开启时 | `model/tbps_model_MoE.py:443-453` | 整数标签，无梯度 |
| TSE 图像/文本特征 | `model/tbps_model_MoE.py:495-500` | 先可微 gather，之后教师支路显式 detach |
| 全局教师特征 | `model/tbps_model_MoE.py:458-461` | 本地与 gathered 教师副本均 detach |
| 桥接分支 | `model/tbps_model_MoE.py:749-807` | 不 gather；只用本 rank 的有桥子集 |

前向 gather 分配每个 rank 同形状的张量 (`model/shared_modules.py:17`)，并不实现变长 gather；各 rank 本地批形状一致是运行条件之一。

## 3. 锚点、候选、归一化与 DDP 归约

设本地批大小为 B，进程数为 W，候选数 N=WB。`model/tbps_model_MoE.py:437-441` 创建 `[B,N]` 的 `sim_targets`；同身份候选按行均分，并非只标记本地配对样本：

```python
idx = ids.view(-1, 1)
gathered_ids = self.all_gather(ids)
idx_all = gathered_ids.view(1, -1)
pos_idx = torch.eq(idx, idx_all).float()
sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
```

各损失的 rank 内归一化：

| 项 | 基线位置 | anchor 与分母 |
| --- | --- | --- |
| NITC 全局和 TSE | `model/tbps_model_MoE.py:465-469,501-505,973-1002` | `[B,N]` 的 i2t/t2i；候选方向 `dim=1` softmax；各方向对本地 B 个锚点 `.mean()`，双向平均 |
| NRW 加权 NITC（非默认） | `model/tbps_model_MoE.py:1004-1070` | 本地 `(w*loss).sum()/(w.sum()+1e-6)`；不是所有 rank 的权重和 |
| RITC 全局和 TSE | `model/tbps_model_MoE.py:575-579,837-846` | `[B,N]`；KL `reduction='batchmean'`，分母为本地 B；双向平均 |
| SDM（非默认） | `model/tbps_model_MoE.py:526-546,848-866` | 每行汇总后本地锚点均值，两个方向相加 |
| TAL（非默认） | `model/tbps_model_MoE.py:548-573,868-940` | 本地锚点对全局候选；最终只对有负例的行平均，各 rank 有效行数可能不同 |
| ID | `model/tbps_model_MoE.py:581-585` | 图像与文本分别本地 cross entropy 均值，再双向平均；无候选 gather |
| view | `model/base_transformer.py:118-125,289-291` | 各 MoE block 本地图像 CE 均值；再对 block 平均 |
| ortho | `model/base_transformer.py:126-128,292-293` | 各 block 本地样本均值；再对 block 平均 |
| bridge（非默认） | `model/tbps_model_MoE.py:749-835` | 仅本地有桥子集 M，局部候选 M；每方向对 M 平均 |

`main.py:1724-1729` 在本 rank 汇总所有名称含 `loss` 的返回项。日志 meter `main.py:1731-1762` 也在本地更新，没有为这些 meter 调用跨 rank 汇总。rank 0 的训练日志不能直接视为全局所有锚点的损失均值。

```python
# main.py:1665-1670
if is_using_distributed():
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.device],
        find_unused_parameters=True,
    )
```

代码没有注册通信 hook、没有改写 DDP 默认归约，也没有使用 DDP `join`。正常等长迭代情况下，DDP 在 backward 同步参数梯度并按 W 平均。它没有在前向替本地 loss 求全局均值；不同 rank 的归一化仍由上表决定。`broadcast_buffers` 未传入，沿用库默认 `True`，它不等价于 SyncBatchNorm。

只有在相同固定特征、相同模型参数、所有 rank 有相同 B、相同全局候选/目标、损失各自按 B 归一、无额外状态差异等条件成立时，数学目标 `mean_r L_r` 才对应全局 N 个锚点的均值。可微 gather 负责跨 rank 的候选侧梯度，DDP 均值归约负责参数侧的 rank 汇总。这个条件解释不是本机多卡梯度等价测试，也不保证浮点逐位一致。

NRW 的本地权重分母、TAL 的本地有效行数分母、bridge 的本地子集与局部候选，均不能自动等同于单卡全局加权/有效行/桥接子集目标。默认配置 `nrw_mode='off'`、`tal_ratio=0`、桥接 ratio=0，但开启它们时必须另做语义核查。

## 4. 不能忽略的 TSE BatchNorm 与模型状态

TSE 的局部特征前向包含 BatchNorm，而非纯逐样本变换。`model/CrossEmbeddingLayer_tse.py:45-54`：

```python
h = [hidden_dim] * (num_layers - 1)
self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

def forward(self, x):
    B, N, D = x.size()
    x = x.reshape(B * N, D)
    for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
        x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
    x = x.view(B, N, self.output_dim)
    return x
```

训练时第一层 BN 的统计来自本 rank 的 B×所选 token 数；最后一层不调用该层 BN。文本与图像 TSE 分别调用该 MLP (`model/CrossEmbeddingLayer_tse.py:90,122`)。仓库没有 SyncBatchNorm 转换。因此单卡 320、四卡本地 80、四次独立 80 前向累积的 BN 统计/特征/反向和运行中 buffers 可以不同。DDP buffer 广播复制 rank 0 的 buffer，并不合并当前各 rank 的 BN batch 统计。

Round A 的“仅在已得到的特征上分块调用对比损失”保留原完整前向的 BN 统计，因而仍能在固定相同特征条件下与独立分块参考实现比较；这不证明它与拆分模型前向或多卡无 gather 等价。

NRW 正常化统计另有显式跨进程路径：`model/tbps_model_MoE.py:607` 使用 `@torch.no_grad()`，`:641-644` 对相似度 detach，`:659-697` 更新 buffers，`:720-734` 对 count/sum/squared_sum 调用 all_reduce 后生成矩。这不是 BN 的同步实现，也不消除本地加权损失分母的差别。

模型 EMA 仅由 rank 0 创建 (`main.py:1673-1678`)，优化步后更新 (`:1769-1778`)，评测时临时切权重 (`:1802-1817`)；关闭时没有这条路径。比较不同运行时必须固定 EMA 开关和用于评测的权重来源。

## 5. DistributedSampler、补齐与每轮迭代数

`misc/data.py:138-159` 没有向 DistributedSampler 提供 `seed`、`shuffle` 或 `drop_last`，沿用库默认 `seed=0, shuffle=True, drop_last=False`。因此 sampler 的 shuffle 种子不是 `config.misc.seed`，尽管增强/模型随机源由该配置设置。`main.py:1707-1708` 每轮调用 `set_epoch(epoch)`。

Sampler 默认对长度 S 的数据集生成 `ceil(S/W)*W` 个索引，必要时复制打乱索引序列的头部，使各 rank 有 `ceil(S/W)` 个样本，再使用 `indices[rank:total_size:W]` 分片。随后训练 DataLoader 使用 `drop_last=True` (`misc/data.py:152`)：这是**本地批级丢尾**，不同于 sampler 的补齐设置。

每 rank 的轮内 batch 数为 `floor(ceil(S/W)/B)`；`main.py:1698` 实际用 `len(train_loader)` 设置 `niter_per_ep`。全局每步 WB；单卡与多卡即使“名义全局 batch”相同，补齐、最后一段丢尾与按 rank 交错索引的顺序也需要比对。`AllGather` 的顺序是 rank 优先、各 rank 本地顺序次之 (`model/shared_modules.py:19-21`)，不自动等于单卡 sampler 同一步的原始索引顺序。

测试 loader 不使用 DistributedSampler，`shuffle=False, drop_last=False` (`misc/data.py:142,154-159`)；训练后仅 rank 0 跑完整评测 (`main.py:1791-1817`)。默认测试 batch 是代码中的 32 (`misc/data.py:156`)，只有明确的 test batch CLI override 才重建 loader (`main.py:771-805`)。

## 6. 每进程训练随机源与数据增强

`main.py:2032` 在模型构建/数据 loader 创建之前调用 `misc/utils.py:87-98` 的 `set_seed(config)`：torch CPU、NumPy、Python random 都使用 `config.misc.seed`；CUDA 可用时 `manual_seed_all`；cudnn deterministic=True / benchmark=False。没有 `seed+rank` 分支。模型构建在 `main.py:1626`，初始化参数会消费 torch 随机源；DDP 初始化会同步模型，不能据此假定不同分片产生相同增强或相同 dropout 样本。

训练增强源：`misc/data.py:109-116` 声明 ColorJitter、RandomRotation、RandomResizedCrop、RandomGrayscale、RandomHorizontalFlip、RandomErasing；`misc/data.py:177-184` 用 NumPy 随机选择其中两种，并在 Resize/ToTensor 之后、Normalize 之前应用。torchvision 的随机变换由其实现使用 torch RNG；主图在 `misc/caption_dataset.py:152-155` 执行 transform。这里既有 NumPy RNG，又有 torch RNG。

DataLoader 未显式设置 `worker_init_fn` 或 `generator` (`misc/data.py:145-159`)；迭代器会通过默认 CPU generator 生成 base_seed；工作进程由 PyTorch 初始化 Python random、torch、NumPy。`num_workers=0` 时增强在主进程消费这些状态；`num_workers>0` 时消费 worker 状态。跨运行要固定 torch/torchvision 版本、worker 数、loader 顺序与创建/迭代时机，而不仅固定配置 seed。

桥接启用时，选择与 transform 是隔离的上下文 (`misc/caption_dataset.py:168-187`)：桥接 seed 为 `(torch.initial_seed()+index)%2**32`；保存/恢复 Python 与 NumPy 状态，使用 `torch.random.fork_rng(devices=[])` 保存/恢复 torch CPU 状态。该分支不应推进主样本的 CPU RNG；不能以此推断其他所有模型随机路径也被隔离。

模型训练随机源包括 attention/dropout (`model/base_transformer.py:53-56,187-198,222,249`)、图像级 Gumbel 路由 (`:130`)，及 token 路由噪声 `torch.randn_like(...)*0.01` (`:145-147`)。梯度检查点使用 `use_reentrant=False` 且不覆盖默认 `preserve_rng_state=True` (`:12-19,198,252`)，仅 `self.training` 开启时重算；这保留一次重算的随机状态，不证明不同设备/批切分间抽样完全相同。

## 7. AMP、显式 half 与 TF32

TAG-PEDES 训练由 `main.py:1703` 的 CUDA GradScaler 和 `:1724` 的 CUDA autocast 包裹；`:1764-1781` 执行 scaled backward、scaler step/update，再清梯度并加 iteration。没有查到训练路径的 `clip_grad_norm_` / `clip_grad_value_` 或显式梯度裁剪。GradScaler 出现 overflow 时可能跳过优化器更新，因此“迭代计数”不必等于成功更新次数。

TSE 还有原有显式 half 转换：`model/CrossEmbeddingLayer_tse.py:86-92` 的文本 projection 在训练时使用 `features.half()`；`:116-124` 的图像局部特征在训练时 half，lengths 张量也 half，输出返回 float。评测明确传 `training=False` (`misc/eval.py:461,473`) 避免这两个特征 half 分支。审计描述现存路径，未降低或调整精度。

TAL 单独关闭 autocast 并把相似度输入转 float (`model/tbps_model_MoE.py:884-905`，非默认损失)；rerank 内部也有关闭 autocast (`misc/rerank.py:27`，非默认推理后处理)。`misc/eval.py:444-478` 标准 TSE 评测没有外层 autocast。

仓库未显式设置 `torch.backends.cuda.matmul.allow_tf32`、`torch.backends.cudnn.allow_tf32`、`torch.set_float32_matmul_precision` 或 `torch.use_deterministic_algorithms`；`cudnn.deterministic=True` 不等价于对全部算子打开确定性检查。TF32/AMP 实际行为需要记录运行时的 PyTorch/CUDA/GPU 与 backend 设置。仅作库实现核查：本审计机器 PyTorch 为 `2.5.1+cu121`，当前 matmul TF32=False、cudnn TF32=True、deterministic_algorithms=False；这些是本机读取值，不是 AutoDL 的配置或训练测试结果。

## 8. 评测 RNG 清单及 Gumbel 具体调用点

| 随机源 | 是否/如何在 baseline 设置 | 评测期间的消费点 |
| --- | --- | --- |
| Python random | `misc/utils.py:92`，启动时配置 seed | 标准 Resize/ToTensor/Normalize 无随机增强；loader worker 的库初始化会设置 worker Python RNG；扩展 dataset/变换需另核实 |
| NumPy | `misc/utils.py:91` | 标准测试 transform 无 NumPy 随机增强；worker 库初始化 NumPy 状态；训练 Choose 不在测试 transform 中 |
| torch CPU | `misc/utils.py:90` | 模型初始化在加载权重前会消费；loader iterator 的 base_seed；CPU 图像模型上的 Gumbel 如适用 |
| torch CUDA | `misc/utils.py:96` | 图像级 Gumbel-Softmax 每个 active MoE block 每批采样 |
| DataLoader worker | `misc/data.py:154-159` 无用户 generator/worker_init_fn | base_seed 由 loader generator/默认 CPU generator 产生；worker_id 派生的库种子 |
| Gumbel 路由 | 没有 baseline eval 专用种子 | `model/base_transformer.py:116,130-131`：`img_router` 对 view token 输出 logits，无条件 `F.gumbel_softmax(view_logits, hard=not training)` |
| token router noise | 非评测消费 | `model/base_transformer.py:145-147` 只在传入 `training=True` 时使用 `randn_like` |
| dropout | eval 中关闭 | `misc/eval.py:446` 调用 `model.eval()`；模块 dropout/attention 使用 module.training，checkpoint 同样在 eval 不执行 |

Gumbel 不是通过 `self.training` 自动关闭的层，传 `training=False` 只使 `hard=True`，仍然采样。标准评测调用链：`misc/eval.py:446,468-473` → `CLIP.encode_image(..., return_dense=True)` (`model/tbps_model_MoE.py:1106-1110`，默认 `training=False`) → `visual_transformer.forward` (`model/visual_transformer_MoE.py:156-169`) → `Transformer_MoE.forward` (`model/base_transformer.py:244-264`) → 每个 active MoE block 的 `forward_experts` (`:109-130`)。

在 `disable_moe=True` 情况下没有 active expert block，view token 在标准 block 前被移除 (`model/base_transformer.py:225-239,250-251`)，不会到 Gumbel 调用点；不应把该消融的确定性推广到默认 HR-MoE。

`main.py:1578` 先构造 loader，`:1626-1636` 再构造/加载 eval_only 模型；启动 seed 设置在 `:2032`，baseline 没有模型加载后的 eval 专用 seed。训练中每轮评测只 rank 0 消费图像 Gumbel 与 loader CPU base_seed (`:1791-1817`)，其他 rank 不走这段；下一轮随机流可能因而不同，不能假定所有 rank 永远共享相同 RNG 状态。

`scripts/E3_fresh_forward.py:64-90` 是另一条分析工具的 RNG/buffer 快照与检查；它不能作为 `main.py` 的默认评测已经隔离 RNG 的证据。

## 9. 库行为的只读核查依据与适用范围

为区分仓库显式逻辑与依赖库默认值，本机只读检查 PyTorch `2.5.1+cu121` 实现（目录 `torch/` 下的相对行号）：

- `torch/utils/data/distributed.py:72` sampler 默认 `drop_last=False`，`:110` 使用 `seed+epoch`，`:117-122` 补齐，`:130` 按 rank stride 分片。
- `torch/utils/data/dataloader.py:666-669` `_BaseDataLoaderIter` 用 `torch.empty(...).random_(generator=loader.generator)` 产生 base_seed。
- `torch/utils/data/_utils/worker.py:260-267` 设置 `base_seed+worker_id` 的 Python/torch 种子和派生 NumPy 种子。
- `torch/nn/parallel/distributed.py:396-397` 说明不同进程间参数梯度默认平均；`:634` 默认 `broadcast_buffers=True`。

这些行号属于本机依赖，不属于仓库基线。服务器 PyTorch 版本不同需按其实际依赖核查；没有把本机 API 实现阅读写成服务器多卡试验通过。

## 10. 可陈述与尚未证明的结论

**由基线源码确认：** gather 候选特征可微且 backward 做跨 rank SUM；主对比项本地锚点、全局候选；DDP 正常路径对参数梯度平均；sampler 会补齐、loader 会本地丢尾；TSE 有本地 BN；默认推理 HR-MoE 存在 Gumbel 随机性。

**尚未证明：** 单卡320与4卡×80的全部参数梯度/BN buffers/更新完全一致；4卡无 gather 与连续分块候选一致；梯度累积与单次完整前向一致；各机器不同精度 backend 的结果逐位一致。要比较必须固定并实际核查特征与 BN 状态、采样索引/顺序及补齐、局部归一分母、随机流/增强/worker、AMP scaling/overflow、候选侧 autograd、DDP 归约、优化步与调度、评测权重和 RNG 来源。

前向损失值相同只验证该次数值输出；不能据此推断这些条件成立或全部梯度相同。本轮 D1 单进程固定特征的独立块参考测试若通过，仅证明它覆盖的损失与参数梯度关系，不构成未运行多进程路径的证据。

## D8. Round B：原作者 TSE BatchNorm 的只读核查

核查基线 `8622dc2540fb28e20c4c344204ef75603c8f3b3b`。原作者 TSE 实现位于
`model/CrossEmbeddingLayer_tse.py:45-54`，定义的是普通 `nn.BatchNorm1d`，
不是 `nn.SyncBatchNorm`；训练时先把本 rank 的 token 特征由 `[B,N,D]`
reshape 为 `[B*N,D]`，再做第一层 BN。最后一层虽然声明 BN，但不调用。
文本/图像调用分别在同文件 `:90` / `:122`。

```python
# model/CrossEmbeddingLayer_tse.py:45-54
h = [hidden_dim] * (num_layers - 1)
self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

def forward(self, x):
    B, N, D = x.size()
    x = x.reshape(B * N, D)
    for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
        x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
```

`main.py:1822-1827` 直接包装 `DistributedDataParallel(...,
find_unused_parameters=True)`。模型/训练入口没有 `SyncBatchNorm`、
`convert_sync_batchnorm` 或跨进程合并 TSE BN batch 统计的操作。
原有可微 all-gather 用于损失候选，不在上述 BN 内。

因此：**4 卡×80 时每卡 BN 统计只覆盖本卡样本，单卡 320 时覆盖全批**。
更准确地说，覆盖对应图像/文本选中 token 的 `B*N` 行。
DDP 默认 buffer 广播不是当前 batch 统计的跨卡聚合。
这是源码语义结论，没有进行多 GPU 试验，不作任何性能影响推断。
