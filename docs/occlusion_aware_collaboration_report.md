# V2X-Gaussians 遮挡感知选择性协作：阶段性汇报

> 更新日期：2026-08-18  
> 数据集：V2X-GOTR（来源于 TUMTraf-V2X，共 9 个真实交通场景）  
> 训练设备：RTX 4090 D（24 GB），多卡数据并行  
> 基础代码：V2X-Gaussians
>
> 工作版本：4090 工程基线（main）→ 遮挡协作第一版 → 自适应遮挡协作第二版

## 1. 工作演进概览

本项目不是直接在原始 V2X-Gaussians 上增加遮挡模块，而是分为三个阶段：

```text
原始 V2X-Gaussians（单卡 A100 设计）
        │
        │  24 GB RTX 4090 OOM
        ▼
4090 工程基线 / main
关键帧 + 逐视角反传 + 多卡同步 + Gaussian 显存预算
        │
        │  解决“能稳定训练和批量评测”
        ▼
第一版：二值遮挡感知协作
        │
        │  平均 PSNR 提升，但部分场景不稳定
        ▼
第二版：连续置信度与有界自适应协作
```

三个阶段分别对应“工程可运行性”“方法可行性验证”和“方法稳定性改进”。后两个方法版本
都建立在 main 分支的 4090 工程基线上。

## 2. 基础工程版本：从单卡 A100 到多卡 RTX 4090

### 2.1 初始问题

原始源码面向单卡 A100。迁移到单张 RTX 4090（24 GB）后，训练会因以下原因 OOM：

- 同一 iteration 同时保留多个 agent 的 rasterizer 计算图；
- 原始流程处理全部同步帧，图像和训练计算冗余较大；
- cross-ray densification 使 Gaussian 数量快速增长；
- Gaussian 参数及 Adam 优化器状态随点数共同增长；
- 缺少针对消费级显卡的点数上限和单次增密预算。

实际调试中曾观察到 Gaussian 从约 82 万增长到约 104.5 万后触及 24 GB 显存上限。
因此，仅降低 batch size 或修改一个配置项不足以解决问题，需要同时调整数据、计算图、
多卡同步和 Gaussian 生命周期。

### 2.2 关键帧筛选

V2X 场景中的三个 agent 必须保持时间同步，因此不能独立随机删除单张图片。main 分支先按
timestamp 将多相机图像组成同步组，再结合低分辨率图像变化分数和均匀时间锚点选择关键帧。

当前 4090 配置从常见的 50 个同步时间戳中保留 30 个，即从 150 张图像缩减到 90 张，
并设置最小时间间隔 2。这样能够保留运动变化与时间覆盖，同时减少约 40% 的输入帧。

### 2.3 逐视角反向传播

原流程容易同时持有多个相机的渲染图。优化后，每个相机执行：

```text
render → loss → backward → 释放当前相机计算图 → 下一个相机
```

因此显存峰值由“多个 agent 图之和”降低为“单个 agent 图 + 共享 Gaussian 参数”。损失仍按
视角数归一化，保持多视角监督。

### 2.4 多卡训练

main 分支增加 `torchrun + NCCL` 多进程训练：

- 每个 rank 只负责一个相机视角的渲染与反传；
- 所有 Gaussian 参数梯度执行 all-reduce；
- 可见性、投影半径和 densification 梯度跨卡同步；
- cross-ray 区域跨 rank 聚合；
- 所有 rank 使用确定性随机种子执行相同增密和剪枝，保持模型结构一致；
- 仅 rank 0 保存模型、执行评测和显示主进度条。

需要强调：当前方案是数据并行，模型在每张卡各保存一份，所以多卡显存不会合并成
`4 × 24 GB`。多卡的作用是拆分相机计算和提高吞吐量；单卡显存安全仍由逐视角反传与
Gaussian 预算保证。

### 2.5 Gaussian 和网络预算

4090 配置的主要参数如下：

| 项目 | 4090 工程基线 |
|---|---:|
| 图像分辨率缩放 | 2 |
| SH degree | 2 |
| 关键时间戳 | 30/50 |
| 总迭代数 | 20,000 |
| coarse 迭代数 | 2,500 |
| densify 截止 | 5,000 |
| densify 间隔 | 200 |
| Gaussian 上限 | 2,000,000 |
| 单次最大新增点 | 50,000 |
| K-Planes 分辨率 | `[48, 48, 48, 25]` |
| deformation width | 96 |

同时关闭当前实验不需要的 scale、rotation、opacity 和 SH 动态形变分支，减少计算和参数
开销。增密前根据剩余 Gaussian 容量分配 clone/split 预算，达到上限后停止继续增长。

### 2.6 数据与实验自动化

main 分支还完成了工程闭环：

- 自动为全部 9 个场景生成或复用 `lidar_downsampled.ply`；
- 自动依次训练所有场景，并跳过已有完整 checkpoint 的场景；
- 每个 rank 自动写入持久化日志和异常 traceback；
- 自动渲染 test 重建视频；后续实验分支进一步补充 novel-view 视频；
- 自动计算 PSNR、SSIM、LPIPS、MS-SSIM 和 D-SSIM；
- 自动生成逐场景及平均指标 CSV/JSON；
- 训练或评测中断后可重新运行并跳过已有产物。

main 分支的批量训练结果构成后续方法对比的工程基线：

| PSNR ↑ | SSIM ↑ | LPIPS-VGG ↓ | LPIPS-Alex ↓ | MS-SSIM ↑ |
|---:|---:|---:|---:|---:|
| 27.599 | 0.8664 | 0.2493 | 0.1723 | 0.8955 |

对应版本为 `main`，提交 `125d2a7`，配置文件为
`arguments/multi_agents/v2x_gaussian_4090.py`。

## 3. 研究背景与问题

V2X-Gaussians 通过 ego 车辆与路侧单元（RSU）的多视角图像共同优化一组动态
Gaussians，并通过 cross-ray densification 补充多视角相交区域的几何。原方法的主要
问题是：不同 agent 的梯度基本被无差别融合，高损失区域也可能直接触发增密。

这种处理没有区分：

- ego 已经可靠观测的区域；
- ego 被遮挡、但协作相机可见的盲区；
- 多视角不一致、位姿误差或光照变化造成的伪高损失区域。

因此，本工作借鉴 FRUC 的“协作信息应主要补充 ego 盲区，而不应破坏 ego 可靠几何”
思想，但保留 V2X-Gaussians 的逐场景优化框架，利用已有相机标定、LiDAR 初始化和显式
Gaussian 几何，研究从无差别协作到遮挡感知选择性协作的改进。

## 4. 总体技术路线

```text
同步 ego / RSU 图像
        │
        ├── ego Gaussian 可见性 ─────────────┐
        │                                    │
        └── collaborator Gaussian 可见性 ────┤
                                             ▼
                                  ego-centric 遮挡置信度
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
               选择性协作梯度融合                      遮挡引导 cross-ray 增密
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             ▼
                                  动态 Gaussian 场景表示
```

多卡角色固定为：rank 0 处理 ego 车载相机（当前数据 UID 为 2），其余 rank 处理路侧
协作视角。每张卡只保留一个 rasterizer 计算图，以适配 24 GB 显存。

## 5. 第一版：二值遮挡感知协作

### 5.1 方法

第一版根据 rasterizer 返回的逐 Gaussian `visibility_filter` 和投影半径 `radii` 构造
二值可见性场：

```text
ego 不可见且 collaborator 可见 → ego blind spot
ego 与 collaborator 都可见     → overlap
```

协作梯度初始规则为：

\[
w_i^{collab}=
\begin{cases}
1.0, & i \in \text{blind spot}\\
0.1C_i, & i \in \text{overlap}\\
0, & \text{otherwise}
\end{cases}
\]

其中，\(C_i\) 是由 ego 与协作视角投影半径差异得到的轻量几何一致性。cross-ray
强制克隆只允许发生在二值 blind-spot mask 内，盲区的增密梯度放大 2 倍。

### 5.2 实测结果

| 指标 | 原 2×4090 基线 | 第一版 4×4090 | 变化 |
|---|---:|---:|---:|
| PSNR ↑ | 27.599 | **28.126** | **+0.527 dB** |
| SSIM ↑ | 0.8664 | **0.8695** | **+0.0031** |
| LPIPS-VGG ↓ | **0.2493** | 0.2638 | -0.0145（变差） |
| LPIPS-Alex ↓ | **0.1723** | 0.1966 | -0.0244（变差） |
| MS-SSIM ↑ | **0.8955** | 0.8741 | -0.0215（变差） |

逐场景 PSNR：

| 场景 | 基线 | 第一版 | 差值 |
|---|---:|---:|---:|
| dense_vru_crossing | 26.798 | 25.831 | -0.967 |
| dense_vru_with_rsu_occlusion | 29.357 | 21.810 | **-7.547** |
| ego_vehicle_occlusion | 22.049 | 21.032 | -1.017 |
| far_distance_pedestrian | 22.382 | 20.944 | -1.437 |
| infrastructure_occlusion | 27.869 | 31.629 | **+3.760** |
| night_scene | 24.687 | 30.169 | **+5.482** |
| pedestrian_crossing | 31.511 | 34.087 | **+2.576** |
| sharp-u_turn_mauever | 32.306 | 34.060 | **+1.754** |
| u_turn_maneuver | 31.432 | 33.571 | **+2.139** |

### 5.3 阶段结论

第一版在 5/9 个场景提升，尤其对 infrastructure occlusion、夜间和转弯场景有效，
说明“优先补充 ego 盲区”的方向成立。但 4 个密集或困难场景下降，且 LPIPS 变差，表明
二值遮挡判断和全强度盲区协作过于激进。

进一步检查还发现，第一版的加权梯度先乘权重、随后又除以权重和。在只有协作视角可见
时，权重会被抵消，导致名义上的权重控制没有真正降低协作更新强度。这是第二版重点修复
的实现与建模问题。

## 6. 第二版：连续置信度与有界自适应协作

### 6.1 连续可见性与遮挡分数

第二版不再只使用可见/不可见二值判断。使用 Gaussian 的投影半径构造连续可见置信度：

\[
q_i=\frac{r_i}{r_i+s},\qquad s=4.0
\]

随后计算：

\[
S_i^{occ}=(1-q_i^{ego})q_i^{collab}
\]

它表达“ego 越不可靠，同时 collaborator 越可靠，协作价值越高”。

### 6.2 真正有界的梯度融合

第二版将梯度同步改为：

\[
\nabla G_i=\nabla G_i^{ego}
+w_i^{collab}\operatorname{Mean}(\nabla G_i^{collab})
\]

不再用权重和归一化，从而避免权重被抵消。主要保护参数为：

| 参数 | 第一版 | 第二版 |
|---|---:|---:|
| blind-spot 基础权重 | 1.00 | 0.75 |
| 最大协作权重 | 无单独上限 | 0.60 |
| overlap 权重 | 0.10 | 0.05 |
| 网络最小协作权重 | 0.10 | 0.05 |

第二版的协作更新由连续遮挡分数调制，并被限制在 `[0, 0.6]`。这能防止密集动态场景中
大量 collaborator-only Gaussians 以全强度更新，同时进一步保护 ego 已观测区域。

### 6.3 更保守的遮挡增密

第一版对全部二值盲区执行强制 cross-ray 增密；第二版要求：

\[
S_i^{occ}\ge 0.65
\]

才允许强制增密，并将最大增密放大倍率从 2.0 降至 1.5。目标是减少因低置信度遮挡、
跨视角误差和动态不一致造成的冗余 Gaussian。

### 6.4 当前状态

第二版代码已经完成，静态编译和配置继承检查通过，但最终 9 场景指标尚未生成。因此，
目前只能报告方法改动，不能宣称第二版优于第一版。

## 7. 三个版本的工程隔离与复现

| 版本 | Git 分支 | 固定提交 | 配置文件 |
|---|---|---|---|
| 4090 工程基线 | `main` | `125d2a7` | `v2x_gaussian_4090.py` |
| 第一版 | `agent/occlusion-aware-collaboration` | `1509cf7` | `v2x_gaussian_occlusion.py` |
| 第二版 | `agent/adaptive-occlusion-collaboration` | `8d8eb44` | `v2x_gaussian_adaptive_occlusion.py` |

第一版结果不会被第二版覆盖。切换版本：

```bash
git switch agent/occlusion-aware-collaboration   # 第一版
git switch agent/adaptive-occlusion-collaboration  # 第二版
```

第二版四卡完整实验：

```bash
python scripts/run_full_pipeline.py \
  --gpus 0,1,4,5 \
  --suffix adaptive_occlusion_4x4090
```

第二版仅自适应梯度消融：

```bash
python scripts/run_full_pipeline.py \
  --gpus 0,1,4,5 \
  --config arguments/multi_agents/v2x_gaussian_adaptive_gradient.py \
  --suffix adaptive_gradient_4x4090
```

## 8. 后续实验设计

建议按以下顺序形成完整消融表：

| 实验 | 选择性梯度 | 连续置信度 | 遮挡增密 | 目的 |
|---|:---:|:---:|:---:|---|
| 4090 baseline | × | × | × | 原始参照 |
| 第一版 | ✓ | × | ✓ | 验证二值选择性协作 |
| 第二版 gradient-only | ✓ | ✓ | × | 隔离自适应梯度收益 |
| 第二版 full | ✓ | ✓ | ✓ | 验证完整方法 |

除全图 PSNR、SSIM、LPIPS 外，还应增加：

- ego blind-spot crop 的 PSNR、LPIPS 或 NIQE；
- 动态区域指标；
- 每场景最终 Gaussian 数量；
- 峰值显存、训练时间和渲染 FPS；
- 对 `dense_vru_with_rsu_occlusion` 等下降场景的单独可视化。

严格比较时，还应补跑同为 4 卡、相同相机分配策略的 baseline。当前基线为 2 卡、第一版
为 4 卡，虽然核心训练配置相同，但硬件并行度和每轮视角分配不同，属于潜在混杂因素。

## 9. 汇报结论

1. 首先完成了从单卡 A100 源码到 2/4 卡 RTX 4090 的工程迁移，解决 OOM、动态点数失控、
   多卡结构同步和批量评测问题，为方法研究建立了可复现基线。
2. 第一版证明了遮挡感知选择性协作具有潜力：平均 PSNR 提升 0.527 dB，5/9 场景提升。
3. 第一版仍存在明显不稳定性：密集动态场景下降，感知指标 LPIPS 变差。
4. 原因包括二值遮挡过硬、盲区更新过强、强制增密范围过大，以及梯度权重归一化抵消。
5. 第二版通过连续遮挡置信度、有界协作平均梯度和高置信度增密进行针对性修复。
6. 下一阶段需要完成第二版 full/gradient-only 消融，并补充 blind-spot 专项指标，才能形成
   最终方法结论。

## 10. 结果文件

- 原 2×4090 基线：`output/evaluation_summary_2x4090.csv/json`
- 第一版 4×4090：`output/evaluation_summary_occlusion_4x4090.csv/json`
- 第二版完成后：`output/evaluation_summary_adaptive_occlusion_4x4090.csv/json`
- 每个场景的视频：`output/<实验名>/test/ours_20000/video_rgb.mp4`
- Novel-view 视频：`output/<实验名>/video/ours_20000/video_rgb.mp4`
