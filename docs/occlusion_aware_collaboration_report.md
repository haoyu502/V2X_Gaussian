# V2X-Gaussians 遮挡感知选择性协作：阶段性汇报

> 更新日期：2026-08-18  
> 数据集：V2X-GOTR（来源于 TUMTraf-V2X，共 9 个真实交通场景）  
> 训练设备：RTX 4090 D（24 GB），多卡数据并行  
> 基础代码：V2X-Gaussians

## 1. 研究背景与问题

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

## 2. 总体技术路线

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

## 3. 第一版：二值遮挡感知协作

### 3.1 方法

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

### 3.2 实测结果

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

### 3.3 阶段结论

第一版在 5/9 个场景提升，尤其对 infrastructure occlusion、夜间和转弯场景有效，
说明“优先补充 ego 盲区”的方向成立。但 4 个密集或困难场景下降，且 LPIPS 变差，表明
二值遮挡判断和全强度盲区协作过于激进。

进一步检查还发现，第一版的加权梯度先乘权重、随后又除以权重和。在只有协作视角可见
时，权重会被抵消，导致名义上的权重控制没有真正降低协作更新强度。这是第二版重点修复
的实现与建模问题。

## 4. 第二版：连续置信度与有界自适应协作

### 4.1 连续可见性与遮挡分数

第二版不再只使用可见/不可见二值判断。使用 Gaussian 的投影半径构造连续可见置信度：

\[
q_i=\frac{r_i}{r_i+s},\qquad s=4.0
\]

随后计算：

\[
S_i^{occ}=(1-q_i^{ego})q_i^{collab}
\]

它表达“ego 越不可靠，同时 collaborator 越可靠，协作价值越高”。

### 4.2 真正有界的梯度融合

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

### 4.3 更保守的遮挡增密

第一版对全部二值盲区执行强制 cross-ray 增密；第二版要求：

\[
S_i^{occ}\ge 0.65
\]

才允许强制增密，并将最大增密放大倍率从 2.0 降至 1.5。目标是减少因低置信度遮挡、
跨视角误差和动态不一致造成的冗余 Gaussian。

### 4.4 当前状态

第二版代码已经完成，静态编译和配置继承检查通过，但最终 9 场景指标尚未生成。因此，
目前只能报告方法改动，不能宣称第二版优于第一版。

## 5. 两个版本的工程隔离与复现

| 版本 | Git 分支 | 固定提交 | 配置文件 |
|---|---|---|---|
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

## 6. 后续实验设计

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

## 7. 汇报结论

1. 第一版证明了遮挡感知选择性协作具有潜力：平均 PSNR 提升 0.527 dB，5/9 场景提升。
2. 第一版仍存在明显不稳定性：密集动态场景下降，感知指标 LPIPS 变差。
3. 原因包括二值遮挡过硬、盲区更新过强、强制增密范围过大，以及梯度权重归一化抵消。
4. 第二版通过连续遮挡置信度、有界协作平均梯度和高置信度增密进行针对性修复。
5. 下一阶段需要完成第二版 full/gradient-only 消融，并补充 blind-spot 专项指标，才能形成
   最终方法结论。

## 8. 结果文件

- 原 2×4090 基线：`output/evaluation_summary_2x4090.csv/json`
- 第一版 4×4090：`output/evaluation_summary_occlusion_4x4090.csv/json`
- 第二版完成后：`output/evaluation_summary_adaptive_occlusion_4x4090.csv/json`
- 每个场景的视频：`output/<实验名>/test/ours_20000/video_rgb.mp4`
- Novel-view 视频：`output/<实验名>/video/ours_20000/video_rgb.mp4`
