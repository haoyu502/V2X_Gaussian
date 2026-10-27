# 场景图首个里程碑

已新增独立的刚体场景图训练入口 `train_scene_graph.py`：静态背景高斯 + 每个车辆 ID 的局部高斯 + 连续时间位姿。Ego、South1 和 South2 通过各自的图像时间监督同一个实例节点。

目前是**固定高斯数量的功能验证版本**，用于验证坐标、时间、三视角渲染、梯度与检查点。尚不能作为完整 E1 重建质量结果；没有实现增密/剪枝、可学习位姿、跨端消息传输、按需协作或 LPIPS/实例区域评估。原来的 `train_v2x_gaussians.py`、全局 GaussianModel 和 HexPlane 训练算法未改动。两处包入口改为延迟导入旧模型，`utils/loss_utils.py` 去掉未使用的 `lpips` 导入，使新路径不依赖旧形变模型的环境包。

## 实现与约定

| 文件 | 用途 |
| --- | --- |
| `scene/scene_graph/actor_model.py` | 可训练的实例局部高斯，位置、SH、尺度、四元数和不透明度 |
| `scene/scene_graph/actor_pose.py` | 平移线性插值、四元数最短路径 SLERP；不越界外推，不跨越长轨迹缺口 |
| `scene/scene_graph/graph_model.py` | 合成背景与当前时间的实例，并返回每个节点在合并数组中的切片 |
| `scene/scene_graph/dataset.py` | 带明确标定、时间原点和训练/测试划分的独立输入接口 |
| `gaussian_renderer/scene_graph_renderer.py` | 复用现有深度光栅化器；SH 在实例局部方向上计算 |
| `scripts/prepare_scene_graph.py` | 从本地 TUMTraf ZIP 按逐帧点云切分背景/实例，生成最小训练数据 |

内部旋转统一为 **wxyz、局部到世界**；OpenLABEL 的 xyzw 在解析时转换。背景坐标、框和相机必须处于同一个世界坐标系。所有包内高斯参数已经激活：尺度是 `exp(log_scale)`，旋转已归一化，不透明度是 `sigmoid(logit)`。实例运动同时变换位置与高斯旋转，等价于协方差变换 `R Σ Rᵀ`。

原始时间戳以**整数纳秒**保留，先减共享 `time_origin_ns` 再转换为秒，避免把 epoch 秒变成 float32 后丢失毫秒级偏移。`sync_group_id` 只用于采样关联，不会把三端时间改为相同。旧数据集额外提供 `scene_graph_metadata(index, time_origin_ns)`，原来的 `camera.time` 不受影响。

## 环境与自检

以下命令在仓库根目录、已激活的 `v2x_gaussian` 环境执行。需要 PyTorch、NumPy、Pillow、OpenCV；GPU 路径使用仓库已有深度 rasterizer，未修改 CUDA 源码。

```bash
export OMP_NUM_THREADS=4
MAX_JOBS=4 CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=8.0 \
  python -m pip install --no-build-isolation --no-deps ./submodules/depth-diff-gaussian-rasterization
python -m unittest discover -s tests -v
```

合成小样本有三个相机、不同的图像时间，目标图像由已知场景图生成。只验证实现，不用其指标作论文结论。输出目录必须为空或使用新的目录名。

```bash
python scripts/smoke_test_scene_graph.py --output data/scene_graph_smoke
python train_scene_graph.py \
  --manifest data/scene_graph_smoke/manifest.json \
  --output output/scene_graph_smoke \
  --iterations 100 --initial-scale 0.12
```

## 本地真实数据的小样本

```bash
python scripts/prepare_scene_graph.py \
  --archive /root/tumtraf_v2x_cooperative_perception_dataset.zip \
  --output data/scene_graph_tumtraf_smoke \
  --start-ns 1688625741046595374 \
  --frames 12 --max-bg-points 20000 --max-actor-points 1000

python train_scene_graph.py \
  --manifest data/scene_graph_tumtraf_smoke/manifest.json \
  --output output/scene_graph_tumtraf_smoke \
  --iterations 30 --downsample 4 --initial-scale 0.12
```

`--downsample 4`、SH0、小点数和短训练只为快速验证；这里没有更改原基线配置。完整质量实验应固定数据划分、恢复足够的高斯容量与分辨率，并在实现增密后与基线公平对比。

预处理行为：

- 跨数据原始 train/val/test 目录按文件名解析声明的传感器关联，不用排序后的 `zip()` 匹配。该命令创建自己的时间留出划分，**不等于原数据集官方划分或论文划分**。
- 每端保留自己的图像时间。主车相机也插值到图像时间；由于 metadata 没有单独给变换加时间，暂以 metadata 参考时刻作为变换时刻，假设写入 manifest。这个假设需要在正式实验前核实。
- 从 12 个参考帧中保留首尾两帧仅作位姿插值支撑，其余 10 组相机每 5 组留出一组，所有端同时留出：24 张训练图、6 张测试图。
- 轨迹使用标注，包含测试时刻的标注位姿，属于 **oracle pose 设置**；只有训练帧的基础设施 LiDAR 用来初始化几何。不会用测试点云、测试图像的颜色或未分离的合并 PLY 初始化几何。
- 不使用已融合的车路点云，因为其中含有不同时刻的观测。当前只有基础设施 LiDAR 初始化，尚不能证明跨端点云补全。
- 在每帧世界点云中先移除**全部**已标注框，再把支持的车辆点变到局部规范空间累计。行人、自行车、摩托车等暂不重建；无训练 LiDAR 点的实例也会省略。背景不会保留这些框内的点，但全图损失尚未屏蔽这些类别，因此当前结果不适合动态质量比较。
- 初始颜色为中性灰。稀疏、高斯不增密、背景无天空、部分实例点缺失，都会限制当前渲染质量。

`overlays/centers_*.jpg` 是标定检查图，绘制参考时刻的标注中心；未作遮挡剔除，因此被遮住车辆的中心也可能画在前景物体上。正式训练前需要进一步检查完整框投影、相机畸变和位姿误差。内置日间投影来自数据作者的 [CoopDet3D 转换代码](https://github.com/tum-traffic-dataset/coopdet3d/blob/master/tools/data_converter/tumtraf_v2x_converter.py)；这与旧加载器的部分硬编码相机参数不同，正式比较必须让两条路径使用同一套核验过的标定，不能把差异归因于场景图。

## 输入输出

数据目录包含 `manifest.json`、`tracks.json`、`background.npz`、`actors/<ID>.npz` 和图像。NPZ 必须有 `[N,3]` 的 `xyz` 和 `[0,1]` 的 `rgb`。背景 XYZ 在世界坐标系，实例 XYZ 在对应局部坐标系。

Manifest 的关键字段：

```json
{
  "schema_version": 1,
  "world_frame": "s110_lidar_ouster_south",
  "time_origin_ns": 1688625741046595374,
  "tracks": "tracks.json",
  "background": "background.npz",
  "actors": {"car_uuid": "actors/car_uuid.npz"},
  "cameras": [{
    "image": "images/example.jpg",
    "agent": "ego",
    "timestamp_ns": 1688625741052165680,
    "sync_group_id": "reference_frame_id",
    "split": "train",
    "intrinsics": [[1000, 0, 960], [0, 1000, 600], [0, 0, 1]],
    "world_to_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  }]
}
```

`tracks.json` 必须声明相同的 `world_frame`、`time_origin_ns`、`quaternion_order: "wxyz"`，每个 `objects` 条目包含 `track_id` 及按时间递增的 `observations`，每个观测提供 `timestamp_ns`、`position` 和 `rotation_wxyz`。不支持超出轨迹范围外推；过大的时间缺口也不会插值。

训练输出为 `checkpoint.pt`、`metrics.json`、`run_args.json` 和 `test/*.png`。测试指标只有全图 PSNR、SSIM，尚无 LPIPS 或实例 mask 指标。`radii > 0` 只代表投影/光栅化可见，不作为真正遮挡或可靠性判断。

```bash
python train_scene_graph.py \
  --manifest data/scene_graph_smoke/manifest.json \
  --output output/scene_graph_smoke_resume \
  --resume output/scene_graph_smoke/checkpoint.pt \
  --iterations 120 --initial-scale 0.12
```

检查点保存节点参数、轨迹缓冲区、时间原点、插值间隔、优化器和随机状态；恢复前核对 manifest/轨迹指纹和相关训练参数。输入文件需要保留。仅加载可信的本地检查点。

下一阶段是增密/剪枝及节点级统计回传，再建立一致标定、固定划分和动态区域指标的 E0/E1 对比；当前代码没有宣称协同效果或通信量收益。

## 本次运行记录

在 NVIDIA A800 80GB、PyTorch 2.4.0+cu124 上完成 CUDA 扩展构建和数学/渲染/梯度测试。合成三视角训练 100 步，训练损失约 `0.020174 -> 0.012863`，另完成从第 100 步恢复至第 120 步的测试。

真实数据的最终检查目录为 `data/scene_graph_tumtraf_validated/` 和 `output/scene_graph_tumtraf_validated/`：20 个实例、20,000 个背景高斯、24 张训练图和 6 张留出图；30 步训练损失约 `0.518366 -> 0.455527`。不同步随机组的损失只说明训练正常推进，不表示收敛。检查点、PSNR/SSIM 和留出渲染已保存。图像仍然稀疏，不能据此声称达到或超过原基线。
