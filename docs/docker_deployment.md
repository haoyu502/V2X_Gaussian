# V2X-Gaussians Docker 部署

该镜像固定以下核心环境：

- Ubuntu 22.04
- CUDA 12.4.1 + cuDNN 开发环境
- Python 3.9
- PyTorch 2.4.0 + CUDA 12.4
- 项目 Python 依赖
- `depth-diff-gaussian-rasterization` CUDA 扩展
- `simple-knn` CUDA 扩展

数据集和训练输出不写入镜像，通过目录挂载提供。

## 1. 构建并导出镜像

构建机器需要安装 Docker。构建过程不要求 GPU，但需要网络下载基础镜像和依赖：

```bash
bash docker/build_and_export.sh
```

默认生成：

```text
v2x-gaussian-v3-docker.tar.gz
```

默认 CUDA 架构包含 A100（8.0）、Ampere 8.6 和 RTX 4090（8.9）。如果只在 4090 上运行，可以减小镜像构建时间和扩展体积：

```bash
TORCH_CUDA_ARCH_LIST=8.9 bash docker/build_and_export.sh
```

## 2. 上传到云服务器

```bash
scp v2x-gaussian-v3-docker.tar.gz user@cloud-server:/path/to/
```

数据集建议单独同步：

```bash
rsync -avP data/ user@cloud-server:/path/to/v2x-data/
```

## 3. 云服务器导入镜像

云服务器需要：

1. NVIDIA 驱动；
2. Docker；
3. NVIDIA Container Toolkit；
4. `docker run --gpus all` 能够正常执行。

导入镜像：

```bash
gzip -dc v2x-gaussian-v3-docker.tar.gz | docker load
```

验证 GPU 和 PyTorch：

```bash
docker run --rm --gpus all v2x-gaussian:v3 \
  python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## 4. 运行第三版 E1

在代码仓库中运行宿主机包装脚本：

```bash
bash docker/run_pipeline.sh \
  /absolute/path/to/v2x-data \
  /absolute/path/to/v2x-output \
  0,1,2,3
```

脚本会把数据只读挂载到容器的 `data/`，并把结果写入宿主机指定的输出目录。

## 5. 运行 E2 和 E3

E2（残差强度 0.50）：

```bash
bash docker/run_pipeline.sh \
  /absolute/path/to/v2x-data \
  /absolute/path/to/v2x-output \
  0,1,2,3 \
  arguments/multi_agents/v2x_gaussian_residual_enhancement_050.py \
  e2_residual_050_4x4090
```

E3（公平 4 卡 baseline）：

```bash
bash docker/run_pipeline.sh \
  /absolute/path/to/v2x-data \
  /absolute/path/to/v2x-output \
  0,1,2,3 \
  arguments/multi_agents/v2x_gaussian_4090.py \
  e3_baseline_4x4090
```

## 6. 兼容性说明

Docker 镜像包含 CUDA 用户态运行库，但不包含宿主机 NVIDIA 内核驱动。云服务器仍必须安装兼容驱动。建议驱动支持 CUDA 12.4；如果云端驱动较旧，应升级驱动或改用匹配的 CUDA 基础镜像重新构建。

如果云端 GPU 不属于默认的 8.0、8.6 或 8.9 架构，需要用对应的 `TORCH_CUDA_ARCH_LIST` 重新构建 CUDA 扩展。
