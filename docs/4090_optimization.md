# RTX 4090 memory-optimized training

The optimized profile preserves synchronized V2X camera groups, but selects the
most informative timestamps using low-resolution image-change scores plus uniform
temporal anchors. It also performs one backward pass per camera view so only one
rasterization graph is retained at a time.

```bash
CUDA_VISIBLE_DEVICES=0 python train_v2x_gaussians.py \
  -s /path/to/V2X_GOTR_scene \
  --expname scene_4090 \
  --configs arguments/multi_agents/v2x_gaussian_4090.py
```

The profile keeps 120 synchronized timestamps, trains at half resolution with
SH degree 2, disables temporal SH deformation, and caps the scene at two million
Gaussians. At most 50,000 points may be added by one densification event.

Useful overrides:

```bash
# Safer for a scene that still exceeds 24 GiB
--resolution 4 --max_gaussians 1200000 --max_densify_points 25000

# Higher quality after confirming memory headroom
--resolution 2 --keyframe_count 180 --max_gaussians 3000000
```

Multi-GPU training uses one process per GPU. Camera-view gradients and
densification statistics are synchronized, while deterministic Gaussian growth
keeps the dynamically resized model identical on every rank:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train_v2x_gaussians.py \
  -s /path/to/V2X_GOTR_scene \
  --expname scene_4x4090 \
  --configs arguments/multi_agents/v2x_gaussian_4090.py
```

The Gaussian model is replicated, so VRAM is not pooled; multi-GPU execution
splits camera rendering and accelerates training while the memory guards keep
each replica below the per-card budget.

Every run automatically writes line-buffered logs under the experiment folder:

```text
output/scene_4x4090/logs/train_<run-id>_rank0.log
output/scene_4x4090/logs/train_<run-id>_rank1.log
```

Rank 0 contains the full training output. Worker logs retain uncaught Python
tracebacks and fatal-signal diagnostics, so shell `tee` redirection is optional.
