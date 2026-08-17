_base_ = ['./v2x_gaussian_adaptive_occlusion.py']

# Ablation for v2: adaptive gradient fusion without guided densification.
OptimizationParams = dict(
    occlusion_guided_densification=False,
)
