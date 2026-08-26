_base_ = './v2x_gaussian_residual_enhancement.py'

# E2 ablation: a stronger residual; all other settings match V3 E1.
OptimizationParams = dict(
    collaboration_enhancement=0.50,
)
