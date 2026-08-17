ModelParams = dict(
    sh_degree=2,
    resolution=2,
    # The released scenes commonly contain 50 synchronized timestamps.
    # Keep 30 high-change/coverage timestamps (90 images across three cameras).
    keyframe_count=30,
    keyframe_min_gap=2,
    render_process=False,
)

ModelHiddenParams = dict(
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 24,
        'resolution': [48, 48, 48, 25],
    },
    multires=[1, 2, 4],
    defor_depth=0,
    net_width=96,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    no_dx=False,
    no_grid=False,
    no_ds=True,
    no_dr=True,
    no_do=True,
    no_dshs=True,
    empty_voxel=False,
    static_mlp=False,
)

OptimizationParams = dict(
    dataloader=False,
    iterations=20_000,
    coarse_iterations=2_500,
    batch_size=1,
    densify_from_iter=500,
    densify_until_iter=5_000,
    densification_interval=200,
    pruning_from_iter=500,
    pruning_interval=100,
    opacity_reset_interval=6_000,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
    max_gaussians=2_000_000,
    max_densify_points=50_000,
    memory_log_interval=250,
)
