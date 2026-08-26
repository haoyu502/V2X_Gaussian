_base_ = './v2x_gaussian_4090.py'

# V3: the original V2X multi-GPU gradient and densification paths remain
# untouched; reliable collaborator evidence is an additive residual only.
OptimizationParams = dict(
    selective_collaboration=True,
    adaptive_collaboration=True,
    residual_collaboration=True,
    ego_camera_uid=2,
    visibility_radius_scale=4.0,
    collaboration_blind_weight=1.0,
    collaboration_overlap_weight=1.0,
    collaboration_max_weight=1.0,
    collaboration_enhancement=0.25,
    collaboration_ramp_start=2000,
    collaboration_ramp_end=4000,
    occlusion_guided_densification=False,
    collaboration_log_interval=250,
)
