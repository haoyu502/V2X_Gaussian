_base_ = ['./v2x_gaussian_4090.py']

# Full proposed method: ego visibility field + selective collaborative gradient
# fusion + occlusion-guided cross-ray densification.
OptimizationParams = dict(
    selective_collaboration=True,
    ego_camera_uid=2,
    collaboration_overlap_weight=0.1,
    collaboration_blind_weight=1.0,
    collaboration_network_min_weight=0.1,
    occlusion_guided_densification=True,
    occlusion_densify_weight=2.0,
    collaboration_log_interval=250,
)
