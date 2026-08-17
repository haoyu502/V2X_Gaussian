_base_ = ['./v2x_gaussian_4090.py']

# Stable v2: continuous visibility confidence, bounded collaborative gradients,
# and conservative occlusion-guided densification.
OptimizationParams = dict(
    selective_collaboration=True,
    adaptive_collaboration=True,
    ego_camera_uid=2,
    visibility_radius_scale=4.0,
    collaboration_overlap_weight=0.05,
    collaboration_blind_weight=0.75,
    collaboration_max_weight=0.6,
    collaboration_network_min_weight=0.05,
    occlusion_guided_densification=True,
    occlusion_densify_weight=1.5,
    occlusion_score_threshold=0.65,
    collaboration_log_interval=250,
)
