"""Compose rigid instances and reuse the repository's depth rasterizer."""

import math
import torch


def render_scene_graph(camera, scene_graph, pipe, bg_color, scaling_modifier=1.0):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    # Deliberately do not fall back to normalized frame-index camera.time.
    if not hasattr(camera, "timestamp_sec"):
        raise ValueError("Scene-graph cameras need timestamp_sec relative to the shared time origin")
    if getattr(camera, "time_origin_ns", None) != scene_graph.tracklets.time_origin_ns:
        raise ValueError("Camera and tracklets must use the same time_origin_ns")
    composed = scene_graph.compose(camera.timestamp_sec)
    pack = composed.pack
    if not len(pack.xyz):
        raise ValueError("No Gaussians active at the requested camera time")
    if not pack.xyz.is_cuda or pack.xyz.dtype != torch.float32:
        raise ValueError("The rasterizer requires CUDA float32 Gaussian parameters")
    device = pack.xyz.device
    screenspace = torch.zeros_like(pack.xyz, requires_grad=True)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*0.5), tanfovy=math.tan(camera.FoVy*0.5),
        bg=bg_color.to(device=device, dtype=torch.float32), scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform.to(device=device, dtype=pack.xyz.dtype),
        projmatrix=camera.full_proj_transform.to(device=device, dtype=pack.xyz.dtype),
        sh_degree=scene_graph.sh_degree, campos=camera.camera_center.to(device=device, dtype=pack.xyz.dtype),
        prefiltered=False, debug=getattr(pipe, "debug", False))
    # Explicit colors ensure actor SH directions rotate with the canonical frame.
    colors = scene_graph.colors(composed, settings.campos)
    image, radii, depth = GaussianRasterizer(settings)(
        means3D=pack.xyz, means2D=screenspace, shs=None,
        colors_precomp=colors.contiguous(), opacities=pack.opacity.contiguous(),
        scales=pack.scaling.contiguous(), rotations=pack.rotation.contiguous(), cov3D_precomp=None)
    return {"render": image, "depth": depth, "radii": radii,
            "viewspace_points": screenspace, "visibility_filter": radii > 0,
            "node_slices": composed.node_slices}
