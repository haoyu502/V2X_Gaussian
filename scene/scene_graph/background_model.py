from .actor_model import ActorGaussianModel


class BackgroundGaussianModel(ActorGaussianModel):
    """World-coordinate Gaussians, constant across timestamps."""
