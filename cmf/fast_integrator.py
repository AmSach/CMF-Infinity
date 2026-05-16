from __future__ import annotations

import torch


def _load_extension():
    try:
        import cmf_cuda  # type: ignore

        return cmf_cuda
    except ImportError:
        return None


def euler_integrate_precomputed(
    z0: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    use_extension: bool = False,
) -> torch.Tensor:
    """Integrate a precomputed velocity field.

    Args:
        z0: Tensor shaped [batch, dim].
        velocity: Tensor shaped [batch, steps, dim].
        dt: Euler step size.
        use_extension: Try the optional C++/CUDA extension.

    Returns:
        Tensor shaped [batch, steps, dim] containing states after each step.
    """

    if z0.ndim != 2:
        raise ValueError(f"z0 must have shape [batch, dim], got {tuple(z0.shape)}")
    if velocity.ndim != 3:
        raise ValueError(
            f"velocity must have shape [batch, steps, dim], got {tuple(velocity.shape)}"
        )
    if z0.shape[0] != velocity.shape[0] or z0.shape[1] != velocity.shape[2]:
        raise ValueError(
            "z0 and velocity shape mismatch: "
            f"z0={tuple(z0.shape)}, velocity={tuple(velocity.shape)}"
        )

    if use_extension:
        extension = _load_extension()
        if extension is not None:
            return extension.euler_integrate(z0.contiguous(), velocity.contiguous(), dt)

    return z0.unsqueeze(1) + torch.cumsum(velocity * dt, dim=1)

