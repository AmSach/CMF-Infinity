"""
CMF Solver — clean numerical integrators.

What was removed from v1:
  • Context-Guided Manifold Projection (CGMP) in every Euler step.
    This normalises z to context statistics at every step which is:
    (a) not a valid ODE integration method
    (b) destroys gradient signal — every step overwrites z scale
    (c) makes the solver non-ablatable (projection is always active)

  • sin(z * 1000.0) * 1e-6 jitter — BF16 mantissa cannot represent 1e-6
    differences reliably; creates periodic artefacts in attention entropy;
    non-reproducible across hardware.

  • Hamiltonian Symplectic Curl: z += 0.10 * cat(delta_p, -delta_q)
    This is a non-conservative perturbation that grows with step count,
    not a real symplectic integrator (which requires paired (q,p) dynamics).
    It adds O(steps) accumulated drift with no physical justification.

What remains:
  • euler_step: z_{t+1} = z_t + dt * v(z_t, c, τ)  [clean, standard]
  • rk4_step:   classical 4th-order Runge-Kutta     [ablation comparison]
  • euler_maruyama: Euler + Wiener noise during training [proper SDE]
  • integrate_fixed: run N steps with either method
  • integrate_adaptive: curvature-based step selection (kept from v1, logic is sound)
"""

from __future__ import annotations
import math
from typing import Callable, Tuple

import torch
from torch import Tensor

FieldFn = Callable[[Tensor, Tensor, Tensor], Tensor]  # (z, context, tau) -> velocity


# ─────────────────────────────────────────────────────────────────────────────
# Single-step integrators
# ─────────────────────────────────────────────────────────────────────────────

def euler_step(
    z: Tensor,
    context: Tensor,
    tau: Tensor,
    field_fn: FieldFn,
    dt: float,
    noise_scale: float = 0.0,
) -> Tensor:
    v = field_fn(z, context, tau)
    z_next = z + dt * v
    if noise_scale > 0.0:
        z_next = z_next + noise_scale * math.sqrt(dt) * torch.randn_like(z_next)
    return z_next


def rk4_step(
    z: Tensor,
    context: Tensor,
    tau: Tensor,
    field_fn: FieldFn,
    dt: float,
) -> Tensor:
    k1 = field_fn(z,                  context, tau)
    k2 = field_fn(z + 0.5 * dt * k1, context, tau + 0.5 * dt)
    k3 = field_fn(z + 0.5 * dt * k2, context, tau + 0.5 * dt)
    k4 = field_fn(z + dt * k3,        context, tau + dt)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-step integration
# ─────────────────────────────────────────────────────────────────────────────

def integrate_fixed(
    z0: Tensor,
    context: Tensor,
    field_fn: FieldFn,
    steps: int,
    method: str = "euler",
    noise_scale: float = 0.0,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Integrate from τ=0 to τ=1 in `steps` equal substeps."""
    z = z0
    dt = 1.0 / steps
    traj = [z] if return_trajectory else None

    for i in range(steps):
        tau = torch.full(z.shape[:-1], i * dt, dtype=z.dtype, device=z.device)
        if method == "rk4":
            z = rk4_step(z, context, tau, field_fn, dt)
        else:
            z = euler_step(z, context, tau, field_fn, dt, noise_scale=noise_scale)
        if traj is not None:
            traj.append(z)

    if return_trajectory:
        return z, traj
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive integration (curvature-based step selection — logic from v1 is fine)
# ─────────────────────────────────────────────────────────────────────────────

def integrate_adaptive(
    z0: Tensor,
    context: Tensor,
    field_fn: FieldFn,
    min_steps: int = 2,
    max_steps: int = 16,
    curvature_threshold: float = 0.05,
    noise_scale: float = 0.0,
) -> Tuple[Tensor, int]:
    """
    Probe curvature of the field and select step count accordingly.
    Returns (z_final, steps_used).

    This implements the adaptive compute test (checklist Phase 3.2):
    easy inputs should converge in min_steps; hard inputs use more.
    """
    if min_steps <= 0:
        raise ValueError("min_steps must be positive")
    if max_steps < min_steps:
        raise ValueError("max_steps >= min_steps required")

    eps = torch.finfo(z0.dtype).eps if torch.is_floating_point(z0) else 1e-7
    batch = z0.shape[0] if z0.ndim >= 2 else 1

    with torch.no_grad():
        tau0 = torch.zeros(batch, dtype=z0.dtype, device=z0.device)
        taum = torch.full((batch,), 0.5, dtype=z0.dtype, device=z0.device)

        # Reshape for field_fn if z0 is [B, T, D]
        _z = z0.reshape(batch, -1) if z0.ndim == 3 else z0
        _c = context.reshape(batch, -1) if context.ndim == 3 else context

        v0 = field_fn(_z, _c, tau0)
        vm = field_fn(_z + 0.5 * v0, _c, taum)

        speed = (torch.linalg.norm(v0, dim=-1) + torch.linalg.norm(vm, dim=-1)).mean()
        curvature = torch.linalg.norm(vm - v0, dim=-1).mean() / torch.clamp(speed, min=eps)

        pressure = float((curvature / max(curvature_threshold, eps)).detach().cpu())
        steps = max(min_steps, min(max_steps, min_steps + max(0, math.ceil(pressure) - 1)))

    z = integrate_fixed(z0, context, field_fn, steps, method="euler", noise_scale=noise_scale)
    return z, steps
