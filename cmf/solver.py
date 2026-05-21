from __future__ import annotations

import math

import torch
from typing import Callable

def euler_step(
    z: torch.Tensor,
    context: torch.Tensor,
    tau: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    dt: float,
) -> torch.Tensor:
    """Perform a single Euler step with Context-Guided Manifold Projection (CGMP)."""
    z_next = z + dt * field_fn(z, context, tau)
    if z_next.size(-1) > 1:
        z_mean = z_next.mean(dim=-1, keepdim=True)
        z_std = z_next.std(dim=-1, keepdim=True, unbiased=False)
        c_mean = context.mean(dim=-1, keepdim=True)
        c_std = context.std(dim=-1, keepdim=True, unbiased=False)
        mask = (c_std > 1e-5) & (z_std > 1e-6)
        projected = ((z_next - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
        z_next = torch.where(mask, projected, z_next)
    return z_next

def rk4_step(
    z: torch.Tensor,
    context: torch.Tensor,
    tau: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    dt: float,
) -> torch.Tensor:
    """Perform a single RK4 step with Context-Guided Manifold Projection (CGMP)."""
    k1 = field_fn(z, context, tau)
    k2 = field_fn(z + 0.5 * dt * k1, context, tau + 0.5 * dt)
    k3 = field_fn(z + 0.5 * dt * k2, context, tau + 0.5 * dt)
    k4 = field_fn(z + dt * k3, context, tau + dt)
    z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if z_next.size(-1) > 1:
        z_mean = z_next.mean(dim=-1, keepdim=True)
        z_std = z_next.std(dim=-1, keepdim=True, unbiased=False)
        c_mean = context.mean(dim=-1, keepdim=True)
        c_std = context.std(dim=-1, keepdim=True, unbiased=False)
        mask = (c_std > 1e-5) & (z_std > 1e-6)
        projected = ((z_next - z_mean) / torch.clamp(z_std, min=1e-6)) * c_std + c_mean
        z_next = torch.where(mask, projected, z_next)
    return z_next

def integrate_symplectic_leapfrog(
    z0: torch.Tensor,
    context: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int,
    dt: float,
) -> torch.Tensor:
    """Integrate for a fixed number of steps using a Symplectic Leapfrog (Verlet) scheme."""
    z = z0.clone()
    batch_size = z.size(0)
    d_half = z.size(-1) // 2
    
    q = z[..., :d_half]
    p = z[..., d_half:]
    
    for step_idx in range(steps):
        tau = torch.full(
            (batch_size,),
            step_idx * dt,
            dtype=z.dtype,
            device=z.device,
        )
        
        # 1. Half-step momentum update
        z_q = torch.cat([q, torch.zeros_like(p)], dim=-1)
        force = field_fn(z_q, context, tau)[..., d_half:]
        p = p + 0.5 * dt * force
        
        # 2. Full-step position update
        q = q + dt * p
        
        # 3. Full-step momentum update
        tau_next = torch.full(
            (batch_size,),
            (step_idx + 1) * dt,
            dtype=z.dtype,
            device=z.device,
        )
        z_q_next = torch.cat([q, torch.zeros_like(p)], dim=-1)
        force_next = field_fn(z_q_next, context, tau_next)[..., d_half:]
        p = p + 0.5 * dt * force_next
        
    return torch.cat([q, p], dim=-1)

def integrate_fixed(
    z0: torch.Tensor,
    context: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int,
    dt: float,
    method: str = "euler"
) -> torch.Tensor:
    """Integrate for a fixed number of steps."""
    if method == "symplectic":
        return integrate_symplectic_leapfrog(z0, context, field_fn, steps, dt)
    z = z0
    batch_size = z.size(0)
    for step_idx in range(steps):
        tau = torch.full(
            (batch_size,),
            step_idx * dt,
            dtype=z.dtype,
            device=z.device,
        )
        if method == "rk4":
            z = rk4_step(z, context, tau, field_fn, dt)
        else:
            z = euler_step(z, context, tau, field_fn, dt)
    return z

def integrate_adaptive(
    z0: torch.Tensor,
    context: torch.Tensor,
    field_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    min_steps: int,
    max_steps: int,
    curvature_threshold: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Integrate one latent-time interval with a curvature-selected step count.

    The controller first probes the field to decide how many Euler substeps are
    worth spending, then integrates with ``dt = 1 / steps``. This keeps every
    token transition on the same latent-time horizon while still allocating
    extra work to curved fields.
    """
    if min_steps <= 0:
        raise ValueError("min_steps must be positive")
    if max_steps < min_steps:
        raise ValueError("max_steps must be greater than or equal to min_steps")

    batch_size = z0.size(0)
    eps = torch.finfo(z0.dtype).eps if torch.is_floating_point(z0) else 1e-7

    with torch.no_grad():
        tau0 = torch.zeros((batch_size,), dtype=z0.dtype, device=z0.device)
        taum = torch.full((batch_size,), 0.5, dtype=z0.dtype, device=z0.device)
        v0 = field_fn(z0, context, tau0)
        probe_z = z0 + 0.5 * v0
        vm = field_fn(probe_z, context, taum)
        speed = torch.linalg.norm(v0, dim=-1).mean() + torch.linalg.norm(vm, dim=-1).mean()
        curvature = torch.linalg.norm(vm - v0, dim=-1).mean() / torch.clamp(speed, min=eps)
        if curvature_threshold <= 0:
            selected_steps = max_steps
        else:
            step_pressure = float((curvature / curvature_threshold).detach().cpu())
            selected_steps = min_steps + max(0, math.ceil(step_pressure) - 1)
            selected_steps = max(min_steps, min(max_steps, selected_steps))

    dt = 1.0 / float(selected_steps)
    z = integrate_fixed(z0, context, field_fn, selected_steps, dt)
    return z, selected_steps
