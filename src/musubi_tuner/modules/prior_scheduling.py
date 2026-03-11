from __future__ import annotations

import math

import torch


def compute_prior_weight_per_sample(
    timesteps: torch.Tensor,
    *,
    base_weight: float,
    schedule: str,
    pivot_timestep: float,
    global_step: int,
    warmup_steps: int,
) -> torch.Tensor:
    """Compute per-sample prior weights for timestep-adaptive scheduling.

    Semantics (with pivot):
      - For t >= pivot: schedule factor == 1.0 (flat at max weight)
      - For 0 <= t < pivot: schedule factor decays toward 0 at t=0

    Warmup:
      - If warmup_steps > 0, weight is scaled by min(1, global_step / warmup_steps).

    Args:
        timesteps: Tensor of diffusion timesteps (typically shape (B,)).
        base_weight: Base prior weight scalar (e.g., args.prior_preservation_weight).
        schedule: "constant", "linear", or "cosine".
        pivot_timestep: Pivot timestep > 0 (e.g., 300). Timesteps above pivot clamp to pivot.
        global_step: Optimizer step index (not microstep).
        warmup_steps: Number of warmup steps (0 to disable).

    Returns:
        Float32 tensor of shape timesteps.shape, on the same device as timesteps.
    """
    if float(base_weight) <= 0.0:
        return torch.zeros_like(timesteps, dtype=torch.float32)

    schedule = str(schedule).lower()
    if schedule not in ("constant", "linear", "cosine"):
        raise ValueError(f"Unsupported schedule: {schedule}. Expected constant|linear|cosine.")

    pivot = float(pivot_timestep)
    if pivot <= 0.0:
        raise ValueError("pivot_timestep must be > 0")

    warmup_factor = 1.0
    warmup_steps = int(warmup_steps)
    if warmup_steps > 0:
        warmup_factor = min(1.0, float(global_step) / float(warmup_steps))

    t_norm = timesteps.to(dtype=torch.float32).clamp(0.0, pivot) / pivot  # 0..1, clamped above pivot
    if schedule == "constant":
        schedule_factor = torch.ones_like(t_norm)
    elif schedule == "linear":
        schedule_factor = t_norm
    else:  # cosine
        schedule_factor = 0.5 - 0.5 * torch.cos(math.pi * t_norm)

    return schedule_factor * (float(base_weight) * warmup_factor)
