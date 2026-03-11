from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_unreduced_target_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "mse",
    loss_delta: float = 1.0,
) -> torch.Tensor:
    """Compute elementwise (unreduced) target loss.

    This is intended to run *before* any mask-weighted reduction, so it must return a tensor
    with the same shape as model_pred/target.

    Args:
        model_pred: Model output tensor.
        target: Target tensor (same shape).
        loss_type: "mse" or "huber".
        loss_delta: Huber delta (only used when loss_type == "huber").

    Returns:
        Unreduced loss tensor (same shape).
    """
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return F.mse_loss(model_pred, target, reduction="none")

    if loss_type != "huber":
        raise ValueError(f"Unsupported loss_type: {loss_type}. Expected 'mse' or 'huber'.")

    delta = float(loss_delta)
    if delta <= 0.0:
        raise ValueError("--loss_delta must be > 0 when --loss_type huber is used.")

    diff = model_pred - target
    abs_diff = diff.abs()
    delta_t = diff.new_tensor(delta)

    # Standard Huber loss:
    #   0.5 * diff^2                         if |diff| <= delta
    #   delta * (|diff| - 0.5 * delta)       otherwise
    quadratic = 0.5 * diff * diff
    linear = delta_t * (abs_diff - 0.5 * delta_t)
    return torch.where(abs_diff <= delta_t, quadratic, linear)
