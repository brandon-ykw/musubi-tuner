from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class LoRAEmaTeacher:
    """EMA tracker + in-place swap context manager for adapter (LoRA) weights.

    Design constraints:
      - Graph-safe for torch.compile: never replace Parameter objects, only copy_ values.
      - DDP-safe: all ranks perform identical local copies; no syncing required.
      - Low overhead: EMA buffers are float32 and kept on the same device as the adapter params.
      - VRAM aware: swap backups are allocated lazily inside apply_to().
    """

    decay: float
    _ema_fp32: dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.decay) < 1.0):
            raise ValueError("LoRAEmaTeacher.decay must be in range (0, 1)")

    def init_from(self, network: torch.nn.Module) -> None:
        """Initialize EMA buffers from current network parameters."""
        self._ema_fp32.clear()

        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            # Keep EMA in float32 for stability; keep it on the same device as params for fast swapping.
            self._ema_fp32[name] = param.detach().to(dtype=torch.float32).clone()

    def update(self, network: torch.nn.Module) -> None:
        """Update EMA buffers from current network parameters (in-place)."""
        if len(self._ema_fp32) == 0:
            raise RuntimeError("LoRAEmaTeacher.update() called before init_from().")

        decay = float(self.decay)
        one_minus = 1.0 - decay

        with torch.no_grad():
            for name, param in network.named_parameters():
                if not param.requires_grad:
                    continue
                ema = self._ema_fp32.get(name, None)
                if ema is None:
                    raise KeyError(f"EMA buffer missing for parameter: {name}")
                # Keep EMA on the same device as the live param if devices change (rare, but safe).
                if ema.device != param.device:
                    ema = ema.to(device=param.device)
                    self._ema_fp32[name] = ema

                # ema = decay * ema + (1 - decay) * param
                ema.mul_(decay).add_(param.detach().to(dtype=torch.float32), alpha=one_minus)

    @contextmanager
    def apply_to(self, network: torch.nn.Module):
        """Temporarily swap network parameters to EMA weights (in-place), then restore."""
        if len(self._ema_fp32) == 0:
            raise RuntimeError("LoRAEmaTeacher.apply_to() called before init_from().")

        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in network.named_parameters():
                if not param.requires_grad:
                    continue
                ema = self._ema_fp32.get(name, None)
                if ema is None:
                    raise KeyError(f"EMA buffer missing for parameter: {name}")

                if ema.device != param.device:
                    ema = ema.to(device=param.device)
                    self._ema_fp32[name] = ema

                # Backup is allocated lazily to avoid persistent VRAM overhead when the teacher is inactive.
                backup[name] = param.detach().clone()
                param.copy_(ema.to(dtype=param.dtype))

        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in network.named_parameters():
                    if not param.requires_grad:
                        continue
                    b = backup.get(name, None)
                    if b is None:
                        # If a failure occurred mid-swap, some params may not have been backed up/swapped.
                        # Best-effort restore without masking the original exception.
                        continue
                    param.copy_(b)
