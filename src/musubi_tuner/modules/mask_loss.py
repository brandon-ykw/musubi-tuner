from __future__ import annotations

import argparse
import logging
from typing import Any, Literal

import torch
import torch.nn.functional as F


MaskLossLayout = Literal["video", "layered"]

# Module-level logger for validation warnings
_logger = logging.getLogger(__name__)


def _compute_default_gaussian_sigma(kernel_size: int) -> float:
    """Compute the default Gaussian sigma used by torchvision's gaussian_blur when sigma is omitted.

    Matches:
      sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    """
    radius = (kernel_size - 1) * 0.5
    return 0.3 * (radius - 1.0) + 0.8


def _gaussian_blur_compact_mask(mask: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    """Gaussian-blur a compact mask in-place over spatial dims only.

    Expects a compact 5D mask with channel dim == 1:
      - video:   (B, 1, F, H, W)
      - layered: (B, L, 1, H, W)

    Returns a tensor with the same shape and dtype as the input.
    """
    if kernel_size <= 1:
        return mask
    if kernel_size < 0:
        raise ValueError("--mask_blur_kernel_size must be >= 0")
    if kernel_size % 2 == 0:
        raise ValueError("--mask_blur_kernel_size must be odd (or 0 to disable)")
    if mask.ndim != 5:
        raise ValueError(f"Expected compact mask to be 5D, got {mask.ndim}D: {tuple(mask.shape)}")
    if mask.shape[-2] <= 0 or mask.shape[-1] <= 0:
        raise ValueError(f"Invalid mask spatial size: H={mask.shape[-2]} W={mask.shape[-1]}")

    pad = kernel_size // 2
    device = mask.device
    dtype = mask.dtype

    # Flatten all leading dimensions into batch. Channel dim is always 1, so this works
    # for both (B,1,F,H,W) and (B,L,1,H,W) without permuting.
    h, w = mask.shape[-2], mask.shape[-1]
    mask_flat = mask.reshape(-1, 1, h, w)

    # Compute Gaussian kernel in float32 for stability / CPU support.
    sigma = _compute_default_gaussian_sigma(kernel_size)
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).to(device=device, dtype=torch.float32)
    weight = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    mask_float = mask_flat.to(dtype=torch.float32)
    mask_padded = F.pad(mask_float, (pad, pad, pad, pad), mode="replicate")
    blurred = F.conv2d(mask_padded, weight, bias=None, stride=1, padding=0)
    blurred = blurred.clamp(0.0, 1.0).to(dtype=dtype)

    return blurred.reshape(mask.shape)


def add_mask_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use_mask_loss",
        action="store_true",
        help="Enable mask-weighted loss training. Requires alpha_mask and/or mask_directory in dataset config. "
        "White regions (255) get full training weight, black regions (0) are ignored. "
        "/ マスク重み付き損失学習を有効にする。データセット設定でalpha_maskおよび/またはmask_directoryが必要。"
        "白い領域(255)は完全な学習重みを取得し、黒い領域(0)は無視される。",
    )
    parser.add_argument(
        "--mask_min_weight",
        type=float,
        default=0.0,
        help="Minimum weight for masked-out regions (default: 0.0). Set to 0.1-0.2 to give some training signal to background regions. "
        "NOTE: When using --prior_preservation_weight, recommend 0.0 for best results. "
        "/ マスク外領域の最小重み（デフォルト：0.0）。0.1-0.2に設定すると背景領域にもある程度の学習シグナルを与える。",
    )
    parser.add_argument(
        "--mask_gamma",
        type=float,
        default=1.0,
        help="Gamma correction for mask weights (default: 1.0). Values < 1.0 soften the mask (more midtones, gradual falloff). "
        "Values > 1.0 sharpen the mask (more binary, stronger face focus). Try 0.5-0.7 for softer or 1.5-2.0 for sharper. "
        "/ マスク重みのガンマ補正（デフォルト：1.0）。1.0未満はマスクを柔らかくし、1.0超はマスクを鋭くして顔への集中を強める。",
    )
    parser.add_argument(
        "--mask_blur_kernel_size",
        "--mask_blur_radius",
        dest="mask_blur_kernel_size",
        type=int,
        default=0,
        help="Optional: Apply a Gaussian blur to mask weights before gamma/min_weight (default: 0 = disabled). "
        "This feathers mask boundaries and can reduce halo/edge artifacts. Value is kernel size in latent-space pixels "
        "(must be odd if > 0). Recommended: 3 or 5. "
        "Note: --mask_blur_radius is an alias for this option.",
    )
    parser.add_argument(
        "--mask_area_scale_beta",
        type=float,
        default=0.0,
        help="Optional: Scale masked target loss by (raw_mask_mean ** beta) to reduce gradient spikes on tiny masks. "
        "beta=0.0 keeps strict weighted-mean normalization (current behavior). "
        "beta=1.0 approximates global-mean-like scaling when mask_gamma=1 and mask_min_weight=0. "
        "Try 0.5 as a middle-ground. Note: raw_mask_mean is computed before gamma/min_weight. Default: 0.0.",
    )

    # Prior preservation arguments
    parser.add_argument(
        "--prior_preservation_weight",
        type=float,
        default=0.0,
        help="Weight for prior preservation loss in unmasked regions (default: 0.0 = disabled). "
        "When enabled, unmasked regions are trained to match base model predictions, preventing "
        "phantom limbs and background hallucinations. Recommended: 0.5-1.0. "
        "NOTE: Recommend --mask_min_weight 0.0 when using this. Requires LoRA training. "
        "/ マスク外領域での事前保存損失の重み（デフォルト：0.0=無効）。",
    )
    parser.add_argument(
        "--prior_decay_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
        help="Optional: Timestep-adaptive scaling schedule for prior preservation weight (default: constant). "
        "When enabled, w_prior stays at maximum at high noise and decays toward 0 at low noise "
        "(see --prior_decay_timestep_start). Requires --normalize_per_sample for correctness.",
    )
    parser.add_argument(
        "--prior_decay_timestep_start",
        type=float,
        default=300.0,
        help="Pivot timestep for prior decay schedule (default: 300). For timesteps >= pivot, prior weight stays at max. "
        "For timesteps < pivot, the schedule decays toward 0 at t=0.",
    )
    parser.add_argument(
        "--prior_decay_warmup_ratio",
        type=float,
        default=0.0,
        help="Optional: Warm up prior preservation weight from 0 to full value over this fraction of total train steps "
        "(default: 0.0). This applies before the timestep schedule. Requires --normalize_per_sample when non-zero.",
    )
    parser.add_argument(
        "--prior_mask_threshold",
        type=float,
        default=None,
        help="Optional: Apply prior preservation only where RAW mask < threshold (before gamma/min_weight). "
        "Default: None (continuous mode - prior preservation scales with inverse mask). "
        "Set to 0.05-0.1 to preserve only true background while body/hair still train to target. "
        "/ 事前保存を適用するマスクしきい値（オプション）。",
    )
    parser.add_argument(
        "--prior_preservation_timestep_threshold",
        type=float,
        default=None,
        help="Optional: Skip the teacher forward pass unless diffusion timesteps are above this value (0-1000). "
        "This can significantly reduce compute/VRAM overhead by running prior preservation only at high-noise "
        "structural timesteps (when hallucinations tend to lock in). Example: --prior_preservation_timestep_threshold 300. "
        "Default: None (always compute teacher when prior preservation is enabled).",
    )
    parser.add_argument(
        "--prior_teacher_eval",
        action="store_true",
        help="Optional: Run the teacher forward pass with the base transformer in eval() mode (disables dropout / uses eval behavior). "
        "This makes teacher targets deterministic if the architecture ever introduces stochastic layers. "
        "Default: disabled (teacher runs in train() mode for OneTrainer compatibility). "
        "Note: toggling train/eval may reduce torch.compile effectiveness.",
    )
    parser.add_argument(
        "--prior_teacher_mode",
        type=str,
        default="base",
        choices=["base", "ema"],
        help="Teacher mode for prior preservation (default: base). "
        "'base' disables adapters to use the pristine base model as teacher. "
        "'ema' uses an EMA-smoothed copy of adapter weights as teacher (adapters remain enabled), which can reduce stylistic clash. "
        "Note: EMA teacher is initialized after warmup and a small minimum number of steps to avoid step-0 adapter noise.",
    )
    parser.add_argument(
        "--prior_teacher_ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for --prior_teacher_mode=ema (default: 0.999). "
        "EMA is applied to adapter (LoRA) parameters only, not the base transformer.",
    )
    parser.add_argument(
        "--normalize_per_sample",
        action="store_true",
        help="Normalize loss per-sample before averaging across batch (default: global normalization). "
        "Recommended when prior preservation is enabled for more predictable behavior. "
        "/ サンプルごとに損失を正規化してからバッチ全体で平均する。",
    )


def validate_mask_loss_args(args: argparse.Namespace) -> None:
    use_mask_loss = bool(getattr(args, "use_mask_loss", False))

    # Back-compat: old configs may still include this key via read_config_from_file()'s Namespace merge.
    if hasattr(args, "mask_loss_scale") and args.mask_loss_scale is not None:
        try:
            mask_loss_scale = float(args.mask_loss_scale)
        except Exception as e:  # noqa: BLE001
            raise ValueError("--mask_loss_scale must be a number") from e

        if mask_loss_scale != 1.0:
            raise ValueError(
                "--mask_loss_scale has been removed (it had no effect with weighted-mean normalization). "
                "Use --mask_gamma and/or --mask_min_weight instead."
            )
        # mask_loss_scale == 1.0 is treated as a no-op for back-compat with old configs.

    # Prior preservation validation
    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    if prior_preservation_weight < 0:
        raise ValueError("--prior_preservation_weight must be >= 0")

    if prior_preservation_weight > 0 and not use_mask_loss:
        raise ValueError(
            "--prior_preservation_weight > 0 requires --use_mask_loss. "
            "Prior preservation uses mask-based region splitting and has no effect without masks."
        )

    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        if prior_mask_threshold <= 0 or prior_mask_threshold >= 1:
            raise ValueError("--prior_mask_threshold must be in range (0, 1)")
        if prior_preservation_weight <= 0:
            _logger.warning(f"--prior_mask_threshold={prior_mask_threshold} has no effect without --prior_preservation_weight > 0")

    prior_timestep_threshold = getattr(args, "prior_preservation_timestep_threshold", None)
    if prior_timestep_threshold is not None:
        try:
            prior_timestep_threshold = float(prior_timestep_threshold)
        except Exception as e:  # noqa: BLE001
            raise ValueError("--prior_preservation_timestep_threshold must be a number") from e

        if prior_timestep_threshold < 0 or prior_timestep_threshold > 1000:
            raise ValueError("--prior_preservation_timestep_threshold must be in range [0, 1000]")

        if prior_preservation_weight <= 0:
            _logger.warning(
                f"--prior_preservation_timestep_threshold={prior_timestep_threshold} has no effect without --prior_preservation_weight > 0"
            )
        else:
            # Optional guidance: large flow shifts can heavily concentrate the timestep distribution at high noise,
            # which makes low gating thresholds (e.g. 300) skip very few teacher passes.
            timestep_sampling = getattr(args, "timestep_sampling", None)
            discrete_flow_shift = getattr(args, "discrete_flow_shift", None)
            if isinstance(timestep_sampling, str) and timestep_sampling.endswith("shift") and discrete_flow_shift is not None:
                try:
                    discrete_flow_shift = float(discrete_flow_shift)
                except Exception:  # noqa: BLE001
                    discrete_flow_shift = None

                if discrete_flow_shift is not None and discrete_flow_shift > 5.0:
                    _logger.warning(
                        f"--prior_preservation_timestep_threshold={prior_timestep_threshold} with "
                        f"--timestep_sampling={timestep_sampling} and --discrete_flow_shift={discrete_flow_shift}: "
                        "High shift values can concentrate most sampled timesteps at high noise, so gating may skip "
                        "very few teacher passes. Use --show_timesteps console to calibrate a threshold."
                    )

    prior_teacher_eval = bool(getattr(args, "prior_teacher_eval", False))
    if prior_teacher_eval and prior_preservation_weight <= 0:
        _logger.warning("--prior_teacher_eval has no effect without --prior_preservation_weight > 0")

    prior_teacher_mode = str(getattr(args, "prior_teacher_mode", "base"))
    if prior_teacher_mode not in ("base", "ema"):
        raise ValueError("--prior_teacher_mode must be one of: base, ema")

    prior_teacher_ema_decay = float(getattr(args, "prior_teacher_ema_decay", 0.999))
    if prior_teacher_ema_decay <= 0.0 or prior_teacher_ema_decay >= 1.0:
        raise ValueError("--prior_teacher_ema_decay must be in range (0, 1)")
    if prior_teacher_mode == "ema" and prior_preservation_weight <= 0:
        _logger.warning("--prior_teacher_mode=ema has no effect without --prior_preservation_weight > 0")

    prior_decay_schedule = str(getattr(args, "prior_decay_schedule", "constant"))
    if prior_decay_schedule not in ("constant", "linear", "cosine"):
        raise ValueError("--prior_decay_schedule must be one of: constant, linear, cosine")

    prior_decay_timestep_start = float(getattr(args, "prior_decay_timestep_start", 300.0))
    if prior_decay_timestep_start <= 0.0 or prior_decay_timestep_start > 1000.0:
        raise ValueError("--prior_decay_timestep_start must be in range (0, 1000]")

    prior_decay_warmup_ratio = float(getattr(args, "prior_decay_warmup_ratio", 0.0))
    if prior_decay_warmup_ratio < 0.0 or prior_decay_warmup_ratio > 1.0:
        raise ValueError("--prior_decay_warmup_ratio must be in range [0, 1]")

    if not use_mask_loss:
        return

    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")

    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")

    mask_blur_kernel_size = int(getattr(args, "mask_blur_kernel_size", 0) or 0)
    if mask_blur_kernel_size < 0:
        raise ValueError("--mask_blur_kernel_size must be >= 0")
    if mask_blur_kernel_size > 0 and mask_blur_kernel_size % 2 == 0:
        raise ValueError("--mask_blur_kernel_size must be odd (or 0 to disable)")

    mask_area_scale_beta = float(getattr(args, "mask_area_scale_beta", 0.0))
    if mask_area_scale_beta < 0:
        raise ValueError("--mask_area_scale_beta must be >= 0")

    if prior_preservation_weight > 0 and mask_min_weight > 0:
        _logger.warning(
            f"--prior_preservation_weight={prior_preservation_weight} with --mask_min_weight={mask_min_weight}: "
            "Non-zero mask_min_weight reduces prior preservation effect. Recommend --mask_min_weight 0.0"
        )

    # TP-11: Warn when prior preservation is active but normalize_per_sample is off.
    normalize_per_sample = getattr(args, "normalize_per_sample", False)
    if prior_preservation_weight > 0 and not normalize_per_sample:
        _logger.warning(
            "--prior_preservation_weight > 0 without --normalize_per_sample: "
            "global normalization may cause per-sample loss imbalance when mask coverage varies across the batch. "
            "Consider adding --normalize_per_sample for more predictable behavior."
        )

    schedule_enabled = (prior_decay_schedule != "constant") or (prior_decay_warmup_ratio > 0.0)
    if schedule_enabled and not normalize_per_sample:
        raise ValueError(
            "Timestep-adaptive prior scheduling requires --normalize_per_sample for correctness. "
            f"Got --prior_decay_schedule={prior_decay_schedule} and --prior_decay_warmup_ratio={prior_decay_warmup_ratio} "
            "without --normalize_per_sample."
        )
    if schedule_enabled and prior_preservation_weight <= 0:
        _logger.warning(
            f"--prior_decay_schedule={prior_decay_schedule} / --prior_decay_warmup_ratio={prior_decay_warmup_ratio} "
            "has no effect without --prior_preservation_weight > 0"
        )


def log_mask_loss_banner(
    logger: Any,
    args: argparse.Namespace,
    cache_hint: str | None = None,
    *,
    cache_mask_transform_pairs: set[tuple[float, float]] | None = None,
    cache_mask_metadata_coverage: tuple[int, int] | None = None,
) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    prior_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    prior_threshold = getattr(args, "prior_mask_threshold", None)
    prior_timestep_threshold = getattr(args, "prior_preservation_timestep_threshold", None)
    prior_decay_schedule = str(getattr(args, "prior_decay_schedule", "constant"))
    prior_decay_timestep_start = float(getattr(args, "prior_decay_timestep_start", 300.0))
    prior_decay_warmup_ratio = float(getattr(args, "prior_decay_warmup_ratio", 0.0))
    prior_teacher_mode = str(getattr(args, "prior_teacher_mode", "base"))
    prior_teacher_ema_decay = float(getattr(args, "prior_teacher_ema_decay", 0.999))
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    mask_blur_kernel_size = int(getattr(args, "mask_blur_kernel_size", 0) or 0)
    mask_area_scale_beta = float(getattr(args, "mask_area_scale_beta", 0.0))
    normalize_per_sample = getattr(args, "normalize_per_sample", False)
    mask_gamma = float(getattr(args, "mask_gamma", 1.0))

    logger.info("=" * 60)
    if prior_weight > 0:
        logger.info("MASKED PRIOR PRESERVATION TRAINING ENABLED")
    else:
        logger.info("MASK-WEIGHTED LOSS TRAINING ENABLED")
    logger.info("=" * 60)
    logger.info(f"  mask_min_weight: {mask_min_weight}")
    logger.info(f"  mask_blur_kernel_size: {mask_blur_kernel_size}")
    logger.info(f"  mask_area_scale_beta: {mask_area_scale_beta}")
    logger.info(f"  mask_gamma: {mask_gamma}")
    logger.info(
        f"  Applying training-time mask gamma/min_weight: gamma={mask_gamma}, min_weight={mask_min_weight}. "
        "(Note: If you already baked these into your latents during caching, keep these at 1.0 and 0.0 to avoid double-application.)"
    )

    # Cache-time mask preprocessing transparency / safety.
    #
    # Cache-time transforms are applied BEFORE latent downsampling, so they change the numerical mask values
    # that training sees as "raw" when using threshold-mode prior preservation.
    if cache_mask_transform_pairs is not None:
        if cache_mask_metadata_coverage is not None:
            with_meta, checked = cache_mask_metadata_coverage
            logger.info(f"  cache_mask_metadata_coverage: {with_meta}/{checked} sampled cache files")

        if len(cache_mask_transform_pairs) == 0:
            logger.warning(
                "  No cache mask metadata found in sampled caches. "
                "This usually means you are using older caches created before cache metadata tracking existed."
            )
        elif len(cache_mask_transform_pairs) == 1:
            baked_gamma, baked_min_weight = next(iter(cache_mask_transform_pairs))
            logger.info(f"  cache_mask_gamma (baked): {baked_gamma}")
            logger.info(f"  cache_mask_min_weight (baked): {baked_min_weight}")

            # Double application warning.
            if baked_gamma != 1.0 and mask_gamma != 1.0:
                logger.warning(
                    f"  WARNING: Cache has baked gamma={baked_gamma}, but training also applies --mask_gamma={mask_gamma}. "
                    "This will apply gamma twice (once in cache-time pixel space, once at training time in latent space). "
                    "If you baked gamma into the cache, keep --mask_gamma=1.0."
                )
            if baked_min_weight != 0.0 and mask_min_weight != 0.0:
                logger.warning(
                    f"  WARNING: Cache has baked min_weight={baked_min_weight}, but training also applies --mask_min_weight={mask_min_weight}. "
                    "This will apply a floor twice. If you baked min_weight into the cache, keep --mask_min_weight=0.0."
                )

            # Threshold-mode safety: floor trap.
            if prior_weight > 0 and prior_threshold is not None and baked_min_weight >= float(prior_threshold):
                logger.warning(
                    "  CRITICAL: Threshold-mode prior preservation may be disabled by baked min_weight. "
                    f"You are using --prior_mask_threshold={prior_threshold}, but cached masks have baked min_weight={baked_min_weight}. "
                    "Because background becomes >= min_weight, (mask < threshold) will be empty or near-empty. "
                    "Fix: set --cache_mask_min_weight=0.0 when using threshold-mode prior, or increase --prior_mask_threshold above baked min_weight."
                )

            if prior_weight > 0 and prior_threshold is not None and (baked_gamma != 1.0 or baked_min_weight != 0.0):
                logger.info(
                    "  NOTE: --prior_mask_threshold is evaluated on the cached mask values. "
                    "If you baked cache-time gamma/min_weight, you may need to adjust the threshold accordingly "
                    "(monotonic transform)."
                )
        else:
            # Mixed caches: usually means stale caches were kept via --skip_existing or cache dirs were reused.
            pairs_str = ", ".join(f"(gamma={g}, min_weight={m})" for g, m in sorted(cache_mask_transform_pairs))
            logger.warning(
                "  WARNING: Mixed cache mask metadata detected across sampled cache files: " + pairs_str + ". "
                "This can cause inconsistent masking behavior across items. "
                "Recommendation: use a fresh cache_directory and recache latents to make these consistent."
            )

            if prior_weight > 0 and prior_threshold is not None:
                max_baked_min_weight = max(m for _, m in cache_mask_transform_pairs)
                if max_baked_min_weight >= float(prior_threshold):
                    logger.warning(
                        "  CRITICAL: Some cached masks have baked min_weight that can disable threshold-mode prior preservation. "
                        f"--prior_mask_threshold={prior_threshold}, max baked min_weight in sample={max_baked_min_weight}. "
                        "Fix: recache with consistent settings and keep baked min_weight < threshold (or avoid threshold mode)."
                    )
    if prior_weight > 0:
        logger.info(f"  prior_preservation_weight: {prior_weight}")
        logger.info(f"  prior_decay_schedule: {prior_decay_schedule}")
        if prior_decay_schedule != "constant" or prior_decay_warmup_ratio > 0.0:
            logger.info(f"  prior_decay_timestep_start: {prior_decay_timestep_start}")
            logger.info(f"  prior_decay_warmup_ratio: {prior_decay_warmup_ratio}")
        if prior_threshold is not None:
            logger.info(f"  prior_mask_threshold: {prior_threshold} (threshold mode)")
        else:
            logger.info("  prior_mask_threshold: None (continuous mode)")
        if prior_timestep_threshold is not None:
            logger.info(f"  prior_preservation_timestep_threshold: {prior_timestep_threshold} (teacher gated)")
        prior_teacher_eval = bool(getattr(args, "prior_teacher_eval", False))
        logger.info(f"  prior_teacher_eval: {prior_teacher_eval}")
        logger.info(f"  prior_teacher_mode: {prior_teacher_mode}")
        if prior_teacher_mode == "ema":
            logger.info(f"  prior_teacher_ema_decay: {prior_teacher_ema_decay}")
        logger.info(f"  normalize_per_sample: {normalize_per_sample}")
        logger.info("-" * 60)
        logger.info("PRIOR PRESERVATION: Unmasked regions will match base model.")
        if prior_timestep_threshold is not None:
            logger.info("                    Teacher pass is gated by timestep threshold (reduced overhead).")
        else:
            logger.info("                    Expect ~1.3-1.7x training time.")
            logger.info("                    Tip: try --prior_preservation_timestep_threshold 300 for a speed/quality tradeoff.")
        if mask_min_weight > 0:
            logger.warning(f"  NOTE: mask_min_weight={mask_min_weight} reduces prior effect.")
            logger.warning("        Recommend --mask_min_weight 0.0 with prior preservation.")
    logger.info("-" * 60)
    logger.info("IMPORTANT: Masks must be baked into latent cache!")
    if cache_hint:
        logger.info(cache_hint)
    logger.info("=" * 60)


def require_mask_weights_if_enabled(batch: dict[str, Any], args: argparse.Namespace, cache_hint: str | None = None) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    if batch.get("mask_weights", None) is not None:
        return

    message = [
        "FATAL: --use_mask_loss is enabled but batch has no mask_weights!",
        "This means masks were NOT baked into your latent cache.",
        "To fix:",
        "  1. Add 'alpha_mask = true' and/or 'mask_directory = \"/path/to/masks\"' in dataset TOML",
        "  2. Use a FRESH cache_directory (masks are stored in cache)",
    ]
    if cache_hint:
        message.append(f"  3. {cache_hint}")
    else:
        message.append("  3. Recache latents with the appropriate cache script")
    message.append("  4. Then re-run training")

    raise ValueError("\n".join(message))


def apply_masked_loss(
    loss: torch.Tensor,
    mask_weights: torch.Tensor | None,
    *,
    args: argparse.Namespace,
    layout: MaskLossLayout = "video",
    drop_base_frame: bool = False,
    accelerator: Any | None = None,
) -> torch.Tensor:
    """Apply mask-weighted loss (no prior preservation). Thin wrapper around apply_masked_loss_with_prior."""
    del accelerator  # reserved for future global reduction support
    return apply_masked_loss_with_prior(
        loss, mask_weights, prior_loss_unreduced=None, args=args, layout=layout, drop_base_frame=drop_base_frame
    )


def _prepare_tensors(
    loss: torch.Tensor,
    mask_weights: torch.Tensor,
    layout: MaskLossLayout,
    drop_base_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare loss and mask tensors for computation.
    Handles 4D/5D tensors and layout-specific transformations.

    Args:
        loss: Loss tensor (B, C, H, W) or (B, C, F, H, W) for video, or (B, L, C, H, W) for layered
        mask_weights: Mask tensor (B, F, H, W) or (B, 1, F, H, W)
        layout: "video" or "layered"
        drop_base_frame: Whether to drop base frame for layered layout

    Returns:
        Tuple of (loss, mask_weights) as 5D tensors. Mask is in compact (unexpanded)
        form: (B,1,F,H,W) for video or (B,L,1,H,W) for layered, broadcast-compatible with loss.
    """
    # Handle 4D vs 5D loss
    if loss.ndim == 4:
        if layout != "video":
            raise ValueError("4D loss is only supported for layout='video'")
        loss = loss.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
    elif loss.ndim != 5:
        raise ValueError(f"Expected loss to be 4D or 5D, got {loss.ndim}D: {tuple(loss.shape)}")

    if drop_base_frame and layout != "layered":
        raise ValueError("drop_base_frame=True is only valid with layout='layered'")

    # Handle mask dimensions
    if mask_weights.ndim == 4:
        mask_weights = mask_weights.unsqueeze(1)  # (B, F, H, W) -> (B, 1, F, H, W)
    elif mask_weights.ndim != 5:
        raise ValueError(f"Unexpected mask_weights shape: {tuple(mask_weights.shape)}")

    # Layout-specific handling
    if layout == "video":
        # loss: (B, C, F, H, W), mask: (B, 1, F, H, W)
        if mask_weights.shape[0] != loss.shape[0] or mask_weights.shape[2:] != loss.shape[2:]:
            raise ValueError(
                "mask_weights shape does not match loss shape for layout='video': "
                f"mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)}"
            )
        # mask stays (B, 1, F, H, W) — broadcast-compatible with (B, C, F, H, W)
    elif layout == "layered":
        # loss: (B, L, C, H, W), mask: (B, 1, F, H, W) where F == (base + layers) or F == L
        if drop_base_frame:
            mask_weights = mask_weights[:, :, 1:, :, :]
        if (
            mask_weights.shape[0] != loss.shape[0]
            or mask_weights.shape[2] != loss.shape[1]
            or mask_weights.shape[3:] != loss.shape[3:]
        ):
            raise ValueError(
                "mask_weights shape does not match loss shape for layout='layered': "
                f"mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)} drop_base_frame={drop_base_frame}"
            )
        mask_weights = mask_weights.permute(0, 2, 1, 3, 4)  # (B, L, 1, H, W)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return loss, mask_weights


def apply_masked_loss_with_prior(
    loss: torch.Tensor,
    mask_weights: torch.Tensor | None,
    *,
    prior_loss_unreduced: torch.Tensor | None = None,
    prior_sample_mask: torch.Tensor | None = None,
    prior_weight_per_sample: torch.Tensor | None = None,
    target_huber_is_linear: torch.Tensor | None = None,
    prior_huber_is_linear: torch.Tensor | None = None,
    stats: dict[str, torch.Tensor] | None = None,
    args: argparse.Namespace,
    layout: MaskLossLayout = "video",
    drop_base_frame: bool = False,
) -> torch.Tensor:
    """
    Apply masked loss with optional prior preservation.

    Uses region-normalized means + explicit weighting:
        L_target = weighted_mean(mse, mask_processed)
        L_prior  = weighted_mean(mse, prior_mask) * w_prior
        loss = L_target + L_prior

    This ensures w_prior acts as a true independent knob.

    Args:
        loss: Unreduced loss tensor (B, C, F, H, W) or (B, C, H, W)
        mask_weights: Mask weights tensor, or None to use uniform weights
        prior_loss_unreduced: Unreduced prior loss tensor (same shape as loss), or None
        prior_sample_mask: Optional per-sample mask (shape: (B,)) that gates prior preservation
            on/off per item in the batch (useful for timestep-gated teacher passes).
        prior_weight_per_sample: Optional per-sample prior weight tensor (shape: (B,)). When provided,
            prior preservation uses per-sample weights (e.g., timestep-adaptive scheduling).
            Requires --normalize_per_sample.
        target_huber_is_linear: Optional boolean tensor (same shape as `loss`) indicating which
            elements are in the linear regime when `--loss_type huber` is used (|diff| > delta).
            Used only for logging/telemetry; does not affect training.
        prior_huber_is_linear: Optional boolean tensor (same shape as `prior_loss_unreduced`) indicating
            which elements are in the linear regime for the prior teacher loss. Used only for telemetry.
        args: Namespace with mask_gamma, mask_min_weight, prior_preservation_weight,
              prior_mask_threshold, normalize_per_sample
        layout: "video" or "layered"
        drop_base_frame: Whether to drop base frame for layered layout

    Returns:
        Scalar loss tensor (float32)
    """
    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    normalize_per_sample = getattr(args, "normalize_per_sample", False)

    # If no mask or mask loss disabled, fall back to simple mean
    if mask_weights is None or not getattr(args, "use_mask_loss", False):
        return loss.float().mean()

    # Guard: prior preservation not yet implemented for layered layout
    if prior_preservation_weight > 0 and layout == "layered":
        raise NotImplementedError(
            "Prior preservation is not yet supported with layout='layered'. "
            "Use layout='video' or disable --prior_preservation_weight."
        )

    # Handle tensor shapes — mask returned in compact form (B,1,F,H,W) or (B,L,1,H,W)
    loss, mask_weights = _prepare_tensors(loss, mask_weights, layout, drop_base_frame)

    if prior_weight_per_sample is not None:
        if not normalize_per_sample:
            raise ValueError("prior_weight_per_sample requires --normalize_per_sample (per-sample reduction).")
        if prior_weight_per_sample.ndim != 1 or prior_weight_per_sample.shape[0] != loss.shape[0]:
            raise ValueError(
                "prior_weight_per_sample must be a 1D tensor with shape (B,), got "
                f"{tuple(prior_weight_per_sample.shape)} for B={loss.shape[0]}"
            )
        if (prior_weight_per_sample < 0).any().item():
            raise ValueError("prior_weight_per_sample must be >= 0 for all samples")
        prior_weight_per_sample = prior_weight_per_sample.to(device=loss.device, dtype=torch.float32)

    # Ensure mask weights match loss device/dtype to prevent mixed-precision collisions.
    # Note: mask_weights may be stored as float16 in cache files to reduce disk I/O.
    mask_weights = mask_weights.to(loss.device, dtype=loss.dtype)

    # Compact mask is broadcast-compatible with loss; compute channel factor for weight sums.
    # Video: mask (B,1,F,H,W), loss (B,C,F,H,W) → C = loss.shape[1]
    # Layered: mask (B,L,1,H,W), loss (B,L,C,H,W) → C = loss.shape[2]
    num_channels = loss.shape[1] if layout == "video" else loss.shape[2]

    # Keep raw mask for thresholding (before gamma/min_weight).
    # All mask processing stays on compact tensor to save VRAM.
    mask_raw_unblurred = mask_weights.clamp(0.0, 1.0)

    # Optional: blur mask boundaries before gamma/min_weight to reduce halo artifacts.
    # This is intentionally applied to the compact mask tensor to keep VRAM minimal.
    mask_blur_kernel_size = int(getattr(args, "mask_blur_kernel_size", 0) or 0)
    if mask_blur_kernel_size < 0:
        raise ValueError("--mask_blur_kernel_size must be >= 0")
    if mask_blur_kernel_size > 0 and mask_blur_kernel_size % 2 == 0:
        raise ValueError("--mask_blur_kernel_size must be odd (or 0 to disable)")
    mask_raw_for_processing = (
        _gaussian_blur_compact_mask(mask_raw_unblurred, kernel_size=mask_blur_kernel_size)
        if mask_blur_kernel_size > 1
        else mask_raw_unblurred
    )

    # Apply gamma and min_weight to get processed mask for target loss
    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))

    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")

    mask_processed = mask_raw_for_processing
    if mask_gamma != 1.0:
        mask_processed = mask_processed**mask_gamma
    if mask_min_weight > 0:
        mask_processed = mask_processed * (1.0 - mask_min_weight) + mask_min_weight

    # Optional: threshold on RAW mask (before gamma/min_weight)
    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        # Binarize: full prior preservation where raw mask < threshold
        prior_mask = (mask_raw_unblurred < prior_mask_threshold).float()
    else:
        # Continuous mode: prior is the complement of processed mask
        prior_mask = 1 - mask_processed

    # Optional: gate prior preservation per-sample (e.g., timestep-based teacher gating).
    # IMPORTANT: This must happen BEFORE overlap prevention so target masks are not modified for gated-off samples.
    if prior_sample_mask is not None:
        if prior_sample_mask.ndim != 1 or prior_sample_mask.shape[0] != loss.shape[0]:
            raise ValueError(
                f"prior_sample_mask must be a 1D tensor with shape (B,), got {tuple(prior_sample_mask.shape)} for B={loss.shape[0]}"
            )
        gate = prior_sample_mask.to(device=loss.device, dtype=loss.dtype).view(loss.shape[0], *([1] * (prior_mask.ndim - 1)))
        prior_mask = prior_mask * gate

    # Prevent target/prior overlap in threshold mode only
    if prior_mask_threshold is not None:
        # Prevent target/prior overlap: zero out target where prior applies
        mask_processed = mask_processed * (1 - prior_mask)

    if stats is not None:
        # Optional telemetry: how often Huber is operating in the linear regime (|diff| > delta).
        # Call sites compute the boolean tensors cheaply from the UNWEIGHTED Huber loss (or diff),
        # then we compute region-weighted fractions using the *actual* masks used for training.
        def _weighted_linear_frac(is_linear: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
            is_linear = is_linear.to(device=loss.device)
            if is_linear.ndim == 4 and loss.ndim == 5 and layout == "video":
                is_linear = is_linear.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W)
            if is_linear.shape != loss.shape:
                raise ValueError(
                    "huber_is_linear tensor must match loss shape after normalization: "
                    f"is_linear={tuple(is_linear.shape)} loss={tuple(loss.shape)} layout={layout}"
                )

            # Reduce across channel dimension without expanding the compact mask.
            channel_dim = 1 if layout == "video" else 2
            linear_count = is_linear.sum(dim=channel_dim, dtype=torch.float32)
            linear_frac_per_voxel = linear_count / float(num_channels)

            # Squeeze the singleton channel dim on the compact mask to align with linear_frac_per_voxel.
            mask_singleton_dim = 1 if layout == "video" else 2
            region_compact = region_mask.detach().to(dtype=torch.float32).squeeze(mask_singleton_dim)

            denom = region_compact.sum(dtype=torch.float32).clamp_min(1e-8)
            return (linear_frac_per_voxel * region_compact).sum(dtype=torch.float32) / denom

        if target_huber_is_linear is not None:
            stats["huber/target_linear_frac"] = _weighted_linear_frac(target_huber_is_linear, mask_processed).detach()

        if prior_huber_is_linear is not None:
            stats["huber/prior_linear_frac"] = _weighted_linear_frac(prior_huber_is_linear, prior_mask).detach()

    # === Target Loss (inside mask) ===
    # Broadcasting: loss (B,C,F,H,W) * compact mask (B,1,F,H,W) → (B,C,F,H,W)
    target_loss_weighted = loss * mask_processed

    mask_area_scale_beta = float(getattr(args, "mask_area_scale_beta", 0.0))

    if normalize_per_sample:
        # Per-sample weighted mean, then average over batch
        # Reduce over C, F, H, W dimensions (keep batch)
        reduce_dims = tuple(range(1, loss.ndim))
        target_sum = target_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
        target_weight = mask_processed.sum(dim=reduce_dims, dtype=torch.float32) * num_channels
        # Handle samples with zero target weight: treat as 0 contribution
        valid_target = target_weight > 1e-8
        per_sample_target = torch.where(valid_target, target_sum / target_weight.clamp_min(1e-8), torch.zeros_like(target_sum))
        if mask_area_scale_beta > 0.0:
            # Use RAW mask mean so this reflects geometric coverage, not post gamma/min_weight distortion.
            # This keeps --mask_area_scale_beta effective even when --mask_min_weight > 0.
            area_ratio = mask_raw_unblurred.mean(dim=reduce_dims, dtype=torch.float32).clamp(0.0, 1.0)
            per_sample_target = per_sample_target * (area_ratio**mask_area_scale_beta)
        L_target = per_sample_target.mean()
    else:
        # Global weighted mean; weight sum scaled by channels (compact mask has 1 where loss has C)
        target_weight_sum = mask_processed.sum(dtype=torch.float32) * num_channels
        if target_weight_sum < 1e-8:
            _logger.warning(
                "All-zero mask weights detected — target loss is zero for this batch. "
                "This means no training signal is being applied. Check your mask images."
            )
            L_target = loss.new_zeros((), dtype=torch.float32)
        else:
            L_target = target_loss_weighted.sum(dtype=torch.float32) / target_weight_sum
            if mask_area_scale_beta > 0.0:
                area_ratio = mask_raw_unblurred.mean(dtype=torch.float32).clamp(0.0, 1.0)
                L_target = L_target * (area_ratio**mask_area_scale_beta)

    # === Prior Loss (outside mask) ===
    use_prior_term = prior_loss_unreduced is not None and (prior_preservation_weight > 0 or prior_weight_per_sample is not None)
    if use_prior_term:
        # Check if prior mask is effectively all zeros (skip computation)
        # Use float32 for the sum so clamp_min(1e-8) is meaningful under fp16/bf16
        prior_mask_sum = prior_mask.sum(dtype=torch.float32) * num_channels
        if prior_mask_sum < 1e-8:
            # No prior loss contribution (e.g., unmasked step or mask=1 everywhere)
            L_prior = loss.new_zeros((), dtype=torch.float32)
        else:
            # Prepare prior loss tensor
            prior_loss_unreduced = prior_loss_unreduced.to(loss.device, dtype=loss.dtype)
            if prior_loss_unreduced.ndim == 4:
                prior_loss_unreduced = prior_loss_unreduced.unsqueeze(2)

            if prior_loss_unreduced.shape != loss.shape:
                raise ValueError(
                    f"prior_loss_unreduced shape {tuple(prior_loss_unreduced.shape)} does not match "
                    f"loss shape {tuple(loss.shape)} after dimension normalization"
                )

            prior_loss_weighted = prior_loss_unreduced * prior_mask

            if normalize_per_sample:
                reduce_dims = tuple(range(1, loss.ndim))
                prior_sum = prior_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
                prior_weight = prior_mask.sum(dim=reduce_dims, dtype=torch.float32) * num_channels
                # Handle samples with zero prior weight: treat as 0 contribution
                valid_prior = prior_weight > 1e-8
                per_sample_prior = torch.where(valid_prior, prior_sum / prior_weight.clamp_min(1e-8), torch.zeros_like(prior_sum))
                if prior_weight_per_sample is not None:
                    L_prior = (per_sample_prior * prior_weight_per_sample).mean()
                else:
                    L_prior = per_sample_prior.mean() * prior_preservation_weight
            else:
                L_prior = (prior_loss_weighted.sum(dtype=torch.float32) / prior_mask_sum) * prior_preservation_weight
    else:
        L_prior = loss.new_zeros((), dtype=torch.float32)

    # === Combine: region-normalized means + explicit weighting ===
    if stats is not None:
        # Loss terms (effective contributions to the combined loss).
        stats["target"] = L_target.detach()
        stats["prior"] = L_prior.detach()

        # Mask summaries (compact tensors; use float32 reduction for stability).
        # These are useful for diagnosing:
        #   - target/prior area coverage
        #   - threshold-mode overlap bugs (should be ~0 after overlap prevention)
        raw_f32 = mask_raw_unblurred.detach().to(dtype=torch.float32)
        processed_f32 = mask_processed.detach().to(dtype=torch.float32)
        prior_f32 = prior_mask.detach().to(dtype=torch.float32)

        stats["mask/raw_mean"] = raw_f32.mean()
        stats["mask/raw_min"] = raw_f32.min()
        stats["mask/raw_max"] = raw_f32.max()

        stats["mask/processed_mean"] = processed_f32.mean()
        stats["mask/processed_min"] = processed_f32.min()
        stats["mask/processed_max"] = processed_f32.max()

        stats["mask/prior_mean"] = prior_f32.mean()
        stats["mask/prior_min"] = prior_f32.min()
        stats["mask/prior_max"] = prior_f32.max()

        stats["mask/overlap_mass"] = (processed_f32 * prior_f32).mean()

    return L_target + L_prior
