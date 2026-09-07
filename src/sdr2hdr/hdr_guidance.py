from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def resolve_anchor_nits(tone: str, peak_nits: float, diffuse_white_nits: float = 203.0) -> float:
    """Resolve anchor luminance in nits for linear conversion."""
    if tone == "vivid":
        return float(peak_nits)
    return float(diffuse_white_nits)


def target_rel_to_linear(
    target_luma_rel: np.ndarray,
    peak_nits: float,
    anchor_nits: float,
) -> np.ndarray:
    """Convert relative target luminance (0..1) to internal linear luminance."""
    return target_luma_rel * float(peak_nits) / max(float(anchor_nits), 1e-4)


def target_rel_to_linear_torch(
    target_luma_rel: torch.Tensor,
    peak_nits: float,
    anchor_nits: float,
) -> torch.Tensor:
    """Torch variant of target_rel_to_linear."""
    return target_luma_rel * float(peak_nits) / max(float(anchor_nits), 1e-4)


def build_reconstruction_gate(
    clipped_white_mask: np.ndarray,
    reconstruction_confidence: np.ndarray,
    reconstruction_strength: float,
    protection: np.ndarray,
) -> np.ndarray:
    """Calculate highlight reconstruction gate."""
    gate = (
        clipped_white_mask
        * reconstruction_confidence
        * float(np.clip(reconstruction_strength, 0.0, 1.0))
        * (1.0 - np.clip(protection, 0.0, 1.0))
    )
    return np.clip(gate, 0.0, 1.0)


def build_reconstruction_gate_torch(
    clipped_white_mask: torch.Tensor,
    reconstruction_confidence: torch.Tensor,
    reconstruction_strength: float,
    protection: torch.Tensor,
) -> torch.Tensor:
    """Torch variant of build_reconstruction_gate."""
    gate = (
        clipped_white_mask
        * reconstruction_confidence
        * float(np.clip(reconstruction_strength, 0.0, 1.0))
        * (1.0 - torch.clamp(protection, 0.0, 1.0))
    )
    return torch.clamp(gate, 0.0, 1.0)


def blend_target_luma(
    legacy_target_luma: np.ndarray,
    model_target_linear: np.ndarray,
    guidance_weight: np.ndarray,
) -> np.ndarray:
    """Blend legacy target luma with model target linear luma."""
    w = np.clip(guidance_weight, 0.0, 1.0)
    return legacy_target_luma * (1.0 - w) + model_target_linear * w


def blend_target_luma_torch(
    legacy_target_luma: torch.Tensor,
    model_target_linear: torch.Tensor,
    guidance_weight: torch.Tensor,
) -> torch.Tensor:
    """Torch variant of blend_target_luma."""
    w = torch.clamp(guidance_weight, 0.0, 1.0)
    return legacy_target_luma * (1.0 - w) + model_target_linear * w


def clamp_to_peak(absolute_nits: np.ndarray, peak_nits: float) -> np.ndarray:
    """Clamp absolute luminance in nits to peak_nits."""
    return np.clip(absolute_nits, 0.0, float(peak_nits))


def clamp_to_peak_torch(absolute_nits: torch.Tensor, peak_nits: float) -> torch.Tensor:
    """Torch variant of clamp_to_peak."""
    return torch.clamp(absolute_nits, 0.0, float(peak_nits))


def linear_nits_to_pq(frame_nits: np.ndarray) -> np.ndarray:
    """Apply SMPTE ST 2084 PQ EOTF inverse directly to absolute nit values."""
    y = np.clip(frame_nits / 10000.0, 0.0, 1.0)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    ym = np.power(y, m1)
    return np.power((c1 + c2 * ym) / (1.0 + c3 * ym), m2)


def linear_nits_to_pq_torch(frame_nits: torch.Tensor) -> torch.Tensor:
    """Torch variant of linear_nits_to_pq."""
    y = torch.clamp(frame_nits / 10000.0, 0.0, 1.0)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    ym = torch.pow(y, m1)
    return torch.pow((c1 + c2 * ym) / (1.0 + c3 * ym), m2)
