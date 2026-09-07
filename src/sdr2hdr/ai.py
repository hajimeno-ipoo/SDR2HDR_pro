from __future__ import annotations

from dataclasses import dataclass
import json
import math

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None


@dataclass
class EnhancementMaps:
    expansion: np.ndarray
    contrast: np.ndarray
    protection: np.ndarray


@dataclass
class EnhancementGuidance:
    expansion: np.ndarray
    contrast: np.ndarray
    protection: np.ndarray
    target_luma_rel: np.ndarray | None = None
    reconstruction_confidence: np.ndarray | None = None
    reference_nits: float | None = None

    @property
    def has_hdr_guidance(self) -> bool:
        return (
            self.target_luma_rel is not None
            and self.reconstruction_confidence is not None
        )

    def to_maps(self) -> EnhancementMaps:
        return EnhancementMaps(
            expansion=self.expansion,
            contrast=self.contrast,
            protection=self.protection,
        )


@dataclass
class EnhancementGuidanceTorch:
    expansion: torch.Tensor
    contrast: torch.Tensor
    protection: torch.Tensor
    target_luma_rel: torch.Tensor | None = None
    reconstruction_confidence: torch.Tensor | None = None
    reference_nits: float | None = None

    @property
    def has_hdr_guidance(self) -> bool:
        return (
            self.target_luma_rel is not None
            and self.reconstruction_confidence is not None
        )

    def to_guidance_numpy(self) -> EnhancementGuidance:
        return EnhancementGuidance(
            expansion=self.expansion.detach().cpu().numpy(),
            contrast=self.contrast.detach().cpu().numpy(),
            protection=self.protection.detach().cpu().numpy(),
            target_luma_rel=self.target_luma_rel.detach().cpu().numpy()
            if self.target_luma_rel is not None
            else None,
            reconstruction_confidence=self.reconstruction_confidence.detach().cpu().numpy()
            if self.reconstruction_confidence is not None
            else None,
            reference_nits=self.reference_nits,
        )


def parse_enhancement_output(
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Parse enhancement model output tensor (1, C, H, W) or (C, H, W).
    
    Returns:
        (expansion_res, contrast_res, protection_res, target_luma_rel, reconstruction_confidence)
    """
    if output.ndim == 4:
        channels = output.shape[1]
        if channels == 3:
            return output[:, 0], output[:, 1], output[:, 2], None, None
        elif channels == 5:
            return output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]
        else:
            raise RuntimeError(
                f"Unsupported enhancement model output: expected 3 or 5 channels, got {channels}"
            )
    elif output.ndim == 3:
        channels = output.shape[0]
        if channels == 3:
            return output[0], output[1], output[2], None, None
        elif channels == 5:
            return output[0], output[1], output[2], output[3], output[4]
        else:
            raise RuntimeError(
                f"Unsupported enhancement model output: expected 3 or 5 channels, got {channels}"
            )
    else:
        raise RuntimeError(f"Unexpected model output dimensions: {output.shape}")


def estimate_heuristic_maps(frame_linear: np.ndarray) -> EnhancementMaps:
    luminance = np.clip(
        0.2126 * frame_linear[..., 0]
        + 0.7152 * frame_linear[..., 1]
        + 0.0722 * frame_linear[..., 2],
        0.0,
        1.0,
    )
    smooth = cv2.GaussianBlur(luminance, (0, 0), 5.0)
    detail = np.clip(luminance - smooth, -0.2, 0.2)
    expansion = np.clip((luminance - 0.55) / 0.45, 0.0, 1.0)
    local_contrast = np.clip(np.abs(detail) * 6.0, 0.0, 1.0)
    chroma_spread = np.max(frame_linear, axis=2) - np.min(frame_linear, axis=2)
    protection = np.clip(1.0 - chroma_spread * 1.2, 0.0, 1.0)
    return EnhancementMaps(expansion=expansion, contrast=local_contrast, protection=protection)


class BaseEnhancer:
    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        raise NotImplementedError

    def estimate_guidance(self, frame_linear: np.ndarray) -> EnhancementGuidance:
        maps = self.estimate(frame_linear)
        return EnhancementGuidance(
            expansion=maps.expansion,
            contrast=maps.contrast,
            protection=maps.protection,
        )


class HeuristicEnhancer(BaseEnhancer):
    """Fallback enhancer that approximates highlight/detail recovery maps."""

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        return estimate_heuristic_maps(frame_linear)

    def estimate_guidance(self, frame_linear: np.ndarray) -> EnhancementGuidance:
        maps = self.estimate(frame_linear)
        return EnhancementGuidance(
            expansion=maps.expansion,
            contrast=maps.contrast,
            protection=maps.protection,
        )


class TorchMapEnhancer(BaseEnhancer):
    """Torch-backed enhancer supporting Legacy 3-channel and HDR Guidance v2 5-channel models."""

    def __init__(self, model_path: str, device: str = "cpu", inference_scale: float = 0.5) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is not installed; install sdr2hdr[ai] to use model inference")
        self.device = torch.device(device)
        extra_files = {"hdr_reference.json": ""}
        self.model = torch.jit.load(model_path, map_location=self.device, _extra_files=extra_files)
        self.reference_nits: float | None = None
        if extra_files["hdr_reference.json"]:
            metadata = json.loads(extra_files["hdr_reference.json"])
            if not isinstance(metadata, dict) or "reference_nits" not in metadata:
                raise ValueError("HDR metadata must contain reference_nits")
            value = metadata["reference_nits"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("reference_nits must be a number")
            self.reference_nits = float(value)
            if not math.isfinite(self.reference_nits) or self.reference_nits <= 0:
                raise ValueError("reference_nits must be finite and positive")
        self.differentiable = False
        self.model.eval()
        self.inference_scale = float(np.clip(inference_scale, 0.1, 1.0))

    @classmethod
    def for_training(cls, model: torch.nn.Module, reference_nits: float, inference_scale: float = 0.5) -> TorchMapEnhancer:
        """Use the production resize/map path without disabling model gradients."""
        if not math.isfinite(reference_nits) or reference_nits <= 0:
            raise ValueError("reference_nits must be finite and positive")
        enhancer = cls.__new__(cls)
        enhancer.device = next(model.parameters()).device
        enhancer.model = model
        enhancer.reference_nits = float(reference_nits)
        enhancer.inference_scale = float(np.clip(inference_scale, 0.1, 1.0))
        enhancer.differentiable = True
        return enhancer

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        if self.inference_scale >= 0.999:
            return height, width
        return (
            max(64, int(round(height * self.inference_scale))),
            max(64, int(round(width * self.inference_scale))),
        )

    def _heuristic_maps_torch(self, frame_linear_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        luminance = torch.clamp(
            frame_linear_t[..., 0] * 0.2126 + frame_linear_t[..., 1] * 0.7152 + frame_linear_t[..., 2] * 0.0722,
            0.0,
            1.0,
        )
        smooth = F.avg_pool2d(
            luminance[None, None], 11, stride=1, padding=5, count_include_pad=False,
        ).squeeze(0).squeeze(0)
        detail = torch.clamp(luminance - smooth, -0.2, 0.2)
        expansion = torch.clamp((luminance - 0.55) / 0.45, 0.0, 1.0)
        local_contrast = torch.clamp(torch.abs(detail) * 6.0, 0.0, 1.0)
        chroma_spread = torch.amax(frame_linear_t, dim=2) - torch.amin(frame_linear_t, dim=2)
        protection = torch.clamp(1.0 - chroma_spread * 1.2, 0.0, 1.0)
        return expansion, local_contrast, protection

    def _run_model(self, tensor: torch.Tensor) -> torch.Tensor:
        assert F is not None
        _, _, height, width = tensor.shape
        target_height, target_width = self._target_size(height, width)
        if (target_height, target_width) != (height, width):
            tensor = F.interpolate(
                tensor,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
        if self.differentiable:
            output = self.model(tensor)
        else:
            with torch.inference_mode():
                output = self.model(tensor)
        if tuple(output.shape[-2:]) != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return output

    def estimate_guidance_torch(self, frame_linear_t: torch.Tensor) -> EnhancementGuidanceTorch:
        tensor = frame_linear_t.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        output = self._run_model(tensor)
        exp_res, cont_res, prot_res, target_luma_rel, recon_conf = parse_enhancement_output(output)

        base_expansion, base_contrast, base_protection = self._heuristic_maps_torch(
            frame_linear_t.to(self.device, dtype=torch.float32)
        )
        expansion = torch.clamp(base_expansion + exp_res.squeeze(0), 0.0, 1.0)
        contrast = torch.clamp(base_contrast + cont_res.squeeze(0), 0.0, 1.0)
        protection = torch.clamp(base_protection + prot_res.squeeze(0), 0.0, 1.0)

        target_luma_rel_out = (
            torch.clamp(target_luma_rel.squeeze(0), 0.0, 1.0)
            if target_luma_rel is not None
            else None
        )
        recon_conf_out = (
            torch.clamp(recon_conf.squeeze(0), 0.0, 1.0)
            if recon_conf is not None
            else None
        )

        return EnhancementGuidanceTorch(
            expansion=expansion,
            contrast=contrast,
            protection=protection,
            target_luma_rel=target_luma_rel_out,
            reconstruction_confidence=recon_conf_out,
            reference_nits=self.reference_nits,
        )

    def estimate_torch(self, frame_linear_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        guidance = self.estimate_guidance_torch(frame_linear_t)
        return guidance.expansion, guidance.contrast, guidance.protection

    def estimate_guidance(self, frame_linear: np.ndarray) -> EnhancementGuidance:
        tensor = torch.from_numpy(frame_linear).to(self.device, dtype=torch.float32)
        guidance_t = self.estimate_guidance_torch(tensor)
        return guidance_t.to_guidance_numpy()

    def estimate(self, frame_linear: np.ndarray) -> EnhancementMaps:
        guidance = self.estimate_guidance(frame_linear)
        return guidance.to_maps()
