"""Versioned output transforms; the existing SDR enhancement remains unchanged."""
from __future__ import annotations

import math

import numpy as np
import cv2

from .hdr_guidance import linear_nits_to_pq


ACES_OUTPUTS = {
    "openexr_acescg_1_3": (
        "cg-config-v2.2.0_aces-v1.3_ocio-v2.4",
        "ACES 1.1 - HDR Video (1000 nits & Rec.2020 lim)",
    ),
    "openexr_acescg_2_0": (
        "cg-config-v4.0.0_aces-v2.0_ocio-v2.5",
        "ACES 2.0 - HDR 1000 nits (Rec.2020)",
    ),
}
HLG_ENCODER = "hevc_hlg"
EXR_ENCODERS = {"openexr", "openexr_acescg", *ACES_OUTPUTS}
PRORES_ENCODERS = {"prores_422hq", "prores_4444", "prores_4444_xq"}

# ITU-R BT.2020 RGB -> CIE XYZ, D65. OCIO display transforms use 1 = 100 nit.
REC2020_TO_XYZ = np.array([
    [0.6369580483, 0.1446169036, 0.1688809752],
    [0.2627002120, 0.6779980715, 0.0593017165],
    [0.0000000000, 0.0280726930, 1.0609850577],
], dtype=np.float32)


class OutputColorTransform:
    """Convert display-linear BT.2020 nits using a fixed 1000-nit reference view."""

    def __init__(self, output: str, peak_nits: float) -> None:
        if not math.isfinite(peak_nits) or not 0 < peak_nits <= 1000:
            raise ValueError("ACEScg / HLG output requires a peak between 0 and 1000 nit.")
        import PyOpenColorIO as ocio

        self.output = output
        if output in ACES_OUTPUTS:
            config_name, view = ACES_OUTPUTS[output]
            config = ocio.Config.CreateFromBuiltinConfig(config_name)
            transform = ocio.DisplayViewTransform(
                src="ACEScg", display="Rec.2100-PQ - Display", view=view,
            )
            self.processor = config.getProcessor(
                transform, ocio.TRANSFORM_DIR_INVERSE,
            ).getDefaultCPUProcessor()
        elif output == HLG_ENCODER:
            transform = ocio.BuiltinTransform(
                style="DISPLAY - CIE-XYZ-D65_to_REC.2100-HLG-1000nit",
            )
            self.processor = ocio.Config.CreateRaw().getProcessor(transform).getDefaultCPUProcessor()
        else:
            raise ValueError(f"Unknown output transform: {output}")

    def render(self, frame_nits: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
        if self.output in ACES_OUTPUTS:
            pixels = np.ascontiguousarray(linear_nits_to_pq(frame_nits), dtype=np.float32)
            if size is not None and pixels.shape[:2] != (size[1], size[0]):
                # Match the existing PQ output's interpolation, before the nonlinear
                # inverse view. Interpolating ACES pixels afterwards changes edges.
                pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_LINEAR)
        else:
            pixels = np.ascontiguousarray(frame_nits @ REC2020_TO_XYZ.T / 100, dtype=np.float32)
        self.processor.applyRGB(pixels)
        if not np.isfinite(pixels).all():
            raise RuntimeError("Output colour transform produced non-finite pixels.")
        if self.output in ACES_OUTPUTS:
            if np.max(np.abs(pixels)) > np.finfo(np.float16).max:
                raise RuntimeError("ACEScg values exceed the half-float EXR range.")
            # Half-float EXR is written from planar G, B, R float32 input.
            return np.ascontiguousarray(pixels[..., [1, 2, 0]].transpose(2, 0, 1))
        if size is not None and pixels.shape[:2] != (size[1], size[0]):
            pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_LINEAR)
        return np.clip(np.round(pixels * 65535), 0, 65535).astype(np.uint16)


class EXRVideoTransform:
    """Render the selected EXR representation as display-referred BT.2020/PQ.

    Versioned ACEScg uses its official forward output transform. Display-linear
    AP1 only changes RGB coordinates and luminance units; it is not scene ACES.
    """

    def __init__(self, output: str, white_luminance: float) -> None:
        if output not in EXR_ENCODERS:
            raise ValueError(f"Not an EXR output: {output}")
        self.output = output
        self.white_luminance = white_luminance
        if output in ACES_OUTPUTS:
            import PyOpenColorIO as ocio
            config_name, view = ACES_OUTPUTS[output]
            config = ocio.Config.CreateFromBuiltinConfig(config_name)
            transform = ocio.DisplayViewTransform(
                src="ACEScg", display="Rec.2100-PQ - Display", view=view,
            )
            self.processor = config.getProcessor(
                transform, ocio.TRANSFORM_DIR_FORWARD,
            ).getDefaultCPUProcessor()
        elif output == "openexr_acescg":
            from .core import REC2020_TO_ACESCG
            self.ap1_to_rec2020 = np.linalg.inv(REC2020_TO_ACESCG)

    def to_pq(self, rgb: np.ndarray) -> np.ndarray:
        pixels = np.array(rgb, dtype=np.float32, order="C", copy=True)
        if self.output in ACES_OUTPUTS:
            self.processor.applyRGB(pixels)
        elif self.output == "openexr_acescg":
            nits = (pixels @ self.ap1_to_rec2020.T) * self.white_luminance
            pixels = linear_nits_to_pq(nits)
        if not np.isfinite(pixels).all():
            raise RuntimeError("EXR video transform produced non-finite pixels.")
        return pixels
