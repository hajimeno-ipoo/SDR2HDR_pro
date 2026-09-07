from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from sdr2hdr.model import EnhancementUNet, HDRGuidedEnhancementUNet


def export_torchscript(model: nn.Module, output_path: Path, reference_nits: float | None = None) -> None:
    # Export TorchScript from a CPU copy so the serialized graph stays portable
    # across CUDA, MPS, and CPU runtimes.
    if isinstance(model, HDRGuidedEnhancementUNet):
        cpu_model: nn.Module = HDRGuidedEnhancementUNet()
    else:
        cpu_model = EnhancementUNet()

    cpu_state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    cpu_model.load_state_dict(cpu_state_dict)
    cpu_model.eval()
    scripted = torch.jit.script(cpu_model)
    extra_files = {}
    if reference_nits is not None:
        if not isinstance(model, HDRGuidedEnhancementUNet):
            raise ValueError("reference_nits requires a v2 luminance head")
        if not math.isfinite(reference_nits) or reference_nits <= 0:
            raise ValueError("reference_nits must be finite and positive")
        extra_files["hdr_reference.json"] = json.dumps({"reference_nits": reference_nits})
    torch.jit.save(scripted, str(output_path), _extra_files=extra_files)

    # Smoke test exported model
    loaded = torch.jit.load(str(output_path), map_location="cpu")
    dummy = torch.rand(1, 3, 256, 256)
    out = loaded(dummy)
    expected_channels = 5 if isinstance(cpu_model, HDRGuidedEnhancementUNet) else 3
    if out.shape[1] != expected_channels:
        raise RuntimeError(
            f"Exported model output channel mismatch: expected {expected_channels}, got {out.shape[1]}"
        )


def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    dummy = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=17,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export enhancement model checkpoint to TorchScript and/or ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["torchscript", "onnx", "both"], default="torchscript")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    
    # Determine model version
    is_v2 = False
    if isinstance(payload, dict):
        if payload.get("model_version") == "v2" or payload.get("output_channels") == 5:
            is_v2 = True
        elif any("map_head" in k or "luma_head" in k for k in state_dict.keys()):
            is_v2 = True
    elif any("map_head" in k or "luma_head" in k for k in state_dict.keys()):
        is_v2 = True

    model: nn.Module = HDRGuidedEnhancementUNet().to(device) if is_v2 else EnhancementUNet().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    output_path = Path(args.output)
    reference_nits = payload.get("reference_nits") if isinstance(payload, dict) else None
    if reference_nits is not None and args.format != "torchscript":
        parser.error("Models with reference_nits currently require TorchScript export")
    if args.format == "torchscript":
        export_torchscript(model, output_path, reference_nits=reference_nits)
    elif args.format == "onnx":
        export_onnx(model, output_path, device)
    else:
        export_torchscript(model, output_path.with_suffix(".pt"))
        export_onnx(model, output_path.with_suffix(".onnx"), device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
