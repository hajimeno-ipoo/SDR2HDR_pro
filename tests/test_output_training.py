from dataclasses import replace
from pathlib import Path
import tempfile
import json

import numpy as np
import torch
import pytest

from scripts.export_model import export_torchscript
from scripts.prepare_live_sequences import adjacent_indices
from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.app import get_presets
from sdr2hdr.core import SDRToHDRProcessor
from sdr2hdr.hdr_guidance import linear_nits_to_pq_torch
from sdr2hdr.model import HDRGuidedEnhancementUNet


def test_training_render_matches_exported_production_and_backpropagates():
    torch.manual_seed(7)
    model = HDRGuidedEnhancementUNet().eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("luma_head.", "reconstruction_head.")))
    config = replace(get_presets()["natural"], backend="torch-cpu", ai_strength=.25,
                     target_luma_temporal_strength=0., luminance_guidance_strength=.7)
    train = SDRToHDRProcessor(config, TorchMapEnhancer.for_training(model, 1000.))
    frames = np.random.default_rng(4).integers(25, 230, (2, 96, 160, 3), dtype=np.uint8)
    # Reconstruction is gated to clipped highlights, absent in random midtones.
    frames[:, :40, :60] = 250
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidate.pt"
        export_torchscript(model, path, reference_nits=1000.)
        enhancer = TorchMapEnhancer(str(path))
        assert enhancer.reference_nits == 1000.
        production = SDRToHDRProcessor(config, enhancer)
        outputs = []
        for frame in frames:
            nits, _ = train.render_frame_nits_torch(frame)
            with torch.no_grad():
                expected = torch.clamp(torch.round(linear_nits_to_pq_torch(nits) * 65535.), 0, 65535)
            actual = production.process_frame(frame)
            np.testing.assert_array_equal(expected.to(torch.uint16).numpy(), actual)
            outputs.append(nits)
        loss = torch.log(outputs[1] + .1).mean() + (outputs[1] - outputs[0]).abs().mean() / 1000.
        loss.backward()
        for head in (model.luma_head, model.reconstruction_head):
            assert head.weight.grad is not None
            assert torch.isfinite(head.weight.grad).all()
            assert head.weight.grad.abs().sum() > 0
        assert model.enc1.block[0].weight.grad is None


def test_absent_reference_preserves_legacy_semantics():
    model = HDRGuidedEnhancementUNet().eval()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "old_v2.pt"
        export_torchscript(model, path)
        enhancer = TorchMapEnhancer(str(path))
        assert enhancer.reference_nits is None
        assert enhancer.estimate_guidance(np.zeros((64, 64, 3), np.float32)).reference_nits is None


def test_sequence_indices_are_adjacent_source_frames():
    assert adjacent_indices(480, [.1, .3, .5, .7, .9]) == [(48, 49), (144, 145), (240, 241), (335, 336), (431, 432)]
    assert adjacent_indices(2, [0., 1.]) == [(0, 1)]


@pytest.mark.parametrize("metadata", [{}, {"reference_nits": None}, {"reference_nits": -1}, {"reference_nits": True}])
def test_malformed_reference_is_not_silently_treated_as_legacy(metadata):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "malformed.pt"
        torch.jit.save(torch.jit.script(HDRGuidedEnhancementUNet().eval()), str(path),
                       _extra_files={"hdr_reference.json": json.dumps(metadata)})
        with pytest.raises(ValueError):
            TorchMapEnhancer(str(path))


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu"])
def test_explicit_reference_is_separate_from_output_ceiling(backend):
    torch.manual_seed(2)
    model = HDRGuidedEnhancementUNet().eval()
    with torch.no_grad():
        model.luma_head.weight.zero_()
        model.luma_head.bias.fill_(-2.)
    config = replace(get_presets()["natural"], backend=backend, disable_auto_exposure=True,
                     peak_nits=800., target_luma_temporal_strength=0.)
    frame = np.full((64, 96, 3), 100, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as temporary:
        outputs = []
        for label, reference in [("legacy", None), ("explicit800", 800.), ("teacher1000", 1000.)]:
            path = Path(temporary) / f"{label}.pt"
            export_torchscript(model, path, reference_nits=reference)
            processor = SDRToHDRProcessor(config, TorchMapEnhancer(str(path)))
            outputs.append(processor.process_frame(frame))
        np.testing.assert_array_equal(outputs[0], outputs[1])
        assert outputs[2].astype(np.float32).mean() > outputs[1].astype(np.float32).mean()
