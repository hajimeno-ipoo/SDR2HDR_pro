from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from scripts.train import (
    compute_loss,
    compute_loss_v2,
    initialize_v2_guidance_heads,
    resolve_training_device,
    set_training_mode,
    validate_content_separation,
    split_train_validation_indices,
    split_train_validation_indices_by_group,
)
from sdr2hdr.model import HDRGuidedEnhancementUNet


class TrainScriptTests(unittest.TestCase):
    def test_separate_validation_rejects_same_content(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_content_separation(['scene_a', 'scene_b'], ['scene_b'])
        with self.assertRaisesRegex(ValueError, 'both contain'):
            validate_content_separation(['scene_a'], [])
        validate_content_separation(['scene_a', 'scene_a'], ['scene_b'])

    def test_frozen_shared_state_survives_head_training(self) -> None:
        model = HDRGuidedEnhancementUNet()
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("luma_head.", "reconstruction_head."))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        set_training_mode(model, freeze_shared=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        model(torch.rand(2, 3, 32, 32))[:, 3:].mean().backward()
        optimizer.step()
        after = model.state_dict()
        for key in before:
            if not key.startswith(("luma_head.", "reconstruction_head.")):
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before["luma_head.weight"], after["luma_head.weight"]))

    def test_resolve_training_device_prefers_cuda(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=True):
            device = resolve_training_device("auto")
        self.assertEqual(device.type, "cuda")

    def test_unfrozen_training_updates_batch_statistics(self) -> None:
        model = HDRGuidedEnhancementUNet()
        model.eval()
        before = model.enc1.block[1].num_batches_tracked.clone()
        set_training_mode(model, freeze_shared=False)
        model(torch.rand(2, 3, 32, 32))
        self.assertEqual(int(model.enc1.block[1].num_batches_tracked - before), 1)

    def test_resolve_training_device_uses_mps_when_cuda_is_unavailable(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=True):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "mps")

    def test_resolve_training_device_falls_back_to_cpu(self) -> None:
        with patch("scripts.train.torch.cuda.is_available", return_value=False):
            with patch("scripts.train.torch.backends.mps.is_available", return_value=False):
                device = resolve_training_device("auto")
        self.assertEqual(device.type, "cpu")

    def test_resolve_training_device_respects_explicit_request(self) -> None:
        device = resolve_training_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_compute_loss_v2_finite(self) -> None:
        batch_size = 2
        pred = torch.sigmoid(torch.randn(batch_size, 5, 32, 32))
        target_maps = torch.zeros(batch_size, 3, 32, 32)
        target_luma_rel = torch.full((batch_size, 1, 32, 32), 0.5)
        reconstruction_target = torch.zeros(batch_size, 1, 32, 32)
        identity_mask = torch.ones(batch_size, 1, 32, 32)
        clip_mask = torch.zeros(batch_size, 1, 32, 32)

        total_loss, metrics = compute_loss_v2(
            pred,
            target_maps,
            target_luma_rel,
            reconstruction_target,
            identity_mask,
            clip_mask,
        )
        self.assertTrue(torch.isfinite(total_loss))
        self.assertIn("luma", metrics)
        self.assertIn("recon", metrics)
        self.assertIn("identity", metrics)
        self.assertIn("grad", metrics)

    def test_compute_loss_v2_penalizes_clipped_reconstruction_error(self) -> None:
        batch_size = 1
        # Prediction with 0.1 luma everywhere
        pred_low = torch.zeros(batch_size, 5, 16, 16)
        pred_low[:, 3:4] = 0.1
        pred_low[:, 4:5] = 0.9

        # Target with 0.9 luma in clipped region
        target_maps = torch.zeros(batch_size, 3, 16, 16)
        target_luma_rel = torch.full((batch_size, 1, 16, 16), 0.9)
        reconstruction_target = torch.ones(batch_size, 1, 16, 16)
        identity_mask = torch.zeros(batch_size, 1, 16, 16)
        clip_mask = torch.ones(batch_size, 1, 16, 16)

        loss_error, _ = compute_loss_v2(
            pred_low,
            target_maps,
            target_luma_rel,
            reconstruction_target,
            identity_mask,
            clip_mask,
        )

        pred_correct = pred_low.clone()
        pred_correct[:, 3:4] = 0.9
        pred_correct[:, 4:5] = 1.0

        loss_correct, _ = compute_loss_v2(
            pred_correct,
            target_maps,
            target_luma_rel,
            reconstruction_target,
            identity_mask,
            clip_mask,
        )
        self.assertLess(float(loss_correct), float(loss_error))

    def test_validation_indices_are_spread_across_sequence(self) -> None:
        train_indices, val_indices = split_train_validation_indices(53, 5)

        self.assertEqual(val_indices, [0, 13, 26, 39, 52])
        self.assertEqual(set(train_indices) | set(val_indices), set(range(53)))
        self.assertEqual(set(train_indices) & set(val_indices), set())

    def test_validation_indices_keep_content_groups_together(self) -> None:
        groups = ["scene_a", "scene_a", "scene_b", "scene_b", "scene_c", "scene_c", "scene_d", "scene_d"]

        train_indices, val_indices = split_train_validation_indices_by_group(groups, validation_size=4)

        train_groups = {groups[index] for index in train_indices}
        val_groups = {groups[index] for index in val_indices}
        self.assertEqual(train_groups & val_groups, set())
        self.assertEqual(val_groups, {"scene_a", "scene_d"})
        self.assertEqual(set(train_indices) | set(val_indices), set(range(len(groups))))

    def test_v2_guidance_heads_start_conservatively(self) -> None:
        model = HDRGuidedEnhancementUNet()
        initialize_v2_guidance_heads(model)

        with torch.no_grad():
            self.assertAlmostEqual(float(torch.sigmoid(model.luma_head.bias).item()), 0.10, places=5)
            self.assertAlmostEqual(float(torch.sigmoid(model.reconstruction_head.bias).item()), 0.03, places=5)
            self.assertLess(float(model.luma_head.weight.abs().mean()), 0.05)
            self.assertLess(float(model.reconstruction_head.weight.abs().mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
