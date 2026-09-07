import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np

from scripts.prepare_data import (
    collect_video_pairs,
    collect_video_pairs_with_metadata,
    convert_paired_frame_to_npz,
    tone_map_hdr_linear_to_sdr_linear,
)


class PrepareDataTests(unittest.TestCase):
    def test_collect_video_pairs_uses_manifest_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hdr_dir = root / "hdr"
            sdr_dir = root / "sdr"
            hdr_dir.mkdir()
            sdr_dir.mkdir()
            hdr_name = "0_Balance_Forest_HDR10_3840x2160_50000k.mp4"
            sdr_name = "0_Balance_Forest_SDR_3840x2160_50000k.mp4"
            (hdr_dir / hdr_name).touch()
            (sdr_dir / sdr_name).touch()
            manifest = root / "JOD_separate.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["video_name", "video_format", "resolution", "bitrate", "content_name", "Type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video_name": hdr_name,
                        "video_format": "HDR10",
                        "resolution": "3840x2160",
                        "bitrate": "50000",
                        "content_name": "0_Balance_Forest",
                        "Type": "Open-source",
                    }
                )
                writer.writerow(
                    {
                        "video_name": sdr_name,
                        "video_format": "SDR",
                        "resolution": "3840x2160",
                        "bitrate": "50000",
                        "content_name": "0_Balance_Forest",
                        "Type": "Open-source",
                    }
                )

            pairs = collect_video_pairs(hdr_dir, sdr_dir, manifest)

            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0], (hdr_dir / hdr_name, sdr_dir / sdr_name))

            pairs_with_metadata = collect_video_pairs_with_metadata(hdr_dir, sdr_dir, manifest)
            self.assertEqual(pairs_with_metadata[0].content_name, "0_Balance_Forest")

    def test_collect_video_pairs_rejects_unpaired_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hdr_dir = root / "hdr"
            sdr_dir = root / "sdr"
            hdr_dir.mkdir()
            sdr_dir.mkdir()
            hdr_name = "scene_HDR10_1920x1080_3000k.mp4"
            (hdr_dir / hdr_name).touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["video_name", "video_format", "resolution", "bitrate", "content_name", "Type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "video_name": hdr_name,
                        "video_format": "HDR10",
                        "resolution": "1920x1080",
                        "bitrate": "3000",
                        "content_name": "scene",
                        "Type": "Open-source",
                    }
                )

            with self.assertRaisesRegex(ValueError, "unpaired records"):
                collect_video_pairs(hdr_dir, sdr_dir, manifest)

    def test_collect_video_pairs_keeps_legacy_same_name_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hdr_dir = root / "hdr"
            sdr_dir = root / "sdr"
            hdr_dir.mkdir()
            sdr_dir.mkdir()
            (hdr_dir / "SolLevante.mov").touch()
            (sdr_dir / "SolLevante.mov").touch()

            pairs = collect_video_pairs(hdr_dir, sdr_dir)

            self.assertEqual(pairs, [(hdr_dir / "SolLevante.mov", sdr_dir / "SolLevante.mov")])

    def test_convert_paired_frame_to_npz_keeps_real_sdr_content(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sdr_path = root / "sdr.png"
            hdr_path = root / "hdr.png"
            output_path = root / "pair.npz"
            sdr_bgr8 = np.full((4, 4, 3), [64, 96, 128], dtype=np.uint8)
            hdr_bgr16 = np.full((4, 4, 3), [16384, 24576, 32768], dtype=np.uint16)
            self.assertTrue(cv2.imwrite(str(sdr_path), sdr_bgr8))
            self.assertTrue(cv2.imwrite(str(hdr_path), hdr_bgr16))

            convert_paired_frame_to_npz(
                sdr_path,
                hdr_path,
                output_path,
                peak_nits=1000.0,
                content_name="scene_a",
            )
            with np.load(output_path) as sample:
                self.assertEqual(sample["sdr_linear"].dtype, np.float16)
                self.assertEqual(sample["hdr_linear"].dtype, np.float16)
                self.assertEqual(str(sample["pair_source"]), "real_sdr_hdr")
                self.assertEqual(str(sample["content_name"]), "scene_a")
                self.assertEqual(sample["hdr_2020_linear"].shape, (4, 4, 3))
                self.assertFalse(np.allclose(sample["sdr_linear"], sample["hdr_linear"]))

    def test_tone_map_hdr_linear_to_sdr_linear_returns_bounded_frame(self) -> None:
        frame = np.full((8, 8, 3), 0.75, dtype=np.float32)
        out = tone_map_hdr_linear_to_sdr_linear(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(np.all(out >= 0.0))
        self.assertTrue(np.all(out <= 1.0))

    def test_tone_map_with_clip_ratio_and_quantization(self) -> None:
        frame = np.full((8, 8, 3), 1.2, dtype=np.float32)
        out_clipped = tone_map_hdr_linear_to_sdr_linear(frame, clip_ratio=0.5, quantize_8bit=True)
        self.assertEqual(out_clipped.shape, frame.shape)
        self.assertTrue(np.all(out_clipped >= 0.0))
        self.assertTrue(np.all(out_clipped <= 1.0))


if __name__ == "__main__":
    unittest.main()
