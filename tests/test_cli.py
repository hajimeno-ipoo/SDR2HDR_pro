from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from sdr2hdr.cli import build_parser, main


class CLITests(unittest.TestCase):
    def test_parser_accepts_guidance_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4",
            "output.mp4",
            "--model-path",
            "model.pt",
            "--hdr-guidance",
            "on",
            "--luminance-guidance-strength",
            "0.85",
            "--reconstruction-strength",
            "0.75",
        ])
        self.assertEqual(args.hdr_guidance, "on")
        self.assertEqual(args.luminance_guidance_strength, 0.85)
        self.assertEqual(args.reconstruction_strength, 0.75)

    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "input.mp4",
            "--model-path",
            "model.pt",
        ])
        self.assertEqual(args.hdr_guidance, "auto")
        self.assertEqual(args.luminance_guidance_strength, 0.70)
        self.assertEqual(args.reconstruction_strength, 0.60)


if __name__ == "__main__":
    unittest.main()
