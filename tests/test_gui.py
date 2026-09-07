from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from sdr2hdr.gui import (
    build_backend_options,
    build_encoder_options,
    filter_models_for_backend,
    format_ai_strength,
    list_available_models,
    model_display_name,
    SDR2HDRGUI,
)


class GuiTests(unittest.TestCase):
    def test_build_encoder_options_for_windows(self) -> None:
        options = build_encoder_options("Windows")
        self.assertIn("libx265", options)
        self.assertIn("hevc_nvenc", options)
        self.assertIn("prores_422hq", options)
        self.assertIn("prores_4444", options)
        self.assertIn("openexr", options)
        self.assertIn("openexr_acescg", options)
        self.assertEqual(options["openexr_acescg"], "OpenEXR 16-bit AP1線形連番（表示基準）")
        self.assertNotIn("hevc_videotoolbox", options)

    def test_build_encoder_options_for_macos(self) -> None:
        options = build_encoder_options("Darwin")
        self.assertIn("libx265", options)
        self.assertIn("hevc_videotoolbox", options)
        self.assertIn("prores_422hq", options)
        self.assertIn("prores_4444", options)
        self.assertIn("prores_4444_xq", options)
        self.assertIn("openexr", options)
        self.assertIn("openexr_acescg", options)
        self.assertNotIn("hevc_nvenc", options)

    def test_build_backend_options_for_windows(self) -> None:
        options = build_backend_options("Windows")
        self.assertIn("auto", options)
        self.assertIn("cuda", options)
        self.assertIn("numpy", options)
        self.assertNotIn("directml", options)

    def test_build_backend_options_for_macos(self) -> None:
        options = build_backend_options("Darwin")
        self.assertIn("auto", options)
        self.assertIn("mps", options)
        self.assertIn("numpy", options)
        self.assertNotIn("cuda", options)

    def test_format_ai_strength(self) -> None:
        self.assertEqual(format_ai_strength(0.25), "0.25")
        self.assertEqual(format_ai_strength(0.2), "0.20")

    def test_missing_retained_model_does_not_select_another_model(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pt").write_bytes(b"pt")
            (root / "b.onnx").write_bytes(b"onnx")
            (root / "notes.txt").write_text("x", encoding="utf-8")
            models = list_available_models(root)
        self.assertEqual(models, [])

    def test_filter_models_for_backend_returns_pt_only(self) -> None:
        models = [Path("a.pt"), Path("b.onnx")]
        filtered = filter_models_for_backend(models, "auto", "Windows")
        self.assertEqual(filtered, [Path("a.pt")])

    def test_model_choices_only_include_retained_model(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [
                "a.pt", "enhancement_model_20260310.pt", "best.pt", "last.pt", "checkpoint.pt",
                "sollevante_v2/best.pt", "sollevante_v2/enhancement_model_v2.pt",
                "sollevante_v2_stable/enhancement_model_v2.pt", "other/best.pt",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            choices = list_available_models(root)
            self.assertEqual([p.relative_to(root).as_posix() for p in choices], [
                "enhancement_model_20260310.pt",
            ])
            labels = [model_display_name(path, root) for path in choices]
            self.assertTrue(labels[0].startswith("使用モデル:"))
            self.assertEqual(len(labels), len(set(labels)))

    def test_retained_model_refreshes_both_tabs(self) -> None:
        root = Path("/test/models")
        choices = [root / "enhancement_model_20260310.pt"]
        gui = SDR2HDRGUI.__new__(SDR2HDRGUI)
        gui.available_models = choices
        gui.filtered_models = choices
        gui.system_name = "Darwin"
        gui._selected_backend = Mock(return_value="auto")
        gui._sync_model_controls = Mock()
        gui._sync_mode_hint = Mock()
        gui.model_name_var = Mock()
        gui.model_path_var = Mock()
        gui.model_combo = Mock()
        gui.img_model_combo = Mock()
        with patch("sdr2hdr.gui.MODELS_DIR", root):
            labels = [model_display_name(path) for path in choices]
            for path, label in zip(choices, labels):
                gui.model_name_var.get.return_value = label
                gui._sync_selected_model()
                gui.model_path_var.set.assert_called_with(str(path))
            gui._refresh_model_choices()
            gui.model_combo.configure.assert_called_with(values=labels)
            gui.img_model_combo.configure.assert_called_with(values=labels)
            gui.model_path_var.set.assert_called_with(str(choices[0]))
            gui.model_name_var.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
