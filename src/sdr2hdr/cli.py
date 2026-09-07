from __future__ import annotations

import argparse
from pathlib import Path

from .app import (
    PRESETS,
    X265_PROFILE_DEFAULTS,
    ConversionCallbacks,
    ConversionRequest,
    build_output_path,
    run_conversion,
    validate_request,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SDR 動画を HDR10 に変換します。")
    parser.add_argument("input_path", help="入力 SDR 動画のパス")
    parser.add_argument("output_path", nargs="?", help="出力 HDR 動画のパス")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="natural")
    parser.add_argument("--encoder", default="libx265")
    parser.add_argument("--x265-mode", choices=sorted(X265_PROFILE_DEFAULTS), default="balanced")
    parser.add_argument("--backend", choices=["auto", "numpy", "cuda", "mps"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", required=True, help="TorchScript .pt 拡張モデルのパス")
    parser.add_argument("--ai-strength", type=float, default=0.25)
    parser.add_argument(
        "--hdr-guidance",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use v2 target-luminance/reconstruction guidance when supported.",
    )
    parser.add_argument(
        "--luminance-guidance-strength",
        type=float,
        default=0.70,
        help="Blend strength for learned HDR target luminance.",
    )
    parser.add_argument(
        "--reconstruction-strength",
        type=float,
        default=0.60,
        help="Maximum learned reconstruction contribution in clipped highlights.",
    )
    parser.add_argument(
        "--no-fallback-to-x265-on-hardware-error",
        action="store_true",
        help="ハードウェアエンコード失敗時に libx265 への自動フォールバックを無効にする",
    )
    parser.add_argument(
        "--discard-partial-output-on-cancel",
        action="store_true",
        help="キャンセル時に部分的な出力を保持せず削除する",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = args.output_path or build_output_path(args.input_path)
    model_path = str(Path(args.model_path))
    if Path(model_path).suffix.lower() != ".pt":
        parser.error("--model-path は .pt の TorchScript モデルを指定する必要があります。")

    request = ConversionRequest(
        input_path=str(Path(args.input_path)),
        output_path=output_path,
        preset=args.preset,
        encoder=args.encoder,
        x265_mode=args.x265_mode,
        backend=args.backend,
        device=args.device,
        model_path=model_path,
        ai_strength=args.ai_strength,
        fallback_to_x265_on_hardware_error=not args.no_fallback_to_x265_on_hardware_error,
        keep_partial_output_on_cancel=not args.discard_partial_output_on_cancel,
        hdr_guidance=args.hdr_guidance,
        luminance_guidance_strength=args.luminance_guidance_strength,
        reconstruction_strength=args.reconstruction_strength,
    )
    validate_request(request)

    callbacks = ConversionCallbacks(
        on_status=lambda message: print(message, flush=True),
        on_progress=lambda processed, total, fps: print(
            f"{processed}/{total or '?'} フレーム ({fps:.1f} fps)" if fps else f"{processed}/{total or '?'} フレーム",
            flush=True,
        ),
        on_complete=lambda result: print(
            f"{result.processed_frames} フレームでキャンセルされました"
            if result.cancelled
            else f"完了しました: {result.output_path}",
            flush=True,
        ),
        on_error=lambda message: print(message, flush=True),
    )
    run_conversion(request, callbacks=callbacks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
