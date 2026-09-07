from __future__ import annotations

import os
import platform
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.io import (
    ffprobe_video,
    finalize_process,
    has_expected_hdr_metadata,
    open_decoder,
    open_encoder,
    read_frame,
    restamp_hdr_metadata,
    is_videotoolbox_available,
    require_exr_metadata_tool,
    stamp_ap1_exr_sequence,
)

# AI libraries are imported lazily inside run_conversion to ensure physical separation
# and avoid loading heavy dependencies when only using Log Passthrough mode.

def get_presets():
    from sdr2hdr.core import ProcessorConfig
    return {
        "poc": ProcessorConfig(
            peak_nits=600.0,
            ai_strength=0.15,
            detail_boost=0.12,
            scene_smoothing=0.82,
            processing_scale=0.75,
            fast_mode=True,
            saturation=1.0,
        ),
        "balanced": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.30,
            detail_boost=0.20,
            scene_smoothing=0.88,
            fast_mode=True,
            saturation=1.10,
        ),
        "high": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.42,
            detail_boost=0.28,
            scene_smoothing=0.92,
            saturation=1.18,
            gamut_expansion=0.12,
        ),
        "portrait": ProcessorConfig(
            peak_nits=800.0,
            ai_strength=0.18,
            detail_boost=0.12,
            scene_smoothing=0.93,
            scene_cut_threshold=0.14,
            highlight_boost=0.72,
            subtitle_protection=0.90,
            shadow_noise_floor=0.10,
            skin_protection=0.82,
            shadow_rolloff=0.62,
            processing_scale=0.85,
            fast_mode=True,
            clipped_white_protection=0.78,
            near_white_rolloff_start=0.74,
            near_white_rolloff_strength=0.72,
            saturation=1.03,
        ),
        "natural": ProcessorConfig(
            peak_nits=800.0,
            ai_strength=0.15,
            detail_boost=0.10,
            scene_smoothing=0.93,
            scene_cut_threshold=0.14,
            highlight_boost=0.55,
            subtitle_protection=0.90,
            shadow_noise_floor=0.10,
            skin_protection=0.82,
            shadow_rolloff=0.62,
            processing_scale=0.85,
            fast_mode=True,
            clipped_white_protection=0.78,
            near_white_rolloff_start=0.74,
            near_white_rolloff_strength=0.72,
            saturation=1.05,
        ),
        "anime": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.22,
            detail_boost=0.08,
            scene_smoothing=0.96,
            highlight_boost=0.75,
            shadow_noise_floor=0.05,
            fast_mode=True,
            saturation=1.12,
        ),
        "cinema": ProcessorConfig(
            peak_nits=1000.0,
            tone="cinema",
            diffuse_white_nits=203.0,
            ai_strength=0.20,
            detail_boost=0.10,
            scene_smoothing=0.95,
            scene_cut_threshold=0.12,
            highlight_boost=0.60,
            subtitle_protection=0.92,
            shadow_noise_floor=0.08,
            skin_protection=0.85,
            shadow_rolloff=0.70,
            clipped_white_protection=0.75,
            near_white_rolloff_start=0.70,
            near_white_rolloff_strength=0.80,
            saturation=1.02,
            gamut_expansion=0.06,
        ),
    }

PRESETS = ("poc", "balanced", "high", "portrait", "natural", "anime", "cinema")
HDR10_ENCODERS = {"hevc_videotoolbox", "hevc_nvenc", "libx265"}

def get_preset_descriptions():
    return {
        "natural": "【推奨】自然な色合いと明るさを保つ、最も汎用的な設定です。",
        "balanced": "明るさとディテールのバランスが取れた、標準的な設定です。",
        "high": "輝度の拡張を強調し、よりダイナミックで力強い印象を与えます。",
        "portrait": "人物の肌の質感を保護し、柔らかなHDR表現を行います。",
        "anime": "アニメ特有の鮮やかな色彩と発光を強調し、輪郭ノイズを抑えた設定です。",
        "cinema": "【映画調】フィルムの滑らかな階調とハイライトロールオフを適用します。",
        "poc": "処理速度を優先した軽量な設定です（検証用）。",
    }

X265_PROFILE_DEFAULTS = {
    "preview": {"preset": "veryfast", "crf": 20},
    "balanced": {"preset": "medium", "crf": 16},
    "final": {"preset": "slow", "crf": 14},
}


@dataclass
class ConversionRequest:
    input_path: str
    output_path: str
    preset: str = "natural"
    encoder: str = "hevc_videotoolbox"
    x265_mode: str = "balanced"
    x265_preset: str | None = None
    x265_crf: int | None = None
    peak_nits: float | None = None
    ai_strength: float | None = None
    highlight_boost: float | None = None
    detail_boost: float | None = None
    processing_scale: float | None = None
    fast_mode: bool = False
    backend: str = "auto"
    model_path: str | None = None
    device: str = "cpu"
    max_frames: int | None = None
    fallback_to_x265_on_hardware_error: bool = False
    keep_partial_output_on_cancel: bool = True
    verify_hdr_metadata: bool = True
    saturation: float | None = None
    hdr_guidance: str = "auto"
    luminance_guidance_strength: float = 0.70
    reconstruction_strength: float = 0.60


@dataclass
class LogConversionRequest:
    input_path: str
    output_path: str
    encoder: str = "hevc_videotoolbox"
    x265_mode: str = "balanced"
    x265_preset: str | None = None
    x265_crf: int | None = None
    max_frames: int | None = None
    keep_partial_output_on_cancel: bool = True
    verify_hdr_metadata: bool = True


@dataclass
class ImageConversionRequest:
    input_path: str
    output_path: str
    preset: str = "natural"
    peak_nits: float | None = None
    ai_strength: float | None = None
    highlight_boost: float | None = None
    detail_boost: float | None = None
    processing_scale: float | None = None
    fast_mode: bool = False
    backend: str = "auto"
    model_path: str | None = None
    device: str = "cpu"
    verify_hdr_metadata: bool = True
    saturation: float | None = None
    hdr_guidance: str = "auto"
    luminance_guidance_strength: float = 0.70
    reconstruction_strength: float = 0.60


@dataclass
class ImageLogConversionRequest:
    input_path: str
    output_path: str
    verify_hdr_metadata: bool = True


@dataclass
class ConversionResult:
    output_path: str
    processed_frames: int
    total_frames: int | None
    cancelled: bool = False


@dataclass
class ConversionCallbacks:
    on_status: Callable[[str], None] | None = None
    on_progress: Callable[[int, int | None, float | None], None] | None = None
    on_complete: Callable[[ConversionResult], None] | None = None
    on_error: Callable[[str], None] | None = None


class CancelToken:
    def __init__(self) -> None:
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True


def build_output_path(input_path: str, extension: Optional[str] = None, encoder: Optional[str] = None) -> str:
    path = Path(input_path)
    if not path.suffix:
        return str(path.with_name(f"{path.name}_hdr"))
    suffix = path.suffix.lower()
    
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".avif", ".tif", ".tiff"}:
        # 静止画の場合
        output_suffix = extension or ".tif"
    else:
        # 動画の場合
        if encoder in {"prores_422hq", "prores_4444", "prores_4444_xq"}:
            output_suffix = ".mov"
        elif encoder in {"openexr", "openexr_acescg"}:
            return str(path.with_name(f"{path.stem}_hdr_%06d.exr"))
        else:
            output_suffix = ".mp4" if suffix in {".m2ts", ".mts", ".m2t", ".ts"} else path.suffix
        
    return str(path.with_name(f"{path.stem}_hdr{output_suffix}"))


def build_request_config(request: ConversionRequest | ImageConversionRequest) -> tuple[object, str, int]:
    from dataclasses import replace
    presets = get_presets()
    config = replace(presets[request.preset])
    if getattr(request, "encoder", None) == "openexr_acescg":
        config.output_color_space = "acescg_linear"
    if request.preset in {"portrait", "natural"} and request.model_path and request.ai_strength is None:
        config.ai_strength = 0.25 if request.preset == "portrait" else 0.15
    if request.peak_nits is not None:
        config.peak_nits = request.peak_nits
    if request.ai_strength is not None:
        config.ai_strength = request.ai_strength
    if request.highlight_boost is not None:
        config.highlight_boost = request.highlight_boost
    if request.detail_boost is not None:
        config.detail_boost = request.detail_boost
    if request.processing_scale is not None:
        config.processing_scale = request.processing_scale
    if request.fast_mode:
        config.fast_mode = True
    if getattr(request, "saturation", None) is not None:
        config.saturation = request.saturation
    if getattr(request, "hdr_guidance", None) is not None:
        config.hdr_guidance = request.hdr_guidance
    if getattr(request, "luminance_guidance_strength", None) is not None:
        config.luminance_guidance_strength = request.luminance_guidance_strength
    if getattr(request, "reconstruction_strength", None) is not None:
        config.reconstruction_strength = request.reconstruction_strength
    config.backend = request.backend
    profile = X265_PROFILE_DEFAULTS.get(getattr(request, "x265_mode", "balanced"), X265_PROFILE_DEFAULTS["balanced"])
    x265_preset = getattr(request, "x265_preset", None) or profile["preset"]
    x265_crf = getattr(request, "x265_crf", None) if getattr(request, "x265_crf", None) is not None else profile["crf"]
    return config, x265_preset, x265_crf


def validate_request(request: ConversionRequest | ImageConversionRequest) -> None:
    if getattr(request, "hdr_guidance", "auto") not in {"auto", "on", "off"}:
        raise ValueError(f"Unknown hdr_guidance option: {request.hdr_guidance}")
    if (
        getattr(request, "luminance_guidance_strength", 0.7) < 0.0
        or getattr(request, "luminance_guidance_strength", 0.7) > 1.0
    ):
        raise ValueError("luminance_guidance_strength must be between 0.0 and 1.0")
    if (
        getattr(request, "reconstruction_strength", 0.6) < 0.0
        or getattr(request, "reconstruction_strength", 0.6) > 1.0
    ):
        raise ValueError("reconstruction_strength must be between 0.0 and 1.0")
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    if not request.output_path.strip():
        raise ValueError("Output path is required.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if not request.model_path or not request.model_path.strip():
        raise ValueError("AI model is required. Select a model from the models folder.")
    model_path = Path(request.model_path)
    suffix = model_path.suffix.lower()
    if suffix != ".pt":
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    if request.preset not in get_presets():
        raise ValueError(f"Unknown preset: {request.preset}")
    valid_encoders = {
        "hevc_videotoolbox",
        "hevc_nvenc",
        "libx265",
        "prores_422hq",
        "prores_4444",
        "prores_4444_xq",
        "openexr",
        "openexr_acescg",
    }
    if hasattr(request, "encoder") and request.encoder not in valid_encoders:
        raise ValueError(f"Unknown encoder: {request.encoder}")
    if getattr(request, "encoder", None) == "openexr_acescg":
        require_exr_metadata_tool()
    if hasattr(request, "x265_mode") and request.x265_mode not in X265_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown x265 mode: {request.x265_mode}")
    if request.model_path and not model_path.exists():
        raise ValueError(f"Model file does not exist: {request.model_path}")
    if request.backend in {"cuda", "mps", "torch-cpu", "numpy"} and suffix != ".pt":
        raise ValueError(f"Backend '{request.backend}' requires a TorchScript model (.pt).")


def resolve_model_device(request: ConversionRequest, torch_device: str | None) -> str:
    if request.device != "auto":
        return request.device
    return torch_device or "cpu"


def resolve_model_backend(request: ConversionRequest, torch_device: str | None) -> str:
    if request.backend == "auto":
        if torch_device is not None:
            return torch_device
        return "numpy"
    if request.backend == "numpy":
        return "torch-cpu"
    return request.backend


def build_enhancer(request: ConversionRequest, torch_device: str | None) -> object:
    model_backend = resolve_model_backend(request, torch_device)
    return TorchMapEnhancer(
        request.model_path or "",
        device=resolve_model_device(request, torch_device if model_backend != "torch-cpu" else "cpu"),
    )


def _emit_status(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_status:
        callbacks.on_status(message)


def _emit_progress(callbacks: ConversionCallbacks | None, processed: int, total: int | None, fps: float | None) -> None:
    if callbacks and callbacks.on_progress:
        callbacks.on_progress(processed, total, fps)


def _emit_complete(callbacks: ConversionCallbacks | None, result: ConversionResult) -> None:
    if callbacks and callbacks.on_complete:
        callbacks.on_complete(result)


def _emit_error(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_error:
        callbacks.on_error(message)


def is_hardware_encoder_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        "videotoolbox" in lowered
        or "compression session" in lowered
        or "nvenc" in lowered
        or "nvidia" in lowered
        or "no capable devices found" in lowered
        or "cannot load nvcuda" in lowered
        or "unsupported device" in lowered
    )


def is_videotoolbox_failure(message: str) -> bool:
    lowered = message.lower()
    return "videotoolbox" in lowered or "compression session" in lowered


def default_encoder_for_platform(system_name: str | None = None) -> str:
    system_name = system_name or platform.system()
    if system_name == "Darwin":
        return "hevc_videotoolbox"
    if system_name == "Windows":
        return "hevc_nvenc"
    return "libx265"


def _terminate_process(process: object | None) -> None:
    if process is None:
        return
    try:
        process.terminate()
    except Exception:
        return


def _wait_terminated_process(process: object | None) -> None:
    if process is None:
        return
    for handle_name in ("stdin", "stdout", "stderr"):
        handle = getattr(process, handle_name, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass


def run_conversion(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    current_request = request
    attempted_fallback = False
    while True:
        try:
            return _run_conversion_once(current_request, callbacks=callbacks, cancel_token=cancel_token)
        except RuntimeError as exc:
            if (
                current_request.fallback_to_x265_on_hardware_error
                and current_request.encoder in {"hevc_videotoolbox", "hevc_nvenc"}
                and not attempted_fallback
                and is_hardware_encoder_failure(str(exc))
            ):
                attempted_fallback = True
                _emit_status(callbacks, f"{current_request.encoder} failed; falling back to libx265")
                try:
                    os.remove(current_request.output_path)
                except FileNotFoundError:
                    pass
                current_request = replace(current_request, encoder="libx265")
                continue
            raise


def _run_conversion_once(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    # Lazy imports to ensure AI engine is only loaded when needed
    from sdr2hdr.ai import HeuristicEnhancer
    from sdr2hdr.core import SDRToHDRProcessor
    import torch

    config, x265_preset, x265_crf = build_request_config(request)
    if request.encoder == "openexr_acescg":
        require_exr_metadata_tool()
    info = ffprobe_video(request.input_path)
    total_frames = request.max_frames if request.max_frames is not None else info.frames
    processor = SDRToHDRProcessor(config, enhancer=HeuristicEnhancer())
    _emit_status(callbacks, "Loading AI model")
    processor.enhancer = build_enhancer(request, processor.torch_device)
    if request.hdr_guidance == "on":
        if isinstance(processor.enhancer, TorchMapEnhancer):
            dummy = torch.rand(1, 3, 64, 64).to(processor.enhancer.device)
            with torch.inference_mode():
                out = processor.enhancer.model(dummy)
            if out.shape[1] != 5:
                raise RuntimeError("HDR Guidance requires a v2 5-channel model.")
        else:
            raise RuntimeError("HDR Guidance requires a v2 5-channel model.")
    decoder = open_decoder(request.input_path, info)
    encoder = open_encoder(
        request.output_path,
        request.input_path,
        info,
        config.peak_nits,
        encoder=request.encoder,
        x265_preset=x265_preset,
        x265_crf=x265_crf,
    )
    processed = 0
    cancelled = False
    encoder_broken_pipe = False
    start = time.monotonic()
    _emit_status(callbacks, "Preparing conversion")

    _SENTINEL = object()
    decode_q: queue.Queue = queue.Queue(maxsize=3)
    encode_q: queue.Queue = queue.Queue(maxsize=3)
    pipeline_error: list[BaseException] = []

    def _decoder_thread() -> None:
        try:
            frame_count = 0
            while True:
                if cancel_token and cancel_token.cancel_requested:
                    break
                if request.max_frames is not None and frame_count >= request.max_frames:
                    break
                frame = read_frame(decoder, info.width, info.height)
                if frame is None:
                    break
                decode_q.put(frame)
                frame_count += 1
        except Exception as exc:
            pipeline_error.append(exc)
        finally:
            decode_q.put(_SENTINEL)

    def _encoder_thread() -> None:
        nonlocal encoder_broken_pipe
        try:
            assert encoder.stdin is not None
            while True:
                item = encode_q.get()
                if item is _SENTINEL:
                    break
                try:
                    encoder.stdin.write(item.tobytes())
                except BrokenPipeError:
                    encoder_broken_pipe = True
                    break
        except Exception as exc:
            pipeline_error.append(exc)

    dec_thread = threading.Thread(target=_decoder_thread, daemon=True)
    enc_thread = threading.Thread(target=_encoder_thread, daemon=True)
    dec_thread.start()
    enc_thread.start()

    try:
        while True:
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                _emit_status(callbacks, "Cancelling")
                break
            if encoder_broken_pipe:
                break
            item = decode_q.get()
            if item is _SENTINEL:
                break
            hdr_frame = processor.process_frame(item)
            encode_q.put(hdr_frame)
            processed += 1
            if processed == 1:
                _emit_status(callbacks, "Converting")
            elapsed = max(time.monotonic() - start, 1e-6)
            fps = processed / elapsed
            _emit_progress(callbacks, processed, total_frames, fps)
        encode_q.put(_SENTINEL)
        enc_thread.join(timeout=30)
        dec_thread.join(timeout=10)
        if pipeline_error:
            raise pipeline_error[0]
    except Exception as exc:
        _emit_error(callbacks, str(exc))
        raise
    finally:
        if cancelled:
            _terminate_process(decoder)
            _wait_terminated_process(decoder)
            finalize_process(encoder, "encoder", allow_broken_pipe=True)
        else:
            encoder_error: RuntimeError | None = None
            try:
                finalize_process(
                    encoder,
                    "encoder",
                    allow_broken_pipe=bool(request.max_frames) or encoder_broken_pipe,
                )
            except RuntimeError as exc:
                encoder_error = exc
            try:
                finalize_process(
                    decoder,
                    "decoder",
                    allow_broken_pipe=bool(request.max_frames) or encoder_broken_pipe or encoder_error is not None,
                )
            except RuntimeError:
                if encoder_error is None:
                    raise
            if encoder_error is not None:
                raise encoder_error
    if request.encoder == "openexr_acescg" and processed > 0:
        from sdr2hdr.hdr_guidance import resolve_anchor_nits
        _emit_status(callbacks, "Writing AP1 EXR color metadata")
        stamp_ap1_exr_sequence(
            request.output_path, processed,
            resolve_anchor_nits(config.tone, config.peak_nits, config.diffuse_white_nits),
        )
    if cancelled:
        if request.keep_partial_output_on_cancel and processed > 0 and request.encoder not in {"openexr", "openexr_acescg"}:
            restamp_hdr_metadata(request.output_path)
        if not request.keep_partial_output_on_cancel or processed == 0:
            try:
                os.remove(request.output_path)
            except FileNotFoundError:
                pass
        result = ConversionResult(
            output_path=request.output_path,
            processed_frames=processed,
            total_frames=total_frames,
            cancelled=True,
        )
        _emit_complete(callbacks, result)
        return result
    if processed == 0:
        raise RuntimeError("No frames were processed. Check the input path and video stream.")
    measured_cll, measured_fall = processor.get_measured_hdr_metadata()
    if request.encoder in HDR10_ENCODERS and request.verify_hdr_metadata:
        _emit_status(callbacks, "HDR metadata missing; repairing output tags")
        restamp_hdr_metadata(request.output_path, int(round(measured_cll)), int(round(measured_fall)))
    result = ConversionResult(output_path=request.output_path, processed_frames=processed, total_frames=total_frames)
    _emit_status(callbacks, "Completed")
    _emit_complete(callbacks, result)
    return result

def run_log_conversion(
    request: LogConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    # バリデーション (AIモデル関連のチェックは一切行わない)
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    if not request.output_path.strip():
        raise ValueError("Output path is required.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")

    info = ffprobe_video(request.input_path)
    total_frames = request.max_frames if request.max_frames is not None else info.frames
    
    _emit_status(callbacks, "変換中 (Log素材用・色域拡張パススルー)")
    
    # 1. zscaleを使用して、見た目を変えずに色域だけをBT.709からBT.2020に拡張します。
    # 2. tin=bt709:t=smpte2084 により、Logの階調を維持したままHDR10の器（PQ）へマッピングします。
    # 3. エラー187を回避するため、全てのパラメータ (p, t, m) を明示的に指定します。
    vf_chain = (
        "zscale=pin=bt709:tin=bt709:min=bt709:"
        "p=bt2020:t=smpte2084:m=bt2020nc:range=tv,"
        "format=gbrpf32le" if request.encoder == "openexr" else "format=yuv420p10le"
    )

    # Professional HDR10 Static Metadata (SEI messages)
    cmd = [
        "ffmpeg", "-y", "-i", request.input_path,
        "-vf", vf_chain,
    ]

    if request.encoder in {"prores_422hq", "prores_4444"}:
        is_4444 = request.encoder == "prores_4444"
        prores_encoder = "prores_videotoolbox" if is_videotoolbox_available() else "prores_ks"
        cmd += [
            "-c:v", prores_encoder,
            "-profile:v", "4" if is_4444 else "3",
            "-pix_fmt", "p410le" if is_4444 and prores_encoder == "prores_videotoolbox" else (
                "yuv444p10le" if is_4444 else "yuv422p10le"
            ),
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
            "-movflags", "+write_colr",
            "-c:a", "copy",
        ]
    elif request.encoder == "openexr":
        cmd += [
            "-f", "image2",
            "-c:v", "exr",
            "-format", "half",
            "-compression", "zip16",
            "-pix_fmt", "gbrpf32le",
        ]
    else:
        x265_params = (
            "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):"
            "max-cll=1000,400:hdr10=1"
        )
        cmd += [
            "-c:v", "libx265",
            "-crf", "10",
            "-preset", "medium",
            "-tag:v", "hvc1",
            "-x265-params", x265_params,
            "-c:a", "copy",
        ]
        
    if request.max_frames is not None:
        cmd.extend(["-frames:v", str(request.max_frames)])
        
    cmd.append(request.output_path)
    
    import subprocess
    import re

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    processed = 0
    cancelled = False
    start = time.monotonic()
    frame_re = re.compile(r"frame=\s*(\d+)")
    
    try:
        assert process.stderr is not None
        for line in process.stderr:
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                _emit_status(callbacks, "キャンセル中")
                _terminate_process(process)
                break
                
            match = frame_re.search(line)
            if match:
                processed = int(match.group(1))
                elapsed = max(time.monotonic() - start, 1e-6)
                fps = processed / elapsed
                _emit_progress(callbacks, processed, total_frames, fps)
                
        process.wait()
    except Exception as exc:
        _terminate_process(process)
        _emit_error(callbacks, str(exc))
        raise
    finally:
        _wait_terminated_process(process)
        
    if cancelled:
        if not request.keep_partial_output_on_cancel or processed == 0:
            try:
                os.remove(request.output_path)
            except FileNotFoundError:
                pass
        result = ConversionResult(
            output_path=request.output_path,
            processed_frames=processed,
            total_frames=total_frames,
            cancelled=True,
        )
        _emit_complete(callbacks, result)
        return result
        
    if process.returncode != 0:
        raise RuntimeError(f"FFmpegがエラーコード {process.returncode} で終了しました。")
        
    if request.encoder in HDR10_ENCODERS and request.verify_hdr_metadata and not has_expected_hdr_metadata(request.output_path):
        _emit_status(callbacks, "HDRメタデータが不足しています。タグを修復中...")
        restamp_hdr_metadata(request.output_path)
        
    result = ConversionResult(output_path=request.output_path, processed_frames=processed, total_frames=total_frames)
    _emit_status(callbacks, "完了")
    _emit_complete(callbacks, result)
    return result


def run_image_conversion(
    request: ImageConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    from sdr2hdr.ai import HeuristicEnhancer
    from sdr2hdr.core import SDRToHDRProcessor
    from sdr2hdr.io import load_image, save_image_hdr

    config, _, _ = build_request_config(request)
    _emit_status(callbacks, "AIモデルの読み込み中")
    processor = SDRToHDRProcessor(config, enhancer=HeuristicEnhancer())
    processor.enhancer = build_enhancer(request, processor.torch_device)
    
    _emit_status(callbacks, "画像を読み込み中")
    img_bgr8 = load_image(request.input_path)
    
    _emit_status(callbacks, "AI処理中")
    hdr_rgb48 = processor.process_frame(img_bgr8)
    
    _emit_status(callbacks, "HDR画像を保存中")
    save_image_hdr(request.output_path, hdr_rgb48, peak_nits=config.peak_nits)
    
    result = ConversionResult(output_path=request.output_path, processed_frames=1, total_frames=1)
    _emit_status(callbacks, "完了")
    _emit_complete(callbacks, result)
    return result


def run_image_log_conversion(
    request: ImageLogConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    _emit_status(callbacks, "変換中 (静止画 Logパススルー)")
    
    vf_chain = (
        "zscale=pin=bt709:tin=bt709:min=bt709:"
        "p=bt2020:t=smpte2084:m=bt2020nc:range=tv,"
        "format=yuv420p10le"
    )
    
    ext = Path(request.output_path).suffix.lower()
    
    cmd = ["ffmpeg", "-y", "-i", request.input_path]
    
    if ext == ".jxl":
        cmd += [
            "-vf", vf_chain,
            "-c:v", "libjxl",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    elif ext == ".avif":
        cmd += [
            "-vf", vf_chain,
            "-c:v", "libaom-av1",
            "-still-picture", "1",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    else:
        # Default to high quality TIFF if unknown
        cmd += [
            "-vf", "zscale=pin=bt709:tin=bt709:min=bt709:p=bt2020:t=smpte2084:m=bt2020nc:range=tv,format=rgb48le",
            "-c:v", "tiff",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]

    cmd.append(request.output_path)
    
    import subprocess
    result_proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if result_proc.returncode != 0:
        raise RuntimeError(f"FFmpeg image conversion failed: {result_proc.stderr}")
        
    result = ConversionResult(output_path=request.output_path, processed_frames=1, total_frames=1)
    _emit_status(callbacks, "完了")
    _emit_complete(callbacks, result)
    return result
