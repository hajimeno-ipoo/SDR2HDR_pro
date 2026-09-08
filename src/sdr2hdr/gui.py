from __future__ import annotations

import os
import platform
import queue
import re
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sdr2hdr.app import (
    CancelToken,
    ConversionCallbacks,
    ConversionRequest,
    LogConversionRequest,
    ImageConversionRequest,
    ImageLogConversionRequest,
    get_presets,
    get_preset_descriptions,
    X265_PROFILE_DEFAULTS,
    build_output_path,
    default_encoder_for_platform,
    run_conversion,
    run_log_conversion,
    run_image_conversion,
    run_image_log_conversion,
    validate_request,
    validate_export_request,
    output_extensions,
)

X265_MODE_OPTIONS = {
    "preview": "プレビュー (高速)",
    "balanced": "バランス",
    "final": "最終 (最高品質)",
}

EXR_DELIVERY_OPTIONS = {
    "zip": "EXR連番のZIP",
    "video": "HDR動画（ProRes 4444 / MOV）",
}

MODELS_DIR = Path.cwd() / "models"


def describe_mode_hint(encoder: str, mode: str, backend: str, preset: str, model_path: str, exr_delivery: str = "zip") -> str:
    if exr_delivery == "video" and encoder.startswith("openexr"):
        if encoder in {"openexr_acescg_1_3", "openexr_acescg_2_0"}:
            version = "1.3" if encoder.endswith("1_3") else "2.0"
            return f"ACES {version}の公式出力変換でBT.2020/PQのHDR動画へ変換します。ACEScgを保持する場合はZIPを選んでください。"
        if encoder == "openexr_acescg":
            return "表示基準のAP1線形データをBT.2020/PQへ変換してHDR動画で保存します。AP1線形データを保持する場合はZIPを選んでください。音声も動画に含めます。"
        return "BT.2020/PQのEXR連番をHDR動画として保存します。音声も動画に含めます。"
    descriptions = {
        "prores_422hq": "編集用のProRes 422 HQをMOVで保存します。入力精度は10bit、色はBT.2020/PQです。",
        "prores_4444": "編集用のProRes 4444をMOVで保存します。このアプリの入力精度は10bitです。色はBT.2020/PQです。",
        "prores_4444_xq": "ProRes 4444 XQを12bit精度でMOV保存します。対応するMacとFFmpegが必要です。色はBT.2020/PQです。",
        "openexr": "BT.2020/PQのEXR連番を、音声があれば音声ファイルと一緒に1つのZIPへ保存します。",
        "openexr_acescg": "表示基準のAP1線形EXR連番をZIPへ保存します。撮影時の光量を復元する形式ではありません。音声があれば同梱します。",
        "openexr_acescg_1_3": "ACES 1.3の公式出力変換を逆変換したACEScgのEXR連番をZIPへ保存します。音声があれば同梱します。",
        "openexr_acescg_2_0": "ACES 2.0の公式出力変換を逆変換したACEScgのEXR連番をZIPへ保存します。音声があれば同梱します。",
        "hevc_hlg": "HEVC/H.265の10bit HLG動画です。基準ピークは1000 nitです。速度/品質で圧縮設定を選べます。",
        "hevc_videotoolbox": "MacのVideoToolboxでHEVC/H.265の10bit PQ動画を保存します。",
        "hevc_nvenc": "NVIDIA GPUのNVENCでHEVC/H.265の10bit PQ動画を保存します。",
        "libx265": "CPUでHEVC/H.265の10bit PQ動画を保存します。速度/品質で圧縮設定を選べます。",
    }
    return descriptions.get(encoder, "")


def describe_image_format(extension: str) -> str:
    return {
        ".jpg": "Ultra HDR JPEGで保存します。10bit PQ入力からHDR復元用のゲインマップを生成します。非可逆圧縮です。非対応アプリではSDR表示になります。",
        ".png": "16bit RGBのPNGでロスレス保存します。BT.2020/PQの色情報を記録します。HDR表示には対応アプリが必要です。",
        ".tif": "16bit RGBのTIFFで保存します。画素はBT.2020/PQです。FFmpegのTIFF色情報の記録には制限があります。",
        ".jxl": "16bit RGBのJPEG XLで保存します。HDR変換後の画素をロスレス圧縮します。色はBT.2020/PQです。",
        ".avif": "10bitのAVIFで保存します。圧縮により画素値が変わる閲覧向けの形式です。色はBT.2020/PQです。",
    }[extension]


def format_ai_strength(value: float) -> str:
    return f"{value:.2f}"

def build_encoder_options(system_name: str | None = None) -> dict[str, str]:
    system_name = system_name or platform.system()
    options = {
        "libx265": "HEVC / H.265（CPU）",
        "prores_422hq": "Apple ProRes 422 HQ（MOV / 10bit）",
        "prores_4444": "Apple ProRes 4444（MOV / 10bit入力）",
        "openexr": "OpenEXR（BT.2020/PQ）",
        "openexr_acescg": "OpenEXR（表示基準AP1線形）",
        "openexr_acescg_1_3": "OpenEXR ACEScg（ACES 1.3）",
        "openexr_acescg_2_0": "OpenEXR ACEScg（ACES 2.0）",
        "hevc_hlg": "HEVC 10-bit HLG（基準1000 nit）",
    }
    if system_name == "Darwin":
        options["prores_4444_xq"] = "Apple ProRes 4444 XQ（MOV / 12bit）"
        options["hevc_videotoolbox"] = "HEVC / H.265（Mac）"
    elif system_name == "Windows":
        options["hevc_nvenc"] = "HEVC / H.265（NVIDIA）"
    return options


def build_backend_options(system_name: str | None = None) -> dict[str, str]:
    system_name = system_name or platform.system()
    options = {"auto": "自動 (推奨)"}
    if system_name == "Darwin":
        options["mps"] = "MPS (Apple GPU)"
    if system_name == "Windows":
        options["cuda"] = "CUDA (NVIDIA GPU)"
    options["numpy"] = "CPU / NumPy"
    return options


def is_exr_sequence(path: str) -> bool:
    output = Path(path)
    return output.suffix.lower() == ".exr" and re.search(r"%0?\d*d", output.name) is not None


def open_path(path: str) -> None:
    system_name = platform.system()
    if system_name == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if system_name == "Darwin":
        subprocess.run(["open", path], check=False)
        return
    subprocess.run(["xdg-open", path], check=False)


class AppState:
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueJob:
    request: ConversionRequest | LogConversionRequest | ImageConversionRequest | ImageLogConversionRequest
    status: str = "queued"


STATUS_LABELS = {
    "queued": "待機中",
    "starting": "開始中",
    "running": "実行中",
    "cancelling": "キャンセル中",
    "completed": "完了",
    "failed": "失敗",
    "cancelled": "キャンセル済",
}


def list_available_models(models_dir: Path | None = None) -> list[Path]:
    models_dir = models_dir or MODELS_DIR
    model = models_dir / "enhancement_model_20260310.pt"
    return [model] if model.is_file() else []


def model_display_name(path: Path, models_dir: Path | None = None) -> str:
    relative = path.relative_to(models_dir or MODELS_DIR).as_posix()
    return f"使用モデル: {relative}"


def filter_models_for_backend(models: list[Path], backend: str, system_name: str | None = None) -> list[Path]:
    return [path for path in models if path.suffix.lower() == ".pt"]


class SDR2HDRGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("sdr2hdr")
        self.root.geometry("1120x720")
        self.state = AppState.IDLE
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_token: CancelToken | None = None
        self.last_output_path: str | None = None
        self.queue_jobs: list[QueueJob] = []
        self._input_batches = {}
        self._path_labels = {}
        self.current_job_index: int | None = None

        self.system_name = platform.system()
        self.encoder_options = build_encoder_options(self.system_name)
        self.backend_options = build_backend_options(self.system_name)
        self.available_models = list_available_models()
        self.filtered_models = filter_models_for_backend(self.available_models, "auto", self.system_name)
        default_encoder = default_encoder_for_platform(self.system_name)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.preset_var = tk.StringVar(value="natural")
        self.encoder_var = tk.StringVar(value=self.encoder_options[default_encoder])
        self.x265_mode_var = tk.StringVar(value=X265_MODE_OPTIONS["balanced"])
        self.backend_var = tk.StringVar(value=self.backend_options["auto"])
        default_model = model_display_name(self.filtered_models[0]) if self.filtered_models else ""
        self.model_name_var = tk.StringVar(value=default_model)
        self.model_path_var = tk.StringVar(value=str(self.filtered_models[0]) if self.filtered_models else "")
        self.ai_strength_var = tk.DoubleVar(value=0.15)
        self.ai_strength_label_var = tk.StringVar(value=format_ai_strength(0.15))
        self.status_var = tk.StringVar(value="待機中")
        self.progress_var = tk.StringVar(value="0 フレーム")
        self.mode_hint_var = tk.StringVar(value="")
        self.log_mode_hint_var = tk.StringVar(value="")
        self.img_mode_hint_var = tk.StringVar(value="")
        self.img_log_mode_hint_var = tk.StringVar(value="")
        self._output_path_state = {}
        self.saturation_var = tk.DoubleVar(value=1.05)
        self.saturation_label_var = tk.StringVar(value="1.05x")
        self.hdr_guidance_var = tk.StringVar(value="auto")
        self.luminance_guidance_var = tk.DoubleVar(value=0.70)
        self.reconstruction_strength_var = tk.DoubleVar(value=0.60)
        
        self.log_input_var = tk.StringVar()
        self.log_output_var = tk.StringVar()
        self.log_encoder_var = tk.StringVar(value=self.encoder_options[default_encoder])
        self.log_x265_mode_var = tk.StringVar(value=X265_MODE_OPTIONS["balanced"])
        self.exr_delivery_var = tk.StringVar(value=EXR_DELIVERY_OPTIONS["zip"])
        self.log_exr_delivery_var = tk.StringVar(value=EXR_DELIVERY_OPTIONS["zip"])
        
        # Image用の変数
        self.img_input_var = tk.StringVar()
        self.img_output_var = tk.StringVar()
        self.img_log_input_var = tk.StringVar()
        self.img_log_output_var = tk.StringVar()
        
        # 画像出力形式
        self.img_format_options = {".tif": "TIFF (16-bit / 編集用)", ".jxl": "JPEG XL (ロスレス / 保存用)", ".avif": "AVIF (高圧縮 / 閲覧用)", ".jpg": "JPEG (Ultra HDR / ゲインマップ)", ".png": "PNG (16-bit / HDR / ロスレス)"}
        self.img_format_var = tk.StringVar(value=self.img_format_options[".tif"])
        self.img_log_format_var = tk.StringVar(value=self.img_format_options[".tif"])

        self._build()
        self._set_state(AppState.IDLE)
        self.root.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(2, weight=1)

        title = ttk.Label(outer, text="SDR to HDR10 コンバーター", font=("Helvetica", 18, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w")
        subtitle = ttk.Label(
            outer,
            text="実写映像・画像変換用のキュー対応デスクトップUI",
            font=("Helvetica", 11),
        )
        subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 16))

        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)

        right = ttk.Frame(outer)
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Tabs
        self.ai_tab = ttk.Frame(self.notebook, padding=12)
        self.log_tab = ttk.Frame(self.notebook, padding=12)
        self.img_ai_tab = ttk.Frame(self.notebook, padding=12)
        self.img_log_tab = ttk.Frame(self.notebook, padding=12)

        self.notebook.add(self.ai_tab, text=" 動画 AI ")
        self.notebook.add(self.log_tab, text=" 動画 Log ")
        self.notebook.add(self.img_ai_tab, text=" 画像 AI ")
        self.notebook.add(self.img_log_tab, text=" 画像 Log ")

        self._build_ai_tab(self.ai_tab)
        self._build_log_tab(self.log_tab)
        self._build_img_ai_tab(self.img_ai_tab)
        self._build_img_log_tab(self.img_log_tab)

        controls = ttk.Frame(left)
        controls.grid(row=1, column=0, sticky="ew", pady=(16, 12))
        self.add_queue_button = ttk.Button(controls, text="キューに追加", command=self._enqueue_current)
        self.add_queue_button.pack(side="left")
        self.start_button = ttk.Button(controls, text="キュー開始", command=self._start)
        self.start_button.pack(side="left", padx=(8, 0))
        self.stop_button = ttk.Button(controls, text="現在の処理を停止", command=self._stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.open_output_button = ttk.Button(controls, text="出力を開く", command=self._open_output)
        self.open_output_button.pack(side="left", padx=(8, 0))
        self.open_folder_button = ttk.Button(controls, text="フォルダを開く", command=self._open_folder)
        self.open_folder_button.pack(side="left", padx=(8, 0))

        status_frame = ttk.Frame(left)
        status_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Helvetica", 11, "bold")).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(anchor="w", pady=(4, 8))
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")

        ttk.Label(left, text="ログ").grid(row=3, column=0, sticky="w", pady=(16, 6))
        self.log = tk.Text(left, height=16, wrap="word", state="disabled")
        self.log.grid(row=4, column=0, sticky="nsew")

        queue_controls = ttk.Frame(right)
        queue_controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.remove_queue_button = ttk.Button(queue_controls, text="選択項目を削除", command=self._remove_selected_job)
        self.remove_queue_button.pack(side="left")
        self.clear_queue_button = ttk.Button(queue_controls, text="キューをクリア", command=self._clear_queue)
        self.clear_queue_button.pack(side="left", padx=(8, 0))

        ttk.Label(right, text="キュー (Queue)").grid(row=1, column=0, sticky="w", pady=(8, 6))
        self.queue_view = ttk.Treeview(right, columns=("status", "input", "output"), show="headings", height=14)
        self.queue_view.heading("status", text="ステータス")
        self.queue_view.heading("input", text="入力ファイル")
        self.queue_view.heading("output", text="出力ファイル")
        self.queue_view.column("status", width=90, anchor="w")
        self.queue_view.column("input", width=170, anchor="w")
        self.queue_view.column("output", width=220, anchor="w")
        self.queue_view.grid(row=2, column=0, sticky="nsew")

        self.input_var.trace_add("write", self._sync_output_path)
        self.log_input_var.trace_add("write", self._sync_log_output_path)
        self.img_input_var.trace_add("write", self._sync_img_output_path)
        self.img_log_input_var.trace_add("write", self._sync_img_log_output_path)
        self.img_format_var.trace_add("write", self._sync_img_output_path)
        self.img_log_format_var.trace_add("write", self._sync_img_log_output_path)
        self.encoder_var.trace_add("write", self._sync_encoder_ui)
        self.encoder_var.trace_add("write", self._sync_output_path)
        self.log_encoder_var.trace_add("write", self._sync_log_encoder_ui)
        self.log_encoder_var.trace_add("write", self._sync_log_output_path)
        self.exr_delivery_var.trace_add("write", self._sync_encoder_ui)
        self.exr_delivery_var.trace_add("write", self._sync_output_path)
        self.log_exr_delivery_var.trace_add("write", self._sync_log_encoder_ui)
        self.log_exr_delivery_var.trace_add("write", self._sync_log_output_path)
        self.x265_mode_var.trace_add("write", self._sync_mode_hint)
        self.backend_var.trace_add("write", self._sync_mode_hint)
        self.backend_var.trace_add("write", self._refresh_model_choices)
        self.preset_var.trace_add("write", self._sync_preset_description)
        self.preset_var.trace_add("write", self._on_preset_change)
        self.model_name_var.trace_add("write", self._sync_selected_model)
        self.model_path_var.trace_add("write", self._sync_model_controls)
        self._sync_encoder_ui()
        self._sync_log_encoder_ui()
        self._sync_ai_strength_label()
        self._refresh_model_choices()
        self._sync_selected_model()
        self._sync_model_controls()
        self._sync_ai_strength_label()
        self._sync_saturation_label()
        self._sync_mode_hint()
        self._sync_preset_description()
        self._sync_img_output_path()
        self._sync_img_log_output_path()
        self._refresh_job_list()

    def _build_ai_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        self.input_entry = self._add_path_row(tab, 0, "入力 (Video)", self.input_var, self._browse_input)
        self.output_entry = self._add_path_row(tab, 1, "出力 (HDR)", self.output_var, self._browse_output)
        self.preset_combo = self._add_combo_row(tab, 2, "プリセット", self.preset_var, list(get_presets().keys()))
        
        # プリセット説明用
        self.preset_desc_label = ttk.Label(tab, text="", font=("Helvetica", 9), foreground="#555", wraplength=400)
        self.preset_desc_label.grid(row=3, column=1, sticky="w", pady=(0, 8), padx=2)

        self.encoder_combo = self._add_format_row(tab, 4, "出力形式", self.encoder_var, list(self.encoder_options.values()), self.mode_hint_var)
        self.encoder_combo.grid_configure(sticky="ew")
        self.x265_combo = self._add_combo_row(tab, 5, "速度/品質", self.x265_mode_var, list(X265_MODE_OPTIONS.values()))
        self.x265_row_widgets = tab.grid_slaves(row=5)
        self.exr_delivery_row, self.exr_delivery_combo = self._add_exr_delivery_row(tab, 5, self.exr_delivery_var)
        self.backend_combo = self._add_combo_row(tab, 6, "バックエンド", self.backend_var, list(self.backend_options.values()))
        self.model_combo = self._add_combo_row(
            tab, 7, "AI モデル", self.model_name_var,
            [model_display_name(path) for path in self.filtered_models] or ["互換性のあるモデルがありません"],
        )
        ttk.Button(tab, text="更新", command=self._refresh_available_models).grid(row=7, column=2, padx=(8, 0))
        ttk.Label(tab, text="AI 強度").grid(row=8, column=0, sticky="w", pady=6, padx=(0, 12))
        ai_slider_row = ttk.Frame(tab)
        ai_slider_row.grid(row=8, column=1, sticky="ew", pady=6)
        ai_slider_row.columnconfigure(0, weight=1)
        self.ai_strength_scale = ttk.Scale(
            ai_slider_row, from_=0.0, to=0.8, orient="horizontal",
            variable=self.ai_strength_var, command=self._sync_ai_strength_label,
        )
        self.ai_strength_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(ai_slider_row, textvariable=self.ai_strength_label_var, width=5).grid(row=0, column=1, padx=(8, 0))
        
        ttk.Label(tab, text="彩度 (Saturation)").grid(row=9, column=0, sticky="w", pady=6, padx=(0, 12))
        sat_slider_row = ttk.Frame(tab)
        sat_slider_row.grid(row=9, column=1, sticky="ew", pady=6)
        sat_slider_row.columnconfigure(0, weight=1)
        self.saturation_scale = ttk.Scale(
            sat_slider_row, from_=0.8, to=1.5, orient="horizontal",
            variable=self.saturation_var, command=self._sync_saturation_label,
        )
        self.saturation_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(sat_slider_row, textvariable=self.saturation_label_var, width=5).grid(row=0, column=1, padx=(8, 0))

        # AI強度説明用
        ttk.Label(tab, text="AIがどれだけ積極的に輝度を拡張するかを調整します。値を上げるとより眩しいHDRになりますが、上げすぎると不自然な階調になる場合があります。", 
                  font=("Helvetica", 9), foreground="#555", wraplength=400).grid(row=10, column=1, sticky="w", pady=(0, 4))


    def _build_log_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="動画の階調を維持したままBT.2020/PQコンテナに格納します。", 
                  font=("Helvetica", 10), foreground="#666").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))
        self.log_input_entry = self._add_path_row(tab, 1, "入力 (Log)", self.log_input_var, self._browse_log_input)
        self.log_output_entry = self._add_path_row(tab, 2, "出力 (HDR)", self.log_output_var, self._browse_log_output)
        log_options = [label for key, label in self.encoder_options.items()
                       if key not in {"hevc_hlg", "openexr_acescg_1_3", "openexr_acescg_2_0"}]
        self.log_encoder_combo = self._add_format_row(
            tab, 3, "出力形式", self.log_encoder_var, log_options, self.log_mode_hint_var,
        )
        self.log_x265_combo = self._add_combo_row(tab, 4, "速度/品質", self.log_x265_mode_var, list(X265_MODE_OPTIONS.values()))
        self.log_x265_row_widgets = tab.grid_slaves(row=4)
        self.log_exr_delivery_row, self.log_exr_delivery_combo = self._add_exr_delivery_row(tab, 4, self.log_exr_delivery_var)

    def _build_img_ai_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        self.img_input_entry = self._add_path_row(tab, 0, "入力 (Image)", self.img_input_var, self._browse_img_input)
        self.img_output_entry = self._add_path_row(tab, 1, "出力 (HDR)", self.img_output_var, self._browse_img_output)
        self._add_format_row(tab, 2, "出力形式", self.img_format_var, list(self.img_format_options.values()), self.img_mode_hint_var)
        self._add_combo_row(tab, 3, "プリセット", self.preset_var, list(get_presets().keys()))
        
        # プリセット説明用（動画タブと共有変数）
        self.img_preset_desc_label = ttk.Label(tab, text="", font=("Helvetica", 9), foreground="#555", wraplength=400)
        self.img_preset_desc_label.grid(row=4, column=1, sticky="w", pady=(0, 8), padx=2)

        self._add_combo_row(tab, 5, "バックエンド", self.backend_var, list(self.backend_options.values()))
        self.img_model_combo = self._add_combo_row(
            tab, 6, "AI モデル", self.model_name_var,
            [model_display_name(path) for path in self.filtered_models] or ["互換性のあるモデルがありません"],
        )
        ttk.Label(tab, text="AI 強度").grid(row=7, column=0, sticky="w", pady=6, padx=(0, 12))
        slider_row = ttk.Frame(tab)
        slider_row.grid(row=7, column=1, sticky="ew", pady=6)
        slider_row.columnconfigure(0, weight=1)
        ttk.Scale(slider_row, from_=0.0, to=0.8, orient="horizontal",
                  variable=self.ai_strength_var, command=self._sync_ai_strength_label).grid(row=0, column=0, sticky="ew")
        ttk.Label(slider_row, textvariable=self.ai_strength_label_var, width=5).grid(row=0, column=1, padx=(8, 0))
        
        ttk.Label(tab, text="彩度 (Saturation)").grid(row=8, column=0, sticky="w", pady=6, padx=(0, 12))
        img_sat_slider_row = ttk.Frame(tab)
        img_sat_slider_row.grid(row=8, column=1, sticky="ew", pady=6)
        img_sat_slider_row.columnconfigure(0, weight=1)
        ttk.Scale(img_sat_slider_row, from_=0.8, to=1.5, orient="horizontal",
                  variable=self.saturation_var, command=self._sync_saturation_label).grid(row=0, column=0, sticky="ew")
        ttk.Label(img_sat_slider_row, textvariable=self.saturation_label_var, width=5).grid(row=0, column=1, padx=(8, 0))

        # AI強度説明用
        ttk.Label(tab, text="AIがどれだけ積極的に輝度を拡張するかを調整します。値を上げるとより眩しいHDRになりますが、上げすぎると不自然な階調になる場合があります。", 
                  font=("Helvetica", 9), foreground="#555", wraplength=400).grid(row=9, column=1, sticky="w", pady=(0, 8))

    def _build_img_log_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(1, weight=1)
        ttk.Label(tab, text="静止画のLog素材をBT.2020/PQの静止画コンテナに変換します。", 
                  font=("Helvetica", 10), foreground="#666").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 16))
        self.img_log_input_entry = self._add_path_row(tab, 1, "入力 (Image)", self.img_log_input_var, self._browse_img_log_input)
        self.img_log_output_entry = self._add_path_row(tab, 2, "出力 (HDR)", self.img_log_output_var, self._browse_img_log_output)
        self._add_format_row(tab, 3, "出力形式", self.img_log_format_var, list(self.img_format_options.values()), self.img_log_mode_hint_var)


    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command: object,
    ) -> ttk.Entry:
        path_label = ttk.Label(parent, text=label)
        path_label.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
        self._path_labels[str(variable)] = (path_label, label)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="参照", command=browse_command).grid(row=row, column=2, padx=(8, 0))
        return entry

    def _add_combo_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: list[str],
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="w", pady=6)
        return combo

    def _add_format_row(self, parent, row, label, variable, values, description):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", pady=6, padx=(0, 12))
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=6)
        frame.columnconfigure(0, weight=1)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.grid(row=0, column=0, sticky="ew")
        hint = ttk.Label(frame, textvariable=description, wraplength=400, foreground="#555")
        hint.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        frame.bind("<Configure>", lambda event: hint.configure(wraplength=max(1, event.width)))
        return combo

    def _add_exr_delivery_row(self, parent, row, variable):
        label = ttk.Label(parent, text="保存方法")
        label.grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=list(EXR_DELIVERY_OPTIONS.values()), state="readonly", width=30)
        combo.grid(row=row, column=1, sticky="ew", pady=6)
        return (label, combo), combo

    def _selected_exr_delivery(self, log_mode=False):
        variable = self.log_exr_delivery_var if log_mode else self.exr_delivery_var
        return next(key for key, label in EXR_DELIVERY_OPTIONS.items() if label == variable.get())

    def _sync_exr_delivery_ui(self, log_mode=False):
        row = self.log_exr_delivery_row if log_mode else self.exr_delivery_row
        combo = self.log_exr_delivery_combo if log_mode else self.exr_delivery_combo
        encoder = self._selected_log_encoder() if log_mode else self._selected_encoder()
        quality_widgets = self.log_x265_row_widgets if log_mode else self.x265_row_widgets
        exr = encoder.startswith("openexr")
        for widget in row:
            widget.grid() if exr else widget.grid_remove()
        for widget in quality_widgets:
            widget.grid_remove() if exr else widget.grid()
        running = self.state in {AppState.RUNNING, AppState.CANCELLING}
        combo.configure(state="disabled" if running else "readonly")

    def _sync_path(self, key, input_var, output_var, extension):
        source = input_var.get().strip()
        current = output_var.get().strip()
        batch = getattr(self, "_input_batches", {}).get(key)
        if batch and source == batch[0]:
            return  # A batch output is a folder, unaffected by format changes.
        if batch:
            del self._input_batches[key]
            widget, label = self._path_labels[str(output_var)]
            widget.configure(text=label)
            current = str(Path(current) / Path(build_output_path(source, extension=extension)).name) if current and source else ""
            output_var.set(current)
        previous_source, previous_auto = self._output_path_state.get(key, ("", ""))
        automatic = not current or current == previous_auto
        if automatic and source and (source != previous_source or not current):
            updated = build_output_path(source, extension=extension)
        elif current:
            updated = str(Path(current).with_suffix(extension))
        else:
            updated = ""
        self._output_path_state[key] = (source, updated if automatic else "")
        if updated != current:
            output_var.set(updated)

    def _sync_output_path(self, *_: object) -> None:
        self._sync_path("video_ai", self.input_var, self.output_var, output_extensions(self._selected_encoder(), self._selected_exr_delivery())[0])

    def _sync_log_output_path(self, *_: object) -> None:
        self._sync_path("video_log", self.log_input_var, self.log_output_var, output_extensions(self._selected_log_encoder(), self._selected_exr_delivery(log_mode=True))[0])

    def _sync_preset_description(self, *_: object) -> None:
        descriptions = get_preset_descriptions()
        desc = descriptions.get(self.preset_var.get(), "")
        if hasattr(self, "preset_desc_label"):
            self.preset_desc_label.config(text=desc)
        if hasattr(self, "img_preset_desc_label"):
            self.img_preset_desc_label.config(text=desc)

    def _selected_img_format(self, log_mode: bool = False) -> str:
        var = self.img_log_format_var if log_mode else self.img_format_var
        for ext, label in self.img_format_options.items():
            if var.get() == label:
                return ext
        return ".tif"

    def _sync_img_output_path(self, *_: object) -> None:
        ext = self._selected_img_format()
        self._sync_path("image_ai", self.img_input_var, self.img_output_var, ext)
        self.img_mode_hint_var.set(describe_image_format(ext))

    def _sync_img_log_output_path(self, *_: object) -> None:
        ext = self._selected_img_format(log_mode=True)
        self._sync_path("image_log", self.img_log_input_var, self.img_log_output_var, ext)
        self.img_log_mode_hint_var.set(describe_image_format(ext))

    def _sync_encoder_ui(self, *_: object) -> None:
        self._sync_exr_delivery_ui()
        if self._selected_encoder() in {"libx265", "hevc_hlg"}:
            self.x265_combo.configure(state="readonly")
        else:
            self.x265_combo.configure(state="disabled")
        self._sync_mode_hint()

    def _sync_log_encoder_ui(self, *_: object) -> None:
        self._sync_exr_delivery_ui(log_mode=True)
        self.log_mode_hint_var.set(describe_mode_hint(self._selected_log_encoder(), self._selected_log_x265_mode(), "", "", "", self._selected_exr_delivery(log_mode=True)))
        if self._selected_log_encoder() == "libx265":
            self.log_x265_combo.configure(state="readonly")
        else:
            self.log_x265_combo.configure(state="disabled")

    def _sync_mode_hint(self, *_: object) -> None:
        self.mode_hint_var.set(
            describe_mode_hint(
                self._selected_encoder(),
                self._selected_x265_mode(),
                self._selected_backend(),
                self.preset_var.get(),
                self.model_path_var.get(),
                self._selected_exr_delivery(),
            )
        )

    def _sync_selected_model(self, *_: object) -> None:
        selected_name = self.model_name_var.get().strip()
        selected_path = next((path for path in self.filtered_models if model_display_name(path) == selected_name), None)
        self.model_path_var.set(str(selected_path) if selected_path else "")
        self._sync_mode_hint()

    def _sync_ai_strength_label(self, *_: object) -> None:
        self.ai_strength_label_var.set(format_ai_strength(self.ai_strength_var.get()))

    def _sync_saturation_label(self, *_: object) -> None:
        self.saturation_label_var.set(f"{self.saturation_var.get():.2f}x")

    def _on_preset_change(self, *_: object) -> None:
        presets = get_presets()
        preset_name = self.preset_var.get()
        if preset_name in presets:
            config = presets[preset_name]
            self.ai_strength_var.set(config.ai_strength)
            self.saturation_var.set(config.saturation)
            self._sync_ai_strength_label()
            self._sync_saturation_label()
        self._sync_mode_hint()

    def _sync_model_controls(self, *_: object) -> None:
        has_model = bool(self.model_path_var.get().strip())
        running = self.state in {AppState.RUNNING, AppState.CANCELLING}
        scale_state = "normal" if has_model and not running else "disabled"
        self.ai_strength_scale.configure(state=scale_state)
        self._sync_ai_strength_label()
        self._sync_saturation_label()
        self._sync_mode_hint()
        combo_state = "disabled" if running else "readonly"
        self.model_combo.configure(state=combo_state if self.filtered_models else "disabled")
        self.img_model_combo.configure(state=combo_state if self.filtered_models else "disabled")

    def _selected_encoder(self) -> str:
        for key, label in self.encoder_options.items():
            if self.encoder_var.get() == label:
                return key
        return "libx265"

    def _selected_x265_mode(self) -> str:
        for key, label in X265_MODE_OPTIONS.items():
            if self.x265_mode_var.get() == label:
                return key
        return "balanced"

    def _selected_log_encoder(self) -> str:
        for key, label in self.encoder_options.items():
            if self.log_encoder_var.get() == label:
                return key
        return "libx265"

    def _selected_log_x265_mode(self) -> str:
        for key, label in X265_MODE_OPTIONS.items():
            if self.log_x265_mode_var.get() == label:
                return key
        return "balanced"

    def _selected_backend(self) -> str:
        for key, label in self.backend_options.items():
            if self.backend_var.get() == label:
                return key
        return "auto"

    def _browse_inputs(self, key, input_var, output_var, image=False):
        if self.state in {AppState.RUNNING, AppState.CANCELLING}:
            return
        pattern = "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.heic *.avif" if image else "*.mp4 *.mov *.mkv *.m2ts *.mts"
        paths = filedialog.askopenfilenames(title="入力ファイルを選択（複数選択可）", filetypes=[("画像" if image else "動画", pattern), ("すべてのファイル", "*.*")])
        if not paths:
            return
        if len(paths) == 1:
            input_var.set(paths[0])
            return
        was_batch = key in self._input_batches
        current = output_var.get().strip()
        folder = current if was_batch else str(Path(current).parent) if current else str(Path(paths[0]).parent)
        display = f"{len(paths)}件: " + "; ".join(paths)
        self._input_batches[key] = (display, tuple(paths))
        input_var.set(display)
        output_var.set(folder)
        self._path_labels[str(output_var)][0].configure(text="保存先フォルダー")

    def _browse_input(self) -> None:
        self._browse_inputs("video_ai", self.input_var, self.output_var)

    def _browse_selected_output(self, variable, extension, key):
        if self.state in {AppState.RUNNING, AppState.CANCELLING}:
            return
        current = Path(variable.get()) if variable.get().strip() else None
        if key in self._input_batches:
            path = filedialog.askdirectory(title="保存先フォルダーを選択", initialdir=str(current) if current else None)
            if path:
                variable.set(path)
            return
        path = filedialog.asksaveasfilename(
            title="出力を保存", defaultextension=extension,
            filetypes=[(extension.lstrip(".").upper(), "*" + extension)],
            initialdir=str(current.parent) if current else None,
            initialfile=current.name if current else None,
        )
        if path:
            variable.set(str(Path(path).with_suffix(extension)))
            # A selected save location is manual even if it matches the default.
            self._output_path_state[key] = (self._output_path_state.get(key, ("", ""))[0], "")

    def _browse_output(self) -> None:
        self._browse_selected_output(self.output_var, output_extensions(self._selected_encoder(), self._selected_exr_delivery())[0], "video_ai")

    def _browse_log_input(self) -> None:
        self._browse_inputs("video_log", self.log_input_var, self.log_output_var, image=False)

    def _browse_log_output(self) -> None:
        self._browse_selected_output(self.log_output_var, output_extensions(self._selected_log_encoder(), self._selected_exr_delivery(log_mode=True))[0], "video_log")

    def _browse_img_input(self) -> None:
        self._browse_inputs("image_ai", self.img_input_var, self.img_output_var, image=True)

    def _browse_img_output(self) -> None:
        self._browse_selected_output(self.img_output_var, self._selected_img_format(), "image_ai")

    def _browse_img_log_input(self) -> None:
        self._browse_inputs("image_log", self.img_log_input_var, self.img_log_output_var, image=True)

    def _browse_img_log_output(self) -> None:
        self._browse_selected_output(self.img_log_output_var, self._selected_img_format(log_mode=True), "image_log")

    def _refresh_available_models(self) -> None:
        self.available_models = list_available_models()
        self._refresh_model_choices()

    def _refresh_model_choices(self, *_: object) -> None:
        self.filtered_models = filter_models_for_backend(self.available_models, self._selected_backend(), self.system_name)
        values = [model_display_name(path) for path in self.filtered_models] or ["No compatible models"]
        self.model_combo.configure(values=values)
        self.img_model_combo.configure(values=values)
        if self.filtered_models:
            current = self.model_name_var.get().strip()
            if current not in values:
                self.model_name_var.set(values[0])
            else:
                self._sync_selected_model()
        else:
            self.model_name_var.set("No compatible models")
            self.model_path_var.set("")
        self._sync_model_controls()

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _build_request(self) -> ConversionRequest | LogConversionRequest | ImageConversionRequest | ImageLogConversionRequest:
        tab_idx = self.notebook.index("current")
        if tab_idx == 0:  # Video AI
            return ConversionRequest(
                input_path=self.input_var.get().strip(),
                output_path=self.output_var.get().strip(),
                preset=self.preset_var.get(),
                encoder=self._selected_encoder(),
                exr_delivery=self._selected_exr_delivery(),
                x265_mode=self._selected_x265_mode(),
                backend=self._selected_backend(),
                model_path=self.model_path_var.get().strip() or None,
                ai_strength=self.ai_strength_var.get() if self.model_path_var.get().strip() else None,
                device="auto",
                fallback_to_x265_on_hardware_error=True,
                keep_partial_output_on_cancel=False,
                saturation=self.saturation_var.get(),
                hdr_guidance=self.hdr_guidance_var.get(),
                luminance_guidance_strength=self.luminance_guidance_var.get(),
                reconstruction_strength=self.reconstruction_strength_var.get(),
            )
        elif tab_idx == 1:  # Video Log
            return LogConversionRequest(
                input_path=self.log_input_var.get().strip(),
                output_path=self.log_output_var.get().strip(),
                encoder=self._selected_log_encoder(),
                exr_delivery=self._selected_exr_delivery(log_mode=True),
                x265_mode=self._selected_log_x265_mode(),
                keep_partial_output_on_cancel=False,
            )
        elif tab_idx == 2:  # Image AI
            return ImageConversionRequest(
                input_path=self.img_input_var.get().strip(),
                output_path=self.img_output_var.get().strip(),
                preset=self.preset_var.get(),
                backend=self._selected_backend(),
                model_path=self.model_path_var.get().strip() or None,
                ai_strength=self.ai_strength_var.get() if self.model_path_var.get().strip() else None,
                device="auto",
                saturation=self.saturation_var.get(),
            )
        else:  # Image Log
            return ImageLogConversionRequest(
                input_path=self.img_log_input_var.get().strip(),
                output_path=self.img_log_output_var.get().strip(),
            )

    def _validate_request(self, request: ConversionRequest | LogConversionRequest | ImageConversionRequest | ImageLogConversionRequest) -> None:
        if not request.input_path:
            raise ValueError("入力パスが指定されていません。")
        if not request.output_path:
            raise ValueError("出力パスが指定されていません。")
        validate_export_request(request)
        if isinstance(request, (ImageConversionRequest, ImageLogConversionRequest)):
            selected = self._selected_img_format(log_mode=isinstance(request, ImageLogConversionRequest))
            actual = Path(request.output_path).suffix.lower()
            if actual != selected and (selected, actual) not in {(".tif", ".tiff"), (".jpg", ".jpeg")}:
                raise ValueError(f"選択した画像形式には {selected} の拡張子が必要です。")
        if isinstance(request, (ConversionRequest, ImageConversionRequest)):
            validate_request(request)

    def _make_job_label(self, request: ConversionRequest | LogConversionRequest) -> tuple[str, str, str]:
        return ("pending", Path(request.input_path).name, Path(request.output_path).name)

    def _refresh_job_list(self) -> None:
        self.queue_view.delete(*self.queue_view.get_children())
        for index, job in enumerate(self.queue_jobs):
            self.queue_view.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    STATUS_LABELS.get(job.status, job.status.upper()),
                    Path(job.request.input_path).name,
                    Path(job.request.output_path).name,
                ),
            )

    def _set_job_status(self, index: int | None, status: str) -> None:
        if index is None:
            return
        if 0 <= index < len(self.queue_jobs):
            self.queue_jobs[index].status = status
            self._refresh_job_list()

    def _enqueue_request(self, request: ConversionRequest | LogConversionRequest) -> None:
        self._validate_request(request)
        self.queue_jobs.append(QueueJob(request=request))
        self.last_output_path = request.output_path
        self._refresh_job_list()
        mode_text = "AI (動画)" if isinstance(request, ConversionRequest) else \
                    "Log (動画)" if isinstance(request, LogConversionRequest) else \
                    "AI (画像)" if isinstance(request, ImageConversionRequest) else "Log (画像)"
        self._log(f"キュー追加 ({mode_text}): {Path(request.input_path).name}")

    def _enqueue_current(self) -> None:
        try:
            self._enqueue_inputs()
        except ValueError as exc:
            messagebox.showerror("キューに追加できません", str(exc))

    def _enqueue_inputs(self):
        template = self._build_request()
        index = self.notebook.index("current")
        key = ("video_ai", "video_log", "image_ai", "image_log")[index]
        batch = getattr(self, "_input_batches", {}).get(key)
        if not batch:
            self._enqueue_request(template)
            return
        if not template.output_path:
            raise ValueError("保存先フォルダーを指定してください。")
        folder = Path(template.output_path)
        if folder.exists() and not folder.is_dir():
            raise ValueError("複数ファイルの保存先にはフォルダーを指定してください。")
        extension = self._selected_img_format(log_mode=index == 3) if index >= 2 else output_extensions(template.encoder, template.exr_delivery)[0]
        requests = []
        used = {Path(job.request.output_path).resolve() for job in self.queue_jobs}
        for source in batch[1]:
            name = Path(build_output_path(source, extension=extension)).name
            destination = folder / name
            number = 2
            while destination.resolve() in used or destination.exists():
                destination = folder / f"{Path(name).stem}_{number}{extension}"
                number += 1
            request = replace(template, input_path=source, output_path=str(destination))
            self._validate_request(request)
            used.add(destination.resolve())
            requests.append(request)
        for request in requests:
            self._enqueue_request(request)

    def _selected_job_indices(self) -> list[int]:
        return sorted((int(item_id) for item_id in self.queue_view.selection()), reverse=True)

    def _remove_selected_job(self) -> None:
        if self.state in {AppState.RUNNING, AppState.CANCELLING}:
            return
        removed = False
        for index in self._selected_job_indices():
            if 0 <= index < len(self.queue_jobs):
                del self.queue_jobs[index]
                removed = True
        if removed:
            self._refresh_job_list()
            self._log("選択されたキュー項目を削除しました")

    def _clear_queue(self) -> None:
        if self.state in {AppState.RUNNING, AppState.CANCELLING}:
            return
        self.queue_jobs.clear()
        self.current_job_index = None
        self._refresh_job_list()
        self._log("キューをクリアしました")

    def _next_pending_job_index(self) -> int | None:
        for index, job in enumerate(self.queue_jobs):
            if job.status == "queued":
                return index
        return None

    def _start_job(self, index: int) -> None:
        request = self.queue_jobs[index].request
        self.current_job_index = index
        self._set_job_status(index, "starting")
        self.cancel_token = CancelToken()
        self.last_output_path = request.output_path
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(10)
        self.status_var.set("開始中")
        if isinstance(request, ConversionRequest):
            self.progress_var.set("AI モデルの読み込み中")
        else:
            self.progress_var.set("変換準備中...")
            
        self._log(f"変換を開始します: {Path(request.input_path).name}")
        self._set_state(AppState.RUNNING)

        callbacks = ConversionCallbacks(
            on_status=lambda message: self.event_queue.put(("detail_status", message)),
            on_progress=lambda processed, total, fps: self.event_queue.put(("progress", (processed, total, fps))),
            on_complete=lambda result: self.event_queue.put(("complete", result)),
            on_error=lambda message: self.event_queue.put(("error", message)),
        )

        def worker() -> None:
            try:
                if isinstance(request, ConversionRequest):
                    run_conversion(request, callbacks=callbacks, cancel_token=self.cancel_token)
                elif isinstance(request, LogConversionRequest):
                    run_log_conversion(request, callbacks=callbacks, cancel_token=self.cancel_token)
                elif isinstance(request, ImageConversionRequest):
                    run_image_conversion(request, callbacks=callbacks, cancel_token=self.cancel_token)
                elif isinstance(request, ImageLogConversionRequest):
                    run_image_log_conversion(request, callbacks=callbacks, cancel_token=self.cancel_token)
            except Exception as exc:
                self.event_queue.put(("failed", str(exc)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _start(self) -> None:
        if self.state in {AppState.RUNNING, AppState.CANCELLING}:
            return
        if not self.queue_jobs:
            try:
                self._enqueue_inputs()
            except ValueError as exc:
                messagebox.showerror("開始できません", str(exc))
                return
        next_index = self._next_pending_job_index()
        if next_index is None:
            return
        self._start_job(next_index)

    def _stop(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.cancel()
            self._set_job_status(self.current_job_index, "cancelling")
            self.status_var.set("キャンセル中")
            self.progress_var.set("現在の処理を停止しています")
            self._set_state(AppState.CANCELLING)
            self._log("停止が要求されました")

    def _open_output(self) -> None:
        if not self.last_output_path:
            return
        output = Path(self.last_output_path)
        open_path(str(output.parent) if is_exr_sequence(self.last_output_path) else self.last_output_path)

    def _open_folder(self) -> None:
        if not self.last_output_path:
            return
        open_path(str(Path(self.last_output_path).parent))

    def _set_state(self, state: str) -> None:
        self.state = state
        running = state in {AppState.RUNNING, AppState.CANCELLING}
        idle_like = state in {AppState.IDLE, AppState.COMPLETED, AppState.FAILED, AppState.CANCELLED}
        field_state = "disabled" if running else "normal"
        combo_state = "disabled" if running else "readonly"
        self.input_entry.configure(state=field_state)
        self.output_entry.configure(state=field_state)
        self.log_input_entry.configure(state=field_state)
        self.log_output_entry.configure(state=field_state)
        self.img_input_entry.configure(state=field_state)
        self.img_output_entry.configure(state=field_state)
        self.img_log_input_entry.configure(state=field_state)
        self.img_log_output_entry.configure(state=field_state)
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.add_queue_button.configure(state="disabled" if running else "normal")
        self.remove_queue_button.configure(state="disabled" if running else "normal")
        self.clear_queue_button.configure(state="disabled" if running else "normal")
        self.open_output_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.open_folder_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.preset_combo.configure(state=combo_state)
        self.encoder_combo.configure(state=combo_state)
        self.log_encoder_combo.configure(state=combo_state)
        self.backend_combo.configure(state=combo_state)
        self._sync_encoder_ui()
        self._sync_log_encoder_ui()
        self._sync_model_controls()

    def _finish_current_job(self) -> None:
        self.cancel_token = None
        self.worker = None
        self.current_job_index = None

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "detail_status":
                if self.state != AppState.CANCELLING:
                    self.status_var.set(str(payload))
                self._log(str(payload))
            elif kind == "progress":
                processed, total, fps = payload
                if self.current_job_index is not None:
                    current_status = self.queue_jobs[self.current_job_index].status
                    if current_status == "starting":
                        self._set_job_status(self.current_job_index, "running")
                if total:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100, value=(processed / total) * 100)
                if self.state == AppState.RUNNING:
                    self.status_var.set("変換中")
                fps_text = f"{fps:.1f} fps" if fps else "n/a"
                self.progress_var.set(f"{processed}/{total or '?'} フレーム, {fps_text}")
            elif kind == "complete":
                result = payload
                self.progress.stop()
                self.progress.configure(value=100 if not result.cancelled else 0)
                if self.current_job_index is not None:
                    self._set_job_status(self.current_job_index, "cancelled" if result.cancelled else "completed")
                    self.last_output_path = self.queue_jobs[self.current_job_index].request.output_path
                if result.cancelled:
                    self.status_var.set("キャンセル済")
                    self.progress_var.set(f"{result.processed_frames} フレームで停止しました。今回の未完成出力は破棄しました")
                    self._log("変換をキャンセルし、今回の未完成出力を破棄しました")
                    self._finish_current_job()
                    self._set_state(AppState.CANCELLED)
                else:
                    self.status_var.set("完了")
                    if Path(result.output_path).suffix.lower() == ".zip":
                        self.progress_var.set(f"EXR画像 {result.processed_frames} 枚をZIPに保存しました")
                        self._log(f"EXR ZIPの保存先: {result.output_path}")
                    else:
                        self.progress_var.set(f"{result.processed_frames} フレーム書き出しました")
                    self._log(f"完了: {result.output_path}")
                    self._finish_current_job()
                    next_index = self._next_pending_job_index()
                    if next_index is not None:
                        self._set_state(AppState.IDLE)
                        self._start_job(next_index)
                    else:
                        self._set_state(AppState.COMPLETED)
            elif kind == "error":
                self._log(str(payload))
            elif kind == "failed":
                self.progress.stop()
                self.progress.configure(value=0)
                self._set_job_status(self.current_job_index, "failed")
                self.status_var.set("失敗")
                self.progress_var.set("変換に失敗しました")
                self._log(str(payload))
                self._finish_current_job()
                next_index = self._next_pending_job_index()
                if next_index is not None:
                    self._set_state(AppState.IDLE)
                    self._start_job(next_index)
                else:
                    self._set_state(AppState.FAILED)
                    messagebox.showerror("変換に失敗しました", str(payload))
        self.root.after(100, self._drain_events)


def main() -> int:
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    app._log("準備完了")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
