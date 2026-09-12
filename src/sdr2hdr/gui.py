from __future__ import annotations

import os
import math
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

from sdr2hdr.gui_output import PreviewOutputs, export_preview

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
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueJob:
    request: ConversionRequest | LogConversionRequest | ImageConversionRequest | ImageLogConversionRequest
    status: str = "queued"
    error: str = ""
    preview_path: str | None = None


STATUS_LABELS = {
    "queued": "待機中",
    "starting": "開始中",
    "running": "実行中",
    "cancelling": "キャンセル中",
    "completed": "未書き出し",
    "exported": "書き出し済",
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
        self.root.title("SDR2HDR Pro")
        self.app_icon = tk.PhotoImage(master=root, file=str(Path(__file__).parent / "assets" / "app-icon.png"))
        self.root.iconphoto(True, self.app_icon)
        self.root.geometry("1760x1000")
        self.root.minsize(1680, 900)
        self.preview_outputs = PreviewOutputs()
        self._closing = False
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
        self.hdr_guidance_label_var = tk.StringVar(value="自動")
        self.peak_nits_var = tk.StringVar(value=f"{get_presets()['natural'].peak_nits:g}")
        self.luminance_guidance_var = tk.DoubleVar(value=0.70)
        self.reconstruction_strength_var = tk.DoubleVar(value=0.60)
        self.hdr_detail_controls = []
        self.hdr_detail_panels = []
        
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
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if self.system_name == "Darwin":
            self.root.createcommand("::tk::mac::Quit", self._close)
        self.root.update_idletasks()
        width = min(self.root.winfo_screenwidth(), max(1760, self.root.winfo_reqwidth()))
        height = min(self.root.winfo_screenheight() - 60, max(1000, self.root.winfo_reqheight()))
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self._set_state(AppState.IDLE)
        self.root.after(100, self._drain_events)
        if self.system_name == "Darwin":
            self.root.after_idle(self._set_macos_menu_name)

    def _set_macos_menu_name(self) -> None:
        from AppKit import NSApplication

        # Tk installs its initial menu before this idle callback runs.
        app_menu = NSApplication.sharedApplication().mainMenu().itemAtIndex_(0)
        app_menu.submenu().setTitle_("SDR2HDR Pro")
        app_menu.setTitle_("SDR2HDR Pro")

    def _close(self):
        if not self._closing:
            self._closing = True
            if self.cancel_token:
                self.cancel_token.cancel()
            self.compare_view.close()
            self.root.withdraw()
        # Wait asynchronously so no writer can recreate files after cleanup.
        if self.worker and self.worker.is_alive():
            self.root.after(50, self._close)
            return
        self.preview_outputs.close()
        self.root.destroy()

    def _build(self) -> None:
        from sdr2hdr.gui_style import apply_theme, ButtonSlot

        self.reduce_motion_var = tk.BooleanVar(value=False)
        apply_theme(self.root, self.reduce_motion_var)
        outer = ttk.Frame(self.root, padding=20, style="Shell.TFrame")
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3, minsize=670)
        outer.columnconfigure(1, weight=2, minsize=500)
        outer.columnconfigure(2, weight=2, minsize=440)
        outer.rowconfigure(1, weight=1)

        hero = tk.Frame(outer, bg="#efb4eb", highlightbackground="#14200e", highlightthickness=3)
        hero.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 16))
        tk.Label(hero, text="SDR2HDR Pro", font=("Impact", 46), bg="#efb4eb", fg="#14200e").pack(side="left", padx=22, pady=14)
        tk.Label(hero, text="いつもの映像を、HDRへ。\n素材と形式を選ぶ → 変換して確認 → 書き出す", justify="left", font=("Helvetica Neue", 12), bg="#efb4eb", fg="#14200e").pack(side="left", padx=20)
        self.hero_art = tk.PhotoImage(file=str(Path(__file__).parent / "assets" / "film-editor.png")).subsample(8)
        tk.Label(hero, image=self.hero_art, bg="#efb4eb", borderwidth=0).pack(side="right", padx=10, pady=8)

        left = ttk.Frame(outer, padding=(16, 12), style="Card.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="01  /  CONVERT", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.notebook = ttk.Notebook(left)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        self.ai_tab = ttk.Frame(self.notebook, padding=8)
        self.log_tab = ttk.Frame(self.notebook, padding=8)
        self.img_ai_tab = ttk.Frame(self.notebook, padding=8)
        self.img_log_tab = ttk.Frame(self.notebook, padding=8)
        for tab, title in ((self.ai_tab, "動画 AI"), (self.log_tab, "動画 Log"), (self.img_ai_tab, "画像 AI"), (self.img_log_tab, "画像 Log")):
            self.notebook.add(tab, text=title)
        self.ai_form = self._add_scrollable_ai_form(self.ai_tab)
        self.img_ai_form = self._add_scrollable_ai_form(self.img_ai_tab)
        self._build_ai_tab(self.ai_form)
        self._build_log_tab(self.log_tab)
        self._build_img_ai_tab(self.img_ai_form)
        self._build_img_log_tab(self.img_log_tab)
        # Group existing controls without changing their variables or processing.
        for tab, groups in ((self.ai_form, ((0, "素材と保存先 / プリセット"), (4, "出力形式"), (6, "AI設定"))),
                            (self.img_ai_form, ((0, "素材と保存先"), (2, "出力形式"), (3, "AI設定")))):
            for row, title in reversed(groups):
                for widget in tab.grid_slaves():
                    info = widget.grid_info()
                    if int(info["row"]) >= row:
                        widget.grid_configure(row=int(info["row"]) + 1)
                ttk.Label(tab, text=title, style="Group.TLabel").grid(row=row, column=0, columnspan=3, sticky="ew", pady=(3, 2))
            self._bind_ai_form_scroll(tab)
        self.feedback_var = tk.StringVar(value="設定した素材を一覧へ追加して、まとめて変換できます。")
        ttk.Label(left, textvariable=self.feedback_var, style="Muted.TLabel", wraplength=540).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        add_queue_slot = ButtonSlot(left, self.reduce_motion_var, text="変換待ちに追加", command=self._enqueue_current, style="Accent.TButton")
        self.add_queue_button = add_queue_slot.button
        add_queue_slot.grid(row=3, column=0, sticky="ew")

        right = ttk.Frame(outer, padding=16, style="Card.TFrame")
        right.grid(row=1, column=1, sticky="nsew")
        from sdr2hdr.compare_view import CompareView
        self.compare_view = CompareView(outer)
        self.compare_view.grid(row=1, column=2, sticky="nsew", padx=(14, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        ttk.Label(right, text="02  /  EXPORT", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(right, text="変換一覧 / 未書き出しの結果をまとめて保存", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 8))
        queue_frame = ttk.Frame(right)
        queue_frame.grid(row=2, column=0, sticky="nsew")
        queue_frame.configure(height=180)
        queue_frame.grid_propagate(False)
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(0, weight=1)
        self.queue_view = ttk.Treeview(queue_frame, columns=("status", "input", "output"), show="headings", height=4)
        for key, label, width in (("status", "状態", 105), ("input", "入力ファイル", 120), ("output", "出力ファイル", 140)):
            self.queue_view.heading(key, text=label)
            self.queue_view.column(key, width=width, minwidth=65, anchor="w")
        self.queue_view.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.queue_view.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.queue_view.configure(yscrollcommand=scroll.set)
        horizontal = ttk.Scrollbar(queue_frame, orient="horizontal", command=self.queue_view.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.queue_view.configure(xscrollcommand=horizontal.set)
        self.job_detail = tk.Text(queue_frame, height=3, width=20, wrap="char", font=("Helvetica Neue", 10), bg="#fffdf6", fg="#14200e", relief="flat", state="disabled")
        self.job_detail.grid(row=2, column=0, sticky="ew")
        detail_scroll = ttk.Scrollbar(queue_frame, command=self.job_detail.yview)
        detail_scroll.grid(row=2, column=1, sticky="ns")
        self.job_detail.configure(yscrollcommand=detail_scroll.set)
        self.detail_scroll = detail_scroll
        self.queue_view.bind("<<TreeviewSelect>>", self._show_job_detail)
        queue_controls = ttk.Frame(right)
        queue_controls.grid(row=3, column=0, sticky="ew", pady=10)
        self.remove_queue_button = ttk.Button(queue_controls, text="選択を削除", command=self._remove_selected_job)
        self.remove_queue_button.pack(side="left")
        self.clear_queue_button = ttk.Button(queue_controls, text="一覧をクリア", command=self._clear_queue)
        self.clear_queue_button.pack(side="right")

        status_frame = ttk.Frame(right, padding=10, style="Green.TFrame")
        status_frame.grid(row=4, column=0, sticky="ew", pady=(6, 12))
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel", wraplength=410)
        self.status_label.pack(anchor="w")
        tk.Label(status_frame, textvariable=self.progress_var, bg="#a9d994", fg="#14200e", font=("Helvetica Neue", 11), height=1, anchor="w", justify="left", wraplength=410).pack(fill="x", pady=(4, 4))
        self.result_var = tk.StringVar()
        tk.Label(status_frame, textvariable=self.result_var, bg="#a9d994", fg="#14200e", font=("Helvetica Neue", 11), height=2, anchor="w", justify="left", wraplength=410).pack(fill="x", pady=(0, 5))
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        conversion_controls = ttk.Frame(right)
        conversion_controls.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        conversion_controls.columnconfigure(0, weight=1, uniform="conversion")
        conversion_controls.columnconfigure(1, weight=1, uniform="conversion")
        start_slot = ButtonSlot(conversion_controls, self.reduce_motion_var, text="変換を開始", command=self._start, style="Primary.TButton")
        self.start_button = start_slot.button
        start_slot.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        stop_slot = ButtonSlot(conversion_controls, self.reduce_motion_var, text="現在の処理を停止", command=self._stop, style="Primary.TButton")
        self.stop_button = stop_slot.button
        stop_slot.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        export_slot = ButtonSlot(right, self.reduce_motion_var, text="変換済みを書き出す", command=self._export, style="Accent.TButton")
        self.export_button = export_slot.button
        export_slot.grid(row=6, column=0, sticky="ew", pady=(0, 4))
        output_controls = ttk.Frame(right)
        output_controls.grid(row=7, column=0, sticky="ew")
        self.open_output_button = ttk.Button(output_controls, text="出力を開く", command=self._open_output)
        self.open_output_button.pack(side="left", fill="x", expand=True)
        self.open_folder_button = ttk.Button(output_controls, text="保存フォルダ", command=self._open_folder)
        self.open_folder_button.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("処理ログ")
        self.log_window.geometry("760x340")
        self.log_window.withdraw()
        self.log_window.protocol("WM_DELETE_WINDOW", self.log_window.withdraw)
        log_frame = ttk.Frame(self.log_window, padding=12)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, height=12, width=65, wrap="word", state="disabled", bg="#f5f3ec", fg="#14200e", relief="flat", padx=10, pady=8, font=("Menlo", 10), highlightthickness=1, highlightbackground="#14200e")
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        log_scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=log_scroll.set)

        from sdr2hdr.gui_style import FeedbackMotion
        self.feedback_motion = FeedbackMotion(self.root, self.reduce_motion_var, self.status_label, self.export_button)
        footer = ttk.Frame(right)
        footer.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(footer, text="動きを減らす", variable=self.reduce_motion_var).pack(side="left")
        self.log_button = ttk.Button(footer, text="処理ログを開く", command=self._show_log)
        self.log_button.pack(side="right")

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
        ttk.Label(tab, text="AI 強度").grid(row=8, column=0, sticky="w", pady=2, padx=(0, 12))
        ai_slider_row = ttk.Frame(tab)
        ai_slider_row.grid(row=8, column=1, sticky="ew", pady=2)
        ai_slider_row.columnconfigure(0, weight=1)
        self.ai_strength_scale = ttk.Scale(
            ai_slider_row, from_=0.0, to=0.8, orient="horizontal",
            variable=self.ai_strength_var, command=self._sync_ai_strength_label,
        )
        self.ai_strength_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(ai_slider_row, textvariable=self.ai_strength_label_var, width=5).grid(row=0, column=1, padx=(8, 0))
        
        ttk.Label(tab, text="彩度 (Saturation)").grid(row=9, column=0, sticky="w", pady=2, padx=(0, 12))
        sat_slider_row = ttk.Frame(tab)
        sat_slider_row.grid(row=9, column=1, sticky="ew", pady=2)
        sat_slider_row.columnconfigure(0, weight=1)
        self.saturation_scale = ttk.Scale(
            sat_slider_row, from_=0.8, to=1.5, orient="horizontal",
            variable=self.saturation_var, command=self._sync_saturation_label,
        )
        self.saturation_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(sat_slider_row, textvariable=self.saturation_label_var, width=5).grid(row=0, column=1, padx=(8, 0))

        # AI強度説明用
        ttk.Label(tab, text="AIがどれだけ積極的に輝度を拡張するかを調整します。値を上げるとより眩しいHDRになりますが、上げすぎると不自然な階調になる場合があります。", 
                  font=("Helvetica", 9), foreground="#555", wraplength=400).grid(row=11, column=1, sticky="w", pady=(4, 4))
        self._add_hdr_details(tab, 10)


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
        ttk.Label(tab, text="AI 強度").grid(row=7, column=0, sticky="w", pady=2, padx=(0, 12))
        slider_row = ttk.Frame(tab)
        slider_row.grid(row=7, column=1, sticky="ew", pady=2)
        slider_row.columnconfigure(0, weight=1)
        ttk.Scale(slider_row, from_=0.0, to=0.8, orient="horizontal",
                  variable=self.ai_strength_var, command=self._sync_ai_strength_label).grid(row=0, column=0, sticky="ew")
        ttk.Label(slider_row, textvariable=self.ai_strength_label_var, width=5).grid(row=0, column=1, padx=(8, 0))
        
        ttk.Label(tab, text="彩度 (Saturation)").grid(row=8, column=0, sticky="w", pady=2, padx=(0, 12))
        img_sat_slider_row = ttk.Frame(tab)
        img_sat_slider_row.grid(row=8, column=1, sticky="ew", pady=2)
        img_sat_slider_row.columnconfigure(0, weight=1)
        ttk.Scale(img_sat_slider_row, from_=0.8, to=1.5, orient="horizontal",
                  variable=self.saturation_var, command=self._sync_saturation_label).grid(row=0, column=0, sticky="ew")
        ttk.Label(img_sat_slider_row, textvariable=self.saturation_label_var, width=5).grid(row=0, column=1, padx=(8, 0))

        # AI強度説明用
        ttk.Label(tab, text="AIがどれだけ積極的に輝度を拡張するかを調整します。値を上げるとより眩しいHDRになりますが、上げすぎると不自然な階調になる場合があります。", 
                  font=("Helvetica", 9), foreground="#555", wraplength=400).grid(row=10, column=1, sticky="w", pady=(4, 8))
        self._add_hdr_details(tab, 9)

    def _add_scrollable_ai_form(self, tab):
        from sdr2hdr.gui_style import PAPER
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        canvas = tk.Canvas(tab, width=1, height=1, background=PAPER, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        canvas.configure(yscrollcommand=scrollbar.set)
        form = ttk.Frame(canvas)
        item = canvas.create_window(0, 0, window=form, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        return form

    def _bind_ai_form_scroll(self, form):
        canvas = form.master

        def scroll(event):
            if canvas.yview() != (0.0, 1.0):
                delta = event.delta if self.system_name == "Darwin" else event.delta / 120
                canvas.yview_scroll(-int(delta), "units")
                return "break"

        def reveal(event):
            widget = event.widget
            top = widget.winfo_rooty() - form.winfo_rooty()
            bottom = top + widget.winfo_height()
            visible_top = canvas.canvasy(0)
            if top < visible_top:
                canvas.yview_moveto(top / form.winfo_height())
            elif bottom > visible_top + canvas.winfo_height():
                canvas.yview_moveto((bottom - canvas.winfo_height()) / form.winfo_height())

        def bind_children(widget):
            widget.bind("<FocusIn>", reveal, add="+")
            if not isinstance(widget, (ttk.Combobox, ttk.Scale)):
                widget.bind("<MouseWheel>", scroll, add="+")
            for child in widget.winfo_children():
                bind_children(child)

        canvas.bind("<MouseWheel>", scroll)
        bind_children(form)

    def _add_hdr_details(self, tab, row):
        section = ttk.Frame(tab)
        section.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        section.columnconfigure(0, weight=1)
        body = ttk.Frame(section, padding=(8, 4, 8, 4))
        body.grid(row=1, column=0, sticky="ew")
        body.columnconfigure(1, weight=1)

        def toggle():
            if body.winfo_manager():
                body.grid_remove()
                button.configure(text="▶ 詳細HDR設定")
                tab.master.yview_moveto(0.0)
            else:
                body.grid()
                button.configure(text="▼ 詳細HDR設定")
                section.update_idletasks()
                canvas = tab.master
                canvas.yview_moveto(1.0)

        button = ttk.Button(section, text="▶ 詳細HDR設定", command=toggle,
                            style="HDRDisclosure.TButton")
        button.grid(row=0, column=0, sticky="ew")
        ttk.Label(body, text="目標ピーク輝度").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=3)
        peak_row = ttk.Frame(body)
        peak_row.grid(row=0, column=1, sticky="ew", pady=3)
        peak = ttk.Entry(peak_row, textvariable=self.peak_nits_var, width=10)
        peak.pack(side="left")
        ttk.Label(peak_row, text="nit").pack(side="left", padx=(8, 0))
        guidance = self._add_combo_row(body, 1, "HDR Guidance", self.hdr_guidance_label_var, ["自動", "ON", "OFF"])
        guidance.bind("<<ComboboxSelected>>", lambda _: self.hdr_guidance_var.set(
            {"自動": "auto", "ON": "on", "OFF": "off"}[self.hdr_guidance_label_var.get()]))
        self.hdr_detail_controls.extend((peak, guidance))
        for index, (label, variable) in enumerate((("輝度ガイダンス", self.luminance_guidance_var),
                                                    ("復元強度", self.reconstruction_strength_var)), 2):
            ttk.Label(body, text=label).grid(row=index, column=0, sticky="w", padx=(0, 12), pady=3)
            slider_row = ttk.Frame(body)
            slider_row.grid(row=index, column=1, sticky="ew", pady=3)
            slider_row.columnconfigure(0, weight=1)
            value = tk.StringVar(value=f"{variable.get():.2f}")
            variable.trace_add("write", lambda *_, v=variable, text=value: text.set(f"{v.get():.2f}"))
            scale = ttk.Scale(slider_row, from_=0.0, to=1.0, variable=variable, orient="horizontal")
            scale.grid(row=0, column=0, sticky="ew")
            ttk.Label(slider_row, textvariable=value, width=5).grid(row=0, column=1, padx=(8, 0))
            self.hdr_detail_controls.append(scale)
        body.grid_remove()
        self.hdr_detail_panels.append((button, body))

    def _peak_nits(self):
        try:
            peak = float(self.peak_nits_var.get())
        except ValueError:
            raise ValueError("目標ピーク輝度には、0より大きい数値を入力してください。") from None
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("目標ピーク輝度には、0より大きい数値を入力してください。")
        return peak

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
        path_label.grid(row=row, column=0, sticky="w", pady=2, padx=(0, 12))
        self._path_labels[str(variable)] = (path_label, label)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
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
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=2)
        return combo

    def _add_format_row(self, parent, row, label, variable, values, description):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", pady=2, padx=(0, 12))
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=2)
        frame.columnconfigure(0, weight=1)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.grid(row=0, column=0, sticky="ew")
        hint = ttk.Label(frame, textvariable=description, wraplength=400, foreground="#555")
        hint.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        frame.bind("<Configure>", lambda event: hint.configure(wraplength=max(1, event.width)))
        return combo

    def _add_exr_delivery_row(self, parent, row, variable):
        label = ttk.Label(parent, text="保存方法")
        label.grid(row=row, column=0, sticky="w", pady=2, padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=list(EXR_DELIVERY_OPTIONS.values()), state="readonly", width=30)
        combo.grid(row=row, column=1, sticky="ew", pady=2)
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
        running = self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}
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
        if source and previous_source and source != previous_source:
            # A chosen folder persists, but a filename belongs to one input.
            name = Path(build_output_path(source, extension=extension)).name
            updated = str(Path(current).with_name(name)) if current else build_output_path(source, extension=extension)
            automatic = True
        elif automatic and source and not current:
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
            self.peak_nits_var.set(f"{config.peak_nits:g}")
            self._sync_ai_strength_label()
            self._sync_saturation_label()
        self._sync_mode_hint()

    def _sync_model_controls(self, *_: object) -> None:
        has_model = bool(self.model_path_var.get().strip())
        running = self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}
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
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
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
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
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
                peak_nits=self._peak_nits(),
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
                peak_nits=self._peak_nits(),
                backend=self._selected_backend(),
                model_path=self.model_path_var.get().strip() or None,
                ai_strength=self.ai_strength_var.get() if self.model_path_var.get().strip() else None,
                device="auto",
                saturation=self.saturation_var.get(),
                hdr_guidance=self.hdr_guidance_var.get(),
                luminance_guidance_strength=self.luminance_guidance_var.get(),
                reconstruction_strength=self.reconstruction_strength_var.get(),
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
        selected = self.queue_view.selection() if hasattr(self, "job_detail") else ()
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

        if hasattr(self, "job_detail"):
            valid = [item for item in selected if self.queue_view.exists(item)]
            if valid:
                self.queue_view.selection_set(valid)
                self._show_job_detail()
            else:
                self.job_detail.configure(state="normal")
                self.job_detail.delete("1.0", "end")
                self.job_detail.configure(state="disabled")

            self.start_button.configure(state="disabled" if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING} or (self.queue_jobs and self._next_pending_job_index() is None) else "normal")
            self._update_export_button()

    def _set_job_status(self, index: int | None, status: str) -> None:
        if index is None:
            return
        if 0 <= index < len(self.queue_jobs):
            self.queue_jobs[index].status = status
            self._refresh_job_list()

    def _enqueue_request(self, request: ConversionRequest | LogConversionRequest) -> None:
        self._validate_request(request)
        destination = Path(request.output_path).resolve()
        if any(Path(job.request.output_path).resolve() == destination for job in self.queue_jobs):
            raise ValueError("同じ保存先がすでに一覧にあります。出力ファイル名を変更してください。")
        self.queue_jobs.append(QueueJob(request=request))
        self._refresh_job_list()
        if hasattr(self, "feedback_motion"):
            self.start_button.configure(state="normal")
            if not getattr(self, "_batch_adding", False):
                self._feedback("added", f"追加しました: {Path(request.input_path).name}", str(len(self.queue_jobs) - 1))
        mode_text = "AI (動画)" if isinstance(request, ConversionRequest) else \
                    "Log (動画)" if isinstance(request, LogConversionRequest) else \
                    "AI (画像)" if isinstance(request, ImageConversionRequest) else "Log (画像)"
        self._log(f"キュー追加 ({mode_text}): {Path(request.input_path).name}")

    def _enqueue_current(self) -> None:
        try:
            self._enqueue_inputs()
        except ValueError as exc:
            self._feedback("error", str(exc))

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
        self._batch_adding = True
        try:
            for request in requests:
                self._enqueue_request(request)
        finally:
            self._batch_adding = False
        self._feedback("added", f"{len(requests)}件を追加しました", tuple(str(i) for i in range(len(self.queue_jobs)-len(requests), len(self.queue_jobs))))

    def _selected_job_indices(self) -> list[int]:
        return sorted((int(item_id) for item_id in self.queue_view.selection()), reverse=True)

    def _remove_selected_job(self) -> None:
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
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
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
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
        request = replace(request, output_path=self.preview_outputs.allocate(request.output_path))
        self.current_job_index = index
        self._set_job_status(index, "starting")
        self.cancel_token = CancelToken()
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
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
            return
        if not self.queue_jobs:
            try:
                self._enqueue_inputs()
            except ValueError as exc:
                self._feedback("error", str(exc))
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

    def _show_log(self):
        self.log_window.deiconify()
        self.log_window.lift()

    def _open_output(self) -> None:
        if not self.last_output_path:
            return
        output = Path(self.last_output_path)
        open_path(str(output.parent) if is_exr_sequence(self.last_output_path) else self.last_output_path)

    def _open_folder(self) -> None:
        if not self.last_output_path:
            return
        open_path(str(Path(self.last_output_path).parent))

    def _update_export_button(self):
        if not hasattr(self, "export_button"):
            return
        pending = sum(job.status == "completed" and bool(job.preview_path) for job in self.queue_jobs)
        busy = self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}
        self.export_button.configure(
            text=f"変換済みを書き出す（{pending}件）" if pending else "変換済みを書き出す",
            state="normal" if pending and not busy else "disabled",
        )

    def _export(self):
        if self.state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}:
            return
        jobs = [job for job in self.queue_jobs if job.status == "completed" and job.preview_path]
        if not jobs:
            return
        destinations = [Path(job.request.output_path).resolve() for job in jobs]
        if len(set(destinations)) != len(destinations):
            messagebox.showerror(
                "保存先が重複しています",
                "同じ保存先の項目があるため、書き出しを停止しました。\n"
                "一覧から重複する項目を外し、異なる出力ファイル名で追加してください。",
            )
            return
        existing = [job.request.output_path for job in jobs if Path(job.request.output_path).exists()]
        if existing and not messagebox.askyesno(
            "既存ファイルを上書き", "次の保存先を上書きしますか？\n" + "\n".join(existing)
        ):
            return
        token = self.cancel_token = CancelToken()
        self._set_state(AppState.EXPORTING)
        self.status_var.set("書き出し中")
        self.result_var.set("変換済みのデータを指定先へ保存しています。")
        self.progress.configure(mode="determinate", maximum=100, value=0)

        def worker():
            error = None
            try:
                for number, job in enumerate(jobs, 1):
                    def progress(copied, size):
                        self.event_queue.put(("export_progress", (number, len(jobs), copied, size)))
                    if not export_preview(job.preview_path, job.request.output_path, token, progress):
                        break
                    self.event_queue.put(("exported", job))
            except Exception as exc:
                error = str(exc)
            self.event_queue.put(("export_finished", (token.cancel_requested, error)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _show_job_detail(self, *_):
        selected = self.queue_view.selection()
        if not selected:
            self.job_detail.configure(state="normal")
            self.job_detail.delete("1.0", "end")
            self.job_detail.configure(state="disabled")
            return
        self.job_detail.grid()
        self.detail_scroll.grid()
        index = int(selected[0])
        if index >= len(self.queue_jobs):
            return
        job = self.queue_jobs[index]
        text = f"入力: {job.request.input_path}\n保存先: {job.request.output_path}"
        if job.error:
            text += f"\n失敗理由: {job.error}"
        self.job_detail.configure(state="normal")
        self.job_detail.delete("1.0", "end")
        self.job_detail.insert("1.0", text)
        self.job_detail.configure(state="disabled")

    def _clear_exported_jobs(self):
        jobs = [job for job in self.queue_jobs if job.status == "exported" and job.preview_path]
        if not jobs:
            return ""
        try:
            self.compare_view.remove_pairs(job.preview_path for job in jobs)
        except Exception as error:
            self._log(f"書き出し済みですが、比較の解除に失敗しました: {error}")
            return "比較を解除できなかったため、一時ファイルを残しています。"
        failed = False
        for job in jobs:
            try:
                self.preview_outputs.discard(job.preview_path)
            except (OSError, ValueError) as error:
                failed = True
                job.error = f"書き出し済みですが、一時ファイルの削除に失敗しました: {error}"
                self._log(job.error)
            else:
                self.queue_jobs.remove(job)
                job.preview_path = None
        self._refresh_job_list()
        return "一部の一時ファイルを削除できませんでした。処理ログを確認してください。" if failed else ""

    def _feedback(self, kind, message, row=None):
        if not hasattr(self, "feedback_motion"):
            return
        if kind in {"start", "complete", "stopped"}:
            self.result_var.set(message)
        else:
            self.feedback_var.set(message)
        target = None
        if kind == "error":
            entries = (
                (self.input_entry, self.output_entry),
                (self.log_input_entry, self.log_output_entry),
                (self.img_input_entry, self.img_output_entry),
                (self.img_log_input_entry, self.img_log_output_entry),
            )[self.notebook.index("current")]
            # Only identify a field when its missing value is directly observable.
            for entry in entries:
                if not entry.get().strip():
                    entry.focus_set()
                    target = entry
                    break
        self.feedback_motion.cue(kind, self.queue_view if row is not None else None, row, target)

    def _set_state(self, state: str) -> None:
        previous = self.state
        self.state = state
        if previous != state:
            cues = {
                AppState.RUNNING: ("start", "変換を開始しました。右側で進捗を確認できます。"),
                AppState.COMPLETED: ("complete", "変換完了。指定先への保存は「書き出す」を押してください。"),
                AppState.CANCELLED: ("stopped", "処理を停止しました。"),
            }
            if state in cues:
                self._feedback(*cues[state])
        running = state in {AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING}
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
        self.start_button.configure(state="disabled" if running or (self.queue_jobs and self._next_pending_job_index() is None) else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.add_queue_button.configure(state="disabled" if running else "normal")
        self.remove_queue_button.configure(state="disabled" if running else "normal")
        self.clear_queue_button.configure(state="disabled" if running else "normal")
        self.open_output_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.open_folder_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self._update_export_button()
        self.preset_combo.configure(state=combo_state)
        self.encoder_combo.configure(state=combo_state)
        self.log_encoder_combo.configure(state=combo_state)
        self.backend_combo.configure(state=combo_state)
        for control in self.hdr_detail_controls:
            control.configure(state=combo_state if isinstance(control, ttk.Combobox) else field_state)
        self._sync_encoder_ui()
        self._sync_log_encoder_ui()
        self._sync_model_controls()

    def _finish_current_job(self) -> None:
        self.cancel_token = None
        self.worker = None
        self.current_job_index = None

    def _drain_events(self) -> None:
        if getattr(self, "_closing", False):
            return
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
                if result.cancelled:
                    self.status_var.set("キャンセル済")
                    self.progress_var.set(f"{result.processed_frames} フレームで停止しました。今回の未完成出力は破棄しました")
                    self._log("変換をキャンセルし、今回の未完成出力を破棄しました")
                    self._finish_current_job()
                    self._set_state(AppState.CANCELLED)
                else:
                    self.status_var.set("完了")
                    if self.current_job_index is not None:
                        job = self.queue_jobs[self.current_job_index]
                        job.preview_path = result.output_path
                        request = job.request
                        self.compare_view.add_pair(
                            request.input_path, result.output_path,
                            image=isinstance(request, (ImageConversionRequest, ImageLogConversionRequest)),
                        )
                    if Path(result.output_path).suffix.lower() == ".zip":
                        self.progress_var.set(f"EXR画像 {result.processed_frames} 枚のZIPを準備しました")
                    else:
                        self.progress_var.set(f"{result.processed_frames} フレーム変換しました")
                    self._log(f"変換完了（未書き出し）: {Path(result.output_path).name}")
                    self._finish_current_job()
                    next_index = self._next_pending_job_index()
                    if next_index is not None:
                        self._set_state(AppState.IDLE)
                        self._start_job(next_index)
                    else:
                        self._set_state(AppState.COMPLETED)
            elif kind == "error":
                self._log(str(payload))
            elif kind == "export_progress":
                number, total, copied, size = payload
                self.progress.configure(value=100 * ((number - 1) + copied / size) / total)
                self.progress_var.set(f"{number}/{total} 件を書き出しています")
            elif kind == "exported":
                payload.status = "exported"
                self.last_output_path = payload.request.output_path
                self._refresh_job_list()
                self._log(f"書き出し完了: {self.last_output_path}")
            elif kind == "export_finished":
                cancelled, error = payload
                self._finish_current_job()
                cleanup_warning = self._clear_exported_jobs()
                self._set_state(AppState.IDLE)
                self.status_var.set("書き出し失敗" if error else "書き出し停止" if cancelled else "書き出し完了")
                self.result_var.set(error or ("未書き出しの結果は再度保存できます。" if cancelled else "指定先へ保存しました。「出力を開く」で確認できます。"))
                if cleanup_warning:
                    self.result_var.set(f"{self.result_var.get()}\n{cleanup_warning}")
                self.progress_var.set("" if error or cancelled else "書き出しが完了しました")
                if error:
                    self._log(error)
                    messagebox.showerror("書き出しに失敗しました", error)
            elif kind == "failed":
                self.progress.stop()
                self.progress.configure(value=0)
                if self.current_job_index is not None:
                    self.queue_jobs[self.current_job_index].error = str(payload)
                self._set_job_status(self.current_job_index, "failed")
                self.status_var.set("失敗")
                self.progress_var.set("変換に失敗しました")
                self._log(str(payload))
                self._feedback("error", str(payload))
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
    root.withdraw()
    app = SDR2HDRGUI(root)
    root.deiconify()
    app._log("準備完了")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
