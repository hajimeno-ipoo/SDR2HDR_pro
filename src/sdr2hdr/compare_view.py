"""Permanent SDR/HDR comparison column in the existing Tk workspace."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from pathlib import Path

from .io import ffprobe_comparison
from .compare_controller import CompareController


@dataclass(frozen=True)
class ComparePair:
    sdr_path: str
    hdr_path: str
    image: bool = False


def color_label(info):
    names = {"bt709": "Rec.709", "bt2020": "BT.2020", "smpte2084": "PQ",
             "arib-std-b67": "HLG", "iec61966-2-1": "sRGB"}
    values = [info.get("color_primaries"), info.get("color_transfer")]
    parts = [names.get(v, v) if v and v not in {"unknown", "unspecified"} else "Unknown" for v in values]
    parts.append(f"{info['bit_depth']}-bit" if info.get("bit_depth") else "Unknown")
    label = " / ".join(parts)
    return f"{label}（{info['color_source']}）" if info.get("color_source") else label


def time_label(seconds):
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class CompareView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=12, style="Card.TFrame", width=440)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1, uniform="video")
        self.rowconfigure(6, weight=1, uniform="video")
        self.pairs = []
        self.controller = None
        self.closed = False
        self._generation = 0
        self._results = queue.SimpleQueue()
        self._timer = None
        self._load_started = None
        self._warning = ""
        self._sync_warning = ""
        self._dragging = False
        self._pan_drag = None
        self._next_sync = self._next_status = 0
        self.duration = 0
        ttk.Label(self, text="03  /  COMPARE", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.selection = ttk.Combobox(self, state="readonly", width=22)
        self.selection.grid(row=1, column=0, sticky="ew", pady=(8, 10))
        self.selection.bind("<<ComboboxSelected>>", self._select)
        ttk.Label(self, text="SDR", style="Group.TLabel").grid(row=2, column=0, sticky="ew")
        self.sdr_surface = tk.Frame(self, bg="black", width=400, height=225)
        self.sdr_surface.grid(row=3, column=0, sticky="nsew")
        self.sdr_info = tk.StringVar(value="変換前")
        ttk.Label(self, textvariable=self.sdr_info, style="Muted.TLabel").grid(row=4, column=0, sticky="w", pady=(3, 8))
        ttk.Label(self, text="HDR", style="Group.TLabel").grid(row=5, column=0, sticky="ew")
        self.hdr_surface = tk.Frame(self, bg="black", width=400, height=225)
        self.hdr_surface.grid(row=6, column=0, sticky="nsew")
        for surface in (self.sdr_surface, self.hdr_surface):
            surface.bind("<ButtonPress-1>", self._pan_start)
            surface.bind("<B1-Motion>", self._pan_move)
            surface.bind("<ButtonRelease-1>", self._pan_end)
        self.hdr_info = tk.StringVar(value="変換後")
        ttk.Label(self, textvariable=self.hdr_info, style="Muted.TLabel").grid(row=7, column=0, sticky="w", pady=(3, 8))
        zoom_controls = ttk.Frame(self)
        zoom_controls.grid(row=8, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(zoom_controls, text="表示倍率（上下共通）", style="Muted.TLabel").pack(side="left")
        self.zoom = tk.StringVar(value="100%")
        self.zoom_combo = ttk.Combobox(zoom_controls, textvariable=self.zoom, width=6,
                                      values=("25%", "50%", "75%", "100%", "150%", "200%", "300%", "400%"), state="disabled")
        self.zoom_combo.pack(side="left", padx=8)
        self.zoom_combo.bind("<<ComboboxSelected>>", self._zoom_changed)
        self.fit_button = ttk.Button(zoom_controls, text="全体表示", command=self._fit, state="disabled")
        self.fit_button.pack(side="right")
        self.controls = ttk.Frame(self)
        self.controls.grid(row=9, column=0, sticky="ew")
        self.controls.columnconfigure(0, weight=1)
        buttons = ttk.Frame(self.controls)
        buttons.grid(row=0, column=0)
        self.buttons = []
        for text, action in (("先頭", lambda: self._operate("seek", 0)),
                             ("◀", lambda: self._operate("frame_step", -1)),
                             ("再生", self._toggle),
                             ("▶", lambda: self._operate("frame_step", 1))):
            button = ttk.Button(buttons, text=text, width=5, command=action, state="disabled")
            button.pack(side="left", padx=2)
            self.buttons.append(button)
        self.position = tk.DoubleVar(value=0)
        self.seekbar = ttk.Scale(self.controls, variable=self.position, from_=0, to=1)
        self.seekbar.grid(row=1, column=0, sticky="ew", pady=6)
        self.seekbar.state(["disabled"])
        self.seekbar.bind("<ButtonPress-1>", self._seek_start)
        self.seekbar.bind("<ButtonRelease-1>", self._seek_end)
        self.seekbar.bind("<KeyRelease>", self._seek_end)
        self.clock = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(self.controls, textvariable=self.clock, style="Muted.TLabel").grid(row=2, column=0)
        self.display_state = tk.StringVar(value="HDR出力未確認")
        ttk.Label(self, textvariable=self.display_state, style="Muted.TLabel").grid(row=10, column=0, sticky="w", pady=(8, 3))
        self.message = tk.StringVar(value="変換が完了すると比較できます。")
        self.message_label = ttk.Label(self, textvariable=self.message, style="Muted.TLabel", wraplength=390)
        self.message_label.grid(row=11, column=0, sticky="ew")
        self.bind("<Configure>", self._configure)
        self.bind("<Destroy>", self._destroy)

    def _configure(self, event):
        if event.widget is self:
            self.message_label.configure(wraplength=max(100, event.width - 24))

    def add_pair(self, sdr_path, hdr_path, *, image=False):
        if Path(hdr_path).suffix.lower() in {".exr", ".zip"}:
            return
        pair = ComparePair(sdr_path, hdr_path, image)
        if pair not in self.pairs:
            self.pairs.append(pair)
        self.selection.configure(values=[f"{i + 1}: {Path(p.hdr_path).name}" for i, p in enumerate(self.pairs)])
        # A newly completed queue job must not replace an active comparison.
        if self.selection.current() < 0:
            self.selection.current(0)
            self._select()

    def _select(self, *_):
        index = self.selection.current()
        if index < 0 or self.closed:
            return
        self._generation += 1
        generation = self._generation
        if self.controller:
            self.controller.pause()
        self._enable(False)
        self._warning = ""
        self._sync_warning = ""
        self.message.set("比較素材を読み込んでいます…")
        pair = self.pairs[index]
        self.controls.grid_remove() if pair.image else self.controls.grid()

        def probe():
            try:
                for path in (pair.sdr_path, pair.hdr_path):
                    if not Path(path).is_file():
                        raise FileNotFoundError(f"ファイルが見つかりません: {Path(path).name}")
                sdr_info, hdr_info = ffprobe_comparison(pair.sdr_path), ffprobe_comparison(pair.hdr_path)
                if pair.image:
                    from .compare_images import prepare_image
                    sdr_info = prepare_image(pair.sdr_path, sdr_info)
                    hdr_info = prepare_image(pair.hdr_path, hdr_info, app_hdr_output=True)
                result = (sdr_info, hdr_info)
            except Exception as error:
                result = error
            self._results.put((generation, pair, result))

        threading.Thread(target=probe, daemon=True).start()
        self._schedule()

    def _create_players(self):
        from .native_surface import create_surface
        from .mpv_player import MpvPlayer
        sdr = None
        try:
            sdr = MpvPlayer(create_surface(self.sdr_surface), hdr=False)
            hdr = MpvPlayer(create_surface(self.hdr_surface), hdr=True)
        except Exception:
            if sdr:
                sdr.close()
            raise
        self.controller = CompareController(sdr, hdr)

    def _load(self, pair, infos):
        if not self.controller:
            self._create_players()
        sdr_info, hdr_info = infos
        self.sdr_info.set(color_label(sdr_info))
        self.hdr_info.set(color_label(hdr_info))
        if hdr_info.get("color_transfer") not in {"smpte2084", "arib-std-b67"}:
            self._warning = "HDRのPQ / HLG色情報を確認できません。"
        if pair.image and hdr_info.get("color_transfer") not in {"smpte2084", "arib-std-b67"}:
            self._warning = "HDR表示未確認（画像形式）"
        self.duration = float(sdr_info.get("duration") or 0) if not pair.image else 0
        self.seekbar.configure(to=max(self.duration, 1))
        self.position.set(0)
        self.controller.load_pair(pair.sdr_path, pair.hdr_path, sdr_info, hdr_info, image=pair.image)
        self.zoom.set("100%")
        self._load_started = time.monotonic()

    def _enable(self, enabled):
        self._pan_drag = None
        self.zoom_combo.configure(state="readonly" if enabled else "disabled")
        self.fit_button.state(["!disabled"] if enabled else ["disabled"])
        for button in self.buttons:
            button.state(["!disabled"] if enabled and self.controller and not self.controller.image else ["disabled"])
        self.seekbar.state(["!disabled"] if enabled and self.duration > 0 else ["disabled"])
        self._pan_cursor()

    def _zoom_changed(self, *_):
        self._pan_drag = None
        self._operate("set_zoom", float(self.zoom.get().removesuffix("%")) / 100)
        self._pan_cursor()

    def _fit(self):
        self.zoom.set("100%")
        self._zoom_changed()

    def _pan_cursor(self):
        enabled = not self.zoom_combo.instate(["disabled"]) and self.controller and self.controller.zoom > 1
        for surface in (self.sdr_surface, self.hdr_surface):
            surface.configure(cursor="fleur" if enabled else "")

    def _pan_start(self, event):
        controller = self.controller
        if not controller or self.zoom_combo.instate(["disabled"]) or controller.zoom <= 1:
            return
        player = controller.sdr if event.widget is self.sdr_surface else controller.hdr
        dimensions = player.mpv.osd_dimensions or {}
        if not all(key in dimensions for key in ("w", "h", "ml", "mr", "mt", "mb")):
            return
        width, height = dimensions["w"], dimensions["h"]
        video_width = width - dimensions["ml"] - dimensions["mr"]
        video_height = height - dimensions["mt"] - dimensions["mb"]
        if min(width, height, video_width, video_height) <= 0:
            return
        # Convert native render pixels to Tk coordinates (including Retina scaling).
        scale_x = video_width / width * event.widget.winfo_width()
        scale_y = video_height / height * event.widget.winfo_height()
        limit_x = max(0.0, (video_width - width) / (2 * video_width))
        limit_y = max(0.0, (video_height - height) / (2 * video_height))
        self._pan_drag = (event.x_root, event.y_root, *controller.pan, scale_x, scale_y, limit_x, limit_y)

    def _pan_move(self, event):
        if self._pan_drag is None:
            return
        x, y, pan_x, pan_y, scale_x, scale_y, limit_x, limit_y = self._pan_drag
        new_x = max(-limit_x, min(limit_x, pan_x + (event.x_root - x) / scale_x))
        new_y = max(-limit_y, min(limit_y, pan_y + (event.y_root - y) / scale_y))
        self._operate("set_pan", new_x, new_y)

    def _pan_end(self, *_):
        self._pan_drag = None

    def _operate(self, method, *args):
        if self.controller:
            try:
                getattr(self.controller, method)(*args)
            except Exception as error:
                self._fail(error)

    def _toggle(self):
        self._operate("pause" if self.controller and self.controller.playing else "play")

    def _seek_start(self, _):
        self._dragging = True

    def _seek_end(self, _):
        self._dragging = False
        self._operate("seek", self.position.get())

    def _schedule(self):
        if self._timer is None and not self.closed:
            self._timer = self.after(16, self._tick)

    def _tick(self):
        self._timer = None
        if self.closed:
            return
        try:
            while not self._results.empty():
                generation, pair, result = self._results.get()
                if generation != self._generation:
                    continue
                if isinstance(result, Exception):
                    raise result
                self._load(pair, result)
            controller = self.controller
            if controller:
                errors = controller.sdr.poll() + controller.hdr.poll()
                if errors:
                    raise RuntimeError(" / ".join(errors))
                if self._load_started is not None:
                    if controller.ready:
                        self._load_started = None
                        self._enable(True)
                    elif time.monotonic() - self._load_started > 20:
                        raise RuntimeError("プレイヤーの読み込みが完了しませんでした")
                now = time.monotonic()
                if not controller.image and now >= self._next_sync:
                    warning = controller.sync_tick()
                    self._next_sync = now + 0.1
                    self._sync_warning = warning or ""
                    position = controller.sdr.get_position()
                    if position is not None and not self._dragging:
                        self.position.set(position)
                    self.clock.set(f"{time_label(position)} / {time_label(self.duration)}")
                    self.buttons[2].configure(text="停止" if controller.playing else "再生")
                if now >= self._next_status:
                    state = controller.hdr.surface.output_status()
                    self.display_state.set(state)
                    self.message.set(self._sync_warning or self._warning or ("" if "Active" in state else "HDR出力を確認できません。"))
                    self._next_status = now + 1
        except Exception as error:
            self._fail(error)
        self._schedule()

    def _fail(self, error):
        self.message.set(f"比較できません: {error}")
        self.display_state.set("HDR出力未確認")
        self._enable(False)
        self._load_started = None
        if self.controller:
            self.controller.close()
            self.controller = None

    def _destroy(self, event):
        if event.widget is self:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        if self.controller:
            self.controller.close()
            self.controller = None
