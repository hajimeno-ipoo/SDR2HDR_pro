"""One libmpv player. Tk and platform rendering live outside this module."""
from __future__ import annotations

import queue
import math


class MpvPlayer:
    def __init__(self, surface, *, hdr: bool):
        try:
            import mpv
        except (ImportError, OSError):
            surface.close()
            raise

        self.surface = surface
        self.hdr = hdr
        self.events = queue.SimpleQueue()
        self.closed = False
        self.loaded = False
        self._image_stream = None
        options = dict(
            config=False, load_scripts=False, osc=False, osd_level=0,
            input_default_bindings=False, input_vo_keyboard=False,
            idle=True, keep_open="always", pause=True, mute=hdr,
            image_display_duration="inf", hwdec="auto-safe",
        )
        options.update(surface.player_options(hdr))
        try:
            self.mpv = mpv.MPV(**options)
        except Exception:
            surface.close()
            raise

        @self.mpv.event_callback("file-loaded", "end-file")
        def on_event(event):
            self.events.put(event.as_dict(decoder=mpv.strict_decoder))

        self._event_callback = on_event
        try:
            surface.attach(self.mpv)
        except Exception:
            try:
                surface.close()
            finally:
                self.mpv.terminate()
            raise

    def load(self, path: str, info: dict):
        self.mpv.command("stop")
        self._release_image_stream()
        while not self.events.empty():
            self.events.get()
        self.loaded = False
        self.mpv.pause = True
        self.surface.configure_color(self.mpv, info, self.hdr)
        data = info.get("image_bytes")
        if data is not None:
            @self.mpv.python_stream(size=len(data))
            def reader():
                yield data
            self._image_stream = reader
            path = reader.stream_uri
        self.mpv.command("loadfile", path, "replace")

    def _release_image_stream(self):
        if self._image_stream is not None:
            self._image_stream.unregister()
            self._image_stream = None

    def poll(self):
        errors = []
        while not self.events.empty():
            event = self.events.get()
            if event["event"] == "file-loaded":
                self.loaded = True
            elif event["event"] == "end-file" and event.get("reason") == "error":
                errors.append(str(event.get("error", "映像の読み込みに失敗しました")))
        self.surface.draw()
        return errors

    def play(self):
        self.mpv.pause = False

    def pause(self):
        self.mpv.pause = True

    def seek_absolute(self, seconds: float):
        self.mpv.seek(max(0.0, seconds), "absolute", "exact")

    def step_frame(self, direction: int):
        self.mpv.command("frame-step" if direction > 0 else "frame-back-step")

    def get_position(self):
        return self.mpv.time_pos

    def set_mute(self, muted: bool):
        self.mpv.mute = muted

    def set_zoom(self, factor: float):
        self.mpv.video_zoom = math.log2(factor)

    def set_pan(self, x: float, y: float):
        self.mpv.video_pan_x = x
        self.mpv.video_pan_y = y

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Free render context before terminating mpv or destroying its drawable.
        try:
            self.surface.close()
        finally:
            try:
                self.mpv.terminate()
            finally:
                self._release_image_stream()
