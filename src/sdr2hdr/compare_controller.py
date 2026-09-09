"""Playback synchronization independent of Tk and platform APIs."""
from __future__ import annotations

import time


class CompareController:
    def __init__(self, sdr, hdr):
        self.sdr, self.hdr = sdr, hdr
        self.playing = False
        self.image = False
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self.sync_tolerance = 0.050
        self._last_correction = 0.0
        self._drift_since = None
        self._align_after_step = False

    def load_pair(self, sdr_path, hdr_path, sdr_info, hdr_info, *, image=False):
        self.pause()
        self.image = image
        self._drift_since = None
        self._align_after_step = False
        self.sdr.load(sdr_path, sdr_info)
        self.hdr.load(hdr_path, hdr_info)
        self.set_zoom(1.0)

    def set_zoom(self, factor):
        if not 0.25 <= factor <= 4.0:
            raise ValueError("表示倍率は25%から400%の範囲で指定してください。")
        self.sdr.set_zoom(factor)
        self.hdr.set_zoom(factor)
        self.zoom = factor
        self.set_pan(0.0, 0.0)

    def set_pan(self, x, y):
        self.sdr.set_pan(x, y)
        self.hdr.set_pan(x, y)
        self.pan = (x, y)

    @property
    def ready(self):
        return self.sdr.loaded and self.hdr.loaded

    def play(self):
        if not self.ready or self.image:
            return
        self.sdr.play()
        self.hdr.play()
        self.playing = True

    def pause(self):
        self.playing = False
        self.sdr.pause()
        self.hdr.pause()

    def seek(self, seconds):
        if not self.ready or self.image:
            return
        self.sdr.seek_absolute(seconds)
        self.hdr.seek_absolute(seconds)
        self._last_correction = time.monotonic()
        self._drift_since = None

    def frame_step(self, direction):
        if not self.ready or self.image:
            return
        self.pause()
        self.sdr.step_frame(direction)
        self.hdr.step_frame(direction)
        self._align_after_step = True
        self._last_correction = time.monotonic()

    def sync_tick(self):
        if self.image or not self.ready:
            return None
        now = time.monotonic()
        if self._align_after_step and now - self._last_correction >= 0.15:
            if not self.sdr.mpv.seeking and not self.hdr.mpv.seeking:
                position = self.sdr.get_position()
                if position is not None:
                    self.hdr.seek_absolute(position)
                self._align_after_step = False
        if not self.playing:
            return None
        if self.sdr.mpv.eof_reached or self.hdr.mpv.eof_reached:
            self.pause()
            return None
        sdr, hdr = self.sdr.get_position(), self.hdr.get_position()
        if sdr is None or hdr is None or self.sdr.mpv.seeking or self.hdr.mpv.seeking:
            return None
        if abs(sdr - hdr) <= self.sync_tolerance:
            self._drift_since = None
            return None
        if self._drift_since is None:
            self._drift_since = now
        # Allow the preceding asynchronous seek to settle before correcting again.
        if now - self._last_correction >= 0.5:
            self.hdr.seek_absolute(sdr)
            self._last_correction = now
        if now - self._drift_since >= 3:
            return "再生位置のずれが続いています。一時停止して位置を確認してください。"
        return None

    def close(self):
        try:
            self.sdr.close()
        finally:
            self.hdr.close()
