"""Display original still files with AppKit's image-specific HDR handling."""
from __future__ import annotations

import AppKit
import Quartz

from .native_surface import create_mac_view


class SDRHDRImageView(AppKit.NSImageView):
    def hitTest_(self, point):
        # The parent forwards gestures to the existing shared Tk controls.
        return None


class MacImagePlayer:
    def __init__(self, widget, *, hdr):
        self.widget = widget
        self.hdr = hdr
        self.loaded = self.closed = False
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self._layout = self._cursor = None
        self.parent, self.view = create_mac_view(widget)
        self.view.setWantsLayer_(True)
        self.view.layer().setBackgroundColor_(AppKit.NSColor.blackColor().CGColor())
        self.view.setClipsToBounds_(True)
        self.image_view = SDRHDRImageView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 1, 1))
        self.image_view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
        self.image_view.setPreferredImageDynamicRange_(
            AppKit.NSImageDynamicRangeHigh if hdr else AppKit.NSImageDynamicRangeStandard)
        self.view.addSubview_(self.image_view)
        self.parent.addSubview_(self.view)

    def load(self, path, info):
        self.loaded = False
        image = AppKit.NSImage.alloc().initWithContentsOfFile_(path)
        if image is None or not image.isValid() or min(image.size()) <= 0:
            raise RuntimeError(f"macOSで画像を読み取れません: {path}")
        # Preserve the original profile and any gain map. Do not apply video
        # HDR10 metadata, a custom gain, or a converted in-memory PNG.
        self.image_view.setImage_(image)
        self._layout = None

    def pause(self):
        pass  # Still images have no playback clock.

    def set_zoom(self, factor):
        self.zoom = factor

    def set_pan(self, x, y):
        self.pan = (x, y)

    def _image_frame(self):
        width, height = self.widget.winfo_width(), self.widget.winfo_height()
        image = self.image_view.image()
        if image is None:
            return (0, 0, width, height)
        size = image.size()
        scale = min(width / size.width, height / size.height) * self.zoom
        w, h = size.width * scale, size.height * scale
        return ((width - w) / 2 + self.pan[0] * w,
                (height - h) / 2 - self.pan[1] * h, w, h)

    def get_dimensions(self):
        x, y, w, h = self._image_frame()
        width, height = self.widget.winfo_width(), self.widget.winfo_height()
        return dict(w=width, h=height, ml=x, mr=width-x-w, mt=height-y-h, mb=y)

    def poll(self):
        while not self.view.pointer_events.empty():
            sequence, x, y = self.view.pointer_events.get()
            self.widget.event_generate(sequence, x=x, y=y)
        cursor = self.widget.cget("cursor")
        if cursor != self._cursor and self.view.window():
            self._cursor = cursor
            self.view.draggable = cursor == "fleur"
            self.view.window().invalidateCursorRectsForView_(self.view)
        top = self.widget.winfo_toplevel()
        width, height = self.widget.winfo_width(), self.widget.winfo_height()
        x = self.widget.winfo_rootx() - top.winfo_rootx()
        y = self.widget.winfo_rooty() - top.winfo_rooty()
        if not self.parent.isFlipped():
            y = self.parent.bounds().size.height - y - height
        frame = self._image_frame()
        layout = (x, y, width, height, frame, bool(self.widget.winfo_ismapped()))
        if layout != self._layout:
            self._layout = layout
            Quartz.CATransaction.begin()
            Quartz.CATransaction.setDisableActions_(True)
            self.view.setFrame_(AppKit.NSMakeRect(x, y, width, height))
            self.view.setHidden_(not layout[-1])
            self.image_view.setFrame_(AppKit.NSMakeRect(*frame))
            Quartz.CATransaction.commit()
        # AppKit reports unspecified until the image has actually displayed.
        self.loaded = self.image_view.imageDynamicRange() != AppKit.NSImageDynamicRangeUnspecified
        return []

    def output_status(self):
        # Report the display mode, not a hardware-output guarantee. On macOS
        # 26.6.2 imageDynamicRange returned standard even while the actual
        # HDR capture contained extended-range pixels. AppKit also suppresses
        # HDR in background apps automatically.
        if self.hdr and self.loaded:
            return "HDR表示：macOS標準"
        return "HDR出力未確認"

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.loaded = False
        self.image_view.setImage_(None)
        self.view.removeFromSuperview()
