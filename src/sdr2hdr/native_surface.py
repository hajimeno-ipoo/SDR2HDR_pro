"""Independent native video surfaces; never copy HDR pixels into Tk images."""
from __future__ import annotations

import platform
import queue
import math

if platform.system() == "Darwin":
    import AppKit
    import objc

    class SDRHDRDraggableView(AppKit.NSView):
        """Forward native mouse gestures to the corresponding Tk video frame."""
        @objc.python_method
        def _pointer(self, sequence, event):
            point = self.convertPoint_fromView_(event.locationInWindow(), None)
            y = point.y if self.isFlipped() else self.bounds().size.height - point.y
            # Do not re-enter Tcl from an AppKit callback inside Tk's event loop.
            self.pointer_events.put((sequence, round(point.x), round(y)))

        def acceptsFirstMouse_(self, event):
            return True

        def mouseDown_(self, event):
            self._pointer("<ButtonPress-1>", event)

        def mouseDragged_(self, event):
            self._pointer("<B1-Motion>", event)

        def mouseUp_(self, event):
            self._pointer("<ButtonRelease-1>", event)

        def resetCursorRects(self):
            dragging = getattr(self, "draggable", False)
            cursor = AppKit.NSCursor.openHandCursor() if dragging else AppKit.NSCursor.arrowCursor()
            self.addCursorRect_cursor_(self.bounds(), cursor)


def create_surface(widget):
    system = platform.system()
    if system == "Darwin":
        return MacSurface(widget)
    if system == "Windows":
        return WindowsSurface(widget)
    raise RuntimeError("比較プレイヤーはmacOSとWindowsに対応しています")


def create_mac_view(widget):
    """Attach a gesture-forwarding view to Tk's exported native root."""
    import _tkinter
    import ctypes

    widget.update_idletasks()
    tk_library = ctypes.CDLL(_tkinter.__file__)
    get_root = tk_library.TkMacOSXGetRootControl
    get_root.argtypes = [ctypes.c_void_p]
    get_root.restype = ctypes.c_void_p
    pointer = get_root(widget.winfo_id())
    if not pointer:
        raise RuntimeError("Tkinterのネイティブ表示面を取得できません")
    parent = objc.objc_object(c_void_p=pointer)
    view = SDRHDRDraggableView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 1, 1))
    view.pointer_events = queue.SimpleQueue()
    view.draggable = False
    return parent, view


class WindowsSurface:
    def __init__(self, widget):
        widget.update_idletasks()
        self.widget = widget

    def player_options(self, hdr):
        return dict(wid=str(self.widget.winfo_id()), vo="gpu-next", gpu_api="d3d11",
                    target_colorspace_hint="auto",
                    target_colorspace_hint_mode="target")

    def attach(self, player):
        self.player = player

    def configure_color(self, player, info, hdr):
        # D3D11 supplies display HDR/SDR state, gamut and peak to GPU-next.
        # Forcing source PQ / 10,000 nits overrides that display information.
        player["target-prim"] = "auto" if hdr else "bt.709"
        player["target-trc"] = "auto" if hdr else "srgb"
        player["target-peak"] = "auto" if hdr else 203

    def draw(self):
        pass  # D3D11 owns the child HWND and its rendering loop.

    def output_status(self):
        # Do not promote a requested setting to a verified Windows HDR state.
        return "HDR出力未確認（Windows実機未確認）"

    def close(self):
        pass  # The player terminates before its Tk child HWND is destroyed.


class MacSurface:
    def __init__(self, widget):
        import ctypes
        import objc
        import AppKit
        import Quartz

        self.widget = widget
        self.view = self.layer = self.renderer = None
        self._bounds = None
        self._color = None
        self._configured_hdr = False
        self._metadata_range = None
        self._file_hdr10 = False
        # Tk's Window id is a MacDrawable, not an Objective-C object.
        # Use the exported Tk macOS accessor instead of casting that id.
        self.parent, self.view = create_mac_view(widget)
        self._cursor = None
        from . import _compare_gl
        self.bridge = ctypes.CDLL(_compare_gl.__file__)
        self.draw_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        for name, args, result in (
            ("create", [], ctypes.c_void_p),
            ("set_callback", [ctypes.c_void_p, self.draw_type], None),
            ("make_current", [ctypes.c_void_p], None),
            ("draw", [ctypes.c_void_p], ctypes.c_int),
            ("error", [ctypes.c_void_p], ctypes.c_char_p),
            ("frames", [ctypes.c_void_p], ctypes.c_ulong),
            ("release", [ctypes.c_void_p], None),
        ):
            function = getattr(self.bridge, "comparison_layer_" + name)
            function.argtypes, function.restype = args, result
        self.layer_pointer = self.bridge.comparison_layer_create()
        if not self.layer_pointer:
            raise RuntimeError("16bit浮動小数の表示面を作成できません")
        self.layer = objc.objc_object(c_void_p=self.layer_pointer)
        self.draw_error = None
        self.layer.setContentsFormat_(Quartz.kCAContentsFormatRGBA16Float)
        self.layer.setOpaque_(True)
        self.layer.setBackgroundColor_(AppKit.NSColor.blackColor().CGColor())
        self.view.setLayer_(self.layer)
        self.view.setWantsLayer_(True)
        self.parent.addSubview_(self.view)
        self._resize()

    def player_options(self, hdr):
        # The destination is RGBA16Float, not an 8-bit display framebuffer.
        # libmpv cannot discover its depth and otherwise dithers to 8 bpc.
        return dict(vo="libmpv", hdr_compute_peak="no", dither_depth="no")

    def attach(self, player):
        import ctypes
        import threading
        import mpv

        self.player = player
        self.pending = threading.Event()
        self.gl_library = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")

        def address(_, name):
            try:
                return ctypes.cast(getattr(self.gl_library, name.decode()), ctypes.c_void_p).value
            except AttributeError:
                return None

        self.get_proc = mpv.MpvGlGetProcAddressFn(address)
        self.bridge.comparison_layer_make_current(self.layer_pointer)
        self.renderer = mpv.MpvRenderContext(player, "opengl",
                                            opengl_init_params=dict(get_proc_address=self.get_proc))
        self.renderer.update_cb = self.pending.set
        def render(fbo, width, height):
            try:
                self.renderer.render(opengl_fbo=dict(fbo=fbo, w=width, h=height,
                                                     internal_format=0x881A), flip_y=False)
            except Exception as error:
                self.draw_error = str(error)
        self.draw_callback = self.draw_type(render)
        self.bridge.comparison_layer_set_callback(self.layer_pointer, self.draw_callback)

    def configure_color(self, player, info, hdr):
        import Quartz

        if hdr:
            # mpv linear HDR uses 203 nits per 1.0. CAEDRMetadata declares that
            # unit; macOS handles the actual display's brightness capacity.
            trc, primaries, color = "linear", "bt.2020", Quartz.kCGColorSpaceExtendedLinearITUR_2020
        else:
            trc, primaries, color = "srgb", "bt.709", Quartz.kCGColorSpaceSRGB
        player["target-trc"] = trc
        player["target-prim"] = primaries
        # Preserve the full PQ encoding range before the OS maps it to the
        # display. This is an intermediate signal limit, not a display peak.
        player["target-peak"] = 10000 if hdr else 203
        self._color = Quartz.CGColorSpaceCreateWithName(color)
        if self._color is None:
            raise RuntimeError("表示面の色空間を作成できません")
        self.layer.setColorspace_(self._color)
        self.layer.setWantsExtendedDynamicRangeContent_(hdr)
        self._metadata_range = None
        self._file_hdr10 = hdr and info.get("color_transfer") == "smpte2084"
        self.layer.setEDRMetadata_(
            Quartz.CAEDRMetadata.HDR10MetadataWithDisplayInfo_contentInfo_opticalOutputScale_(
                info.get("hdr10_display_info") if self._file_hdr10 else None,
                info.get("hdr10_content_info") if self._file_hdr10 else None,
                203.0)
            if hdr else None)
        if self.layer.respondsToSelector_("setToneMapMode:"):
            self.layer.setToneMapMode_(Quartz.CAToneMapModeAutomatic if hdr else Quartz.CAToneMapModeNever)
        if self.layer.respondsToSelector_("setPreferredDynamicRange:"):
            self.layer.setPreferredDynamicRange_(Quartz.CADynamicRangeHigh if hdr else Quartz.CADynamicRangeStandard)
        self._configured_hdr = hdr
        self.pending.set()

    def _update_hdr_metadata(self):
        import Quartz

        # PQ images and videos use recorded metadata, or Apple's
        # defaults when absent. mpv fills absent PQ peaks with 10,000 nits;
        # that inferred value is not mastering-display metadata.
        if not self._configured_hdr or self._file_hdr10:
            return
        # HLG is made display-linear by mpv's source OOTF before reaching
        # the Metal layer. Keep its existing decoded-input metadata handling.
        params = self.player.video_out_params or {}
        minimum, maximum = params.get("min-luma"), params.get("max-luma")
        values = (minimum, maximum)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return
        if not 0 <= minimum < maximum or values == self._metadata_range:
            return
        self.layer.setEDRMetadata_(
            Quartz.CAEDRMetadata.HDR10MetadataWithMinLuminance_maxLuminance_opticalOutputScale_(
                minimum, maximum, 203.0))
        self._metadata_range = values

    def _resize(self):
        import AppKit
        import Quartz

        if not self.view:
            return False
        widget = self.widget
        top = widget.winfo_toplevel()
        width, height = widget.winfo_width(), widget.winfo_height()
        x = widget.winfo_rootx() - top.winfo_rootx()
        y = widget.winfo_rooty() - top.winfo_rooty()
        if not self.parent.isFlipped():
            y = self.parent.bounds().size.height - y - height
        scale = self.parent.window().backingScaleFactor()
        bounds = (x, y, width, height, scale, bool(widget.winfo_ismapped()))
        if bounds == self._bounds:
            return False
        self._bounds = bounds
        Quartz.CATransaction.begin()
        Quartz.CATransaction.setDisableActions_(True)
        self.view.setFrame_(AppKit.NSMakeRect(x, y, width, height))
        self.view.setHidden_(not bounds[-1])
        self.layer.setContentsScale_(scale)
        Quartz.CATransaction.commit()
        return True

    def draw(self):
        if self.renderer is None:
            return
        while not self.view.pointer_events.empty():
            sequence, x, y = self.view.pointer_events.get()
            self.widget.event_generate(sequence, x=x, y=y)
        cursor = self.widget.cget("cursor")
        if cursor != self._cursor and self.view.window():
            self._cursor = cursor
            self.view.draggable = cursor == "fleur"
            self.view.window().invalidateCursorRectsForView_(self.view)
        changed = self._resize()
        if changed or self.pending.is_set():
            self.pending.clear()
            self.bridge.comparison_layer_make_current(self.layer_pointer)
            self.renderer.update()
            self._update_hdr_metadata()
            result = self.bridge.comparison_layer_draw(self.layer_pointer)
            if result < 0:
                error = self.bridge.comparison_layer_error(self.layer_pointer)
                raise RuntimeError(error.decode() if error else "Metalの描画に失敗しました")
            if result:
                self.renderer.report_swap()
            else:
                self.pending.set()
        if self.draw_error:
            raise RuntimeError(self.draw_error)

    def output_status(self):
        screen = self.view.window().screen() if self.view and self.view.window() else None
        # libmpv's OpenGL render API does not expose video-target-params.
        # The app owns this layer and its matching output target explicitly.
        target_trc = self.player["target-trc"]
        source = self.player.video_params or {}
        if (self._configured_hdr and self.bridge.comparison_layer_frames(self.layer_pointer) and screen is not None
                and self.layer.wantsExtendedDynamicRangeContent()
                and source.get("gamma") in {"pq", "hlg"}
                and target_trc == "linear" and self.layer.EDRMetadata() is not None
                and self.player["target-prim"] == "bt.2020"
                and screen.maximumExtendedDynamicRangeColorComponentValue() > 1):
            return "HDR Output: Active（EDR）"
        return "HDR出力未確認"

    def close(self):
        if self.layer is not None:
            self.bridge.comparison_layer_set_callback(self.layer_pointer, self.draw_type())
        if self.renderer is not None:
            self.bridge.comparison_layer_make_current(self.layer_pointer)
            self.renderer.update_cb = None
            self.renderer.free()
            self.renderer = None
        if self.view is not None:
            self.view.removeFromSuperview()
            self.view.setLayer_(None)
            self.view = None
        if self.layer is not None:
            self.bridge.comparison_layer_release(self.layer_pointer)
            self.layer = None
