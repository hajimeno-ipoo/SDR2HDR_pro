"""AppKit image routing and the controls connected to that display."""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sdr2hdr.compare_images import prepare_image
from sdr2hdr.compare_view import CompareView


def test_native_image_keeps_original_file_and_skips_video_decoding():
    with patch('sdr2hdr.compare_images._run') as decode:
        for extension in ('tif', 'tiff', 'png', 'jxl', 'avif', 'jpg', 'jpeg'):
            info = prepare_image('original.' + extension, {}, app_hdr_output=True, native=True)
            assert 'image_bytes' not in info
        decode.assert_not_called()


@pytest.mark.skipif(sys.platform != 'darwin', reason='AppKit')
def test_only_mac_stills_choose_native_image_player():
    view = CompareView.__new__(CompareView)
    view.sdr_surface, view.hdr_surface = Mock(), Mock()
    with patch('sdr2hdr.mac_image_player.MacImagePlayer') as native, \
            patch('sdr2hdr.mpv_player.MpvPlayer') as video, \
            patch('sdr2hdr.native_surface.create_surface'):
        view._create_players(image=True)
        assert native.call_count == 2
        video.assert_not_called()
        native.reset_mock()
        view._create_players(image=False)
        assert video.call_count == 2
        native.assert_not_called()
        video.reset_mock()
        with patch('sdr2hdr.compare_view.sys.platform', 'win32'):
            view._create_players(image=True)
        assert video.call_count == 2
        native.assert_not_called()


@pytest.mark.skipif(sys.platform != 'darwin' or not os.environ.get('SDR2HDR_TEST_IMAGE'),
                    reason='Requires an existing user image and native GUI access')
def test_native_user_image_display_zoom_drag_fit_and_cleanup():
    import AppKit
    import tkinter as tk
    from sdr2hdr.gui import SDR2HDRGUI
    from sdr2hdr.mac_image_player import MacImagePlayer

    path = Path(os.environ['SDR2HDR_TEST_IMAGE'])
    original = path.read_bytes()
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    view = app.compare_view

    def until(predicate):
        deadline = time.monotonic() + 20
        while not predicate():
            root.update()
            assert time.monotonic() < deadline, view.message.get()
            time.sleep(.01)

    try:
        view.add_pair(str(path), str(path), image=True)
        until(lambda: view.controller is not None and view.controller.ready and
              view._load_started is None)
        players = (view.controller.sdr, view.controller.hdr)
        assert all(isinstance(p, MacImagePlayer) for p in players)
        until(lambda: view.display_state.get() == 'HDR表示：macOS標準')
        # These verify the connection, not screen luminance. The latter is
        # compared separately using HDR captures of this app and Preview.
        assert players[0].image_view.preferredImageDynamicRange() == AppKit.NSImageDynamicRangeStandard
        assert players[1].image_view.preferredImageDynamicRange() == AppKit.NSImageDynamicRangeHigh
        assert not view.controls.winfo_ismapped()
        fit = [p.image_view.frame() for p in players]
        view.zoom.set('200%')
        view._zoom_changed()
        until(lambda: all(p.image_view.frame().size.width > f.size.width for p, f in zip(players, fit)))
        for p, f in zip(players, fit):
            assert p.image_view.frame().size.width == pytest.approx(f.size.width * 2)
            assert p.image_view.frame().size.height == pytest.approx(f.size.height * 2)
            assert p.view.clipsToBounds()
        p = players[1]
        native = p.view
        point = AppKit.NSMakePoint(native.bounds().size.width / 2, native.bounds().size.height / 2)
        # Real AppKit hit testing must reach the gesture-forwarding parent.
        assert native.hitTest_(native.convertPoint_toView_(point, native.superview())) == native
        before = [p.image_view.frame().origin.y for p in players]

        def event(kind, dy):
            location = native.convertPoint_toView_(AppKit.NSMakePoint(point.x, point.y-dy), None)
            return AppKit.NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(
                kind, location, 0, 0, native.window().windowNumber(), None, 0, 1, 1)

        native.mouseDown_(event(AppKit.NSEventTypeLeftMouseDown, 0))
        until(lambda: view._pan_drag is not None)
        native.mouseDragged_(event(AppKit.NSEventTypeLeftMouseDragged, 24))
        native.mouseUp_(event(AppKit.NSEventTypeLeftMouseUp, 24))
        until(lambda: all(p.image_view.frame().origin.y < y for p, y in zip(players, before)))
        assert view.controller.pan[1] > 0
        assert view._pan_drag is None
        view.fit_button.invoke()
        until(lambda: all(p.image_view.frame() == f for p, f in zip(players, fit)))
        assert view.controller.pan == (0, 0)
        print('AppKit dynamic ranges:', [p.image_view.imageDynamicRange() for p in players],
              'HDR:', view.display_state.get(), 'zoom/drag/fit: passed')
    finally:
        app._close()
    assert all(p.closed and p.view.superview() is None for p in players)
    assert path.read_bytes() == original
