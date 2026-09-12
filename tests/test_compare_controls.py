"""Continuous user seeking and comparison-only audio muting."""
import os
import time
import tkinter as tk
from unittest.mock import Mock

import pytest

from sdr2hdr.compare_controller import CompareController
from sdr2hdr.compare_view import CompareView
from sdr2hdr.gui_style import apply_theme, GREEN
from sdr2hdr.mpv_player import MpvPlayer


def test_mute_controls_only_comparison_audio_and_never_unmutes_hdr():
    sdr, hdr = Mock(), Mock()
    controller = CompareController(sdr, hdr)
    for muted in (True, False):
        controller.set_mute(muted)
        sdr.set_mute.assert_called_with(muted)
        hdr.set_mute.assert_called_with(True)
    sdr.reset_mock(); hdr.reset_mock()
    controller.image = True
    controller.set_mute(False)
    sdr.set_mute.assert_not_called()
    hdr.set_mute.assert_not_called()


def scale_point(scale, seconds):
    return tuple(round(float(v)) for v in scale.tk.call(str(scale), 'coords', seconds))


def test_seek_and_position_do_not_wait_on_mpv_and_errors_are_reported():
    from unittest.mock import patch
    from types import SimpleNamespace
    surface, mpv = Mock(), Mock()
    surface.player_options.return_value = {}
    with patch.dict('sys.modules', mpv=SimpleNamespace(MPV=Mock(return_value=mpv))):
        player = MpvPlayer(surface, hdr=False)
    observers = {call.args[0]:call.args[1] for call in mpv.observe_property.call_args_list}
    observers['time-pos']('time-pos', 2.5)
    observers['seeking']('seeking', True)
    observers['eof-reached']('eof-reached', False)
    assert player.get_position() == 2.5
    assert player.seeking and not player.eof_reached
    player.seek_absolute(3)
    mpv.command_async.assert_called_once_with('seek', 3, 'absolute+exact', callback=player._seek_finished)
    mpv.seek.assert_not_called()
    player._seek_finished(RuntimeError('seek failed'), None)
    assert player.poll() == ['seek failed']
    player.load('next.mov', {})
    assert player.get_position() is None
    assert not player.seeking and not player.eof_reached
    player.close()


def test_comparison_buttons_match_and_zoom_adjusts_one_percent():
    from tkinter import ttk
    root = tk.Tk()
    try:
        apply_theme(root)
        view = CompareView(root); view.pack(fill='both', expand=True)
        root.geometry('440x900')
        sdr, hdr = Mock(loaded=True), Mock(loaded=True)
        view.controller = CompareController(sdr, hdr)
        for button in (view.zoom_combo, view.zoom_in, view.zoom_out, view.fit_button, view.mute_button):
            assert button.instate(['disabled'])
        view._enable(True)
        root.update()
        buttons = [*view.buttons, view.mute_button, view.fit_button]
        assert len({(b.winfo_width(), b.winfo_height()) for b in buttons}) == 1
        assert all(b.winfo_height() == buttons[0].winfo_height() for b in (view.zoom_in, view.zoom_out))
        for button in [*buttons, view.zoom_combo, view.zoom_in, view.zoom_out]:
            assert button.winfo_rootx() >= view.winfo_rootx()
            assert button.winfo_rootx() + button.winfo_width() <= view.winfo_rootx() + view.winfo_width()
        # Keep the zoom controls together, with breathing room on both sides
        # of the preset list. Fit must not drift to the far edge on resize.
        def gap(left, right):
            return right.winfo_rootx() - left.winfo_rootx() - left.winfo_width()
        for width in (440, 540):
            root.geometry(f'{width}x900')
            root.update()
            assert gap(view.zoom_out, view.zoom_combo) == 8
            assert gap(view.zoom_combo, view.zoom_in) == 8
            assert gap(view.zoom_in, view.fit_button) == 12
            assert view.fit_button.winfo_width() == view.buttons[0].winfo_width()
            zoom_row = view.zoom_combo.master
            playback_row = view.buttons[0].master
            assert abs((zoom_row.winfo_rootx() + zoom_row.winfo_width()/2)
                       - (playback_row.winfo_rootx() + playback_row.winfo_width()/2)) <= 1
            assert playback_row.winfo_rooty() - zoom_row.winfo_rooty() - zoom_row.winfo_height() == 8
        view.zoom_in.invoke()
        assert view.zoom.get() == '101%'
        sdr.set_zoom.assert_called_with(1.01)
        hdr.set_zoom.assert_called_with(1.01)
        view.zoom_out.invoke()
        assert view.zoom.get() == '100%'
        assert view.zoom_combo.instate(['readonly'])
        assert tuple(view.zoom_combo['values']) == ('25%', '50%', '75%', '100%', '150%', '200%', '300%', '400%')
        for preset in view.zoom_combo['values']:
            view.zoom_combo.set(preset)
            view.zoom_combo.event_generate('<<ComboboxSelected>>')
            factor = float(preset.removesuffix('%')) / 100
            sdr.set_zoom.assert_called_with(factor)
            hdr.set_zoom.assert_called_with(factor)
        view.zoom_combo.set('150%')
        view.zoom_combo.event_generate('<<ComboboxSelected>>')
        view.zoom_in.invoke()
        assert view.zoom.get() == view.zoom_combo.get() == '151%'
        sdr.set_zoom.assert_called_with(1.51)
        hdr.set_zoom.assert_called_with(1.51)
        view.zoom_out.invoke()
        assert view.zoom.get() == view.zoom_combo.get() == '150%'
        for value, button in ((25, view.zoom_out), (400, view.zoom_in)):
            view.zoom.set(f'{value}%'); view._zoom_changed()
            button.invoke()
            assert view.zoom.get() == f'{value}%'
        view.fit_button.invoke()
        assert view.zoom.get() == '100%' and view.controller.pan == (0, 0)
        view.mute_button.invoke()
        assert view.mute_button.instate(['selected'])
        assert ttk.Style(root).lookup('Compare.TButton', 'background', ('selected',)) == GREEN
        assert view.mute_button.winfo_width() == view.fit_button.winfo_width()
    finally:
        root.destroy()


def test_scale_moves_both_videos_before_release_without_playback_feedback():
    root = tk.Tk()
    try:
        view = CompareView(root); view.pack(fill='both', expand=True)
        sdr, hdr = Mock(loaded=True), Mock(loaded=True)
        view.controller = CompareController(sdr, hdr)
        view.duration = 10
        view.seekbar.configure(to=10)
        view._enable(True)
        root.update()
        view.position.set(1)
        root.update()
        sdr.seek_absolute.assert_not_called()
        x, y = scale_point(view.seekbar, 1)
        view.seekbar.event_generate('<ButtonPress-1>', x=x, y=y)
        for seconds in (3, 5, 7):
            x, y = scale_point(view.seekbar, seconds)
            view.seekbar.event_generate('<B1-Motion>', x=x, y=y)
            root.update()
            assert view._dragging
            assert sdr.seek_absolute.call_args.args[0] == pytest.approx(seconds, abs=.05)
            assert hdr.seek_absolute.call_args == sdr.seek_absolute.call_args
        view.seekbar.event_generate('<ButtonRelease-1>', x=x, y=y)
        assert not view._dragging
        assert sdr.seek_absolute.call_args.args[0] == view.position.get()
        sdr.play.assert_not_called(); sdr.pause.assert_not_called()
        view.mute_button.invoke()
        sdr.set_mute.assert_called_with(True)
        view.mute_button.invoke()
        sdr.set_mute.assert_called_with(False)
        hdr.set_mute.assert_called_with(True)
        view.mute_button.invoke()
        view._load(type('Pair', (), dict(sdr_path='sdr.mov', hdr_path='hdr.mov', image=False))(),
                   ({'duration':10}, {'color_transfer':'smpte2084'}))
        sdr.set_mute.assert_called_with(True)
        assert view.muted.get()
    finally:
        root.destroy()


@pytest.mark.skipif(not os.environ.get('SDR2HDR_TEST_VIDEO') or not os.environ.get('SDR2HDR_TEST_HDR_VIDEO'),
                    reason='Requires existing user videos and native GUI access')
def test_real_video_scrubbing_updates_rendered_frames_and_mute():
    from sdr2hdr.gui import SDR2HDRGUI

    root = tk.Tk(); app = SDR2HDRGUI(root); view = app.compare_view
    root.title('SDR2HDR Pro 動画操作確認')

    def until(predicate):
        deadline = time.monotonic() + 20
        outcome = []
        def check():
            if predicate():
                outcome.append(True)
                root.quit()
            elif time.monotonic() >= deadline:
                root.quit()
            else:
                root.after(10, check)
        root.after(0, check)
        root.mainloop()
        assert outcome, view.message.get()

    try:
        view.add_pair(os.environ['SDR2HDR_TEST_VIDEO'], os.environ['SDR2HDR_TEST_HDR_VIDEO'])
        until(lambda: view.controller is not None and view.controller.ready and view._load_started is None)
        players = (view.controller.sdr, view.controller.hdr)
        # Confirm the native renderer presents new frames before button release,
        # not merely that a seek option was set.
        last_frames = [p.surface.bridge.comparison_layer_frames(p.surface.layer_pointer) for p in players]
        last_positions = [p.get_position() or 0 for p in players]
        view.position.set(0)
        root.update()
        x, y = scale_point(view.seekbar, 0)
        view.seekbar.event_generate('<ButtonPress-1>', x=x, y=y)
        positions = []
        for fraction in (.2, .45, .7):
            target = view.duration * fraction
            x, y = scale_point(view.seekbar, target)
            view.seekbar.event_generate('<B1-Motion>', x=x, y=y)
            until(lambda: all(not p.seeking and abs((p.get_position() or 0)-target) < .12
                              and p.surface.bridge.comparison_layer_frames(p.surface.layer_pointer) > n
                              for p, n in zip(players, last_frames)))
            assert view._dragging
            assert all(p.mpv.pause for p in players)
            current = [p.get_position() for p in players]
            assert all(b > a for a, b in zip(last_positions, current))
            positions.append(current)
            last_positions = current
            last_frames = [p.surface.bridge.comparison_layer_frames(p.surface.layer_pointer) for p in players]
        view.seekbar.event_generate('<ButtonRelease-1>', x=x, y=y)
        until(lambda: all(not p.seeking and abs(p.get_position()-target) < .12 for p in players))
        assert not view._dragging
        view.mute_button.invoke()
        assert players[0].mpv.mute and players[1].mpv.mute
        view._toggle()
        until(lambda: (players[0].get_position() or 0) > last_positions[0]+.15)
        view.mute_button.invoke()
        assert not players[0].mpv.mute and players[1].mpv.mute
        assert view.controller.playing
        view._toggle()
        # Send continuous input, without waiting for each seek to settle. The
        # former test's three settled positions did not exercise this case.
        view.controller.seek(view.duration * .85)
        until(lambda: all(not p.seeking and abs((p.get_position() or 0)-view.duration*.85) < .12 for p in players))
        x, y = scale_point(view.seekbar, view.position.get())
        view.seekbar.event_generate('<ButtonPress-1>', x=x, y=y)
        samples = [[], []]
        callback_ms = []
        began = time.monotonic()
        def sweep():
            elapsed = time.monotonic() - began
            target = view.duration * (.85 - .7 * min(elapsed/2, 1))
            x, y = scale_point(view.seekbar, target)
            before = time.monotonic()
            view.seekbar.event_generate('<B1-Motion>', x=x, y=y)
            callback_ms.append((time.monotonic()-before)*1000)
            for player, sample in zip(players, samples):
                count = player.surface.bridge.comparison_layer_frames(player.surface.layer_pointer)
                if not sample or sample[-1][1] != count:
                    sample.append((elapsed, count))
            if elapsed < 2:
                root.after(16, sweep)
            else:
                view.seekbar.event_generate('<ButtonRelease-1>', x=x, y=y)
                root.quit()
        root.after(0, sweep)
        root.mainloop()
        final_target = view.duration * .15
        until(lambda: all(not p.seeking and abs((p.get_position() or 0)-final_target) < .12 for p in players))
        assert all(len(sample) > 10 for sample in samples), samples
        assert all(max(b[0]-a[0] for a,b in zip(sample,sample[1:])) < .3 for sample in samples), samples
        assert max(callback_ms) < 100, callback_ms
        assert all(p.mpv.pause for p in players)
        print('Continuous reverse drag: frame updates', [len(sample) for sample in samples],
              'max input callback ms', round(max(callback_ms), 2), flush=True)
        print('Frames presented while dragging, SDR/HDR seconds:', positions,
              'mute/unmute while playing: passed', flush=True)
    finally:
        app._close()
