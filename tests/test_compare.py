"""Requirements of the comparison controller and pair registration."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk
import pytest

from sdr2hdr.compare_controller import CompareController
from sdr2hdr.compare_view import CompareView, color_label
from sdr2hdr.mpv_player import MpvPlayer


def test_player_initialization_failure_releases_surface():
    surface = Mock()
    surface.player_options.return_value = {}
    module = SimpleNamespace(MPV=Mock(side_effect=RuntimeError('initialization failed')))
    with patch.dict('sys.modules', mpv=module), pytest.raises(RuntimeError):
        MpvPlayer(surface, hdr=True)
    surface.close.assert_called_once()


def test_render_initialization_failure_releases_player_and_surface():
    surface = Mock()
    surface.player_options.return_value = {}
    surface.attach.side_effect = RuntimeError('render initialization failed')
    player = Mock()
    module = SimpleNamespace(MPV=Mock(return_value=player))
    with patch.dict('sys.modules', mpv=module), pytest.raises(RuntimeError):
        MpvPlayer(surface, hdr=True)
    surface.close.assert_called_once()
    player.terminate.assert_called_once()


def players():
    result = []
    for position in (1.0, 1.0):
        p = Mock(loaded=True, seeking=False, eof_reached=False)
        p.get_position.return_value = position
        result.append(p)
    return result


def test_small_difference_is_not_corrected_and_large_difference_is_throttled():
    sdr, hdr = players()
    c = CompareController(sdr, hdr)
    c.play()
    with patch('sdr2hdr.compare_controller.time.monotonic', return_value=10):
        hdr.get_position.return_value = .97
        assert c.sync_tick() is None
        hdr.seek_absolute.assert_not_called()
        hdr.get_position.return_value = .8
        assert c.sync_tick() is None
        hdr.seek_absolute.assert_called_once_with(1.0)
        c.sync_tick()
        hdr.seek_absolute.assert_called_once()
    with patch('sdr2hdr.compare_controller.time.monotonic', return_value=13.1):
        assert 'ずれ' in c.sync_tick()
    hdr.get_position.return_value = 1
    assert c.sync_tick() is None


def test_images_do_not_play_or_synchronize():
    sdr, hdr = players()
    c = CompareController(sdr, hdr)
    c.load_pair('sdr.png', 'hdr.png', {}, {}, image=True)
    c.play()
    c.seek(3)
    c.frame_step(1)
    assert c.sync_tick() is None
    for p in (sdr, hdr):
        p.play.assert_not_called()
        p.seek_absolute.assert_not_called()
        p.step_frame.assert_not_called()


def test_eof_pauses_both_players():
    sdr, hdr = players()
    c = CompareController(sdr, hdr)
    c.play()
    hdr.eof_reached = True
    c.sync_tick()
    assert not c.playing
    sdr.pause.assert_called_once()
    hdr.pause.assert_called_once()


def test_close_attempts_both_players_even_when_first_fails():
    sdr, hdr = players()
    sdr.close.side_effect = RuntimeError('close error')
    c = CompareController(sdr, hdr)
    import pytest
    with pytest.raises(RuntimeError):
        c.close()
    hdr.close.assert_called_once()


def test_unknown_metadata_is_not_labeled_hdr():
    assert color_label({}) == 'Unknown / Unknown / Unknown'
    assert color_label({'color_primaries': 'bt2020', 'color_transfer': 'smpte2084', 'bit_depth': 10}) == 'BT.2020 / PQ / 10-bit'


def test_registered_jobs_do_not_replace_selected_pair_and_exr_is_excluded():
    root = tk.Tk()
    try:
        view = CompareView(root)
        view.pack()
        with patch.object(view, '_select') as select:
            view.add_pair('a.mp4', 'a.mov')
            select.assert_called_once()
            view.add_pair('b.mp4', 'b.mov')
            view.add_pair('a.mp4', 'a.mov')
            view.add_pair('c.mp4', 'c.zip')
            view.add_pair('d.mp4', 'd.exr')
            assert len(view.pairs) == 2
            assert view.selection.current() == 0
            select.assert_called_once()
    finally:
        root.destroy()


def test_four_job_modes_register_only_successful_results():
    import queue
    from sdr2hdr.gui import SDR2HDRGUI, QueueJob
    from sdr2hdr.app import (ConversionRequest, LogConversionRequest,
                            ImageConversionRequest, ImageLogConversionRequest, ConversionResult)
    for request_type, image in ((ConversionRequest, False), (LogConversionRequest, False),
                                (ImageConversionRequest, True), (ImageLogConversionRequest, True)):
        for cancelled in (False, True):
            app = SDR2HDRGUI.__new__(SDR2HDRGUI)
            app.root = Mock()
            app.queue_jobs = [QueueJob(request_type('source', 'requested-output'))]
            app.current_job_index = 0
            app.event_queue = queue.Queue()
            app.event_queue.put(('complete', ConversionResult('actual-output', 1, 1, cancelled)))
            for name in ('progress', 'progress_var', 'status_var', 'compare_view', '_set_job_status',
                         '_finish_current_job', '_set_state', '_log'):
                setattr(app, name, Mock())
            app._next_pending_job_index = Mock(return_value=1)
            app._start_job = Mock()
            app._drain_events()
            if cancelled:
                app.compare_view.add_pair.assert_not_called()
                app._start_job.assert_not_called()
            else:
                app.compare_view.add_pair.assert_called_once_with('source', 'actual-output', image=image)
                app._start_job.assert_called_once_with(1)


def test_removing_pairs_preserves_other_selection_and_discards_late_probe():
    root = tk.Tk()
    try:
        view = CompareView(root)
        view.pack()
        with patch.object(view, '_select'):
            view.add_pair('a.png', 'a_hdr.png', image=True)
            view.add_pair('b.png', 'b_hdr.png', image=True)
        view.selection.current(1)
        controller = Mock(image=True)
        view.controller = controller
        view._loaded_pair = view.pairs[1]
        view.remove_pairs(['a_hdr.png'])
        controller.close.assert_not_called()
        assert view.selection.current() == 0 and view.pairs[0].hdr_path == 'b_hdr.png'
        late_pair = view.pairs[0]
        generation = view._generation
        view.remove_pairs(['b_hdr.png'])
        controller.close.assert_called_once()
        assert view.controller is None and not view.pairs and not view.selection.get()
        view._results.put((generation, late_pair, ({}, {})))
        with patch.object(view, '_load') as load:
            view._tick()
            load.assert_not_called()
        assert view.sdr_info.get() == '変換前' and view.hdr_info.get() == '変換後'
        with patch.object(view, '_select') as select:
            view.add_pair('c.png', 'c_hdr.png', image=True)
            select.assert_called_once()
    finally:
        root.destroy()


def test_removing_loaded_pair_during_another_selection_releases_old_player():
    root = tk.Tk()
    try:
        view = CompareView(root)
        with patch.object(view, '_select'):
            view.add_pair('a.png', 'a_hdr.png', image=True)
            view.add_pair('b.png', 'b_hdr.png', image=True)
        controller = Mock(image=True)
        view.controller = controller
        view._loaded_pair = view.pairs[0]
        view.selection.current(1)
        generation = view._generation
        view.remove_pairs(['a_hdr.png'])
        controller.close.assert_called_once()
        assert view._generation == generation  # The pending probe for B remains valid.
        assert view.pairs[view.selection.current()].hdr_path == 'b_hdr.png'
    finally:
        root.destroy()


def test_drag_is_disabled_at_fit_and_clamps_to_image_edges():
    from types import SimpleNamespace
    view = CompareView.__new__(CompareView)
    view._pan_drag = None
    view.fit_button = Mock()
    view.fit_button.instate.return_value = False
    view.sdr_surface = Mock()
    view.sdr_surface.winfo_width.return_value = 400
    view.sdr_surface.winfo_height.return_value = 200
    view.hdr_surface = Mock()
    player = SimpleNamespace(get_dimensions=lambda: {'w':800,'h':400,'ml':-400,'mr':-400,'mt':-200,'mb':-200})
    view.controller = SimpleNamespace(zoom=1,pan=(0,0),sdr=player,hdr=player)
    view._operate = Mock()
    event = SimpleNamespace(widget=view.sdr_surface,x_root=100,y_root=100)
    view._pan_start(event)
    assert view._pan_drag is None
    view.controller.zoom = 2
    view._pan_start(event)
    event.x_root = event.y_root = 10000
    view._pan_move(event)
    view._operate.assert_called_with('set_pan',.25,.25)
    event.x_root = event.y_root = -10000
    view._pan_move(event)
    view._operate.assert_called_with('set_pan',-.25,-.25)
    view._pan_end()
    assert view._pan_drag is None
