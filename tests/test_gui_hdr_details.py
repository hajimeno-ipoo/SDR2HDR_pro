"""Exercise the HDR controls through the real Tk form and queue configuration."""
import tkinter as tk
from tkinter import ttk

import pytest

from sdr2hdr.app import build_request_config, get_presets
from sdr2hdr.gui import AppState, SDR2HDRGUI


@pytest.fixture
def gui():
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    root.geometry("1680x900")
    root.update()
    try:
        yield app
    finally:
        app._close()


@pytest.mark.parametrize("index", [0, 2])
def test_details_default_preset_and_queue_settings(gui, tmp_path, index):
    gui.notebook.select(index)
    gui.root.update()
    for _, body in gui.hdr_detail_panels:
        assert not body.winfo_manager()
    default = build_request_config(gui._build_request())[0]
    assert default.peak_nits == get_presets()["natural"].peak_nits
    assert (default.hdr_guidance, default.luminance_guidance_strength, default.reconstruction_strength) == ("auto", .7, .6)

    source = tmp_path / ("input.mov" if index == 0 else "input.png")
    source.touch()
    (gui.input_var if index == 0 else gui.img_input_var).set(str(source))
    (gui.output_var if index == 0 else gui.img_output_var).set(str(tmp_path / ("output.mov" if index == 0 else "output.tif")))
    button, body = gui.hdr_detail_panels[index // 2]
    button.invoke()
    gui.root.update()
    controls = gui.hdr_detail_controls[index // 2 * 4:index // 2 * 4 + 4]
    peak, guidance, luminance, reconstruction = controls
    peak.delete(0, "end")
    peak.insert(0, "1200")
    luminance.set(.35)
    reconstruction.set(.25)
    for label, mode in (("ON", "on"), ("OFF", "off"), ("自動", "auto")):
        guidance.set(label)
        guidance.event_generate("<<ComboboxSelected>>")
        gui._enqueue_inputs()
        request = gui.queue_jobs[-1].request
        config = build_request_config(request)[0]
        assert (config.peak_nits, config.hdr_guidance, config.luminance_guidance_strength, config.reconstruction_strength) == (1200, mode, .35, .25)
    button.invoke()
    gui.root.update()
    assert not body.winfo_ismapped()
    assert gui._build_request().peak_nits == 1200
    gui.preset_var.set("poc")
    assert gui._build_request().peak_nits == get_presets()["poc"].peak_nits
    assert gui.queue_jobs[-1].request.peak_nits == 1200
    gui.notebook.select(2 if index == 0 else 0)
    assert gui._build_request().luminance_guidance_strength == .35


def test_invalid_peak_is_not_queued_and_log_modes_unchanged(gui):
    for invalid in ("", "bad", "0", "-1", "nan", "inf"):
        gui.peak_nits_var.set(invalid)
        with pytest.raises(ValueError, match="目標ピーク輝度"):
            gui._enqueue_inputs()
        assert not gui.queue_jobs
    for index in (1, 3):
        gui.notebook.select(index)
        assert not hasattr(gui._build_request(), "peak_nits")


def test_details_fit_and_preserve_other_panels_when_open_and_closed(gui):
    def bounds(widget):
        return (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height())

    fixed = (gui.compare_view, gui.queue_view.master.master, gui.add_queue_button)
    before = [bounds(widget) for widget in fixed]
    for index, form in ((0, gui.ai_form), (2, gui.img_ai_form)):
        gui.notebook.select(index)
        gui.root.update()
        button, body = gui.hdr_detail_panels[index // 2]
        canvas = form.master
        assert button.winfo_rooty() + button.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
        button.invoke()
        gui.root.update()
        canvas = form.master
        assert body.winfo_ismapped()
        assert [bounds(widget) for widget in fixed] == before
        assert body.winfo_rooty() >= canvas.winfo_rooty()
        assert body.winfo_rooty() + body.winfo_height() <= canvas.winfo_rooty() + canvas.winfo_height()
        for control in gui.hdr_detail_controls[index // 2 * 4:index // 2 * 4 + 4]:
            assert control.winfo_width() >= control.winfo_reqwidth()
            assert control.winfo_rootx() + control.winfo_width() <= canvas.winfo_rootx() + canvas.winfo_width()
        button.invoke()
        gui.root.update()
        assert not body.winfo_ismapped()
        assert [bounds(widget) for widget in fixed] == before
    for state in (AppState.RUNNING, AppState.EXPORTING, AppState.CANCELLING):
        gui._set_state(state)
        assert all(control.instate(["disabled"]) for control in gui.hdr_detail_controls)
    gui._set_state(AppState.IDLE)
    for control in gui.hdr_detail_controls:
        assert not control.instate(["disabled"])
        if isinstance(control, ttk.Combobox):
            assert control.instate(["readonly"])
