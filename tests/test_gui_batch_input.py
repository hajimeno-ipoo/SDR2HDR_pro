"""Tk integration for the single input route and per-file queue entries."""
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

import pytest

from sdr2hdr.gui import SDR2HDRGUI


@pytest.mark.parametrize('index', range(4))
def test_multiple_inputs_are_separate_jobs_and_single_input_restores_filename(tmp_path, index):
    root = tk.Tk()
    root.withdraw()
    try:
        gui = SDR2HDRGUI(root)
        gui.notebook.select(index)
        root.update()
        keys = ['video_ai','video_log','image_ai','image_log']
        key = keys[index]
        inputs = [gui.input_var,gui.log_input_var,gui.img_input_var,gui.img_log_input_var]
        outputs = [gui.output_var,gui.log_output_var,gui.img_output_var,gui.img_log_output_var]
        browsers = [gui._browse_input,gui._browse_log_input,gui._browse_img_input,gui._browse_img_log_input]
        source_a = tmp_path/'a'/'same.png'
        source_b = tmp_path/'b'/'same.png'
        for source in (source_a,source_b):
            source.parent.mkdir(); source.touch()
        folder = tmp_path/'destination.with.dots'; folder.mkdir()
        outputs[index].set(str(folder/'custom.tif'))
        with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(source_a),str(source_b))):
            browsers[index]()
        assert outputs[index].get() == str(folder)
        assert gui._path_labels[str(outputs[index])][0].cget('text') == '保存先フォルダー'
        assert not hasattr(gui, 'add_files_button')
        if index >= 2:
            (gui.img_format_var if index == 2 else gui.img_log_format_var).set(gui.img_format_options['.png'])
        else:
            (gui.encoder_var if index == 0 else gui.log_encoder_var).set(gui.encoder_options['libx265'])
        assert outputs[index].get() == str(folder)
        gui._enqueue_inputs()
        assert len(gui.queue_jobs) == 2
        assert len(gui.queue_view.get_children()) == 2
        requests = [job.request for job in gui.queue_jobs]
        assert [r.input_path for r in requests] == [str(source_a),str(source_b)]
        assert all(Path(r.output_path).parent == folder for r in requests)
        assert requests[0].output_path != requests[1].output_path
        gui.queue_view.selection_set('0')
        gui._remove_selected_job()
        assert len(gui.queue_jobs) == 1 and gui.queue_jobs[0].request.input_path == str(source_b)
        with patch('sdr2hdr.gui.filedialog.askopenfilenames',return_value=(str(source_a),)):
            browsers[index]()
        assert key not in gui._input_batches
        assert gui._path_labels[str(outputs[index])][0].cget('text') != '保存先フォルダー'
        assert Path(outputs[index].get()).parent == folder
    finally:
        root.destroy()


@pytest.mark.parametrize('index', range(4))
def test_sequential_inputs_replace_filename_keep_folder_and_reject_duplicate(tmp_path, index):
    root = tk.Tk()
    gui = SDR2HDRGUI(root)
    try:
        gui.notebook.select(index)
        root.update()
        inputs = [gui.input_var, gui.log_input_var, gui.img_input_var, gui.img_log_input_var]
        outputs = [gui.output_var, gui.log_output_var, gui.img_output_var, gui.img_log_output_var]
        browsers = [gui._browse_input, gui._browse_log_input, gui._browse_img_input, gui._browse_img_log_input]
        saves = [gui._browse_output, gui._browse_log_output, gui._browse_img_output, gui._browse_img_log_output]
        if index < 2:
            (gui.encoder_var if index == 0 else gui.log_encoder_var).set(gui.encoder_options['prores_422hq'])
        else:
            (gui.img_format_var if index == 2 else gui.img_log_format_var).set(gui.img_format_options['.png'])
        extension = '.mov' if index < 2 else '.png'
        destination = tmp_path / 'chosen.folder' / ('custom.first' + extension)
        source_a, source_b, source_c = (tmp_path / name for name in ('first.mp4', 'second.mp4', 'third.mp4'))
        for source in (source_a, source_b, source_c):
            source.touch()
        with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(source_a),)):
            browsers[index]()
        with patch('sdr2hdr.gui.filedialog.asksaveasfilename', return_value=str(destination)):
            saves[index]()
        # Selecting the same source must not throw away its custom filename.
        with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(source_a),)):
            browsers[index]()
        assert outputs[index].get() == str(destination)
        gui._enqueue_inputs()
        for source in (source_b, source_c):
            with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(source),)):
                browsers[index]()
            assert outputs[index].get() == str(destination.with_name(source.stem + '_hdr' + extension))
            gui._enqueue_inputs()
        assert [job.request.input_path for job in gui.queue_jobs] == [str(p) for p in (source_a, source_b, source_c)]
        assert [Path(job.request.output_path).name for job in gui.queue_jobs] == [destination.name, 'second_hdr' + extension, 'third_hdr' + extension]
        assert all(not variable.get() for i, variable in enumerate(inputs) if i != index)
        # Manually reusing the first destination cannot add an overwriting job.
        outputs[index].set(str(destination))
        gui._enqueue_current()
        assert len(gui.queue_jobs) == 3
        assert '同じ保存先' in gui.feedback_var.get()
    finally:
        gui._close()
