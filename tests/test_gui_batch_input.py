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
