"""Check that the redesigned workspace keeps controls reachable."""
import tkinter as tk
from sdr2hdr.gui import SDR2HDRGUI


def test_workspace_tabs_and_actions_fit_minimum_window():
    root = tk.Tk()
    try:
        app = SDR2HDRGUI(root)
        root.update()
        left = app.notebook.master
        right = app.queue_view.master.master
        assert left.winfo_rooty() + left.winfo_height() == right.winfo_rooty() + right.winfo_height()
        root.geometry('1280x900')
        root.update()
        for tab in (app.ai_tab, app.log_tab, app.img_ai_tab, app.img_log_tab):
            app.notebook.select(tab)
            root.update()
            assert app.notebook.nametowidget(app.notebook.select()) is tab
            for child in tab.winfo_children():
                if child.winfo_ismapped():
                    assert child.winfo_rooty() + child.winfo_height() <= tab.winfo_rooty() + tab.winfo_height()
            for widget in (tab, app.add_queue_button, app.start_button, app.open_folder_button, app.log):
                assert widget.winfo_ismapped()
                assert widget.winfo_rooty() + widget.winfo_height() <= root.winfo_rooty() + root.winfo_height()
                assert widget.winfo_rootx() + widget.winfo_width() <= root.winfo_rootx() + root.winfo_width()
        button = app.add_queue_button
        button.event_generate('<ButtonPress-1>', x=5, y=5)
        root.update()
        button.event_generate('<Leave>')
        root.after(250, root.quit)
        root.mainloop()
        assert not button.cget('padding')
    finally:
        root.destroy()


def test_feedback_interrupt_and_reduced_motion():
    root = tk.Tk()
    try:
        app = SDR2HDRGUI(root)
        root.update()
        app.queue_view.insert('', 'end', iid='cue', values=('待機中', 'input', 'output'))
        app._feedback('added', '追加しました', 'cue')
        assert app.queue_view.item('cue', 'tags') == ('recent',)
        app._feedback('complete', '保存が完了しました')
        assert not app.queue_view.item('cue', 'tags')
        assert app.open_output_button.cget('style') == 'Accent.TButton'
        app.reduce_motion_var.set(True)
        assert app.feedback_motion.pending is None
        assert app.result_var.get() == '保存が完了しました'
        app._feedback('error', '入力ファイルを指定してください')
        assert app.feedback_motion.pending is None
        assert app.feedback_var.get() == '入力ファイルを指定してください'
        app.reduce_motion_var.set(False)
        app._feedback('start', '変換を開始しました')
        root.after(400, root.quit)
        root.mainloop()
        assert app.feedback_motion.pending is None
        assert app.status_label.pack_info()['padx'] == 0
    finally:
        root.destroy()


def test_completed_queue_details_and_reduced_addition():
    from types import SimpleNamespace
    from sdr2hdr.gui import QueueJob, AppState
    root = tk.Tk()
    try:
        app = SDR2HDRGUI(root)
        app.queue_jobs = [QueueJob(SimpleNamespace(input_path='/tmp/long-input.mov', output_path='/tmp/output.mov'), status='failed', error='保存先へ書き込めません')]
        app._refresh_job_list()
        app.queue_view.selection_set('0')
        app._show_job_detail()
        assert '/tmp/output.mov' in app.job_detail.get('1.0', 'end')
        assert '保存先へ書き込めません' in app.job_detail.get('1.0', 'end')
        app._set_state(AppState.COMPLETED)
        assert app.start_button.instate(['disabled'])
        app.reduce_motion_var.set(True)
        app._feedback('added', '1件追加', '0')
        assert app.queue_view.item('0', 'tags') == ('recent',)
        assert app.feedback_motion.pending is None
        app._feedback('start', '開始')
        assert '保存先へ書き込めません' in app.job_detail.get('1.0', 'end')
    finally:
        root.destroy()


def test_button_motion_preserves_surrounding_geometry():
    root = tk.Tk()
    try:
        app = SDR2HDRGUI(root)
        root.update()
        slot = app.open_output_button.motion_slot
        nearby = (app.start_button, app.stop_button, app.log, app.status_label)
        def bounds(widget):
            return (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height())
        before = [bounds(w) for w in nearby]
        moved = []
        slot.bounce()
        for delay in (50, 100, 150, 200, 250):
            root.after(delay, lambda: moved.append((app.open_output_button.winfo_y(), [bounds(w) for w in nearby])))
        root.after(360, root.quit)
        root.mainloop()
        assert any(y != 6 for y, _ in moved)
        assert all(rects == before for _, rects in moved)
        assert app.open_output_button.winfo_y() == 6
        app.reduce_motion_var.set(True)
        slot.bounce()
        assert slot.pending is None
    finally:
        root.destroy()
