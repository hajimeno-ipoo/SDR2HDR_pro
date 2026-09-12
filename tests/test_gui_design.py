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
        root.geometry('1680x900')
        root.update()
        hero = next(w for w in app.compare_view.master.winfo_children() if w.winfo_class() == 'Frame')
        for label in hero.winfo_children():
            assert label.winfo_ismapped()
            assert label.winfo_width() >= label.winfo_reqwidth()
            assert label.winfo_x() + label.winfo_width() <= hero.winfo_width()
        for tab in (app.ai_tab, app.log_tab, app.img_ai_tab, app.img_log_tab):
            app.notebook.select(tab)
            root.update()
            assert app.notebook.nametowidget(app.notebook.select()) is tab
            for child in tab.winfo_children():
                if child.winfo_ismapped():
                    assert child.winfo_rooty() + child.winfo_height() <= tab.winfo_rooty() + tab.winfo_height()
            for widget in (tab, app.add_queue_button, app.start_button, app.open_folder_button, app.export_button, app.log_button):
                assert widget.winfo_ismapped()
                parent = widget.master.master if hasattr(widget, "motion_slot") else widget.master
                assert widget.winfo_rooty() + widget.winfo_height() <= parent.winfo_rooty() + parent.winfo_height()
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
        assert app.export_button.cget('style') == 'Accent.TButton'
        app.reduce_motion_var.set(True)
        assert app.feedback_motion.pending is None
        assert app.result_var.get() == '保存が完了しました'
        app._feedback('error', '入力ファイルを指定してください')
        assert app.feedback_motion.pending is None
        assert app.feedback_var.get() == '入力ファイルを指定してください'
        app.reduce_motion_var.set(False)
        app._feedback('start', '変換を開始しました')
        # Tk's after delay is a minimum, not a completion deadline. Check that
        # the cue finishes and releases its timer, independent of redraw cost.
        def finish_when_idle():
            if app.feedback_motion.pending is None:
                root.quit()
            else:
                root.after(20, finish_when_idle)
        timeout = root.after(2000, root.quit)
        root.after(20, finish_when_idle)
        root.mainloop()
        root.after_cancel(timeout)
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
        slot = app.export_button.motion_slot
        nearby = (app.start_button, app.stop_button, app.log_button, app.status_label)
        def bounds(widget):
            return (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_width(), widget.winfo_height())
        before = [bounds(w) for w in nearby]
        moved = []
        slot.bounce()
        for delay in (50, 100, 150, 200, 250):
            root.after(delay, lambda: moved.append((app.export_button.winfo_y(), [bounds(w) for w in nearby])))
        root.after(360, root.quit)
        root.mainloop()
        assert any(y != 6 for y, _ in moved)
        assert all(rects == before for _, rects in moved)
        assert app.export_button.winfo_y() == 6
        app.reduce_motion_var.set(True)
        slot.bounce()
        assert slot.pending is None
    finally:
        root.destroy()


def test_completion_keeps_export_button_inside_slot_and_details_above_actions():
    from types import SimpleNamespace
    from sdr2hdr.gui import QueueJob, AppState
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    try:
        root.geometry('1680x900')
        root.update()
        before = app.export_button.motion_slot.winfo_rooty()
        app.queue_jobs = [QueueJob(SimpleNamespace(input_path='/tmp/input.mov', output_path='/tmp/output.mov'), status='completed', preview_path='/tmp/preview.mov')]
        app._refresh_job_list()
        app.queue_view.selection_set('0')
        app._show_job_detail()
        app.progress_var.set('243 フレーム変換しました')
        app._set_state(AppState.COMPLETED)
        app.reduce_motion_var.set(True)
        root.update()
        assert app.export_button.motion_slot.winfo_rooty() == before
        for widget in (app.export_button, app.open_output_button, app.open_folder_button, app.start_button, app.job_detail, app.queue_view):
            parent = widget.master
            assert widget.winfo_rooty() >= parent.winfo_rooty()
            assert widget.winfo_rooty()+widget.winfo_height() <= parent.winfo_rooty()+parent.winfo_height()
            assert widget.winfo_rooty()+widget.winfo_height() <= root.winfo_rooty()+root.winfo_height()
        assert app.queue_view.winfo_rooty()+app.queue_view.winfo_height() <= app.job_detail.winfo_rooty()
        outer = app.compare_view.master
        hero = next(w for w in outer.winfo_children() if w.winfo_class() == 'Frame')
        assert hero.winfo_rootx()+hero.winfo_width() == app.compare_view.winfo_rootx()+app.compare_view.winfo_width()
        assert app.export_button.winfo_width() > app.start_button.winfo_width()
        assert app.export_button.winfo_height() >= 44
        app.log_button.invoke(); root.update()
        assert app.log.winfo_ismapped()
        assert app.export_button.motion_slot.winfo_rooty() == before
    finally:
        app._close()
