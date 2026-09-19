"""Native GUI smoke checks that must run through run_macos_gui_tests.py."""

import os
import sys
import tkinter as tk

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("SDR2HDR_MACOS_GUI_TEST_APP") != "1",
    reason="Requires the dedicated macOS Python.app GUI test runner",
)


def test_tk_window_can_process_an_update():
    root = tk.Tk()
    try:
        root.withdraw()
        root.update()
    finally:
        root.destroy()


def test_application_menu_uses_japanese_labels_and_own_name():
    from AppKit import NSApplication
    from sdr2hdr.gui import SDR2HDRGUI

    root = tk.Tk()
    try:
        root.withdraw()
        SDR2HDRGUI(root)
        root.deiconify()
        root.focus_force()
        root.update()
        menu = NSApplication.sharedApplication().mainMenu()
        assert [menu.itemAtIndex_(i).title() for i in range(menu.numberOfItems())] == [
            "SDR2HDR Pro", "ファイル", "編集", "ウインドウ", "ヘルプ"
        ]
        app_menu = menu.itemAtIndex_(0).submenu()
        labels = [app_menu.itemAtIndex_(i).title() for i in range(app_menu.numberOfItems())]
        assert "SDR2HDR Pro について" in labels
        assert "SDR2HDR Pro を終了" in labels
        assert "Python" not in " ".join(labels)
        assert root.tk.call("info", "commands", "::tk::mac::Quit")
    finally:
        root.destroy()


def test_libmpv_can_initialize_without_registering_a_cli_process():
    import mpv

    player = mpv.MPV(config=False, vo="null", ao="null", idle=True, pause=True)
    player.terminate()
