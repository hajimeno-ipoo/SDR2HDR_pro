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


def test_libmpv_can_initialize_without_registering_a_cli_process():
    import mpv

    player = mpv.MPV(config=False, vo="null", ao="null", idle=True, pause=True)
    player.terminate()
