#!/usr/bin/env python3
"""Run pytest through Python.app so macOS GUI frameworks get an app context."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import sysconfig
import tempfile
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_FLAG = "--python-app-worker"


def _python_app() -> Path:
    return Path(sys.base_prefix) / "Resources" / "Python.app"


def _run_worker(arguments: list[str]) -> int:
    if len(arguments) < 5 or arguments[4] != "--":
        print("GUI test worker received invalid arguments.", file=sys.stderr)
        return 2

    project_root = Path(arguments[0]).resolve()
    site_packages = Path(arguments[1]).resolve()
    result_path = Path(arguments[2]).resolve()
    log_path = Path(arguments[3]).resolve()
    pytest_arguments = arguments[5:]

    site.addsitedir(str(site_packages))
    site_path = str(site_packages)
    if site_path in sys.path:
        sys.path.remove(site_path)
    sys.path.insert(0, site_path)
    sys.path.insert(0, str(project_root / "src"))
    os.chdir(project_root)
    os.environ["SDR2HDR_MACOS_GUI_TEST_APP"] = "1"

    return_code = 2
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = log
        try:
            import pytest

            return_code = int(pytest.main(pytest_arguments))
        except BaseException:
            traceback.print_exc()
        finally:
            log.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr

    temporary_result = result_path.with_suffix(".tmp")
    temporary_result.write_text(
        json.dumps({"return_code": return_code}), encoding="utf-8"
    )
    temporary_result.replace(result_path)
    return return_code


def _launch_test(python_app: Path, site_packages: Path, node_id: str) -> int:
    with tempfile.TemporaryDirectory(prefix="sdr2hdr-gui-tests-") as directory:
        temporary_dir = Path(directory)
        result_path = temporary_dir / "result.json"
        log_path = temporary_dir / "pytest.log"
        command = [
            "/usr/bin/open",
            "-n",
            "-W",
            str(python_app),
            "--args",
            str(Path(__file__).resolve()),
            WORKER_FLAG,
            str(PROJECT_ROOT),
            str(site_packages),
            str(result_path),
            str(log_path),
            "--",
            node_id,
            "-q",
        ]
        launch = subprocess.run(command, check=False)

        if log_path.exists():
            print(log_path.read_text(encoding="utf-8"), end="")
        if launch.returncode != 0:
            print(
                f"Could not launch Python.app (open exited {launch.returncode}).",
                file=sys.stderr,
            )
            return 2
        if not result_path.exists():
            print(
                "Python.app ended before the GUI test result was written.",
                file=sys.stderr,
            )
            return 2

        result = json.loads(result_path.read_text(encoding="utf-8"))
        return int(result["return_code"])


class _NodeCollector:
    def __init__(self) -> None:
        self.node_ids: list[str] = []

    def pytest_collection_finish(self, session) -> None:
        self.node_ids = [item.nodeid for item in session.items]


def _collect_node_ids(pytest_arguments: list[str]) -> tuple[int, list[str]]:
    import pytest

    collector = _NodeCollector()
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        return_code = int(
            pytest.main([*pytest_arguments, "--collect-only"], plugins=[collector])
        )
    if return_code != 0:
        print(output.getvalue(), end="", file=sys.stderr)
    return return_code, collector.node_ids


def _run_controller(pytest_arguments: list[str]) -> int:
    if sys.platform != "darwin":
        print("This runner is only for macOS GUI tests.", file=sys.stderr)
        return 2
    if not pytest_arguments:
        print(
            "Usage: python scripts/run_macos_gui_tests.py <pytest arguments>",
            file=sys.stderr,
        )
        return 2

    python_app = _python_app()
    if not python_app.is_dir():
        print(f"Python.app was not found: {python_app}", file=sys.stderr)
        return 2

    collection_code, node_ids = _collect_node_ids(pytest_arguments)
    if collection_code != 0:
        return collection_code
    if not node_ids:
        print("No tests were selected.", file=sys.stderr)
        return 5

    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    final_code = 0
    for index, node_id in enumerate(node_ids, start=1):
        print(f"[{index}/{len(node_ids)}] {node_id}")
        return_code = _launch_test(python_app, site_packages, node_id)
        if return_code != 0 and final_code == 0:
            final_code = return_code
        if return_code == 2:
            break
    return final_code


def main() -> int:
    arguments = sys.argv[1:]
    if arguments and arguments[0] == WORKER_FLAG:
        return_code = _run_worker(arguments[1:])
        os._exit(return_code)
    return _run_controller(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
