"""GUI-owned previews and explicit, byte-preserving delivery."""
from pathlib import Path
import tempfile


class PreviewOutputs:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sdr2hdr-preview-")

    def allocate(self, destination: str) -> str:
        folder = Path(tempfile.mkdtemp(dir=self.directory.name))
        return str(folder / Path(destination).name)

    def close(self):
        self.directory.cleanup()


def export_preview(source: str, destination: str, cancel_token, progress=None) -> bool:
    """Copy without re-encoding; cancellation/failure leaves the destination intact."""
    source, destination = Path(source), Path(destination)
    size = source.stat().st_size
    if not source.is_file() or not size:
        raise ValueError("書き出せる変換結果がありません。")
    if cancel_token.cancel_requested:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sdr2hdr-export-", dir=destination.parent) as folder:
        working = Path(folder) / destination.name
        copied = 0
        with source.open("rb") as reader, working.open("wb") as writer:
            while chunk := reader.read(4 * 1024 * 1024):
                if cancel_token.cancel_requested:
                    return False
                writer.write(chunk)
                copied += len(chunk)
                if progress:
                    progress(copied, size)
        if cancel_token.cancel_requested:
            return False
        working.replace(destination)
    return True
