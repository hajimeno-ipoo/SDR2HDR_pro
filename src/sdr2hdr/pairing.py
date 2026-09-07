"""Build aligned SDR/HDR video pairs from a CSV manifest."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _clean(value: str | None) -> str:
    return (value or "").strip().lstrip("\ufeff")


def _safe_manifest_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest video_name must be relative to its root: {name}")
    return root / relative


@dataclass(frozen=True)
class ManifestVideoPair:
    """One HDR/SDR pair identified by the manifest metadata."""

    content_name: str
    resolution: str
    bitrate: str
    hdr_path: Path
    sdr_path: Path
    hdr_name: str
    sdr_name: str


def load_manifest_video_pairs(
    manifest_path: str | Path,
    hdr_dir: str | Path,
    sdr_dir: str | Path,
    *,
    dataset_type: str = "Open-source",
    hdr_format: str = "HDR10",
    sdr_format: str = "SDR",
) -> list[ManifestVideoPair]:
    """Load unique HDR/SDR pairs from a CSV manifest.

    Rows are paired by ``content_name``, ``resolution`` and ``bitrate``.  The
    exact filenames from ``video_name`` are used for the returned paths, so
    the source files never need to be renamed or guessed from one another.
    """

    manifest = Path(manifest_path)
    hdr_root = Path(hdr_dir)
    sdr_root = Path(sdr_dir)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not hdr_root.is_dir():
        raise NotADirectoryError(f"HDR directory does not exist: {hdr_root}")
    if not sdr_root.is_dir():
        raise NotADirectoryError(f"SDR directory does not exist: {sdr_root}")

    requested_type = _clean(dataset_type).casefold()
    requested_hdr = _clean(hdr_format).casefold()
    requested_sdr = _clean(sdr_format).casefold()
    by_format: dict[str, dict[tuple[str, str, str], tuple[str, Path]]] = {
        requested_hdr: {},
        requested_sdr: {},
    }

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {manifest}")
        fields = {_clean(name): name for name in reader.fieldnames}
        required = {"video_name", "video_format", "content_name", "resolution", "bitrate", "Type"}
        missing_fields = sorted(required - fields.keys())
        if missing_fields:
            raise ValueError(f"manifest is missing required columns: {', '.join(missing_fields)}")

        for row_number, row in enumerate(reader, start=2):
            values = {name: _clean(row.get(column)) for name, column in fields.items()}
            if values["Type"].casefold() != requested_type:
                continue
            video_format = values["video_format"].casefold()
            if video_format not in by_format:
                continue
            key = (values["content_name"], values["resolution"], values["bitrate"])
            if not all(key) or not values["video_name"]:
                raise ValueError(f"manifest row {row_number} has an incomplete pairing key")
            if key in by_format[video_format]:
                previous = by_format[video_format][key][0]
                raise ValueError(
                    f"duplicate {values['video_format']} manifest pair at row {row_number}: "
                    f"{previous} and {values['video_name']}"
                )
            root = hdr_root if video_format == requested_hdr else sdr_root
            by_format[video_format][key] = (
                values["video_name"],
                _safe_manifest_path(root, values["video_name"]),
            )

    hdr_pairs = by_format[requested_hdr]
    sdr_pairs = by_format[requested_sdr]
    missing_sdr = sorted(set(hdr_pairs) - set(sdr_pairs))
    missing_hdr = sorted(set(sdr_pairs) - set(hdr_pairs))
    if missing_sdr or missing_hdr:
        details: list[str] = []
        if missing_sdr:
            details.append(f"missing SDR pairs: {len(missing_sdr)}")
        if missing_hdr:
            details.append(f"missing HDR pairs: {len(missing_hdr)}")
        raise ValueError(f"manifest has unpaired records ({'; '.join(details)})")

    pairs: list[ManifestVideoPair] = []
    for content_name, resolution, bitrate in sorted(hdr_pairs):
        hdr_name, hdr_path = hdr_pairs[(content_name, resolution, bitrate)]
        sdr_name, sdr_path = sdr_pairs[(content_name, resolution, bitrate)]
        if not hdr_path.is_file():
            raise FileNotFoundError(f"HDR file listed by manifest does not exist: {hdr_path}")
        if not sdr_path.is_file():
            raise FileNotFoundError(f"SDR file listed by manifest does not exist: {sdr_path}")
        pairs.append(
            ManifestVideoPair(
                content_name=content_name,
                resolution=resolution,
                bitrate=bitrate,
                hdr_path=hdr_path,
                sdr_path=sdr_path,
                hdr_name=hdr_name,
                sdr_name=sdr_name,
            )
        )
    return pairs
