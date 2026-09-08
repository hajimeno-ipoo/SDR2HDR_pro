"""Update HEVC static HDR SEI without decoding or encoding picture data."""
from __future__ import annotations

import struct
from pathlib import Path

import av


def _escape(raw: bytes) -> bytes:
    result = bytearray()
    zeros = 0
    for value in raw:
        if zeros >= 2 and value <= 3:
            result.append(3)
            zeros = 0
        result.append(value)
        zeros = zeros + 1 if value == 0 else 0
    return bytes(result)


def _strip_static_sei(nal: bytes) -> bytes:
    if len(nal) < 2:
        raise ValueError("Truncated HEVC NAL header")
    if (nal[0] >> 1) & 63 not in (39, 40):
        return nal
    raw = nal[2:].replace(b"\x00\x00\x03", b"\x00\x00")
    kept = bytearray()
    pos = 0
    while pos < len(raw) and raw[pos:] != b"\x80":
        start = pos
        values = []
        for _ in range(2):
            value = 0
            while True:
                if pos >= len(raw):
                    raise ValueError("Truncated HEVC SEI header")
                byte = raw[pos]
                pos += 1
                value += byte
                if byte != 255:
                    break
            values.append(value)
        kind, size = values
        if pos + size > len(raw):
            raise ValueError("Truncated HEVC SEI payload")
        pos += size
        if kind not in (137, 144):
            kept.extend(raw[start:pos])
    if raw[pos:] != b"\x80":
        raise ValueError("Invalid HEVC SEI trailing bits")
    return nal[:2] + _escape(bytes(kept) + b"\x80") if kept else b""


def _clean_config(config: bytes) -> tuple[bytes, int]:
    # ISO/IEC 14496-15 HEVCDecoderConfigurationRecord (hvcC).
    if len(config) < 23 or config[0] != 1:
        raise ValueError("Expected MP4/MOV HEVC configuration")
    length_size = (config[21] & 3) + 1
    result = bytearray(config[:23])
    pos = 23
    arrays = 0
    for _ in range(config[22]):
        if pos + 3 > len(config):
            raise ValueError("Truncated HEVC configuration array")
        header = config[pos]
        count = int.from_bytes(config[pos + 1:pos + 3], "big")
        pos += 3
        kept = []
        for _ in range(count):
            if pos + 2 > len(config):
                raise ValueError("Truncated HEVC configuration NAL length")
            size = int.from_bytes(config[pos:pos + 2], "big")
            pos += 2
            if pos + size > len(config):
                raise ValueError("Truncated HEVC configuration NAL")
            nal = _strip_static_sei(config[pos:pos + size])
            pos += size
            if nal:
                kept.append(len(nal).to_bytes(2, "big") + nal)
        if kept:
            result.extend(bytes([header]) + len(kept).to_bytes(2, "big") + b"".join(kept))
            arrays += 1
    if pos != len(config):
        raise ValueError("Unexpected HEVC configuration data")
    result[22] = arrays
    return bytes(result), length_size


def _update_packet(data: bytes, length_size: int, sei: bytes, keyframe: bool) -> bytes:
    result = bytearray()
    pos = 0
    inserted = False
    while pos < len(data):
        if pos + length_size > len(data):
            raise ValueError("Truncated HEVC packet NAL length")
        size = int.from_bytes(data[pos:pos + length_size], "big")
        pos += length_size
        if size < 2 or pos + size > len(data):
            raise ValueError("Invalid HEVC packet NAL size")
        nal = data[pos:pos + size]
        pos += size
        if (nal[0] >> 1) & 63 <= 31 and keyframe and not inserted:
            result.extend(len(sei).to_bytes(length_size, "big") + sei)
            inserted = True
        nal = _strip_static_sei(nal)
        if nal:
            result.extend(len(nal).to_bytes(length_size, "big") + nal)
    return bytes(result)


def remux_hdr_metadata(source: Path, output: Path, max_cll: int, max_fall: int) -> None:
    """Copy compressed video/audio packets, replacing only static HDR SEI.

    Mastering values match the application's existing HDR10 encoder settings.
    SEI 137/144 syntax follows FFmpeg's cbs_sei_syntax_template.c definitions.
    """
    cll, fall = max(int(max_cll), 1), max(int(max_fall), 1)
    if cll > 65535 or fall > 65535:
        raise ValueError("HDR content light levels exceed the HEVC uint16 range")
    mastering = struct.pack(">8H2I", 13250, 34500, 7500, 3000, 34000, 16000,
                            15635, 16450, 10000000, 1)
    sei = b"\x4e\x01" + _escape(
        bytes([137, 24]) + mastering + bytes([144, 4]) + struct.pack(">HH", cll, fall) + b"\x80"
    )
    with av.open(str(source)) as input_file, av.open(
        str(output), "w", options={"movflags": "+faststart"}
    ) as output_file:
        output_file.metadata.update(input_file.metadata)
        streams = {}
        lengths = {}
        for stream in input_file.streams:
            if stream.type not in ("video", "audio"):
                continue  # Same stream selection as the previous FFmpeg command.
            target = output_file.add_stream_from_template(stream, opaque=True)
            target.time_base = stream.time_base
            target.metadata.update(stream.metadata)
            streams[stream.index] = target
            if stream.type == "video":
                if stream.codec_context.name != "hevc":
                    raise ValueError("HDR metadata update requires HEVC video")
                config, lengths[stream.index] = _clean_config(stream.codec_context.extradata)
                target.codec_context.extradata = config
                target.codec_context.codec_tag = "hvc1"
        if not lengths:
            raise ValueError("No HEVC video stream found")
        pictures = 0
        for packet in input_file.demux(list(stream for stream in input_file.streams
                                           if stream.index in streams)):
            if packet.size == 0:
                continue  # Demux flush packets are not media samples.
            index = packet.stream.index
            if index in lengths:
                updated = av.Packet(_update_packet(bytes(packet), lengths[index], sei, packet.is_keyframe))
                updated.pts, updated.dts = packet.pts, packet.dts
                updated.duration, updated.time_base = packet.duration, packet.time_base
                updated.is_keyframe = packet.is_keyframe
                updated.is_corrupt = packet.is_corrupt
                for side_data in packet.iter_sidedata():
                    side_data.to_packet(updated)
                packet = updated
                pictures += 1
            packet.stream = streams[index]
            output_file.mux(packet)
        if not pictures:
            raise ValueError("No HEVC video packets found")


if __name__ == "__main__":
    import sys

    remux_hdr_metadata(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
