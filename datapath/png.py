"""PNG integrity and dimensions, standard library only.

``inspect_png`` reads the signature and walks every chunk. It verifies each chunk's CRC,
requires a 13-byte IHDR first with positive width and height, at least one IDAT, IEND
last and nothing after it, and decompresses the image data. For non-interlaced images
it also checks that the decompressed size matches the dimensions.
"""

import struct
import zlib
from pathlib import Path

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def inspect_png(path: Path) -> dict:
    """Return {'ok', 'width', 'height', 'bytes', 'error'} for a PNG file."""
    data = Path(path).read_bytes()
    out = {"ok": False, "width": None, "height": None, "bytes": len(data), "error": None}
    if not data.startswith(_SIGNATURE):
        out["error"] = "bad signature"
        return out
    pos, chunks, idat, ihdr = len(_SIGNATURE), [], [], None
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body_end = pos + 8 + length
        if body_end + 4 > len(data):
            out["error"] = f"truncated chunk {ctype!r}"
            return out
        body = data[pos + 8:body_end]
        (crc,) = struct.unpack(">I", data[body_end:body_end + 4])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            out["error"] = f"CRC mismatch in {ctype!r}"
            return out
        chunks.append(ctype)
        if ctype == b"IHDR":
            if len(body) != 13:
                out["error"] = "IHDR is not 13 bytes"
                return out
            ihdr = struct.unpack(">IIBBBBB", body)
            out["width"], out["height"] = ihdr[0], ihdr[1]
        elif ctype == b"IDAT":
            idat.append(body)
        pos = body_end + 4
        if ctype == b"IEND":
            break
    if not chunks or chunks[0] != b"IHDR":
        out["error"] = "IHDR not first"
    elif chunks[-1] != b"IEND":
        out["error"] = "no IEND"
    elif pos != len(data):
        out["error"] = "data after IEND"
    elif not idat:
        out["error"] = "no IDAT"
    elif not (out["width"] and out["height"]):
        out["error"] = "zero width or height"
    else:
        out["error"] = _check_pixels(ihdr, b"".join(idat))
        out["ok"] = out["error"] is None
    return out


def _check_pixels(ihdr, compressed: bytes):
    width, height, bit_depth, color_type, _, _, interlace = ihdr
    if color_type not in _CHANNELS:
        return f"unknown color type {color_type}"
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as e:
        return f"IDAT does not decompress: {e}"
    if interlace == 0:
        row = (width * _CHANNELS[color_type] * bit_depth + 7) // 8 + 1
        if len(raw) != row * height:
            return f"pixel data is {len(raw)} bytes, expected {row * height}"
    return None


def make_png(width: int, height: int, rgb=(200, 200, 200)) -> bytes:
    """Build a tiny valid RGB PNG (used by the synthetic test fixture)."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(ctype: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + ctype + body
                + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
