"""Screenshot sizes and the shared downscale fallback, standard library only.

``png_size`` reads a PNG's dimensions from its IHDR chunk. ``fit_long_edge`` is the single
fallback both judges share: it scales a screenshot so its long edge is ``long_edge``
pixels, keeping the aspect ratio, and never enlarges; a screenshot already within the
limit keeps its exact bytes.

The resize is deterministic by construction, so both judges, every run and every Python
version get the same bytes:

- decoding supports 8-bit, non-interlaced PNGs (grayscale, RGB, palette, grayscale with
  alpha, RGBA); a palette image becomes RGB and its transparency table is ignored. Any
  other PNG cannot be downscaled, and the caller records why;
- each output sample is the rounded mean of the source samples in its box, computed
  with integers, first along rows and then along columns;
- the output PNG uses filter 0 on every row and stores the pixel data in uncompressed
  deflate blocks, written here rather than by zlib, whose compressed bytes can differ
  between library versions.
"""

import struct
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}


def png_size(data: bytes):
    """(width, height) from the IHDR chunk. Raises ValueError if this is not a PNG."""
    if not data.startswith(SIGNATURE) or len(data) < 33 or data[12:16] != b"IHDR":
        raise ValueError("not a PNG with an IHDR chunk first")
    (length,) = struct.unpack(">I", data[8:12])
    width, height = struct.unpack(">II", data[16:24])
    if length != 13 or width < 1 or height < 1:
        raise ValueError("bad IHDR")
    return width, height


def _chunks(data: bytes):
    pos = len(SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise ValueError(f"truncated chunk {ctype!r}")
        yield ctype, body
        pos += 12 + length
        if ctype == b"IEND":
            return
    raise ValueError("no IEND chunk")


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode(data: bytes):
    """(width, height, channels, pixel bytes) for a supported PNG; ValueError otherwise."""
    width, height = png_size(data)
    ihdr, idat, palette = None, [], None
    for ctype, body in _chunks(data):
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"PLTE":
            palette = body
        elif ctype == b"IDAT":
            idat.append(body)
    _, _, depth, color, _, _, interlace = ihdr
    if depth != 8 or interlace != 0 or color not in _CHANNELS:
        raise ValueError(f"unsupported PNG for downscaling: bit depth {depth}, "
                         f"color type {color}, interlace {interlace}")
    ch = _CHANNELS[color]
    raw = zlib.decompress(b"".join(idat))
    stride = width * ch
    if len(raw) != (stride + 1) * height:
        raise ValueError("pixel data does not match the dimensions")
    out = bytearray(stride * height)
    prev = bytearray(stride)
    for y in range(height):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if f == 1:
            for i in range(ch, stride):
                line[i] = (line[i] + line[i - ch]) & 0xFF
        elif f == 2:
            line = bytearray((a + b) & 0xFF for a, b in zip(line, prev))
        elif f == 3:
            for i in range(stride):
                left = line[i - ch] if i >= ch else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                left = line[i - ch] if i >= ch else 0
                up_left = prev[i - ch] if i >= ch else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 0xFF
        elif f != 0:
            raise ValueError(f"unknown PNG filter {f}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    if color == 3:
        if palette is None:
            raise ValueError("palette image without PLTE")
        rgb = bytearray(width * height * 3)
        for i, index in enumerate(out):
            if 3 * index + 3 > len(palette):
                raise ValueError("palette index out of range")
            rgb[3 * i:3 * i + 3] = palette[3 * index:3 * index + 3]
        return width, height, 3, bytes(rgb)
    return width, height, ch, bytes(out)


def target_size(width: int, height: int, long_edge: int):
    """The size whose long edge is ``long_edge``, rounding the short edge half up."""
    if max(width, height) <= long_edge:
        return width, height
    if width >= height:
        return long_edge, max(1, (2 * height * long_edge + width) // (2 * width))
    return max(1, (2 * width * long_edge + height) // (2 * height)), long_edge


def _spans(src: int, dst: int):
    return [(i * src // dst, max(i * src // dst + 1, -(-(i + 1) * src // dst))) for i in range(dst)]


def downscale(width, height, ch, pixels: bytes, new_width, new_height) -> bytes:
    """Box-filter ``pixels`` to (new_width, new_height): integer means, rows then columns."""
    xs, ys = _spans(width, new_width), _spans(height, new_height)
    stride = width * ch
    rows = []
    for y in range(height):
        row = pixels[y * stride:(y + 1) * stride]
        out = bytearray(new_width * ch)
        for ox, (x0, x1) in enumerate(xs):
            n = x1 - x0
            for c in range(ch):
                out[ox * ch + c] = (sum(row[x0 * ch + c:x1 * ch:ch]) + n // 2) // n
        rows.append(out)
    result = bytearray(new_width * ch * new_height)
    nstride = new_width * ch
    for oy, (y0, y1) in enumerate(ys):
        n = y1 - y0
        col = [sum(v) for v in zip(*rows[y0:y1])]
        result[oy * nstride:(oy + 1) * nstride] = bytes((s + n // 2) // n for s in col)
    return bytes(result)


def _stored_zlib(data: bytes) -> bytes:
    """A zlib stream of uncompressed deflate blocks: the same bytes on every zlib version."""
    out = bytearray(b"\x78\x01")
    blocks = [data[i:i + 65535] for i in range(0, len(data), 65535)] or [b""]
    for i, block in enumerate(blocks):
        n = len(block)
        out += bytes([1 if i == len(blocks) - 1 else 0]) + struct.pack("<HH", n, n ^ 0xFFFF) + block
    return bytes(out + struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF))


def encode(width: int, height: int, ch: int, pixels: bytes) -> bytes:
    """An 8-bit PNG with filter 0 on every row and uncompressed deflate blocks."""
    stride = width * ch
    raw = b"".join(b"\x00" + pixels[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(ctype: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + ctype + body
                + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, _COLOR_TYPE[ch], 0, 0, 0)
    return SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", _stored_zlib(raw)) + chunk(b"IEND", b"")


def fit_long_edge(data: bytes, long_edge: int) -> bytes:
    """The screenshot with its long edge at most ``long_edge``; unchanged bytes if it fits."""
    width, height = png_size(data)
    new_width, new_height = target_size(width, height, long_edge)
    if (new_width, new_height) == (width, height):
        return data
    w, h, ch, pixels = decode(data)
    return encode(new_width, new_height, ch, downscale(w, h, ch, pixels, new_width, new_height))
