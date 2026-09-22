"""Procedural RGBA8_UNORM texture generators for the example programs.

The WebGPU samples ship their textures as image assets and copy them with
``copyExternalImageToTexture``. EasyGPU is dependency-free and headless, so the
examples synthesize equivalent images as raw RGBA8 byte rows. Every generator
returns ``bytes`` of length ``width * height * 4`` in top-down row order, ready
for ``Queue.write_texture``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Pixel = tuple[int, int, int, int]


def _to_bytes(pixels: Sequence[int]) -> bytes:
    return bytes(pixels)


def checkerboard(
    size: int,
    cell: int = 1,
    color_a: Pixel = (255, 255, 255, 255),
    color_b: Pixel = (0, 0, 0, 255),
) -> bytes:
    """Alternating RGBA squares of ``cell`` pixels."""
    rows: list[int] = []
    for y in range(size):
        for x in range(size):
            rows.extend(color_a if ((x // cell) + (y // cell)) % 2 == 0 else color_b)
    return _to_bytes(rows)


def radial_gradient(
    size: int,
    inner: Pixel = (255, 255, 255, 255),
    outer: Pixel = (255, 255, 255, 0),
) -> bytes:
    """A soft radial falloff used as a point-sprite/spot texture."""
    center = (size - 1) / 2.0
    radius = size / 2.0
    rows: list[int] = []
    for y in range(size):
        for x in range(size):
            dx, dy = x - center, y - center
            t = min(1.0, math.sqrt(dx * dx + dy * dy) / radius)
            rows.extend(
                (
                    round(inner[0] + (outer[0] - inner[0]) * t),
                    round(inner[1] + (outer[1] - inner[1]) * t),
                    round(inner[2] + (outer[2] - inner[2]) * t),
                    round(inner[3] + (outer[3] - inner[3]) * t),
                )
            )
    return _to_bytes(rows)


def three_circles(size: int) -> bytes:
    """Three overlapping soft-edged circles with alpha falloff (source image)."""
    center = (size - 1) / 2.0
    radius = size / 2.0
    rows: list[int] = []
    for y in range(size):
        for x in range(size):
            px, py = x - center, y - center
            best: tuple[float, Pixel] = (0.0, (0, 0, 0, 0))
            for i in range(3):
                angle = 2.0 * math.pi * i / 3.0
                cx, cy = math.cos(angle) * size / 6.0, math.sin(angle) * size / 6.0
                dx, dy = px - cx, py - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d >= radius:
                    continue
                t = d / radius
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                alpha = round(255 * (1.0 - t))
                if alpha > best[0]:
                    best = (float(alpha), _hsl_rgba(i / 3.0, 1.0, 0.5, alpha))
            rows.extend(best[1])
    return _to_bytes(rows)


def color_stripes(size: int) -> bytes:
    """A rainbow gradient with opaque and transparent diagonal stripes."""
    rows: list[int] = []
    for y in range(size):
        for x in range(size):
            t = ((x + y) / (2.0 * size)) % 1.0
            alpha = 255 if ((x + y) // 32) % 2 == 0 else 0
            rows.extend(_hsl_rgba(t, 1.0, 0.5, alpha))
    return _to_bytes(rows)


def _hsl_rgba(h: float, s: float, lightness: float, alpha: int) -> Pixel:
    c = (1.0 - abs(2.0 * lightness - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - abs(hp % 2.0 - 1.0))
    if hp < 1.0:
        r, g, b = c, x, 0.0
    elif hp < 2.0:
        r, g, b = x, c, 0.0
    elif hp < 3.0:
        r, g, b = 0.0, c, x
    elif hp < 4.0:
        r, g, b = 0.0, x, c
    elif hp < 5.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    m = lightness - c / 2.0
    return (
        round((r + m) * 255),
        round((g + m) * 255),
        round((b + m) * 255),
        alpha,
    )


def cube_texture(size: int = 32) -> bytes:
    """A colorful tile gradient standing in for the original click-me cube."""
    rows: list[int] = []
    tile = size // 4
    for y in range(size):
        for x in range(size):
            tx, ty = x // tile, y // tile
            hue = ((tx * 3 + ty * 5) % 8) / 8.0
            rows.extend(_hsl_rgba(hue, 0.9, 0.5, 255))
    return _to_bytes(rows)


def alpha_grid(size: int = 64, cell: int = 8) -> bytes:
    """Opaque/transparent checkerboard for alpha-to-coverage sampling."""
    return checkerboard(size, cell, (255, 255, 255, 255), (255, 255, 255, 0))


__all__ = [
    "alpha_grid",
    "checkerboard",
    "color_stripes",
    "cube_texture",
    "radial_gradient",
    "three_circles",
]
