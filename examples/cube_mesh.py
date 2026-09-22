"""Cube mesh data, ported 1:1 from ``webgpu-samples`` ``meshes/cube.ts``.

The reference samples use a 36-vertex cube in which every vertex carries
``float4 position + float4 color + float2 uv`` (a 40-byte stride). EasyGPU
re-drives the exact same data so scene setup matches the TypeScript originals.
We also derive a minimal index/edge set so samples that demonstrate
``set_index_buffer``/``draw_indexed`` (e.g. the wireframe port) can share the
mesh module.
"""

from __future__ import annotations

import struct

CUBE_VERTEX_SIZE = 4 * 10  # Byte size of one cube vertex.
CUBE_POSITION_OFFSET = 0
CUBE_COLOR_OFFSET = 4 * 4  # Byte offset of cube vertex color attribute.
CUBE_UV_OFFSET = 4 * 8
CUBE_VERTEX_COUNT = 36

CUBE_NORMAL_OFFSET = 12
CUBE_INDEX_COUNT = 36

# float4 position, float4 color, float2 uv,  (from cube.ts)
_CUBE_VERTEX_ARRAY: tuple[
    tuple[float, float, float, float, float, float, float, float, float, float], ...
] = (
    (1, -1, 1, 1, 1, 0, 1, 1, 0, 1),
    (-1, -1, 1, 1, 0, 0, 1, 1, 1, 1),
    (-1, -1, -1, 1, 0, 0, 0, 1, 1, 0),
    (1, -1, -1, 1, 1, 0, 0, 1, 0, 0),
    (1, -1, 1, 1, 1, 0, 1, 1, 0, 1),
    (-1, -1, -1, 1, 0, 0, 0, 1, 1, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
    (1, -1, 1, 1, 1, 0, 1, 1, 1, 1),
    (1, -1, -1, 1, 1, 0, 0, 1, 1, 0),
    (1, 1, -1, 1, 1, 1, 0, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
    (1, -1, -1, 1, 1, 0, 0, 1, 1, 0),
    (-1, 1, 1, 1, 0, 1, 1, 1, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, -1, 1, 1, 1, 0, 1, 1, 0),
    (-1, 1, -1, 1, 0, 1, 0, 1, 0, 0),
    (-1, 1, 1, 1, 0, 1, 1, 1, 0, 1),
    (1, 1, -1, 1, 1, 1, 0, 1, 1, 0),
    (-1, -1, 1, 1, 0, 0, 1, 1, 0, 1),
    (-1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    (-1, 1, -1, 1, 0, 1, 0, 1, 1, 0),
    (-1, -1, -1, 1, 0, 0, 0, 1, 0, 0),
    (-1, -1, 1, 1, 0, 0, 1, 1, 0, 1),
    (-1, 1, -1, 1, 0, 1, 0, 1, 1, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
    (-1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    (-1, -1, 1, 1, 0, 0, 1, 1, 1, 0),
    (-1, -1, 1, 1, 0, 0, 1, 1, 1, 0),
    (1, -1, 1, 1, 1, 0, 1, 1, 0, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
    (1, -1, -1, 1, 1, 0, 0, 1, 0, 1),
    (-1, -1, -1, 1, 0, 0, 0, 1, 1, 1),
    (-1, 1, -1, 1, 0, 1, 0, 1, 1, 0),
    (1, 1, -1, 1, 1, 1, 0, 1, 0, 0),
    (1, -1, -1, 1, 1, 0, 0, 1, 0, 1),
    (-1, 1, -1, 1, 0, 1, 0, 1, 1, 0),
)

CUBE_VERTICES = b"".join(struct.pack("<10f", *vertex) for vertex in _CUBE_VERTEX_ARRAY)

# The 8 unique cube corners (±1), indexed for set_index_buffer/draw_indexed.
CUBE_CORNERS: tuple[tuple[float, float, float], ...] = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)

CUBE_CORNER_POSITIONS = b"".join(struct.pack("<3f", *corner) for corner in CUBE_CORNERS)

# 36 triangle-list indices referencing CUBE_CORNERS.
CUBE_INDICES = struct.pack(
    "<36I",
    *(
        0,
        4,
        6,
        0,
        6,
        2,  # -X face
        1,
        3,
        7,
        1,
        7,
        5,  # +X face
        0,
        1,
        5,
        0,
        5,
        4,  # -Y face
        2,
        6,
        7,
        2,
        7,
        3,  # +Y face
        0,
        2,
        3,
        0,
        3,
        1,  # -Z face
        4,
        5,
        7,
        4,
        7,
        6,  # +Z face
    ),
)

# The 12 cube edges as line-list vertices (48 floats = 24 points).
CUBE_EDGES = struct.pack(
    "<72f",
    *(
        v
        for pair in (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),  # bottom/top back, bottom/top front
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),  # verticals
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # back-to-front
        )
        for corner in pair
        for v in CUBE_CORNERS[corner]
    ),
)


_CUBE_VERTICES_PN_ARRAY: tuple[tuple[float, ...], ...] = (
    (-1, -1, -1, -1, 0, 0),
    (-1, -1, 1, -1, 0, 0),
    (-1, 1, 1, -1, 0, 0),
    (-1, -1, -1, -1, 0, 0),
    (-1, 1, 1, -1, 0, 0),
    (-1, 1, -1, -1, 0, 0),
    (1, -1, -1, 1, 0, 0),
    (1, 1, -1, 1, 0, 0),
    (1, 1, 1, 1, 0, 0),
    (1, -1, -1, 1, 0, 0),
    (1, 1, 1, 1, 0, 0),
    (1, -1, 1, 1, 0, 0),
    (-1, -1, -1, 0, -1, 0),
    (1, -1, -1, 0, -1, 0),
    (1, -1, 1, 0, -1, 0),
    (-1, -1, -1, 0, -1, 0),
    (1, -1, 1, 0, -1, 0),
    (-1, -1, 1, 0, -1, 0),
    (-1, 1, -1, 0, 1, 0),
    (-1, 1, 1, 0, 1, 0),
    (1, 1, 1, 0, 1, 0),
    (-1, 1, -1, 0, 1, 0),
    (1, 1, 1, 0, 1, 0),
    (1, 1, -1, 0, 1, 0),
    (-1, -1, -1, 0, 0, -1),
    (-1, 1, -1, 0, 0, -1),
    (1, 1, -1, 0, 0, -1),
    (-1, -1, -1, 0, 0, -1),
    (1, 1, -1, 0, 0, -1),
    (1, -1, -1, 0, 0, -1),
    (-1, -1, 1, 0, 0, 1),
    (1, -1, 1, 0, 0, 1),
    (1, 1, 1, 0, 0, 1),
    (-1, -1, 1, 0, 0, 1),
    (1, 1, 1, 0, 0, 1),
    (-1, 1, 1, 0, 0, 1),
)

CUBE_VERTICES_PN: tuple[float, ...] = tuple(v for vertex in _CUBE_VERTICES_PN_ARRAY for v in vertex)

CUBE_PN_INDICES: bytes = struct.pack("<36H", *range(36))


__all__ = [
    "CUBE_COLOR_OFFSET",
    "CUBE_CORNER_POSITIONS",
    "CUBE_CORNERS",
    "CUBE_EDGES",
    "CUBE_INDEX_COUNT",
    "CUBE_INDICES",
    "CUBE_NORMAL_OFFSET",
    "CUBE_PN_INDICES",
    "CUBE_POSITION_OFFSET",
    "CUBE_UV_OFFSET",
    "CUBE_VERTEX_COUNT",
    "CUBE_VERTEX_SIZE",
    "CUBE_VERTICES",
    "CUBE_VERTICES_PN",
]
