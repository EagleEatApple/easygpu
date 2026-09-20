"""WebGPU enums used by :mod:`easygpu`.

Discriminants mirror the WebGPU IDL bit flags (``ShaderStage``, ``BufferUsage``)
and the wgpu-native / wgpu-types ABI encodings (``VertexFormat``, ``LoadOp``,
``StoreOp``, ``PrimitiveTopology``, ``TextureFormat``) so a future real backend
can translate them without remapping. The exact ABI numbers are the backend's
source of truth in 0.1.0; do not assume stability across wgpu releases.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag


class ShaderStage(IntFlag):
    VERTEX = 0x1
    FRAGMENT = 0x2
    COMPUTE = 0x4


class BufferUsage(IntFlag):
    MAP_READ = 0x0001
    MAP_WRITE = 0x0002
    COPY_SRC = 0x0004
    COPY_DST = 0x0008
    INDEX = 0x0010
    VERTEX = 0x0020
    UNIFORM = 0x0040
    STORAGE = 0x0080
    INDIRECT = 0x0100
    QUERY_RESOLVE = 0x0200


class PrimitiveTopology(IntEnum):
    TRIANGLE_LIST = 4


class VertexFormat(IntEnum):
    FLOAT32 = 19
    FLOAT32X2 = 20
    FLOAT32X3 = 21


class LoadOp(IntEnum):
    CLEAR = 0
    LOAD = 1


class StoreOp(IntEnum):
    STORE = 0
    DISCARD = 1


class TextureFormat(IntEnum):
    BGRA8_UNORM = 23


__all__ = [
    "BufferUsage",
    "LoadOp",
    "PrimitiveTopology",
    "ShaderStage",
    "StoreOp",
    "TextureFormat",
    "VertexFormat",
]
