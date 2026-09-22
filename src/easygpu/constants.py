"""WebGPU enums used by :mod:`easygpu`.

Discriminants mirror the WebGPU IDL bit flags (``ShaderStage``, ``BufferUsage``)
and the wgpu-native / wgpu-types ABI encodings (``VertexFormat``, ``LoadOp``,
``StoreOp``, ``PrimitiveTopology``, ``TextureFormat``) so a future real backend
can translate them without remapping. The exact ABI numbers are the backend's
source of truth once a real backend ships; do not assume stability across
wgpu releases.
0.0.1 extended the same style with the enums needed by the webgpu-samples port
(textures, samplers, bind groups, depth/stencil and blend state).
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


class MapMode(IntFlag):
    READ = 1
    WRITE = 2


class TextureUsage(IntFlag):
    COPY_SRC = 0x0001
    COPY_DST = 0x0002
    TEXTURE_BINDING = 0x0004
    STORAGE_BINDING = 0x0008
    RENDER_ATTACHMENT = 0x0010


class PrimitiveTopology(IntEnum):
    POINT_LIST = 0
    LINE_LIST = 1
    LINE_STRIP = 2
    TRIANGLE_LIST = 4


class VertexFormat(IntEnum):
    FLOAT16X2 = 8
    FLOAT16X4 = 9
    FLOAT32 = 19
    FLOAT32X2 = 20
    FLOAT32X3 = 21
    FLOAT32X4 = 22
    SINT32 = 32
    UINT32 = 36


class IndexFormat(IntEnum):
    UINT16 = 1
    UINT32 = 2


class LoadOp(IntEnum):
    CLEAR = 0
    LOAD = 1


class StoreOp(IntEnum):
    STORE = 0
    DISCARD = 1


class TextureFormat(IntEnum):
    BGRA8_UNORM = 23
    RGBA8_UNORM = 24
    DEPTH24PLUS = 25
    DEPTH32FLOAT = 26
    DEPTH24PLUS_STENCIL8 = 27
    DEPTH32FLOAT_STENCIL8 = 28


class TextureSampleType(IntEnum):
    FLOAT = 0
    UNFILTERABLE_FLOAT = 1
    DEPTH = 2


class BufferBindingType(IntEnum):
    UNIFORM = 0
    STORAGE = 1
    READ_ONLY_STORAGE = 2


class QueryType(IntEnum):
    OCCLUSION = 0
    TIMESTAMP = 1


class SamplerBindingType(IntEnum):
    FILTERING = 0
    NON_FILTERING = 1
    COMPARISON = 2


class CullMode(IntEnum):
    NONE = 0
    FRONT = 1
    BACK = 2


class FrontFace(IntEnum):
    CCW = 0
    CW = 1


class CompareFunction(IntEnum):
    NEVER = 0
    LESS = 1
    EQUAL = 2
    LESS_EQUAL = 3
    GREATER = 4
    NOT_EQUAL = 5
    GREATER_EQUAL = 6
    ALWAYS = 7


class FilterMode(IntEnum):
    NEAREST = 0
    LINEAR = 1


class MipmapFilterMode(IntEnum):
    NEAREST = 0
    LINEAR = 1


class AddressMode(IntEnum):
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRROR_REPEAT = 2


class BlendOperation(IntEnum):
    ADD = 0
    SUBTRACT = 1
    REVERSE_SUBTRACT = 2
    MIN = 3
    MAX = 4


class BlendFactor(IntEnum):
    ZERO = 0
    ONE = 1
    SRC = 2
    ONE_MINUS_SRC = 3
    SRC_ALPHA = 4
    ONE_MINUS_SRC_ALPHA = 5
    DST = 6
    ONE_MINUS_DST = 7
    DST_ALPHA = 8
    ONE_MINUS_DST_ALPHA = 9
    SRC_ALPHA_SATURATED = 10
    CONSTANT = 11
    ONE_MINUS_CONSTANT = 12


__all__ = [
    "AddressMode",
    "BlendFactor",
    "BlendOperation",
    "BufferBindingType",
    "BufferUsage",
    "CompareFunction",
    "CullMode",
    "FilterMode",
    "FrontFace",
    "IndexFormat",
    "LoadOp",
    "MapMode",
    "MipmapFilterMode",
    "PrimitiveTopology",
    "QueryType",
    "SamplerBindingType",
    "ShaderStage",
    "StoreOp",
    "TextureFormat",
    "TextureSampleType",
    "TextureUsage",
    "VertexFormat",
]
