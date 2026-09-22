"""WebGPU enum constants for easygpu."""

from __future__ import annotations

import easygpu.constants as constants
from easygpu.constants import (
    AddressMode,
    BlendFactor,
    BlendOperation,
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    CullMode,
    FilterMode,
    FrontFace,
    IndexFormat,
    LoadOp,
    MipmapFilterMode,
    PrimitiveTopology,
    ShaderStage,
    StoreOp,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
    VertexFormat,
)


def test_shader_stage_values() -> None:
    assert (
        ShaderStage.VERTEX,
        ShaderStage.FRAGMENT,
        ShaderStage.COMPUTE,
    ) == (0x1, 0x2, 0x4)


def test_shader_stage_composes() -> None:
    combined = ShaderStage.VERTEX | ShaderStage.FRAGMENT
    assert combined == 0x3
    assert isinstance(combined, int)


def test_buffer_usage_spec_flags() -> None:
    assert (
        BufferUsage.MAP_READ,
        BufferUsage.MAP_WRITE,
        BufferUsage.COPY_SRC,
        BufferUsage.COPY_DST,
        BufferUsage.INDEX,
        BufferUsage.VERTEX,
        BufferUsage.UNIFORM,
        BufferUsage.STORAGE,
        BufferUsage.INDIRECT,
        BufferUsage.QUERY_RESOLVE,
    ) == (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200)


def test_buffer_usage_composes() -> None:
    combined = BufferUsage.VERTEX | BufferUsage.COPY_DST
    assert combined == 0x28
    assert isinstance(combined, int)
    assert combined & BufferUsage.VERTEX


def test_primitive_topology() -> None:
    assert PrimitiveTopology.TRIANGLE_LIST == 4
    assert (PrimitiveTopology.POINT_LIST, PrimitiveTopology.LINE_LIST) == (0, 1)
    assert PrimitiveTopology.LINE_STRIP == 2


def test_vertex_format_values() -> None:
    assert (
        VertexFormat.FLOAT32,
        VertexFormat.FLOAT32X2,
        VertexFormat.FLOAT32X3,
    ) == (19, 20, 21)
    assert (
        VertexFormat.FLOAT16X2,
        VertexFormat.FLOAT16X4,
        VertexFormat.FLOAT32X4,
        VertexFormat.SINT32,
        VertexFormat.UINT32,
    ) == (8, 9, 22, 32, 36)


def test_index_format() -> None:
    assert (IndexFormat.UINT16, IndexFormat.UINT32) == (1, 2)


def test_load_store_ops() -> None:
    assert (LoadOp.CLEAR, LoadOp.LOAD) == (0, 1)
    assert (StoreOp.STORE, StoreOp.DISCARD) == (0, 1)


def test_texture_format() -> None:
    assert TextureFormat.BGRA8_UNORM == 23
    assert (TextureFormat.RGBA8_UNORM, TextureFormat.DEPTH24PLUS) == (24, 25)
    assert TextureFormat.DEPTH32FLOAT == 26


def test_texture_usage_spec_flags() -> None:
    assert (
        TextureUsage.COPY_SRC,
        TextureUsage.COPY_DST,
        TextureUsage.TEXTURE_BINDING,
        TextureUsage.STORAGE_BINDING,
        TextureUsage.RENDER_ATTACHMENT,
    ) == (0x0001, 0x0002, 0x0004, 0x0008, 0x0010)


def test_texture_sample_type() -> None:
    assert (TextureSampleType.FLOAT, TextureSampleType.UNFILTERABLE_FLOAT) == (0, 1)
    assert TextureSampleType.DEPTH == 2


def test_buffer_binding_type() -> None:
    assert (BufferBindingType.UNIFORM, BufferBindingType.STORAGE) == (0, 1)
    assert BufferBindingType.READ_ONLY_STORAGE == 2


def test_cull_mode_and_front_face() -> None:
    assert (CullMode.NONE, CullMode.FRONT, CullMode.BACK) == (0, 1, 2)
    assert (FrontFace.CCW, FrontFace.CW) == (0, 1)


def test_compare_function() -> None:
    assert (CompareFunction.NEVER, CompareFunction.LESS, CompareFunction.EQUAL) == (0, 1, 2)
    assert (CompareFunction.LESS_EQUAL, CompareFunction.GREATER) == (3, 4)
    assert (CompareFunction.NOT_EQUAL, CompareFunction.GREATER_EQUAL) == (5, 6)
    assert CompareFunction.ALWAYS == 7


def test_sampler_filters_and_address_modes() -> None:
    assert (FilterMode.NEAREST, FilterMode.LINEAR) == (0, 1)
    assert (MipmapFilterMode.NEAREST, MipmapFilterMode.LINEAR) == (0, 1)
    assert (AddressMode.CLAMP_TO_EDGE, AddressMode.REPEAT, AddressMode.MIRROR_REPEAT) == (0, 1, 2)


def test_blend_operations_and_factors() -> None:
    assert (BlendOperation.ADD, BlendOperation.SUBTRACT, BlendOperation.REVERSE_SUBTRACT) == (
        0,
        1,
        2,
    )
    assert (BlendOperation.MIN, BlendOperation.MAX) == (3, 4)
    assert (BlendFactor.ZERO, BlendFactor.ONE) == (0, 1)
    assert (BlendFactor.SRC, BlendFactor.ONE_MINUS_SRC) == (2, 3)
    assert (
        BlendFactor.SRC_ALPHA,
        BlendFactor.ONE_MINUS_SRC_ALPHA,
        BlendFactor.DST,
        BlendFactor.ONE_MINUS_DST,
    ) == (4, 5, 6, 7)
    assert (
        BlendFactor.DST_ALPHA,
        BlendFactor.ONE_MINUS_DST_ALPHA,
        BlendFactor.SRC_ALPHA_SATURATED,
        BlendFactor.CONSTANT,
        BlendFactor.ONE_MINUS_CONSTANT,
    ) == (8, 9, 10, 11, 12)


def test_query_type_values() -> None:
    assert constants.QueryType.OCCLUSION == 0
    assert constants.QueryType.TIMESTAMP == 1


def test_map_mode_flags() -> None:
    assert constants.MapMode.READ == 1
    assert constants.MapMode.WRITE == 2
    assert (constants.MapMode.READ | constants.MapMode.WRITE) == 3


def test_sampler_binding_type_values() -> None:
    assert constants.SamplerBindingType.FILTERING == 0
    assert constants.SamplerBindingType.NON_FILTERING == 1
    assert constants.SamplerBindingType.COMPARISON == 2


def test_stencil_texture_formats() -> None:
    assert constants.TextureFormat.DEPTH24PLUS_STENCIL8 == 27
    assert constants.TextureFormat.DEPTH32FLOAT_STENCIL8 == 28
