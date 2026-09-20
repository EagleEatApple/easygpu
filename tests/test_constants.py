"""WebGPU enum constants for easygpu."""

from __future__ import annotations

from easygpu.constants import (
    BufferUsage,
    LoadOp,
    PrimitiveTopology,
    ShaderStage,
    StoreOp,
    TextureFormat,
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


def test_vertex_format_values() -> None:
    assert (
        VertexFormat.FLOAT32,
        VertexFormat.FLOAT32X2,
        VertexFormat.FLOAT32X3,
    ) == (19, 20, 21)


def test_load_store_ops() -> None:
    assert (LoadOp.CLEAR, LoadOp.LOAD) == (0, 1)
    assert (StoreOp.STORE, StoreOp.DISCARD) == (0, 1)


def test_texture_format() -> None:
    assert TextureFormat.BGRA8_UNORM == 23
