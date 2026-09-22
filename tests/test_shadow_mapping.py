"""shadowMapping example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import CompareFunction, IndexFormat, TextureFormat, TextureUsage
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.cube_mesh import (
    CUBE_INDEX_COUNT,
    CUBE_NORMAL_OFFSET,
    CUBE_PN_INDICES,
    CUBE_POSITION_OFFSET,
    CUBE_VERTICES_PN,
)
from examples.shadow_mapping import encode_frame

expected_sequence = [
    "create_buffer",
    "get_queue",
    "write_buffer",
    "create_buffer",
    "write_buffer",
    "create_texture",
    "create_texture_view",
    "create_buffer",
    "create_buffer",
    "create_shader_module",
    "create_shader_module",
    "create_shader_module",
    "create_bind_group_layout",
    "create_bind_group_layout",
    "create_pipeline_layout",
    "create_pipeline_layout",
    "create_render_pipeline",
    "create_render_pipeline",
    "create_texture",
    "create_texture_view",
    "create_sampler",
    "create_bind_group",
    "create_bind_group",
    "create_bind_group",
    "write_buffer",
    "write_buffer",
    "write_buffer",
    "write_buffer",
    "create_command_encoder",
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "set_bind_group",
    "set_vertex_buffer",
    "set_index_buffer",
    "draw_indexed",
    "end_render_pass",
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "set_bind_group",
    "set_vertex_buffer",
    "set_index_buffer",
    "draw_indexed",
    "end_render_pass",
    "finish",
    "submit",
]

submit_phase = [
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "set_bind_group",
    "set_vertex_buffer",
    "set_index_buffer",
    "draw_indexed",
    "end_render_pass",
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "set_bind_group",
    "set_vertex_buffer",
    "set_index_buffer",
    "draw_indexed",
    "end_render_pass",
    "finish",
    "submit",
]


def test_shadow_mapping_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_shadow_mapping_submit_phase(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    names = [name for name, _ in fake_gpu.calls]
    begin_idx = names.index("begin_render_pass")
    assert names[begin_idx:] == submit_phase


def test_shadow_mapping_sampler_compare(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    sampler_calls = fake_gpu.calls_of("create_sampler")
    assert sampler_calls
    descriptor = sampler_calls[0][1]
    assert descriptor.compare == CompareFunction.LESS


def test_shadow_mapping_fragment_constants(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    pipeline_calls = fake_gpu.calls_of("create_render_pipeline")
    assert len(pipeline_calls) >= 2
    color_pipeline_descriptor = pipeline_calls[1][1]
    assert color_pipeline_descriptor.fragment_stage.constants == {
        "shadow_depth_texture_size": 1024.0
    }


def test_shadow_mapping_pass_sequences(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    set_pipelines = fake_gpu.calls_of("set_pipeline")
    shadow_pass_handle = set_pipelines[0][0]
    color_pass_handle = set_pipelines[1][0]

    assert fake_gpu.render_pass_sequence(shadow_pass_handle) == [
        "set_pipeline",
        "set_bind_group",
        "set_bind_group",
        "set_vertex_buffer",
        "set_index_buffer",
        "draw_indexed",
        "end_render_pass",
    ]

    assert fake_gpu.render_pass_sequence(color_pass_handle) == [
        "set_pipeline",
        "set_bind_group",
        "set_bind_group",
        "set_vertex_buffer",
        "set_index_buffer",
        "draw_indexed",
        "end_render_pass",
    ]


def test_shadow_mapping_draw_indexed(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    draw_calls = fake_gpu.calls_of("draw_indexed")
    assert len(draw_calls) == 2
    for call in draw_calls:
        assert call[1] == 36
        assert call[2] == 1
        assert call[3] == 0
        assert call[4] == 0
        assert call[5] == 0


def test_shadow_mapping_index_buffer_format(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    index_calls = fake_gpu.calls_of("set_index_buffer")
    assert len(index_calls) == 2
    for call in index_calls:
        assert call[2] == IndexFormat.UINT16


def test_shadow_mapping_binds_pn_index_buffer(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    index_calls = fake_gpu.calls_of("set_index_buffer")
    assert len(index_calls) == 2
    for call in index_calls:
        buffer_handle = call[1]
        bound = bytes(fake_gpu.buffers[buffer_handle])
        assert bound == CUBE_PN_INDICES
        assert struct.unpack("<36H", bound) == tuple(range(36))


def test_shadow_mapping_vertex_layout_stride(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    descriptor = fake_gpu.calls_of("create_render_pipeline")[0][1]
    layout = descriptor.vertex_buffers[0]
    assert layout.array_stride == 24
    assert [attribute.shader_location for attribute in layout.attributes] == [0, 1]
    assert [attribute.offset for attribute in layout.attributes] == [0, 12]


def test_shadow_mapping_depth_textures(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    textures = fake_gpu.calls_of("create_texture")
    assert len(textures) == 2
    shadow = textures[0][1]
    assert shadow.format == TextureFormat.DEPTH32FLOAT
    assert shadow.usage == (TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING)
    scene = textures[1][1]
    assert scene.format == TextureFormat.DEPTH24PLUS_STENCIL8
    assert scene.usage == TextureUsage.RENDER_ATTACHMENT


def test_shadow_mapping_no_blend_constant(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    assert fake_gpu.calls_of("set_blend_constant") == []


def test_shadow_mapping_empty_color_pass(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_frame(device)

    begin_calls = fake_gpu.calls_of("begin_render_pass")
    assert len(begin_calls) == 2
    shadow_pass_colors = begin_calls[0][1]
    assert shadow_pass_colors == []
    color_pass_colors = begin_calls[1][1]
    assert len(color_pass_colors) == 1


def test_cube_pn_mesh_layout() -> None:
    assert CUBE_POSITION_OFFSET == 0
    assert CUBE_NORMAL_OFFSET == 12
    assert CUBE_INDEX_COUNT == 36
    assert len(CUBE_VERTICES_PN) == 36 * 6
    assert len(CUBE_PN_INDICES) == 36 * 2


def test_cube_vertices_pn_length() -> None:
    assert len(CUBE_VERTICES_PN) == 36 * 6
