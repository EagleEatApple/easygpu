"""rotatingCube example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import TextureFormat
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.rotating_cube import encode_rotating_cube, mvp_matrix

expected_sequence = [
    "create_buffer",
    "get_queue",
    "write_buffer",
    "create_shader_module",
    "create_render_pipeline",
    "create_texture",
    "create_buffer",
    "get_bind_group_layout",
    "create_bind_group",
    "write_buffer",
    "create_command_encoder",
    "create_texture_view",
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "set_vertex_buffer",
    "draw",
    "end_render_pass",
    "finish",
    "submit",
]


def test_rotating_cube_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_rotating_cube(device, width=640, height=480, time=0.0)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_rotating_cube_draws_36_vertices(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_rotating_cube(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    assert fake_gpu.calls_of("draw") == [(set_pipeline[0], 36, 1, 0, 0)]


def test_rotating_cube_pass_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_rotating_cube(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    pass_handle = set_pipeline[0]
    assert fake_gpu.render_pass_sequence(pass_handle) == [
        "set_pipeline",
        "set_bind_group",
        "set_vertex_buffer",
        "draw",
        "end_render_pass",
    ]


def test_rotating_cube_uniform_bytes(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_rotating_cube(device, time=0.0)

    write_buffer = fake_gpu.calls_of("write_buffer")
    uniform_write = write_buffer[1]
    assert uniform_write[2] == 0
    assert uniform_write[3] == struct.pack("<16f", *mvp_matrix(640, 480, 0.0))


def test_rotating_cube_depth_attachment(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_rotating_cube(device)

    begin = fake_gpu.calls_of("begin_render_pass")[0]
    depth_attachment = begin[2]
    assert depth_attachment.depth_clear_value == 1.0
    assert depth_attachment.view.texture.format == TextureFormat.DEPTH24PLUS
    assert begin[1][0].clear_value == (0.5, 0.5, 0.5, 1.0)
