"""cameras example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.cameras import camera_matrix, encode_cameras

expected_sequence = [
    "create_buffer",
    "get_queue",
    "write_buffer",
    "create_shader_module",
    "create_render_pipeline",
    "create_texture",
    "create_buffer",
    "create_texture",
    "write_texture",
    "create_sampler",
    "get_bind_group_layout",
    "create_texture_view",
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


def test_cameras_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_cameras(device)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_cameras_uses_look_at_view(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_cameras(device, width=800, height=600)

    uniform_write = fake_gpu.calls_of("write_buffer")[1]
    assert uniform_write[3] == struct.pack("<16f", *camera_matrix(800, 600))


def test_cameras_draws_cube(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_cameras(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    assert fake_gpu.calls_of("draw") == [(set_pipeline[0], 36, 1, 0, 0)]
