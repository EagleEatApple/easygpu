"""instancedCube example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.instanced_cube import (
    NUM_INSTANCES,
    encode_instanced_cube,
    instance_matrices,
)

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


def test_instanced_cube_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_instanced_cube(device, time=0.0)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_instanced_cube_single_draw_of_16_instances(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_instanced_cube(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    assert fake_gpu.calls_of("draw") == [(set_pipeline[0], 36, NUM_INSTANCES, 0, 0)]


def test_instanced_cube_uniform_bytes(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_instanced_cube(device, time=0.0)

    uniform_write = fake_gpu.calls_of("write_buffer")[1]
    expected = struct.pack("<256f", *instance_matrices(640, 480, 0.0))
    assert uniform_write[2] == 0
    assert uniform_write[3] == expected
    assert len(uniform_write[3]) == NUM_INSTANCES * 64
