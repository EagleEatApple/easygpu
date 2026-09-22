"""twoCubes example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.bindgroup import BufferSlice
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.two_cubes import UNIFORM_OFFSET, cube_matrices, encode_two_cubes

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
    "create_bind_group",
    "write_buffer",
    "write_buffer",
    "create_command_encoder",
    "create_texture_view",
    "begin_render_pass",
    "set_pipeline",
    "set_vertex_buffer",
    "set_bind_group",
    "draw",
    "set_bind_group",
    "draw",
    "end_render_pass",
    "finish",
    "submit",
]


def test_two_cubes_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_two_cubes(device, time=0.0)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_two_cubes_draws_both_cubes(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_two_cubes(device)

    draws = fake_gpu.calls_of("draw")
    assert [d[1:] for d in draws] == [(36, 1, 0, 0), (36, 1, 0, 0)]


def test_two_cubes_uniform_offsets(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_two_cubes(device, time=0.0)

    writes = fake_gpu.calls_of("write_buffer")
    assert writes[1][2] == 0
    assert writes[2][2] == UNIFORM_OFFSET

    mvp_1, mvp_2 = cube_matrices(640, 480, 0.0)
    assert writes[1][3] == struct.pack("<16f", *mvp_1)
    assert writes[2][3] == struct.pack("<16f", *mvp_2)


def test_two_cubes_bind_group_slices(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_two_cubes(device)

    groups = fake_gpu.calls_of("create_bind_group")
    slice_1 = groups[0][1].entries[0].resource
    slice_2 = groups[1][1].entries[0].resource
    assert isinstance(slice_1, BufferSlice) and slice_1.offset == 0
    assert isinstance(slice_2, BufferSlice) and slice_2.offset == UNIFORM_OFFSET
