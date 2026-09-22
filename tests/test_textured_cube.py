"""texturedCube example verified via FakeGPU."""

from __future__ import annotations

from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from easygpu.sampler import Sampler
from examples.textured_cube import TEXTURE_SIZE, encode_textured_cube
from examples.textures import cube_texture

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


def test_textured_cube_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_textured_cube(device, time=0.0)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_textured_cube_uploads_procedural_texture(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_textured_cube(device)

    write = fake_gpu.calls_of("write_texture")[0]
    assert write[2] == cube_texture(TEXTURE_SIZE)
    assert (write[3], write[4]) == (TEXTURE_SIZE, TEXTURE_SIZE)

    texture = fake_gpu.calls_of("create_texture")[1][1]
    assert texture.size == (TEXTURE_SIZE, TEXTURE_SIZE, 1)


def test_textured_cube_bind_group_entries(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_textured_cube(device)

    entries = fake_gpu.calls_of("create_bind_group")[0][1].entries
    assert [entry.binding for entry in entries] == [0, 1, 2]
    assert isinstance(entries[1].resource, Sampler)


def test_textured_cube_draws_cube(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_textured_cube(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    assert fake_gpu.calls_of("draw") == [(set_pipeline[0], 36, 1, 0, 0)]
