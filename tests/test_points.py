"""points example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import VertexFormat
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.points import (
    MAX_POINTS,
    UNIFORM_FLOATS,
    encode_points,
    fibonacci_sphere_vertices,
    points_matrix,
)


def test_points_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_points(device, time=0.0)

    names = [name for name, _ in fake_gpu.calls]
    assert names == [
        "create_buffer",
        "get_queue",
        "write_buffer",
        "create_shader_module",
        "create_shader_module",
        "create_shader_module",
        "create_shader_module",
        "create_bind_group_layout",
        "create_pipeline_layout",
        "create_render_pipeline",
        "create_render_pipeline",
        "create_render_pipeline",
        "create_render_pipeline",
        "create_texture",
        "create_buffer",
        "create_sampler",
        "create_texture",
        "write_texture",
        "create_texture_view",
        "create_bind_group",
        "write_buffer",
        "create_command_encoder",
        "create_texture_view",
        "begin_render_pass",
        "set_pipeline",
        "set_vertex_buffer",
        "set_bind_group",
        "draw",
        "end_render_pass",
        "finish",
        "submit",
    ]


def test_points_draws_six_vertices_per_instance(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_points(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    assert fake_gpu.calls_of("draw") == [(set_pipeline[0], 6, MAX_POINTS, 0, 0)]


def test_points_vertex_buffer_is_instanced(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_points(device)

    descriptor = fake_gpu.calls_of("create_render_pipeline")[0][1]
    layout = descriptor.vertex_buffers[0]
    assert layout.step_mode == "instance"
    assert layout.array_stride == 12
    assert layout.attributes[0].format == VertexFormat.FLOAT32X3


def test_points_bind_group_layout(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_points(device)

    entries = fake_gpu.calls_of("create_bind_group_layout")[0][1].entries
    assert [entry.binding for entry in entries] == [0, 1, 2]


def test_points_uniform_bytes(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_points(device, width=800, height=600, time=0.0, size=10.0)

    uniform_write = fake_gpu.calls_of("write_buffer")[1]
    expected = struct.pack(
        f"<{UNIFORM_FLOATS}f",
        *points_matrix(800, 600, 0.0),
        800.0,
        600.0,
        10.0,
        0.0,
    )
    assert uniform_write[3] == expected


def test_fibonacci_sphere_vertex_count() -> None:
    vertices = fibonacci_sphere_vertices(MAX_POINTS, radius=1.0)
    assert len(vertices) == MAX_POINTS * 3 * 4
