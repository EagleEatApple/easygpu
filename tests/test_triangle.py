"""The 0.1.0 triangle example verified via FakeGPU."""

from __future__ import annotations

from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.triangle import VERTICES, encode_triangle

expected_prefix = [
    "create_buffer",
    "get_queue",
    "write_buffer",
    "create_shader_module",
    "create_render_pipeline",
    "create_command_encoder",
    "begin_render_pass",
    "set_pipeline",
    "set_vertex_buffer",
    "draw",
    "end_render_pass",
    "finish",
    "submit",
]


def test_triangle_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_triangle(device)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_prefix

    set_pipeline = fake_gpu.calls_of("set_pipeline")
    draw = fake_gpu.calls_of("draw")
    assert set_pipeline
    assert draw == [(set_pipeline[0][0], 3, 1, 0, 0)]

    submit = fake_gpu.calls_of("submit")
    assert submit and submit[0][1] == fake_gpu.command_buffers


def test_triangle_pass_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_triangle(device)

    set_pipeline = fake_gpu.calls_of("set_pipeline")[0]
    pass_handle = set_pipeline[0]
    assert fake_gpu.render_pass_sequence(pass_handle) == [
        "set_pipeline",
        "set_vertex_buffer",
        "draw",
        "end_render_pass",
    ]


def test_vertex_data_is_exact_bytes() -> None:
    assert len(VERTICES) == 3 * 20
