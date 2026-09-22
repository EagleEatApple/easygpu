"""blending example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import BlendFactor, BlendOperation
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.blending import BLEND_CONSTANT, blend_matrix, encode_blending

expected_sequence = [
    "create_shader_module",
    "create_bind_group_layout",
    "create_pipeline_layout",
    "create_texture",
    "get_queue",
    "write_texture",
    "create_texture",
    "write_texture",
    "create_sampler",
    "create_buffer",
    "create_buffer",
    "create_texture_view",
    "create_bind_group",
    "create_texture_view",
    "create_bind_group",
    "create_render_pipeline",
    "create_render_pipeline",
    "write_buffer",
    "write_buffer",
    "create_command_encoder",
    "begin_render_pass",
    "set_pipeline",
    "set_bind_group",
    "draw",
    "set_pipeline",
    "set_bind_group",
    "set_blend_constant",
    "draw",
    "end_render_pass",
    "finish",
    "submit",
]


def test_blending_encoding_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_blending(device)

    names = [name for name, _ in fake_gpu.calls]
    assert names == expected_sequence


def test_blending_source_over_blend_state(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_blending(device)

    pipelines = [c[1] for c in fake_gpu.calls_of("create_render_pipeline")]
    assert pipelines[0].fragment_targets[0].blend is None

    blend = pipelines[1].fragment_targets[0].blend
    assert blend is not None
    assert blend.color.operation == BlendOperation.ADD
    assert blend.color.src_factor == BlendFactor.ONE
    assert blend.color.dst_factor == BlendFactor.ONE_MINUS_SRC_ALPHA
    assert blend.alpha.src_factor == BlendFactor.ONE


def test_blending_draws_destination_then_source(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_blending(device)

    draws = fake_gpu.calls_of("draw")
    assert [d[1:] for d in draws] == [(6, 1, 0, 0), (6, 1, 0, 0)]

    constants = fake_gpu.calls_of("set_blend_constant")
    assert constants == [(draws[1][0], BLEND_CONSTANT)]


def test_blending_uniforms_use_ortho_scale(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_blending(device, width=800, height=600)

    matrix = blend_matrix(800, 600, 300)
    writes = fake_gpu.calls_of("write_buffer")
    assert writes[0][3] == struct.pack("<16f", *matrix)
    assert writes[1][3] == struct.pack("<16f", *matrix)
