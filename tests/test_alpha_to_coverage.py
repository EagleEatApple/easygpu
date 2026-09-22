"""alphaToCoverage example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import StoreOp
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.alpha_to_coverage import (
    SAMPLE_COUNT,
    encode_alpha_to_coverage,
    full_screen_matrix,
)


def test_alpha_to_coverage_msaa_pipeline(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_alpha_to_coverage(device)

    descriptor = fake_gpu.calls_of("create_render_pipeline")[0][1]
    assert descriptor.multisample is not None
    assert descriptor.multisample.count == SAMPLE_COUNT
    assert descriptor.multisample.alpha_to_coverage_enabled is True


def test_alpha_to_coverage_textures(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_alpha_to_coverage(device, width=800, height=600)

    descriptors = [c[1] for c in fake_gpu.calls_of("create_texture")]
    msaa, resolve, depth, grid = descriptors
    assert msaa.sample_count == SAMPLE_COUNT
    assert resolve.sample_count == 1
    assert depth.sample_count == SAMPLE_COUNT
    assert grid.size == (64, 64, 1)


def test_alpha_to_coverage_resolve_attachment(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_alpha_to_coverage(device)

    begin = fake_gpu.calls_of("begin_render_pass")[0]
    color = begin[1][0]
    assert color.resolve_target is not None
    assert color.store_op == StoreOp.DISCARD
    assert begin[2].depth_clear_value == 1.0


def test_alpha_to_coverage_uniform_and_draw(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_alpha_to_coverage(device, width=800, height=600)

    uniform_write = fake_gpu.calls_of("write_buffer")[0]
    assert uniform_write[3] == struct.pack("<16f", *full_screen_matrix(800, 600))
    assert fake_gpu.calls_of("draw")[0][1:] == (6, 1, 0, 0)
