"""reversedZ example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import CompareFunction, LoadOp
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.reversed_z import (
    camera_matrices,
    encode_reversed_z,
    model_matrices,
)


def test_reversed_z_two_pipelines(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_reversed_z(device)

    descriptors = [c[1] for c in fake_gpu.calls_of("create_render_pipeline")]
    assert descriptors[0].depth_stencil.depth_compare == CompareFunction.LESS
    assert descriptors[1].depth_stencil.depth_compare == CompareFunction.GREATER


def test_reversed_z_split_viewport_draws(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_reversed_z(device, width=800, height=600)

    viewports = [c[1:] for c in fake_gpu.calls_of("set_viewport")]
    assert viewports == [
        (0.0, 0.0, 400.0, 600.0, 0.0, 1.0),
        (400.0, 0.0, 400.0, 600.0, 0.0, 1.0),
    ]
    assert [d[1:] for d in fake_gpu.calls_of("draw")] == [(12, 5, 0, 0), (12, 5, 0, 0)]


def test_reversed_z_depth_clear_values(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_reversed_z(device)

    begins = fake_gpu.calls_of("begin_render_pass")
    assert begins[0][2].depth_clear_value == 1.0
    assert begins[1][2].depth_clear_value == 0.0
    assert begins[0][1][0].load_op == LoadOp.CLEAR
    assert begins[1][1][0].load_op == LoadOp.LOAD


def test_reversed_z_camera_buffers(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_reversed_z(device, width=800, height=600)

    view_projection, reversed_range = camera_matrices(800, 600)
    writes = fake_gpu.calls_of("write_buffer")
    assert writes[1][3] == struct.pack("<16f", *view_projection)
    assert writes[2][3] == struct.pack("<16f", *reversed_range)


def test_reversed_z_model_buffer(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_reversed_z(device, time=0.0)

    models = model_matrices(0.0)
    model_write = fake_gpu.calls_of("write_buffer")[3]
    assert model_write[3] == struct.pack(f"<{len(models)}f", *models)
    assert len(models) == 5 * 16
