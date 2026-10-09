"""The offscreen triangle: FakeGPU recording vs WgpuPyGPU pixels."""

from __future__ import annotations

import pytest

from easygpu.device import request_adapter
from easygpu.errors import GPUValidationError
from easygpu.fake_gpu import FakeGPU
from easygpu.gpu import GPU
from easygpu.offscreen import OffscreenTarget, offscreen_target, readback_rgba8
from examples.triangle_offscreen import encode_triangle_offscreen


@pytest.mark.offscreen
def test_triangle_offscreen_dual_engine(fake_gpu: GPU) -> None:
    device = request_adapter().request_device(label="tri")
    target = offscreen_target(256, 256)
    encode_triangle_offscreen(device, target)

    if isinstance(fake_gpu, FakeGPU):
        assert [name for name, _ in fake_gpu.calls] == [
            "request_adapter",
            "request_device",
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
    else:
        pass_commands = [
            name
            for name, _ in fake_gpu.calls
            if name in ("set_pipeline", "set_vertex_buffer", "draw", "end_render_pass")
        ]
        assert pass_commands == ["set_pipeline", "set_vertex_buffer", "draw", "end_render_pass"]

    draw = fake_gpu.calls_of("draw")
    assert draw == [(draw[0][0], 3, 1, 0, 0)]
    if isinstance(fake_gpu, FakeGPU):
        with pytest.raises(GPUValidationError, match="real GPU backend"):
            readback_rgba8(target)


@pytest.mark.offscreen
@pytest.mark.wgpu
def test_triangle_offscreen_wgpu_pixels(wgpu_gpu: GPU) -> None:
    device = request_adapter().request_device(label="tri")
    target = OffscreenTarget(width=256, height=256)
    encode_triangle_offscreen(device, target)

    data = readback_rgba8(target)
    assert len(data) == 256 * 256 * 4
    center = data[(256 // 2) * 256 * 4 + (256 // 2) * 4 : (256 // 2) * 256 * 4 + (256 // 2) * 4 + 4]
    assert center[0] > 200 and center[1] < 60 and center[2] < 60
    assert data[:4] == bytes((38, 38, 38, 255))


def test_offscreen_target_defaults() -> None:
    target = OffscreenTarget(width=32, height=16)
    assert (target.width, target.height, target.label) == (32, 16, None)
    assert target.format == 24  # TextureFormat.RGBA8_UNORM
