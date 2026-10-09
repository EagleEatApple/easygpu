"""Offscreen textured cube: FakeGPU recording vs WgpuPyGPU pixels."""

from __future__ import annotations

import pytest

from easygpu.device import request_adapter
from easygpu.fake_gpu import FakeGPU
from easygpu.gpu import GPU
from easygpu.offscreen import OffscreenTarget, offscreen_target, readback_rgba8
from examples.textured_cube_offscreen import encode_textured_cube_offscreen


@pytest.mark.offscreen
def test_textured_cube_offscreen_dual_engine(fake_gpu: GPU) -> None:
    device = request_adapter().request_device(label="cube")
    target = offscreen_target(256, 256)
    encode_textured_cube_offscreen(device, target, time=0.25)

    pass_commands = [
        name
        for name, _ in fake_gpu.calls
        if name
        in ("set_pipeline", "set_bind_group", "set_vertex_buffer", "draw", "end_render_pass")
    ]
    assert pass_commands == [
        "set_pipeline",
        "set_bind_group",
        "set_vertex_buffer",
        "draw",
        "end_render_pass",
    ]
    draw = fake_gpu.calls_of("draw")
    assert draw == [(draw[0][0], 36, 1, 0, 0)]

    present = {name for name, _ in fake_gpu.calls}
    assert {
        "write_texture",
        "create_sampler",
        "get_bind_group_layout",
        "create_bind_group",
    } <= present

    if isinstance(fake_gpu, FakeGPU):
        entries = fake_gpu.calls_of("create_bind_group")[0][1].entries
        assert [entry.binding for entry in entries] == [0, 1, 2]
        with pytest.raises(Exception, match="real GPU backend"):
            readback_rgba8(target)


@pytest.mark.offscreen
@pytest.mark.wgpu
def test_textured_cube_offscreen_wgpu_pixels(wgpu_gpu: GPU) -> None:
    device = request_adapter().request_device(label="cube")
    target = OffscreenTarget(width=256, height=256)
    encode_textured_cube_offscreen(device, target, time=0.25)

    data = readback_rgba8(target)
    assert len(data) == 256 * 256 * 4
    clear = bytes((128, 128, 128, 255))
    assert data != clear * (256 * 256)
