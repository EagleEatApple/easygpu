"""EasyGPU 0.2.0 offscreen triangle — rendered headlessly, then read back.

Runs the 0.1.0 triangle on a real wgpu backend (WgpuPyGPU) with no window: the
render target is an :class:`~easygpu.offscreen.OffscreenTarget` and the frame is
pulled back to pixels with :func:`~easygpu.offscreen.readback_rgba8`.

Run with ``EASYGPU_SOFTWARE=1`` (plus ``WGPU_BACKEND=dx12`` on Windows when the
default backend finds nothing) in front of ``uv run python
examples/triangle_offscreen.py`` to force a software adapter.
"""

from __future__ import annotations

import struct

from easygpu.buffer import Buffer
from easygpu.constants import VertexFormat
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment
from easygpu.offscreen import OffscreenTarget
from easygpu.pipeline import (
    ColorTargetState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.shader import ShaderModule

VERTEX_SOURCE = """@vertex
fn vs_main(@location(0) a_pos: vec2f, @location(1) a_color: vec3f) -> @builtin(position) vec4f {
    return vec4f(a_pos, 0.0, 1.0);
}
"""

FRAGMENT_SOURCE = """@fragment
fn fs_main() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }
"""

VERTICES = b"".join(
    struct.pack("<5f", *v)
    for v in [
        (-0.5, -0.5, 1.0, 0.0, 0.0),
        (0.5, -0.5, 0.0, 1.0, 0.0),
        (0.0, 0.5, 0.0, 0.0, 1.0),
    ]
)


def encode_triangle_offscreen(device: Device, target: OffscreenTarget) -> None:
    """Create a triangle pipeline, then encode+submit one clear+draw pass."""
    vertex_buffer = Buffer.from_data(device, VERTICES, label="triangle-vertices")

    shader = ShaderModule.from_wgsl(device, VERTEX_SOURCE, FRAGMENT_SOURCE, label="tri")

    pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="vs_main"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="fs_main"),
            vertex_buffers=[
                VertexBufferLayout(
                    array_stride=20,
                    attributes=[
                        VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X2, offset=0),
                        VertexAttribute(shader_location=1, format=VertexFormat.FLOAT32X3, offset=8),
                    ],
                )
            ],
            label="triangle-pipeline",
            fragment_targets=[ColorTargetState(format=target.format)],
        )
    )

    encoder = device.create_command_encoder(label="triangle-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=target, clear_value=(0.15, 0.15, 0.15, 1.0))]
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_vertex_buffer(0, vertex_buffer)
        pass_.draw(3)
    command_buffer = encoder.finish()
    device.queue.submit([command_buffer])


def main() -> None:
    from easygpu.device import request_adapter
    from easygpu.gpu import configure
    from easygpu.offscreen import readback_rgba8
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    configure(gpu)
    device = request_adapter().request_device(label="triangle-offscreen-device")
    target = OffscreenTarget(width=256, height=256)
    encode_triangle_offscreen(device, target)
    data = readback_rgba8(target)
    center = data[(256 // 2) * 256 * 4 + (256 // 2) * 4 : (256 // 2) * 256 * 4 + (256 // 2) * 4 + 4]
    print("frame bytes:", len(data), "| center pixel:", center.hex(" "))


if __name__ == "__main__":
    main()
