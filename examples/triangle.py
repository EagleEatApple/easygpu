"""EasyGPU 0.1.0 triangle — command encoding verified headlessly.

No GPU is needed: every call is recorded by :class:`FakeGPU`. Run with
``uv run pytest tests/test_triangle.py``.
"""

from __future__ import annotations

import struct

from easygpu.buffer import Buffer
from easygpu.constants import VertexFormat
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment
from easygpu.pipeline import (
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


def encode_triangle(device: Device) -> None:
    """Create a triangle pipeline and encode+submit one clear+draw pass."""
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
        )
    )

    encoder = device.create_command_encoder(label="triangle-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=(0.15, 0.15, 0.15, 1.0))]
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_vertex_buffer(0, vertex_buffer)
        pass_.draw(3)
    command_buffer = encoder.finish()
    device.queue.submit([command_buffer])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode the triangle."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="triangle-device")
    encode_triangle(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
