"""EasyGPU instancedCube — 16 cubes drawn with a single instanced draw call.

A faithful port of ``webgpu-samples/sample/instancedCube``. All 16
model-view-projection matrices live in one uniform buffer as a
``array<mat4x4f, 16>``, indexed by ``@builtin(instance_index)`` in
``instanced.vert.wgsl``; one ``draw(36, 16)`` renders the whole grid.
"""

from __future__ import annotations

import math
import struct

from easygpu.bindgroup import BindGroupDescriptor, BindGroupEntry
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import (
    BufferUsage,
    CompareFunction,
    CullMode,
    TextureFormat,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import multiply, perspective, rotation, translation
from easygpu.pipeline import (
    ColorTargetState,
    DepthStencilState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.shader import ShaderModule
from easygpu.texture import TextureDescriptor
from examples.cube_mesh import (
    CUBE_UV_OFFSET,
    CUBE_VERTEX_COUNT,
    CUBE_VERTEX_SIZE,
    CUBE_VERTICES,
)
from examples.shaders import INSTANCED_VERT, VERTEX_POSITION_COLOR_FRAG

CUBE_POSITION_OFFSET = 0
CLEAR_COLOR = (0.5, 0.5, 0.5, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
X_COUNT = 4
Y_COUNT = 4
NUM_INSTANCES = X_COUNT * Y_COUNT
STEP = 4.0


def instance_matrices(width: int, height: int, time: float) -> list[float]:
    """Return 16 concatenated column-major MVP matrices for ``time``.

    Mirrors ``updateTransformationMatrix()``: every instance sits on a 4x4
    grid (``step`` apart), spins about its own pseudo-random axis, then is
    transformed by the shared view (``(0, 0, -12)``) and projection.
    """
    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = translation((0.0, 0.0, -12.0))

    values: list[float] = []
    for x in range(X_COUNT):
        for y in range(Y_COUNT):
            model = translation(
                (
                    STEP * (x - X_COUNT / 2 + 0.5),
                    STEP * (y - Y_COUNT / 2 + 0.5),
                    0.0,
                )
            )
            rotated = multiply(
                model,
                rotation((math.sin((x + 0.5) * time), math.cos((y + 0.5) * time), 0.0), 1.0),
            )
            values.extend(multiply(projection, multiply(view, rotated)))
    return values


def encode_instanced_cube(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the instancedCube pipeline and encode+submit one instanced draw."""
    vertices = Buffer.from_data(device, CUBE_VERTICES, label="cube-vertices")

    shader = ShaderModule.from_wgsl(
        device, INSTANCED_VERT, VERTEX_POSITION_COLOR_FRAG, label="instanced-cube-shader"
    )

    pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="main"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="main"),
            vertex_buffers=[
                VertexBufferLayout(
                    array_stride=CUBE_VERTEX_SIZE,
                    attributes=[
                        VertexAttribute(
                            shader_location=0,
                            format=VertexFormat.FLOAT32X4,
                            offset=CUBE_POSITION_OFFSET,
                        ),
                        VertexAttribute(
                            shader_location=1,
                            format=VertexFormat.FLOAT32X2,
                            offset=CUBE_UV_OFFSET,
                        ),
                    ],
                )
            ],
            cull_mode=CullMode.BACK,
            depth_stencil=DepthStencilState(
                format=DEPTH_FORMAT,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS,
            ),
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="instanced-cube-pipeline",
        )
    )

    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="cube-depth",
        )
    )

    uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=NUM_INSTANCES * 64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="instance-uniforms",
        )
    )
    uniform_bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=pipeline.get_bind_group_layout(0),
            entries=[BindGroupEntry(binding=0, resource=uniform_buffer)],
            label="instance-bind-group",
        )
    )

    matrices = instance_matrices(width, height, time)
    device.queue.write_buffer(uniform_buffer, 0, struct.pack("<256f", *matrices))

    encoder = device.create_command_encoder(label="instanced-cube-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)],
        RenderPassDepthStencilAttachment(view=depth_texture.create_view(), depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, uniform_bind_group)
        pass_.set_vertex_buffer(0, vertices)
        pass_.draw(CUBE_VERTEX_COUNT, NUM_INSTANCES)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one instancedCube frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="instanced-cube-device")
    encode_instanced_cube(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
