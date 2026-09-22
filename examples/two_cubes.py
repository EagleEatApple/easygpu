"""EasyGPU twoCubes — two cubes sharing one uniform buffer at 256-byte offsets.

A faithful port of ``webgpu-samples/sample/twoCubes``. The two model matrices
live in a single uniform buffer at byte offsets ``0`` and ``256`` and are
selected per draw with two :class:`BufferSlice` bind groups — the canonical
WebGPU 256-byte-alignment lesson.
"""

from __future__ import annotations

import math
import struct

from easygpu.bindgroup import BindGroupDescriptor, BindGroupEntry, BufferSlice
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
from examples.shaders import BASIC_VERT, VERTEX_POSITION_COLOR_FRAG

CUBE_POSITION_OFFSET = 0
CLEAR_COLOR = (0.5, 0.5, 0.5, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
MATRIX_SIZE = 64
UNIFORM_OFFSET = 256  # uniformBindGroup offset must be 256-byte aligned


def cube_matrices(width: int, height: int, time: float) -> tuple[list[float], list[float]]:
    """Return the two model-view-projection matrices for ``time``."""
    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = translation((0.0, 0.0, -7.0))

    model1 = multiply(
        view,
        multiply(
            translation((-2.0, 0.0, 0.0)),
            rotation((math.sin(time), math.cos(time), 0.0), 1.0),
        ),
    )
    model2 = multiply(
        view,
        multiply(
            translation((2.0, 0.0, 0.0)),
            rotation((math.cos(time), math.sin(time), 0.0), 1.0),
        ),
    )
    return multiply(projection, model1), multiply(projection, model2)


def encode_two_cubes(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the twoCubes pipeline and encode+submit both cube draws."""
    vertices = Buffer.from_data(device, CUBE_VERTICES, label="cube-vertices")

    shader = ShaderModule.from_wgsl(
        device, BASIC_VERT, VERTEX_POSITION_COLOR_FRAG, label="cube-shader"
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
            label="two-cubes-pipeline",
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
            size=UNIFORM_OFFSET + MATRIX_SIZE,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="cube-uniforms",
        )
    )
    layout = pipeline.get_bind_group_layout(0)
    bind_group_1 = device.create_bind_group(
        BindGroupDescriptor(
            layout=layout,
            entries=[
                BindGroupEntry(
                    binding=0,
                    resource=BufferSlice(uniform_buffer, offset=0, size=MATRIX_SIZE),
                )
            ],
            label="cube-bind-group-1",
        )
    )
    bind_group_2 = device.create_bind_group(
        BindGroupDescriptor(
            layout=layout,
            entries=[
                BindGroupEntry(
                    binding=0,
                    resource=BufferSlice(uniform_buffer, offset=UNIFORM_OFFSET, size=MATRIX_SIZE),
                )
            ],
            label="cube-bind-group-2",
        )
    )

    mvp_1, mvp_2 = cube_matrices(width, height, time)
    device.queue.write_buffer(uniform_buffer, 0, struct.pack("<16f", *mvp_1))
    device.queue.write_buffer(uniform_buffer, UNIFORM_OFFSET, struct.pack("<16f", *mvp_2))

    encoder = device.create_command_encoder(label="two-cubes-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)],
        RenderPassDepthStencilAttachment(view=depth_texture.create_view(), depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_vertex_buffer(0, vertices)
        pass_.set_bind_group(0, bind_group_1)
        pass_.draw(CUBE_VERTEX_COUNT)
        pass_.set_bind_group(0, bind_group_2)
        pass_.draw(CUBE_VERTEX_COUNT)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one twoCubes frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="two-cubes-device")
    encode_two_cubes(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
