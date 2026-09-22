"""EasyGPU rotatingCube — a depth-tested perspective cube.

A faithful port of ``webgpu-samples/sample/rotatingCube``: the same 36-vertex
cube mesh (``meshes/cube.ts``) driven by ``basic.vert.wgsl`` +
``vertexPositionColor.frag.wgsl``, with backface culling and a ``depth24plus``
render target. The browser canvas is not modeled headlessly, so the color
attachment binds view handle ``0``; the depth texture is real.

No GPU is needed — ``encode_rotating_cube`` records every call on a
:class:`FakeGPU`. Test with ``uv run pytest tests/test_rotating_cube.py``.
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
from examples.shaders import BASIC_VERT, VERTEX_POSITION_COLOR_FRAG

CUBE_POSITION_OFFSET = 0
CLEAR_COLOR = (0.5, 0.5, 0.5, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS


def mvp_matrix(width: int, height: int, time: float) -> list[float]:
    """Return the rotatingCube model-view-projection matrix for ``time``.

    Mirrors ``getTransformationMatrix()``: identity view translated to
    ``(0, 0, -4)``, rotated about the axis ``(sin t, cos t, 0)`` by one
    radian, then multiplied by the 72-degree-fov perspective projection.
    """
    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = multiply(
        translation((0.0, 0.0, -4.0)),
        rotation((math.sin(time), math.cos(time), 0.0), 1.0),
    )
    return multiply(projection, view)


def _mvp_bytes(width: int, height: int, time: float) -> bytes:
    return struct.pack("<16f", *mvp_matrix(width, height, time))


def encode_rotating_cube(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the rotatingCube pipeline and encode+submit one cube pass."""
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
            label="rotating-cube-pipeline",
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
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="cube-uniforms",
        )
    )
    uniform_bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=pipeline.get_bind_group_layout(0),
            entries=[BindGroupEntry(binding=0, resource=uniform_buffer)],
            label="cube-bind-group",
        )
    )

    device.queue.write_buffer(uniform_buffer, 0, _mvp_bytes(width, height, time))

    encoder = device.create_command_encoder(label="rotating-cube-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)],
        RenderPassDepthStencilAttachment(view=depth_texture.create_view(), depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, uniform_bind_group)
        pass_.set_vertex_buffer(0, vertices)
        pass_.draw(CUBE_VERTEX_COUNT)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one rotatingCube frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="rotating-cube-device")
    encode_rotating_cube(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
