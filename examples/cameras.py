"""EasyGPU cameras — a textured cube viewed through a look-at camera.

A port of ``webgpu-samples/sample/cameras``. The original ships an interactive
``ArcballCamera``/``WASDCamera`` pair; EasyGPU records static frames, so this
port fixes the camera at the sample's initial position ``(3, 2, 5)`` looking at
the origin, expressed with :func:`easygpu.matrix.look_at`. The pipeline and
``cube.wgsl`` (uniforms@0, sampler@1, texture@2) match the original.
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
    FilterMode,
    TextureFormat,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import look_at, multiply, perspective
from easygpu.pipeline import (
    ColorTargetState,
    DepthStencilState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.sampler import SamplerDescriptor
from easygpu.shader import ShaderModule
from easygpu.texture import TextureDescriptor
from examples.cube_mesh import (
    CUBE_UV_OFFSET,
    CUBE_VERTEX_COUNT,
    CUBE_VERTEX_SIZE,
    CUBE_VERTICES,
)
from examples.shaders import CAMERA_CUBE
from examples.textures import cube_texture

CUBE_POSITION_OFFSET = 0
CLEAR_COLOR = (0.5, 0.5, 0.5, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
TEXTURE_SIZE = 32
CAMERA_POSITION = (3.0, 2.0, 5.0)
CAMERA_TARGET = (0.0, 0.0, 0.0)
CAMERA_UP = (0.0, 1.0, 0.0)


def camera_matrix(width: int, height: int) -> list[float]:
    """Return the projection * look-at view matrix for the fixed camera."""
    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = look_at(CAMERA_POSITION, CAMERA_TARGET, CAMERA_UP)
    return multiply(projection, view)


def encode_cameras(device: Device, *, width: int = 640, height: int = 480) -> None:
    """Create the cameras pipeline and encode+submit one textured cube draw."""
    vertices = Buffer.from_data(device, CUBE_VERTICES, label="cube-vertices")

    shader = ShaderModule.from_wgsl(device, CAMERA_CUBE, label="camera-cube-shader")

    pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="vertex_main"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="fragment_main"),
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
            label="camera-cube-pipeline",
        )
    )

    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="camera-depth",
        )
    )

    uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="camera-uniforms",
        )
    )

    cube_tex = device.create_texture(
        TextureDescriptor(
            size=(TEXTURE_SIZE, TEXTURE_SIZE, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=(
                TextureUsage.TEXTURE_BINDING
                | TextureUsage.COPY_DST
                | TextureUsage.RENDER_ATTACHMENT
            ),
            label="cube-texture",
        )
    )
    device.queue.write_texture(
        cube_tex,
        cube_texture(TEXTURE_SIZE),
        width=TEXTURE_SIZE,
        height=TEXTURE_SIZE,
    )

    sampler = device.create_sampler(
        SamplerDescriptor(mag_filter=FilterMode.LINEAR, min_filter=FilterMode.LINEAR)
    )

    uniform_bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                BindGroupEntry(binding=0, resource=uniform_buffer),
                BindGroupEntry(binding=1, resource=sampler),
                BindGroupEntry(binding=2, resource=cube_tex.create_view()),
            ],
            label="camera-bind-group",
        )
    )

    mvp = camera_matrix(width, height)
    device.queue.write_buffer(uniform_buffer, 0, struct.pack("<16f", *mvp))

    encoder = device.create_command_encoder(label="cameras-encoder")
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
    """Wire a :class:`FakeGPU` and encode one cameras frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="cameras-device")
    encode_cameras(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
