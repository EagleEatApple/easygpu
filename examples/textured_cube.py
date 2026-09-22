"""EasyGPU texturedCube — a rotating cube with a sampled texture.

A faithful port of ``webgpu-samples/sample/texturedCube``. Instead of fetching
``assets/img/Di-3d.png`` it synthesizes an equivalent tile texture with
:func:`examples.textures.cube_texture` and uploads it via ``Queue.write_texture``
(the samples use ``copyExternalImageToTexture``). The fragment shader
``sampleTextureMixColor.frag.wgsl`` multiplies the sample by the cube's
per-vertex color.
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
from easygpu.matrix import multiply, perspective, rotation, translation
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
from examples.shaders import BASIC_VERT, TEXTURED_CUBE_FRAG
from examples.textures import cube_texture

CUBE_POSITION_OFFSET = 0
CLEAR_COLOR = (0.5, 0.5, 0.5, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
TEXTURE_SIZE = 32


def encode_textured_cube(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the texturedCube pipeline and encode+submit one textured draw."""
    vertices = Buffer.from_data(device, CUBE_VERTICES, label="cube-vertices")

    shader = ShaderModule.from_wgsl(
        device, BASIC_VERT, TEXTURED_CUBE_FRAG, label="textured-cube-shader"
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
            label="textured-cube-pipeline",
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
            label="textured-cube-bind-group",
        )
    )

    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = multiply(
        translation((0.0, 0.0, -4.0)),
        rotation((math.sin(time), math.cos(time), 0.0), 1.0),
    )
    mvp = multiply(projection, view)
    device.queue.write_buffer(uniform_buffer, 0, struct.pack("<16f", *mvp))

    encoder = device.create_command_encoder(label="textured-cube-encoder")
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
    """Wire a :class:`FakeGPU` and encode one texturedCube frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="textured-cube-device")
    encode_textured_cube(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
