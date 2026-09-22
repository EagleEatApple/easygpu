"""EasyGPU blending — compositing a source texture over a destination texture.

A faithful port of ``webgpu-samples/sample/blending`` (the default
"premultiplied blend (source-over)" preset). The destination quad is drawn
unblended, then the source quad is drawn with

    color = src * one + dst * (1 - src.alpha)

using a ``set_blend_constant`` value. The original uses canvas-generated
images; this port synthesizes equivalent source/destination images with
:mod:`examples.textures`.
"""

from __future__ import annotations

import struct

from easygpu.bindgroup import (
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import BufferDescriptor
from easygpu.constants import (
    BlendFactor,
    BlendOperation,
    BufferBindingType,
    BufferUsage,
    FilterMode,
    MipmapFilterMode,
    ShaderStage,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment
from easygpu.matrix import multiply, ortho, scale
from easygpu.pipeline import (
    BlendComponent,
    BlendState,
    ColorTargetState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
)
from easygpu.sampler import SamplerDescriptor
from easygpu.shader import ShaderModule
from easygpu.texture import TextureDescriptor
from examples.shaders import TEXTURED_QUAD
from examples.textures import color_stripes, three_circles

TEXTURE_SIZE = 300
CLEAR_VALUE = (0.0, 0.0, 0.0, 0.0)
BLEND_CONSTANT = (1.0, 0.5, 0.25, 1.0)


def blend_matrix(width: int, height: int, texture_size: int) -> list[float]:
    """Return the ortho * scale matrix that places the unit quad on screen."""
    projection = ortho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
    return multiply(projection, scale((float(texture_size), float(texture_size), 1.0)))


def _source_over_blend() -> BlendState:
    component = BlendComponent(
        operation=BlendOperation.ADD,
        src_factor=BlendFactor.ONE,
        dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
    )
    return BlendState(color=component, alpha=component)


def encode_blending(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
) -> None:
    """Create the blending pipelines and encode+submit the two quads."""
    shader = ShaderModule.from_wgsl(device, TEXTURED_QUAD, label="textured-quad")

    bind_group_layout = device.create_bind_group_layout(
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(binding=0, visibility=ShaderStage.FRAGMENT, sampler=True),
                BindGroupLayoutEntry(
                    binding=1,
                    visibility=ShaderStage.FRAGMENT,
                    texture=TextureSampleType.FLOAT,
                ),
                BindGroupLayoutEntry(
                    binding=2,
                    visibility=ShaderStage.VERTEX,
                    buffer=BufferBindingType.UNIFORM,
                ),
            ],
            label="blending-bind-group-layout",
        )
    )
    pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[bind_group_layout])
    )

    src_texture = device.create_texture(
        TextureDescriptor(
            size=(TEXTURE_SIZE, TEXTURE_SIZE, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=(
                TextureUsage.TEXTURE_BINDING
                | TextureUsage.COPY_DST
                | TextureUsage.RENDER_ATTACHMENT
            ),
            label="source-texture",
        )
    )
    device.queue.write_texture(
        src_texture, three_circles(TEXTURE_SIZE), width=TEXTURE_SIZE, height=TEXTURE_SIZE
    )
    dst_texture = device.create_texture(
        TextureDescriptor(
            size=(TEXTURE_SIZE, TEXTURE_SIZE, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=(
                TextureUsage.TEXTURE_BINDING
                | TextureUsage.COPY_DST
                | TextureUsage.RENDER_ATTACHMENT
            ),
            label="destination-texture",
        )
    )
    device.queue.write_texture(
        dst_texture, color_stripes(TEXTURE_SIZE), width=TEXTURE_SIZE, height=TEXTURE_SIZE
    )

    sampler = device.create_sampler(
        SamplerDescriptor(
            mag_filter=FilterMode.LINEAR,
            min_filter=FilterMode.LINEAR,
            mipmap_filter=MipmapFilterMode.LINEAR,
        )
    )

    src_uniform = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="source-uniforms",
        )
    )
    dst_uniform = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="destination-uniforms",
        )
    )

    src_bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[
                BindGroupEntry(binding=0, resource=sampler),
                BindGroupEntry(binding=1, resource=src_texture.create_view()),
                BindGroupEntry(binding=2, resource=src_uniform),
            ],
            label="source-bind-group",
        )
    )
    dst_bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[
                BindGroupEntry(binding=0, resource=sampler),
                BindGroupEntry(binding=1, resource=dst_texture.create_view()),
                BindGroupEntry(binding=2, resource=dst_uniform),
            ],
            label="destination-bind-group",
        )
    )

    dst_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="fs"),
            layout=pipeline_layout,
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="destination-pipeline",
        )
    )
    src_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="fs"),
            layout=pipeline_layout,
            fragment_targets=[
                ColorTargetState(format=TextureFormat.BGRA8_UNORM, blend=_source_over_blend())
            ],
            label="source-pipeline",
        )
    )

    matrix = blend_matrix(width, height, TEXTURE_SIZE)
    packed = struct.pack("<16f", *matrix)
    device.queue.write_buffer(src_uniform, 0, packed)
    device.queue.write_buffer(dst_uniform, 0, packed)

    encoder = device.create_command_encoder(label="blending-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_VALUE)]
    ) as pass_:
        pass_.set_pipeline(dst_pipeline)
        pass_.set_bind_group(0, dst_bind_group)
        pass_.draw(6)

        pass_.set_pipeline(src_pipeline)
        pass_.set_bind_group(0, src_bind_group)
        pass_.set_blend_constant(BLEND_CONSTANT)
        pass_.draw(6)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one blending frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="blending-device")
    encode_blending(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
