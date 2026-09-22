"""EasyGPU shadowMapping — dual-pass shadow mapping with a cube mesh.

Faithful port of ``webgpu-samples/sample/shadowMapping`` using the EasyGPU
headless encoding API. Two render passes: a shadow depth pass (depth32float)
and a color pass (depth24plus-stencil8) with percentage-closer filtering via
a comparison sampler.

No GPU is needed — ``encode_frame`` records every call on a :class:`FakeGPU`.
Test with ``uv run pytest tests/test_shadow_mapping.py``.
"""

from __future__ import annotations

import math
import struct

from easygpu.bindgroup import (
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import Buffer, BufferDescriptor, BufferUsage
from easygpu.constants import (
    BufferBindingType,
    CompareFunction,
    CullMode,
    IndexFormat,
    SamplerBindingType,
    ShaderStage,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import look_at, multiply, ortho, perspective, translation
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
    CUBE_INDEX_COUNT,
    CUBE_NORMAL_OFFSET,
    CUBE_PN_INDICES,
    CUBE_POSITION_OFFSET,
    CUBE_VERTICES_PN,
)
from examples.shadow_shaders import FRAGMENT_SOURCE, VERTEX_SHADOW_SOURCE, VERTEX_SOURCE

SHADOW_SIZE = 1024


def encode_frame(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the shadowMapping pipelines and encode+submit one frame."""
    vertex_buffer = Buffer.from_data(device, CUBE_VERTICES_PN, label="shadow-vertices")

    index_buffer = Buffer.from_data(
        device,
        CUBE_PN_INDICES,
        usage=BufferUsage.INDEX | BufferUsage.COPY_DST,
        label="shadow-indices",
    )

    shadow_depth_texture = device.create_texture(
        TextureDescriptor(
            size=(SHADOW_SIZE, SHADOW_SIZE, 1),
            format=TextureFormat.DEPTH32FLOAT,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.TEXTURE_BINDING,
            label="shadow-depth",
        )
    )
    shadow_depth_view = shadow_depth_texture.create_view()

    scene_uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=144,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="scene-uniforms",
        )
    )
    model_uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="model-uniforms",
        )
    )

    shadow_shader = ShaderModule.from_wgsl(device, VERTEX_SHADOW_SOURCE, label="shadow-vert")
    color_vertex_shader = ShaderModule.from_wgsl(device, VERTEX_SOURCE, label="color-vert")
    fragment_shader = ShaderModule.from_wgsl(device, FRAGMENT_SOURCE, label="color-frag")

    uniform_bgl = device.create_bind_group_layout(
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(
                    binding=0,
                    visibility=ShaderStage.VERTEX,
                    buffer=BufferBindingType.UNIFORM,
                ),
            ],
        )
    )
    bgl_for_render = device.create_bind_group_layout(
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(
                    binding=0,
                    visibility=ShaderStage.VERTEX | ShaderStage.FRAGMENT,
                    buffer=BufferBindingType.UNIFORM,
                ),
                BindGroupLayoutEntry(
                    binding=1,
                    visibility=ShaderStage.VERTEX | ShaderStage.FRAGMENT,
                    texture=TextureSampleType.DEPTH,
                ),
                BindGroupLayoutEntry(
                    binding=2,
                    visibility=ShaderStage.VERTEX | ShaderStage.FRAGMENT,
                    sampler=SamplerBindingType.COMPARISON,
                ),
            ],
        )
    )

    shadow_pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[uniform_bgl, uniform_bgl])
    )
    color_pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[bgl_for_render, uniform_bgl])
    )

    vertex_buffer_layout = VertexBufferLayout(
        array_stride=24,
        attributes=[
            VertexAttribute(
                shader_location=0,
                format=VertexFormat.FLOAT32X3,
                offset=CUBE_POSITION_OFFSET,
            ),
            VertexAttribute(
                shader_location=1,
                format=VertexFormat.FLOAT32X3,
                offset=CUBE_NORMAL_OFFSET,
            ),
        ],
    )

    shadow_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shadow_shader, entry_point="main"),
            layout=shadow_pipeline_layout,
            vertex_buffers=[vertex_buffer_layout],
            depth_stencil=DepthStencilState(
                format=TextureFormat.DEPTH32FLOAT,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS,
            ),
            cull_mode=CullMode.BACK,
            label="shadow-pipeline",
        )
    )

    color_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=color_vertex_shader, entry_point="main"),
            fragment_stage=ShaderStageEntry(
                module=fragment_shader,
                entry_point="main",
                constants={"shadow_depth_texture_size": float(SHADOW_SIZE)},
            ),
            layout=color_pipeline_layout,
            vertex_buffers=[vertex_buffer_layout],
            depth_stencil=DepthStencilState(
                format=TextureFormat.DEPTH24PLUS_STENCIL8,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS,
            ),
            cull_mode=CullMode.BACK,
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="color-pipeline",
        )
    )

    scene_depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=TextureFormat.DEPTH24PLUS_STENCIL8,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="scene-depth",
        )
    )
    scene_depth_view = scene_depth_texture.create_view()

    comparison_sampler = device.create_sampler(SamplerDescriptor(compare=CompareFunction.LESS))

    scene_shadow_bg = device.create_bind_group(
        BindGroupDescriptor(
            layout=uniform_bgl,
            entries=[BindGroupEntry(binding=0, resource=scene_uniform_buffer)],
            label="scene-shadow-bg",
        )
    )
    scene_render_bg = device.create_bind_group(
        BindGroupDescriptor(
            layout=bgl_for_render,
            entries=[
                BindGroupEntry(binding=0, resource=scene_uniform_buffer),
                BindGroupEntry(binding=1, resource=shadow_depth_view),
                BindGroupEntry(binding=2, resource=comparison_sampler),
            ],
            label="scene-render-bg",
        )
    )
    model_bg = device.create_bind_group(
        BindGroupDescriptor(
            layout=uniform_bgl,
            entries=[BindGroupEntry(binding=0, resource=model_uniform_buffer)],
            label="model-bg",
        )
    )

    origin = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    eye = (0.0, 50.0, -100.0)
    light_position = (50.0, 100.0, -100.0)

    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 2000.0)

    rad = math.pi * time
    c, s = math.cos(rad), math.sin(rad)
    rotated_eye = (c * eye[0] + s * eye[2], eye[1], -s * eye[0] + c * eye[2])
    view = look_at(rotated_eye, origin, up)
    camera_view_proj = multiply(projection, view)

    light_view = look_at(light_position, origin, up)
    light_projection = ortho(-80, 80, -80, 80, -200, 300)
    light_view_proj = multiply(light_projection, light_view)

    model_matrix = translation((0, -45, 0))

    device.queue.write_buffer(scene_uniform_buffer, 0, struct.pack("<16f", *light_view_proj))
    device.queue.write_buffer(scene_uniform_buffer, 64, struct.pack("<16f", *camera_view_proj))
    device.queue.write_buffer(scene_uniform_buffer, 128, struct.pack("<3f", *light_position))
    device.queue.write_buffer(model_uniform_buffer, 0, struct.pack("<16f", *model_matrix))

    encoder = device.create_command_encoder(label="shadow-encoder")

    with encoder.begin_render_pass(
        [],
        RenderPassDepthStencilAttachment(
            view=shadow_depth_view,
            depth_clear_value=1.0,
        ),
    ) as shadow_pass:
        shadow_pass.set_pipeline(shadow_pipeline)
        shadow_pass.set_bind_group(0, scene_shadow_bg)
        shadow_pass.set_bind_group(1, model_bg)
        shadow_pass.set_vertex_buffer(0, vertex_buffer)
        shadow_pass.set_index_buffer(index_buffer, IndexFormat.UINT16)
        shadow_pass.draw_indexed(CUBE_INDEX_COUNT)

    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=(0.5, 0.5, 0.5, 1.0))],
        RenderPassDepthStencilAttachment(
            view=scene_depth_view,
            depth_clear_value=1.0,
        ),
    ) as color_pass:
        color_pass.set_pipeline(color_pipeline)
        color_pass.set_bind_group(0, scene_render_bg)
        color_pass.set_bind_group(1, model_bg)
        color_pass.set_vertex_buffer(0, vertex_buffer)
        color_pass.set_index_buffer(index_buffer, IndexFormat.UINT16)
        color_pass.draw_indexed(CUBE_INDEX_COUNT)

    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one shadowMapping frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="shadow-device")
    encode_frame(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
