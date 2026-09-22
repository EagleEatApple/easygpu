"""EasyGPU points — 1000 Fibonacci-sphere points drawn as instanced billboards.

A faithful port of ``webgpu-samples/sample/points``. Each point is a quad that
is expanded in the vertex shader (``@builtin(vertex_index)``) and positioned by
``@builtin(instance_index)``; a single ``draw(6, 1000)`` renders the whole
sphere. Four pipelines are built from the distance/fixed-size vertex shaders
crossed with the orange/textured fragment shaders. The original loads a
butterfly emoji sprite; this port synthesizes a soft radial sprite instead.
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
from easygpu.buffer import BufferDescriptor
from easygpu.constants import (
    BlendFactor,
    BlendOperation,
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    ShaderStage,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import look_at, multiply, perspective, rotation_x, rotation_y
from easygpu.pipeline import (
    BlendComponent,
    BlendState,
    ColorTargetState,
    DepthStencilState,
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.sampler import SamplerDescriptor
from easygpu.shader import ShaderModule
from easygpu.texture import TextureDescriptor
from examples.shaders import (
    POINT_DISTANCE_VERT,
    POINT_FIXED_VERT,
    POINT_ORANGE_FRAG,
    POINT_TEXTURED_FRAG,
)
from examples.textures import radial_gradient

CLEAR_COLOR = (0.3, 0.3, 0.3, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
SPRITE_SIZE = 64
MAX_POINTS = 1000
# matrix (16) + resolution (2) + size (1) + padding (1)
UNIFORM_FLOATS = 20


def fibonacci_sphere_vertices(num_samples: int, radius: float) -> bytes:
    """Return ``num_samples`` xyz positions evenly spread on a sphere."""
    values: list[float] = []
    increment = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_samples):
        offset = 2.0 / num_samples
        y = i * offset - 1.0 + offset / 2.0
        r = math.sqrt(1.0 - y * y)
        phi = (i % num_samples) * increment
        values.append((math.cos(phi) * r) * radius)
        values.append(y * radius)
        values.append((math.sin(phi) * r) * radius)
    return struct.pack(f"<{len(values)}f", *values)


def points_matrix(width: int, height: int, time: float) -> list[float]:
    """Return the rotated view-projection matrix for ``time``."""
    projection = perspective(math.pi / 2.0, width / height, 0.1, 50.0)
    view = look_at((0.0, 0.0, 1.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    view_projection = multiply(projection, view)
    # mat4.rotateY(vp, time) then mat4.rotateX(result, time * 0.1)
    return multiply(
        multiply(view_projection, rotation_y(time)),
        rotation_x(time * 0.1),
    )


def blend_state() -> BlendState:
    component = BlendComponent(
        operation=BlendOperation.ADD,
        src_factor=BlendFactor.ONE,
        dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
    )
    return BlendState(color=component, alpha=component)


def encode_points(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
    fixed_size: bool = False,
    textured: bool = False,
    size: float = 10.0,
) -> None:
    """Create the points pipelines and encode+submit one sphere draw."""
    vertex_data = fibonacci_sphere_vertices(MAX_POINTS, radius=1.0)
    vertex_buffer = device.create_buffer(
        BufferDescriptor(
            size=len(vertex_data),
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
            label="point-vertices",
        ),
    )
    device.queue.write_buffer(vertex_buffer, 0, vertex_data)

    distance_module = ShaderModule.from_wgsl(
        device, POINT_DISTANCE_VERT, label="point-distance-vert"
    )
    fixed_module = ShaderModule.from_wgsl(device, POINT_FIXED_VERT, label="point-fixed-vert")
    orange_module = ShaderModule.from_wgsl(device, POINT_ORANGE_FRAG, label="point-orange-frag")
    textured_module = ShaderModule.from_wgsl(
        device, POINT_TEXTURED_FRAG, label="point-textured-frag"
    )

    bind_group_layout = device.create_bind_group_layout(
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(
                    binding=0,
                    visibility=ShaderStage.VERTEX,
                    buffer=BufferBindingType.UNIFORM,
                ),
                BindGroupLayoutEntry(binding=1, visibility=ShaderStage.FRAGMENT, sampler=True),
                BindGroupLayoutEntry(
                    binding=2,
                    visibility=ShaderStage.FRAGMENT,
                    texture=TextureSampleType.FLOAT,
                ),
            ],
            label="point-bind-group-layout",
        )
    )
    pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[bind_group_layout])
    )

    depth_format = DEPTH_FORMAT
    blend = blend_state()
    pipelines: list[list[RenderPipeline]] = []
    for vert_module in (distance_module, fixed_module):
        row: list[RenderPipeline] = []
        for frag_module in (orange_module, textured_module):
            row.append(
                device.create_render_pipeline(
                    RenderPipelineDescriptor(
                        vertex_stage=ShaderStageEntry(module=vert_module, entry_point="vs"),
                        fragment_stage=ShaderStageEntry(module=frag_module, entry_point="fs"),
                        vertex_buffers=[
                            VertexBufferLayout(
                                array_stride=12,
                                step_mode="instance",
                                attributes=[
                                    VertexAttribute(
                                        shader_location=0,
                                        format=VertexFormat.FLOAT32X3,
                                        offset=0,
                                    )
                                ],
                            )
                        ],
                        layout=pipeline_layout,
                        depth_stencil=DepthStencilState(
                            format=depth_format,
                            depth_write_enabled=True,
                            depth_compare=CompareFunction.LESS,
                        ),
                        fragment_targets=[
                            ColorTargetState(format=TextureFormat.BGRA8_UNORM, blend=blend)
                        ],
                        label=f"point-pipeline-{len(pipelines)}-{len(row)}",
                    )
                )
            )
        pipelines.append(row)

    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=depth_format,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="point-depth",
        )
    )

    uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=UNIFORM_FLOATS * 4,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="point-uniforms",
        )
    )

    sampler = device.create_sampler(SamplerDescriptor())
    sprite = device.create_texture(
        TextureDescriptor(
            size=(SPRITE_SIZE, SPRITE_SIZE, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=(
                TextureUsage.COPY_DST
                | TextureUsage.TEXTURE_BINDING
                | TextureUsage.RENDER_ATTACHMENT
            ),
            label="point-sprite",
        )
    )
    device.queue.write_texture(
        sprite, radial_gradient(SPRITE_SIZE), width=SPRITE_SIZE, height=SPRITE_SIZE
    )
    sprite_view = sprite.create_view()

    bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[
                BindGroupEntry(binding=0, resource=uniform_buffer),
                BindGroupEntry(binding=1, resource=sampler),
                BindGroupEntry(binding=2, resource=sprite_view),
            ],
            label="point-bind-group",
        )
    )

    matrix = points_matrix(width, height, time)
    uniform_values = [
        *matrix,
        float(width),
        float(height),
        float(size),
        0.0,
    ]
    device.queue.write_buffer(
        uniform_buffer, 0, struct.pack(f"<{UNIFORM_FLOATS}f", *uniform_values)
    )

    pipeline = pipelines[1 if fixed_size else 0][1 if textured else 0]

    encoder = device.create_command_encoder(label="points-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)],
        RenderPassDepthStencilAttachment(view=depth_texture.create_view(), depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_vertex_buffer(0, vertex_buffer)
        pass_.set_bind_group(0, bind_group)
        pass_.draw(6, MAX_POINTS)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one points frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="points-device")
    encode_points(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
