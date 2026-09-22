"""EasyGPU alphaToCoverage — alpha-to-coverage on a 4x MSAA target.

``webgpu-samples`` only links out for alpha-to-coverage (the
``alphaToCoverageEmulator`` sample has no shaders), so this port implements the
classic technique: a checkerboard texture with transparent cells is sampled
with ``alphaToCoverageEnabled`` on a ``sampleCount = 4`` pipeline. The fragment
alpha becomes *coverage* on the MSAA target, which is then resolved to a
single-sample texture bound as the color attachment's ``resolveTarget`` —
turning hard texture edges into smooth ones without blending.
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
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    FilterMode,
    LoadOp,
    ShaderStage,
    StoreOp,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import multiply, ortho, scale
from easygpu.pipeline import (
    ColorTargetState,
    DepthStencilState,
    MultisampleState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
)
from easygpu.sampler import SamplerDescriptor
from easygpu.shader import ShaderModule
from easygpu.texture import TextureDescriptor
from examples.shaders import TEXTURED_QUAD
from examples.textures import alpha_grid

SAMPLE_COUNT = 4
CLEAR_COLOR = (0.1, 0.1, 0.15, 1.0)
TEXTURE_SIZE = 64


def full_screen_matrix(width: int, height: int) -> list[float]:
    """Return an ortho * scale matrix mapping the unit quad to the full target."""
    projection = ortho(0.0, float(width), float(height), 0.0, -1.0, 1.0)
    return multiply(projection, scale((float(width), float(height), 1.0)))


def encode_alpha_to_coverage(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
) -> None:
    """Create the MSAA alpha-to-coverage pipeline and encode+submit one frame."""
    shader = ShaderModule.from_wgsl(device, TEXTURED_QUAD, label="alpha-quad")

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
            label="alpha-to-coverage-bind-group-layout",
        )
    )
    pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[bind_group_layout])
    )

    msaa_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT,
            sample_count=SAMPLE_COUNT,
            label="msaa-color",
        )
    )
    resolve_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
            label="resolved-color",
        )
    )
    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=TextureFormat.DEPTH24PLUS,
            usage=TextureUsage.RENDER_ATTACHMENT,
            sample_count=SAMPLE_COUNT,
            label="msaa-depth",
        )
    )

    grid_texture = device.create_texture(
        TextureDescriptor(
            size=(TEXTURE_SIZE, TEXTURE_SIZE, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
            label="alpha-grid",
        )
    )
    device.queue.write_texture(
        grid_texture, alpha_grid(TEXTURE_SIZE), width=TEXTURE_SIZE, height=TEXTURE_SIZE
    )

    sampler = device.create_sampler(
        SamplerDescriptor(mag_filter=FilterMode.LINEAR, min_filter=FilterMode.LINEAR)
    )

    uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="alpha-uniforms",
        )
    )

    msaa_view = msaa_texture.create_view()
    resolve_view = resolve_texture.create_view()
    depth_view = depth_texture.create_view()

    bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[
                BindGroupEntry(binding=0, resource=sampler),
                BindGroupEntry(binding=1, resource=grid_texture.create_view()),
                BindGroupEntry(binding=2, resource=uniform_buffer),
            ],
            label="alpha-to-coverage-bind-group",
        )
    )

    pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=shader, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=shader, entry_point="fs"),
            layout=pipeline_layout,
            depth_stencil=DepthStencilState(
                format=TextureFormat.DEPTH24PLUS,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS,
            ),
            fragment_targets=[ColorTargetState(format=TextureFormat.RGBA8_UNORM)],
            multisample=MultisampleState(count=SAMPLE_COUNT, alpha_to_coverage_enabled=True),
            label="alpha-to-coverage-pipeline",
        )
    )

    device.queue.write_buffer(
        uniform_buffer, 0, struct.pack("<16f", *full_screen_matrix(width, height))
    )

    encoder = device.create_command_encoder(label="alpha-to-coverage-encoder")
    with encoder.begin_render_pass(
        [
            RenderPassColorAttachment(
                view=msaa_view,
                clear_value=CLEAR_COLOR,
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.DISCARD,
                resolve_target=resolve_view,
            )
        ],
        RenderPassDepthStencilAttachment(view=depth_view, depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.draw(6)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one alpha-to-coverage frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="alpha-to-coverage-device")
    encode_alpha_to_coverage(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
