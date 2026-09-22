"""EasyGPU wireframe — a solid cube with a line-list wireframe overlay.

Simplified port of ``webgpu-samples/sample/wireframe``. The original renders
~200 ``models`` with STORAGE buffers and a ``buffer_view`` (WGSL storage read)
plus barycentric-coordinate edge highlighting. EasyGPU keeps the *core lesson*
— a solid pass with ``depthBias`` so the overlaid line-list pass does not
z-fight — and uses an indexed cube (``set_index_buffer`` + ``draw_indexed``)
plus a line-list edge buffer. Storage buffers/``buffer_view`` are out of scope.
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
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import (
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    CullMode,
    IndexFormat,
    PrimitiveTopology,
    ShaderStage,
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
    CUBE_CORNER_POSITIONS,
    CUBE_EDGES,
    CUBE_INDICES,
)
from examples.shaders import CUBE_SOLID_FRAG, CUBE_SOLID_VERT, CUBE_WIRE_FRAG, CUBE_WIRE_VERT

CLEAR_COLOR = (0.15, 0.15, 0.2, 1.0)
DEPTH_FORMAT = TextureFormat.DEPTH24PLUS
CUBE_INDEX_COUNT = len(CUBE_INDICES) // 4
CUBE_EDGE_VERTEX_COUNT = len(CUBE_EDGES) // 12


def wireframe_matrix(width: int, height: int, time: float) -> list[float]:
    """Return the rotating view-projection matrix for ``time``."""
    projection = perspective(2.0 * math.pi / 5.0, width / height, 1.0, 100.0)
    view = multiply(
        translation((0.0, 0.0, -4.0)),
        rotation((math.sin(time), math.cos(time), 0.0), 1.0),
    )
    return multiply(projection, view)


def _position_layout() -> VertexBufferLayout:
    return VertexBufferLayout(
        array_stride=12,
        attributes=[VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X3, offset=0)],
    )


def encode_wireframe(
    device: Device,
    *,
    width: int = 640,
    height: int = 480,
    time: float = 0.0,
) -> None:
    """Create the wireframe pipelines and encode+submit both passes."""
    corner_buffer = Buffer.from_data(
        device,
        CUBE_CORNER_POSITIONS,
        usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
        label="cube-corners",
    )
    index_buffer = Buffer.from_data(
        device,
        CUBE_INDICES,
        usage=BufferUsage.INDEX | BufferUsage.COPY_DST,
        label="cube-indices",
    )
    edge_buffer = Buffer.from_data(
        device,
        CUBE_EDGES,
        usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
        label="cube-edges",
    )

    solid_shader = ShaderModule.from_wgsl(
        device, CUBE_SOLID_VERT, CUBE_SOLID_FRAG, label="cube-solid"
    )
    wire_shader = ShaderModule.from_wgsl(device, CUBE_WIRE_VERT, CUBE_WIRE_FRAG, label="cube-wire")

    bind_group_layout = device.create_bind_group_layout(
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(
                    binding=0,
                    visibility=ShaderStage.VERTEX,
                    buffer=BufferBindingType.UNIFORM,
                )
            ],
            label="wireframe-bind-group-layout",
        )
    )
    pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[bind_group_layout])
    )

    lit_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=solid_shader, entry_point="main"),
            fragment_stage=ShaderStageEntry(module=solid_shader, entry_point="main"),
            primitive_topology=PrimitiveTopology.TRIANGLE_LIST,
            vertex_buffers=[_position_layout()],
            layout=pipeline_layout,
            cull_mode=CullMode.BACK,
            depth_stencil=DepthStencilState(
                format=DEPTH_FORMAT,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS,
                depth_bias=1,
                depth_bias_slope_scale=0.5,
            ),
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="wireframe-lit-pipeline",
        )
    )

    wire_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=wire_shader, entry_point="main"),
            fragment_stage=ShaderStageEntry(module=wire_shader, entry_point="main"),
            primitive_topology=PrimitiveTopology.LINE_LIST,
            vertex_buffers=[_position_layout()],
            layout=pipeline_layout,
            depth_stencil=DepthStencilState(
                format=DEPTH_FORMAT,
                depth_write_enabled=True,
                depth_compare=CompareFunction.LESS_EQUAL,
            ),
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="wireframe-line-pipeline",
        )
    )

    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="wireframe-depth",
        )
    )

    uniform_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="wireframe-uniforms",
        )
    )
    bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=bind_group_layout,
            entries=[BindGroupEntry(binding=0, resource=uniform_buffer)],
            label="wireframe-bind-group",
        )
    )

    device.queue.write_buffer(
        uniform_buffer, 0, struct.pack("<16f", *wireframe_matrix(width, height, time))
    )

    encoder = device.create_command_encoder(label="wireframe-encoder")
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)],
        RenderPassDepthStencilAttachment(view=depth_texture.create_view(), depth_clear_value=1.0),
    ) as pass_:
        pass_.set_pipeline(lit_pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.set_index_buffer(index_buffer, IndexFormat.UINT32)
        pass_.set_vertex_buffer(0, corner_buffer)
        pass_.draw_indexed(CUBE_INDEX_COUNT)

        pass_.set_pipeline(wire_pipeline)
        pass_.set_vertex_buffer(0, edge_buffer)
        pass_.draw(CUBE_EDGE_VERTEX_COUNT)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one wireframe frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="wireframe-device")
    encode_wireframe(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
