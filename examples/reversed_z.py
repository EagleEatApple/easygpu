"""EasyGPU reversedZ — reversed-Z depth precision, color mode.

Port of ``webgpu-samples/sample/reversedZ``. The original offers three GUI
modes (``color``, ``precision-error``, ``depth-texture``); this port implements
``color``: the same scene is drawn twice into a split viewport, once with a
normal depth buffer (``less``, cleared to 1) and once with a reversed-Z depth
buffer (``greater``, cleared to 0) produced by pre-multiplying the projection
with ``depthRangeRemapMatrix``. The two halves let you compare z-fighting on
nearly-coplanar planes at large distances.
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
    LoadOp,
    ShaderStage,
    TextureFormat,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.matrix import multiply, perspective, rotation, scale, translation
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
from examples.shaders import REVERSED_Z_FRAG, REVERSED_Z_VERT

DEPTH_FORMAT = TextureFormat.DEPTH32FLOAT
GEOMETRY_VERTEX_SIZE = 32
GEOMETRY_POSITION_OFFSET = 0
GEOMETRY_COLOR_OFFSET = 16
GEOMETRY_DRAW_COUNT = 12
X_COUNT = 1
Y_COUNT = 5
NUM_INSTANCES = X_COUNT * Y_COUNT
MATRIX_STRIDE = 64
CLEAR_COLOR = (0.0, 0.0, 0.5, 1.0)


def geometry_vertices() -> bytes:
    """Return the two nearly-coplanar plane vertex arrays (from the sample)."""
    d = 0.0001
    o = 0.5
    red = (1.0, 0.0, 0.0, 1.0)
    green = (0.0, 1.0, 0.0, 1.0)
    rows: list[float] = [
        -1 - o,
        -1,
        d,
        1,
        *red,
        1 - o,
        -1,
        d,
        1,
        *red,
        -1 - o,
        1,
        d,
        1,
        *red,
        1 - o,
        -1,
        d,
        1,
        *red,
        1 - o,
        1,
        d,
        1,
        *red,
        -1 - o,
        1,
        d,
        1,
        *red,
        -1 + o,
        -1,
        -d,
        1,
        *green,
        1 + o,
        -1,
        -d,
        1,
        *green,
        -1 + o,
        1,
        -d,
        1,
        *green,
        1 + o,
        -1,
        -d,
        1,
        *green,
        1 + o,
        1,
        -d,
        1,
        *green,
        -1 + o,
        1,
        -d,
        1,
        *green,
    ]
    return struct.pack(f"<{len(rows)}f", *rows)


def depth_range_remap_matrix() -> list[float]:
    """Return the matrix that maps the [0, 1] depth range onto itself reversed."""
    matrix = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    matrix[10] = -1.0
    matrix[14] = 1.0
    return matrix


def camera_matrices(width: int, height: int) -> tuple[list[float], list[float]]:
    """Return the normal and reversed-range view-projection matrices."""
    aspect = (0.5 * width) / height
    projection = perspective(2.0 * math.pi / 5.0, aspect, 5.0, 9999.0)
    view = translation((0.0, 0.0, -12.0))
    view_projection = multiply(projection, view)
    reversed_range = multiply(depth_range_remap_matrix(), view_projection)
    return view_projection, reversed_range


def model_matrices(time: float) -> list[float]:
    """Return the 5 concatenated per-instance model matrices for ``time``."""
    values: list[float] = []
    m = 0
    for _x in range(X_COUNT):
        for y in range(Y_COUNT):
            z = -800.0 * m
            s = 1.0 + 50.0 * m
            model = multiply(
                translation(
                    (
                        _x - X_COUNT / 2 + 0.5,
                        (4.0 - 0.2 * z) * (y - Y_COUNT / 2 + 1.0),
                        z,
                    )
                ),
                scale((s, s, s)),
            )
            rotated = multiply(
                model,
                rotation((math.sin(time), math.cos(time), 0.0), math.pi / 6.0),
            )
            values.extend(rotated)
            m += 1
    return values


def _uniform_layout() -> BindGroupLayoutDescriptor:
    return BindGroupLayoutDescriptor(
        entries=[
            BindGroupLayoutEntry(
                binding=0,
                visibility=ShaderStage.VERTEX,
                buffer=BufferBindingType.UNIFORM,
            ),
            BindGroupLayoutEntry(
                binding=1,
                visibility=ShaderStage.VERTEX,
                buffer=BufferBindingType.UNIFORM,
            ),
        ],
        label="reversed-z-uniform-layout",
    )


def encode_reversed_z(
    device: Device,
    *,
    width: int = 800,
    height: int = 600,
    time: float = 0.0,
) -> None:
    """Create the reversedZ pipelines and encode both split-viewport passes."""
    geometry = Buffer.from_data(device, geometry_vertices(), label="geometry-vertices")

    shader = ShaderModule.from_wgsl(
        device, REVERSED_Z_VERT, REVERSED_Z_FRAG, label="reversed-z-shader"
    )

    uniform_layout = device.create_bind_group_layout(_uniform_layout())
    pipeline_layout = device.create_pipeline_layout(
        PipelineLayoutDescriptor(bind_group_layouts=[uniform_layout])
    )

    vertex_buffers = [
        VertexBufferLayout(
            array_stride=GEOMETRY_VERTEX_SIZE,
            attributes=[
                VertexAttribute(
                    shader_location=0,
                    format=VertexFormat.FLOAT32X4,
                    offset=GEOMETRY_POSITION_OFFSET,
                ),
                VertexAttribute(
                    shader_location=1,
                    format=VertexFormat.FLOAT32X4,
                    offset=GEOMETRY_COLOR_OFFSET,
                ),
            ],
        )
    ]

    pipelines = []
    for depth_compare in (CompareFunction.LESS, CompareFunction.GREATER):
        pipelines.append(
            device.create_render_pipeline(
                RenderPipelineDescriptor(
                    vertex_stage=ShaderStageEntry(module=shader, entry_point="main"),
                    fragment_stage=ShaderStageEntry(module=shader, entry_point="main"),
                    vertex_buffers=vertex_buffers,
                    layout=pipeline_layout,
                    cull_mode=CullMode.BACK,
                    depth_stencil=DepthStencilState(
                        format=DEPTH_FORMAT,
                        depth_write_enabled=True,
                        depth_compare=depth_compare,
                    ),
                    fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
                    label=f"reversed-z-pipeline-{depth_compare.name.lower()}",
                )
            )
        )

    depth_texture = device.create_texture(
        TextureDescriptor(
            size=(width, height, 1),
            format=DEPTH_FORMAT,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="reversed-z-depth",
        )
    )

    model_buffer = device.create_buffer(
        BufferDescriptor(
            size=NUM_INSTANCES * MATRIX_STRIDE,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="reversed-z-models",
        )
    )
    camera_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="reversed-z-camera",
        )
    )
    reversed_camera_buffer = device.create_buffer(
        BufferDescriptor(
            size=64,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="reversed-z-camera-reversed",
        )
    )

    bind_groups = [
        device.create_bind_group(
            BindGroupDescriptor(
                layout=uniform_layout,
                entries=[
                    BindGroupEntry(binding=0, resource=model_buffer),
                    BindGroupEntry(binding=1, resource=camera_buffer),
                ],
                label="reversed-z-bind-group-normal",
            )
        ),
        device.create_bind_group(
            BindGroupDescriptor(
                layout=uniform_layout,
                entries=[
                    BindGroupEntry(binding=0, resource=model_buffer),
                    BindGroupEntry(binding=1, resource=reversed_camera_buffer),
                ],
                label="reversed-z-bind-group-reversed",
            )
        ),
    ]

    view_projection, reversed_range = camera_matrices(width, height)
    device.queue.write_buffer(camera_buffer, 0, struct.pack("<16f", *view_projection))
    device.queue.write_buffer(reversed_camera_buffer, 0, struct.pack("<16f", *reversed_range))
    models = model_matrices(time)
    device.queue.write_buffer(model_buffer, 0, struct.pack(f"<{len(models)}f", *models))

    depth_view = depth_texture.create_view()
    depth_clear_values = (1.0, 0.0)

    encoder = device.create_command_encoder(label="reversed-z-encoder")
    for mode in (0, 1):
        color_attachment = RenderPassColorAttachment(
            view=0,
            clear_value=CLEAR_COLOR,
            load_op=LoadOp.CLEAR if mode == 0 else LoadOp.LOAD,
        )
        with encoder.begin_render_pass(
            [color_attachment],
            RenderPassDepthStencilAttachment(
                view=depth_view, depth_clear_value=depth_clear_values[mode]
            ),
        ) as pass_:
            pass_.set_pipeline(pipelines[mode])
            pass_.set_bind_group(0, bind_groups[mode])
            pass_.set_vertex_buffer(0, geometry)
            pass_.set_viewport((width * mode) / 2, 0.0, width / 2, float(height), 0.0, 1.0)
            pass_.draw(GEOMETRY_DRAW_COUNT, NUM_INSTANCES)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one reversedZ color-mode frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="reversed-z-device")
    encode_reversed_z(device, time=0.0)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
