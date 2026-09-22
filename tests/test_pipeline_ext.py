"""Pipeline extension tests: blend/depth/multisample state, compute, layouts."""

from __future__ import annotations

from easygpu.bindgroup import BindGroupLayout
from easygpu.constants import (
    BlendFactor,
    BlendOperation,
    CompareFunction,
    CullMode,
    FrontFace,
    PrimitiveTopology,
    TextureFormat,
    VertexFormat,
)
from easygpu.fake_gpu import FakeGPU
from easygpu.pipeline import (
    BlendComponent,
    BlendState,
    ColorTargetState,
    ComputePipelineDescriptor,
    DepthStencilState,
    MultisampleState,
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)


def _descriptor() -> RenderPipelineDescriptor:
    return RenderPipelineDescriptor(
        vertex_stage=ShaderStageEntry(module=0),
        vertex_buffers=[
            VertexBufferLayout(
                array_stride=40,
                attributes=[
                    VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X4, offset=0),
                    VertexAttribute(shader_location=1, format=VertexFormat.FLOAT32X4, offset=16),
                    VertexAttribute(shader_location=2, format=VertexFormat.FLOAT32X2, offset=32),
                ],
            )
        ],
    )


def test_descriptor_focus_fields_default() -> None:
    d = _descriptor()
    assert d.layout is None
    assert d.cull_mode == CullMode.NONE
    assert d.front_face == FrontFace.CCW
    assert d.depth_stencil is None
    assert d.fragment_targets is None
    assert d.multisample is None
    assert d.primitive_topology == PrimitiveTopology.TRIANGLE_LIST


def test_descriptor_allows_instance_step_mode() -> None:
    d = RenderPipelineDescriptor(
        vertex_stage=ShaderStageEntry(module=0),
        vertex_buffers=[
            VertexBufferLayout(
                array_stride=16,
                step_mode="instance",
                attributes=[
                    VertexAttribute(shader_location=1, format=VertexFormat.FLOAT32X4, offset=0)
                ],
            )
        ],
    )
    assert d.vertex_buffers[0].step_mode == "instance"


def test_blend_state_round_trip() -> None:
    blend = BlendState(
        color=BlendComponent(
            operation=BlendOperation.ADD,
            src_factor=BlendFactor.SRC_ALPHA,
            dst_factor=BlendFactor.ONE_MINUS_SRC_ALPHA,
        )
    )
    target = ColorTargetState(format=TextureFormat.RGBA8_UNORM, blend=blend)
    d = RenderPipelineDescriptor(
        vertex_stage=ShaderStageEntry(module=0),
        fragment_stage=None,
        fragment_targets=[target],
    )
    assert d.fragment_targets == [target]
    assert blend.alpha.operation == BlendOperation.ADD


def test_depth_stencil_state_defaults() -> None:
    ds = DepthStencilState(format=TextureFormat.DEPTH24PLUS)
    assert ds.depth_write_enabled is True
    assert ds.depth_compare == CompareFunction.LESS
    assert ds.depth_bias == 0
    assert ds.depth_bias_slope_scale == 0.0


def test_multisample_state_defaults() -> None:
    ms = MultisampleState()
    assert ms.count == 1
    assert ms.alpha_to_coverage_enabled is False
    ms4 = MultisampleState(count=4, alpha_to_coverage_enabled=True)
    assert ms4.count == 4
    assert ms4.alpha_to_coverage_enabled is True


def test_compute_pipeline_descriptor_defaults() -> None:
    stage = ShaderStageEntry(module=0)
    d = ComputePipelineDescriptor(compute=stage)
    assert d.compute is stage
    assert d.layout is None
    assert d.label is None


def test_shader_stage_entry_constants() -> None:
    assert ShaderStageEntry(module=0).constants is None
    stage = ShaderStageEntry(module=0, constants={"n": 1024})
    assert stage.constants == {"n": 1024}


def test_get_bind_group_layout_forwards(fake_gpu: FakeGPU) -> None:
    pipe = RenderPipeline(id_=5)
    layout = pipe.get_bind_group_layout(0)
    assert isinstance(layout, BindGroupLayout)
    assert fake_gpu.calls_of("get_bind_group_layout") == [(5, 0)]
