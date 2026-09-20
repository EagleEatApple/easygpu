"""RenderPipeline descriptor tests."""

from __future__ import annotations

from easygpu.constants import PrimitiveTopology, VertexFormat
from easygpu.pipeline import (
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)


def test_shader_stage_entry_holds_module_and_entry_point() -> None:
    stage = ShaderStageEntry(module=5, entry_point="vs_main")
    assert stage.module == 5
    assert stage.entry_point == "vs_main"


def test_vertex_attribute_shape() -> None:
    attr = VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X2, offset=0)
    assert attr.shader_location == 0
    assert attr.format == VertexFormat.FLOAT32X2
    assert attr.offset == 0


def test_vertex_buffer_layout_shape() -> None:
    attrs = [VertexAttribute(0, VertexFormat.FLOAT32X3, 0)]
    layout = VertexBufferLayout(array_stride=12, attributes=attrs)
    assert layout.array_stride == 12
    assert layout.attributes == attrs
    assert layout.step_mode == "vertex"


def test_render_pipeline_descriptor_defaults() -> None:
    desc = RenderPipelineDescriptor(vertex_stage=ShaderStageEntry(module=5))
    assert desc.fragment_stage is None
    assert desc.primitive_topology == PrimitiveTopology.TRIANGLE_LIST
    assert desc.vertex_buffers is None
    assert desc.label is None


def test_render_pipeline_exposes_id() -> None:
    pipe = RenderPipeline(id_=2, label="tri")
    assert pipe.id == 2
    assert pipe.label == "tri"


def test_render_pipeline_delete_forwards_destroy(fake_gpu) -> None:
    pipe = RenderPipeline(id_=6)
    pipe.delete()
    assert fake_gpu.calls_of("destroy_render_pipeline") == [(6,)]
