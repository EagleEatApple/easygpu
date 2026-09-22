"""wireframe example verified via FakeGPU."""

from __future__ import annotations

from easygpu.constants import CompareFunction, IndexFormat, PrimitiveTopology
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.cube_mesh import CUBE_INDICES
from examples.wireframe import encode_wireframe

CUBE_INDEX_COUNT = len(CUBE_INDICES) // 4


def test_wireframe_shared_bind_group_layout(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_wireframe(device)

    layouts = fake_gpu.calls_of("create_pipeline_layout")
    assert len(layouts) == 1

    groups = fake_gpu.calls_of("create_bind_group")
    assert len(groups) == 1


def test_wireframe_two_pipelines(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_wireframe(device)

    descriptors = [c[1] for c in fake_gpu.calls_of("create_render_pipeline")]
    assert descriptors[0].primitive_topology == PrimitiveTopology.TRIANGLE_LIST
    assert descriptors[1].primitive_topology == PrimitiveTopology.LINE_LIST
    assert descriptors[0].depth_stencil.depth_bias == 1
    assert descriptors[0].depth_stencil.depth_compare == CompareFunction.LESS
    assert descriptors[1].depth_stencil.depth_compare == CompareFunction.LESS_EQUAL


def test_wireframe_calls_indexed_then_lines(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_wireframe(device)

    index_buffer = fake_gpu.calls_of("set_index_buffer")[0]
    assert index_buffer[2] == IndexFormat.UINT32
    assert fake_gpu.calls_of("draw_indexed")[0][1:] == (CUBE_INDEX_COUNT, 1, 0, 0, 0)
    assert fake_gpu.calls_of("draw")[0][1:] == (24, 1, 0, 0)


def test_wireframe_pass_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_wireframe(device)

    pass_handle = fake_gpu.calls_of("set_pipeline")[0][0]
    assert fake_gpu.render_pass_sequence(pass_handle) == [
        "set_pipeline",
        "set_bind_group",
        "set_index_buffer",
        "set_vertex_buffer",
        "draw_indexed",
        "set_pipeline",
        "set_vertex_buffer",
        "draw",
        "end_render_pass",
    ]
