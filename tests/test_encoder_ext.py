"""CommandEncoder / RenderPassEncoder extension tests."""

from __future__ import annotations

import pytest

from easygpu.bindgroup import BindGroup
from easygpu.buffer import Buffer
from easygpu.constants import BufferUsage, IndexFormat, LoadOp, StoreOp
from easygpu.device import Device
from easygpu.encoder import (
    CommandEncoder,
    ComputePassEncoder,
    RenderBundle,
    RenderBundleDescriptor,
    RenderBundleEncoder,
    RenderPassColorAttachment,
    RenderPassDepthStencilAttachment,
)
from easygpu.errors import GPUValidationError
from easygpu.fake_gpu import FakeGPU
from easygpu.pipeline import ComputePipeline, RenderPipeline
from easygpu.query import QuerySet
from easygpu.texture import TextureDescriptor, TextureFormat


def test_color_attachment_accepts_texture_view_and_resolve(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=16)
    )
    view = tex.create_view()
    att = RenderPassColorAttachment(view=view, resolve_target=view)
    assert att.resolve_target is view


def test_depth_stencil_attachment_defaults() -> None:
    d = RenderPassDepthStencilAttachment(view=2)
    assert d.depth_clear_value == 1.0
    assert d.depth_load_op == LoadOp.CLEAR
    assert d.depth_store_op == StoreOp.STORE
    assert d.stencil_clear_value == 0


def test_begin_render_pass_records_depth_attachment(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    depth = RenderPassDepthStencilAttachment(view=2)
    enc.begin_render_pass(
        [RenderPassColorAttachment(view=0)],
        depth_stencil_attachment=depth,
    )
    args = fake_gpu.calls_of("begin_render_pass")
    assert args[0][1] == [RenderPassColorAttachment(view=0)]
    assert args[0][2] == depth


def test_set_index_buffer_and_draw_indexed(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    ib = Buffer(id_=3, size=72, usage=BufferUsage.INDEX)
    rp.set_index_buffer(ib, IndexFormat.UINT32)
    rp.draw_indexed(36)
    rp.draw_indexed(36, 2)
    rp.end()
    assert fake_gpu.calls_of("set_index_buffer") == [(1, 3, IndexFormat.UINT32, 0, None)]
    assert fake_gpu.calls_of("draw_indexed") == [(1, 36, 1, 0, 0, 0), (1, 36, 2, 0, 0, 0)]


def test_set_bind_group(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    group = BindGroup(id_=7)
    rp.set_bind_group(0, group)
    rp.end()
    assert fake_gpu.calls_of("set_bind_group") == [(1, 0, 7)]


def test_set_viewport(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.set_viewport(0.0, 0.0, 2.0, 2.0, 0.25, 0.75)
    rp.end()
    assert fake_gpu.calls_of("set_viewport") == [(1, 0.0, 0.0, 2.0, 2.0, 0.25, 0.75)]


def test_set_blend_constant(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.set_blend_constant((0.5, 0.5, 0.5, 1.0))
    rp.end()
    assert fake_gpu.calls_of("set_blend_constant") == [(1, (0.5, 0.5, 0.5, 1.0))]


def test_extended_calls_after_end_raise(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.end()
    with pytest.raises(GPUValidationError):
        rp.set_index_buffer(Buffer(id_=3, size=4, usage=BufferUsage.INDEX), IndexFormat.UINT32)
    with pytest.raises(GPUValidationError):
        rp.set_bind_group(0, BindGroup(id_=7))
    with pytest.raises(GPUValidationError):
        rp.set_viewport(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(GPUValidationError):
        rp.set_blend_constant((1.0, 1.0, 1.0, 1.0))
    with pytest.raises(GPUValidationError):
        rp.draw_indexed(6)


def test_compute_pass_sequence(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    with enc.begin_compute_pass() as cp:
        cp.set_pipeline(ComputePipeline(id_=4))
        cp.set_bind_group(0, BindGroup(id_=7))
        cp.dispatch_workgroups(8)
    assert fake_gpu.compute_pass_sequence(cp.id) == [
        "compute_set_pipeline",
        "compute_set_bind_group",
        "dispatch_workgroups",
        "end_compute_pass",
    ]
    assert fake_gpu.calls_of("begin_compute_pass") == [(1, None)]
    assert fake_gpu.calls_of("compute_set_pipeline") == [(cp.id, 4)]
    assert fake_gpu.calls_of("compute_set_bind_group") == [(cp.id, 0, 7)]
    assert fake_gpu.calls_of("dispatch_workgroups") == [(cp.id, 8, 1, 1)]
    assert fake_gpu.calls_of("end_compute_pass") == [(cp.id,)]


def test_compute_pass_dispatch_indirect_and_timestamp(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    cp = enc.begin_compute_pass(label="add")
    assert isinstance(cp, ComputePassEncoder)
    indirect = Buffer(id_=3, size=16, usage=BufferUsage.INDIRECT)
    cp.dispatch_workgroups(2, 3, 4)
    cp.dispatch_workgroups_indirect(indirect, offset=8)
    cp.write_timestamp(QuerySet(id_=9), 1)
    cp.end()
    assert fake_gpu.calls_of("begin_compute_pass") == [(1, "add")]
    assert fake_gpu.calls_of("dispatch_workgroups") == [(cp.id, 2, 3, 4)]
    assert fake_gpu.calls_of("dispatch_workgroups_indirect") == [(cp.id, 3, 8)]
    assert fake_gpu.calls_of("write_compute_timestamp") == [(cp.id, 9, 1)]
    assert fake_gpu.calls_of("end_compute_pass") == [(cp.id,)]


def test_render_bundle_round_trip(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    dev = Device(id_=2)
    be = dev.create_render_bundle_encoder(RenderBundleDescriptor(label="b"))
    assert isinstance(be, RenderBundleEncoder)
    be.set_pipeline(RenderPipeline(id_=5))
    be.set_bind_group(0, BindGroup(id_=7))
    be.set_vertex_buffer(0, Buffer(id_=3, size=64, usage=BufferUsage.VERTEX), offset=8)
    be.set_index_buffer(
        Buffer(id_=4, size=72, usage=BufferUsage.INDEX), IndexFormat.UINT16, offset=16
    )
    be.draw(6)
    be.draw_indexed(3)
    bundle = be.finish()
    rp.execute_render_bundles([bundle])
    rp.end()
    assert isinstance(bundle, RenderBundle)
    assert bundle.id in fake_gpu.bundles
    assert fake_gpu.calls_of("create_render_bundle_encoder") == [
        (2, RenderBundleDescriptor(label="b"))
    ]
    assert fake_gpu.calls_of("bundle_set_pipeline") == [(2, 5)]
    assert fake_gpu.calls_of("bundle_set_bind_group") == [(2, 0, 7)]
    assert fake_gpu.calls_of("bundle_set_vertex_buffer") == [(2, 0, 3, 8)]
    assert fake_gpu.calls_of("bundle_set_index_buffer") == [(2, 4, IndexFormat.UINT16, 16, None)]
    assert fake_gpu.calls_of("bundle_draw") == [(2, 6, 1)]
    assert fake_gpu.calls_of("bundle_draw_indexed") == [(2, 3, 1)]
    assert fake_gpu.calls_of("bundle_finish") == [(2,)]
    assert fake_gpu.calls_of("execute_render_bundles") == [(1, [bundle.id])]
    assert fake_gpu.bundles[bundle.id] == [
        ("bundle_set_pipeline", (5,)),
        ("bundle_set_bind_group", (0, 7)),
        ("bundle_set_vertex_buffer", (0, 3, 8)),
        ("bundle_set_index_buffer", (4, IndexFormat.UINT16, 16, None)),
        ("bundle_draw", (6, 1)),
        ("bundle_draw_indexed", (3, 1)),
    ]


def test_scissor_indirect_and_write_timestamp(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.set_scissor_rect(0, 0, 32, 32)
    indirect = Buffer(id_=3, size=16, usage=BufferUsage.INDIRECT)
    rp.draw_indirect(indirect)
    rp.draw_indexed_indirect(indirect, offset=4)
    rp.write_timestamp(QuerySet(id_=9), 0)
    rp.end()
    assert fake_gpu.calls_of("set_scissor_rect") == [(1, 0, 0, 32, 32)]
    assert fake_gpu.calls_of("draw_indirect") == [(1, 3, 0)]
    assert fake_gpu.calls_of("draw_indexed_indirect") == [(1, 3, 4)]
    assert fake_gpu.calls_of("write_timestamp") == [(1, 9, 0)]
    assert fake_gpu.calls_of("end_render_pass") == [(1,)]


def test_copies_and_query_resolve(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    device = Device(id_=2)
    src = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=16)
    ).create_view()
    dst = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=16)
    ).create_view()
    buf = Buffer(id_=3, size=128, usage=BufferUsage.COPY_DST)
    enc.copy_texture_to_texture(src, dst, size=(4, 4, 1))
    enc.copy_texture_to_buffer(src, buf, size=(4, 4, 1))
    dest = Buffer(id_=5, size=32, usage=BufferUsage.QUERY_RESOLVE)
    enc.resolve_query_set(QuerySet(id_=9), 0, 2, dest, 0)
    assert fake_gpu.calls_of("copy_texture_to_texture") == [(1, src.id, dst.id, (4, 4, 1))]
    assert fake_gpu.calls_of("copy_texture_to_buffer") == [(1, src.id, 3, (4, 4, 1))]
    assert fake_gpu.calls_of("resolve_query_set") == [(1, 9, 0, 2, 5, 0)]


def test_render_pass_buffer_setters_wire_order(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.set_index_buffer(
        Buffer(id_=3, size=72, usage=BufferUsage.INDEX), IndexFormat.UINT32, offset=4
    )
    rp.set_vertex_buffer(0, Buffer(id_=5, size=60, usage=BufferUsage.VERTEX), offset=16)
    rp.draw(6, 2, 1, 0)
    rp.draw_indexed(6, 2, 1, 0, 1)
    rp.end()
    assert fake_gpu.calls_of("set_index_buffer") == [(1, 3, IndexFormat.UINT32, 4, None)]
    assert fake_gpu.calls_of("set_vertex_buffer") == [(1, 0, 5, 16)]
    assert fake_gpu.calls_of("draw") == [(1, 6, 2, 1, 0)]
    assert fake_gpu.calls_of("draw_indexed") == [(1, 6, 2, 1, 0, 1)]
    assert fake_gpu.calls_of("end_render_pass") == [(1,)]
