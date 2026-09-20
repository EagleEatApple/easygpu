"""CommandEncoder / RenderPassEncoder tests."""

from __future__ import annotations

import pytest

from easygpu.buffer import Buffer
from easygpu.constants import BufferUsage, LoadOp, StoreOp
from easygpu.encoder import (
    CommandBuffer,
    CommandEncoder,
    RenderPassColorAttachment,
    RenderPassEncoder,
)
from easygpu.errors import GPUValidationError
from easygpu.fake_gpu import FakeGPU
from easygpu.pipeline import RenderPipeline


def test_color_attachment_defaults() -> None:
    att = RenderPassColorAttachment(view=0)
    assert att.clear_value == (0.0, 0.0, 0.0, 1.0)
    assert att.load_op == LoadOp.CLEAR
    assert att.store_op == StoreOp.STORE


def test_command_buffer_exposes_id() -> None:
    assert CommandBuffer(id_=8).id == 8


def test_begin_render_pass_returns_encoder(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    assert isinstance(rp, RenderPassEncoder)
    assert fake_gpu.calls_of("begin_render_pass") == [(1, [RenderPassColorAttachment(view=0)])]


def test_finish_records_and_returns_command_buffer(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    buf = enc.finish()
    assert isinstance(buf, CommandBuffer)
    assert fake_gpu.calls_of("finish")[0][0] == 1


def test_use_after_finish_raises(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    enc.finish()
    with pytest.raises(GPUValidationError):
        enc.finish()
    with pytest.raises(GPUValidationError):
        enc.begin_render_pass([RenderPassColorAttachment(view=0)])


def test_render_pass_records_draw_sequence(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    pipe = RenderPipeline(id_=2)
    vb = Buffer(id_=3, size=60, usage=BufferUsage.VERTEX)
    rp.set_pipeline(pipe)
    rp.set_vertex_buffer(0, vb)
    rp.draw(3)
    rp.draw(3, 2)
    rp.end()
    assert fake_gpu.calls_of("set_pipeline") == [(1, 2)]
    assert fake_gpu.calls_of("set_vertex_buffer") == [(1, 0, 3)]
    assert fake_gpu.calls_of("draw") == [(1, 3, 1), (1, 3, 2)]
    assert fake_gpu.calls_of("end") == [(1,)]


def test_double_end_raises(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    rp = enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    rp.end()
    with pytest.raises(GPUValidationError):
        rp.end()


def test_render_pass_context_manager_auto_ends(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    with enc.begin_render_pass([RenderPassColorAttachment(view=0)]) as rp:
        assert isinstance(rp, RenderPassEncoder)
        rp.draw(3)
    assert fake_gpu.calls_of("end") == [(1,)]


def test_render_pass_context_manager_no_second_end(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    with enc.begin_render_pass([RenderPassColorAttachment(view=0)]) as rp:
        rp.end()
    assert fake_gpu.calls_of("end") == [(1,)]


def test_render_pass_context_manager_preserves_body_exception(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=1)
    with (
        pytest.raises(ValueError),
        enc.begin_render_pass([RenderPassColorAttachment(view=0)]),
    ):
        raise ValueError("boom")
    assert fake_gpu.calls_of("end") == [(1,)]


def test_command_encoder_delete_forwards_destroy(fake_gpu: FakeGPU) -> None:
    enc = CommandEncoder(id_=7)
    enc.delete()
    assert fake_gpu.calls_of("destroy_command_encoder") == [(7,)]
