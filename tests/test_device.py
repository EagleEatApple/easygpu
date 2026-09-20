"""Adapter / Device / Queue wrappers integration tests."""

from __future__ import annotations

from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import BufferUsage
from easygpu.device import Adapter, Device, Queue, request_adapter
from easygpu.encoder import CommandEncoder, RenderPassColorAttachment
from easygpu.fake_gpu import FakeGPU
from easygpu.pipeline import RenderPipeline, RenderPipelineDescriptor, ShaderStageEntry
from easygpu.shader import ShaderModule, ShaderModuleDescriptor


def test_request_adapter_returns_wrapper(fake_gpu: FakeGPU) -> None:
    adapter = request_adapter()
    assert isinstance(adapter, Adapter)
    assert fake_gpu.calls_of("request_adapter") == [(False,)]


def test_device_creates_resources(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    buf = device.create_buffer(BufferDescriptor(size=60, usage=BufferUsage.VERTEX))
    sm = device.create_shader_module(ShaderModuleDescriptor(code="@vertex fn m() {}"))
    pipe = device.create_render_pipeline(
        RenderPipelineDescriptor(vertex_stage=ShaderStageEntry(module=sm))
    )
    enc = device.create_command_encoder()
    assert isinstance(buf, Buffer)
    assert isinstance(sm, ShaderModule)
    assert isinstance(pipe, RenderPipeline)
    assert isinstance(enc, CommandEncoder)
    assert fake_gpu.calls_of("create_buffer")[0][0] == 2
    assert fake_gpu.calls_of("create_shader_module")[0][0] == 2
    assert fake_gpu.calls_of("create_render_pipeline")[0][0] == 2
    assert fake_gpu.calls_of("create_command_encoder")[0][0] == 2


def test_queue_write_and_submit(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    queue = device.queue
    assert isinstance(queue, Queue)
    buf = device.create_buffer(BufferDescriptor(size=60, usage=BufferUsage.VERTEX))
    queue.write_buffer(buf, 0, b"\x00" * 60)
    assert fake_gpu.calls_of("write_buffer") == [(queue.id, buf.id, 0, b"\x00" * 60)]


def test_queue_submit_accepts_command_buffer(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    enc = device.create_command_encoder()
    enc.begin_render_pass([RenderPassColorAttachment(view=0)])
    cb = enc.finish()
    device.queue.submit([cb])
    assert fake_gpu.calls_of("submit") == [(device.queue.id, [cb.id])]
