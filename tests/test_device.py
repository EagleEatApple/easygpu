"""Adapter / Device / Queue wrappers integration tests."""

from __future__ import annotations

from easygpu.bindgroup import BindGroup, BindGroupDescriptor
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import BufferUsage, QueryType, TextureFormat, TextureUsage
from easygpu.device import Adapter, Device, Queue, request_adapter
from easygpu.encoder import (
    CommandEncoder,
    RenderBundleDescriptor,
    RenderBundleEncoder,
    RenderPassColorAttachment,
)
from easygpu.fake_gpu import FakeGPU
from easygpu.pipeline import (
    ComputePipeline,
    ComputePipelineDescriptor,
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
)
from easygpu.query import QuerySet, QuerySetDescriptor
from easygpu.sampler import Sampler
from easygpu.shader import ShaderModule, ShaderModuleDescriptor
from easygpu.texture import Texture, TextureDescriptor


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
    tex = device.create_texture(
        TextureDescriptor(
            size=(2, 2, 1), format=TextureFormat.RGBA8_UNORM, usage=TextureUsage.RENDER_ATTACHMENT
        )
    )
    sampler = device.create_sampler()
    group = device.create_bind_group(BindGroupDescriptor(layout=0, entries=[]))
    assert isinstance(buf, Buffer)
    assert isinstance(sm, ShaderModule)
    assert isinstance(pipe, RenderPipeline)
    assert isinstance(enc, CommandEncoder)
    assert isinstance(tex, Texture)
    assert isinstance(sampler, Sampler)
    assert isinstance(group, BindGroup)
    assert fake_gpu.calls_of("create_buffer")[0][0] == 2
    assert fake_gpu.calls_of("create_shader_module")[0][0] == 2
    assert fake_gpu.calls_of("create_render_pipeline")[0][0] == 2
    assert fake_gpu.calls_of("create_command_encoder")[0][0] == 2
    assert fake_gpu.calls_of("create_texture")[0][0] == 2
    assert fake_gpu.calls_of("create_sampler")[0][0] == 2
    assert fake_gpu.calls_of("create_bind_group")[0][0] == 2


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


def test_adapter_info_returns_recorded_dict(fake_gpu: FakeGPU) -> None:
    adapter = request_adapter()
    assert adapter.info == {
        "vendor": "fake",
        "architecture": "record",
        "device": "fake",
        "description": "FakeGPU",
    }
    assert fake_gpu.calls_of("get_adapter_info") == [(adapter.id,)]


def test_device_uncaptured_error_defaults_to_none(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    assert device.uncaptured_error is None
    assert fake_gpu.calls_of("on_uncaptured_error") == [(2,)]


def test_device_uncaptured_error_reads_recorded_error(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    fake_gpu.uncaptured_errors[2] = "boom"
    assert device.uncaptured_error == "boom"
    assert fake_gpu.calls_of("on_uncaptured_error") == [(2,)]


def test_device_creates_compute_pipeline(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    desc = ComputePipelineDescriptor(compute=ShaderStageEntry(module=0), label="add")
    pipe = device.create_compute_pipeline(desc)
    assert isinstance(pipe, ComputePipeline)
    assert pipe.label == "add"
    assert fake_gpu.compute_pipelines[pipe.id] is desc
    assert fake_gpu.calls_of("create_compute_pipeline") == [(2, desc)]


def test_device_creates_query_set(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    desc = QuerySetDescriptor(type=QueryType.TIMESTAMP, count=8, label="ts")
    query_set = device.create_query_set(desc)
    assert isinstance(query_set, QuerySet)
    assert query_set.label == "ts"
    assert fake_gpu.query_sets[query_set.id] is desc
    assert fake_gpu.calls_of("create_query_set") == [(2, desc)]


def test_queue_on_submitted_work_done(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    queue = device.queue
    queue.on_submitted_work_done()
    assert fake_gpu.calls_of("on_submitted_work_done") == [(queue.id,)]


def test_device_creates_render_bundle_encoder(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    desc = RenderBundleDescriptor(label="b")
    be = device.create_render_bundle_encoder(desc)
    assert isinstance(be, RenderBundleEncoder)
    assert fake_gpu.calls_of("create_render_bundle_encoder") == [(2, desc)]
