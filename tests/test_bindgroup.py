"""Bind group layout / pipeline layout / bind group tests."""

from __future__ import annotations

from easygpu.bindgroup import (
    BindGroup,
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BufferSlice,
    PipelineLayout,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import BufferDescriptor
from easygpu.constants import (
    BufferBindingType,
    BufferUsage,
    SamplerBindingType,
    ShaderStage,
    TextureSampleType,
)
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from easygpu.sampler import Sampler
from easygpu.texture import Texture, TextureView


def test_layout_entry_defaults_only_binding_visibility() -> None:
    e = BindGroupLayoutEntry(binding=0, visibility=ShaderStage.VERTEX)
    assert e.buffer is None
    assert e.sampler is None
    assert e.texture is None


def test_create_bind_group_layout(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    entries = [
        BindGroupLayoutEntry(
            binding=0, visibility=ShaderStage.VERTEX, buffer=BufferBindingType.UNIFORM
        ),
        BindGroupLayoutEntry(binding=1, visibility=ShaderStage.FRAGMENT, sampler=True),
        BindGroupLayoutEntry(
            binding=2,
            visibility=ShaderStage.FRAGMENT,
            texture=TextureSampleType.FLOAT,
        ),
    ]
    descriptor = BindGroupLayoutDescriptor(entries=entries)
    layout = device.create_bind_group_layout(descriptor)
    assert isinstance(layout, BindGroupLayout)
    assert fake_gpu.calls_of("create_bind_group_layout") == [(2, descriptor)]


def test_create_pipeline_layout(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    bg0 = device.create_bind_group_layout(BindGroupLayoutDescriptor(entries=[]))
    descriptor = PipelineLayoutDescriptor(bind_group_layouts=[bg0])
    layout = device.create_pipeline_layout(descriptor)
    assert isinstance(layout, PipelineLayout)
    assert fake_gpu.calls_of("create_pipeline_layout") == [(2, descriptor)]


def test_create_bind_group_with_slices(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    buf = device.create_buffer(BufferDescriptor(size=256, usage=BufferUsage.UNIFORM))
    layout = device.create_bind_group_layout(BindGroupLayoutDescriptor(entries=[]))
    group = device.create_bind_group(
        BindGroupDescriptor(
            layout=layout,
            entries=[BindGroupEntry(binding=0, resource=BufferSlice(buf, offset=16))],
        )
    )
    assert isinstance(group, BindGroup)
    entries = fake_gpu.calls_of("create_bind_group")[0][1].entries
    assert entries == [BindGroupEntry(binding=0, resource=BufferSlice(buf, offset=16))]


def test_bind_group_delete_forwards_destroy(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    group = device.create_bind_group(BindGroupDescriptor(layout=0, entries=[]))
    group_id = group.id
    group.delete()
    assert fake_gpu.calls_of("destroy_bind_group") == [(group_id,)]


def test_layout_entry_sampler_accepts_sampler_binding_type() -> None:
    e = BindGroupLayoutEntry(
        binding=0, visibility=ShaderStage.COMPUTE, sampler=SamplerBindingType.COMPARISON
    )
    assert e.sampler == SamplerBindingType.COMPARISON


def test_layout_entry_sampler_true_equality_preserved() -> None:
    a = BindGroupLayoutEntry(binding=0, visibility=ShaderStage.COMPUTE, sampler=True)
    b = BindGroupLayoutEntry(binding=0, visibility=ShaderStage.COMPUTE, sampler=True)
    assert a == b


def test_pipeline_layout_descriptor_in_module_all() -> None:
    import easygpu.bindgroup

    assert "PipelineLayoutDescriptor" in easygpu.bindgroup.__all__


def test_bind_group_entry_accepts_sampler_and_view() -> None:
    sampler = Sampler(id_=9)
    assert BindGroupEntry(binding=1, resource=sampler).resource.id == 9
    tex = Texture(id_=11, size=(1, 1, 1), format=24, usage=1)
    view = TextureView(id_=10, texture=tex)
    assert BindGroupEntry(binding=2, resource=view).resource.id == 10
