"""Texture / TextureView / Sampler tests."""

from __future__ import annotations

from easygpu.constants import TextureFormat, TextureUsage
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from easygpu.sampler import Sampler, SamplerDescriptor
from easygpu.texture import (
    Texture,
    TextureDescriptor,
    TextureView,
    TextureViewDescriptor,
)


def test_texture_descriptor_defaults() -> None:
    d = TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    assert d.mip_level_count == 1
    assert d.sample_count == 1
    assert d.dimension == "2d"
    assert d.label is None


def test_sampler_descriptor_defaults() -> None:
    d = SamplerDescriptor()
    assert d.mag_filter == 0
    assert d.min_filter == 0
    assert d.mipmap_filter == 0
    assert d.address_mode_u == 0
    assert d.compare is None


def test_create_texture_records_descriptor(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    descriptor = TextureDescriptor(
        size=(4, 4, 1),
        format=TextureFormat.RGBA8_UNORM,
        usage=TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST,
    )
    tex = device.create_texture(descriptor)
    assert isinstance(tex, Texture)
    assert fake_gpu.calls_of("create_texture") == [(2, descriptor)]
    assert tex.width == 4
    assert tex.height == 4
    assert tex.size == (4, 4, 1)
    assert tex.format == TextureFormat.RGBA8_UNORM
    assert tex.usage == TextureUsage.TEXTURE_BINDING | TextureUsage.COPY_DST


def test_texture_create_view(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    )
    view = tex.create_view()
    assert isinstance(view, TextureView)
    assert view.texture is tex
    assert fake_gpu.calls_of("create_texture_view") == [(tex.id, None)]


def test_create_sampler_records(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    sampler = device.create_sampler()
    assert isinstance(sampler, Sampler)
    assert fake_gpu.calls_of("create_sampler") == [(2, SamplerDescriptor())]


def test_texture_delete_forwards_destroy(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    )
    view = tex.create_view()
    view_id = view.id
    tex_id = tex.id
    view.delete()
    assert fake_gpu.calls_of("destroy_texture_view") == [(view_id,)]
    tex.delete()
    assert fake_gpu.calls_of("destroy_texture") == [(tex_id,)]


def test_queue_write_texture(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(
            size=(2, 2, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.COPY_DST | TextureUsage.TEXTURE_BINDING,
        )
    )
    device.queue.write_texture(tex, b"\xff" * 16, width=2, height=2)
    args = fake_gpu.calls_of("write_texture")
    assert args == [(device.queue.id, tex.id, b"\xff" * 16, 2, 2, 0, (0, 0, 0))]
    assert fake_gpu.textures[tex.id].width == 2
    assert fake_gpu.textures[tex.id].height == 2


def test_create_view_records_descriptor(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    )
    tex.create_view(None)
    assert fake_gpu.calls_of("create_texture_view") == [(tex.id, None)]


def test_create_view_with_descriptor(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    )
    view = tex.create_view(TextureViewDescriptor(dimension="2d"))
    assert isinstance(view, TextureView)
    assert fake_gpu.calls_of("create_texture_view")[-1] == (
        tex.id,
        TextureViewDescriptor(dimension="2d"),
    )


def test_generate_mipmap_records(fake_gpu: FakeGPU) -> None:
    device = Device(id_=2)
    tex = device.create_texture(
        TextureDescriptor(size=(4, 4, 1), format=TextureFormat.RGBA8_UNORM, usage=1)
    )
    tex.generate_mipmap()
    assert fake_gpu.calls_of("generate_mipmap") == [(tex.id,)]
