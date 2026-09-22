"""Adapter, Device and Queue wrappers for :mod:`easygpu`.

The GPU protocol seam (:class:`easygpu.gpu.GPU`) speaks raw int handle ids.
These wrappers give type-safe Python objects to library users and translate
wrapper calls into handle calls on the configured GPU implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.bindgroup import (
    BindGroup,
    BindGroupDescriptor,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    PipelineLayout,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.encoder import (
    CommandBuffer,
    CommandEncoder,
    RenderBundleDescriptor,
    RenderBundleEncoder,
)
from easygpu.pipeline import (
    ComputePipeline,
    ComputePipelineDescriptor,
    RenderPipeline,
    RenderPipelineDescriptor,
)
from easygpu.query import QuerySet, QuerySetDescriptor
from easygpu.sampler import Sampler, SamplerDescriptor
from easygpu.shader import ShaderModule, ShaderModuleDescriptor
from easygpu.texture import Texture, TextureDescriptor

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


class Adapter(GPUObject):
    """A GPU adapter that can request a :class:`Device`."""

    @property
    def info(self) -> dict[str, object]:
        return _gpu().get_adapter_info(self.id)

    def request_device(self, *, label: str | None = None) -> Device:
        device_id = _gpu().request_device(self.id, label=label)
        return Device(device_id, label=label)


class Device(GPUObject):
    """The logical GPU device; owns resource factories."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)
        self._queue: Queue | None = None

    def create_buffer(self, descriptor: BufferDescriptor) -> Buffer:
        buffer_id = _gpu().create_buffer(self.id, descriptor)
        return Buffer(
            buffer_id, size=descriptor.size, usage=descriptor.usage, label=descriptor.label
        )

    def create_shader_module(self, descriptor: ShaderModuleDescriptor) -> ShaderModule:
        module_id = _gpu().create_shader_module(self.id, descriptor)
        return ShaderModule(module_id, label=descriptor.label)

    def create_render_pipeline(self, descriptor: RenderPipelineDescriptor) -> RenderPipeline:
        pipeline_id = _gpu().create_render_pipeline(self.id, descriptor)
        return RenderPipeline(pipeline_id, label=descriptor.label)

    def create_compute_pipeline(self, descriptor: ComputePipelineDescriptor) -> ComputePipeline:
        pipeline_id = _gpu().create_compute_pipeline(self.id, descriptor)
        return ComputePipeline(pipeline_id, label=descriptor.label)

    def create_query_set(self, descriptor: QuerySetDescriptor) -> QuerySet:
        query_set_id = _gpu().create_query_set(self.id, descriptor)
        return QuerySet(query_set_id, label=descriptor.label)

    def create_texture(self, descriptor: TextureDescriptor) -> Texture:
        texture_id = _gpu().create_texture(self.id, descriptor)
        return Texture(
            texture_id,
            size=descriptor.size,
            format=descriptor.format,
            usage=descriptor.usage,
            sample_count=descriptor.sample_count,
            mip_level_count=descriptor.mip_level_count,
            label=descriptor.label,
        )

    def create_sampler(self, descriptor: SamplerDescriptor | None = None) -> Sampler:
        descriptor = descriptor or SamplerDescriptor()
        sampler_id = _gpu().create_sampler(self.id, descriptor)
        return Sampler(sampler_id, label=descriptor.label)

    def create_bind_group_layout(self, descriptor: BindGroupLayoutDescriptor) -> BindGroupLayout:
        layout_id = _gpu().create_bind_group_layout(self.id, descriptor)
        return BindGroupLayout(layout_id, label=descriptor.label)

    def create_pipeline_layout(self, descriptor: PipelineLayoutDescriptor) -> PipelineLayout:
        layout_id = _gpu().create_pipeline_layout(self.id, descriptor)
        return PipelineLayout(layout_id, label=descriptor.label)

    def create_bind_group(self, descriptor: BindGroupDescriptor) -> BindGroup:
        group_id = _gpu().create_bind_group(self.id, descriptor)
        return BindGroup(group_id, label=descriptor.label)

    def create_command_encoder(self, *, label: str | None = None) -> CommandEncoder:
        encoder_id = _gpu().create_command_encoder(self.id, label=label)
        return CommandEncoder(encoder_id, label=label)

    def create_render_bundle_encoder(
        self, descriptor: RenderBundleDescriptor
    ) -> RenderBundleEncoder:
        encoder_id = _gpu().create_render_bundle_encoder(self.id, descriptor)
        return RenderBundleEncoder(self, encoder_id)

    @property
    def uncaptured_error(self) -> str | None:
        return _gpu().on_uncaptured_error(self.id)

    def _delete_impl(self) -> None:
        _gpu().destroy_device(self._id)

    @property
    def queue(self) -> Queue:
        if self._queue is None:
            queue_id = _gpu().get_queue(self.id)
            self._queue = Queue(queue_id, self)
        return self._queue


class Queue(GPUObject):
    """A submission queue belonging to a :class:`Device`."""

    def __init__(self, id_: int, device: Device) -> None:
        super().__init__(id_)
        self._device = device

    def write_buffer(self, buffer: Buffer, buffer_offset: int, data: bytes) -> None:
        _gpu().write_buffer(self.id, buffer.id, buffer_offset, data)

    def write_texture(
        self,
        texture: Texture,
        data: bytes,
        *,
        width: int,
        height: int,
    ) -> None:
        _gpu().write_texture(self.id, texture.id, data, width=width, height=height)

    def submit(self, command_buffers: list[CommandBuffer]) -> None:
        _gpu().submit(
            self.id, [cb.id if isinstance(cb, CommandBuffer) else cb for cb in command_buffers]
        )

    def on_submitted_work_done(self) -> None:
        _gpu().on_submitted_work_done(self.id)


def request_adapter(*, force_fallback_adapter: bool = False) -> Adapter:
    adapter_id = _gpu().request_adapter(force_fallback_adapter=force_fallback_adapter)
    return Adapter(adapter_id)


__all__ = [
    "Adapter",
    "BindGroup",
    "BindGroupLayout",
    "BindGroupLayoutDescriptor",
    "Device",
    "PipelineLayout",
    "PipelineLayoutDescriptor",
    "Queue",
    "Sampler",
    "Texture",
    "request_adapter",
]
