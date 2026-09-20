"""Adapter, Device and Queue wrappers for :mod:`easygpu`.

The GPU protocol seam (:class:`easygpu.gpu.GPU`) speaks raw int handle ids.
These wrappers give type-safe Python objects to library users and translate
wrapper calls into handle calls on the configured GPU implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.encoder import CommandBuffer, CommandEncoder
from easygpu.pipeline import RenderPipeline, RenderPipelineDescriptor
from easygpu.shader import ShaderModule, ShaderModuleDescriptor

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


class Adapter(GPUObject):
    """A GPU adapter that can request a :class:`Device`."""

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

    def create_command_encoder(self, *, label: str | None = None) -> CommandEncoder:
        encoder_id = _gpu().create_command_encoder(self.id, label=label)
        return CommandEncoder(encoder_id, label=label)

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

    def submit(self, command_buffers: list[CommandBuffer]) -> None:
        _gpu().submit(
            self.id, [cb.id if isinstance(cb, CommandBuffer) else cb for cb in command_buffers]
        )


def request_adapter(*, force_fallback_adapter: bool = False) -> Adapter:
    adapter_id = _gpu().request_adapter(force_fallback_adapter=force_fallback_adapter)
    return Adapter(adapter_id)


__all__ = ["Adapter", "Device", "Queue", "request_adapter"]
