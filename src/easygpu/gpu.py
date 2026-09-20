"""The WebGPU interface surface used by :mod:`easygpu`.

Every wrapper talks to "the GPU" only through a :class:`GPU` implementation.
This makes the library testable without a GPU: tests inject a recording
:class:`~easygpu.fake_gpu.FakeGPU`, while future versions will ship a real
wgpu-backed implementation. 0.0.0 ships no real backend — the Protocol IS
the deliverable.

The protocol speaks in named int handle aliases (``BufferHandle`` etc.) so
readers can tell which object each id refers to. Resource wrappers in
:mod:`easygpu.buffer` and friends translate between this handle seam and
type-safe Python objects; :mod:`easygpu.device` holds the Adapter/Device/Queue
wrappers. This module intentionally imports none of them at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from easygpu.errors import GPUValidationError

if TYPE_CHECKING:
    from easygpu.buffer import BufferDescriptor
    from easygpu.encoder import RenderPassColorAttachment
    from easygpu.pipeline import RenderPipelineDescriptor
    from easygpu.shader import ShaderModuleDescriptor

AdapterHandle = int
DeviceHandle = int
QueueHandle = int
BufferHandle = int
ShaderModuleHandle = int
RenderPipelineHandle = int
CommandBufferHandle = int
CommandEncoderHandle = int
RenderPassHandle = int


@runtime_checkable
class GPU(Protocol):
    """Minimal subset of the WebGPU API used by :mod:`easygpu`."""

    # --- adapter / device / queue -------------------------------------
    def request_adapter(self, *, force_fallback_adapter: bool = False) -> AdapterHandle: ...
    def request_device(
        self, adapter: AdapterHandle, *, label: str | None = None
    ) -> DeviceHandle: ...
    def get_queue(self, device: DeviceHandle) -> QueueHandle: ...

    # --- resource creation --------------------------------------------
    def create_buffer(self, device: DeviceHandle, descriptor: BufferDescriptor) -> BufferHandle: ...
    def create_shader_module(
        self, device: DeviceHandle, descriptor: ShaderModuleDescriptor
    ) -> ShaderModuleHandle: ...
    def create_render_pipeline(
        self, device: DeviceHandle, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineHandle: ...
    def create_command_encoder(
        self, device: DeviceHandle, *, label: str | None = None
    ) -> CommandEncoderHandle: ...

    # --- queue data path ----------------------------------------------
    def write_buffer(
        self, queue: QueueHandle, buffer: BufferHandle, buffer_offset: int, data: bytes
    ) -> None: ...
    def submit(self, queue: QueueHandle, command_buffers: list[CommandBufferHandle]) -> None: ...

    # --- command encoding ---------------------------------------------
    def begin_render_pass(
        self, encoder: CommandEncoderHandle, color_attachments: list[RenderPassColorAttachment]
    ) -> RenderPassHandle: ...
    def set_pipeline(
        self, render_pass: RenderPassHandle, pipeline: RenderPipelineHandle
    ) -> None: ...
    def set_vertex_buffer(
        self, render_pass: RenderPassHandle, slot: int, buffer: BufferHandle
    ) -> None: ...
    def draw(
        self, render_pass: RenderPassHandle, vertex_count: int, instance_count: int = 1
    ) -> None: ...
    def end(self, render_pass: RenderPassHandle) -> None: ...
    def finish(self, encoder: CommandEncoderHandle) -> CommandBufferHandle: ...

    # --- teardown -----------------------------------------------------
    def destroy_buffer(self, buffer: BufferHandle) -> None: ...
    def destroy_shader_module(self, module: ShaderModuleHandle) -> None: ...
    def destroy_render_pipeline(self, pipeline: RenderPipelineHandle) -> None: ...
    def destroy_command_encoder(self, encoder: CommandEncoderHandle) -> None: ...
    def destroy_device(self, device: DeviceHandle) -> None: ...


class _DefaultGPU:
    """In-process placeholder used before any :func:`configure` call.

    Records nothing and rejects every driver call so an unconfigured
    library fails loudly instead of silently doing nothing.
    """

    def _fail(self) -> None:
        raise GPUValidationError("no GPU configured: call easygpu.gpu.configure(gpu)")

    def request_adapter(self, *, force_fallback_adapter: bool = False) -> AdapterHandle:
        self._fail()
        return -1

    def request_device(self, adapter: AdapterHandle, *, label: str | None = None) -> DeviceHandle:
        self._fail()
        return -1

    def get_queue(self, device: DeviceHandle) -> QueueHandle:
        self._fail()
        return -1

    def create_buffer(self, device: DeviceHandle, descriptor: BufferDescriptor) -> BufferHandle:
        self._fail()
        return -1

    def create_shader_module(
        self, device: DeviceHandle, descriptor: ShaderModuleDescriptor
    ) -> ShaderModuleHandle:
        self._fail()
        return -1

    def create_render_pipeline(
        self, device: DeviceHandle, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineHandle:
        self._fail()
        return -1

    def create_command_encoder(
        self, device: DeviceHandle, *, label: str | None = None
    ) -> CommandEncoderHandle:
        self._fail()
        return -1

    def write_buffer(
        self, queue: QueueHandle, buffer: BufferHandle, buffer_offset: int, data: bytes
    ) -> None:
        self._fail()

    def submit(self, queue: QueueHandle, command_buffers: list[CommandBufferHandle]) -> None:
        self._fail()

    def begin_render_pass(
        self, encoder: CommandEncoderHandle, color_attachments: list[RenderPassColorAttachment]
    ) -> RenderPassHandle:
        self._fail()
        return -1

    def set_pipeline(self, render_pass: RenderPassHandle, pipeline: RenderPipelineHandle) -> None:
        self._fail()

    def set_vertex_buffer(
        self, render_pass: RenderPassHandle, slot: int, buffer: BufferHandle
    ) -> None:
        self._fail()

    def draw(
        self, render_pass: RenderPassHandle, vertex_count: int, instance_count: int = 1
    ) -> None:
        self._fail()

    def end(self, render_pass: RenderPassHandle) -> None:
        self._fail()

    def finish(self, encoder: CommandEncoderHandle) -> CommandBufferHandle:
        self._fail()
        return -1

    def destroy_buffer(self, buffer: BufferHandle) -> None:
        self._fail()

    def destroy_shader_module(self, module: ShaderModuleHandle) -> None:
        self._fail()

    def destroy_render_pipeline(self, pipeline: RenderPipelineHandle) -> None:
        self._fail()

    def destroy_command_encoder(self, encoder: CommandEncoderHandle) -> None:
        self._fail()

    def destroy_device(self, device: DeviceHandle) -> None:
        self._fail()


_default = _DefaultGPU()
_current: GPU = _default


def configure(gpu: GPU) -> None:
    """Replace the process-wide GPU implementation (used by tests)."""
    global _current
    _current = gpu


def get_gpu() -> GPU:
    """Return the configured :class:`GPU` implementation."""
    return _current


__all__ = ["GPU", "configure", "get_gpu"]
