"""The full WebGPU surface used by EasyGPU.

Every wrapper talks to "the GPU" only through a :class:`GPU` implementation.
This makes the library testable without a GPU: tests inject a recording
:class:`~easygpu.fake_gpu.FakeGPU`, while future versions will ship a real
wgpu-backed implementation. 0.1.0 ships no real backend — the Protocol IS
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
    from easygpu.bindgroup import (
        BindGroupDescriptor,
        BindGroupLayoutDescriptor,
        PipelineLayoutDescriptor,
    )
    from easygpu.buffer import BufferDescriptor
    from easygpu.encoder import (
        RenderBundleDescriptor,
        RenderPassColorAttachment,
        RenderPassDepthStencilAttachment,
    )
    from easygpu.pipeline import ComputePipelineDescriptor, RenderPipelineDescriptor
    from easygpu.query import QuerySetDescriptor
    from easygpu.sampler import SamplerDescriptor
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.texture import TextureDescriptor, TextureViewDescriptor

AdapterHandle = int
DeviceHandle = int
QueueHandle = int
BufferHandle = int
TextureHandle = int
TextureViewHandle = int
SamplerHandle = int
ShaderModuleHandle = int
BindGroupLayoutHandle = int
BindGroupHandle = int
PipelineLayoutHandle = int
RenderPipelineHandle = int
ComputePipelineHandle = int
QuerySetHandle = int
CommandEncoderHandle = int
RenderPassHandle = int
ComputePassHandle = int
RenderBundleEncoderHandle = int
RenderBundleHandle = int
CommandBufferHandle = int


@runtime_checkable
class GPU(Protocol):
    """The full WebGPU surface used by EasyGPU."""

    # --- adapter / device / queue -------------------------------------
    def request_adapter(self, *, force_fallback_adapter: bool = False) -> AdapterHandle: ...
    def request_device(
        self, adapter: AdapterHandle, *, label: str | None = None
    ) -> DeviceHandle: ...
    def get_queue(self, device: DeviceHandle) -> QueueHandle: ...
    def get_adapter_info(self, adapter: AdapterHandle) -> dict[str, object]: ...
    def on_uncaptured_error(self, device: DeviceHandle) -> str | None: ...

    # --- resource creation --------------------------------------------
    def create_buffer(self, device: DeviceHandle, descriptor: BufferDescriptor) -> BufferHandle: ...
    def create_texture(
        self, device: DeviceHandle, descriptor: TextureDescriptor
    ) -> TextureHandle: ...
    def create_texture_view(
        self, texture: TextureHandle, descriptor: TextureViewDescriptor | None = None
    ) -> TextureViewHandle: ...
    def generate_mipmap(self, texture: TextureHandle) -> None: ...
    def create_sampler(
        self, device: DeviceHandle, descriptor: SamplerDescriptor
    ) -> SamplerHandle: ...
    def create_shader_module(
        self, device: DeviceHandle, descriptor: ShaderModuleDescriptor
    ) -> ShaderModuleHandle: ...
    def create_bind_group_layout(
        self, device: DeviceHandle, descriptor: BindGroupLayoutDescriptor
    ) -> BindGroupLayoutHandle: ...
    def create_pipeline_layout(
        self, device: DeviceHandle, descriptor: PipelineLayoutDescriptor
    ) -> PipelineLayoutHandle: ...
    def create_bind_group(
        self, device: DeviceHandle, descriptor: BindGroupDescriptor
    ) -> BindGroupHandle: ...
    def get_bind_group_layout(
        self,
        pipeline: RenderPipelineHandle | ComputePipelineHandle,
        group_index: int,
    ) -> BindGroupLayoutHandle: ...
    def create_render_pipeline(
        self, device: DeviceHandle, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineHandle: ...
    def create_compute_pipeline(
        self, device: DeviceHandle, descriptor: ComputePipelineDescriptor
    ) -> ComputePipelineHandle: ...
    def create_query_set(
        self, device: DeviceHandle, descriptor: QuerySetDescriptor
    ) -> QuerySetHandle: ...
    def create_command_encoder(
        self, device: DeviceHandle, *, label: str | None = None
    ) -> CommandEncoderHandle: ...

    # --- queue data path ----------------------------------------------
    def write_buffer(
        self,
        queue: QueueHandle,
        buffer: BufferHandle,
        buffer_offset: int,
        data: bytes,
    ) -> None: ...
    def map_async(self, buffer: BufferHandle, mode: int, offset: int, size: int) -> None: ...
    def get_mapped_range(self, buffer: BufferHandle, offset: int, size: int) -> bytes: ...
    def unmap(self, buffer: BufferHandle) -> None: ...
    def write_texture(
        self,
        queue: QueueHandle,
        texture: TextureHandle,
        data: bytes,
        *,
        width: int,
        height: int,
        mip_level: int = 0,
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None: ...
    def submit(
        self,
        queue: QueueHandle,
        command_buffers: list[CommandBufferHandle],
    ) -> None: ...
    def on_submitted_work_done(self, queue: QueueHandle) -> None: ...

    # --- command encoding ---------------------------------------------
    def begin_render_pass(
        self,
        encoder: CommandEncoderHandle,
        color_attachments: list[RenderPassColorAttachment],
        depth_stencil_attachment: RenderPassDepthStencilAttachment | None = None,
    ) -> RenderPassHandle: ...
    def set_pipeline(
        self, render_pass: RenderPassHandle, pipeline: RenderPipelineHandle
    ) -> None: ...
    def set_bind_group(
        self,
        render_pass: RenderPassHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None: ...
    def set_vertex_buffer(
        self,
        render_pass: RenderPassHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None: ...
    def set_index_buffer(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None: ...
    def draw(
        self,
        render_pass: RenderPassHandle,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None: ...
    def draw_indexed(
        self,
        render_pass: RenderPassHandle,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None: ...
    def draw_indirect(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None: ...
    def draw_indexed_indirect(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None: ...
    def set_viewport(
        self,
        render_pass: RenderPassHandle,
        x: float,
        y: float,
        w: float,
        h: float,
        min_depth: float,
        max_depth: float,
    ) -> None: ...
    def set_scissor_rect(
        self,
        render_pass: RenderPassHandle,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None: ...
    def set_blend_constant(
        self, render_pass: RenderPassHandle, color: tuple[float, float, float, float]
    ) -> None: ...
    def execute_render_bundles(
        self,
        render_pass: RenderPassHandle,
        bundles: list[RenderBundleHandle],
    ) -> None: ...
    def write_timestamp(
        self,
        render_pass: RenderPassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None: ...
    def end_render_pass(self, render_pass: RenderPassHandle) -> None: ...
    def create_render_bundle_encoder(
        self, device: DeviceHandle, descriptor: RenderBundleDescriptor
    ) -> RenderBundleEncoderHandle: ...
    def begin_compute_pass(
        self, encoder: CommandEncoderHandle, *, label: str | None = None
    ) -> ComputePassHandle: ...
    def compute_set_pipeline(
        self,
        compute_pass: ComputePassHandle,
        pipeline: ComputePipelineHandle,
    ) -> None: ...
    def compute_set_bind_group(
        self,
        compute_pass: ComputePassHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None: ...
    def dispatch_workgroups(
        self,
        compute_pass: ComputePassHandle,
        x: int,
        y: int = 1,
        z: int = 1,
    ) -> None: ...
    def dispatch_workgroups_indirect(
        self,
        compute_pass: ComputePassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None: ...
    def write_compute_timestamp(
        self,
        compute_pass: ComputePassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None: ...
    def end_compute_pass(self, compute_pass: ComputePassHandle) -> None: ...
    def bundle_set_pipeline(
        self,
        encoder: RenderBundleEncoderHandle,
        pipeline: RenderPipelineHandle,
    ) -> None: ...
    def bundle_set_bind_group(
        self,
        encoder: RenderBundleEncoderHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None: ...
    def bundle_set_vertex_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None: ...
    def bundle_set_index_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None: ...
    def bundle_draw(
        self,
        encoder: RenderBundleEncoderHandle,
        vertex_count: int,
        instance_count: int = 1,
    ) -> None: ...
    def bundle_draw_indexed(
        self,
        encoder: RenderBundleEncoderHandle,
        index_count: int,
        instance_count: int = 1,
    ) -> None: ...
    def bundle_finish(self, encoder: RenderBundleEncoderHandle) -> RenderBundleHandle: ...
    def copy_texture_to_texture(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: TextureHandle,
        *,
        size: tuple[int, int, int],
    ) -> None: ...
    def copy_texture_to_buffer(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: BufferHandle,
        *,
        size: tuple[int, int, int],
    ) -> None: ...
    def resolve_query_set(
        self,
        encoder: CommandEncoderHandle,
        query_set: QuerySetHandle,
        first_query: int,
        query_count: int,
        destination: BufferHandle,
        destination_offset: int,
    ) -> None: ...
    def finish(self, encoder: CommandEncoderHandle) -> CommandBufferHandle: ...

    # --- teardown ------------------------------------------------------
    def destroy_buffer(self, buffer: BufferHandle) -> None: ...
    def destroy_texture(self, texture: TextureHandle) -> None: ...
    def destroy_texture_view(self, view: TextureViewHandle) -> None: ...
    def destroy_sampler(self, sampler: SamplerHandle) -> None: ...
    def destroy_shader_module(self, module: ShaderModuleHandle) -> None: ...
    def destroy_bind_group_layout(self, layout: BindGroupLayoutHandle) -> None: ...
    def destroy_pipeline_layout(self, layout: PipelineLayoutHandle) -> None: ...
    def destroy_bind_group(self, group: BindGroupHandle) -> None: ...
    def destroy_render_pipeline(self, pipeline: RenderPipelineHandle) -> None: ...
    def destroy_compute_pipeline(self, pipeline: ComputePipelineHandle) -> None: ...
    def destroy_query_set(self, query_set: QuerySetHandle) -> None: ...
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

    def get_adapter_info(self, adapter: AdapterHandle) -> dict[str, object]:
        self._fail()
        return {}

    def on_uncaptured_error(self, device: DeviceHandle) -> str | None:
        self._fail()
        return None

    def create_buffer(self, device: DeviceHandle, descriptor: BufferDescriptor) -> BufferHandle:
        self._fail()
        return -1

    def create_texture(self, device: DeviceHandle, descriptor: TextureDescriptor) -> TextureHandle:
        self._fail()
        return -1

    def create_texture_view(
        self, texture: TextureHandle, descriptor: TextureViewDescriptor | None = None
    ) -> TextureViewHandle:
        self._fail()
        return -1

    def generate_mipmap(self, texture: TextureHandle) -> None:
        self._fail()

    def create_sampler(self, device: DeviceHandle, descriptor: SamplerDescriptor) -> SamplerHandle:
        self._fail()
        return -1

    def create_shader_module(
        self, device: DeviceHandle, descriptor: ShaderModuleDescriptor
    ) -> ShaderModuleHandle:
        self._fail()
        return -1

    def create_bind_group_layout(
        self, device: DeviceHandle, descriptor: BindGroupLayoutDescriptor
    ) -> BindGroupLayoutHandle:
        self._fail()
        return -1

    def create_pipeline_layout(
        self, device: DeviceHandle, descriptor: PipelineLayoutDescriptor
    ) -> PipelineLayoutHandle:
        self._fail()
        return -1

    def create_bind_group(
        self, device: DeviceHandle, descriptor: BindGroupDescriptor
    ) -> BindGroupHandle:
        self._fail()
        return -1

    def get_bind_group_layout(
        self,
        pipeline: RenderPipelineHandle | ComputePipelineHandle,
        group_index: int,
    ) -> BindGroupLayoutHandle:
        self._fail()
        return -1

    def create_render_pipeline(
        self, device: DeviceHandle, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineHandle:
        self._fail()
        return -1

    def create_compute_pipeline(
        self, device: DeviceHandle, descriptor: ComputePipelineDescriptor
    ) -> ComputePipelineHandle:
        self._fail()
        return -1

    def create_query_set(
        self, device: DeviceHandle, descriptor: QuerySetDescriptor
    ) -> QuerySetHandle:
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

    def map_async(self, buffer: BufferHandle, mode: int, offset: int, size: int) -> None:
        self._fail()

    def get_mapped_range(self, buffer: BufferHandle, offset: int, size: int) -> bytes:
        self._fail()
        return b""

    def unmap(self, buffer: BufferHandle) -> None:
        self._fail()

    def write_texture(
        self,
        queue: QueueHandle,
        texture: TextureHandle,
        data: bytes,
        *,
        width: int,
        height: int,
        mip_level: int = 0,
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._fail()

    def submit(self, queue: QueueHandle, command_buffers: list[CommandBufferHandle]) -> None:
        self._fail()

    def on_submitted_work_done(self, queue: QueueHandle) -> None:
        self._fail()

    def begin_render_pass(
        self,
        encoder: CommandEncoderHandle,
        color_attachments: list[RenderPassColorAttachment],
        depth_stencil_attachment: RenderPassDepthStencilAttachment | None = None,
    ) -> RenderPassHandle:
        self._fail()
        return -1

    def set_pipeline(self, render_pass: RenderPassHandle, pipeline: RenderPipelineHandle) -> None:
        self._fail()

    def set_bind_group(
        self,
        render_pass: RenderPassHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None:
        self._fail()

    def set_vertex_buffer(
        self,
        render_pass: RenderPassHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None:
        self._fail()

    def set_index_buffer(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._fail()

    def draw(
        self,
        render_pass: RenderPassHandle,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._fail()

    def draw_indexed(
        self,
        render_pass: RenderPassHandle,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._fail()

    def draw_indirect(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None:
        self._fail()

    def draw_indexed_indirect(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None:
        self._fail()

    def set_viewport(
        self,
        render_pass: RenderPassHandle,
        x: float,
        y: float,
        w: float,
        h: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        self._fail()

    def set_scissor_rect(
        self,
        render_pass: RenderPassHandle,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        self._fail()

    def set_blend_constant(
        self, render_pass: RenderPassHandle, color: tuple[float, float, float, float]
    ) -> None:
        self._fail()

    def execute_render_bundles(
        self,
        render_pass: RenderPassHandle,
        bundles: list[RenderBundleHandle],
    ) -> None:
        self._fail()

    def write_timestamp(
        self,
        render_pass: RenderPassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None:
        self._fail()

    def end_render_pass(self, render_pass: RenderPassHandle) -> None:
        self._fail()

    def create_render_bundle_encoder(
        self, device: DeviceHandle, descriptor: RenderBundleDescriptor
    ) -> RenderBundleEncoderHandle:
        self._fail()
        return -1

    def finish(self, encoder: CommandEncoderHandle) -> CommandBufferHandle:
        self._fail()
        return -1

    def begin_compute_pass(
        self, encoder: CommandEncoderHandle, *, label: str | None = None
    ) -> ComputePassHandle:
        self._fail()
        return -1

    def compute_set_pipeline(
        self,
        compute_pass: ComputePassHandle,
        pipeline: ComputePipelineHandle,
    ) -> None:
        self._fail()

    def compute_set_bind_group(
        self,
        compute_pass: ComputePassHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None:
        self._fail()

    def dispatch_workgroups(
        self,
        compute_pass: ComputePassHandle,
        x: int,
        y: int = 1,
        z: int = 1,
    ) -> None:
        self._fail()

    def dispatch_workgroups_indirect(
        self,
        compute_pass: ComputePassHandle,
        buffer: BufferHandle,
        offset: int,
    ) -> None:
        self._fail()

    def write_compute_timestamp(
        self,
        compute_pass: ComputePassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None:
        self._fail()

    def end_compute_pass(self, compute_pass: ComputePassHandle) -> None:
        self._fail()

    def bundle_set_pipeline(
        self,
        encoder: RenderBundleEncoderHandle,
        pipeline: RenderPipelineHandle,
    ) -> None:
        self._fail()

    def bundle_set_bind_group(
        self,
        encoder: RenderBundleEncoderHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None:
        self._fail()

    def bundle_set_vertex_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None:
        self._fail()

    def bundle_set_index_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._fail()

    def bundle_draw(
        self,
        encoder: RenderBundleEncoderHandle,
        vertex_count: int,
        instance_count: int = 1,
    ) -> None:
        self._fail()

    def bundle_draw_indexed(
        self,
        encoder: RenderBundleEncoderHandle,
        index_count: int,
        instance_count: int = 1,
    ) -> None:
        self._fail()

    def bundle_finish(self, encoder: RenderBundleEncoderHandle) -> RenderBundleHandle:
        self._fail()
        return -1

    def copy_texture_to_texture(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: TextureHandle,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self._fail()

    def copy_texture_to_buffer(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: BufferHandle,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self._fail()

    def resolve_query_set(
        self,
        encoder: CommandEncoderHandle,
        query_set: QuerySetHandle,
        first_query: int,
        query_count: int,
        destination: BufferHandle,
        destination_offset: int,
    ) -> None:
        self._fail()

    def destroy_buffer(self, buffer: BufferHandle) -> None:
        self._fail()

    def destroy_texture(self, texture: TextureHandle) -> None:
        self._fail()

    def destroy_texture_view(self, view: TextureViewHandle) -> None:
        self._fail()

    def destroy_sampler(self, sampler: SamplerHandle) -> None:
        self._fail()

    def destroy_shader_module(self, module: ShaderModuleHandle) -> None:
        self._fail()

    def destroy_bind_group_layout(self, layout: BindGroupLayoutHandle) -> None:
        self._fail()

    def destroy_pipeline_layout(self, layout: PipelineLayoutHandle) -> None:
        self._fail()

    def destroy_bind_group(self, group: BindGroupHandle) -> None:
        self._fail()

    def destroy_render_pipeline(self, pipeline: RenderPipelineHandle) -> None:
        self._fail()

    def destroy_compute_pipeline(self, pipeline: ComputePipelineHandle) -> None:
        self._fail()

    def destroy_query_set(self, query_set: QuerySetHandle) -> None:
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
