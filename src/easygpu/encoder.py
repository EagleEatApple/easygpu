"""Command encoder / render pass wrappers for :mod:`easygpu`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from easygpu.base import GPUObject
from easygpu.bindgroup import BindGroup
from easygpu.buffer import Buffer
from easygpu.constants import IndexFormat, LoadOp, StoreOp
from easygpu.errors import GPUValidationError
from easygpu.pipeline import ComputePipeline, RenderPipeline
from easygpu.query import QuerySet
from easygpu.texture import TextureView

if TYPE_CHECKING:
    from easygpu.device import Device
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class ComputePassDescriptor:
    """WebGPU ``GPUComputePassDescriptor`` placeholder."""

    label: str | None = None


@dataclass(frozen=True)
class RenderPassColorAttachment:
    """WebGPU ``GPURenderPassColorAttachment``."""

    view: int | TextureView
    clear_value: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    load_op: LoadOp = LoadOp.CLEAR
    store_op: StoreOp = StoreOp.STORE
    resolve_target: int | TextureView | None = None


@dataclass(frozen=True)
class RenderPassDepthStencilAttachment:
    """WebGPU ``GPURenderPassDepthStencilAttachment`` (depth subset)."""

    view: int | TextureView
    depth_clear_value: float = 1.0
    depth_load_op: LoadOp = LoadOp.CLEAR
    depth_store_op: StoreOp = StoreOp.STORE
    stencil_clear_value: int = 0
    stencil_load_op: LoadOp = LoadOp.CLEAR
    stencil_store_op: StoreOp = StoreOp.STORE


@dataclass(frozen=True)
class RenderBundleDescriptor:
    """WebGPU ``GPURenderBundleEncoderDescriptor`` placeholder."""

    label: str | None = None


class CommandBuffer:
    """Opaque handle to a finished command buffer (ready for queue.submit)."""

    def __init__(self, id_: int) -> None:
        self._id = id_

    @property
    def id(self) -> int:
        return self._id


class RenderPassEncoder:
    """Encodes draw commands inside a render pass."""

    def __init__(self, encoder: CommandEncoder, id_: int) -> None:
        self._encoder = encoder
        self._id = id_
        self._ended = False

    def _require_open(self) -> None:
        if self._ended:
            raise GPUValidationError("render pass has already ended")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self._ended:
            self.end()

    def set_pipeline(self, pipeline: RenderPipeline) -> None:
        self._require_open()
        _gpu().set_pipeline(self._id, pipeline.id)

    def set_vertex_buffer(self, slot: int, buffer: Buffer, offset: int = 0) -> None:
        self._require_open()
        _gpu().set_vertex_buffer(self._id, slot, buffer.id, offset)

    def set_index_buffer(
        self,
        buffer: Buffer,
        index_format: IndexFormat,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._require_open()
        _gpu().set_index_buffer(self._id, buffer.id, index_format, offset, size)

    def set_bind_group(self, group_index: int, bind_group: BindGroup) -> None:
        self._require_open()
        _gpu().set_bind_group(self._id, group_index, bind_group.id)

    def set_viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
    ) -> None:
        self._require_open()
        _gpu().set_viewport(self._id, x, y, width, height, min_depth, max_depth)

    def set_blend_constant(self, color: tuple[float, float, float, float]) -> None:
        self._require_open()
        _gpu().set_blend_constant(self._id, color)

    def draw(
        self,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._require_open()
        _gpu().draw(self._id, vertex_count, instance_count, first_vertex, first_instance)

    def draw_indexed(
        self,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._require_open()
        _gpu().draw_indexed(
            self._id, index_count, instance_count, first_index, base_vertex, first_instance
        )

    def set_scissor_rect(self, x: int, y: int, w: int, h: int) -> None:
        self._require_open()
        _gpu().set_scissor_rect(self._id, x, y, w, h)

    def draw_indirect(self, buffer: Buffer, offset: int = 0) -> None:
        self._require_open()
        _gpu().draw_indirect(self._id, buffer.id, offset)

    def draw_indexed_indirect(self, buffer: Buffer, offset: int = 0) -> None:
        self._require_open()
        _gpu().draw_indexed_indirect(self._id, buffer.id, offset)

    def execute_render_bundles(self, bundles: list[RenderBundle]) -> None:
        self._require_open()
        _gpu().execute_render_bundles(self._id, [bundle.id for bundle in bundles])

    def write_timestamp(self, query_set: QuerySet, query_index: int) -> None:
        self._require_open()
        _gpu().write_timestamp(self._id, query_set.id, query_index)

    def end(self) -> None:
        self._require_open()
        _gpu().end_render_pass(self._id)
        self._ended = True


class RenderBundle:
    """Opaque handle to an encoded render bundle (ready for pass execution)."""

    def __init__(self, id_: int) -> None:
        self._id = id_

    @property
    def id(self) -> int:
        return self._id


class RenderBundleEncoder:
    """Encodes a reusable batch of draw commands (a render bundle)."""

    def __init__(self, device: Device, id_: int) -> None:
        self._device = device
        self._id = id_

    def set_pipeline(self, pipeline: RenderPipeline) -> None:
        _gpu().bundle_set_pipeline(self._id, pipeline.id)

    def set_bind_group(self, group_index: int, group: BindGroup) -> None:
        _gpu().bundle_set_bind_group(self._id, group_index, group.id)

    def set_vertex_buffer(self, slot: int, buffer: Buffer, offset: int = 0) -> None:
        _gpu().bundle_set_vertex_buffer(self._id, slot, buffer.id, offset)

    def set_index_buffer(
        self,
        buffer: Buffer,
        index_format: IndexFormat,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        _gpu().bundle_set_index_buffer(self._id, buffer.id, index_format, offset, size)

    def draw(self, vertex_count: int, instance_count: int = 1) -> None:
        _gpu().bundle_draw(self._id, vertex_count, instance_count)

    def draw_indexed(self, index_count: int, instance_count: int = 1) -> None:
        _gpu().bundle_draw_indexed(self._id, index_count, instance_count)

    def finish(self) -> RenderBundle:
        bundle_id = _gpu().bundle_finish(self._id)
        return RenderBundle(bundle_id)


class ComputePassEncoder:
    """Encodes compute dispatches inside a compute pass."""

    def __init__(self, encoder: CommandEncoder, id_: int) -> None:
        self._encoder = encoder
        self._id = id_
        self._ended = False

    @property
    def id(self) -> int:
        return self._id

    def _require_open(self) -> None:
        if self._ended:
            raise GPUValidationError("compute pass has already ended")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if not self._ended:
            self.end()

    def set_pipeline(self, pipeline: ComputePipeline) -> None:
        self._require_open()
        _gpu().compute_set_pipeline(self._id, pipeline.id)

    def set_bind_group(self, group_index: int, group: BindGroup) -> None:
        self._require_open()
        _gpu().compute_set_bind_group(self._id, group_index, group.id)

    def dispatch_workgroups(self, x: int, y: int = 1, z: int = 1) -> None:
        self._require_open()
        _gpu().dispatch_workgroups(self._id, x, y, z)

    def dispatch_workgroups_indirect(self, buffer: Buffer, offset: int = 0) -> None:
        self._require_open()
        _gpu().dispatch_workgroups_indirect(self._id, buffer.id, offset)

    def write_timestamp(self, query_set: QuerySet, query_index: int) -> None:
        self._require_open()
        _gpu().write_compute_timestamp(self._id, query_set.id, query_index)

    def end(self) -> None:
        self._require_open()
        _gpu().end_compute_pass(self._id)
        self._ended = True


class CommandEncoder(GPUObject):
    """Encodes a command/graphics stream; finished encoders are immutable."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)
        self._finished = False

    def _require_open(self) -> None:
        self._require_alive()
        if self._finished:
            raise GPUValidationError("command encoder already finished")

    def begin_render_pass(
        self,
        color_attachments: list[RenderPassColorAttachment],
        depth_stencil_attachment: RenderPassDepthStencilAttachment | None = None,
    ) -> RenderPassEncoder:
        self._require_open()
        pass_id = _gpu().begin_render_pass(
            self._id, list(color_attachments), depth_stencil_attachment
        )
        return RenderPassEncoder(self, pass_id)

    def begin_compute_pass(self, *, label: str | None = None) -> ComputePassEncoder:
        self._require_open()
        pass_id = _gpu().begin_compute_pass(self._id, label=label)
        return ComputePassEncoder(self, pass_id)

    def copy_texture_to_texture(
        self, src: TextureView, dst: TextureView, *, size: tuple[int, int, int]
    ) -> None:
        self._require_open()
        _gpu().copy_texture_to_texture(self._id, src.id, dst.id, size=size)

    def copy_texture_to_buffer(
        self, src: TextureView, dst: Buffer, *, size: tuple[int, int, int]
    ) -> None:
        self._require_open()
        _gpu().copy_texture_to_buffer(self._id, src.id, dst.id, size=size)

    def resolve_query_set(
        self,
        query_set: QuerySet,
        first_query: int,
        query_count: int,
        destination: Buffer,
        destination_offset: int,
    ) -> None:
        self._require_open()
        _gpu().resolve_query_set(
            self._id, query_set.id, first_query, query_count, destination.id, destination_offset
        )

    def finish(self) -> CommandBuffer:
        self._require_open()
        buffer_id = _gpu().finish(self._id)
        self._finished = True
        return CommandBuffer(buffer_id)

    def _delete_impl(self) -> None:
        _gpu().destroy_command_encoder(self._id)


__all__ = [
    "CommandBuffer",
    "CommandEncoder",
    "ComputePassDescriptor",
    "ComputePassEncoder",
    "RenderBundle",
    "RenderBundleDescriptor",
    "RenderBundleEncoder",
    "RenderPassColorAttachment",
    "RenderPassDepthStencilAttachment",
    "RenderPassEncoder",
]
