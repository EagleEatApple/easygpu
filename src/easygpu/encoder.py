"""Command encoder / render pass wrappers for :mod:`easygpu`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from easygpu.base import GPUObject
from easygpu.buffer import Buffer
from easygpu.constants import LoadOp, StoreOp
from easygpu.errors import GPUValidationError
from easygpu.pipeline import RenderPipeline

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class RenderPassColorAttachment:
    """WebGPU ``GPURenderPassColorAttachment`` (0.0.0 subset)."""

    view: int
    clear_value: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    load_op: LoadOp = LoadOp.CLEAR
    store_op: StoreOp = StoreOp.STORE


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

    def set_vertex_buffer(self, slot: int, buffer: Buffer) -> None:
        self._require_open()
        _gpu().set_vertex_buffer(self._id, slot, buffer.id)

    def draw(self, vertex_count: int, instance_count: int = 1) -> None:
        self._require_open()
        _gpu().draw(self._id, vertex_count, instance_count)

    def end(self) -> None:
        self._require_open()
        _gpu().end(self._id)
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
        self, color_attachments: list[RenderPassColorAttachment]
    ) -> RenderPassEncoder:
        self._require_open()
        pass_id = _gpu().begin_render_pass(self._id, list(color_attachments))
        return RenderPassEncoder(self, pass_id)

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
    "RenderPassColorAttachment",
    "RenderPassEncoder",
]
