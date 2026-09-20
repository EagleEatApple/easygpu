"""Render pipeline wrapper for :mod:`easygpu`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.constants import PrimitiveTopology, VertexFormat
from easygpu.shader import ShaderModule

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class ShaderStageEntry:
    """A pipeline stage referencing a :class:`ShaderModule`."""

    module: ShaderModule | int
    entry_point: str = "main"


@dataclass(frozen=True)
class VertexAttribute:
    shader_location: int
    format: VertexFormat
    offset: int


@dataclass(frozen=True)
class VertexBufferLayout:
    array_stride: int
    attributes: list[VertexAttribute]
    step_mode: str = "vertex"


@dataclass(frozen=True)
class RenderPipelineDescriptor:
    """WebGPU ``GPURenderPipelineDescriptor`` (0.0.0 subset)."""

    vertex_stage: ShaderStageEntry
    fragment_stage: ShaderStageEntry | None = None
    primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST
    vertex_buffers: list[VertexBufferLayout] | None = None
    label: str | None = None


class RenderPipeline(GPUObject):
    """A render pipeline (vertex + fragment stages and topology)."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)

    def _delete_impl(self) -> None:
        _gpu().destroy_render_pipeline(self._id)


__all__ = [
    "RenderPipeline",
    "RenderPipelineDescriptor",
    "ShaderStageEntry",
    "VertexAttribute",
    "VertexBufferLayout",
]
