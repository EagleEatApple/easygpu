"""Render pipeline wrapper for :mod:`easygpu`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.bindgroup import BindGroupLayout, PipelineLayout
from easygpu.constants import (
    BlendFactor,
    BlendOperation,
    CompareFunction,
    CullMode,
    FrontFace,
    PrimitiveTopology,
    TextureFormat,
    VertexFormat,
)
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
    constants: dict[str, float] | None = None


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
class BlendComponent:
    """WebGPU ``GPUBlendComponent``."""

    operation: BlendOperation = BlendOperation.ADD
    src_factor: BlendFactor = BlendFactor.ONE
    dst_factor: BlendFactor = BlendFactor.ZERO


@dataclass(frozen=True)
class BlendState:
    """WebGPU ``GPUBlendState``."""

    color: BlendComponent = BlendComponent()
    alpha: BlendComponent = BlendComponent()


@dataclass(frozen=True)
class ColorTargetState:
    """WebGPU ``GPUColorTargetState``."""

    format: TextureFormat | int
    blend: BlendState | None = None


@dataclass(frozen=True)
class DepthStencilState:
    """WebGPU ``GPUDepthStencilState`` (depth subset)."""

    format: TextureFormat | int
    depth_write_enabled: bool = True
    depth_compare: CompareFunction = CompareFunction.LESS
    depth_bias: int = 0
    depth_bias_slope_scale: float = 0.0
    depth_bias_clamp: float = 0.0


@dataclass(frozen=True)
class MultisampleState:
    """WebGPU ``GPUMultisampleState``."""

    count: int = 1
    mask: int = 0xFFFFFFFF
    alpha_to_coverage_enabled: bool = False


@dataclass(frozen=True)
class RenderPipelineDescriptor:
    """WebGPU ``GPURenderPipelineDescriptor`` (graphics subset)."""

    vertex_stage: ShaderStageEntry
    fragment_stage: ShaderStageEntry | None = None
    primitive_topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST
    vertex_buffers: list[VertexBufferLayout] | None = None
    label: str | None = None
    layout: PipelineLayout | None = None
    cull_mode: CullMode = CullMode.NONE
    front_face: FrontFace = FrontFace.CCW
    depth_stencil: DepthStencilState | None = None
    fragment_targets: list[ColorTargetState] | None = None
    multisample: MultisampleState | None = None


@dataclass(frozen=True)
class ComputePipelineDescriptor:
    """WebGPU ``GPUComputePipelineDescriptor``."""

    compute: ShaderStageEntry
    layout: PipelineLayout | None = None
    label: str | None = None


class ComputePipeline(GPUObject):
    """A compute pipeline (one compute shader stage)."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)

    def get_bind_group_layout(self, group_index: int) -> BindGroupLayout:
        """Return the bind group layout for ``group_index`` (works with 'auto' layouts)."""
        layout_id = _gpu().get_bind_group_layout(self.id, group_index)
        return BindGroupLayout(layout_id)

    def _delete_impl(self) -> None:
        _gpu().destroy_compute_pipeline(self._id)


class RenderPipeline(GPUObject):
    """A render pipeline (vertex + fragment stages, topology and state)."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)

    def get_bind_group_layout(self, group_index: int) -> BindGroupLayout:
        """Return the bind group layout for ``group_index`` (works with 'auto' layouts)."""
        layout_id = _gpu().get_bind_group_layout(self.id, group_index)
        return BindGroupLayout(layout_id)

    def _delete_impl(self) -> None:
        _gpu().destroy_render_pipeline(self._id)


__all__ = [
    "BlendComponent",
    "BlendState",
    "ColorTargetState",
    "ComputePipeline",
    "ComputePipelineDescriptor",
    "DepthStencilState",
    "MultisampleState",
    "RenderPipeline",
    "RenderPipelineDescriptor",
    "ShaderStageEntry",
    "VertexAttribute",
    "VertexBufferLayout",
]
