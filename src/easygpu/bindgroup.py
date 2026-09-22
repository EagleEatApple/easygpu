"""Bind group resources for :mod:`easygpu`.

WebGPU gathers the resources a shader reads each draw into a *bind group*,
using a matching *bind group layout*. The layout describes the resource kinds
per binding; the group supplies the actual buffer/sampler/texture objects.
A pipeline can either declare an explicit layout or use ``layout='auto'`` and
ask :meth:`easygpu.pipeline.RenderPipeline.get_bind_group_layout` for the one
the driver synthesized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.buffer import Buffer
from easygpu.constants import (
    BufferBindingType,
    SamplerBindingType,
    ShaderStage,
    TextureSampleType,
)
from easygpu.sampler import Sampler
from easygpu.texture import TextureView

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class BindGroupLayoutEntry:
    """WebGPU ``GPUBindGroupLayoutEntry`` (``sampler=True`` == ``SamplerBindingType.FILTERING``)."""

    binding: int
    visibility: ShaderStage
    buffer: BufferBindingType | None = None
    sampler: bool | SamplerBindingType | None = None
    texture: TextureSampleType | None = None


@dataclass(frozen=True)
class BindGroupLayoutDescriptor:
    """WebGPU ``GPUBindGroupLayoutDescriptor``."""

    entries: list[BindGroupLayoutEntry]
    label: str | None = None


@dataclass(frozen=True)
class PipelineLayoutDescriptor:
    """WebGPU ``GPUPipelineLayoutDescriptor``."""

    bind_group_layouts: list[BindGroupLayout]
    label: str | None = None


@dataclass(frozen=True)
class BufferSlice:
    """A sub-range of :class:`Buffer` used as a uniform resource (WebGPU buffer binding)."""

    buffer: Buffer
    offset: int = 0
    size: int | None = None


@dataclass(frozen=True)
class BindGroupEntry:
    """WebGPU ``GPUBindGroupEntry``."""

    binding: int
    resource: Buffer | BufferSlice | Sampler | TextureView


@dataclass(frozen=True)
class BindGroupDescriptor:
    """WebGPU ``GPUBindGroupDescriptor``."""

    layout: BindGroupLayout | int
    entries: list[BindGroupEntry]
    label: str | None = None


class BindGroupLayout(GPUObject):
    """A layout describing the resource kinds a bind group must supply."""

    def _delete_impl(self) -> None:
        _gpu().destroy_bind_group_layout(self._id)


class PipelineLayout(GPUObject):
    """A full pipeline layout: an ordered list of bind group layouts."""

    def _delete_impl(self) -> None:
        _gpu().destroy_pipeline_layout(self._id)


class BindGroup(GPUObject):
    """A concrete set of resources for one bind group slot."""

    def _delete_impl(self) -> None:
        _gpu().destroy_bind_group(self._id)


__all__ = [
    "BindGroup",
    "BindGroupDescriptor",
    "BindGroupEntry",
    "BindGroupLayout",
    "BindGroupLayoutDescriptor",
    "BindGroupLayoutEntry",
    "BufferSlice",
    "PipelineLayout",
    "PipelineLayoutDescriptor",
]
