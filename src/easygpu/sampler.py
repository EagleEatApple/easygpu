"""Sampler / SamplerDescriptor live here (moved from easygpu.texture)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.constants import AddressMode, CompareFunction, FilterMode, MipmapFilterMode

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class SamplerDescriptor:
    """WebGPU ``GPUSamplerDescriptor``."""

    address_mode_u: AddressMode = AddressMode.CLAMP_TO_EDGE
    address_mode_v: AddressMode = AddressMode.CLAMP_TO_EDGE
    address_mode_w: AddressMode = AddressMode.CLAMP_TO_EDGE
    mag_filter: FilterMode = FilterMode.NEAREST
    min_filter: FilterMode = FilterMode.NEAREST
    mipmap_filter: MipmapFilterMode = MipmapFilterMode.NEAREST
    compare: CompareFunction | None = None
    label: str | None = None


class Sampler(GPUObject):
    """A texture sampler (filtering / wrapping / compare mode)."""

    def _delete_impl(self) -> None:
        _gpu().destroy_sampler(self._id)


__all__ = ["Sampler", "SamplerDescriptor"]
