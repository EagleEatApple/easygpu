"""Shader module wrapper for :mod:`easygpu`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject

if TYPE_CHECKING:
    from easygpu.device import Device
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class ShaderModuleDescriptor:
    """WebGPU ``GPUShaderModuleDescriptor``."""

    code: str
    label: str | None = None


class ShaderModule(GPUObject):
    """A compiled WGSL shader module."""

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)

    @classmethod
    def from_wgsl(
        cls,
        device: Device,
        *sources: str,
        label: str | None = None,
    ) -> ShaderModule:
        code = "\n".join(sources)
        return device.create_shader_module(ShaderModuleDescriptor(code=code, label=label))

    def _delete_impl(self) -> None:
        _gpu().destroy_shader_module(self._id)


__all__ = ["ShaderModule", "ShaderModuleDescriptor"]
