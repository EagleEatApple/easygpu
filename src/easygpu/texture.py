"""Texture / TextureView wrappers for :mod:`easygpu`.

WebGPU textures are the key missing piece that the webgpu-samples port needed:
color attachments, depth buffers and sampled textures are all ``GPUTexture``
objects exposed to shaders through a :class:`TextureView`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.constants import TextureFormat, TextureUsage

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class TextureDescriptor:
    """WebGPU ``GPUTextureDescriptor`` (2d subset)."""

    size: tuple[int, int, int]
    format: TextureFormat | int
    usage: TextureUsage | int
    label: str | None = None
    mip_level_count: int = 1
    sample_count: int = 1
    dimension: str = "2d"


@dataclass(frozen=True)
class TextureViewDescriptor:
    """WebGPU ``GPUTextureViewDescriptor``."""

    format: TextureFormat | int | None = None
    dimension: str | None = None
    aspect: str | None = None
    base_mip_level: int = 0
    mip_level_count: int | None = None
    base_array_layer: int = 0
    array_layer_count: int | None = None


class TextureView(GPUObject):
    """A view of a :class:`Texture` (what slots and attachments bind)."""

    def __init__(self, id_: int, texture: Texture, *, label: str | None = None) -> None:
        super().__init__(id_, label=label)
        self._texture = texture

    @property
    def texture(self) -> Texture:
        return self._texture

    def _delete_impl(self) -> None:
        _gpu().destroy_texture_view(self._id)


class Texture(GPUObject):
    """A GPU texture (render target, depth buffer or sampled texture).

    Owns zero or more :class:`TextureView` objects created with
    :meth:`create_view`.
    """

    def __init__(
        self,
        id_: int,
        *,
        size: tuple[int, int, int],
        format: TextureFormat | int,
        usage: TextureUsage | int,
        sample_count: int = 1,
        mip_level_count: int = 1,
        label: str | None = None,
    ) -> None:
        super().__init__(id_, label=label)
        self._size = size
        self._format = format
        self._usage = usage
        self._sample_count = sample_count
        self._mip_level_count = mip_level_count

    @property
    def width(self) -> int:
        return self._size[0]

    @property
    def height(self) -> int:
        return self._size[1]

    @property
    def size(self) -> tuple[int, int, int]:
        return self._size

    @property
    def format(self) -> TextureFormat | int:
        return self._format

    @property
    def usage(self) -> TextureUsage | int:
        return self._usage

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def create_view(self, descriptor: TextureViewDescriptor | None = None) -> TextureView:
        view_id = _gpu().create_texture_view(self.id, descriptor)
        return TextureView(view_id, self)

    def generate_mipmap(self) -> None:
        _gpu().generate_mipmap(self.id)

    def _delete_impl(self) -> None:
        _gpu().destroy_texture(self._id)


__all__ = [
    "Texture",
    "TextureDescriptor",
    "TextureView",
    "TextureViewDescriptor",
]
