"""Surface (swapchain) render target for canvas-backed presentation.

A :class:`SurfaceTarget` is a backend-unaware render attachment: the canvas is
a plain duck contract (``get_physical_size()`` and ``get_wgpu_context()``), so
:mod:`easygpu` holds zero rendercanvas concepts. ``FakeGPU`` records the
attachment untouched; :class:`~easygpu.wgpu_backend.WgpuPyGPU` resolves it by
acquiring the current surface texture view each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from easygpu.constants import TextureFormat


@dataclass(frozen=True)
class SurfaceTarget:
    """A render attachment backed by a canvas surface texture.

    ``canvas`` must provide ``get_physical_size() -> tuple[int, int]`` and
    ``get_wgpu_context()`` whose ``get_current_texture()`` result exposes
    ``create_view()``. Configuration belongs to the caller (the app owning the
    canvas); this type only carries format metadata and acquires the view.
    """

    canvas: Any
    format: TextureFormat | int = TextureFormat.RGBA8_UNORM
    label: str | None = None

    @property
    def width(self) -> int:
        """Current surface width in physical pixels (re-read on every access)."""
        width: int = self.canvas.get_physical_size()[0]
        return width

    @property
    def height(self) -> int:
        """Current surface height in physical pixels (re-read on every access)."""
        height: int = self.canvas.get_physical_size()[1]
        return height

    def acquire(self) -> object:
        """Return this frame's surface texture view (one acquire per frame)."""
        view: object = self.canvas.get_wgpu_context().get_current_texture().create_view()
        return view


__all__ = ["SurfaceTarget"]
