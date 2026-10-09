"""Offscreen render targets for headless GPU backends (0.2.0).

An :class:`OffscreenTarget` is a backend-unaware render attachment: the current
engine resolves it (``FakeGPU`` records the calls, :class:`WgpuPyGPU` lazily
creates an offscreen texture). :func:`readback_rgba8` pulls the rendered pixels
back to a tight RGBA byte string and needs a real backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from easygpu.constants import TextureFormat
from easygpu.errors import GPUValidationError
from easygpu.gpu import get_gpu


@dataclass(frozen=True)
class OffscreenTarget:
    """A render attachment backed by a lazily created offscreen texture."""

    width: int
    height: int
    format: TextureFormat | int = TextureFormat.RGBA8_UNORM
    label: str | None = None


def offscreen_target(
    width: int,
    height: int,
    *,
    format: TextureFormat | int = TextureFormat.RGBA8_UNORM,
    label: str | None = None,
) -> OffscreenTarget:
    """Construct an :class:`OffscreenTarget` resolved by the current engine."""
    return OffscreenTarget(width=width, height=height, format=format, label=label)


def readback_rgba8(target: OffscreenTarget) -> bytes:
    """Copy an offscreen target's pixels to a tight ``width * height * 4`` string.

    The copy size comes from the target itself, so callers only pass the
    target. Only a real backend has pixels: :class:`FakeGPU` raises
    :class:`GPUValidationError`.
    """
    from easygpu.fake_gpu import FakeGPU

    engine = get_gpu()
    if isinstance(engine, FakeGPU):
        raise GPUValidationError(
            "readback_rgba8 needs a real GPU backend (WgpuPyGPU); FakeGPU only records calls"
        )
    readback = getattr(engine, "readback_offscreen_rgba8", None)
    if readback is None:
        raise GPUValidationError(
            f"configured backend {type(engine).__name__} cannot read back pixels"
        )
    return bytes(readback(target, target.width, target.height))


__all__ = ["OffscreenTarget", "offscreen_target", "readback_rgba8"]
