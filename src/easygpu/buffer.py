"""Buffer wrapper for :mod:`easygpu`."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.constants import BufferUsage, MapMode

if TYPE_CHECKING:
    from easygpu.device import Device
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class BufferDescriptor:
    """WebGPU ``GPUBufferDescriptor``."""

    size: int
    usage: BufferUsage | int
    label: str | None = None
    mapped_at_creation: bool = False


def _coerce_bytes(data: bytes | bytearray | memoryview | Sequence[float]) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    try:
        values = tuple(float(value) for value in data)
    except (TypeError, ValueError) as exc:
        raise TypeError("data must be bytes-like or a sequence of float values") from exc
    return struct.pack(f"<{len(values)}f", *values)


class Buffer(GPUObject):
    """A GPU vertex/transform buffer."""

    def __init__(
        self,
        id_: int,
        *,
        size: int,
        usage: BufferUsage | int,
        label: str | None = None,
    ) -> None:
        if size < 0:
            raise ValueError(f"buffer size must be >= 0, got {size}")
        super().__init__(id_, label=label)
        self._size = size
        self._usage = usage

    @property
    def size(self) -> int:
        return self._size

    @property
    def usage(self) -> BufferUsage | int:
        return self._usage

    def map_async(self, mode: MapMode, *, offset: int = 0, size: int | None = None) -> None:
        _gpu().map_async(self.id, mode, offset, self.size - offset if size is None else size)

    def get_mapped_range(self, offset: int = 0, size: int | None = None) -> bytes:
        end = self.size if size is None else offset + size
        return _gpu().get_mapped_range(self.id, offset, end - offset)

    def unmap(self) -> None:
        _gpu().unmap(self.id)

    def _delete_impl(self) -> None:
        _gpu().destroy_buffer(self._id)

    @classmethod
    def from_data(
        cls,
        device: Device,
        data: bytes | bytearray | memoryview | Sequence[float],
        *,
        usage: BufferUsage | int = BufferUsage.VERTEX | BufferUsage.COPY_DST,
        label: str | None = None,
    ) -> Buffer:
        raw = _coerce_bytes(data)
        descriptor = BufferDescriptor(size=len(raw), usage=usage, label=label)
        buffer = device.create_buffer(descriptor)
        device.queue.write_buffer(buffer, 0, raw)
        return buffer


__all__ = ["Buffer", "BufferDescriptor"]
