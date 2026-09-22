"""Query set resources for :mod:`easygpu`.

A *query set* holds the results WebGPU writes when the GPU records
occlusion or timestamp queries during a render or compute pass; the host
later resolves them into a buffer via an encoder command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easygpu.base import GPUObject
from easygpu.constants import QueryType

if TYPE_CHECKING:
    from easygpu.gpu import GPU


def _gpu() -> GPU:
    """Lazily reach the configured GPU without holding a module-cycle."""
    from easygpu.gpu import get_gpu

    return get_gpu()


@dataclass(frozen=True)
class QuerySetDescriptor:
    """WebGPU ``GPUQuerySetDescriptor``."""

    type: QueryType
    count: int
    label: str | None = None


class QuerySet(GPUObject):
    """A set of occlusion or timestamp query slots."""

    def _delete_impl(self) -> None:
        _gpu().destroy_query_set(self._id)


__all__ = ["QuerySet", "QuerySetDescriptor"]
