"""Base class shared by every GPU-backed object in :mod:`easygpu`."""

from __future__ import annotations

from easygpu.errors import GPUObjectDeletedError


class GPUObject:
    """A GPU resource that owns a WebGPU handle.

    Subclasses receive their handle from a ``device.create_*`` call and
    implement :meth:`_delete_impl`. Calling :meth:`delete` twice, or using a
    deleted object, is a no-op / guarded error respectively.
    """

    _id: int
    _deleted: bool
    _label: str | None

    def __init__(self, id_: int, *, label: str | None = None) -> None:
        self._id = id_
        self._deleted = False
        self._label = label

    @property
    def id(self) -> int:
        """The raw WebGPU handle of this object."""
        self._require_alive()
        return self._id

    @property
    def label(self) -> str | None:
        return self._label

    @property
    def deleted(self) -> bool:
        return self._deleted

    def _require_alive(self) -> None:
        if self._deleted:
            raise GPUObjectDeletedError(f"{type(self).__name__} has been deleted")

    def delete(self) -> None:
        """Release the GPU resource. Safe to call multiple times."""
        if self._deleted:
            return
        self._require_alive()
        self._delete_impl()
        self._deleted = True

    def _delete_impl(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _delete_impl")


__all__ = ["GPUObject"]
