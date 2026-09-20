"""Exceptions raised by :mod:`easygpu`."""

from __future__ import annotations


class EasyGPUError(Exception):
    """Base class for every error raised by the library."""


class GPUValidationError(EasyGPUError):
    """A WebGPU validation error (bad usage, already-finished encoder, ...)."""


class GPUObjectDeletedError(EasyGPUError):
    """Accessing a deleted GPU resource."""


__all__ = ["EasyGPUError", "GPUObjectDeletedError", "GPUValidationError"]
