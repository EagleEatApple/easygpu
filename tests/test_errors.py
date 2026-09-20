"""Error hierarchy tests for easygpu."""

from __future__ import annotations

from easygpu.errors import EasyGPUError, GPUObjectDeletedError, GPUValidationError


def test_validation_is_easygpu_error() -> None:
    assert issubclass(GPUValidationError, EasyGPUError)


def test_deleted_is_easygpu_error() -> None:
    assert issubclass(GPUObjectDeletedError, EasyGPUError)


def test_message_roundtrips() -> None:
    assert str(GPUValidationError("boom")) == "boom"
