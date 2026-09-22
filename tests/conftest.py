"""Shared pytest fixtures for the easygpu test-suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from easygpu import gpu as gpu_module
from easygpu.fake_gpu import FakeGPU


@pytest.fixture
def fake_gpu() -> Iterator[FakeGPU]:
    """A fresh :class:`FakeGPU` wired up as the process-wide GPU implementation."""
    engine = FakeGPU()
    previous = gpu_module.get_gpu()
    gpu_module.configure(engine)
    try:
        yield engine
    finally:
        gpu_module.configure(previous)
