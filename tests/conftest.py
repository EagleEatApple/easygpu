"""Shared pytest fixtures for the easygpu test-suite."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator

import pytest

from easygpu import gpu as gpu_module
from easygpu.fake_gpu import FakeGPU
from easygpu.gpu import GPU, get_gpu

warnings.filterwarnings(
    "ignore",
    message="easygpu .*",
    category=UserWarning,
)


def _make_engine() -> GPU:
    """Return the engine flavor selected by ``EASYGPU_ENGINE`` (default FakeGPU)."""
    if os.environ.get("EASYGPU_ENGINE") == "wgpu":
        from easygpu.wgpu_backend import WgpuPyGPU

        return WgpuPyGPU()
    return FakeGPU()


@pytest.fixture
def fake_gpu() -> Iterator[GPU]:
    """A fresh engine wired up as the process-wide GPU implementation.

    Defaults to :class:`FakeGPU`; set ``EASYGPU_ENGINE=wgpu`` to run the same
    suite against :class:`~easygpu.wgpu_backend.WgpuPyGPU`.
    """
    engine = _make_engine()
    previous = gpu_module.get_gpu()
    gpu_module.configure(engine)
    try:
        yield engine
    finally:
        gpu_module.configure(previous)


@pytest.fixture
def wgpu_gpu() -> Iterator[GPU]:
    """A fresh WgpuPyGPU per test, skipped when no adapter/wgpu is available."""
    pytest.importorskip("wgpu")
    from easygpu.errors import GPUValidationError
    from easygpu.wgpu_backend import WgpuPyGPU

    engine = WgpuPyGPU()
    try:
        adapter = engine.request_adapter()
    except (ImportError, GPUValidationError):
        pytest.skip("no wgpu adapter available (install a driver or enable software rendering)")
    device = engine.request_device(adapter)
    previous = get_gpu()
    gpu_module.configure(engine)
    try:
        yield engine
    finally:
        gpu_module.configure(previous)
        engine.destroy_device(device)
