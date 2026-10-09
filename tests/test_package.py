"""Package-level sanity tests for :mod:`easygpu`."""

from __future__ import annotations

import warnings

import pytest

import easygpu

EXPECTED_EXPORTS = (
    "Buffer",
    "BufferDescriptor",
    "BufferUsage",
    "CommandBuffer",
    "CommandEncoder",
    "Device",
    "EasyGPUError",
    "GPU",
    "GPUObjectDeletedError",
    "GPUValidationError",
    "LoadOp",
    "OffscreenTarget",
    "PrimitiveTopology",
    "Queue",
    "RenderPassColorAttachment",
    "RenderPassEncoder",
    "RenderPipeline",
    "RenderPipelineDescriptor",
    "ShaderModule",
    "ShaderModuleDescriptor",
    "ShaderStage",
    "ShaderStageEntry",
    "StoreOp",
    "SurfaceTarget",
    "TextureFormat",
    "VertexAttribute",
    "VertexBufferLayout",
    "VertexFormat",
    "configure",
    "get_gpu",
    "offscreen_target",
    "readback_rgba8",
    "request_adapter",
)


def test_package_imports() -> None:
    assert easygpu.__version__ is not None


def test_version_is_020() -> None:
    assert easygpu.__version__ == "0.2.0"


def test_core_exports_present() -> None:
    for name in EXPECTED_EXPORTS:
        assert hasattr(easygpu, name), f"missing package export: {name}"


def test_fake_gpu_not_exported() -> None:
    assert not hasattr(easygpu, "FakeGPU")


def test_import_warns_only_when_wgpu_missing(monkeypatch) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(easygpu)
    assert any("0.2.0" in str(item.message) for item in caught)


def test_import_does_not_warn_when_wgpu_installed() -> None:
    import importlib.util

    if importlib.util.find_spec("wgpu") is None:
        pytest.skip("wgpu not installed in this environment")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(easygpu)
    assert not any("0.2.0" in str(item.message) for item in caught)
