"""Package-level sanity tests for :mod:`easygpu`."""

from __future__ import annotations

import importlib
import warnings

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
    "TextureFormat",
    "VertexAttribute",
    "VertexBufferLayout",
    "VertexFormat",
    "configure",
    "get_gpu",
    "request_adapter",
)


def test_package_imports() -> None:
    assert easygpu.__version__ is not None


def test_version_is_010() -> None:
    assert easygpu.__version__ == "0.1.0"


def test_core_exports_present() -> None:
    for name in EXPECTED_EXPORTS:
        assert hasattr(easygpu, name), f"missing package export: {name}"


def test_fake_gpu_not_exported() -> None:
    assert not hasattr(easygpu, "FakeGPU")


def test_import_warns_about_api_preview() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(easygpu)
    assert any("0.1.0" in str(item.message) for item in caught)
