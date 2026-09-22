"""Top-level export surface tests for :mod:`easygpu`."""

from __future__ import annotations

import pytest

import easygpu
import easygpu.constants as constants_module
import easygpu.encoder as encoder_module

NEW_EXPORTS = (
    ("ComputePassEncoder", encoder_module),
    ("MapMode", constants_module),
    ("QueryType", constants_module),
    ("RenderBundle", encoder_module),
    ("RenderBundleEncoder", encoder_module),
    ("SamplerBindingType", constants_module),
)

ALREADY_EXPORTED = (
    "QuerySet",
    "QuerySetDescriptor",
    "ComputePipeline",
    "ComputePipelineDescriptor",
    "TextureViewDescriptor",
    "Sampler",
    "SamplerDescriptor",
    "RenderBundleDescriptor",
    "ComputePassDescriptor",
)


@pytest.mark.parametrize(("name", "module"), NEW_EXPORTS)
def test_new_export_present_and_aliased(name: str, module: object) -> None:
    assert hasattr(easygpu, name), f"missing package export: {name}"
    assert getattr(easygpu, name) is getattr(module, name), f"mis-aliased export: {name}"


@pytest.mark.parametrize("name", ALREADY_EXPORTED)
def test_existing_names_still_importable(name: str) -> None:
    assert hasattr(easygpu, name), f"missing package export: {name}"
