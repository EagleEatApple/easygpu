"""Sampler / SamplerDescriptor module tests."""

from __future__ import annotations

from easygpu.constants import CompareFunction


def test_sampler_module_path() -> None:
    from easygpu import sampler

    assert hasattr(sampler, "Sampler")
    assert hasattr(sampler, "SamplerDescriptor")


def test_sampler_descriptor_compare() -> None:
    from easygpu.sampler import SamplerDescriptor

    assert SamplerDescriptor(compare=CompareFunction.LESS).compare == CompareFunction.LESS
