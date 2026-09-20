"""ShaderModule wrapper tests."""

from __future__ import annotations

from easygpu.device import Device
from easygpu.shader import ShaderModule, ShaderModuleDescriptor


def test_descriptor_holds_source() -> None:
    desc = ShaderModuleDescriptor(code="@vertex fn main() {}", label="tri_vs")
    assert "@vertex" in desc.code
    assert desc.label == "tri_vs"
    assert ShaderModuleDescriptor(code="x").label is None


def test_shader_module_exposes_id_and_label() -> None:
    mod = ShaderModule(id_=9, label="tri_vs")
    assert mod.id == 9
    assert mod.label == "tri_vs"


def test_shader_module_from_wgsl_concatenates_sources(fake_gpu) -> None:
    device = Device(id_=1)
    mod = ShaderModule.from_wgsl(device, "@vertex", "@fragment", label="tri")
    assert fake_gpu.shader_modules[mod.id] == "@vertex\n@fragment"
    assert mod.label == "tri"


def test_shader_module_delete_forwards_destroy(fake_gpu) -> None:
    mod = ShaderModule(id_=5)
    mod.delete()
    assert fake_gpu.calls_of("destroy_shader_module") == [(5,)]
