"""Tables in :mod:`easygpu.wgpu_translate` are pure and GPU-free — a missed
enum member surfaces as an exhaustive-table failure, not a runtime surprise."""

from __future__ import annotations

from typing import Any

import pytest

from easygpu import constants as c
from easygpu import wgpu_translate as t
from easygpu.bindgroup import (
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BufferSlice,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
from easygpu.errors import GPUValidationError
from easygpu.pipeline import (
    BlendComponent,
    BlendState,
    ColorTargetState,
    ComputePipelineDescriptor,
    DepthStencilState,
    MultisampleState,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.sampler import Sampler, SamplerDescriptor
from easygpu.shader import ShaderModule, ShaderModuleDescriptor
from easygpu.texture import (
    Texture,
    TextureDescriptor,
    TextureView,
    TextureViewDescriptor,
)

MEMBERS = {
    c.TextureFormat: {
        c.TextureFormat.BGRA8_UNORM: "bgra8unorm",
        c.TextureFormat.RGBA8_UNORM: "rgba8unorm",
        c.TextureFormat.DEPTH24PLUS: "depth24plus",
        c.TextureFormat.DEPTH32FLOAT: "depth32float",
        c.TextureFormat.DEPTH24PLUS_STENCIL8: "depth24plus-stencil8",
        c.TextureFormat.DEPTH32FLOAT_STENCIL8: "depth32float-stencil8",
    },
    c.PrimitiveTopology: {
        c.PrimitiveTopology.POINT_LIST: "point-list",
        c.PrimitiveTopology.LINE_LIST: "line-list",
        c.PrimitiveTopology.LINE_STRIP: "line-strip",
        c.PrimitiveTopology.TRIANGLE_LIST: "triangle-list",
        c.PrimitiveTopology.TRIANGLE_STRIP: "triangle-strip",
    },
    c.VertexFormat: {
        c.VertexFormat.FLOAT16X2: "float16x2",
        c.VertexFormat.FLOAT16X4: "float16x4",
        c.VertexFormat.FLOAT32: "float32",
        c.VertexFormat.FLOAT32X2: "float32x2",
        c.VertexFormat.FLOAT32X3: "float32x3",
        c.VertexFormat.FLOAT32X4: "float32x4",
        c.VertexFormat.SINT32: "sint32",
        c.VertexFormat.UINT32: "uint32",
    },
    c.IndexFormat: {
        c.IndexFormat.UINT16: "uint16",
        c.IndexFormat.UINT32: "uint32",
    },
    c.LoadOp: {
        c.LoadOp.CLEAR: "clear",
        c.LoadOp.LOAD: "load",
    },
    c.StoreOp: {
        c.StoreOp.STORE: "store",
        c.StoreOp.DISCARD: "discard",
    },
    c.CompareFunction: {
        c.CompareFunction.NEVER: "never",
        c.CompareFunction.LESS: "less",
        c.CompareFunction.EQUAL: "equal",
        c.CompareFunction.LESS_EQUAL: "less-equal",
        c.CompareFunction.GREATER: "greater",
        c.CompareFunction.NOT_EQUAL: "not-equal",
        c.CompareFunction.GREATER_EQUAL: "greater-equal",
        c.CompareFunction.ALWAYS: "always",
    },
    c.FilterMode: {
        c.FilterMode.NEAREST: "nearest",
        c.FilterMode.LINEAR: "linear",
    },
    c.MipmapFilterMode: {
        c.MipmapFilterMode.NEAREST: "nearest",
        c.MipmapFilterMode.LINEAR: "linear",
    },
    c.AddressMode: {
        c.AddressMode.CLAMP_TO_EDGE: "clamp-to-edge",
        c.AddressMode.REPEAT: "repeat",
        c.AddressMode.MIRROR_REPEAT: "mirror-repeat",
    },
    c.BlendOperation: {
        c.BlendOperation.ADD: "add",
        c.BlendOperation.SUBTRACT: "subtract",
        c.BlendOperation.REVERSE_SUBTRACT: "reverse-subtract",
        c.BlendOperation.MIN: "min",
        c.BlendOperation.MAX: "max",
    },
    c.BlendFactor: {
        c.BlendFactor.ZERO: "zero",
        c.BlendFactor.ONE: "one",
        c.BlendFactor.SRC: "src",
        c.BlendFactor.ONE_MINUS_SRC: "one-minus-src",
        c.BlendFactor.SRC_ALPHA: "src-alpha",
        c.BlendFactor.ONE_MINUS_SRC_ALPHA: "one-minus-src-alpha",
        c.BlendFactor.DST: "dst",
        c.BlendFactor.ONE_MINUS_DST: "one-minus-dst",
        c.BlendFactor.DST_ALPHA: "dst-alpha",
        c.BlendFactor.ONE_MINUS_DST_ALPHA: "one-minus-dst-alpha",
        c.BlendFactor.SRC_ALPHA_SATURATED: "src-alpha-saturated",
        c.BlendFactor.CONSTANT: "constant",
        c.BlendFactor.ONE_MINUS_CONSTANT: "one-minus-constant",
    },
    c.CullMode: {
        c.CullMode.NONE: "none",
        c.CullMode.FRONT: "front",
        c.CullMode.BACK: "back",
    },
    c.FrontFace: {
        c.FrontFace.CCW: "ccw",
        c.FrontFace.CW: "cw",
    },
    c.TextureSampleType: {
        c.TextureSampleType.FLOAT: "float",
        c.TextureSampleType.UNFILTERABLE_FLOAT: "unfilterable-float",
        c.TextureSampleType.DEPTH: "depth",
    },
    c.BufferBindingType: {
        c.BufferBindingType.UNIFORM: "uniform",
        c.BufferBindingType.STORAGE: "storage",
        c.BufferBindingType.READ_ONLY_STORAGE: "read-only-storage",
    },
    c.SamplerBindingType: {
        c.SamplerBindingType.FILTERING: "filtering",
        c.SamplerBindingType.NON_FILTERING: "non-filtering",
        c.SamplerBindingType.COMPARISON: "comparison",
    },
    c.QueryType: {
        c.QueryType.OCCLUSION: "occlusion",
        c.QueryType.TIMESTAMP: "timestamp",
    },
}


@pytest.mark.parametrize("enum_cls,expected", MEMBERS.items())
def test_string_enum_table_complete(enum_cls: Any, expected: dict[int, str]) -> None:
    """Every member of a string enum is translated, to exactly the WebGPU string."""
    got = t.enum_names(enum_cls)
    assert set(got) == set(enum_cls)
    assert got == {int(member): name for member, name in expected.items()}


def test_flag_types_are_pass_through() -> None:
    """IntFlag types are not string-translated; they pass through as ints."""
    assert t.PASSTHROUGH_FLAGS == {
        c.ShaderStage,
        c.BufferUsage,
        c.TextureUsage,
        c.MapMode,
    }


def test_texture_format_str_member_and_int() -> None:
    assert t.texture_format_str(c.TextureFormat.RGBA8_UNORM) == "rgba8unorm"
    assert (
        t.texture_format_str(int(c.TextureFormat.DEPTH32FLOAT_STENCIL8)) == "depth32float-stencil8"
    )


def test_name_raises_on_unknown_int() -> None:
    with pytest.raises(GPUValidationError, match="TextureFormat"):
        t._name(c.TextureFormat, 9999)


def test_name_raises_on_unknown_class() -> None:
    with pytest.raises(GPUValidationError, match="is not a string-translated enum"):
        t._name(c.BufferUsage, int(c.BufferUsage.VERTEX))


def test_buffer_descriptor_dict() -> None:
    d = BufferDescriptor(
        size=256, usage=int(c.BufferUsage.VERTEX | c.BufferUsage.COPY_DST), label="verts"
    )
    assert t.buffer_descriptor_dict(d) == {
        "label": "verts",
        "size": 256,
        "usage": 0x28,
        "mapped_at_creation": False,
    }


def test_buffer_descriptor_omits_none_label() -> None:
    d = BufferDescriptor(size=4, usage=c.BufferUsage.MAP_READ)
    out = t.buffer_descriptor_dict(d)
    assert "label" not in out
    assert out["usage"] == 1


def test_texture_descriptor_dict() -> None:
    d = TextureDescriptor(
        size=(4, 2, 1),
        format=c.TextureFormat.RGBA8_UNORM,
        usage=c.TextureUsage.RENDER_ATTACHMENT | c.TextureUsage.COPY_SRC,
        label="rt",
        sample_count=2,
    )
    assert t.texture_descriptor_dict(d) == {
        "label": "rt",
        "size": (4, 2, 1),
        "mip_level_count": 1,
        "sample_count": 2,
        "dimension": "2d",
        "format": "rgba8unorm",
        "usage": 17,
    }


def test_texture_view_descriptor_dict() -> None:
    d = TextureViewDescriptor(format=c.TextureFormat.RGBA8_UNORM, aspect="all", base_mip_level=0)
    assert t.texture_view_descriptor_dict(d) == {
        "format": "rgba8unorm",
        "aspect": "all",
        "base_mip_level": 0,
        "base_array_layer": 0,
    }


def test_texure_view_descriptor_dict_defaults_format_none() -> None:
    assert t.texture_view_descriptor_dict(TextureViewDescriptor()) == {
        "base_mip_level": 0,
        "base_array_layer": 0,
    }


def test_sampler_descriptor_dict_defaults() -> None:
    d = SamplerDescriptor()
    assert t.sampler_descriptor_dict(d) == {
        "address_mode_u": "clamp-to-edge",
        "address_mode_v": "clamp-to-edge",
        "address_mode_w": "clamp-to-edge",
        "mag_filter": "nearest",
        "min_filter": "nearest",
        "mipmap_filter": "nearest",
    }


def test_sampler_descriptor_dict_compare() -> None:
    d = SamplerDescriptor(
        address_mode_u=c.AddressMode.REPEAT,
        mag_filter=c.FilterMode.LINEAR,
        min_filter=c.FilterMode.LINEAR,
        mipmap_filter=c.MipmapFilterMode.LINEAR,
        compare=c.CompareFunction.ALWAYS,
        label="s",
    )
    out = t.sampler_descriptor_dict(d)
    assert out["compare"] == "always"
    assert out["address_mode_u"] == "repeat"
    assert out["mag_filter"] == "linear"
    assert out["label"] == "s"


def test_shader_module_dict() -> None:
    d = ShaderModuleDescriptor(code="fn main() {}", label="tri")
    assert t.shader_module_dict(d) == {"code": "fn main() {}", "label": "tri"}


def test_bind_group_layout_dict() -> None:
    d = BindGroupLayoutDescriptor(
        label="bgl",
        entries=[
            BindGroupLayoutEntry(
                binding=0, visibility=c.ShaderStage.VERTEX, buffer=c.BufferBindingType.UNIFORM
            ),
            BindGroupLayoutEntry(binding=1, visibility=c.ShaderStage.FRAGMENT, sampler=True),
            BindGroupLayoutEntry(
                binding=2,
                visibility=c.ShaderStage.FRAGMENT,
                sampler=c.SamplerBindingType.COMPARISON,
            ),
            BindGroupLayoutEntry(
                binding=3, visibility=c.ShaderStage.FRAGMENT, texture=c.TextureSampleType.FLOAT
            ),
        ],
    )
    assert t.bind_group_layout_dict(d) == {
        "label": "bgl",
        "entries": [
            {"binding": 0, "visibility": 1, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": 2, "sampler": {"type": "filtering"}},
            {"binding": 2, "visibility": 2, "sampler": {"type": "comparison"}},
            {
                "binding": 3,
                "visibility": 2,
                "texture": {"sample_type": "float", "view_dimension": "2d"},
            },
        ],
    }


def test_bind_group_layout_entry_view_dimension_passthrough() -> None:
    d = BindGroupLayoutDescriptor(
        entries=[
            BindGroupLayoutEntry(
                binding=0,
                visibility=c.ShaderStage.FRAGMENT,
                texture=c.TextureSampleType.FLOAT,
                view_dimension="cube",
            )
        ]
    )
    out = t.bind_group_layout_dict(d)
    assert out["entries"][0]["texture"] == {
        "sample_type": "float",
        "view_dimension": "cube",
    }


def test_bind_group_layout_entry_sampler_false_raises() -> None:
    d = BindGroupLayoutDescriptor(
        entries=[BindGroupLayoutEntry(binding=0, visibility=c.ShaderStage.FRAGMENT, sampler=False)]
    )
    with pytest.raises(GPUValidationError, match="sampler binding"):
        t.bind_group_layout_dict(d)


def _resolve(x: Any) -> str:
    return f"resolved:{getattr(x, 'id', x)}"


def test_bind_group_dict_binding_kinds() -> None:
    buf = Buffer(1, size=64, usage=c.BufferUsage.UNIFORM)
    sampler = Sampler(2)
    view = TextureView(
        3,
        Texture(
            4,
            size=(1, 1, 1),
            format=c.TextureFormat.RGBA8_UNORM,
            usage=c.TextureUsage.TEXTURE_BINDING,
        ),
    )
    d = BindGroupDescriptor(
        label="bg",
        layout=7,
        entries=[
            BindGroupEntry(binding=0, resource=buf),
            BindGroupEntry(binding=1, resource=BufferSlice(buf, offset=16)),
            BindGroupEntry(binding=2, resource=BufferSlice(buf, offset=16, size=32)),
            BindGroupEntry(binding=3, resource=sampler),
            BindGroupEntry(binding=4, resource=view),
        ],
    )
    assert t.bind_group_dict(d, _resolve) == {
        "label": "bg",
        "layout": "resolved:7",
        "entries": [
            {"binding": 0, "resource": "resolved:1"},
            {"binding": 1, "resource": {"buffer": "resolved:1", "offset": 16}},
            {"binding": 2, "resource": {"buffer": "resolved:1", "offset": 16, "size": 32}},
            {"binding": 3, "resource": "resolved:2"},
            {"binding": 4, "resource": "resolved:3"},
        ],
    }


def test_color_attachment_dict() -> None:
    a = RenderPassColorAttachment(view=11, clear_value=(0.1, 0.2, 0.3, 1.0), resolve_target=12)
    assert t.color_attachment_dict(a, _resolve) == {
        "view": "resolved:11",
        "clear_value": (0.1, 0.2, 0.3, 1.0),
        "load_op": "clear",
        "store_op": "store",
        "resolve_target": "resolved:12",
    }


def test_color_attachment_dict_default_resolve_target() -> None:
    a = RenderPassColorAttachment(view=11)
    out = t.color_attachment_dict(a, _resolve)
    assert out["view"] == "resolved:11"
    assert "resolve_target" not in out


def test_depth_stencil_attachment_dict() -> None:
    a = RenderPassDepthStencilAttachment(view=9)
    assert t.depth_stencil_attachment_dict(a, _resolve) == {
        "view": "resolved:9",
        "depth_clear_value": 1.0,
        "depth_load_op": "clear",
        "depth_store_op": "store",
        "depth_read_only": False,
        "stencil_clear_value": 0,
        "stencil_load_op": "clear",
        "stencil_store_op": "store",
        "stencil_read_only": False,
    }


def _module(idx: int) -> ShaderModule:
    return ShaderModule(idx)


def test_render_pipeline_dict_full() -> None:
    d = RenderPipelineDescriptor(
        label="pipe",
        layout=None,
        vertex_stage=ShaderStageEntry(module=_module(3), entry_point="vs", constants={"x": 2.0}),
        primitive_topology=c.PrimitiveTopology.TRIANGLE_LIST,
        cull_mode=c.CullMode.BACK,
        front_face=c.FrontFace.CW,
        vertex_buffers=[
            VertexBufferLayout(
                array_stride=24,
                step_mode="vertex",
                attributes=[
                    VertexAttribute(shader_location=0, format=c.VertexFormat.FLOAT32X3, offset=0)
                ],
            )
        ],
        fragment_stage=ShaderStageEntry(module=_module(4)),
        fragment_targets=[
            ColorTargetState(
                format=c.TextureFormat.BGRA8_UNORM,
                blend=BlendState(
                    color=BlendComponent(
                        operation=c.BlendOperation.ADD,
                        src_factor=c.BlendFactor.ONE,
                        dst_factor=c.BlendFactor.ONE_MINUS_SRC_ALPHA,
                    ),
                    alpha=BlendComponent(operation=c.BlendOperation.REVERSE_SUBTRACT),
                ),
            )
        ],
        depth_stencil=DepthStencilState(
            format=c.TextureFormat.DEPTH24PLUS,
            depth_write_enabled=True,
            depth_compare=c.CompareFunction.LESS_EQUAL,
        ),
        multisample=MultisampleState(count=4, alpha_to_coverage_enabled=True),
    )
    assert t.render_pipeline_dict(d, _resolve) == {
        "label": "pipe",
        "layout": "auto",
        "vertex": {
            "module": "resolved:3",
            "entry_point": "vs",
            "constants": {"x": 2.0},
            "buffers": [
                {
                    "array_stride": 24,
                    "step_mode": "vertex",
                    "attributes": [{"format": "float32x3", "offset": 0, "shader_location": 0}],
                }
            ],
        },
        "primitive": {"topology": "triangle-list", "cull_mode": "back", "front_face": "cw"},
        "depth_stencil": {
            "format": "depth24plus",
            "depth_write_enabled": True,
            "depth_compare": "less-equal",
            "depth_bias": 0,
            "depth_bias_slope_scale": 0.0,
            "depth_bias_clamp": 0.0,
        },
        "multisample": {"count": 4, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": True},
        "fragment": {
            "module": "resolved:4",
            "entry_point": "main",
            "targets": [
                {
                    "format": "bgra8unorm",
                    "blend": {
                        "color": {
                            "operation": "add",
                            "src_factor": "one",
                            "dst_factor": "one-minus-src-alpha",
                        },
                        "alpha": {
                            "operation": "reverse-subtract",
                            "src_factor": "one",
                            "dst_factor": "zero",
                        },
                    },
                    "write_mask": 0xF,
                }
            ],
        },
    }


def test_render_pipeline_dict_minimal() -> None:
    d = RenderPipelineDescriptor(vertex_stage=ShaderStageEntry(module=_module(5)))
    out = t.render_pipeline_dict(d, _resolve)
    assert out["layout"] == "auto"
    assert out["vertex"] == {"module": "resolved:5", "entry_point": "main", "buffers": []}
    assert out["primitive"] == {
        "topology": "triangle-list",
        "cull_mode": "none",
        "front_face": "ccw",
    }
    assert "fragment" not in out
    assert "depth_stencil" not in out
    assert "multisample" not in out
    assert "constants" not in out["vertex"]


def test_color_target_write_mask_passthrough() -> None:
    d = RenderPipelineDescriptor(
        vertex_stage=ShaderStageEntry(module=_module(5)),
        fragment_stage=ShaderStageEntry(module=_module(6)),
        fragment_targets=[ColorTargetState(format=c.TextureFormat.RGBA8_UNORM, write_mask=0x5)],
    )
    out = t.render_pipeline_dict(d, _resolve)
    assert out["fragment"]["targets"][0]["write_mask"] == 0x5


def test_compute_pipeline_dict() -> None:
    d = ComputePipelineDescriptor(
        compute=ShaderStageEntry(module=_module(8), entry_point="main"), layout=None, label="cp"
    )
    assert t.compute_pipeline_dict(d, _resolve) == {
        "label": "cp",
        "layout": "auto",
        "compute": {"module": "resolved:8", "entry_point": "main"},
    }


def test_pipeline_layout_resolved() -> None:
    d = RenderPipelineDescriptor(vertex_stage=ShaderStageEntry(module=_module(5)))
    d2 = ComputePipelineDescriptor(compute=ShaderStageEntry(module=_module(5)))
    assert t.compute_pipeline_dict(d2, _resolve)["layout"] == "auto"
    assert t.render_pipeline_dict(d, _resolve)["layout"] == "auto"


def test_pipeline_layout_dict() -> None:
    d = PipelineLayoutDescriptor(bind_group_layouts=[], label="pl")
    assert t.pipeline_layout_dict(d, _resolve) == {"label": "pl", "bind_group_layouts": []}
