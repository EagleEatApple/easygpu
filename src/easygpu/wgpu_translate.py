"""Translate EasyGPU enums and descriptors into wgpu-py dict shapes.

Pure module: no ``wgpu`` import, fully unit-testable without a GPU (Task 3).
Flag types (``ShaderStage``, ``BufferUsage``, ``TextureUsage``, ``MapMode``)
pass through as their integer bit values — they match wgpu-py / WebGPU IDL
exactly (verified on wgpu 0.32.0, e.g. ``flags.MapMode READ=1, WRITE=2``).
Named enums translate through explicit, exhaustively-tested tables so a missed
or unknown member raises :class:`~easygpu.errors.GPUValidationError` instead of
being silently mapped to the wrong WebGPU string.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from easygpu import constants as c
from easygpu.bindgroup import BufferSlice
from easygpu.errors import GPUValidationError

if TYPE_CHECKING:
    from easygpu.bindgroup import (
        BindGroupDescriptor,
        BindGroupEntry,
        BindGroupLayoutDescriptor,
        BindGroupLayoutEntry,
        PipelineLayoutDescriptor,
    )
    from easygpu.buffer import BufferDescriptor
    from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
    from easygpu.pipeline import (
        ComputePipelineDescriptor,
        DepthStencilState,
        RenderPipelineDescriptor,
        ShaderStageEntry,
        VertexBufferLayout,
    )
    from easygpu.sampler import SamplerDescriptor
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.texture import TextureDescriptor, TextureViewDescriptor

_Resolver: TypeAlias = Callable[[Any], Any]

PASSTHROUGH_FLAGS: frozenset[type[enum.IntFlag]] = frozenset(
    {c.ShaderStage, c.BufferUsage, c.TextureUsage, c.MapMode}
)


def _table(rows: dict[Any, str]) -> dict[int, str]:
    return {int(key): value for key, value in rows.items()}


_STRING_TABLES: dict[type[enum.IntEnum], dict[int, str]] = {
    c.TextureFormat: _table(
        {
            c.TextureFormat.BGRA8_UNORM: "bgra8unorm",
            c.TextureFormat.RGBA8_UNORM: "rgba8unorm",
            c.TextureFormat.DEPTH24PLUS: "depth24plus",
            c.TextureFormat.DEPTH32FLOAT: "depth32float",
            c.TextureFormat.DEPTH24PLUS_STENCIL8: "depth24plus-stencil8",
            c.TextureFormat.DEPTH32FLOAT_STENCIL8: "depth32float-stencil8",
        }
    ),
    c.PrimitiveTopology: _table(
        {
            c.PrimitiveTopology.POINT_LIST: "point-list",
            c.PrimitiveTopology.LINE_LIST: "line-list",
            c.PrimitiveTopology.LINE_STRIP: "line-strip",
            c.PrimitiveTopology.TRIANGLE_LIST: "triangle-list",
            c.PrimitiveTopology.TRIANGLE_STRIP: "triangle-strip",
        }
    ),
    c.VertexFormat: _table(
        {
            c.VertexFormat.FLOAT16X2: "float16x2",
            c.VertexFormat.FLOAT16X4: "float16x4",
            c.VertexFormat.FLOAT32: "float32",
            c.VertexFormat.FLOAT32X2: "float32x2",
            c.VertexFormat.FLOAT32X3: "float32x3",
            c.VertexFormat.FLOAT32X4: "float32x4",
            c.VertexFormat.SINT32: "sint32",
            c.VertexFormat.UINT32: "uint32",
        }
    ),
    c.IndexFormat: _table(
        {
            c.IndexFormat.UINT16: "uint16",
            c.IndexFormat.UINT32: "uint32",
        }
    ),
    c.LoadOp: _table(
        {
            c.LoadOp.CLEAR: "clear",
            c.LoadOp.LOAD: "load",
        }
    ),
    c.StoreOp: _table(
        {
            c.StoreOp.STORE: "store",
            c.StoreOp.DISCARD: "discard",
        }
    ),
    c.CompareFunction: _table(
        {
            c.CompareFunction.NEVER: "never",
            c.CompareFunction.LESS: "less",
            c.CompareFunction.EQUAL: "equal",
            c.CompareFunction.LESS_EQUAL: "less-equal",
            c.CompareFunction.GREATER: "greater",
            c.CompareFunction.NOT_EQUAL: "not-equal",
            c.CompareFunction.GREATER_EQUAL: "greater-equal",
            c.CompareFunction.ALWAYS: "always",
        }
    ),
    c.FilterMode: _table(
        {
            c.FilterMode.NEAREST: "nearest",
            c.FilterMode.LINEAR: "linear",
        }
    ),
    c.MipmapFilterMode: _table(
        {
            c.MipmapFilterMode.NEAREST: "nearest",
            c.MipmapFilterMode.LINEAR: "linear",
        }
    ),
    c.AddressMode: _table(
        {
            c.AddressMode.CLAMP_TO_EDGE: "clamp-to-edge",
            c.AddressMode.REPEAT: "repeat",
            c.AddressMode.MIRROR_REPEAT: "mirror-repeat",
        }
    ),
    c.BlendOperation: _table(
        {
            c.BlendOperation.ADD: "add",
            c.BlendOperation.SUBTRACT: "subtract",
            c.BlendOperation.REVERSE_SUBTRACT: "reverse-subtract",
            c.BlendOperation.MIN: "min",
            c.BlendOperation.MAX: "max",
        }
    ),
    c.BlendFactor: _table(
        {
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
        }
    ),
    c.CullMode: _table(
        {
            c.CullMode.NONE: "none",
            c.CullMode.FRONT: "front",
            c.CullMode.BACK: "back",
        }
    ),
    c.FrontFace: _table(
        {
            c.FrontFace.CCW: "ccw",
            c.FrontFace.CW: "cw",
        }
    ),
    c.TextureSampleType: _table(
        {
            c.TextureSampleType.FLOAT: "float",
            c.TextureSampleType.UNFILTERABLE_FLOAT: "unfilterable-float",
            c.TextureSampleType.DEPTH: "depth",
        }
    ),
    c.BufferBindingType: _table(
        {
            c.BufferBindingType.UNIFORM: "uniform",
            c.BufferBindingType.STORAGE: "storage",
            c.BufferBindingType.READ_ONLY_STORAGE: "read-only-storage",
        }
    ),
    c.SamplerBindingType: _table(
        {
            c.SamplerBindingType.FILTERING: "filtering",
            c.SamplerBindingType.NON_FILTERING: "non-filtering",
            c.SamplerBindingType.COMPARISON: "comparison",
        }
    ),
    c.QueryType: _table(
        {
            c.QueryType.OCCLUSION: "occlusion",
            c.QueryType.TIMESTAMP: "timestamp",
        }
    ),
}


def enum_names(enum_cls: type[enum.IntEnum]) -> dict[int, str]:
    """Return a copy of the WebGPU string table for ``enum_cls``."""
    return dict(_STRING_TABLES[enum_cls])


def _name(enum_cls: type[enum.IntEnum], value: int) -> str:
    table = _STRING_TABLES.get(enum_cls)
    if table is None:
        raise GPUValidationError(
            f"{enum_cls.__name__} is not a string-translated enum (flags pass through as ints)"
        )
    name = table.get(int(value))
    if name is None:
        raise GPUValidationError(f"unknown {enum_cls.__name__} value {value!r}")
    return name


def texture_format_str(value: c.TextureFormat | int) -> str:
    """Convenience wrapper translating an EasyGPU texture format to WebGPU."""
    return _name(c.TextureFormat, int(value))


def _add_label(out: dict[str, Any], label: str | None) -> None:
    if label is not None:
        out["label"] = label


def buffer_descriptor_dict(descriptor: BufferDescriptor) -> dict[str, Any]:
    """Translate :class:`easygpu.buffer.BufferDescriptor` for ``create_buffer``."""
    out: dict[str, Any] = {
        "size": descriptor.size,
        "usage": int(descriptor.usage),
        "mapped_at_creation": descriptor.mapped_at_creation,
    }
    _add_label(out, descriptor.label)
    return out


def texture_descriptor_dict(descriptor: TextureDescriptor) -> dict[str, Any]:
    """Translate :class:`easygpu.texture.TextureDescriptor` for ``create_texture``."""
    out: dict[str, Any] = {
        "size": descriptor.size,
        "mip_level_count": descriptor.mip_level_count,
        "sample_count": descriptor.sample_count,
        "dimension": descriptor.dimension,
        "format": _name(c.TextureFormat, int(descriptor.format)),
        "usage": int(descriptor.usage),
    }
    _add_label(out, descriptor.label)
    return out


def texture_view_descriptor_dict(descriptor: TextureViewDescriptor) -> dict[str, Any]:
    """Translate :class:`easygpu.texture.TextureViewDescriptor` for ``create_view``."""
    out: dict[str, Any] = {
        "base_mip_level": descriptor.base_mip_level,
        "base_array_layer": descriptor.base_array_layer,
    }
    if descriptor.format is not None:
        out["format"] = _name(c.TextureFormat, int(descriptor.format))
    if descriptor.dimension is not None:
        out["dimension"] = descriptor.dimension
    if descriptor.aspect is not None:
        out["aspect"] = descriptor.aspect
    if descriptor.mip_level_count is not None:
        out["mip_level_count"] = descriptor.mip_level_count
    if descriptor.array_layer_count is not None:
        out["array_layer_count"] = descriptor.array_layer_count
    return out


def sampler_descriptor_dict(descriptor: SamplerDescriptor) -> dict[str, Any]:
    """Translate :class:`easygpu.sampler.SamplerDescriptor` for ``create_sampler``."""
    out: dict[str, Any] = {
        "address_mode_u": _name(c.AddressMode, int(descriptor.address_mode_u)),
        "address_mode_v": _name(c.AddressMode, int(descriptor.address_mode_v)),
        "address_mode_w": _name(c.AddressMode, int(descriptor.address_mode_w)),
        "mag_filter": _name(c.FilterMode, int(descriptor.mag_filter)),
        "min_filter": _name(c.FilterMode, int(descriptor.min_filter)),
        "mipmap_filter": _name(c.MipmapFilterMode, int(descriptor.mipmap_filter)),
    }
    if descriptor.compare is not None:
        out["compare"] = _name(c.CompareFunction, int(descriptor.compare))
    _add_label(out, descriptor.label)
    return out


def shader_module_dict(descriptor: ShaderModuleDescriptor) -> dict[str, Any]:
    """Translate :class:`easygpu.shader.ShaderModuleDescriptor` for ``create_shader_module``."""
    out: dict[str, Any] = {"code": descriptor.code}
    _add_label(out, descriptor.label)
    return out


def _bind_group_layout_entry_dict(entry: BindGroupLayoutEntry) -> dict[str, Any]:
    out: dict[str, Any] = {"binding": entry.binding, "visibility": int(entry.visibility)}
    if entry.buffer is not None:
        out["buffer"] = {"type": _name(c.BufferBindingType, int(entry.buffer))}
    if entry.sampler is not None:
        if entry.sampler is True:
            kind = c.SamplerBindingType.FILTERING
        elif isinstance(entry.sampler, c.SamplerBindingType):
            kind = entry.sampler
        else:
            raise GPUValidationError(
                f"sampler binding must be True or a SamplerBindingType, got {entry.sampler!r}"
            )
        out["sampler"] = {"type": _name(c.SamplerBindingType, int(kind))}
    if entry.texture is not None:
        out["texture"] = {
            "sample_type": _name(c.TextureSampleType, int(entry.texture)),
            "view_dimension": entry.view_dimension,
        }
    return out


def bind_group_layout_dict(descriptor: BindGroupLayoutDescriptor) -> dict[str, Any]:
    """Translate a bind group layout descriptor for ``create_bind_group_layout``."""
    out: dict[str, Any] = {
        "entries": [_bind_group_layout_entry_dict(entry) for entry in descriptor.entries]
    }
    _add_label(out, descriptor.label)
    return out


def _bind_group_entry_dict(entry: BindGroupEntry, resolve: _Resolver) -> dict[str, Any]:
    resource = entry.resource
    if isinstance(resource, BufferSlice):
        binding: dict[str, Any] = {"buffer": resolve(resource.buffer), "offset": resource.offset}
        if resource.size is not None:
            binding["size"] = resource.size
        return {"binding": entry.binding, "resource": binding}
    return {"binding": entry.binding, "resource": resolve(resource)}


def bind_group_dict(descriptor: BindGroupDescriptor, resolve: _Resolver) -> dict[str, Any]:
    """Translate a bind group descriptor for ``create_bind_group``.

    ``resolve`` maps EasyGPU handles/objects (buffers, samplers, views, layout)
    to the corresponding wgpu objects.
    """
    out: dict[str, Any] = {
        "layout": resolve(descriptor.layout),
        "entries": [_bind_group_entry_dict(entry, resolve) for entry in descriptor.entries],
    }
    _add_label(out, descriptor.label)
    return out


def pipeline_layout_dict(
    descriptor: PipelineLayoutDescriptor, resolve: _Resolver
) -> dict[str, Any]:
    """Translate a pipeline layout descriptor for ``create_pipeline_layout``."""
    out: dict[str, Any] = {
        "bind_group_layouts": [resolve(layout) for layout in descriptor.bind_group_layouts]
    }
    _add_label(out, descriptor.label)
    return out


def _programmable_stage_dict(entry: ShaderStageEntry, resolve: _Resolver) -> dict[str, Any]:
    out: dict[str, Any] = {"module": resolve(entry.module), "entry_point": entry.entry_point}
    if entry.constants:
        out["constants"] = dict(entry.constants)
    return out


def _vertex_buffer_layout_dict(layout: VertexBufferLayout) -> dict[str, Any]:
    return {
        "array_stride": layout.array_stride,
        "step_mode": layout.step_mode,
        "attributes": [
            {
                "format": _name(c.VertexFormat, int(attribute.format)),
                "offset": attribute.offset,
                "shader_location": attribute.shader_location,
            }
            for attribute in layout.attributes
        ],
    }


def _depth_stencil_state_dict(state: DepthStencilState) -> dict[str, Any]:
    return {
        "format": texture_format_str(state.format),
        "depth_write_enabled": state.depth_write_enabled,
        "depth_compare": _name(c.CompareFunction, int(state.depth_compare)),
        "depth_bias": state.depth_bias,
        "depth_bias_slope_scale": state.depth_bias_slope_scale,
        "depth_bias_clamp": state.depth_bias_clamp,
    }


def _blend_component_dict(component: Any) -> dict[str, Any]:
    return {
        "operation": _name(c.BlendOperation, int(component.operation)),
        "src_factor": _name(c.BlendFactor, int(component.src_factor)),
        "dst_factor": _name(c.BlendFactor, int(component.dst_factor)),
    }


def _color_target_dict(target: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "format": _name(c.TextureFormat, int(target.format)),
        "write_mask": target.write_mask,
    }
    if target.blend is not None:
        out["blend"] = {
            "color": _blend_component_dict(target.blend.color),
            "alpha": _blend_component_dict(target.blend.alpha),
        }
    return out


def render_pipeline_dict(
    descriptor: RenderPipelineDescriptor, resolve: _Resolver
) -> dict[str, Any]:
    """Translate a render pipeline descriptor for ``create_render_pipeline``."""
    vertex = _programmable_stage_dict(descriptor.vertex_stage, resolve)
    if descriptor.vertex_buffers is None:
        vertex["buffers"] = []
    else:
        vertex["buffers"] = [
            _vertex_buffer_layout_dict(layout) for layout in descriptor.vertex_buffers
        ]
    out: dict[str, Any] = {
        "layout": resolve(descriptor.layout) if descriptor.layout is not None else "auto",
        "vertex": vertex,
        "primitive": {
            "topology": _name(c.PrimitiveTopology, int(descriptor.primitive_topology)),
            "cull_mode": _name(c.CullMode, int(descriptor.cull_mode)),
            "front_face": _name(c.FrontFace, int(descriptor.front_face)),
        },
    }
    _add_label(out, descriptor.label)
    if descriptor.depth_stencil is not None:
        out["depth_stencil"] = _depth_stencil_state_dict(descriptor.depth_stencil)
    if descriptor.multisample is not None:
        out["multisample"] = {
            "count": descriptor.multisample.count,
            "mask": descriptor.multisample.mask,
            "alpha_to_coverage_enabled": descriptor.multisample.alpha_to_coverage_enabled,
        }
    if descriptor.fragment_stage is not None:
        fragment = _programmable_stage_dict(descriptor.fragment_stage, resolve)
        fragment["targets"] = [
            _color_target_dict(target) for target in (descriptor.fragment_targets or [])
        ]
        out["fragment"] = fragment
    return out


def compute_pipeline_dict(
    descriptor: ComputePipelineDescriptor, resolve: _Resolver
) -> dict[str, Any]:
    """Translate a compute pipeline descriptor for ``create_compute_pipeline``."""
    out: dict[str, Any] = {
        "layout": resolve(descriptor.layout) if descriptor.layout is not None else "auto",
        "compute": _programmable_stage_dict(descriptor.compute, resolve),
    }
    _add_label(out, descriptor.label)
    return out


def color_attachment_dict(
    attachment: RenderPassColorAttachment, resolve: _Resolver
) -> dict[str, Any]:
    """Translate a render-pass color attachment for ``begin_render_pass``."""
    out: dict[str, Any] = {
        "view": resolve(attachment.view),
        "clear_value": attachment.clear_value,
        "load_op": _name(c.LoadOp, int(attachment.load_op)),
        "store_op": _name(c.StoreOp, int(attachment.store_op)),
    }
    if attachment.resolve_target is not None:
        out["resolve_target"] = resolve(attachment.resolve_target)
    return out


def depth_stencil_attachment_dict(
    attachment: RenderPassDepthStencilAttachment, resolve: _Resolver
) -> dict[str, Any]:
    """Translate a render-pass depth/stencil attachment for ``begin_render_pass``.

    The stencil load/store ops and read-only flags follow WebGPU: when the
    attachment format has no stencil aspect (or the view is read-only)
    wgpu-py expects the corresponding keys to be absent, so
    ``WgpuPyGPU`` trims them for non-stencil formats after this translation.
    """
    return {
        "view": resolve(attachment.view),
        "depth_clear_value": attachment.depth_clear_value,
        "depth_load_op": _name(c.LoadOp, int(attachment.depth_load_op)),
        "depth_store_op": _name(c.StoreOp, int(attachment.depth_store_op)),
        "depth_read_only": attachment.depth_read_only,
        "stencil_clear_value": attachment.stencil_clear_value,
        "stencil_load_op": _name(c.LoadOp, int(attachment.stencil_load_op)),
        "stencil_store_op": _name(c.StoreOp, int(attachment.stencil_store_op)),
        "stencil_read_only": attachment.stencil_read_only,
    }


__all__ = [
    "PASSTHROUGH_FLAGS",
    "bind_group_dict",
    "bind_group_layout_dict",
    "buffer_descriptor_dict",
    "color_attachment_dict",
    "compute_pipeline_dict",
    "depth_stencil_attachment_dict",
    "enum_names",
    "pipeline_layout_dict",
    "render_pipeline_dict",
    "sampler_descriptor_dict",
    "shader_module_dict",
    "texture_descriptor_dict",
    "texture_format_str",
    "texture_view_descriptor_dict",
]
