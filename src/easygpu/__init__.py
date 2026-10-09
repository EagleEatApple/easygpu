"""EasyGPU — a Python wrapper for WebGPU.

The core class for tests is :class:`easygpu.fake_gpu.FakeGPU`: a recording
implementation of the :class:`easygpu.gpu.GPU` protocol that lets you TDD
your command-encoding logic without a GPU.
"""

import importlib.util
import warnings

__version__ = "0.2.0"

from easygpu.bindgroup import (
    BindGroup,
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BufferSlice,
    PipelineLayout,
    PipelineLayoutDescriptor,
)
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import (
    AddressMode,
    BlendFactor,
    BlendOperation,
    BufferBindingType,
    BufferUsage,
    CompareFunction,
    CullMode,
    FilterMode,
    FrontFace,
    IndexFormat,
    LoadOp,
    MapMode,
    MipmapFilterMode,
    PrimitiveTopology,
    QueryType,
    SamplerBindingType,
    ShaderStage,
    StoreOp,
    TextureFormat,
    TextureSampleType,
    TextureUsage,
    VertexFormat,
)
from easygpu.device import Adapter, Device, Queue, request_adapter
from easygpu.encoder import (
    CommandBuffer,
    CommandEncoder,
    ComputePassDescriptor,
    ComputePassEncoder,
    RenderBundle,
    RenderBundleDescriptor,
    RenderBundleEncoder,
    RenderPassColorAttachment,
    RenderPassDepthStencilAttachment,
    RenderPassEncoder,
)
from easygpu.errors import EasyGPUError, GPUObjectDeletedError, GPUValidationError
from easygpu.gpu import GPU, configure, get_gpu
from easygpu.matrix import (
    create,
    identity,
    look_at,
    multiply,
    ortho,
    perspective,
    rotation,
    rotation_x,
    rotation_y,
    rotation_z,
    scale,
    translation,
)
from easygpu.offscreen import OffscreenTarget, offscreen_target, readback_rgba8
from easygpu.pipeline import (
    BlendComponent,
    BlendState,
    ColorTargetState,
    ComputePipeline,
    ComputePipelineDescriptor,
    DepthStencilState,
    MultisampleState,
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.query import QuerySet, QuerySetDescriptor
from easygpu.sampler import Sampler, SamplerDescriptor
from easygpu.shader import ShaderModule, ShaderModuleDescriptor
from easygpu.surface import SurfaceTarget
from easygpu.texture import (
    Texture,
    TextureDescriptor,
    TextureView,
    TextureViewDescriptor,
)

if importlib.util.find_spec("wgpu") is None:
    warnings.warn(
        "easygpu 0.2.0 ships a wgpu-py backend (install with "
        "`uv sync --extra wgpu` or `pip install easygpu[wgpu]`); without the "
        "extra, FakeGPU still records command encoding with no GPU",
        UserWarning,
        stacklevel=2,
    )

__all__ = [
    "AddressMode",
    "Adapter",
    "BindGroup",
    "BindGroupDescriptor",
    "BindGroupEntry",
    "BindGroupLayout",
    "BindGroupLayoutDescriptor",
    "BindGroupLayoutEntry",
    "BlendComponent",
    "BlendFactor",
    "BlendOperation",
    "BlendState",
    "Buffer",
    "BufferBindingType",
    "BufferDescriptor",
    "BufferSlice",
    "BufferUsage",
    "ColorTargetState",
    "CommandBuffer",
    "CommandEncoder",
    "CompareFunction",
    "ComputePassDescriptor",
    "ComputePassEncoder",
    "ComputePipeline",
    "ComputePipelineDescriptor",
    "CullMode",
    "DepthStencilState",
    "Device",
    "EasyGPUError",
    "FilterMode",
    "FrontFace",
    "GPU",
    "GPUObjectDeletedError",
    "GPUValidationError",
    "IndexFormat",
    "LoadOp",
    "MapMode",
    "MipmapFilterMode",
    "MultisampleState",
    "OffscreenTarget",
    "PipelineLayout",
    "PipelineLayoutDescriptor",
    "PrimitiveTopology",
    "Queue",
    "QuerySet",
    "QuerySetDescriptor",
    "QueryType",
    "RenderBundle",
    "RenderBundleDescriptor",
    "RenderBundleEncoder",
    "RenderPassColorAttachment",
    "RenderPassDepthStencilAttachment",
    "RenderPassEncoder",
    "RenderPipeline",
    "RenderPipelineDescriptor",
    "Sampler",
    "SamplerBindingType",
    "SamplerDescriptor",
    "ShaderModule",
    "ShaderModuleDescriptor",
    "ShaderStage",
    "ShaderStageEntry",
    "StoreOp",
    "SurfaceTarget",
    "Texture",
    "TextureDescriptor",
    "TextureFormat",
    "TextureSampleType",
    "TextureUsage",
    "TextureView",
    "TextureViewDescriptor",
    "VertexAttribute",
    "VertexBufferLayout",
    "VertexFormat",
    "configure",
    "create",
    "get_gpu",
    "identity",
    "look_at",
    "multiply",
    "offscreen_target",
    "ortho",
    "perspective",
    "readback_rgba8",
    "request_adapter",
    "rotation",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale",
    "translation",
]
