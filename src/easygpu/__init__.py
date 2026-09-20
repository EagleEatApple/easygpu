"""EasyGPU — a Python wrapper for WebGPU.

The core class for tests is :class:`easygpu.fake_gpu.FakeGPU`: a recording
implementation of the :class:`easygpu.gpu.GPU` protocol that lets you TDD
your command-encoding logic without a GPU.
"""

import warnings

__version__ = "0.0.0"

from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import (
    BufferUsage,
    LoadOp,
    PrimitiveTopology,
    ShaderStage,
    StoreOp,
    TextureFormat,
    VertexFormat,
)
from easygpu.device import Adapter, Device, Queue, request_adapter
from easygpu.encoder import (
    CommandBuffer,
    CommandEncoder,
    RenderPassColorAttachment,
    RenderPassEncoder,
)
from easygpu.errors import EasyGPUError, GPUObjectDeletedError, GPUValidationError
from easygpu.gpu import GPU, configure, get_gpu
from easygpu.pipeline import (
    RenderPipeline,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.shader import ShaderModule, ShaderModuleDescriptor

warnings.warn(
    "easygpu 0.0.0 is an API preview: the GPU Protocol is the deliverable, "
    "no real backend ships yet",
    UserWarning,
    stacklevel=2,
)

__all__ = [
    "GPU",
    "Adapter",
    "Buffer",
    "BufferDescriptor",
    "BufferUsage",
    "CommandBuffer",
    "CommandEncoder",
    "Device",
    "EasyGPUError",
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
]
