"""wgpu-py backend: second implementation of the :class:`easygpu.gpu.GPU` protocol.

Importing this module does NOT import wgpu. The real adapter is created lazily on
first ``request_adapter`` so ``easygpu`` keeps zero required dependencies; the
optional ``wgpu`` extra provides wgpu-py.

Task 4 ships the skeleton: adapter/device/queue lifecycle, structural
conformance to the full protocol, ``from_wgpu_device`` single-device adoption,
the public ``underlying_device`` escape hatch, and a ``calls`` recording seam
for FakeGPU-parity. Every remaining protocol method whose name is listed in
``easygpu.gpu.WGPU_BACKEND_UNSUPPORTED`` raises
``GPUValidationError("<name> not yet supported by WgpuPyGPU")``.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any, NoReturn

from easygpu.constants import IndexFormat
from easygpu.errors import GPUValidationError
from easygpu.gpu import (
    WGPU_BACKEND_UNSUPPORTED,
    AdapterHandle,
    BindGroupHandle,
    BindGroupLayoutHandle,
    BufferHandle,
    CommandBufferHandle,
    CommandEncoderHandle,
    ComputePassHandle,
    ComputePipelineHandle,
    DeviceHandle,
    PipelineLayoutHandle,
    QuerySetHandle,
    QueueHandle,
    RenderBundleEncoderHandle,
    RenderBundleHandle,
    RenderPassHandle,
    RenderPipelineHandle,
    SamplerHandle,
    ShaderModuleHandle,
    TextureHandle,
    TextureViewHandle,
)
from easygpu.surface import SurfaceTarget
from easygpu.wgpu_translate import (
    bind_group_dict,
    bind_group_layout_dict,
    buffer_descriptor_dict,
    color_attachment_dict,
    compute_pipeline_dict,
    depth_stencil_attachment_dict,
    enum_names,
    pipeline_layout_dict,
    render_pipeline_dict,
    sampler_descriptor_dict,
    shader_module_dict,
    texture_descriptor_dict,
    texture_format_str,
    texture_view_descriptor_dict,
)

if TYPE_CHECKING:
    from easygpu.bindgroup import (
        BindGroupDescriptor,
        BindGroupLayoutDescriptor,
        PipelineLayoutDescriptor,
    )
    from easygpu.buffer import BufferDescriptor
    from easygpu.encoder import (
        RenderBundleDescriptor,
        RenderPassColorAttachment,
        RenderPassDepthStencilAttachment,
    )
    from easygpu.pipeline import ComputePipelineDescriptor, RenderPipelineDescriptor
    from easygpu.query import QuerySetDescriptor
    from easygpu.sampler import SamplerDescriptor
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.texture import TextureDescriptor, TextureViewDescriptor

_ADOPTED = object()

_FORMAT_BPP: dict[str, int] = {
    "bgra8unorm": 4,
    "rgba8unorm": 4,
    "depth24plus": 0,
    "depth32float": 4,
    "depth24plus-stencil8": 4,
    "depth32float-stencil8": 4,
}

_INDEX_FORMAT_NAMES = enum_names(IndexFormat)


def _trim_depth_stencil_keys(depth: dict[str, Any], wgpu_view: Any) -> dict[str, Any]:
    """Drop load/store-op keys whose aspect the view format does not provide.

    wgpu-py (verified on 0.32.0) requires ``depth_load_op``/``depth_store_op``
    exactly when the view format has a depth aspect and is not read-only, and
    the stencil pair likewise; passing them otherwise only emits a one-time
    warning. The translator always includes all four plus the read-only flags,
    so the backend trims what the aspect does not warrant.
    """
    if wgpu_view is None:
        return depth
    fmt = getattr(getattr(wgpu_view, "texture", None), "format", "") or ""
    keep_depth = "depth" in fmt and not depth.get("depth_read_only", False)
    keep_stencil = "stencil" in fmt and not depth.get("stencil_read_only", False)
    if keep_depth and keep_stencil:
        return depth
    out = dict(depth)
    if not keep_depth:
        out.pop("depth_load_op", None)
        out.pop("depth_store_op", None)
    if not keep_stencil:
        out.pop("stencil_load_op", None)
        out.pop("stencil_store_op", None)
    return out


def _load_wgpu() -> Any:
    try:
        return importlib.import_module("wgpu")
    except Exception as exc:
        raise GPUValidationError("wgpu is not installed: run `uv sync --extra wgpu`") from exc


class WgpuPyGPU:
    """Delegates the GPU Protocol to wgpu-py. Structurally conforms to ``GPU``."""

    def __init__(self, *, force_fallback_adapter: bool | None = None) -> None:
        self._force_fallback = (
            bool(os.environ.get("EASYGPU_SOFTWARE", ""))
            if force_fallback_adapter is None
            else force_fallback_adapter
        )
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._next = 1
        self._adapters: dict[int, Any] = {}
        self._devices: dict[int, Any] = {}
        self._queues: dict[int, Any] = {}
        self._queue_devices: dict[int, DeviceHandle] = {}
        self._buffers: dict[int, Any] = {}
        self._buffer_shadow: dict[int, bytearray] = {}
        self._buffer_devices: dict[int, DeviceHandle] = {}
        self._mapped: dict[int, tuple[int, int, int]] = {}
        self._objects: dict[int, Any] = {}
        self._owners: dict[int, DeviceHandle] = {}
        self._bind_group_layout_cache: dict[tuple[int, int], int] = {}
        self._textures: dict[int, Any] = {}
        self._texture_formats: dict[int, str] = {}
        self._texture_sizes: dict[int, tuple[int, int, int]] = {}
        self._view_texture: dict[int, TextureHandle] = {}
        self._command_encoders: dict[int, Any] = {}
        self._render_passes: dict[int, Any] = {}
        self._compute_passes: dict[int, Any] = {}
        self._command_buffers: dict[int, Any] = {}
        self._encoder_devices: dict[int, DeviceHandle] = {}
        self._offscreen_views: dict[int, TextureViewHandle] = {}
        self._offscreen_devices: dict[int, DeviceHandle] = {}
        self._adopted_device: Any | None = None
        self._last_device: Any | None = None

    @classmethod
    def from_wgpu_device(cls, wgpu_device: Any) -> WgpuPyGPU:
        """Adopt a live ``wgpu.GPUDevice`` (migration-guide commitment).

        ``request_adapter()`` returns an adapter handle whose ``request_device()``
        returns the adopted device itself; every resource factory delegates to it.
        Requesting a different device is not possible via this path.
        """
        if wgpu_device is None:
            raise GPUValidationError("from_wgpu_device requires a live wgpu.GPUDevice, got None")
        gpu = cls()
        gpu._adopted_device = wgpu_device
        gpu._last_device = wgpu_device
        return gpu

    @property
    def underlying_device(self) -> Any:
        """The wrapped ``wgpu.GPUDevice`` (public escape hatch)."""
        device = self._last_device if self._last_device is not None else self._adopted_device
        if device is None:
            raise GPUValidationError(
                "no wgpu device yet: request an adapter+device or use from_wgpu_device"
            )
        return device

    def _handle(self) -> int:
        handle = self._next
        self._next += 1
        return handle

    def _record(self, name: str, args: tuple[Any, ...]) -> None:
        self.calls.append((name, args))

    def calls_of(self, name: str) -> list[tuple[Any, ...]]:
        """Return the recorded arguments of every call named ``name``."""
        return [args for recorded, args in self.calls if recorded == name]

    def _fail(self, message: str) -> NoReturn:
        raise GPUValidationError(message)

    def _unsupported(self, name: str) -> NoReturn:
        """Raise for a protocol method listed in :data:`WGPU_BACKEND_UNSUPPORTED`."""
        if name not in WGPU_BACKEND_UNSUPPORTED:
            raise RuntimeError(f"{name} missing from gpu.WGPU_BACKEND_UNSUPPORTED")
        self._fail(f"{name} not yet supported by WgpuPyGPU")

    def _require_texture(self, texture: TextureHandle) -> Any:
        wgpu_texture = self._textures.get(texture)
        if wgpu_texture is None:
            self._fail(f"unknown texture handle {texture}")
        return wgpu_texture

    def _require_render_pass(self, render_pass: RenderPassHandle) -> Any:
        wgpu_pass = self._render_passes.get(render_pass)
        if wgpu_pass is None:
            self._fail(f"unknown render pass {render_pass}")
        return wgpu_pass

    def _require_compute_pass(self, compute_pass: ComputePassHandle) -> Any:
        wgpu_pass = self._compute_passes.get(compute_pass)
        if wgpu_pass is None:
            self._fail(f"unknown compute pass {compute_pass}")
        return wgpu_pass

    def _require_encoder(self, encoder: CommandEncoderHandle) -> Any:
        wgpu_encoder = self._command_encoders.get(encoder)
        if wgpu_encoder is None:
            self._fail(f"unknown command encoder handle {encoder}")
        return wgpu_encoder

    def _resolve_target(self, view: Any, device: DeviceHandle) -> Any:
        """Resolve a render-attachment target to its wgpu texture view.

        Accepts an int view handle, a ``SurfaceTarget``, a ``TextureView``
        wrapper, or an offscreen target (migration-guide commitment: lazy
        offscreen texture creation).
        """
        if isinstance(view, int):
            if view == 0:
                self._fail(
                    "swapchain surface (view=0) is not supported by WgpuPyGPU; "
                    "create a texture instead"
                )
            return self._resolve(view)
        if isinstance(view, SurfaceTarget):
            return view.acquire()
        object_id = getattr(view, "id", None)
        if object_id is not None:
            return self._resolve_target(object_id, device)
        if hasattr(view, "width") and hasattr(view, "height") and hasattr(view, "format"):
            return self._offscreen_view(view, device)
        return self._resolve(view)

    def _offscreen_view(self, target: Any, device: DeviceHandle) -> Any:
        """Lazily create (and cache per target) the offscreen texture+view."""
        cached = self._offscreen_views.get(id(target))
        if cached is not None:
            return self._resolve(cached)
        from easygpu.constants import TextureUsage
        from easygpu.texture import TextureDescriptor

        texture = self.create_texture(
            device,
            TextureDescriptor(
                size=(target.width, target.height, 1),
                format=target.format,
                usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
            ),
        )
        view = self.create_texture_view(texture)
        self._offscreen_views[id(target)] = view
        self._offscreen_devices[id(target)] = device
        return self._resolve(view)

    def _require_device(self, device: DeviceHandle) -> Any:
        wgpu_device = self._devices.get(device)
        if wgpu_device is None:
            self._fail(f"unknown device handle {device}")
        return wgpu_device

    def _require_buffer(self, buffer: BufferHandle) -> Any:
        wgpu_buffer = self._buffers.get(buffer)
        if wgpu_buffer is None:
            self._fail(f"unknown buffer handle {buffer}")
        return wgpu_buffer

    def _queue_of_buffer(self, buffer: BufferHandle) -> Any:
        device = self._buffer_devices.get(buffer)
        if device is None:
            self._fail(f"unknown buffer handle {buffer}")
        return self._require_device(device).queue

    def _resolve(self, resource: Any) -> Any:
        """Map an EasyGPU handle/object to its wrapped wgpu object."""
        if isinstance(resource, int):
            found = self._objects.get(resource)
            if found is None:
                found = self._buffers.get(resource)
            if found is None:
                self._fail(f"unknown resource handle {resource}")
            return found
        object_id = getattr(resource, "id", None)
        if object_id is not None:
            return self._resolve(object_id)
        return resource

    # --- adapter / device / queue -------------------------------------

    def request_adapter(self, *, force_fallback_adapter: bool = False) -> AdapterHandle:
        self._record("request_adapter", (force_fallback_adapter,))
        if self._adopted_device is not None:
            handle = self._handle()
            self._adapters[handle] = _ADOPTED
            return handle
        wgpu = _load_wgpu()
        force = force_fallback_adapter or self._force_fallback
        try:
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance", force_fallback_adapter=force
            )
        except Exception:  # noqa: BLE001
            adapter = None
        if adapter is None:
            try:
                adapters = wgpu.gpu.enumerate_adapters_sync()
                adapter = adapters[0] if adapters else None
            except Exception:  # noqa: BLE001
                adapter = None
        if adapter is None:
            try:
                adapter = wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
            except Exception:  # noqa: BLE001
                adapter = None
        if adapter is None:
            self._fail(
                "no GPU adapter available (install a driver or enable a software rasterizer)"
            )
        handle = self._handle()
        self._adapters[handle] = adapter
        return handle

    def request_device(self, adapter: AdapterHandle, *, label: str | None = None) -> DeviceHandle:
        self._record("request_device", (adapter, label))
        source = self._adapters.get(adapter)
        if source is None:
            self._fail(f"unknown adapter handle {adapter}")
        if source is _ADOPTED:
            if self._adopted_device is None:
                self._fail("from_wgpu_device not configured with a live wgpu device")
            wgpu_device = self._adopted_device
        else:
            wgpu_device = source.request_device_sync(label=label or "")
        handle = self._handle()
        self._devices[handle] = wgpu_device
        self._last_device = wgpu_device
        return handle

    def get_queue(self, device: DeviceHandle) -> QueueHandle:
        self._record("get_queue", (device,))
        wgpu_device = self._devices.get(device)
        if wgpu_device is None:
            self._fail(f"unknown device handle {device}")
        handle = self._handle()
        self._queues[handle] = wgpu_device.queue
        self._queue_devices[handle] = device
        return handle

    def get_adapter_info(self, adapter: AdapterHandle) -> dict[str, object]:
        self._record("get_adapter_info", (adapter,))
        source = self._adapters.get(adapter)
        if source is None or source is _ADOPTED:
            return {
                "vendor": "unknown",
                "architecture": "adopted",
                "device": "unknown",
                "description": "wgpu device adopted via WgpuPyGPU.from_wgpu_device",
            }
        try:
            raw = dict(source.info)
        except Exception:  # noqa: BLE001
            raw = {}
        return {
            "vendor": str(raw.get("vendor_id", raw.get("vendor", "unknown"))),
            "architecture": str(raw.get("architecture", "") or "unknown"),
            "device": str(raw.get("device", "unknown")),
            "description": str(raw.get("description", "") or ""),
        }

    def on_uncaptured_error(self, device: DeviceHandle) -> str | None:
        self._record("on_uncaptured_error", (device,))
        return None

    def on_submitted_work_done(self, queue: QueueHandle) -> None:
        self._record("on_submitted_work_done", (queue,))

    # --- resource creation --------------------------------------------

    def create_buffer(self, device: DeviceHandle, descriptor: BufferDescriptor) -> BufferHandle:
        self._record("create_buffer", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_buffer = wgpu_device.create_buffer(**buffer_descriptor_dict(descriptor))
        handle = self._handle()
        self._buffers[handle] = wgpu_buffer
        self._buffer_shadow[handle] = bytearray(descriptor.size)
        self._buffer_devices[handle] = device
        return handle

    def create_texture(self, device: DeviceHandle, descriptor: TextureDescriptor) -> TextureHandle:
        self._record("create_texture", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_texture = wgpu_device.create_texture(**texture_descriptor_dict(descriptor))
        handle = self._handle()
        self._textures[handle] = wgpu_texture
        self._objects[handle] = wgpu_texture
        self._owners[handle] = device
        size = tuple(descriptor.size)[:3]
        self._texture_sizes[handle] = (size[0], size[1], size[2])
        self._texture_formats[handle] = texture_format_str(int(descriptor.format))
        return handle

    def create_texture_view(
        self, texture: TextureHandle, descriptor: TextureViewDescriptor | None = None
    ) -> TextureViewHandle:
        self._record("create_texture_view", (texture, descriptor))
        wgpu_texture = self._require_texture(texture)
        view_dict = texture_view_descriptor_dict(descriptor) if descriptor else {}
        wgpu_view = wgpu_texture.create_view(**view_dict)
        handle = self._handle()
        self._objects[handle] = wgpu_view
        self._view_texture[handle] = texture
        self._owners[handle] = self._owners[texture]
        return handle

    def generate_mipmap(self, texture: TextureHandle) -> None:
        del texture
        self._unsupported("generate_mipmap")

    def create_sampler(self, device: DeviceHandle, descriptor: SamplerDescriptor) -> SamplerHandle:
        self._record("create_sampler", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_sampler = wgpu_device.create_sampler(**sampler_descriptor_dict(descriptor))
        handle = self._handle()
        self._objects[handle] = wgpu_sampler
        self._owners[handle] = device
        return handle

    def create_shader_module(
        self, device: DeviceHandle, descriptor: ShaderModuleDescriptor
    ) -> ShaderModuleHandle:
        self._record("create_shader_module", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_shader = wgpu_device.create_shader_module(**shader_module_dict(descriptor))
        handle = self._handle()
        self._objects[handle] = wgpu_shader
        self._owners[handle] = device
        return handle

    def create_bind_group_layout(
        self, device: DeviceHandle, descriptor: BindGroupLayoutDescriptor
    ) -> BindGroupLayoutHandle:
        self._record("create_bind_group_layout", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_layout = wgpu_device.create_bind_group_layout(**bind_group_layout_dict(descriptor))
        handle = self._handle()
        self._objects[handle] = wgpu_layout
        self._owners[handle] = device
        return handle

    def create_pipeline_layout(
        self, device: DeviceHandle, descriptor: PipelineLayoutDescriptor
    ) -> PipelineLayoutHandle:
        self._record("create_pipeline_layout", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_layout = wgpu_device.create_pipeline_layout(
            **pipeline_layout_dict(descriptor, self._resolve)
        )
        handle = self._handle()
        self._objects[handle] = wgpu_layout
        self._owners[handle] = device
        return handle

    def create_bind_group(
        self, device: DeviceHandle, descriptor: BindGroupDescriptor
    ) -> BindGroupHandle:
        self._record("create_bind_group", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_group = wgpu_device.create_bind_group(**bind_group_dict(descriptor, self._resolve))
        handle = self._handle()
        self._objects[handle] = wgpu_group
        self._owners[handle] = device
        return handle

    def get_bind_group_layout(
        self,
        pipeline: RenderPipelineHandle | ComputePipelineHandle,
        group_index: int,
    ) -> BindGroupLayoutHandle:
        key = (pipeline, group_index)
        cached = self._bind_group_layout_cache.get(key)
        if cached is not None:
            return cached
        self._record("get_bind_group_layout", (pipeline, group_index))
        wgpu_pipeline = self._objects.get(pipeline)
        if wgpu_pipeline is None:
            self._fail(f"unknown pipeline handle {pipeline}")
        wgpu_layout = wgpu_pipeline.get_bind_group_layout(group_index)
        handle = self._handle()
        self._objects[handle] = wgpu_layout
        self._bind_group_layout_cache[key] = handle
        return handle

    def create_render_pipeline(
        self, device: DeviceHandle, descriptor: RenderPipelineDescriptor
    ) -> RenderPipelineHandle:
        self._record("create_render_pipeline", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_pipeline = wgpu_device.create_render_pipeline(
            **render_pipeline_dict(descriptor, self._resolve)
        )
        handle = self._handle()
        self._objects[handle] = wgpu_pipeline
        self._owners[handle] = device
        return handle

    def create_compute_pipeline(
        self, device: DeviceHandle, descriptor: ComputePipelineDescriptor
    ) -> ComputePipelineHandle:
        self._record("create_compute_pipeline", (device, descriptor))
        wgpu_device = self._require_device(device)
        wgpu_pipeline = wgpu_device.create_compute_pipeline(
            **compute_pipeline_dict(descriptor, self._resolve)
        )
        handle = self._handle()
        self._objects[handle] = wgpu_pipeline
        self._owners[handle] = device
        return handle

    def create_query_set(
        self, device: DeviceHandle, descriptor: QuerySetDescriptor
    ) -> QuerySetHandle:
        del device, descriptor
        self._unsupported("create_query_set")

    def create_command_encoder(
        self, device: DeviceHandle, *, label: str | None = None
    ) -> CommandEncoderHandle:
        self._record("create_command_encoder", (device, label))
        wgpu_device = self._require_device(device)
        wgpu_encoder = wgpu_device.create_command_encoder(label=label or "")
        handle = self._handle()
        self._command_encoders[handle] = wgpu_encoder
        self._encoder_devices[handle] = device
        return handle

    # --- queue data path ----------------------------------------------

    def write_buffer(
        self, queue: QueueHandle, buffer: BufferHandle, buffer_offset: int, data: bytes
    ) -> None:
        self._record("write_buffer", (queue, buffer, buffer_offset, data))
        if not data:
            return
        wgpu_queue = self._queues.get(queue)
        if wgpu_queue is None:
            self._fail(f"unknown queue handle {queue}")
        wgpu_buffer = self._require_buffer(buffer)
        if self._queue_devices.get(queue) != self._buffer_devices.get(buffer):
            self._fail("queue and buffer belong to different devices")
        wgpu_queue.write_buffer(wgpu_buffer, buffer_offset, data)
        shadow = self._buffer_shadow[buffer]
        shadow[buffer_offset : buffer_offset + len(data)] = data

    def map_async(self, buffer: BufferHandle, mode: int, offset: int, size: int) -> None:
        self._record("map_async", (buffer, mode, offset, size))
        self._require_buffer(buffer)
        self._mapped[buffer] = (mode, offset, size)

    def get_mapped_range(self, buffer: BufferHandle, offset: int, size: int) -> bytes:
        self._record("get_mapped_range", (buffer, offset, size))
        mapped = self._mapped.get(buffer)
        if mapped is None:
            self._fail(f"buffer {buffer} not mapped")
        wgpu_buffer = self._require_buffer(buffer)
        data = self._queue_of_buffer(buffer).read_buffer(wgpu_buffer)
        return bytes(data[offset : offset + size])

    def unmap(self, buffer: BufferHandle) -> None:
        self._record("unmap", (buffer,))
        self._require_buffer(buffer)
        self._mapped.pop(buffer, None)

    def write_texture(
        self,
        queue: QueueHandle,
        texture: TextureHandle,
        data: bytes,
        *,
        width: int,
        height: int,
        mip_level: int = 0,
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self._record("write_texture", (queue, texture, data, width, height, mip_level, origin))
        wgpu_queue = self._queues.get(queue)
        if wgpu_queue is None:
            self._fail(f"unknown queue handle {queue}")
        wgpu_texture = self._require_texture(texture)
        tight = width * _FORMAT_BPP[self._texture_formats[texture]]
        stride = (tight + 255) & ~255
        padded = bytearray(stride * height)
        for row in range(height):
            padded[row * stride : row * stride + tight] = data[row * tight : (row + 1) * tight]
        wgpu_queue.write_texture(
            destination={"texture": wgpu_texture, "mip_level": mip_level, "origin": origin},
            data=bytes(padded),
            data_layout={"bytes_per_row": stride, "rows_per_image": height},
            size=(width, height, 1),
        )

    def submit(self, queue: QueueHandle, command_buffers: list[CommandBufferHandle]) -> None:
        self._record("submit", (queue, list(command_buffers)))
        wgpu_queue = self._queues.get(queue)
        if wgpu_queue is None:
            self._fail(f"unknown queue handle {queue}")
        wgpu_buffers = []
        for handle in command_buffers:
            wgpu_command_buffer = self._command_buffers.get(handle)
            if wgpu_command_buffer is None:
                self._fail(f"unknown command buffer handle {handle}")
            wgpu_buffers.append(wgpu_command_buffer)
        wgpu_queue.submit(wgpu_buffers)

    # --- command encoding ---------------------------------------------

    def begin_render_pass(
        self,
        encoder: CommandEncoderHandle,
        color_attachments: list[RenderPassColorAttachment],
        depth_stencil_attachment: RenderPassDepthStencilAttachment | None = None,
    ) -> RenderPassHandle:
        self._record(
            "begin_render_pass", (encoder, list(color_attachments), depth_stencil_attachment)
        )
        wgpu_encoder = self._require_encoder(encoder)
        device = self._encoder_devices.get(encoder)
        if device is None:
            self._fail(f"unknown command encoder handle {encoder}")

        def resolve_view(v: Any) -> Any:
            return self._resolve_target(v, device)

        color = [
            color_attachment_dict(attachment, resolve_view) for attachment in color_attachments
        ]
        depth = None
        if depth_stencil_attachment is not None:
            depth = depth_stencil_attachment_dict(depth_stencil_attachment, resolve_view)
            depth = _trim_depth_stencil_keys(depth, depth["view"])
        wgpu_pass = wgpu_encoder.begin_render_pass(
            color_attachments=color, depth_stencil_attachment=depth
        )
        handle = self._handle()
        self._render_passes[handle] = wgpu_pass
        self._owners[handle] = device
        return handle

    def set_pipeline(self, render_pass: RenderPassHandle, pipeline: RenderPipelineHandle) -> None:
        self._record("set_pipeline", (render_pass, pipeline))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pipeline = self._objects.get(pipeline)
        if wgpu_pipeline is None:
            self._fail(f"unknown pipeline handle {pipeline}")
        wgpu_pass.set_pipeline(wgpu_pipeline)

    def set_bind_group(
        self,
        render_pass: RenderPassHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None:
        self._record("set_bind_group", (render_pass, index, group))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_group = self._objects.get(group)
        if wgpu_group is None:
            self._fail(f"unknown bind group handle {group}")
        wgpu_pass.set_bind_group(index, wgpu_group)

    def set_vertex_buffer(
        self,
        render_pass: RenderPassHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None:
        self._record("set_vertex_buffer", (render_pass, slot, buffer, offset))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.set_vertex_buffer(slot, self._require_buffer(buffer), offset)

    def set_index_buffer(
        self,
        render_pass: RenderPassHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        self._record("set_index_buffer", (render_pass, buffer, format, offset, size))
        wgpu_pass = self._require_render_pass(render_pass)
        index_format = _INDEX_FORMAT_NAMES.get(format)
        if index_format is None:
            self._fail(f"unknown index format {format}")
        wgpu_pass.set_index_buffer(self._require_buffer(buffer), index_format, offset, size)

    def draw(
        self,
        render_pass: RenderPassHandle,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._record(
            "draw", (render_pass, vertex_count, instance_count, first_vertex, first_instance)
        )
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.draw(vertex_count, instance_count, first_vertex, first_instance)

    def draw_indexed(
        self,
        render_pass: RenderPassHandle,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        self._record(
            "draw_indexed",
            (
                render_pass,
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            ),
        )
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.draw_indexed(
            index_count, instance_count, first_index, base_vertex, first_instance
        )

    def draw_indirect(
        self, render_pass: RenderPassHandle, buffer: BufferHandle, offset: int
    ) -> None:
        self._record("draw_indirect", (render_pass, buffer, offset))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.draw_indirect(self._require_buffer(buffer), offset)

    def draw_indexed_indirect(
        self, render_pass: RenderPassHandle, buffer: BufferHandle, offset: int
    ) -> None:
        self._record("draw_indexed_indirect", (render_pass, buffer, offset))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.draw_indexed_indirect(self._require_buffer(buffer), offset)

    def set_viewport(
        self,
        render_pass: RenderPassHandle,
        x: float,
        y: float,
        w: float,
        h: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        self._record("set_viewport", (render_pass, x, y, w, h, min_depth, max_depth))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.set_viewport(x, y, w, h, min_depth, max_depth)

    def set_scissor_rect(
        self, render_pass: RenderPassHandle, x: int, y: int, w: int, h: int
    ) -> None:
        self._record("set_scissor_rect", (render_pass, x, y, w, h))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.set_scissor_rect(x, y, w, h)

    def set_blend_constant(
        self, render_pass: RenderPassHandle, color: tuple[float, float, float, float]
    ) -> None:
        self._record("set_blend_constant", (render_pass, color))
        wgpu_pass = self._require_render_pass(render_pass)
        wgpu_pass.set_blend_constant(color)

    def execute_render_bundles(
        self, render_pass: RenderPassHandle, bundles: list[RenderBundleHandle]
    ) -> None:
        del render_pass, bundles
        self._unsupported("execute_render_bundles")

    def write_timestamp(
        self,
        render_pass: RenderPassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None:
        del render_pass, query_set, query_index
        self._unsupported("write_timestamp")

    def end_render_pass(self, render_pass: RenderPassHandle) -> None:
        self._record("end_render_pass", (render_pass,))
        wgpu_pass = self._render_passes.pop(render_pass, None)
        if wgpu_pass is None:
            self._fail(f"unknown render pass {render_pass}")
        wgpu_pass.end()

    def create_render_bundle_encoder(
        self, device: DeviceHandle, descriptor: RenderBundleDescriptor
    ) -> RenderBundleEncoderHandle:
        del device, descriptor
        self._unsupported("create_render_bundle_encoder")

    def begin_compute_pass(
        self, encoder: CommandEncoderHandle, *, label: str | None = None
    ) -> ComputePassHandle:
        self._record("begin_compute_pass", (encoder, label))
        wgpu_encoder = self._require_encoder(encoder)
        wgpu_pass = wgpu_encoder.begin_compute_pass(label=label or "")
        handle = self._handle()
        self._compute_passes[handle] = wgpu_pass
        device = self._encoder_devices.get(encoder)
        if device is None:
            self._fail(f"unknown command encoder handle {encoder}")
        self._owners[handle] = device
        return handle

    def compute_set_pipeline(
        self, compute_pass: ComputePassHandle, pipeline: ComputePipelineHandle
    ) -> None:
        self._record("compute_set_pipeline", (compute_pass, pipeline))
        wgpu_pass = self._require_compute_pass(compute_pass)
        wgpu_pipeline = self._objects.get(pipeline)
        if wgpu_pipeline is None:
            self._fail(f"unknown pipeline handle {pipeline}")
        wgpu_pass.set_pipeline(wgpu_pipeline)

    def compute_set_bind_group(
        self, compute_pass: ComputePassHandle, index: int, group: BindGroupHandle
    ) -> None:
        self._record("compute_set_bind_group", (compute_pass, index, group))
        wgpu_pass = self._require_compute_pass(compute_pass)
        wgpu_group = self._objects.get(group)
        if wgpu_group is None:
            self._fail(f"unknown bind group handle {group}")
        wgpu_pass.set_bind_group(index, wgpu_group)

    def dispatch_workgroups(
        self, compute_pass: ComputePassHandle, x: int, y: int = 1, z: int = 1
    ) -> None:
        self._record("dispatch_workgroups", (compute_pass, x, y, z))
        wgpu_pass = self._require_compute_pass(compute_pass)
        wgpu_pass.dispatch_workgroups(x, y, z)

    def dispatch_workgroups_indirect(
        self, compute_pass: ComputePassHandle, buffer: BufferHandle, offset: int
    ) -> None:
        self._record("dispatch_workgroups_indirect", (compute_pass, buffer, offset))
        wgpu_pass = self._require_compute_pass(compute_pass)
        wgpu_pass.dispatch_workgroups_indirect(self._require_buffer(buffer), offset)

    def write_compute_timestamp(
        self,
        compute_pass: ComputePassHandle,
        query_set: QuerySetHandle,
        query_index: int,
    ) -> None:
        del compute_pass, query_set, query_index
        self._unsupported("write_compute_timestamp")

    def end_compute_pass(self, compute_pass: ComputePassHandle) -> None:
        self._record("end_compute_pass", (compute_pass,))
        wgpu_pass = self._compute_passes.pop(compute_pass, None)
        if wgpu_pass is None:
            self._fail(f"unknown compute pass {compute_pass}")
        wgpu_pass.end()

    def bundle_set_pipeline(
        self,
        encoder: RenderBundleEncoderHandle,
        pipeline: RenderPipelineHandle,
    ) -> None:
        del encoder, pipeline
        self._unsupported("bundle_set_pipeline")

    def bundle_set_bind_group(
        self,
        encoder: RenderBundleEncoderHandle,
        index: int,
        group: BindGroupHandle,
    ) -> None:
        del encoder, index, group
        self._unsupported("bundle_set_bind_group")

    def bundle_set_vertex_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        slot: int,
        buffer: BufferHandle,
        offset: int = 0,
    ) -> None:
        del encoder, slot, buffer, offset
        self._unsupported("bundle_set_vertex_buffer")

    def bundle_set_index_buffer(
        self,
        encoder: RenderBundleEncoderHandle,
        buffer: BufferHandle,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        del encoder, buffer, format, offset, size
        self._unsupported("bundle_set_index_buffer")

    def bundle_draw(
        self,
        encoder: RenderBundleEncoderHandle,
        vertex_count: int,
        instance_count: int = 1,
    ) -> None:
        del encoder, vertex_count, instance_count
        self._unsupported("bundle_draw")

    def bundle_draw_indexed(
        self,
        encoder: RenderBundleEncoderHandle,
        index_count: int,
        instance_count: int = 1,
    ) -> None:
        del encoder, index_count, instance_count
        self._unsupported("bundle_draw_indexed")

    def bundle_finish(self, encoder: RenderBundleEncoderHandle) -> RenderBundleHandle:
        del encoder
        self._unsupported("bundle_finish")

    def copy_texture_to_texture(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: TextureHandle,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self._record("copy_texture_to_texture", (encoder, src, dst, size))
        wgpu_encoder = self._require_encoder(encoder)
        src_handle = self._view_texture.get(src, src)
        dst_handle = self._view_texture.get(dst, dst)
        wgpu_src = self._require_texture(src_handle)
        wgpu_dst = self._require_texture(dst_handle)
        wgpu_encoder.copy_texture_to_texture(
            source={"texture": wgpu_src, "mip_level": 0, "origin": (0, 0, 0)},
            destination={"texture": wgpu_dst, "mip_level": 0, "origin": (0, 0, 0)},
            copy_size=(size[0], size[1], size[2]),
        )

    def copy_texture_to_buffer(
        self,
        encoder: CommandEncoderHandle,
        src: TextureHandle,
        dst: BufferHandle,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self._record("copy_texture_to_buffer", (encoder, src, dst, size))
        wgpu_encoder = self._require_encoder(encoder)
        texture_handle = self._view_texture.get(src, src)
        wgpu_texture = self._require_texture(texture_handle)
        wgpu_buffer = self._require_buffer(dst)
        bytes_per_row = (size[0] * _FORMAT_BPP[self._texture_formats[texture_handle]] + 255) & ~255
        wgpu_encoder.copy_texture_to_buffer(
            source={"texture": wgpu_texture, "mip_level": 0, "origin": (0, 0, 0)},
            destination={
                "buffer": wgpu_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": size[1],
            },
            copy_size=(size[0], size[1], size[2]),
        )

    def resolve_query_set(
        self,
        encoder: CommandEncoderHandle,
        query_set: QuerySetHandle,
        first_query: int,
        query_count: int,
        destination: BufferHandle,
        destination_offset: int,
    ) -> None:
        del encoder, query_set, first_query, query_count, destination, destination_offset
        self._unsupported("resolve_query_set")

    def readback_offscreen_rgba8(self, target: Any, width: int, height: int) -> bytes:
        """Copy an offscreen target back to a tight ``width * height * 4`` string."""
        self._record("readback_offscreen_rgba8", (target, width, height))
        view = self._offscreen_views.get(id(target))
        device = self._offscreen_devices.get(id(target))
        if view is None or device is None:
            self._fail("offscreen target has not been rendered to yet")
        from easygpu.buffer import BufferDescriptor
        from easygpu.constants import BufferUsage, MapMode

        bytes_per_row = (width * 4 + 255) & ~255
        staging = self.create_buffer(
            device,
            BufferDescriptor(
                size=bytes_per_row * height,
                usage=BufferUsage.COPY_DST | BufferUsage.COPY_SRC,
            ),
        )
        encoder = self.create_command_encoder(device)
        self.copy_texture_to_buffer(encoder, view, staging, size=(width, height, 1))
        self.submit(self.get_queue(device), [self.finish(encoder)])
        self.map_async(staging, MapMode.READ, 0, bytes_per_row * height)
        mapped = self.get_mapped_range(staging, 0, bytes_per_row * height)
        self.unmap(staging)
        self.destroy_buffer(staging)
        return b"".join(
            mapped[row * bytes_per_row : row * bytes_per_row + width * 4] for row in range(height)
        )

    def finish(self, encoder: CommandEncoderHandle) -> CommandBufferHandle:
        self._record("finish", (encoder,))
        wgpu_encoder = self._command_encoders.pop(encoder, None)
        device = self._encoder_devices.get(encoder)
        if wgpu_encoder is None:
            self._fail(f"unknown command encoder handle {encoder}")
        if device is None:
            self._fail(f"unknown command encoder handle {encoder}")
        wgpu_command_buffer = wgpu_encoder.finish()
        handle = self._handle()
        self._command_buffers[handle] = wgpu_command_buffer
        self._owners[handle] = device
        return handle

    # --- teardown ------------------------------------------------------

    def destroy_buffer(self, buffer: BufferHandle) -> None:
        self._record("destroy_buffer", (buffer,))
        wgpu_buffer = self._buffers.pop(buffer, None)
        self._buffer_shadow.pop(buffer, None)
        self._buffer_devices.pop(buffer, None)
        self._mapped.pop(buffer, None)
        if wgpu_buffer is not None:
            try:
                wgpu_buffer.destroy()
            except Exception:  # noqa: BLE001
                pass

    def destroy_texture(self, texture: TextureHandle) -> None:
        self._record("destroy_texture", (texture,))
        wgpu_texture = self._textures.pop(texture, None)
        self._objects.pop(texture, None)
        self._owners.pop(texture, None)
        self._texture_formats.pop(texture, None)
        self._texture_sizes.pop(texture, None)
        for view, owner in list(self._view_texture.items()):
            if owner == texture:
                self._view_texture.pop(view, None)
                self._objects.pop(view, None)
        if wgpu_texture is not None:
            try:
                wgpu_texture.destroy()
            except Exception:  # noqa: BLE001
                pass

    def destroy_texture_view(self, view: TextureViewHandle) -> None:
        self._record("destroy_texture_view", (view,))
        self._view_texture.pop(view, None)
        self._objects.pop(view, None)
        self._owners.pop(view, None)

    def destroy_sampler(self, sampler: SamplerHandle) -> None:
        self._record("destroy_sampler", (sampler,))
        self._objects.pop(sampler, None)
        self._owners.pop(sampler, None)

    def destroy_shader_module(self, module: ShaderModuleHandle) -> None:
        self._record("destroy_shader_module", (module,))
        self._objects.pop(module, None)
        self._owners.pop(module, None)

    def destroy_bind_group_layout(self, layout: BindGroupLayoutHandle) -> None:
        self._record("destroy_bind_group_layout", (layout,))
        self._objects.pop(layout, None)
        self._owners.pop(layout, None)

    def destroy_pipeline_layout(self, layout: PipelineLayoutHandle) -> None:
        self._record("destroy_pipeline_layout", (layout,))
        self._objects.pop(layout, None)
        self._owners.pop(layout, None)

    def destroy_bind_group(self, group: BindGroupHandle) -> None:
        self._record("destroy_bind_group", (group,))
        self._objects.pop(group, None)
        self._owners.pop(group, None)

    def destroy_render_pipeline(self, pipeline: RenderPipelineHandle) -> None:
        self._record("destroy_render_pipeline", (pipeline,))
        self._objects.pop(pipeline, None)
        self._owners.pop(pipeline, None)

    def destroy_compute_pipeline(self, pipeline: ComputePipelineHandle) -> None:
        self._record("destroy_compute_pipeline", (pipeline,))
        self._objects.pop(pipeline, None)
        self._owners.pop(pipeline, None)

    def destroy_query_set(self, query_set: QuerySetHandle) -> None:
        # Query sets are not creatable yet (see WGPU_BACKEND_UNSUPPORTED), so this
        # is a no-op: it keeps wrapper-side ``delete`` symmetric and future-proof.
        self._record("destroy_query_set", (query_set,))

    def destroy_command_encoder(self, encoder: CommandEncoderHandle) -> None:
        self._record("destroy_command_encoder", (encoder,))
        self._command_encoders.pop(encoder, None)
        self._encoder_devices.pop(encoder, None)

    def destroy_device(self, device: DeviceHandle) -> None:
        self._record("destroy_device", (device,))
        wgpu_device = self._devices.pop(device, None)
        if wgpu_device is None:
            return
        if self._adopted_device is wgpu_device:
            self._adopted_device = None
        if self._last_device is wgpu_device:
            self._last_device = None
        try:
            wgpu_device.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._purge_device(device)

    def _purge_device(self, device: DeviceHandle) -> None:
        """Drop every internal handle owned by ``device`` once it is destroyed.

        ``wgpu`` already invalidates the underlying resources when the device
        is destroyed, so this only releases EasyGPU-side references: otherwise
        ``requests`` and offscreen targets would keep the (useless) handles
        alive and leak across device lifecycles.
        """
        owned = {handle for handle, owner in self._owners.items() if owner == device}
        self._bind_group_layout_cache = {
            key: layout
            for key, layout in self._bind_group_layout_cache.items()
            if key[0] not in owned
        }
        for handle in owned:
            self._objects.pop(handle, None)
            self._owners.pop(handle, None)
            self._textures.pop(handle, None)
            self._texture_formats.pop(handle, None)
            self._texture_sizes.pop(handle, None)
            self._view_texture.pop(handle, None)
            self._render_passes.pop(handle, None)
            self._compute_passes.pop(handle, None)
            self._command_buffers.pop(handle, None)
        buffer_handles = [h for h, owner in self._buffer_devices.items() if owner == device]
        for handle in buffer_handles:
            self._buffers.pop(handle, None)
            self._buffer_shadow.pop(handle, None)
            self._mapped.pop(handle, None)
            self._buffer_devices.pop(handle, None)
        encoder_handles = [h for h, owner in self._encoder_devices.items() if owner == device]
        for handle in encoder_handles:
            self._command_encoders.pop(handle, None)
            self._encoder_devices.pop(handle, None)
        queue_handles = [h for h, owner in self._queue_devices.items() if owner == device]
        for handle in queue_handles:
            self._queues.pop(handle, None)
            self._queue_devices.pop(handle, None)
        for target, owner in list(self._offscreen_devices.items()):
            if owner == device:
                self._offscreen_views.pop(target, None)
                self._offscreen_devices.pop(target, None)


def configure_service(*, force_fallback_adapter: bool | None = None) -> WgpuPyGPU:
    """Build a :class:`WgpuPyGPU`` for ``easygpu.gpu.configure()`` (lazy wgpu import)."""
    return WgpuPyGPU(force_fallback_adapter=force_fallback_adapter)


__all__ = ["WgpuPyGPU", "configure_service"]
