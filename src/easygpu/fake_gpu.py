"""In-memory recording fake of the :class:`easygpu.gpu.GPU` protocol.

Ships with the package so users can TDD their command-encoding logic without a
GPU (also used by the test-suite). Every call is recorded in
:attr:`FakeGPU.calls` (list of ``(name, args)``) so tests assert the exact
command-encoding sequence. Mirrors gl46's FakeGL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from easygpu.errors import GPUValidationError


@dataclass
class EncoderState:
    finished: bool = False
    passes: list[int] | None = None


@dataclass
class TextureState:
    width: int
    height: int
    depth_or_array_layers: int
    format: object
    usage: object
    sample_count: int
    mip_level_count: int
    data: bytearray = field(default_factory=bytearray)
    views: set[int] = field(default_factory=set)


class FakeGPU:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.adapters: set[int] = set()
        self.devices: set[int] = set()
        self.queues: dict[int, int] = {}  # queue -> device
        self.buffers: dict[int, bytearray] = {}
        self.buffer_descriptors: dict[int, object] = {}  # buffer -> descriptor (label lookup)
        self.shader_modules: dict[int, str] = {}  # id -> wgsl code
        self.pipelines: dict[int, object] = {}
        self.compute_pipelines: dict[int, object] = {}
        self.textures: dict[int, TextureState] = {}
        self.texture_views: dict[int, int] = {}  # view id -> texture id
        self.samplers: dict[int, object] = {}
        self.bind_group_layouts: dict[int, object] = {}
        self.pipeline_layouts: dict[int, object] = {}
        self.bind_groups: dict[int, object] = {}
        self.query_sets: dict[int, object] = {}
        self.pipeline_bind_group_layouts: dict[tuple[int, int], int] = {}
        self.encoders: dict[int, EncoderState] = {}
        self.render_passes: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
        self._ended_passes: set[int] = set()
        self.compute_passes: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
        self._ended_compute_passes: set[int] = set()
        self.bundle_encoders: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
        self.bundles: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
        self.command_buffers: list[int] = []
        self.submitted: list[list[int]] = []
        self.mapped: dict[int, tuple[int, int, int]] = {}  # buffer -> (mode, offset, size)
        self.query_results: dict[int, list[int]] = {}  # query set -> list of result ints
        self.adapter_info: dict[int, dict[str, object]] = {}
        self.uncaptured_errors: dict[int, str | None] = {}
        self._owners: dict[int, int] = {}  # resource handle -> owning device handle
        self._next = 1

    # -- recording helpers ---------------------------------------------
    def _next_handle(self) -> int:
        handle = self._next
        self._next += 1
        return handle

    def _record(self, name: str, args: tuple[Any, ...]) -> None:
        self.calls.append((name, args))

    def calls_of(self, name: str) -> list[tuple[Any, ...]]:
        return [args for n, args in self.calls if n == name]

    def render_pass_sequence(self, render_pass: int) -> list[str]:
        return [name for name, _ in self.render_passes.get(render_pass, [])]

    def compute_pass_sequence(self, compute_pass: int) -> list[str]:
        return [name for name, _ in self.compute_passes.get(compute_pass, [])]

    # -- adapter / device / queue --------------------------------------
    def request_adapter(self, *, force_fallback_adapter: bool = False) -> int:
        handle = self._next_handle()
        self.calls.append(("request_adapter", (force_fallback_adapter,)))
        self.adapters.add(handle)
        return handle

    def request_device(self, adapter: int, *, label: str | None = None) -> int:
        handle = self._next_handle()
        self.calls.append(("request_device", (adapter, label)))
        self.devices.add(handle)
        return handle

    def get_queue(self, device: int) -> int:
        if device in self.queues:
            return self.queues[device]
        handle = self._next_handle()
        self.calls.append(("get_queue", (device,)))
        self.queues[device] = handle
        return handle

    def get_adapter_info(self, adapter: int) -> dict[str, object]:
        self.calls.append(("get_adapter_info", (adapter,)))
        info = self.adapter_info.get(adapter)
        if info is None:
            info = {
                "vendor": "fake",
                "architecture": "record",
                "device": "fake",
                "description": "FakeGPU",
            }
            self.adapter_info[adapter] = info
        return info

    def on_uncaptured_error(self, device: int) -> str | None:
        self.calls.append(("on_uncaptured_error", (device,)))
        return self.uncaptured_errors.get(device)

    def on_submitted_work_done(self, queue: int) -> None:
        self.calls.append(("on_submitted_work_done", (queue,)))

    # -- resource creation ---------------------------------------------
    def create_buffer(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_buffer", (device, descriptor)))
        size = int(getattr(descriptor, "size", 0))
        self.buffers[handle] = bytearray(size)
        self.buffer_descriptors[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_shader_module(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_shader_module", (device, descriptor)))
        self.shader_modules[handle] = str(getattr(descriptor, "code", ""))
        self._owners[handle] = device
        return handle

    def create_render_pipeline(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_render_pipeline", (device, descriptor)))
        self.pipelines[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_compute_pipeline(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_compute_pipeline", (device, descriptor)))
        self.compute_pipelines[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_texture(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_texture", (device, descriptor)))
        size = tuple(getattr(descriptor, "size", (0, 0, 1)))[:3]
        state = TextureState(
            width=int(size[0]),
            height=int(size[1]),
            depth_or_array_layers=int(size[2]) if len(size) > 2 else 1,
            format=getattr(descriptor, "format", 0),
            usage=getattr(descriptor, "usage", 0),
            sample_count=int(getattr(descriptor, "sample_count", 1)),
            mip_level_count=int(getattr(descriptor, "mip_level_count", 1)),
            data=bytearray(int(size[0]) * int(size[1]) * 4),
        )
        self.textures[handle] = state
        self._owners[handle] = device
        return handle

    def create_texture_view(self, texture: int, descriptor: object = None) -> int:
        handle = self._next_handle()
        self.calls.append(("create_texture_view", (texture, descriptor)))
        self.texture_views[handle] = texture
        state = self.textures.get(texture)
        if state is None:
            self.textures[texture] = TextureState(0, 0, 1, 0, 0, 1, 1)
            state = self.textures[texture]
        state.views.add(handle)
        return handle

    def generate_mipmap(self, texture: int) -> None:
        self.calls.append(("generate_mipmap", (texture,)))

    def create_sampler(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_sampler", (device, descriptor)))
        self.samplers[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_bind_group_layout(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_bind_group_layout", (device, descriptor)))
        self.bind_group_layouts[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_pipeline_layout(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_pipeline_layout", (device, descriptor)))
        self.pipeline_layouts[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_bind_group(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_bind_group", (device, descriptor)))
        self.bind_groups[handle] = descriptor
        self._owners[handle] = device
        return handle

    def create_query_set(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_query_set", (device, descriptor)))
        self.query_sets[handle] = descriptor
        self.query_results[handle] = [0] * int(getattr(descriptor, "count", 0))
        self._owners[handle] = device
        return handle

    def get_bind_group_layout(self, pipeline: int, group_index: int) -> int:
        key = (pipeline, group_index)
        if key not in self.pipeline_bind_group_layouts:
            handle = self._next_handle()
            self.calls.append(("get_bind_group_layout", (pipeline, group_index)))
            self.pipeline_bind_group_layouts[key] = handle
            self.bind_group_layouts[handle] = None
            self._owners[handle] = self._owners.get(pipeline, -1)
            return handle
        return self.pipeline_bind_group_layouts[key]

    def create_command_encoder(self, device: int, *, label: str | None = None) -> int:
        handle = self._next_handle()
        self.calls.append(("create_command_encoder", (device, label)))
        self.encoders[handle] = EncoderState()
        self._owners[handle] = device
        return handle

    def create_render_bundle_encoder(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_render_bundle_encoder", (device, descriptor)))
        self.bundle_encoders[handle] = []
        self._owners[handle] = device
        return handle

    # -- queue data path -----------------------------------------------
    def write_buffer(self, queue: int, buffer: int, buffer_offset: int, data: bytes) -> None:
        self.calls.append(("write_buffer", (queue, buffer, buffer_offset, data)))
        store = self.buffers.setdefault(buffer, bytearray())
        need = buffer_offset + len(data)
        if need > len(store):
            store.extend(b"\x00" * (need - len(store)))
        store[buffer_offset:need] = data

    def map_async(self, buffer: int, mode: int, offset: int, size: int) -> None:
        self.calls.append(("map_async", (buffer, mode, offset, size)))
        self.mapped[buffer] = (mode, offset, size)

    def get_mapped_range(self, buffer: int, offset: int, size: int) -> bytes:
        self.calls.append(("get_mapped_range", (buffer, offset, size)))
        if buffer not in self.mapped:
            raise GPUValidationError("buffer not mapped")
        return bytes(self.buffers[buffer][offset : offset + size])

    def unmap(self, buffer: int) -> None:
        self.calls.append(("unmap", (buffer,)))
        self.mapped.pop(buffer, None)

    def write_texture(
        self,
        queue: int,
        texture: int,
        data: bytes,
        *,
        width: int,
        height: int,
        mip_level: int = 0,
        origin: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.calls.append(
            ("write_texture", (queue, texture, data, width, height, mip_level, origin))
        )
        store = self.textures.setdefault(texture, TextureState(width, height, 1, 0, 0, 1, 1))
        store.width = width
        store.height = height
        need = width * height * 4
        if len(store.data) < need:
            store.data.extend(b"\x00" * (need - len(store.data)))
        store.data[: min(len(data), need)] = data[:need]

    def submit(self, queue: int, command_buffers: list[int]) -> None:
        self.calls.append(("submit", (queue, list(command_buffers))))
        self.submitted.append(list(command_buffers))

    # -- command encoding ----------------------------------------------
    def begin_render_pass(
        self,
        encoder: int,
        color_attachments: list[object],
        depth_stencil_attachment: object | None = None,
    ) -> int:
        state = self.encoders.get(encoder)
        if state is None:
            state = EncoderState()
            self.encoders[encoder] = state
        if state.finished:
            raise GPUValidationError("encoder already finished")
        handle = self._next_handle()
        self.calls.append(
            ("begin_render_pass", (encoder, list(color_attachments), depth_stencil_attachment))
        )
        self.render_passes[handle] = []
        if state.passes is None:
            state.passes = []
        state.passes.append(handle)
        return handle

    def begin_compute_pass(self, encoder: int, *, label: str | None = None) -> int:
        state = self.encoders.get(encoder)
        if state is None:
            state = EncoderState()
            self.encoders[encoder] = state
        if state.finished:
            raise GPUValidationError("encoder already finished")
        handle = self._next_handle()
        self.calls.append(("begin_compute_pass", (encoder, label)))
        self.compute_passes[handle] = []
        if state.passes is None:
            state.passes = []
        state.passes.append(handle)
        return handle

    def _require_pass(self, render_pass: int) -> list[tuple[str, tuple[Any, ...]]]:
        if render_pass not in self.render_passes or render_pass in self._ended_passes:
            raise GPUValidationError(f"unknown render pass {render_pass}")
        return self.render_passes[render_pass]

    def _require_compute_pass(self, compute_pass: int) -> list[tuple[str, tuple[Any, ...]]]:
        if compute_pass not in self.compute_passes or compute_pass in self._ended_compute_passes:
            raise GPUValidationError(f"unknown compute pass {compute_pass}")
        return self.compute_passes[compute_pass]

    def _require_bundle_encoder(self, encoder: int) -> list[tuple[str, tuple[Any, ...]]]:
        if encoder not in self.bundle_encoders:
            raise GPUValidationError(f"unknown render bundle encoder {encoder}")
        return self.bundle_encoders[encoder]

    def set_pipeline(self, render_pass: int, pipeline: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_pipeline", (render_pass, pipeline)))
        events.append(("set_pipeline", (pipeline,)))

    def set_vertex_buffer(self, render_pass: int, slot: int, buffer: int, offset: int = 0) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_vertex_buffer", (render_pass, slot, buffer, offset)))
        events.append(("set_vertex_buffer", (slot, buffer, offset)))

    def set_index_buffer(
        self,
        render_pass: int,
        buffer: int,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_index_buffer", (render_pass, buffer, format, offset, size)))
        events.append(("set_index_buffer", (buffer, format, offset, size)))

    def set_bind_group(self, render_pass: int, group_index: int, bind_group: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_bind_group", (render_pass, group_index, bind_group)))
        events.append(("set_bind_group", (group_index, bind_group)))

    def set_viewport(
        self,
        render_pass: int,
        x: float,
        y: float,
        w: float,
        h: float,
        min_depth: float,
        max_depth: float,
    ) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(
            (
                "set_viewport",
                (render_pass, x, y, w, h, min_depth, max_depth),
            )
        )
        events.append(("set_viewport", (x, y, w, h, min_depth, max_depth)))

    def set_scissor_rect(self, render_pass: int, x: int, y: int, w: int, h: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_scissor_rect", (render_pass, x, y, w, h)))
        events.append(("set_scissor_rect", (x, y, w, h)))

    def set_blend_constant(
        self, render_pass: int, color: tuple[float, float, float, float]
    ) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("set_blend_constant", (render_pass, color)))
        events.append(("set_blend_constant", (color,)))

    def draw(
        self,
        render_pass: int,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(
            ("draw", (render_pass, vertex_count, instance_count, first_vertex, first_instance))
        )
        events.append(("draw", (vertex_count, instance_count, first_vertex, first_instance)))

    def draw_indexed(
        self,
        render_pass: int,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        base_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(
            (
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
        )
        events.append(
            (
                "draw_indexed",
                (index_count, instance_count, first_index, base_vertex, first_instance),
            )
        )

    def draw_indirect(self, render_pass: int, buffer: int, offset: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("draw_indirect", (render_pass, buffer, offset)))
        events.append(("draw_indirect", (buffer, offset)))

    def draw_indexed_indirect(self, render_pass: int, buffer: int, offset: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("draw_indexed_indirect", (render_pass, buffer, offset)))
        events.append(("draw_indexed_indirect", (buffer, offset)))

    def execute_render_bundles(self, render_pass: int, bundles: list[int]) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("execute_render_bundles", (render_pass, list(bundles))))
        events.append(("execute_render_bundles", (list(bundles),)))

    def write_timestamp(self, render_pass: int, query_set: int, query_index: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("write_timestamp", (render_pass, query_set, query_index)))
        events.append(("write_timestamp", (query_set, query_index)))

    def end_render_pass(self, render_pass: int) -> None:
        events = self._require_pass(render_pass)
        self.calls.append(("end_render_pass", (render_pass,)))
        events.append(("end_render_pass", ()))
        self._ended_passes.add(render_pass)

    def compute_set_pipeline(self, compute_pass: int, pipeline: int) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("compute_set_pipeline", (compute_pass, pipeline)))
        events.append(("compute_set_pipeline", (pipeline,)))

    def compute_set_bind_group(self, compute_pass: int, index: int, group: int) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("compute_set_bind_group", (compute_pass, index, group)))
        events.append(("compute_set_bind_group", (index, group)))

    def dispatch_workgroups(self, compute_pass: int, x: int, y: int = 1, z: int = 1) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("dispatch_workgroups", (compute_pass, x, y, z)))
        events.append(("dispatch_workgroups", (x, y, z)))

    def dispatch_workgroups_indirect(self, compute_pass: int, buffer: int, offset: int) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("dispatch_workgroups_indirect", (compute_pass, buffer, offset)))
        events.append(("dispatch_workgroups_indirect", (buffer, offset)))

    def write_compute_timestamp(self, compute_pass: int, query_set: int, query_index: int) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("write_compute_timestamp", (compute_pass, query_set, query_index)))
        events.append(("write_compute_timestamp", (query_set, query_index)))

    def end_compute_pass(self, compute_pass: int) -> None:
        events = self._require_compute_pass(compute_pass)
        self.calls.append(("end_compute_pass", (compute_pass,)))
        events.append(("end_compute_pass", ()))
        self._ended_compute_passes.add(compute_pass)

    def bundle_set_pipeline(self, encoder: int, pipeline: int) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_set_pipeline", (encoder, pipeline)))
        events.append(("bundle_set_pipeline", (pipeline,)))

    def bundle_set_bind_group(self, encoder: int, index: int, group: int) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_set_bind_group", (encoder, index, group)))
        events.append(("bundle_set_bind_group", (index, group)))

    def bundle_set_vertex_buffer(
        self, encoder: int, slot: int, buffer: int, offset: int = 0
    ) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_set_vertex_buffer", (encoder, slot, buffer, offset)))
        events.append(("bundle_set_vertex_buffer", (slot, buffer, offset)))

    def bundle_set_index_buffer(
        self,
        encoder: int,
        buffer: int,
        format: int,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_set_index_buffer", (encoder, buffer, format, offset, size)))
        events.append(("bundle_set_index_buffer", (buffer, format, offset, size)))

    def bundle_draw(self, encoder: int, vertex_count: int, instance_count: int = 1) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_draw", (encoder, vertex_count, instance_count)))
        events.append(("bundle_draw", (vertex_count, instance_count)))

    def bundle_draw_indexed(self, encoder: int, index_count: int, instance_count: int = 1) -> None:
        events = self._require_bundle_encoder(encoder)
        self.calls.append(("bundle_draw_indexed", (encoder, index_count, instance_count)))
        events.append(("bundle_draw_indexed", (index_count, instance_count)))

    def bundle_finish(self, encoder: int) -> int:
        events = self._require_bundle_encoder(encoder)
        handle = self._next_handle()
        self.calls.append(("bundle_finish", (encoder,)))
        self.bundles[handle] = list(events)
        events.clear()
        self.bundle_encoders.pop(encoder, None)
        self._owners[handle] = self._owners.get(encoder, -1)
        return handle

    def copy_texture_to_texture(
        self,
        encoder: int,
        src: int,
        dst: int,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self.calls.append(("copy_texture_to_texture", (encoder, src, dst, size)))
        src_state = self.textures.setdefault(src, TextureState(0, 0, 1, 0, 0, 1, 1))
        dst_state = self.textures.setdefault(dst, TextureState(0, 0, 1, 0, 0, 1, 1))
        nbytes = size[0] * size[1] * size[2] * 4
        if len(dst_state.data) < nbytes:
            dst_state.data.extend(b"\x00" * (nbytes - len(dst_state.data)))
        dst_state.data[:nbytes] = src_state.data[:nbytes]

    def copy_texture_to_buffer(
        self,
        encoder: int,
        src: int,
        dst: int,
        *,
        size: tuple[int, int, int],
    ) -> None:
        self.calls.append(("copy_texture_to_buffer", (encoder, src, dst, size)))
        src_state = self.textures.setdefault(src, TextureState(0, 0, 1, 0, 0, 1, 1))
        nbytes = size[0] * size[1] * 4
        store = self.buffers.setdefault(dst, bytearray())
        if len(store) < nbytes:
            store.extend(b"\x00" * (nbytes - len(store)))
        store[:nbytes] = src_state.data[:nbytes]

    def resolve_query_set(
        self,
        encoder: int,
        query_set: int,
        first_query: int,
        query_count: int,
        destination: int,
        destination_offset: int,
    ) -> None:
        self.calls.append(
            (
                "resolve_query_set",
                (encoder, query_set, first_query, query_count, destination, destination_offset),
            )
        )
        results = self.query_results.setdefault(query_set, [])
        data = b"".join(
            int(v).to_bytes(4, "little") for v in results[first_query : first_query + query_count]
        )
        store = self.buffers.setdefault(destination, bytearray())
        start = destination_offset * 4
        need = start + len(data)
        if need > len(store):
            store.extend(b"\x00" * (need - len(store)))
        store[start:need] = data

    def finish(self, encoder: int) -> int:
        state = self.encoders.get(encoder)
        if state is None:
            state = EncoderState()
            self.encoders[encoder] = state
        if state.finished:
            raise GPUValidationError("encoder already finished")
        handle = self._next_handle()
        self.calls.append(("finish", (encoder,)))
        state.finished = True
        self.command_buffers.append(handle)
        self._owners[handle] = self._owners.get(encoder, -1)
        return handle

    # -- teardown ------------------------------------------------------
    def destroy_buffer(self, buffer: int) -> None:
        self.calls.append(("destroy_buffer", (buffer,)))
        self.buffers.pop(buffer, None)
        self.buffer_descriptors.pop(buffer, None)
        self.mapped.pop(buffer, None)
        self._owners.pop(buffer, None)

    def destroy_shader_module(self, module: int) -> None:
        self.calls.append(("destroy_shader_module", (module,)))
        self.shader_modules.pop(module, None)
        self._owners.pop(module, None)

    def destroy_render_pipeline(self, pipeline: int) -> None:
        self.calls.append(("destroy_render_pipeline", (pipeline,)))
        self.pipelines.pop(pipeline, None)
        self._owners.pop(pipeline, None)

    def destroy_compute_pipeline(self, pipeline: int) -> None:
        self.calls.append(("destroy_compute_pipeline", (pipeline,)))
        self.compute_pipelines.pop(pipeline, None)
        self._owners.pop(pipeline, None)

    def destroy_texture(self, texture: int) -> None:
        self.calls.append(("destroy_texture", (texture,)))
        self.textures.pop(texture, None)
        for view, owner in list(self.texture_views.items()):
            if owner == texture:
                self.texture_views.pop(view, None)
        self._owners.pop(texture, None)

    def destroy_texture_view(self, view: int) -> None:
        self.calls.append(("destroy_texture_view", (view,)))
        owner = self.texture_views.pop(view, None)
        if owner is not None:
            state = self.textures.get(owner)
            if state is not None:
                state.views.discard(view)

    def destroy_sampler(self, sampler: int) -> None:
        self.calls.append(("destroy_sampler", (sampler,)))
        self.samplers.pop(sampler, None)
        self._owners.pop(sampler, None)

    def destroy_bind_group_layout(self, layout: int) -> None:
        self.calls.append(("destroy_bind_group_layout", (layout,)))
        self.bind_group_layouts.pop(layout, None)
        self._owners.pop(layout, None)

    def destroy_pipeline_layout(self, layout: int) -> None:
        self.calls.append(("destroy_pipeline_layout", (layout,)))
        self.pipeline_layouts.pop(layout, None)
        self._owners.pop(layout, None)

    def destroy_bind_group(self, bind_group: int) -> None:
        self.calls.append(("destroy_bind_group", (bind_group,)))
        self.bind_groups.pop(bind_group, None)
        self._owners.pop(bind_group, None)

    def destroy_query_set(self, query_set: int) -> None:
        self.calls.append(("destroy_query_set", (query_set,)))
        self.query_sets.pop(query_set, None)
        self.query_results.pop(query_set, None)
        self._owners.pop(query_set, None)

    def destroy_command_encoder(self, encoder: int) -> None:
        self.calls.append(("destroy_command_encoder", (encoder,)))
        state = self.encoders.pop(encoder, None)
        self._owners.pop(encoder, None)
        if state is not None and state.passes is not None:
            for pass_ in state.passes:
                self.render_passes.pop(pass_, None)
                self._ended_passes.discard(pass_)
                self.compute_passes.pop(pass_, None)
                self._ended_compute_passes.discard(pass_)

    def destroy_device(self, device: int) -> None:
        self.calls.append(("destroy_device", (device,)))
        owned = [handle for handle, owner in self._owners.items() if owner == device]
        for handle in owned:
            self._owners.pop(handle, None)
            if handle in self.buffers:
                self.buffers.pop(handle, None)
                self.buffer_descriptors.pop(handle, None)
                self.mapped.pop(handle, None)
            elif handle in self.shader_modules:
                self.shader_modules.pop(handle, None)
            elif handle in self.pipelines:
                self.pipelines.pop(handle, None)
            elif handle in self.compute_pipelines:
                self.compute_pipelines.pop(handle, None)
            elif handle in self.textures:
                self.textures.pop(handle, None)
            elif handle in self.samplers:
                self.samplers.pop(handle, None)
            elif handle in self.bind_group_layouts:
                self.bind_group_layouts.pop(handle, None)
            elif handle in self.pipeline_layouts:
                self.pipeline_layouts.pop(handle, None)
            elif handle in self.bind_groups:
                self.bind_groups.pop(handle, None)
            elif handle in self.query_sets:
                self.query_sets.pop(handle, None)
                self.query_results.pop(handle, None)
            elif handle in self.bundles:
                self.bundles.pop(handle, None)
            elif handle in self.bundle_encoders:
                self.bundle_encoders.pop(handle, None)
            elif handle in self.command_buffers:
                self.command_buffers.remove(handle)
            elif handle in self.encoders:
                state = self.encoders.pop(handle, None)
                if state is not None and state.passes is not None:
                    for pass_ in state.passes:
                        self.render_passes.pop(pass_, None)
                        self._ended_passes.discard(pass_)
                        self.compute_passes.pop(pass_, None)
                        self._ended_compute_passes.discard(pass_)
        for queue in [q for q, owner in self.queues.items() if owner == device]:
            self.queues.pop(queue, None)
        self.devices.discard(device)


__all__ = ["FakeGPU"]
