"""In-memory recording fake of the :class:`easygpu.gpu.GPU` protocol.

Ships with the package so users can TDD their command-encoding logic without a
GPU (also used by the test-suite). Every call is recorded in
:attr:`FakeGPU.calls` (list of ``(name, args)``) so tests assert the exact
command-encoding sequence. Mirrors gl46's FakeGL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from easygpu.errors import GPUValidationError


@dataclass
class EncoderState:
    finished: bool = False
    passes: list[int] | None = None


class FakeGPU:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.adapters: set[int] = set()
        self.devices: set[int] = set()
        self.queues: dict[int, int] = {}  # queue -> device
        self.buffers: dict[int, bytearray] = {}
        self.shader_modules: dict[int, str] = {}  # id -> wgsl code
        self.pipelines: dict[int, object] = {}
        self.encoders: dict[int, EncoderState] = {}
        self.render_passes: dict[int, list[tuple[str, tuple[Any, ...]]]] = {}
        self._ended_passes: set[int] = set()
        self.command_buffers: list[int] = []
        self.submitted: list[list[int]] = []
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

    # -- resource creation ---------------------------------------------
    def create_buffer(self, device: int, descriptor: object) -> int:
        handle = self._next_handle()
        self.calls.append(("create_buffer", (device, descriptor)))
        size = int(getattr(descriptor, "size", 0))
        self.buffers[handle] = bytearray(size)
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

    def create_command_encoder(self, device: int, *, label: str | None = None) -> int:
        handle = self._next_handle()
        self.calls.append(("create_command_encoder", (device, label)))
        self.encoders[handle] = EncoderState()
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

    def submit(self, queue: int, command_buffers: list[int]) -> None:
        self.calls.append(("submit", (queue, list(command_buffers))))
        self.submitted.append(list(command_buffers))

    # -- command encoding ----------------------------------------------
    def begin_render_pass(self, encoder: int, color_attachments: list[object]) -> int:
        state = self.encoders.get(encoder)
        if state is None:
            state = EncoderState()
            self.encoders[encoder] = state
        if state.finished:
            raise GPUValidationError("encoder already finished")
        handle = self._next_handle()
        self.calls.append(("begin_render_pass", (encoder, list(color_attachments))))
        self.render_passes[handle] = []
        if state.passes is None:
            state.passes = []
        state.passes.append(handle)
        return handle

    def set_pipeline(self, render_pass: int, pipeline: int) -> None:
        if render_pass not in self.render_passes or render_pass in self._ended_passes:
            raise GPUValidationError(f"unknown render pass {render_pass}")
        self.calls.append(("set_pipeline", (render_pass, pipeline)))
        self.render_passes[render_pass].append(("set_pipeline", (pipeline,)))

    def set_vertex_buffer(self, render_pass: int, slot: int, buffer: int) -> None:
        if render_pass not in self.render_passes or render_pass in self._ended_passes:
            raise GPUValidationError(f"unknown render pass {render_pass}")
        self.calls.append(("set_vertex_buffer", (render_pass, slot, buffer)))
        self.render_passes[render_pass].append(("set_vertex_buffer", (slot, buffer)))

    def draw(self, render_pass: int, vertex_count: int, instance_count: int = 1) -> None:
        if render_pass not in self.render_passes or render_pass in self._ended_passes:
            raise GPUValidationError(f"unknown render pass {render_pass}")
        self.calls.append(("draw", (render_pass, vertex_count, instance_count)))
        self.render_passes[render_pass].append(("draw", (vertex_count, instance_count)))

    def end(self, render_pass: int) -> None:
        if render_pass not in self.render_passes or render_pass in self._ended_passes:
            raise GPUValidationError(f"unknown render pass {render_pass}")
        self.calls.append(("end", (render_pass,)))
        self.render_passes[render_pass].append(("end", ()))
        self._ended_passes.add(render_pass)

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
        self._owners.pop(buffer, None)

    def destroy_shader_module(self, module: int) -> None:
        self.calls.append(("destroy_shader_module", (module,)))
        self.shader_modules.pop(module, None)
        self._owners.pop(module, None)

    def destroy_render_pipeline(self, pipeline: int) -> None:
        self.calls.append(("destroy_render_pipeline", (pipeline,)))
        self.pipelines.pop(pipeline, None)
        self._owners.pop(pipeline, None)

    def destroy_command_encoder(self, encoder: int) -> None:
        self.calls.append(("destroy_command_encoder", (encoder,)))
        state = self.encoders.pop(encoder, None)
        self._owners.pop(encoder, None)
        if state is not None and state.passes is not None:
            for pass_ in state.passes:
                self.render_passes.pop(pass_, None)
                self._ended_passes.discard(pass_)

    def destroy_device(self, device: int) -> None:
        self.calls.append(("destroy_device", (device,)))
        owned = [handle for handle, owner in self._owners.items() if owner == device]
        for handle in owned:
            self._owners.pop(handle, None)
            if handle in self.buffers:
                self.buffers.pop(handle, None)
            elif handle in self.shader_modules:
                self.shader_modules.pop(handle, None)
            elif handle in self.pipelines:
                self.pipelines.pop(handle, None)
            elif handle in self.command_buffers:
                self.command_buffers.remove(handle)
            elif handle in self.encoders:
                state = self.encoders.pop(handle, None)
                if state is not None and state.passes is not None:
                    for pass_ in state.passes:
                        self.render_passes.pop(pass_, None)
                        self._ended_passes.discard(pass_)
        for queue in [q for q, owner in self.queues.items() if owner == device]:
            self.queues.pop(queue, None)
        self.devices.discard(device)


__all__ = ["FakeGPU"]
