"""WgpuPyGPU skeleton: structural conformance, lazy import, adapter/device/queue."""

from __future__ import annotations

import struct
import sys

import pytest

from easygpu.errors import GPUValidationError


def _device_gated() -> bool:
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync(force_fallback_adapter=True) is not None
    except Exception:  # noqa: BLE001
        return False


HAS_ADAPTER = _device_gated()


@pytest.fixture(scope="module")
def wgpu_device():
    if not HAS_ADAPTER:
        pytest.skip("no wgpu adapter available (install a driver or WARP/lavapipe)")
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
    return adapter.request_device_sync(label="wgpu-backend-test")


def test_wgpu_backend_is_a_gpu_implementation() -> None:
    from easygpu.gpu import GPU
    from easygpu.wgpu_backend import WgpuPyGPU

    assert isinstance(WgpuPyGPU(), GPU)


def test_import_does_not_load_wgpu() -> None:
    sys.modules.pop("wgpu", None)
    from easygpu import wgpu_backend  # noqa: F401

    assert "wgpu" not in sys.modules


def test_configure_service_returns_wgpu_backend() -> None:
    from easygpu.gpu import configure, get_gpu
    from easygpu.wgpu_backend import configure_service

    gpu = configure_service()
    configure(gpu)
    assert get_gpu() is gpu


def test_unimplemented_factories_fail_loudly() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    with pytest.raises(GPUValidationError, match="generate_mipmap not yet supported"):
        gpu.generate_mipmap(1)


_CAPABILITY_ARGS: dict[str, tuple[object, ...]] = {
    "generate_mipmap": (1,),
    "create_query_set": (1, None),
    "execute_render_bundles": (1, []),
    "write_timestamp": (1, 1, 0),
    "create_render_bundle_encoder": (1, None),
    "write_compute_timestamp": (1, 1, 0),
    "bundle_set_pipeline": (1, 1),
    "bundle_set_bind_group": (1, 0, 1),
    "bundle_set_vertex_buffer": (1, 0, 1),
    "bundle_set_index_buffer": (1, 1, 1),
    "bundle_draw": (1, 3),
    "bundle_draw_indexed": (1, 3),
    "bundle_finish": (1,),
    "resolve_query_set": (1, 1, 0, 1, 1, 0),
}


def test_wgpu_capability_marker_claims_existing_protocol_methods() -> None:
    from easygpu.gpu import GPU, WGPU_BACKEND_UNSUPPORTED
    from easygpu.wgpu_backend import WgpuPyGPU

    protocol_names = set(dir(GPU))
    assert WGPU_BACKEND_UNSUPPORTED <= protocol_names
    for name in sorted(WGPU_BACKEND_UNSUPPORTED):
        assert name in _CAPABILITY_ARGS, f"missing test invocation for {name}"
        assert callable(getattr(WgpuPyGPU, name, None)), f"missing WgpuPyGPU.{name}"


@pytest.mark.parametrize("name", sorted(_CAPABILITY_ARGS))
def test_wgpu_capability_marker_methods_raise(name: str) -> None:
    from easygpu.gpu import WGPU_BACKEND_UNSUPPORTED
    from easygpu.wgpu_backend import WgpuPyGPU

    assert name in WGPU_BACKEND_UNSUPPORTED
    gpu = WgpuPyGPU()
    with pytest.raises(GPUValidationError, match=f"{name} not yet supported by WgpuPyGPU"):
        getattr(gpu, name)(*_CAPABILITY_ARGS[name])


def test_wgpu_destroy_query_set_and_command_encoder_are_idempotent_noops() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    gpu.destroy_query_set(7)
    gpu.destroy_command_encoder(7)
    gpu.destroy_query_set(7)
    assert [name for name, _ in gpu.calls] == [
        "destroy_query_set",
        "destroy_command_encoder",
        "destroy_query_set",
    ]


def test_request_adapter_records_call() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    try:
        gpu.request_adapter(force_fallback_adapter=True)
    except GPUValidationError:
        pass
    assert gpu.calls and gpu.calls[0][0] == "request_adapter"


def test_on_submitted_work_done_is_a_noop() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    assert gpu.on_submitted_work_done(1) is None
    assert gpu.calls == [("on_submitted_work_done", (1,))]


def test_on_uncaptured_error_is_none() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    assert gpu.on_uncaptured_error(1) is None


def test_underlying_device_raises_before_any_device() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    with pytest.raises(GPUValidationError, match="no wgpu device yet"):
        assert gpu.underlying_device


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_real_adapter_device_queue_chain(wgpu_device: object) -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    del wgpu_device
    gpu = WgpuPyGPU()
    adapter = gpu.request_adapter()
    device = gpu.request_device(adapter, label="real")
    queue = gpu.get_queue(device)
    assert isinstance(adapter, int)
    assert isinstance(device, int)
    assert isinstance(queue, int)
    info = gpu.get_adapter_info(adapter)
    assert set(info) == {"vendor", "architecture", "device", "description"}
    assert all(isinstance(v, str) for v in info.values())
    assert gpu.underlying_device is not None


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_from_wgpu_device_adoption(wgpu_device: object) -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU.from_wgpu_device(wgpu_device)
    assert gpu.underlying_device is wgpu_device
    adapter = gpu.request_adapter()
    device = gpu.request_device(adapter)
    queue = gpu.get_queue(device)
    assert isinstance(device, int)
    assert isinstance(queue, int)
    info = gpu.get_adapter_info(adapter)
    assert set(info) == {"vendor", "architecture", "device", "description"}
    gpu.destroy_device(device)


def test_destroy_device_for_unknown_handle_is_noop() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    gpu.destroy_device(12345)
    assert gpu.calls == [("destroy_device", (12345,))]


def test_wgpu_buffer_api_shape_is_synchronous() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    for name in (
        "create_buffer",
        "write_buffer",
        "map_async",
        "get_mapped_range",
        "unmap",
        "destroy_buffer",
    ):
        assert hasattr(gpu, name)


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_buffer_roundtrip_through_queue_and_readback() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage, MapMode
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)
    b = gpu.create_buffer(
        d,
        BufferDescriptor(size=16, usage=BufferUsage.COPY_SRC | BufferUsage.COPY_DST),
    )
    payload = b"0123456789abcdef"
    gpu.write_buffer(q, b, 0, payload)
    assert bytes(gpu._buffer_shadow[b]) == payload  # noqa: SLF001 — parity with FakeGPU
    gpu.map_async(b, int(MapMode.READ), 0, 16)
    assert gpu.get_mapped_range(b, 0, 16) == payload  # real readback over WARP
    gpu.unmap(b)
    gpu.destroy_buffer(b)
    assert [name for name, _ in gpu.calls] == [
        "request_adapter",
        "request_device",
        "get_queue",
        "create_buffer",
        "write_buffer",
        "map_async",
        "get_mapped_range",
        "unmap",
        "destroy_buffer",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_get_mapped_range_unmapped_raises() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage
    from easygpu.errors import GPUValidationError
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)
    b = gpu.create_buffer(
        d,
        BufferDescriptor(size=4, usage=BufferUsage.COPY_SRC | BufferUsage.COPY_DST),
    )
    gpu.write_buffer(q, b, 0, b"abcd")
    with pytest.raises(GPUValidationError, match="not mapped"):
        gpu.get_mapped_range(b, 0, 4)


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_render_pipeline_creation_with_auto_layout() -> None:
    from easygpu.constants import TextureFormat
    from easygpu.pipeline import ColorTargetState, RenderPipelineDescriptor, ShaderStageEntry
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    code = (
        "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f "
        "{ return vec4f(0.0, 0.0, 0.0, 1.0); }\n"
        "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n"
    )
    module = gpu.create_shader_module(d, ShaderModuleDescriptor(code=code))
    pipeline = gpu.create_render_pipeline(
        d,
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=module, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=module, entry_point="fs"),
            fragment_targets=[ColorTargetState(format=TextureFormat.RGBA8_UNORM)],
        ),
    )
    assert isinstance(pipeline, int)
    assert [name for name, _ in gpu.calls][2:] == [
        "create_shader_module",
        "create_render_pipeline",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_compute_pipeline_creation_with_auto_layout() -> None:
    from easygpu.pipeline import ComputePipelineDescriptor, ShaderStageEntry
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    code = "@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) gid: vec3u) {}\n"
    module = gpu.create_shader_module(d, ShaderModuleDescriptor(code=code))
    pipeline = gpu.create_compute_pipeline(
        d, ComputePipelineDescriptor(compute=ShaderStageEntry(module=module))
    )
    assert isinstance(pipeline, int)
    assert [name for name, _ in gpu.calls][2:] == [
        "create_shader_module",
        "create_compute_pipeline",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_get_bind_group_layout_cached_and_recorded_once() -> None:
    from easygpu.constants import TextureFormat
    from easygpu.pipeline import ColorTargetState, RenderPipelineDescriptor, ShaderStageEntry
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    code = (
        "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f "
        "{ return vec4f(0.0, 0.0, 0.0, 1.0); }\n"
        "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n"
    )
    module = gpu.create_shader_module(d, ShaderModuleDescriptor(code=code))
    pipeline = gpu.create_render_pipeline(
        d,
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=module, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=module, entry_point="fs"),
            fragment_targets=[ColorTargetState(format=TextureFormat.RGBA8_UNORM)],
        ),
    )
    layout_a = gpu.get_bind_group_layout(pipeline, 0)
    layout_b = gpu.get_bind_group_layout(pipeline, 0)
    assert layout_a == layout_b
    assert isinstance(layout_a, int)
    assert [name for name, _ in gpu.calls].count("get_bind_group_layout") == 1


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_texture_and_view_creation() -> None:
    from easygpu.constants import TextureFormat, TextureUsage
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    tex = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(4, 4, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
        ),
    )
    view = gpu.create_texture_view(tex, None)
    assert isinstance(tex, int)
    assert isinstance(view, int)
    assert [name for name, _ in gpu.calls][2:] == [
        "create_texture",
        "create_texture_view",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_sampler_and_bind_group_creation() -> None:
    from easygpu.bindgroup import (
        BindGroupDescriptor,
        BindGroupEntry,
        BindGroupLayoutDescriptor,
        BindGroupLayoutEntry,
        PipelineLayoutDescriptor,
    )
    from easygpu.constants import SamplerBindingType, ShaderStage
    from easygpu.sampler import SamplerDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    bgl = gpu.create_bind_group_layout(
        d,
        BindGroupLayoutDescriptor(
            entries=[
                BindGroupLayoutEntry(
                    binding=0,
                    visibility=ShaderStage.FRAGMENT,
                    buffer=None,
                    sampler=SamplerBindingType.FILTERING,
                    texture=None,
                )
            ]
        ),
    )
    pgl = gpu.create_pipeline_layout(d, PipelineLayoutDescriptor(bind_group_layouts=[bgl]))
    sampler = gpu.create_sampler(d, SamplerDescriptor())
    group = gpu.create_bind_group(
        d,
        BindGroupDescriptor(
            layout=bgl,
            entries=[BindGroupEntry(binding=0, resource=sampler)],
        ),
    )
    for handle in (bgl, pgl, sampler, group):
        assert isinstance(handle, int)
    assert [name for name, _ in gpu.calls][2:] == [
        "create_bind_group_layout",
        "create_pipeline_layout",
        "create_sampler",
        "create_bind_group",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_write_texture_records_and_does_not_raise() -> None:
    from easygpu.constants import TextureFormat, TextureUsage
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)
    tex = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(2, 2, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.COPY_DST | TextureUsage.COPY_SRC,
        ),
    )
    payload = bytes(range(16))
    gpu.write_texture(q, tex, payload, width=2, height=2)
    assert gpu.calls[-1] == (
        "write_texture",
        (q, tex, payload, 2, 2, 0, (0, 0, 0)),
    )


def test_wgpu_destroy_texture_view_sampler_layouts_record_and_noop() -> None:
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    gpu.destroy_texture_view(1)
    gpu.destroy_sampler(1)
    gpu.destroy_bind_group_layout(1)
    gpu.destroy_pipeline_layout(1)
    gpu.destroy_bind_group(1)
    gpu.destroy_shader_module(1)
    gpu.destroy_render_pipeline(1)
    gpu.destroy_compute_pipeline(1)
    assert [name for name, _ in gpu.calls] == [
        "destroy_texture_view",
        "destroy_sampler",
        "destroy_bind_group_layout",
        "destroy_pipeline_layout",
        "destroy_bind_group",
        "destroy_shader_module",
        "destroy_render_pipeline",
        "destroy_compute_pipeline",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_render_to_texture_submit_and_readback() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage, TextureFormat, TextureUsage, VertexFormat
    from easygpu.pipeline import (
        ColorTargetState,
        RenderPipelineDescriptor,
        ShaderStageEntry,
        VertexAttribute,
        VertexBufferLayout,
    )
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)

    width, height = 256, 256
    positions = b"".join(
        struct.pack("<2f", *pos) for pos in ((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))
    )
    vertex_buffer = gpu.create_buffer(
        d,
        BufferDescriptor(size=len(positions), usage=BufferUsage.VERTEX | BufferUsage.COPY_DST),
    )
    gpu.write_buffer(q, vertex_buffer, 0, positions)

    code = (
        "@vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f "
        "{ return vec4f(pos, 0.0, 1.0); }\n"
        "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.0, 0.0, 1.0); }\n"
    )
    module = gpu.create_shader_module(d, ShaderModuleDescriptor(code=code))
    pipeline = gpu.create_render_pipeline(
        d,
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=module, entry_point="vs"),
            fragment_stage=ShaderStageEntry(module=module, entry_point="fs"),
            vertex_buffers=[
                VertexBufferLayout(
                    array_stride=8,
                    attributes=[
                        VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X2, offset=0)
                    ],
                )
            ],
            fragment_targets=[ColorTargetState(format=TextureFormat.RGBA8_UNORM)],
        ),
    )

    from easygpu.encoder import RenderPassColorAttachment

    texture = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(width, height, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
        ),
    )
    view = gpu.create_texture_view(texture, None)
    readback = gpu.create_buffer(
        d,
        BufferDescriptor(
            size=width * height * 4, usage=BufferUsage.COPY_SRC | BufferUsage.COPY_DST
        ),
    )

    encoder = gpu.create_command_encoder(d)
    render_pass = gpu.begin_render_pass(
        encoder,
        [RenderPassColorAttachment(view=view, clear_value=(0.0, 0.0, 0.0, 0.0))],
    )
    gpu.set_pipeline(render_pass, pipeline)
    gpu.set_vertex_buffer(render_pass, 0, vertex_buffer, 0)
    gpu.draw(render_pass, 3)
    gpu.end_render_pass(render_pass)
    gpu.copy_texture_to_buffer(encoder, view, readback, size=(width, height, 1))
    command_buffer = gpu.finish(encoder)
    gpu.submit(q, [command_buffer])

    gpu.map_async(readback, 0, 0, width * height * 4)
    data = gpu.get_mapped_range(readback, 0, width * height * 4)
    gpu.unmap(readback)
    assert len(data) == width * height * 4
    center = data[(height // 2) * width * 4 + (width // 2) * 4 :]
    assert center[0] > 200, "expected red center pixel"
    assert center[1] < 60
    assert center[2] < 60

    assert [name for name, _ in gpu.calls] == [
        "request_adapter",
        "request_device",
        "get_queue",
        "create_buffer",
        "write_buffer",
        "create_shader_module",
        "create_render_pipeline",
        "create_texture",
        "create_texture_view",
        "create_buffer",
        "create_command_encoder",
        "begin_render_pass",
        "set_pipeline",
        "set_vertex_buffer",
        "draw",
        "end_render_pass",
        "copy_texture_to_buffer",
        "finish",
        "submit",
        "map_async",
        "get_mapped_range",
        "unmap",
    ]


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_compute_dispatch_and_storage_readback() -> None:
    from easygpu.bindgroup import BindGroupDescriptor, BindGroupEntry
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage
    from easygpu.pipeline import ComputePipelineDescriptor, ShaderStageEntry
    from easygpu.shader import ShaderModuleDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)

    storage = gpu.create_buffer(
        d, BufferDescriptor(size=16, usage=BufferUsage.STORAGE | BufferUsage.COPY_SRC)
    )
    code = (
        "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n"
        "@compute @workgroup_size(1)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3u) {\n"
        "    data[gid.x] = 0xABCDEF01u;\n"
        "}\n"
    )
    module = gpu.create_shader_module(d, ShaderModuleDescriptor(code=code))
    pipeline = gpu.create_compute_pipeline(
        d, ComputePipelineDescriptor(compute=ShaderStageEntry(module=module))
    )
    bgl = gpu.get_bind_group_layout(pipeline, 0)
    group = gpu.create_bind_group(
        d, BindGroupDescriptor(layout=bgl, entries=[BindGroupEntry(binding=0, resource=storage)])
    )

    encoder = gpu.create_command_encoder(d)
    compute_pass = gpu.begin_compute_pass(encoder)
    gpu.compute_set_pipeline(compute_pass, pipeline)
    gpu.compute_set_bind_group(compute_pass, 0, group)
    gpu.dispatch_workgroups(compute_pass, 4)
    gpu.end_compute_pass(compute_pass)
    gpu.submit(q, [gpu.finish(encoder)])

    gpu.map_async(storage, 0, 0, 16)
    data = gpu.get_mapped_range(storage, 0, 16)
    gpu.unmap(storage)
    assert data == bytes([0x01, 0xEF, 0xCD, 0xAB]) * 4

    assert [name for name, _ in gpu.calls] == [
        "request_adapter",
        "request_device",
        "get_queue",
        "create_buffer",
        "create_shader_module",
        "create_compute_pipeline",
        "get_bind_group_layout",
        "create_bind_group",
        "create_command_encoder",
        "begin_compute_pass",
        "compute_set_pipeline",
        "compute_set_bind_group",
        "dispatch_workgroups",
        "end_compute_pass",
        "finish",
        "submit",
        "map_async",
        "get_mapped_range",
        "unmap",
    ]


def test_trim_depth_stencil_keys_keeps_only_present_aspects() -> None:
    from easygpu.wgpu_backend import _trim_depth_stencil_keys

    class _StubView:
        def __init__(self, format: str) -> None:
            self.texture = type("T", (), {"format": format})()

    depth = {
        "view": 1,
        "depth_load_op": "clear",
        "depth_store_op": "store",
        "depth_read_only": False,
        "stencil_clear_value": 0,
        "stencil_load_op": "clear",
        "stencil_store_op": "store",
        "stencil_read_only": False,
    }
    combined = _trim_depth_stencil_keys(dict(depth), _StubView("depth24plus-stencil8"))
    assert combined["depth_load_op"] == "clear"
    assert combined["stencil_load_op"] == "clear"
    depth_only = _trim_depth_stencil_keys(dict(depth), _StubView("depth32float"))
    assert depth_only["depth_load_op"] == "clear"
    assert "stencil_load_op" not in depth_only
    assert "stencil_store_op" not in depth_only
    read_only = _trim_depth_stencil_keys(
        dict(depth, depth_read_only=True), _StubView("depth32float")
    )
    assert "depth_load_op" not in read_only
    assert read_only is not depth
    assert _trim_depth_stencil_keys(depth, None) is depth


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_render_pass_with_stencil_combined_attachment(wgpu_device: object) -> None:
    del wgpu_device
    from easygpu.constants import (
        LoadOp,
        StoreOp,
        TextureFormat,
        TextureUsage,
    )
    from easygpu.encoder import RenderPassColorAttachment, RenderPassDepthStencilAttachment
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    depth_tex = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(4, 4, 1),
            format=TextureFormat.DEPTH24PLUS_STENCIL8,
            usage=TextureUsage.RENDER_ATTACHMENT,
        ),
    )
    depth_view = gpu.create_texture_view(depth_tex)
    color_tex = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(4, 4, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT,
        ),
    )
    color_view = gpu.create_texture_view(color_tex)
    encoder = gpu.create_command_encoder(d)
    render_pass = gpu.begin_render_pass(
        encoder,
        [
            RenderPassColorAttachment(
                view=color_view,
                clear_value=(0.0, 0.0, 0.0, 1.0),
                load_op=LoadOp.CLEAR,
                store_op=StoreOp.STORE,
            )
        ],
        depth_stencil_attachment=RenderPassDepthStencilAttachment(
            view=depth_view,
            depth_load_op=LoadOp.CLEAR,
            depth_store_op=StoreOp.STORE,
            stencil_load_op=LoadOp.CLEAR,
            stencil_store_op=StoreOp.STORE,
        ),
    )
    gpu.end_render_pass(render_pass)
    gpu.destroy_command_encoder(encoder)
    assert gpu.calls[-1] == ("destroy_command_encoder", (encoder,))


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_write_buffer_rejects_cross_device_queue() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage
    from easygpu.errors import GPUValidationError
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    adapter = gpu.request_adapter(force_fallback_adapter=True)
    d1 = gpu.request_device(adapter, label="d1")
    d2 = gpu.request_device(adapter, label="d2")
    q2 = gpu.get_queue(d2)
    buffer = gpu.create_buffer(d1, BufferDescriptor(size=4, usage=BufferUsage.COPY_DST))
    with pytest.raises(GPUValidationError, match="different devices"):
        gpu.write_buffer(q2, buffer, 0, b"abcd")


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_destroy_device_purges_internal_state() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage, TextureFormat, TextureUsage
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)
    gpu.create_buffer(d, BufferDescriptor(size=4, usage=BufferUsage.COPY_SRC))
    tex = gpu.create_texture(
        d,
        TextureDescriptor(
            size=(2, 2, 1),
            format=TextureFormat.RGBA8_UNORM,
            usage=TextureUsage.RENDER_ATTACHMENT,
        ),
    )
    gpu.create_texture_view(tex)
    gpu.destroy_device(d)
    assert gpu._devices == {}  # noqa: SLF001
    assert gpu._queues == {}  # noqa: SLF001
    assert gpu._buffers == {}  # noqa: SLF001
    assert gpu._buffer_shadow == {}  # noqa: SLF001
    assert gpu._mapped == {}  # noqa: SLF001
    assert gpu._textures == {}  # noqa: SLF001
    assert gpu._view_texture == {}  # noqa: SLF001
    assert gpu._owners == {}  # noqa: SLF001
    assert gpu._queue_devices == {}  # noqa: SLF001
    assert gpu._buffer_devices == {}  # noqa: SLF001
    assert q not in gpu._queues  # noqa: SLF001


@pytest.mark.skipif(not HAS_ADAPTER, reason="no wgpu adapter available")
def test_wgpu_copy_texture_to_texture_content() -> None:
    from easygpu.buffer import BufferDescriptor
    from easygpu.constants import BufferUsage, TextureFormat, TextureUsage
    from easygpu.texture import TextureDescriptor
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    d = gpu.request_device(gpu.request_adapter(force_fallback_adapter=True))
    q = gpu.get_queue(d)

    def make_texture() -> int:
        return gpu.create_texture(
            d,
            TextureDescriptor(
                size=(4, 4, 1),
                format=TextureFormat.RGBA8_UNORM,
                usage=TextureUsage.COPY_DST | TextureUsage.COPY_SRC,
            ),
        )

    src, dst = make_texture(), make_texture()
    payload = bytes(range(64))
    gpu.write_texture(q, src, payload, width=4, height=4)

    encoder = gpu.create_command_encoder(d)
    gpu.copy_texture_to_texture(encoder, src, dst, size=(4, 4, 1))
    gpu.submit(q, [gpu.finish(encoder)])

    # wgpu requires the destination layout's bytes_per_row to be a multiple of
    # 256, so the staging buffer is padded per row; reconstruct the tight rows.
    bytes_per_row = 4 * 4
    padded_row = (bytes_per_row + 255) & ~255
    staging = gpu.create_buffer(
        d, BufferDescriptor(size=padded_row * 4, usage=BufferUsage.COPY_SRC | BufferUsage.COPY_DST)
    )
    encoder = gpu.create_command_encoder(d)
    gpu.copy_texture_to_buffer(encoder, dst, staging, size=(4, 4, 1))
    gpu.submit(q, [gpu.finish(encoder)])
    gpu.map_async(staging, 0, 0, padded_row * 4)
    readback = b"".join(
        gpu.get_mapped_range(staging, row * padded_row, bytes_per_row) for row in range(4)
    )
    assert readback == payload
    gpu.unmap(staging)
