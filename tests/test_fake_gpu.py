"""FakeGPU conformance and recording behaviour."""

from __future__ import annotations

import inspect

import pytest

from easygpu.buffer import BufferDescriptor
from easygpu.constants import BufferUsage, MapMode, QueryType, TextureFormat, TextureUsage
from easygpu.errors import GPUValidationError
from easygpu.fake_gpu import FakeGPU
from easygpu.gpu import GPU
from easygpu.query import QuerySetDescriptor
from easygpu.texture import TextureDescriptor


def test_fake_gpu_satisfies_gpu_protocol() -> None:
    fake = FakeGPU()
    assert isinstance(fake, GPU)


def _protocol_method_names() -> set[str]:
    return {
        name for name, _ in inspect.getmembers(GPU, inspect.isfunction) if not name.startswith("_")
    }


def test_interface_has_no_drift() -> None:
    expected = _protocol_method_names()
    missing = expected - set(dir(FakeGPU()))
    assert not missing, f"FakeGPU missing protocol methods: {sorted(missing)}"


def test_fake_gpu_implements_full_protocol() -> None:
    for name in _protocol_method_names():
        assert hasattr(FakeGPU, name), name
    assert isinstance(FakeGPU(), GPU)


def test_calls_of_filters_by_name() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.draw(rp, 3)
    fake.draw(rp, 6, 2)
    fake.end_render_pass(rp)
    assert fake.calls_of("draw") == [(rp, 3, 1, 0, 0), (rp, 6, 2, 0, 0)]
    assert fake.calls_of("end_render_pass") == [(rp,)]


def test_handle_allocation_is_unique_and_increasing() -> None:
    fake = FakeGPU()
    a = fake.request_adapter()
    b = fake.request_adapter()
    c = fake.request_device(a)
    assert a != b and c != a and c != b


def test_finish_twice_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    fake.begin_render_pass(enc, [])
    fake.finish(enc)
    with pytest.raises(GPUValidationError):
        fake.finish(enc)


def test_begin_render_pass_after_finish_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    fake.finish(enc)
    with pytest.raises(GPUValidationError):
        fake.begin_render_pass(enc, [])


def test_end_render_pass_twice_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.end_render_pass(rp)
    with pytest.raises(GPUValidationError):
        fake.end_render_pass(rp)


def test_render_pass_sequence_tracks_structured_calls() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    pass_a = fake.begin_render_pass(enc, [])
    pass_b = fake.begin_render_pass(enc, [])
    fake.draw(pass_a, 3)
    fake.set_pipeline(pass_a, 7)
    fake.draw(pass_b, 6, 2)
    fake.end_render_pass(pass_a)
    assert fake.render_pass_sequence(pass_a) == ["draw", "set_pipeline", "end_render_pass"]
    assert fake.render_pass_sequence(pass_b) == ["draw"]


def test_render_pass_sequence_unknown_returns_empty() -> None:
    fake = FakeGPU()
    assert fake.render_pass_sequence(999) == []


def test_draw_after_end_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.end_render_pass(rp)
    with pytest.raises(GPUValidationError):
        fake.draw(rp, 3)


def test_destroy_device_cascades_to_its_resources() -> None:
    fake = FakeGPU()
    device = fake.request_device(fake.request_adapter())
    queue = fake.get_queue(device)
    buf = fake.create_buffer(device, object())
    mod = fake.create_shader_module(device, object())
    pipe = fake.create_render_pipeline(device, object())
    enc = fake.create_command_encoder(device)
    pass_id = fake.begin_render_pass(enc, [])
    command_buffer = fake.finish(enc)
    fake.submit(queue, [command_buffer])

    fake.destroy_device(device)

    assert device not in fake.devices
    assert queue not in fake.queues
    assert buf not in fake.buffers
    assert mod not in fake.shader_modules
    assert pipe not in fake.pipelines
    assert enc not in fake.encoders
    assert pass_id not in fake.render_passes
    assert command_buffer not in fake.command_buffers


def test_destroy_device_keeps_other_devices_resources() -> None:
    fake = FakeGPU()
    a = fake.request_device(fake.request_adapter())
    b = fake.request_device(fake.request_adapter())
    buf_a = fake.create_buffer(a, object())
    buf_b = fake.create_buffer(b, object())
    fake.destroy_device(a)
    assert buf_a not in fake.buffers
    assert buf_b in fake.buffers


def test_map_and_readback_roundtrip() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    q = fake.get_queue(d)
    b = fake.create_buffer(d, BufferDescriptor(size=8, usage=BufferUsage.MAP_READ))
    fake.write_buffer(q, b, 0, b"\x01\x02\x03\x04" * 2)
    fake.map_async(b, MapMode.READ, 0, 8)
    assert fake.mapped[b] == (MapMode.READ, 0, 8)
    assert fake.get_mapped_range(b, 2, 4) == b"\x03\x04\x01\x02"
    fake.unmap(b)
    assert b not in fake.mapped
    assert fake.calls_of("map_async") == [(b, MapMode.READ, 0, 8)]
    assert fake.calls_of("get_mapped_range") == [(b, 2, 4)]
    assert fake.calls_of("unmap") == [(b,)]


def test_get_mapped_range_unmapped_raises() -> None:
    fake = FakeGPU()
    b = fake.create_buffer(1, object())
    with pytest.raises(GPUValidationError):
        fake.get_mapped_range(b, 0, 4)


def test_copy_texture_to_buffer_contents() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    q = fake.get_queue(d)
    enc = fake.create_command_encoder(d)
    t = fake.create_texture(
        d,
        TextureDescriptor(
            size=(2, 1, 1), format=TextureFormat.RGBA8_UNORM, usage=TextureUsage.COPY_SRC
        ),
    )
    fake.write_texture(q, t, b"ABCDEFGH", width=2, height=1)
    dst = fake.create_buffer(d, BufferDescriptor(size=8, usage=BufferUsage.COPY_DST))
    fake.copy_texture_to_buffer(enc, t, dst, size=(2, 1, 1))
    assert fake.buffers[dst] == bytearray(b"ABCDEFGH")
    assert fake.calls_of("copy_texture_to_buffer") == [(enc, t, dst, (2, 1, 1))]


def test_copy_texture_to_texture_contents() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    q = fake.get_queue(d)
    enc = fake.create_command_encoder(d)
    src = fake.create_texture(
        d,
        TextureDescriptor(
            size=(2, 1, 1), format=TextureFormat.RGBA8_UNORM, usage=TextureUsage.COPY_SRC
        ),
    )
    dst = fake.create_texture(
        d,
        TextureDescriptor(
            size=(2, 1, 1), format=TextureFormat.RGBA8_UNORM, usage=TextureUsage.COPY_DST
        ),
    )
    fake.write_texture(q, src, b"ABCDEFGH", width=2, height=1)
    fake.copy_texture_to_texture(enc, src, dst, size=(2, 1, 1))
    assert fake.textures[dst].data[:8] == bytearray(b"ABCDEFGH")
    assert fake.calls_of("copy_texture_to_texture") == [(enc, src, dst, (2, 1, 1))]


def test_resolve_query_set_writes_zero_results_four_byte_le() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    enc = fake.create_command_encoder(d)
    qs = fake.create_query_set(d, QuerySetDescriptor(type=QueryType.TIMESTAMP, count=4))
    dest = fake.create_buffer(d, BufferDescriptor(size=16, usage=BufferUsage.QUERY_RESOLVE))
    fake.resolve_query_set(enc, qs, 0, 4, dest, 0)
    assert fake.buffers[dest] == bytearray(b"\x00\x00\x00\x00" * 4)
    assert fake.calls_of("resolve_query_set") == [(enc, qs, 0, 4, dest, 0)]
    assert fake.query_results[qs] == [0, 0, 0, 0]


def test_compute_pass_sequence() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    enc = fake.create_command_encoder(d)
    pipe = fake.create_compute_pipeline(d, object())
    group = fake.create_bind_group(d, object())
    buf = fake.create_buffer(d, object())
    qs = fake.create_query_set(d, QuerySetDescriptor(type=QueryType.TIMESTAMP, count=1))
    cp = fake.begin_compute_pass(enc, label="compute")
    fake.compute_set_pipeline(cp, pipe)
    fake.compute_set_bind_group(cp, 0, group)
    fake.dispatch_workgroups(cp, 8, 4, 2)
    fake.dispatch_workgroups_indirect(cp, buf, 16)
    fake.write_compute_timestamp(cp, qs, 0)
    fake.end_compute_pass(cp)
    assert fake.compute_pass_sequence(cp) == [
        "compute_set_pipeline",
        "compute_set_bind_group",
        "dispatch_workgroups",
        "dispatch_workgroups_indirect",
        "write_compute_timestamp",
        "end_compute_pass",
    ]
    assert fake.calls_of("begin_compute_pass") == [(enc, "compute")]
    assert fake.calls_of("end_compute_pass") == [(cp,)]


def test_compute_pass_after_finish_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    fake.finish(enc)
    with pytest.raises(GPUValidationError):
        fake.begin_compute_pass(enc)


def test_end_compute_pass_twice_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    cp = fake.begin_compute_pass(enc)
    fake.end_compute_pass(cp)
    with pytest.raises(GPUValidationError):
        fake.end_compute_pass(cp)


def test_render_bundle_record_and_finish() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    rbe = fake.create_render_bundle_encoder(d, object())
    pipe = fake.create_render_pipeline(d, object())
    group = fake.create_bind_group(d, object())
    vb = fake.create_buffer(d, object())
    ib = fake.create_buffer(d, object())
    fake.bundle_set_pipeline(rbe, pipe)
    fake.bundle_set_bind_group(rbe, 0, group)
    fake.bundle_set_vertex_buffer(rbe, 0, vb, 16)
    fake.bundle_set_index_buffer(rbe, ib, 1, 8, 12)
    fake.bundle_draw(rbe, 3)
    fake.bundle_draw_indexed(rbe, 6, 2)
    bundle = fake.bundle_finish(rbe)
    assert fake.bundles[bundle] == [
        ("bundle_set_pipeline", (pipe,)),
        ("bundle_set_bind_group", (0, group)),
        ("bundle_set_vertex_buffer", (0, vb, 16)),
        ("bundle_set_index_buffer", (ib, 1, 8, 12)),
        ("bundle_draw", (3, 1)),
        ("bundle_draw_indexed", (6, 2)),
    ]
    assert rbe not in fake.bundle_encoders
    assert fake.calls_of("bundle_finish") == [(rbe,)]
    assert bundle in fake.bundles


def test_destroy_device_cascades_compute_and_bundles() -> None:
    fake = FakeGPU()
    device = fake.request_device(fake.request_adapter())
    query = fake.create_query_set(device, QuerySetDescriptor(type=QueryType.TIMESTAMP, count=2))
    compute = fake.create_compute_pipeline(device, object())
    encoder = fake.create_command_encoder(device)
    bundle_encoder = fake.create_render_bundle_encoder(device, object())
    bundle = fake.bundle_finish(bundle_encoder)
    compute_pass = fake.begin_compute_pass(encoder)
    fake.destroy_device(device)
    assert query not in fake.query_sets
    assert query not in fake.query_results
    assert compute not in fake.compute_pipelines
    assert bundle_encoder not in fake.bundle_encoders
    assert bundle not in fake.bundles
    assert compute_pass not in fake.compute_passes


def test_get_bind_group_layout_cached_for_compute_pipeline() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    pipe = fake.create_compute_pipeline(d, object())
    layout1 = fake.get_bind_group_layout(pipe, 0)
    layout2 = fake.get_bind_group_layout(pipe, 0)
    layout3 = fake.get_bind_group_layout(pipe, 1)
    assert layout1 == layout2
    assert layout1 != layout3
    assert fake.calls_of("get_bind_group_layout") == [(pipe, 0), (pipe, 1)]


def test_generate_mipmap_records() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    t = fake.create_texture(d, object())
    fake.generate_mipmap(t)
    assert fake.calls_of("generate_mipmap") == [(t,)]


def test_get_adapter_info_records_and_returns() -> None:
    fake = FakeGPU()
    a = fake.request_adapter()
    info = fake.get_adapter_info(a)
    assert info == {
        "vendor": "fake",
        "architecture": "record",
        "device": "fake",
        "description": "FakeGPU",
    }
    assert fake.get_adapter_info(a) is info
    assert fake.adapter_info[a] is info
    assert fake.calls_of("get_adapter_info") == [(a,), (a,)]


def test_uncaptured_error_and_submitted_work_done_records() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    q = fake.get_queue(d)
    fake.uncaptured_errors[d] = "validation failure"
    assert fake.on_uncaptured_error(d) == "validation failure"
    assert fake.calls_of("on_uncaptured_error") == [(d,)]
    fake.on_submitted_work_done(q)
    assert fake.calls_of("on_submitted_work_done") == [(q,)]


def test_write_texture_records_full_shape() -> None:
    fake = FakeGPU()
    d = fake.request_device(fake.request_adapter())
    q = fake.get_queue(d)
    t = fake.create_texture(d, object())
    fake.write_texture(q, t, b"ABCD", width=2, height=2, mip_level=1, origin=(1, 2, 0))
    assert fake.calls_of("write_texture")[-1] == (q, t, b"ABCD", 2, 2, 1, (1, 2, 0))
    assert fake.textures[t].data[:4] == bytearray(b"ABCD")


def test_render_pass_extras_are_recorded() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    indirect = fake.create_buffer(1, object())
    qs = fake.create_query_set(1, QuerySetDescriptor(type=QueryType.TIMESTAMP, count=2))
    bundle_enc = fake.create_render_bundle_encoder(1, object())
    bundle = fake.bundle_finish(bundle_enc)
    fake.set_scissor_rect(rp, 0, 0, 100, 100)
    fake.draw_indirect(rp, indirect, 8)
    fake.draw_indexed_indirect(rp, indirect, 16)
    fake.execute_render_bundles(rp, [bundle])
    fake.write_timestamp(rp, qs, 0)
    fake.end_render_pass(rp)
    assert fake.render_pass_sequence(rp) == [
        "set_scissor_rect",
        "draw_indirect",
        "draw_indexed_indirect",
        "execute_render_bundles",
        "write_timestamp",
        "end_render_pass",
    ]
    assert fake.calls_of("set_scissor_rect") == [(rp, 0, 0, 100, 100)]
    assert fake.calls_of("draw_indirect") == [(rp, indirect, 8)]
    assert fake.calls_of("draw_indexed_indirect") == [(rp, indirect, 16)]
    assert fake.calls_of("execute_render_bundles") == [(rp, [bundle])]
    assert fake.calls_of("write_timestamp") == [(rp, qs, 0)]


def test_set_index_buffer_and_vertex_buffer_new_shapes() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    ib = fake.create_buffer(1, object())
    vb = fake.create_buffer(1, object())
    fake.set_index_buffer(rp, ib, 2)
    fake.set_vertex_buffer(rp, 0, vb, 16)
    assert fake.calls_of("set_index_buffer") == [(rp, ib, 2, 0, None)]
    assert fake.calls_of("set_vertex_buffer") == [(rp, 0, vb, 16)]
