"""FakeGPU conformance and recording behaviour."""

from __future__ import annotations

import inspect

import pytest

from easygpu.errors import GPUValidationError
from easygpu.fake_gpu import FakeGPU
from easygpu.gpu import GPU


def test_fake_gpu_satisfies_gpu_protocol() -> None:
    fake = FakeGPU()
    assert isinstance(fake, GPU)


def _protocol_method_names() -> set[str]:
    return {name for name, _ in inspect.getmembers(GPU, inspect.isfunction)}


def test_interface_has_no_drift() -> None:
    expected = _protocol_method_names()
    missing = expected - set(dir(FakeGPU()))
    assert not missing, f"FakeGPU missing protocol methods: {sorted(missing)}"


def test_calls_of_filters_by_name() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.draw(rp, 3)
    fake.draw(rp, 6, 2)
    fake.end(rp)
    assert fake.calls_of("draw") == [(rp, 3, 1), (rp, 6, 2)]
    assert fake.calls_of("end") == [(rp,)]


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


def test_end_twice_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.end(rp)
    with pytest.raises(GPUValidationError):
        fake.end(rp)


def test_render_pass_sequence_tracks_structured_calls() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    pass_a = fake.begin_render_pass(enc, [])
    pass_b = fake.begin_render_pass(enc, [])
    fake.draw(pass_a, 3)
    fake.set_pipeline(pass_a, 7)
    fake.draw(pass_b, 6, 2)
    fake.end(pass_a)
    assert fake.render_pass_sequence(pass_a) == ["draw", "set_pipeline", "end"]
    assert fake.render_pass_sequence(pass_b) == ["draw"]


def test_render_pass_sequence_unknown_returns_empty() -> None:
    fake = FakeGPU()
    assert fake.render_pass_sequence(999) == []


def test_draw_after_end_raises() -> None:
    fake = FakeGPU()
    enc = fake.create_command_encoder(1)
    rp = fake.begin_render_pass(enc, [])
    fake.end(rp)
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
