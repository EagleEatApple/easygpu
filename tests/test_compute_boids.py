"""computeBoids example verified via FakeGPU."""

from __future__ import annotations

import struct

from easygpu.constants import BufferUsage
from easygpu.device import Device
from easygpu.fake_gpu import FakeGPU
from examples.compute_boids import (
    NUM_PARTICLES,
    NUM_WORKGROUPS,
    PARAMS_BYTE_SIZE,
    PARTICLE_STRIDE,
    encode_compute_boids,
)

SUBMIT_PHASE = [
    "begin_compute_pass",
    "compute_set_pipeline",
    "compute_set_bind_group",
    "dispatch_workgroups",
    "end_compute_pass",
    "begin_render_pass",
    "set_pipeline",
    "set_vertex_buffer",
    "set_vertex_buffer",
    "draw",
    "end_render_pass",
    "finish",
    "submit",
]

PARTICLE_USAGE = BufferUsage.VERTEX | BufferUsage.STORAGE | BufferUsage.COPY_DST


def test_compute_boids_submit_phase_sequence(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_compute_boids(device)

    names = [name for name, _ in fake_gpu.calls]
    start = names.index("create_command_encoder") + 1
    assert names[start:] == SUBMIT_PHASE


def test_compute_boids_compute_api_records(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_compute_boids(device)

    assert fake_gpu.calls_of("create_compute_pipeline")
    compute_handle = next(iter(fake_gpu.compute_pipelines))
    assert fake_gpu.calls_of("get_bind_group_layout") == [(compute_handle, 0)]

    pass_id = fake_gpu.calls_of("compute_set_pipeline")[0][0]
    assert (
        fake_gpu.calls_of("begin_compute_pass")
        and len(fake_gpu.calls_of("begin_compute_pass")) == 1
    )
    assert fake_gpu.calls_of("compute_set_pipeline") == [(pass_id, compute_handle)]
    assert fake_gpu.calls_of("compute_set_bind_group")[0][:2] == (pass_id, 0)
    assert fake_gpu.calls_of("dispatch_workgroups") == [(pass_id, NUM_WORKGROUPS, 1, 1)]
    assert fake_gpu.calls_of("end_compute_pass") == [(pass_id,)]


def test_compute_boids_draws_instanced_sprites(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_compute_boids(device)

    render_pass_id = fake_gpu.calls_of("set_pipeline")[0][0]
    assert fake_gpu.calls_of("draw")[-1] == (render_pass_id, 3, NUM_PARTICLES, 0, 0)

    sprite_handles = {
        handle
        for handle, descriptor in fake_gpu.buffer_descriptors.items()
        if getattr(descriptor, "label", None) == "sprite-vertices"
    }
    sprite_writes = [
        args for args in fake_gpu.calls_of("write_buffer") if args[1] in sprite_handles
    ]
    assert len(sprite_writes) == 1
    assert sprite_writes[0][3] == struct.pack("<6f", -0.01, -0.02, 0.01, -0.02, 0.0, 0.02)


def test_compute_boids_buffer_descriptors(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_compute_boids(device)

    descriptors = [args[1] for args in fake_gpu.calls_of("create_buffer")]
    particle_descriptors = [d for d in descriptors if d.usage == PARTICLE_USAGE]
    assert len(particle_descriptors) == 2
    assert all(d.size == NUM_PARTICLES * PARTICLE_STRIDE for d in particle_descriptors)
    params_descriptors = [d for d in descriptors if d.size == PARAMS_BYTE_SIZE]
    assert len(params_descriptors) == 1
    assert params_descriptors[0].usage == BufferUsage.UNIFORM | BufferUsage.COPY_DST


def test_compute_boids_deterministic_particle_seed(fake_gpu: FakeGPU) -> None:
    device = Device(id_=1)
    encode_compute_boids(device)

    payloads = [args[3] for args in fake_gpu.calls_of("write_buffer")]
    particle_payloads = [p for p in payloads if len(p) == NUM_PARTICLES * PARTICLE_STRIDE]
    assert len(particle_payloads) == 2
    expected_particle_zero = struct.pack("<4f", -1.0, -0.5, 0.0, 0.0)
    for payload in particle_payloads:
        assert payload[:16] == expected_particle_zero
