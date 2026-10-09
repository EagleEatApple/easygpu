"""Offscreen compute boids: FakeGPU recording vs WgpuPyGPU output change."""

from __future__ import annotations

import pytest

from easygpu.device import request_adapter
from easygpu.gpu import GPU
from examples.compute_boids import initial_particle_bytes
from examples.compute_boids_offscreen import (
    encode_compute_boids_offscreen,
    read_particles,
)


@pytest.mark.offscreen
def test_compute_boids_offscreen_dual_engine(fake_gpu: GPU) -> None:
    device = request_adapter().request_device(label="boids")
    encode_compute_boids_offscreen(device, workgroups=4)

    pass_commands = [
        name
        for name, _ in fake_gpu.calls
        if name
        in (
            "compute_set_pipeline",
            "compute_set_bind_group",
            "dispatch_workgroups",
            "end_compute_pass",
        )
    ]
    assert pass_commands == [
        "compute_set_pipeline",
        "compute_set_bind_group",
        "dispatch_workgroups",
        "end_compute_pass",
    ]
    dispatch = fake_gpu.calls_of("dispatch_workgroups")
    assert dispatch == [(dispatch[0][0], 4, 1, 1)]

    present = {name for name, _ in fake_gpu.calls}
    assert {
        "create_compute_pipeline",
        "get_bind_group_layout",
        "create_bind_group",
        "begin_compute_pass",
    } <= present


@pytest.mark.offscreen
@pytest.mark.wgpu
def test_compute_boids_offscreen_wgpu_output_change(wgpu_gpu: GPU) -> None:
    device = request_adapter().request_device(label="boids")
    seed = initial_particle_bytes()
    output = encode_compute_boids_offscreen(device)

    data = read_particles(output)
    assert len(data) == len(seed)
    assert data != seed, "compute step must move at least one particle"
    assert any(a != b for a, b in zip(seed, data, strict=True))
