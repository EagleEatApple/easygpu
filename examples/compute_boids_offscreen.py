"""EasyGPU 0.2.0 offscreen computeBoids — one simulated step, headless.

Runs only the compute half of :mod:`examples.compute_boids`: one dispatch of
``updateSprites.wgsl`` over 1500 particles ping-ponged through two storage
buffers (the sprite render pass is skipped — there is no canvas). The output
buffer keeps ``COPY_SRC`` so it can be mapped back to bytes and compared with
the deterministic seed.
"""

from __future__ import annotations

import struct

from easygpu.bindgroup import BindGroupDescriptor, BindGroupEntry
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import BufferUsage
from easygpu.device import Device
from easygpu.pipeline import ComputePipelineDescriptor, ShaderStageEntry
from easygpu.shader import ShaderModule
from examples.boids_shaders import UPDATE_SPRITES_SOURCE
from examples.compute_boids import (
    NUM_WORKGROUPS,
    PARAMS_BYTE_SIZE,
    SIM_PARAMS,
    initial_particle_bytes,
)

PARTICLE_STRIDE = 16
PARTICLE_USAGE = BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC


def encode_compute_boids_offscreen(device: Device, *, workgroups: int = NUM_WORKGROUPS) -> Buffer:
    """Run one compute-boids step and return the output particle buffer.

    The returned buffer is the ``(frame + 1) % 2`` particle buffer, i.e. the
    result the sprite pass would have drawn. It carries ``COPY_SRC`` so a real
    backend can map it back for the output-change assertion.
    """
    seed = initial_particle_bytes()
    particle_buffers = [
        Buffer.from_data(device, seed, usage=PARTICLE_USAGE, label=f"particle-buffer-{i}")
        for i in range(2)
    ]

    params_buffer = device.create_buffer(
        BufferDescriptor(
            size=PARAMS_BYTE_SIZE,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="sim-params",
        )
    )
    device.queue.write_buffer(params_buffer, 0, struct.pack("<7f", *SIM_PARAMS))

    update_module = ShaderModule.from_wgsl(device, UPDATE_SPRITES_SOURCE, label="update-sprites")

    compute_pipeline = device.create_compute_pipeline(
        ComputePipelineDescriptor(
            compute=ShaderStageEntry(module=update_module, entry_point="main"),
            label="boids-compute-pipeline",
        )
    )
    compute_layout = compute_pipeline.get_bind_group_layout(0)
    bind_group = device.create_bind_group(
        BindGroupDescriptor(
            layout=compute_layout,
            entries=[
                BindGroupEntry(binding=0, resource=params_buffer),
                BindGroupEntry(binding=1, resource=particle_buffers[0]),
                BindGroupEntry(binding=2, resource=particle_buffers[1]),
            ],
            label="particle-bind-group",
        )
    )

    encoder = device.create_command_encoder(label="compute-boids-encoder")
    with encoder.begin_compute_pass() as compute_pass:
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(workgroups, 1, 1)
    device.queue.submit([encoder.finish()])

    return particle_buffers[1]


def read_particles(buffer: Buffer) -> bytes:
    """Synchronously map a storage buffer back to bytes (real backends only)."""
    from easygpu.constants import MapMode

    buffer.map_async(MapMode.READ)
    data = buffer.get_mapped_range()
    buffer.unmap()
    return data


def main() -> None:
    """Run one compute-boids step and check that the particles moved."""
    from easygpu.device import request_adapter
    from easygpu.gpu import configure
    from easygpu.wgpu_backend import WgpuPyGPU

    gpu = WgpuPyGPU()
    configure(gpu)
    device = request_adapter().request_device(label="compute-boids-offscreen-device")
    seed = initial_particle_bytes()
    output = encode_compute_boids_offscreen(device)
    data = read_particles(output)
    moved = sum(a != b for a, b in zip(seed, data, strict=True))
    print("initial bytes:", len(seed), "| changed bytes:", moved)


if __name__ == "__main__":
    main()
