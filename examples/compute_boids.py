"""EasyGPU computeBoids — 1500 boid particles simulated on the GPU.

A faithful port of ``webgpu-samples/sample/computeBoids``: one compute pass
runs ``updateSprites.wgsl`` over a storage buffer (ping-ponged between two
particle buffers) and one render pass draws ``sprite.wgsl`` as 1500 instanced
three-vertex sprites. A single frame is encoded, matching the sample's per-frame
shape.

Differences from the browser sample, by design:

* Timestamp-query timing and the ``timestamp-query`` feature are omitted
  entirely (the sample measures both passes and displays an average); this port
  performs no readback and no occlusion queries.
* Particle positions come from a deterministic seed instead of ``Math.random``:
  particle ``i`` starts at ``(2 * (i / 1500 - 0.5), 2 * ((i * 7) % 1500) / 1500
  - 0.5, 0.0, 0.0)``.
* This port initializes the particle buffers through ``write_buffer``, so they
  carry ``VERTEX | STORAGE | COPY_DST``; ``COPY_DST`` is required by that
  initialization path. The browser sample instead uses only ``VERTEX | STORAGE``
  and fills the buffers with ``mappedAtCreation``.
* The canvas is not modeled headlessly, so the color attachment binds view handle
  ``0`` and clears to the sample's black.

No GPU is needed: ``encode_compute_boids`` records every call on a
:class:`easygpu.fake_gpu.FakeGPU`. Test with
``uv run pytest tests/test_compute_boids.py``.
"""

from __future__ import annotations

import struct

from easygpu.bindgroup import BindGroupDescriptor, BindGroupEntry
from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import BufferUsage, TextureFormat, VertexFormat
from easygpu.device import Device
from easygpu.encoder import RenderPassColorAttachment
from easygpu.pipeline import (
    ColorTargetState,
    ComputePipelineDescriptor,
    RenderPipelineDescriptor,
    ShaderStageEntry,
    VertexAttribute,
    VertexBufferLayout,
)
from easygpu.shader import ShaderModule
from examples.boids_shaders import SPRITE_SOURCE, UPDATE_SPRITES_SOURCE

NUM_PARTICLES = 1500
PARTICLE_STRIDE = 16
PARAMS_FLOAT_COUNT = 7
PARAMS_BYTE_SIZE = PARAMS_FLOAT_COUNT * 4
WORKGROUP_SIZE = 64
NUM_WORKGROUPS = (NUM_PARTICLES + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
PARTICLE_USAGE = BufferUsage.VERTEX | BufferUsage.STORAGE | BufferUsage.COPY_DST
CLEAR_COLOR = (0.0, 0.0, 0.0, 1.0)
SPRITE_VERTICES = [-0.01, -0.02, 0.01, -0.02, 0.0, 0.02]
SIM_PARAMS = (0.04, 0.1, 0.025, 0.025, 0.02, 0.05, 0.005)


def initial_particle_bytes() -> bytes:
    """Return the deterministic seed for every particle (four floats each)."""
    values: list[float] = []
    for i in range(NUM_PARTICLES):
        values.extend(
            (
                2.0 * (i / NUM_PARTICLES - 0.5),
                2.0 * ((i * 7) % NUM_PARTICLES) / NUM_PARTICLES - 0.5,
                0.0,
                0.0,
            )
        )
    return struct.pack(f"<{len(values)}f", *values)


def encode_compute_boids(device: Device) -> None:
    """Create both pipelines and encode+submit one compute-then-render frame."""
    sprite_vertices = Buffer.from_data(device, SPRITE_VERTICES, label="sprite-vertices")

    particle_seed = initial_particle_bytes()
    particle_buffers = [
        Buffer.from_data(
            device,
            particle_seed,
            usage=PARTICLE_USAGE,
            label=f"particle-buffer-{i}",
        )
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
    sprite_module = ShaderModule.from_wgsl(device, SPRITE_SOURCE, label="sprite")

    compute_pipeline = device.create_compute_pipeline(
        ComputePipelineDescriptor(
            compute=ShaderStageEntry(module=update_module, entry_point="main"),
            layout=None,
            label="boids-compute-pipeline",
        )
    )
    compute_layout = compute_pipeline.get_bind_group_layout(0)
    particle_bind_groups = [
        device.create_bind_group(
            BindGroupDescriptor(
                layout=compute_layout,
                entries=[
                    BindGroupEntry(binding=0, resource=params_buffer),
                    BindGroupEntry(binding=1, resource=particle_buffers[i]),
                    BindGroupEntry(binding=2, resource=particle_buffers[(i + 1) % 2]),
                ],
                label=f"particle-bind-group-{i}",
            )
        )
        for i in range(2)
    ]

    render_pipeline = device.create_render_pipeline(
        RenderPipelineDescriptor(
            vertex_stage=ShaderStageEntry(module=sprite_module, entry_point="vert_main"),
            fragment_stage=ShaderStageEntry(module=sprite_module, entry_point="frag_main"),
            vertex_buffers=[
                VertexBufferLayout(
                    array_stride=PARTICLE_STRIDE,
                    step_mode="instance",
                    attributes=[
                        VertexAttribute(shader_location=0, format=VertexFormat.FLOAT32X2, offset=0),
                        VertexAttribute(shader_location=1, format=VertexFormat.FLOAT32X2, offset=8),
                    ],
                ),
                VertexBufferLayout(
                    array_stride=8,
                    step_mode="vertex",
                    attributes=[
                        VertexAttribute(shader_location=2, format=VertexFormat.FLOAT32X2, offset=0),
                    ],
                ),
            ],
            fragment_targets=[ColorTargetState(format=TextureFormat.BGRA8_UNORM)],
            label="boids-sprite-pipeline",
        )
    )

    encoder = device.create_command_encoder(label="compute-boids-encoder")
    with encoder.begin_compute_pass() as compute_pass:
        compute_pass.set_pipeline(compute_pipeline)
        compute_pass.set_bind_group(0, particle_bind_groups[0])
        compute_pass.dispatch_workgroups(NUM_WORKGROUPS, 1, 1)
    with encoder.begin_render_pass(
        [RenderPassColorAttachment(view=0, clear_value=CLEAR_COLOR)]
    ) as render_pass:
        render_pass.set_pipeline(render_pipeline)
        render_pass.set_vertex_buffer(0, particle_buffers[1])
        render_pass.set_vertex_buffer(1, sprite_vertices)
        render_pass.draw(3, NUM_PARTICLES)
    device.queue.submit([encoder.finish()])


def main() -> None:
    """Wire a :class:`FakeGPU` and encode one computeBoids frame."""
    from easygpu.device import request_adapter
    from easygpu.fake_gpu import FakeGPU
    from easygpu.gpu import configure

    gpu = FakeGPU()
    configure(gpu)
    device = request_adapter().request_device(label="compute-boids-device")
    encode_compute_boids(device)
    print("encoded calls:", [name for name, _ in gpu.calls])


if __name__ == "__main__":
    main()
