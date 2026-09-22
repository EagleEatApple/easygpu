# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-22

First functional release — **command encoding + `FakeGPU` complete;
real backend on the roadmap.**

### Added

- Protocol extended from 20 to 73 methods, covering the full WebGPU surface.
- New wrapper objects: `Texture`/`TextureView`/`Sampler`, `BindGroup` +
  `BindGroupLayout` + `PipelineLayout`, `ComputePipeline`, `QuerySet`,
  `RenderBundleEncoder`/`RenderBundle`, `ComputePassEncoder`.
- Full-protocol capabilities: render bundles, indirect + indexed indirect
  draws, query sets (`resolve_query_set`), buffer mapping, texture copies,
  comparison samplers, pipeline shader constants, depth/stencil, blend,
  multisample + alpha-to-coverage, MRT.
- `FakeGPU` state machine: per-pass call trees, owner tracking,
  `destroy_device` cascade, `buffer_descriptors` label lookup.
- 13 examples (11 `webgpu-samples` ports + `triangle` + `alpha_to_coverage`),
  each guarded by pytest.
- Pure-Python `easygpu.matrix` (column-major, `wgpu-matrix`-style).
- Procedural texture generators for the examples.

### Changed

- Package version bumped `0.0.0` → `0.1.0`; `UserWarning` reworded.
- `_DefaultGPU` no longer inherits `GPU` (structural conformance only).
- `destroy_render_bundle` removed from the protocol (value-object semantics).
- `PipelineLayoutDescriptor` added to `easygpu.bindgroup.__all__`.
- `conftest.fake_gpu` annotated `Iterator[FakeGPU]`.
- `examples/points.py` `_blend_state` renamed to public `blend_state()`.

### Fixed

- Stray `\x20\x20` bytes in `examples/boids_shaders.py`.

## [0.0.0] - 2026-09-20

Initial API-preview release.

### Added

- `GPU` Protocol seam (20 methods) + `configure` / `get_gpu` routing.
- `FakeGPU` — records the command-encoding sequence with no GPU.
- Wrapper objects: `Adapter`, `Device`, `Queue`, `Buffer`, `ShaderModule`,
  `RenderPipeline`, `CommandEncoder` / `RenderPassEncoder` (+ context-manager
  auto-end), `CommandBuffer`.
- `easygpu.errors` hierarchy and minimal spec-aligned enum constants.
- `Buffer.from_data`, `ShaderModule.from_wgsl` convenience constructors.
- One triangle example driven against `FakeGPU` and guarded by pytest.
- Zero runtime dependencies.