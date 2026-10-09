# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-10

Real backend release — **`WgpuPyGPU` behind the stable `GPU` Protocol, with
offscreen render targets that render headlessly on lavapipe and WARP.**

### Added

- `WgpuPyGPU` (`easygpu.wgpu_backend`): delegates the existing `GPU` Protocol to
  wgpu-py. Adapter/device/queue creation (`EASYGPU_SOFTWARE=1` forces a software
  adapter), buffers with synchronous readback (`map_async`/`get_mapped_range`),
  shader modules, render/compute pipelines with auto layouts, textures/views/
  samplers, bind groups, full render-pass and compute-pass encoding, `submit`,
  `write_buffer`/`write_texture` (256-byte-row safe), `copy_texture_to_buffer`.
  `WgpuPyGPU.from_wgpu_device()` adopts an existing wgpu device;
  `WgpuPyGPU.underlying_device` exposes it.
- `easygpu.gpu.WGPU_BACKEND_UNSUPPORTED`: a `frozenset[str]` naming every
  Protocol method `WgpuPyGPU` does not yet implement (14 in 0.2.0). The
  backend raises `GPUValidationError("<name> not yet supported by WgpuPyGPU")`
  for these; a parametrized capability test walks the set so the marker stays
  in sync with the backend.
- `easygpu.testing`: assertion helpers over `FakeGPU.calls` —
  `assert_call_order`, `assert_no_call`, `diff_against`, `call_names`.
  Kept separate from `fake_gpu.py` so the recording path stays import-free
  of assertion concerns.
- `easygpu.offscreen`: `OffscreenTarget`, `offscreen_target()`, and
  `readback_rgba8()` (copy + map + 256-stride unpad → tight RGBA). FakeGPU
  records the same encoding; only a real backend has pixels.
- `easygpu.wgpu_translate`: pure descriptor/enum translation, tested without
  a GPU (18 enum tables + descriptor translators).
- Optional `wgpu` extra (`pip install easygpu[wgpu]`); plain `import easygpu`
  stays zero-dependency.
- Dual-engine offscreen suite in `tests/offscreen/` — every test runs once
  against FakeGPU recording and once against live WgpuPyGPU pixels.
- 3 offscreen examples: `triangle_offscreen`, `textured_cube_offscreen`,
  `compute_boids_offscreen`.
- CI renders offscreen on Linux (lavapipe) and Windows (WARP) with software
  adapters, plus the zero-dependency core job.

### Changed

- `__version__` → `0.2.0`; import warning now nudges installing the `wgpu` extra.
- `RenderPassColorAttachment.view` accepts int handles, `TextureView`, or an
  `OffscreenTarget` (the recommended headless attachment).
- Pipeline color targets must declare a format (as WebGPU requires) — offscreen
  examples use `fragment_targets=[ColorTargetState(format=target.format)]`.
- `tests/conftest.py` gains a `wgpu_gpu` fixture and an `EASYGPU_ENGINE=wgpu`
  switch on the `fake_gpu` fixture for same-suite smoke runs.

### Fixed

- Basic vertex + textured fragment shaders (both `fn main`) now build as
  separate modules to satisfy WGSL redefinition rules on the real backend.

> **Note**: includes the 0.1.1 README fix — examples run as modules
> (`python -m examples.xxx`).

## [0.1.1] - 2026-10-09

### Fixed

- README: run examples as modules (`uv run python -m examples.xxx`) instead
  of `uv run python examples/xxx.py`, which failed with
  `ModuleNotFoundError: No module named 'examples'` because the script
  directory (not the repository root) goes on `sys.path`.

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