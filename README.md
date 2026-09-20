# EasyGPU

[![PyPI version](https://img.shields.io/pypi/v/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![CI](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml/badge.svg)](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status: 0.0.0 preview — API design validation only.**

EasyGPU is a Python wrapper for WebGPU with a built-in `FakeGPU` that
lets you TDD your command-encoding logic without a GPU.

## What works in 0.0.0

- ✅ `GPU` Protocol (the full interface EasyGPU uses) + `configure`/`get_gpu`
- ✅ `FakeGPU` — record and assert command-encoding calls (in `src/easygpu/`),
  including `render_pass_sequence(pass_handle)` for per-pass call trees
- ✅ `Buffer`, `ShaderModule`, `RenderPipeline`, `CommandEncoder` (+ `Adapter`, `Device`, `Queue`)
- ✅ Convenience constructors: `Buffer.from_data(device, data)` (bytes or a float
  sequence), `ShaderModule.from_wgsl(device, *sources)`
- ✅ Render passes work as context managers — `with encoder.begin_render_pass(...):`
  auto-calls `end()`
- ✅ Spec-aligned enums: `BufferUsage`/`ShaderStage` use the WebGPU IDL bit flags
  (`IntFlag`), the rest mirror wgpu-native ABI encodings
- ✅ `examples/triangle.py` — a triangle example that verifies command encoding via `FakeGPU`

## What does NOT work in 0.0.0

- ❌ No real GPU backend — the `GPU` Protocol is the deliverable; the wrappers
  raise `GPUValidationError` unless you `configure()` a fake or real backend
- ❌ No swapchain / window output, no textures, no compute passes
- ❌ No runtime dependencies: vertex data is raw `bytes` (float sequences are
  struct-packed); the reference numpy upload path is intentionally deferred
- ❌ WGSL execution (FakeGPU records calls, it does not run shaders)


## Quickstart

### From source (contributors)

```bash
uv sync
uv run pytest                # headless, no GPU needed
uv run python examples/triangle.py
```

### From PyPI (users)

```bash
pip install easygpu
python -c "from easygpu.fake_gpu import FakeGPU; print(FakeGPU())"
```

> **Note**: importing `easygpu` emits a `UserWarning` explaining that
> 0.0.0 is an API preview. That warning disappears in 0.1.0.

## Roadmap

- 0.1.0: real backend behind the `GPU` Protocol (wgpu/native), keeping the
  Protocol stable; translate the spec-aligned enums to the backend ABI
- Later: swapchain + windowed output, textures, compute
- 0.2.0: compute pipelines + bind groups + samplers
- 0.3.0: more examples (triangle, compute, multi-pass, instancing etc.)

## Project layout

- `src/easygpu/` — library (Protocol seam + wrapper objects + `fake_gpu.py`)
- `examples/` — runnable examples
