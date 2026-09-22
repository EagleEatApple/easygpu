# EasyGPU

[![PyPI version](https://img.shields.io/pypi/v/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![CI](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml/badge.svg)](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status: 0.1.0 — command encoding + FakeGPU complete; real backend on the roadmap.**

EasyGPU is a Python wrapper for WebGPU with a built-in `FakeGPU` that
lets you TDD your command-encoding logic without a GPU.

## What works in 0.1.0

- ✅ **`GPU` Protocol** (73 methods) + `configure`/`get_gpu`
- ✅ **`FakeGPU`** — records and asserts command encoding without a GPU,
  including `render_pass_sequence()` / `compute_pass_sequence()` call trees
- ✅ **Core objects** — `Buffer`, `Texture`/`TextureView`/`Sampler`,
  `ShaderModule`, `RenderPipeline`/`ComputePipeline`, `BindGroup` +
  `PipelineLayout`, `QuerySet`, `Adapter`/`Device`/`Queue`
- ✅ **Complete render pipeline** — depth/stencil, blend, multisample + A2C,
  indexed draws, viewport/scissor, MRT via multiple color targets
- ✅ **Compute pipeline** — storage buffers, dispatch, indirect dispatch
- ✅ **Render bundles** — reusable draw batches
- ✅ **Indirect draws** on render passes
- ✅ **Query sets** — timestamp/occlusion + `resolve_query_set`
- ✅ **Buffer mapping** — `map_async`/`get_mapped_range`/`unmap`
- ✅ **Texture ops** — `write_texture`, `copy_texture_to_*`, `generate_mipmap`
- ✅ **Context managers** — `with encoder.begin_render_pass(...):` auto-ends
- ✅ **Spec-aligned enums** — IDL flags for `ShaderStage`/`BufferUsage`/`TextureUsage`
- ✅ **Zero runtime dependencies** — `struct.pack` for data, pure-Python
  `easygpu.matrix`
- ✅ **13 examples** — 11 `webgpu-samples` ports + 2 self-authored (triangle, alpha_to_coverage)

## What does NOT work in 0.1.0

- ❌ No real GPU backend — the `GPU` Protocol is the deliverable; the wrappers
  raise `GPUValidationError` unless you `configure()` a fake or real backend
- ❌ No swapchain / window output (examples bind color attachment handle `0`)
- ❌ No dynamic-offset bind groups (subset-range bindings work via `BufferSlice`;
  see `two_cubes`)
- ❌ No numpy integration: vertex data is raw `bytes` (float sequences are
  struct-packed); the numpy upload path is deferred
- ❌ WGSL execution (FakeGPU records calls, it does not run shaders)

## Examples

All examples live in the [`examples/`](https://github.com/EagleEatApple/easygpu/tree/main/examples)
directory of the repository. They are not shipped in the wheel — run them
from a clone or from the source distribution.

Ports of [`webgpu-samples`](https://github.com/webgpu/webgpu-samples), each
encoding one frame against `FakeGPU` and verified by a pytest. 11
`webgpu-samples` ports + 2 self-authored examples (`triangle`,
`alpha_to_coverage`); some ports are simplified (`wireframe`, `reversed_z`).
WGSL and data are inlined/adapted (no browser, no image assets).

| Example | What it teaches |
| --- | --- |
| `examples/triangle.py` | minimal vertex/fragment pipeline |
| `examples/rotating_cube.py` | perspective, MVP uniforms, `depth24plus` |
| `examples/two_cubes.py` | one uniform buffer, two bind groups at 256-byte offsets |
| `examples/instanced_cube.py` | instancing, `array<mat4x4f, 16>`, `instance_index` |
| `examples/textured_cube.py` | texture + sampler + `write_texture` |
| `examples/cameras.py` | look-at view matrix, `cube.wgsl` |
| `examples/points.py` | Fibonacci sphere, instanced billboards, blending, 4 pipelines |
| `examples/wireframe.py` | `set_index_buffer` + `draw_indexed`, depth bias line overlay (simplified) |
| `examples/reversed_z.py` | reversed-Z depth precision, split viewport (color mode) |
| `examples/blending.py` | source-over blend state + `set_blend_constant` |
| `examples/alpha_to_coverage.py` | 4x MSAA + alpha-to-coverage + resolve target |
| `examples/shadow_mapping.py` | dual-pass shadow mapping, `depth32float`, comparison sampler |
| `examples/compute_boids.py` | compute pipeline, storage ping-pong, instanced sprites |

Run one directly, e.g.:

```bash
uv run python examples/textured_cube.py
uv run pytest tests/test_textured_cube.py -v
```

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
> 0.1.0 ships no real backend. It will go away once one lands.

## Roadmap

- **0.1.0** (current): command encoding + `FakeGPU`, 13 examples, zero runtime deps
- 0.2.0: real backend behind the `GPU` Protocol (wgpu-py), keeping the Protocol stable
- 0.3.0: swapchain + windowed output
- 0.4.0: dynamic-offset bind groups, GPU timing readback
- Later: more `webgpu-samples` ports

## Project layout

- `src/easygpu/` — library (Protocol seam + wrapper objects + `fake_gpu.py`)
- `examples/` — runnable examples (11 `webgpu-samples` ports + 2 self-authored)
- `tests/` — pytest suite for the library and every example (headless)

> The layout above describes the **source repository**. The published wheel
> contains only `src/easygpu/` — `examples/` and `tests/` live in the
> repository and in the source distribution (`easygpu-0.1.0.tar.gz`).
