# EasyGPU

[![PyPI version](https://img.shields.io/pypi/v/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EasyGPU)](https://pypi.org/project/EasyGPU/)
[![CI](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml/badge.svg)](https://github.com/EagleEatApple/easygpu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Status: 0.2.0 — offscreen rendering on a real wgpu-py backend; CI renders
headlessly on lavapipe and WARP.**

EasyGPU is a Python wrapper for WebGPU. The 0.1.0 `GPU` Protocol + `FakeGPU`
let you TDD command encoding without a GPU; 0.2.0 adds `WgpuPyGPU`, a real
wgpu-py backend behind that same protocol, with offscreen render targets for
headless CI and readback.

## What works in 0.2.0

> **Backend capability note**: `FakeGPU` implements the **full** `GPU`
> Protocol (73 methods). `WgpuPyGPU` implements everything listed below except 
> the FakeGPU-only items marked ⚠️: **render bundles**, **query sets**, and
> **`generate_mipmap`** — calling one of them on the real backend raises
> `GPUValidationError("... not yet supported by WgpuPyGPU")`. The method set is
> the single source of truth `easygpu.gpu.WGPU_BACKEND_UNSUPPORTED`; check it 
> when a call behaves differently than you expect.

- ✅ **`GPU` Protocol** (73 methods) + `configure`/`get_gpu`
- ✅ **`FakeGPU`** — records and asserts command encoding without a GPU,
  including `render_pass_sequence()` / `compute_pass_sequence()` call trees
- ✅ **`WgpuPyGPU`** — keeps the Protocol stable and delegates to wgpu-py:
  adapters/devices/queues, buffers + synchronous readback, shaders, render &
  compute pipelines (auto layouts), textures/views/samplers, bind groups,
  render-pass encoding, dispatch, `write_buffer`/`write_texture`
- ✅ **Offscreen targets** — `easygpu.offscreen.OffscreenTarget` +
  `readback_rgba8()` let real pipelines render into textures and pull pixels
  back (headless CI green on Linux/lavapipe and Windows/WARP)
- ✅ **Core objects** — `Buffer`, `Texture`/`TextureView`/`Sampler`,
  `ShaderModule`, `RenderPipeline`/`ComputePipeline`, `BindGroup` +
  `PipelineLayout`, `QuerySet`, `Adapter`/`Device`/`Queue`
- ✅ **Complete render pipeline** — depth/stencil, blend, multisample + A2C,
  indexed draws, viewport/scissor, MRT via multiple color targets
- ✅ **Compute pipeline** — storage buffers, dispatch, indirect dispatch
- ⚠️ **Render bundles** — reusable draw batches (FakeGPU-only in 0.2.0)
- ✅ **Indirect draws** on render passes
- ⚠️ **Query sets** — timestamp/occlusion + `resolve_query_set`
  (FakeGPU-only in 0.2.0; timestamps are a future 0.4.0 item)
- ✅ **Buffer mapping** — `map_async`/`get_mapped_range`/`unmap`
- ✅ **Texture ops** — `write_texture`, `copy_texture_to_buffer`,
  `copy_texture_to_texture`
- ⚠️ **`generate_mipmap`** — FakeGPU-only in 0.2.0 (real mip generation is
  staged for 0.3.0)
- ✅ **Context managers** — `with encoder.begin_render_pass(...):` auto-ends
- ✅ **Spec-aligned enums** — IDL flags for `ShaderStage`/`BufferUsage`/`TextureUsage`
- ✅ **Zero runtime dependencies** by default — the wgpu backend is an optional
  extra (`easygpu[wgpu]`)
- ✅ **16 examples** — 13 command-encoding ports (asserted against FakeGPU
  recording) + 3 offscreen (dual-engine tested: FakeGPU recording + live wgpu
  pixels)

## What does NOT work in 0.2.0

- ❌ No swapchain / window output (offscreen targets work; `view=0` canvas
  sentinels do not on the real backend — use `OffscreenTarget`)
- ❌ No dynamic-offset bind groups (subset-range bindings work via `BufferSlice`;
  see `two_cubes`)
- ❌ No numpy integration: vertex data is raw `bytes` (float sequences are
  struct-packed); the numpy upload path is deferred
- ❌ `generate_mipmap` on the real backend (FakeGPU records it; real mips are
  staged for 0.3.0)

## Examples

All examples live in the [`examples/`](https://github.com/EagleEatApple/easygpu/tree/main/examples)
directory of the repository and are not shipped in the wheel — run them from
a clone or from the source distribution. They run from the repository root;
use Python's `-m` form (e.g. `uv run python -m examples.triangle`) so the
shared helper modules under `examples/` resolve.

Ports of [`webgpu-samples`](https://github.com/webgpu/webgpu-samples) plus two
self-authored examples (`triangle`, `alpha_to_coverage`), each encoding one
frame and guarded by a pytest. 13 command-encoding examples + 3 offscreen;
some ports are simplified (`wireframe`, `reversed_z`). WGSL and data are
inlined/adapted (no browser, no image assets, no window).

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
| `examples/triangle_offscreen.py` | 0.2.0: real wgpu backend + `OffscreenTarget` + `readback_rgba8` |
| `examples/textured_cube_offscreen.py` | 0.2.0: sampled texture rendered headlessly, non-clear pixels |
| `examples/compute_boids_offscreen.py` | 0.2.0: compute dispatch + buffer readback (output-change) |

The offscreen examples render on a real backend and pull pixels back. With a
software adapter they run anywhere:

```bash
EASYGPU_SOFTWARE=1 uv run python -m examples.triangle_offscreen
EASYGPU_SOFTWARE=1 uv run python -m examples.textured_cube_offscreen
EASYGPU_SOFTWARE=1 uv run python -m examples.compute_boids_offscreen
```

Run one of the 0.1.0 encoding examples directly, e.g.:

```bash
uv run python -m examples.textured_cube
uv run pytest tests/test_textured_cube.py -v
```

## Quickstart

### From source (contributors)

```bash
uv sync --extra wgpu
uv run pytest                # headless: FakeGPU suite + offscreen wgpu pixels
uv run python -m examples.triangle
EASYGPU_SOFTWARE=1 uv run python -m examples.triangle_offscreen
```

### From PyPI (users)

```bash
pip install "easygpu[wgpu]"
python -c "from easygpu.offscreen import OffscreenTarget; print(OffscreenTarget(256, 256))"
```

> **Note**: importing `easygpu` emits a `UserWarning` only when wgpu-py is
> **not** installed: without the `wgpu` extra just `FakeGPU` recording works;
> the real backend needs `pip install easygpu[wgpu]`.

## Roadmap

- **0.1.0**: command encoding + `FakeGPU`, 13 examples, zero runtime deps
- **0.1.1**: README fix — examples run as modules (`python -m examples.xxx`)
- **0.2.0** (current): real wgpu-py backend (`WgpuPyGPU`) behind the stable
  Protocol, offscreen targets + pixel readback, headless CI on lavapipe/WARP
- 0.3.0: swapchain + windowed output; `generate_mipmap` and timestamp
  queries on the real backend
- 0.4.0: dynamic-offset bind groups, render bundles on the real backend
- Later: more `webgpu-samples` ports

## Project layout

- `src/easygpu/` — library (Protocol seam + wrapper objects + `fake_gpu.py` +
  `wgpu_translate.py` + `wgpu_backend.py` + `offscreen.py` + `surface.py` +
  `testing.py`)
- `examples/` — 16 runnable examples (13 command-encoding + 3 offscreen),
  plus 5 shared helper modules (`shaders.py`, `shadow_shaders.py`,
  `boids_shaders.py`, `cube_mesh.py`, `textures.py`) that the examples
  import
- `tests/` — pytest suite for the library and every example (headless;
  `tests/offscreen/` covers the wgpu pixels + FakeGPU recording dual engine)

> The layout above describes the **source repository**. The published wheel
> contains only `src/easygpu/` — `examples/` and `tests/` live in the
> repository and in the source distribution (`easygpu-0.2.0.tar.gz`).
