"""SurfaceTarget surface render attachment tests."""

from __future__ import annotations

from easygpu.constants import TextureFormat
from easygpu.encoder import RenderPassColorAttachment
from easygpu.fake_gpu import FakeGPU
from easygpu.surface import SurfaceTarget
from easygpu.wgpu_backend import WgpuPyGPU, configure_service


class _CanvasStub:
    def __init__(self, width: int, height: int) -> None:
        self._size = (width, height)
        self.context_calls: list[str] = []

    def get_physical_size(self) -> tuple[int, int]:
        return self._size

    def get_wgpu_context(self) -> _ContextStub:
        return _ContextStub(self.context_calls)


class _ContextStub:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def get_current_texture(self) -> _TextureStub:
        self._calls.append("get_current_texture")
        return _TextureStub(self._calls)


class _TextureStub:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def create_view(self) -> object:
        self._calls.append("create_view")
        return "view-sentinel"


def test_surface_target_exposes_physical_size() -> None:
    target = SurfaceTarget(canvas=_CanvasStub(64, 48))
    assert target.width == 64
    assert target.height == 48
    assert target.format == TextureFormat.RGBA8_UNORM
    assert target.label is None


def test_surface_target_acquire_pulls_current_texture_view() -> None:
    canvas = _CanvasStub(32, 16)
    target = SurfaceTarget(canvas=canvas)
    assert target.acquire() == "view-sentinel"
    assert canvas.context_calls == ["get_current_texture", "create_view"]


def test_resolve_target_routes_surface_before_duck_branch() -> None:
    gpu = object.__new__(WgpuPyGPU)
    target = SurfaceTarget(canvas=_CanvasStub(64, 64))
    assert gpu._resolve_target(target, device=None) == "view-sentinel"


def test_resolve_target_rejects_zero_int_view() -> None:
    gpu = configure_service()
    try:
        gpu._resolve_target(0, device=None)
    except Exception as exc:
        assert "swapchain surface" in str(exc)
    else:
        raise AssertionError("view=0 must fail loud")


def test_fake_gpu_records_surface_attachment_without_acquire() -> None:
    class _ForbiddenCanvas:
        def get_physical_size(self) -> tuple[int, int]:
            return (64, 64)

        def get_wgpu_context(self) -> object:
            raise AssertionError("FakeGPU must not acquire the surface")

    gpu = FakeGPU()
    encoder = gpu.create_command_encoder(1, label="pass")
    attachment = RenderPassColorAttachment(view=SurfaceTarget(canvas=_ForbiddenCanvas()))
    gpu.begin_render_pass(encoder, [attachment], None)
    passes = [c[1] for c in gpu.calls if c[0] == "begin_render_pass"]
    assert len(passes) == 1
    assert isinstance(passes[0][1][0].view, SurfaceTarget)
