"""GPU protocol seam and process routing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import easygpu
from easygpu.gpu import GPU, configure, get_gpu


class _Probe:
    """Minimal protocol object used to prove routing works."""

    def request_adapter(self, *, force_fallback_adapter: bool = False) -> int:
        return 1


def test_configure_and_get_gpu_roundtrip() -> None:
    assert isinstance(get_gpu(), GPU)
    probe = _Probe()
    previous = get_gpu()
    configure(probe)  # type: ignore[arg-type]  # structural: probe implements GPU subset
    try:
        assert get_gpu() is probe
    finally:
        configure(previous)


def test_default_implementation_satisfies_protocol_structurally() -> None:
    from easygpu.gpu import _DefaultGPU

    assert isinstance(_DefaultGPU(), GPU)


def test_gpu_module_stays_a_protocol_leaf() -> None:
    """easygpu.gpu must not pull wrapper modules in at runtime."""
    code = (
        "import os, sys, types\n"
        "pkg = types.ModuleType('easygpu')\n"
        "pkg.__path__ = [os.environ['EASYGPU_SRC']]\n"
        "sys.modules['easygpu'] = pkg\n"
        "import easygpu.gpu\n"
        "leaked = {'easygpu.buffer', 'easygpu.shader', 'easygpu.pipeline',"
        " 'easygpu.encoder', 'easygpu.device', 'easygpu.base'} & set(sys.modules)\n"
        "assert not leaked, leaked\n"
    )
    env = {**os.environ, "EASYGPU_SRC": str(Path(easygpu.__file__).parent)}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
