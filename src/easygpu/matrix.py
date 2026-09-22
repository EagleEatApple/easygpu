"""Pure-Python WebGPU-style math helpers (zero runtime dependencies).

A small, dependency-free port of the subset of ``wgpu-matrix`` used by the
webgpu-samples that EasyGPU re-implements. Matrices are **column-major**
4x16 ``float`` lists (matching WGSL ``mat4x4f`` memory layout); ``Buffer``
packs them straight into uniform buffers via :func:`struct.pack`.

The helpers are functional (they return new matrices) rather than mutating
``out`` arguments — the JS samples mutate for performance, which a Python
re-implementation does not need.
"""

from __future__ import annotations

import math

Mat4 = list[float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]

_TYPE = "Mat4 is a 16-element column-major list of floats"


def _chk(m: list[float] | tuple[float, ...]) -> None:
    if len(m) != 16:
        raise ValueError(_TYPE)


def identity() -> Mat4:
    """Return the 4x4 identity matrix."""
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def create() -> Mat4:
    """Alias of :func:`identity` (mirrors ``wgpu-matrix`` ``mat4.create``)."""
    return identity()


def multiply(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> Mat4:
    """Return ``a * b`` (both must be column-major 4x4 matrices)."""
    _chk(a)
    _chk(b)
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


def perspective(fovy: float, aspect: float, near: float, far: float) -> Mat4:
    """Return a WebGPU-style (depth mapped to [0, 1]) perspective matrix."""
    f = 1.0 / math.tan(fovy / 2.0)
    m = [0.0] * 16
    m[0] = f / aspect
    m[5] = f
    m[10] = far / (near - far)
    m[11] = far * near / (near - far)
    m[14] = -1.0
    return m


def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4:
    """Return an orthographic projection (WebGPU depth [0, 1])."""
    m = [0.0] * 16
    m[0] = 2.0 / (right - left)
    m[5] = 2.0 / (top - bottom)
    m[10] = 1.0 / (near - far)
    m[12] = (left + right) / (left - right)
    m[13] = (bottom + top) / (bottom - top)
    m[14] = near / (near - far)
    m[15] = 1.0
    return m


def translation(xyz: tuple[float, float, float]) -> Mat4:
    """Return a translation matrix."""
    m = identity()
    m[12], m[13], m[14] = xyz
    return m


def scale(xyz: tuple[float, float, float]) -> Mat4:
    """Return a non-uniform scale matrix."""
    m = identity()
    m[0] *= xyz[0]
    m[5] *= xyz[1]
    m[10] *= xyz[2]
    return m


def rotation_x(angle: float) -> Mat4:
    """Return a rotation about the X axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return [1.0, 0.0, 0.0, 0.0, 0.0, c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0]


def rotation_y(angle: float) -> Mat4:
    """Return a rotation about the Y axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return [c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0]


def rotation_z(angle: float) -> Mat4:
    """Return a rotation about the Z axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    return [c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def rotation(axis: tuple[float, float, float], angle: float) -> Mat4:
    """Return a rotation matrix for ``angle`` radians around a unit ``axis``.

    Mirrors ``wgpu-matrix`` ``mat4.rotation`` (Rodrigues' rotation formula).
    """
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        t * x * x + c,
        t * x * y + s * z,
        t * x * z - s * y,
        0.0,
        t * x * y - s * z,
        t * y * y + c,
        t * y * z + s * x,
        0.0,
        t * x * z + s * y,
        t * y * z - s * x,
        t * z * z + c,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float],
) -> Mat4:
    """Return a right-handed view matrix looking from ``eye`` to ``target``."""
    z = _normalize3((eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]))
    x = _normalize3(_cross3(up, z))
    y = _cross3(z, x)
    m = [0.0] * 16
    m[0], m[1], m[2] = x
    m[4], m[5], m[6] = y
    m[8], m[9], m[10] = z
    m[12] = -_dot3(x, eye)
    m[13] = -_dot3(y, eye)
    m[14] = -_dot3(z, eye)
    m[15] = 1.0
    return m


__all__ = [
    "create",
    "identity",
    "look_at",
    "multiply",
    "ortho",
    "perspective",
    "rotation",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "scale",
    "translation",
]
