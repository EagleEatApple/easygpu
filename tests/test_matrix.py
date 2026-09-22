"""Matrix helper tests."""

from __future__ import annotations

import math

import pytest

from easygpu import matrix


def test_identity_is_column_major() -> None:
    assert matrix.identity() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_create_aliases_identity() -> None:
    assert matrix.create() == matrix.identity()


def test_multiply_with_identity_is_noop() -> None:
    m = matrix.translation((1.0, 2.0, 3.0))
    assert matrix.multiply(m, matrix.identity()) == m
    assert matrix.multiply(matrix.identity(), m) == m


def test_multiply_requires_16_elements() -> None:
    with pytest.raises(ValueError):
        matrix.multiply([1.0, 2.0, 3.0], matrix.identity())


def test_multiply_applies_a_then_b() -> None:
    t = matrix.translation((5.0, 0.0, 0.0))
    r = matrix.rotation_z(math.pi / 2.0)
    combo = matrix.multiply(t, r)
    point = (1.0, 0.0, 0.0, 1.0)
    out = [sum(combo[k * 4 + i] * point[k] for k in range(4)) for i in range(4)]
    assert pytest.approx(out[0], abs=1e-6) == 5.0
    assert pytest.approx(out[1], abs=1e-6) == 1.0


def test_perspective_webgpu_depth_range() -> None:
    near, far = 0.1, 100.0
    m = matrix.perspective(math.pi / 2.0, 1.0, near, far)
    assert m[0] == pytest.approx(1.0)
    assert m[5] == pytest.approx(1.0)
    assert m[10] == pytest.approx(far / (near - far))
    assert m[11] == pytest.approx(far * near / (near - far))
    assert m[14] == pytest.approx(-1.0)


def test_ortho_row() -> None:
    m = matrix.ortho(0.0, 1.0, 0.0, 1.0, 0.1, 1.0)
    assert m[0] == pytest.approx(2.0)
    assert m[5] == pytest.approx(2.0)
    assert m[15] == pytest.approx(1.0)


def test_translation_sets_column4() -> None:
    m = matrix.translation((1.0, 2.0, 3.0))
    assert m[12:15] == [1.0, 2.0, 3.0]
    assert m[0] == 1.0


def test_scale_non_uniform() -> None:
    m = matrix.scale((2.0, 3.0, 4.0))
    assert m[0] == 2.0
    assert m[5] == 3.0
    assert m[10] == 4.0
    assert m[15] == 1.0


def test_rotation_axis_zero_angle_is_identity() -> None:
    assert matrix.rotation_x(0.0) == matrix.identity()
    assert matrix.rotation_y(0.0) == matrix.identity()
    assert matrix.rotation_z(0.0) == matrix.identity()


def test_rotation_z_quarter_turn() -> None:
    m = matrix.rotation_z(math.pi / 2.0)
    assert m[0] == pytest.approx(0.0, abs=1e-6)
    assert m[1] == pytest.approx(1.0)
    assert m[4] == pytest.approx(-1.0)
    assert m[5] == pytest.approx(0.0, abs=1e-6)


def test_rotation_about_axis_matches_axis_helpers() -> None:
    angle = math.pi / 3.0
    assert matrix.rotation((0.0, 0.0, 1.0), angle) == pytest.approx(matrix.rotation_z(angle))
    assert matrix.rotation((1.0, 0.0, 0.0), angle) == pytest.approx(matrix.rotation_x(angle))
    assert matrix.rotation((0.0, 1.0, 0.0), angle) == pytest.approx(matrix.rotation_y(angle))


def test_look_at_z_axis() -> None:
    m = matrix.look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert m[0:3] == pytest.approx([1.0, 0.0, 0.0])
    assert m[4:7] == pytest.approx([0.0, 1.0, 0.0])
    assert m[8:11] == pytest.approx([0.0, 0.0, 1.0])
    assert m[12:15] == pytest.approx([0.0, 0.0, -5.0])
