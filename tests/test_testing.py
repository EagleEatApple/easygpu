"""Tests for the :mod:`easygpu.testing` call-log helpers."""

from __future__ import annotations

import pytest

from easygpu.fake_gpu import FakeGPU
from easygpu.testing import assert_call_order, assert_no_call, call_names, diff_against


def _recorded(fake: FakeGPU, names: list[str]) -> None:
    for name in names:
        fake.calls.append((name, (1,)))


def test_call_names_in_order() -> None:
    fake = FakeGPU()
    _recorded(fake, ["get_queue", "create_buffer", "write_buffer"])
    assert call_names(fake) == ["get_queue", "create_buffer", "write_buffer"]


def test_assert_call_order_accepts_interleaved_calls() -> None:
    fake = FakeGPU()
    _recorded(fake, ["create_buffer", "write_buffer", "create_buffer", "submit"])
    assert_call_order(fake, "create_buffer", "write_buffer", "submit")


def test_assert_call_order_rejects_reordered_calls() -> None:
    fake = FakeGPU()
    _recorded(fake, ["write_buffer", "create_buffer"])
    with pytest.raises(AssertionError, match="out of expected order"):
        assert_call_order(fake, "create_buffer", "write_buffer")


def test_assert_call_order_rejects_missing_calls() -> None:
    fake = FakeGPU()
    _recorded(fake, ["create_buffer"])
    with pytest.raises(AssertionError, match="never recorded"):
        assert_call_order(fake, "create_buffer", "submit")


def test_assert_no_call_passes_when_absent() -> None:
    fake = FakeGPU()
    _recorded(fake, ["create_buffer"])
    assert_no_call(fake, "write_timestamp")


def test_assert_no_call_rejects_when_present() -> None:
    fake = FakeGPU()
    _recorded(fake, ["create_buffer", "write_timestamp"])
    with pytest.raises(AssertionError, match="unexpected call"):
        assert_no_call(fake, "write_timestamp")


def test_diff_against_exact_match_is_empty() -> None:
    fake = FakeGPU()
    _recorded(fake, ["a", "b"])
    assert diff_against(fake, [("a", (1,)), ("b", (1,))]) == []


def test_diff_against_flags_element_mismatch() -> None:
    fake = FakeGPU()
    fake.calls.append(("a", (2,)))
    [(expected, actual)] = diff_against(fake, [("a", (1,))])
    assert expected == ("a", (1,))
    assert actual == ("a", (2,))


def test_diff_against_flags_length_difference() -> None:
    fake = FakeGPU()
    _recorded(fake, ["a"])
    golden = [("a", (1,)), ("b", (1,))]
    [(missing, placeholder)] = diff_against(fake, golden)
    assert missing == ("b", (1,))
    assert placeholder == ("<missing>", ())
