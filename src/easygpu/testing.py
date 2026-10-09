"""Assertion helpers for driving :class:`~easygpu.fake_gpu.FakeGPU` call logs.

The 0.2.x workflow (TDD-first and AI-agent friendly) records every protocol
call in ``FakeGPU.calls``. These helpers make "these calls happened, in this
order, and nothing else" expressible without hand-rolling list comprehensions
in every test.

They are intentionally small and separate from :mod:`easygpu.fake_gpu` so the
production backend stays lean: import :mod:`easygpu.testing` only in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from easygpu.fake_gpu import FakeGPU

Call = tuple[str, tuple[Any, ...]]


def call_names(fake: FakeGPU) -> list[str]:
    """Return the recorded call names, in order."""
    return [name for name, _ in fake.calls]


def assert_call_order(fake: FakeGPU, *names: str) -> list[str]:
    """Assert ``names`` appear in the recorded log in this relative order.

    Other calls may occur between them; each ``name`` must appear at least
    once. Raises ``AssertionError`` listing missing or out-of-order calls and
    returns the recorded names on success for chained assertions.
    """
    recorded = call_names(fake)
    positions: dict[str, int] = {}
    for index, name in enumerate(recorded):
        positions.setdefault(name, index)
    missing = [name for name in names if name not in positions]
    assert not missing, f"these calls were never recorded: {missing}"
    ordered = [positions[name] for name in names]
    assert ordered == sorted(ordered), (
        f"calls recorded out of expected order: expected {names}, got {recorded}"
    )
    return recorded


def assert_no_call(fake: FakeGPU, name: str) -> None:
    """Assert ``name`` never appears in the recorded log."""
    if any(recorded == name for recorded, _ in fake.calls):
        raise AssertionError(f"unexpected call recorded: {name}")


def diff_against(fake: FakeGPU, golden: Sequence[Call]) -> list[tuple[Call, Call]]:
    """Return ``(expected, actual)`` mismatches between ``golden`` and the log.

    A length mismatch and every element mismatch both surface as entries;
    an empty list means the log matches ``golden`` exactly. Placeholders
    ``("<missing>", ())`` / ``("<unexpected>", ())`` mark the tail when
    lengths differ.
    """
    actual: list[Call] = list(fake.calls)
    mismatches: list[tuple[Call, Call]] = [
        (expected, recorded)
        for expected, recorded in zip(golden, actual, strict=False)
        if expected != recorded
    ]
    if len(golden) > len(actual):
        for offset in range(len(actual), len(golden)):
            mismatches.append((golden[offset], ("<missing>", ())))
    elif len(actual) > len(golden):
        for offset in range(len(golden), len(actual)):
            mismatches.append((("<unexpected>", ()), actual[offset]))
    return mismatches
