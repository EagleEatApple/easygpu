"""GPUObject lifecycle tests."""

from __future__ import annotations

import pytest

from easygpu.base import GPUObject
from easygpu.buffer import Buffer
from easygpu.constants import BufferUsage
from easygpu.errors import GPUObjectDeletedError
from easygpu.fake_gpu import FakeGPU


def test_id_exposes_handle() -> None:
    obj = GPUObject(id_=7)
    assert obj.id == 7


def test_label_stored() -> None:
    obj = GPUObject(id_=1, label="vbo")
    assert obj.label == "vbo"
    assert GPUObject(id_=1).label is None


def test_deleted_starts_false() -> None:
    assert GPUObject(id_=1).deleted is False


def test_delete_forwards_type_specific_destroy(fake_gpu: FakeGPU) -> None:
    buf = Buffer(id_=4, size=8, usage=BufferUsage.VERTEX)
    buf.delete()
    assert fake_gpu.calls_of("destroy_buffer") == [(4,)]
    assert 4 not in fake_gpu.buffers


def test_delete_is_idempotent(fake_gpu: FakeGPU) -> None:
    buf = Buffer(id_=4, size=8, usage=BufferUsage.VERTEX)
    buf.delete()
    buf.delete()
    assert len(fake_gpu.calls_of("destroy_buffer")) == 1
    assert buf.deleted is True


def test_id_after_delete_raises(fake_gpu: FakeGPU) -> None:
    buf = Buffer(id_=4, size=8, usage=BufferUsage.VERTEX)
    buf.delete()
    with pytest.raises(GPUObjectDeletedError):
        _ = buf.id


def test_base_delete_requires_concrete_type(fake_gpu: FakeGPU) -> None:
    obj = GPUObject(id_=4)
    with pytest.raises(NotImplementedError):
        obj.delete()
