"""Buffer wrapper tests."""

from __future__ import annotations

import struct

import pytest

from easygpu.buffer import Buffer, BufferDescriptor
from easygpu.constants import BufferUsage, MapMode
from easygpu.device import Device
from easygpu.errors import GPUObjectDeletedError
from easygpu.gpu import GPUValidationError


def test_descriptor_holds_shape() -> None:
    desc = BufferDescriptor(size=60, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST, label="vert")
    assert desc.size == 60
    assert desc.usage == (BufferUsage.VERTEX | BufferUsage.COPY_DST)
    assert desc.label == "vert"
    assert desc.mapped_at_creation is False


def test_descriptor_defaults() -> None:
    desc = BufferDescriptor(size=8, usage=BufferUsage.VERTEX)
    assert desc.label is None
    assert desc.mapped_at_creation is False


def test_buffer_exposes_attrs() -> None:
    buf = Buffer(id_=3, size=60, usage=BufferUsage.VERTEX, label="positions")
    assert buf.id == 3
    assert buf.size == 60
    assert buf.usage == BufferUsage.VERTEX
    assert buf.label == "positions"


def test_buffer_delete_guards_id(fake_gpu) -> None:
    buf = Buffer(id_=3, size=60, usage=BufferUsage.VERTEX)
    buf.delete()
    with pytest.raises(GPUObjectDeletedError):
        _ = buf.id


def test_buffer_extracts_size_from_descriptor() -> None:
    desc = BufferDescriptor(size=24, usage=BufferUsage.VERTEX)
    assert desc.size == 24


def test_buffer_negative_size_rejected() -> None:
    with pytest.raises(ValueError):
        Buffer(id_=1, size=-1, usage=BufferUsage.VERTEX)


def test_buffer_from_data_uploads_bytes(fake_gpu) -> None:
    device = Device(id_=1)
    data = b"\x01\x02\x03\x04"
    buf = Buffer.from_data(device, data, usage=BufferUsage.VERTEX | BufferUsage.COPY_DST)
    assert buf.size == 4
    assert bytes(fake_gpu.buffers[buf.id]) == data
    assert fake_gpu.calls_of("write_buffer")[0][2] == 0
    assert [name for name, _ in fake_gpu.calls] == [
        "create_buffer",
        "get_queue",
        "write_buffer",
    ]


def test_buffer_from_data_packs_float_sequence(fake_gpu) -> None:
    device = Device(id_=1)
    buf = Buffer.from_data(device, [1.0, -2.5])
    assert buf.size == 8
    assert bytes(fake_gpu.buffers[buf.id]) == struct.pack("<2f", 1.0, -2.5)


def test_buffer_from_data_rejects_non_numeric(fake_gpu) -> None:
    device = Device(id_=1)
    with pytest.raises(TypeError):
        Buffer.from_data(device, 42)
    with pytest.raises(TypeError):
        Buffer.from_data(device, ["not", "numeric"])


def test_buffer_map_and_get_mapped_range_defaults(fake_gpu) -> None:
    device = Device(id_=1)
    buf = Buffer(id_=2, size=11, usage=BufferUsage.MAP_READ)
    device.queue.write_buffer(buf, 0, b"hello world")
    buf.map_async(MapMode.READ)
    assert fake_gpu.calls_of("map_async") == [(2, MapMode.READ, 0, 11)]
    assert buf.get_mapped_range() == b"hello world"
    assert fake_gpu.calls_of("get_mapped_range") == [(2, 0, 11)]
    assert fake_gpu.mapped[buf.id] == (MapMode.READ, 0, 11)


def test_buffer_map_async_offset_size_and_get_mapped_range_slice(fake_gpu) -> None:
    device = Device(id_=1)
    buf = Buffer(id_=2, size=11, usage=BufferUsage.MAP_READ)
    device.queue.write_buffer(buf, 0, b"hello world")
    buf.map_async(MapMode.READ, offset=2)
    assert fake_gpu.calls_of("map_async") == [(2, MapMode.READ, 2, 9)]
    assert buf.get_mapped_range(2, 4) == b"llo "
    assert fake_gpu.calls_of("get_mapped_range") == [(2, 2, 4)]


def test_buffer_unmap_records_and_clears_mapping(fake_gpu) -> None:
    device = Device(id_=1)
    buf = Buffer(id_=2, size=11, usage=BufferUsage.MAP_READ)
    device.queue.write_buffer(buf, 0, b"hello world")
    buf.map_async(MapMode.READ)
    buf.unmap()
    assert fake_gpu.calls_of("unmap") == [(2,)]
    assert buf.id not in fake_gpu.mapped
    with pytest.raises(GPUValidationError):
        buf.get_mapped_range()
