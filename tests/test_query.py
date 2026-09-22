"""QuerySet / QuerySetDescriptor module tests."""

from __future__ import annotations

from easygpu.constants import QueryType
from easygpu.fake_gpu import FakeGPU


def test_query_module_path() -> None:
    from easygpu import query

    assert hasattr(query, "QuerySet")
    assert hasattr(query, "QuerySetDescriptor")


def test_query_set_descriptor_defaults() -> None:
    from easygpu.query import QuerySetDescriptor

    descriptor = QuerySetDescriptor(type=QueryType.TIMESTAMP, count=8)
    assert descriptor.type == QueryType.TIMESTAMP
    assert descriptor.count == 8
    assert descriptor.label is None


def test_query_set_descriptor_label() -> None:
    from easygpu.query import QuerySetDescriptor

    descriptor = QuerySetDescriptor(type=QueryType.OCCLUSION, count=1, label="occ")
    assert descriptor.label == "occ"


def test_query_set_delete_forwards_destroy(fake_gpu: FakeGPU) -> None:
    from easygpu.query import QuerySet

    query_set = QuerySet(id_=5)
    query_set_id = query_set.id
    query_set.delete()
    assert fake_gpu.calls_of("destroy_query_set") == [(query_set_id,)]
