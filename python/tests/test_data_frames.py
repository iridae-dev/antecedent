"""Data-frame constructors and ingest helpers (no native calls)."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import data
from antecedent._data import (
    as_columns,
    as_multi_env_columns,
    coerce_data_args,
    ingest_columns,
    to_f64,
    try_as_arrow_c_columns,
)


def test_event_and_panel_and_multi_env_constructors():
    frame = {"a": [0.0, 1.0], "y": [2.0, 3.0]}
    with pytest.raises(ValueError, match="align_interval_ns"):
        data.event(frame, [1, 2], align_interval_ns=0)
    with pytest.raises(ValueError, match="1-d"):
        data.event(frame, [[1, 2]], align_interval_ns=1)
    with pytest.raises(ValueError, match="length"):
        data.event(frame, [1], align_interval_ns=1)
    events = data.event(frame, [10, 20], align_interval_ns=5)
    assert events.align_interval_ns == 5
    assert list(events.event_times_ns) == [10, 20]

    with pytest.raises(ValueError, match="≥1 unit"):
        data.panel([])
    with pytest.raises(ValueError, match="do not match"):
        data.panel([frame, {"a": [0.0]}])
    by_id = data.panel({7: frame, 8: {"a": [4.0, 5.0], "y": [6.0, 7.0]}})
    assert by_id.unit_ids == [7, 8]
    sequential = data.panel([frame])
    assert sequential.unit_ids == [0]

    envs = data.multi_env([frame, {"a": [1.0, 2.0], "y": [3.0, 4.0]}])
    assert envs.names == ["a", "y"]
    with pytest.raises(ValueError, match="non-empty"):
        as_multi_env_columns([])
    with pytest.raises(ValueError, match="do not match"):
        as_multi_env_columns([frame, {"a": [0.0]}])


def test_column_ingest_and_arrow_probes():
    names, cols = as_columns({"a": [1, 2]})
    assert names == ["a"]
    assert cols[0].dtype == np.float64
    with pytest.raises(TypeError, match="mapping"):
        as_columns([1, 2])
        with pytest.raises(ValueError, match="1-d"):
            to_f64([[1.0, 2.0]])

    names2, cols2 = coerce_data_args(names=["z"], columns=[[1.0, 2.0]])
    assert names2 == ["z"]
    assert list(cols2[0]) == [1.0, 2.0]
    names3, cols3 = coerce_data_args(data={"z": [1.0, 2.0]})
    assert names3 == ["z"]
    assert list(cols3[0]) == [1.0, 2.0]
    with pytest.raises(TypeError, match="names="):
        coerce_data_args()

    class Series:
        def to_numpy(self) -> np.ndarray:
            return np.array([1.0, 2.0])

    class Frame:
        columns = ["a"]

        def __getitem__(self, key: str) -> Series:
            return Series()

        def to_numpy(self) -> np.ndarray:
            return np.array([[1.0], [2.0]])

    frame_names, frame_cols = as_columns(Frame())
    assert frame_names == ["a"]
    assert list(frame_cols[0]) == [1.0, 2.0]

    class ArrowCol:
        def __arrow_c_array__(self, requested_schema=None):
            return (None, None)

    mapping = {"a": ArrowCol()}
    assert try_as_arrow_c_columns(mapping)[0] == ["a"]
    assert try_as_arrow_c_columns({"a": [1.0]}) is None
    assert ingest_columns({"a": [1.0]})[0] == ["a"]
    assert ingest_columns(mapping)[0] == ["a"]

    class Table:
        column_names = ["a"]

        def column(self, i: int) -> ArrowCol:
            return ArrowCol()

    assert try_as_arrow_c_columns(Table())[0] == ["a"]

    class Schema:
        names = ["a"]

    class SchemaTable:
        schema = Schema()

        def column(self, i: int) -> ArrowCol:
            return ArrowCol()

    assert try_as_arrow_c_columns(SchemaTable())[0] == ["a"]

    class Chunked:
        def combine_chunks(self) -> ArrowCol:
            return ArrowCol()

    class ChunkedTable:
        column_names = ["a"]

        def column(self, i: int) -> Chunked:
            return Chunked()

    assert try_as_arrow_c_columns(ChunkedTable())[0] == ["a"]

    class BadColumns:
        columns = ["a"]

        def __getitem__(self, key: str) -> None:
            raise KeyError(key)

    assert try_as_arrow_c_columns(BadColumns()) is None

    class FrameCols:
        columns = ["a"]

        def __getitem__(self, key: str) -> ArrowCol:
            return ArrowCol()

    assert try_as_arrow_c_columns(FrameCols())[0] == ["a"]
