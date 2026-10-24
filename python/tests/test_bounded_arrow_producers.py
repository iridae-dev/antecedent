"""Producers are asked for bounded record batches before their Arrow stream is exported."""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from antecedent._data import bounded_arrow_reader  # noqa: E402
from antecedent.errors import CausalValueError  # noqa: E402

ROWS = 2_500
LIMIT = 400


def _table() -> pa.Table:
    return pa.table({"x": [float(i) for i in range(ROWS)], "k": list(range(ROWS))})


def _check(reader: pa.RecordBatchReader) -> None:
    sizes = [batch.num_rows for batch in reader]
    assert sum(sizes) == ROWS
    assert max(sizes) <= LIMIT


def test_pyarrow_table_is_rebatched_at_the_producer() -> None:
    _check(bounded_arrow_reader(_table(), LIMIT))


def test_one_oversized_batch_reader_is_split_and_keeps_every_value() -> None:
    table = _table()
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(max_chunksize=ROWS))
    rebatched = bounded_arrow_reader(reader, LIMIT)
    assert pa.Table.from_batches(list(rebatched)).equals(table)


def test_a_stream_exporter_is_split_as_its_batches_arrive() -> None:
    class Exporter:
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            return _table().to_reader(max_chunksize=ROWS).__arrow_c_stream__(requested_schema)

    _check(bounded_arrow_reader(Exporter(), LIMIT))


def test_polars_frames_are_sliced_at_the_producer() -> None:
    pl = pytest.importorskip("polars")
    _check(bounded_arrow_reader(pl.from_arrow(_table()), LIMIT))


def test_duckdb_results_request_the_batch_size_from_the_engine() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.register("t", _table())
    _check(bounded_arrow_reader(connection.sql("select * from t"), LIMIT))


def test_a_zero_row_bound_is_refused() -> None:
    with pytest.raises(CausalValueError):
        bounded_arrow_reader(_table(), 0)
