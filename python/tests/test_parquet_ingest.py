"""Parquet and Arrow IPC tables from Python: lazy handles, bounded reads, identity, analyses."""

from __future__ import annotations

import math
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalDataError, CausalResourceError, CausalValueError
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect("t", "y")


def _sample(seed: int = 7, n: int = 400) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z))).astype(np.float64)
    y = 2.0 * t + z + rng.normal(scale=0.3, size=n)
    return {"t": t, "y": y, "z": z}


def _write(path: Path, data: dict[str, object], row_group_size: int = 128) -> Path:
    pq.write_table(pa.table(data), path, row_group_size=row_group_size)
    return path


def _with_extras(data: dict[str, np.ndarray]) -> dict[str, object]:
    n = len(data["t"])
    labels = np.array(["a", "b", "c"])[np.arange(n) % 3]
    stamps = np.datetime64("2024-03-01T00:00:00") + np.arange(n).astype("timedelta64[m]")
    return {
        **data,
        "label": pa.array(labels).dictionary_encode(),
        "at": pa.array(stamps.astype("datetime64[ms]")),
        "nested": pa.array([[i, i + 1] for i in range(n)]),
    }


def _analyze(data: object) -> object:
    return require_success(
        ant.analyze(
            data,
            query=QUERY,
            graph=GRAPH,
            inference=ant.Frequentist(bootstrap=0),
            refute="none",
            execution=ant.Execution(seed=11),
        )
    )


def test_analyze_over_parquet_equals_the_same_analysis_in_memory(tmp_path):
    data = _sample()
    path = _write(tmp_path / "rows.parquet", _with_extras(data))
    in_memory = _analyze(data)
    from_handle = _analyze(ant.data.from_parquet(path, chunk_rows=64))
    from_path = _analyze(str(path))
    from_pathlike = _analyze(path)
    for result in (from_handle, from_path, from_pathlike):
        assert math.isclose(result.effect, in_memory.effect, rel_tol=0, abs_tol=1e-12)
    assert abs(in_memory.effect - 2.0) < 0.25


def test_prepare_estimate_and_refresh_accept_files(tmp_path):
    data = _sample()
    first = _write(tmp_path / "first.parquet", data)
    shifted = {**data, "y": data["y"] + data["t"]}
    second = _write(tmp_path / "second.parquet", shifted)
    study = ant.prepare(
        first, query=QUERY, graph=GRAPH, inference=ant.Frequentist(bootstrap=0), refute="none"
    )
    baseline = require_success(ant.estimate(study))
    refreshed = require_success(ant.estimate(ant.rebind(study, data=second)))
    assert refreshed.effect == pytest.approx(baseline.effect + 1.0)
    other = require_success(ant.estimate(ant.rebind(study, data=ant.data.from_parquet(second))))
    assert other.effect == pytest.approx(refreshed.effect)


def test_schema_types_are_kept_and_nulls_are_masks(tmp_path):
    table = pa.table(
        {
            "f": pa.array([1.5, None, float("nan")], type=pa.float64()),
            "i": pa.array([1, None, 3], type=pa.int32()),
            "b": pa.array([True, None, False]),
            "ts": pa.array([1_000, None, 3_000], type=pa.timestamp("ms", tz="UTC")),
            "d": pa.array([19_000, 19_001, 19_002], type=pa.date32()),
            "cat": pa.array(["lo", None, "hi"]).dictionary_encode(),
            "s": pa.array(["x", "y", None]),
        }
    )
    path = tmp_path / "types.parquet"
    pq.write_table(table, path)
    handle = ant.data.from_parquet(path)
    kinds = {column.name: column.kind for column in handle.schema}
    assert kinds == {
        "f": "float64",
        "i": "int64",
        "b": "boolean",
        "ts": "timestamp",
        "d": "timestamp",
        "cat": "categorical",
        "s": "categorical",
    }
    assert handle.num_rows == 3
    columns = handle.read()
    assert columns["f"][0] == 1.5 and np.isnan(columns["f"][1]) and np.isnan(columns["f"][2])
    assert list(columns["i"]) == [1, None, 3]
    assert list(columns["b"]) == [True, None, False]
    assert columns["ts"][0] == np.datetime64(1_000_000_000, "ns")
    assert np.isnat(columns["ts"][1])
    assert columns["d"][0] == np.datetime64("2022-01-08", "ns")
    assert list(columns["cat"]) == ["lo", None, "hi"]
    assert list(columns["s"]) == ["x", "y", None]
    restricted = ant.data.from_parquet(path, columns=["s", "f"])
    assert restricted.names == ["s", "f"]
    assert "i" not in restricted


def test_only_the_needed_columns_are_read_and_nested_columns_are_refused(tmp_path):
    data = _sample(n=90)
    path = _write(tmp_path / "wide.parquet", _with_extras(data), row_group_size=30)
    handle = ant.data.from_parquet(path)
    nested = next(column for column in handle.schema if column.name == "nested")
    assert nested.kind == "unsupported"
    assert math.isclose(_analyze(handle).effect, _analyze(data).effect, abs_tol=1e-12)
    assert "nested" not in handle._cache, "a column the analysis does not name is not read"
    with pytest.raises(CausalValueError) as refused:
        handle.read(["nested"])
    assert refused.value.reason_code == "invalid_argument"
    assert "flatten" in (refused.value.remedy or "")


def test_content_identity_follows_values_not_layout(tmp_path):
    data = _sample()
    one = _write(tmp_path / "one.parquet", data, row_group_size=50)
    two = _write(tmp_path / "two.parquet", data, row_group_size=333)
    identity = ant.data.from_parquet(one, chunk_rows=7).content_identity()
    assert ant.data.from_parquet(two, chunk_rows=1000).content_identity() == identity
    changed = {**data, "y": data["y"].copy()}
    changed["y"][-1] += 1e-9
    three = _write(tmp_path / "three.parquet", changed)
    assert ant.data.from_parquet(three).content_identity() != identity


def test_a_small_pool_refuses_a_resident_read_and_streams_the_summary(tmp_path):
    data = _sample(n=3_000)
    weights = 0.5 + (np.arange(3_000) % 4) / 4.0
    path = _write(tmp_path / "rows.parquet", {**data, "w": weights}, row_group_size=500)
    quote = ant.data.from_parquet(path, chunk_rows=256).quote(["t", "y", "z", "w"])
    assert quote.total_bytes is not None and quote.chunk_bytes < quote.total_bytes
    small = ant.data.from_parquet(
        path,
        chunk_rows=256,
        execution=Execution(limits=ResourceLimits(memory_bytes=quote.total_bytes // 2)),
    )
    with pytest.raises(CausalResourceError) as refused:
        small.read(["t", "y", "z", "w"])
    assert "summary route" in (refused.value.remedy or "")
    summary = small.summarize_weighted_gaussian(["t", "z"], "y", weight="w", batch_rows=256)
    assert summary.design_columns == ["(intercept)", "t", "z"]
    assert summary.rows_used == 3_000 and summary.source_passes == 2
    design = np.column_stack([np.ones(3_000), data["t"], data["z"]])
    root = np.sqrt(weights)
    expected, *_ = np.linalg.lstsq(design * root[:, None], data["y"] * root, rcond=None)
    np.testing.assert_allclose(summary.coefficients, expected, rtol=1e-9, atol=1e-9)
    assert summary.robust_covariance.shape == (3, 3)


def test_the_summary_refuses_or_drops_missing_rows_as_declared(tmp_path):
    data = _sample(n=200)
    y = data["y"].astype(object)
    y[17] = None
    path = tmp_path / "missing.parquet"
    pq.write_table(
        pa.table({"t": data["t"], "z": data["z"], "y": pa.array(list(y), type=pa.float64())}), path
    )
    handle = ant.data.from_parquet(path)
    with pytest.raises(CausalValueError) as refused:
        handle.summarize_weighted_gaussian(["t", "z"], "y")
    assert "complete-case" in (refused.value.remedy or "")
    summary = handle.summarize_weighted_gaussian(["t", "z"], "y", missing="complete_case")
    assert summary.rows_dropped == 1 and summary.rows_used == 199


def test_arrow_ipc_files_and_streams(tmp_path):
    data = _sample(n=120)
    table = pa.table(data)
    file_path = tmp_path / "rows.arrow"
    with pa.OSFile(str(file_path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=50):
            writer.write_batch(batch)
    stream_path = tmp_path / "rows.arrows"
    with pa.OSFile(str(stream_path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    from_file = ant.data.from_arrow_ipc(file_path, chunk_rows=32)
    from_stream = ant.data.from_arrow_ipc(stream_path)
    assert from_file.format == "arrow_ipc_file" and from_stream.format == "arrow_ipc_stream"
    assert from_file.num_rows is None
    np.testing.assert_array_equal(from_file["y"], data["y"])
    np.testing.assert_array_equal(from_stream["z"], data["z"])
    assert from_file.content_identity() == from_stream.content_identity()
    assert math.isclose(_analyze(from_stream).effect, _analyze(data).effect, abs_tol=1e-12)


def test_a_directory_of_parquet_files_is_one_table(tmp_path):
    data = _sample(n=300)
    folder = tmp_path / "dataset"
    folder.mkdir()
    head = {name: column[:120] for name, column in data.items()}
    tail = {name: column[120:] for name, column in data.items()}
    _write(folder / "part-000.parquet", head)
    _write(folder / "part-001.parquet", tail)
    handle = ant.data.from_parquet(folder)
    assert handle.num_rows == 300 and len(handle.paths) == 2
    assert math.isclose(_analyze(folder).effect, _analyze(data).effect, abs_tol=1e-12)


def test_paths_that_are_not_tables_are_refused():
    with pytest.raises(CausalValueError) as refused:
        _analyze("rows.csv")
    assert refused.value.reason_code == "invalid_argument"
    with pytest.raises(CausalDataError):
        ant.data.from_parquet("does-not-exist.parquet")


def test_unknown_columns_are_refused_before_reading(tmp_path):
    path = _write(tmp_path / "rows.parquet", _sample(n=10))
    with pytest.raises(CausalValueError) as unknown:
        ant.data.from_parquet(path, columns=["t", "absent"])
    assert unknown.value.reason_code == "invalid_argument"


def test_loader_execution_cancels_a_read_and_owns_the_memory_limit(tmp_path):
    from antecedent import _native
    from antecedent.errors import CausalCancelledError

    data = _sample(n=2_000)
    path = _write(tmp_path / "rows.parquet", data, row_group_size=250)
    token = _native.CancellationToken()
    handle = ant.data.from_parquet(path, chunk_rows=128, execution=Execution(cancel=token))
    assert handle.read(["t"])["t"].shape == (2_000,)  # an unfired token changes nothing
    token.cancel()
    with pytest.raises(CausalCancelledError):
        handle.read(["y", "z"])
    with pytest.raises(CausalCancelledError):
        handle.content_identity()
    with pytest.raises(CausalCancelledError):
        handle.summarize_weighted_gaussian(["t"], "y")
    # A table that was opened without an execution is unaffected by that token.
    free = ant.data.from_arrow_ipc(_arrow(tmp_path, data), chunk_rows=128)
    assert free.read()["y"].shape == (2_000,)
    # The memory limit has one spelling; a loose keyword is not accepted.
    for opener in (ant.data.from_parquet, ant.data.TableFile.open):
        with pytest.raises(TypeError, match="unexpected keyword"):
            opener(path, memory_limit_bytes=1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ant.data.from_arrow_ipc(path, memory_limit_bytes=1)
    with pytest.raises(CausalValueError):
        ant.data.from_parquet(path, execution=Execution(limits=ResourceLimits(memory_bytes=0)))
    with pytest.raises(Exception, match="Execution"):
        ant.data.from_parquet(path, execution={"limits": 1})


def _arrow(tmp_path: Path, data: dict[str, np.ndarray]) -> Path:
    import pyarrow.feather as feather

    path = tmp_path / "rows.arrow"
    feather.write_feather(pa.table(data), path, compression="uncompressed")
    return path
