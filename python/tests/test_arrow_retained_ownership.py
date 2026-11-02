"""A031: a study that retains Arrow data owns it.

PyArrow can wrap a writable NumPy array without copying. A retained study must not read
through that alias: writing the source array afterwards changes neither the study's results nor
the bytes its recorded identity describes. Call-scoped probes may still borrow.
"""

from __future__ import annotations

import gc

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalResourceError
from antecedent.outcomes import require_success

from _scm import confounded_scm

pa = pytest.importorskip("pyarrow")


def _aliased_table(data):
    # ``pa.array`` over a contiguous float64 array without nulls is zero-copy.
    return pa.table({name: pa.array(values, type=pa.float64()) for name, values in data.items()})


def _analyze(table, edges):
    return require_success(
        antecedent.analyze(
            table,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(seed=1),
        )
    )


def test_mutating_the_source_array_does_not_change_a_retained_arrow_study():
    data, edges = confounded_scm(n=500, seed=3)
    result = _analyze(_aliased_table(data), edges)
    before = result.ate

    data["y"][:] += 100.0 * data["t"]
    data["t"][::2] = 1.0 - data["t"][::2]

    assert require_success(result.study.estimate()).ate == pytest.approx(before, abs=0.0)


def test_a_fresh_study_over_the_mutated_array_sees_the_new_values():
    data, edges = confounded_scm(n=500, seed=3)
    first = _analyze(_aliased_table(data), edges)
    data["y"][:] += 100.0 * data["t"]
    second = _analyze(_aliased_table(data), edges)
    assert abs(second.ate - first.ate) > 50.0


def test_the_conversion_probe_still_borrows_for_the_duration_of_the_call():
    values = np.arange(64, dtype=np.float64)
    info = antecedent.data.load_float64_arrow_c_columns(["a"], [pa.array(values)])
    assert info.bytes_borrowed >= values.nbytes


def test_an_explicit_borrow_snapshot_reports_its_borrowed_bytes():
    values = np.arange(64, dtype=np.float64)
    borrowed = antecedent.data.table(pa.table({"a": pa.array(values)}), borrow=True)
    owned = antecedent.data.table(pa.table({"a": pa.array(values)}))
    assert borrowed.bytes_borrowed == values.nbytes
    assert owned.bytes_borrowed == 0
    assert owned.bytes_copied == values.nbytes


def _tight(table):
    # Less than the bytes the owned copy of the table retains: three float64 columns.
    return antecedent.Execution(
        seed=1, limits=antecedent.ResourceLimits(memory_bytes=table.num_rows)
    )


def test_the_owned_copy_of_arrow_data_is_funded_before_it_is_made():
    data, edges = confounded_scm(n=500, seed=3)
    table = _aliased_table(data)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    with pytest.raises(CausalResourceError):
        require_success(
            antecedent.analyze(
                table,
                graph=edges,
                query=query,
                inference=antecedent.Frequentist(bootstrap=0),
                refute=False,
                execution=_tight(table),
            )
        )
    with pytest.raises(CausalResourceError):
        antecedent.prepare(
            table,
            graph=edges,
            query=query,
            inference=antecedent.Frequentist(bootstrap=0),
            refute="none",
            execution=_tight(table),
        )
    # A pool that can fund the copy and the fit runs, and the mutation guarantee holds.
    roomy = antecedent.Execution(seed=1, limits=antecedent.ResourceLimits(memory_bytes=1 << 30))
    result = require_success(
        antecedent.analyze(
            table,
            graph=edges,
            query=query,
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=roomy,
        )
    )
    assert np.isfinite(result.ate)


def test_a_probe_retains_nothing_and_so_is_not_funded():
    values = np.arange(64, dtype=np.float64)
    info = antecedent.data.load_float64_arrow_c_columns(["a"], [pa.array(values)])
    assert info.bytes_borrowed >= values.nbytes


def test_the_lease_lives_as_long_as_the_table_whatever_its_column_types():
    # Two int64 columns, no float64 one: nothing but the table itself can carry the lease.
    rows = 8
    columns = [pa.array(np.arange(rows, dtype=np.int64)), pa.array(np.arange(rows)[::-1] + 0)]
    assert all(column.type == pa.int64() for column in columns)
    policy = antecedent.Execution(limits=antecedent.ResourceLimits(memory_bytes=256))
    snapshot = policy._call(antecedent._native._TableSnapshot, ["a", "b"], columns)
    charged = rows * 2 * 8
    assert policy._handle.available_memory_bytes == 256 - charged
    gc.collect()
    assert policy._handle.available_memory_bytes == 256 - charged
    del snapshot
    gc.collect()
    assert policy._handle.available_memory_bytes == 256


def test_the_owned_copy_of_numpy_columns_is_funded_before_it_is_made():
    data, edges = confounded_scm(n=500, seed=3)
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    def prepare(execution):
        return antecedent.prepare(
            data,
            graph=edges,
            query=query,
            inference=antecedent.Frequentist(bootstrap=0),
            refute="none",
            execution=execution,
        )

    rows = len(next(iter(data.values())))
    tight = antecedent.Execution(
        seed=1, limits=antecedent.ResourceLimits(memory_bytes=rows * len(data))
    )
    with pytest.raises(CausalResourceError, match="admission wait"):
        prepare(tight)
    assert tight._handle.available_memory_bytes == rows * len(data)
    with pytest.raises(CausalResourceError):
        require_success(
            antecedent.analyze(
                data,
                graph=edges,
                query=query,
                inference=antecedent.Frequentist(bootstrap=0),
                refute=False,
                execution=tight,
            )
        )
    assert tight._handle.available_memory_bytes == rows * len(data)


def test_the_numpy_copy_stays_charged_until_the_retained_study_is_dropped():
    data, edges = confounded_scm(n=500, seed=3)
    limit = 1 << 30
    policy = antecedent.Execution(seed=1, limits=antecedent.ResourceLimits(memory_bytes=limit))
    study = antecedent.prepare(
        data,
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        inference=antecedent.Frequentist(bootstrap=0),
        refute="none",
        execution=policy,
    )
    rows = len(next(iter(data.values())))
    gc.collect()
    assert limit - policy._handle.available_memory_bytes >= rows * len(data) * 8
    del study
    gc.collect()
    assert policy._handle.available_memory_bytes == limit
