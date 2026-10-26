#!/usr/bin/env python3
"""Start from a Parquet file: analyze it, check its identity, and summarize it under a small pool.

Writes a small simulated orders table to Parquet with PyArrow, then reads it the way
a large table is read: only the columns an analysis needs are decoded, in bounded
chunks charged to an admission pool. The average effect from the file equals the
one from the same table in memory. A pool smaller than the table refuses a resident
read with its exact quote, and the weighted Gaussian summary streams the same file
one chunk at a time.

Install with `python -m pip install antecedent pyarrow`; see examples/README.md."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import antecedent
import numpy as np
from antecedent.errors import CausalResourceError


def _orders(n: int = 2_000, seed: int = 5) -> dict[str, np.ndarray]:
    """Discount (t) raises basket value (y); loyalty (z) drives both."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-0.8 * z))).astype(np.float64)
    y = 1.5 * t + 0.9 * z + rng.normal(scale=0.5, size=n)
    return {"t": t, "y": y, "z": z}


def main() -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "parquet_e2e: skipped, PyArrow is not installed (python -m pip install pyarrow)"
        )
        return

    columns = _orders()
    n = len(columns["t"])
    regions = np.array(["north", "south", "east", "west"])[np.arange(n) % 4]
    stamps = np.datetime64("2024-01-01T00:00:00") + np.arange(n).astype(
        "timedelta64[h]"
    )
    table = pa.table(
        {
            **columns,
            "region": pa.array(regions).dictionary_encode(),
            "ordered_at": pa.array(stamps.astype("datetime64[ms]")),
        }
    )

    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "orders.parquet"
        pq.write_table(table, path, row_group_size=500)

        orders = antecedent.data.from_parquet(path, chunk_rows=512)
        kinds = {column.name: column.kind for column in orders.schema}
        print("columns:", kinds)
        assert kinds["region"] == "categorical"
        assert kinds["ordered_at"] == "timestamp"
        assert orders.num_rows == n

        graph = [("z", "t"), ("z", "y"), ("t", "y")]
        query = antecedent.AverageEffect("t", "y")
        from_file = antecedent.analyze(
            orders,
            query=query,
            graph=graph,
            inference=antecedent.Frequentist(bootstrap=0),
            refute="none",
            execution=antecedent.Execution(seed=3),
        )
        in_memory = antecedent.analyze(
            columns,
            query=query,
            graph=graph,
            inference=antecedent.Frequentist(bootstrap=0),
            refute="none",
            execution=antecedent.Execution(seed=3),
        )
        print(
            f"effect from Parquet {from_file.effect:.4f}, in memory {in_memory.effect:.4f}"
        )
        assert math.isclose(
            from_file.effect, in_memory.effect, rel_tol=1e-12, abs_tol=1e-12
        )
        assert abs(from_file.effect - 1.5) < 0.2

        by_path = antecedent.analyze(
            str(path),
            query=query,
            graph=graph,
            inference=antecedent.Frequentist(bootstrap=0),
            refute="none",
            execution=antecedent.Execution(seed=3),
        )
        assert math.isclose(
            by_path.effect, in_memory.effect, rel_tol=1e-12, abs_tol=1e-12
        )

        identity = orders.content_identity(["t", "y", "z"])
        regrouped = Path(scratch) / "orders_regrouped.parquet"
        pq.write_table(table, regrouped, row_group_size=137)
        again = antecedent.data.from_parquet(regrouped, chunk_rows=100)
        assert again.content_identity(["t", "y", "z"]) == identity, (
            "layout is not content"
        )
        print("content identity:", identity[:16], "...")

        quote = orders.quote(["t", "y", "z"])
        assert quote.total_bytes is not None
        print(f"resident read of three columns: {quote.total_bytes} bytes")

        small = antecedent.data.from_parquet(
            path, chunk_rows=256, memory_limit_bytes=quote.total_bytes // 2
        )
        try:
            small.read(["t", "y", "z"])
        except CausalResourceError as refused:
            print("resident read refused:", refused.remedy)
            assert "summary route" in (refused.remedy or "")
        else:
            raise AssertionError("a pool below the quote must refuse a resident read")

        summary = small.summarize_weighted_gaussian(["t", "z"], "y", batch_rows=256)
        coefficient = summary.coefficients[summary.design_columns.index("t")]
        print(
            f"streamed summary: {summary.rows_used} rows in {summary.batches} batches, "
            f"coefficient on t {coefficient:.4f}"
        )
        assert summary.rows_used == n
        assert abs(coefficient - 1.5) < 0.2


if __name__ == "__main__":
    main()
