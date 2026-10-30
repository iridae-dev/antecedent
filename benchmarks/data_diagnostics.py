"""Bounded diagnostics timings; run with the native extension built from this tree.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from time import perf_counter

import antecedent as ant
import numpy as np

for rows, columns in [(100_000, 4), (2_000, 128)]:
    rng = np.random.default_rng(73)
    table = {f"x{i}": rng.normal(size=rows) for i in range(columns)}
    start = perf_counter()
    report = ant.diagnose_data(table)
    print(
        f"rows={rows} columns={columns} sample={report.sample_rows} seconds={perf_counter() - start:.3f}"
    )
    assert report.sample_rows == min(rows, 4096)

rows = {"t": np.arange(10_000, dtype=float) % 2, "y": np.arange(10_000, dtype=float)}
from antecedent._batch import prepare_many

start = perf_counter()
batch = prepare_many(
    rows,
    graph=[("t", "y")],
    queries=[ant.AverageEffect("t", "y")] * 8,
    estimator="linear.adjustment.ate",
    refute=False,
)
print(f"batch_members=8 seconds={perf_counter() - start:.3f}")
assert len({report.identity for report in batch.data_diagnostics}) == 1
