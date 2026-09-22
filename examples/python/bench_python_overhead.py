"""Benchmark Python↔Rust call and data-ingest overhead.

Compares dict / pandas / Arrow inputs on a fixed interactive ``analyze``
workload (and a modest ``PCMCI`` discovery). Separates ingest+convert time
from the end-to-end native call where practical.

This is an FFI/ingest sanity check, not a Criterion replacement. Absolute
timings are machine-noise; prefer relative ordering on one machine.

Optional deps (pandas, pyarrow): skipped when unavailable.

Usage::

    python examples/python/bench_python_overhead.py
    python examples/python/bench_python_overhead.py --smoke
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import antecedent
import numpy as np
from antecedent._data import as_columns, try_as_arrow_c_columns

# Modest interactive estimate fixture (same structure as Arrow CDI smoke).
_N_ANALYZE = 600
_N_PCMCI = 200
_SEED = 7
_EDGES = [("z", "t"), ("z", "y"), ("t", "y")]


@dataclass(frozen=True)
class TimingRow:
    workload: str
    format: str
    iters: int
    ingest_ms: float
    total_ms: float
    native_est_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _confounded_dict(n: int = _N_ANALYZE, seed: int = _SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(-0.4 + 0.9 * z)))
    t = (rng.random(n) < p).astype(np.float64)
    y = 2.0 * t + z + rng.normal(scale=0.4, size=n)
    return {"t": t, "y": y, "z": z}


def _pcmci_dict(n: int = _N_PCMCI, seed: int = _SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    x = np.sin(t * 0.01) + 0.05 * rng.normal(size=n)
    y = np.zeros(n, dtype=np.float64)
    y[1:] = 0.8 * x[:-1] + 0.01 * np.cos(t[1:] * 0.03)
    return {"x": x, "y": y}


def _formats(base: Mapping[str, np.ndarray]) -> list[tuple[str, Any]]:
    """Build available input formats; pandas / Arrow are optional."""
    out: list[tuple[str, Any]] = [("dict", dict(base))]
    try:
        import pandas as pd

        out.append(("pandas", pd.DataFrame(base)))
    except ImportError:
        pass  # pandas is an optional format for this benchmark; skip it if absent
    try:
        import pyarrow as pa

        out.append(
            (
                "arrow",
                pa.table({k: pa.array(v, type=pa.float64()) for k, v in base.items()}),
            )
        )
    except ImportError:
        pass  # pyarrow is an optional format for this benchmark; skip it if absent
    return out


def _ingest(data: Any) -> tuple[list[str], list[Any]]:
    arrow = try_as_arrow_c_columns(data)
    if arrow is not None:
        return arrow
    return as_columns(data)


def _median_ms(fn: Callable[[], Any], iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(samples))


def _run_analyze(data: Any) -> None:
    result = antecedent.analyze(
        data,
        graph=_EDGES,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        latency="interactive",
        bootstrap=0,
        refute=False,
        seed=1,
    )
    if not math.isfinite(result.ate):
        raise RuntimeError("analyze returned non-finite ATE")


def _run_pcmci(data: Any) -> None:
    # Discovery coerce_data uses as_columns (copy path); still useful for
    # comparing frame conversion cost ahead of the native PCMCI call.
    result = antecedent.discovery.PCMCI(max_lag=1, alpha=0.05, fdr=False).run(data, seed=1)
    if result.ci_tests < 0:
        raise RuntimeError("pcmci returned invalid ci_tests")


def run_bench(
    *,
    iters: int = 25,
    warmup: int = 3,
    workloads: tuple[str, ...] = ("analyze", "pcmci"),
) -> list[TimingRow]:
    """Time ingest helpers and end-to-end calls; return structured rows."""
    rows: list[TimingRow] = []
    fixtures: dict[str, tuple[Mapping[str, np.ndarray], Callable[[Any], None]]] = {
        "analyze": (_confounded_dict(), _run_analyze),
        "pcmci": (_pcmci_dict(), _run_pcmci),
    }
    for workload in workloads:
        base, call = fixtures[workload]
        for fmt, data in _formats(base):
            # PCMCI's coerce_data path rejects bare Arrow tables (no to_numpy);
            # map of arrays still exercises conversion via as_columns.
            if workload == "pcmci" and fmt == "arrow":
                import pyarrow as pa

                assert isinstance(data, pa.Table)
                data = {
                    name: data.column(i).combine_chunks()
                    for i, name in enumerate(data.column_names)
                }

            ingest_ms = _median_ms(lambda d=data: _ingest(d), iters, warmup)
            total_ms = _median_ms(lambda d=data: call(d), iters, warmup)
            native_est = max(0.0, total_ms - ingest_ms)
            rows.append(
                TimingRow(
                    workload=workload,
                    format=fmt,
                    iters=iters,
                    ingest_ms=round(ingest_ms, 3),
                    total_ms=round(total_ms, 3),
                    native_est_ms=round(native_est, 3),
                )
            )
    return rows


def format_table(rows: list[TimingRow]) -> str:
    headers = ("workload", "format", "iters", "ingest_ms", "total_ms", "native_est_ms")
    body = [
        (
            r.workload,
            r.format,
            str(r.iters),
            f"{r.ingest_ms:.3f}",
            f"{r.total_ms:.3f}",
            f"{r.native_est_ms:.3f}",
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(headers)]
    lines = [
        " ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        " ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in body:
        lines.append(" ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append("")
    lines.append(
        "native_est_ms ≈ total_ms − ingest_ms (ingest also runs inside the call; "
        "treat as a lower bound on non-ingest work)."
    )
    lines.append("Not a merge gate; machine noise is expected.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[TimingRow]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="few iterations for CI / earning tests (not for timing claims)",
    )
    parser.add_argument("--iters", type=int, default=None, help="timed iterations per cell")
    parser.add_argument("--warmup", type=int, default=None, help="warmup iterations per cell")
    parser.add_argument(
        "--workload",
        choices=("analyze", "pcmci", "both"),
        default="both",
        help="which fixed workload(s) to time",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        iters, warmup = 2, 1
    else:
        iters = args.iters if args.iters is not None else 25
        warmup = args.warmup if args.warmup is not None else 3
    if args.iters is not None and args.smoke:
        iters = args.iters
    if args.warmup is not None and args.smoke:
        warmup = args.warmup
    workloads: tuple[str, ...]
    if args.workload == "both":
        workloads = ("analyze", "pcmci")
    else:
        workloads = (args.workload,)
    rows = run_bench(iters=iters, warmup=warmup, workloads=workloads)
    print(format_table(rows))
    return rows


if __name__ == "__main__":
    main()
