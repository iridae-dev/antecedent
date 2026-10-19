#!/usr/bin/env python3
"""Relative bench regression check over a Criterion baseline comparison.

    python3 scripts/bench_regression.py [CRITERION_DIR]

Run after `cargo bench ... --save-baseline base` on the base commit and
`cargo bench ... --baseline base` on the head commit, on the same machine in the
same job, so absolute wall times (the Apple-M1 numbers in benches/baselines/) never
enter. Criterion writes `<id>/change/estimates.json` for each benchmark of the
comparison; `mean.confidence_interval.lower_bound` is the lower end of the
confidence interval of the relative change of the mean. The check fails when that
lower bound exceeds +25 % (the interval excludes noise: the head is slower by more
than a quarter) and warns when the point estimate is beyond +10 % with a lower
bound above zero. A run that compared nothing fails: an empty comparison is not a
pass.

An optional `<bench>/measurement.json` beside `change/estimates.json` (or a
`measurement.json` in `change/`) is attached to the comparison result when it
carries native counters: peak RSS or an explicit unsupported reason,
allocation/copy/borrow bytes, source passes, fit/test counts, retained outputs,
spill bytes, thread/device use, stage times, cancellation, failure, and
variability. Keys from a Python allocator trace are ignored. Pass/fail stays
the Criterion relative change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAIL_LOWER_BOUND = 0.25
WARN_POINT = 0.10

# Native counters. A Python tracemalloc key is not one of these.
_COUNT_KEYS = (
    "peak_rss_bytes",
    "allocation_bytes",
    "copy_bytes",
    "borrow_bytes",
    "source_passes",
    "fit_count",
    "test_count",
    "retained_outputs",
    "spill_bytes",
    "threads",
)
_VARIABILITY_PARTS = ("count", "min", "median", "max")


def _nonneg_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _stage_times(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("stage_times_ns must be a list")
    stages = []
    for stage in value:
        if not isinstance(stage, dict):
            raise ValueError("stage_times_ns entries must be objects")
        name = stage.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("stage name must be a non-empty string")
        stages.append({"name": name, "nanos": _nonneg_int(stage.get("nanos"), "stage nanos")})
    return stages


def _variability(value: object) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        raise ValueError("variability must be an object")
    checked: dict[str, dict[str, int]] = {}
    for name, summary in value.items():
        if not isinstance(name, str) or not isinstance(summary, dict):
            raise ValueError("variability entries must be objects")
        checked[name] = {
            part: _nonneg_int(summary[part], f"variability.{name}.{part}")
            for part in _VARIABILITY_PARTS
        }
    return checked


def measurement_fields(payload: dict) -> dict:
    """Known native counters in `payload`. Other keys, including Python traces, are dropped."""
    if not isinstance(payload, dict):
        raise ValueError("measurement record must be an object")
    fields: dict = {}
    for key in _COUNT_KEYS:
        if key in payload:
            fields[key] = _nonneg_int(payload[key], key)
    if "peak_rss_unsupported" in payload:
        reason = payload["peak_rss_unsupported"]
        if not isinstance(reason, str) or not reason:
            raise ValueError("peak_rss_unsupported must be a non-empty string")
        fields["peak_rss_unsupported"] = reason
    if "device" in payload:
        device = payload["device"]
        if device is not None and not isinstance(device, str):
            raise ValueError("device must be a string or null")
        if device == "":
            raise ValueError("device must be a non-empty string or null")
        fields["device"] = device
    if "stage_times_ns" in payload:
        fields["stage_times_ns"] = _stage_times(payload["stage_times_ns"])
    for key in ("cancelled", "failed"):
        if key in payload:
            flag = payload[key]
            if not isinstance(flag, bool):
                raise ValueError(f"{key} must be a boolean")
            fields[key] = flag
    if "variability" in payload:
        fields["variability"] = _variability(payload["variability"])
    return fields


def measurement_beside(estimate_path: Path) -> dict | None:
    """JSON object at `<bench>/measurement.json`, else `change/measurement.json`."""
    for candidate in (
        estimate_path.parent.parent / "measurement.json",
        estimate_path.parent / "measurement.json",
    ):
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{candidate} must be a JSON object")
        return payload
    return None


def comparison_result(estimate: dict, bench: str, measurement: dict | None = None) -> dict:
    """One Criterion comparison, plus a `measurement` object when native fields are present."""
    mean = estimate["mean"]
    result = {
        "bench": bench,
        "point_estimate": float(mean["point_estimate"]),
        "lower_bound": float(mean["confidence_interval"]["lower_bound"]),
    }
    if measurement is not None:
        fields = measurement_fields(measurement)
        if fields:
            result["measurement"] = fields
    return result


def main(argv: list[str]) -> int:
    criterion = Path(argv[0]) if argv else Path("target/criterion")
    estimates = sorted(criterion.glob("**/change/estimates.json"))
    if not estimates:
        print(f"FAIL: no baseline comparison under {criterion} (was --baseline base used?)")
        return 1
    failures, warnings = [], []
    for path in estimates:
        bench = "/".join(path.parent.parent.relative_to(criterion).parts)
        try:
            result = comparison_result(
                json.loads(path.read_text(encoding="utf-8")),
                bench,
                measurement_beside(path),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
            print(f"FAIL: {bench}: {err}")
            return 1
        lower = result["lower_bound"]
        point = result["point_estimate"]
        if lower > FAIL_LOWER_BOUND:
            failures.append(f"{bench}: {point:+.1%} (lower bound {lower:+.1%})")
        elif point > WARN_POINT and lower > 0.0:
            warnings.append(f"{bench}: {point:+.1%} (lower bound {lower:+.1%})")
    for line in warnings:
        print(f"WARN: slower than the base: {line}")
    if failures:
        print(f"FAIL: {len(failures)} benchmark(s) more than {FAIL_LOWER_BOUND:.0%} slower:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"bench regression: {len(estimates)} benchmark(s) compared, none beyond the bound")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
