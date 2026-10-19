#!/usr/bin/env python3
"""Relative bench regression check over a Criterion baseline comparison.

    python3 scripts/bench_regression.py [CRITERION_DIR] [RESOURCE_DIR]

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

RESOURCE_DIR is a separate end-to-end/resource path. Each `resource.json` under
it is one attempt, or a list of attempts. Those attempts are printed after the
Criterion comparison, including unsuccessful and OOM attempts and a missed lane
target. The process exit status is only the Criterion relative change. Omitting
RESOURCE_DIR, or pointing it at a directory with no `resource.json`, does not
fail that comparison.
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
_LANES = ("first_answer", "reuse_throughput", "row_capacity", "exact_capacity")
_HARDWARE = ("apple_silicon", "x86_64_linux")
_BUILDS = ("portable", "runtime_dispatched")
_ROW_TIERS = ("1e4", "1e5", "1e6", "1e7")
_THREADS = (1, 4, 16)
_OUTCOMES = ("completed", "unsuccessful", "oom")
_SMALL_WORKLOAD_TIER = "1e4"


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
        if name.startswith("python_"):
            continue
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


def _choice(value: object, allowed: tuple, label: str) -> object:
    if isinstance(value, bool) or value not in allowed:
        names = ", ".join(str(item) for item in allowed)
        raise ValueError(f"{label} must be one of {names}")
    return value


def _relative(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("relative must be an object")
    checked: dict[str, float] = {}
    for key in ("point_estimate", "lower_bound"):
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"relative.{key} must be a number")
        checked[key] = float(number)
    return checked


def _small_workload_ceiling(relative: dict[str, float]) -> str:
    """How a 1e4 relative comparison sits against the Criterion bounds. Not an exit status."""
    lower = relative["lower_bound"]
    point = relative["point_estimate"]
    if lower > FAIL_LOWER_BOUND:
        return "beyond"
    if point > WARN_POINT and lower > 0.0:
        return "warn"
    return "within"


def resource_result(payload: dict) -> dict:
    """One end-to-end attempt. Python tracemalloc keys are ignored. OOM and misses stay."""
    if not isinstance(payload, dict):
        raise ValueError("resource record must be an object")
    control = payload.get("control")
    if not isinstance(control, str) or not control:
        raise ValueError("control must be a non-empty string")
    target_met = payload.get("target_met")
    if not isinstance(target_met, bool):
        raise ValueError("target_met must be a boolean")
    record = {
        "lane": _choice(payload.get("lane"), _LANES, "lane"),
        "control": control,
        "hardware_class": _choice(payload.get("hardware_class"), _HARDWARE, "hardware_class"),
        "build": _choice(payload.get("build"), _BUILDS, "build"),
        "threads": _choice(payload.get("threads"), _THREADS, "threads"),
        "row_tier": _choice(payload.get("row_tier"), _ROW_TIERS, "row_tier"),
        "outcome": _choice(payload.get("outcome"), _OUTCOMES, "outcome"),
        "target_met": target_met,
    }
    fields = measurement_fields(payload)
    if fields:
        record["measurement"] = fields
    if "relative" in payload:
        relative = _relative(payload["relative"])
        record["relative"] = relative
        if record["row_tier"] == _SMALL_WORKLOAD_TIER:
            record["small_workload_ceiling"] = _small_workload_ceiling(relative)
    return record


def format_resource_line(record: dict) -> str:
    """One attempt, including unsuccessful, OOM, and missed-target rows."""
    parts = [
        f"lane={record['lane']}",
        f"control={record['control']}",
        f"hardware={record['hardware_class']}",
        f"build={record['build']}",
        f"threads={record['threads']}",
        f"tier={record['row_tier']}",
        f"outcome={record['outcome']}",
        f"target_met={str(record['target_met']).lower()}",
    ]
    measurement = record.get("measurement")
    if measurement:
        parts.append("measurement=" + ",".join(sorted(measurement)))
    relative = record.get("relative")
    if relative is not None:
        parts.append(f"relative_point={relative['point_estimate']:+.1%}")
        parts.append(f"relative_lower={relative['lower_bound']:+.1%}")
    ceiling = record.get("small_workload_ceiling")
    if ceiling is not None:
        parts.append(f"small_workload_ceiling={ceiling}")
    return "resource: " + " ".join(parts)


def report_resources(root: Path) -> list[str]:
    """Lines for every attempt under `root`. A missing directory supplies none.

    Unreadable files are listed. They are not omitted, and they do not decide
    the Criterion exit status. Unsuccessful and OOM attempts are kept.
    """
    if not root.is_dir():
        return ["resource: none supplied"]
    paths = sorted(path for path in root.glob("**/resource.json") if path.is_file())
    if not paths:
        return ["resource: none supplied"]
    lines: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            lines.append(f"resource: unreadable {path}: {err}")
            continue
        items = payload if isinstance(payload, list) else [payload]
        for index, item in enumerate(items):
            try:
                lines.append(format_resource_line(resource_result(item)))
            except ValueError as err:
                lines.append(f"resource: unreadable {path}[{index}]: {err}")
    return lines or ["resource: none supplied"]


def _emit_resources(root: Path | None) -> None:
    if root is None:
        return
    for line in report_resources(root):
        print(line)


def main(argv: list[str]) -> int:
    criterion = Path(argv[0]) if argv else Path("target/criterion")
    resource_root = Path(argv[1]) if len(argv) > 1 else None
    estimates = sorted(criterion.glob("**/change/estimates.json"))
    if not estimates:
        print(f"FAIL: no baseline comparison under {criterion} (was --baseline base used?)")
        _emit_resources(resource_root)
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
            _emit_resources(resource_root)
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
        _emit_resources(resource_root)
        return 1
    print(f"bench regression: {len(estimates)} benchmark(s) compared, none beyond the bound")
    _emit_resources(resource_root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
