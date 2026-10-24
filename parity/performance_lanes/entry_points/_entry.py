"""Shared harness for the lane entry points.

Each entry point is one script that runs unchanged under two builds of the library: the
frozen predecessor (published v2.4.1, `--role control`) and the current candidate
(`--role candidate`). The data, queries, requested uncertainty, diagnostics and seeds are
built by the same code in both roles, so a pair differs only in the installed library.

Native resource measurement runs inside the timed procedure: the process peak resident set
(`getrusage`, which counts every native allocation, not Python's) is read immediately before
and after the section, together with the process CPU time and page faults, and the next
repeat starts from a fresh process when the lane runner launches one process per repeat.
When the result carries the engine's own performance record, its copy, borrow, stage and
thread figures are printed too (the frozen build has none).

Row synthesis and caller-supplied inputs are built before the timer starts. The timed
section is one complete first answer (or one complete prepared-family session) that returns
the uncertainty and diagnostics the control requests; `required` fields that come back empty
stop the run, so a point estimate alone can never be timed. Every repeat is retained and
reported; the harness never selects the fastest run and never writes a measurement record.
"""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:  # not on Windows
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

import antecedent
import numpy as np

PREDECESSOR = "2.4.1"
ROLES = ("control", "candidate")


def arguments(
    description: str, configure: Callable[[argparse.ArgumentParser], None] | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="rows of the row-tier sweep; each script names its default",
    )
    if configure is not None:
        configure(parser)
    parsed = parser.parse_args()
    if parsed.repeats < 1:
        parser.error("--repeats must be at least 1")
    if parsed.rows is not None and parsed.rows < 100:
        parser.error("--rows must be at least 100")
    return parsed


def require_build(role: str) -> str:
    """The installed build must match the role: control is v2.4.1, candidate is not."""
    version = str(antecedent.__version__)
    # The development branch keeps the predecessor's version string until release, so the
    # current build is told apart by a module only it ships.
    current = importlib.util.find_spec("antecedent.host") is not None
    frozen = version.startswith(PREDECESSOR) and not current
    if role == "control" and not frozen:
        sys.exit(f"role control needs the published {PREDECESSOR} build, found {version}")
    if role == "candidate" and frozen:
        sys.exit(f"role candidate needs the current build, found the frozen {version}")
    return version


def process_usage() -> dict[str, int] | None:
    """Peak resident set (bytes), CPU time (ns) and page faults of this process so far.

    `ru_maxrss` is the high-water mark of the whole process, native allocations included. It
    is bytes on macOS and kibibytes elsewhere.
    """
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1 if sys.platform == "darwin" else 1024
    return {
        "peak_rss_bytes": int(usage.ru_maxrss) * scale,
        "cpu_user_ns": int(usage.ru_utime * 1e9),
        "cpu_system_ns": int(usage.ru_stime * 1e9),
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
    }


_PERFORMANCE_FIELDS = (
    "copy_count",
    "bytes_borrowed",
    "worker_threads",
    "n_draws",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_ok",
    "early_stopped",
    "cancelled",
)


_FACT_PROPERTIES = (
    "observed_scans",
    "observed_copies",
    "observed_copied_bytes",
    "observed_retained_bytes",
    "observed_model_fits",
    "peak_admitted_bytes",
    "peak_admitted_threads",
    "observed_spill_bytes",
)


def engine_performance(result: Any) -> dict[str, Any]:
    """The engine's own performance record for `result`, when the build attaches one.

    Reads `performance` (copy and borrow counts, stage times, threads) and, where a result
    exposes it, the native `execution_facts` (scans, copies, copied/retained/spill bytes,
    model fits, admitted threads). A fact the engine did not observe stays absent; it is
    never filled with zero.
    """
    stack = [result]
    while stack:
        item = stack.pop()
        if isinstance(item, (tuple, list)):
            stack.extend(item)
            continue
        facts = getattr(item, "execution_facts", None)
        if facts is not None and getattr(facts, "is_native", False):
            observed = {
                f"facts_{name}": getattr(facts, name)
                for name in _FACT_PROPERTIES
                if isinstance(getattr(facts, name, None), int)
            }
            if observed:
                return observed
        performance = getattr(item, "performance", None)
        if performance is None:
            continue
        found = {
            name: getattr(performance, name)
            for name in _PERFORMANCE_FIELDS
            if isinstance(getattr(performance, name, None), (int, bool))
        }
        stages = getattr(performance, "stage_timings", None)
        if isinstance(stages, Mapping):
            found["stage_timings_ns"] = {str(k): int(v) for k, v in stages.items()}
        elif isinstance(stages, (list, tuple)) and stages:
            found["stage_timings_ns"] = {str(k): int(v) for k, v in stages}
        if found:
            return found
    return {}


def digest_arrays(columns: Mapping[str, Any]) -> str:
    """Digest of the input columns: equal digests mean identical data in both roles."""
    h = hashlib.sha256()
    for name in sorted(columns):
        h.update(name.encode())
        h.update(np.ascontiguousarray(np.asarray(columns[name], dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def digest_fields(fields: Mapping[str, Any]) -> str:
    text = json.dumps(fields, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_entry(
    control: str,
    arguments_: argparse.Namespace,
    data_digest: str,
    section: Callable[[], Any],
    fields: Callable[[Any], dict[str, Any]],
    required: Sequence[str],
) -> None:
    """Time `section` `repeats` times, check the requested outputs, print the samples."""
    version = require_build(arguments_.role)
    samples: list[int] = []
    section_usage: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    native: dict[str, Any] = {}
    for _ in range(arguments_.repeats):
        before = process_usage()
        started = time.perf_counter_ns()
        result = section()
        samples.append(time.perf_counter_ns() - started)
        after = process_usage()
        if before is not None and after is not None:
            section_usage.append(
                {
                    "peak_rss_before_bytes": before["peak_rss_bytes"],
                    "peak_rss_bytes": after["peak_rss_bytes"],
                    "cpu_user_ns": after["cpu_user_ns"] - before["cpu_user_ns"],
                    "cpu_system_ns": after["cpu_system_ns"] - before["cpu_system_ns"],
                    "minor_faults": after["minor_faults"] - before["minor_faults"],
                    "major_faults": after["major_faults"] - before["major_faults"],
                }
            )
        last = fields(result)
        native = engine_performance(result)
        absent = [name for name in required if last.get(name) in (None, (), [], {})]
        if absent:
            sys.exit(f"{control}: the timed section returned no {absent}")
    print(
        f"entry_point control={control} role={arguments_.role} version={version} "
        f"threads={arguments_.threads} data={data_digest} samples_ns={samples}"
    )
    print(f"entry_point control={control} role={arguments_.role} fields={digest_fields(last)}")
    for name in sorted(last):
        print(f"  {name}={last[name]!r}")
    # One machine-readable line per run: the lane runner parses it. The result values are
    # reduced to their digest plus the integer/string accounting fields.
    print(
        "entry_point_json "
        + json.dumps(
            {
                "control": control,
                "role": arguments_.role,
                "version": version,
                "threads": arguments_.threads,
                "rows": getattr(arguments_, "rows", None),
                "data_digest": data_digest,
                "fields_digest": digest_fields(last),
                "accounting": accounting_fields(last),
                "samples_ns": samples,
                "section_usage": section_usage,
                "engine_performance": native,
            },
            sort_keys=True,
            default=repr,
        )
    )


def accounting_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a result that hold no floating-point number: counts, statuses, labels.

    Two builds that did the same work agree on these exactly even when their numerics
    differ in the last digits, so the lane runner compares them to reject a speedup that
    came from omitted work or a changed target. Names starting with an underscore hold free
    text and are left out.
    """

    def exact(value: Any) -> bool:
        if isinstance(value, float):
            return False
        if isinstance(value, Mapping):
            return all(exact(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(exact(item) for item in value)
        return True

    return {
        name: value
        for name, value in values.items()
        if not name.startswith("_") and exact(value)
    }
