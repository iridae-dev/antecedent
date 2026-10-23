"""Shared harness for the lane entry points.

Each entry point is one script that runs unchanged under two builds of the library: the
frozen predecessor (published v2.4.1, `--role control`) and the current candidate
(`--role candidate`). The data, queries, requested uncertainty, diagnostics and seeds are
built by the same code in both roles, so a pair differs only in the installed library.

Row synthesis and caller-supplied inputs are built before the timer starts. The timed
section is one complete first answer (or one complete prepared-family session) that returns
the uncertainty and diagnostics the control requests; `required` fields that come back empty
stop the run, so a point estimate alone can never be timed. Every repeat is retained and
reported; the harness never selects the fastest run and never writes a measurement record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import antecedent
import numpy as np

PREDECESSOR = "2.4.1"
ROLES = ("control", "candidate")


def arguments(description: str) -> argparse.Namespace:
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
    parsed = parser.parse_args()
    if parsed.repeats < 1:
        parser.error("--repeats must be at least 1")
    if parsed.rows is not None and parsed.rows < 100:
        parser.error("--rows must be at least 100")
    return parsed


def require_build(role: str) -> str:
    """The installed build must match the role: control is v2.4.1, candidate is not."""
    version = str(antecedent.__version__)
    frozen = version.startswith(PREDECESSOR)
    if role == "control" and not frozen:
        sys.exit(f"role control needs the published {PREDECESSOR} build, found {version}")
    if role == "candidate" and frozen:
        sys.exit(f"role candidate needs the current build, found the frozen {version}")
    return version


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
    last: dict[str, Any] = {}
    for _ in range(arguments_.repeats):
        started = time.perf_counter_ns()
        result = section()
        samples.append(time.perf_counter_ns() - started)
        last = fields(result)
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
