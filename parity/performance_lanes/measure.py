"""Run one command and measure it with the operating system, not with Python.

`os.wait4` returns the kernel's resource usage for the child and the descendants it waited
for, so the peak resident set it reports includes every native allocation of an extension
module, a cargo-built bench binary or a compiled example. The same call serves the frozen
build and the candidate, because it needs nothing from the library under test.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RSS_SCALE = (
    1 if sys.platform == "darwin" else 1024
)  # ru_maxrss: bytes on macOS, KiB elsewhere
OOM_MARKERS = (
    "MemoryError",
    "memory allocation of",
    "Cannot allocate memory",
    "out of memory",
    "std::bad_alloc",
)


def tail(path: Path, limit: int = 4000) -> str:
    data = path.read_bytes() if path.exists() else b""
    return data[-limit:].decode("utf-8", errors="replace")


def classify(returncode: int, timed_out: bool, stderr_tail: str) -> str:
    """`completed`, `oom` or `unsuccessful` (the protocol's three outcomes)."""
    if timed_out:
        return "unsuccessful"
    if returncode == 0:
        return "completed"
    if returncode == -signal.SIGKILL or any(
        marker in stderr_tail for marker in OOM_MARKERS
    ):
        return "oom"
    return "unsuccessful"


def run_measured(
    argv: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float | None = None,
) -> dict:
    """Run `argv` to completion; return exit status, wall time and kernel resource usage.

    Output goes to the two files. A timeout kills the whole process group and is reported as
    `timed_out`, never as a shorter measurement.
    """
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        started = time.perf_counter_ns()
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        if timeout is None:
            _, status, usage = os.wait4(proc.pid, 0)
        else:
            deadline = time.monotonic() + timeout
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    break
                if time.monotonic() > deadline:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGKILL)
                    _, status, usage = os.wait4(proc.pid, 0)
                    break
                time.sleep(0.01)
        wall_ns = time.perf_counter_ns() - started
    # wait4 reaped the child; tell Popen so it does not wait again.
    proc.returncode = os.waitstatus_to_exitcode(status)
    stderr_tail = tail(stderr_path)
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "outcome": classify(proc.returncode, timed_out, stderr_tail),
        "wall_ns": wall_ns,
        "cpu_user_ns": int(usage.ru_utime * 1e9),
        "cpu_system_ns": int(usage.ru_stime * 1e9),
        "peak_rss_bytes": int(usage.ru_maxrss) * RSS_SCALE,
        "minor_faults": int(usage.ru_minflt),
        "major_faults": int(usage.ru_majflt),
        "stderr_tail": stderr_tail,
    }
