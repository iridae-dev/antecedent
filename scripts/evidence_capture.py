"""Running coverage-instrumented test binaries, shared by scientific assertions and
calibration measurements.

One run executes a Nix-built test binary with its own LLVM profile and returns its exit
status, its output and the code regions it executed. Runs go through one process-wide
queue: first come, first served, at most `CAPTURE_WORKERS` at once, so the units of
work submitted first finish (and are recorded) first. A real run executes in a spawned
worker process, because decoding a coverage export is Python work that would otherwise
hold the interpreter lock the recording thread needs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import types
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

# Half the cores: tests may use threads internally, and the earning process records
# results alongside.
CAPTURE_WORKERS = max(1, (os.cpu_count() or 2) // 2)

_LOCK = threading.Lock()
_NO_TESTS = re.compile(r"^running 0 tests$", re.M)
_PROCESSES: ProcessPoolExecutor | None = None
_THREADS: ThreadPoolExecutor | None = None


def region_coverage(export: dict) -> dict[str, set[tuple[int, int]]]:
    """Resolved source path -> (line, column) of every executed code region, from an llvm-cov export.

    A region is `[line, column, end line, end column, count, file, expanded file, kind]`;
    kind 0 is code. A function that never ran contributes nothing.
    """
    coverage: dict[str, set[tuple[int, int]]] = {}
    for block in export.get("data", []):
        for function in block.get("functions", []):
            if function.get("count", 0) <= 0:
                continue
            filenames = function.get("filenames", [])
            for region in function.get("regions", []):
                if region[4] > 0 and region[7] == 0:
                    path = str(Path(filenames[region[5]]).resolve())
                    coverage.setdefault(path, set()).add((region[0], region[1]))
    return coverage


def merge_coverage(*parts: dict[str, set[tuple[int, int]]] | None) -> dict[str, set[tuple[int, int]]]:
    """The regions any of several runs executed."""
    merged: dict[str, set[tuple[int, int]]] = {}
    for part in parts:
        for path, regions in (part or {}).items():
            merged.setdefault(path, set()).update(regions)
    return merged


def instrumented_run(binary: str, args: list[str], *, cwd: Path, env: dict[str, str],
                     tools: dict[str, str], run=subprocess.run, merge_output: bool = False,
                     coverage: bool = True) -> tuple[int, str, dict | None, str | None]:
    """Run `binary args` under its own profile: (exit status, output, executed regions, problem).

    The regions are `None`, with a problem, when the run failed or its profile could not
    be read; `merge_output` interleaves stderr into stdout as a log would."""
    with tempfile.TemporaryDirectory(prefix="antecedent-capture-") as tmp:
        raw, data = Path(tmp) / "run.profraw", Path(tmp) / "run.profdata"
        streams = ({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT} if merge_output
                   else {"capture_output": True})
        done = run([binary, *args], cwd=cwd, env={**env, "LLVM_PROFILE_FILE": str(raw)}, text=True, **streams)
        output = (done.stdout or "") + ("" if merge_output else (done.stderr or ""))
        if done.returncode != 0:
            return done.returncode, output, None, "the test binary failed"
        if not coverage:
            return 0, output, None, None
        if _NO_TESTS.search(output):
            return 0, output, None, "the test binary ran no test"
        merged = run([tools["llvm-profdata"], "merge", "-sparse", str(raw), "-o", str(data)],
                     text=True, capture_output=True)
        if merged.returncode != 0:
            return 0, output, None, f"profile merge failed\n{merged.stderr[-400:]}"
        export = run([tools["llvm-cov"], "export", f"-instr-profile={data}", binary],
                     text=True, capture_output=True)
        if export.returncode != 0:
            return 0, output, None, f"llvm-cov export failed\n{export.stderr[-400:]}"
        return 0, output, region_coverage(json.loads(export.stdout)), None


def configure(workers: int) -> None:
    """Set how many runs execute at once, before the first submission."""
    global CAPTURE_WORKERS
    with _LOCK:
        if _PROCESSES is not None or _THREADS is not None:
            if workers != CAPTURE_WORKERS:
                raise RuntimeError("the capture queue is already running")
            return
        CAPTURE_WORKERS = max(1, int(workers))


def submit(function, *args, local: bool = False, **kwargs) -> Future:
    """Queue `function(*args, **kwargs)` behind every earlier submission.

    It runs in a spawned worker process, or on a thread of this process when `local`
    (a test double that a worker process would not see) or when it cannot be sent to one."""
    global _PROCESSES, _THREADS
    remote = not local and isinstance(function, types.FunctionType) \
        and function.__qualname__ == function.__name__ \
        and all(_picklable(value) for value in (*args, *kwargs.values()))
    with _LOCK:
        if remote:
            if _PROCESSES is None:
                # Spawned, not forked: the parent runs threads.
                _PROCESSES = ProcessPoolExecutor(max_workers=CAPTURE_WORKERS, mp_context=get_context("spawn"))
            return _PROCESSES.submit(function, *args, **kwargs)
        if _THREADS is None:
            _THREADS = ThreadPoolExecutor(max_workers=CAPTURE_WORKERS, thread_name_prefix="capture")
        return _THREADS.submit(function, *args, **kwargs)


def _picklable(value) -> bool:
    if value is subprocess.run or value is None or isinstance(value, (str, int, float, bool, Path)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_picklable(v) for v in value)
    if isinstance(value, dict):
        return all(_picklable(k) and _picklable(v) for k, v in value.items())
    return False


def longest_first(items: list, seconds) -> list:
    """Order work by its last recorded duration, longest first (longest-processing-time
    scheduling), so a long unit does not start late and set the wall time; units with no
    recorded duration (`seconds(item)` is None) keep their order after the timed ones."""
    def key(entry):
        position, item = entry
        value = seconds(item)
        return (0, -value, position) if value is not None else (1, 0.0, position)
    return [item for _, item in sorted(enumerate(items), key=key)]


def record_duration(path: Path, label: str, seconds: float) -> None:
    """Append one unit's wall time (`label<TAB>seconds`); a single short append is atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(f"{label}\t{seconds:.1f}\n")


def last_durations(path: Path) -> dict[str, float]:
    """The most recent recorded wall time of each label."""
    out: dict[str, float] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            try:
                out[parts[0]] = float(parts[1])
            except (IndexError, ValueError):
                continue
    return out
