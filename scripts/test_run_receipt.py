#!/usr/bin/env python3
"""Record ordinary test feedback costs; never an acceptance attestation.

The child inherits its terminal and exit status. Resource data covers the whole
process tree where the host supports it; it is not a release benchmark. A dirty
or changing checkout is recorded explicitly rather than stamped as reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _repo import REPO_ROOT


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def checkout(root: Path) -> dict:
    paths = set(git(root, "diff", "--name-only", "HEAD", "-z").split(b"\0"))
    paths.update(git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"))
    changes = {}
    for raw in sorted(paths - {b""}):
        rel = os.fsdecode(raw)
        path = root / rel
        changes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "deleted"
    return {
        "commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "changed_inputs": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("an ordinary test command is required")
    started = datetime.now(timezone.utc)
    output = args.output or REPO_ROOT / "target/test-performance" / (
        started.strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}.json"
    )
    before = checkout(REPO_ROOT)
    clock = time.perf_counter()
    environment = {**os.environ, "ANTECEDENT_TEST_RECEIPT_ACTIVE": "1"}
    try:
        process = subprocess.Popen(command, env=environment)
        _, status, usage = os.wait4(process.pid, 0)
        result = os.waitstatus_to_exitcode(status)
        process.returncode = result
    except OSError as error:
        print(f"test command could not start: {error}", file=sys.stderr)
        result = 127
        usage = None
    elapsed = time.perf_counter() - clock
    after = checkout(REPO_ROOT)
    record = {
        "schema": 1,
        "purpose": "ordinary_test_feedback_measurement_not_acceptance",
        "started_utc": started.isoformat(),
        "command": command,
        "exit_code": result,
        "wall_seconds": elapsed,
        "user_seconds": usage.ru_utime if usage else None,
        "system_seconds": usage.ru_stime if usage else None,
        "peak_child_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024) if usage else None,
        "resource_scope": "host child-process accounting; peak RSS is not summed concurrent-process memory",
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "settings": {
            key: value for key, value in os.environ.items()
            if key.startswith("CARGO_PROFILE_TEST_") or key in {
                "CARGO_BUILD_JOBS", "CARGO_TARGET_DIR", "RUSTFLAGS", "RUSTUP_TOOLCHAIN",
                "NEXTEST_TEST_THREADS", "PYTEST_ADDOPTS", "ANTECEDENT_SKIP_NATIVE_BUILD",
            }
        },
        "checkout_before": before,
        "checkout_after": after,
        "checkout_unchanged": before == after,
        "calibration_attestation": False,
        "limitations": "One local run; cache warmth, contention and checkout changes affect comparison. Cargo build timings and nextest/pytest execution reports must be compared separately under matching settings.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Ordinary test run receipt: {output}", file=sys.stderr)
    return result if result >= 0 else 128 - result


if __name__ == "__main__":
    raise SystemExit(main())
