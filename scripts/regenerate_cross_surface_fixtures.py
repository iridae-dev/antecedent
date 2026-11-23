#!/usr/bin/env python3
"""Regenerate cross-surface fixtures: Python writers, then Rust ignored tests.

    python3 scripts/regenerate_cross_surface_fixtures.py
    python3 scripts/regenerate_cross_surface_fixtures.py --dry-run
    python3 scripts/regenerate_cross_surface_fixtures.py --python-only
    python3 scripts/regenerate_cross_surface_fixtures.py --rust-only

Python runs first so ``py_*`` exists before any consumer. ``--dry-run`` prints
the commands and does not write fixtures.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from _repo import REPO_ROOT


def python_executable() -> str:
    venv = REPO_ROOT / "python" / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def python_steps() -> tuple[list[str], ...]:
    executable = python_executable()
    return (
        [executable, "python/tests/generate_cross_surface_fixtures.py"],
        [executable, "python/tests/generate_cross_surface_bundle_fixtures.py"],
    )


RUST_STEPS = (
    [
        "cargo",
        "test",
        "-p",
        "antecedent-io",
        "--test",
        "suite_01",
        "--",
        "--ignored",
        "--exact",
        "cross_surface_external_claim::regenerate_rust_fixtures",
    ],
    [
        "cargo",
        "test",
        "-p",
        "antecedent-design",
        "--test",
        "suite_01",
        "--",
        "--ignored",
        "--exact",
        "cross_surface_decision::regenerate_rust_fixtures",
    ],
    [
        "cargo",
        "test",
        "-p",
        "antecedent-design",
        "--test",
        "suite_01",
        "--",
        "--ignored",
        "--exact",
        "cross_surface_bundle::regenerate_rust_fixtures",
    ],
)


def _display(command: list[str], *, rust: bool) -> str:
    rendered = " ".join(command)
    if rust:
        return f"ANTECEDENT_WRITE_FIXTURES=1 {rendered}"
    return rendered


def _run(command: list[str], *, rust: bool, dry_run: bool) -> int:
    print(_display(command, rust=rust))
    if dry_run:
        return 0
    env = os.environ.copy()
    if rust:
        env["ANTECEDENT_WRITE_FIXTURES"] = "1"
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true", help="write py_* only")
    parser.add_argument("--rust-only", action="store_true", help="write rust_* only")
    parser.add_argument("--dry-run", action="store_true", help="print commands without writing")
    args = parser.parse_args(argv)
    if args.python_only and args.rust_only:
        parser.error("pass only one of --python-only and --rust-only")
    steps: list[tuple[list[str], bool]] = []
    if not args.rust_only:
        steps.extend((command, False) for command in python_steps())
    if not args.python_only:
        steps.extend((command, True) for command in RUST_STEPS)
    for command, rust in steps:
        status = _run(command, rust=rust, dry_run=args.dry_run)
        if status != 0:
            print(f"fixture regeneration failed ({status})", file=sys.stderr)
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
