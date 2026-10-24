#!/usr/bin/env python3
"""Verify every registered control and candidate entry point before any measurement.

    python parity/performance_lanes/verify_entry_points.py                  # static
    python parity/performance_lanes/verify_entry_points.py --level import   # + both interpreters
    python parity/performance_lanes/verify_entry_points.py --level build    # + cargo prebuilds

Levels are cumulative.

static   Reads `lanes.toml` and `workloads.toml`. Every lane control has a frozen and a
         candidate command; every file the command names exists in the current tree and, when
         a staged frozen tree is present, in that tree; each Python file compiles; each Rust
         bench is registered in the current crate manifest and in the frozen one; the
         `--role` switches pair control with candidate; the frozen command of the primary
         entry is the control's `workloads.toml` command.
import   Runs every Python entry point's `--help` under the frozen interpreter (role control)
         and the candidate interpreter (role candidate), and checks that each interpreter
         holds the build its role needs. A missing library, a wrong build or a script that
         cannot parse its arguments is reported here, not in the middle of a measurement.
build    Compiles, without running, every Rust bench and example in the current tree and in
         the frozen tree (`cargo bench --no-run` / `cargo build`), each in its own target
         directory.

Exit status is 0 only when every check of the requested level passed.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from lanes_common import (
    BENCH,
    EXAMPLE,
    command_kind,
    entries_by_control,
    lane_rows,
    load_tables,
    parse_command,
    prebuild_argv,
)
from stage_frozen import DEFAULT_ROOT, Layout

LEVELS = ("static", "import", "build")
ROLE = re.compile(r"--role\s+(\w+)")


def lane_controls(lanes: dict) -> list[str]:
    return sorted(
        {name for row in lane_rows(lanes).values() for name in row["controls"]}
    )


def check_static(layout: Layout, lanes: dict, workloads: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    grouped = entries_by_control(lanes)
    staged = (layout.frozen_root / ".git").exists()
    for control in lane_controls(lanes):
        rows = grouped.get(control, [])
        if not rows:
            problems.append(f"{control}: no registered entry point")
            continue
        for entry in rows:
            where = f"{control}/{entry['surface']}"
            for label in ("frozen_command", "candidate_command"):
                try:
                    parse_command(entry[label])
                except ValueError as error:
                    problems.append(f"{where}: {label}: {error}")
            roles = [
                ROLE.search(entry[k]) for k in ("frozen_command", "candidate_command")
            ]
            if any(roles) and [m and m[1] for m in roles] != ["control", "candidate"]:
                problems.append(f"{where}: --role must pair control with candidate")
            for name in entry["files"]:
                if not (layout.repo / name).is_file():
                    problems.append(f"{where}: {name} is missing from the current tree")
                elif name.endswith(".py"):
                    try:
                        with tempfile.TemporaryDirectory() as scratch:
                            py_compile.compile(
                                str(layout.repo / name),
                                cfile=str(Path(scratch) / "x.pyc"),
                                doraise=True,
                            )
                    except py_compile.PyCompileError as error:
                        problems.append(
                            f"{where}: {name} does not compile: {error.msg}"
                        )
                if (
                    staged
                    and not name.startswith("parity/")
                    and not (layout.frozen_root / name).is_file()
                ):
                    problems.append(
                        f"{where}: {name} is missing from the staged frozen tree"
                    )
            for label, root in (
                ("candidate_command", layout.repo),
                ("frozen_command", layout.frozen_root),
            ):
                if root is layout.frozen_root and not staged:
                    continue
                match = BENCH.search(entry[label])
                if match and not EXAMPLE.search(entry[label]):
                    manifest = root / "crates" / match["crate"] / "Cargo.toml"
                    text = (
                        manifest.read_text(encoding="utf-8")
                        if manifest.is_file()
                        else ""
                    )
                    if f'name = "{match["bench"]}"' not in text:
                        problems.append(
                            f"{where}: bench {match['bench']} is not registered in {manifest}"
                        )
        wanted = workloads[control].get("command")
        if not any(row["frozen_command"] == wanted for row in rows):
            problems.append(
                f"{control}: no entry's frozen command is the workloads.toml command"
            )
    return problems


def check_import(layout: Layout, lanes: dict) -> list[str]:
    problems: list[str] = []
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); import _entry; "
        "print(_entry.require_build(sys.argv[2]))"
    )
    entry_dir = str(layout.repo / "parity" / "performance_lanes" / "entry_points")
    for role, python in (
        ("control", layout.frozen_python),
        ("candidate", layout.candidate_python),
    ):
        if not python.is_file():
            problems.append(
                f"{role}: interpreter {python} does not exist; run stage_frozen.py"
            )
            continue
        build = subprocess.run(
            [str(python), "-c", probe, entry_dir, role],
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            problems.append(
                f"{role}: wrong or missing build: {(build.stderr or build.stdout).strip()}"
            )
            continue
        for control, rows in sorted(entries_by_control(lanes).items()):
            for entry in rows:
                command = parse_command(
                    entry[
                        "candidate_command" if role == "candidate" else "frozen_command"
                    ]
                )
                kind = command_kind(command)
                if kind == "entry":
                    script = layout.repo / command.argv[1]
                elif kind == "example":
                    script = (
                        layout.repo
                        / "parity/performance_lanes/entry_points/run_example.py"
                    )
                else:
                    continue
                run = subprocess.run(
                    [str(python), str(script), "--role", role, "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=layout.repo,
                )
                if run.returncode != 0:
                    problems.append(
                        f"{control}/{role}: {script.name} --help failed: {run.stderr.strip()[-300:]}"
                    )
    return problems


def check_build(layout: Layout, lanes: dict) -> list[str]:
    problems: list[str] = []
    built: set[tuple[str, ...]] = set()
    for _control, rows in sorted(entries_by_control(lanes).items()):
        for entry in rows:
            for label, root, target in (
                ("candidate_command", layout.repo, layout.candidate_target),
                ("frozen_command", layout.frozen_root, layout.frozen_target),
            ):
                command = parse_command(entry[label])
                argv = prebuild_argv(command)
                if argv is None or (argv, str(root)) in built:
                    continue
                built.add((argv, str(root)))
                env = {
                    **os.environ,
                    **command.environment,
                    "CARGO_TARGET_DIR": str(target),
                }
                print("+ " + " ".join(argv) + f"   # {root}", flush=True)
                run = subprocess.run(argv, cwd=root, env=env, check=False)
                if run.returncode != 0:
                    problems.append(
                        f"{entry['control']}: `{' '.join(argv)}` failed in {root}"
                    )
    return problems


def verify(layout: Layout, level: str = "static") -> list[str]:
    lanes, workloads = load_tables(layout.repo)
    problems = check_static(layout, lanes, workloads)
    if level in ("import", "build"):
        problems += check_import(layout, lanes)
    if level == "build":
        problems += check_build(layout, lanes)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--level", choices=LEVELS, default="static")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    layout = Layout.under(args.work_dir.resolve())
    lanes, _ = load_tables(layout.repo)
    problems = verify(layout, args.level)
    controls = lane_controls(lanes)
    pairs = sum(len(rows) for rows in entries_by_control(lanes).values())
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print(
        f"ok ({args.level}): {len(controls)} lane controls, {pairs} control/candidate entry points"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
