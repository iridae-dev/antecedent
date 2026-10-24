#!/usr/bin/env python3
"""Run the four performance and capacity lanes with one command.

    python parity/performance_lanes/run_lanes.py                 # stage, verify, run, report
    python parity/performance_lanes/run_lanes.py plan            # print every command, run nothing
    python parity/performance_lanes/run_lanes.py --lanes first_answer,reuse_throughput

`all` (the default) does, in order:

1. `prepare`  stage_frozen.py: the v2.4.1 worktree, the bench and example files the tag lacks,
              the frozen and candidate environments (see that file).
2. `verify`   verify_entry_points.py at level `build`: every registered control and candidate
              entry point imports, parses and compiles before anything is timed.
3. `run`      each control, per lane, as one OS process per repeat and per arm, interleaved so
              both arms see the same host conditions. Arms: `frozen` (published v2.4.1),
              `candidate` (this tree), `frozen_solo` (reuse lane: independent calls, the prepared
              family control only) and `scalar_direct` (recorded unsuccessful: no direct-baseline
              entry point is registered). Every repeat is retained; none is dropped.
4. `report`   `summary.json`, `summary.md` and one `resource.json` per lane, control, arm,
              tier and thread count under `<run>/resource/`, in the format
              `scripts/bench_regression.py <criterion-dir> <run>/resource` reads.

Measurement. Each process is measured by the operating system (`wait4`: peak resident set,
CPU time, faults), which counts native allocations. Python entry points also read the same
counters inside the timed section, and Criterion benches report their own mean per bench id.
The candidate's engine counters (scans, copies, fits, spill) are recorded where the result
exposes `execution_facts` or `performance`; a counter the engine did not report is absent.

Grid. Threads `--threads` (default 1,4,16); row tiers `--tiers` (default 1e4..1e7) for the
capacity lanes on controls that take a row count (`--rows`, or `USER_PATH_N`); every other
cell runs the control at its registered size under tier `control_default`. `--hardware-class`
and `--build` label the cell (`portable` or `runtime_dispatched`); the build named must be
the one installed in `--candidate-python` / compiled by cargo on this host.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lane_verdicts as verdicts
from lanes_common import (
    DEFAULT_TIER,
    ENTRY_DIR,
    LANE_IDS,
    TIERS,
    Command,
    command_kind,
    lane_rows,
    load_tables,
    parse_command,
    prebuild_argv,
    primary_entries,
    rows_mechanism,
    with_rows,
)
from measure import run_measured
from stage_frozen import DEFAULT_ROOT, Layout, stage
from verify_entry_points import verify

THREADS = (1, 4, 16)
ARMS = ("frozen", "candidate")
ENGINE_TO_MEASUREMENT = {
    "facts_observed_scans": "source_passes",
    "facts_observed_copied_bytes": "copy_bytes",
    "facts_observed_model_fits": "fit_count",
    "facts_observed_spill_bytes": "spill_bytes",
    "bytes_borrowed": "borrow_bytes",
}


@dataclass(frozen=True)
class Config:
    layout: Layout
    run_dir: Path
    lanes: tuple[str, ...]
    tiers: tuple[str, ...]
    threads: tuple[int, ...]
    repeats: int
    timeout: float | None
    hardware_class: str
    build: str
    time_factor: float


@dataclass(frozen=True)
class Cell:
    """One control run by one arm: the unit that is repeated."""

    control: str
    surface: str
    arm: str
    tier: str
    threads: int

    @property
    def key(self) -> str:
        return f"{self.control}.{self.surface}.{self.arm}.{self.tier}.t{self.threads}"


def detect_hardware_class() -> str:
    machine, system = platform.machine().lower(), platform.system()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "apple_silicon"
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "x86_64_linux"
    sys.exit(
        f"hardware class of {system}/{machine} is not registered; pass --hardware-class"
    )


def lane_tiers(config: Config, lane: str, command: Command) -> tuple[str, ...]:
    if lane in verdicts.CAPACITY_TIER_STEPS and rows_mechanism(command):
        return config.tiers
    return (DEFAULT_TIER,)


def arms_for(lane_row: dict, control: str) -> tuple[str, ...]:
    if lane_row["id"] == "reuse_throughput" and control in lane_row.get(
        "prepared_controls", []
    ):
        return (*ARMS, "frozen_solo")
    return ARMS


def plan_cells(
    config: Config, lanes: dict, workloads: dict[str, dict]
) -> dict[str, list[Cell]]:
    """Unique cells to run, per lane membership. A cell shared by two lanes runs once."""
    entries = primary_entries(lanes, workloads)
    lane_by_id = lane_rows(lanes)
    cells: dict[str, Cell] = {}
    membership: dict[str, list[Cell]] = {lane: [] for lane in config.lanes}
    for lane in config.lanes:
        row = lane_by_id[lane]
        for control in row["controls"]:
            for entry in entries[control]:
                for arm in arms_for(row, control):
                    base = parse_command(
                        entry[
                            "frozen_command"
                            if arm.startswith("frozen")
                            else "candidate_command"
                        ]
                    )
                    for tier in lane_tiers(config, lane, base):
                        for threads in config.threads:
                            cell = Cell(control, entry["surface"], arm, tier, threads)
                            cells.setdefault(cell.key, cell)
                            membership[lane].append(cells[cell.key])
    return membership


def invocation(
    config: Config, entries: dict, cell: Cell
) -> tuple[list[str], dict, Path, str]:
    """argv, environment, working directory and kind for one process of `cell`."""
    layout = config.layout
    entry = next(e for e in entries[cell.control] if e["surface"] == cell.surface)
    frozen = cell.arm.startswith("frozen")
    command = with_rows(
        parse_command(entry["frozen_command" if frozen else "candidate_command"]),
        cell.tier,
    )
    kind = command_kind(command)
    root = layout.frozen_root if frozen else layout.repo
    python = str(layout.frozen_python if frozen else layout.candidate_python)
    env = {
        **os.environ,
        **command.environment,
        "RAYON_NUM_THREADS": str(cell.threads),
        "OMP_NUM_THREADS": str(cell.threads),
        "PYTHONHASHSEED": "0",
        "CARGO_TARGET_DIR": str(
            layout.frozen_target if frozen else layout.candidate_target
        ),
    }
    role = "control" if frozen else "candidate"
    tail = ["--threads", str(cell.threads), "--repeats", "1"]
    if kind == "entry":
        script = str(layout.repo / command.argv[1])
        extra = ["--mode", "solo"] if cell.arm == "frozen_solo" else []
        return [python, script, *command.argv[2:], *tail, *extra], env, root, kind
    if kind == "example":
        wrapper = str(layout.repo / ENTRY_DIR / "run_example.py")
        return (
            [python, wrapper, "--role", role, "--example", command.argv[1], *tail],
            env,
            root,
            kind,
        )
    return list(command.argv), env, root, kind


def criterion_legs(target: Path, since: float) -> dict[str, int]:
    """Mean ns of every Criterion bench written under `target` since `since` (epoch seconds)."""
    legs: dict[str, int] = {}
    base = target / "criterion"
    for estimates in base.glob("**/new/estimates.json"):
        if estimates.stat().st_mtime >= since:
            mean = json.loads(estimates.read_text(encoding="utf-8"))["mean"][
                "point_estimate"
            ]
            legs[str(estimates.parent.parent.relative_to(base))] = int(mean)
    return legs


def run_once(config: Config, entries: dict, cell: Cell, repeat: int) -> dict:
    argv, env, cwd, kind = invocation(config, entries, cell)
    raw = config.run_dir / "raw" / cell.key
    started = time.time() - 1.0
    measured = run_measured(
        argv,
        env=env,
        cwd=cwd,
        stdout_path=raw / f"{repeat}.stdout",
        stderr_path=raw / f"{repeat}.stderr",
        timeout=config.timeout,
    )
    result = {
        "repeat": repeat,
        "argv": argv,
        **{k: v for k, v in measured.items() if k != "stderr_tail"},
    }
    if measured["outcome"] != "completed":
        result["reason"] = measured["stderr_tail"][-600:] or (
            "timed out" if measured["timed_out"] else "exited without output"
        )
        return result
    if kind in ("entry", "example"):
        lines = (raw / f"{repeat}.stdout").read_text(encoding="utf-8").splitlines()
        payload = next(
            (
                json.loads(line.split(" ", 1)[1])
                for line in lines
                if line.startswith("entry_point_json ")
            ),
            None,
        )
        if payload is None:
            result.update(
                outcome="unsuccessful",
                reason="the entry point printed no entry_point_json line",
            )
            return result
        usage = payload["section_usage"][0] if payload["section_usage"] else {}
        result.update(
            section_ns=payload["samples_ns"][0],
            data_digest=payload["data_digest"],
            accounting=payload["accounting"],
            engine_performance=payload["engine_performance"],
            section_peak_rss_bytes=usage.get("peak_rss_bytes"),
            peak_rss_before_section_bytes=usage.get("peak_rss_before_bytes"),
            fields_digest=payload["fields_digest"],
        )
        return result
    if kind == "cargo_bench":
        legs = criterion_legs(Path(env["CARGO_TARGET_DIR"]), started)
        if not legs:
            result.update(
                outcome="unsuccessful", reason="no Criterion estimates were written"
            )
            return result
        result.update(
            section_ns=sum(legs.values()),
            legs=legs,
            data_digest="criterion:" + ",".join(sorted(legs)),
            accounting={"criterion_ids": sorted(legs)},
        )
        return result
    result.update(section_ns=measured["wall_ns"], data_digest="process", accounting={})
    return result


def prebuild(config: Config, entries: dict, cells: list[Cell]) -> None:
    """Compile every cargo cell once, outside the timed processes."""
    done: set[tuple] = set()
    for cell in cells:
        argv, env, cwd, kind = invocation(config, entries, cell)
        if kind not in ("cargo_bench", "cargo_run"):
            continue
        build = prebuild_argv(parse_command(" ".join(argv)))
        if build is None or (build, str(cwd)) in done:
            continue
        done.add((build, str(cwd)))
        print("+ " + " ".join(build), flush=True)
        subprocess.run(build, cwd=cwd, env=env, check=True)


def group_key(cell: Cell) -> tuple:
    return (cell.control, cell.surface, cell.tier, cell.threads)


def run_cells(
    config: Config, entries: dict, membership: dict[str, list[Cell]]
) -> dict[str, dict]:
    unique = {cell.key: cell for cells in membership.values() for cell in cells}
    prebuild(config, entries, list(unique.values()))
    groups: dict[tuple, list[Cell]] = {}
    for cell in unique.values():
        groups.setdefault(group_key(cell), []).append(cell)
    results: dict[str, dict] = {}
    # Ascending tiers per (control, surface, threads): once every arm failed a tier, higher
    # tiers cannot say more and are recorded as not attempted.
    ordered = sorted(groups, key=lambda g: (g[0], g[1], g[3], _tier_index(g[2])))
    blocked: set[tuple] = set()
    for group in ordered:
        control, surface, tier, threads = group
        track = (control, surface, threads)
        arms = groups[group]
        if tier != DEFAULT_TIER and track in blocked:
            for cell in arms:
                results[cell.key] = {
                    "outcome": "unsuccessful",
                    "repeats": [],
                    "reason": "not attempted: a lower tier failed for every arm",
                }
            continue
        cells = {cell.key: {"outcome": "unsuccessful", "repeats": []} for cell in arms}
        for repeat in range(config.repeats):
            for cell in arms:
                one = run_once(config, entries, cell, repeat)
                print(f"{cell.key} repeat {repeat}: {one['outcome']}", flush=True)
                cells[cell.key]["repeats"].append(one)
        for cell in arms:
            reps = cells[cell.key]["repeats"]
            cells[cell.key]["outcome"] = (
                "completed"
                if all(r["outcome"] == "completed" for r in reps)
                else "oom"
                if any(r["outcome"] == "oom" for r in reps)
                else "unsuccessful"
            )
            results[cell.key] = cells[cell.key]
        if tier != DEFAULT_TIER and all(
            cells[c.key]["outcome"] != "completed" for c in arms
        ):
            blocked.add(track)
        (config.run_dir / "cells.json").write_text(
            json.dumps(results, indent=1, default=repr), encoding="utf-8"
        )
    return results


def _tier_index(tier: str) -> int:
    return TIERS.index(tier) if tier in TIERS else -1


def measurement_of(cell: dict) -> dict:
    """The `measurement` fields of a cell, from its completed repeats."""
    reps = [r for r in cell["repeats"] if r["outcome"] == "completed"]
    if not reps:
        return {"failed": True, "cancelled": False}
    peaks = [r["peak_rss_bytes"] for r in reps]
    fields: dict = {
        "peak_rss_bytes": verdicts.median(peaks),
        "stage_times_ns": [
            {
                "name": "section",
                "nanos": verdicts.median([r["section_ns"] for r in reps]),
            },
            {
                "name": "process_wall",
                "nanos": verdicts.median([r["wall_ns"] for r in reps]),
            },
        ],
        "variability": {
            "peak_rss_bytes": verdicts.spread(peaks),
            "section_ns": verdicts.spread([r["section_ns"] for r in reps]),
            "process_cpu_ns": verdicts.spread(
                [r["cpu_user_ns"] + r["cpu_system_ns"] for r in reps]
            ),
        },
        "cancelled": False,
        "failed": len(reps) != len(cell["repeats"]),
    }
    for source, target in ENGINE_TO_MEASUREMENT.items():
        values = [
            r["engine_performance"][source]
            for r in reps
            if source in r.get("engine_performance", {})
        ]
        if len(values) == len(reps) and values:
            fields[target] = verdicts.median(values)
    return fields


def verdict_for(
    config: Config,
    lane_row: dict,
    cells: dict[str, dict],
    control: str,
    surface: str,
    tier: str,
    threads: int,
    goals: dict[str, float],
) -> dict:
    def get(arm: str, at: str = tier) -> dict | None:
        return cells.get(Cell(control, surface, arm, at, threads).key)

    lane = lane_row["id"]
    if lane in verdicts.CAPACITY_TIER_STEPS:
        if tier == DEFAULT_TIER:
            return {
                "target_met": False,
                "reason": "this control takes no row count, so no row tier was run",
            }
        frozen = {t: get("frozen", t) for t in config.tiers if get("frozen", t)}
        candidate = {
            t: get("candidate", t) for t in config.tiers if get("candidate", t)
        }
        return verdicts.capacity_verdict(
            frozen,
            candidate,
            config.tiers,
            verdicts.CAPACITY_TIER_STEPS[lane],
            config.time_factor,
        )
    frozen, candidate = get("frozen"), get("candidate")
    if not (frozen and candidate):
        return {"target_met": False, "reason": "an arm has no cell"}
    if lane == "reuse_throughput" and control in lane_row.get("prepared_controls", []):
        return verdicts.reuse_verdict(
            get("frozen_solo"), frozen, candidate, goals[lane]
        )
    return verdicts.first_answer_verdict(frozen, candidate, goals[lane])


def reproducible(cell: dict) -> bool | None:
    """Whether every repeat returned the same result digest (same seed, same answer).
    None when the control prints no result digest (a Criterion bench)."""
    digests = {r["fields_digest"] for r in cell["repeats"] if r.get("fields_digest")}
    return None if not digests else len(digests) == 1


def write_records(
    config: Config,
    lanes: dict,
    workloads: dict[str, dict],
    membership: dict[str, list[Cell]],
    cells: dict[str, dict],
) -> list[dict]:
    goals = verdicts.GOALS
    lane_by_id = lane_rows(lanes)
    summary: list[dict] = []
    for lane, members in membership.items():
        row = lane_by_id[lane]
        seen: set[str] = set()
        for cell in members:
            if cell.arm != "candidate" or cell.key in seen:
                continue
            seen.add(cell.key)
            candidate = cells[cell.key]
            frozen = cells.get(
                Cell(cell.control, cell.surface, "frozen", cell.tier, cell.threads).key
            )
            verdict = verdict_for(
                config,
                row,
                cells,
                cell.control,
                cell.surface,
                cell.tier,
                cell.threads,
                goals,
            )
            relative = None
            if verdicts.completed(frozen) and verdicts.completed(candidate):
                relative = verdicts.relative_change(
                    verdicts.column(frozen, "section_ns"),
                    verdicts.column(candidate, "section_ns"),
                )
            record = {
                "lane": lane,
                "control": cell.control,
                "hardware_class": config.hardware_class,
                "build": config.build,
                "threads": cell.threads,
                "row_tier": cell.tier,
                "outcome": candidate["outcome"],
                "target_met": bool(verdict["target_met"]),
                "arm": "candidate",
                "surface": cell.surface,
                "verdict": verdict,
                "reproducible": reproducible(candidate),
                "repeats": candidate["repeats"],
                **measurement_of(candidate),
            }
            if relative is not None:
                record["relative"] = relative
            if frozen is not None:
                record["frozen"] = {
                    "outcome": frozen["outcome"],
                    "repeats": frozen["repeats"],
                    **measurement_of(frozen),
                }
            record["scalar_direct"] = {
                "outcome": "unsuccessful",
                "reason": "no scalar/direct baseline entry point is registered for this control",
            }
            target = (
                config.run_dir / "resource" / lane / f"{cell.key}" / "resource.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(record, indent=1, default=repr), encoding="utf-8"
            )
            summary.append(
                {
                    k: record[k]
                    for k in (
                        "lane",
                        "control",
                        "surface",
                        "threads",
                        "row_tier",
                        "outcome",
                        "target_met",
                    )
                }
                | {
                    "reason": verdict["reason"],
                    "relative": relative,
                    "reproducible": record["reproducible"],
                }
            )
    return summary


def bootstrap_before_after(cells: dict[str, dict]) -> list[dict]:
    """Wall-clock of the bootstrap control in the frozen build (before) and this tree (after)."""
    rows = []
    for key, cell in sorted(cells.items()):
        if (
            not key.startswith("bootstrap.")
            or ".frozen." not in key
            or not verdicts.completed(cell)
        ):
            continue
        after = cells.get(key.replace(".frozen.", ".candidate."))
        if not verdicts.completed(after):
            continue
        rows.append(
            {
                "cell": key.replace(".frozen.", ".{arm}."),
                "before_section_ns": verdicts.spread(
                    verdicts.column(cell, "section_ns")
                ),
                "after_section_ns": verdicts.spread(
                    verdicts.column(after, "section_ns")
                ),
                "before_peak_rss_bytes": verdicts.spread(
                    verdicts.column(cell, "peak_rss_bytes")
                ),
                "after_peak_rss_bytes": verdicts.spread(
                    verdicts.column(after, "peak_rss_bytes")
                ),
                "legs_before": cell["repeats"][0].get("legs"),
                "legs_after": after["repeats"][0].get("legs"),
            }
        )
    return rows


def environment_record(config: Config) -> dict:
    def git(*args: str, cwd: Path) -> str:
        done = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
        )
        return done.stdout.strip()

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "hardware_class": config.hardware_class,
        "build": config.build,
        "candidate_revision": git("rev-parse", "HEAD", cwd=config.layout.repo),
        "candidate_dirty": bool(git("status", "--porcelain", cwd=config.layout.repo)),
        "frozen_revision": git("rev-parse", "HEAD", cwd=config.layout.frozen_root)
        if config.layout.frozen_root.exists()
        else None,
    }


def write_report(config: Config, summary: list[dict], cells: dict[str, dict]) -> None:
    before_after = bootstrap_before_after(cells)
    document = {
        "environment": environment_record(config),
        "records": summary,
        "bootstrap_before_after": before_after,
    }
    (config.run_dir / "summary.json").write_text(
        json.dumps(document, indent=1, default=repr), encoding="utf-8"
    )
    lines = [
        "# Lane run",
        "",
        "| lane | control | surface | threads | tier | outcome | goal met | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['lane']} | {row['control']} | {row['surface']} | {row['threads']} | {row['row_tier']} "
            f"| {row['outcome']} | {'yes' if row['target_met'] else 'no'} | {row['reason']} |"
        )
    lines += ["", "## Bootstrap before and after", ""]
    for item in before_after:
        lines.append(
            f"- `{item['cell']}`: before {item['before_section_ns']}, after {item['after_section_ns']}"
        )
    (config.run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"report: {config.run_dir / 'summary.md'}")


def plan_text(config: Config, entries: dict, membership: dict[str, list[Cell]]) -> str:
    unique = {cell.key: cell for cells in membership.values() for cell in cells}
    out = [
        f"{len(unique)} cells x {config.repeats} repeats; lanes {', '.join(config.lanes)}"
    ]
    for cell in unique.values():
        argv, env, cwd, kind = invocation(config, entries, cell)
        out.append(f"{cell.key}  [{kind}]  cwd={cwd}\n    {' '.join(argv)}")
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=("all", "plan", "prepare", "verify", "run", "report"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=DEFAULT_ROOT, help="builds and run output"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="default: <work-dir>/runs/<timestamp>",
    )
    parser.add_argument("--lanes", default=",".join(LANE_IDS))
    parser.add_argument("--tiers", default=",".join(TIERS))
    parser.add_argument("--threads", default=",".join(str(t) for t in THREADS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--timeout", type=float, default=3600.0, help="seconds per process; 0 for none"
    )
    parser.add_argument("--hardware-class", default=None)
    parser.add_argument(
        "--build", choices=("portable", "runtime_dispatched"), default="portable"
    )
    parser.add_argument(
        "--time-factor",
        type=float,
        default=1.0,
        help="capacity lanes: candidate time at the larger tier may be this many times the frozen time",
    )
    parser.add_argument("--frozen-from-source", action="store_true")
    parser.add_argument("--skip-candidate-build", action="store_true")
    args = parser.parse_args(argv)
    lanes = tuple(name for name in args.lanes.split(",") if name)
    unknown = sorted(set(lanes) - set(LANE_IDS))
    if unknown or args.repeats < 1:
        parser.error(f"unknown lanes {unknown} or repeats < 1")
    work = args.work_dir.resolve()
    run_dir = (args.run_dir or work / "runs" / time.strftime("%Y%m%dT%H%M%S")).resolve()
    config = Config(
        layout=Layout.under(work),
        run_dir=run_dir,
        lanes=lanes,
        tiers=tuple(args.tiers.split(",")),
        threads=tuple(int(t) for t in args.threads.split(",")),
        repeats=args.repeats,
        timeout=args.timeout or None,
        hardware_class=args.hardware_class or detect_hardware_class(),
        build=args.build,
        time_factor=args.time_factor,
    )
    return args, config


def main(argv: list[str] | None = None) -> int:
    args, config = parse_args(argv)
    lanes, workloads = load_tables(config.layout.repo)
    entries = primary_entries(lanes, workloads)
    membership = plan_cells(config, lanes, workloads)
    if args.step == "plan":
        print(plan_text(config, entries, membership))
        return 0
    if args.step in ("all", "prepare"):
        stage(
            config.layout,
            frozen_from_source=args.frozen_from_source,
            build_candidate=not args.skip_candidate_build,
        )
    if args.step in ("all", "verify"):
        problems = verify(config.layout, "build")
        if problems:
            print("\n".join(f"FAIL: {p}" for p in problems))
            return 1
    if args.step in ("all", "run"):
        config.run_dir.mkdir(parents=True, exist_ok=True)
        (config.run_dir / "plan.txt").write_text(
            plan_text(config, entries, membership), encoding="utf-8"
        )
        cells = run_cells(config, entries, membership)
        (config.run_dir / "cells.json").write_text(
            json.dumps(cells, indent=1, default=repr), encoding="utf-8"
        )
    if args.step in ("all", "run", "report"):
        cells = json.loads((config.run_dir / "cells.json").read_text(encoding="utf-8"))
        summary = write_records(config, lanes, workloads, membership, cells)
        write_report(config, summary, cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
