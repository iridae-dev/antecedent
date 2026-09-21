#!/usr/bin/env python3
"""The calibration gate's groups: which records each measures, which owe a run,
and running them in parallel on this machine.

`scripts/gate_calibration.sh` owns the groups, their commands and the
2000-replicate recheck, which runs inline right after a group that asks for it.
This module never edits or re-implements a group. It reads the group list from
the gate's dry run and runs each selected group alone through the unchanged
gate (`ANTECEDENT_CALIBRATION_SHARD=<index-1>/<groups>` selects exactly that
group), so the gate's recheck and its record logs in `target/calibration-records/`
are the ones `scripts/collect_coverage_records.py` reads.

Which records owe a re-measurement is decided by `scripts/calibration_facets.py`
(the only drift computation); this module maps those records to groups.

    python3 scripts/calibration_groups.py plan [--all]          # what would run, with estimates
    python3 scripts/calibration_groups.py run [--all] [--jobs N]

A coverage group that emits records is measured at every point of the
sample-size grid (`ANTECEDENT_CALIBRATION_GRID_POINT`); each (group, grid point)
pair is its own parallel job, selected in the gate with
`ANTECEDENT_CALIBRATION_GRID_POINTS=<k>`, so the three points of a long group
run side by side instead of one after another.

`scripts/measure_calibration.sh` is the one command a developer runs: it
refuses a dirty tree, runs `run`, collects the records and re-runs the
attestation gate.

Selection without `--all`:

* every group that measures a record owing a re-measurement (a facet it depends
  on drifted since its `calibration_sha` with no valid replay waiver, or that
  commit is missing from this clone);
* every group in a record-emitting suite file that has no record in the
  registry yet (a new coverage cell), whether or not anything else is owed;
* when anything above runs, the pass/fail gates that emit no record (CI Type I,
  discovery FPR, SBC, `gate_response_calibration.sh`): nothing attests them, so
  they run alongside every re-measurement rather than never.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate_calibration.sh"
LOG_DIR = ROOT / "target" / "calibration-records"
PREVIOUS_LOG_DIR = ROOT / "target" / "calibration-records.previous"
CONSOLE_DIR = ROOT / "target" / "calibration-console"
# Wall-clock seconds of each group from earlier local runs (label, seconds, exit, sha).
TIMINGS = ROOT / "target" / "calibration-timings.tsv"

# Groups whose 2000-replicate recheck, when it fires, has run for hours on its
# own on an M-series laptop: the recheck of
# point_derivative_bayesian_default_nominal_coverage ran for more than
# 2 h 50 min, and that of
# interventional_distribution_admg_frontdoor_bayesian_nominal_coverage 1419 s
# (1.10.0 merged-tree measurement).
LONG_PATTERNS = (
    "*: ade_bayesian_*",
    "*: point_derivative_bayesian_*",
    "*: point_derivative_order_2_bayesian_*",
    "*: average_derivative_bayesian_*",
    "*: semi_elasticity_*_bayesian_*",
    "*: elasticity_bayesian_*",
    "*: directional_derivative_bayesian_*",
    "*: response_jacobian_bayesian_*",
    "*: interventional_distribution_admg_frontdoor_bayesian_*",
)

# Suite wall-clock totals of the 1.10.0 merged-tree measurement on an M-series
# laptop (release build, the `test result: ... finished in` line of each
# suite's log): (seconds, tests run, test threads). All suites ran at once, so
# these are loaded-machine numbers. A group's estimate is its suite's average
# wall time per test; a local run's own timing replaces it.
SUITE_SECONDS = {
    "antecedent-estimate": (2.06, 19, 2),
    "v110_calibration_bayesian_static": (14448.34, 22, 2),
    "v110_calibration_admg": (709.34, 3, 2),
    "v110_calibration_design": (51.55, 2, 2),
    "v110_calibration_estimate": (35.83, 1, 2),  # one test when measured; two now
    "v110_calibration_response": (6.67, 6, 2),
    "v110_calibration_temporal": (2.30, 2, 2),  # two tests when measured; three now
    "v110_panel_calibration": (4.58, 4, 2),
    "v19_bayesian_temporal": (17.88, 47, 2),
    "v19_calibration": (206.66, 27, 2),
    "v19_derivative_calibration": (2032.69, 13, 2),
    "v19_static_calibration": (664.20, 25, 2),
    "v19_static_envelope_calibration": (11.65, 13, 2),
    "v19_static_mixture_calibration": (3.25, 4, 2),
    "v19_temporal_class_calibration": (452.48, 24, 2),
    "v19_temporal_frequentist": (138.58, 26, 2),
    "v19_temporal_response_calibration": (443.90, 28, 2),
}
# Suites the gate runs as one whole-file group.
WHOLE_FILE = {"v19_temporal_response_calibration", "v110_panel_calibration"}

# Sample-size grid points of a record-emitting group (`GRID_POINTS` in
# crates/antecedent/tests/common/calibration.rs; `grid_group` in the gate).
GRID_POINTS = (0, 1, 2)
# Cost of the three points relative to the base point for a design whose run
# time is linear in n: `SampleGrid::STANDARD` is n/2 + n + 2n. The heavy grid
# (n/2, n, 3n/2) costs 3.0 and the short-series grid (3n/4, n, 2n) 3.75; this
# is the estimate for all three, since the plan cannot see a design's grid.
GRID_COST = 3.5


def is_grid(label: str) -> bool:
    """Is `label` measured over the sample-size grid (the gate's `grid_group`)?"""
    if label.startswith("antecedent-estimate: bayesian_"):
        return False
    return label.startswith(("antecedent-estimate:", "v19_", "v110_", "v20_"))


@dataclass
class Group:
    index: int  # 1-based position in the gate
    label: str
    long: bool

    @property
    def head(self) -> str:
        return self.label.partition(": ")[0]

    @property
    def test_filter(self) -> str:
        return self.label.partition(": ")[2]

    @property
    def points(self) -> tuple[int | None, ...]:
        """The grid points the gate measures this group at (`None`: run once)."""
        return GRID_POINTS if is_grid(self.label) else (None,)

    @property
    def safe(self) -> str:
        """The log name the gate derives (`tr ' /:' '___'`)."""
        return self.label.translate(str.maketrans(" /:", "___"))


def is_long(label: str) -> bool:
    return any(fnmatch.fnmatchcase(label, p) for p in LONG_PATTERNS)


def gate_groups() -> list[Group]:
    env = dict(os.environ, ANTECEDENT_CALIBRATION_DRY_RUN="1")
    env.pop("ANTECEDENT_CALIBRATION_SHARD", None)
    out = subprocess.run(
        ["bash", str(GATE)], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout
    groups = []
    for line in out.splitlines():
        m = re.fullmatch(r"group (\d+): (.+)", line)
        if m:
            groups.append(Group(int(m.group(1)), m.group(2), is_long(m.group(2))))
    if not groups or [g.index for g in groups] != list(range(1, len(groups) + 1)):
        raise SystemExit("could not read the group list from the gate's dry run")
    return groups


def measures(head: str, test_filter: str, rec: dict) -> bool:
    """Does the gate group `<head>: <test_filter>` (or whole file `<head>`) run `rec`'s test?"""
    path, _, fn = str(rec["test"]).rpartition("::")
    stem = Path(path).stem
    owner = path.split("/")[1] if path.startswith("crates/") else ""
    if not test_filter:  # the group runs every test in the file
        return head == stem
    # A cargo filter matches by substring unless --exact: a superset.
    same_file = head == stem or (head == owner and "/src/" in path)
    return same_file and test_filter in fn


def _suite_file(head: str) -> bool:
    return (ROOT / "crates" / "antecedent" / "tests" / f"{head}.rs").is_file()


def _package(head: str) -> bool:
    return (ROOT / "crates" / head / "Cargo.toml").is_file()


@dataclass
class Selection:
    groups: list[Group]
    owed_records: int
    reasons: dict[int, str]  # group index -> why it runs


def select(groups: list[Group], everything: bool) -> Selection:
    """The groups to run; see the module docstring for the rule."""
    if everything:
        return Selection(groups, 0, {g.index: "--all" for g in groups})
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_facets as facets

    records = facets.load_records()
    assessments = facets.assess(facets.load_surface(), records)
    wanted = [rec for a in assessments for rec in a.stale]
    wanted += [rec for a in assessments if not a.resolved for rec in a.records]
    reasons: dict[int, str] = {}
    matched: set[str] = set()
    recordless: list[Group] = []
    for group in groups:
        if not any(measures(group.head, group.test_filter, rec) for rec in records):
            if _suite_file(group.head):
                reasons[group.index] = "no record in the registry"
            else:
                recordless.append(group)
            continue
        owing = [rec for rec in wanted if measures(group.head, group.test_filter, rec)]
        if owing:
            matched.update(str(rec["id"]) for rec in owing)
            reasons[group.index] = f"{len(owing)} record(s) owed"
    if reasons:
        for group in recordless:
            reasons[group.index] = "pass/fail gate, no record"
    for test in sorted({str(r["test"]) for r in wanted if str(r["id"]) not in matched}):
        print(f"warning: no gate group measures {test}", file=sys.stderr)
    chosen = [g for g in groups if g.index in reasons]
    return Selection(chosen, len({str(r["id"]) for r in wanted}), reasons)


# --------------------------------------------------------------------------
# Duration estimates.
# --------------------------------------------------------------------------


def local_timings() -> dict[str, float]:
    """The most recent passing wall time of each group from earlier local runs."""
    out: dict[str, float] = {}
    if TIMINGS.is_file():
        for line in TIMINGS.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 4 and parts[2] == "0":
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return out


def task_label(group: Group, point: int | None) -> str:
    return group.label if point is None else f"{group.label} [grid point {point}]"


def estimate(group: Group, timings: dict[str, float]) -> tuple[float | None, str]:
    """Seconds over every grid point and where the number comes from (None when
    there is no data)."""
    labels = [task_label(group, point) for point in group.points]
    if all(label in timings for label in labels):
        return sum(timings[label] for label in labels), "local run"
    measured = SUITE_SECONDS.get(group.head)
    if measured is None:
        return None, "no data"
    seconds, tests, threads = measured
    # The 1.10.0 sweep measured one sample size per design.
    grid = GRID_COST if len(group.points) > 1 else 1.0
    if group.head in WHOLE_FILE:
        return seconds * grid, "1.10.0 sweep x grid"
    return seconds * min(threads, tests) / tests * grid, "1.10.0 sweep, suite average x grid"


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def print_plan(selection: Selection, total: int, jobs: int) -> None:
    timings = local_timings()
    known: list[float] = []
    unknown = 0
    if selection.owed_records:
        print(f"records owing a re-measurement: {selection.owed_records}")
    print(f"groups to run: {len(selection.groups)} of {total} (parallel jobs: {jobs})")
    for g in selection.groups:
        seconds, source = estimate(g, timings)
        if seconds is None:
            unknown += 1
            shown = "estimate: no data"
        else:
            known.append(seconds)
            shown = f"~{_clock(seconds)} ({source})"
        long_note = "; its recheck, if it fires, can take hours" if g.long else ""
        print(f"  group {g.index}: {g.label}  [{selection.reasons[g.index]}; {shown}{long_note}]")
    if not selection.groups:
        return
    # A rough wall-clock bound: the jobs share the known work (each grid point
    # of a group is its own job), and no run ends before its longest grid point.
    # Rechecks are not included.
    wall = max(max(known, default=0.0) / GRID_COST, sum(known) / max(jobs, 1))
    print(
        f"rough estimate: {_clock(sum(known))} of group time, about {_clock(wall)} wall-clock "
        f"with {jobs} jobs, for the {len(known)} group(s) with timing data"
        + (f"; {unknown} group(s) have none" if unknown else "")
        + ". 2000-replicate rechecks come on top."
    )
    if any(g.long for g in selection.groups):
        print(
            "long groups are selected: suite averages understate them. In the 1.10.0 sweep (one "
            "sample size per design) the Bayesian static suite ran 4h01m, and one Bayesian "
            "derivative recheck at 2000 replicates ran for more than 2h50m on its own; the "
            "grid measures each such design at three sample sizes."
        )


# --------------------------------------------------------------------------
# Running.
# --------------------------------------------------------------------------


def prebuild(groups: list[Group]) -> int:
    """Build every selected group's test binary once, serially, so the parallel
    runs neither wait on cargo's build-directory lock nor count its time."""
    commands: list[list[str]] = []
    packages = sorted({g.head for g in groups if _package(g.head)})
    for package in packages:
        commands.append(["cargo", "test", "--release", "-p", package, "--lib", "--no-run"])
    suites = sorted({g.head for g in groups if _suite_file(g.head)})
    if suites:
        tests = [arg for suite in suites for arg in ("--test", suite)]
        commands.append(["cargo", "test", "--release", "-p", "antecedent", *tests, "--no-run"])
    for command in commands:
        print(f"== build: {' '.join(command)}", flush=True)
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            return 1
    return 0


def run(selection: Selection, total: int, jobs: int) -> int:
    groups = selection.groups
    if not groups:
        print("nothing to measure: no record owes a re-measurement")
        return 0
    # The collector stamps every log in LOG_DIR with this commit, so logs from an
    # earlier measurement must not stay there. They are moved aside, not deleted.
    if LOG_DIR.is_dir() and any(LOG_DIR.iterdir()):
        shutil.rmtree(PREVIOUS_LOG_DIR, ignore_errors=True)
        shutil.move(str(LOG_DIR), str(PREVIOUS_LOG_DIR))
        print(f"moved earlier logs to {PREVIOUS_LOG_DIR.relative_to(ROOT)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(CONSOLE_DIR, ignore_errors=True)
    CONSOLE_DIR.mkdir(parents=True)
    if prebuild(groups) != 0:
        print("FAIL: the calibration tests do not build")
        return 1
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "ANTECEDENT_CALIBRATION_NSIM",
            "ANTECEDENT_CALIBRATION_RECHECK_NSIM",
            "ANTECEDENT_CALIBRATION_DRY_RUN",
            "ANTECEDENT_CALIBRATION_GRID_POINT",
            "ANTECEDENT_CALIBRATION_GRID_POINTS",
        )
    }
    started = time.monotonic()
    lock = threading.Lock()
    done: list[int] = []
    failed: list[Group] = []

    tasks = [(group, point) for group in groups for point in group.points]

    def one(task: tuple[Group, int | None]) -> None:
        group, point = task
        label = task_label(group, point)
        suffix = "" if point is None else f".p{point}"
        console = CONSOLE_DIR / f"{group.safe}{suffix}.txt"
        task_env = {**env, "ANTECEDENT_CALIBRATION_SHARD": f"{group.index - 1}/{total}"}
        if point is not None:
            task_env["ANTECEDENT_CALIBRATION_GRID_POINTS"] = str(point)
        begin = time.monotonic()
        with lock:
            print(f"[{_clock(begin - started)}] start group {group.index}: {label}")
        with console.open("w") as out:
            status = subprocess.run(
                ["bash", str(GATE)],
                cwd=ROOT,
                env=task_env,
                stdout=out,
                stderr=subprocess.STDOUT,
            ).returncode
        elapsed = time.monotonic() - begin
        rechecked = (LOG_DIR / f"{group.safe}{suffix}.recheck.log").exists()
        with lock:
            done.append(group.index)
            if status != 0:
                failed.append(label)
            with TIMINGS.open("a") as timings:
                timings.write(f"{label}\t{elapsed:.1f}\t{status}\t{sha}\n")
            verdict = "ok" if status == 0 else f"FAILED (see {console.relative_to(ROOT)})"
            print(
                f"[{_clock(time.monotonic() - started)}] {len(done)}/{len(tasks)} "
                f"group {group.index}: {verdict} in {_clock(elapsed)}"
                + (" (rechecked at 2000 replicates)" if rechecked else "")
                + f": {label}",
                flush=True,
            )

    # Long groups first, so the ones that set the wall clock start at once.
    ordered = sorted(tasks, key=lambda t: (not t[0].long, t[0].index, t[1] or 0))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, ordered))
    print(
        f"measured {len(groups)} group(s) as {len(tasks)} grid job(s) in "
        f"{_clock(time.monotonic() - started)}"
    )
    if failed:
        print(f"FAIL: {len(failed)} grid job(s) failed the calibration gate:")
        for label in sorted(failed):
            print(f"  {label}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "list the groups that would run"), ("run", "run them")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--all", action="store_true", help="every group, not only the owed ones")
        p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    groups = gate_groups()
    selection = select(groups, args.all)
    print_plan(selection, len(groups), args.jobs)
    if args.command == "plan":
        return 0
    return run(selection, len(groups), args.jobs)


if __name__ == "__main__":
    sys.exit(main())
