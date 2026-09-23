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

Each job's `map_replicates` gets `ceil(cores / jobs)` worker threads
(`ANTECEDENT_CALIBRATION_THREADS`), so `--jobs` jobs share the machine instead
of each taking all of it; the default is one job per core, one thread each.
The gate's recheck extends a first run (it computes only the replicates the
first run did not), so the plan books it at
`(RECHECK_NSIM - NSIM) / NSIM` of the first run, weighted by the share of
each suite's grid points the last measurement rechecked.

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
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate_calibration.sh"
LOG_DIR = ROOT / "target" / "calibration-records"
PREVIOUS_LOG_DIR = ROOT / "target" / "calibration-records.previous"
CONSOLE_DIR = ROOT / "target" / "calibration-console"
REGISTRY = ROOT / "parity" / "coverage_records.toml"
# Wall-clock seconds of each (group, grid point) task from earlier local runs:
# label, seconds, exit, sha, worker threads the task ran with. Older rows have
# no threads column and are read as whole-machine runs.
TIMINGS = ROOT / "target" / "calibration-timings.tsv"
CORES = os.cpu_count() or 1
# The harness default (DEFAULT_N_SIM / RECHECK_N_SIM in
# crates/antecedent/tests/common/calibration.rs), unless the gate is told otherwise.
FIRST_NSIM = int(os.environ.get("ANTECEDENT_CALIBRATION_NSIM", "400"))
RECHECK_NSIM = int(os.environ.get("ANTECEDENT_CALIBRATION_RECHECK_NSIM", "2000"))

# Groups whose 2000-replicate recheck, when it fires, has run for hours on its
# own on an M-series laptop: the recheck of
# point_derivative_bayesian_default_nominal_coverage ran for more than
# 2 h 50 min, and that of
# interventional_distribution_admg_frontdoor_bayesian_nominal_coverage 1419 s
# (recorded merged-tree measurement).
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

# Per-replicate cost of a LONG_PATTERNS design relative to the other designs
# of its suite: measured at 400 replicates, the Bayesian derivative designs of
# v110_calibration_bayesian_static run 2.4-4.6 s per replicate per run while
# its other designs run about 1 s per whole run, a ratio of roughly 50 in a
# suite's wall time. Until a local run has timed a group, a suite's recorded
# seconds are apportioned to its tests by this weight instead of evenly.
LONG_WEIGHT = 50.0

# Suite wall-clock totals of the recorded merged-tree measurement on an M-series
# laptop (release build, the `test result: ... finished in` line of each
# suite's log): (seconds, tests run, test threads). All suites ran at once, so
# these are loaded-machine numbers. A group's estimate is its weighted share
# of its suite's wall time (LONG_WEIGHT); a local run's own timing replaces it.
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


def threads_per_job(jobs: int) -> int:
    """Worker threads each job's `map_replicates` gets (ANTECEDENT_CALIBRATION_THREADS),
    so `jobs` concurrent jobs share the cores instead of each taking all of them."""
    return max(1, math.ceil(CORES / jobs))


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
    """The most recent passing time of each (group, grid point) task from earlier
    local runs, in machine-seconds: a task timed with `t` worker threads on
    `CORES` cores is scaled by `t / CORES`, so a whole-machine run and a
    one-thread run of the same task estimate the same work."""
    out: dict[str, float] = {}
    if TIMINGS.is_file():
        for line in TIMINGS.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) in (4, 5) and parts[2] == "0":
                try:
                    seconds = float(parts[1])
                    threads = int(parts[4]) if len(parts) == 5 else CORES
                except ValueError:
                    continue
                out[parts[0]] = seconds * min(threads, CORES) / CORES
    return out


def task_label(group: Group, point: int | None) -> str:
    return group.label if point is None else f"{group.label} [grid point {point}]"


def suite_weights(groups: list[Group]) -> dict[str, float]:
    """Total weight of each suite's groups in the gate (LONG_WEIGHT per long
    group, 1 per other), for apportioning a suite's recorded seconds."""
    out: dict[str, float] = {}
    for group in groups:
        out[group.head] = out.get(group.head, 0.0) + (LONG_WEIGHT if group.long else 1.0)
    return out


def estimate(
    group: Group, timings: dict[str, float], weights: dict[str, float]
) -> tuple[float | None, str]:
    """Machine-seconds over every grid point, first run only, and where the
    number comes from (None when there is no data)."""
    labels = [task_label(group, point) for point in group.points]
    if all(label in timings for label in labels):
        return sum(timings[label] for label in labels), "local run"
    measured = SUITE_SECONDS.get(group.head)
    if measured is None:
        return None, "no data"
    seconds, _tests, _threads = measured
    # The recorded sweep measured one sample size per design, every suite at once
    # on the whole machine, so a suite's wall time is machine time. It is not
    # scaled by the suite's libtest threads: those shared the machine (each test
    # already ran on every core), they did not each get a machine of their own.
    # The suite's seconds are apportioned to its groups by weight: a long
    # design costs LONG_WEIGHT of a cheap one, and an even split books every
    # cheap design at the suite average (18 one-second designs of the Bayesian
    # static suite at 0.64 h each).
    grid = GRID_COST if len(group.points) > 1 else 1.0
    if group.head in WHOLE_FILE:
        return seconds * grid, "recorded sweep x grid"
    weight = LONG_WEIGHT if group.long else 1.0
    share = weight / weights.get(group.head, weight)
    return seconds * share * grid, "recorded sweep, weighted suite share x grid"


def recheck_rates() -> dict[str, float]:
    """Share of each suite's grid points that the last measurement rechecked
    (a registry grid entry at RECHECK_NSIM replicates or more), by test file
    stem. A suite with no record has rate 0."""
    if not REGISTRY.is_file():
        return {}
    counts: dict[str, list[int]] = {}
    for rec in tomllib.loads(REGISTRY.read_text()).get("record", []):
        stem = Path(str(rec["test"]).rpartition("::")[0]).stem
        tally = counts.setdefault(stem, [0, 0])
        for point in rec.get("grid", []):
            tally[1] += 1
            tally[0] += int(int(point["replicates"]) >= RECHECK_NSIM)
    return {stem: hit / total for stem, (hit, total) in counts.items() if total}


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def print_plan(selection: Selection, groups: list[Group], jobs: int) -> None:
    timings = local_timings()
    weights = suite_weights(groups)
    rates = recheck_rates()
    threads = threads_per_job(jobs)
    known: list[float] = []
    recheck_extra = 0.0
    longest_task = 0.0
    unknown = 0
    if selection.owed_records:
        print(f"records owing a re-measurement: {selection.owed_records}")
    print(
        f"groups to run: {len(selection.groups)} of {len(groups)} "
        f"(parallel jobs: {jobs}, worker threads per job: {threads}, cores: {CORES})"
    )
    # A recheck extends the first run from FIRST_NSIM to RECHECK_NSIM replicates,
    # so it costs (RECHECK_NSIM - FIRST_NSIM) / FIRST_NSIM of the first run.
    recheck_factor = (RECHECK_NSIM - FIRST_NSIM) / FIRST_NSIM
    for g in selection.groups:
        seconds, source = estimate(g, timings, weights)
        if seconds is None:
            unknown += 1
            shown = "estimate: no data"
        else:
            known.append(seconds)
            recheck_extra += seconds * rates.get(g.head, 0.0) * recheck_factor
            # The largest grid point (factor 2 of GRID_COST) is a grid group's
            # longest task; a run-once group is one task.
            longest_task = max(
                longest_task, seconds * (2.0 / GRID_COST if len(g.points) > 1 else 1.0)
            )
            shown = f"~{_clock(seconds)} ({source})"
        long_note = "; its recheck, if it fires, can take hours" if g.long else ""
        print(f"  group {g.index}: {g.label}  [{selection.reasons[g.index]}; {shown}{long_note}]")
    if not selection.groups:
        return
    # Estimates are machine-seconds (the whole machine on one task). Each job
    # gets `threads` of the CORES cores (ANTECEDENT_CALIBRATION_THREADS), so
    # the parallel phase takes about the total work, and no schedule beats the
    # longest single task at its thread budget: wall ~ max(total, longest).
    work = sum(known)
    longest_wall = longest_task * CORES / threads
    print(
        f"rough estimate: about {_clock(work)} of machine time for the {len(known)} group(s) with "
        f"timing data, first runs only"
        + (f"; {unknown} group(s) have none" if unknown else "")
        + "."
    )
    print(
        f"expected recheck overhead: about {_clock(recheck_extra)} more (each suite's share of "
        f"grid points rechecked in the last measurement, from {REGISTRY.relative_to(ROOT)}, "
        f"times {recheck_factor:g}x its first run: a recheck extends {FIRST_NSIM} to "
        f"{RECHECK_NSIM} replicates instead of restarting)."
    )
    print(
        f"wall-clock: about {_clock(max(work + recheck_extra, longest_wall))}: total work "
        f"spread over {CORES} cores by {jobs} job(s) of {threads} thread(s), bounded below by "
        f"the longest single task at {threads} thread(s) (~{_clock(longest_wall)} before its "
        f"recheck, {1 + recheck_factor:g}x that if it fires)."
    )
    if any(g.long for g in selection.groups):
        print(
            "long groups are selected: their estimates are weighted suite shares, not "
            "measurements. In the recorded sweep (one sample size per design) the Bayesian "
            "static suite ran 4h01m, and one Bayesian derivative recheck at 2000 replicates "
            "ran for more than 2h50m on its own; the grid measures each such design at three "
            "sample sizes."
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
    threads = threads_per_job(jobs)
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
            "ANTECEDENT_CALIBRATION_THREADS",
            "ANTECEDENT_CALIBRATION_REPLICATE_START",
            "ANTECEDENT_CALIBRATION_PRIOR_TALLIES",
            "ANTECEDENT_CALIBRATION_TALLY_OUT",
        )
    }
    # Each job's `map_replicates` gets its share of the cores, so `jobs` jobs
    # fill the machine without oversubscribing it. RUST_TEST_THREADS is left
    # alone: every gate invocation runs one exact test.
    env["ANTECEDENT_CALIBRATION_THREADS"] = str(threads)
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
                timings.write(f"{label}\t{elapsed:.1f}\t{status}\t{sha}\t{threads}\n")
            verdict = "ok" if status == 0 else f"FAILED (see {console.relative_to(ROOT)})"
            print(
                f"[{_clock(time.monotonic() - started)}] {len(done)}/{len(tasks)} "
                f"group {group.index}: {verdict} in {_clock(elapsed)}"
                + (f" (rechecked at {RECHECK_NSIM} replicates)" if rechecked else "")
                + f": {label}",
                flush=True,
            )

    # Long groups first, so the ones that set the wall clock start at once.
    ordered = sorted(tasks, key=lambda t: (not t[0].long, t[0].index, t[1] or 0))
    print(f"running {len(tasks)} grid job(s), {jobs} at a time with {threads} thread(s) each")
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
        # One job per core, each on one worker thread: list-scheduling the
        # many heterogeneous tasks packs the machine better than nesting each
        # task's own parallelism. `--jobs N` gives each job ceil(cores / N)
        # threads (threads_per_job).
        p.add_argument("--jobs", type=int, default=CORES)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    groups = gate_groups()
    selection = select(groups, args.all)
    print_plan(selection, groups, args.jobs)
    if args.command == "plan":
        return 0
    return run(selection, len(groups), args.jobs)


if __name__ == "__main__":
    sys.exit(main())
