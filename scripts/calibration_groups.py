#!/usr/bin/env python3
"""The calibration gate's groups: which records each measures, which owe a run,
and running them in parallel on this machine.

`scripts/gate_calibration.sh` owns the groups, their commands and the
2000-replicate recheck, which runs inline right after a group that asks for it.
This module never edits or re-implements a group. It reads the group list from
the gate's dry run and runs each selected group alone through the
gate (`ANTECEDENT_CALIBRATION_SHARD=<index-1>/<groups>` selects exactly that
group), so the gate's recheck and its record logs in `target/calibration-records/`
are the ones `scripts/collect_coverage_records.py` reads.

Which records owe a re-measurement is decided by `scripts/calibration_facets.py`
(the only drift computation); this module maps those records to groups.

    python3 scripts/calibration_groups.py plan [--all]          # what would run
    python3 scripts/calibration_groups.py pilot [--all] [--jobs N]
    python3 scripts/calibration_groups.py plan [--all]          # pilot-based projection
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
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

GATE = ROOT / "scripts" / "gate_calibration.sh"
LOG_DIR = ROOT / "target" / "calibration-records"
ARCHIVE_DIR = ROOT / "target" / "calibration-archives"
CONSOLE_DIR = ROOT / "target" / "calibration-console"
REGISTRY = ROOT / "parity" / "coverage_records.toml"
GATE_REGISTRY = ROOT / "parity" / "calibration_gates.toml"
# Wall-clock seconds of each (group, grid point) task from earlier local runs:
# label, seconds, exit, sha, worker threads the task ran with. Older rows have
# no threads column and are read as whole-machine runs.
TIMINGS = ROOT / "target" / "calibration-timings.tsv"
PILOT_TIMINGS = ROOT / "target" / "calibration-pilot-timings.tsv"
PILOT_LOG_DIR = ROOT / "target" / "calibration-pilot-records"
PILOT_NSIM = 8
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

# Sample-size grid points of a record-emitting group (`GRID_POINTS` in
# crates/antecedent/tests/common/calibration.rs). Which groups are grid groups is
# the gate's `grid_group`, printed as `grid <index>` lines by its dry run.
GRID_POINTS = (0, 1, 2)

def threads_per_job(jobs: int) -> int:
    """Worker threads each job's `map_replicates` gets (ANTECEDENT_CALIBRATION_THREADS),
    so `jobs` concurrent jobs share the cores instead of each taking all of them."""
    return max(1, math.ceil(CORES / jobs))


@dataclass
class Group:
    index: int  # 1-based position in the gate
    label: str
    long: bool
    grid: bool  # measured over the sample-size grid (the gate's `grid_group`, read from its dry run)

    @property
    def head(self) -> str:
        return self.label.partition(": ")[0]

    @property
    def test_filter(self) -> str:
        return self.label.partition(": ")[2]

    @property
    def points(self) -> tuple[int | None, ...]:
        """The grid points the gate measures this group at (`None`: run once)."""
        return GRID_POINTS if self.grid else (None,)

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
    grid = {int(m.group(1)) for m in re.finditer(r"^grid (\d+)$", out, re.M)}
    for line in out.splitlines():
        m = re.fullmatch(r"group (\d+): (.+)", line)
        if m:
            index = int(m.group(1))
            groups.append(Group(index, m.group(2), is_long(m.group(2)), index in grid))
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
    # The gate labels its sampled transport integration suite by package (the
    # same convention as lib tests), while its records name the integration file.
    package_suite = head == "antecedent-estimate" and stem == "counterfactual_transport_sampled"
    same_file = head == stem or (head == owner and "/src/" in path) or package_suite
    return same_file and test_filter in fn


def measures_current_protocol(group: Group, record: dict) -> bool:
    """Normal measurements target published defaults; replay still maps old protocols."""
    if not measures(group.head, group.test_filter, record):
        return False
    if group.head == "evidence_synthesis_calibration" and group.test_filter.startswith("pooled_random_effects_"):
        return (record.get("interval_method") == "independent_gaussian_study_t"
                and record.get("role") == "conservative_point")
    if group.head == "sample_split_weighted_effect_calibration":
        marker = "discovery_fraction_0_75.conflict_resolution_treatment_invariant.variance_hc3"
        return (str(record.get("functional", "")) == "weighted_population_total_effect." + marker
                and record.get("se_kind") == "hc3_sampling_weighted"
                and record.get("interval_method") == "student_t_sandwich_effective_n"
                and str(record.get("id", "")).endswith("." + marker.replace(".", "_")))
    if group.head == "discovered_transport_calibration" and group.test_filter == "discovered_pag_transport_contrast_complete_procedure_coverage":
        marker = "declared_selection_joint_pretreatment_standardization"
        return (str(record.get("functional", "")).endswith("." + marker)
                and str(record.get("id", "")).endswith("_" + marker)
                and record.get("interval_method") == "percentile_bootstrap")
    if group.head == "sample_split_parent_set_effect_calibration":
        return ".discovery_fraction_0_75" in str(record.get("functional", ""))
    protocol = current_group_protocol(group)
    if protocol:
        if protocol not in str(record.get("functional", "")):
            return False
        if protocol == "reference=cluster_df":
            return record.get("interval_method") == "student_analytic"
        if protocol == "interval_variance=heteroskedastic_arm_linear_training_hc3":
            return (record.get("interval_method") == "population_linear_atc_hc3"
                    and record.get("se_kind") == "population_linear_heteroskedastic_hc3"
                    and str(record.get("functional", "")).startswith("untreated.mean+"))
        if protocol == "interval_variance=homoskedastic_gaussian_arm_linear_population":
            return (record.get("interval_method") == "population_linear_aipw_score"
                    and record.get("se_kind") == "population_linear_homoskedastic"
                    and "point_correction=arm_linear_projection" in str(record.get("functional", "")))
        if protocol == "weighting=balanced_linear_hc3":
            if group.test_filter == "weighted_ipw_hajek_interval":
                return (record.get("interval_method") == "balanced_population_linear_score"
                        and record.get("se_kind") == "population_linear_homoskedastic"
                        and "interval_variance=homoskedastic_gaussian_arm_linear_population"
                        in str(record.get("functional", "")))
            return record.get("interval_method") == "balanced_population_score"
        if protocol in {"matching=population_linear", "stratification=regression_linear_hc3"}:
            return str(record.get("id", "")).endswith("." + protocol.partition("=")[2])
    return True


def current_group_protocol(group: Group) -> str | None:
    """Construction identity required before old test names count as measured."""
    if group.head == "weighted_interval_calibration":
        return {
            "weighted_aipw_cluster_dml_interval": "reference=cluster_df",
            "weighted_dml_cluster_interval": "reference=cluster_df",
            "weighted_dr_learner_cluster_interval": "reference=cluster_df",
            "weighted_aipw_atc_interval": "interval_variance=heteroskedastic_arm_linear_training_hc3",
            "weighted_aipw_population_linear_interval": "interval_variance=homoskedastic_gaussian_arm_linear_population",
            "weighted_dml_aipw_population_linear_interval": "interval_variance=homoskedastic_gaussian_arm_linear_population",
            "weighted_dr_learner_population_linear_interval": "interval_variance=homoskedastic_gaussian_arm_linear_population",
            "weighted_matching_interval": "matching=population_linear",
            "weighted_distance_matching_interval": "matching=population_linear",
            "weighted_stratification_interval": "stratification=regression_linear_hc3",
            "weighted_ipw_hajek_interval": "weighting=balanced_linear_hc3",
            "weighted_ipw_hajek_cluster_interval": "weighting=balanced_linear_hc3",
        }.get(group.test_filter)
    if group.head == "knn_matching_calibration" and group.test_filter == "knn_matching_interval_ci_coverage":
        return "population_linear"
    return None


@cache
def _workspace_packages(root: Path) -> tuple[tuple[str, Path], ...]:
    """Actual Cargo member names and directories, including member globs."""
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]
    excluded = {path for pattern in workspace.get("exclude", []) for path in root.glob(pattern)}
    members = {path for pattern in workspace["members"] for path in root.glob(pattern)} - excluded
    packages = []
    for member in sorted(members):
        manifest = tomllib.loads((member / "Cargo.toml").read_text(encoding="utf-8"))
        if "package" in manifest:
            packages.append((manifest["package"]["name"], member))
    return tuple(packages)


@cache
def _suite_owner(root: Path, head: str) -> str | None:
    owners = []
    for package, member in _workspace_packages(root):
        manifest = tomllib.loads((member / "Cargo.toml").read_text(encoding="utf-8"))
        explicit = any(
            test.get("name") == head
            and (member / test.get("path", f"tests/{head}.rs")).is_file()
            for test in manifest.get("test", [])
        )
        automatic = manifest["package"].get("autotests", True) and any(
            path.is_file() for path in (member / "tests" / f"{head}.rs", member / "tests" / head / "main.rs")
        )
        if explicit or automatic:
            owners.append(package)
    if len(owners) > 1:
        raise SystemExit(f"ambiguous calibration test target {head!r}: {', '.join(owners)}")
    return owners[0] if owners else None


@cache
def _suite_features(root: Path, package: str, suite: str) -> tuple[str, ...]:
    """The Cargo features a test target declares as required, in manifest order."""
    for name, member in _workspace_packages(root):
        if name != package:
            continue
        manifest = tomllib.loads((member / "Cargo.toml").read_text(encoding="utf-8"))
        for test in manifest.get("test", []):
            if test.get("name") == suite:
                return tuple(test.get("required-features", []))
    return ()


def _suite_file(head: str) -> bool:
    return _suite_owner(ROOT, head) is not None


def _package(head: str) -> bool:
    return any(package == head for package, _ in _workspace_packages(ROOT))


@dataclass
class Selection:
    groups: list[Group]
    owed_records: int
    reasons: dict[int, str]  # group index -> why it runs


def select(groups: list[Group], everything: bool, missing_only: bool = False) -> Selection:
    """The groups to run; see the module docstring for the rule."""
    if everything:
        return Selection(groups, 0, {g.index: "--all" for g in groups})
    ensure_scripts_on_path()
    import calibration_facets as facets

    records = facets.load_records()
    if missing_only:
        ledger = tomllib.loads(GATE_REGISTRY.read_text(encoding="utf-8")) if GATE_REGISTRY.is_file() else {}
        passed = {str(row["group"]): row for row in ledger.get("gate", []) if int(row.get("passed", 0)) > 0}
        chosen = []
        reasons = {}
        for group in groups:
            if any(measures_current_protocol(group, rec) for rec in records):
                continue
            names = [group.safe + ("" if point is None else f".p{point}") for point in group.points]
            # A repeated test name does not prove a revised construction. Parent
            # and weighted record emitters must carry their current identity;
            # recordless KNN must have emitted its explicit protocol marker.
            protocol = current_group_protocol(group)
            record_required = group.head in {"sample_split_parent_set_effect_calibration", "sample_split_weighted_effect_calibration"} or (
                group.head == "weighted_interval_calibration" and protocol is not None
            )
            record_required = record_required or (
                group.head == "discovered_transport_calibration" and
                group.test_filter == "discovered_pag_transport_contrast_complete_procedure_coverage"
            )
            if not record_required and all(
                name in passed and (protocol is None or passed[name].get("protocol") == protocol)
                for name in names
            ):
                continue
            chosen.append(group)
            reasons[group.index] = "no coverage record or complete passed gate ledger"
        return Selection(chosen, 0, reasons)
    assessments = facets.assess(facets.load_surface(), records)
    wanted = [rec for a in assessments for rec in a.stale]
    wanted += [rec for a in assessments if not a.resolved for rec in a.records]
    reasons: dict[int, str] = {}
    matched: set[str] = set()
    recordless: list[Group] = []
    for group in groups:
        if not any(measures_current_protocol(group, rec) for rec in records):
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


def filter_selection(selection: Selection, patterns: list[str]) -> Selection:
    """Restrict the selected work with full group-label globs; typos fail closed."""
    if not patterns:
        return selection
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(group.label, pattern) for group in selection.groups):
            raise SystemExit(f"--group {pattern!r} matches no selected group")
    chosen = [group for group in selection.groups
              if any(fnmatch.fnmatchcase(group.label, pattern) for pattern in patterns)]
    return Selection(chosen, selection.owed_records,
                     {g.index: selection.reasons[g.index] for g in chosen})


# --------------------------------------------------------------------------
# Duration estimates.
# --------------------------------------------------------------------------


def pilot_timings() -> dict[str, float]:
    """Per-task first-pass machine seconds projected from a smoke pilot at HEAD.

    The pilot exercises the real estimator and grid point with fewer replicates.
    Its output is never coverage evidence. The projection is approximate because
    fixed process startup and replicate-dependent convergence are not linear.
    """
    out: dict[str, float] = {}
    if not PILOT_TIMINGS.is_file():
        return out
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for line in PILOT_TIMINGS.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 7 or parts[3] != sha or parts[2] != "0":
            continue
        try:
            elapsed, threads, replicates, test_seconds = (
                float(parts[1]), int(parts[4]), int(parts[5]), float(parts[6])
            )
        except ValueError:
            continue
        if elapsed > 0 and 0 <= test_seconds <= elapsed and threads > 0 and replicates == PILOT_NSIM:
            # Scale the test's replicate work, but pay Cargo/gate startup only
            # once. The old projection multiplied the startup of ~1,100 jobs
            # by 50 and substantially overstated their full-pass cost.
            projected = test_seconds * FIRST_NSIM / replicates + elapsed - test_seconds
            out[parts[0]] = projected * min(threads, CORES) / CORES
    return out


def task_label(group: Group, point: int | None) -> str:
    return group.label if point is None else f"{group.label} [grid point {point}]"


def estimate(group: Group, timings: dict[str, float]) -> tuple[float | None, str]:
    """Machine-seconds over every grid point, first run only, and where the
    number comes from (None when there is no data)."""
    labels = [task_label(group, point) for point in group.points]
    if all(label in timings for label in labels):
        return sum(timings[label] for label in labels), "pilot at HEAD"
    # Historical whole-suite wall times were measured while many suites shared
    # the same CPU. Treating each as exclusive machine time inflated the old
    # full-pass forecast to 28 hours. There is no defensible per-group estimate
    # until the exact group and grid point have a pilot timing at this commit.
    return None, "no pilot timing"


def recheck_rates() -> dict[str, float]:
    """Share of each suite's grid points that the last measurement rechecked
    (a registry grid entry at RECHECK_NSIM replicates or more), by test file
    stem. A suite with no record has rate 0."""
    if not REGISTRY.is_file():
        return {}
    counts: dict[str, list[int]] = {}
    for rec in tomllib.loads(REGISTRY.read_text(encoding="utf-8")).get("record", []):
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
    timings = pilot_timings()
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
        f"(parallel jobs: {jobs}, worker threads per job: {threads}, cores: {CORES}; first-pass replicates: {FIRST_NSIM})"
    )
    # A recheck extends the first run from FIRST_NSIM to RECHECK_NSIM replicates,
    # so it costs (RECHECK_NSIM - FIRST_NSIM) / FIRST_NSIM of the first run.
    recheck_factor = (RECHECK_NSIM - FIRST_NSIM) / FIRST_NSIM if FIRST_NSIM < 2000 else 0.0
    for g in selection.groups:
        seconds, source = estimate(g, timings)
        if seconds is None:
            unknown += 1
            shown = "estimate: no data"
        else:
            known.append(seconds)
            recheck_extra += seconds * rates.get(g.head, 0.0) * recheck_factor
            longest_task = max(longest_task, *(timings[task_label(g, p)] for p in g.points))
            shown = f"~{_clock(seconds)} ({source})"
        long_note = "; its recheck, if it fires, can take hours" if g.long else ""
        print(f"  group {g.index}: {g.label}  [{selection.reasons[g.index]}; {shown}{long_note}]")
    if not selection.groups:
        return
    if unknown:
        print(
            f"No full-pass duration estimate: {unknown} group(s) lack a pilot at HEAD. "
            "Run `bash scripts/measure_calibration.sh --pilot` before budgeting the measurement."
        )
        return
    # Estimates are machine-seconds (the whole machine on one task). Each job
    # gets `threads` of the CORES cores (ANTECEDENT_CALIBRATION_THREADS), so
    # the parallel phase takes about the total work, and no schedule beats the
    # longest single task at its thread budget: wall ~ max(total, longest).
    work = sum(known)
    longest_wall = longest_task * CORES / threads
    print(
        f"rough first-pass projection: about {_clock(work)} of machine time for the {len(known)} group(s) with "
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
        f"wall-clock projection: about {_clock(max(work + recheck_extra, longest_wall))}: total work "
        f"spread over {CORES} cores by {jobs} job(s) of {threads} thread(s), bounded below by "
        f"the longest single task at {threads} thread(s) (~{_clock(longest_wall)} before its "
        f"recheck, {1 + recheck_factor:g}x that if it fires)."
    )
    if any(g.long for g in selection.groups):
        print(
            "This is a projection, not a measured full-pass duration. Run "
            "`bash scripts/measure_calibration.sh --pilot` for timings of the "
            "selected groups on this commit; old loaded-machine suite times "
            "can greatly overstate the duration. Rechecks and convergence "
            "can still vary from the pilot."
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
    # A suite that declares required features is built with exactly those features,
    # the way scripts/gate_calibration.sh runs it, so the binary is reused and the
    # package's other suites are not rebuilt with a different feature set.
    suites_by_package: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for group in groups:
        owner = _suite_owner(ROOT, group.head)
        if owner is not None:
            features = _suite_features(ROOT, owner, group.head)
            suites_by_package.setdefault((owner, features), set()).add(group.head)
    for (package, features), suites in sorted(suites_by_package.items()):
        tests = [arg for suite in sorted(suites) for arg in ("--test", suite)]
        feature_args = ["--features", ",".join(features)] if features else []
        commands.append(
            ["cargo", "test", "--release", "-p", package, *feature_args, *tests, "--no-run"]
        )
    for command in commands:
        print(f"== build: {' '.join(command)}", flush=True)
        if subprocess.run(command, cwd=ROOT).returncode != 0:
            return 1
    return 0


def pilot(selection: Selection, total: int, jobs: int) -> int:
    """Time selected groups with smoke replicates, isolated from coverage logs."""
    groups = selection.groups
    if not groups:
        print("nothing to pilot")
        return 0
    if prebuild(groups) != 0:
        return 1
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    shutil.rmtree(PILOT_LOG_DIR, ignore_errors=True)
    PILOT_LOG_DIR.mkdir(parents=True)
    PILOT_TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    tasks = [(group, point) for group in groups for point in group.points]
    lock = threading.Lock()
    failed: list[str] = []
    finished = 0
    started = time.monotonic()
    base_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("ANTECEDENT_CALIBRATION_") and k not in {"ANTECEDENT_PARENT_PROTOCOL_VERSION", "ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION", "ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL", "ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL"}
    }
    base_env.update({
        "ANTECEDENT_CALIBRATION_SMOKE": "1",
        "ANTECEDENT_CALIBRATION_NSIM": str(PILOT_NSIM),
        "ANTECEDENT_CALIBRATION_RECHECK_NSIM": str(PILOT_NSIM + 1),
        "ANTECEDENT_CALIBRATION_THREADS": "1",
        "ANTECEDENT_CALIBRATION_LOG_DIR": str(PILOT_LOG_DIR),
    })

    def one(task: tuple[Group, int | None]) -> None:
        nonlocal finished
        group, point = task
        label = task_label(group, point)
        suffix = "" if point is None else f".p{point}"
        console = PILOT_LOG_DIR / f"{group.safe}{suffix}.console.txt"
        task_env = {**base_env, "ANTECEDENT_CALIBRATION_SHARD": f"{group.index - 1}/{total}"}
        if point is not None:
            task_env["ANTECEDENT_CALIBRATION_GRID_POINTS"] = str(point)
        begin = time.monotonic()
        with console.open("w", encoding="utf-8", newline="\n") as out:
            status = subprocess.run(
                ["bash", str(GATE)], cwd=ROOT, env=task_env,
                stdout=out, stderr=subprocess.STDOUT,
            ).returncode
        elapsed = time.monotonic() - begin
        log = PILOT_LOG_DIR / f"{group.safe}{suffix}.log"
        test_seconds = sum(
            float(value) for value in re.findall(
                r"^test result: ok\..*?finished in ([0-9.]+)s",
                log.read_text(encoding="utf-8") if log.is_file() else "",
                re.M,
            )
        )
        # Libtest rounds very short tests to 0.00s. A small positive floor
        # avoids declaring their future 400-replicate work to be exactly zero.
        test_seconds = min(elapsed, max(0.01, test_seconds))
        with lock:
            finished += 1
            if status != 0:
                failed.append(label)
            with PILOT_TIMINGS.open("a", encoding="utf-8", newline="\n") as timings:
                timings.write(
                    f"{label}\t{elapsed:.2f}\t{status}\t{sha}\t1\t{PILOT_NSIM}"
                    f"\t{test_seconds:.2f}\n"
                )
            print(
                f"[{_clock(time.monotonic() - started)}] pilot {finished}/{len(tasks)}: "
                f"{label}: {_clock(elapsed)} {'ok' if status == 0 else 'FAILED'}",
                flush=True,
            )

    PILOT_TIMINGS.write_text("", encoding="utf-8", newline="\n")
    print(f"piloting {len(tasks)} grid job(s) with {PILOT_NSIM} smoke replicates; no coverage records")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, tasks))
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != sha or subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    ).strip():
        PILOT_TIMINGS.unlink(missing_ok=True)
        print("FAIL: HEAD or the working tree changed during the pilot")
        return 1
    if failed:
        print(f"FAIL: {len(failed)} pilot job(s) failed; see {PILOT_LOG_DIR.relative_to(ROOT)}")
        return 1
    print_plan(selection, gate_groups(), jobs)
    return 0


def last_durations() -> dict[str, float]:
    """The most recent recorded wall time of each task label (target/calibration-timings.tsv),
    falling back to the latest pilot of the label at any commit, projected to the first pass.

    These order and size the work; they are never a budget claim, so a pilot from an earlier
    commit is close enough where no full run has timed the task."""
    out: dict[str, float] = {}
    if PILOT_TIMINGS.is_file():
        for line in PILOT_TIMINGS.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 7 or parts[2] != "0":
                continue
            try:
                elapsed, replicates, test_seconds = float(parts[1]), int(parts[5]), float(parts[6])
            except ValueError:
                continue
            if replicates == PILOT_NSIM and 0 <= test_seconds <= elapsed:
                out[parts[0]] = test_seconds * FIRST_NSIM / replicates + elapsed - test_seconds
    if not TIMINGS.is_file():
        return out
    for line in TIMINGS.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def longest_first(tasks: list[tuple[Group, int | None]]) -> list[tuple[Group, int | None]]:
    """Order tasks by their last recorded duration, longest first (longest-processing-time
    scheduling), so a long task does not start late and set the wall time; tasks with no
    record keep their gate order after the timed ones."""
    durations = last_durations()

    def key(item: tuple[int, tuple[Group, int | None]]) -> tuple[int, float, int]:
        position, (group, point) = item
        seconds = durations.get(task_label(group, point))
        return (0, -seconds, position) if seconds is not None else (1, 0.0, position)

    return [task for _, task in sorted(enumerate(tasks), key=key)]


# Most worker threads a single long task is given (see tail_threads).
MAX_TAIL_THREADS = 4


def tail_threads(
    tasks: list[tuple[Group, int | None]], jobs: int, threads: int
) -> dict[str, int]:
    """Extra worker threads for the tasks that would otherwise set the wall time.

    With `jobs` tasks running at once the run cannot finish before the total recorded work over
    `jobs`; a task whose last single-threaded duration exceeds that share would finish after
    everything else. Such a task gets `ceil(duration / share)` threads (at most
    MAX_TAIL_THREADS), so its replicates spread over cores the short tasks leave idle at the
    tail. Durations come from the last recorded run; untimed tasks keep `threads`."""
    durations = last_durations()
    timed = [durations[task_label(g, p)] for g, p in tasks if task_label(g, p) in durations]
    if not timed or jobs <= 1:
        return {}
    share = sum(timed) / jobs
    out: dict[str, int] = {}
    for group, point in tasks:
        label = task_label(group, point)
        seconds = durations.get(label)
        if seconds is not None and seconds > share:
            out[label] = min(MAX_TAIL_THREADS, CORES, max(threads, math.ceil(seconds / share)))
    return out


def archive_logs() -> None:
    """Keep prior evidence and consoles permanently, partitioned by measured SHA.

    Unknown or mixed stamps are retained as such instead of attributed to HEAD.
    Moving directories also preserves incomplete jobs, tallies and failed logs.
    """
    sources = [path for path in (LOG_DIR, CONSOLE_DIR) if path.is_dir() and any(path.iterdir())]
    if not sources:
        return
    stamps = set()
    if LOG_DIR.is_dir():
        for log in LOG_DIR.glob("*.log"):
            stamps.update(re.findall(r"^calibration-measured-at ([0-9a-f]{40})$",
                                     log.read_text(encoding="utf-8"), re.M))
    sha = next(iter(stamps)) if len(stamps) == 1 else ("mixed" if stamps else "unstamped")
    destination = ARCHIVE_DIR / sha / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(parents=True)
    for source in sources:
        shutil.move(str(source), str(destination / source.name))
    print(f"archived earlier logs and consoles to {destination}")


def formal_decision_environment() -> dict[str, str]:
    """Activate only a reviewed complete family, independently of the selected shard.

    Historical gate dry runs, smoke pilots and legacy replay do not call this.
    A caller's inherited config cannot substitute a smaller family or seed domain.
    """
    import calibration_decision as decision
    import calibration_family as family

    if os.environ.get("ANTECEDENT_CALIBRATION_CONFIRMATION"):
        raise SystemExit("fresh-seed confirmation has no allocated risk in suite coverage protocol v1")
    if FIRST_NSIM not in {look["planned_replicates"] for look in decision.LOOKS}:
        raise SystemExit("suite coverage protocol requires a registered 400, 2000 or 20000 attempt look")
    config = family.current_config(ROOT)
    return {decision.ENV: decision.canonical_json(config)}


def measurement_source_sha() -> str:
    """Direct runner entry points must obey the same source binding as the wrapper."""
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True
    ).strip():
        raise SystemExit("formal measurement requires a clean committed source checkout; raw logs are retained")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run(selection: Selection, total: int, jobs: int) -> int:
    groups = selection.groups
    if not groups:
        print("nothing to measure: no record owes a re-measurement")
        return 0
    sha = measurement_source_sha()
    decision_env = formal_decision_environment()
    archive_logs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)
    if prebuild(groups) != 0:
        print("FAIL: the calibration tests do not build")
        return 1
    if measurement_source_sha() != sha:
        raise SystemExit("measurement source changed during prebuild; raw logs are retained")
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
            "ANTECEDENT_CALIBRATION_LOG_DIR",
            "ANTECEDENT_CALIBRATION_SMOKE",
            "ANTECEDENT_PARENT_PROTOCOL_VERSION",
            "ANTECEDENT_SYNTHESIS_PROTOCOL_VERSION",
            "ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL",
            "ANTECEDENT_WEIGHTED_CLUSTER_PROTOCOL",
            "ANTECEDENT_CALIBRATION_DECISION_CONFIG",
            "ANTECEDENT_CALIBRATION_VALIDATION_DOMAIN",
        )
    }
    env.update(decision_env)
    # Each job's `map_replicates` gets its share of the cores, so `jobs` jobs
    # fill the machine without oversubscribing it. RUST_TEST_THREADS is left
    # alone: every gate invocation runs one exact test.
    env["ANTECEDENT_CALIBRATION_THREADS"] = str(threads)
    env["ANTECEDENT_CALIBRATION_NSIM"] = str(FIRST_NSIM)
    env["ANTECEDENT_CALIBRATION_RECHECK_NSIM"] = str(RECHECK_NSIM)
    started = time.monotonic()
    lock = threading.Lock()
    done: list[int] = []
    failed: list[Group] = []

    tasks = longest_first([(group, point) for group in groups for point in group.points])
    widened = tail_threads(tasks, jobs, threads)

    def one(task: tuple[Group, int | None]) -> None:
        group, point = task
        label = task_label(group, point)
        suffix = "" if point is None else f".p{point}"
        console = CONSOLE_DIR / f"{group.safe}{suffix}.txt"
        task_threads = widened.get(label, threads)
        task_env = {
            **env,
            "ANTECEDENT_CALIBRATION_SHARD": f"{group.index - 1}/{total}",
            "ANTECEDENT_CALIBRATION_THREADS": str(task_threads),
        }
        if point is not None:
            task_env["ANTECEDENT_CALIBRATION_GRID_POINTS"] = str(point)
        begin = time.monotonic()
        with lock:
            print(f"[{_clock(begin - started)}] start group {group.index}: {label}")
        with console.open("w", encoding="utf-8", newline="\n") as out:
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
            with TIMINGS.open("a", encoding="utf-8", newline="\n") as timings:
                timings.write(f"{label}\t{elapsed:.1f}\t{status}\t{sha}\t{task_threads}\t{FIRST_NSIM}\n")
            verdict = "ok" if status == 0 else f"FAILED (see {console.relative_to(ROOT)})"
            print(
                f"[{_clock(time.monotonic() - started)}] {len(done)}/{len(tasks)} "
                f"group {group.index}: {verdict} in {_clock(elapsed)}"
                + (f" (rechecked at {RECHECK_NSIM} replicates)" if rechecked else "")
                + f": {label}",
                flush=True,
            )

    # last_durations includes measured wall times and falls back to old pilots.
    # Keep its longest-first schedule even when no pilot exists at this HEAD.
    ordered = tasks
    print(f"running {len(tasks)} grid job(s), {jobs} at a time with {threads} thread(s) each; first-pass replicates={FIRST_NSIM}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, ordered))
    print(
        f"measured {len(groups)} group(s) as {len(tasks)} grid job(s) in "
        f"{_clock(time.monotonic() - started)}"
    )
    if measurement_source_sha() != sha:
        raise SystemExit("measurement source changed during execution; raw logs are retained")
    if failed:
        print(f"FAIL: {len(failed)} grid job(s) failed the calibration gate:")
        for label in sorted(failed):
            print(f"  {label}")
        return 1
    return 0


def main() -> int:
    global FIRST_NSIM, RECHECK_NSIM
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("plan", "list the groups that would run"), ("pilot", "time smoke runs"), ("run", "run them")):
        p = sub.add_parser(name, help=text)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--all", action="store_true", help="every group, not only the owed ones")
        mode.add_argument("--missing-only", action="store_true",
                          help="only groups without coverage records or complete passed gate evidence; ignore drift")
        p.add_argument("--group", action="append", default=[], metavar="GLOB",
                       help="restrict selected groups by full label glob (repeatable; quote wildcards)")
        p.add_argument("--replicates", type=int, default=FIRST_NSIM, metavar="N",
                       help="first-pass simulation count for selected groups (minimum 400; default 400)")
        # One job per core, each on one worker thread: list-scheduling the
        # many heterogeneous tasks packs the machine better than nesting each
        # task's own parallelism. `--jobs N` gives each job ceil(cores / N)
        # threads (threads_per_job).
        p.add_argument("--jobs", type=int, default=CORES)
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.replicates < 400:
        raise SystemExit("--replicates must be at least 400 for measured calibration")
    FIRST_NSIM = args.replicates
    RECHECK_NSIM = max(2000, FIRST_NSIM + 1)
    groups = gate_groups()
    selection = filter_selection(select(groups, args.all, args.missing_only), args.group)
    print_plan(selection, groups, args.jobs)
    if args.command == "plan":
        return 0
    if args.command == "pilot":
        return pilot(selection, len(groups), args.jobs)
    return run(selection, len(groups), args.jobs)


if __name__ == "__main__":
    sys.exit(main())
