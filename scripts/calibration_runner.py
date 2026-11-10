"""Measuring one calibration group at one grid point: its first pass and, when the
harness asks for it, its recheck.

A unit of work runs the group's Nix-built test binary through the shared runner
(`evidence_capture`) and writes the same files the collector,
the decision audit and the probe read: `<stem>.log`, `<stem>.tallies.jsonl` and,
after a recheck, `<stem>.recheck.log` and `<stem>.recheck.tallies.jsonl`, each log
ending in its `calibration-measured-at` stamp (and decision context). It returns the
code regions its runs executed, so each record it emits is bound to the items its
measurement depended on.

With a release build (`nix/calibration-release.nix`) the replicates run on that
uninstrumented, optimized binary, and the instrumented binary of the same source runs a
smoke slice of `SLICE_NSIM` replicates, one thread, for coverage: counters in hot loops
cost several times the work, and threads sharing them far more. Its records bind every
function the executed code calls whole (`evidence_items.Selection.callee_bodies`).

The first pass runs `first_nsim` replicates. When it passes, ran at least one test,
is not a smoke run, ran fewer than `recheck_nsim` replicates and printed
`calibration-recheck`, the recheck extends it to `recheck_nsim`: the harness computes
only the replicates the first pass did not and folds them onto its tallies
(crates/antecedent/tests/common/calibration.rs), so the recheck's lines equal a fresh
run's. A group that names no exact tests rechecks only the tests that asked, and names
them in `calibration-rechecked-test` lines.
"""
from __future__ import annotations

import collections
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

GROUPS = "parity/calibration_groups.toml"
ACCEPTANCE = "parity/acceptance_manifest.toml"
_PASSED = re.compile(r"^test result: ok\. (\d+) passed", re.M)
_RECHECK = re.compile(r"^calibration-recheck ", re.M)
_RECHECK_TEST = re.compile(r"^calibration-recheck-test (\S+)", re.M)
_REPLICATED = {"replicated_interval", "replicated_statistical_test"}
# Replicates the instrumented binary runs, single-threaded, beside a release-binary measurement.
SLICE_NSIM = 16


@dataclass(frozen=True)
class GroupSpec:
    index: int  # 1-based position in the gate: the group's identity
    label: str
    package: str
    target: str  # "lib" or an integration test target
    features: tuple[str, ...]
    tests: tuple[str, ...] | None  # exact names; None: every ignored test of the binary
    ignored: bool
    grid: bool

    @property
    def build_target(self) -> tuple[str, str, str]:
        return (self.package, "lib", "") if self.target == "lib" else (self.package, "test", self.target)

    @property
    def safe(self) -> str:
        return self.label.translate(str.maketrans(" /:", "___"))


def load_groups(root: Path) -> list[GroupSpec]:
    rows = tomllib.loads((root / GROUPS).read_text(encoding="utf-8"))["group"]
    return [GroupSpec(index=i, label=r["label"], package=r["package"], target=r["target"],
                      features=tuple(r.get("features", ())),
                      tests=tuple(r["tests"]) if "tests" in r else None,
                      ignored=bool(r["ignored"]), grid=bool(r["grid"]))
            for i, r in enumerate(rows, start=1)]


@dataclass(frozen=True)
class Acceptance:
    """The replicate minima `parity/acceptance_manifest.toml` binds to calibration tests."""
    tests: frozenset[str]  # bound tests: their base point runs the acceptance count
    grid_tests: frozenset[str]  # bound at all three grid sizes (n, 2n, 4n)
    point_tests: frozenset[tuple[str, int]]  # bound at an explicit non-base grid point

    @classmethod
    def load(cls, root: Path) -> "Acceptance":
        cases = tomllib.loads((root / ACCEPTANCE).read_text(encoding="utf-8"))["case"]
        bound = [c for c in cases
                 if c["expected"]["kind"] in _REPLICATED and c["evidence"]["status"] == "bound"]
        name = lambda case: case["evidence"]["test"].rsplit("::", 1)[1]  # noqa: E731
        sizes: dict[str, set[int]] = collections.defaultdict(set)
        points = set()
        for case in bound:
            participants = case["configuration"].get("participants")
            if isinstance(participants, int):
                sizes[name(case)].add(participants)
        for case in cases:
            point = case["configuration"].get("calibration_grid_point")
            if point is None:
                continue
            if type(point) is not int or point not in (0, 1, 2):
                raise ValueError("invalid declared calibration_grid_point")
            if case in bound:
                points.add((name(case), point))
        grid = {t for t, v in sizes.items() if len(v) == 3 and sorted(v) == [min(v), 2 * min(v), 4 * min(v)]}
        return cls(frozenset(name(c) for c in bound), frozenset(grid), frozenset(points))


def first_pass_nsim(group: GroupSpec, point: int | None, acceptance: Acceptance, *, first: int,
                    acceptance_nsim: int, smoke: bool, parent_protocol: str = "2") -> int:
    """Replicates of a first pass (the gate's former `first_pass_nsim`)."""
    test = group.label.partition(": ")[2]
    if point in (0, 2):
        # Legacy parent records replay the version-1 400/2000/400 plan.
        if parent_protocol == "1" and group.label.startswith("sample_split_parent_set_effect_calibration:"):
            return first
        if test not in acceptance.grid_tests and (test, point) not in acceptance.point_tests:
            return first
    if not smoke and acceptance_nsim > first and test in acceptance.tests:
        return acceptance_nsim
    return first


def libtest_args(group: GroupSpec, only: list[str] | None = None) -> list[str]:
    """The test binary's arguments: what `cargo test ... -- ` passed it."""
    names = list(only) if only else list(group.tests or ())
    args = [*names, "--exact"] if names else []
    if group.ignored:
        args.append("--ignored")
    return [*args, "--nocapture", "--test-threads", "1"]


def passed_tests(output: str) -> int:
    return sum(int(n) for n in _PASSED.findall(output))


def stem_of(log_dir: Path, group: GroupSpec, point: int | None, confirm: bool) -> Path:
    suffix = "" if point is None else f".p{point}"
    return log_dir / f"{group.safe}{suffix}{'.confirm' if confirm else ''}"


def measure(spec: dict) -> dict:
    """One (group, grid point): first pass, recheck when asked. Runs in a worker process.

    `spec` holds: group (GroupSpec fields), point, stem, binary, cwd, env, tools,
    first_nsim, recheck_nsim, smoke, sha, decision_context. Returns the verdict, the
    regions every run executed, and whether it rechecked."""
    from evidence_capture import instrumented_run, merge_coverage
    group = GroupSpec(**spec["group"])
    release = spec.get("release_binary")

    def measured_run(args: list[str], env: dict[str, str]):
        """A measurement run: on the release binary when there is one, else instrumented."""
        if release is None:
            return instrumented_run(spec["binary"], args, cwd=Path(spec["cwd"]), env=env,
                                    tools=spec["tools"], merge_output=True)
        import subprocess
        done = subprocess.run([release, *args], cwd=spec["cwd"], env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return done.returncode, done.stdout or "", {}, None
    stem = Path(spec["stem"])
    log, tallies = stem.with_name(stem.name + ".log"), stem.with_name(stem.name + ".tallies.jsonl")
    recheck_log = stem.with_name(stem.name + ".recheck.log")
    recheck_tallies = stem.with_name(stem.name + ".recheck.tallies.jsonl")
    for stale in (tallies, recheck_tallies, recheck_log):
        stale.unlink(missing_ok=True)
    base_env = {k: v for k, v in spec["env"].items() if k != "ANTECEDENT_CALIBRATION_GRID_POINT"}
    if spec["point"] is not None:
        base_env["ANTECEDENT_CALIBRATION_GRID_POINT"] = str(spec["point"])

    def stamp(path: Path, text: str) -> None:
        lines = [text.rstrip("\n") + "\n" if text else "", f"calibration-measured-at {spec['sha']}\n"]
        if spec.get("decision_context"):
            lines.append(f"calibration-decision-context {spec['decision_context']}\n")
        path.write_text("".join(lines), encoding="utf-8", newline="\n")

    first = spec["first_nsim"]
    env = {**base_env, "ANTECEDENT_CALIBRATION_NSIM": str(first), "ANTECEDENT_CALIBRATION_TALLY_OUT": str(tallies)}
    status, output, regions, problem = measured_run(libtest_args(group), env)
    problems = []
    if status == 0 and passed_tests(output) < 1:
        problems.append(f"no test ran for this group (renamed or filtered away): {log}")
        status = 1
    if status == 0 and regions is None:
        problems.append(f"the coverage of {log} could not be read: {problem}")
        status = 1
    stamp(log, output)
    coverage = [regions]
    rechecked = False
    if (status == 0 and not spec["smoke"] and first < spec["recheck_nsim"] and _RECHECK.search(output)):
        rechecked = True
        tests = sorted(set(_RECHECK_TEST.findall(output))) if group.tests is None else []
        env = {**base_env, "ANTECEDENT_CALIBRATION_NSIM": str(spec["recheck_nsim"]),
               "ANTECEDENT_CALIBRATION_REPLICATE_START": str(first),
               "ANTECEDENT_CALIBRATION_PRIOR_TALLIES": str(tallies),
               "ANTECEDENT_CALIBRATION_TALLY_OUT": str(recheck_tallies)}
        status, output, regions, problem = measured_run(libtest_args(group, tests), env)
        if status == 0 and passed_tests(output) < 1:
            problems.append(f"no test ran for this group (renamed or filtered away): {recheck_log}")
            status = 1
        if status == 0 and regions is None:
            problems.append(f"the coverage of {recheck_log} could not be read: {problem}")
            status = 1
        output += "".join(f"calibration-rechecked-test {t}\n" for t in tests)
        stamp(recheck_log, output)
        coverage.append(regions)
    if release is not None and status == 0:
        # The measurement ran uninstrumented: a smoke slice of the same test, one thread, on
        # the instrumented binary says which code it runs. Its logs and tallies are discarded.
        slice_env = {k: v for k, v in base_env.items()
                     if k not in ("ANTECEDENT_CALIBRATION_TALLY_OUT", "ANTECEDENT_CALIBRATION_PRIOR_TALLIES",
                                  "ANTECEDENT_CALIBRATION_REPLICATE_START")}
        slice_env.update({"ANTECEDENT_CALIBRATION_NSIM": str(min(SLICE_NSIM, first)),
                          "ANTECEDENT_CALIBRATION_SMOKE": "1", "ANTECEDENT_CALIBRATION_THREADS": "1"})
        _, slice_output, regions, problem = instrumented_run(
            spec["binary"], libtest_args(group), cwd=Path(spec["cwd"]), env=slice_env,
            tools=spec["tools"], merge_output=True)
        if regions is None or passed_tests(slice_output) + slice_output.count("... FAILED") < 1:
            problems.append(f"the coverage slice of {log} ran no test or could not be read: {problem}")
            status = 1
        coverage = [regions or {}]
    return {"status": status, "rechecked": rechecked, "problems": problems,
            "coverage": merge_coverage(*coverage)}


# ---------------------------------------------------------------------- running tasks

@dataclass(frozen=True)
class Task:
    group: GroupSpec
    point: int | None  # grid point, or None for a group measured once
    threads: int = 1  # `map_replicates` workers (ANTECEDENT_CALIBRATION_THREADS)
    nsim: int | None = None  # first-pass replicates; None: the gate's rule

    @property
    def key(self) -> tuple[int, int | None]:
        return self.group.index, self.point


BUILDS = "evidence-builds.json"


def coverage_path(stem: Path) -> Path:
    return stem.with_name(stem.name + ".coverage.json")


def execute(root: Path, tasks: list[Task], *, env: dict[str, str], log_dir: Path, jobs: int,
            first_nsim: int, recheck_nsim: int, acceptance_nsim: int = 2000, smoke: bool = False,
            confirm: bool = False, decision_context: str | None = None, on_done=None,
            release: bool = True) -> dict:
    """Build every task's test binary in one Nix derivation from clean HEAD, then measure
    the tasks in the order given, `jobs` at a time, through the shared runner.

    Each task's logs land under `log_dir` with the code regions its runs executed beside
    them (`<stem>.coverage.json`), and the build records in `log_dir/evidence-builds.json`,
    so the collector binds every record it accepts to the items its measurement ran.
    `on_done(task, result)` is called as each task finishes. Returns `{task.key: result}`.
    With `release` (the default) the replicates run on the uninstrumented release build of
    the same source and the instrumented binary runs a coverage slice (module docstring)."""
    import json
    from concurrent.futures import as_completed
    from dataclasses import asdict

    import evidence_capture
    from nix_evidence_build import build_committed_targets
    from nix_evidence_policy import validate_build
    from nix_evidence_source import _clean_head

    root = root.resolve()
    if not tasks:
        return {}
    if not smoke:
        # A formal measurement carries the complete frozen decision context, whoever calls.
        import calibration_family
        from calibration_preflight import validate_inherited
        validate_inherited(decision_context, calibration_family.current_config(root))
    sha, _ = _clean_head(root)
    targets = sorted({t.group.build_target for t in tasks})
    features = {t.group.build_target: list(t.group.features) for t in tasks if t.group.features}
    built = build_committed_targets(root, targets, features=features)
    for record in built.values():
        validate_build(root, record)
    if release:
        from nix_evidence_build import attach_release_builds
        attach_release_builds(root, built)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / BUILDS).write_text(json.dumps({"|".join(k): v for k, v in built.items()}, sort_keys=True),
                                  encoding="utf-8")
    acceptance = Acceptance.load(root)
    parent_protocol = env.get("ANTECEDENT_PARENT_PROTOCOL_VERSION", "2")
    # `env` is the caller's whole environment for the tests; the runner sets the profile.
    base = {**env, "ANTECEDENT_CALIBRATION_SHA": sha, "RUST_TEST_THREADS": "1"}
    if decision_context:
        base["ANTECEDENT_CALIBRATION_DECISION_CONFIG"] = decision_context
    if confirm:
        base["ANTECEDENT_CALIBRATION_CONFIRMATION"] = "1"
    if smoke:
        base["ANTECEDENT_CALIBRATION_SMOKE"] = "1"
    evidence_capture.configure(jobs)
    futures = {}
    for task in tasks:
        record = built[task.group.build_target]
        output = Path(record["artifact"]["output"])
        nsim = task.nsim if task.nsim is not None else first_pass_nsim(
            task.group, task.point, acceptance, first=first_nsim, acceptance_nsim=acceptance_nsim,
            smoke=smoke, parent_protocol=parent_protocol)
        spec = {
            "group": asdict(task.group), "point": task.point,
            "stem": str(stem_of(log_dir, task.group, task.point, confirm)),
            "binary": str(output / record["artifact"]["binary"]),
            "release_binary": (str(Path(record["release"]["output"]) / record["release"]["binary"])
                               if "release" in record else None),
            "cwd": str(root / "crates" / task.group.package),
            "env": {**base, "ANTECEDENT_CALIBRATION_THREADS": str(task.threads)},
            "tools": {name: str(output / "bin" / name) for name in ("llvm-profdata", "llvm-cov")},
            "first_nsim": nsim, "recheck_nsim": recheck_nsim, "smoke": smoke, "sha": sha,
            "decision_context": decision_context,
        }
        futures[evidence_capture.submit(_timed_measure, spec)] = task
    results = {}
    for future in as_completed(futures):
        task = futures[future]
        result = future.result()
        stem = stem_of(log_dir, task.group, task.point, confirm)
        coverage_path(stem).write_text(json.dumps(
            {path: sorted(regions) for path, regions in sorted(result["coverage"].items())}), encoding="utf-8")
        results[task.key] = result
        if on_done is not None:
            on_done(task, result)
    if _clean_head(root)[0] != sha:
        raise SystemExit("measurement source changed during execution; raw logs are retained")
    return results


def _timed_measure(spec: dict) -> dict:
    import time
    begin = time.monotonic()
    result = measure(spec)
    result["seconds"] = time.monotonic() - begin
    return result


# ---------------------------------------------------------------------- binding records

def record_runs(log_dir: Path) -> dict[str, set[str]]:
    """Record id -> the task stems (`<group>[.p<k>][.confirm]`) whose logs emitted it."""
    import json
    runs: dict[str, set[str]] = {}
    for log in log_dir.glob("*.log"):
        stem = log.name.removesuffix(".log").removesuffix(".recheck")
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("calibration-record "):
                runs.setdefault(str(json.loads(line.split(" ", 1)[1])["id"]), set()).add(stem)
    return runs


def run_evidence(root: Path, log_dir: Path, stems: set[str]) -> tuple[dict, dict]:
    """The Nix build record and the executed regions behind a record measured by `stems`:
    every stem must come from one test binary and carry its coverage."""
    import json
    by_safe = {g.safe: g for g in load_groups(root)}
    builds = json.loads((log_dir / BUILDS).read_text(encoding="utf-8"))
    targets, regions = set(), {}
    for stem in stems:
        safe = re.sub(r"(\.p[012])?(\.confirm)?$", "", stem)
        group = by_safe.get(safe)
        path = log_dir / f"{stem}.coverage.json"
        if group is None or not path.is_file():
            raise RuntimeError(f"no measured coverage for {stem}")
        targets.add(group.build_target)
        for source, points in json.loads(path.read_text(encoding="utf-8")).items():
            regions.setdefault(source, set()).update(tuple(p) for p in points)
    if len(targets) != 1:
        raise RuntimeError(f"a record is measured by several test binaries: {sorted(targets)}")
    return builds["|".join(targets.pop())], regions
