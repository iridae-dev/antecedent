#!/usr/bin/env python3
"""The one reader of `.github/workflows/ci.yml` for the gates.

Workflow job ids (the keys under `jobs:`) are what `parity/release.toml`
names in `required_jobs`. `gh run view --json jobs` reports display names
instead: the job's `name:` with every `${{ matrix.* }}` expanded, one job
per matrix combination ("Rust ubuntu-latest", "Wheel macos-14 py3.12").
This module parses the workflow as YAML and owns that id -> display-name
mapping, so no gate re-derives it with a regex.

Run through the Python dev environment (PyYAML is a dev dependency):

    uv run --project python python scripts/ci_workflow.py job-ids
    uv run --project python python scripts/ci_workflow.py expected-names rust
    uv run --project python python scripts/ci_workflow.py check-run RUN.json \\
        [--head-sha SHA] rust gates
    uv run --project python python scripts/ci_workflow.py synth-run SHA rust gates
    uv run --project python python scripts/ci_workflow.py required-jobs [--json]

`required-jobs` is also the one reader of `parity/release.toml`'s
`required_jobs` key, so no gate re-parses that TOML either.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

# Relative to the working directory: every gate runs from the repo root, and a
# gate self-test runs from an overlay whose ci.yml may be deliberately broken.
CI_YML = Path(".github") / "workflows" / "ci.yml"
# The capability table naming the CI jobs a release candidate must have green.
RELEASE_TOML = Path("parity") / "release.toml"

_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")
_ANY_EXPR = re.compile(r"\$\{\{.*?\}\}")


class WorkflowError(ValueError):
    """The workflow uses a shape this reader does not model."""


def load_jobs(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Parsed `jobs:` mapping, keyed by workflow job id."""
    path = CI_YML if path is None else path
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkflowError(f"{path}: not a YAML mapping")
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise WorkflowError(f"{path}: no `jobs:` mapping")
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            raise WorkflowError(f"{path}: job {job_id!r} is not a mapping")
    return jobs


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def matrix_combinations(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand `strategy.matrix` with GitHub's include/exclude semantics."""
    matrix = (job.get("strategy") or {}).get("matrix")
    if matrix is None:
        return [{}]
    if isinstance(matrix, str):
        raise WorkflowError("dynamic matrix expressions cannot be expanded statically")
    if not isinstance(matrix, dict):
        raise WorkflowError("strategy.matrix is not a mapping")
    axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise WorkflowError(f"matrix axis {key!r} is not a non-empty list")
    keys = list(axes)
    combos: list[dict[str, Any]] = (
        [dict(zip(keys, row, strict=True)) for row in itertools.product(*(axes[k] for k in keys))]
        if keys
        else []
    )
    for rule in matrix.get("exclude") or []:
        combos = [c for c in combos if not all(c.get(k) == v for k, v in rule.items())]
    extra: list[dict[str, Any]] = []
    for entry in matrix.get("include") or []:
        if not isinstance(entry, dict):
            raise WorkflowError("matrix include entry is not a mapping")
        # An include entry extends every combination whose original axis values
        # it does not contradict; if it extends none, it is a new combination.
        merged = False
        for combo in combos:
            if all(combo[k] == v for k, v in entry.items() if k in axes):
                combo.update({k: v for k, v in entry.items() if k not in axes})
                merged = True
        if not merged:
            extra.append(dict(entry))
    combos.extend(extra)
    if not combos:
        raise WorkflowError("strategy.matrix expands to zero jobs")
    return combos


def expected_names(job_id: str, job: dict[str, Any]) -> list[str]:
    """Every display name GitHub gives this job id in a run."""
    combos = matrix_combinations(job)
    template = job.get("name")
    names = []
    for combo in combos:
        if template is None:
            if combo:
                values = ", ".join(_scalar(v) for v in combo.values())
                names.append(f"{job_id} ({values})")
            else:
                names.append(job_id)
            continue

        def sub(match: re.Match[str], combo: dict[str, Any] = combo) -> str:
            key = match.group(1)
            if key not in combo:
                raise WorkflowError(f"job {job_id!r} name uses matrix.{key}, which is unset")
            return _scalar(combo[key])

        name = _MATRIX_EXPR.sub(sub, str(template))
        if _ANY_EXPR.search(name):
            raise WorkflowError(f"job {job_id!r} name has a non-matrix expression: {template!r}")
        names.append(name)
    if len(set(names)) != len(names):
        raise WorkflowError(f"job {job_id!r} expands to duplicate display names")
    return names


def check_run(
    run_jobs: list[dict[str, Any]], required: list[str], path: Path | None = None
) -> list[str]:
    """Problems with a `gh run view --json jobs` payload; empty means accepted."""
    path = CI_YML if path is None else path
    jobs = load_jobs(path)
    by_name: dict[str, str | None] = {}
    for job in run_jobs:
        by_name[str(job.get("name", ""))] = job.get("conclusion")
    problems = []
    for job_id in required:
        if job_id not in jobs:
            problems.append(f"required job {job_id!r} is not a job id in {path.name}")
            continue
        for name in expected_names(job_id, jobs[job_id]):
            if name not in by_name:
                problems.append(f"{job_id}: job {name!r} missing from the run")
            elif by_name[name] != "success":
                problems.append(f"{job_id}: job {name!r} concluded {by_name[name]!r}")
    return problems


def required_jobs(
    release: Path | None = None, path: Path | None = None
) -> tuple[list[str], list[str]]:
    """The `required_jobs` every `parity/release.toml` capability names.

    Returns `(job_ids, problems)`. A row without the key states no requirement
    and is skipped; a row whose value is not a list, a job id that is not a job
    id in the workflow, and an empty overall list are each a problem. This is
    the one reader of that key — both the release-candidate gate and the parity
    schema gate call it rather than re-parsing the TOML.
    """
    release = RELEASE_TOML if release is None else release
    rows = tomllib.loads(release.read_text(encoding="utf-8")).get("capabilities", [])
    wanted: list[str] = []
    problems: list[str] = []
    for row in rows:
        jobs = row.get("required_jobs")
        if jobs is None:
            continue
        if not isinstance(jobs, list):
            problems.append(f"release.toml {row.get('id')}: required_jobs must be a list")
            continue
        wanted.extend(jobs)
    if not wanted and not problems:
        problems.append(f"{release.name} names no required_jobs")
    try:
        job_ids = set(load_jobs(path))
    except (OSError, WorkflowError) as error:
        problems.append(f"{CI_YML if path is None else path}: {error}")
        return wanted, problems
    for row in rows:
        jobs = row.get("required_jobs")
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job not in job_ids:
                problems.append(
                    f"release.toml {row.get('id')}: required job {job!r} missing from ci.yml"
                )
    return wanted, problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    command, *rest = argv
    # `--workflow PATH` reads a workflow elsewhere (a gate self-test overlay).
    global CI_YML, RELEASE_TOML
    if "--workflow" in rest:
        i = rest.index("--workflow")
        CI_YML = Path(rest[i + 1])
        del rest[i : i + 2]
    if "--release" in rest:
        i = rest.index("--release")
        RELEASE_TOML = Path(rest[i + 1])
        del rest[i : i + 2]
    try:
        if command == "required-jobs":
            # required-jobs [--json] : the release.toml job ids, checked.
            wanted, problems = required_jobs()
            if "--json" in rest:
                print(json.dumps({"jobs": wanted, "problems": problems}))
                return 0
            if problems:
                print("FAIL: parity/release.toml required_jobs:", file=sys.stderr)
                for problem in problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 1
            print(" ".join(wanted))
            return 0
        if command == "job-ids":
            print(json.dumps(sorted(load_jobs())))
            return 0
        if command == "expected-names":
            jobs = load_jobs()
            out = {job_id: expected_names(job_id, jobs[job_id]) for job_id in rest}
            print(json.dumps(out, indent=2))
            return 0
        if command == "synth-run":
            # synth-run HEAD_SHA JOB_ID... : an all-green payload for self-tests.
            head, *wanted = rest
            jobs = load_jobs()
            payload = {
                "headSha": head,
                "jobs": [
                    {"name": name, "conclusion": "success"}
                    for job_id in wanted
                    for name in expected_names(job_id, jobs[job_id])
                ],
            }
            print(json.dumps(payload))
            return 0
        if command == "check-run":
            # check-run RUN.json [--head-sha SHA] JOB_ID...
            head_sha = None
            if len(rest) >= 3 and rest[1] == "--head-sha":
                head_sha = rest[2]
                rest = [rest[0], *rest[3:]]
            if len(rest) < 2:
                print("FAIL: check-run needs a run payload and at least one required job id")
                return 1
            payload = json.loads(Path(rest[0]).read_text(encoding="utf-8"))
            problems = []
            if head_sha is not None and payload.get("headSha") != head_sha:
                problems.append(f"CI run headSha {payload.get('headSha')} != HEAD {head_sha}")
            problems += check_run(payload.get("jobs") or [], rest[1:])
            if problems:
                print("FAIL: required CI jobs not successful:")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
            total = sum(len(expected_names(j, load_jobs()[j])) for j in rest[1:])
            print(f"RC CI jobs: {', '.join(rest[1:])} ({total} display names) all success")
            return 0
    except WorkflowError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"unknown command {command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
