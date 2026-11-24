#!/usr/bin/env python3
"""Adversarial campaign: generated cases against production code and the defects they catch.

`parity/adversarial_campaign.toml` lists the required cases (one generated test each, of
kind preserving, changing, invalid or bound) and the mutations: defects deliberately
introduced into production code, each naming the cases that must catch it.

    python3 scripts/adversarial.py check                # static: cases exist, mutations apply
    python3 scripts/adversarial.py run                  # run every case; all must run and pass
    python3 scripts/adversarial.py run --mutations      # then apply each mutation and rerun
    python3 scripts/adversarial.py run --runners rust   # only the cases of the named runners (CI)

A required case that did not run is a failure, never a pass. A mutation is reported as
detected (a named case failed), undetected (every named case passed) or unexecuted (the
mutation did not apply or a runner could not run); undetected and unexecuted both fail
the campaign. Each mutated file is restored, and the run fails if any is left changed.

The `aeneas` runner retranslates Rust and needs AENEAS_DIR (formal/rust/extract.sh); the
`kani` runner needs cargo kani at the pinned version (formal/toolchain.toml) and checks one named
harness per invocation under its timeout; the `python` runner runs in the pinned interpreter the
review digests are bound under, with the extension already built into that environment.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from _repo import REPO_ROOT, load_toml

CAMPAIGN = Path("parity/adversarial_campaign.toml")
KINDS = {"preserving", "changing", "invalid", "bound"}
# Where each runner's case names are declared, for the static check.
SOURCES = {
    "rust": ("formal/adversarial/tests", "*.rs", r"fn\s+{name}\b"),
    "python": ("python/tests", "test_adversarial_*.py", r"def\s+{name}\b"),
    "kani": ("formal/kani/src", "*.rs", r"#\[kani::proof\][^{{]*?fn\s+{name}\b"),
}
LEAN_MODULES = {"lean": "formal", "aeneas": "formal/rust"}


def load(root: Path) -> dict:
    return load_toml(root / CAMPAIGN)


def _declared(root: Path, runner: str, name: str) -> bool:
    if runner in LEAN_MODULES:
        return (root / LEAN_MODULES[runner] / (name.replace(".", "/") + ".lean")).is_file()
    directory, pattern, regex = SOURCES[runner]
    rx = re.compile(regex.format(name=re.escape(name)), re.S)
    return any(rx.search(p.read_text(encoding="utf-8")) for p in (root / directory).glob(pattern))


def validate(root: Path, campaign: dict) -> list[str]:
    errors: list[str] = []
    runners = campaign.get("runners", {})
    cases = campaign.get("cases", [])
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case ids")
    for case in cases:
        if case.get("kind") not in KINDS:
            errors.append(f"{case['id']}: kind must be one of {sorted(KINDS)}")
        if case.get("runner") not in runners:
            errors.append(f"{case['id']}: unknown runner {case.get('runner')!r}")
        elif not _declared(root, case["runner"], case["test"]):
            errors.append(f"{case['id']}: test {case['test']} is not declared for runner {case['runner']}")
    for kind in ("preserving", "changing", "invalid"):
        if not any(c.get("kind") == kind for c in cases):
            errors.append(f"no {kind} case: the campaign needs positive and negative cases")
    mutations = campaign.get("mutations", [])
    if not mutations:
        errors.append("no mutations: the campaign must show it catches a deliberate defect")
    for mutation in mutations:
        mid = mutation["id"]
        path = root / mutation["file"]
        if not path.is_file():
            errors.append(f"{mid}: no file {mutation['file']}")
        elif path.read_text(encoding="utf-8").count(mutation["old"]) != 1:
            errors.append(f"{mid}: the text to mutate does not occur exactly once in {mutation['file']}")
        unknown = set(mutation.get("caught_by", [])) - set(ids)
        if unknown or not mutation.get("caught_by"):
            errors.append(f"{mid}: caught_by must name existing cases (unknown {sorted(unknown)})")
    return errors


def check(root: Path = REPO_ROOT) -> int:
    campaign = load(root)
    errors = validate(root, campaign)
    for error in errors:
        print(f"adversarial: {error}", file=sys.stderr)
    if not errors:
        print(f"adversarial: {len(campaign['cases'])} cases and {len(campaign['mutations'])} mutations declared")
    return 1 if errors else 0


def _outcomes(runner: str, output: str, returncode: int, names: list[str]) -> dict[str, str]:
    """passed / failed / not run, per required case name, from one runner's output."""
    result: dict[str, str] = {}
    for name in names:
        if runner == "rust":
            m = re.search(rf"^test (?:\S+::)?{re.escape(name)} \.\.\. (ok|FAILED|ignored)", output, re.M)
            result[name] = {"ok": "passed", "FAILED": "failed"}.get(m.group(1), "not run") if m else "not run"
        elif runner == "python":
            m = re.search(rf"^(PASSED|FAILED|ERROR|SKIPPED)\S* \S*::{re.escape(name)}\b", output, re.M)
            result[name] = {"PASSED": "passed", "FAILED": "failed", "ERROR": "failed"}.get(
                m.group(1), "not run") if m else "not run"
        elif runner == "kani":
            m = re.search(rf"Checking harness (?:\S+::)?{re.escape(name)}\.\.\..*?VERIFICATION:- (SUCCESSFUL|FAILED)",
                          output, re.S)
            result[name] = {"SUCCESSFUL": "passed", "FAILED": "failed"}[m.group(1)] if m else "not run"
        else:  # a Lean build checks every module or fails
            failed = re.search(rf"✖ .*{re.escape(name)}", output) or returncode != 0
            result[name] = "failed" if failed else "passed"
    return result


def run_cases(root: Path, campaign: dict, only: set[str] | None = None) -> dict[str, str]:
    runners = campaign["runners"]
    by_runner: dict[str, list[dict]] = {}
    for case in campaign["cases"]:
        if only is None or case["id"] in only:
            by_runner.setdefault(case["runner"], []).append(case)
    outcomes: dict[str, str] = {}
    for runner, cases in by_runner.items():
        spec = runners[runner]
        # A runner with `case_args` runs one invocation per case (a Kani harness each, under the
        # runner's `timeout`); otherwise one invocation runs every case of the runner.
        groups = [[case] for case in cases] if "case_args" in spec else [cases]
        for group in groups:
            outcomes.update(_run_group(root, runner, spec, group))
    return outcomes


def _run_group(root: Path, runner: str, spec: dict, cases: list[dict]) -> dict[str, str]:
    command = list(spec["command"])
    if "case_args" in spec:
        command += [arg.format(test=cases[0]["test"]) for arg in spec["case_args"]]
    try:
        # A session of its own, so a timeout ends the whole process group (cargo kani leaves
        # its CBMC solver running when only cargo is killed).
        proc = subprocess.Popen(
            command, cwd=root / spec["directory"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
    except OSError as error:
        print(f"adversarial: runner {runner} could not start: {error}", file=sys.stderr)
        return {c["id"]: "not run" for c in cases}
    try:
        stdout, stderr = proc.communicate(timeout=spec.get("timeout"))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        names = ", ".join(c["test"] for c in cases)
        print(f"adversarial: runner {runner} exceeded {spec['timeout']} s on {names}", file=sys.stderr)
        return {c["id"]: "not run" for c in cases}
    output = stdout + stderr
    names = _outcomes(runner, output, proc.returncode, [c["test"] for c in cases])
    if proc.returncode != 0 and all(names[c["test"]] == "not run" for c in cases):
        print(f"adversarial: runner {runner} failed before running its cases:\n{output[-3000:]}", file=sys.stderr)
    return {case["id"]: names[case["test"]] for case in cases}


def run(root: Path = REPO_ROOT, mutations: bool = False, runners: set[str] | None = None) -> int:
    campaign = load(root)
    errors = validate(root, campaign)
    unknown = (runners or set()) - set(campaign["runners"])
    if unknown:
        errors.append(f"unknown runner(s) {sorted(unknown)}")
    if mutations and runners:
        errors.append("--mutations runs every runner a mutation names; it takes no --runners")
    if errors:
        for error in errors:
            print(f"adversarial: {error}", file=sys.stderr)
        return 1
    selected = None if runners is None else {c["id"] for c in campaign["cases"] if c["runner"] in runners}
    if selected is not None:
        campaign = {**campaign, "cases": [c for c in campaign["cases"] if c["id"] in selected]}
    outcomes = run_cases(root, campaign)
    bad = {k: v for k, v in outcomes.items() if v != "passed"}
    for case in campaign["cases"]:
        print(f"  {outcomes[case['id']]:8} {case['kind']:10} {case['id']}")
    if bad:
        print(f"adversarial: {len(bad)} required case(s) did not pass: {sorted(bad)}", file=sys.stderr)
        return 1
    if not mutations:
        print(f"adversarial: {len(outcomes)} cases passed")
        return 0
    report: dict[str, str] = {}
    for mutation in campaign["mutations"]:
        path = root / mutation["file"]
        original = path.read_text(encoding="utf-8")
        if original.count(mutation["old"]) != 1:
            report[mutation["id"]] = "unexecuted"
            continue
        try:
            path.write_text(original.replace(mutation["old"], mutation["new"], 1), encoding="utf-8")
            results = run_cases(root, campaign, set(mutation["caught_by"]))
        finally:
            path.write_text(original, encoding="utf-8")
        if any(v == "failed" for v in results.values()):
            report[mutation["id"]] = "detected"
        elif any(v == "not run" for v in results.values()):
            report[mutation["id"]] = "unexecuted"
        else:
            report[mutation["id"]] = "undetected"
    # Restoring a mutated Rust file must also restore any translation regenerated from it.
    if any(m["file"].startswith("crates/") for m in campaign["mutations"]) and "aeneas" in campaign["runners"]:
        subprocess.run(["bash", "formal/rust/extract.sh"], cwd=root, capture_output=True)
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", *sorted({m["file"] for m in campaign["mutations"]}),
         "formal/rust/AntecedentRust/Extracted"],
        cwd=root, capture_output=True, text=True,
    ).stdout.strip()
    for mid, status in report.items():
        print(f"  {status:10} {mid}")
    if changed:
        print(f"adversarial: files left changed after the mutations:\n{changed}", file=sys.stderr)
        return 1
    failing = {k: v for k, v in report.items() if v != "detected"}
    if failing:
        print(f"adversarial: mutations not detected: {failing}", file=sys.stderr)
        return 1
    print(f"adversarial: {len(outcomes)} cases passed; {len(report)} mutations detected")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        return check()
    if argv[:1] == ["run"]:
        rest = argv[1:]
        runners = None
        if "--runners" in rest:
            index = rest.index("--runners")
            runners = set(rest[index + 1].split(",")) if index + 1 < len(rest) else set()
        return run(mutations="--mutations" in rest, runners=runners)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
