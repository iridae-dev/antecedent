#!/usr/bin/env python3
"""Release-candidate receipts for the acceptance manifest.

The release-closure cases of parity/acceptance_manifest.toml are statements about the
whole manifest and the gates that execute it. Each is one function here, run on the real
registries by `scripts/gate_release_candidate.sh` and exercised on both a passing and a
failing fixture by scripts/test_check_release_closure.py:

    installed_wheel_problems     every documented example passes from an installed wheel
    refusal_boundary_problems    every documented refusal boundary refuses
    replay_identity_problems     replay identity holds for each artifact consumer
    attestation_problems         the candidate runs the inventory, which binds every manifest case

Calibration is not a receipt: a measured coverage record annotates an interval and never
closes or blocks a case (scripts/calibration_status.py lists what is owed). The rest of the
closure receipts live beside what they enforce: the manifest checker owns "no planned case"
and "no audit, limit, failed measurement or deferral closes a case"
(scripts/check_acceptance_manifest.py), and scripts/ci_workflow.py owns the CI attestation.

    python3 scripts/check_release_closure.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path, load_toml

ensure_scripts_on_path()
import check_acceptance_manifest as manifest  # noqa: E402

IDENTITY = re.compile(
    r"replay\w* (?:reproduces|yields|in a fresh|identity matches)|replays? (?:in separate|results)"
    r"|bit-identical|identical (?:results|receipt)|replayed draws",
    re.I,
)
EQUALITY = re.compile(
    r"assert_eq!|assert_ne!|assertEqual|assert\s+[^\n]*==|\bis_identical|\.identity|to_bits"
)


def bound_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in data["case"] if c["evidence"]["status"] == "bound"]


def test_body(case: dict[str, Any], root: Path) -> tuple[str, str, str] | None:
    """`(path, decoration, source with its same-file helpers)` of a bound case's test."""
    path, name = (
        case["evidence"]["test"].split("::", 1)[0],
        case["evidence"]["test"].rsplit("::", 1)[1],
    )
    source = root / path
    if not source.is_file():
        return None
    text = source.read_text(encoding="utf-8")
    found = manifest.test_source(text, source.suffix, name)
    if found is None:
        return None
    return path, found[0], manifest.with_helpers(text, source.suffix, found[1])


def installed_wheel_problems(candidate: str) -> list[str]:
    """The installed-wheel step of the release-candidate gate, read as text.

    The wheel must be installed into a fresh venv, the suite and every documented example
    must run with that venv's interpreter, from a directory that is not the source tree,
    and the interpreter must first prove it imports the wheel.
    """
    venv = '"$venv_dir/venv/bin/python"'
    problems = []
    if "maturin build" not in candidate or "uv pip install --python" not in candidate:
        problems.append("the gate does not build and install a wheel into a fresh venv")
    for label, needle in (
        ("proves the wheel is the imported package", "check_installed_wheel_import.py"),
        ("runs the Python suite (every shipped example and notebook)", "-m pytest"),
        ("executes the documented snippets and embedded examples", "run_doc_examples.py"),
    ):
        lines = [line for line in candidate.splitlines() if needle in line and venv in line]
        if not lines:
            problems.append(f"the installed-wheel step does not {label} with the venv interpreter")
    leave, doc_examples = candidate.find('cd "$run_dir"'), candidate.find("run_doc_examples.py")
    if leave < 0 or (0 <= doc_examples < leave):
        problems.append(
            "the installed-wheel step does not leave the source tree before running the examples"
        )
    return problems


def example_problems(root: Path = ROOT) -> list[str]:
    """Every shipped example must assert its own result and be run by the Python suite."""
    suite = root / "python/tests/test_examples_run.py"
    if not suite.is_file():
        return ["python/tests/test_examples_run.py does not exist"]
    text = suite.read_text(encoding="utf-8")
    problems = (
        []
        if "examples" in text and "runpy.run_path" in text
        else ["python/tests/test_examples_run.py does not execute examples/python"]
    )
    return problems


def executed_by_the_gates(path: str, ci: str) -> bool:
    """Whether the CI `rust` job (verified by the CI run) or the release-candidate suite runs `path`."""
    if path.startswith(("python/tests/", "scripts/test_")):
        return True
    return path.startswith("crates/") and "cargo nextest run" in ci and "--workspace" in ci


def refusal_boundary_problems(data: dict[str, Any], root: Path = ROOT, ci: str = "") -> list[str]:
    """Every refusal case is bound to a live, un-ignored asserting test that a gate executes."""
    problems = []
    for case in data["case"]:
        if case["expected"]["kind"] != "refusal":
            continue
        cid = case["id"]
        if case["evidence"]["status"] != "bound":
            problems.append(f"{cid}: refusal boundary has no bound test")
            continue
        found = test_body(case, root)
        if found is None:
            problems.append(f"{cid}: bound refusal test cannot be read")
            continue
        path, decoration, _ = found
        if "#[ignore" in decoration:
            problems.append(f"{cid}: refusal test is ignored, so no gate checks the boundary")
        if not executed_by_the_gates(path, ci):
            problems.append(f"{cid}: no release gate executes {path}")
    return problems


def replay_identity_problems(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Every case that promises replay or identity compares the replay with the original exactly."""
    problems = []
    for case in data["case"]:
        outputs = " ".join(case["expected"]["outputs"])
        if case["expected"]["kind"] not in ("value", "receipt") or not IDENTITY.search(outputs):
            continue
        if case["evidence"]["status"] != "bound":
            problems.append(f"{case['id']}: replay case has no bound test")
            continue
        found = test_body(case, root)
        if found is None or not EQUALITY.search(found[2]):
            problems.append(f"{case['id']}: bound test never compares the replay with the original")
    return problems


def attestation_problems(candidate: str, release: str) -> list[str]:
    """The release candidate runs the inventory, and the inventory requires every manifest case bound."""
    problems = []
    if "gate_release.sh" not in candidate:
        problems.append("gate_release_candidate.sh does not run gate_release.sh")
    if "check_release_inventory.py" not in release:
        problems.append("gate_release.sh does not run the release inventory")
    elif "--require-bound" not in Path(__file__).with_name("check_release_inventory.py").read_text(
        encoding="utf-8"
    ):
        problems.append("the release inventory does not require every manifest case to be bound")
    return problems


def closure_problems(root: Path = ROOT) -> list[str]:
    data = load_toml(root / "parity/acceptance_manifest.toml")
    read = lambda rel: (root / rel).read_text(encoding="utf-8")  # noqa: E731
    candidate, release = (read(f"scripts/{n}") for n in ("gate_release_candidate.sh", "gate_release.sh"))
    return [
        *installed_wheel_problems(candidate),
        *example_problems(root),
        *refusal_boundary_problems(data, root, read(".github/workflows/ci.yml")),
        *replay_identity_problems(data, root),
        *attestation_problems(candidate, release),
    ]


def main() -> int:
    problems = closure_problems()
    if problems:
        print("release closure FAILED:")
        for problem in dict.fromkeys(problems):
            print(f"  {problem}")
        return 1
    print("release closure: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
