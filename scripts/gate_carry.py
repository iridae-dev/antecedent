#!/usr/bin/env python3
"""Carry a pass/fail calibration gate's earlier pass to the commit being measured.

A calibration group that emits no coverage record is attested by a row of
`parity/calibration_gates.toml`: the group ran at `calibration_sha`, passed and failed
nothing. Such a row has no input fingerprint, so planning could not tell whether the pass
still describes the code, and re-planned every one of them.

A gate run writes the executed source regions of each group beside its log
(`<group>.coverage.json`). The collector stores the executed files in the row
(`executed_files`). A pass is carried to the current tree when nothing it executed
changed since `calibration_sha`:

* no executed source file differs between `calibration_sha` and the working tree;
* no non-Rust file inside an executed crate differs (manifests, data, build scripts);
* the workspace manifest, lock file, toolchain pin and cargo configuration are identical.

File granularity is deliberately conservative: a changed file the group executed re-plans
it, whatever the hunk. Inputs the test reads at run time that no line coverage can name
(fixture files outside the crates) are not seen by this check; the report counts the
changed ones so the reader can judge them. A reviewed carry in
`parity/calibration_gate_carries.toml` binds the original complete receipt set
and test selection to a reviewed source tree. Subsequent relevant changes are
checked against that tree. The original measurement SHA and counts stay in the ledger.

    python3 scripts/gate_carry.py report            # which recorded passes are carried now
    python3 scripts/gate_carry.py attach --logs DIR   # dry run: rows that gain executed_files
    python3 scripts/gate_carry.py attach --logs DIR --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import load_toml

GATES = ROOT / "parity" / "calibration_gates.toml"
SOURCE_PREFIX = "/antecedent-source/"
# Files whose change can alter every build, whichever crate a group executed.
WORKSPACE_INPUTS = ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "rust-toolchain")
WORKSPACE_INPUT_DIRS = (".cargo/",)
MEASURED_AT = re.compile(r"^calibration-measured-at ([0-9a-f]{40})$", re.M)


def executed_files(coverage: dict) -> list[str]:
    """Repository paths of the workspace source files a coverage capture executed."""
    return sorted(
        key.removeprefix(SOURCE_PREFIX)
        for key in coverage
        if key.startswith(SOURCE_PREFIX) and coverage[key]
    )


def executed_crates(files: list[str]) -> set[str]:
    return {"/".join(f.split("/")[:2]) for f in files if f.startswith("crates/")}


@cache
def changed_since(sha: str, root: Path = ROOT) -> frozenset[str]:
    """Paths that differ between `sha` and the working tree (renames reported as both ends)."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", sha],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return frozenset(p for p in out.split("\0") if p)


def stale_inputs(files: list[str], sha: str, changed: frozenset[str] | None = None) -> list[str]:
    """The changed inputs that stop a pass at `sha` describing the tree, empty when carried."""
    changed = changed_since(sha) if changed is None else changed
    executed = set(files)
    crates = executed_crates(files)
    stale = []
    for path in sorted(changed):
        crate = "/".join(path.split("/")[:2])
        if (
            path in executed
            or path in WORKSPACE_INPUTS
            or path.startswith(WORKSPACE_INPUT_DIRS)
            or (crate in crates and (not path.endswith(".rs") or path.endswith("/build.rs")))
        ):
            stale.append(path)
    return stale


@cache
def reviewed_receipts() -> dict[tuple[str, str], tuple[str, frozenset[str]]]:
    """Map an exact gate and evidence digest to the tree where its changes were reviewed."""
    path = ROOT / "parity/calibration_gate_carries.toml"
    reviewed = {}
    for review in load_toml(path)["review"]:
        sha = review["reviewed_at"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or not review["basis"].strip():
            raise ValueError(f"invalid calibration gate review at {sha!r}")
        for label, digest in review["receipts"].items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid gate receipt digest for {label}")
            if (label, digest) in reviewed:
                raise ValueError(f"duplicate gate carry for {label}")
            reviewed[label, digest] = sha, frozenset(review["inputs"])
    return reviewed


@cache
def gate_definitions() -> dict[str, dict]:
    rows = load_toml(ROOT / "parity/calibration_groups.toml")["group"]
    return {row["label"].translate(str.maketrans(" /:", "___")): row for row in rows}


def receipt_digest(definition: dict, rows: list[dict]) -> str:
    """Bind exact test selection and every original grid receipt, including decisions and files."""
    data = json.dumps([definition, rows], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


# Inputs shared by the reviewed measurement jobs beyond the recorded Rust footprints.
# Each review also checked the literal conformance fixtures read by its test sources.
REVIEW_INPUTS = {
    "flake.nix", "flake.lock", "scripts/calibration_groups.py", "scripts/calibration_runner.py",
    "scripts/calibration_shards.py",
    "scripts/calibration_decision.py", "scripts/calibration_family_manifest.json",
    "parity/acceptance_manifest.toml",
}


def group_carried(rows: list[dict | None], changed: dict[str, frozenset[str]] | None = None) -> tuple[bool, str]:
    """Is every grid point of a group a passing row whose executed inputs are unchanged?"""
    if not rows or any(row is None for row in rows):
        return False, "no recorded pass for every grid point"
    first = rows[0]
    assert first is not None
    name = re.sub(r"\.p[012]$", "", first["group"])
    definition = gate_definitions().get(name)
    review = None
    if definition is not None:
        digest = receipt_digest(definition, rows)
        review = reviewed_receipts().get((definition["label"], digest))
    for row in rows:
        assert row is not None
        if int(row.get("passed", 0)) <= 0:
            return False, f"{row['group']}: recorded no passing test"
        files = row.get("executed_files")
        if not files:
            return False, f"{row['group']}: pass is bound to a commit only (no executed_files)"
        sha = review[0] if review else str(row["calibration_sha"])
        try:
            paths = changed[sha] if changed is not None and sha in changed else changed_since(sha)
            stale = stale_inputs(list(files), sha, paths)
            if review:
                stale.extend(sorted(paths.intersection(REVIEW_INPUTS | review[1])))
                stale.extend(sorted(p for p in paths if p.startswith("nix/")))
        except subprocess.CalledProcessError:
            return False, f"{row['group']}: cannot verify changes since {sha}"
        if stale:
            return False, f"{row['group']}: input changed since {sha[:10]}: {stale[0]}"
    return True, f"carried by review at {review[0][:10]}" if review else "carried"


def gate_rows(path: Path = GATES) -> dict[str, dict]:
    return {str(row["group"]): row for row in load_toml(path).get("gate", [])} if path.is_file() else {}


def rows_of(group_safe: str, points: tuple, ledger: dict[str, dict]) -> list[dict | None]:
    return [ledger.get(group_safe + ("" if p is None else f".p{p}")) for p in points]


def attach(logs: Path, apply: bool) -> int:
    """Add `executed_files` to rows measured at the commit their retained coverage names."""
    ledger = gate_rows()
    gained, skipped = 0, []
    for name, row in sorted(ledger.items()):
        if row.get("executed_files"):
            continue
        coverage, log = logs / f"{name}.coverage.json", logs / f"{name}.log"
        if not coverage.is_file() or not log.is_file():
            skipped.append((name, "no retained coverage capture"))
            continue
        measured = MEASURED_AT.search(log.read_text(encoding="utf-8", errors="ignore"))
        if not measured or measured.group(1) != row["calibration_sha"]:
            skipped.append((name, "retained log is not of the recorded commit"))
            continue
        files = executed_files(json.loads(coverage.read_text(encoding="utf-8")))
        if not files:
            skipped.append((name, "capture executed no workspace source"))
            continue
        row["executed_files"] = files
        gained += 1
    print(f"rows gaining executed_files: {gained} of {len(ledger)}; left bound to a commit only: {len(skipped)}")
    for name, why in skipped[:10]:
        print(f"  {name}: {why}")
    if apply and gained:
        import collect_coverage_records as collector

        collector.write_gate_ledger(ledger)
        print(f"wrote {GATES.relative_to(ROOT)}")
    return 0


def report() -> int:
    import calibration_groups as groups

    ledger = gate_rows()
    carried, owed = 0, {}
    for group in groups.gate_groups():
        ok, why = group_carried(rows_of(group.safe, group.points, ledger))
        if not any(rows_of(group.safe, group.points, ledger)):
            continue
        if ok:
            carried += 1
        else:
            owed[group.label] = why
    print(f"gate groups with a recorded pass carried to the tree: {carried}; not carried: {len(owed)}")
    for label, why in sorted(owed.items())[:20]:
        print(f"  {label}: {why}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    b = sub.add_parser("attach")
    b.add_argument("--logs", type=Path, required=True)
    b.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    return report() if args.command == "report" else attach(args.logs, args.apply)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
