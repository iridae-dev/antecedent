#!/usr/bin/env python3
"""Recorded passes of the scientific assertions, carried forward while nothing they depend on changed.

    scientific_records.py status [--check]   owed versus current records
    scientific_records.py run [--all]        run the owed assertions, record passes

`parity/scientific_records.toml` holds one row per scientific assertion that
passed: its exact selector, roles, the commit it passed at, a fingerprint of
everything the assertion depends on, and when. A row is current while the
assertion's fingerprint is unchanged, so a no-change rerun executes nothing and
a leaf-crate edit owes only the assertions whose dependency closure contains
that crate. `--all` or ANTECEDENT_SCIENTIFIC_ALL=1 (a release candidate) runs
and re-records everything.

A recorded pass is evidence for support only. It is not a calibration
measurement and no coverage or support claim reads it as one.

The fingerprint is `scripts/evidence_fingerprint.py`: one implementation shared with the
calibration flow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from _repo import REPO_ROOT, read_text
from evidence_fingerprint import Fingerprinter, locate

RECORDS = "parity/scientific_records.toml"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

HEADER = """# Recorded passes of the scientific assertions (see scripts/scientific_records.py).
#
# One [[record]] per scientific assertion that passed, written by
# `python3 scripts/scientific_records.py run` and by the gates that execute
# scientific rows. A record is current while its fingerprint matches; an owed
# assertion reruns, a current one carries forward. A recorded pass is evidence for
# support only. It is never a calibration measurement or a coverage claim.
#
# test         exact selector: rust `crate::binary::module::name` (`lib` for the
#              library target) or python node id relative to python/
# roles        evidence roles of the citing claims
# passed       always true; failures are reported, never recorded
# commit       HEAD when the pass was recorded
# dirty        whether the working tree differed from that commit
# fingerprint  sha256 over the assertion's dependency closure
# recorded_at  UTC time of the pass
"""


def load_records(root: Path) -> dict[str, dict]:
    path = root / RECORDS
    if not path.is_file():
        return {}
    data = tomllib.loads(read_text(path))
    return {r["test"]: r for r in data.get("record", [])}


def write_records(root: Path, records: dict[str, dict]) -> None:
    out = [HEADER]
    for test in sorted(records):
        r = records[test]
        out.append(
            "\n[[record]]\n"
            f"test = {json.dumps(test)}\n"
            f"roles = {json.dumps(list(r['roles']))}\n"
            "passed = true\n"
            f"commit = {json.dumps(r['commit'])}\n"
            f"dirty = {'true' if r.get('dirty') else 'false'}\n"
            f"fingerprint = {json.dumps(r['fingerprint'])}\n"
            f"recorded_at = {json.dumps(r['recorded_at'])}\n"
        )
    (root / RECORDS).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORDS).write_text("".join(out), encoding="utf-8", newline="\n")


def git_state(root: Path) -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    return head.stdout.strip() or "unknown", bool(dirty.stdout.strip())


def is_current(records: dict[str, dict], selector: str, fingerprint: str) -> bool:
    rec = records.get(selector)
    return bool(rec and rec.get("passed") is True and rec.get("fingerprint") == fingerprint)


def classify(root: Path, rows: list[dict], fp: Fingerprinter) -> tuple[list[dict], list[dict]]:
    """(owed, current) rows, each annotated with selector and fingerprint."""
    records = load_records(root)
    owed, current = [], []
    for row in rows:
        info = locate(root, row)
        item = {**row, "selector": info["selector"], "fingerprint": fp.fingerprint(row)}
        (current if is_current(records, item["selector"], item["fingerprint"]) else owed).append(item)
    return owed, current


def record_passes(root: Path, items: list[dict]) -> int:
    """Merge passing rows (annotated by classify) into the ledger."""
    if not items:
        return 0
    commit, dirty = git_state(root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = load_records(root)  # reread: another gate may have written meanwhile
    for item in items:
        records[item["selector"]] = {
            "roles": item.get("roles") or ["scientific"],
            "commit": commit,
            "dirty": dirty,
            "fingerprint": item["fingerprint"],
            "recorded_at": now,
        }
    write_records(root, records)
    return len(items)


def all_scientific_rows(root: Path) -> list[dict]:
    import evidence_audit

    return evidence_audit.scientific_rows(evidence_audit.build_report(root)["claims"])


class GateSession:
    """What run_evidence_rows.py uses: carry current scientific rows, record new passes."""

    def __init__(self, root: Path, metadata: dict | None = None):
        self.root = root
        self.rows = {(r["evidence_test"], r["evidence_assertion"]): r for r in all_scientific_rows(root)}
        self.fp = Fingerprinter(root, metadata)
        self.records = load_records(root)
        self.annotated: dict[tuple[str, str], dict] = {}

    def annotate(self, row: dict) -> dict | None:
        """The scientific row this registry row cites, or None when it is not carryable."""
        key = (row.get("evidence_test"), row.get("evidence_assertion"))
        if key not in self.rows or row.get("composition_filter"):
            return None
        if key not in self.annotated:
            item = self.rows[key]
            info = locate(self.root, item)
            self.annotated[key] = {**item, "selector": info["selector"], "fingerprint": self.fp.fingerprint(item)}
        return self.annotated[key]

    def current(self, row: dict) -> bool:
        item = self.annotate(row)
        return bool(item and is_current(self.records, item["selector"], item["fingerprint"]))

    def record(self, rows: list[dict]) -> int:
        items = {}
        for row in rows:
            item = self.annotate(row)
            if item:
                items[item["selector"]] = item
        return record_passes(self.root, list(items.values()))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def force_all(args: argparse.Namespace | None = None) -> bool:
    return bool(getattr(args, "all", False)) or os.environ.get("ANTECEDENT_SCIENTIFIC_ALL") == "1"


def cmd_status(root: Path, args: argparse.Namespace) -> int:
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows, Fingerprinter(root))
    print(f"scientific records: {len(current)} current, {len(owed)} owed of {len(rows)}")
    for item in owed:
        print(f"  owed: {item['selector']}")
    return int(args.check and bool(owed))


def python_environment_ok(root: Path) -> bool:
    for candidate in (root / "python/.venv/bin/python", root / "python/.venv/Scripts/python.exe"):
        if candidate.is_file():
            return subprocess.run([str(candidate), str(root / "scripts/native_build_identity.py"), "check"]).returncode == 0
    print("FAIL: the Python conformance environment is missing; run bash scripts/test.sh python")
    return False


def cmd_run(root: Path, args: argparse.Namespace) -> int:
    rows = all_scientific_rows(root)
    owed, current = classify(root, rows, Fingerprinter(root))
    run_items = (owed + current) if force_all(args) else owed
    print(
        f"scientific records: running {len(run_items)} of {len(rows)} "
        f"({len(current) if not force_all(args) else 0} carried forward)",
        flush=True,
    )
    if not run_items:
        return 0
    if any(i["evidence_test"].endswith(".py") for i in run_items) and not python_environment_ok(root):
        return 1
    with tempfile.TemporaryDirectory(prefix="antecedent-scientific-") as tmp:
        ledger = Path(tmp) / "selection.toml"
        passed_out = Path(tmp) / "passed.json"
        ledger.write_text(
            "".join(
                "[[capabilities]]\n"
                f"id = {json.dumps(i['selector'])}\n"
                f"evidence_test = {json.dumps(i['evidence_test'])}\n"
                f"evidence_assertion = {json.dumps(i['evidence_assertion'])}\n\n"
                for i in run_items
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "ANTECEDENT_SCIENTIFIC_ALL": "1",  # run exactly what was selected
            "ANTECEDENT_SCIENTIFIC_RECORD": "0",  # this command records, from the proven ids
            "ANTECEDENT_EVIDENCE_PASSED_OUT": str(passed_out),
        }
        code = subprocess.run(
            [sys.executable, str(root / "scripts/run_evidence_rows.py"), str(root), str(ledger), str(root),
             "capabilities", "scientific"],
            cwd=root, env=env,
        ).returncode
        passed = set(json.loads(passed_out.read_text(encoding="utf-8"))) if passed_out.is_file() else set()
    recorded = record_passes(root, [i for i in run_items if i["selector"] in passed])
    print(f"scientific records: recorded {recorded} pass(es); {len(run_items) - recorded} not recorded")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--check", action="store_true", help="exit 1 when any assertion is owed")
    run = sub.add_parser("run")
    run.add_argument("--all", action="store_true", help="rerun and re-record every assertion (release candidate)")
    args = parser.parse_args()
    return {"status": cmd_status, "run": cmd_run}[args.command](REPO_ROOT, args)


if __name__ == "__main__":
    raise SystemExit(main())
