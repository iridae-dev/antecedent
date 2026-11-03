#!/usr/bin/env python3
"""Calibration status of the coverage records: an annotation, never a gate on code.

A coverage record in `parity/coverage_records.toml` was measured at some state of
the code. Its `fingerprint` is the dependency fingerprint of its `test` target
(`scripts/evidence_fingerprint.py`, the one the scientific records use) taken
when it was measured. A record is

* `current`            its stored `fingerprint` equals the fingerprint of the tree
                       now, or a passing probe attestation
                       (`parity/calibration_probe_attestations.toml`) covers its
                       measurement at a tree whose fingerprint equals the one now;
* `probe_owed`         not current, and its measurement stored replicate digests,
                       so `scripts/calibration_probe.py` can reproduce them;
* `measurement_owed`   not current and nothing to probe: only a re-measurement
                       (`scripts/measure_calibration.sh`) makes it current.

A record without a stored `fingerprint` is never current through its
measurement. Nothing here fails because of a code change: the exit status is 0
whatever is owed. Calibration is measured on a development machine; engineering
checks never read this.

    python3 scripts/calibration_status.py status            # counts, then the owed ids
    python3 scripts/calibration_status.py status --owed     # ids of the owed records, one per line
    python3 scripts/calibration_status.py status --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import calibration_probe as probe
from _repo import REPO_ROOT as ROOT
from _repo import read_text
from evidence_fingerprint import Fingerprinter, coverage_fingerprint

RECORDS = ROOT / "parity" / "coverage_records.toml"

CURRENT = "current"
PROBE_OWED = "probe_owed"
MEASUREMENT_OWED = "measurement_owed"
STATUSES = (CURRENT, PROBE_OWED, MEASUREMENT_OWED)


@dataclass(frozen=True)
class Status:
    status: str
    via: str  # "measurement" | "probe" | ""


def load_records(path: Path = RECORDS) -> list[dict]:
    if not path.is_file():
        return []
    return tomllib.loads(read_text(path)).get("record", [])


def load_attestations(path: Path = probe.ATTESTATIONS) -> list[dict]:
    if not path.is_file():
        return []
    return tomllib.loads(read_text(path)).get("attestation", [])


def tree_fingerprinter(root: Path = ROOT, metadata: dict | None = None) -> Fingerprinter:
    """The fingerprinter calibration compares with: platform-independent, since a
    record is measured on one machine and read on others."""
    return Fingerprinter(root, metadata, platform=False)


def tree_fingerprints(records: list[dict], fp: Fingerprinter) -> dict[str, str | None]:
    """record id -> fingerprint of its test target in the tree `fp` reads."""
    by_test: dict[tuple[str, str], str | None] = {}
    out: dict[str, str | None] = {}
    for rec in records:
        key = (str(rec.get("test", "")), str(rec.get("dgp", "")))
        if key not in by_test:
            by_test[key] = coverage_fingerprint(fp, rec)
        out[str(rec.get("id"))] = by_test[key]
    return out


def covered_by_probe(rec: dict, rows: list[dict], now: str | None) -> bool:
    """A passing probe of exactly this measurement, run at a tree whose fingerprint is `now`."""
    return bool(now) and any(
        probe.covers(rec, row) and row.get("fingerprint") == now for row in rows
    )


def classify(
    records: list[dict], attestations: list[dict], fingerprints: dict[str, str | None]
) -> dict[str, Status]:
    """record id -> status; `fingerprints` holds the fingerprint of each record's test now."""
    rows: dict[str, list[dict]] = {}
    for row in attestations:
        rows.setdefault(str(row.get("record_id", "")), []).append(row)
    out: dict[str, Status] = {}
    for rec in records:
        rid = str(rec.get("id"))
        now = fingerprints.get(rid)
        stored = rec.get("fingerprint")
        if now and stored == now:
            out[rid] = Status(CURRENT, "measurement")
        elif covered_by_probe(rec, rows.get(rid, []), now):
            out[rid] = Status(CURRENT, "probe")
        elif probe.stored_prefixes(rec):
            out[rid] = Status(PROBE_OWED, "")
        else:
            out[rid] = Status(MEASUREMENT_OWED, "")
    return out


def current_ids(root: Path = ROOT) -> list[str]:
    """Ids of the records that are current in the tree at `root`: the set the runtime snapshot carries."""
    records = load_records(root / "parity" / "coverage_records.toml")
    statuses = classify(
        records,
        load_attestations(root / "parity" / "calibration_probe_attestations.toml"),
        tree_fingerprints(records, tree_fingerprinter(root)),
    )
    return sorted(rid for rid, s in statuses.items() if s.status == CURRENT)


def summary(statuses: dict[str, Status]) -> dict[str, int]:
    counts = {name: 0 for name in STATUSES}
    for s in statuses.values():
        counts[s.status] += 1
    counts["current_by_probe"] = sum(1 for s in statuses.values() if s.via == "probe")
    return counts


def cmd_status(args: argparse.Namespace) -> int:
    records = load_records()
    statuses = classify(records, load_attestations(), tree_fingerprints(records, tree_fingerprinter()))
    counts = summary(statuses)
    owed = {rid: s.status for rid, s in sorted(statuses.items()) if s.status != CURRENT}
    if args.json:
        json.dump({"counts": counts, "records": {r: s.status for r, s in sorted(statuses.items())}}, sys.stdout, indent=2)
        print()
    elif args.owed:
        for rid in owed:
            print(rid)
    else:
        print(
            f"calibration records: {counts[CURRENT]} current "
            f"({counts['current_by_probe']} by probe), {counts[PROBE_OWED]} probe owed, "
            f"{counts[MEASUREMENT_OWED]} measurement owed of {len(statuses)}"
        )
        for rid, status in owed.items():
            print(f"  {status}: {rid}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status")
    status.add_argument("--owed", action="store_true", help="print only the ids of the records that are not current")
    status.add_argument("--json", action="store_true", help="machine-readable counts and per-record status")
    return cmd_status(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
