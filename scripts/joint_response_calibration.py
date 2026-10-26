#!/usr/bin/env python3
"""List or measure the registered Q08/E14 joint-response procedures on a stable SHA.

This runner emits candidate-bound execution logs. It never updates coverage
licenses or borrows a risk allocation from the frozen historical family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "parity/joint_response_calibration.toml"


def procedures() -> list[dict]:
    document = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    if document["schema"] != 1 or document["grants_release_license"]:
        raise ValueError("invalid unmeasured procedure registry")
    rows = document["procedure"]
    ids = set()
    for row in rows:
        if row["id"] in ids or row["cargo_target"] not in {"lib", "suite_01"}:
            raise ValueError("duplicate procedure or unsupported Cargo target")
        ids.add(row["id"])
        source = ROOT / row["source"]
        if not source.is_file() or not re.search(
            rf"\bfn\s+{re.escape(row['test'].split('::')[-1])}\s*\(",
            source.read_text(encoding="utf-8"),
        ):
            raise ValueError(f"missing executable calibration test: {row['id']}")
        if row["minimum_replicates"] < 400 or not row["cells"] or not row["stages"]:
            raise ValueError("incomplete calibration procedure")
    return rows


def command(row: dict) -> list[str]:
    target = (
        ["--lib"] if row["cargo_target"] == "lib" else ["--test", row["cargo_target"]]
    )
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        row["cargo_package"],
        *target,
        row["test"],
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-stable-candidate", metavar="GIT_SHA")
    parser.add_argument("--only", help="one registered procedure ID")
    parser.add_argument("--replicates", type=int, default=400)
    args = parser.parse_args()
    rows = procedures()
    if args.only:
        rows = [row for row in rows if row["id"] == args.only]
        if not rows:
            parser.error("unknown procedure")
    if not args.run_stable_candidate:
        print(
            json.dumps(
                [
                    {
                        "id": row["id"],
                        "cells": row["cells"],
                        "command": command(row),
                        "status": "registered_unmeasured",
                    }
                    for row in rows
                ],
                indent=2,
            )
        )
        return 0
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if sha != args.run_stable_candidate:
        parser.error("requested stable candidate does not equal HEAD")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    if dirty.strip():
        parser.error(
            "final measurement requires a clean stable candidate, including untracked sources"
        )
    if any(args.replicates < row["minimum_replicates"] for row in rows):
        parser.error("replicate count is below the registered procedure minimum")
    output = ROOT / "target/calibration-records/joint-response" / sha
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, ANTECEDENT_CALIBRATION_NSIM=str(args.replicates))
    failures = 0
    for row in rows:
        log = output / f"{row['id']}.log"
        metadata = {
            "candidate_sha": sha,
            "procedure": row["id"],
            "replicates": args.replicates,
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "grants_release_license": False,
        }
        with log.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata) + "\n")
            stream.flush()
            result = subprocess.run(
                command(row),
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        failures += int(result.returncode != 0)
        print(json.dumps(dict(metadata, exit_code=result.returncode, log=str(log))))
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
