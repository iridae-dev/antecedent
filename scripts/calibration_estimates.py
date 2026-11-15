#!/usr/bin/env python3
"""Turn a per-group cost model into the duration-estimate file a first run orders by.

A clean checkout has no `target/calibration-timings.tsv`, so the first run takes gate order
and a single long job can start last. This writes `label<TAB>seconds` rows, one per
(group, grid point) task label the planner uses, for `calibration_groups.py run
--estimates FILE` (or `ANTECEDENT_CALIBRATION_ESTIMATES=FILE`). The file is an ordering and
sizing hint only: recorded timings and pilots replace it label by label, no gate reads it, and
it belongs under `target/`, not in the repository.

    python3 scripts/calibration_estimates.py MODEL.json --out target/calibration-estimates.tsv

MODEL.json is a list of `{"harness": ..., "fn": ..., "secs": [seconds per grid point]}`: the
group's label is `harness: fn`; three values are its three grid points, one value a group run once.
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT

GROUPS = "parity/calibration_groups.toml"


def estimate_rows(model: list[dict], labels: set[str]) -> tuple[list[tuple[str, float]], list[str]]:
    """The estimate rows and the model entries that name no group."""
    rows, unknown = [], []
    for entry in model:
        label = f"{entry['harness']}: {entry['fn']}"
        if label not in labels:
            unknown.append(label)
            continue
        secs = entry["secs"]
        if len(secs) == 1:
            rows.append((label, float(secs[0])))
        elif len(secs) == 3:
            rows.extend((f"{label} [grid point {point}]", float(s)) for point, s in enumerate(secs))
        else:
            raise SystemExit(f"{label}: want 1 or 3 durations, got {len(secs)}")
    return rows, unknown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = {g["label"] for g in tomllib.loads((REPO_ROOT / GROUPS).read_text(encoding="utf-8"))["group"]}
    rows, unknown = estimate_rows(json.loads(args.model.read_text(encoding="utf-8")), labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Duration ESTIMATES (seconds, one thread) from a cost model, not measurements. Ordering hint only.\n"
    args.out.write_text(header + "".join(f"{label}\t{seconds:.1f}\n" for label, seconds in rows),
                        encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} estimate(s) to {args.out}" + (f"; {len(unknown)} unknown group(s) skipped" if unknown else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
