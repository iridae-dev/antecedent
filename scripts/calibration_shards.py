#!/usr/bin/env python3
"""Split the calibration gate across runners by duration, not by position.

`scripts/gate_calibration.sh` owns the groups, their commands and the
2000-replicate recheck, which runs inline in the same runner as the group.
Groups are far from equal in duration: a Bayesian derivative cell whose recheck
fires runs for hours on its own, while most groups take seconds to minutes.
Assigning groups to shards by index modulo N can put two long groups on one
runner and fail it on its time limit instead of on its statistics.

This scheduler reads the group list from the gate's dry run and splits it
deterministically: each long group (named below, with the measurements that
make it long) gets a shard to itself, in gate order, and the short groups are
dealt in gate order across the remaining shards. Every runner computes the same
plan, so no coordination is needed. A shard count too small for that layout is
refused: raise the shard count, not the runner timeout. Each assigned group
then runs alone through the unchanged gate (`ANTECEDENT_CALIBRATION_SHARD=<index-1>/<groups>`
selects exactly that group), keeping the gate's recheck and record logs.

    python3 scripts/calibration_shards.py plan 24              # every shard
    python3 scripts/calibration_shards.py run 3/24             # run shard 3 of 24
    python3 scripts/calibration_shards.py run 3/24 --only-stale
    python3 scripts/calibration_shards.py run all --only-stale # locally, one process
    ANTECEDENT_CALIBRATION_DRY_RUN=1 python3 scripts/calibration_shards.py run 3/24

`--only-stale` keeps the groups that measure a record owing a
re-measurement (`scripts/calibration_facets.py`) and the pass/fail gates that
emit no record: records stand until the surface they measured changes, so
nothing else needs to run.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate_calibration.sh"

# Long groups: a single one can hold a runner for hours, so each gets a shard
# to itself. Evidence, from the per-group logs of the 1.10.0 merged-tree
# measurement (release build, two test threads):
#   * the v110_calibration_bayesian_static suite ran 14448 s, dominated by its
#     Bayesian derivative cells; the v19_derivative_calibration suite 2033 s;
#   * the recheck of point_derivative_bayesian_default_nominal_coverage alone
#     had run for more than 2 h 50 min without finishing;
#   * the recheck of interventional_distribution_admg_frontdoor_bayesian_nominal_coverage
#     alone ran 1419 s.
# Every other group measured seconds to a few minutes (the whole
# v19_temporal_response_calibration file: 444 s; v110_panel_calibration: 5 s).
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
# Short groups per shard. The index-modulo layout this replaces put 32 groups
# (255 over 8 shards) on a runner; a shard of short groups is held to that.
MAX_SHORT_PER_SHARD = 32


@dataclass
class Group:
    index: int  # 1-based position in the gate
    label: str
    long: bool
    shard: int = -1


def is_long(label: str) -> bool:
    return any(fnmatch.fnmatchcase(label, p) for p in LONG_PATTERNS)


def gate_groups() -> list[Group]:
    env = dict(os.environ, ANTECEDENT_CALIBRATION_DRY_RUN="1")
    env.pop("ANTECEDENT_CALIBRATION_SHARD", None)
    out = subprocess.run(
        ["bash", str(GATE)], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout
    groups = []
    for line in out.splitlines():
        m = re.fullmatch(r"group (\d+): (.+)", line)
        if m:
            groups.append(Group(int(m.group(1)), m.group(2), is_long(m.group(2))))
    if not groups or [g.index for g in groups] != list(range(1, len(groups) + 1)):
        raise SystemExit("could not read the group list from the gate's dry run")
    return groups


def plan(groups: list[Group], shards: int) -> list[Group]:
    """Long groups one per shard and alone; short groups spread by count over the rest."""
    longs = [g for g in groups if g.long]
    shorts = [g for g in groups if not g.long]
    free = shards - len(longs)
    if free < 1 or len(shorts) > free * MAX_SHORT_PER_SHARD:
        need = len(longs) + max(1, -(-len(shorts) // MAX_SHORT_PER_SHARD))
        raise SystemExit(
            f"{len(longs)} long groups each need a shard of their own and {len(shorts)} short "
            f"groups need {need - len(longs)} more at {MAX_SHORT_PER_SHARD} per shard: "
            f"{need} shards, not {shards}. Raise the shard count, not the timeout."
        )
    for shard, group in enumerate(longs):
        group.shard = shard
    for i, group in enumerate(shorts):
        group.shard = len(longs) + i % free
    return groups


def stale_filter(groups: list[Group]) -> list[Group]:
    """Groups that measure a record owing a re-measurement, plus every group that
    emits no record: a pass/fail gate (CI Type-I, discovery FPR, SBC) leaves no
    record behind, so nothing attests it and it always runs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibration_facets as facets

    records = facets.load_records()
    assessments = facets.assess(facets.load_surface(), records)
    wanted = [rec for a in assessments for rec in a.stale]
    wanted += [rec for a in assessments if not a.resolved for rec in a.records]
    selected = []
    matched: set[str] = set()
    for group in groups:
        head, _, test_filter = group.label.partition(": ")
        if not any(_measures(head, test_filter, rec) for rec in records):
            selected.append(group)
            continue
        hit = False
        for rec in wanted:
            if _measures(head, test_filter, rec):
                matched.add(str(rec["id"]))
                hit = True
        if hit:
            selected.append(group)
    orphans = sorted({str(r["test"]) for r in wanted if str(r["id"]) not in matched})
    for test in orphans:
        print(f"warning: no gate group measures {test}", file=sys.stderr)
    return selected


def _measures(head: str, test_filter: str, rec: dict) -> bool:
    """Does the gate group `<head>: <test_filter>` (or whole file `<head>`) run `rec`'s test?"""
    path, _, fn = str(rec["test"]).rpartition("::")
    stem = Path(path).stem
    owner = path.split("/")[1] if path.startswith("crates/") else ""
    if not test_filter:  # the group runs every test in the file
        return head == stem
    # A cargo filter matches by substring unless --exact: a superset.
    same_file = head == stem or (head == owner and "/src/" in path)
    return same_file and test_filter in fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="print every shard's groups")
    p_plan.add_argument("shards", type=int)
    p_plan.add_argument("--only-stale", action="store_true")
    p_run = sub.add_parser("run", help="run one shard")
    p_run.add_argument("shard", help="k/N, or `all` for every group in one process")
    p_run.add_argument("--only-stale", action="store_true")
    args = parser.parse_args()

    whole = args.command == "run" and args.shard == "all"
    if args.command == "plan":
        shards = args.shards
    elif whole:
        shards = 1
    else:
        m = re.fullmatch(r"(\d+)/(\d+)", args.shard)
        if not m or int(m.group(1)) >= int(m.group(2)):
            raise SystemExit(f"bad shard {args.shard!r} (want k/N with 0 <= k < N, or all)")
        shards = int(m.group(2))
    groups = gate_groups()
    total = len(groups)
    if whole:
        for group in groups:
            group.shard = 0
    else:
        plan(groups, shards)
    if args.only_stale:
        groups = stale_filter(groups)

    if args.command == "plan":
        for shard in range(shards):
            mine = [g for g in groups if g.shard == shard]
            kind = "long" if any(g.long for g in mine) else "short"
            print(f"shard {shard}: {len(mine)} {kind} group(s)")
            for g in mine:
                print(f"  group {g.index}: {g.label}")
        return 0

    shard = 0 if whole else int(m.group(1))
    mine = [g for g in groups if g.shard == shard]
    print(f"calibration shard {shard}/{shards}: {len(mine)} of {total} groups")
    if os.environ.get("ANTECEDENT_CALIBRATION_DRY_RUN"):
        for g in mine:
            print(f"group {g.index}{' (long)' if g.long else ''}: {g.label}")
        return 0
    failed = []
    for g in mine:
        env = dict(os.environ, ANTECEDENT_CALIBRATION_SHARD=f"{g.index - 1}/{total}")
        print(f"== group {g.index}: {g.label} ==", flush=True)
        if subprocess.run(["bash", str(GATE)], cwd=ROOT, env=env).returncode != 0:
            failed.append(g.label)
    if failed:
        print(f"calibration shard {shard}/{shards}: {len(failed)} group(s) failed:")
        for label in failed:
            print(f"  {label}")
        return 1
    print(f"calibration shard {shard}/{shards}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
