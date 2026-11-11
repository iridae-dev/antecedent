#!/usr/bin/env python3
"""Keep a coverage record current by re-running its first replicates.

A coverage record stands as long as the numbers it measured are unchanged. Each
measurement stores, per grid point, the chained digests of its first
10 / 32 / 100 / 320 / 1000 replicates (`probe` in `parity/coverage_records.toml`;
`prefix_digests` in `crates/antecedent/tests/common/calibration.rs`). Replicate `k`
is a deterministic function of `k`, so a run at `ANTECEDENT_CALIBRATION_NSIM=k` at
another commit reproduces the stored prefix digest exactly when the first `k`
replicates are bit-identical. A match for `k` replicates bounds the per-replicate
rate of any change at `3/k` (rule of three, 95%); that bound is what the record's
margin to its acceptance floor has to cover.

    python3 scripts/calibration_probe.py quick   [--record ID ...] [--dry-run]
    python3 scripts/calibration_probe.py bounded [--record ID ...] [--dry-run]

* `quick` probes k = 10 at every grid point: a cheap detector. A mismatch marks
  the record owed; a match proves nothing unless the margin is large.
* `bounded` probes the smallest stored prefix `k >= 3/margin`, where `margin` is
  the observed coverage minus the acceptance floor at the record's worst grid
  point (3 MCSE for a boundary record, whose band is centered on its measurement).
  A record whose margin needs more than the longest stored prefix, or that has no
  stored digests, is reported `owed`: only a re-measurement keeps it.

Without `--record` the probes cover the records that are not current
(`scripts/calibration_status.py status --owed`) and carry digests. Each group runs through
the shared calibration runner (`scripts/calibration_runner.py`) exactly as a measurement does (same frozen decision
environment, so the seeds match), in smoke mode so a short run is never gated,
with logs in `target/calibration-probe/`. Results go to
`parity/calibration_probe_attestations.toml`, one row per record and measurement:

    record_id, calibration_sha (the measurement compared against), sha (HEAD),
    fingerprint (the record's dependency fingerprint at HEAD,
    `scripts/evidence_fingerprint.py`), k, bound = 3/k, result = "pass" | "mismatch"

`scripts/calibration_status.py` counts a record as current when a passing row
covers it (`covers`) and its fingerprint equals the tree's. Commit the attestation
file with the change it attests. A real run needs a
clean tree (the gate refuses a dirty one); `--dry-run` prints the commands only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import _repo
from _repo import REPO_ROOT as ROOT
from _repo import load_toml

ATTESTATIONS = ROOT / "parity" / "calibration_probe_attestations.toml"
PROBE_LOG_DIR = ROOT / "target" / "calibration-probe"

# `DIGEST_PREFIXES` in crates/antecedent/tests/common/calibration.rs.
PREFIXES = (10, 32, 100, 320, 1000)
QUICK_K = PREFIXES[0]
GRID_POINTS = (0, 1, 2)
RULE_OF_THREE = 3.0
RESULTS = ("pass", "mismatch")
SHA = re.compile(r"[0-9a-f]{40}")
HEX_DIGEST = re.compile(r"[0-9a-f]{32}")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")

HEADER = """# Probe attestations for coverage records (scripts/calibration_probe.py).
#
# A row says: the first `k` replicates of `record_id`, re-run at commit `sha`,
# reproduced the digests stored with the measurement made at `calibration_sha`
# (`result = "pass"`), or did not (`"mismatch"`). `bound = 3/k` bounds the
# per-replicate rate of an undetected change. `fingerprint` is the record's test
# fingerprint (scripts/evidence_fingerprint.py) at `sha`; scripts/calibration_status.py
# counts the record as current while a passing row covers it and that fingerprint
# equals the tree's. Written by the probe; do not edit rows by hand.
"""


# --------------------------------------------------------------------------
# Sizing: how many replicates a record's margin needs.
# --------------------------------------------------------------------------


def acceptance_floor(nominal: float, replicates: int) -> float:
    """Lowest observed coverage the gate accepts at `replicates` (the two-sided
    3-MCSE band's lower edge, raised to the 2-MCSE precision floor from 1000)."""
    mcse = math.sqrt(nominal * (1.0 - nominal) / max(replicates, 1))
    floor = max(nominal - 3.0 * mcse, 0.0)
    if replicates >= 1000:
        floor = max(floor, nominal - 2.0 * mcse)
    return floor


def point_margin(record: dict, entry: dict) -> float:
    """Coverage the measured point could lose before it failed its acceptance.

    A boundary point (a documented under-coverage or a failing cell) is held to its
    own measurement `± 3 MCSE`, so its margin is 3 MCSE."""
    if entry.get("boundary") or entry.get("role") == "named_boundary":
        return 3.0 * float(entry["mcse"])
    return float(entry["observed"]) - acceptance_floor(
        float(record["nominal"]), int(entry["replicates"])
    )


def record_margin(record: dict) -> float | None:
    """Margin at the record's worst grid point; `None` when it has no grid."""
    grid = record.get("grid")
    if not isinstance(grid, list) or not grid:
        return None
    return min(point_margin(record, entry) for entry in grid)


def stored_prefixes(record: dict) -> list[int]:
    """Prefix lengths stored at every grid point, ascending."""
    table = record.get("probe")
    if not isinstance(table, dict):
        return []
    common: set[int] | None = None
    for point in GRID_POINTS:
        digests = table.get(f"p{point}")
        if not isinstance(digests, dict):
            return []
        keys = {int(k) for k in digests if str(k).isdigit()}
        common = keys if common is None else common & keys
    return sorted(common or [])


def required_k(record: dict) -> int | None:
    """Replicates whose reproduction bounds a change below the margin; `None`
    when the record has no positive margin."""
    margin = record_margin(record)
    if margin is None or margin <= 0.0:
        return None
    return math.ceil(RULE_OF_THREE / margin - 1e-9)


def bounded_k(record: dict) -> int | None:
    """Smallest stored prefix at least `3/margin`; `None` (owed) when the record
    stores no digests or its margin needs more replicates than any stored prefix."""
    need = required_k(record)
    if need is None:
        return None
    return next((k for k in stored_prefixes(record) if k >= need), None)


def quick_k(record: dict) -> int | None:
    return QUICK_K if QUICK_K in stored_prefixes(record) else None


# --------------------------------------------------------------------------
# Comparison.
# --------------------------------------------------------------------------


def compare(record: dict, observed: dict[int, dict[str, str]], k: int) -> list[str]:
    """What differs between the record's stored digests and a probe at `k`
    replicates (`observed`: grid point -> prefix digests); empty means a match.

    Every stored prefix up to `k` must be reproduced, so a divergence is located
    to the shortest prefix that shows it."""
    problems = []
    table = record.get("probe") or {}
    for point in GRID_POINTS:
        stored = table.get(f"p{point}")
        if not isinstance(stored, dict) or str(k) not in stored:
            problems.append(f"grid point {point}: no stored digest for the first {k} replicates")
            continue
        seen = observed.get(point)
        if seen is None:
            problems.append(f"grid point {point}: the probe emitted no line for this record")
            continue
        if str(k) not in seen:
            problems.append(f"grid point {point}: the probe produced fewer than {k} replicates")
            continue
        for prefix in sorted(int(p) for p in stored if str(p).isdigit() and int(p) <= k):
            if seen.get(str(prefix)) != stored[str(prefix)]:
                problems.append(f"grid point {point}: the first {prefix} replicates differ")
                break
    return problems


# --------------------------------------------------------------------------
# Attestation rows.
# --------------------------------------------------------------------------


def covers(record: dict, row: dict) -> bool:
    """Whether `row` is a passing probe of exactly this measurement that is long
    enough for the record's margin."""
    if row.get("result") != "pass" or row.get("record_id") != record.get("id"):
        return False
    if row.get("calibration_sha") != record.get("calibration_sha"):
        return False
    if row.get('dependency_manifest'):
        from evidence_earning import measurement_digest
        if row.get('measurement_digest') != measurement_digest(record):
            return False
    k, bound = row.get("k"), row.get("bound")
    if type(k) is not int or isinstance(bound, bool) or not isinstance(bound, (int, float)):
        return False
    need = bounded_k(record)
    if need is None or k < need or k not in stored_prefixes(record):
        return False
    return math.isclose(float(bound), RULE_OF_THREE / k, rel_tol=1e-9)


def attestation_problems(rows: list[dict], records: list[dict]) -> list[str]:
    """Malformed rows, and rows that no longer describe a record's measurement."""
    by_id = {str(rec.get("id")): rec for rec in records}
    problems = []
    for index, row in enumerate(rows):
        label = f"calibration_probe_attestations.toml row {index + 1} ({row.get('record_id')})"
        missing = [
            key
            for key in ("record_id", "calibration_sha", "sha", "fingerprint", "k", "bound", "result")
            if key not in row
        ]
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue
        if row["result"] not in RESULTS:
            problems.append(f"{label}: result must be one of {', '.join(RESULTS)}")
        if not SHA.fullmatch(str(row["sha"])) or not SHA.fullmatch(str(row["calibration_sha"])):
            problems.append(f"{label}: sha and calibration_sha must be full commit ids")
        if not FINGERPRINT.fullmatch(str(row["fingerprint"])):
            problems.append(f"{label}: fingerprint must be a sha256 hex digest")
        if type(row["k"]) is not int or row["k"] not in PREFIXES:
            problems.append(f"{label}: k must be one of {', '.join(map(str, PREFIXES))}")
        elif not math.isclose(float(row["bound"]), RULE_OF_THREE / row["k"], rel_tol=1e-9):
            problems.append(f"{label}: bound must be 3/k")
        record = by_id.get(str(row["record_id"]))
        if record is None:
            problems.append(f"{label}: no such coverage record; delete the row")
        elif record.get("calibration_sha") != row["calibration_sha"]:
            problems.append(
                f"{label}: it probed a measurement the record no longer holds; "
                "delete the row or probe again"
            )
    return problems


def prune_attestations(records: list[dict], path: Path = ATTESTATIONS) -> int:
    """Drop rows whose record is gone or was measured again; returns how many."""
    if not path.is_file():
        return 0
    rows = _repo.load_toml(path).get("attestation", [])
    current = {str(r.get("id")): r.get("calibration_sha") for r in records}
    kept = [
        row for row in rows if current.get(str(row.get("record_id"))) == row.get("calibration_sha")
    ]
    if len(kept) != len(rows):
        path.write_text(render_attestations(kept), encoding="utf-8", newline="\n")
    return len(rows) - len(kept)


def render_attestations(rows: list[dict]) -> str:
    lines = [HEADER]
    for row in sorted(rows, key=lambda r: (str(r["record_id"]), str(r["sha"]))):
        lines += [
            "[[attestation]]",
            f"record_id = {json.dumps(str(row['record_id']))}",
            f"calibration_sha = {json.dumps(str(row['calibration_sha']))}",
            f"sha = {json.dumps(str(row['sha']))}",
            f"fingerprint = {json.dumps(str(row['fingerprint']))}",
            f"k = {int(row['k'])}",
            f"bound = {float(row['bound'])!r}",
            f"result = {json.dumps(str(row['result']))}",
            "",
        ]
        if row.get('dependency_manifest'):
            lines.insert(len(lines)-1, f"dependency_manifest = {json.dumps(row['dependency_manifest'])}")
            lines.insert(len(lines)-1, f"measurement_digest = {json.dumps(row['measurement_digest'])}")
    return "\n".join(lines)


def merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Fold fresh probe rows into the file.

    A mismatch is the whole story of its measurement: it replaces every earlier
    row of that record and measurement, so no older pass survives it. A pass
    replaces the row at the same commit only when it ran at least as many
    replicates, so a quick probe never downgrades a bounded one."""
    rows = list(existing)
    for row in new:
        record, measurement = str(row["record_id"]), str(row["calibration_sha"])

        def same_measurement(other: dict, key: tuple[str, str] = (record, measurement)) -> bool:
            return (str(other.get("record_id")), str(other.get("calibration_sha"))) == key

        if row["result"] != "pass":
            rows = [r for r in rows if not same_measurement(r)]
            rows.append(row)
            continue
        if any(same_measurement(r) and r.get("result") != "pass" for r in rows):
            rows = [r for r in rows if not same_measurement(r)]
        twins = [r for r in rows if same_measurement(r) and r.get("sha") == row["sha"]]
        if any(int(t["k"]) > int(row["k"]) for t in twins):
            continue
        rows = [r for r in rows if r not in twins]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Plan: which probes a set of records needs.
# --------------------------------------------------------------------------


@dataclass
class Probe:
    record: dict
    k: int | None
    owed_reason: str | None = None


def plan(records: list[dict], mode: str) -> list[Probe]:
    """One `Probe` per record: its `k`, or why it is owed without a run."""
    out = []
    for rec in records:
        if not stored_prefixes(rec):
            out.append(Probe(rec, None, "no digests are stored with this measurement"))
        elif mode == "quick":
            k = quick_k(rec)
            out.append(Probe(rec, k, None if k else "no stored prefix of 10 replicates"))
        else:
            k = bounded_k(rec)
            if k is not None:
                out.append(Probe(rec, k))
                continue
            margin = record_margin(rec)
            if margin is None or margin <= 0.0:
                reason = "its margin to the acceptance floor is not positive"
            else:
                reason = (
                    f"its margin {margin:.4f} needs {required_k(rec)} replicates; "
                    f"the longest stored prefix is {stored_prefixes(rec)[-1]}"
                )
            out.append(Probe(rec, None, reason))
    return out


# --------------------------------------------------------------------------
# Running probes through the gate.
# --------------------------------------------------------------------------


@dataclass
class Task:
    group: object  # calibration_groups.Group
    point: int
    k: int
    records: list[dict] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return f"{self.group.safe}.p{self.point}"


def tasks_for(probes: list[Probe], groups: list) -> tuple[list[Task], list[str]]:
    """Gate (group, grid point) jobs covering the planned probes, each run at the
    longest `k` any of its records needs, and the ids of records no group measures."""
    import calibration_groups

    tasks: dict[tuple[int, int], Task] = {}
    unmatched = []
    for probe in probes:
        if probe.k is None:
            continue
        rec = probe.record
        found = [g for g in groups if calibration_groups.measures(g.head, g.test_filter, rec)]
        fn = str(rec["test"]).rpartition("::")[2]
        exact = [g for g in found if g.label.partition(": ")[2] == fn]
        chosen = exact or found
        if not chosen:
            unmatched.append(str(rec["id"]))
            continue
        for group in chosen:
            for point in GRID_POINTS:
                task = tasks.setdefault((group.index, point), Task(group, point, probe.k))
                task.k = max(task.k, probe.k)
                task.records.append(rec)
    return list(tasks.values()), unmatched


def dry_run_line(task: Task, total: int, threads: int) -> str:
    ids = ", ".join(sorted({str(r["id"]) for r in task.records}))
    return (f"# {task.group.label} [grid point {task.point}] k={task.k}: {ids}\n"
            f"measure group {task.group.index} of {total} at grid point {task.point}: {task.k} smoke "
            f"replicate(s), {threads} thread(s), frozen decision context")


def parse_log(text: str) -> tuple[dict[tuple[str, int], dict[str, str]], str | None]:
    """`(record id, grid point) -> prefix digests` from a probe log, and the
    commit the gate stamped on it."""
    seen: dict[tuple[str, int], dict[str, str]] = {}
    measured = None
    for line in text.splitlines():
        if line.startswith("calibration-measured-at "):
            measured = line.split()[1]
        elif line.startswith("calibration-record "):
            payload = json.loads(line.split(" ", 1)[1])
            digests = payload.get("prefix_digests")
            if isinstance(digests, dict) and "grid_point" in payload:
                seen[(str(payload["id"]), int(payload["grid_point"]))] = {
                    str(k): str(v) for k, v in digests.items()
                }
    return seen, measured


def head_sha() -> str:
    return _repo.strict_git_head(ROOT)


def worktree_dirty() -> bool:
    return _repo.strict_git_dirty(ROOT)


def run_tasks(tasks: list[Task], total: int, jobs: int, decision_json: str) -> dict[str, dict]:
    """Run each task as a measurement does (`calibration_runner.execute`) at its `k`
    replicates in smoke mode: the same frozen decision context and seeds, never gated.
    `stem -> {status, observed, measured_at}`."""
    import calibration_groups
    from calibration_runner import Task as RunTask
    from calibration_runner import execute

    PROBE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    for stale in PROBE_LOG_DIR.glob("*"):
        stale.unlink()
    threads = max(1, (os.cpu_count() or 1) // max(jobs, 1))
    results: dict[str, dict] = {}
    by_key = {(task.group.index, task.point): task for task in tasks}

    def done(run: RunTask, result: dict) -> None:
        task = by_key[run.key]
        log = PROBE_LOG_DIR / f"{task.stem}.log"
        observed, measured = (
            parse_log(log.read_text(encoding="utf-8", errors="ignore")) if log.is_file() else ({}, None)
        )
        results[task.stem] = {"status": result["status"], "observed": observed, "measured_at": measured}
        print(f"  {'ok' if result['status'] == 0 else 'FAILED'}: {task.group.label} [p{task.point}] k={task.k}",
              flush=True)

    execute(ROOT, [RunTask(task.group.spec, task.point, threads, task.k) for task in tasks],
            env=calibration_groups._measurement_env(strip_prefix=True), log_dir=PROBE_LOG_DIR,
            jobs=max(jobs, 1), first_nsim=calibration_groups.FIRST_NSIM,
            recheck_nsim=calibration_groups.RECHECK_NSIM, smoke=True, decision_context=decision_json,
            on_done=done)
    return results


def judge(
    probes: list[Probe],
    tasks: list[Task],
    results: dict[str, dict],
    sha: str,
    fingerprints: dict[str, str | None],
) -> tuple[list[dict], dict[str, str]]:
    """Attestation rows and a per-record verdict line from the finished runs."""
    rows, verdicts = [], {}
    for probe in probes:
        rid = str(probe.record["id"])
        if probe.k is None:
            verdicts[rid] = f"owed: {probe.owed_reason}"
            continue
        observed: dict[int, dict[str, str]] = {}
        failed = []
        for task in tasks:
            if not any(r is probe.record or r.get("id") == rid for r in task.records):
                continue
            result = results.get(task.stem)
            if result is None or result["status"] != 0 or result["measured_at"] != sha:
                failed.append(task.stem)
                continue
            seen = result["observed"].get((rid, task.point))
            if seen is not None:
                observed[task.point] = seen
        if failed:
            verdicts[rid] = f"unprobed: the gate failed for {', '.join(sorted(failed))}"
            continue
        fingerprint = fingerprints.get(rid)
        if not fingerprint:
            verdicts[rid] = "unprobed: the record's test has no computable fingerprint"
            continue
        problems = compare(probe.record, observed, probe.k)
        rows.append(
            {
                "record_id": rid,
                "calibration_sha": probe.record["calibration_sha"],
                "sha": sha,
                "fingerprint": fingerprint,
                "k": probe.k,
                "bound": RULE_OF_THREE / probe.k,
                "result": "mismatch" if problems else "pass",
            }
        )
        verdicts[rid] = (
            "owed (mismatch): " + "; ".join(problems)
            if problems
            else f"pass at k={probe.k} (bound {RULE_OF_THREE / probe.k:.4f})"
        )
    return rows, verdicts


# --------------------------------------------------------------------------
# Command line.
# --------------------------------------------------------------------------


def select_records(ids: list[str]) -> list[dict]:
    import calibration_status as status

    records = status.load_records()
    if ids:
        by_id = {str(r["id"]): r for r in records}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"no such coverage record(s): {', '.join(missing)}")
        return [by_id[i] for i in ids]
    verdicts = status.classify(
        records,
        status.load_attestations(),
        status.tree_fingerprints(records),
    )
    return [r for r in records if verdicts[str(r["id"])].status != status.CURRENT]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("quick", "bounded"))
    parser.add_argument("--record", action="append", default=[], help="probe this record id")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--jobs", type=int, default=2, help="gate jobs at a time (default 2)")
    args = parser.parse_args(argv)

    records = select_records(args.record)
    if not records:
        print("nothing to probe: no record owes a run")
        return 0
    probes = plan(records, args.mode)
    import calibration_groups

    groups = calibration_groups.gate_groups()
    tasks, unmatched = tasks_for(probes, groups)
    for rid in unmatched:
        print(f"warning: no gate group measures {rid}", file=sys.stderr)
    threads = max(1, (os.cpu_count() or 1) // max(args.jobs, 1))
    if args.dry_run:
        for task in tasks:
            print(dry_run_line(task, len(groups), threads))
        for probe in probes:
            if probe.k is None:
                print(f"# {probe.record['id']}: owed without a run: {probe.owed_reason}")
        return 0
    if worktree_dirty():
        print("FAIL: the working tree is dirty; a probe attests a real commit. Commit first.")
        return 2
    import calibration_decision as decision
    import calibration_family as family

    decision_json = decision.canonical_json(family.current_config(ROOT))
    sha = head_sha()
    import calibration_status as status
    from evidence_earning import Snapshot, measurement_digest, probe_binding
    from evidence_fingerprint import coverage_row
    fp = status.tree_fingerprinter()
    fingerprints = status.identity_fingerprints(records, fp)
    probed = {planned.record['id']: planned.record for planned in probes}
    print(f"== probing {len(probes)} record(s) at {sha} ({len(tasks)} gate job(s)) ==")
    results = run_tasks(tasks, len(groups), args.jobs, decision_json)
    if head_sha() != sha or worktree_dirty():
        print("FAIL: HEAD or the working tree changed during the probe; nothing was written.")
        return 2
    import calibration_status as status

    rows, verdicts = judge(probes, tasks, results, sha, fingerprints)
    # A passing probe is bound, like a measurement, to the Nix-built binary it ran and
    # the items its runs executed (`publish_executed`).
    from calibration_runner import record_runs, run_evidence
    from evidence_earning import publish_executed
    from scientific_records import _remember_nix_output
    runs = record_runs(PROBE_LOG_DIR)
    allowed: dict[str, str] = {}
    for row in rows:
        if row['result'] == 'pass':
            rec = probed[row['record_id']]
            row['measurement_digest'] = measurement_digest(rec)
            built, coverage = run_evidence(ROOT, PROBE_LOG_DIR, runs.get(row['record_id'], set()))
            snapshot = Snapshot(ROOT, coverage_row(rec['test'], rec.get('dgp', '')), probe_binding(row), fp,
                                nix_build=built, allowed_outputs=allowed)
            row['dependency_manifest'] = publish_executed(snapshot, built, coverage)
            for output in snapshot.outputs:
                _remember_nix_output(ROOT, allowed, output)
    existing = (
        load_toml(ATTESTATIONS).get("attestation", [])
        if ATTESTATIONS.is_file()
        else []
    )
    ATTESTATIONS.write_text(
        render_attestations(merge_results(existing, rows)), encoding="utf-8", newline="\n"
    )
    for rid in sorted(verdicts):
        print(f"{rid}: {verdicts[rid]}")
    owed = [rid for rid, text in verdicts.items() if text.startswith(("owed", "unprobed"))]
    print(
        f"{len(verdicts) - len(owed)} record(s) reproduced their digests; {len(owed)} owed. "
        f"Wrote {ATTESTATIONS.relative_to(ROOT)}; "
        "`python3 scripts/calibration_status.py status` shows what is still owed."
    )
    return 1 if owed else 0


if __name__ == "__main__":
    sys.exit(main())
