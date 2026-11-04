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
    python3 scripts/calibration_status.py intervals         # per interval route: calibrated, outside_measured_scope,
                                                            #   measured_below_nominal or not_measured
    python3 scripts/calibration_status.py backlog           # cells whose interval has no measurement yet
    python3 scripts/calibration_status.py readiness         # which of those have measurement wiring
    python3 scripts/calibration_status.py plan              # records owning coverage ids, with their status
    python3 scripts/calibration_status.py --check           # shape of parity/coverage_records.toml

Every subcommand is informational and exits 0. Only `--check` can fail, and only on a
malformed coverage record (a missing field, an inconsistent grid, a bad digest): it
reads the shape of the records, never what they measured.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
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


def tree_fingerprinter(
    root: Path = ROOT, metadata: dict | None = None
) -> Fingerprinter:
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
    statuses = classify(
        records, load_attestations(), tree_fingerprints(records, tree_fingerprinter())
    )
    counts = summary(statuses)
    owed = {rid: s.status for rid, s in sorted(statuses.items()) if s.status != CURRENT}
    if args.json:
        json.dump(
            {
                "counts": counts,
                "records": {r: s.status for r, s in sorted(statuses.items())},
            },
            sys.stdout,
            indent=2,
        )
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


# ---------------------------------------------------------------- interval routes
CALIBRATED = "calibrated"
OUTSIDE_MEASURED_SCOPE = "outside_measured_scope"
MEASURED_BELOW_NOMINAL = "measured_below_nominal"
NOT_MEASURED = "not_measured"
INTERVAL_STATES = (
    CALIBRATED,
    OUTSIDE_MEASURED_SCOPE,
    MEASURED_BELOW_NOMINAL,
    NOT_MEASURED,
)
CAPABILITY_RECORDS = ROOT / "parity" / "capability_records.toml"


def interval_state(allocated: list[str], by_id: dict[str, dict]) -> str:
    """The calibration annotation of one interval route.

    * `not_measured`          none of the coverage records the route's record allocates exists;
    * `measured_below_nominal` a measured record misses its nominal band at some grid point;
    * `outside_measured_scope` the measured records pass but do not cover every allocated
                               coordinate, so the route reaches past what was measured;
    * `calibrated`            every allocated record exists and none misses its band.
    """
    present = [by_id[rid] for rid in allocated if rid in by_id]
    if not present:
        return NOT_MEASURED
    if any(rec.get("boundary") for rec in present):
        return MEASURED_BELOW_NOMINAL
    if len(present) < len(allocated):
        return OUTSIDE_MEASURED_SCOPE
    return CALIBRATED


def interval_routes(capability: list[dict], records: list[dict]) -> list[dict]:
    """One row per supported route that reports an interval, with its annotation."""
    by_id = {str(rec.get("id")): rec for rec in records}
    rows = []
    for rec in capability:
        claim = rec.get("inference_claim")
        for route in rec.get("routes") or []:
            if (
                route.get("status") != "supported"
                or (route.get("claim") or claim) != "interval"
            ):
                continue
            allocated = [str(x) for x in rec.get("coverage_records") or []]
            rows.append(
                {
                    "record": str(rec["id"]),
                    "route": str(route.get("name", "")),
                    "allocated": len(allocated),
                    "measured": sum(1 for rid in allocated if rid in by_id),
                    "state": interval_state(allocated, by_id),
                }
            )
    return rows


def cmd_intervals(args: argparse.Namespace) -> int:
    capability = tomllib.loads(read_text(CAPABILITY_RECORDS)).get("record", [])
    rows = interval_routes(capability, load_records())
    counts = collections.Counter(row["state"] for row in rows)
    if args.json:
        json.dump({"counts": dict(counts), "routes": rows}, sys.stdout, indent=2)
        print()
        return 0
    print(
        "interval routes: "
        + ", ".join(f"{counts[name]} {name}" for name in INTERVAL_STATES)
        + f" of {len(rows)}"
    )
    for row in rows:
        print(
            f"  {row['state']}: {row['record']} / {row['route']} ({row['measured']}/{row['allocated']} records)"
        )
    return 0


# ---------------------------------------------------------------- inventories
SUPPORT_CELLS = ROOT / "parity" / "support_cells.toml"
UNMEASURED = "estimator_grid_not_measured"
AXES = ("query", "graph_class", "structure", "inference")


def unmeasured_coordinates() -> dict[tuple[str, ...], list[dict]]:
    cells = tomllib.loads(read_text(SUPPORT_CELLS)).get("cell", [])
    grouped: dict[tuple[str, ...], list[dict]] = collections.defaultdict(list)
    for cell in cells:
        if cell.get("calibration_reason") == UNMEASURED:
            grouped[tuple(cell[axis] for axis in AXES)].append(cell)
    return dict(sorted(grouped.items()))


def cmd_backlog(args: argparse.Namespace) -> int:
    grouped = unmeasured_coordinates()
    print(
        f"calibration backlog: {sum(map(len, grouped.values()))} cells without a measurement "
        f"across {len(grouped)} coordinates"
    )
    for key, cells in grouped.items():
        print(f"  {' / '.join(key)}: {len(cells)}")
    return 0


def _slug(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _phrase(label: str, phrase: str, *, static_graph: bool = False) -> bool:
    """Underscore-delimited components are tokens: dag does not match cpdag or the dag of temporal_dag."""
    normalized = re.sub(r"[^a-z0-9_]+", "_", label.lower())
    token = _slug(phrase)
    for match in re.finditer(
        r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", normalized
    ):
        if static_graph and normalized[: match.start()].endswith("temporal_"):
            continue
        return True
    return False


def calibration_groups() -> list[tuple[str, str]]:
    """Group names from the measurement driver's dry run; no test is executed."""
    import os
    import subprocess

    env = dict(os.environ, ANTECEDENT_CALIBRATION_DRY_RUN="1")
    env.pop("ANTECEDENT_CALIBRATION_SHARD", None)
    env.pop("ANTECEDENT_JOINT_RESPONSE_STABLE_CANDIDATE", None)
    result = subprocess.run(
        ["bash", "scripts/gate_calibration.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    groups = []
    for line in result.stdout.splitlines():
        prefix, separator, label = line.partition(": ")
        if separator and prefix.startswith("group "):
            head, separator, test_filter = label.partition(": ")
            groups.append((head, test_filter if separator else ""))
    return groups


def candidate_groups(key: tuple[str, ...], groups: list[tuple[str, str]]) -> list[str]:
    """Groups whose name matches the coordinate's query, graph and inference: candidates, never proof of fit."""
    query, graph, _structure, inference = key
    static = graph in {"Dag", "Cpdag", "Pag", "Admg"}
    out = []
    for head, test_filter in groups:
        label = f"{head}: {test_filter}" if test_filter else head
        if (
            _phrase(label, query)
            and _phrase(label, graph, static_graph=static)
            and _phrase(label, inference)
        ):
            out.append(label)
    return out


def cmd_readiness(args: argparse.Namespace) -> int:
    grouped = unmeasured_coordinates()
    groups = calibration_groups()
    print(
        f"calibration readiness: {len(grouped)} unmeasured coordinates; no calibration test executed"
    )
    for key in grouped:
        candidates = candidate_groups(key, groups)
        print(f"  {' / '.join(key)}: {len(candidates)} candidate group(s)")
        for label in candidates:
            print(f"    {label}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    capability = tomllib.loads(read_text(CAPABILITY_RECORDS)).get("record", [])
    records = load_records()
    statuses = classify(
        records, load_attestations(), tree_fingerprints(records, tree_fingerprinter())
    )
    by_id = {str(rec.get("id")) for rec in records}
    owners = [rec for rec in capability if rec.get("coverage_records")]
    print(f"capability records owning coverage ids: {len(owners)}")
    for rec in owners:
        ids = [str(x) for x in rec["coverage_records"]]
        measured = [rid for rid in ids if rid in by_id]
        current = [rid for rid in measured if statuses[rid].status == CURRENT]
        print(
            f"  {rec['id']} ({rec.get('inference_claim')}): {len(ids)} allocated, "
            f"{len(measured)} measured, {len(current)} current"
        )
    return 0


# ---------------------------------------------------------------- shape check
def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _sanitize(label: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in label).strip("_")


def _resolves(root: Path, spec: str, collector) -> bool:
    """Does `<file>::<item>` name a function (or a macro-table entry, or a dgp label) that file defines?"""
    if "::" not in spec:
        return False
    rel, fn = spec.rsplit("::", 1)
    path = root / rel
    if not path.is_file():
        return False
    text = path.read_text(errors="ignore", encoding="utf-8")
    if re.search(rf"fn\s+{re.escape(fn)}\s*[(<]", text):
        return True
    if re.search(r'dgp:\s*"' + re.escape(fn) + r'"', text):
        return True
    if re.search(r'"' + re.escape(fn) + r'"', text):
        return True
    if fn in collector.conditional_record_dgp_labels(path):
        return True
    if re.search(rf"\buse\b[^;]*\b{re.escape(fn)}\b[^;]*;", text):
        import test_evidence

        target = test_evidence.target_root(path)
        if target and any(
            item.kind == "fn" and item.name == fn
            for module in test_evidence.target_files(target[0].resolve())
            for item in test_evidence.rust_items(module)[1]
        ):
            return True
    return "macro_rules!" in text and bool(re.search(rf"\b{re.escape(fn)}\b", text))


def _unit_interval(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0 <= float(value) <= 1
    )


def record_shape_problems(rec: dict, collector, root: Path) -> list[str]:
    """Why one coverage record is malformed; empty when its shape is sound.

    Shape only: which fields exist, that the grid is internally consistent and that the
    summary is the governing grid point's. Whether a measured rate is acceptable is a
    calibration reading, never a shape problem."""
    problems: list[str] = []
    rid = str(rec.get("id", ""))
    label = f"coverage_records.toml {rid}"
    if problem := collector.validation_record_problem(rec):
        problems.append(f"{label}: {problem}")
    missing = [key for key in collector.FIELDS if key not in rec]
    if missing:
        return [*problems, f"{label}: missing {', '.join(missing)}"]
    test = str(rec["test"])
    test_fn = test.rsplit("::", 1)[-1]
    if "fingerprint" in rec and not re.fullmatch(
        r"[0-9a-f]{64}", str(rec["fingerprint"])
    ):
        problems.append(f"{label}: fingerprint must be a sha256 hex digest")
    expected = (
        f"cov.{_snake(rec['query'])}.{_snake(rec['graph_class'])}."
        f"{str(rec['inference']).lower()}.{rec['interval_method']}."
        f"l{round(float(rec['nominal']) * 100)}.{test_fn}"
    )
    if rid != expected and not rid.startswith(expected + "."):
        problems.append(f"{label}: id does not match its fields ({expected}[.<label>])")
    if (
        rid.startswith(expected + ".")
        and _sanitize(rid[len(expected) + 1 :]) != rid[len(expected) + 1 :]
    ):
        problems.append(f"{label}: record label is not sanitized")
    sha = str(rec["calibration_sha"])
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        problems.append(f"{label}: calibration_sha must be 40 lowercase hex")
    elif set(sha) == {"0"}:
        problems.append(
            f"{label}: calibration_sha is the zero SHA; nothing was measured"
        )
    for field in ("observed", "mcse", "nominal", "unidentified_mass_max"):
        if not _unit_interval(rec[field]):
            problems.append(f"{label}: {field} not in [0,1]")
    if int(rec["replicates"]) < 1:
        problems.append(f"{label}: replicates must be positive")
    if int(rec["n_min"]) < 1 or int(rec["n_max"]) < int(rec["n_min"]):
        problems.append(f"{label}: measured row-count range is empty")
    if not _resolves(root, test, collector):
        problems.append(f"{label}: test {test} does not resolve")
    if not _resolves(root, str(rec["dgp"]), collector):
        problems.append(f"{label}: dgp {rec['dgp']} does not resolve")
    role, boundary = str(rec["role"]), bool(rec["boundary"])
    grid = rec["grid"]
    if (
        not isinstance(grid, list)
        or len(grid) != collector.GRID_POINTS
        or not all(isinstance(p, dict) for p in grid)
    ):
        return [
            *problems,
            f"{label}: grid must hold one measurement per sample-size grid point "
            f"({collector.GRID_POINTS}); re-measure with scripts/measure_calibration.sh",
        ]
    if any(not collector.grid_fields_valid(p) for p in grid):
        return [
            *problems,
            f"{label}: grid points must carry exactly {', '.join(collector.GRID_ENTRY_FIELDS)}",
        ]
    if [int(p["point"]) for p in grid] != list(range(collector.GRID_POINTS)):
        problems.append(
            f"{label}: grid points must be 0..{collector.GRID_POINTS - 1} in order"
        )
    if any(
        int(lo["n_max"]) >= int(hi["n_min"])
        for lo, hi in zip(grid, grid[1:], strict=False)
    ):
        problems.append(
            f"{label}: grid sample sizes must strictly increase point to point"
        )
    if (int(rec["n_min"]), int(rec["n_max"])) != (
        min(int(p["n_min"]) for p in grid),
        max(int(p["n_max"]) for p in grid),
    ):
        problems.append(f"{label}: n_min..n_max must span the grid points")
    failing = [p for p in grid if p["boundary"]]
    if boundary != bool(failing):
        problems.append(f"{label}: boundary must be true exactly when a grid point is")
    governing = min(
        failing or grid, key=lambda p: (float(p["observed"]), int(p["point"]))
    )
    if (rec["observed"], rec["mcse"], rec["replicates"]) != (
        governing["observed"],
        governing["mcse"],
        governing["replicates"],
    ):
        problems.append(
            f"{label}: observed / mcse / replicates must be the governing grid point's "
            f"(point {governing['point']})"
        )
    for point in grid:
        where = f"{label} grid point {point['point']}"
        for field in ("observed", "mcse"):
            if not _unit_interval(point[field]):
                problems.append(f"{where}: {field} not in [0,1]")
        if int(point["replicates"]) < 1:
            problems.append(f"{where}: replicates must be positive")
        if int(point["n_min"]) < 1 or int(point["n_max"]) < int(point["n_min"]):
            problems.append(f"{where}: measured row-count range is empty")
        if str(point["role"]) not in collector.ROLES:
            problems.append(f"{where}: unknown role {point['role']}")
    if "probe" in rec:
        table = rec["probe"]
        if not (
            isinstance(table, dict)
            and sorted(table) == ["p0", "p1", "p2"]
            and all(
                isinstance(digests, dict)
                and digests
                and all(
                    str(k).isdigit()
                    and int(k) in collector.DIGEST_PREFIXES
                    and isinstance(v, str)
                    and re.fullmatch(r"[0-9a-f]{32}", v)
                    for k, v in digests.items()
                )
                for digests in table.values()
            )
        ):
            problems.append(
                f"{label}: probe must hold prefix digests ({', '.join(map(str, collector.DIGEST_PREFIXES))}) "
                "for grid points p0, p1 and p2"
            )
    confirmation = rec.get("confirmation")
    stated_range = rec.get("confirmed_passing_range")
    if confirmation is None and stated_range is not None:
        problems.append(
            f"{label}: confirmed_passing_range without the confirmation behind it"
        )
    elif confirmation is not None:
        if not boundary or role == "named_boundary":
            problems.append(
                f"{label}: a confirmation belongs to a boundary record that is not a named boundary"
            )
        if (
            not isinstance(confirmation, list)
            or not confirmation
            or not all(isinstance(p, dict) for p in confirmation)
            or any(not collector.grid_fields_valid(p) for p in confirmation)
        ):
            problems.append(
                f"{label}: confirmation must list points carrying exactly "
                f"{', '.join(collector.GRID_ENTRY_FIELDS)}"
            )
        else:
            indices = [int(p["point"]) for p in confirmation]
            if indices != sorted(set(indices)) or not set(indices) <= set(
                range(len(grid))
            ):
                problems.append(
                    f"{label}: confirmation points must be distinct grid points in order"
                )
            elif stated_range != collector.confirmed_passing_range(grid, confirmation):
                problems.append(
                    f"{label}: confirmed_passing_range must be the contiguous passing run the "
                    "grid and confirmation establish (collect_coverage_records.confirmed_passing_range)"
                )
    if problem := collector.conservative_point_problem(rec):
        problems.append(f"{label}: {problem}")
    one_sided = rec.get("one_sided")
    if role == "one_sided":
        if (
            not isinstance(one_sided, dict)
            or sorted(one_sided) != ["side", "target"]
            or one_sided.get("side") not in ("lower", "upper")
            or not isinstance(one_sided.get("target"), str)
            or not one_sided["target"].strip()
        ):
            problems.append(
                f"{label}: a one_sided record must state one_sided = {{ side = "
                '"lower"|"upper", target = "<non-empty>" }'
            )
    elif one_sided is not None:
        problems.append(f"{label}: one_sided on a {role} record")
    roles = {str(p["role"]) for p in grid}
    if len(roles) > 1 and roles != {"gated", "named_boundary"}:
        problems.append(
            f"{label}: grid points carry incompatible roles {sorted(roles)}"
        )
    else:
        expected_role = "named_boundary" if len(roles) > 1 else next(iter(roles))
        if role != expected_role:
            problems.append(
                f"{label}: role {role} is not the grid points' {expected_role}"
            )
    return problems


def shape_problems(root: Path = ROOT) -> list[str]:
    """Malformed records of `parity/coverage_records.toml`; the only thing this script can fail on."""
    import collect_coverage_records as collector

    path = root / "parity" / "coverage_records.toml"
    if not path.is_file():
        return ["parity/coverage_records.toml missing"]
    records = tomllib.loads(read_text(path)).get("record", [])
    problems: list[str] = []
    for rec in records:
        try:
            problems.extend(record_shape_problems(rec, collector, root))
        except (KeyError, TypeError, ValueError) as error:
            problems.append(
                f"coverage_records.toml {rec.get('id')}: malformed ({error!r})"
            )
    ids = [rec.get("id") for rec in records]
    if len(set(ids)) != len(ids):
        problems.append("parity/coverage_records.toml: duplicate record ids")
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    problems = shape_problems()
    if problems:
        print("coverage record shape violations:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("coverage record shape: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the shape of the coverage records (the one thing that can fail)",
    )
    sub = parser.add_subparsers(dest="command")
    status = sub.add_parser("status")
    status.add_argument(
        "--owed",
        action="store_true",
        help="print only the ids of the records that are not current",
    )
    status.add_argument(
        "--json",
        action="store_true",
        help="machine-readable counts and per-record status",
    )
    intervals = sub.add_parser("intervals")
    intervals.add_argument(
        "--json",
        action="store_true",
        help="machine-readable counts and per-route state",
    )
    sub.add_parser("backlog")
    sub.add_parser("readiness")
    sub.add_parser("plan")
    args = parser.parse_args(argv)
    if args.check:
        return cmd_check(args)
    handlers = {
        "status": cmd_status,
        "intervals": cmd_intervals,
        "backlog": cmd_backlog,
        "readiness": cmd_readiness,
        "plan": cmd_plan,
    }
    if args.command not in handlers:
        parser.error("give a subcommand or --check")
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
