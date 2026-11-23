"""Closing rules for manifest routes whose support unit is not a graphless row.

Two kinds of hand-written route in `parity/capabilities/route_manifest.toml` close without a
row in `parity/support_graphless.toml`:

* `status = "capability"`: a standalone or facade route whose unit is a capability record. It
  names the record (or several) in `capability_record`. The route is closed only while every
  named record exists in `parity/capability_records.toml` with `status = "supported"` and,
  when the route claims an interval, every named record carries coverage record ids for it
  or has its interval routes mapped to gate groups in `parity/calibration_routes.toml`.
  The claim is the route's `capability_claim` (`interval`, `point_only` or `none`); a route
  that records none claims an interval whenever a named record does. An allocated id or a
  mapped group with no measured record is reported as an owed measurement, never accepted as
  measured.
* `status = "supported"` on a route that is not a `supported.<query>.<graph class>` record: an
  estimator of support-matrix cells. It must name an estimator that
  `parity/support_cells.toml` lists, with its evidence, on a cell of the route's graph class
  and inference.

Every reader of route statuses takes the vocabulary from here.
"""

from __future__ import annotations

import re
from pathlib import Path

CAPABILITY = "capability"
CLAIMS = ("interval", "point_only", "none")
# Statuses of a hand-written route that executes and owns a `method.<route id>` record.
METHOD_ROUTE_STATUSES = ("graphless", CAPABILITY, "execution_only", "fitting_gap")
# Statuses that leave a route open: it executes without a support unit.
OPEN_STATUSES = ("execution_only", "fitting_gap")
COVERAGE_RECORDS = "parity/coverage_records.toml"
CALIBRATION_ROUTES = "parity/calibration_routes.toml"
CALIBRATION_GROUPS = "parity/calibration_groups.toml"
Groups = dict[str, list[str]]


def is_estimator_route(route: dict) -> bool:
    """A supported route that describes one estimator of supported cells."""
    return route.get("status") == "supported" and not str(route["id"]).startswith("supported.")


def owns_method_record(route: dict) -> bool:
    return route.get("status") in METHOD_ROUTE_STATUSES or is_estimator_route(route)


def record_ids(route: dict) -> list[str]:
    """The capability records a route names: one id or a list of ids."""
    named = route.get("capability_record")
    if isinstance(named, str):
        return [named] if named else []
    if isinstance(named, list):
        return [str(item) for item in named]
    return []


def coverage_ids(record: dict) -> list[str]:
    """Coverage record ids a capability record carries for an interval it states."""
    ids = [str(item) for item in record.get("coverage_records") or []]
    for output in record.get("inference_outputs") or []:
        ids += [str(item) for item in output.get("allocated_coverage_records") or []]
    return list(dict.fromkeys(ids))


def claims_interval(record: dict, groups: Groups | None = None) -> bool:
    return (
        record.get("inference_claim") == "interval"
        or bool(coverage_ids(record))
        or bool((groups or {}).get(str(record.get("id"))))
    )


def route_claim(route: dict, by_id: dict[str, dict], groups: Groups | None = None) -> str:
    """The route's declared claim, else the strongest claim of the records it names."""
    declared = route.get("capability_claim")
    if declared is not None:
        return str(declared)
    named = [by_id[rid] for rid in record_ids(route) if rid in by_id]
    if any(claims_interval(record, groups) for record in named):
        return "interval"
    claims = {str(record.get("inference_claim", "none")) for record in named}
    return "point_only" if "point_only" in claims else "none"


def capability_route_problems(
    route: dict, by_id: dict[str, dict], groups: Groups | None = None
) -> list[str]:
    """Why a `capability` route is not closed by the records it names; empty when it is."""
    rid = route["id"]
    named = record_ids(route)
    if not named:
        return [f"{rid}: capability route names no capability_record"]
    problems: list[str] = []
    declared = route.get("capability_claim")
    if declared is not None and declared not in CLAIMS:
        problems.append(f"{rid}: capability_claim {declared!r} is not one of {', '.join(CLAIMS)}")
    groups = groups or {}
    interval = route_claim(route, by_id, groups) == "interval"
    for name in named:
        record = by_id.get(name)
        if record is None:
            problems.append(f"{rid}: capability_record {name!r} is not a capability record")
            continue
        if record.get("status") != "supported":
            problems.append(
                f"{rid}: capability record {name} is {record.get('status')!r}, not supported"
            )
        if interval and not coverage_ids(record) and not groups.get(name):
            problems.append(
                f"{rid}: claims an interval but capability record {name} carries no coverage "
                "record id and no mapped gate group for it"
            )
    return problems


def manifest_capability_problems(
    routes: list[dict], records: list[dict], groups: Groups | None = None
) -> list[str]:
    """Every `capability` route against its records, and stray fields on other routes."""
    by_id = {str(record["id"]): record for record in records}
    problems: list[str] = []
    for route in routes:
        if route.get("status") == CAPABILITY:
            problems += capability_route_problems(route, by_id, groups)
        elif "capability_record" in route or "capability_claim" in route:
            problems.append(
                f"{route['id']}: capability_record and capability_claim belong to a route "
                f"whose status is capability, not {route.get('status')!r}"
            )
    return problems


def group_measured(label: str, measured: set[str], group_tests: Groups) -> bool:
    """Whether a gate group (`<test file>: <test filter>`) has a measured coverage record.

    A record id carries the test function that produced it; a filter is a substring of that
    name, and a label without a filter is every test of the file's groups."""
    head, _, test_filter = label.partition(":")
    test_filter = test_filter.strip()
    if test_filter:
        return any(test_filter in cov for cov in measured)
    names = group_tests.get(head.strip(), [])
    return any(f".{name}." in cov for name in names for cov in measured)


def owed_measurements(
    routes: list[dict],
    records: list[dict],
    measured: set[str],
    groups: Groups | None = None,
    group_tests: Groups | None = None,
) -> list[tuple[str, str]]:
    """(route id, what is owed) for an interval route: every allocated coverage id and every
    mapped gate group that has no measured record."""
    by_id = {str(record["id"]): record for record in records}
    groups, group_tests = groups or {}, group_tests or {}
    owed: list[tuple[str, str]] = []
    for route in routes:
        if route.get("status") != CAPABILITY or route_claim(route, by_id, groups) != "interval":
            continue
        for name in record_ids(route):
            allocated = coverage_ids(by_id.get(name, {}))
            for cov in allocated:
                if cov not in measured:
                    owed.append((route["id"], f"coverage record {cov}"))
            for label in groups.get(name, []):
                # A group whose record id the capability record allocates is listed by that id.
                test_filter = label.partition(":")[2].strip()
                named = bool(test_filter) and any(test_filter in cov for cov in allocated)
                if not named and not group_measured(label, measured, group_tests):
                    owed.append((route["id"], f"gate group {label}"))
    return list(dict.fromkeys(owed))


def mapped_groups(root: Path) -> Groups:
    """Capability record id -> the gate groups its interval routes are mapped to."""
    import tomllib

    path = root / CALIBRATION_ROUTES
    out: Groups = {}
    if path.is_file():
        for row in tomllib.loads(path.read_text(encoding="utf-8")).get("route", []):
            labels = out.setdefault(str(row.get("record", "")), [])
            labels += [str(g) for g in row.get("groups", []) if str(g) not in labels]
    return out


def tests_by_target(root: Path) -> Groups:
    """Test target -> the tests of its gate groups."""
    import tomllib

    path = root / CALIBRATION_GROUPS
    out: Groups = {}
    if path.is_file():
        for group in tomllib.loads(path.read_text(encoding="utf-8")).get("group", []):
            out.setdefault(str(group.get("target", "")), []).extend(group.get("tests", []))
    return out


def measured_coverage_ids(root: Path) -> set[str]:
    """Ids of the measured coverage records (read by line: the ledger is large)."""
    path = root / COVERAGE_RECORDS
    if not path.is_file():
        return set()
    return set(re.findall(r'^id = "(cov\.[^"]+)"', path.read_text(encoding="utf-8"), re.M))


def route_estimator(route: dict, estimators: set[str]) -> str | None:
    """The estimator wire id a route describes: its own id, or `estimator <id>` in its notes."""
    if route["id"] in estimators:
        return route["id"]
    notes = str(route.get("notes", ""))
    for estimator in sorted(estimators, key=len, reverse=True):
        if re.search(rf"\b[Ee]stimator {re.escape(estimator)}(?![\w.]*\w)", notes):
            return estimator
    return None


def estimator_route_problems(routes: list[dict], cells: list[dict]) -> list[str]:
    """Each supported estimator route names an estimator a supported cell lists with evidence."""
    listed = {estimator for cell in cells for estimator in cell.get("estimators", [])}
    problems: list[str] = []
    for route in routes:
        if not is_estimator_route(route):
            continue
        rid = route["id"]
        estimator = route_estimator(route, listed)
        if estimator is None:
            problems.append(
                f"{rid}: supported route names no estimator that parity/support_cells.toml lists"
            )
            continue
        inferences = {name for name in ("Frequentist", "Bayesian") if name in route["inference"]}
        on_cells = [
            cell
            for cell in cells
            if estimator in cell.get("estimators", [])
            and cell["graph_class"] == route["graph_class"]
            and cell["inference"] in inferences
        ]
        if not on_cells:
            problems.append(
                f"{rid}: estimator {estimator} is on no supported {route['graph_class']} cell "
                f"under {'/'.join(sorted(inferences)) or 'its'} inference"
            )
        elif not any(estimator_evidenced(cell, estimator) for cell in on_cells):
            problems.append(
                f"{rid}: no supported cell cites evidence for estimator {estimator}"
            )
    return problems


def estimator_evidenced(cell: dict, estimator: str) -> bool:
    """Whether a cell cites a check for this estimator: its own, or a per-estimator entry."""
    if cell.get("estimators", [None])[0] == estimator and cell.get("evidence_test"):
        return True
    for value in cell.values():
        if isinstance(value, list) and any(
            isinstance(item, dict) and item.get("estimator") == estimator and item.get("test")
            for item in value
        ):
            return True
    return False
