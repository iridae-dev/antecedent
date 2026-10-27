#!/usr/bin/env python3
"""Generate the verified counterpart inventory (parity/counterparts.toml, docs/counterparts.md).

Every advertised method has four quadrants: {static, temporal} x {frequentist, bayesian}.
The inventory records one row per quadrant, joined from

  parity/method_children/children.toml      method-level children (family x design x inference)
  parity/capabilities/route_manifest.toml   public routes (licensed and closed routes by target)
  parity/licensed_routes.toml               compiler lowering of each licensed cell
  parity/release_scope/scope.toml           release scope entry of each child

Row fields: route, status (licensed | executing_unlicensed | closed_with_next_attempt |
not_applicable_with_theory_reason), uncertainty kind (analytic | design_aware_resampling |
posterior | enclosure | none), the sampling design honoured, the reuse key, the decision port,
the artifact, and the closure kind (delivered | narrowed_promise). A closed_with_next_attempt
row, and any executing row whose claim is narrowed, is a narrowed_promise and records the
reduced claim, the failed attempt and the next attempt. A not_applicable_with_theory_reason
row records the boundary itself and is delivered.

    python3 scripts/generate_counterparts.py            # rewrite both outputs
    python3 scripts/generate_counterparts.py --check    # fail when an output is stale
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from method_evidence import UNLISTED, Evidence

CHILDREN = ROOT / "parity/method_children/children.toml"
ROUTES = ROOT / "parity/capabilities/route_manifest.toml"
LICENSED = ROOT / "parity/licensed_routes.toml"
SCOPE = ROOT / "parity/release_scope/scope.toml"
OUT_TOML = ROOT / "parity/counterparts.toml"
OUT_DOC = ROOT / "docs/counterparts.md"

DESIGNS = ("static", "temporal")
INFERENCES = ("frequentist", "bayesian")
QUADRANTS = tuple(f"{d}.{i}" for d in DESIGNS for i in INFERENCES)
STATUSES = (
    "licensed",
    "executing_unlicensed",
    "closed_with_next_attempt",
    "not_applicable_with_theory_reason",
)
UNCERTAINTY_KINDS = ("analytic", "design_aware_resampling", "posterior", "enclosure", "none")
BASES = (
    "measured_record",
    "point_only_cell",
    "asserted_in_test",
    "estimator_records",
    "declared_unmeasured",
    "declared_and_corroborated",
    "graphless_claim",
    "no_executing_route",
    "refused_route",
    "no_method_record",
)
CLOSURE_KINDS = ("delivered", "narrowed_promise")

# What each kind of recorded gap asks for next. The child record names the gap; the reason text
# is the child's own sentence.
NEXT_ATTEMPT = {
    "validation": "measure the whole-procedure coverage of this interval and record its boundary",
    "statistical": "derive the dependence-aware likelihood or resampling law for this counterpart",
    "implementation": "build the missing route through the compiler with the same uncertainty ports",
    "temporal-identification": (
        "prove identification under the declared temporal assumptions or record the "
        "counterexample as the boundary"
    ),
    "theory": "derive the identification result for this counterpart or record the boundary proof",
}
EXECUTING = ("licensed", "executing_unlicensed")
NO_INTERVAL_NEXT = (
    "derive the dependence-aware interval or posterior for this counterpart and publish it through "
    "the same route with the uncertainty port of its static twin"
)
CLOSED_ROUTE_NEXT = (
    "identify the target on this graph class and lower it through the compiler, or keep the "
    "typed refusal with the boundary proof"
)
SPECS = ROOT / "parity/method_specs/specs.toml"
TEMPORAL_MARK = re.compile(r"temporal|panel|sequential|series|period|time", re.I)
UNCERTAINTY_RANK = {
    "none": 0,
    "enclosure": 1,
    "analytic": 2,
    "design_aware_resampling": 2,
    "posterior": 2,
}


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def split_id(child_id: str) -> tuple[str, str, str]:
    """Method, design and inference of a child id such as P01.linear.temporal.bayesian."""
    parts = child_id.split(".")
    return ".".join(parts[:-2]), parts[-2], parts[-1]


class Evidenced:
    """The method records of the route manifest and the evidence each row is derived from."""

    def __init__(self, manifest: dict, specs: list[dict], ev: Evidence) -> None:
        self.ev = ev
        self.methods = manifest.get("method", [])
        self.by_route: dict[str, list[dict]] = {}
        self.by_estimator: dict[str, list[dict]] = {}
        self.by_child: dict[str, list[dict]] = {}
        for method in self.methods:
            if method["child"]:
                self.by_child.setdefault(method["child"], []).append(method)
                continue
            self.by_route.setdefault(method["route"], []).append(method)
            if method["estimator"] not in UNLISTED:
                self.by_estimator.setdefault(method["estimator"], []).append(method)
        self.specs = {spec["id"]: spec for spec in specs}

    def for_side(self, methods: list[dict], inference: str) -> list[dict]:
        return [m for m in methods if m["inference"] in (inference, "both", "unlisted")]

    def methods_of_child(self, child: dict, inference: str) -> list[dict]:
        """Method records that implement a child: its route's, narrowed by the estimators it names."""
        if child["id"] in self.by_child:
            return self.by_child[child["id"]]
        tokens = [
            t for t in re.split(r"[^A-Za-z0-9_.]+", child["inference"]) if t in self.by_estimator
        ]
        candidates = self.for_side(self.by_route.get(child["route"], []), inference)
        if tokens:
            narrowed = [m for m in candidates if m["estimator"] in tokens]
            if narrowed:
                return narrowed
            if not candidates:
                return self.for_side([m for t in tokens for m in self.by_estimator[t]], inference)
        return candidates

    def route_kind(self, route_id: str, inference: str) -> tuple[str, str]:
        """The best uncertainty kind of a route's methods on one inference side, with its basis."""
        methods = self.for_side(self.by_route.get(route_id, []), inference)
        if not methods:
            return "none", "no_method_record"
        best = max(methods, key=lambda m: UNCERTAINTY_RANK[m["uncertainty"]])
        return best["uncertainty"], best["uncertainty_basis"].split("|")[0].split(":")[0]


def child_status(child: dict, methods: list[dict], verified: bool) -> tuple[str, str]:
    """Status and closure kind of a child, from the methods and verified evidence that back it.

    A child is licensed when a licensed method record implements it, and executing when a method
    record or its own verified module and fixture do. A boundary is not applicable when its
    gap is a target that does not exist on the axis; every other open gap is a closed row that
    names its next attempt.
    """
    gap, missing = child["gap"], child["missing"]
    if gap == "target" or missing.lower().startswith("not applicable"):
        return "not_applicable_with_theory_reason", "delivered"
    executes = gap in ("none", "validation") or (
        gap == "implementation" and child["module"] != "none"
    )
    if executes and (methods or (verified and child["module"] != "none")):
        licensed = any(m["status"] == "licensed" for m in methods)
        status = "licensed" if licensed else "executing_unlicensed"
        return status, "delivered" if gap == "none" else "narrowed_promise"
    return "closed_with_next_attempt", "narrowed_promise"


def scope_by_id(scope: dict) -> dict[str, dict]:
    return {entry["id"]: entry for entry in scope["entry"]}


def child_rows(
    children: list[dict], routes: dict[str, dict], scope: dict[str, dict], evidenced: Evidenced
) -> list[dict]:
    ev = evidenced.ev
    from check_method_children import child_findings

    rows = []
    for child in children:
        method, design, inference = split_id(child["id"])
        route = routes.get(child["route"])
        methods = evidenced.methods_of_child(child, inference)
        verified = not child_findings(ev, child, set(routes))
        status, closure = child_status(child, methods, verified)
        reason = "" if child["missing"] == "none" else child["missing"]
        if status in ("closed_with_next_attempt", "not_applicable_with_theory_reason"):
            kind, basis = "none", "no_executing_route"
        elif methods:
            best = max(methods, key=lambda m: UNCERTAINTY_RANK[m["uncertainty"]])
            kind, basis = best["uncertainty"], best["uncertainty_basis"].split("|")[0].split(":")[0]
        else:
            kind, basis = "unresolved", "no_method_record"
        spec = evidenced.specs.get(child["id"], {})
        sampling = next(
            (m["design_time_model"] for m in methods if m["design_time_model"] not in UNLISTED), ""
        )
        sampling = sampling or spec.get("sampling_units", "unlisted")
        fixtures = ev.tokens(child["fixtures"])
        consumers = ev.tokens(child["consumer"])
        entry = scope.get(child["id"], {})
        rows.append(
            {
                "id": child["id"],
                "kind": "method",
                "method": method,
                "design": design,
                "inference": inference,
                "route": child["route"],
                "status": status,
                "uncertainty": kind,
                "uncertainty_basis": basis,
                "target": child["target"],
                "sampling_design": sampling,
                "execution": fixtures[0] if fixtures else "unlisted",
                "reuse_key": child["inference"]
                if child["inference"] != "unlisted"
                else child["route"],
                "decision_port": methods[0]["decision_ports"]
                if methods
                else (route["decision_ports"] if route else "unlisted"),
                "artifact": next(
                    (c for c in consumers if not ev.is_test_path(ev.split(c)[0])),
                    consumers[0] if consumers else "unlisted",
                ),
                "closure": closure,
                "scope_status": entry.get("status", "unlisted"),
                "gap": child["gap"],
                "reason": reason,
                "next_attempt": (
                    NO_INTERVAL_NEXT
                    if closure == "narrowed_promise" and kind == "none" and status in EXECUTING
                    else NEXT_ATTEMPT.get(child["gap"], "")
                    if closure == "narrowed_promise"
                    else ""
                ),
            }
        )
    return rows


def id_design(route: dict) -> str | None:
    """The design a route id names as a dotted segment, leading or inner, or None."""
    for design in DESIGNS:
        if re.search(rf"(?:^|\.){design}\.", route["id"]):
            return design
    return None


def route_method(route: dict) -> str:
    """The method a route's rows belong to.

    A measurement route is a quadrant of the measurement-corrected functional whichever design
    its manifest target names, so the temporal measurement routes (whose target is the
    sequentially randomized design they correct) join the static ones.
    """
    return (
        "MeasurementCorrectedFunctional"
        if route["id"].startswith("measurement.")
        else route["target"]
    )


def route_design(route: dict) -> str:
    if "counterpart_design" in route:
        return route["counterpart_design"]
    named = id_design(route)
    if named is not None:
        return named
    first = route["design_time_model"].split(";")[0].strip().lower()
    if first in DESIGNS:
        return first
    return "temporal" if TEMPORAL_MARK.search(route["graph_class"]) else "static"


def route_inferences(route: dict) -> list[str]:
    text = route["inference"].replace("refused:", "")
    found = [name for name in INFERENCES if name.capitalize() in text]
    return found


def route_rows(routes: list[dict], lowerings: dict[str, str], evidenced: Evidenced) -> list[dict]:
    """One row per quadrant of each licensed or closed route, grouped by its target.

    A route outside the dotted ``static``/``temporal`` id scheme is included when it declares its
    ``counterpart_design``. ``narrowed_reason`` and ``narrowed_next_attempt`` narrow the rows of
    the route's own target the way ``counterpart_reason`` narrows a counterpart's.

    A route may also be the counterpart of another target (``counterpart_targets``): the
    temporal side of mediation is the temporal mediation route, and a measurement correction on
    a sequentially randomized panel is the temporal side of the measurement-corrected
    functional. Each such target gets a row of its own, narrowed when the route declares a
    ``counterpart_reason`` and a ``counterpart_next_attempt`` (the reduced claim, and what to
    attempt next).
    """
    rows = []
    for route in routes:
        if route["status"] not in ("licensed", "closed", "execution_only"):
            continue
        design = route_design(route)
        closed = route["status"] == "closed"
        inferences = route_inferences(route)
        counterparts = route.get("counterpart_targets", [])
        if (
            route["status"] == "execution_only"
            and id_design(route) is None
            and "counterpart_design" not in route
        ):
            continue
        targets = list(dict.fromkeys([route_method(route), *counterparts]))
        for inference in inferences:
            kind, basis = (
                evidenced.route_kind(route["id"], inference)
                if not closed
                else ("none", "refused_route")
            )
            methods = evidenced.for_side(evidenced.by_route.get(route["id"], []), inference)
            status = (
                "closed_with_next_attempt"
                if closed
                else "licensed"
                if (
                    any(m["status"] == "licensed" for m in methods)
                    if methods
                    else route["status"] == "licensed"
                )
                else "executing_unlicensed"
            )
            reason = route["notes"] if closed else ""
            weak = kind == "none" and not closed
            for target in targets:
                derived = target != route_method(route)
                reason_key = "counterpart_reason" if derived else "narrowed_reason"
                attempt_key = "counterpart_next_attempt" if derived else "narrowed_next_attempt"
                narrowed = reason_key in route
                suffix = f">{target}" if derived else ""
                rows.append(
                    {
                        "id": f"{route['id']}#{design}.{inference}{suffix}",
                        "kind": "route",
                        "method": target,
                        "design": design,
                        "inference": inference,
                        "route": route["id"],
                        "status": status,
                        "uncertainty": kind,
                        "uncertainty_basis": basis,
                        "target": target,
                        "sampling_design": route["design_time_model"],
                        "execution": methods[0]["evidence"][0]
                        if methods and methods[0]["evidence"]
                        else "unlisted",
                        "reuse_key": lowerings.get(route["id"], route["id"]),
                        "decision_port": route["decision_ports"],
                        "artifact": next(
                            (
                                c
                                for m in methods
                                for c in m["artifact_consumer"]
                                if not evidenced.ev.is_test_path(evidenced.ev.split(c)[0])
                            ),
                            "unlisted",
                        ),
                        "closure": "narrowed_promise"
                        if closed or weak or narrowed
                        else "delivered",
                        "scope_status": "mandatory",
                        "gap": "target"
                        if closed
                        else ("validation" if weak or narrowed else "none"),
                        "reason": (
                            route[reason_key]
                            if narrowed
                            else reason
                            or (
                                f"the route publishes a point value and no interval ({basis})"
                                if weak
                                else ""
                            )
                        ),
                        "next_attempt": (
                            route[attempt_key]
                            if narrowed
                            else CLOSED_ROUTE_NEXT
                            if closed
                            else (NEXT_ATTEMPT["validation"] if weak else "")
                        ),
                    }
                )
    return rows + static_boundaries(rows)


def static_boundaries(rows: list[dict]) -> list[dict]:
    """A target recorded only over time has no static graph class: record that boundary."""
    designs: dict[str, set[str]] = {}
    for row in rows:
        designs.setdefault(row["method"], set()).add(row["design"])
    extra = []
    for target, seen in sorted(designs.items()):
        if seen != {"temporal"}:
            continue
        for inference in INFERENCES:
            extra.append(
                {
                    "id": f"{target}#static.{inference}",
                    "kind": "route",
                    "method": target,
                    "design": "static",
                    "inference": inference,
                    "route": "unlisted",
                    "status": "not_applicable_with_theory_reason",
                    "uncertainty": "none",
                    "uncertainty_basis": "no_executing_route",
                    "target": target,
                    "sampling_design": "unlisted",
                    "execution": "unlisted",
                    "reuse_key": "unlisted",
                    "decision_port": "unlisted",
                    "artifact": "unlisted",
                    "closure": "delivered",
                    "scope_status": "mandatory",
                    "gap": "target",
                    "reason": f"{target} is defined over a time-indexed process, so it has no static graph class",
                    "next_attempt": "",
                }
            )
    return extra


def licensed_lowerings(licensed: dict) -> dict[str, str]:
    """Route id (licensed.<Target>.<Graph>) to its compiled lowering, where unambiguous."""
    out: dict[str, str] = {}
    for route in licensed["route"]:
        target, graph = route["coordinate"].split(":")[:2]
        out.setdefault(f"licensed.{target}.{graph}", route["lowering"])
    return out


def build() -> list[dict]:
    children = load(CHILDREN)["child"]
    manifest_data = load(ROUTES)
    manifest = manifest_data["route"]
    routes = {route["id"]: route for route in manifest}
    scope = scope_by_id(load(SCOPE))
    evidenced = Evidenced(manifest_data, load(SPECS)["spec"], Evidence(ROOT))
    rows = child_rows(children, routes, scope, evidenced)
    rows += route_rows(manifest, licensed_lowerings(load(LICENSED)), evidenced)
    rows.sort(
        key=lambda row: (
            row["kind"],
            row["method"],
            QUADRANTS.index(f"{row['design']}.{row['inference']}"),
            row["id"],
        )
    )
    return rows


FIELDS = (
    "id",
    "kind",
    "method",
    "design",
    "inference",
    "route",
    "status",
    "uncertainty",
    "uncertainty_basis",
    "target",
    "sampling_design",
    "execution",
    "reuse_key",
    "decision_port",
    "artifact",
    "closure",
    "scope_status",
    "gap",
    "reason",
    "next_attempt",
)


def render_toml(rows: list[dict]) -> str:
    out = [
        "# Verified counterpart inventory. Generated by scripts/generate_counterparts.py; do not edit.",
        "# One row per advertised method x {static, temporal} x {frequentist, bayesian} quadrant,",
        "# joined from the method children, the route manifest, the compiled lowerings and the",
        "# release scope. scripts/check_counterparts.py fails on a quadrant no row records.",
        "",
        "schema = 1",
        f"statuses = {json.dumps(list(STATUSES))}",
        f"uncertainty_kinds = {json.dumps(list(UNCERTAINTY_KINDS))}",
        f"closure_kinds = {json.dumps(list(CLOSURE_KINDS))}",
        "",
    ]
    for row in rows:
        out.append("[[row]]")
        out.extend(f"{field} = {toml_str(row[field])}" for field in FIELDS)
        out.append("")
    return "\n".join(out)


def methods(rows: list[dict]) -> dict[tuple[str, str], dict[str, list[dict]]]:
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in rows:
        grouped.setdefault((row["kind"], row["method"]), {}).setdefault(
            f"{row['design']}.{row['inference']}", []
        ).append(row)
    return grouped


STATUS_ORDER = {
    "licensed": 0,
    "executing_unlicensed": 1,
    "not_applicable_with_theory_reason": 2,
    "closed_with_next_attempt": 3,
}


def best(rows: list[dict]) -> dict:
    return min(rows, key=lambda row: STATUS_ORDER.get(row["status"], len(STATUS_ORDER)))


def render_doc(rows: list[dict]) -> str:
    grouped = methods(rows)
    short = {
        "licensed": "licensed",
        "executing_unlicensed": "executing",
        "closed_with_next_attempt": "closed",
        "not_applicable_with_theory_reason": "n/a (theory)",
    }
    out = [
        "# Counterpart inventory",
        "",
        "Generated by `scripts/generate_counterparts.py` from `parity/method_children/children.toml`,",
        "the route manifest, the compiled lowerings and the release scope. The full rows are in",
        "`parity/counterparts.toml`; `scripts/check_counterparts.py` fails on a quadrant no row",
        "records, on a temporal route with weaker uncertainty than its static twin and no recorded",
        "reason, and on a reason that reads as not implemented.",
        "",
        "A route is complete only when it executes through the compiler, meets its scientific claim,",
        "carries the uncertainty the claim needs and satisfies its consumers. `delivered` means the",
        "implementation meets the claim. `narrowed_promise` means the public claim was reduced; the",
        "reduced claim, the failed attempt and the next attempt are recorded beside it. A",
        "`closed` quadrant is always a narrowed promise.",
        "",
    ]
    for kind, title in (("method", "Method children"), ("route", "Routes by target")):
        out += [
            f"## {title}",
            "",
            "| Method | static / frequentist | static / bayesian | temporal / frequentist | temporal / bayesian |",
            "| --- | --- | --- | --- | --- |",
        ]
        for (k, method), quads in sorted(grouped.items()):
            if k != kind:
                continue
            cells = []
            for quadrant in QUADRANTS:
                cells.append(
                    short[best(quads[quadrant])["status"]] if quadrant in quads else "**missing**"
                )
            out.append(f"| `{method}` | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Narrowed promises", "", "| Row | Status | Next attempt |", "| --- | --- | --- |"]
    for row in rows:
        if row["closure"] == "narrowed_promise" and row["kind"] == "method":
            out.append(f"| `{row['id']}` | {short[row['status']]} | {row['next_attempt']} |")
    out.append("")
    return "\n".join(out)


def main_check() -> int:
    """Nonzero when either generated output is stale."""
    sys.argv.append("--check") if "--check" not in sys.argv else None
    return main()


def main() -> int:
    rows = build()
    toml_text, doc_text = render_toml(rows), render_doc(rows)
    if "--check" in sys.argv[1:]:
        stale = [
            p
            for p, t in ((OUT_TOML, toml_text), (OUT_DOC, doc_text))
            if not p.exists() or p.read_text(encoding="utf-8") != t
        ]
        for path in stale:
            print(
                f"{path.relative_to(ROOT)} is stale; run python3 scripts/generate_counterparts.py"
            )
        return 1 if stale else 0
    OUT_TOML.write_text(toml_text, encoding="utf-8")
    OUT_DOC.write_text(doc_text, encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT_TOML.relative_to(ROOT)} and {OUT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
