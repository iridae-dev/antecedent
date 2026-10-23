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
CLOSED_ROUTE_NEXT = (
    "identify the target on this graph class and lower it through the compiler, or keep the "
    "typed refusal with the boundary proof"
)
NO_INTERVAL = re.compile(r"no[_ ]interval|reports no interval|no .{0,40}interval|point value only", re.I)
RESAMPLING = re.compile(
    r"bootstrap|resampl|permutation|conformal|placebo|jackknife|multiplier|randomi[sz]ation inference"
    r"|neyman|whole-cluster|simultaneous",
    re.I,
)
ENCLOSURE = re.compile(r"enclosure|partial identification|sensitivity interval|bound interval", re.I)
TEMPORAL_MARK = re.compile(r"temporal|panel|sequential|series|period|time", re.I)
UNCERTAINTY_RANK = {"none": 0, "enclosure": 1, "analytic": 2, "design_aware_resampling": 2, "posterior": 2}


def toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def split_id(child_id: str) -> tuple[str, str, str]:
    """Method, design and inference of a child id such as P01.linear.temporal.bayesian."""
    parts = child_id.split(".")
    return ".".join(parts[:-2]), parts[-2], parts[-1]


def uncertainty_segment(text: str, inference: str) -> str:
    """The part of a route's uncertainty sentence that speaks for one inference side."""
    side = "Frequentist" if inference == "frequentist" else "Bayesian"
    other = "Bayesian" if inference == "frequentist" else "Frequentist"
    match = re.search(rf"{side}:(.*?)(?:;\s*{other}:|$)", text, re.S)
    return match.group(1) if match else text


def uncertainty_kind(inference: str, status: str, reason: str, procedure: str, ident: str) -> str:
    if status in ("closed_with_next_attempt", "not_applicable_with_theory_reason"):
        return "none"
    if NO_INTERVAL.search(reason) or NO_INTERVAL.search(procedure):
        return "none"
    if inference == "bayesian":
        return "posterior"
    if ENCLOSURE.search(procedure) or ENCLOSURE.search(ident):
        return "enclosure"
    if RESAMPLING.search(procedure) or RESAMPLING.search(reason) or RESAMPLING.search(ident):
        return "design_aware_resampling"
    return "analytic"


def child_status(child: dict, route: dict | None) -> tuple[str, str]:
    """Status and closure kind of a child; the reason text is the child's own sentence."""
    gap, missing = child["gap"], child["missing"]
    if gap == "none":
        licensed = route is not None and route["status"] == "licensed"
        return ("licensed" if licensed else "executing_unlicensed"), "delivered"
    if gap == "target" or missing.lower().startswith("not applicable"):
        return "not_applicable_with_theory_reason", "delivered"
    executes = gap == "validation" or (gap == "implementation" and child["module"] != "none")
    if executes:
        licensed = route is not None and route["status"] == "licensed"
        return ("licensed" if licensed else "executing_unlicensed"), "narrowed_promise"
    return "closed_with_next_attempt", "narrowed_promise"


def scope_by_id(scope: dict) -> dict[str, dict]:
    return {entry["id"]: entry for entry in scope["entry"]}


def child_rows(children: list[dict], routes: dict[str, dict], scope: dict[str, dict]) -> list[dict]:
    rows = []
    for child in children:
        method, design, inference = split_id(child["id"])
        route = routes.get(child["route"])
        status, closure = child_status(child, route)
        reason = "" if child["missing"] == "none" else child["missing"]
        segment = uncertainty_segment(route["uncertainty"], inference) if route else ""
        kind = uncertainty_kind(inference, status, reason, segment, child["inference"])
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
                "sampling_design": route["design_time_model"] if route else "unlisted",
                "reuse_key": child["inference"] if child["inference"] != "unlisted" else child["route"],
                "decision_port": route["decision_ports"] if route else "unlisted",
                "artifact": route["artifact_consumer"] if route else "unlisted",
                "closure": closure,
                "scope_status": entry.get("status", "unlisted"),
                "gap": child["gap"],
                "reason": reason,
                "next_attempt": NEXT_ATTEMPT.get(child["gap"], "") if closure == "narrowed_promise" else "",
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
    return "MeasurementCorrectedFunctional" if route["id"].startswith("measurement.") else route["target"]


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


def route_rows(routes: list[dict], lowerings: dict[str, str]) -> list[dict]:
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
        targets = [route_method(route), *counterparts]
        for inference in inferences:
            segment = uncertainty_segment(route["uncertainty"], inference)
            status = (
                "closed_with_next_attempt" if closed
                else "licensed" if route["status"] == "licensed" else "executing_unlicensed"
            )
            reason = route["notes"] if closed else ""
            weak = NO_INTERVAL.search(segment) is not None
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
                        "uncertainty": uncertainty_kind(inference, status, "", segment, route["id"]),
                        "sampling_design": route["design_time_model"],
                        "reuse_key": lowerings.get(route["id"], route["id"]),
                        "decision_port": route["decision_ports"],
                        "artifact": route["artifact_consumer"],
                        "closure": "narrowed_promise" if closed or weak or narrowed else "delivered",
                        "scope_status": "mandatory",
                        "gap": "target" if closed else ("validation" if weak or narrowed else "none"),
                        "reason": (
                            route[reason_key] if narrowed
                            else reason or (segment.strip()[:300] if weak else "")
                        ),
                        "next_attempt": (
                            route[attempt_key] if narrowed
                            else CLOSED_ROUTE_NEXT if closed
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
                    "id": f"{target}#static.{inference}", "kind": "route", "method": target,
                    "design": "static", "inference": inference, "route": "unlisted",
                    "status": "not_applicable_with_theory_reason", "uncertainty": "none",
                    "sampling_design": "unlisted", "reuse_key": "unlisted",
                    "decision_port": "unlisted", "artifact": "unlisted", "closure": "delivered",
                    "scope_status": "mandatory", "gap": "target",
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
    manifest = load(ROUTES)["route"]
    routes = {route["id"]: route for route in manifest}
    scope = scope_by_id(load(SCOPE))
    rows = child_rows(children, routes, scope)
    rows += route_rows(manifest, licensed_lowerings(load(LICENSED)))
    rows.sort(key=lambda row: (row["kind"], row["method"], QUADRANTS.index(f"{row['design']}.{row['inference']}"), row["id"]))
    return rows


FIELDS = (
    "id", "kind", "method", "design", "inference", "route", "status", "uncertainty",
    "sampling_design", "reuse_key", "decision_port", "artifact", "closure", "scope_status",
    "gap", "reason", "next_attempt",
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
                cells.append(short[best(quads[quadrant])["status"]] if quadrant in quads else "**missing**")
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
        stale = [p for p, t in ((OUT_TOML, toml_text), (OUT_DOC, doc_text)) if not p.exists() or p.read_text(encoding="utf-8") != t]
        for path in stale:
            print(f"{path.relative_to(ROOT)} is stale; run python3 scripts/generate_counterparts.py")
        return 1 if stale else 0
    OUT_TOML.write_text(toml_text, encoding="utf-8")
    OUT_DOC.write_text(doc_text, encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT_TOML.relative_to(ROOT)} and {OUT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
