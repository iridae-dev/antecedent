#!/usr/bin/env bash
# Support-matrix honesty: axes match the live public surface; n/a and licensed
# rows are well-formed; unspecified cells do not exist (default is refused).
#
# Every licensed row's evidence_test / evidence_assertion must be an executing
# test (resolved through `cargo test -- --list`, not ignored; or a collected
# pytest), and, read with the helpers it calls (scripts/test_evidence.py), must
# consume the row's known_truth_fixture when the row claims known truth and build
# every axis value of the row.
#
# Every licensed row's checked_execution names, per licensed estimator, the
# executing test that drops its builder, executes the retained checked plan and
# inspects it (scripts/test_evidence.py checked_execution_problems); the cited
# tests run in scripts/gate_checked_execution.sh.
#
# Run standalone or via scripts/gate_release.sh.
#   bash scripts/gate_support_matrix.sh --self-test   # broken evidence must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/test_evidence_selftest.py"
fi

python3 - <<'PY'
from __future__ import annotations

import ast
import re
import sys
import tomllib
from itertools import product
from pathlib import Path

root = Path(".")
fail: list[str] = []

EVIDENCE_KINDS = {
    "implementation_exists",
    "internal_known_truth",
    "internal_cross_check",
    # frozen output of this library itself (a seeded run or a copied value): a change
    # detector, never truth. Names its test, never a known_truth_fixture.
    "regression_pin",
    "frozen_external_oracle",
    "behavioral_parity",
    "contract_equivalence",
}

def regression_pin_problem(fixture: str, inference: str) -> str | None:
    """Why `fixture` cannot back known truth for a cell of this inference mode.

    A fixture whose `oracle.kind` is `regression_pin` holds frozen output of this library. Only
    the inference modes its oracle lists in `independent_inferences` (a part checked against an
    independent reference beside the pin) may cite it as truth."""
    import json

    path = root / fixture / "expected.json"
    if not path.is_file():
        return None
    try:
        oracle = json.loads(path.read_text()).get("oracle")
    except (OSError, ValueError):
        return None
    if isinstance(oracle, dict) and oracle.get("kind") == "regression_pin":
        if inference not in (oracle.get("independent_inferences") or []):
            return (
                f"known_truth_fixture {fixture} is a regression_pin fixture (frozen output of "
                f"this library) and lists no independent {inference} reference; the cell is "
                "evidence_kind = regression_pin, not known truth"
            )
    return None


def load(rel: str) -> dict:
    path = root / rel
    if not path.is_file():
        fail.append(f"{rel}: missing")
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        fail.append(f"{rel}: not valid TOML: {exc}")
        return {}

axes = load("parity/support_axes.toml")
na_doc = load("parity/support_n_a.toml")
closed_doc = load("parity/support_closed.toml")
allow_doc = load("parity/support_allowlist.toml")
lic_doc = load("parity/support_licensed.toml")

queries = list(axes.get("queries") or [])
stage_queries = list(axes.get("stage_queries") or [])
graph_classes = list(axes.get("graph_classes") or [])
structures = list(axes.get("structures") or [])
inferences = list(axes.get("inferences") or [])
validations = list(axes.get("validations") or [])
all_queries = queries + stage_queries

# --- live Python root query names -------------------------------------------
init_text = (root / "python/antecedent/__init__.py").read_text()
mod = ast.parse(init_text)
public: list[str] | None = None
for node in mod.body:
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "__all__" in names and isinstance(node.value, (ast.List, ast.Tuple)):
            public = []
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    public.append(elt.value)
            break
if public is None:
    fail.append("python/antecedent/__init__.py: could not parse __all__")
    public = []

# Queries live between the "Queries" and "Graphs" comment blocks in __all__.
q_start = init_text.find("# Queries")
q_end = init_text.find("# Graphs")
if q_start < 0 or q_end < 0 or q_end <= q_start:
    fail.append("python/antecedent/__init__.py: missing # Queries / # Graphs markers in __all__")
    live_queries: list[str] = []
else:
    block = init_text[q_start:q_end]
    live_queries = re.findall(r'"([A-Za-z][A-Za-z0-9]+)"', block)

# TransportQuery stays on the support-matrix query axis (licensed trial-IPW
# cell) but lives on `antecedent.transport.advanced`, not root `__all__`.
root_queries = [q for q in queries if q != "TransportQuery"]
# Public query class names can differ from the stable support query axis.
# Keep those mappings explicit so adding a user-facing type does not silently
# rename support cells or their evidence records.
query_axis_aliases = {"NestedCounterfactual": "NestedCounterfactualEffect"}
mapped_live_queries = [query_axis_aliases.get(q, q) for q in live_queries]
if sorted(root_queries) != sorted(mapped_live_queries):
    fail.append(
        "parity/support_axes.toml queries != python __all__ query names: "
        f"axes={sorted(root_queries)} live={sorted(mapped_live_queries)}"
    )

for name in ["Frequentist", "Bayesian"]:
    if name not in public:
        fail.append(f"{name} is an inference axis value but is not in python __all__")

for name in stage_queries:
    if name == "TransportQuery":
        src_path = root / "python/antecedent/transport/_impl.py"
    else:
        src_path = root / "python/antecedent/interference.py"
    src = src_path.read_text() if src_path.is_file() else ""
    if f"class {name}" not in src:
        fail.append(f"{name}: no class {name} in {src_path.relative_to(root)}")

# --- live GraphClass variants ------------------------------------------------
accepted = (root / "crates/antecedent/src/accepted.rs").read_text()
m = re.search(r"pub enum GraphClass \{([^}]+)\}", accepted, re.S)
if not m:
    fail.append("crates/antecedent/src/accepted.rs: could not parse GraphClass")
    live_graphs: list[str] = []
else:
    live_graphs = re.findall(r"^\s+([A-Z][A-Za-z0-9]+),", m.group(1), re.M)
# GraphClass must be on the axis. Classification-only extras (tier-rule
# backgrounds) may appear in addition; they are not GraphClass variants.
TIER_EXTRAS = {"CoDetermined", "Unknown"}
if not set(live_graphs) <= set(graph_classes):
    fail.append(
        "parity/support_axes.toml graph_classes must contain GraphClass: "
        f"axes={sorted(graph_classes)} live={sorted(live_graphs)}"
    )
unknown_extras = set(graph_classes) - set(live_graphs) - TIER_EXTRAS
if unknown_extras:
    fail.append(
        "parity/support_axes.toml graph_classes has extras that are not "
        f"GraphClass or tier-rule axes: {sorted(unknown_extras)}"
    )
if not TIER_EXTRAS <= set(graph_classes):
    fail.append(
        "parity/support_axes.toml graph_classes must include "
        f"{sorted(TIER_EXTRAS)} (classification-only tier-rule axes)"
    )

expected_structures = {"explicit", "accepted", "graph_posterior"}
if set(structures) != expected_structures:
    fail.append(f"structures must be {sorted(expected_structures)}; got {structures}")
expected_inferences = {"Frequentist", "Bayesian"}
if set(inferences) != expected_inferences:
    fail.append(f"inferences must be {sorted(expected_inferences)}; got {inferences}")
expected_validations = {"none", "cheap", "full"}
if set(validations) != expected_validations:
    fail.append(f"validations must be {sorted(expected_validations)}; got {validations}")

for seq, label in (
    (queries, "queries"),
    (stage_queries, "stage_queries"),
    (graph_classes, "graph_classes"),
    (structures, "structures"),
    (inferences, "inferences"),
    (validations, "validations"),
):
    if len(seq) != len(set(seq)):
        fail.append(f"parity/support_axes.toml {label} has duplicates")
    if any(not isinstance(x, str) or not x.strip() for x in seq):
        fail.append(f"parity/support_axes.toml {label} has an empty value")

# --- n/a predicates ----------------------------------------------------------
na_rules = na_doc.get("n_a") or []
AXIS_KEYS = {
    "queries": set(all_queries),
    "graph_classes": set(graph_classes),
    "structures": set(structures),
    "inferences": set(inferences),
    "validations": set(validations),
}
for i, rule in enumerate(na_rules, 1):
    if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
        fail.append(f"parity/support_n_a.toml rule #{i}: missing reason")
    for key, legal in AXIS_KEYS.items():
        vals = rule.get(key)
        if vals is None:
            continue
        if not isinstance(vals, list) or not vals:
            fail.append(f"parity/support_n_a.toml rule #{i}: {key} must be a non-empty list")
            continue
        for v in vals:
            if v not in legal:
                fail.append(f"parity/support_n_a.toml rule #{i}: {key} value {v!r} is not an axis value")

def rule_matches(rule: dict, cell: dict) -> bool:
    mapping = {
        "queries": "query",
        "graph_classes": "graph_class",
        "structures": "structure",
        "inferences": "inference",
        "validations": "validation",
    }
    for rule_key, cell_key in mapping.items():
        allowed = rule.get(rule_key)
        if allowed is not None and cell[cell_key] not in allowed:
            return False
    return True

def is_n_a(cell: dict) -> bool:
    return any(rule_matches(rule, cell) for rule in na_rules)

# Cited evidence is read through scripts/test_evidence.py, the one reader of test
# source for the gates: the cited function must be an executing test and, with the
# helpers it calls, must consume its fixture and build the row's axis values.
sys.path.insert(0, str((root / "scripts").resolve()))
import test_evidence  # noqa: E402

closed_rules = closed_doc.get("closed") or []
for i, rule in enumerate(closed_rules, 1):
    if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
        fail.append(f"parity/support_closed.toml rule #{i}: missing reason")
    for key, legal in AXIS_KEYS.items():
        vals = rule.get(key)
        if vals is None:
            continue
        if not isinstance(vals, list) or not vals:
            fail.append(f"parity/support_closed.toml rule #{i}: {key} must be a non-empty list")
            continue
        for v in vals:
            if v not in legal:
                fail.append(
                    f"parity/support_closed.toml rule #{i}: {key} value {v!r} is not an axis value"
                )

def is_closed(cell: dict) -> bool:
    return (not is_n_a(cell)) and any(rule_matches(rule, cell) for rule in closed_rules)

# n/a and reason-backed refusals are disjoint. A too-broad n/a must not mask
# a reason row in legacy-named support_closed.toml.
if all_queries and graph_classes and structures and inferences and validations:
    lic_keys_for_overlap = {
        (
            row.get("query"),
            row.get("graph_class"),
            row.get("structure"),
            row.get("inference"),
            row.get("validation"),
        )
        for row in (lic_doc.get("cell") or [])
    }
    overlap_n = 0
    for q, g, s, inf, v in product(
        all_queries, graph_classes, structures, inferences, validations
    ):
        cell = {
            "query": q,
            "graph_class": g,
            "structure": s,
            "inference": inf,
            "validation": v,
        }
        closed_hit = next(
            (i for i, rule in enumerate(closed_rules, 1) if rule_matches(rule, cell)),
            None,
        )
        if closed_hit is None:
            continue
        if is_n_a(cell):
            overlap_n += 1
            if overlap_n <= 25:
                fail.append(
                    f"parity/support_closed.toml rule #{closed_hit} overlaps n/a cell "
                    f"{(q, g, s, inf, v)}"
                )
        if (q, g, s, inf, v) in lic_keys_for_overlap:
            fail.append(
                f"parity/support_closed.toml rule #{closed_hit} overlaps licensed cell "
                f"{(q, g, s, inf, v)}"
            )
    if overlap_n > 25:
        fail.append(f"... {overlap_n - 25} more n/a/refusal-reason overlaps")

# --- allowlist rules -----------------------------------------------------------
allowed_rules = allow_doc.get("allowed") or []
if allowed_rules:
    fail.append(
        "parity/support_allowlist.toml: 0.9 requires an empty allowlist; "
        f"found {len(allowed_rules)} active rule(s)"
    )
for i, rule in enumerate(allowed_rules, 1):
    label = f"parity/support_allowlist.toml rule #{i}"
    if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
        fail.append(f"{label}: missing reason")
    if not isinstance(rule.get("parent"), str) or not rule["parent"].strip():
        fail.append(f"{label}: missing parent")
    for key, legal in AXIS_KEYS.items():
        vals = rule.get(key)
        if vals is None:
            continue
        if not isinstance(vals, list) or not vals:
            fail.append(f"{label}: {key} must be a non-empty list")
            continue
        for v in vals:
            if v not in legal:
                fail.append(f"{label}: {key} value {v!r} is not an axis value")

def is_allowed(cell: dict) -> bool:
    return (
        (not is_n_a(cell))
        and (not is_closed(cell))
        and any(rule_matches(rule, cell) for rule in allowed_rules)
    )

# Retained compatibility rules must not match any licensed, n/a, or
# reason-backed refused cell. The 0.9 invariant above requires zero rules;
# these checks remain so a bad legacy entry reports all of its defects.
if all_queries and graph_classes and structures and inferences and validations:
    lic_keys_for_disjointness = {
        (row.get("query"), row.get("graph_class"), row.get("structure"), row.get("inference"), row.get("validation"))
        for row in (lic_doc.get("cell") or [])
    }
    for i, rule in enumerate(allowed_rules, 1):
        label = f"parity/support_allowlist.toml rule #{i}"
        for q, g, s, inf, v in product(
            all_queries, graph_classes, structures, inferences, validations
        ):
            if not rule_matches(rule, {"query": q, "graph_class": g, "structure": s, "inference": inf, "validation": v}):
                continue
            cell = {"query": q, "graph_class": g, "structure": s, "inference": inf, "validation": v}
            if (q, g, s, inf, v) in lic_keys_for_disjointness:
                fail.append(f"{label}: matches licensed cell {(q, g, s, inf, v)}")
            elif is_n_a(cell):
                fail.append(f"{label}: matches an n/a cell {(q, g, s, inf, v)}")
            elif is_closed(cell):
                fail.append(f"{label}: matches a reason-backed refused cell {(q, g, s, inf, v)}")

# --- licensed cells ----------------------------------------------------------
cells = lic_doc.get("cell") or []
seen: set[tuple[str, ...]] = set()
required = (
    "query",
    "graph_class",
    "structure",
    "inference",
    "validation",
    "staged",
    "evidence_kind",
)
legal_q = set(all_queries)

# Every staged licensed row names the executing test and test function that
# carries its evidence. Rows licensed before this rule are frozen in
# parity/_evidence_test_backlog.txt; the backlog only shrinks.
missing_evidence: set[str] = set()

# Estimator wire-ids recorded on licensed rows (secondary axis; not cartesian).
# When present, every entry must be a non-empty string. Empty lists are allowed
# only when the row honestly has no estimator evidence (classify_estimator then
# refuses every concrete EstimatorId, including the five unmeasured families).
def check_estimators(label: str, row: dict) -> None:
    named = row.get("estimators")
    if named is None:
        return
    if not isinstance(named, list):
        fail.append(f"{label}: estimators must be a list of wire-ids")
        return
    for est in named:
        if not isinstance(est, str) or not est.strip():
            fail.append(f"{label}: estimators entries must be non-empty strings")


def check_evidence_test(label: str, row: dict) -> None:
    for problem in test_evidence.row_evidence_problems(row):
        fail.append(f"{label}: {problem}")


# Every licensed estimator on a row executes through a retained checked
# operation: one checked_execution entry per estimator, each citing an
# executing test that discards its builder, executes the plan, and inspects it.
def check_checked_execution(label: str, row: dict) -> None:
    for problem in test_evidence.checked_execution_problems(row):
        fail.append(f"{label}: {problem}")


for i, row in enumerate(cells, 1):
    label = f"parity/support_licensed.toml cell #{i}"
    for key in required:
        if key not in row:
            fail.append(f"{label}: missing {key}")
    check_estimators(label, row)
    q = row.get("query")
    g = row.get("graph_class")
    s = row.get("structure")
    inf = row.get("inference")
    v = row.get("validation")
    if q not in legal_q:
        fail.append(f"{label}: query {q!r} is not an axis value")
    if g not in set(graph_classes):
        fail.append(f"{label}: graph_class {g!r} is not an axis value")
    if s not in set(structures):
        fail.append(f"{label}: structure {s!r} is not an axis value")
    if inf not in set(inferences):
        fail.append(f"{label}: inference {inf!r} is not an axis value")
    if v not in set(validations):
        fail.append(f"{label}: validation {v!r} is not an axis value")
    if row.get("staged") is not True:
        fail.append(f"{label}: staged must be true (analyze-only is not a license)")
    kind = row.get("evidence_kind")
    if not isinstance(kind, str) or kind not in EVIDENCE_KINDS:
        fail.append(f"{label}: evidence_kind is required and must be a known kind")
    if row.get("status") is not None:
        fail.append(f"{label}: status is illegal on a matrix cell")
    fixture = row.get("known_truth_fixture")
    if kind in {"internal_known_truth", "frozen_external_oracle"}:
        if not isinstance(fixture, str) or not fixture.strip():
            fail.append(f"{label}: {kind} requires known_truth_fixture")
        elif not (root / fixture).exists():
            fail.append(f"{label}: known_truth_fixture {fixture!r} does not exist")
        elif (why := regression_pin_problem(fixture, inf)) is not None:
            fail.append(f"{label}: {why}")
        elif not (row.get("evidence_test") and row.get("evidence_assertion")):
            fail.append(
                f"{label}: {kind} requires evidence_test/evidence_assertion whose own "
                "function consumes the fixture"
            )
    elif kind == "regression_pin":
        if fixture is not None:
            fail.append(
                f"{label}: regression_pin must not name known_truth_fixture; a frozen "
                "output of this library is not truth (name the pin fixture in limitations)"
            )
        if not str(row.get("limitations", "")).strip():
            fail.append(f"{label}: regression_pin requires limitations saying what is pinned")
        if not (row.get("evidence_test") and row.get("evidence_assertion")):
            fail.append(f"{label}: regression_pin requires evidence_test/evidence_assertion")
    elif kind == "internal_cross_check":
        if fixture is not None:
            fail.append(
                f"{label}: internal_cross_check must not name known_truth_fixture; "
                "that would present contextual data as independent truth evidence"
            )
        test_rel = row.get("evidence_test")
        assertion = row.get("evidence_assertion")
        if not isinstance(test_rel, str) or not test_rel.strip():
            fail.append(f"{label}: internal_cross_check requires evidence_test")
        elif not isinstance(assertion, str) or not assertion.strip():
            fail.append(f"{label}: internal_cross_check requires evidence_assertion")
    # Every named test must be an executing test function, whatever the kind.
    test_rel = row.get("evidence_test")
    assertion = row.get("evidence_assertion")
    has_test = isinstance(test_rel, str) and bool(test_rel.strip())
    has_assertion = isinstance(assertion, str) and bool(assertion.strip())
    if has_test != has_assertion:
        fail.append(f"{label}: evidence_test and evidence_assertion must be set together")
    elif has_test:
        check_evidence_test(label, row)
    check_checked_execution(label, row)
    if row.get("staged") is True and not (has_test and has_assertion):
        missing_evidence.add("|".join(str(x) for x in (q, g, s, inf, v)))
    key = (q, g, s, inf, v)
    if key in seen:
        fail.append(f"{label}: duplicate cell {key}")
    seen.add(key)
    cell = {
        "query": q,
        "graph_class": g,
        "structure": s,
        "inference": inf,
        "validation": v,
    }
    if all(cell.values()) and is_n_a(cell):
        fail.append(f"{label}: cell is n/a under support_n_a.toml; cannot license it")
    if all(cell.values()) and is_closed(cell):
        fail.append(
            f"{label}: cell has a refusal reason under support_closed.toml; cannot license it"
        )
    if all(cell.values()) and is_allowed(cell):
        fail.append(f"{label}: cell matches support_allowlist.toml; cannot license it")

# --- cross-check-only cells need a known-truth sibling -------------------------
# A cell whose only evidence is an internal cross-check (the estimator against an
# in-test recomputation that may share its convention) and that carries no measured
# coverage has nothing tying it to the truth. It is licensed only while another cell of
# the same query and inference mode has known-truth or external-oracle evidence, so the
# estimand itself is checked against a simulation truth somewhere.
_truth_kinds = {"internal_known_truth", "frozen_external_oracle"}
_truthful = {
    (c.get("query"), c.get("inference"))
    for c in cells
    if c.get("evidence_kind") in _truth_kinds
}
for row in cells:
    if row.get("evidence_kind") in {"internal_cross_check", "regression_pin"} and not row.get(
        "calibration"
    ):
        if (row.get("query"), row.get("inference")) not in _truthful:
            fail.append(
                f"{row.get('query')}/{row.get('graph_class')}/{row.get('structure')}/"
                f"{row.get('inference')}/{row.get('validation')}: {row.get('evidence_kind')} with no "
                "calibration and no known-truth sibling cell of the same query and inference "
                "mode; add known-truth evidence or a measured calibration"
            )

# --- evidence_test ratchet ---------------------------------------------------
backlog_rel = "parity/_evidence_test_backlog.txt"
backlog_path = root / backlog_rel
evidence_backlog: set[str] = set()
if backlog_path.is_file():
    for line in backlog_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            evidence_backlog.add(line)
for key in sorted(missing_evidence - evidence_backlog):
    fail.append(
        f"licensed cell {key} is staged but has no evidence_test/evidence_assertion; "
        "name the executing test file and test function (do not add rows to "
        f"{backlog_rel} -- it is frozen)"
    )
for key in sorted(evidence_backlog - missing_evidence):
    fail.append(
        f"{backlog_rel} lists {key} but that row now names its evidence (or is no "
        "longer licensed) -- delete the line in the same change (the list only shrinks)"
    )

# --- counts ------------------------------------------------------------------
if all_queries and graph_classes and structures and inferences and validations:
    cartesian = 0
    n_a_count = 0
    reason_backed_refused_count = 0
    allowed_count = 0
    for q, g, s, inf, v in product(
        all_queries, graph_classes, structures, inferences, validations
    ):
        cartesian += 1
        cell = {
            "query": q,
            "graph_class": g,
            "structure": s,
            "inference": inf,
            "validation": v,
        }
        if is_n_a(cell):
            n_a_count += 1
        elif is_closed(cell):
            reason_backed_refused_count += 1
        elif is_allowed(cell):
            allowed_count += 1
    refused = cartesian - n_a_count - len(cells)
    unreasoned = refused - reason_backed_refused_count - allowed_count
    if refused < 0:
        fail.append("licensed + n/a exceeds the cartesian product")
    if reason_backed_refused_count + allowed_count > refused:
        fail.append("reason-backed refusals + compatibility entries exceed refused cells")
    if unreasoned:
        fail.append(
            f"{unreasoned} meaningful refused cell(s) have no reason in "
            "parity/support_closed.toml; every refused cell must name why"
        )
else:
    cartesian = n_a_count = reason_backed_refused_count = allowed_count = refused = 0
    fail.append("axes are incomplete; cannot form a cartesian product")

if fail:
    print("Support matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    f"Support matrix OK ({cartesian} cells; {len(cells)} licensed; "
    f"{n_a_count} n/a; {reason_backed_refused_count} refused with reasons; "
    f"{refused - reason_backed_refused_count - allowed_count} refused without reasons; "
    f"{allowed_count} active allowed_unlicensed compatibility entries)"
)
PY

python3 "$ROOT/scripts/check_transport_stages.py"
python3 "$ROOT/scripts/generate_calibration_backlog.py" --check
