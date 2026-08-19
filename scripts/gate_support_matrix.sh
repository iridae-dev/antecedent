#!/usr/bin/env bash
# Support-matrix honesty: axes match the live public surface; n/a and licensed
# rows are well-formed; unspecified cells do not exist (default is refused).
#
# Run standalone or via scripts/gate_release.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

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
    "frozen_external_oracle",
    "behavioral_parity",
    "contract_equivalence",
}

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

if sorted(queries) != sorted(live_queries):
    fail.append(
        "parity/support_axes.toml queries != python __all__ query names: "
        f"axes={sorted(queries)} live={sorted(live_queries)}"
    )

for name in ["Frequentist", "Bayesian"]:
    if name not in public:
        fail.append(f"{name} is an inference axis value but is not in python __all__")

for name in stage_queries:
    module = "transport" if name == "TransportQuery" else "interference"
    src = (root / "python/antecedent" / f"{module}.py").read_text()
    if f"class {name}" not in src:
        fail.append(f"{name}: no class {name} in python/antecedent/{module}.py")

# --- live GraphClass variants ------------------------------------------------
accepted = (root / "crates/antecedent/src/accepted.rs").read_text()
m = re.search(r"pub enum GraphClass \{([^}]+)\}", accepted, re.S)
if not m:
    fail.append("crates/antecedent/src/accepted.rs: could not parse GraphClass")
    live_graphs: list[str] = []
else:
    live_graphs = re.findall(r"^\s+([A-Z][A-Za-z0-9]+),", m.group(1), re.M)
if sorted(graph_classes) != sorted(live_graphs):
    fail.append(
        "parity/support_axes.toml graph_classes != GraphClass: "
        f"axes={sorted(graph_classes)} live={sorted(live_graphs)}"
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

corpus_files = []
for pat in (
    "crates/**/*.rs",
    "python/tests/**/*.py",
    "python/antecedent/**/*.py",
    "scripts/*.sh",
    "scripts/*.py",
):
    corpus_files.extend(root.glob(pat))
corpus = "\n".join(
    p.read_text(errors="ignore")
    for p in corpus_files
    if "target" not in p.parts and ".venv" not in p.parts
)

def referenced(name: str) -> bool:
    for m in re.finditer(re.escape(name), corpus):
        before = corpus[m.start() - 1] if m.start() > 0 else ""
        after = corpus[m.end()] if m.end() < len(corpus) else ""
        if not re.match(r"[A-Za-z0-9_]", before) and not re.match(r"[A-Za-z0-9_]", after):
            return True
    return False

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

# --- allowlist rules -----------------------------------------------------------
allowed_rules = allow_doc.get("allowed") or []
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

# An allowed rule must not match any licensed, n/a, or closed cell -- the
# allowlist is a true partition of the running-but-unlicensed remainder.
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
                fail.append(f"{label}: matches a closed cell {(q, g, s, inf, v)}")

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
for i, row in enumerate(cells, 1):
    label = f"parity/support_licensed.toml cell #{i}"
    for key in required:
        if key not in row:
            fail.append(f"{label}: missing {key}")
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
    if not isinstance(fixture, str) or not fixture.strip():
        fail.append(f"{label}: known_truth_fixture is required")
    elif not (root / fixture).exists():
        fail.append(f"{label}: known_truth_fixture {fixture!r} does not exist")
    elif not referenced(Path(fixture).name):
        fail.append(
            f"{label}: known_truth_fixture {fixture!r} is not named in executing tests"
        )
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
        fail.append(f"{label}: cell is closed under support_closed.toml; cannot license it")
    if all(cell.values()) and is_allowed(cell):
        fail.append(f"{label}: cell matches support_allowlist.toml; cannot license it")

# --- counts ------------------------------------------------------------------
if all_queries and graph_classes and structures and inferences and validations:
    cartesian = 0
    n_a_count = 0
    closed_count = 0
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
            closed_count += 1
        elif is_allowed(cell):
            allowed_count += 1
    refused = cartesian - n_a_count - len(cells)
    if refused < 0:
        fail.append("licensed + n/a exceeds the cartesian product")
    if closed_count + allowed_count > refused:
        fail.append("closed + allowlisted exceed remaining refused cells")
else:
    cartesian = n_a_count = closed_count = allowed_count = refused = 0
    fail.append("axes are incomplete; cannot form a cartesian product")

if fail:
    print("Support matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    f"Support matrix OK ({cartesian} cells; {len(cells)} licensed; "
    f"{n_a_count} n/a; {closed_count} closed; {allowed_count} allowlisted; "
    f"{refused - closed_count - allowed_count} refused (enforced, no allowlist match))"
)
PY
