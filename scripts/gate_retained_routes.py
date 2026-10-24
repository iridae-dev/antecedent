"""Static checks that every route reaches its retained program and never re-enters the builder.

The route set is derived from three records and the sources that implement them:

* the route manifest (`parity/capabilities/route_manifest.toml`): every licensed,
  graphless and execution-only route, and its `retained_operation` field;
* the `route_lowerings!` registry and the sealed-execution table in
  `crates/antecedent/src/analysis`: which cells lower, which sealed operations
  exist, which sealer admits which cell;
* the licensed-route record (`parity/licensed_routes.toml`).

`retained_route_problems(root)` returns one message per violation:

* a licensed, graphless or execution-only route with no lowered program entry;
* a registered route cell no sealed operation serves, so its prepared execution is
  only a program-only operation (not the route's own retained operation);
* a manifest `retained_operation` that names no sealed operation, or one that does
  not serve the route's cell, or a graphless row without the field and its test;
* a sealed operation without a sealer, a retained-operation variant or a name;
* an `OrdinaryDispatch` or `LoweredDispatch` identifier in library code, an `EstimatorId` the
  estimator-family sealer's table omits, or a program-only operation built outside its owner;
* a dispatch entry that does not require the lowered program and enter its scope;
* a missing `public_builder_reentry` refusal at a compile or run entry, or a
  `route_not_lowered` refusal that a caller swallows.
"""

import datetime
import re
import tomllib
from pathlib import Path

from _rust_source import lex, rust_mask
from retained_computation import retained_computation_problems

ROOT = Path(__file__).resolve().parents[1]
SRC = "crates/antecedent/src"
ANALYSIS = f"{SRC}/analysis"
REGISTRY = f"{ANALYSIS}/lowering/registry.rs"
MANIFEST = "parity/capabilities/route_manifest.toml"
LICENSED_ROUTES = "parity/licensed_routes.toml"
EXECUTING = ("licensed", "graphless", "execution_only")

# Plan routes that are not registry cells and the module that lowers each.
PLAN_ROUTE_LOWERING = {
    "PropagationChain": f"{ANALYSIS}/lowering/chain.rs",
    "CyclicFamily": f"{ANALYSIS}/lowering/cyclic.rs",
    "ExperimentPlanning": f"{SRC}/planner/lowered/planning.rs",
}

# The cell each non-geometric manifest route lowers as. First match wins. A route that
# matches nothing has no lowered program entry and fails the gate.
ROUTE_CELLS = (
    (r"^discovery\.static$", "DiscoverStatic"),
    (r"^discovery\.temporal$", "DiscoverTemporal"),
    (r"^gcm\.attribute\.temporal\.", "GcmTemporal"),
    (r"^(?:gcm\.attribute|gcm\.attribution\.|attribution\.)", "GcmAttributeDag"),
    (r"^(?:gcm\.simulate|gcm\.model\.intervention_programs)$", "GcmSimulateDag"),
    (r"^gcm\.model\.mechanism_selection$", "GcmDag"),
    (r"^bayesian\.gcomp\.", "StaticAteDag"),
    (r"^composition\.temporal\.", "TemporalResponseDag"),
    (r"^(?:measurement|composition\.static)\.", "PropagationChain"),
    (r"^(?:bayesian\.)?temporal\.(?:glm_link_contrast|lagged_instrument)$", "TemporalEffectDag"),
    (r"^bayesian\.temporal\.white_innovation_linear$", "TemporalEffectDag"),
    (r"(?:^|\.)interference", "InterferenceDag"),
    (r"survival", "Survival"),
    (r"continuous_dose|dose_grid", "ContinuousDoseResponse"),
    (r"policy_value", "PolicyValue"),
    (
        r"difference_in_differences|panel_did|staggered_heterogeneity|"
        r"covariate_adjusted_two_period|augmented_panel",
        "PanelDid",
    ),
    (r"synthetic", "SyntheticControl"),
    (
        r"local_polynomial_ratio|local_ratio|fuzzy_rd|regression_kink|temporal_local_ratio",
        "LocalPolynomialRatio",
    ),
    (r"longitudinal_regime|sequential_regime_value", "LongitudinalRegime"),
    (r"sequential[._]crossfit", "SequentialContrast"),
    (r"period_history", "HistoryMatch"),
    (r"^(?:bayesian\.)?temporal_design\.", "TemporalDesign"),
    (r"^(?:bayesian\.)?iv_bounds\.", "IvBounds"),
    (r"^(?:bayesian\.)?association\.descriptive\.static$", "AssociationStatic"),
    (r"^(?:bayesian\.)?association\.descriptive\.temporal$", "AssociationTemporal"),
    (r"randomized|complier|switchback", "RandomizedEffect"),
)

# Sealers that do not read the parsed route (they are admitted by a graph posterior, a
# class envelope, a compiled cross-world operation or the query alone) declare the
# cells they serve here. A sealer that reads neither must be declared.
GRAPH_POSTERIOR = "GraphPosterior*"
DECLARED_SEALER_CELLS = {
    "posterior::graph_posterior_effect": (GRAPH_POSTERIOR,),
    "posterior::class_graph_posterior_effect": (GRAPH_POSTERIOR,),
    "posterior::bayesian_graph_posterior_ate": (GRAPH_POSTERIOR,),
    "posterior::admg_graph_posterior_response": (GRAPH_POSTERIOR,),
    "posterior::graph_posterior_response": (GRAPH_POSTERIOR,),
    "posterior::temporal_graph_posterior_effect": (GRAPH_POSTERIOR,),
    "posterior::temporal_graph_posterior_response": (GRAPH_POSTERIOR,),
    "posterior::bayesian_class_conditional": ("ConditionalCpdag", "ConditionalPag"),
    "posterior::bayesian_specialist": ("StaticAteDag",),
    "posterior::bayesian_temporal_class_effect": ("TemporalEffectClass",),
    "posterior::static_class_effect": ("StaticAtePag", "StaticAteCpdag"),
    "posterior::static_class_response": ("StaticResponsePag", "StaticResponseCpdag"),
    "posterior::temporal_class_response": ("TemporalResponseClass",),
    "posterior::transport_trial": ("TransportMixed",),
    "temporal::temporal_class_mediation": ("TemporalMediationPag", "TemporalMediationCpdag"),
    "estimator::estimator_route": ("*",),
    "estimator::fresh_run": (),
    "static_query::attribution": ("GcmDag", "GcmTemporal", "GcmAttributeDag", "GcmSimulateDag"),
    "static_query::counterfactual": ("GcmDag",),
    "static_query::nested_counterfactual": ("StaticMediationDag",),
}

# No execution is "ordinary" or an unnamed fallback: those variants are gone. A study the
# route-cell sealers decline is sealed by the estimator-family sealer, which lists every
# `EstimatorId`; a fresh run (`Study::execute`, one-shot `run`) is the only other place a
# program-only operation is built, once, where the run dispatches.
FORBIDDEN_IDENTIFIER = re.compile(r"\b(?:OrdinaryDispatch|LoweredDispatch)\b")
ESTIMATOR_KIND = "EstimatorRoute"
FRESH_KIND = "FreshRun"
PROGRAM_ONLY_KINDS = (ESTIMATOR_KIND, FRESH_KIND)
PROGRAM_ONLY_SITES = {
    ESTIMATOR_KIND: (f"{ANALYSIS}/prepared/sealers/estimator.rs", 1),
    FRESH_KIND: (f"{ANALYSIS}/execute/dispatch.rs", 1),
}
ESTIMATOR_IDS = f"{SRC}/strategy_table/ids.rs"
ESTIMATOR_TABLE = f"{ANALYSIS}/prepared/sealers/estimator.rs"
_SWALLOWED = re.compile(
    r"reason_code\(\)\s*==\s*Some\(\s*\"route_not_lowered\"\s*\)|\"route_not_lowered\"\s*\)\s*=>\s*None"
)
_CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test|all\s*\(\s*test\b[^\]]*)\)\s*\]")

# Compile and run entries that refuse a context already inside a lowered dispatch.
REENTRY_GUARDED = (
    f"{ANALYSIS}/execute/compile.rs",
    f"{ANALYSIS}/execute/dispatch.rs",
    f"{ANALYSIS}/execute/compile_planning.rs",
    f"{ANALYSIS}/execute/compile_decision.rs",
)
# Entries that require the plan's lowered program and run inside its dispatch scope.
DISPATCH_ENTRIES = (
    (f"{ANALYSIS}/execute/dispatch.rs", "fn execute_on("),
    (f"{ANALYSIS}/execute/dispatch.rs", "fn execute_tabular("),
    (f"{ANALYSIS}/prepared/estimate.rs", "fn estimate_with_shared_unnamed("),
    (f"{ANALYSIS}/prepared/lifecycle.rs", "fn run_sealed_series("),
)


def _read(root: Path, rel: str) -> str | None:
    path = root / rel
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _blank_test_items(masked: str) -> str:
    out = list(masked)
    for attribute in _CFG_TEST.finditer(masked):
        brace = masked.find("{", attribute.end())
        semicolon = masked.find(";", attribute.end())
        if semicolon != -1 and (brace == -1 or semicolon < brace):
            end = semicolon + 1
        elif brace == -1:
            continue
        else:
            depth, end = 0, len(masked)
            for index in range(brace, len(masked)):
                depth += (masked[index] == "{") - (masked[index] == "}")
                if depth == 0:
                    end = index + 1
                    break
        for index in range(attribute.start(), end):
            if out[index] != "\n":
                out[index] = " "
    return "".join(out)


def _is_test_path(path: Path) -> bool:
    return path.name == "tests.rs" or path.name.endswith("_tests.rs") or "tests" in path.parts[:-1]


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def registry_cells(root: Path) -> tuple[dict[str, str], list[str]]:
    """Registry cell -> family module, plus the graph-posterior cell names."""
    text = _read(root, REGISTRY) or ""
    start = text.find("route_lowerings! {")
    table = text[start:] if start >= 0 else ""
    table = table[: table.find("\n}")] if "\n}" in table else table
    cells = dict(re.findall(r"^\s*([A-Z][A-Za-z0-9]*)\s*=>\s*([a-z_]+)\s+as\s+[A-Za-z]+\s*,", table, re.M))
    posterior = re.findall(r'=>\s*"(GraphPosterior[A-Za-z]+)"', text)
    return cells, posterior


def sealed_kinds(root: Path) -> list[str]:
    text = rust_mask(_read(root, f"{ANALYSIS}/prepared/execution.rs") or "")
    start = text.find("sealed_executions! {\n")
    block = text[start:] if start >= 0 else ""
    block = block[: block.find("\n}")] if "\n}" in block else block
    return re.findall(r"^    ([A-Z][A-Za-z0-9]*)\(", block, re.M)


def sealer_table(root: Path) -> list[tuple[str, str]]:
    """(kind, 'module::fn') for every SEALERS entry."""
    text = rust_mask(_read(root, f"{ANALYSIS}/prepared/sealers/mod.rs") or "")
    start = text.find("pub(super) const SEALERS")
    table = text[start:] if start >= 0 else ""
    table = table[: table.find("\n];")] if "\n];" in table else table
    pattern = re.compile(
        r"sealer\(\s*ExecutionKind::(\w+),\s*(\w+::\w+)\s*,?\s*\)"
        r"|kind:\s*ExecutionKind::(\w+),\s*seal:\s*(\w+::\w+)"
    )
    entries = [
        (match.group(1) or match.group(3), match.group(2) or match.group(4))
        for match in pattern.finditer(table)
    ]
    return entries


def retained_variants(root: Path) -> dict[str, str]:
    """RetainedOperation variant -> stable name."""
    text = lex(_read(root, f"{ANALYSIS}/prepared/info.rs") or "").kept
    start = text.find("pub const fn name(&self)")
    body = text[start : text.find("\n    }\n", start)] if start >= 0 else ""
    pairs = re.findall(r"Self::(\w+)(?:\([^)]*\)|\s*\{[^}]*\})?\s*=>\s*\"(\w+)\"", body)
    return dict(pairs)


def retained_enum_variants(root: Path) -> set[str]:
    text = rust_mask(_read(root, f"{ANALYSIS}/prepared/info.rs") or "")
    start = text.find("pub enum RetainedOperation<'a> {")
    block = text[start : text.find("\n}\n", start)] if start >= 0 else ""
    return set(re.findall(r"^    ([A-Z][A-Za-z0-9]*)\b", block, re.M))


def sealer_cells(root: Path, table: list[tuple[str, str]], cells: set[str], posterior: list[str]) -> tuple[dict[str, set[str]], list[str]]:
    """kind -> cells its sealer serves, derived from the sealer sources and the declared table."""
    problems: list[str] = []
    directory = root / ANALYSIS / "prepared" / "sealers"
    mod_text = rust_mask(_read(root, f"{ANALYSIS}/prepared/sealers/mod.rs") or "")
    accessors: dict[str, set[str]] = {}
    for match in re.finditer(r"pub\(super\)\s+fn\s+(\w+)\s*\(", mod_text):
        rest = mod_text[match.end():]
        following = re.search(r"\n    (?:pub\(super\)\s+)?fn\s", rest)
        body = rest[: following.start()] if following else rest[:1500]
        accessors[match.group(1)] = set(re.findall(r"RouteInput::(\w+)", body))
    served: dict[str, set[str]] = {}
    for kind, path in table:
        module, name = path.split("::")
        source = rust_mask(_read(root, f"{ANALYSIS}/prepared/sealers/{module}.rs") or "")
        part = re.search(rf"(?m)^(?:pub\(super\)\s+)?fn\s+{name}\b.*?(?=^(?:pub\(super\)\s+)?fn\s|\Z)", source, re.S)
        found: set[str] = set()
        if part:
            found |= set(re.findall(r"RouteInput::(\w+)", part.group(0)))
            for accessor, accessor_cells in accessors.items():
                if re.search(rf"\.{accessor}\s*\(", part.group(0)):
                    found |= accessor_cells
        declared = DECLARED_SEALER_CELLS.get(path)
        if declared is not None:
            for cell in declared:
                if cell == "*":
                    found |= cells | set(posterior)
                else:
                    found |= set(posterior) if cell == GRAPH_POSTERIOR else {cell}
        if not found and kind != FRESH_KIND:
            problems.append(
                f"{ANALYSIS}/prepared/sealers/{module}.rs: sealer {path} for {kind} reads no route "
                "cell and declares none; declare the cells it serves in the retained-route gate"
            )
        for cell in found - cells - set(posterior):
            problems.append(f"sealer {path} names {cell}, which is not a registered route cell")
        if kind not in PROGRAM_ONLY_KINDS:
            served.setdefault(kind, set()).update(found)
    return served, problems


def route_cell(route: dict) -> str | None:
    for pattern, cell in ROUTE_CELLS:
        if re.search(pattern, route["id"]):
            return cell
    return None


def _test_citation_problems(root: Path, route: dict) -> list[str]:
    citation = route.get("retained_route_test")
    if citation is None:
        return []
    path, _, name = str(citation).partition("::")
    text = _read(root, path)
    if text is None or not re.search(rf"\bfn\s+{re.escape(name)}\b", text):
        return [f"{route['id']}: retained_route_test {citation!r} names no test function"]
    return []


def manifest_problems(root: Path, cells: dict[str, str], posterior: list[str], kinds: set[str],
                      names: dict[str, str], served: dict[str, set[str]]) -> list[str]:
    problems: list[str] = []
    text = _read(root, MANIFEST)
    if text is None:
        return [f"{MANIFEST}: not found"]
    routes = tomllib.loads(text).get("route", [])
    licensed_text = _read(root, LICENSED_ROUTES)
    licensed = tomllib.loads(licensed_text).get("route", []) if licensed_text else []
    plan_routes = set(PLAN_ROUTE_LOWERING)
    registered = set(cells) | set(posterior)
    name_to_variant = {name: variant for variant, name in names.items()}

    # Licensed geometric routes: manifest records and licensed-route rows agree.
    manifest_keys = {
        (route["target"], route["graph_class"]) for route in routes if route["status"] == "licensed"
    }
    record_keys = set()
    for row in licensed:
        target, graph_class = row["coordinate"].split(":")[:2]
        record_keys.add((target, graph_class))
        if row.get("lowering") not in registered:
            problems.append(f"{row['coordinate']}: lowering {row.get('lowering')!r} is not a registered route cell")
    for target, graph_class in sorted(record_keys - manifest_keys):
        problems.append(f"licensed_routes.toml holds {target}:{graph_class}, which the route manifest does not license")
    for target, graph_class in sorted(manifest_keys - record_keys):
        problems.append(f"route manifest licenses {target}:{graph_class}, which licensed_routes.toml lowers nowhere")

    for route in routes:
        status, rid = route["status"], route["id"]
        operation = route.get("retained_operation")
        if operation in ("estimator_route", "fresh_run"):
            problems.append(f"{rid}: retained_operation names a program-only operation, not a route's own")
        elif operation is not None and operation not in name_to_variant:
            problems.append(f"{rid}: retained_operation {operation!r} names no sealed operation")
        if route.get("retained_route_test") is not None and operation is None:
            problems.append(f"{rid}: retained_route_test without a retained_operation")
        problems.extend(_test_citation_problems(root, route))
        if status == "graphless" and (operation is None or route.get("retained_route_test") is None):
            problems.append(f"{rid}: graphless route records no retained_operation and retained_route_test")
        if status not in EXECUTING or status == "licensed":
            continue
        cell = route_cell(route)
        if cell is None or (cell not in registered and cell not in plan_routes):
            problems.append(f"{rid}: executing route has no lowered program entry (cell {cell!r})")
            continue
        if operation in name_to_variant and cell in registered:
            variant = name_to_variant[operation]
            if variant not in kinds:
                problems.append(f"{rid}: retained_operation {operation!r} has no sealer")
            elif cell not in served.get(variant, set()):
                problems.append(
                    f"{rid}: retained_operation {operation!r} is not served for cell {cell}; "
                    f"its sealer admits {sorted(served.get(variant, set()))}"
                )
    return problems


# Estimator ids a cell sealer serves through the route cell its family lowers as: the cell
# sealer admits the study by cell, and the operation it seals selects the estimator. First
# match wins.
ESTIMATOR_FAMILY_CELLS = (
    (r"^Interference", "InterferenceDag"),
    (r"Survival$|PiecewiseSurvival|RandomizedSurvival", "Survival"),
    (r"TemporalDesign|TemporalRegimeSurvival", "TemporalDesign"),
    (r"Policy|BinaryPolicyValue", "PolicyValue"),
    (r"ContinuousDose|DosePolicy", "ContinuousDoseResponse"),
    (r"Randomized|ComplierEffect|Switchback", "RandomizedEffect"),
    (r"Longitudinal|SequentialRegimeValue", "LongitudinalRegime"),
    (r"PanelDid", "PanelDid"),
    (r"SyntheticControl", "SyntheticControl"),
    (r"LocalRatio|LocalPolynomial", "LocalPolynomialRatio"),
    (r"SequentialCrossfit", "SequentialContrast"),
    (r"PeriodHistory", "HistoryMatch"),
    (r"^TransportTrial", "TransportMixed"),
    (r"^Gcm", "GcmDag"),
)
ESTIMATOR_RESIDUAL = "parity/estimator_route_residual.toml"


def estimator_table_problems(root: Path, served: dict[str, set[str]]) -> list[str]:
    """The family sealer's table is the committed residual; every other id has a specialised sealer."""
    ids_text = rust_mask(_read(root, ESTIMATOR_IDS) or "")
    start = ids_text.find("pub enum EstimatorId")
    block = ids_text[start : ids_text.find("\n}", start)] if start >= 0 else ""
    ids = set(re.findall(r"^    ([A-Z][A-Za-z0-9]*)\s*[,=({]", block, re.M))
    table_text = rust_mask(_read(root, ESTIMATOR_TABLE) or "")
    start = table_text.find("FAMILY_ESTIMATORS: &[EstimatorId] = &[")
    table = table_text[start : table_text.find("\n];", start)] if start >= 0 else ""
    listed = set(re.findall(r"EstimatorId::([A-Z][A-Za-z0-9]*)", table))
    residual_text = _read(root, ESTIMATOR_RESIDUAL)
    if not ids or not listed or residual_text is None:
        return [f"{ESTIMATOR_TABLE} or {ESTIMATOR_RESIDUAL}: not found; update the gate"]
    rows = tomllib.loads(residual_text).get("estimator", [])
    residual = {row["id"] for row in rows}
    flagged = {
        row["id"] for row in rows if row.get("served_by_specialised_sealer_for_other_shapes") is True
    }
    problems = [
        f"{ESTIMATOR_RESIDUAL}: {row['id']} needs a reason"
        for row in rows
        if not str(row.get("reason", "")).strip()
    ]
    problems.extend(f"the estimator-family table lists {n}, which is not an EstimatorId" for n in sorted(listed - ids))
    problems.extend(f"{ESTIMATOR_RESIDUAL} lists {n}, which is not an EstimatorId" for n in sorted(residual - ids))
    problems.extend(
        f"the estimator-family table lists {n}, which {ESTIMATOR_RESIDUAL} does not" for n in sorted(listed - residual)
    )
    problems.extend(
        f"{ESTIMATOR_RESIDUAL} lists {n}, which the estimator-family table omits" for n in sorted(residual - listed)
    )
    sealer_sources = "".join(
        rust_mask(path.read_text(encoding="utf-8"))
        for path in sorted((root / ANALYSIS / "prepared" / "sealers").glob("*.rs"))
        if path.name != "estimator.rs"
    )
    specialised_cells = set().union(*served.values()) if served else set()

    def serves(name: str) -> bool:
        if re.search(rf"EstimatorId::{name}\b", sealer_sources):
            return True
        return any(
            re.search(pattern, name) and cell in specialised_cells
            for pattern, cell in ESTIMATOR_FAMILY_CELLS
        )

    for name in sorted(ids):
        if name in residual and serves(name) and name not in flagged:
            problems.append(
                f"{ESTIMATOR_RESIDUAL} lists {name}, which a specialised sealer serves; flag it "
                "served_by_specialised_sealer_for_other_shapes and state the shapes that fall here"
            )
        if name in flagged and not serves(name):
            problems.append(
                f"{ESTIMATOR_RESIDUAL} flags {name} as served by a specialised sealer, which names it nowhere"
            )
        if name not in residual and not serves(name):
            problems.append(
                f"estimator {name} is served by no specialised sealer and is not in {ESTIMATOR_RESIDUAL}"
            )
    return problems


def ordinary_dispatch_problems(root: Path) -> list[str]:
    problems: list[str] = []
    base = root / SRC
    sites: dict[str, dict[str, int]] = {kind: {} for kind in PROGRAM_ONLY_KINDS}
    for path in sorted(base.rglob("*.rs")):
        rel = path.relative_to(root)
        if _is_test_path(rel):
            continue
        masked = _blank_test_items(rust_mask(path.read_text(encoding="utf-8")))
        for match in FORBIDDEN_IDENTIFIER.finditer(masked):
            problems.append(
                f"{rel}:{_line(masked, match.start())}: {match.group(0)} is reintroduced; every "
                "prepared or fresh execution holds a named sealed operation"
            )
        for kind in PROGRAM_ONLY_KINDS:
            uses = len(re.findall(rf"PreparedExecution::{kind}\s*\((?!\s*_\s*\))", masked))
            if uses:
                sites[kind][str(rel)] = uses
    for kind, (owner, count) in PROGRAM_ONLY_SITES.items():
        if sites[kind] != {owner: count}:
            problems.append(
                f"PreparedExecution::{kind} is built at {sites[kind]}; only {owner} "
                f"({count} site) may build it"
            )
    return problems


def reentry_guard_problems(root: Path) -> list[str]:
    problems: list[str] = []
    mod = lex(_read(root, f"{ANALYSIS}/lowering/mod.rs") or "").kept
    if not mod or not re.search(r"fn refuse_reentry\b", mod) or "public_builder_reentry" not in mod:
        problems.append(f"{ANALYSIS}/lowering/mod.rs: the public_builder_reentry refusal (refuse_reentry) is missing")
    elif "lowered_dispatch().is_some()" not in mod:
        problems.append(f"{ANALYSIS}/lowering/mod.rs: refuse_reentry no longer checks the lowered-dispatch scope")
    for rel in REENTRY_GUARDED:
        text = _read(root, rel)
        if text is None or "refuse_reentry(" not in rust_mask(text):
            problems.append(f"{rel}: the compile or run entry no longer calls refuse_reentry")
    chain = lex(_read(root, f"{ANALYSIS}/propagation_chain.rs") or "").kept
    if "lowered_dispatch().is_some()" not in chain or "public_builder_reentry" not in chain:
        problems.append(f"{ANALYSIS}/propagation_chain.rs: chain compile no longer refuses public_builder_reentry")
    if "enter_lowered_dispatch(" not in chain:
        problems.append(f"{ANALYSIS}/propagation_chain.rs: chain execution no longer enters its lowered dispatch")
    planning = _read(root, f"{ANALYSIS}/execute/compile_planning.rs") or ""
    if "enter_lowered_dispatch(" not in planning:
        problems.append(f"{ANALYSIS}/execute/compile_planning.rs: planning execution no longer enters its lowered dispatch")
    codes = _read(root, "parity/reason_codes.toml") or ""
    for code in ("public_builder_reentry", "route_not_lowered"):
        if f'id = "{code}"' not in codes:
            problems.append(f"parity/reason_codes.toml: {code} is not registered")
    for rel, marker in DISPATCH_ENTRIES:
        text = rust_mask(_read(root, rel) or "")
        start = text.find(marker)
        if start < 0:
            problems.append(f"{rel}: dispatch entry {marker.removeprefix('fn ').rstrip('(')} not found; update the gate")
            continue
        body = text[start : start + 900]
        if "require_lowered(" not in body or "enter_lowered_dispatch(" not in body:
            problems.append(
                f"{rel}: {marker.removeprefix('fn ').rstrip('(')} does not require the lowered program "
                "and enter its dispatch scope"
            )
    base = root / SRC
    for path in sorted(base.rglob("*.rs")):
        rel = path.relative_to(root)
        if _is_test_path(rel):
            continue
        source = lex(path.read_text(encoding="utf-8"))
        live = _blank_test_items(source.skeleton)
        for match in _SWALLOWED.finditer(source.kept):
            if live[match.start()] == " ":
                continue
            problems.append(
                f"{rel}:{_line(source.kept, match.start())}: swallows route_not_lowered; "
                "a plan with no lowered program is refused, not reported without it"
            )
    return problems


def retained_route_problems(root: Path = ROOT) -> list[str]:
    cells, posterior = registry_cells(root)
    if not cells:
        return [f"{REGISTRY}: no route cells found"]
    problems: list[str] = []
    for cell, family in sorted(cells.items()):
        source = _read(root, f"{ANALYSIS}/lowering/{family}.rs")
        if source is None or not re.search(r"\bfn\s+lower\s*[(<]", source):
            problems.append(f"{cell}: family lowering {family} has no lower entry")
    for route, rel in PLAN_ROUTE_LOWERING.items():
        if not (root / rel).is_file():
            problems.append(f"{route}: lowering {rel} not found")
    kinds = sealed_kinds(root)
    table = sealer_table(root)
    variants = retained_variants(root)
    enum_variants = retained_enum_variants(root)
    if not kinds or not table:
        return problems + ["sealed-execution table or sealer table not found; update the retained-route gate"]
    table_kinds = [kind for kind, _ in table]
    for kind in kinds:
        if kind not in table_kinds:
            problems.append(f"sealed operation {kind} has no sealer")
        if kind not in enum_variants or kind not in variants:
            problems.append(f"sealed operation {kind} has no RetainedOperation variant or stable name")
    for kind in sorted(set(table_kinds) - set(kinds)):
        problems.append(f"sealer table names {kind}, which is not a sealed operation")
    for kind in sorted({k for k in table_kinds if table_kinds.count(k) > 1}):
        problems.append(f"sealed operation {kind} has two sealers")
    if table_kinds[-2:] != list(PROGRAM_ONLY_KINDS):
        problems.append(
            f"the {ESTIMATOR_KIND} and {FRESH_KIND} sealers must be the last two in the sealer table"
        )
    served, sealer_problems = sealer_cells(root, table, set(cells), posterior)
    problems.extend(sealer_problems)
    reached = set().union(*served.values()) if served else set()
    for cell in sorted(cells):
        if cell not in reached:
            problems.append(
                f"{cell}: no sealed operation serves this route cell, so its prepared execution is "
                "OrdinaryDispatch (the ordinary dispatcher) and not a retained operation"
            )
    problems.extend(manifest_problems(root, cells, posterior, set(kinds), variants, served))
    manifest_text = _read(root, MANIFEST)
    routes = tomllib.loads(manifest_text).get("route", []) if manifest_text else []
    problems.extend(
        retained_computation_problems(
            root, routes, route_cell, served, variants, datetime.date.today().isoformat()
        )
    )
    problems.extend(estimator_table_problems(root, served))
    problems.extend(ordinary_dispatch_problems(root))
    problems.extend(reentry_guard_problems(root))
    return problems
