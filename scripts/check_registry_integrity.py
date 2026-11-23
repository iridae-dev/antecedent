#!/usr/bin/env python3
"""Registry integrity: reason codes, Python products, composition rows, generated lists, CI jobs, identity.

Run by scripts/gate_parity_schema.sh after the capability-row schema check. It reads the
closed reason-code vocabulary and checks that every registry, generated list and workflow
that cites it agrees. It does not read calibration: the shape of the coverage records is
`scripts/calibration_status.py --check`.
"""
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from _repo import ensure_scripts_on_path, load_toml
from parity_rows import raw_reason_emissions, reason_obligation, runtime_code_literals

root = Path(".")
print_counts = "--print-counts" in sys.argv
problems = []

# --- reason codes ---
vocab_path = root / "parity/reason_codes.toml"
if not vocab_path.is_file():
    problems.append("parity/reason_codes.toml missing")
    vocab = {"code": []}
else:
    vocab = load_toml(vocab_path)
codes = {row["id"]: row for row in vocab.get("code", [])}
uses = {cid: 0 for cid in codes}
# support_{closed,n_a,axes} `reason` fields are matrix prose, not vocabulary ids.
# Do not re-code those pre-existing typed refusals.
for path in sorted(root.glob("parity/*.toml")):
    if path.name == "reason_codes.toml":
        continue
    text = path.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{path}: {exc}")
        continue

    def walk(obj, path=path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                obligation = reason_obligation(path, key)
                if obligation and isinstance(value, str) and value.strip():
                    cid = value.strip()
                    if cid not in codes:
                        problems.append(f"{path}: unknown reason {cid!r}")
                    else:
                        uses[cid] += 1
                        if obligation not in codes[cid].get("applies_to", []):
                            problems.append(f"{path}: {cid} does not apply to {obligation}")
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

if print_counts:
    print("reason-code uses:")
    for cid, n in uses.items():
        print(f"  {cid}: {n} (max_uses={codes[cid].get('max_uses')})")

for cid, n in uses.items():
    # A runtime-refusal-only code is spent in Rust, where this gate cannot count it,
    # so a registry ratchet is meaningful only when the registries themselves cite it.
    if "max_uses" not in codes[cid]:
        if codes[cid].get("applies_to") != ["runtime_refusal"]:
            problems.append(f"parity/reason_codes.toml: {cid} needs max_uses")
        continue
    max_uses = int(codes[cid]["max_uses"])
    if n > max_uses:
        problems.append(f"parity/reason_codes.toml: {cid} uses={n} > max_uses={max_uses}")

# Refusal codes a Rust `fn code(&self)` returns are stable machine-readable strings
# callers switch on; each must be in the closed vocabulary.
for src in sorted(root.glob("crates/*/src/**/*.rs")):
    text = src.read_text(errors="ignore", encoding="utf-8")
    for literal in runtime_code_literals(src, text):
        if literal not in codes:
            problems.append(
                f"{src}: fn code returns {literal!r}, absent from parity/reason_codes.toml"
            )

# --- python_products ---
pp = root / "parity/python_products.toml"
if not pp.is_file():
    problems.append("parity/python_products.toml missing")
else:
    products = load_toml(pp)
    route_tests = []
    for i, row in enumerate(products.get("route", []), 1):
        for key in ("kind", "data", "structure", "test"):
            if key not in row:
                problems.append(f"python_products.toml route #{i} missing {key}")
        test = row.get("test")
        if isinstance(test, str) and test not in route_tests:
            route_tests.append(test)

    def _collects(tests):
        return (
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    "--collect-only",
                    "-q",
                    *(t.removeprefix("python/") for t in tests),
                ],
                cwd=root / "python",
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )

    # One collection for every route node; only on failure, one per node to
    # name the uncollectable ones.
    if route_tests and not _collects(route_tests):
        for test in route_tests:
            if not _collects([test]):
                problems.append(f"python_products.toml route test not collected: {test}")
    for i, row in enumerate(products.get("parameter", []), 1):
        for key in ("name", "entry_points", "binding", "test"):
            if key not in row:
                problems.append(f"python_products.toml parameter #{i} missing {key}")
        if row.get("binding") == "contract" and not row.get("contract_key"):
            problems.append(
                f"python_products.toml parameter {row.get('name')} missing contract_key"
            )

ensure_scripts_on_path()

# --- supported cells: reason-code eligibility comes from the registry row ---
lic = load_toml(root / "parity/support_cells.toml").get(
    "cell", []
)
for cell in lic:
    label = f"{cell.get('query')}/{cell.get('graph_class')}/{cell.get('inference')}"
    cited = cell.get("calibration_reason")
    if cited == "no_interval_reported" and cell.get("interval_method") not in (
        None,
        "none",
        "None",
    ):
        problems.append(
            f"support_cells.toml {label}: no_interval_reported cannot declare an interval method"
        )
    scope = codes.get(cited, {}).get("queries") if isinstance(cited, str) else None
    if scope is not None and cell.get("query") not in scope:
        problems.append(
            f"support_cells.toml {label}: {cited} on a {cell.get('query')} cell; "
            f"reason_codes.toml limits it to {scope}"
        )

# ---- [gates] registry scopes name real queries ----
axes_queries = set(
    load_toml(root / "parity/support_axes.toml").get(
        "queries", []
    )
)
for cid, code in codes.items():
    for query in code.get("queries") or []:
        if query not in axes_queries:
            problems.append(f"reason_codes.toml {cid}: queries names unknown query {query!r}")

# ---- [gates] required composition rows ----
# Deleting a composition row must fail here, not silently shrink what
# gate_composition.sh executes. Additions are free; removals are a reviewed
# edit of this list.
REQUIRED_COMPOSITION_ROWS = {
    "compiler.inspect_supported_cells",
    "compiler.e2e_supported_cells",
    "compiler.inspect_does_not_identify",
    "compiler.metadata_read_no_identify",
    "compiler.validate_result_bindings",
    "compiler.claim_handoff",
    "compiler.request_lifecycle",
    "compiler.discovery_class",
    "compiler.prior_transfer_composition",
    "compiler.prepared_matches_fresh",
    "compiler.adversarial_boundaries",
    "compiler.panel_response_cluster_bands",
    "compiler.panel_class_pulse_completion_masses",
    "compiler.panel_response_bayesian_unit_surfaces",
    "compiler.panel_class_pulse_bayesian",
    "compiler.panel_class_response_surfaces",
    "compiler.panel_class_multi_step_sustained",
    "compiler.contract_section",
    "compiler.request_identity",
    "compiler.provenance_ancestry",
    "compiler.identity_contract_file",
    "compiler.supported_family",
    "compiler.prepared_second_shot",
    "compiler.prepared_family",
    "compiler.identity_encoding",
    "compiler.dbn_atom_identity",
    "compiler.score_reuse_identity",
    "compiler.state_expected_version",
    "compiler.series_refresh_atomicity",
    "compiler.series_state_reuse",
    "compiler.score_batch_reuse",
    "compiler.capability_reports",
    "compiler.support_neighbors",
    "compiler.capability_prefix",
    "compiler.claim_core",
    "compiler.design_rank",
    "compiler.composition_prefix",
    "compiler.python_contract_smoke",
    "compiler.python_stub_conformance",
}
compiler_rows = {
    row.get("id"): row
    for row in load_toml(root / "parity/compiler.toml").get(
        "capabilities", []
    )
}
for rid in sorted(REQUIRED_COMPOSITION_ROWS - compiler_rows.keys()):
    problems.append(f"compiler.toml: required composition row {rid} missing")
for rid, row in compiler_rows.items():
    if not row.get("evidence_test") or not row.get("evidence_assertion"):
        problems.append(
            f"compiler.toml {rid}: composition row without evidence_test/evidence_assertion"
        )

# ---- [gates] runtime reason-code list matches the registry ----
# crates/antecedent-core/src/reason_codes_data.rs is generated from
# parity/reason_codes.toml; the Rust `unsupported_reason!` / `reason_code!`
# macros and Python CausalUnsupportedError validate against it.
data_rs = root / "crates/antecedent-core/src/reason_codes_data.rs"
if not data_rs.is_file():
    problems.append(f"{data_rs}: missing; run scripts/generate_support_matrix_docs.py")
else:
    generated = data_rs.read_text(encoding="utf-8")

    def _const(name: str) -> list[str]:
        m = re.search(rf"pub const {name}: &\[&str\] = &\[(.*?)\];", generated, re.S)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []

    want_all = sorted(codes)
    want_runtime = sorted(
        c for c, row in codes.items() if "runtime_refusal" in row.get("applies_to", [])
    )
    if _const("REASON_CODES") != want_all or _const("RUNTIME_REFUSAL_CODES") != want_runtime:
        problems.append(
            "crates/antecedent-core/src/reason_codes_data.rs is stale against "
            "parity/reason_codes.toml; run scripts/generate_support_matrix_docs.py"
        )
# Inspect compiled product code using the shared module-ownership parser. Test
# children may live in separate files, and inline tests need not end a module.
for path in sorted(root.glob("crates/*/src/**/*.rs")) + sorted(root.glob("python/src/**/*.rs")):
    for lineno in raw_reason_emissions(path):
        problems.append(
            f'{path}:{lineno}: raw "reason=..." literal; use antecedent::unsupported_reason!'
        )

# ---- [gates] job ids from a YAML parse of ci.yml ----
# scripts/ci_workflow.py is the one reader of both ci.yml and release.toml's
# `required_jobs`; gate_release_candidate.sh calls the same subcommand.
listed = subprocess.run(
    [
        "uv",
        "run",
        "--quiet",
        "--project",
        ".",
        "--only-group",
        "dev",
        "python",
        str((root / "scripts/ci_workflow.py").resolve()),
        "required-jobs",
        "--json",
        "--workflow",
        str((root / ".github/workflows/ci.yml").resolve()),
        "--release",
        str((root / "parity/release.toml").resolve()),
    ],
    cwd=root / "python",
    capture_output=True,
    text=True,
)
if listed.returncode != 0:
    problems.append(f"scripts/ci_workflow.py required-jobs failed: {listed.stdout}{listed.stderr}")
else:
    problems.extend(json.loads(listed.stdout)["problems"])

# ---- [gates] supported cells link to the external oracles their routes rest on ----
# parity/supported_routes.toml (generated from executed plans) must route exactly
# the supported cells, and every `oracle_components` entry must sit on an external
# oracle row and name a component a supported cell actually runs
# (scripts/external_evidence.py; rendered by generate_support_matrix_docs.py).
import external_evidence  # noqa: E402

problems.extend(f"external evidence: {p}" for p in external_evidence.check())

# --- identity / claims files exist ---
for rel_path in ("parity/identity.toml", "parity/claims.toml"):
    if not (root / rel_path).is_file():
        problems.append(f"{rel_path} missing")

# --- every identity row names its layer and its bindings ---
identity_rows = load_toml(root / "parity/identity.toml").get(
    "identity", []
)
for row in identity_rows:
    for field in (
        "domain",
        "rust",
        "contract_key",
        "python",
        "availability",
        "naming_row",
        "covers",
    ):
        if not row.get(field):
            problems.append(f"identity.toml {row.get('domain')}: missing {field}")

if problems:
    print("registry integrity violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("registry integrity: ok")
