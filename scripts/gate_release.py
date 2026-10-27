#!/usr/bin/env python3
import sys
from pathlib import Path

import gate_hot_path_baselines as hp
import parity_rows as pr
from _rust_source import has_lint_attribute

root = Path(".")
missing = []

# Inventories: allow pending/in_progress; forbid the retired waiver status and
# rows that never closed.
for manifest in [
    "parity/estimate.toml",
    "parity/discovery.toml",
    "parity/bayesian.toml",
    "parity/pag.toml",
    "parity/context.toml",
    "parity/design_state.toml",
    "parity/gcm.toml",
    "parity/attribution.toml",
    "parity/response.toml",
    "parity/compiler.toml",
]:
    for c in pr.rows(manifest):
        if c.get("status") == "intentional_deviation":
            missing.append(f"{manifest}: {c.get('id')} still intentional_deviation (retired)")
        if c.get("status") in ("planned", "blocked"):
            missing.append(f"{manifest}: {c.get('id')} still {c.get('status')}")

# Evidence is a TOML (or code) record. ADR and doc markdown are not evidence.
EVIDENCE = {
    "release.parity_closure": "parity/release.toml",
    "release.graph_dot_json": "crates/antecedent-io-core/src/graph_gml.rs",
    "release.artifact_schema": "crates/antecedent-io-core/src/migrate.rs",
    "release.artifact_mmap_stream_skip": "crates/antecedent-io-core/src/reader.rs",
    "release.wheel_matrix": ".github/workflows/ci.yml",
    "release.conformance_docs": "parity/oracle_closure.toml",
    "release.hot_path_baselines": "benches/baselines/baselines.toml",
    "release.security_review": "deny.toml",
    "release.ci_required_jobs": ".github/workflows/ci.yml",
}

missing += pr.honesty_problems("parity/release.toml", EVIDENCE, all_done=True)

for path in [
    "parity/release.toml",
    "parity/support_axes.toml",
    "parity/support_licensed.toml",
    "parity/support_n_a.toml",
    "parity/support_closed.toml",
    "parity/oracle_closure.toml",
    "benches/baselines/baselines.toml",
    "deny.toml",
    "conformance/interchange/graph_dot_json/expected.json",
    "conformance/interchange/graph_gml_networkx/expected.json",
    "conformance/interchange/artifact_migrate/expected.json",
    "crates/antecedent-io-core/src/graph_dot.rs",
    "crates/antecedent-io-core/src/graph_gml.rs",
    "crates/antecedent-io-core/src/graph_networkx.rs",
    "crates/antecedent-io-core/src/graph_json.rs",
    "crates/antecedent-io-core/src/migrate.rs",
    "crates/antecedent-io/src/model_bundle.rs",
    "scripts/generate_conformance_docs.py",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Semantic crates: forbid unsafe by default. antecedent-data / antecedent-io keep
# #![deny(unsafe_code)] with scoped allows (Arrow FFI / foreign buffers / mmap, which
# lives in antecedent-io-core; antecedent-io itself forbids unsafe).
forbid_crates = [
    "crates/antecedent-core",
    "crates/antecedent-contract",
    "crates/antecedent-graph",
    "crates/antecedent-expr",
    "crates/antecedent-identify",
    "crates/antecedent-io",
    "crates/antecedent-stats",
    "crates/antecedent-prob",
    "crates/antecedent-estimate",
    "crates/antecedent-validate",
    "crates/antecedent-model",
    "crates/antecedent-counterfactual",
    "crates/antecedent-attribution",
    "crates/antecedent-design",
    "crates/antecedent-state",
    "crates/antecedent-discovery",
    "crates/antecedent",
]
deny_escape_crates = {
    "crates/antecedent-data": ("buffer.rs", "arrow_ffi.rs"),
    "crates/antecedent-io-core": ("mmap_file.rs",),
}
for crate in forbid_crates:
    lib = root / crate / "src" / "lib.rs"
    text = lib.read_text(encoding="utf-8")
    if not has_lint_attribute(text, "forbid", "unsafe_code", inner=True):
        missing.append(f"{crate} missing #![forbid(unsafe_code)]")
for crate, allow_mods in deny_escape_crates.items():
    lib = root / crate / "src" / "lib.rs"
    text = lib.read_text(encoding="utf-8")
    if not has_lint_attribute(text, "deny", "unsafe_code", inner=True):
        missing.append(f"{crate} missing #![deny(unsafe_code)] (scoped unsafe escape)")
    if not has_lint_attribute(text, "allow", "unsafe_code") and not any(
        has_lint_attribute(
            (root / crate / "src" / m).read_text(encoding="utf-8"),
            "allow",
            "unsafe_code",
        )
        for m in allow_mods
        if (root / crate / "src" / m).exists()
    ):
        missing.append(f"{crate} missing allow(unsafe_code) for scoped escape modules")
    for mod_name in allow_mods:
        if not (root / crate / "src" / mod_name).exists():
            missing.append(f"{crate} expected unsafe escape module missing: {mod_name}")

# Required baselines are [[baseline]] rows with required=true. The name list
# is the registry, not a second copy here, and not a scrape of hot_paths.md.
missing += hp.registry_problems()

if missing:
    print("Release gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Release inventory / artifact evidence map OK")
