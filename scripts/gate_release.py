#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

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

EVIDENCE = {
    "release.parity_closure": "parity/README.md",
    "release.graph_dot_json": "crates/antecedent-io/src/graph_gml.rs",
    "release.artifact_schema": "crates/antecedent-io/src/migrate.rs",
    "release.artifact_mmap_stream_skip": "crates/antecedent-io/src/reader.rs",
    "release.wheel_matrix": ".github/workflows/ci.yml",
    "release.conformance_docs": "docs/conformance/README.md",
    "release.hot_path_baselines": "docs/hot_paths.md",
    "release.security_review": "docs/security_review.md",
    "release.ci_required_jobs": ".github/workflows/ci.yml",
}

missing += pr.honesty_problems("parity/release.toml", EVIDENCE, all_done=True)

for path in [
    "adr/0017-release-prep.md",
    "parity/release.toml",
    "parity/README.md",
    "docs/artifacts.md",
    "docs/hot_paths.md",
    "docs/security_review.md",
    "docs/conformance/README.md",
    "docs/support-matrix.md",
    "parity/support_axes.toml",
    "parity/support_licensed.toml",
    "parity/support_n_a.toml",
    "parity/support_closed.toml",
    "adr/0020-support-matrix-and-prepared-workflow.md",
    "deny.toml",
    "conformance/interchange/graph_dot_json/expected.json",
    "conformance/interchange/graph_gml_networkx/expected.json",
    "conformance/interchange/artifact_migrate/expected.json",
    "crates/antecedent-io/src/graph_dot.rs",
    "crates/antecedent-io/src/graph_gml.rs",
    "crates/antecedent-io/src/graph_networkx.rs",
    "crates/antecedent-io/src/graph_json.rs",
    "crates/antecedent-io/src/migrate.rs",
    "crates/antecedent-io/src/model_bundle.rs",
    "scripts/generate_conformance_docs.py",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Semantic crates: forbid unsafe by default. antecedent-data / antecedent-io keep
# #![deny(unsafe_code)] with scoped allows (Arrow FFI / foreign buffers / mmap).
forbid_crates = [
    "crates/antecedent-core",
    "crates/antecedent-graph",
    "crates/antecedent-expr",
    "crates/antecedent-identify",
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
    "crates/antecedent-io": ("mmap_file.rs",),
}
for crate in forbid_crates:
    lib = root / crate / "src" / "lib.rs"
    text = lib.read_text()
    if "#![forbid(unsafe_code)]" not in text:
        missing.append(f"{crate} missing #![forbid(unsafe_code)]")
for crate, allow_mods in deny_escape_crates.items():
    lib = root / crate / "src" / "lib.rs"
    text = lib.read_text()
    if "#![deny(unsafe_code)]" not in text:
        missing.append(f"{crate} missing #![deny(unsafe_code)] (scoped unsafe escape)")
    if "allow(unsafe_code)" not in text and not any(
        "allow(unsafe_code)" in (root / crate / "src" / m).read_text()
        for m in allow_mods
        if (root / crate / "src" / m).exists()
    ):
        missing.append(f"{crate} missing allow(unsafe_code) for scoped escape modules")
    for mod_name in allow_mods:
        if not (root / crate / "src" / mod_name).exists():
            missing.append(f"{crate} expected unsafe escape module missing: {mod_name}")

# Baselines the release requires, and that docs/hot_paths.md must reference.
hot = (root / "docs/hot_paths.md").read_text()

required_baselines = [
    "gather.md",
    "kernel_reductions.md",
    "graph_traversal.md",
    "dseparation.md",
    "adjustment.md",
    "partial_correlation.md",
    "pcmci.md",
    "ci_orientation.md",
    "propensity.md",
    "matching.md",
    "pag.md",
    "regime_mediation.md",
    "shapley.md",
    "design_state.md",
    "response_interference.md",
    "laplace_glm.md",
    "hmc.md",
    "mcmc_stats.md",
    "sample_overlay.md",
    "counterfactual_batch.md",
    "posterior_functional.md",
]
for name in required_baselines:
    if not (root / "benches/baselines" / name).exists():
        missing.append(f"missing baseline {name}")
    if name not in hot:
        missing.append(f"docs/hot_paths.md does not reference {name}")

if missing:
    print("Release gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Release inventory / artifact evidence map OK")
