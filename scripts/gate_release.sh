#!/usr/bin/env bash
# Release gate (also run in CI on every PR via the `gates` job).
#
# Inventory honesty, docs, artifacts, security, Criterion smokes, and prior
# feature gates. CI's `gates` job runs this on every PR; the separate `rust` job
# runs fmt + clippy + cargo test --workspace (+ DCO). `gate_calibration.sh` is
# NOT invoked from here — it runs weekly via .github/workflows/calibration.yml.
# Run this when cutting a release or when a change might break a domain:
#   bash scripts/gate_release.sh
#
# Invokes prior feature gates unless SKIP_PRIOR_GATES=1.
# Optional: cargo deny check when cargo-deny is on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Runs unconditionally: the inventory honesty pass below reads `status` with a
# regex default, so a row missing the key would sail through it. SKIP_PRIOR_GATES
# must not skip the schema contract that pass depends on.
echo "== parity manifest schema =="
bash scripts/gate_parity_schema.sh

echo "== algorithm provenance schema and paths =="
bash scripts/gate_provenance_schema.sh

# Also unconditional: cross-file metadata drift is exactly the class of failure
# that survives when a check is skippable.
echo "== cross-file metadata consistency =="
bash scripts/gate_metadata_consistency.sh

echo "== hot-path baseline metadata =="
bash scripts/gate_hot_path_baselines.sh

echo "== public support matrix =="
bash scripts/gate_support_matrix.sh

echo "== docs vs support matrix =="
bash scripts/gate_docs_support_matrix.sh

echo "== evidence reachability (cited fixtures execute; deviations ratchet) =="
bash scripts/gate_evidence_reachability.sh

if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  echo "== prior feature gates =="
  bash scripts/gate_estimate_ci.sh
  bash scripts/gate_bayesian.sh
  bash scripts/gate_gcm.sh
  bash scripts/gate_pag.sh
  bash scripts/gate_context.sh
  bash scripts/gate_attribution.sh
  bash scripts/gate_design_state.sh
  bash scripts/gate_upstream_names.sh
  bash scripts/gate_response_calibration.sh
  bash scripts/gate_causal_artifacts.sh
  bash scripts/gate_estimate_reuse.sh
fi

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")

def caps(path: Path):
    text = path.read_text()
    blocks = re.split(r"\n\[\[capabilities\]\]\n", text)[1:]
    out = []
    for b in blocks:
        def g(k, default=None):
            m = re.search(rf'^{k}\s*=\s*"([^"]*)"', b, re.M)
            if m:
                return m.group(1)
            m = re.search(rf'^{k}\s*=\s*(\d+)', b, re.M)
            return m.group(1) if m else default
        out.append({"id": g("id"), "status": g("status")})
    return out

missing = []

# Inventories: allow pending/in_progress; forbid retired waiver status.
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
]:
    for c in caps(Path(manifest)):
        if c["status"] == "intentional_deviation":
            missing.append(f"{manifest}: {c['id']} still intentional_deviation (retired)")
        if c["status"] in ("planned", "blocked"):
            missing.append(f"{manifest}: {c['id']} still {c['status']}")

EVIDENCE = {
    "release.parity_closure": "parity/README.md",
    "release.graph_dot_json": "crates/antecedent-io/src/graph_gml.rs",
    "release.artifact_schema": "crates/antecedent-io/src/migrate.rs",
    "release.artifact_mmap_stream_skip": "crates/antecedent-io/src/reader.rs",
    "release.wheel_matrix": ".github/workflows/ci.yml",
    "release.conformance_docs": "docs/conformance/README.md",
    "release.hot_path_baselines": "docs/hot_paths.md",
    "release.security_review": "docs/security_review.md",
}

for c in caps(Path("parity/release.toml")):
    if c["status"] != "done":
        missing.append(f"release.toml {c['id']} status={c['status']}")
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} has no evidence mapping")
    elif not (root / ev).exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

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

# Baseline files referenced by hot_paths index.
hot = (root / "docs/hot_paths.md").read_text()
for base in (root / "benches/baselines").glob("*.md"):
    if base.name not in hot and "baselines/" + base.name not in hot:
        # Allow baselines not linked if mentioned via relative link path
        if f"baselines/{base.name}" not in hot and base.name.replace(".md", "") not in hot:
            pass  # soft: index is curated; require at least the release-listed set below

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
PY

echo "== regenerate conformance docs (must be clean) =="
python3 scripts/generate_conformance_docs.py
if ! git diff --exit-code -- docs/conformance >/dev/null; then
  echo "docs/conformance is stale; commit regenerated output"
  git diff --stat -- docs/conformance
  exit 1
fi

echo "== regenerate support-matrix docs (must be clean) =="
python3 scripts/generate_support_matrix_docs.py
VERSION="$(python3 -c "import tomllib; print(tomllib.load(open('Cargo.toml','rb'))['workspace']['package']['version'])")"
if ! git diff --exit-code -- docs/support-matrix.md \
    crates/antecedent/src/support_matrix_data.rs \
    "docs/release-notes/v${VERSION}.md" >/dev/null; then
  echo "support-matrix generated files are stale; commit regenerated output"
  git diff --stat -- docs/support-matrix.md crates/antecedent/src/support_matrix_data.rs \
    "docs/release-notes/v${VERSION}.md"
  exit 1
fi
if ! git diff --exit-code -- docs/release-notes/ \
    ":!docs/release-notes/v${VERSION}.md" >/dev/null; then
  echo "generator rewrote historical release notes; freeze those licensed blocks"
  git diff --stat -- docs/release-notes/ ":!docs/release-notes/v${VERSION}.md"
  exit 1
fi

echo "== cargo test release surfaces =="
cargo test -p antecedent-io --lib
cargo test -p antecedent --test graph_interchange
cargo test -p antecedent --test artifact_migrate

echo "== criterion smoke (designated hot paths) =="
cargo bench -p antecedent-kernels --bench gather -- --test
cargo bench -p antecedent-kernels --bench reductions -- --test
cargo bench -p antecedent-graph --bench traversal -- --test
cargo bench -p antecedent-graph --bench dseparation -- --test
cargo bench -p antecedent-identify --bench adjustment -- --test
cargo bench -p antecedent-kernels --bench partial_correlation -- --test
cargo bench -p antecedent-discovery --bench pcmci -- --test
cargo bench -p antecedent-design --bench design_rank -- --test
cargo bench -p antecedent-state --bench state_append -- --test
cargo bench -p antecedent-estimate --bench response_interference -- --test
cargo bench -p antecedent-estimate --bench temporal_response -- --test

if command -v cargo-deny >/dev/null 2>&1; then
  echo "== cargo deny check =="
  cargo deny check
else
  echo "WARN: cargo-deny not installed; skipping deny check (optional local tool)."
fi

echo "Release gate PASSED"
