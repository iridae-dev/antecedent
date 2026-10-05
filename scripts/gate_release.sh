#!/usr/bin/env bash
# PR inventory / composition umbrella (also run in CI on every PR via `gates`).
#
# This script is not a release candidate. Cut with
# `scripts/gate_release_candidate.sh` (a green CI run on the SHA via CI_RUN_ID,
# calibration-surface identity, this inventory, Python lint/pytest, the Python
# suite against one locally built wheel). The CI `python-wheels` matrix is
# required on the candidate SHA through CI_RUN_ID.
#
# `gate_calibration.sh` is NOT invoked from here, nor anywhere in CI: the
# statistical measurement is made on a development machine before upload
# (scripts/measure_calibration.sh). This gate, on every PR, push and release
# cut alike, requires every coverage record to match the code it was measured
# at (gate_calibration_attestation.sh, a git comparison that runs in seconds).
#   CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh
#
# Invokes prior feature gates unless SKIP_PRIOR_GATES=1 (a local shortcut; the
# release-candidate gate refuses it).
# cargo deny check runs when cargo-deny is on PATH and is mandatory with
# REQUIRE_CARGO_DENY=1 (set by the release-candidate gate; CI has its own job).
# Every Python smoke requires `uv` and fails rather than skips, unless
# ALLOW_SKIP_PYTHON_SMOKE=1 is given for a local run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# CI runs these independent sections in parallel. With no argument the local
# release-candidate gate still runs every check in this file.
section=all
if [[ "${1:-}" == "--section" && $# -eq 2 ]]; then
  section="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--section inventory|promotion|support|matrix|checked|features|benches]" >&2
  exit 2
fi
case "$section" in all|inventory|promotion|support|matrix|checked|features|benches) ;; *)
  echo "unknown release-gate section: $section" >&2; exit 2 ;;
esac
section_start() {
  printf -v "section_since_$1" '%s' "$SECONDS"
  echo "== $1 started $(date -u +%FT%TZ) =="
}
section_end() {
  local since="section_since_$1"
  echo "== $1 finished in $((SECONDS - ${!since}))s =="
}

if [[ "$section" == all || "$section" == inventory ]]; then
section_start inventory

# Runs unconditionally: the inventory honesty pass below reads `status` with a
# regex default, so a row missing the key would sail through it. SKIP_PRIOR_GATES
# must not skip the schema contract that pass depends on.
echo "== parity manifest schema =="
bash scripts/gate_parity_schema.sh

# The gates' self-tests (each gate must fail on deliberately broken input) are not
# run here: every case rebuilds a repo overlay and runs a whole gate, which roughly
# doubled this script's runtime. They run on demand: scripts/gate_selftests.sh.

echo "== algorithm provenance schema and paths =="
bash scripts/gate_provenance_schema.sh

# Also unconditional: cross-file metadata drift is exactly the class of failure
# that survives when a check is skippable.
echo "== cross-file metadata consistency =="
bash scripts/gate_metadata_consistency.sh

echo "== hot-path baseline metadata =="
bash scripts/gate_hot_path_baselines.sh
fi

if [[ "$section" == all || "$section" == promotion ]]; then
section_start promotion
# 2.2 cells stay closed until their frozen promotion record carries executed
# positive, negative and artifact evidence (parity/promotion_2_2.toml).
echo "== 2.2 promotion records =="
bash scripts/gate_promotion.sh
section_end promotion
fi

if [[ "$section" == all || "$section" == support || "$section" == matrix ]]; then
section_start matrix
echo "== public support matrix =="
bash scripts/gate_support_matrix.sh
section_end matrix
fi

if [[ "$section" == all || "$section" == support || "$section" == checked ]]; then
section_start checked

# Every licensed cell executes each licensed estimator from a retained checked
# operation with no builder alive: the checked_execution citations on
# parity/support_licensed.toml run here, grouped by test target.
echo "== checked execution evidence =="
bash scripts/gate_checked_execution.sh
section_end checked
fi

if [[ "$section" == all || "$section" == inventory ]]; then
echo "== docs vs support matrix =="
bash scripts/gate_docs_support_matrix.sh

echo "== published docs links resolve =="
python3 scripts/check_doc_links.py

# 2.2 release closure: declared bounds agree across records, docs, Rust constants and Python
# docstrings; the release notes describe each cell with exactly its record's claim word.
echo "== 2.2 limits agreement (records, docs, Rust constants, Python) =="
python3 scripts/check_limits_agreement.py

echo "== 2.2 release notes claim vocabulary =="
python3 scripts/check_release_claims.py

# The calibration inventories are generated; a stale checked-in copy fails here (B8 gap closed).
echo "== calibration backlog and readiness inventories are current =="
python3 scripts/generate_calibration_backlog.py --check
python3 scripts/calibration_readiness.py --check

echo "== lint allows: no file-wide allow of lossy casts or exact float compares in library code =="
bash scripts/gate_lint_allows.sh

echo "== evidence reachability (cited fixtures execute; deviations ratchet) =="
bash scripts/gate_evidence_reachability.sh

echo "== coverage citations name existing test fns; coverage figures cite records =="
bash scripts/gate_coverage_citations.sh

echo "== calibration attestation (every coverage record matches the code) =="
bash scripts/gate_calibration_attestation.sh
fi

if [[ "$section" == all || "$section" == features ]]; then
section_start features
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
  # gate_response_calibration.sh and gate_estimate_reuse.sh only re-run tests
  # from `cargo test --workspace`; scripts/gate_named_tests.sh in the Rust job
  # checks their filters still select tests. The calibration script stays on
  # the measurement surface; this job does not invoke it.
  bash scripts/gate_causal_artifacts.sh
  bash scripts/gate_composition.sh
  bash scripts/gate_transport.sh
  # 2.2 A exit gate: six end-to-end stories (Rust and Python) plus the zero-unmeasured-interval
  # check. PENDING_CALIBRATION (story 3 / X1 coverage records unmeasured) exits 0 here; the
  # release-candidate cut can add --require-calibrated.
  echo "== 2.2 A exit gate =="
  bash scripts/gate_a_exit.sh
  # 2.2 B exit gate scaffold: per-package table (PENDING_IMPLEMENTATION until a package registers
  # its stories in scripts/b_exit_report.py) plus the interval check over every 2.2 record. The
  # cut runs it with --release.
  echo "== 2.2 B exit gate =="
  bash scripts/gate_b_exit.sh
fi
section_end features
fi

if [[ "$section" == all || "$section" == inventory ]]; then

python3 - <<'PY'
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
VERSION="$(python3 -c "import tomllib; from pathlib import Path; p=Path('docs/release-notes/preparation.toml'); d=tomllib.load(open(p,'rb')) if p.is_file() else tomllib.load(open('Cargo.toml','rb'))['workspace']['package']; print(d.get('target_version', d.get('version')))" )"
if ! git diff --exit-code -- docs/support-matrix.md \
    crates/antecedent/src/support_matrix_data.rs \
    crates/antecedent-io/src/coverage_records_data.rs \
    "docs/release-notes/v${VERSION}.md" >/dev/null; then
  echo "support-matrix generated files are stale; commit regenerated output"
  git diff --stat -- docs/support-matrix.md crates/antecedent/src/support_matrix_data.rs \
    crates/antecedent-io/src/coverage_records_data.rs \
    "docs/release-notes/v${VERSION}.md"
  exit 1
fi
if ! git diff --exit-code -- docs/release-notes/ \
    ":!docs/release-notes/v${VERSION}.md" >/dev/null; then
  echo "generator rewrote historical release notes; freeze those licensed blocks"
  git diff --stat -- docs/release-notes/ ":!docs/release-notes/v${VERSION}.md"
  exit 1
fi

fi

if [[ "$section" == all || "$section" == benches ]]; then
section_start benches
# One cargo invocation, so a bench added to a manifest cannot ship unexecuted.
# Pull requests skip it unless they touch a bench file or a [[bench]] entry;
# main and a local release run always smoke. The extension module and the
# optional Burn provider have no bench targets.
run_criterion_smoke=1
if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
  base="${PR_BASE_SHA:-}"
  if [[ -z "$base" ]]; then
    echo "pull request base SHA is missing; running the criterion smoke" >&2
  elif ! changed_files="$(git diff --name-only "$base" HEAD)" \
    || ! manifest_diff="$(git diff -U0 "$base" HEAD -- '*Cargo.toml')"; then
    echo "git diff against the pull request base failed; running the criterion smoke" >&2
  elif grep -Eq '(^|/)benches/' <<<"$changed_files" \
    || grep -Eq '^[+-]\[\[bench\]\]' <<<"$manifest_diff"; then
    echo "== criterion smoke (bench files changed) =="
  else
    echo "== criterion smoke skipped (no bench file or [[bench]] change; main runs it) =="
    run_criterion_smoke=0
  fi
else
  echo "== criterion smoke (every bench target of the workspace) =="
fi
if [[ "$run_criterion_smoke" -eq 1 ]]; then
  bench_targets="$(python3 scripts/bench_targets.py)"
  if awk '{print $1}' <<<"$bench_targets" | grep -Eq '^(antecedent-py|antecedent-learn-burn)$'; then
    echo "criterion smoke excludes a package that now has a bench target" >&2
    exit 1
  fi
  # `--bench '*'` selects bench targets only: plain `cargo bench` and `--benches`
  # also run every library's unit tests under the bench profile (lib is bench = true).
  cargo bench --workspace --exclude antecedent-py --exclude antecedent-learn-burn \
    --bench '*' -- --test
fi
section_end benches
fi

if [[ "$section" == all || "$section" == inventory ]]; then

if command -v cargo-deny >/dev/null 2>&1; then
  echo "== cargo deny check =="
  cargo deny check
elif [[ "${REQUIRE_CARGO_DENY:-0}" == "1" ]]; then
  echo "FAIL: cargo-deny is required (REQUIRE_CARGO_DENY=1) and is not installed" >&2
  exit 1
else
  echo "WARN: cargo-deny not installed; skipping deny check (the CI deny job and the RC gate enforce it)."
fi

echo "PR inventory / composition gate PASSED (not an RC)."
echo "Cut with: CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh"
section_end inventory
fi
