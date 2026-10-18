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
  echo "usage: $0 [--section inventory|promotion|promotion-python|promotion-facade-0|promotion-facade-1|promotion-estimate|promotion-identify|promotion-other|support|graphless|matrix|matrix-0..7|transport-contracts|checked|checked-0|checked-1|checked-2|checked-3|features|feature-core|feature-composition|feature-transport|feature-conformance|feature-exits|benches]" >&2
  exit 2
fi
case "$section" in all|inventory|promotion|promotion-python|promotion-facade-0|promotion-facade-1|promotion-estimate|promotion-identify|promotion-other|support|graphless|matrix|matrix-[0-7]|transport-contracts|checked|checked-0|checked-1|checked-2|checked-3|features|feature-core|feature-composition|feature-transport|feature-conformance|feature-exits|benches) ;; *)
  echo "unknown release-gate section: $section" >&2; exit 2 ;;
esac
section_start() {
  printf -v "section_since_${1//-/_}" '%s' "$SECONDS"
  echo "== $1 started $(date -u +%FT%TZ) =="
}
section_end() {
  local since="section_since_${1//-/_}"
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

echo "== promotion evidence inventory =="
python3 scripts/release_evidence_report.py

echo "== inherited calibration debt and current inference contract inventory =="
python3 scripts/generate_calibration_plan.py --check

echo "== hot-path baseline metadata =="
bash scripts/gate_hot_path_baselines.sh

# Every published crate's public surface matches parity/public_api/*.txt, so an
# API change is a reviewed snapshot diff, not a side effect.
echo "== public API snapshots =="
bash scripts/gate_public_api.sh
fi

if [[ "$section" == all || "$section" == promotion ]]; then
section_start promotion
# Frozen or carried-forward cells remain closed until their cited evidence executes.
echo "== promotion records =="
bash scripts/gate_promotion.sh
section_end promotion
fi

if [[ "$section" == promotion-* ]]; then
section_start "$section"
bash scripts/gate_promotion.sh --evidence-group "${section#promotion-}"
section_end "$section"
fi

if [[ "$section" == all || "$section" == support || "$section" == graphless ]]; then
section_start graphless
echo "== graphless support evidence =="
bash scripts/gate_support_matrix.sh --graphless-only
section_end graphless
fi

if [[ "$section" == all || "$section" == support || "$section" == matrix ]]; then
section_start matrix
echo "== public support matrix =="
bash scripts/gate_support_matrix.sh --matrix-only
section_end matrix
fi

if [[ "$section" == matrix-* ]]; then
section_start "$section"
echo "== public support matrix cited tests ${section#matrix-}/8 =="
bash scripts/gate_support_matrix.sh "--matrix-shard-${section#matrix-}"
section_end "$section"
fi

if [[ "$section" == all || "$section" == support || "$section" == transport-contracts ]]; then
section_start transport_contracts
echo "== transport support contracts =="
bash scripts/gate_support_matrix.sh --transport-only
section_end transport_contracts
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

if [[ "$section" == checked-* ]]; then
section_start "$section"
echo "== checked execution evidence ${section#checked-}/4 =="
bash scripts/gate_checked_execution.sh --shard "${section#checked-}/4"
section_end "$section"
fi

if [[ "$section" == all || "$section" == inventory ]]; then
echo "== docs vs support matrix =="
bash scripts/gate_docs_support_matrix.sh

echo "== acceptance manifest: every case bound to a live test =="
python3 scripts/check_acceptance_manifest.py --require-bound
echo "== published docs links resolve =="
python3 scripts/check_doc_links.py

# Release closure: declared bounds agree across records, docs, Rust constants and Python
# docstrings; the release notes describe each cell with exactly its record's claim word.
echo "== limits agreement (records, docs, Rust constants, Python) =="
python3 scripts/check_limits_agreement.py

echo "== interval ownership and measurement wiring =="
python3 scripts/check_interval_coordinates.py

echo "== release notes claim vocabulary =="
python3 scripts/check_release_claims.py

# The calibration inventories are generated; a stale checked-in copy fails here.
echo "== calibration backlog and readiness inventories are current =="
python3 scripts/generate_calibration_backlog.py --check
python3 scripts/calibration_readiness.py --check

echo "== conformance reference generators =="
python3 scripts/check_conformance_references.py

echo "== lint allows: no file-wide allow of lossy casts or exact float compares in library code =="
bash scripts/gate_lint_allows.sh

echo "== evidence reachability (cited fixtures execute; deviations ratchet) =="
bash scripts/gate_evidence_reachability.sh

echo "== coverage citations name existing test fns; coverage figures cite records =="
bash scripts/gate_coverage_citations.sh

echo "== calibration attestation (every coverage record matches the code) =="
bash scripts/gate_calibration_attestation.sh
fi

if [[ "$section" == all || "$section" == features || "$section" == feature-core ]]; then
section_start feature_core
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
fi
section_end feature_core
fi

if [[ "$section" == all || "$section" == features || "$section" == feature-composition ]]; then
section_start feature_composition
if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  if [[ "$section" == feature-composition ]]; then
    # Inventory already runs the common schema and reachability checks in CI.
    bash scripts/gate_composition.sh --evidence-only
  else
    bash scripts/gate_composition.sh
  fi
fi
section_end feature_composition
fi

if [[ "$section" == all || "$section" == features || "$section" == feature-transport ]]; then
section_start feature_transport
if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  bash scripts/gate_transport.sh
fi
section_end feature_transport
fi

if [[ "$section" == all || "$section" == features || "$section" == feature-conformance ]]; then
section_start feature_conformance
if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  # Conformance of programmable interventions, mechanism selection, graph falsification, noise
  # replay and intrinsic influence. The benches section smokes every bench target, including the
  # three this gate registers, so the tests run here without repeating them.
  bash scripts/gate_gcm_conformance.sh --tests-only
fi
section_end feature_conformance
fi

if [[ "$section" == all || "$section" == features || "$section" == feature-exits ]]; then
section_start feature_exits
if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  # Composition exit: six end-to-end stories (Rust and Python) plus the zero-unmeasured-interval
  # check. PENDING_CALIBRATION (story 3 / multi-source coverage records unmeasured) exits 0 here;
  # the release-candidate gate adds --require-calibrated.
  echo "== composition exit stories =="
  bash scripts/gate_composition_exit.sh
  # Decision exit scaffold: per-story table (PENDING_IMPLEMENTATION until a story registers its
  # tests in scripts/decision_exit_report.py) plus the interval check over every record. The
  # release-candidate gate runs it with --release.
  echo "== decision exit stories =="
  bash scripts/gate_decision_exit.sh
fi
section_end feature_exits
fi

if [[ "$section" == all || "$section" == inventory ]]; then

python3 "$ROOT/scripts/gate_release.py"

echo "== regenerate conformance docs (must be clean) =="
python3 scripts/generate_conformance_docs.py
if ! git diff --exit-code -- docs/conformance >/dev/null; then
  echo "docs/conformance is stale; commit regenerated output"
  git diff --stat -- docs/conformance
  exit 1
fi

echo "== regenerate support-matrix docs (must be clean) =="
python3 scripts/generate_support_matrix_docs.py
VERSION="$(python3 "$ROOT/scripts/release_notes_version.py")"
if ! git diff --exit-code -- docs/support-matrix.md \
    crates/antecedent/data/support_matrix.json \
    crates/antecedent-io-core/data/coverage_records.json \
    "docs/release-notes/v${VERSION}.md" >/dev/null; then
  echo "support-matrix generated files are stale; commit regenerated output"
  git diff --stat -- docs/support-matrix.md crates/antecedent/data/support_matrix.json \
    crates/antecedent-io-core/data/coverage_records.json \
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
