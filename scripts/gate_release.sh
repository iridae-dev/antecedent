#!/usr/bin/env bash
# The pull-request gate, in the order CI runs it. Cheap checks first, so a failure
# is reported before an expensive step starts:
#
#   tier 0  static         source reading only, seconds, no build        (section: static)
#   tier 1  builds         compile, list or lint; run no test            (sections: public-api,
#                                                                         cited-tests, benches, deny)
#   tier 2  engineering    tests of the software, no scientific claim    (sections: python-lint,
#                                                                         engineering, development)
#   tier 3  records        every scientific and calibration record is   (section: scientific)
#                          current; nothing is executed
#
# Scientific assertions and calibration are earned on development machines
# (bash scripts/test.sh), rerunning only the records that are owed, and committed. This
# gate never executes either lane: it checks that every record is current
# (scripts/scientific_records.py status --check, scripts/calibration_status.py --check),
# in the local static gate or the dedicated CI job. An owed record fails the change.
#
# Tier 4 is the release candidate (scripts/gate_release_candidate.sh): this gate, then the
# CI-run check, scale cases and the installed wheel. It requires the same current records.
#
#   bash scripts/gate_release.sh                    # every section, in tier order
#   bash scripts/gate_release.sh --section NAME     # one section (CI runs them as parallel jobs)
#
# cargo deny check runs when cargo-deny is on PATH and is mandatory with
# REQUIRE_CARGO_DENY=1 (set by the release-candidate gate; CI has its own job).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SECTIONS="static ci-static public-api cited-tests cited-tests-0..7 benches deny python-lint engineering development scientific"
section=all
if [[ "${1:-}" == "--section" && $# -eq 2 ]]; then
  section="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--section ${SECTIONS// /|}]" >&2
  exit 2
fi
case "$section" in all|static|ci-static|public-api|cited-tests|cited-tests-[0-7]|benches|deny|python-lint|engineering|development|scientific) ;; *)
  echo "unknown gate section: $section" >&2; exit 2 ;;
esac
wanted() { [[ "$section" == all || "$section" == "$1" ]]; }
section_since=0
section_start() { section_since=$SECONDS; echo "== $1 started $(date -u +%FT%TZ) =="; }
section_end() { echo "== $1 finished in $((SECONDS - section_since))s =="; }
# Static checks run to the end and report every failure together, so one run lists them all.
failed_steps=()
parallel_steps=0
step_pids=()
step_names=()
next_step=0
active_steps=()
finish_step() {
  local id="$1"
  if ! wait "${step_pids[$id]}"; then
    failed_steps+=("${step_names[$id]}")
  fi
  cat "$step_logs/$id"
}
start_readers() {
  step_logs="$(mktemp -d)"
  trap 'rm -rf "$step_logs"' EXIT
  parallel_steps="$1"
}
finish_readers() {
  local id
  for id in "${active_steps[@]}"; do finish_step "$id"; done
  parallel_steps=0
}
step() {
  if (( ! parallel_steps )); then
    "$@" || failed_steps+=("$*")
    return
  fi
  local id
  local pending=()
  while (( ${#active_steps[@]} >= parallel_steps )); do
    pending=()
    for id in "${active_steps[@]}"; do
      if kill -0 "${step_pids[$id]}" 2>/dev/null; then
        pending+=("$id")
      else
        finish_step "$id"
      fi
    done
    active_steps=(${pending[@]+"${pending[@]}"})
    if (( ${#active_steps[@]} >= parallel_steps )); then sleep 0.2; fi
  done
  "$@" > "$step_logs/$next_step" 2>&1 &
  step_pids[$next_step]=$!
  active_steps+=("$next_step")
  step_names[$next_step]="$*"
  next_step=$((next_step + 1))
}


check_records() {
  if [[ -n "${ANTECEDENT_RC_RECEIPTS:-}" ]]; then
    step python3 scripts/gate_support.py --check-scientific --write-current "$ANTECEDENT_RC_RECEIPTS/current.json"
  else
    step python3 scripts/gate_support.py --check-scientific
  fi
  step python3 scripts/calibration_status.py --check --check-support-docs
}

# ---------------------------------------------------------------------------------------
# Tier 0: static. One job, no build, no test execution.
# ---------------------------------------------------------------------------------------
if wanted static || wanted ci-static; then
section_start static
start_readers "${ANTECEDENT_STATIC_JOBS:-3}"
case "$parallel_steps" in 1|2|3) ;; *) echo "ANTECEDENT_STATIC_JOBS must be 1, 2, or 3" >&2; exit 2 ;; esac

echo "== format, lint allows, test targets, crate ownership =="
step bash scripts/gate_rustfmt.sh
step bash scripts/gate_lint_allows.sh
step python3 scripts/check_test_targets.py
step python3 scripts/check_architecture.py
step python3 scripts/test_check_architecture.py
step python3 scripts/check_rust_ci_feature_parity.py

echo "== test registry, lanes and ignored tests =="
step python3 scripts/check_test_registry.py
step python3 scripts/evidence_audit.py rust-engineering-filter --check
step python3 scripts/check_ignored_tests.py
step python3 scripts/check_child_test_selectors.py

echo "== parity schema, inventories and provenance =="
step bash scripts/gate_parity_schema.sh
step python3 scripts/check_parity_inventories.py
step bash scripts/gate_provenance_schema.sh
step python3 scripts/assurance.py check
step python3 scripts/assurance.py --self-test
step python3 scripts/operation_records.py check
step python3 scripts/adversarial.py check
step python3 scripts/check_retained_artifact_loaders.py
step bash scripts/gate_metadata_consistency.sh
step bash scripts/gate_evidence_reachability.sh
step bash scripts/gate_upstream_names.sh
step bash scripts/gate_hot_path_baselines.sh
step python3 scripts/release_evidence_report.py

echo "== capability records, composition, transport, checked execution, conformance =="
step bash scripts/gate_capability_records.sh
step bash scripts/gate_composition.sh --evidence-only
step bash scripts/gate_transport.sh
step bash scripts/gate_checked_execution.sh
step python3 scripts/gate_compiled_path.py
step python3 scripts/test_gate_compiled_path.py
step python3 scripts/gate_unit_claims.py
step python3 scripts/gate_unit_claims.py --self-test

echo "== evidence records: execution links, unit index, closure and monotonicity (ADR 0031) =="
step python3 scripts/unit_index.py --check
step python3 scripts/gate_evidence_closure.py
step python3 scripts/gate_evidence_closure.py --self-test
step bash scripts/gate_gcm_conformance.sh --tests-only

echo "== support: the one count, the matrix, the graphless supports, the transport stages =="
if [[ "$section" != ci-static ]]; then check_records; fi
step python3 scripts/gate_not_supported_sites.py
step bash scripts/gate_docs_support_matrix.sh

echo "== method children, method specs and acceptance =="
step python3 scripts/check_method_children.py
step python3 scripts/reconcile_method_specs.py --check
step python3 scripts/check_acceptance_manifest.py --require-bound
step python3 scripts/check_missing_policy_support.py

echo "== bounds, release claims, links, conformance references =="
step python3 scripts/generate_reason_details.py --check
step python3 scripts/gate_reason_detail_literals.py
step python3 scripts/gate_python_bindings_use_facade.py
if [[ "$section" != ci-static && -n "${ANTECEDENT_RC_RECEIPTS:-}" ]]; then
  step python3 scripts/check_limits_agreement.py --strict
else
  step python3 scripts/check_limits_agreement.py
fi
step python3 scripts/check_release_claims.py
step python3 scripts/check_doc_links.py
step python3 scripts/check_conformance_references.py

echo "== generated files are current =="
step python3 scripts/generate_conformance_docs.py --check
step python3 scripts/generate_host_schema.py --check
step python3 scripts/generate_supported_route_records.py --check
step python3 scripts/support_functional_graph_posterior_rows.py --check
step python3 scripts/support_mixed_graph_functional_rows.py --check
step python3 scripts/support_distribution_validation_rows.py --check
step python3 scripts/support_temporal_window_rows.py --check

finish_readers
if (( ${#failed_steps[@]} )); then
  echo "== static: ${#failed_steps[@]} check(s) failed ==" >&2
  printf '  FAILED: %s\n' "${failed_steps[@]}" >&2
  exit 1
fi
section_end static
fi

# ---------------------------------------------------------------------------------------
# Tier 1: builds. Compile, list or lint; no test runs.
# ---------------------------------------------------------------------------------------
if wanted public-api; then
section_start public-api
# Every published crate's public surface matches parity/public_api/*.txt, so an
# API change is a reviewed snapshot diff, not a side effect.
bash scripts/gate_public_api.sh
section_end public-api
fi

# Every cited test of the support matrix, the graphless supports and the transport stages is
# listed by cargo as a compiled, non-ignored test. `cited-tests` is the unsplit form; CI runs
# the eight shards in parallel.
if wanted cited-tests; then
section_start cited-tests
python3 scripts/gate_support.py --cited-tests
section_end cited-tests
fi
if [[ "$section" == cited-tests-* ]]; then
section_start "$section"
python3 scripts/gate_support.py --cited-tests --matrix-shard "${section#cited-tests-}"
section_end "$section"
fi

if wanted benches; then
section_start benches
# One cargo invocation, so a bench added to a manifest cannot ship unexecuted.
# Pull requests skip it unless they touch a bench file or a [[bench]] entry;
# main and a local run always smoke. The extension module and the optional Burn
# provider have no bench targets.
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
  # The conformance benchmarks run in Criterion's smoke mode, which executes their in-process
  # regression and reuse assertions.
  while IFS=' ' read -r package bench; do
    cargo bench -p "$package" --bench "$bench" -- --test
  done < <(python3 - <<'PY'
import tomllib

for row in tomllib.load(open("parity/gcm_conformance.toml", "rb"))["performance"]:
    print(row["package"], row["bench"])
PY
  )
fi
section_end benches
fi

if wanted deny; then
section_start deny
if command -v cargo-deny >/dev/null 2>&1; then
  cargo deny check
elif [[ "${REQUIRE_CARGO_DENY:-0}" == "1" ]]; then
  echo "FAIL: cargo-deny is required (REQUIRE_CARGO_DENY=1) and is not installed" >&2
  exit 1
else
  echo "WARN: cargo-deny not installed; skipping deny check (the CI deny job and the release-candidate gate enforce it)."
fi
section_end deny
fi

# ---------------------------------------------------------------------------------------
# Tier 2: engineering tests. No scientific assertion, no calibration.
# ---------------------------------------------------------------------------------------
if wanted python-lint; then
section_start python-lint
bash scripts/gate_python_lint.sh
section_end python-lint
fi

if wanted engineering; then
section_start engineering
# Script tests, Rust tests and doctests, optional-feature tests, Python tests.
bash scripts/test.sh engineering
section_end engineering
fi

if wanted development; then
section_start development
# The development build runs `not_supported` routes and lists `would_refuse`.
cargo nextest run -P development -p antecedent --features development --test development_build
cargo nextest run -P development -p antecedent-core --features development --lib
section_end development
fi

# ---------------------------------------------------------------------------------------
# Tier 3: record currency. Nothing executes; an owed record fails the change.
# ---------------------------------------------------------------------------------------
if [[ "$section" == scientific ]]; then
section_start scientific
start_readers 2
check_records
finish_readers
if (( ${#failed_steps[@]} )); then
  echo "== scientific: ${#failed_steps[@]} check(s) failed; earn the owed records with bash scripts/test.sh and commit them ==" >&2
  printf '  FAILED: %s\n' "${failed_steps[@]}" >&2
  exit 1
fi
section_end scientific
fi

if [[ "$section" == all ]]; then
  echo "Pull-request gate PASSED (not a release candidate)."
  echo "Cut with: CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh"
fi
