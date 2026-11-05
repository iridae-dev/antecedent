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
# in the static tier and again as its own CI job. An owed record fails the change.
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

SECTIONS="static public-api cited-tests cited-tests-0..7 benches deny python-lint engineering development scientific"
section=all
if [[ "${1:-}" == "--section" && $# -eq 2 ]]; then
  section="$2"
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--section ${SECTIONS// /|}]" >&2
  exit 2
fi
case "$section" in all|static|public-api|cited-tests|cited-tests-[0-7]|benches|deny|python-lint|engineering|development|scientific) ;; *)
  echo "unknown gate section: $section" >&2; exit 2 ;;
esac
wanted() { [[ "$section" == all || "$section" == "$1" ]]; }
section_since=0
section_start() { section_since=$SECONDS; echo "== $1 started $(date -u +%FT%TZ) =="; }
section_end() { echo "== $1 finished in $((SECONDS - section_since))s =="; }
# Static checks run to the end and report every failure together, so one run lists them all.
failed_steps=()
step() { "$@" || failed_steps+=("$*"); }

# ---------------------------------------------------------------------------------------
# Tier 0: static. One job, no build, no test execution.
# ---------------------------------------------------------------------------------------
if wanted static; then
section_start static

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
step bash scripts/gate_gcm_conformance.sh --tests-only

echo "== support: the one count, the matrix, the graphless supports, the transport stages =="
step python3 scripts/gate_support.py
step python3 scripts/gate_not_supported_sites.py
step bash scripts/gate_docs_support_matrix.sh

echo "== release inventory =="
step python3 scripts/check_release_inventory.py
step python3 scripts/check_acceptance_inventory.py
step python3 scripts/check_acceptance_manifest.py --require-bound
step python3 scripts/check_missing_policy_support.py

echo "== bounds, release claims, links, conformance references =="
step python3 scripts/generate_reason_details.py --check
step python3 scripts/gate_reason_detail_literals.py
step python3 scripts/gate_python_bindings_use_facade.py
step python3 scripts/check_limits_agreement.py
step python3 scripts/check_release_claims.py
step python3 scripts/check_doc_links.py
step python3 scripts/check_conformance_references.py

echo "== generated files are current =="
step python3 scripts/generate_conformance_docs.py
if ! git diff --exit-code -- docs/conformance >/dev/null; then
  echo "docs/conformance is stale; commit regenerated output"
  git diff --stat -- docs/conformance
  exit 1
fi
step python3 scripts/generate_support_matrix_docs.py
VERSION="$(python3 "$ROOT/scripts/release_notes_version.py")"
generated=(docs/support-matrix.md crates/antecedent/data/support_matrix.json
  crates/antecedent-io-core/data/coverage_records.json crates/antecedent-core/src/reason_codes_data.rs)
if ! git diff --exit-code -- "${generated[@]}" >/dev/null; then
  echo "support-matrix generated files are stale; commit regenerated output"
  git diff --stat -- "${generated[@]}"
  exit 1
fi
# Only the generated block of the release notes is compared; the hand-written text around it is not read.
if ! python3 - "docs/release-notes/v${VERSION}.md" <<'PY'
import subprocess
import sys

begin = "<!-- generated:support-matrix:supported:begin -->"
end = "<!-- generated:support-matrix:supported:end -->"
path = sys.argv[1]


def block(text: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


committed = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True, check=True).stdout
sys.exit(0 if block(committed) == block(open(path, encoding="utf-8").read()) else 1)
PY
then
  echo "the generated supported block of the release notes is stale; commit regenerated output"
  exit 1
fi
step python3 scripts/generate_host_schema.py --check

echo "== scientific and calibration records are current (earned locally, checked here) =="
step python3 scripts/scientific_records.py status --check
step python3 scripts/calibration_status.py --check
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
# The static tier runs these two checks inside its larger list; this section is the same
# pair as a job of their own, so CI's `scientific` job keeps meaning "the scientific and
# calibration evidence is current" and is quick (a checkout and Python, no build).
if wanted scientific; then
section_start scientific
step python3 scripts/scientific_records.py status --check
step python3 scripts/calibration_status.py --check
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
