#!/usr/bin/env bash
# Conformance gate for programmable interventions, mechanism selection, graph falsification,
# batch noise replay and intrinsic causal influence: parity/gcm_conformance.toml drives every
# invocation.
#
# The registry contract (every clause cited, every cited test and benchmark present, every
# owned acceptance case bound to a test the gate runs) is checked statically, then every
# [[conformance]] row is executed by the shared row runner: Rust rows must pass exactly one
# test, Python rows at least one, none may fail. The tree-ensemble recovery tests need the
# tree providers and run with them. Each [[performance]] benchmark runs in Criterion's smoke
# mode, which executes its in-process regression and reuse assertions; a full timing run is
# `cargo bench -p <package> --bench <bench>`.
#
#   bash scripts/gate_gcm_conformance.sh              # full gate
#   bash scripts/gate_gcm_conformance.sh --tests-only # without the benchmarks (gate_release.sh
#                                                     # smokes every bench target itself)
#   bash scripts/gate_gcm_conformance.sh --self-test  # broken registries must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGISTRY=parity/gcm_conformance.toml

if [[ "${1:-}" == "--self-test" ]]; then
  python3 scripts/check_gcm_conformance.py --self-test
  exit $?
fi
benches=1
if [[ "${1:-}" == "--tests-only" && $# -eq 1 ]]; then
  benches=0
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--tests-only|--self-test]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required; unexecuted Python rows are not conformance evidence"
  exit 1
fi

echo "== conformance registry contract =="
python3 scripts/check_gcm_conformance.py

echo "== acceptance manifest =="
python3 scripts/check_acceptance_manifest.py

echo "== conformance tests =="
python3 scripts/run_evidence_rows.py "$ROOT" "$REGISTRY" "$ROOT" conformance gate_gcm_conformance

echo "== nonlinear mechanism recovery with the tree providers =="
cargo test -p antecedent-model --features ml-gbdt,ml-forest --test mechanism_selection -- with_providers::

if [[ "$benches" -eq 1 ]]; then
  echo "== batch execution performance requirements =="
  while IFS=' ' read -r package bench; do
    cargo bench -p "$package" --bench "$bench" -- --test
  done < <(python3 - <<'PY'
import tomllib

for row in tomllib.load(open("parity/gcm_conformance.toml", "rb"))["performance"]:
    print(row["package"], row["bench"])
PY
  )
fi

echo "revision: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "gate_gcm_conformance: ok"
