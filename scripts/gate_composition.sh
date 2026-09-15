#!/usr/bin/env bash
# 1.10 composition gate: ledger rows in parity/compiler.toml drive invocations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1.10 composition consuming tests =="
REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "== existing evidence gates over composition records =="
bash scripts/gate_parity_schema.sh
bash scripts/gate_provenance_schema.sh
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh

run_and_count() {
  local label="$1"
  shift
  local log
  log="$(mktemp)"
  if ! "$@" >"$log" 2>&1; then
    echo "FAIL: $label"
    cat "$log"
    rm -f "$log"
    exit 1
  fi
  local ran
  ran="$(grep -cE '^test .* \.\.\. ok$|^python/tests/.* PASSED$|passed' "$log" || true)"
  if [[ "$ran" -lt 1 ]]; then
    echo "FAIL: $label reported no executed tests"
    cat "$log"
    rm -f "$log"
    exit 1
  fi
  echo "ok: $label ($ran tests)"
  rm -f "$log"
}

python3 - <<'PY' > /tmp/antecedent_composition_rows.txt
import tomllib
from pathlib import Path
rows = tomllib.loads(Path("parity/compiler.toml").read_text()).get("capabilities", [])
for row in rows:
    test = row.get("evidence_test")
    assertion = row.get("evidence_assertion")
    if not test or not assertion:
        continue
    filt = row.get("composition_filter")
    print(f"{row['id']}\t{test}\t{assertion}\t{filt if filt is not None else ''}")
PY

while IFS=$'\t' read -r cid test_rel assertion filt; do
  [[ -z "${cid}" ]] && continue
  if [[ "${test_rel}" == *.py ]]; then
    run_and_count "${cid}" uv run pytest -q "${test_rel}::${assertion}"
    continue
  fi
  crate="${test_rel#crates/}"
  crate="${crate%%/*}"
  if [[ "${test_rel}" == crates/*/tests/* ]]; then
    tfile="${test_rel##*/}"
    tfile="${tfile%.rs}"
    run_and_count "${cid}" cargo test -p "${crate}" --test "${tfile}" "${assertion}" -- --exact --nocapture
    if [[ -n "${filt}" ]]; then
      run_and_count "${cid}.filter" cargo test -p "${crate}" --test "${tfile}" "${filt}" -- --nocapture
    fi
  elif [[ "${test_rel}" == crates/*/src/* ]]; then
    run_and_count "${cid}" cargo test -p "${crate}" --lib "${assertion}" -- --exact --nocapture
    if [[ -n "${filt}" ]]; then
      run_and_count "${cid}.filter" cargo test -p "${crate}" --lib "${filt}" -- --nocapture
    fi
  else
    echo "FAIL: ${cid} evidence_test ${test_rel} is not a known layout"
    exit 1
  fi
done < /tmp/antecedent_composition_rows.txt

if [[ "${SKIP_PYTHON_SMOKE:-0}" == "1" ]]; then
  echo "FAIL: SKIP_PYTHON_SMOKE=1 is not composition evidence"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is required for the composition Python smoke"
  exit 1
fi

echo "revision: ${REVISION}"
echo "gate_composition: ok"
