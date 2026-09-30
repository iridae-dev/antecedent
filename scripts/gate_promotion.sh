#!/usr/bin/env bash
# 2.2 promotion gate: parity/promotion_2_2.toml freezes each 2.2 cell before
# implementation, licenses a route only with its record's evidence, and rejects a
# capability that executes without its proof, refusal and artifact evidence.
#
#   bash scripts/gate_promotion.sh              # static contract, then execute cited evidence
#   bash scripts/gate_promotion.sh --self-test  # broken registries must fail
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
REGISTRY=parity/promotion_2_2.toml

# Each case corrupts temporary copies (the registry, an owning registry, or an
# extra source file) and must make the checker report a failure that the
# committed tree does not already report. Judging by the added failure, not by the
# exit code alone, keeps each case meaningful while the committed registry is
# itself failing during reconciliation. Committed files are never written.
self_test() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  TMP="$tmp" REGISTRY="$REGISTRY" python3 - <<'PY'
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

tmp = Path(os.environ["TMP"])
registry = Path(os.environ["REGISTRY"])
text = registry.read_text()
stages = Path("parity/transport_stages.toml").read_text()
support = Path("parity/support_licensed.toml").read_text()
support_closed = Path("parity/support_closed.toml").read_text()


def run(reg: Path, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/check_promotion_records.py", str(reg)],
        capture_output=True, text=True, env={**os.environ, **env},
    )
    return proc.returncode, proc.stdout + proc.stderr


code, baseline = run(registry, {})
if code == 0:
    print("self-test: committed registry passes the contract")
else:
    failing = sum(1 for line in baseline.splitlines() if line.startswith(" - "))
    print(f"self-test: committed registry currently reports {failing} failure(s); cases must add their own")

# Frozen records flipped to in_progress, so implemented-work rules apply to them.
X4_LIVE = ('work_package = "A3"\nstatus = "frozen"', 'work_package = "A3"\nstatus = "in_progress"')
X5_LIVE = ('work_package = "A4"\nstatus = "frozen"', 'work_package = "A4"\nstatus = "in_progress"')
X4_MENU = '{ name = "antecedent.transport.estimator_menu", stage = "representation", status = "closed", reason_code = "cell_not_licensed" }'
X4_UNC = '{ name = "antecedent.transport.learned_continuous.uncertainty_joint_outer_bootstrap", stage = "uncertainty", status = "closed", reason_code = "cell_not_licensed" }'
X9_ID = '{ name = "antecedent_identify.mixed_source_search", stage = "identify", status = "closed", reason_code = "cell_not_licensed" }'
X4_FIRST_REFUSAL = 'refusals = [\n  { code = "transport_support_failure", detail = "learned_transport.membership_overlap"'
X9_ROW = 'route = "antecedent_identify.mixed_source_search"\nstage = "identify"\nstatus = "closed"\nreason_code = "cell_not_licensed"'
MENU_ROW = 'route = "antecedent.transport.estimator_menu"\nstage = "representation"\nstatus = "closed"\nreason_code = "cell_not_licensed"'

(tmp / "surface.py").write_text(
    "def self_test_unrouted_symbol():\n    pass\n\n\ndef _self_test_private():\n    pass\n\n\n"
    "class SelfTestValue:\n    pass\n"
)
(tmp / "nocharge.rs").write_text(
    "pub fn search(limits: SearchLimits, ctx: &ExecutionContext) {\n"
    "    let budget = SearchBudget::new(limits, ctx);\n}\n\n"
    "#[cfg(test)]\nmod tests {\n    fn metered() { budget.charge(1, 0).unwrap(); }\n}\n"
)
(tmp / "extra.rs").write_text(
    'pub fn refuse() -> String { format!("learned_transport.self_test_rust_detail: {}", 1) }\n\n'
    '#[cfg(test)]\nmod tests {\n    const ONLY_IN_TESTS: &str = "learned_transport.self_test_test_only";\n}\n'
)
(tmp / "extra.py").write_text('DETAIL = "learned_transport.self_test_python_detail"\n')

# (label, registry edits, owning-registry/source overrides, must report, must not report)
cases = [
    ("route licensed while frozen", [(X9_ID, X9_ID.replace('status = "closed", reason_code = "cell_not_licensed"', 'status = "licensed"'))], {},
     "mixed_source_search: licensed while its record is frozen", None),
    ("promoted without evidence", [('status = "frozen"', 'status = "promoted"')], {},
     "promoted record lacks executed evidence for fixture", None),
    ("nominal interval claim", [('inference_claim = "point_only"', 'inference_claim = "nominal"')], {},
     "inference_claim 'nominal' is not a 2.2 claim", None),
    ("unregistered refusal code", [('code = "transport_missing_evidence"', 'code = "no_such_code"')], {},
     "refusal code 'no_such_code' is not a registered", None),
    ("search without depth limit", [('depth_limit = "SearchLimits.depth", ', '')], {},
     "a bounded search must declare bounds.depth_limit", None),
    ("search off the shared contract", [('contract = "antecedent_core::SearchBudget"', 'contract = "ZTransportLimits"')], {},
     "a bounded search must run under antecedent_core::SearchBudget", None),
    ("missing frozen theorem", [('theorem = "mz-transportability', 'theorem_draft = "mz-transportability')], {},
     "missing frozen field theorem", None),
    ("missing negative fixture", [('{ id = "x9.missing_joint.negative", role = "negative"', '# ')], {},
     "2.2A.X9.mixed_source_proof_search: missing fixture roles negative", None),
    ("calibrated without records", [('coverage_records = ["cov.classical', 'coverage_records = [] # ["cov.classical')], {},
     "a calibrated interval must allocate its coverage record ids", None),
    ("unnamespaced refusal detail", [('detail = "mz_transport.search_incomplete"', 'detail = "search incomplete"')], {},
     "must be <namespace>.<snake_case>", None),
    ("uncancellable bound", [('max_history_states = 4096, cancellation = true', 'max_history_states = 4096, cancellation = false')], {},
     "2.2A.X5.two_step_temporal_transport: bounds.cancellation must be true", None),
    # Per-route licensing ahead of promotion.
    ("licensed uncertainty route before promotion",
     [X4_LIVE, (X4_UNC, X4_UNC.replace('status = "closed", reason_code = "cell_not_licensed"', 'status = "licensed", claim = "point_only"'))], {},
     "uncertainty_joint_outer_bootstrap: uncertainty route licensed before its record is promoted", None),
    ("licensed point route without evidence",
     [X4_LIVE, (X4_MENU, X4_MENU.replace('status = "closed", reason_code = "cell_not_licensed"', 'status = "licensed", claim = "point_only"'))], {},
     "estimator_menu: licensed before promotion but the record cites no executed evidence", None),
    ("licensed route without a point claim",
     [X4_LIVE, (X4_MENU, X4_MENU.replace('status = "closed", reason_code = "cell_not_licensed"', 'status = "licensed"'))], {},
     "estimator_menu: licensed before promotion requires claim", None),
    # Closed routes of implemented work refuse at runtime.
    ("closed route without refusal evidence", [X4_LIVE], {},
     "estimator_menu: closed route of a record at status in_progress needs refusal_test", None),
    ("refusal test that never names the reason",
     [X4_LIVE, (X4_MENU, X4_MENU.replace(' }', ', refusal_test = "python/tests/test_transport_scenarios.py", refusal_assertion = "test_cross_scenario_inference_and_equivalence_classes_are_refused" }'))], {},
     "estimator_menu: refusal test python/tests/test_transport_scenarios.py::test_cross_scenario_inference_and_equivalence_classes_are_refused never names", None),
    # Refusal boundary == non-test source literals.
    ("record detail absent from code",
     [X4_LIVE, (X4_FIRST_REFUSAL, X4_FIRST_REFUSAL.replace('refusals = [\n', 'refusals = [\n  { code = "invalid_argument", detail = "learned_transport.self_test_absent_from_code", when = "self-test" },\n'))], {},
     "refusal detail learned_transport.self_test_absent_from_code is not emitted by non-test source", None),
    ("rust code detail absent from record", [X4_LIVE],
     {"PROMOTION_EXTRA_SOURCES": f"{tmp / 'extra.rs'}{os.pathsep}{tmp / 'extra.py'}"},
     "emits undeclared refusal detail learned_transport.self_test_rust_detail", "learned_transport.self_test_test_only"),
    ("python code detail absent from record", [X4_LIVE],
     {"PROMOTION_EXTRA_SOURCES": f"{tmp / 'extra.py'}"},
     "emits undeclared refusal detail learned_transport.self_test_python_detail", None),
    # Route inventory covers the declared public surface.
    ("surface symbol without a route",
     [(X4_LIVE[0], X4_LIVE[1] + f'\nsurface = ["{tmp / "surface.py"}"]')], {},
     "surface symbol self_test_unrouted_symbol", "_self_test_private"),
    ("stale surface_values entry",
     [(X4_LIVE[0], X4_LIVE[1] + f'\nsurface = ["{tmp / "surface.py"}"]\nsurface_values = ["SelfTestValue", "SelfTestMissing"]')], {},
     "surface_values entry SelfTestMissing is not a public symbol", "surface symbol SelfTestValue"),
    ("function exempted by surface_values",
     [(X4_LIVE[0], X4_LIVE[1] + f'\nsurface = ["{tmp / "surface.py"}"]\nsurface_values = ["SelfTestValue", "self_test_unrouted_symbol"]')], {},
     "surface_values entry self_test_unrouted_symbol", "surface symbol SelfTestValue"),
    # Every route has an agreeing owning-registry row.
    ("route without an owning-registry row",
     [('name = "antecedent.transport.estimator_menu"', 'name = "antecedent.transport.self_test_unregistered"')], {},
     "antecedent.transport.self_test_unregistered has no row in transport_stages.toml", None),
    ("owning row with another reason code", [], {"stages": stages.replace(X9_ROW, X9_ROW.replace('"cell_not_licensed"', '"route_not_supported"'), 1)},
     "mixed_source_search: transport_stages.toml reason_code 'route_not_supported'", None),
    ("transport stage licensed before promotion", [],
     {"stages": stages + '\n[[routes]]\nroute = "antecedent_identify.mixed_source_search"\nstage = "identify"\nstatus = "licensed"\n'},
     "mixed_source_search is licensed in transport_stages.toml but closed in its record", None),
    ("support cell licensed before promotion", [],
     {"support": support + '\n[[cell]]\nquery = "NestedCounterfactualEffect"\ncontrast = "natural_indirect"\n'},
     "natural_indirect is licensed in support_licensed.toml but closed in its record", None),
    ("closed support route without a closed row", [],
     {"support_closed": support_closed.replace('contrast = "natural_indirect"', 'contrast = "self_test_other"')},
     "natural_indirect has no support_closed.toml row", None),
    # Search metered by the shared contract in code.
    ("search_impl without charge",
     [(X5_LIVE[0], X5_LIVE[1] + f'\nsearch_impl = ["{tmp / "nocharge.rs"}"]')], {},
     "search_impl files lack .charge(", None),
    ("in_progress search without search_impl", [X5_LIVE], {},
     "2.2A.X5.two_step_temporal_transport: a search at status in_progress must declare search_impl", None),
    # No nominal-only interval anywhere in implemented work's owning rows.
    ("nominal interval on an owning row", [X4_LIVE],
     {"stages": stages.replace(MENU_ROW, MENU_ROW + '\nnotes = "estimator_grid_not_measured"', 1)},
     "estimator_menu: its transport_stages row carries estimator_grid_not_measured", None),
]

OVERRIDES = {
    "stages": "PROMOTION_TRANSPORT_STAGES",
    "support": "PROMOTION_SUPPORT_LICENSED",
    "support_closed": "PROMOTION_SUPPORT_CLOSED",
}


def check(index: int, case) -> tuple[bool, str]:
    label, edits, overrides, expect, forbid = case
    corrupted = text
    for old, new in edits:
        if old not in corrupted:
            return False, f"SELF-TEST FAIL: '{label}': {old[:60]!r} not found in the registry"
        corrupted = corrupted.replace(old, new, 1)
    env = {}
    for key, value in overrides.items():
        if key in OVERRIDES:
            path = tmp / f"case{index}_{key}.toml"
            path.write_text(value)
            env[OVERRIDES[key]] = str(path)
        else:
            env[key] = value
    changed = corrupted != text or any(
        overrides.get(k) not in (None, orig) for k, orig in (("stages", stages), ("support", support), ("support_closed", support_closed))
    ) or "PROMOTION_EXTRA_SOURCES" in env
    if not changed:
        return False, f"SELF-TEST FAIL: corruption '{label}' changed nothing"
    reg = tmp / f"case{index}.toml"
    reg.write_text(corrupted)
    code, out = run(reg, env)
    if expect in baseline:
        return False, f"SELF-TEST FAIL: '{label}': the committed tree already reports {expect!r}"
    if code == 0 or expect not in out:
        return False, f"SELF-TEST FAIL: '{label}' did not report {expect!r}\n{out[-1500:]}"
    if forbid and forbid in out:
        return False, f"SELF-TEST FAIL: '{label}' wrongly reported {forbid!r}"
    return True, f"self-test ok: '{label}' fails"


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda pair: check(*pair), enumerate(cases)))
for _, message in results:
    print(message)
if not all(ok for ok, _ in results):
    sys.exit(1)
print("gate_promotion self-test: ok")
PY
}

if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit $?
fi

echo "== 2.2 promotion records =="
EVIDENCE="$(mktemp)"
trap 'rm -f "$EVIDENCE"' EXIT
python3 scripts/check_promotion_records.py --emit-evidence "$EVIDENCE"

# Cited fixtures and closed-route refusal tests are executed, not just resolved.
# Frozen records cite nothing yet, so there may be nothing to run.
if [[ -s "$EVIDENCE" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "FAIL: uv is required; unexecuted Python rows are not promotion evidence"
    exit 1
  fi
  echo "== promotion evidence: executing cited fixtures and refusals =="
  python3 scripts/run_evidence_rows.py "$ROOT" "$EVIDENCE" "$ROOT" fixture_evidence gate_promotion
else
  echo "no promotion fixture cites evidence yet; nothing to execute"
fi
echo "gate_promotion: ok"
