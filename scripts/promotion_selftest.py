"""Self-test and mutation check for scripts/check_promotion_records.py.

    python3 scripts/promotion_selftest.py                   # gate_promotion.sh --self-test
    python3 scripts/promotion_selftest.py --mutation-check  # manual: every rule has a live case

The self-test builds ONE synthetic record (2.2A.XT.self_test_cell) with its own owning
registries and source files in a temp dir, so it never depends on the state of any
committed record. The baseline must pass the checker; each case mutates only
synthetic files and must fail with its specific message AND its `[rule-id]`. Cases
are tagged with the rule they exercise; a rule of the checker with no tagged case,
or a case tagged with an unknown rule, fails the self-test. Positive cases must
pass (they prove a construct is accepted, e.g. raw-string literals, shared
evidence). One sanity case checks the committed registry passes (statically).

--mutation-check switches each rule off in turn (PROMOTION_DISABLE_RULES) and requires
every case tagged with it to stop reporting: a rule whose case survives its own
removal is not really tested.

Evidence: Python fixtures cite real cheap tests (collected by pytest for real);
Rust and body-analysis fixtures cite synthetic test files outside the repo, resolved
statically (PROMOTION_SYNTHETIC_EVIDENCE), so the self-test does not depend on any
cargo target compiling. Committed files are never written.
"""

import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_promotion_records.py"
os.chdir(ROOT)

# Real, cheap Python tests (each distinct: one test proves one fixture role).
PY_TEST = "python/tests/test_transport_scenarios.py"
PY_POS = "test_identified_scenarios_disagree_and_the_envelope_spans_them"
PY_REFUSAL = "test_cross_scenario_inference_and_equivalence_classes_are_refused"  # names route_not_supported
PY_ART = "test_artifact_round_trip_keeps_failed_scenarios_and_refuses_tampering"
PY_OTHER = "test_prepare_retains_checked_scenario_plans_after_builder_disposal"  # never names it
SYN_RS = "{D}/synth_tests.rs"
SYN_PY = "{D}/test_synth.py"

REASON = "route_not_supported"

SYNTH_RS = """\
#[test]
fn budget_observes_stop() {
    let receipt = run_budget();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_consumed, Some(3));
}
fn run_budget() -> Receipt { let b = SearchBudget::new(); b.receipt() }
#[test]
fn budget_names_only() { let _b = SearchBudget::new(); assert!(1 + 1 == 2); }
#[test]
fn asserts_things() { assert_eq!(2, 2); }
#[test]
fn asserts_nothing() { let _x = 1; }
#[test]
#[should_panic]
fn only_panics() { panic!("boom"); }
#[test]
fn trivial_assert() { assert!(true); }
#[test]
fn helper_asserts() { check(); }
fn check() { assert_eq!(2, 2); }
#[test]
fn expects_err() { let r: Result<u8, u8> = Err(1); r.unwrap_err(); }
"""
SYNTH_PY = """\
import pytest


def _check():
    assert 1 + 1 == 2


def test_asserts():
    assert 1 + 1 == 2


def test_nothing():
    x = 1
    return x


def test_trivial():
    assert True


def test_raises():
    with pytest.raises(ValueError):
        int("x")


def test_helper_asserts():
    _check()
"""


def fx(fid, role, test, assertion, extra=""):
    return (
        f'  {{ id = "{fid}", role = "{role}", intent = "self-test {role}", '
        f'evidence_test = "{test}", evidence_assertion = "{assertion}"{extra} }},\n'
    )


POS = fx("xt.point_a.positive", "positive", PY_TEST, PY_POS)
NEG = fx("xt.refusal_a.negative", "negative", PY_TEST, PY_REFUSAL)
ART = fx("xt.roundtrip_a.artifact", "artifact", PY_TEST, PY_ART)
BUD = fx("xt.budget_a.budget", "budget", SYN_RS, "budget_observes_stop")
LIFE = '  { id = "xt.extra_a.lifecycle", role = "lifecycle", intent = "self-test lifecycle" },\n'
UNC_CLOSED = (
    f'  {{ name = "self_test_cell.uncertainty_route", stage = "uncertainty", status = "closed", '
    f'reason_code = "{REASON}", refusal_test = "{PY_TEST}", refusal_assertion = "{PY_REFUSAL}" }},\n'
)
UNC_LICENSED = '  { name = "self_test_cell.uncertainty_route", stage = "uncertainty", status = "licensed", claim = "point_only" },\n'
POINT = '  { name = "self_test_cell.point_route", stage = "evaluate", status = "licensed", claim = "point_only" },\n'
SUPPORT = (
    f'  {{ name = "self_test_cell.support_route", stage = "evaluate", status = "closed", reason_code = "{REASON}", '
    f'registry = "support_licensed", query = "NestedCounterfactualEffect", contrast = "self_test_contrast", '
    f'refusal_test = "{PY_TEST}", refusal_assertion = "{PY_REFUSAL}" }},\n'
)
SUPPORT_LICENSED = (
    '  { name = "self_test_cell.support_route", stage = "evaluate", status = "licensed", claim = "point_only", '
    'registry = "support_licensed", query = "NestedCounterfactualEffect", contrast = "self_test_contrast" },\n'
)
FIRST_REFUSAL = '  { code = "invalid_argument", detail = "self_test_cell.invalid_query", when = "malformed query" },\n'
STAGE_UNC = 'route = "self_test_cell.uncertainty_route"\nstage = "uncertainty"\nstatus = "closed"'
STAGE_POINT = 'route = "self_test_cell.point_route"\nstage = "evaluate"\nstatus = "licensed"'

# The one synthetic record, generated here so no committed record can drift it.
RECORD = f'''version = 1
release = "2.2"

[[record]]
id = "2.2A.XT.self_test_cell"
workstream = "XT"
milestone = "A"
work_package = "AT"
status = "in_progress"
consumer_question = "Does the self-test cell hold every frozen field?"
theorem = "self-test theorem."
reference = "https://example.invalid/self-test"
guarantee = "sound_incomplete"
graph_class = "self-test graph class"
population_semantics = "self-test population semantics"
evidence_family = "self-test evidence family"
provider = "self-test provider"
estimand = "self-test estimand"
inference_claim = "point_only"
owners = ["self-test"]
identity_inputs = ["self-test identity"]
wire_changes = ["self-test wire change"]
compatibility = "self-test compatibility"
search = true
search_impl = ["{{D}}/search_ok.rs"]
surface = ["{{D}}/surface.py"]
surface_rust = ["{{D}}/surface_api.rs"]
surface_pyo3 = ["{{D}}/pyo3_api.rs"]
surface_exports = ["{{D}}/exports.py"]
owned_exports = ["point_route", "SelfTestValue"]
surface_values = ["SelfTestValue"]
surface_internal = [{{ name = "hidden_helper", reason = "self-test hidden helper" }}]
bounds = {{ max_states = 16, cancellation = true, contract = "antecedent_core::SearchBudget", operation_limit = "SearchLimits.operations", depth_limit = "SearchLimits.depth", memory_limit = "SearchLimits.memory" }}
refusals = [
{FIRST_REFUSAL}  {{ code = "transport_not_certified", detail = "self_test_cell.search_incomplete", when = "search finished without a verdict" }},
  {{ code = "transport_missing_evidence", detail = "self_test_cell.checked_obstruction", when = "an obstruction was replayed" }},
]
routes = [
  {{ name = "self_test_cell.identify_route", stage = "identify", status = "licensed", claim = "none" }},
{POINT}{UNC_CLOSED}{SUPPORT}]
fixtures = [
{POS}{NEG}{ART}{BUD}]
'''
REC_BODY = RECORD.split("[[record]]", 1)[1]

STAGES = f'''version = 1

[[routes]]
route = "self_test_cell.identify_route"
stage = "identify"
status = "licensed"

[[routes]]
{STAGE_POINT}

[[routes]]
{STAGE_UNC}
reason_code = "{REASON}"
'''

SUPPORT_CLOSED = f'''version = 1

[[closed_contrast]]
query = "NestedCounterfactualEffect"
contrast = "self_test_contrast"
reason_code = "{REASON}"
record = "2.2A.XT.self_test_cell"
'''

SEARCH_OK = (
    "pub fn search(limits: SearchLimits, ctx: &ExecutionContext) {\n"
    "    let budget = SearchBudget::new(limits, ctx);\n"
    "    for state in 0..3 {\n        budget.charge(1, state).unwrap();\n    }\n}\n"
)
CODE_RS = (
    "pub fn refuse(kind: u8) -> (&'static str, &'static str) {\n    match kind {\n"
    '        0 => ("invalid_argument", "self_test_cell.invalid_query"),\n'
    '        1 => ("transport_not_certified", "self_test_cell.search_incomplete: no budget left"),\n'
    '        _ => ("transport_missing_evidence", "self_test_cell.checked_obstruction"),\n    }\n}\n'
)
SURFACE_PY = "def point_route():\n    pass\n\n\nclass SelfTestValue:\n    pass\n"
SURFACE_API = """\
pub struct SelfTestValue;

impl SelfTestValue {
    pub fn point_route(&self) {}
    pub(crate) fn crate_only(&self) {}
    fn private_method(&self) {}
    #[doc(hidden)]
    pub fn hidden_helper(&self) {}
}

pub fn identify_route() {}

struct PrivateType;

impl PrivateType {
    pub fn private_type_method(&self) {}
}

#[cfg(test)]
mod tests {
    pub fn only_in_tests() {}
}
"""
PYO3_API = """\
#[pyfunction]
fn point_route() {}

#[pyclass]
struct SelfTestValue {}

#[pymethods]
impl SelfTestValue {
    #[new]
    fn new() -> Self { SelfTestValue {} }
    fn identify_route(&self) {}
}
"""
EXPORTS = (
    '__all__ = ["point_route", "SelfTestValue", "other_export"]\n\n\n'
    "def point_route():\n    pass\n\n\nclass SelfTestValue:\n    pass\n\n\ndef other_export():\n    pass\n"
)

FILES = {
    "record.toml": RECORD,
    "stages.toml": STAGES,
    "support.toml": "version = 1\n",
    "support_closed.toml": SUPPORT_CLOSED,
    "surface.py": SURFACE_PY,
    "surface_api.rs": SURFACE_API,
    "pyo3_api.rs": PYO3_API,
    "exports.py": EXPORTS,
    "search_ok.rs": SEARCH_OK,
    "search_nobudget.rs": "fn walk() {\n    for x in xs {\n        s.charge(1, x);\n    }\n}\n",
    "synth_tests.rs": SYNTH_RS,
    "test_synth.py": SYNTH_PY,
    "code.rs": CODE_RS,
    "code.py": "NOTHING = 1\n",
}
ENV_FILES = {
    "PROMOTION_TRANSPORT_STAGES": "stages.toml",
    "PROMOTION_SUPPORT_LICENSED": "support.toml",
    "PROMOTION_SUPPORT_CLOSED": "support_closed.toml",
}
UNROUTED_SURFACE = (
    SURFACE_PY + "\n\ndef self_test_unrouted_symbol():\n    pass\n\n\ndef _self_test_private():\n    pass\n"
)
CODE_TEST_ONLY = '\n#[cfg(test)]\nmod tests {\n    const ONLY_IN_TESTS: &str = "self_test_cell.self_test_test_only";\n}\n'
RUST_EXTRA = 'pub fn extra() -> &\'static str { "self_test_cell.self_test_rust_detail" }\n'
NO_SURFACE = [
    ("record.toml", 'surface = ["{D}/surface.py"]\n', ""),
    ("record.toml", 'surface_rust = ["{D}/surface_api.rs"]\n', ""),
    ("record.toml", 'surface_pyo3 = ["{D}/pyo3_api.rs"]\n', ""),
    ("record.toml", 'surface_exports = ["{D}/exports.py"]\n', ""),
    ("record.toml", 'owned_exports = ["point_route", "SelfTestValue"]\n', ""),
    ("record.toml", 'surface_internal = [{ name = "hidden_helper", reason = "self-test hidden helper" }]\n', ""),
]
INTERNAL = 'surface_internal = [{ name = "hidden_helper", reason = "self-test hidden helper" }]'


def rec(old, new):
    return ("record.toml", old, new)


def syn(fid, role, test, assertion, extra=""):
    return fx(fid, role, test, assertion, extra)


def unc_without(field):
    return UNC_CLOSED.replace(field, "")


REFUSAL_TEST = f', refusal_test = "{PY_TEST}"'
REFUSAL_ASSERT = f', refusal_assertion = "{PY_REFUSAL}"'

# (rule, label, edits to the synthetic files, must report, must not report)
cases = [
    # -- registry and frozen fields
    ("registry_header", "wrong release", [rec('release = "2.2"', 'release = "2.1"')],
     "promotion registry requires version 1 and release 2.2", None),
    ("frozen_field", "missing frozen theorem", [rec('\ntheorem = ', '\ntheorem_draft = ')],
     "missing frozen field theorem", None),
    ("duplicate_record", "duplicate record id", [rec(REC_BODY, REC_BODY + "\n[[record]]" + REC_BODY)],
     "duplicate record id", None),
    ("unknown_status", "unknown status", [rec('\nstatus = "in_progress"\n', '\nstatus = "banana"\n')],
     "unknown status 'banana'", None),
    ("claim_invalid", "nominal interval claim", [rec('inference_claim = "point_only"', 'inference_claim = "nominal"')],
     "inference_claim 'nominal' is not a 2.2 claim", None),
    ("coverage_required", "calibrated without records", [rec('inference_claim = "point_only"', 'inference_claim = "calibrated"')],
     "a calibrated interval must allocate its coverage record ids", None),
    ("coverage_unexpected", "coverage ids for a point claim",
     [rec('inference_claim = "point_only"\n', 'inference_claim = "point_only"\ncoverage_records = ["2.2A.NOPE"]\n')],
     "coverage records allocated for a point_only claim", None),
    ("coverage_unknown", "promoted record with an unregistered coverage id",
     [rec('\nstatus = "in_progress"\n', '\nstatus = "promoted"\n'),
      rec('inference_claim = "point_only"\n', 'inference_claim = "calibrated"\ncoverage_records = ["2.2A.NOPE"]\n')],
     "unknown coverage record 2.2A.NOPE", None),
    # -- bounds and search
    ("bounds_cancellation", "uncancellable bound", [rec('cancellation = true', 'cancellation = false')],
     "2.2A.XT.self_test_cell: bounds.cancellation must be true", None),
    ("bounds_limit", "search without depth limit", [rec('depth_limit = "SearchLimits.depth", ', '')],
     "a bounded search must declare bounds.depth_limit", None),
    ("bounds_contract", "search off the shared contract", [rec('contract = "antecedent_core::SearchBudget"', 'contract = "ZTransportLimits"')],
     "a bounded search must run under antecedent_core::SearchBudget", None),
    ("search_flag", "search is not a boolean", [rec('search = true\n', 'search = "yes"\n')],
     "search must be declared true or false", None),
    ("search_impl_shape", "search_impl is not a list", [rec('search_impl = ["{D}/search_ok.rs"]', 'search_impl = "x.rs"')],
     "search_impl must be a list of source paths", None),
    ("search_impl_missing", "in_progress search without search_impl", [rec('search_impl = ["{D}/search_ok.rs"]\n', '')],
     "2.2A.XT.self_test_cell: a search at status in_progress must declare search_impl", None),
    ("search_impl_file", "search_impl names a missing file",
     [rec('search_impl = ["{D}/search_ok.rs"]', 'search_impl = ["{D}/search_ok.rs", "{D}/missing.rs"]')],
     "missing.rs is not a Rust source file", None),
    ("search_impl_names_budget", "a search_impl file never names SearchBudget",
     [rec('search_impl = ["{D}/search_ok.rs"]', 'search_impl = ["{D}/search_ok.rs", "{D}/search_nobudget.rs"]')],
     "search_nobudget.rs non-test source never names SearchBudget", None),
    ("search_impl_charge", "search_impl charge only in tests",
     [("search_ok.rs", "    for state in 0..3 {\n        budget.charge(1, state).unwrap();\n    }\n}\n",
       "}\n\n#[cfg(test)]\nmod tests {\n    fn metered() { budget.charge(1, 0).unwrap(); }\n}\n")],
     "search_impl files lack .charge(", None),
    ("search_charge_loop", "one straight-line charge is a pre-flight",
     [("search_ok.rs", "    for state in 0..3 {\n        budget.charge(1, state).unwrap();\n    }\n",
       "    budget.charge(1, 0).unwrap();\n")],
     "no search_impl fn charges a SearchBudget/SharedSearch inside a loop or recursion", None),
    # -- refusals
    ("refusal_code", "unregistered refusal code", [rec('code = "transport_not_certified"', 'code = "no_such_code"')],
     "refusal code 'no_such_code' is not a registered", None),
    ("refusal_when", "refusal without a condition", [rec(', when = "malformed query"', '')],
     "needs its condition", None),
    ("refusal_detail_shape", "unnamespaced refusal detail", [rec('detail = "self_test_cell.search_incomplete"', 'detail = "search incomplete"')],
     "must be <namespace>.<snake_case>", None),
    ("refusal_detail_duplicate", "duplicate refusal detail", [rec(FIRST_REFUSAL, FIRST_REFUSAL + FIRST_REFUSAL)],
     "duplicate refusal detail self_test_cell.invalid_query", None),
    ("refusal_namespace", "two namespaces in one record",
     [rec(FIRST_REFUSAL, FIRST_REFUSAL + '  { code = "invalid_argument", detail = "other_ns.thing", when = "self-test" },\n')],
     "refusal details must share one namespace", None),
    ("refusal_detail_missing", "record detail absent from code",
     [rec(FIRST_REFUSAL, FIRST_REFUSAL + '  { code = "invalid_argument", detail = "self_test_cell.absent_from_code", when = "self-test" },\n')],
     "refusal detail self_test_cell.absent_from_code is not emitted by non-test source", None),
    ("refusal_detail_undeclared", "rust code detail absent from record",
     [("code.rs", None, RUST_EXTRA + CODE_TEST_ONLY)],
     "emits undeclared refusal detail self_test_cell.self_test_rust_detail", "self_test_test_only"),
    ("refusal_detail_undeclared", "python code detail absent from record",
     [("code.py", None, 'DETAIL = "self_test_cell.self_test_python_detail"\n')],
     "emits undeclared refusal detail self_test_cell.self_test_python_detail", None),
    ("refusal_detail_undeclared", "raw string detail absent from record",
     [("code.rs", None, 'pub fn raw() -> &\'static str { r#"self_test_cell.self_test_raw_detail"# }\n')],
     "emits undeclared refusal detail self_test_cell.self_test_raw_detail", None),
    ("refusal_detail_undeclared", "plain raw string detail absent from record",
     [("code.rs", None, 'pub fn raw2() -> &\'static str { r"self_test_cell.self_test_raw2_detail" }\n')],
     "emits undeclared refusal detail self_test_cell.self_test_raw2_detail", None),
    ("refusal_detail_undeclared", "byte string detail absent from record",
     [("code.rs", None, 'pub fn bytes() -> &\'static [u8] { b"self_test_cell.self_test_bytes_detail" }\n')],
     "emits undeclared refusal detail self_test_cell.self_test_bytes_detail", None),
    ("refusal_detail_undeclared", "continued literal detail absent from record",
     [("code.rs", None, 'pub fn cont() -> &\'static str { "self_test_cell.\\\n        self_test_cont_detail" }\n')],
     "emits undeclared refusal detail self_test_cell.self_test_cont_detail", None),
    ("refusal_detail_undeclared", "python f-string detail absent from record",
     [("code.py", None, 'def f(x):\n    return f"self_test_cell.self_test_fstring_detail: {x}"\n')],
     "emits undeclared refusal detail self_test_cell.self_test_fstring_detail", None),
    ("refusal_detail_undeclared", "python adjacent literals build an undeclared detail",
     [("code.py", None, 'ADJ = ("self_test_cell."\n       "self_test_adjacent_detail")\n')],
     "emits undeclared refusal detail self_test_cell.self_test_adjacent_detail", None),
    ("refusal_dynamic", "rust format! builds a detail",
     [("code.rs", None, 'pub fn dynamic(k: &str) -> String { format!("self_test_cell.{}", k) }\n')],
     "dynamic refusal detail", None),
    ("refusal_dynamic", "python f-string builds a detail",
     [("code.py", None, 'def g(k):\n    return f"self_test_cell.{k}"\n')],
     "dynamic refusal detail", None),
    ("refusal_dynamic", "python concatenation builds a detail",
     [("code.py", None, 'def h(k):\n    return "self_test_cell." + k\n')],
     "dynamic refusal detail", None),
    ("refusal_pair_code", "the detail is emitted without its code",
     [("code.rs", '"transport_missing_evidence"', '"transport_other"')],
     "refusal (transport_missing_evidence, self_test_cell.checked_obstruction)", "invalid_query"),
    ("refusal_dead_const", "a detail held in a const nothing uses",
     [rec(FIRST_REFUSAL, FIRST_REFUSAL + '  { code = "invalid_argument", detail = "self_test_cell.dead_detail", when = "self-test" },\n'),
      ("code.rs", None, 'const DEAD: &str = "self_test_cell.dead_detail";\n')],
     "self_test_cell.dead_detail exists only as const DEAD", None),
    # -- fixtures and evidence
    ("fixture_duplicate", "duplicate fixture id", [rec(POS, POS + POS)],
     "duplicate fixture id xt.point_a.positive", None),
    ("fixture_id_format", "fixture id does not end in its role",
     [rec('id = "xt.point_a.positive", role = "positive"', 'id = "xt.point_a.wrong", role = "positive"')],
     "fixture xt.point_a.wrong must be <workstream>.<name>.<role>", None),
    ("fixture_intent", "fixture without an intent", [rec('intent = "self-test positive"', 'intent = ""')],
     "fixture xt.point_a.positive needs an intent", None),
    ("evidence_pair", "evidence_test without an assertion", [rec(POS, POS.replace(f', evidence_assertion = "{PY_POS}"', ""))],
     "fixture xt.point_a.positive needs both evidence_test and evidence_assertion", None),
    ("evidence_unresolved", "evidence that does not resolve",
     [rec(POS, POS.replace(PY_POS, "test_no_such_test_exists"))],
     "no def test_no_such_test_exists", None),
    ("evidence_no_assertion", "rust test without an assertion",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_RS, "asserts_nothing"))],
     "test asserts_nothing contains no assertion", None),
    ("evidence_no_assertion", "rust test with only assert!(true)",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_RS, "trivial_assert"))],
     "test trivial_assert contains no assertion", None),
    ("evidence_no_assertion", "python test without an assertion",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_PY, "test_nothing"))],
     "test test_nothing contains no assert statement", None),
    ("evidence_no_assertion", "python test with only assert True",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_PY, "test_trivial"))],
     "test test_trivial contains no assert statement", None),
    ("evidence_no_assertion", "closed route refusal test without an assertion",
     [rec(UNC_CLOSED, UNC_CLOSED.replace(f'refusal_test = "{PY_TEST}", refusal_assertion = "{PY_REFUSAL}"',
                                         f'refusal_test = "{SYN_RS}", refusal_assertion = "asserts_nothing"'))],
     "uncertainty_route: refusal evidence: test asserts_nothing contains no assertion", None),
    ("evidence_should_panic", "should_panic test as evidence",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_RS, "only_panics"))],
     "test only_panics is #[should_panic]", None),
    ("evidence_shared", "one test backs two fixture roles",
     [rec(NEG, syn("xt.refusal_a.negative", "negative", PY_TEST, PY_POS))],
     f"test {PY_TEST}::{PY_POS} backs more than one fixture role", None),
    ("budget_symbol", "budget fixture that never names the contract",
     [rec(BUD, syn("xt.budget_a.budget", "budget", SYN_RS, "asserts_things"))],
     "budget fixture xt.budget_a.budget does not exercise antecedent_core::SearchBudget", None),
    ("budget_asserts", "budget fixture that names but never observes the stop",
     [rec(BUD, syn("xt.budget_a.budget", "budget", SYN_RS, "budget_names_only"))],
     "no budget fixture asserts on SearchStop or a SearchReceipt field", None),
    ("promoted_evidence", "promoted without evidence",
     [rec('\nstatus = "in_progress"\n', '\nstatus = "promoted"\n'), rec(ART, ART + LIFE)],
     "promoted record lacks executed evidence for fixture xt.extra_a.lifecycle", None),
    ("fixture_roles", "missing negative fixture", [rec(NEG, '')],
     "2.2A.XT.self_test_cell: missing fixture roles negative", None),
    ("fixture_roles", "calibrated claim without a calibration fixture",
     [rec('inference_claim = "point_only"\n', 'inference_claim = "calibrated"\ncoverage_records = ["2.2A.NOPE"]\n')],
     "missing fixture roles calibration", None),
    # -- routes
    ("route_name", "duplicate route name", [rec(POINT, POINT + POINT)],
     "duplicate or missing route name 'self_test_cell.point_route'", None),
    ("route_stage", "invalid stage", [rec(POINT, POINT.replace('stage = "evaluate"', 'stage = "bogus"'))],
     "invalid stage 'bogus'", None),
    ("route_registry", "unknown owning registry", [rec(POINT, POINT.replace('status = "licensed",', 'status = "licensed", registry = "nowhere",'))],
     "unknown owning registry 'nowhere'", None),
    ("route_claim", "route claim nominal", [rec(POINT, POINT.replace('"point_only"', '"nominal"'))],
     "claim 'nominal' is not a 2.2 claim", None),
    ("route_status", "route status neither closed nor licensed", [rec(POINT, POINT.replace('"licensed"', '"pending"'))],
     "point_route: status must be closed or licensed", None),
    ("closed_reason", "closed route with an unregistered reason",
     [rec(UNC_CLOSED, UNC_CLOSED.replace(f'reason_code = "{REASON}"', 'reason_code = "no_such_code"'))],
     "uncertainty_route: closed route needs a registered runtime reason_code", None),
    ("closed_refusal_pair", "refusal_test without refusal_assertion", [rec(UNC_CLOSED, unc_without(REFUSAL_ASSERT))],
     "uncertainty_route: closed route needs both refusal_test and refusal_assertion", None),
    ("closed_refusal_unresolved", "refusal test that does not resolve",
     [rec(UNC_CLOSED, UNC_CLOSED.replace(PY_REFUSAL, "test_no_such_refusal"))],
     "uncertainty_route: refusal evidence: no def test_no_such_refusal", None),
    ("closed_refusal_reason_named", "refusal test that never names the reason",
     [rec(UNC_CLOSED, UNC_CLOSED.replace(f'refusal_assertion = "{PY_REFUSAL}"', f'refusal_assertion = "{PY_OTHER}"'))],
     f"uncertainty_route: refusal test {PY_TEST}::{PY_OTHER} never names", None),
    ("closed_refusal_required", "closed route without refusal evidence",
     [rec(UNC_CLOSED, unc_without(REFUSAL_TEST + REFUSAL_ASSERT))],
     "uncertainty_route: closed route of a record at status in_progress needs refusal_test", None),
    ("licensed_permanent", "licensed route marked permanent",
     [rec(POINT, POINT.replace('claim = "point_only"', 'claim = "point_only", permanent_in_release = true'))],
     "point_route: licensed but marked permanent_in_release", None),
    ("licensed_uncertainty", "licensed uncertainty route before promotion", [rec(UNC_CLOSED, UNC_LICENSED)],
     "uncertainty_route: uncertainty route licensed before its record is promoted", None),
    ("licensed_claim", "licensed route without a point claim", [rec(POINT, POINT.replace(', claim = "point_only"', ''))],
     "point_route: licensed before promotion requires claim", None),
    ("licensed_unevidenced", "licensed point route without evidence",
     [rec(BUD, BUD.split(", evidence_test")[0] + " },\n")],
     "point_route: licensed before promotion but the record cites no executed evidence", None),
    ("licensed_status", "route licensed while frozen", [rec('\nstatus = "in_progress"\n', '\nstatus = "frozen"\n')],
     "identify_route: licensed while its record is frozen", None),
    ("promoted_closed_route", "promoted record with a closed non-permanent route",
     [rec('\nstatus = "in_progress"\n', '\nstatus = "promoted"\n')],
     "promoted record leaves non-permanent route self_test_cell.uncertainty_route closed", None),
    ("nominal_route", "licensed route carries the nominal marker",
     [rec(POINT, POINT.replace('claim = "point_only"', 'claim = "point_only", notes = "estimator_grid_not_measured"'))],
     "licensed route self_test_cell.point_route carries estimator_grid_not_measured", None),
    ("nominal_row", "nominal interval on an owning row", [("stages.toml", None, 'notes = "estimator_grid_not_measured"\n')],
     "uncertainty_route: its transport_stages row carries estimator_grid_not_measured", None),
    # -- owning registries
    ("stages_missing", "route without an owning-registry row", [rec('name = "self_test_cell.point_route"', 'name = "self_test_cell.unregistered_route"')],
     "self_test_cell.unregistered_route has no row in transport_stages.toml", None),
    ("stages_stage", "owning row with another stage", [("stages.toml", STAGE_POINT, STAGE_POINT.replace('"evaluate"', '"consume"'))],
     "point_route: transport_stages.toml stage 'consume' != record stage 'evaluate'", None),
    ("stages_licensed_only_there", "transport stage licensed before promotion", [("stages.toml", STAGE_UNC, STAGE_UNC.replace('"closed"', '"licensed"'))],
     "uncertainty_route is licensed in transport_stages.toml but closed in its record", None),
    ("stages_licensed_only_here", "route licensed in the record only", [("stages.toml", STAGE_POINT, STAGE_POINT.replace('"licensed"', '"closed"'))],
     "point_route is licensed in its record but not in transport_stages.toml", None),
    ("stages_reason", "owning row with another reason code", [("stages.toml", f'reason_code = "{REASON}"', 'reason_code = "cell_not_licensed"')],
     "uncertainty_route: transport_stages.toml reason_code 'cell_not_licensed'", None),
    ("support_query_contrast", "support route without a query", [rec('query = "NestedCounterfactualEffect", ', '')],
     "a support_licensed route names its query and contrast", None),
    ("support_licensed_only_there", "support cell licensed before promotion",
     [("support.toml", None, '\n[[cell]]\nquery = "NestedCounterfactualEffect"\ncontrast = "self_test_contrast"\n')],
     "support_route is licensed in support_licensed.toml but closed in its record", None),
    ("support_licensed_no_cell", "support route licensed without a cell", [rec(SUPPORT, SUPPORT_LICENSED)],
     "support_route is licensed in its record but has no support_licensed.toml cell", None),
    ("support_licensed_still_closed", "support route licensed but still closed",
     [rec(SUPPORT, SUPPORT_LICENSED),
      ("support.toml", None, '\n[[cell]]\nquery = "NestedCounterfactualEffect"\ncontrast = "self_test_contrast"\n')],
     "support_route is licensed in its record but still closed in support_closed.toml", None),
    ("support_no_closed_row", "closed support route without a closed row", [("support_closed.toml", SUPPORT_CLOSED, "version = 1\n")],
     "support_route has no support_closed.toml row", None),
    ("support_closed_record", "closed row names another record", [("support_closed.toml", 'record = "2.2A.XT.self_test_cell"', 'record = "2.2A.XT.other"')],
     "support_closed.toml row names record '2.2A.XT.other'", None),
    ("support_closed_reason", "closed row with another reason", [("support_closed.toml", f'reason_code = "{REASON}"', 'reason_code = "cell_not_licensed"')],
     "support_route: support_closed.toml reason_code 'cell_not_licensed'", None),
    # -- public surface
    ("surface_shape", "surface is not a list", [rec('surface = ["{D}/surface.py"]', 'surface = "surface.py"')],
     "surface must be a list of source paths", None),
    ("surface_shape", "surface_rust holds a non-string", [rec('surface_rust = ["{D}/surface_api.rs"]', 'surface_rust = [1]')],
     "surface_rust must be a list of non-empty strings", None),
    ("surface_file", "surface names a missing file", [rec('surface = ["{D}/surface.py"]', 'surface = ["{D}/surface.py", "{D}/missing.py"]')],
     "missing.py is not a Python or Rust source file", None),
    ("surface_file", "surface_pyo3 names a missing file", [rec('surface_pyo3 = ["{D}/pyo3_api.rs"]', 'surface_pyo3 = ["{D}/nope.rs"]')],
     "nope.rs is not a Rust source file", None),
    ("surface_symbol", "python surface symbol without a route", [("surface.py", SURFACE_PY, UNROUTED_SURFACE)],
     "surface symbol self_test_unrouted_symbol", "_self_test_private"),
    ("surface_symbol", "public rust method without a route",
     [("surface_api.rs", "    fn private_method(&self) {}\n", "    fn private_method(&self) {}\n    pub fn self_test_new_public(&self) {}\n")],
     "surface symbol self_test_new_public", "private_type_method"),
    ("surface_symbol", "public rust function without a route",
     [("surface_api.rs", None, "pub fn self_test_new_function() {}\n")],
     "surface symbol self_test_new_function", "crate_only"),
    ("surface_symbol", "hidden rust item not listed as internal", [rec(INTERNAL + "\n", "")],
     "surface symbol hidden_helper", None),
    ("surface_symbol", "pyo3 function without a route", [("pyo3_api.rs", None, "#[pyfunction]\nfn self_test_unrouted_py() {}\n")],
     "surface symbol self_test_unrouted_py", None),
    ("surface_symbol", "pyo3 method without a route",
     [("pyo3_api.rs", "    fn identify_route(&self) {}\n", "    fn identify_route(&self) {}\n    fn self_test_unrouted_method(&self) {}\n")],
     "surface symbol self_test_unrouted_method", "surface symbol new"),
    ("surface_symbol", "owned export without a route",
     [rec('owned_exports = ["point_route", "SelfTestValue"]', 'owned_exports = ["point_route", "SelfTestValue", "self_test_unrouted_export"]'),
      ("exports.py", '"other_export"]', '"other_export", "self_test_unrouted_export"]'),
      ("exports.py", None, "\n\ndef self_test_unrouted_export():\n    pass\n")],
     "surface symbol self_test_unrouted_export", "other_export"),
    ("surface_values_shape", "surface_values is not a list", [rec('surface_values = ["SelfTestValue"]', 'surface_values = "SelfTestValue"')],
     "surface_values must be a list of symbol names", None),
    ("surface_values_orphan", "surface_values without any surface", NO_SURFACE,
     "surface_values SelfTestValue listed without a surface", None),
    ("surface_value_stale", "stale surface_values entry", [rec('surface_values = ["SelfTestValue"]', 'surface_values = ["SelfTestValue", "SelfTestMissing"]')],
     "surface_values entry SelfTestMissing is not a public symbol", "surface symbol SelfTestValue"),
    ("surface_value_kind", "function exempted by surface_values",
     [("surface.py", SURFACE_PY, UNROUTED_SURFACE),
      rec('surface_values = ["SelfTestValue"]', 'surface_values = ["SelfTestValue", "self_test_unrouted_symbol"]')],
     "surface_values entry self_test_unrouted_symbol", "surface symbol SelfTestValue"),
    ("surface_internal_shape", "surface_internal entry without a reason", [rec(', reason = "self-test hidden helper"', '')],
     "surface_internal entry", None),
    ("surface_internal_stale", "surface_internal names nothing",
     [rec(INTERNAL, 'surface_internal = [{ name = "hidden_helper", reason = "r" }, { name = "no_such_item", reason = "r" }]')],
     "surface_internal entry no_such_item names no Rust/pyo3 item", None),
    ("surface_internal_visible", "surface_internal names a visible pub item",
     [rec(INTERNAL, 'surface_internal = [{ name = "hidden_helper", reason = "r" }, { name = "identify_route", reason = "r" }]')],
     "surface_internal entry identify_route is a public item that is not #[doc(hidden)]", None),
    ("surface_owned_orphan", "surface_exports without owned_exports", [rec('owned_exports = ["point_route", "SelfTestValue"]\n', '')],
     "surface_exports and owned_exports go together", None),
    ("surface_export_stale", "owned export not in __all__",
     [rec('owned_exports = ["point_route", "SelfTestValue"]', 'owned_exports = ["point_route", "SelfTestValue", "not_exported"]')],
     "owned_exports entry not_exported is not in __all__", None),
]

# Constructs the checker must ACCEPT: (label, edits).
positives = [
    ("shared evidence on every fixture that shares the test",
     [rec(POS, fx("xt.point_a.positive", "positive", PY_TEST, PY_POS, ", shared_evidence = true")),
      rec(NEG, fx("xt.refusal_a.negative", "negative", PY_TEST, PY_POS, ", shared_evidence = true"))]),
    ("a helper's assertion, an Err-forcing unwrap and pytest.raises are assertions",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_RS, "helper_asserts")),
      rec(NEG, syn("xt.refusal_a.negative", "negative", SYN_RS, "expects_err")),
      rec(ART, syn("xt.roundtrip_a.artifact", "artifact", SYN_PY, "test_raises"))]),
    ("a python helper's assert counts", [rec(POS, syn("xt.point_a.positive", "positive", SYN_PY, "test_helper_asserts"))]),
    ("a const detail used by live code",
     [rec(FIRST_REFUSAL, FIRST_REFUSAL + '  { code = "invalid_argument", detail = "self_test_cell.live_detail", when = "self-test" },\n'),
      ("code.rs", None, 'const LIVE: &str = "self_test_cell.live_detail";\npub fn live() -> (&\'static str, &\'static str) { ("invalid_argument", LIVE) }\n')]),
    ("raw, byte and continued literals still emit the declared details",
     [("code.rs", '"self_test_cell.invalid_query"', 'r#"self_test_cell.invalid_query"#'),
      ("code.rs", '"self_test_cell.checked_obstruction"', 'b"self_test_cell.checked_obstruction"'),
      ("code.rs", '"self_test_cell.search_incomplete: no budget left"', '"self_test_cell.\\\n            search_incomplete: no budget left"')]),
    ("a python f-string with a static detail emits the declared detail",
     [("code.rs", '"self_test_cell.invalid_query"', '"x"'), ("code.py", None, 'def f(x):\n    return f"self_test_cell.invalid_query: {x}"\n'),
      ("code.py", None, '\nCODE = "invalid_argument"\n')]),
]


# ------------------------------------------------------------------------ runner

CACHE = None
TMP = None


def run(reg: Path, env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECKER), str(reg)], capture_output=True, text=True, env={**os.environ, **env}
    )
    return proc.returncode, proc.stdout + proc.stderr


def synthetic(index: int, edits, *, baseline: bool = False, disable: str | None = None):
    """Write the (possibly mutated) synthetic registry and its owning files."""
    case = TMP / f"case{index}"
    case.mkdir()
    texts = dict(FILES)
    for name, old, new in edits:
        if old is None:  # append
            texts[name] += new
            continue
        if old not in texts[name]:
            raise LookupError(f"{old[:70]!r} not found in synthetic {name}")
        if old == new:
            raise LookupError(f"edit of {name} changes nothing")
        texts[name] = texts[name].replace(old, new, 1)
    for name, body in texts.items():
        (case / name).write_text(body.replace("{D}", str(case)))
    env = {key: str(case / name) for key, name in ENV_FILES.items()}
    env["PROMOTION_EXTRA_SOURCES"] = os.pathsep.join(str(case / n) for n in ("code.rs", "code.py"))
    env["PROMOTION_SYNTHETIC_EVIDENCE"] = "1"
    env["PROMOTION_EVIDENCE_CACHE"] = str(CACHE)
    if disable:
        env["PROMOTION_DISABLE_RULES"] = disable
    if texts == FILES and not baseline:
        raise LookupError("case changed nothing")
    return case / "record.toml", env


def check(index: int, case, disable: str | None = None) -> tuple[bool, str]:
    rule, label, edits, expect, forbid = case
    try:
        reg, env = synthetic(index, edits, disable=disable)
    except LookupError as err:
        return False, f"SELF-TEST FAIL: [{rule}] '{label}': {err}"
    code, out = run(reg, env)
    if disable:  # mutation: with the rule off, its message must be gone
        if expect in out:
            return False, f"MUTATION SURVIVED: [{rule}] '{label}' still reports {expect!r} with the rule off"
        return True, f"mutation ok: [{rule}] '{label}' stops failing with the rule off"
    if code == 0 or expect not in out or f"[{rule}]" not in out:
        return False, f"SELF-TEST FAIL: [{rule}] '{label}' did not report {expect!r} tagged [{rule}]\n{out[-1500:]}"
    if forbid and forbid in out:
        return False, f"SELF-TEST FAIL: [{rule}] '{label}' wrongly reported {forbid!r}"
    return True, f"self-test ok: [{rule}] '{label}' fails"


def positive(index: int, label: str, edits) -> tuple[bool, str]:
    try:
        reg, env = synthetic(index, edits, baseline=True)
    except LookupError as err:
        return False, f"SELF-TEST FAIL: positive '{label}': {err}"
    code, out = run(reg, env)
    if code != 0:
        return False, f"SELF-TEST FAIL: positive '{label}' must pass\n{out[-1500:]}"
    return True, f"self-test ok: '{label}' passes"


def baseline() -> tuple[bool, str]:
    reg, env = synthetic(999, [], baseline=True)
    code, out = run(reg, env)
    if code != 0 or "Promotion records OK (1 records" not in out:
        return False, f"SELF-TEST FAIL: the synthetic baseline registry must pass\n{out[-2500:]}"
    return True, "self-test ok: synthetic baseline registry passes"


def committed() -> tuple[bool, str]:
    """The committed registry passes every rule (evidence resolved statically here;
    the gate itself lists and executes it for real)."""
    code, out = run(Path("parity/promotion_2_2.toml"), {"PROMOTION_STATIC_ONLY": "1"})
    if code != 0:
        return False, f"SELF-TEST FAIL: the committed registry does not pass\n{out[-2500:]}"
    return True, "self-test ok: committed registry passes the checker"


def rule_ids() -> tuple[list[str], set[str]]:
    listed = subprocess.run([sys.executable, str(CHECKER), "--list-rules"], capture_output=True, text=True).stdout.split()
    used = set(re.findall(r'\bfail\(\s*"([a-z_]+)"', CHECKER.read_text()))
    # rules the source analysers hand back as (rule, message) pairs
    used |= {r for r in listed if f'("{r}",' in (ROOT / "scripts/promotion_source.py").read_text()}
    return listed, used


def coverage_problems() -> list[str]:
    """A rule cannot exist without a case: RULES == the ids `fail()` is called with
    == the ids the cases are tagged with."""
    listed, used = rule_ids()
    tags = {c[0] for c in cases}
    problems = [f"rule {r} is in RULES but never reported by fail()" for r in listed if r not in used]
    problems += [f"rule {r} is reported by fail() but missing from RULES" for r in sorted(used - set(listed))]
    problems += [f"rule {r} has no self-test case tagged with it" for r in listed if r not in tags]
    problems += [f"a self-test case is tagged with unknown rule {r}" for r in sorted(tags - set(listed))]
    return problems


def main() -> int:
    global CACHE, TMP
    mutation = "--mutation-check" in sys.argv
    problems = coverage_problems()
    for p in problems:
        print(f"SELF-TEST FAIL: {p}")
    if problems:
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        TMP = Path(tmp)
        CACHE = TMP / "evidence-cache"
        CACHE.mkdir()
        ok, message = baseline()  # warms the evidence cache before the parallel cases
        print(message)
        if not ok:
            return 1
        workers = max(2, min(8, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = []
            if mutation:
                by_rule: dict[str, list[int]] = {}
                for i, case in enumerate(cases):
                    by_rule.setdefault(case[0], []).append(i)
                jobs += [pool.submit(check, i, cases[i], rule) for rule, idx in by_rule.items() for i in idx]
            else:
                if os.environ.get("PROMOTION_SELFTEST_SKIP_COMMITTED") != "1":  # analysis only
                    jobs.append(pool.submit(committed))
                jobs += [pool.submit(check, i, c) for i, c in enumerate(cases)]
                jobs += [pool.submit(positive, 1000 + i, label, edits) for i, (label, edits) in enumerate(positives)]
            results = [j.result() for j in jobs]
    for _, message in results:
        print(message)
    if not all(ok for ok, _ in results):
        return 1
    if mutation:
        print(f"gate_promotion mutation-check: ok ({len({c[0] for c in cases})} rules, {len(cases)} cases)")
    else:
        print("gate_promotion self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
