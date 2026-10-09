"""Self-test and mutation check for scripts/check_promotion_records.py.

    python3 scripts/promotion_selftest.py                   # gate_promotion.sh --self-test
    python3 scripts/promotion_selftest.py --mutation-check  # manual: every rule has a live case
    python3 scripts/promotion_selftest.py --release 2.3     # the broken-record cases against a 2.3 registry

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
#[test]
fn names_reason_substring() { assert_eq!(code(), "route_not_supported_extended"); }
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
surface_internal = [{{ name = "hidden_helper", why = "self-test hidden helper" }}]
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
# The looping charge moved into a private fn nothing calls; the pub fn charges once.
SEARCH_DEAD = (
    "pub fn search(limits: SearchLimits, ctx: &ExecutionContext) {\n"
    "    let budget = SearchBudget::new(limits, ctx);\n"
    "    budget.charge(1, 0).unwrap();\n}\n"
    "fn dead_walk(budget: &SearchBudget) {\n"
    "    for state in 0..3 {\n        budget.charge(1, state).unwrap();\n    }\n}\n"
)
# The looping charge sits in a private helper a pub fn calls, and in a trait impl.
SEARCH_HELPER = (
    "pub fn search(limits: SearchLimits, ctx: &ExecutionContext) {\n"
    "    let budget = SearchBudget::new(limits, ctx);\n    walk(&budget);\n}\n"
    "fn walk(budget: &SearchBudget) {\n"
    "    for state in 0..3 {\n        budget.charge(1, state).unwrap();\n    }\n}\n"
)
SEARCH_TRAIT = (
    "pub struct Walker;\n"
    "impl Search for Walker {\n"
    "    fn run(&self, limits: SearchLimits, ctx: &ExecutionContext) {\n"
    "        let budget = SearchBudget::new(limits, ctx);\n"
    "        for state in 0..3 {\n            budget.charge(1, state).unwrap();\n        }\n    }\n}\n"
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

# An impl of a listed value type in a file no surface field names: never scanned.
UNSCANNED_IMPL = "impl SelfTestValue {\n    pub fn self_test_shared_name(&self) {}\n    pub fn run_unscanned(&self) {}\n}\n"
IMPL_ANCHOR = "    pub fn point_route(&self) {}\n"
PLAIN_TYPE = "pub struct SelfTestPlain;\n\nimpl SelfTestPlain {\n    pub fn self_test_shared_name(&self) {}\n}\n"

FILES = {
    "record.toml": RECORD,
    "stages.toml": STAGES,
    "support.toml": "version = 1\n",
    "support_closed.toml": SUPPORT_CLOSED,
    "surface.py": SURFACE_PY,
    "surface_api.rs": SURFACE_API,
    "pyo3_api.rs": PYO3_API,
    "exports.py": EXPORTS,
    "unscanned_impl.rs": UNSCANNED_IMPL,
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
    ("record.toml", 'surface_internal = [{ name = "hidden_helper", why = "self-test hidden helper" }]\n', ""),
]
INTERNAL = 'surface_internal = [{ name = "hidden_helper", why = "self-test hidden helper" }]'
QUALIFIED_PLUMBING = """\
#[doc(hidden)]
pub struct HiddenToken;
impl HiddenToken {
    #[doc(hidden)]
    pub fn inspect(&self) {}
}
impl SelfTestValue {
    pub fn inspect(&self) {}
}
"""
QUALIFIED_INTERNAL = 'surface_internal = [{ name = "hidden_helper", why = "self-test hidden helper" }, { name = "HiddenToken", why = "hidden token" }, { name = "HiddenToken.inspect", why = "hidden own method, not another public same-name accessor" }]'



def rec(old, new):
    return ("record.toml", old, new)


def covering(route_line, *entries):
    """`route_line` with an added `covers` list."""
    listed = ", ".join(f'"{e}"' for e in entries)
    return route_line.replace(" },\n", f", covers = [{listed}] }},\n")


PY_ANCHOR = "    fn identify_route(&self) {}\n"
PY_PLAIN = "#[pyclass]\nstruct SelfTestPlainPy {}\n\n#[pymethods]\nimpl SelfTestPlainPy {\n    fn self_test_plain_getter(&self) {}\n}\n"
EXTRA_FN = "pub fn self_test_extra_fn() {}\n"


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
    ("search_charge_loop", "a looping charge in a dead private fn meters nothing",
     [("search_ok.rs", SEARCH_OK, SEARCH_DEAD)],
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
    ("closed_refusal_reason_named", "refusal test that names the reason only as a longer identifier",
     [rec(UNC_CLOSED, UNC_CLOSED.replace(f'refusal_test = "{PY_TEST}", refusal_assertion = "{PY_REFUSAL}"',
                                         f'refusal_test = "{SYN_RS}", refusal_assertion = "names_reason_substring"'))],
     "synth_tests.rs::names_reason_substring never names the route's reason_code route_not_supported", None),
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
    ("surface_symbol", "method of a type that is not a listed value type",
     [("surface_api.rs", None, PLAIN_TYPE.replace("self_test_shared_name", "self_test_new_public"))],
     "surface symbol self_test_new_public", "private_type_method"),
    ("surface_symbol", "value type's accessor is uncovered once the type is not in surface_values",
     [("surface_api.rs", IMPL_ANCHOR, IMPL_ANCHOR + "    pub fn self_test_accessor(&self) {}\n"),
      rec('surface_values = ["SelfTestValue"]', 'surface_values = []')],
     "surface symbol self_test_accessor", None),
    ("surface_symbol", "a listed type does not cover the same-named method of an unlisted type; an impl in a non-scanned file covers nothing",
     [("surface_api.rs", None, PLAIN_TYPE)],
     "surface symbol self_test_shared_name", "surface_value_method_verb"),
    ("surface_symbol", "a free decide_* function next to a listed value type still needs a route",
     [("surface_api.rs", None, "pub fn decide_self_test() {}\n")],
     "surface symbol decide_self_test", "surface_value_method_verb"),
    ("surface_value_method_verb", "executing-verb method on a listed value type",
     [("surface_api.rs", IMPL_ANCHOR, IMPL_ANCHOR + "    pub fn run_self_test(&self) {}\n")],
     "SelfTestValue.run_self_test", "surface symbol run_self_test"),
    ("surface_value_method_verb", "exact executing verb on a listed value type",
     [("surface_api.rs", IMPL_ANCHOR, IMPL_ANCHOR + "    pub fn consume(self) {}\n")],
     "SelfTestValue.consume", None),
    ("surface_symbol", "public rust function without a route",
     [("surface_api.rs", None, "pub fn self_test_new_function() {}\n")],
     "surface symbol self_test_new_function", "crate_only"),
    ("surface_symbol", "hidden rust item not listed as internal",
     [("surface_api.rs", None, "#[doc(hidden)]\npub fn hidden_free() {}\n")],
     "surface symbol hidden_free", None),
    ("surface_symbol", "pyo3 function without a route", [("pyo3_api.rs", None, "#[pyfunction]\nfn self_test_unrouted_py() {}\n")],
     "surface symbol self_test_unrouted_py", None),
    ("surface_symbol", "a renamed pyo3 function is exposed under its #[pyfunction(name)] name",
     [("pyo3_api.rs", "#[pyfunction]\nfn point_route() {}\n", '#[pyfunction(name = "self_test_renamed")]\nfn point_route() {}\n')],
     "surface symbol self_test_renamed", "surface symbol point_route"),
    ("surface_symbol", "a #[pyo3(name)] before #[pyfunction] renames the export too",
     [("pyo3_api.rs", "#[pyfunction]\nfn point_route() {}\n", '#[pyo3(name = "self_test_renamed")]\n#[pyfunction]\nfn point_route() {}\n')],
     "surface symbol self_test_renamed", "surface symbol point_route"),
    ("surface_symbol", "a pub use re-export is a public symbol of a surface_rust file",
     [("surface_api.rs", None, "pub use crate::other::{self_test_reexport, SelfTestValue as SelfTestAlias};\n")],
     "surface symbol self_test_reexport", "surface symbol SelfTestValue"),
    ("surface_symbol", "pyo3 method without a route",
     [("pyo3_api.rs", None, PY_PLAIN)],
     "surface symbol self_test_plain_getter", "surface symbol new"),
    ("surface_symbol", "a pyclass getter is uncovered once its class is not in surface_values",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn self_test_getter(&self) {}\n"),
      rec('surface_values = ["SelfTestValue"]', 'surface_values = []')],
     "surface symbol self_test_getter", None),
    ("surface_symbol", "a route's covers does not reach an unlisted symbol of another name",
     [("surface_api.rs", None, EXTRA_FN + "pub fn self_test_other_fn() {}\n"),
      rec(POINT, covering(POINT, "self_test_extra_fn"))],
     "surface symbol self_test_other_fn", "surface symbol self_test_extra_fn"),
    ("surface_value_method_verb", "refresh method of a listed pyclass",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn refresh_self_test(&self) {}\n")],
     "SelfTestValue.refresh_self_test", "surface symbol refresh_self_test"),
    ("surface_value_method_verb", "exact executing verb (export) on a listed pyclass",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn export(&self) {}\n")],
     "SelfTestValue.export", None),
    ("surface_value_method_verb", "executing-verb method (prepare_*) on a listed pyclass",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn prepare_self_test(&self) {}\n")],
     "SelfTestValue.prepare_self_test", None),
    ("route_covers_shape", "covers is not a list of names",
     [rec(POINT, POINT.replace(" },\n", ', covers = "self_test_extra_fn" },\n'))],
     "covers must be a list of non-empty symbol names", None),
    ("route_covers_stale", "covers names a symbol no surface file declares",
     [rec(POINT, covering(POINT, "self_test_no_such_symbol"))],
     "covers entry self_test_no_such_symbol is not a public symbol", None),
    ("route_covers_stale", "covers names a Type.method of another owner",
     [rec(POINT, covering(POINT, "SelfTestValue.self_test_no_such_method"))],
     "covers entry SelfTestValue.self_test_no_such_method is not a public symbol", None),
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
    ("surface_internal_shape", "surface_internal entry without a why", [rec(', why = "self-test hidden helper"', '')],
     "surface_internal entry", None),
    ("surface_internal_stale", "surface_internal names nothing",
     [rec(INTERNAL, 'surface_internal = [{ name = "hidden_helper", why = "r" }, { name = "no_such_item", why = "r" }]')],
     "surface_internal entry no_such_item names no Rust/pyo3 item", None),
    ("surface_internal_visible", "surface_internal names a visible pub item",
     [rec(INTERNAL, 'surface_internal = [{ name = "hidden_helper", why = "r" }, { name = "identify_route", why = "r" }]')],
     "surface_internal entry identify_route is a public item that is not #[doc(hidden)]", None),
    ("surface_internal_visible", "qualified internal method does not hide another public same-name accessor",
     [("surface_api.rs", None, QUALIFIED_PLUMBING),
      rec(INTERNAL, QUALIFIED_INTERNAL.replace("HiddenToken.inspect", "SelfTestValue.inspect"))],
     "surface_internal entry SelfTestValue.inspect is a public item that is not #[doc(hidden)]", None),
    ("surface_internal_stale", "qualified internal wrong owner cannot borrow a same-name hidden method",
     [("surface_api.rs", None, QUALIFIED_PLUMBING),
      rec(INTERNAL, QUALIFIED_INTERNAL.replace("HiddenToken.inspect", "MissingToken.inspect"))],
     "surface_internal entry MissingToken.inspect names no Rust/pyo3 item", None),
    ("surface_owned_orphan", "surface_exports without owned_exports", [rec('owned_exports = ["point_route", "SelfTestValue"]\n', '')],
     "surface_exports and owned_exports go together", None),
    ("surface_export_stale", "owned export not in __all__",
     [rec('owned_exports = ["point_route", "SelfTestValue"]', 'owned_exports = ["point_route", "SelfTestValue", "not_exported"]')],
     "owned_exports entry not_exported is not in __all__", None),
]

# Constructs the checker must ACCEPT: (label, edits).
positives = [
    ("qualified hidden method leaves public same-name accessor visible",
     [("surface_api.rs", None, QUALIFIED_PLUMBING), rec(INTERNAL, QUALIFIED_INTERNAL)]),
    ("shared evidence on every fixture that shares the test",
     [rec(POS, fx("xt.point_a.positive", "positive", PY_TEST, PY_POS, ", shared_evidence = true")),
      rec(NEG, fx("xt.refusal_a.negative", "negative", PY_TEST, PY_POS, ", shared_evidence = true"))]),
    ("a helper's assertion, an Err-forcing unwrap and pytest.raises are assertions",
     [rec(POS, syn("xt.point_a.positive", "positive", SYN_RS, "helper_asserts")),
      rec(NEG, syn("xt.refusal_a.negative", "negative", SYN_RS, "expects_err")),
      rec(ART, syn("xt.roundtrip_a.artifact", "artifact", SYN_PY, "test_raises"))]),
    ("a listed value type covers its accessors and constructors (verb-like prefixes need a word boundary)",
     [("surface_api.rs", IMPL_ANCHOR,
       IMPL_ANCHOR + "    pub fn self_test_accessor(&self) {}\n    pub fn from_record_checked() -> Self { SelfTestValue }\n"
       "    pub fn identifiable(&self) {}\n    pub fn fitted_values(&self) {}\n")]),
    ("a #[pymethods] getter of a listed pyclass is covered (interval/plan/outcome style accessors)",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn interval(&self) {}\n    fn plan(&self) {}\n    fn refreshing(&self) {}\n")]),
    ("a pyclass verb method is fine when it is internal (underscore name) or covered by a route",
     [("pyo3_api.rs", PY_ANCHOR, PY_ANCHOR + "    fn _refresh_hidden(&self) {}\n    fn export_wire(&self) {}\n"),
      rec(POINT, covering(POINT, "SelfTestValue.export_wire"))]),
    ("a route covers a free function and a Type.method verb; route names stay unchanged",
     [("surface_api.rs", None, EXTRA_FN),
      ("surface_api.rs", IMPL_ANCHOR, IMPL_ANCHOR + "    pub fn run_self_test(&self) {}\n"),
      rec(POINT, covering(POINT, "self_test_extra_fn", "SelfTestValue.run_self_test"))]),
    ("a covers entry no evidence closure or entry file mentions still passes but warns",
     [("surface_api.rs", None, EXTRA_FN),
      rec(UNC_CLOSED, covering(UNC_CLOSED, "self_test_extra_fn"))],
     "warning: 2.2A.XT.self_test_cell: self_test_cell.uncertainty_route: covers entry self_test_extra_fn"),
    ("an executing-verb method on a listed value type is fine when it is a route component or internal",
     [("surface_api.rs", IMPL_ANCHOR, IMPL_ANCHOR + "    pub fn identify_route(&self) {}\n    #[doc(hidden)]\n    pub fn run_hidden(&self) {}\n"),
      rec('why = "self-test hidden helper" }]', 'why = "self-test hidden helper" }, { name = "run_hidden", why = "r" }]')]),
    ("trait-impl methods of a listed value type are not scanned (no verb guard, no coverage needed)",
     [("surface_api.rs", None, "impl Default for SelfTestValue {\n    pub fn run_default() -> Self { SelfTestValue }\n}\n")]),
    ("an impl of a listed value type in a non-scanned file is not scanned",
     [("unscanned_impl.rs", UNSCANNED_IMPL, UNSCANNED_IMPL + "\n")]),
    ("a python helper's assert counts", [rec(POS, syn("xt.point_a.positive", "positive", SYN_PY, "test_helper_asserts"))]),
    ("a looping charge in a private helper a pub fn calls, or in a trait impl, is metered",
     [("search_ok.rs", SEARCH_OK, SEARCH_HELPER + SEARCH_TRAIT)]),
    ("a renamed pyo3 function is covered under its exposed name; a pub(crate) use or a re-export of a routed name needs nothing",
     [("pyo3_api.rs", "#[pyfunction]\nfn point_route() {}\n", '#[pyfunction(name = "point_route")]\nfn point_route_py() {}\n'),
      ("surface_api.rs", None, "pub use crate::other::identify_route;\npub(crate) use crate::other::self_test_crate_reexport;\n")]),
    ("a Rust file listed as a facade `surface` is scanned for its top-level items only",
     [rec('surface = ["{D}/surface.py"]', 'surface = ["{D}/surface.py", "{D}/surface_api.rs"]')]),
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


# ------------------------------------------------------------------ 2.3 release
#
#   python3 scripts/promotion_selftest.py --release 2.3 [--mutation-check]
#
# The same broken-record machinery against a synthetic registry named
# promotion_2_3.toml (the checker derives the expected release from the file name):
# release = "2.3", record id 2.3A.XT.self_test_cell, and the 2.3 record fields
# prerequisite_records / prerequisite_reason. The checked-in cases are re-rendered
# for 2.3 (to_23) and a curated subset is selected (SELECT_23); the prerequisite
# graph rules live in scripts/check_2_3_prerequisites.py, not in the checker's
# RULES, so they are exercised in-process by prereq_case below. Nothing under
# parity/ is read or written.

FILES_ACTIVE = FILES
REGISTRY_FILE = "record.toml"

PREREQ_TOKEN = "@@OLD_RELEASE@@"
PREREQ_LINE = 'prerequisite_records = ["2.2A.XT.self_test_base"]\n'
PREREQ_REASON = 'prerequisite_reason = "self-test prerequisite"\n'
OLD_SYN = [{"id": "2.2A.XT.self_test_base", "status": "promoted"}]


def to_23(text):
    """Render a 2.2 synthetic string (file, edit or expected message) for 2.3."""
    if text is None:
        return None
    with_prereq = REC_BODY.replace(
        'owners = ["self-test"]\n',
        'owners = ["self-test"]\n' + PREREQ_LINE.replace("2.2A", PREREQ_TOKEN) + PREREQ_REASON,
        1,
    )
    text = text.replace(REC_BODY, with_prereq)
    text = (
        text.replace("2.2A.XT", "2.3A.XT")
        .replace('release = "2.2"', 'release = "2.3"')
        .replace("and release 2.2", "and release 2.3")
    )
    return text.replace(PREREQ_TOKEN, "2.2A")


# (rule, label) of the 2.2 cases re-run against 2.3, grouped by the plan's list.
SELECT_23 = [
    # missing frozen fields / registry header / duplicate and unknown records
    ("frozen_field", "missing frozen theorem"),
    ("registry_header", "wrong release"),
    ("duplicate_record", "duplicate record id"),
    ("unknown_status", "unknown status"),
    # orphan public surface
    ("surface_symbol", "python surface symbol without a route"),
    ("surface_symbol", "public rust function without a route"),
    ("surface_symbol", "hidden rust item not listed as internal"),
    ("surface_symbol", "pyo3 function without a route"),
    ("surface_symbol", "pyo3 method without a route"),
    ("surface_symbol", "owned export without a route"),
    ("surface_value_method_verb", "executing-verb method on a listed value type"),
    # unexecuted fixture
    ("evidence_pair", "evidence_test without an assertion"),
    ("evidence_unresolved", "evidence that does not resolve"),
    ("evidence_no_assertion", "rust test without an assertion"),
    ("evidence_no_assertion", "python test without an assertion"),
    ("evidence_should_panic", "should_panic test as evidence"),
    ("evidence_shared", "one test backs two fixture roles"),
    ("promoted_evidence", "promoted without evidence"),
    ("licensed_unevidenced", "licensed point route without evidence"),
    ("closed_refusal_required", "closed route without refusal evidence"),
    ("closed_refusal_reason_named", "refusal test that never names the reason"),
    # missing artifact / budget / calibration evidence
    ("fixture_roles", "missing negative fixture"),
    ("fixture_roles", "calibrated claim without a calibration fixture"),
    ("coverage_required", "calibrated without records"),
    ("coverage_unexpected", "coverage ids for a point claim"),
    ("coverage_unknown", "promoted record with an unregistered coverage id"),
    ("budget_symbol", "budget fixture that never names the contract"),
    ("budget_asserts", "budget fixture that names but never observes the stop"),
    ("bounds_limit", "search without depth limit"),
    ("search_charge_loop", "one straight-line charge is a pre-flight"),
    # false licensed status
    ("licensed_status", "route licensed while frozen"),
    ("licensed_uncertainty", "licensed uncertainty route before promotion"),
    ("promoted_closed_route", "promoted record with a closed non-permanent route"),
    ("nominal_route", "licensed route carries the nominal marker"),
    ("stages_missing", "route without an owning-registry row"),
    ("stages_licensed_only_here", "route licensed in the record only"),
    ("stages_licensed_only_there", "transport stage licensed before promotion"),
    ("stages_reason", "owning row with another reason code"),
    ("support_licensed_no_cell", "support route licensed without a cell"),
    ("support_no_closed_row", "closed support route without a closed row"),
    # refusal mismatch
    ("refusal_code", "unregistered refusal code"),
    ("refusal_detail_missing", "record detail absent from code"),
    ("refusal_detail_undeclared", "rust code detail absent from record"),
    ("refusal_detail_undeclared", "python code detail absent from record"),
    ("refusal_pair_code", "the detail is emitted without its code"),
    ("refusal_dynamic", "rust format! builds a detail"),
    ("refusal_dead_const", "a detail held in a const nothing uses"),
]
SELECT_23_POSITIVE = ["shared evidence on every fixture that shares the test"]

IDENT_LICENSED = '  { name = "self_test_cell.identify_route", stage = "identify", status = "licensed", claim = "none" },\n'
IDENT_CLOSED = (
    f'  {{ name = "self_test_cell.identify_route", stage = "identify", status = "closed", reason_code = "{REASON}" }},\n'
)
POINT_CLOSED = (
    f'  {{ name = "self_test_cell.point_route", stage = "evaluate", status = "closed", reason_code = "{REASON}" }},\n'
)
STAGE_IDENT = 'route = "self_test_cell.identify_route"\nstage = "identify"\nstatus = "licensed"'
STATUS_IN_PROGRESS = '\nstatus = "in_progress"\n'
STATUS_CARRIED = '\nstatus = "carried_forward"\n'
PREREQ_BLOCK = PREREQ_LINE + PREREQ_REASON
SECOND_RECORD = (
    '\n[[record]]\nid = "2.3A.XT.self_test_second"\nworkstream = "XT"\n'
    'prerequisite_records = ["2.2A.XT.self_test_base", "2.3A.XT.self_test_cell"]\n'
)

# Cases written directly for 2.3 (already in 2.3 form; not passed through to_23).
NEW_23_CASES = [
    ("registry_header", "a 2.3 registry file marked release 2.2", [rec('release = "2.3"', 'release = "2.2"')],
     "promotion registry requires version 1 and release 2.3", None),
    ("frozen_field", "missing frozen consumer question", [rec('\nconsumer_question = ', '\nconsumer_question_draft = ')],
     "missing frozen field consumer_question", None),
    ("frozen_field", "missing frozen refusal boundary", [rec('\nrefusals = [', '\nrefusals_draft = [')],
     "missing frozen field refusals", None),
    ("fixture_roles", "missing artifact fixture", [rec(ART, "")],
     "2.3A.XT.self_test_cell: missing fixture roles artifact", None),
    ("fixture_roles", "missing budget fixture on a search record", [rec(BUD, "")],
     "2.3A.XT.self_test_cell: missing fixture roles budget", None),
    ("licensed_status", "licensed routes on a carried_forward record", [rec(STATUS_IN_PROGRESS, STATUS_CARRIED)],
     "identify_route: licensed while its record is carried_forward", None),
]
# Constructs the 2.3 checker must accept.
NEW_23_POSITIVES = [
    ("a carried_forward record with every route closed and its owning rows closed",
     [rec(STATUS_IN_PROGRESS, STATUS_CARRIED),
      rec(IDENT_LICENSED, IDENT_CLOSED),
      rec(POINT, POINT_CLOSED),
      ("stages.toml", STAGE_IDENT, STAGE_IDENT.replace('"licensed"', '"closed"') + f'\nreason_code = "{REASON}"'),
      ("stages.toml", STAGE_POINT, STAGE_POINT.replace('"licensed"', '"closed"') + f'\nreason_code = "{REASON}"')]),
]
# Prerequisite-graph cases (check_2_3_prerequisites.problems): (label, edits, expected, old registry).
PREREQ_CASES = [
    ("a 2.3 record with no prerequisite", [rec(PREREQ_BLOCK, "prerequisite_records = []\n")],
     "needs a named 2.2 prerequisite", None),
    ("a 2.3 record with no prerequisite and no reason", [rec(PREREQ_BLOCK, "prerequisite_records = []\n")],
     "no 2.2 executing base needs an explicit prerequisite_reason", None),
    ("a 2.3 record without the prerequisite field", [rec(PREREQ_BLOCK, "")],
     "prerequisite_records must be a list of record IDs", None),
    ("a self-prerequisite",
     [rec(PREREQ_LINE, 'prerequisite_records = ["2.2A.XT.self_test_base", "2.3A.XT.self_test_cell"]\n')],
     "2.3A.XT.self_test_cell: self prerequisite", None),
    ("an unknown prerequisite",
     [rec(PREREQ_LINE, 'prerequisite_records = ["2.2A.XT.self_test_base", "2.3A.XT.nope"]\n')],
     "unknown prerequisite 2.3A.XT.nope", None),
    ("a cyclic prerequisite",
     [rec(PREREQ_LINE, 'prerequisite_records = ["2.2A.XT.self_test_base", "2.3A.XT.self_test_second"]\n'),
      ("record.toml", None, SECOND_RECORD)],
     "cyclic 2.3 prerequisites", None),
    ("a 2.2 prerequisite that is not an executing base", [("record.toml", None, "\n")],
     "is not an executing base", [{"id": "2.2A.XT.self_test_base", "status": "carried_forward"}]),
]


def release_23_cases() -> list:
    by_key = {(c[0], c[1]): c for c in cases}
    missing = [key for key in SELECT_23 if key not in by_key]
    if missing:
        raise SystemExit(f"SELF-TEST FAIL: --release 2.3 selects unknown 2.2 cases {missing}")
    rendered = []
    for key in SELECT_23:
        rule, label, edits, expect, forbid = by_key[key]
        rendered.append((rule, label, [(n, to_23(o), to_23(w)) for n, o, w in edits], to_23(expect), to_23(forbid)))
    return rendered + NEW_23_CASES


def release_23_positives() -> list:
    by_label = {p[0]: p for p in positives}
    missing = [label for label in SELECT_23_POSITIVE if label not in by_label]
    if missing:
        raise SystemExit(f"SELF-TEST FAIL: --release 2.3 selects unknown 2.2 positives {missing}")
    rendered = []
    for label in SELECT_23_POSITIVE:
        _, edits, *rest = by_label[label]
        rendered.append((label, [(n, to_23(o), to_23(w)) for n, o, w in edits], *[to_23(r) for r in rest]))
    return rendered + NEW_23_POSITIVES


def configure_release_23() -> None:
    """Switch the module to the 2.3 rendering (before any case runs)."""
    global FILES_ACTIVE, REGISTRY_FILE
    FILES_ACTIVE = {name: to_23(body) for name, body in FILES.items()}
    REGISTRY_FILE = "promotion_2_3.toml"
    new_cases, new_positives = release_23_cases(), release_23_positives()
    cases[:] = new_cases
    positives[:] = new_positives


def coverage_problems_23() -> list[str]:
    listed, _ = rule_ids()
    return [f"a 2.3 self-test case is tagged with unknown rule {r}" for r in sorted({c[0] for c in cases} - set(listed))]


def prereq_case(index: int, label: str, edits, expect: str | None, old=None) -> tuple[bool, str]:
    """Run check_2_3_prerequisites.problems over the synthetic 2.3 registry."""
    import tomllib

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_2_3_prerequisites as prereq

    try:
        reg, _ = synthetic(index, edits, baseline=expect is None)
    except LookupError as err:
        return False, f"SELF-TEST FAIL: prerequisite '{label}': {err}"
    found = prereq.problems(old if old is not None else OLD_SYN, tomllib.loads(reg.read_text())["record"])
    if expect is None:
        if found:
            return False, f"SELF-TEST FAIL: prerequisite baseline must pass, got {found}"
        return True, f"self-test ok: prerequisite '{label}' passes"
    if not any(expect in item for item in found):
        return False, f"SELF-TEST FAIL: prerequisite '{label}' did not report {expect!r}; got {found}"
    return True, f"self-test ok: prerequisite '{label}' fails"


def release_arg() -> str:
    for i, arg in enumerate(sys.argv):
        value = arg.split("=", 1)[1] if arg.startswith("--release=") else (
            sys.argv[i + 1] if arg == "--release" and i + 1 < len(sys.argv) else None
        )
        if value is not None:
            if value not in ("2.2", "2.3"):
                raise SystemExit("usage: promotion_selftest.py [--release 2.2|2.3] [--mutation-check]")
            return value
        if arg == "--release":
            raise SystemExit("usage: promotion_selftest.py [--release 2.2|2.3] [--mutation-check]")
    return "2.2"


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
    base = FILES_ACTIVE  # FILES for 2.2; the 2.3 rendering of FILES under --release 2.3
    texts = dict(base)
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
        target = REGISTRY_FILE if name == "record.toml" else name
        (case / target).write_text(body.replace("{D}", str(case)))
    env = {key: str(case / name) for key, name in ENV_FILES.items()}
    env["PROMOTION_EXTRA_SOURCES"] = os.pathsep.join(str(case / n) for n in ("code.rs", "code.py"))
    env["PROMOTION_SYNTHETIC_EVIDENCE"] = "1"
    env["PROMOTION_EVIDENCE_CACHE"] = str(CACHE)
    if disable:
        env["PROMOTION_DISABLE_RULES"] = disable
    if texts == base and not baseline:
        raise LookupError("case changed nothing")
    return case / REGISTRY_FILE, env


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


def positive(index: int, label: str, edits, expect_out: str | None = None) -> tuple[bool, str]:
    try:
        reg, env = synthetic(index, edits, baseline=True)
    except LookupError as err:
        return False, f"SELF-TEST FAIL: positive '{label}': {err}"
    code, out = run(reg, env)
    if code != 0:
        return False, f"SELF-TEST FAIL: positive '{label}' must pass\n{out[-1500:]}"
    if expect_out and expect_out not in out:
        return False, f"SELF-TEST FAIL: positive '{label}' must print {expect_out!r}\n{out[-1500:]}"
    return True, f"self-test ok: '{label}' passes"


def baseline() -> tuple[bool, str]:
    reg, env = synthetic(999, [], baseline=True)
    code, out = run(reg, env)
    if code != 0 or "Promotion records OK (1 records" not in out:
        return False, f"SELF-TEST FAIL: the synthetic baseline registry must pass\n{out[-2500:]}"
    return True, "self-test ok: synthetic baseline registry passes"


def committed() -> tuple[bool, str]:
    """Both committed registries pass every rule (evidence resolved statically here;
    the gate itself lists and executes it for real)."""
    for release in ("2_2", "2_3"):
        path = Path(f"parity/promotion_{release}.toml")
        code, out = run(path, {"PROMOTION_STATIC_ONLY": "1"})
        if code != 0:
            return False, f"SELF-TEST FAIL: {path} does not pass\n{out[-2500:]}"
    # The 2.3 filename must not accidentally accept the 2.2 release marker.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "promotion_2_3.toml"
        source = (ROOT / "parity/promotion_2_3.toml").read_text()
        path.write_text(source.replace('release = "2.3"', 'release = "2.2"', 1))
        code, out = run(path, {"PROMOTION_STATIC_ONLY": "1"})
        if code == 0 or "promotion registry requires version 1 and release 2.3 [registry_header]" not in out:
            return False, f"SELF-TEST FAIL: 2.3 release mismatch was accepted\n{out[-2500:]}"
    return True, "self-test ok: both committed registries pass the checker"


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
    release = release_arg()
    if release == "2.3":
        configure_release_23()
    problems = coverage_problems() if release == "2.2" else coverage_problems_23()
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
                # The 2.3 run leaves the committed-registry sanity check to the 2.2 run (it covers both).
                if release == "2.2" and os.environ.get("PROMOTION_SELFTEST_SKIP_COMMITTED") != "1":  # analysis only
                    jobs.append(pool.submit(committed))
                jobs += [pool.submit(check, i, c) for i, c in enumerate(cases)]
                jobs += [pool.submit(positive, 1000 + i, label, *rest) for i, (label, *rest) in enumerate(positives)]
                if release == "2.3":
                    jobs.append(pool.submit(prereq_case, 1999, "baseline (one 2.2 base, a reason)", [], None))
                    jobs += [pool.submit(prereq_case, 2000 + i, *c) for i, c in enumerate(PREREQ_CASES)]
            results = [j.result() for j in jobs]
    for _, message in results:
        print(message)
    if not all(ok for ok, _ in results):
        return 1
    if mutation:
        print(f"gate_promotion mutation-check: ok ({len({c[0] for c in cases})} rules, {len(cases)} cases)")
    elif release == "2.3":
        print(f"gate_promotion 2.3 self-test: ok ({len(cases)} cases, {len(positives)} positives, {len(PREREQ_CASES) + 1} prerequisite cases)")
    else:
        print("gate_promotion self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
