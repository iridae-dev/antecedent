"""Portable scope metadata never becomes a whole-result validation verdict."""

from antecedent.results._slots import portable_evidence


def _report(scope):
    return {
        "refuter": "target.subset_stability",
        "original_ate": 12.0,
        "refuted_ate": 12.1,
        "comparison": 0.4,
        "informative": True,
        "passed": True,
        "failure_condition": None,
        "replicates": 30,
        "scope": scope,
    }


def test_scoped_portable_report_is_visible_without_whole_verdict():
    scoped = {
        "kind": "temporal_class_completion",
        "outer_atom": 2,
        "horizon_index": 0,
        "horizon": 1,
        "completion_digest": [3] * 32,
        "target": {"identity": [9] * 32},
    }
    validation = portable_evidence({}, {"refutations": [_report(scoped)]})["support"]["validation"]
    assert validation["ran"]
    assert validation["passed"] is None
    assert validation["reports"][0]["scope_target_identity"] == "09" * 32
    assert validation["reports"][0]["scope_completion_digest"] == "03" * 32
    assert validation["reports"][0]["scope_horizon"] == 1
    mixed = portable_evidence({}, {"refutations": [_report(None), _report(scoped)]})["support"][
        "validation"
    ]
    assert mixed["ran"] and mixed["passed"] is None
    failed = {
        "code": "refute.validator.failed",
        "fields": [["validator", "subset"], ["reason", "failed"]],
    }
    with_failure = portable_evidence(
        {}, {"refutations": [_report(scoped)], "diagnostics": [failed]}
    )["support"]["validation"]
    assert with_failure["ran"] and with_failure["passed"] is None
    assert with_failure["computation_failures"][0]["validator"] == "subset"


def test_scoped_validation_displays_unknown_and_names_scope():
    from antecedent.results._html import _refutation_table_html
    from antecedent.results._views import ValidationView

    report = _report(None)
    report.pop("scope")
    report["scope_kind"] = "temporal_class_completion"
    view = ValidationView(passed=None, ran=True, count=1, reports=[report])
    assert "unknown" in repr(view)
    html = _refutation_table_html(view)
    assert "Whole-result verdict unavailable" in html
    assert "temporal_class_completion" in html


def test_live_report_conversion_retains_scope_and_accepts_legacy_views():
    from types import SimpleNamespace

    from antecedent._from_raw import _refutation_reports_from_raw

    raw = _report(None)
    raw.pop("scope")
    raw.update(
        scope_kind="temporal_class_completion",
        scope_target_identity="09" * 32,
        scope_outer_atom=2,
        scope_completion_digest="03" * 32,
        scope_horizon_index=0,
        scope_horizon=1,
    )
    scoped = _refutation_reports_from_raw(SimpleNamespace(reports=[SimpleNamespace(**raw)]))[0]
    assert scoped.scope_kind == "temporal_class_completion"
    assert scoped.scope_target_identity == "09" * 32
    assert scoped.scope_completion_digest == "03" * 32
    assert scoped.scope_outer_atom == 2 and scoped.scope_horizon == 1
    legacy = {k: v for k, v in raw.items() if not k.startswith("scope_")}
    old = _refutation_reports_from_raw(SimpleNamespace(reports=[SimpleNamespace(**legacy)]))[0]
    assert old.scope_kind is None and old.scope_target_identity is None
