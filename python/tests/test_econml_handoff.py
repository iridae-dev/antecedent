"""EconML adjustment-set handoff."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success

from _repo_text import REPO_ROOT, load_json


def _backdoor_data(n: int = 200) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = (z > 0.5).astype(np.float64)
    y = 1.0 + 2.0 * t + 3.0 * z
    return {"t": t, "y": y, "z": z}


def _backdoor_graph() -> antecedent.Dag:
    return antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])


def test_adjustment_api_keeps_econml_aliases_layout_and_tool() -> None:
    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect(treatment="t", outcome="y")
    )
    spec = antecedent.handoff.adjustment(identified)
    assert type(spec) is antecedent.handoff.AdjustmentHandoff
    assert antecedent.handoff.econml is antecedent.handoff.adjustment
    assert antecedent.handoff.EconMLSpec is antecedent.handoff.AdjustmentHandoff
    assert antecedent.handoff.EconMLProviderAdapter is antecedent.handoff.AdjustmentProviderAdapter
    assert spec.tool == "antecedent.handoff.EconMLSpec"
    assert dict(spec.spec.provenance) == {"adapter": "antecedent.handoff.EconMLSpec"}
    data = _backdoor_data()
    econml_cols = spec.columns(data)
    assert set(econml_cols) == {"Y", "T", "W"}
    named = spec.columns(data, layout="named")
    assert set(named) == {"outcome", "treatment", "confounders", "modifiers", "weights", "origins"}
    np.testing.assert_array_equal(named["outcome"], econml_cols["Y"])
    np.testing.assert_array_equal(named["treatment"], econml_cols["T"])
    np.testing.assert_array_equal(named["confounders"], econml_cols["W"])
    assert named["modifiers"] is None and named["weights"] is None and named["origins"] is None
    with pytest.raises(antecedent.errors.CausalValueError, match="layout"):
        spec.columns(data, layout="sklearn")  # type: ignore[arg-type]

    def fit(columns, request):
        assert "Y" in columns and "outcome" not in columns
        return {"learner": "caller.learner", "effect": 1.0, "label": "ate"}

    default_run = require_success(spec.as_provider(fit).execute({"data": data}))
    assert default_run.provenance["adapter"] == "antecedent.handoff.EconMLProviderAdapter"
    assert default_run.provenance["tool"] == "antecedent.handoff.EconMLSpec"
    custom = antecedent.handoff.adjustment(identified, tool="caller.learner")
    assert dict(custom.spec.provenance) == {"adapter": "caller.learner"}
    custom_run = require_success(custom.as_provider(fit).execute({"data": data}))
    assert custom_run.provenance["adapter"] == "caller.learner"
    assert custom_run.provenance["tool"] == "caller.learner"


def test_backdoor_ate_emits_adjustment_set() -> None:
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=_backdoor_graph(), query=query)
    spec = antecedent.handoff.econml(identified)
    assert spec.treatment == "t"
    assert spec.outcome == "y"
    assert spec.confounders == ("z",)
    assert spec.identifier == "backdoor.adjustment"
    assert "Identified" in spec.status
    cols = spec.columns(_backdoor_data())
    assert cols["T"].shape == cols["Y"].shape
    assert cols["W"].shape == (200, 1)
    np.testing.assert_array_equal(cols["W"][:, 0], _backdoor_data()["z"])


def test_econml_runtime_adapter_registers_and_round_trips_attested_artifact(monkeypatch) -> None:
    import hashlib

    import antecedent.extensibility as extension
    from antecedent.extensibility import (
        ProviderRegistry,
        ProviderTrust,
        ProviderVerificationFixture,
    )

    data = _backdoor_data()
    identified = antecedent.identify(
        graph=_backdoor_graph(),
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
    )
    spec = antecedent.handoff.econml(identified)
    seen = {}

    def caller_fit(columns, request):
        seen["columns"] = columns
        seen["request"] = request
        return {
            "learner": "econml.dml.CausalForestDML",
            "learner_config": {"n_estimators": 5},
            "effect": 2.0,
            "interval": [1.5, 2.5],
            "label": "ate",
        }

    class EconMLEntryPoint:
        value = "caller_package:create_econml_provider"

        def load(self):
            return lambda: spec.as_provider(caller_fit)

    class EntryPoints:
        def select(self, *, group, name):
            assert (group, name) == ("antecedent.providers", "econml")
            return [EconMLEntryPoint()]

    monkeypatch.setattr(extension.metadata, "entry_points", lambda: EntryPoints())
    registry = ProviderRegistry()
    registry.load_entry_point("econml")
    executed = require_success(registry.execute("econml", {"data": data}))
    loaded = require_success(antecedent.load(executed.artifact))
    assert seen["columns"]["W"].shape == (len(data["t"]), 1)
    assert loaded.calibration.status == "unavailable"
    assert loaded.calibration.reason == "attested_not_reverifiable"
    assert executed.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert executed.provenance["adapter"] == "antecedent.handoff.EconMLProviderAdapter"
    assert executed.provenance["entry_point"] == EconMLEntryPoint.value

    # Reference receipt is prepared independently of the provider execution.
    expected_artifact = bytes(
        spec.attach(
            learner="econml.dml.CausalForestDML",
            learner_config={"n_estimators": 5},
            effect=2.0,
            interval=[1.5, 2.5],
            data=data,
            label="ate",
        ).export()
    )
    report = registry.verify(
        "econml",
        [
            ProviderVerificationFixture(
                name="independent-known-effect",
                request={"data": data},
                expected_estimate=2.0,
                expected_uncertainty=None,
                expected_assumptions=spec.spec.identification_requirements,
                expected_support_status=spec.status,
                expected_provenance={
                    "adapter": "antecedent.handoff.EconMLProviderAdapter",
                    "entry_point": EconMLEntryPoint.value,
                },
                artifact_digest=hashlib.sha256(expected_artifact).hexdigest(),
                artifact_decoder=lambda encoded: require_success(antecedent.load(encoded)).claim_id,
                expected_decoded_artifact=require_success(
                    antecedent.load(expected_artifact)
                ).claim_id,
            )
        ],
        evidence_origin="independent-econml-fixture/v1",
    )
    verified = require_success(registry.execute("econml", {"data": data}))
    assert verified.trust is ProviderTrust.VERIFIED_EXTENSION
    assert verified.provenance["verification_evidence_digest"] == report.evidence_digest
    assert require_success(antecedent.load(verified.artifact)).claim_id == loaded.claim_id


def test_analyze_result_handoff_matches_identify() -> None:
    data = _backdoor_data()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    result = require_success(
        antecedent.analyze(
            data,
            graph=_backdoor_graph(),
            query=query,
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    spec = antecedent.handoff.econml(result, treatment="t", outcome="y")
    assert result.plan.structure_source == "explicit"
    assert spec.confounders == ("z",)
    assert spec.identifier == "backdoor.adjustment"


def test_partial_id_refuses() -> None:
    pin = load_json(REPO_ROOT / "conformance" / "estimate" / "cpdag_ate_envelope" / "expected.json")
    graph = antecedent.Cpdag.from_directed_undirected(
        pin["columns"],
        [tuple(edge) for edge in pin["graph"]["directed_edges"]],
        [tuple(edge) for edge in pin["graph"]["undirected_edges"]],
    )
    values: dict[str, list[float]] = {name: [] for name in pin["columns"]}
    for cell in pin["contingency_table"]:
        count = int(cell["count"])
        for name in pin["columns"]:
            values[name].extend([float(cell[name])] * count)
    data = {name: np.asarray(col, dtype=np.float64) for name, col in values.items()}
    result = require_success(
        antecedent.analyze(
            data,
            graph=graph,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
        )
    )
    assert result.plan.structure_source == "explicit"
    assert result.identification.adjustment_set == tuple([])
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="point identification"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")
    spec = antecedent.handoff.EconMLSpec(
        "t", "y", (), "backdoor.adjustment", result.identification.status, _parent=result
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="point identification"):
        spec.attach(learner="econml.dml.LinearDML", effect=0.0, data=_backdoor_data())


def test_frontdoor_refuses() -> None:
    graph = antecedent.Dag.from_edges(
        ["u", "t", "m", "y"], [("u", "t"), ("u", "y"), ("t", "m"), ("m", "y")]
    )
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=graph, query=query, identifier="frontdoor")
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="frontdoor"):
        antecedent.handoff.econml(identified)
    spec = antecedent.handoff.EconMLSpec(
        "t",
        "y",
        (),
        identified.identifier or identified.method,
        identified.status,
        _parent=identified,
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="frontdoor"):
        spec.attach(learner="econml.dml.LinearDML", effect=0.0, data=_backdoor_data())


def test_frequentist_graph_posterior_refuses() -> None:
    from known_truth import FREQ, STATIC, static_data, static_posterior

    result = require_success(
        antecedent.analyze(
            static_data(int(STATIC["n"])),
            discovery=static_posterior(),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=FREQ,
            refute=False,
            bootstrap=0,
            seed=1,
        )
    )
    assert result.plan.structure_source == "graph_posterior"
    assert result.identification.status == "GraphDependent"
    assert result.identification.adjustment_set == tuple([])
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")
    spec = antecedent.handoff.EconMLSpec(
        "t", "y", (), "backdoor.adjustment", "NonparametricallyIdentified", _parent=result
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        spec.attach(learner="econml.dml.LinearDML", effect=0.0, data=_backdoor_data())


def test_graph_posterior_refuses() -> None:
    data = _backdoor_data()
    result = require_success(
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.ExactDagPosterior(),
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Bayesian(n_draws=32, prior_scale=100.0, backend="conjugate"),
            refute=False,
            bootstrap=0,
            seed=7,
        )
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        antecedent.handoff.econml(result, treatment="t", outcome="y")
    spec = antecedent.handoff.EconMLSpec(
        "t", "y", (), "backdoor.adjustment", "NonparametricallyIdentified", _parent=result
    )
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="graph-posterior"):
        spec.attach(learner="econml.dml.LinearDML", effect=0.0, data=_backdoor_data())


def test_temporal_dag_pulse_emits_adjustment_set() -> None:
    graph = antecedent.TemporalDag.from_lagged_edges(
        ["t", "y", "z"],
        [("z", 0, "t", 0), ("z", 1, "y", 0), ("t", 1, "y", 0)],
    )
    query = antecedent.PulseEffect("t", "y", treatment_lag=1)
    identified = antecedent.identify(graph=graph, query=query)
    spec = antecedent.handoff.econml(identified)
    cols = spec.columns(_backdoor_data())
    np.testing.assert_array_equal(cols["T"], _backdoor_data()["t"][:-1])
    np.testing.assert_array_equal(cols["Y"], _backdoor_data()["y"][1:])
    np.testing.assert_array_equal(cols["W"][:, 0], _backdoor_data()["z"][:-1])
    np.testing.assert_array_equal(cols["origins"], np.arange(1, 200))


def test_fully_oriented_cpdag_stays_cpdag_and_emits_w() -> None:
    graph = antecedent.Cpdag.from_directed_undirected(
        ["z", "t", "y"],
        [("z", "t"), ("z", "y"), ("t", "y")],
        [],
    )
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=graph, query=query)
    assert isinstance(identified.graph, antecedent.Cpdag)
    spec = antecedent.handoff.econml(identified)
    assert spec.confounders == ("z",)
    assert spec.identifier == "generalized.adjustment"
    assert "Identified" in spec.status
    assert "Partial" not in spec.status


def test_identify_result_needs_names() -> None:
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    identified = antecedent.identify(graph=_backdoor_graph(), query=query)
    legacy = identified.to_identify_result()
    assert antecedent.handoff.econml(legacy).treatment == "t"
    spec = antecedent.handoff.econml(legacy, treatment="t", outcome="y")
    assert spec.confounders == ("z",)


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros(3), "must have the same number of rows"),
        (np.zeros((200, 1)), "requires one-dimensional columns"),
    ],
)
def test_handoff_rejects_misaligned_columns(bad, message) -> None:
    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect("t", "y")
    )
    data = _backdoor_data()
    data["z"] = bad
    with pytest.raises(antecedent.errors.CausalValueError, match=message):
        antecedent.handoff.econml(identified).columns(data)


def test_multiple_controls_are_sample_major() -> None:
    spec = antecedent.handoff.EconMLSpec(
        "t", "y", ("z", "w"), "backdoor.adjustment", "NonparametricallyIdentified"
    )
    data = _backdoor_data()
    data["w"] = data["z"] ** 2
    cols = spec.columns(data)
    assert cols["W"].shape == (200, 2)
    np.testing.assert_array_equal(cols["W"][:, 0], data["z"])
    np.testing.assert_array_equal(cols["W"][:, 1], data["w"])


@pytest.mark.parametrize("analyzed", [False, True])
def test_relabeling_certificate_is_rejected(analyzed):
    query = antecedent.AverageEffect("t", "y")
    result = (
        require_success(
            antecedent.analyze(
                _backdoor_data(),
                graph=_backdoor_graph(),
                query=query,
                refute=False,
                inference=antecedent.Frequentist(bootstrap=0),
            )
        )
        if analyzed
        else antecedent.identify(graph=_backdoor_graph(), query=query)
    )
    with pytest.raises(antecedent.errors.CausalValueError, match="certified query"):
        antecedent.handoff.econml(result, treatment="z", outcome="y")


def test_joint_handoff_retains_both_treatment_columns():
    from test_joint_class_response import _data, _graph

    query = antecedent.InterventionResponse(
        "y", intervention=[antecedent.intervention.Set("t", 1), antecedent.intervention.Set("z", 1)]
    )
    identified = antecedent.identify(graph=_graph("cpdag", False), query=query)
    spec = antecedent.handoff.econml(identified)
    data = _data()
    cols = spec.columns(data)
    assert spec.treatments == ("t", "z")
    np.testing.assert_array_equal(cols["T"], np.column_stack([data["t"], data["z"]]))
    np.testing.assert_array_equal(cols["W"], data["w"][:, None])
    with pytest.raises(antecedent.errors.CausalValueError, match="every joint treatment"):
        antecedent.handoff.econml(identified, treatment="t")


def test_attach_export_load_keeps_attested_external_estimate() -> None:
    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect("t", "y")
    )
    spec = antecedent.handoff.econml(identified)
    data = _backdoor_data()
    attached = spec.attach(
        learner="econml.dml.CausalForestDML",
        learner_config={"n_estimators": 100},
        effect=np.full(len(data["t"]), 2.0),
        data=data,
    )
    loaded = require_success(antecedent.load(bytes(attached.export())))
    forwarded = require_success(antecedent.load(bytes(loaded.export())))
    attested = loaded.decoded_artifact.contract["claim"]["attested"]
    assert attested
    assert attested[0]["kind"] == "external_estimate"
    assert attested[0]["reverifiable"] is False
    assert attested[0]["name"] == "econml.dml.CausalForestDML"
    assert attested[0]["payload_digest"]
    assert attested[0]["name"] == "econml.dml.CausalForestDML"
    identities = loaded.decoded_artifact.contract["identities"]
    assert identities["identification"]
    assert identities["data_snapshot"]
    assert forwarded.decoded_artifact.contract["identities"] == identities
    assert (
        forwarded.decoded_artifact.contract["claim"]["claim_id"]
        == loaded.decoded_artifact.contract["claim"]["claim_id"]
    )
    assert forwarded.decoded_artifact.contract["claim"]["attested"] == attested
    assert loaded.calibration.status == "unavailable"
    assert loaded.calibration.reason == "attested_not_reverifiable"
    inspect = loaded.inspect()
    assert inspect.calibration.status == "unavailable"
    assert inspect.calibration.reason == "attested_not_reverifiable"
    assert attached.inspect().calibration.status == "unavailable"
    other = spec.attach(
        learner="econml.dml.CausalForestDML",
        learner_config={"n_estimators": 100},
        effect=np.full(len(data["t"]), 2.1),
        data=data,
    )
    assert attached.claim_id != other.claim_id
    again = spec.attach(
        learner="econml.dml.CausalForestDML",
        learner_config={"n_estimators": 100},
        effect=np.full(len(data["t"]), 2.0),
        data=data,
    )
    assert again.claim_id == attached.claim_id
    contract = loaded.decoded_artifact.contract
    assert contract.get("estimator") not in {"econml", "econml.dml.CausalForestDML"}
    assert contract.get("score_reuse") in (None, {})


def test_attach_from_analyzed_parent_does_not_claim_native_calibration() -> None:
    data = _backdoor_data()
    result = require_success(
        antecedent.analyze(
            data,
            graph=_backdoor_graph(),
            query=antecedent.AverageEffect("t", "y"),
            refute=False,
            inference=antecedent.Frequentist(bootstrap=0),
        )
    )
    attached = antecedent.handoff.econml(result).attach(
        learner="econml.dml.CausalForestDML",
        learner_config={"n_estimators": 50},
        effect=1.5,
        label="ate",
    )
    loaded = require_success(antecedent.load(bytes(attached.export())))
    assert loaded.calibration.status == "unavailable"
    assert loaded.calibration.reason == "attested_not_reverifiable"
    claim = loaded.decoded_artifact.contract["claim"]
    assert claim["kind"] == "point"
    assert claim["attested"][0]["kind"] == "external_estimate"


def test_attach_scalar_ate_is_bound_through_external_bind() -> None:
    from antecedent.extensibility import ProviderTrust

    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect("t", "y")
    )
    spec = antecedent.handoff.econml(identified)
    data = _backdoor_data()
    attached = spec.attach(
        learner="econml.dml.LinearDML",
        learner_config={"cv": 3},
        effect=2.0,
        interval=[1.5, 2.5],
        data=data,
        label="ate",
        contrast=(0.0, 1.0),
        learner_version="0.15.1",
        outcome_units="mmHg",
        support=["supported"],
    )
    claim = attached.claim
    assert claim is not None and not claim.native
    assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert attached.value == 2.0 and list(claim.values) == [2.0]
    assert attached.support == ("supported",)
    assert claim.uncertainty_method == "caller_attested_interval"
    assert (
        "econml.dml.LinearDML/" in attached.provider_label and "@0.15.1#" in attached.provider_label
    )
    assert {"causal_contract", "external_provider", "claim"} <= claim.stages_behind()
    (quantity,) = claim.quantities
    assert quantity.units == "mmHg" and quantity.functional_id == "ate"
    assert quantity.regime_id == "do(t=1) - do(t=0)"
    assert bytes(attached.export(quantity="claim"))
    # The attested-estimate receipt is unchanged.
    assert (
        require_success(antecedent.load(bytes(attached.export()))).calibration.status
        == "unavailable"
    )

    bare = spec.attach(learner="x", effect=1.0, data=data, label="ate")
    assert bare.support == ("missing_evidence",)
    assert bare.claim.quantities[0].units == "unspecified"
    assert "@unspecified#" in bare.provider_label

    opaque = spec.attach(learner="x", effect=np.full(len(data["t"]), 2.0), data=data)
    assert opaque.claim is None and opaque.support == () and opaque.lineage == ()
    with pytest.raises(antecedent.errors.CausalValueError, match="label='ate'"):
        spec.attach(learner="x", effect=np.ones(3), data=data, learner_version="1")
    with pytest.raises(antecedent.errors.CausalValueError):
        spec.attach(learner="x", effect=float("nan"), data=data, label="ate")


def test_attach_cate_column_is_bound_with_lineage_support_and_provider_version() -> None:
    from antecedent.extensibility import ProviderTrust
    from antecedent.external import BoundExternalColumn, ExternalSpec

    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect("t", "y")
    )
    spec = antecedent.handoff.econml(identified)
    data = _backdoor_data(5_000)
    rng = np.random.default_rng(3)
    cate = 2.0 + 0.5 * data["z"] + rng.normal(scale=0.01, size=len(data["t"]))
    labels = tuple("weak_overlap" if i % 5 == 0 else "supported" for i in range(len(cate)))
    attached = spec.attach(
        learner="econml.dml.CausalForestDML",
        learner_config={"n_estimators": 50},
        effect=cate,
        data=data,
        label="cate",
        contrast=(0.0, 1.0),
        learner_version="0.15.1",
        outcome_units="mmHg",
        support=labels,
    )
    claim = attached.claim
    assert isinstance(claim, BoundExternalColumn)
    assert not claim.native and claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert claim.rows == len(cate) and claim.dtype == "float64"
    assert claim.support_counts == {"supported": 4_000, "weak_overlap": 1_000}
    assert attached.support == ("weak_overlap",) == claim.support
    assert claim.support_status == "weak_overlap"
    assert "econml.dml.CausalForestDML/cate:y~t@0.15.1#" in attached.provider_label
    assert {"causal_contract", "external_provider", "claim"} <= claim.stages_behind()
    (quantity,) = claim.quantities
    assert quantity.functional_id == "cate" and quantity.units == "mmHg"
    assert quantity.regime_id == "do(t=1) - do(t=0)"
    # The claim names one quantity and keeps the column by its digest, never its rows.
    assert claim.covers(cate) and not claim.covers(cate + 1e-9) and not claim.covers(cate[:-1])
    with pytest.raises(antecedent.errors.CausalUnsupportedError, match="rows"):
        _ = claim.values
    exported = bytes(attached.export(quantity="claim"))
    assert len(exported) < 16 * 1024
    reloaded = ExternalSpec(contract_id="", graph_id="", identification="", quantities=()).load(
        exported, expected_identity=claim.identity
    )
    assert isinstance(reloaded, BoundExternalColumn) and reloaded.covers(cate)
    assert reloaded.identity_fields == claim.identity_fields
    # The attested-estimate receipt is built exactly as for any attach.
    assert (
        require_success(antecedent.load(bytes(attached.export()))).calibration.status
        == "unavailable"
    )
    plain = spec.attach(learner="econml.dml.CausalForestDML", effect=cate, data=data)
    assert plain.claim is None

    uniform = spec.attach(
        learner="x", effect=cate, data=data, label="cate", support="supported"
    ).claim
    assert uniform.support_counts == {"supported": len(cate)}
    missing = spec.attach(learner="x", effect=cate, data=data, label="cate").claim
    assert missing.support_counts == {"missing_evidence": len(cate)}
    assert missing.support_status == "missing_evidence"


def test_attach_refusals_are_typed_with_reason_codes_and_remedies() -> None:
    from antecedent.errors import CausalUnsupportedError, CausalValueError
    from antecedent.extensibility import ProviderRefusal

    identified = antecedent.identify(
        graph=_backdoor_graph(), query=antecedent.AverageEffect("t", "y")
    )
    spec = antecedent.handoff.econml(identified)
    data = _backdoor_data(20)

    def refused(call, detail, code):
        with pytest.raises(ProviderRefusal) as caught:
            call()
        error = caught.value
        assert error.detail == detail
        assert error.reason_code == code
        assert error.remedy
        assert error.stage == "handoff"
        return error

    error = refused(
        lambda: spec.attach(learner="x", effect=np.ones(20), data=data, label="ate"),
        "handoff.label_ate_not_scalar",
        "invalid_argument",
    )
    assert isinstance(error, CausalValueError) and isinstance(error, ValueError)
    refused(
        lambda: spec.attach(learner="x", effect=np.ones(7), data=data, label="cate"),
        "handoff.cate_rows_misaligned",
        "schema_mismatch",
    )
    refused(
        lambda: spec.attach(learner="x", effect=np.ones((20, 2)), data=data, label="cate"),
        "handoff.label_cate_not_column",
        "invalid_argument",
    )
    refused(
        lambda: spec.attach(learner="x", effect=1.0, data=data, label="att"),  # type: ignore[arg-type]
        "handoff.label_unknown",
        "invalid_argument",
    )
    refused(
        lambda: spec.attach(learner="x", effect=np.full(20, np.nan), data=data, label="cate"),
        "handoff.effect_not_finite",
        "invalid_argument",
    )
    refused(
        lambda: spec.columns({"t": data["t"]}),
        "handoff.columns_missing",
        "unknown_variable",
    )
    opaque = spec.attach(learner="x", effect=np.ones(20), data=data)
    error = refused(opaque.export_claim, "handoff.no_bound_claim", "attested_not_reverifiable")
    assert isinstance(error, CausalUnsupportedError)
