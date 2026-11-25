import warnings
from dataclasses import replace

import numpy as np
import pytest
from antecedent.extensibility import (
    CausalProviderSpec,
    FoldPolicy,
    ProviderCapability,
    ProviderExecution,
    ProviderRegistry,
    ProviderTrust,
    ProviderVerificationFixture,
    connect_to_decision,
)
from antecedent.interface_contract import InterfaceContractRefusal
from antecedent.joint_distribution import ScientificQuantity
from antecedent.external_learners import (
    ExternalLearner,
    ForecastNuisanceAdapter,
    Learners,
    aipw_ate,
    analyze_ate,
    causal_forest_ate,
    cross_fit_aipw,
    declare_portable_route,
    dml_ate,
    dr_learner_ate,
)


class MeanProvider:
    spec = CausalProviderSpec(
        query_family="nuisance_learner",
        identification_requirements=("caller_identifies",),
        observed_distributions=("X,Y",),
        nuisance_functions=("mean",),
        support_conditions=("finite",),
        data_dependence=("training_rows", "sample_weights"),
        inference_claims=("point_only",),
        influence_function=None,
        fold_policy="caller_owned",
        output_shape="one per prediction row",
        uncertainty_semantics="point_only",
        artifact_codec="ascii",
        deterministic=True,
        provenance={"package": "test", "version": "1"},
        capabilities=(ProviderCapability.MEAN,),
    )

    def execute(self, request):
        if request["operation"] == "fit":
            mean = np.average(request["y"], weights=request["weights"])
            return ProviderExecution(
                [], None, (), "supported", {"version": "1"}, str(mean).encode()
            )
        import base64

        value = float(base64.b64decode(request["model_base64"]))
        return ProviderExecution(
            np.full(len(request["x"]), value), None, (), "supported", {"version": "1"}
        )


def test_fit_predict_use_separate_registry_requests_and_validate_weights():
    registry = ProviderRegistry()
    provider = MeanProvider()
    registry.register("mean", provider)
    learner = ExternalLearner(registry, "mean", "regression", supports_weights=True)
    fitted = learner.fit([[1], [2]], [1, 3], [1, 3])
    prediction = fitted.predict([[4], [5]])
    assert prediction.estimate.tolist() == [2.5, 2.5]
    assert fitted.fit_receipt.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert prediction.trust is ProviderTrust.EXTERNALLY_ATTESTED
    with pytest.raises(ValueError, match="feature count"):
        fitted.predict([[1, 2]])
    provider.spec = replace(provider.spec, provenance={"package": "test", "version": "2"})
    with pytest.raises(ValueError, match="changed since fit"):
        fitted.predict([[1]])


def test_probability_prediction_outside_unit_interval_is_refused():
    class InvalidProbability(MeanProvider):
        def execute(self, request):
            result = super().execute(request)
            if request["operation"] == "predict":
                return replace(result, estimate=[1.1] * len(request["x"]))
            return result

    registry = ProviderRegistry()
    registry.register("mean", InvalidProbability())
    fitted = ExternalLearner(registry, "mean", "binary_probability").fit([[1], [2]], [0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fitted.predict([[3]])
    with pytest.raises(ValueError, match="0 or 1"):
        ExternalLearner(registry, "mean", "binary_probability").fit([[1]], [2])


def test_provider_cannot_change_fit_request_or_hide_callback_failure():
    class MutatingProvider(MeanProvider):
        def execute(self, request):
            result = super().execute(request)
            request["x"].flags.writeable = True
            request["x"][0, 0] = 100
            return result

    class FailingProvider(MeanProvider):
        def execute(self, request):
            raise RuntimeError("external model failed")

    registry = ProviderRegistry()
    registry.register("mutating", MutatingProvider())
    registry.register("failing", FailingProvider())
    with pytest.raises(ValueError, match="changed its fit request"):
        ExternalLearner(registry, "mutating", "regression").fit([[1]], [1])
    with pytest.raises(RuntimeError, match="external model failed"):
        ExternalLearner(registry, "failing", "regression").fit([[1]], [1])


def test_verification_applies_only_to_the_exact_prediction_request():
    registry = ProviderRegistry()
    registry.register("mean", MeanProvider())
    request = {
        "operation": "predict",
        "version": 1,
        "task": "binary_probability",
        "configuration": {},
        "x": np.array([[3.0]]),
        "model_base64": "MC41",
    }
    fixture = ProviderVerificationFixture(
        name="prediction",
        request=request,
        expected_estimate=[0.5],
        expected_uncertainty=None,
        expected_assumptions=(),
        expected_support_status="supported",
        expected_provenance={"package": "test", "version": "1"},
    )
    registry.verify("mean", [fixture], evidence_origin="independent test")
    fitted = ExternalLearner(registry, "mean", "binary_probability").fit([[1], [2]], [0, 1])
    assert fitted.fit_receipt.trust is ProviderTrust.EXTERNALLY_ATTESTED
    verified = fitted.predict([[3]])
    assert verified.trust is ProviderTrust.VERIFIED_EXTENSION
    assert verified.trust is not ProviderTrust.NATIVE_SUPPORTED
    assert fitted.predict([[4]]).trust is ProviderTrust.EXTERNALLY_ATTESTED


def test_unhonoured_fold_policy_is_refused_when_the_learner_is_declared():
    class Temporal(MeanProvider):
        spec = replace(MeanProvider.spec, fold_policy="origin_scoped")

    registry = ProviderRegistry()
    registry.register("temporal", Temporal())
    assert Temporal.spec.fold_policy is FoldPolicy.ORIGIN_SCOPED
    assert not Temporal.spec.fold_policy.estimator_honours()
    with pytest.raises(ValueError, match="temporal cross-fitting"):
        ExternalLearner(registry, "temporal", "regression")
    decoded = replace(
        MeanProvider.spec, fold_policy="external_learner_owned_not_recorded_by_handoff"
    )
    assert decoded.fold_policy is FoldPolicy.HANDOFF_UNKNOWN


def test_forecast_mean_adapter_requires_an_explicit_iid_information_contract():
    class ForecastMean(MeanProvider):
        spec = replace(
            MeanProvider.spec,
            query_family="forecast",
            data_dependence=("iid_row_conditional_mean",),
        )

    registry = ProviderRegistry()
    registry.register("forecast", ForecastMean())
    with pytest.raises(InterfaceContractRefusal) as nuisance:
        ExternalLearner(registry, "forecast", "regression").fit([[1]], [2])
    decision = ScientificQuantity.of(
        "outcome", "effect", units="1", population="trial", regime="do(a=1)"
    )
    with pytest.raises(InterfaceContractRefusal) as law:
        connect_to_decision(ForecastMean.spec, decision, ("law",))
    assert nuisance.value.reason_code.startswith("interface_contract")
    assert law.value.reason_code == "interface_contract_operation_unavailable"
    adapter = ForecastNuisanceAdapter(registry, "forecast", "regression")
    fitted = adapter.fit([[1]], [2])
    assert fitted.fold_policy is FoldPolicy.CALLER_OWNED
    assert fitted.predict([[3]]).estimate.tolist() == [2]
    incompatible = ForecastNuisanceAdapter(
        registry, "forecast", "regression", information_contract="temporal"
    )
    with pytest.raises(InterfaceContractRefusal):
        incompatible.fit([[1]], [2])


def test_native_crossfit_isolates_folds_and_treatment_arms():
    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression", supports_weights=True)
    treatment = ExternalLearner(registry, "treatment", "binary_probability", supports_weights=True)
    x = [[float(i)] for i in range(8)]
    y = [1, 3, 5, 7, 9, 11, 13, 15]
    t = [0, 1, 0, 1, 0, 1, 0, 1]
    folds = [0, 0, 1, 1, 0, 0, 1, 1]
    result = cross_fit_aipw(outcome, treatment, x, y, t, folds, weights=[1] * 8)
    for row in range(8):
        train = [i for i in range(8) if folds[i] != folds[row]]
        assert result.control[row] == np.mean([y[i] for i in train if t[i] == 0])
        assert result.treated[row] == np.mean([y[i] for i in train if t[i] == 1])
        assert result.propensity[row] == 0.5
    assert result.fold_policy is FoldPolicy.CALLER_OWNED
    assert len(result.receipts) == 6
    assert len(result.provenance) == 6
    assert {int(p[1].split(":")[1]) for p in result.provenance} == {r[0] for r in result.receipts}
    with pytest.raises(ValueError, match="temporal cross-fitting"):
        cross_fit_aipw(outcome, treatment, x, y, t, folds, fold_policy="temporal")


def test_native_crossfit_preserves_callback_failure_and_cancellation():
    from antecedent import _native
    from antecedent.execution import Execution, ResourceLimits

    class Broken(MeanProvider):
        def execute(self, request):
            raise RuntimeError("fit failed in provider")

    registry = ProviderRegistry()
    registry.register("broken", Broken())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "broken", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    x, y, t, folds = [[1], [2], [3], [4]], [1, 2, 3, 4], [0, 1, 0, 1], [0, 0, 1, 1]
    with pytest.raises(RuntimeError, match="fit failed in provider"):
        cross_fit_aipw(outcome, treatment, x, y, t, folds)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(Exception, match="cancel"):
        cross_fit_aipw(outcome, treatment, x, y, t, folds, execution=Execution(cancel=token))
    with pytest.raises(Exception, match="admission|memory|budget"):
        cross_fit_aipw(
            outcome,
            treatment,
            x,
            y,
            t,
            folds,
            execution=Execution(limits=ResourceLimits(memory_bytes=1)),
        )
    shared = Execution(threads=4, limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(Exception, match="admission|memory|budget"):
        shared._call(cross_fit_aipw, outcome, treatment, x, y, t, folds)


def test_native_crossfit_rejects_invalid_provider_probabilities():
    class InvalidProbability(MeanProvider):
        def execute(self, request):
            result = super().execute(request)
            if request["operation"] == "predict":
                return replace(result, estimate=[-0.1] * len(request["x"]))
            return result

    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", InvalidProbability())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cross_fit_aipw(
            outcome, treatment, [[1], [2], [3], [4]], [1, 2, 3, 4], [0, 1, 0, 1], [0, 0, 1, 1]
        )


def test_native_crossfit_rejects_a_provider_specification_change_between_fits():
    class ChangingProvider(MeanProvider):
        def __init__(self):
            self.fit_count = 0

        def execute(self, request):
            result = super().execute(request)
            if request["operation"] == "fit":
                self.fit_count += 1
                if self.fit_count == 2:
                    self.spec = replace(self.spec, provenance={"package": "test", "version": "2"})
            return result

    registry = ProviderRegistry()
    registry.register("outcome", ChangingProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    with pytest.raises(ValueError, match="specification changed"):
        cross_fit_aipw(
            outcome, treatment, [[1], [2], [3], [4]], [1, 2, 3, 4], [0, 1, 0, 1], [0, 0, 1, 1]
        )


def test_native_crossfit_cancellation_after_callback_returns_no_result():
    import threading

    from antecedent import _native
    from antecedent.execution import Execution

    entered, release = threading.Event(), threading.Event()

    class BlockingProvider(MeanProvider):
        def execute(self, request):
            entered.set()
            assert release.wait(10)
            return super().execute(request)

    registry = ProviderRegistry()
    registry.register("outcome", BlockingProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    token = _native.CancellationToken()
    captured = []

    def run():
        try:
            captured.append(
                cross_fit_aipw(
                    outcome,
                    treatment,
                    [[1], [2], [3], [4]],
                    [1, 2, 3, 4],
                    [0, 1, 0, 1],
                    [0, 0, 1, 1],
                    execution=Execution(cancel=token),
                )
            )
        except Exception as error:
            captured.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(10)
        token.cancel()
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert len(captured) == 1 and isinstance(captured[0], Exception)
    assert "cancel" in str(captured[0]).lower()


def test_native_crossfit_serializes_callbacks_under_parallel_execution_policy():
    import threading
    import time

    from antecedent.execution import Execution

    lock = threading.Lock()
    active = 0
    maximum = 0

    class ObservedProvider(MeanProvider):
        def execute(self, request):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.002)
                return super().execute(request)
            finally:
                with lock:
                    active -= 1

    registry = ProviderRegistry()
    registry.register("outcome", ObservedProvider())
    registry.register("treatment", ObservedProvider())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    execution = Execution(threads=4)
    execution._call(
        cross_fit_aipw,
        outcome,
        treatment,
        [[i] for i in range(8)],
        list(range(8)),
        [0, 1] * 4,
        [0, 0, 1, 1, 0, 0, 1, 1],
    )
    assert maximum == 1


def test_registry_serializes_callbacks_across_concurrent_runs():
    import threading
    import time

    lock = threading.Lock()
    active = 0
    maximum = 0

    class ObservedProvider(MeanProvider):
        def execute(self, request):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.002)
                return super().execute(request)
            finally:
                with lock:
                    active -= 1

    registry = ProviderRegistry()
    registry.register("outcome", ObservedProvider())
    registry.register("treatment", ObservedProvider())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    failures = []

    def run():
        try:
            cross_fit_aipw(
                outcome,
                treatment,
                [[i] for i in range(8)],
                list(range(8)),
                [0, 1] * 4,
                [0, 0, 1, 1, 0, 0, 1, 1],
            )
        except Exception as error:
            failures.append(error)

    workers = [threading.Thread(target=run) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
    assert not any(worker.is_alive() for worker in workers)
    assert not failures
    assert maximum == 1


def test_external_dml_identifies_graph_and_fits_registered_providers():
    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression", supports_weights=True)
    treatment = ExternalLearner(registry, "treatment", "binary_probability", supports_weights=True)
    t = [float(i % 2) for i in range(40)]
    data = {"t": t, "y": [1 + 2 * value for value in t]}
    result = dml_ate(
        data,
        [("t", "y")],
        "t",
        "y",
        outcome=outcome,
        treatment=treatment,
        folds=2,
        weights=[1] * 40,
    )
    assert result.estimate == pytest.approx(2.0)
    assert result.adjustment_set == ()
    assert len(result.row_ids) == len(result.fold_ids) == 40
    assert any("dml.caller_factory.unverified" in str(item) for item in result.assumptions)
    assert len(result.receipts) == len(result.provenance) == 6
    plr = dml_ate(
        data,
        [("t", "y")],
        "t",
        "y",
        outcome=outcome,
        treatment=treatment,
        folds=2,
        score="partially_linear",
    )
    assert plr.estimate == pytest.approx(2.0)
    assert len(plr.receipts) == len(plr.provenance) == 4
    aipw = aipw_ate(
        data, [("t", "y")], "t", "y", outcome=outcome, treatment=treatment, weights=[1] * 40
    )
    assert aipw.estimate == pytest.approx(2.0)
    assert len(aipw.receipts) == len(aipw.provenance) == 15
    dr = dr_learner_ate(data, [("t", "y")], "t", "y", outcome=outcome, treatment=treatment, folds=2)
    assert dr.estimate == pytest.approx(2.0)
    assert len(dr.cate) == 40
    assert dr.cate == pytest.approx([2.0] * 40)
    assert len(dr.provenance) == len(dr.receipts) + 1


def test_installed_entry_point_runs_through_native_dml(tmp_path, monkeypatch):
    module = tmp_path / "external_fixture.py"
    module.write_text(
        "from test_external_learners import MeanProvider\ndef create(): return MeanProvider()\n"
    )
    dist = tmp_path / "external_fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: external-fixture\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[antecedent.providers]\nfixture = external_fixture:create\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = ProviderRegistry()
    registry.load_entry_point("fixture")
    outcome = ExternalLearner(registry, "fixture", "regression")
    treatment = ExternalLearner(registry, "fixture", "binary_probability")
    arms = [float(i % 2) for i in range(40)]
    result = dml_ate(
        {"t": arms, "y": [1 + 2 * v for v in arms]},
        [("t", "y")],
        "t",
        "y",
        outcome=outcome,
        treatment=treatment,
        folds=2,
    )
    assert result.estimate == pytest.approx(2)
    assert all(
        fit.provenance["entry_point"] == "external_fixture:create" for _, fit, _ in result.receipts
    )


def test_causal_forest_uses_external_factories_for_its_marginal_ate():
    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", MeanProvider())
    outcome = ExternalLearner(registry, "outcome", "regression")
    treatment = ExternalLearner(registry, "treatment", "binary_probability")
    arms = [float(i % 2) for i in range(240)]
    data = {"z": [float(i // 2) for i in range(240)], "t": arms, "y": [1 + 2 * a for a in arms]}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    result = causal_forest_ate(data, graph, "t", "y", outcome=outcome, treatment=treatment)
    assert result.estimate == pytest.approx(2.0)
    assert len(result.cate) == 240
    assert len(result.receipts) == 15


def _binary_study():
    registry = ProviderRegistry()
    registry.register("outcome", MeanProvider())
    registry.register("treatment", MeanProvider())
    learners = Learners(
        ExternalLearner(registry, "outcome", "regression", supports_weights=True),
        ExternalLearner(registry, "treatment", "binary_probability", supports_weights=True),
    )
    arms = [float(i % 2) for i in range(40)]
    data = {"z": [float(i) for i in range(40)], "t": arms, "y": [1 + 2 * arm for arm in arms]}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    return data, graph, learners


def test_analyze_fits_declared_learners_as_a_study_setting():
    import antecedent as ant
    from antecedent.estimators import learner_estimator
    from antecedent.query import AverageEffect

    assert learner_estimator("dml") == ("dml", "partially_linear")
    assert learner_estimator("dr.learner") == ("dr_learner", "aipw")
    assert learner_estimator("causal.forest") == ("causal_forest", "aipw")
    data, graph, learners = _binary_study()
    query = AverageEffect("t", "y")
    outcome = ant.analyze(data, graph=graph, query=query, learners=learners, estimator="dml")
    assert outcome.refusal is None
    assert outcome.answer.value == pytest.approx(2.0)
    assert outcome.value.estimate.estimator_id == "dml"
    assert outcome.value.estimate.method == "partially_linear"
    assert outcome.value.identification.adjustment_set == ("z",)
    assert outcome.value.provenance["fold_receipts"]
    assert outcome.value.provenance["fold_ids"]
    study = ant.prepare(data, graph=graph, query=query, learners=learners, estimator="aipw")
    aipw = ant.estimate(study)
    assert aipw.answer.value == pytest.approx(2.0)
    assert aipw.value.estimate.method == "aipw"
    assert aipw.value.identification.adjustment_set == ("z",)
    with pytest.raises(TypeError):
        ant.analyze(data, graph=graph, query=query, learners=learners, score="aipw")
    with pytest.raises(ValueError, match="aipw, dml"):
        ant.analyze(data, graph=graph, query=query, learners=learners, estimator="linear.adjustment.ate")
    kept = analyze_ate(
        data, graph, "t", "y", learners=learners, estimator="dml", score="partially_linear", folds=2
    )
    assert kept.estimate == pytest.approx(2.0)
    with pytest.raises(ValueError, match="score"):
        analyze_ate(data, graph, "t", "y", learners=learners, estimator="aipw", score="partially_linear")


def test_direct_entries_name_analyze_and_keep_the_legacy_score():
    from antecedent._deprecation import AntecedentDeprecationWarning
    from antecedent.external_learners import dml_ate

    data, graph, learners = _binary_study()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AntecedentDeprecationWarning)
        dml_ate(
            data,
            graph,
            "t",
            "y",
            outcome=learners.outcome,
            treatment=learners.treatment,
            folds=2,
            score="partially_linear",
        )
    messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, AntecedentDeprecationWarning)
    ]
    assert messages
    assert all("antecedent.analyze" in message for message in messages)
    assert all("analyze_ate" not in message for message in messages)


def test_portable_learners_are_admitted_only_on_artifact_routes():
    import antecedent as ant
    from antecedent.errors import CausalUnsupportedError
    from antecedent.query import AverageEffect

    class PortableMean(MeanProvider):
        spec = replace(MeanProvider.spec, artifact_codec="portable_prediction_map_v1")

    portable_registry = ProviderRegistry()
    portable_registry.register("outcome", PortableMean())
    portable_registry.register("treatment", PortableMean())
    portable = Learners(
        ExternalLearner(portable_registry, "outcome", "regression"),
        ExternalLearner(portable_registry, "treatment", "binary_probability"),
    )
    assert portable.outcome.portable and portable.treatment.portable
    for route in ("smoothed_dose", "trial_aipw", "joint_cells"):
        assert declare_portable_route(route, portable) is portable

    data, graph, learners = _binary_study()
    assert learners.outcome.portable is False
    for route in ("smoothed_dose", "trial_aipw", "joint_cells"):
        with pytest.raises(CausalUnsupportedError, match="dose_response.learner_not_portable") as caught:
            declare_portable_route(route, learners)
        assert caught.value.reason_code == "route_not_supported"
    outcome = ant.analyze(
        data,
        graph=graph,
        query=AverageEffect("t", "y"),
        learners=learners,
        estimator="aipw",
    )
    assert outcome.answer.value == pytest.approx(2.0)
