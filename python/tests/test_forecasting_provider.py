from dataclasses import replace

import pytest
from antecedent import forecasting as fc
from antecedent import predictive_assessment as pa
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderCapability,
    ProviderExecution,
    ProviderRegistry,
    ProviderTrust,
    ProviderVerificationFixture,
)


class SampleForecaster:
    spec = CausalProviderSpec(
        query_family="forecast",
        identification_requirements=("predictive_only",),
        observed_distributions=("history",),
        nuisance_functions=(),
        support_conditions=("history",),
        data_dependence=("origin_history",),
        inference_claims=("sample_law",),
        influence_function=None,
        fold_policy="origin_scoped",
        output_shape="samples",
        uncertainty_semantics="sample_trajectories",
        artifact_codec="none",
        deterministic=True,
        provenance={"package": "test", "version": "1"},
        capabilities=(ProviderCapability.SAMPLE,),
    )

    def __init__(self):
        self.origins = []

    def execute(self, request):
        self.origins.append(request["case"]["origin"])
        return ProviderExecution([[1, 2], [2, 3]], None, (), "supported", {"version": "1"})


def test_origin_scoped_joint_forecast_preserves_provider_trust_and_unknown_lineage():
    provider = SampleForecaster()
    registry = ProviderRegistry()
    registry.register("sample", provider)
    keys = pa.keys("forecast", [1, 2])
    target = pa.Target("temperature", "site", "C", horizon_steps=2)
    first = fc.forecast(
        registry,
        "sample",
        fc.ForecastOrigin(2, [1, 2], [[0], [1]], keys),
        target=target,
        kind="joint_samples",
    )
    second = fc.forecast(
        registry,
        "sample",
        fc.ForecastOrigin(3, [1, 2, 3], [[0], [1], [2]], [*keys, pa.keys("forecast", [3])[0]]),
        target=target,
        kind="joint_samples",
    )
    assert provider.origins == [2, 3]
    assert first.execution.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert first.predictor("forecast").laws[0].to_wire()["dimension"] == 2
    assert first.fit.wire["kind"] == "undeclared"
    assert second.validation_design().wire["origins"] == [3]
    with pytest.raises(ValueError, match="not exceed"):
        fc.forecast(
            registry,
            "sample",
            fc.ForecastOrigin(1, [2], [[1]], [keys[0]]),
            target=target,
            kind="joint_samples",
        )


def test_forecast_rejects_changed_specification_and_point_only_uncertainty():
    class ChangesSpec(SampleForecaster):
        def execute(self, request):
            result = super().execute(request)
            self.spec = replace(self.spec, provenance={"package": "test", "version": "2"})
            return result

    registry = ProviderRegistry()
    registry.register("sample", ChangesSpec())
    target = pa.Target("x", "site", "m", horizon_steps=2)
    origin = fc.ForecastOrigin(1, [1], [[1]], pa.keys("x", [1]))
    with pytest.raises(ValueError, match="changed during execution"):
        fc.forecast(registry, "sample", origin, target=target, kind="joint_samples")


def test_forecast_refuses_untimed_or_future_training_lineage_before_execution():
    provider = SampleForecaster()
    registry = ProviderRegistry()
    registry.register("sample", provider)
    key = pa.keys("lineage", [1])[0]
    origin = fc.ForecastOrigin(1, [1], [[1]], [key])
    target = pa.Target("x", "site", "m", horizon_steps=2)
    with pytest.raises(ValueError, match="training times"):
        fc.forecast(
            registry,
            "sample",
            origin,
            target=target,
            kind="joint_samples",
            prior_training_keys=[key],
        )
    with pytest.raises(ValueError, match="not exceed"):
        fc.forecast(
            registry,
            "sample",
            origin,
            target=target,
            kind="joint_samples",
            prior_training_keys=[key],
            prior_training_times=[2],
        )
    assert provider.origins == []


def test_forecast_builds_an_assessable_single_origin_law():
    registry = ProviderRegistry()
    registry.register("sample", SampleForecaster())
    target = pa.Target("x", "site", "m", horizon_steps=2)
    origin = fc.ForecastOrigin(1, [1], [[1]], pa.keys("x", [1]))
    result = fc.forecast(
        registry, "sample", origin, target=target, kind="joint_samples", prior_training_keys=[]
    )
    observations = pa.Observations(target, pa.keys("x", [3]), "vector", [[1.5, 2.5]])
    assessment = pa.assess(
        observations,
        [result.predictor("sample")],
        result.validation_design(),
        scores=["energy_score"],
        require_held_out=True,
    )
    assert assessment.report.predictor("sample").score("energy_score").mean >= 0


def test_point_forecast_is_not_promoted_to_a_distribution():
    class PointForecaster(SampleForecaster):
        spec = replace(
            SampleForecaster.spec,
            capabilities=(ProviderCapability.MEAN,),
            uncertainty_semantics="point_only",
        )

        def execute(self, request):
            return ProviderExecution([2.0], None, (), "supported", {"version": "1"})

    registry = ProviderRegistry()
    registry.register("point", PointForecaster())
    target = pa.Target("x", "site", "m", horizon_steps=1)
    origin = fc.ForecastOrigin(1, [1], [[1]], pa.keys("x", [1]))
    result = fc.forecast(registry, "point", origin, target=target, kind="point")
    assert result.execution.estimate.tolist() == [2.0]
    with pytest.raises(ValueError, match="not a predictive law"):
        result.predictor("point")


def test_rolling_batch_does_not_promote_one_verified_origin():
    registry = ProviderRegistry()
    registry.register("sample", SampleForecaster())
    keys = pa.keys("forecast-verification", [1, 2])
    target = pa.Target("x", "site", "m", horizon_steps=2)
    origins = [
        fc.ForecastOrigin(1, [1], [[1]], keys[:1]),
        fc.ForecastOrigin(2, [1, 2], [[1], [2]], keys),
    ]
    request = {
        "operation": "forecast",
        "version": 1,
        "target": target.to_wire(),
        "kind": "joint_samples",
        "case": origins[0]._wire(2),
        "training_lineage": None,
        "configuration": {},
    }
    fixture = ProviderVerificationFixture(
        name="first-origin",
        request=request,
        expected_estimate=[[1, 2], [2, 3]],
        expected_uncertainty=None,
        expected_assumptions=(),
        expected_support_status="supported",
        expected_provenance={"package": "test", "version": "1"},
    )
    registry.verify("sample", [fixture], evidence_origin="independent fixture")
    result = fc.rolling(registry, "sample", origins, target=target, kind="joint_samples")
    assert result.results[0].execution.trust is ProviderTrust.VERIFIED_EXTENSION
    assert result.results[1].execution.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert result.source.wire["trust"] == "externally_attested"


def test_rolling_forecasts_compare_each_outcome_only_with_its_own_fit_and_replay():
    registry = ProviderRegistry()
    registry.register("sample", SampleForecaster())
    target = pa.Target("x", "site", "m", horizon_steps=1)
    keys = pa.keys("rolling", [1, 2, 3, 4])
    origins = [
        fc.ForecastOrigin(i, list(range(1, i + 1)), [[float(j)] for j in range(1, i + 1)], keys[:i])
        for i in range(1, 4)
    ]
    batch = fc.rolling(
        registry,
        "sample",
        origins,
        target=target,
        kind="joint_samples",
        prior_training_keys=[[], [], []],
    )
    observed = pa.Observations(target, keys[1:], "vector", [[1, 2], [2, 3], [3, 4]])
    result = pa.assess(
        observed,
        [batch.predictor("forecast")],
        batch.validation_design(),
        scores=["energy_score"],
        dependence=pa.Dependence.serial([1, 2, 3], 1),
        require_held_out=True,
    )
    assert len(batch.results) == 3
    artifact = bytes(result.export("rolling-provider"))
    replayed = pa.consume_predictive_assessment_artifact(artifact, result.identity)
    assert replayed.identity == result.identity
    with pytest.raises(ValueError, match="different times"):
        fc.rolling(
            registry,
            "sample",
            origins,
            target=target,
            kind="joint_samples",
            prior_training_keys=[[keys[1]], [], []],
            prior_training_times=[[1], [], []],
        )
    future_leak = fc.rolling(
        registry,
        "sample",
        origins,
        target=target,
        kind="joint_samples",
        prior_training_keys=[[keys[3]], [], []],
        prior_training_times=[[1], [], []],
    )
    with pytest.raises(Exception, match="assessed outcome unavailable"):
        pa.assess(
            observed,
            [future_leak.predictor("future")],
            future_leak.validation_design(),
            scores=["energy_score"],
            require_held_out=True,
        )
    unknown = fc.rolling(registry, "sample", origins, target=target, kind="joint_samples")
    with pytest.raises(Exception, match="unknown|declared|lineage"):
        pa.assess(
            observed,
            [unknown.predictor("unknown")],
            unknown.validation_design(),
            scores=["energy_score"],
            require_held_out=True,
        )
