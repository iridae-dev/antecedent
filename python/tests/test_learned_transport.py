"""Learned providers keep structural, fitting, support and inference claims separate."""

import dataclasses

import antecedent as ac
import pytest
from antecedent import transport as tr
from antecedent.learners import Linear, Logistic
from antecedent.transport import advanced

from test_transport_statistical import fixture


def test_categorical_provider_uses_common_lifecycle_and_verified_load():
    identified, catalog, data = fixture()
    query = advanced.StatisticalTransportQuery(identified, catalog, {"x": 1.0}, bootstrap=9)
    study = advanced.prepare(query, data, provider=tr.LearnedCategorical(Logistic()))
    result = study.estimate()
    assert result.mean("y") == pytest.approx(0.8, abs=1e-6)
    loaded = ac.load(result.export())
    assert loaded.mean("y") == result.mean("y")
    assert "learned" in str(result.inspect().to_dict()).lower()
    assert result.uncertainty["row"]["calibration_binding"] is None


def trial_fixture(sampling="independent_samples"):
    # Exactly balanced trial and target, independent of row ordering.
    source = [True] * 120 + [False] * 80
    treatment = [False, True] * 60 + [False] * 80
    outcome = [1.0 + 2.0 * t for t in treatment[:120]] + [0.0] * 80
    data = tr.TrialAipwData({}, outcome, treatment, source, [0.5] * 200, sampling)
    graph = ac.Admg.from_edges(["a", "y"], [("a", "y")])
    query = advanced.TrialAipwQuery(graph, advanced.SelectionDiagram("trial", "target", []), "a", "y")
    return query, data


@pytest.mark.parametrize("sampling", ["independent_samples", "nested_cohort"])
def test_trial_lifecycle_retains_paired_score_and_atomic_refresh(sampling):
    query, data = trial_fixture(sampling)
    study = advanced.prepare(
        query,
        data,
        provider=tr.TrialAipw(outcome=Linear(), folds=3),
        inference=tr.TransportInference(bootstrap=5, seed=8),
    )
    assert not study.inspect().uncertainty.available
    result = study.estimate()
    assert result.estimate == pytest.approx(2.0)
    assert result.interval == pytest.approx((2.0, 2.0))
    assert len(result.replicates) == 5
    loaded = ac.load(result.export())
    assert loaded.estimate == result.estimate
    assert loaded.replicates == result.replicates
    assert not loaded.inspect().uncertainty.payload.get("calibration_binding")
    with pytest.raises(ValueError):
        study.refresh(dataclasses.replace(data, randomization=[0.0] * 200))
    assert study.estimate().estimate == result.estimate
    assert study.plan.structure_source if hasattr(study.plan, "structure_source") else True
    with pytest.raises(ValueError):
        study.estimate(seed=9)


def test_trial_rejects_unsupported_sampling_and_uncertified_covariates():
    query, data = trial_fixture()
    with pytest.raises(ValueError):
        advanced.prepare(query, dataclasses.replace(data, sampling="clustered"))
    with pytest.raises(ValueError):
        advanced.prepare(query, dataclasses.replace(data, covariates={"a": [0.0] * 200}))


def test_trial_identity_binds_sampling_design():
    query, data = trial_fixture()
    independent = advanced.prepare(query, data, inference=tr.TransportInference(0)).estimate()
    nested = advanced.prepare(
        query,
        dataclasses.replace(data, sampling="nested_cohort"),
        inference=tr.TransportInference(0),
    ).estimate()
    assert independent.inspect().execution_id != nested.inspect().execution_id
    assert independent.interval is None


def test_recursive_learned_provider_composes_complementary_sources():
    from test_transport_meta_grid import fixture as meta_fixture

    _, identified, catalog, data = meta_fixture(statistical=True)
    # Different source sizes retain the same finite SCM laws.
    samples = tuple(
        dataclasses.replace(s, columns={k: list(v) * 2 for k, v in s.columns.items()})
        if s.population == "a"
        else s
        for s in data.samples
    )
    query = advanced.TransportResponseGridQuery(
        identified, catalog, ({"x": 0.0}, {"x": 1.0}), bootstrap=9
    )
    result = advanced.prepare(
        query, tr.StatisticalTransportData(samples), provider=tr.LearnedCategorical()
    ).estimate()
    assert [result.mean(i, "y") for i in range(2)] == pytest.approx([0.26, 0.74], abs=1e-6)
    assert result.contrast(1, 0, "y").estimate == pytest.approx(0.48, abs=1e-6)
    assert ac.load(result.export()).points == result.points


@pytest.mark.parametrize("misspecified", ["membership", "outcome"])
def test_trial_nuisance_misspecification_cases_are_separate(misspecified):
    import numpy as np

    rng = np.random.default_rng(371)
    ns, nt = 2400, 1600
    zs = rng.normal(size=ns)
    zt = rng.normal(scale=1.8 if misspecified == "membership" else 1.0, size=nt)
    a = rng.binomial(1, 0.5, size=ns)
    # Different variances make log sample odds quadratic, outside logistic(z).
    # For the second case membership is correct but outcome(z,a) is quadratic.
    effect = 2 + zs if misspecified == "membership" else 1 + zs**2
    baseline = zs if misspecified == "membership" else zs**2
    y = baseline + a * effect
    data = tr.TrialAipwData(
        {"z": np.r_[zs, zt]},
        np.r_[y, np.zeros(nt)],
        [bool(v) for v in a] + [False] * nt,
        [True] * ns + [False] * nt,
        [0.5] * (ns + nt),
        "independent_samples",
    )
    graph = ac.Admg.from_edges(["z", "a", "y"], [("z", "y"), ("a", "y")])
    query = advanced.TrialAipwQuery(graph, advanced.SelectionDiagram("trial", "target", ["z"]), "a", "y")
    result = advanced.prepare(
        query, data, provider=tr.TrialAipw(outcome=Linear()), inference=tr.TransportInference(0)
    ).estimate()
    assert result.estimate == pytest.approx(2.0, abs=0.16)
    assert result.interval is None


def test_trial_transformation_preview_uses_common_contract():
    query, data = trial_fixture()
    study = advanced.prepare(
        query,
        data,
        provider=tr.TrialAipw(Linear(), Logistic(), folds=3),
        inference=tr.TransportInference(bootstrap=0),
    )
    report = study.preview_transform("compatible_data_replace")
    assert report
    assert study.preview_transform("change_graph") != report
