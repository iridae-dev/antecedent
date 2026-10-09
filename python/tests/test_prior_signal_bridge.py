"""Original artifact to canonical ranker; independent two-state binomial oracle."""

import math
from dataclasses import replace

import antecedent as ac
import pytest
from antecedent import design as dr
from antecedent.errors import CausalTypeError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.priors import (
    DesignVariable,
    EstimandFingerprint,
    PriorCatalog,
    PriorSource,
    PriorSourceMeta,
)


def quantity(name="state", population="target"):
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id=population,
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


def catalog(
    *, summary=False, name="ate", states=(0.25, 0.75), identification="NonparametricallyIdentified"
):
    mean = sum(states) / 2
    delta = states[1] - states[0]
    kwargs = dict(
        n_draws=2,
        mean=[mean],
        sd=[abs(delta) / math.sqrt(2)],
        q025=[states[0] + 0.025 * delta],
        q975=[states[0] + 0.975 * delta],
        backend_id="conjugate",
        identification=identification,
        quantity_names=[name],
    )
    posterior = (
        ac.inference.PosteriorArtifact.from_moments(**kwargs)
        if summary
        else ac.inference.PosteriorArtifact(draws=list(states), **kwargs)
    )
    artifact = bytes(ac.inference.encode_posterior_artifact(posterior))
    meta = PriorSourceMeta(
        "historic",
        EstimandFingerprint("ate", "t", "y"),
        "NonparametricallyIdentified",
        design=(DesignVariable("t", "treatment"), DesignVariable("y", "outcome")),
    )
    return PriorCatalog([PriorSource(meta, artifact)])


def adapt(**kwargs):
    return dr.adapt_prior_to_signal(
        kwargs.pop("catalog", catalog()),
        query=ac.AverageEffect("t", "y"),
        sources=kwargs.pop(
            "sources",
            [dr.PriorSignalSource("historic", "ate", "target", ("original-study",), ("obs:1",))],
        ),
        signal=dr.BinomialSignal(),
        state=kwargs.pop("state", quantity()),
        target_population="target",
        prior_id="historic-prior",
        resolution=2,
        **kwargs,
    )


def rank(checked, *, provider=None, reuse=(), state=None, prior_id="historic-prior", **options):
    problem = dr.DesignDecision(
        "contract-bet",
        (dr.ActionUtility("abstain", 0.0), dr.ActionUtility("bet", -0.5, 1.0)),
        checked.prior,
        "utility",
    )
    spec = dr.SignalSpec(
        prior_id, quantity() if state is None else state, quantity("signal"), ("new-study",)
    )
    return dr.rank_designs(
        [
            dr.Candidate(
                "future",
                2,
                checked.signal if provider is None else provider,
                reused_observations=reuse,
            )
        ],
        decision=problem,
        signal=spec,
        **options,
    )


def test_original_checked_prior_reaches_canonical_ranker_and_replays():
    checked = adapt()
    result = rank(checked)
    assert result.basis == "evsi"
    assert result.prior_expected_utility == pytest.approx(0.0)
    assert result.evpi == pytest.approx(0.125)
    assert result.candidates[0].evsi == pytest.approx(0.0625)
    diagnostics = checked.diagnostics
    assert diagnostics["source_ids"] == ["historic"]
    assert diagnostics["observations_checked"] == 1
    assert diagnostics["pooled_mean"] == 0.5
    assert diagnostics["source_digest"] in result.source_digests
    assert (
        dr.consume(result.export(), expected_identity=result.expectation())
        .entries[0]
        .natively_replayed
    )
    diagnostics["source_ids"].clear()
    assert checked.diagnostics["source_ids"] == ["historic"]


@pytest.mark.parametrize(
    "options,detail",
    [
        ({"reuse": ("obs:1",)}, ".source_overlap"),
        ({"provider": dr.GaussianMeanSignal(1.0)}, "prior_signal.signal_family_mismatch"),
        ({"prior_id": "other"}, "prior_signal.prior_identity_mismatch"),
        ({"state": quantity("other")}, "prior_signal.state_quantity_mismatch"),
    ],
)
def test_ranking_cannot_drop_original_overlap_or_rebind_checked_source(options, detail):
    with pytest.raises(dr.DesignRankingRefusal) as error:
        rank(adapt(), **options)
    assert error.value.detail.endswith(detail)


def test_transport_and_overlap_use_existing_checked_native_adapter():
    source = dr.PriorSignalSource("historic", "ate", "elsewhere", ("original-study",), ("obs:1",))
    with pytest.raises(dr.DesignRankingRefusal) as error:
        adapt(sources=[source])
    assert error.value.detail == "prior_signal.transport_policy_required"
    assert error.value.reason_code == "transport_policy_required"
    checked = adapt(sources=[source], transport_policy_id="named-policy")
    assert checked.diagnostics["transport_policy_id"] == "named-policy"
    with pytest.raises(dr.DesignRankingRefusal) as reused:
        adapt(candidate_observation_ids=["obs:1"])
    assert reused.value.detail == "prior_signal.observation_reuse"


@pytest.mark.parametrize(
    "options,detail",
    [
        ({"catalog": catalog(summary=True)}, "prior_signal.named_state_draws_required"),
        ({"catalog": catalog(name="coef_0")}, "prior_signal.named_state_draws_required"),
        (
            {"sources": [dr.PriorSignalSource("historic", "ate", "target", (), ("obs:1",))]},
            "prior_signal.lineage_missing",
        ),
        ({"state": quantity(population="elsewhere")}, "prior_signal.target_population_mismatch"),
    ],
)
def test_incomplete_or_relabelled_sources_refuse(options, detail):
    with pytest.raises(dr.DesignRankingRefusal) as error:
        adapt(**options)
    assert error.value.detail == detail


def test_coefficient_moments_are_not_prior_signal_inputs():
    with pytest.raises(CausalTypeError):
        dr.adapt_prior_to_signal(
            None,
            query=ac.AverageEffect("t", "y"),
            sources=[],
            signal=dr.BinomialSignal(),
            state=quantity(),
            target_population="target",
            prior_id="p",
        )


def test_actual_native_bayesian_producer_catalog_adapter_ranking_handoff():
    import numpy as np

    rng = np.random.default_rng(823)
    t = np.tile([0.0, 1.0], 40)
    y = 0.25 * t + rng.normal(0.0, 0.7, len(t))
    query = ac.AverageEffect("t", "y")
    result = ac.analyze(
        {"t": t, "y": y},
        graph=[("t", "y")],
        query=query,
        inference=ac.Bayesian(n_draws=32),
        return_posterior_artifact=True,
        refute=False,
        seed=19,
    )
    assert result.posterior is not None and result.posterior.artifact is not None
    source = PriorSource(
        PriorSourceMeta(
            "actual-fit",
            EstimandFingerprint("ate", "t", "y"),
            "IdentifiedUnderParametricRestrictions",
            design=(DesignVariable("t", "treatment"), DesignVariable("y", "outcome")),
        ),
        bytes(result.posterior.artifact),
    )
    checked = dr.adapt_prior_to_signal(
        PriorCatalog([source]),
        query=query,
        sources=[
            dr.PriorSignalSource(
                "actual-fit",
                "ate",
                "target",
                ("actual-fit-study",),
                tuple(f"training:{i}" for i in range(len(t))),
            )
        ],
        signal=dr.GaussianMeanSignal(0.49),
        state=quantity(),
        target_population="target",
        prior_id="historic-prior",
        resolution=32,
    )
    ranked = rank(checked, monte_carlo=dr.MonteCarlo(4, 4, 16, 0.0), mc_error_tolerance=1.0)
    assert math.isfinite(ranked.candidates[0].evsi)
    assert checked.diagnostics["source_posteriors"][0]["posterior"]["n_draws"] == 32
    assert (
        checked.diagnostics["prior_approximation"]
        == "deterministic_resampled_original_posterior_draws"
    )
    assert ranked.calibration == "unmeasured"
    retained = ranked.expectation()
    consumed = dr.consume(ranked.export(), expected_identity=retained)
    assert not consumed.entries[0].natively_replayed
    assert ranked.candidates[0].claim == "monte_carlo_estimate"
    assert consumed.source_digests == ranked.source_digests
    with pytest.raises(dr.DesignRankingRefusal):
        dr.consume(
            ranked.export(), expected_identity=replace(retained, source_digests=("altered-source",))
        )
    with pytest.raises(dr.SourceOverlapRefusal):
        rank(checked, reuse=("training:0",))


def test_checked_prior_projection_cannot_be_relabelled():
    checked = adapt()
    object.__setattr__(checked, "prior", replace(checked.prior, states=(1000.0,)))
    with pytest.raises(dr.DesignRankingRefusal) as error:
        rank(checked)
    assert error.value.detail == "prior_signal.prior_projection_mismatch"
    assert error.value.reason_code == "invalid_argument"


def test_source_draw_mutation_changes_retained_source_expectations():
    original = rank(adapt())
    changed = rank(adapt(catalog=catalog(states=(0.0, 1.0))))
    assert changed.candidates[0].evsi == pytest.approx(0.25)
    assert original.candidates[0].evsi == pytest.approx(0.0625)
    assert original.source_digests != changed.source_digests
    with pytest.raises(dr.DesignRankingRefusal):
        dr.consume(changed.export(), expected_identity=original.expectation())


def test_catalog_metadata_cannot_upgrade_original_unidentified_posterior():
    with pytest.raises(dr.DesignRankingRefusal) as error:
        adapt(catalog=catalog(identification="NotIdentified"))
    assert error.value.detail == "prior_signal.original_posterior_unidentified"


def test_original_population_tag_cannot_be_relabelled():
    original = catalog().sources[0]
    tagged = PriorSource(
        replace(original.meta, tags={"population": "elsewhere"}), original.artifact
    )
    with pytest.raises(dr.DesignRankingRefusal) as error:
        adapt(catalog=PriorCatalog([tagged]))
    assert error.value.detail == "prior_signal.source_population_mismatch"


def test_two_original_sources_cannot_double_count_observations():
    original = catalog().sources[0]
    second = PriorSource(replace(original.meta, artifact_id="second"), original.artifact)
    sources = [
        dr.PriorSignalSource("historic", "ate", "target", ("study:1",), ("obs:1",)),
        dr.PriorSignalSource("second", "ate", "target", ("study:2",), ("obs:1",)),
    ]
    with pytest.raises(dr.DesignRankingRefusal) as error:
        adapt(catalog=PriorCatalog([original, second]), sources=sources)
    assert error.value.detail == "prior_signal.prior_sources_overlap"
