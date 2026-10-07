"""External candidate signals, EVSI with a cost mapping, and the durable design ranking.

Oracles are the constants of ``crates/antecedent-design/tests/{evsi,design_ranking,
design_ranking_artifact}.rs``, derived by hand and not read from the code under test:

* frozen F11/F12/F14 decision: guess a binary state (utility 1 if right), prior 1/2; a signal of
  accuracy ``a`` has ``EVSI = a - 1/2``, so accuracies 3/4 and 5/8 give 1/4 and 1/8, ``EVPI = 1/2``,
  and with an identical utility-unit cost 1/10 the net values are 3/20 and 1/40;
* binomial bet, two-point prior ``{1/4, 3/4}``, ``n = 2``: ``EVSI = 1/16`` and ``EVPI = 1/8``;
* Gaussian mean, prior ``N(1/2, 1)``, ``n = 4``, noise variance 12: ``EVSI = 0.04165773529384315``.
"""

from __future__ import annotations

import pytest
from antecedent import design_ranking as dr
from antecedent.errors import CausalSerializationError, CausalUnsupportedError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity

EVSI_GAUSSIAN = 0.04165773529384315


def _quantity(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


def _signal() -> dr.SignalSpec:
    return dr.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=3,
    )


GUESS = dr.Decision(
    contract="contract-1",
    actions=(dr.ActionUtility("guess0", 1.0, -1.0), dr.ActionUtility("guess1", 0.0, 1.0)),
    prior=dr.Prior.draws([0.0, 1.0]),
    utility_units="utility",
)
UTILITY_MAP = dr.CostMap("utility", "utility", 1.0)


def _bet(scale: float = 1.0, shift: float = 0.0) -> dr.Decision:
    return dr.Decision(
        contract="contract-bet",
        actions=(
            dr.ActionUtility("abstain", shift, 0.0),
            dr.ActionUtility("bet", shift - 0.5 * scale, scale),
        ),
        prior=dr.Prior.draws([0.25, 0.75]),
        utility_units="utility",
    )


def _external(
    law: dr.ExternalLaw, object_id: str = "signal-object", **kw: object
) -> dr.ExternalSignal:
    return dr.ExternalSignal("lab", object_id, "v1", "snap", "lab-qa", law, **kw)  # type: ignore[arg-type]


def _accuracy_posterior(accuracy: float) -> dr.ExternalLaw:
    return dr.ExternalLaw.posterior(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        predictive=[0.5, 0.5],
        posterior=[[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]],
    )


def _guess_candidate(
    cid: str, accuracy: float, cost: float = 0.1, unit: str = "utility", **kw: object
) -> dr.Candidate:
    return dr.Candidate(
        cid,
        1,
        _external(_accuracy_posterior(accuracy)),
        cost=cost,
        cost_unit=unit,
        **kw,  # type: ignore[arg-type]
    )


def _frozen() -> list[dr.Candidate]:
    return [_guess_candidate("cand-1", 0.75), _guess_candidate("cand-2", 0.625)]


def _rank(candidates: list[dr.Candidate], **kw: object) -> dr.DesignRankingResult:
    return dr.rank_designs(GUESS, candidates, signal=_signal(), **kw)  # type: ignore[arg-type]


BET_LIKELIHOOD = dr.ExternalLaw.likelihood(
    states=[0.25, 0.75],
    statistics=[0.0, 1.0, 2.0],
    probabilities=[[0.5625, 0.0625], [0.375, 0.375], [0.0625, 0.5625]],
)
BET_POSTERIOR = dr.ExternalLaw.posterior(
    states=[0.25, 0.75],
    statistics=[0.0, 1.0, 2.0],
    predictive=[0.3125, 0.375, 0.3125],
    posterior=[[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]],
)
BET_VALUES = dr.ExternalLaw.decision_values(
    branch_probabilities=[0.3125, 0.375, 0.3125],
    action_ids=["abstain", "bet"],
    values=[[0.0, -0.2], [0.0, 0.0], [0.0, 0.2]],
)


def _bet_candidate(cid: str, provider: dr.SignalProvider) -> dr.Candidate:
    return dr.Candidate(cid, 2, provider, cost=0.0, cost_unit="utility")


# -- F11: external signal and update ---------------------------------------------------------


def test_f11_external_signal_update_has_the_frozen_branch_values() -> None:
    # P(signal=1) = 1/2; P(theta=1 | signal=1) = 3/4 and P(theta=1 | signal=0) = 1/4.
    sensitivity_specificity = dr.ExternalLaw.likelihood(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        probabilities=[[0.75, 0.25], [0.25, 0.75]],
    )
    candidate = dr.Candidate("study", 1, _external(sensitivity_specificity), cost=0.0)
    value = dr.evsi(GUESS, candidate, signal=_signal())
    assert value.evsi == pytest.approx(0.25, abs=1e-12)  # post-signal 3/4 minus current 1/2
    assert value.evpi == pytest.approx(0.5, abs=1e-12)
    assert value.update_mode == "native_update"
    assert value.provider_trust == "externally_attested"
    assert value.attestor == "lab-qa"
    assert value.claim == "point_only"


def test_f11_native_and_equivalent_external_signal_agree_on_known_truth() -> None:
    native = dr.evsi(_bet(), _bet_candidate("native", dr.BinomialSignal()), signal=_signal())
    external = dr.evsi(
        _bet(), _bet_candidate("external", _external(BET_POSTERIOR)), signal=_signal()
    )
    for value in (native, external):
        assert value.evsi == pytest.approx(1.0 / 16.0, abs=1e-12)
        assert value.evpi == pytest.approx(0.125, abs=1e-12)
    assert native.provider_trust == "native_licensed"
    assert native.natively_replayed
    assert native.provider is None
    assert external.provider_trust == "externally_attested"
    assert not external.natively_replayed  # an equivalent external signal is never native
    assert external.provider is not None
    assert external.provider.request_id == external.request_fingerprint


def test_f11_every_update_mode_executes_end_to_end() -> None:
    ranking = dr.rank_designs(
        _bet(),
        [
            _bet_candidate("a_likelihood", _external(BET_LIKELIHOOD)),
            _bet_candidate("b_posterior", _external(BET_POSTERIOR)),
            _bet_candidate("c_values", _external(BET_VALUES)),
        ],
        signal=_signal(),
    )
    modes = {c.id: c.update_mode for c in ranking.candidates}
    assert modes == {
        "a_likelihood": "native_update",
        "b_posterior": "external_posterior",
        "c_values": "external_decision_values",
    }
    methods = {c.id: c.integration.method for c in ranking.candidates}
    assert methods["c_values"] == "externally_computed"
    assert methods["a_likelihood"] == methods["b_posterior"] == "exact"
    for candidate in ranking.candidates:
        assert candidate.evsi == pytest.approx(1.0 / 16.0, abs=1e-12)
        assert candidate.provider_trust == "externally_attested"
        assert not candidate.natively_replayed
    assert next(c for c in ranking.candidates if c.id == "c_values").replay == (
        "attested_values_combined"
    )
    assert any(
        a == "update_computed_externally_not_verified_by_antecedent"
        for a in ranking.candidate("b_posterior").assumptions
    )


@pytest.mark.parametrize(
    ("kwargs", "detail"),
    [
        ({"attested_candidate_id": "someone-else"}, "signal_provider.candidate_mismatch"),
        ({"attested_prior_id": "other-prior"}, "signal_provider.prior_mismatch"),
        ({"attested_sample_size": 3}, "signal_provider.sample_size_mismatch"),
    ],
)
def test_f11_wrong_candidate_prior_or_sample_id_refuses(
    kwargs: dict[str, object], detail: str
) -> None:
    provider = _external(_accuracy_posterior(0.75), **kwargs)
    with pytest.raises(dr.SignalProviderRefusal) as caught:
        dr.evsi(GUESS, dr.Candidate("study", 1, provider), signal=_signal())
    assert isinstance(caught.value, CausalUnsupportedError)
    assert caught.value.detail == detail
    assert caught.value.reason_code == "design_signal_invalid"


def test_f11_predictive_law_without_a_coherent_update_refuses() -> None:
    incoherent = dr.ExternalLaw.posterior(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        predictive=[0.5, 0.5],
        posterior=[[0.9, 0.1], [0.9, 0.1]],  # does not average back to the prior
    )
    with pytest.raises(dr.SignalProviderRefusal) as caught:
        dr.evsi(GUESS, dr.Candidate("study", 1, _external(incoherent)), signal=_signal())
    assert caught.value.detail == "signal_provider.posterior_incoherent"
    unnormalised = dr.ExternalLaw.likelihood(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        probabilities=[[0.75, 0.25], [0.5, 0.75]],
    )
    with pytest.raises(dr.SignalProviderRefusal) as caught:
        dr.evsi(GUESS, dr.Candidate("study", 1, _external(unnormalised)), signal=_signal())
    assert caught.value.detail == "signal_provider.law_incoherent"


def test_f11_a_request_without_lineage_or_independence_refuses() -> None:
    bad = dr.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=(),
    )
    with pytest.raises(dr.SignalProviderRefusal) as caught:
        dr.evsi(_bet(), _bet_candidate("native", dr.BinomialSignal()), signal=bad)
    assert caught.value.detail == "signal_provider.missing_lineage"
    with pytest.raises(CausalValueError):
        dr.evsi(_bet(), _bet_candidate("native", dr.BinomialSignal()))  # no signal request


# -- F12: EVSI, EVPI and the cost mapping -----------------------------------------------------


def test_f12_frozen_values_and_net_value_only_with_a_cost_mapping() -> None:
    separate = _rank([_guess_candidate("study", 0.75, 0.1, "usd")])
    row = separate.candidates[0]
    assert row.evsi == pytest.approx(0.25, abs=1e-12)
    assert row.evpi == pytest.approx(0.5, abs=1e-12)
    assert separate.prior_expected_utility == pytest.approx(0.5, abs=1e-12)
    assert row.net_value is None and row.study_cost_utility is None
    assert (row.study_cost, row.cost_unit) == (0.1, "usd")
    assert separate.basis == "evsi"
    assert separate.calibration == "unmeasured"

    net = _rank([_guess_candidate("study", 0.75)], cost_map=UTILITY_MAP).candidates[0]
    assert net.study_cost_utility == pytest.approx(0.1, abs=1e-12)
    assert net.net_value == pytest.approx(0.15, abs=1e-12)  # 1/4 - 1/10

    priced = _rank(
        [_guess_candidate("study", 0.75, 0.1, "usd")],
        cost_map=dr.CostMap("usd", "utility", 2.0),
    )
    assert priced.basis == "net_value"
    assert priced.candidates[0].net_value == pytest.approx(0.25 - 0.2, abs=1e-12)


def test_f12_currency_without_a_mapping_and_missing_maps_refuse() -> None:
    with pytest.raises(dr.CostUnitsRefusal) as caught:
        _rank([_guess_candidate("study", 0.75, 0.1, "usd")], cost_map=UTILITY_MAP)
    assert caught.value.reason_code == "design_cost_units_mismatch"
    assert caught.value.detail == "evsi.cost_units_mismatch"
    with pytest.raises(dr.CostUnitsRefusal) as caught:
        _rank(
            [_guess_candidate("study", 0.75, 0.1, "usd")],
            cost_map=dr.CostMap("usd", "qaly", 1.0),
        )
    assert caught.value.detail == "evsi.cost_units_mismatch"
    with pytest.raises(dr.CostUnitsRefusal) as caught:
        _rank([_guess_candidate("study", 0.75, 0.1, "usd")], require_net_value=True)
    assert caught.value.detail == "evsi.cost_map_required"


def test_f12_changed_terminal_action_set_refuses() -> None:
    other_actions = dr.ExternalLaw.decision_values(
        branch_probabilities=[0.3125, 0.375, 0.3125],
        action_ids=["abstain", "other"],
        values=[[0.0, -0.2], [0.0, 0.0], [0.0, 0.2]],
    )
    with pytest.raises(dr.DesignRankingRefusal) as caught:
        dr.evsi(_bet(), _bet_candidate("values", _external(other_actions)), signal=_signal())
    assert caught.value.detail == "evsi.action_set_changed"
    assert caught.value.reason_code == "decision_contract_unsatisfied"
    incoherent = dr.ExternalLaw.decision_values(
        branch_probabilities=[0.3125, 0.375, 0.3125],
        action_ids=["abstain", "bet"],
        values=[[0.0, -0.2], [0.0, 0.0], [0.0, 0.5]],
    )
    with pytest.raises(dr.DesignRankingRefusal) as caught:
        dr.evsi(_bet(), _bet_candidate("values", _external(incoherent)), signal=_signal())
    assert caught.value.detail == "evsi.decision_values_incoherent"


def test_f12_evsi_is_nonnegative_bounded_by_evpi_and_positive_affine_invariant() -> None:
    candidate = _bet_candidate("native", dr.BinomialSignal())
    base = dr.evsi(_bet(), candidate, signal=_signal())
    assert 0.0 <= base.evsi <= base.evpi + 1e-12
    scaled = dr.evsi(_bet(3.0, 10.0), candidate, signal=_signal())  # U' = 3U + 10
    assert scaled.evsi == pytest.approx(3.0 * base.evsi, abs=1e-12)
    assert scaled.evpi == pytest.approx(3.0 * base.evpi, abs=1e-12)


def test_f12_gaussian_mean_matches_the_closed_form_and_is_a_point_value() -> None:
    decision = dr.Decision(
        contract="contract-gauss",
        actions=(dr.ActionUtility("stay", 0.0, 0.0), dr.ActionUtility("treat", 0.0, 1.0)),
        prior=dr.Prior.normal(0.5, 1.0),
        utility_units="utility",
    )
    candidate = dr.Candidate("gauss", 4, dr.GaussianMeanSignal(12.0))
    value = dr.evsi(decision, candidate, signal=_signal())
    assert value.evsi == pytest.approx(EVSI_GAUSSIAN, abs=1e-7)
    assert value.integration.method == "exact"
    assert value.integration.stderr == 0.0 and value.integration.converged
    assert value.claim == "point_only"
    assert value.natively_replayed and value.replay == "native_exact_recomputed"


def test_f12_monte_carlo_value_reports_error_and_makes_no_coverage_claim() -> None:
    decision = dr.Decision(
        contract="contract-mc",
        actions=(dr.ActionUtility("abstain", 0.0, 0.0), dr.ActionUtility("bet", -0.45, 1.0)),
        prior=dr.Prior.draws([0.2, 0.5, 0.8]),
        utility_units="utility",
    )
    ranking = dr.rank_designs(
        decision,
        [dr.Candidate("mc", 4, dr.GaussianMeanSignal(0.25))],
        signal=_signal(),
        monte_carlo=dr.MonteCarlo(16, 16, 64, 0.0),
        mc_error_tolerance=1.0,
    )
    value = ranking.candidates[0]
    assert value.integration.method == "monte_carlo"
    assert value.integration.replicates == 16 * 64
    assert value.integration.stderr > 0.0 and value.integration.converged
    assert value.integration.ess == 1024.0
    assert value.evpi == pytest.approx(1.0 / 12.0, abs=1e-12)
    slack = 4.0 * value.integration.stderr + 1e-9
    assert -slack <= value.evsi <= value.evpi + slack
    assert value.claim == "monte_carlo_estimate"
    assert not value.natively_replayed and value.replay == "monte_carlo_not_replayed"
    assert ranking.calibration == "unmeasured"


def test_f12_ties_and_truncated_search_are_reported() -> None:
    def make(cid: str) -> dr.Candidate:
        return _bet_candidate(cid, dr.BinomialSignal())

    tied = dr.rank_designs(_bet(), [make("b"), make("a")], signal=_signal())
    assert [c.id for c in tied.candidates] == ["a", "b"]
    assert tied.ties == (("a", "b"),)
    assert all(c.rank_uncertain for c in tied.candidates)
    cut = dr.rank_designs(
        _bet(), [make("c"), make("a"), make("b")], signal=_signal(), max_candidates=2
    )
    assert cut.search.truncated and cut.search.supplied == 3 and cut.search.evaluated == 2
    assert cut.search.unevaluated_ids == ("c",)


# -- F14: the ranking artifact -----------------------------------------------------------------


def test_f14_net_values_are_three_twentieths_and_one_fortieth_in_any_input_order() -> None:
    forward = _rank(_frozen(), cost_map=UTILITY_MAP)
    reversed_ = _rank(list(reversed(_frozen())), cost_map=UTILITY_MAP)
    assert [c.id for c in forward.candidates] == ["cand-1", "cand-2"]
    assert forward.candidates[0].net_value == pytest.approx(0.15, abs=1e-12)
    assert forward.candidates[1].net_value == pytest.approx(0.025, abs=1e-12)
    assert forward.candidates[0].evsi == pytest.approx(0.25, abs=1e-12)
    assert forward.candidates[1].evsi == pytest.approx(0.125, abs=1e-12)
    assert forward.identity == reversed_.identity
    assert forward.ranking_identity == reversed_.ranking_identity
    assert forward.export() == reversed_.export()


def test_f14_export_consume_round_trip_with_retained_expectation() -> None:
    result = _rank(_frozen(), cost_map=UTILITY_MAP, source_digests=("digest-b", "digest-a"))
    data = result.export()
    consumed = dr.consume(data, expected=result.expectation())
    assert consumed.identity == result.identity
    assert consumed.ranking_identity == result.ranking_identity
    assert consumed.calibration == "unmeasured"
    assert consumed.source_digests == ("digest-a", "digest-b")
    assert [e.id for e in consumed.entries] == ["cand-1", "cand-2"]
    assert consumed.entries[0].net_value == pytest.approx(0.15, abs=1e-12)
    assert consumed.cost_map == UTILITY_MAP
    assert dr.DesignRankingResult.consume(data).identity == result.identity
    for entry in consumed.entries:
        assert entry.provider_trust == "externally_attested"
        assert entry.update_mode == "external_posterior"
        assert not entry.natively_replayed
        assert "not verified" in entry.trust_limit


def test_f14_altered_signal_or_update_refuses_replay_against_retained_identities() -> None:
    original = _rank(_frozen(), cost_map=UTILITY_MAP)
    # Same EVSI (1/4), but updated natively from a supplied likelihood instead of an externally
    # computed posterior: a different update mode, hence a different signal identity.
    likelihood = dr.ExternalLaw.likelihood(
        states=[0.0, 1.0], statistics=[0.0, 1.0], probabilities=[[0.75, 0.25], [0.25, 0.75]]
    )
    altered = _rank(
        [dr.Candidate("cand-1", 1, _external(likelihood), cost=0.1), _frozen()[1]],
        cost_map=UTILITY_MAP,
    )
    assert altered.candidates[0].evsi == pytest.approx(0.25, abs=1e-12)
    assert altered.identity != original.identity
    with pytest.raises(dr.DesignRankingRefusal) as caught:
        dr.consume(altered.export(), expected=original.expectation())
    assert caught.value.detail == "design_ranking.identity_mismatch"
    signals_only = dr.Expectation(signal_identities=original.expectation().signal_identities)
    with pytest.raises(dr.DesignRankingRefusal) as caught:
        dr.consume(altered.export(), expected=signals_only)
    assert caught.value.detail == "design_ranking.signal_mismatch"
    assert caught.value.offending == "cand-1"
    # An altered RNG seed is a different exact request, hence a different fingerprint.
    reseeded_spec = dr.SignalSpec(
        prior_id="prior-1",
        state=_quantity("state"),
        observation=_quantity("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=99,
    )
    reseeded = dr.rank_designs(GUESS, _frozen(), signal=reseeded_spec, cost_map=UTILITY_MAP)
    assert reseeded.candidates[0].request_fingerprint != original.candidates[0].request_fingerprint
    with pytest.raises(dr.DesignRankingRefusal):
        dr.consume(
            reseeded.export(),
            expected=dr.Expectation(signal_identities=original.expectation().signal_identities),
        )


def test_f14_overlapping_source_data_and_incompatible_cost_units_refuse() -> None:
    overlapping = [_guess_candidate("cand-1", 0.75, reused_observations=("obs-prior",))]
    with pytest.raises(dr.SourceOverlapRefusal) as caught:
        _rank(overlapping, cost_map=UTILITY_MAP, prior_observations=("obs-prior",))
    assert caught.value.detail == "evsi.source_overlap"
    assert caught.value.offending == "obs-prior"
    ok = _rank(
        [_guess_candidate("cand-1", 0.75, reused_observations=("obs-future",))],
        cost_map=UTILITY_MAP,
        prior_observations=("obs-prior",),
    )
    assert ok.candidates[0].source_overlap.overlapping == ()
    assert ok.candidates[0].source_overlap.observations_checked == 2

    ranking = _rank(_frozen(), cost_map=UTILITY_MAP)
    other = dr.Expectation(cost_map=dr.CostMap("utility", "utility", 2.0))
    with pytest.raises(dr.CostUnitsRefusal) as caught:
        dr.consume(ranking.export(), expected=other)
    assert caught.value.reason_code == "design_cost_units_mismatch"
    assert caught.value.detail == "design_ranking.cost_units_mismatch"
    with pytest.raises(dr.CostUnitsRefusal):
        dr.consume(ranking.export(), expected=dr.Expectation(no_cost_map=True))


def test_f14_external_values_retain_attested_values_and_are_never_marked_native() -> None:
    ranking = dr.rank_designs(
        _bet(), [_bet_candidate("values", _external(BET_VALUES))], signal=_signal()
    )
    row = ranking.candidates[0]
    assert row.provider_trust == "externally_attested"
    assert row.attestor == "lab-qa"
    assert row.provider is not None and row.provider.request_id == row.request_fingerprint
    assert row.update_mode == "external_decision_values"
    assert not row.natively_replayed
    assert "not verified" in row.trust_limit
    consumed = dr.consume(ranking.export())
    assert not consumed.entries[0].natively_replayed
    assert consumed.entries[0].replay == "attested_values_combined"


def test_f14_corrupt_or_truncated_bytes_raise_a_serialization_error() -> None:
    data = bytearray(_rank(_frozen(), cost_map=UTILITY_MAP).export())
    flipped = bytearray(data)
    flipped[len(flipped) // 2] ^= 0x55
    with pytest.raises(CausalSerializationError):
        dr.consume(bytes(flipped))
    with pytest.raises(CausalSerializationError):
        dr.consume(bytes(data[: len(data) // 2]))


def test_f14_structural_order_is_preserved_when_no_probabilistic_model_is_licensed() -> None:
    structural = [
        dr.StructuralCandidate("precise", True, 5, 100),
        dr.StructuralCandidate("noisy", True, 1, 100),
    ]
    ordering = dr.rank_structural(structural)
    assert ordering.basis == "structural_sufficiency_cost"
    assert [e.id for e in ordering.entries] == ["noisy", "precise"]
    assert dr.rank_structural(list(reversed(structural))).identity == ordering.identity
    # A noisy candidate's regret order differs from its structural unlock order.
    probabilistic = _rank(
        [_guess_candidate("noisy", 0.51, 0.0), _guess_candidate("precise", 0.75, 0.0)]
    )
    assert [c.id for c in probabilistic.candidates] == ["precise", "noisy"]
    assert probabilistic.candidates[1].evsi == pytest.approx(0.01, abs=1e-12)
    full = [
        dr.StructuralCandidate("tie_b", True, 3, 10),
        dr.StructuralCandidate("impossible", False, 1, 1),
        dr.StructuralCandidate("repair", True, 3, 10),
        dr.StructuralCandidate("cheap", True, 2, 99),
        dr.StructuralCandidate("tie_a", True, 3, 10),
        dr.StructuralCandidate("small_budget", True, 3, 5),
    ]
    assert [e.id for e in dr.rank_structural(full).entries] == [
        "cheap",
        "small_budget",
        "repair",
        "tie_a",
        "tie_b",
        "impossible",
    ]
    with pytest.raises(dr.DesignRankingRefusal) as caught:
        dr.rank_structural(
            [dr.StructuralCandidate("a", True, 1), dr.StructuralCandidate("a", False, 2)]
        )
    assert caught.value.detail == "design_ranking.duplicate_candidate"
