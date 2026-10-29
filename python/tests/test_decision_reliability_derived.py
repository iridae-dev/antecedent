"""Derived decision-reliability uncertainty through Python, against a hand-enumerated law.

The law is four equally likely states repeated ten times (forty serially independent draws on one
shared simulation, declared known): A = (3, 3, 2, 2), B = (2, 1, 3, 2), C = (1, 2, 1, 2). Utilities
are 5/2, 2 and 3/2; the standard error of A is ``0.5 / sqrt(39)`` (sample variance), A - B is
``(1, 2, -1, 0)`` (population variance 5/4) and P(A best) is 7/12 with a Wilson interval at
effective sample size 40 computed here from the closed form.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest
from antecedent import decision_reliability as dr
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success

DR = dr.DecisionReliabilityRefusal
Z_975 = 1.959963984540054
Z_BONFERRONI_3 = 2.39398
SE_A = 0.5 / math.sqrt(39.0)


def _stream(purpose: str = "selection") -> dr.StreamId:
    return dr.StreamId(run_seed=1, purpose=purpose, replicate=0, label="states")


def _repeat(pattern: list[float], times: int = 10) -> np.ndarray:
    return np.array(pattern * times, dtype=float)


def _law(times: int = 10, **kwargs: object) -> dr.DecisionDraws:
    options: dict[str, object] = {
        "stream": _stream(),
        "source": "known_law",
        "coupling": "shared_simulation",
        "within_stream": "independent",
    }
    options.update(kwargs)
    return dr.DecisionDraws.from_arrays(
        {
            "A": _repeat([3, 3, 2, 2], times),
            "B": _repeat([2, 1, 3, 2], times),
            "C": _repeat([1, 2, 1, 2], times),
        },
        **options,  # type: ignore[arg-type]
    )


def _request() -> dr.ReliabilityRequest:
    return dr.ReliabilityRequest(ranking=dr.RankingRequest())


def _wilson(p: float, n: float, z: float) -> tuple[float, float]:
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return centre - half, centre + half


def _utility(derived: dr.DerivedUncertainty, candidate: str) -> dict[str, float]:
    return next(u["interval"] for u in derived.numerical["utilities"] if u["id"] == candidate)


def test_three_components_match_the_enumerated_law() -> None:
    derived = require_success(
        dr.derive_uncertainty(_law(), _request(), dr.IntervalRequest(level=0.95))
    )
    assert derived.standing == "derived_unmeasured"
    assert derived.decided_id == "A"
    assert derived.source == "known_law"
    assert derived.assumptions
    a = _utility(derived, "A")
    assert a["estimate"] == pytest.approx(2.5)
    assert a["standard_error"] == pytest.approx(SE_A, rel=1e-9)
    assert a["lower"] == pytest.approx(2.5 - Z_975 * SE_A, abs=1e-6)
    assert a["upper"] == pytest.approx(2.5 + Z_975 * SE_A, abs=1e-6)
    assert derived.numerical["intervals_reported"] == 8
    ab = next(d for d in derived.numerical["differences_vs_decision"] if d["id"] == "B")
    assert ab["interval"]["estimate"] == pytest.approx(0.5)
    assert ab["interval"]["standard_error"] == pytest.approx(math.sqrt(1.25 / 39.0), rel=1e-9)
    best = next(
        p for p in derived.numerical["probabilities"] if p["id"] == "A" and p["quantity"] == "best"
    )
    lower, upper = _wilson(7.0 / 12.0, 40.0, Z_975)
    assert best["interval"]["lower"] == pytest.approx(lower, abs=1e-6)
    assert best["interval"]["upper"] == pytest.approx(upper, abs=1e-6)


def test_selection_component_is_a_confidence_set_and_a_corrected_interval() -> None:
    derived = require_success(
        dr.derive_uncertainty(_law(), _request(), dr.IntervalRequest(level=0.95))
    )
    cset = derived.confidence_set
    assert cset["basis"] == "paired_comparisons_with_best"
    assert list(cset["members"]) == ["A"]
    assert list(cset["excluded"]) == ["B", "C"]
    assert cset["critical_value"] == pytest.approx(Z_975, abs=1e-6)
    assert cset["unretained_candidates"] == 0
    corrected = derived.corrected_interval
    assert corrected is not None
    assert corrected["selected_id"] == "A"
    assert corrected["n_considered"] == 3
    assert corrected["critical_value"] == pytest.approx(Z_BONFERRONI_3, abs=1e-4)
    assert corrected["numerical_half_width"] == pytest.approx(Z_975 * SE_A, abs=1e-6)
    assert corrected["selection_extra_half_width"] > 0.0
    width = corrected["upper"] - corrected["lower"]
    assert width == pytest.approx(2.0 * Z_BONFERRONI_3 * SE_A, abs=1e-4)


def test_a_prespecified_decision_has_no_selection_correction() -> None:
    request = dr.ReliabilityRequest(decision="B")
    derived = require_success(dr.derive_uncertainty(_law(), request, dr.IntervalRequest(level=0.9)))
    assert derived.corrected_interval is None
    assert derived.decided_id == "B"


def test_derived_uncertainty_is_never_published() -> None:
    result = require_success(dr.assess(_law(), _request()))
    derived = require_success(result.derive_uncertainty(dr.IntervalRequest(level=0.9)))
    for route in (
        "ranking_probability",
        "optimality_confidence_set",
        "utility_interval",
        "selection_corrected_interval",
    ):
        with pytest.raises(DR) as raised:
            require_success(derived.publish(route))
        assert raised.value.reason_code == "cell_not_licensed"
        assert raised.value.detail == "decision_reliability.cell_not_licensed"
    with pytest.raises(DR) as raised:
        require_success(result.combined_uncertainty())
    assert raised.value.reason_code == "uncertainty_components_not_combined"


def test_refusals_for_small_samples_levels_and_undeclared_dependence() -> None:
    with pytest.raises(DR) as raised:
        require_success(
            dr.derive_uncertainty(_law(times=1), _request(), dr.IntervalRequest(level=0.95))
        )
    assert raised.value.detail == "decision_reliability.effective_sample_size_too_small"
    assert raised.value.reason_code == "route_not_supported"
    with pytest.raises(DR) as raised:
        require_success(dr.derive_uncertainty(_law(), _request(), dr.IntervalRequest(level=0.3)))
    assert raised.value.reason_code == "invalid_argument"
    with pytest.raises(DR) as raised:
        require_success(
            dr.derive_uncertainty(
                _law(within_stream="unknown"), _request(), dr.IntervalRequest(level=0.95)
            )
        )
    assert raised.value.reason_code == "sampling_dependence_unknown"


def test_simultaneous_multiplicity_widens_every_interval() -> None:
    marginal = require_success(
        dr.derive_uncertainty(_law(), _request(), dr.IntervalRequest(level=0.95))
    )
    simultaneous = require_success(
        dr.derive_uncertainty(
            _law(), _request(), dr.IntervalRequest(level=0.95, multiplicity="simultaneous")
        )
    )
    narrow, wide = _utility(marginal, "A"), _utility(simultaneous, "A")
    assert wide["upper"] - wide["lower"] > narrow["upper"] - narrow["lower"]
    assert wide["level"] == pytest.approx(1.0 - 0.05 / 8.0)
    # Bonferroni over the 8 reported intervals: the half-width is the 1 - 0.05/16 normal
    # quantile times the same standard error.
    critical = NormalDist().inv_cdf(1.0 - 0.05 / 16.0)
    assert (wide["upper"] - wide["lower"]) / 2.0 == pytest.approx(critical * SE_A, abs=1e-6)
    assert wide["standard_error"] == pytest.approx(SE_A, rel=1e-9)


def test_a_fresh_consumer_rederives_and_refuses_an_edited_derivation() -> None:
    result = require_success(dr.assess(_law(), _request()))
    derived = require_success(result.derive_uncertainty(dr.IntervalRequest(level=0.95)))
    artifact = bytes(result.export())
    consumed = require_success(
        dr.consume_derived_uncertainty(
            artifact, dr.IntervalRequest(level=0.95), draws_digest=_law().digests()[0]
        )
    )
    assert consumed.to_dict() == derived.to_dict()
    require_success(derived.verify(artifact))
    edited = derived.to_dict()
    edited["selection"]["corrected"]["lower"] -= 0.25
    with pytest.raises(DR):
        require_success(dr.DerivedUncertainty(edited).verify(artifact))
    with pytest.raises(DR):
        require_success(dr.consume_derived_uncertainty(artifact, decided_id="B"))


def _cohort(**kwargs: object) -> dr.CohortUtilityDraws:
    options: dict[str, object] = {
        "source_identity": "cohort-source",
        "actions": {"treat": _repeat([3, 3, 2, 2]), "wait": _repeat([1, 2, 1, 2])},
        "run_seed": 7,
        "label": "posterior",
        "joint_law": "one_shared_law",
        "reported_means": {"treat": 2.5, "wait": 1.5},
    }
    options.update(kwargs)
    return dr.CohortUtilityDraws(**options)  # type: ignore[arg-type]


def test_cohort_utilities_reach_the_carrier_and_derive() -> None:
    carrier = dr.reach_carrier(_cohort())
    assert carrier.producer == "cohort_decision"
    assert carrier.joint_law["standing"] == "one_shared_law"
    wire = carrier.draws.to_wire()
    labels = {c["segment"]["stream"]["label"] for c in wire["candidates"]}
    assert labels == {"cohort_decision/cohort-source/posterior"}
    derived = require_success(
        carrier.derive_uncertainty(_request(), dr.IntervalRequest(level=0.95))
    )
    assert derived.decided_id == "treat"
    wait = next(d for d in derived.numerical["differences_vs_decision"] if d["id"] == "wait")
    assert wait["interval"]["estimate"] == pytest.approx(1.0)
    assert wait["interval"]["standard_error"] == pytest.approx(math.sqrt(0.5 / 39.0), rel=1e-9)
    assessed = require_success(carrier.assess(_request()))
    assert assessed.report.decided_id == "treat"


def test_point_only_and_undeclared_outputs_refuse_with_typed_reasons() -> None:
    cases = [
        (dr.PointOnlyOutput("cohort_mean_decision", "cohort-source"), "joint_law_required"),
        (dr.PointOnlyOutput("sequential_rollout", "model"), "diagnostic_not_available"),
        (dr.PointOnlyOutput("sequential_off_policy", "log"), "diagnostic_not_available"),
        (_cohort(joint_law="undeclared"), "sampling_dependence_unknown"),
        (_cohort(joint_law="separate_marginals"), "joint_law_required"),
    ]
    for output, code in cases:
        with pytest.raises(DR) as raised:
            dr.reach_carrier(output)
        assert raised.value.reason_code == code
    with pytest.raises(DR) as raised:
        dr.reach_carrier(_cohort(reported_means={"treat": 9.0}))
    assert raised.value.detail == "decision_reliability.producer_mean_disagrees"


def _rollout(noise: str) -> dr.RolloutUtilityDraws:
    return dr.RolloutUtilityDraws(
        model_identity="model",
        stream_scheme="keyed",
        policies={"p": _repeat([3, 3, 2, 2]), "q": _repeat([2, 1, 3, 2])},
        seed=3,
        noise=noise,
        reported_means={"p": 2.5},
    )


def test_rollout_noise_decides_whether_ranking_is_licensed() -> None:
    keyed = dr.reach_carrier(_rollout("implementation_keyed"))
    assert keyed.joint_law["standing"] == "marginal_only"
    derived = require_success(
        keyed.derive_uncertainty(dr.ReliabilityRequest(), dr.IntervalRequest(level=0.95))
    )
    assert list(derived.numerical["probabilities"]) == []
    with pytest.raises(DR) as raised:
        require_success(keyed.derive_uncertainty(_request(), dr.IntervalRequest(level=0.95)))
    assert raised.value.reason_code == "joint_law_required"
    with pytest.raises(DR) as raised:
        require_success(keyed.assess(_request()))
    assert raised.value.reason_code == "joint_law_required"
    structural = dr.reach_carrier(_rollout("structural_exogenous"))
    ranked = require_success(
        structural.derive_uncertainty(_request(), dr.IntervalRequest(level=0.95))
    )
    assert list(ranked.numerical["probabilities"])


def test_off_policy_returns_carry_the_expected_utility_only() -> None:
    carrier = dr.reach_carrier(
        dr.OffPolicyReturnDraws(
            model_identity="model",
            log_identity="log",
            policies={"p": _repeat([3, 3, 2, 2]), "q": _repeat([1, 2, 1, 2])},
            run_seed=5,
        )
    )
    assert carrier.producer == "sequential_off_policy"
    request = dr.ReliabilityRequest(functional=dr.Functional.cvar(0.5))
    with pytest.raises(DR) as raised:
        require_success(carrier.derive_uncertainty(request, dr.IntervalRequest(level=0.95)))
    assert raised.value.reason_code == "route_not_supported"


def test_coverage_design_is_inspectable_data_that_measures_nothing() -> None:
    design = dr.coverage_design()
    expected = {
        "ranking_probability_interval",
        "optimality_confidence_set",
        "selection_corrected_interval",
        "held_out_interval",
        "winners_curse_bound",
        "utility_interval",
    }
    assert expected <= set(design.routes)
    for cell in design.cells:
        assert cell["min_replicates"] >= 2000
        assert cell["boundary_probes"]
        assert cell["nominal_levels"]
    assert design.cell("ranking_probability_interval")["truth"] == "closed_form"
    assert design.cell("optimality_confidence_set")["truth"] == "simulation_known_optimum"
    assert design.to_dict()["measured"] is False
    with pytest.raises(CausalValueError, match="no coverage cell"):
        design.cell("no_such_route")
