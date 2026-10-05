"""2.2 E7: conditional odds ratio of matched case-control sets through the Python surface.

Truth is hand-enumerated: with 1:1 matching the conditional MLE is the discordant-pair
ratio b / c, and a single type of 1:2 set has theta = 2a / (m - a). The profile-likelihood
oracle below is a separate pure-Python implementation (subset enumeration), not the
hypergeometric-weight solver under test.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest
from antecedent.errors import (
    CausalResourceError,
    CausalSerializationError,
    CausalTypeError,
    CausalUnsupportedError,
)
from antecedent.matched import matched_case_control_odds_ratio, replay_matched_case_control
from antecedent.state import CancellationToken

from _refusal import assert_registered_refusal

DECLARED = "matched_case_control"


def rows(sets):
    """Flatten ``[[(case, exposed), ...], ...]`` into parallel row lists."""
    stratum, case, exposed = [], [], []
    for i, members in enumerate(sets):
        for y, x in members:
            stratum.append(f"set{i}")
            case.append(y)
            exposed.append(x)
    return stratum, case, exposed


def fit(sets, **options):
    stratum, case, exposed = rows(sets)
    return matched_case_control_odds_ratio(
        stratum=stratum, case=case, exposed=exposed, sampling=DECLARED, **options
    )


def pairs(b, c, both=0, neither=0):
    return (
        [[(1, 1), (0, 0)]] * b
        + [[(1, 0), (0, 1)]] * c
        + [[(1, 1), (0, 1)]] * both
        + [[(1, 0), (0, 0)]] * neither
    )


def profile_log_odds(sets):
    """Brute-force argmax of the conditional log likelihood (case subsets of each set)."""

    def loglik(beta):
        total = 0.0
        for members in sets:
            k = sum(y for y, _ in members)
            observed = sum(x for y, x in members if y)
            denominator = sum(
                math.exp(beta * sum(members[j][1] for j in subset))
                for subset in itertools.combinations(range(len(members)), k)
            )
            total += beta * observed - math.log(denominator)
        return total

    lo, hi = -8.0, 8.0
    for _ in range(200):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if loglik(a) < loglik(b):
            lo = a
        else:
            hi = b
    return 0.5 * (lo + hi)


def test_one_to_one_matching_is_the_discordant_pair_ratio():
    sets = [*pairs(7, 3, both=4, neither=5), [(1, 1)], [(0, 1), (0, 0)]]
    result = fit(sets)
    assert result.odds_ratio == pytest.approx(7 / 3, abs=1e-9)
    assert result.log_odds_ratio == pytest.approx(math.log(7 / 3), abs=1e-9)
    counts = result.counts
    assert (counts.total, counts.informative) == (21, 10)
    assert (counts.exposure_concordant, counts.outcome_degenerate, counts.singleton) == (9, 2, 1)
    assert counts.informative + counts.exposure_concordant + counts.outcome_degenerate == (
        counts.total
    )
    assert result.claim == "point_only"
    assert result.estimand == "conditional_odds_ratio"


@pytest.mark.parametrize(("a", "m", "truth"), [(3, 5, 3.0), (1, 4, 2 / 3), (4, 6, 4.0)])
def test_one_to_two_sets_of_one_type_have_the_closed_form(a, m, truth):
    sets = [[(1, 1), (0, 0), (0, 0)]] * a + [[(1, 0), (0, 1), (0, 0)]] * (m - a)
    assert fit(sets).odds_ratio == pytest.approx(truth, abs=1e-8)


def test_mixed_set_sizes_match_the_brute_force_profile_likelihood():
    sets = (
        [[(1, 1), (0, 0), (0, 0)]] * 3
        + [[(1, 0), (0, 1), (0, 0)]] * 2
        + [[(1, 1), (0, 1), (0, 0)]] * 2
        + [[(1, 1), (0, 0), (0, 0), (0, 0)]]
        + [[(1, 0), (0, 1), (0, 0), (0, 0)]] * 2
        + [[(1, 1), (0, 1), (0, 1), (0, 0)]]
        + [[(1, 1), (1, 0), (0, 0), (0, 0)]]
        + [[(1, 0), (1, 0), (0, 1), (0, 0)]]
        + [[(1, 1), (1, 1), (0, 0), (0, 1)]]
        + [[(1, 0), (0, 0), (0, 0)], [(1, 1), (0, 1)], [(0, 1), (0, 0)], [(1, 1)]]
    )
    result = fit(sets)
    assert result.log_odds_ratio == pytest.approx(profile_log_odds(sets), abs=1e-6)
    assert result.counts.informative == 14


def test_uninformative_sets_are_counted_and_leave_the_estimate_unchanged():
    base = fit([[(1, 1), (0, 0), (0, 0)]] * 3 + [[(1, 0), (0, 1), (0, 0)]] * 2)
    padded = fit(
        [[(1, 1), (0, 0), (0, 0)]] * 3
        + [[(1, 0), (0, 1), (0, 0)]] * 2
        + [[(1, 1), (0, 1), (0, 1)]] * 50
        + [[(1, 0), (0, 0), (0, 0)]] * 50
        + [[(0, 1), (0, 0)]] * 20
        + [[(1, 1)]] * 10
    )
    assert padded.log_odds_ratio == base.log_odds_ratio
    assert padded.counts.exposure_concordant == 100
    assert padded.counts.outcome_degenerate == 30
    assert padded.counts.singleton == 10


def test_stratum_labels_are_compared_as_strings_and_order_is_irrelevant():
    stratum, case, exposed = rows(pairs(2, 1))
    ids = [int(label.removeprefix("set")) for label in stratum]
    forward = matched_case_control_odds_ratio(
        stratum=ids, case=case, exposed=exposed, sampling=DECLARED
    )
    backward = matched_case_control_odds_ratio(
        stratum=[str(i) for i in ids[::-1]],
        case=case[::-1],
        exposed=exposed[::-1],
        sampling=DECLARED,
    )
    assert forward.odds_ratio == pytest.approx(2.0, abs=1e-9)
    assert backward.log_odds_ratio == forward.log_odds_ratio


def refusal(call):
    with pytest.raises(CausalUnsupportedError) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_risk_scale_estimands_are_refused_without_prevalence():
    for estimand in ("population_risk", "absolute_risk", "risk_difference", "risk_ratio"):
        error = refusal(lambda estimand=estimand: fit(pairs(7, 3), estimand=estimand))
        assert error.reason_code == "effect_not_identified"
        assert "matched_case_control.absolute_risk_not_identified" in str(error)


def test_another_estimand_is_not_supported():
    error = refusal(lambda: fit(pairs(7, 3), estimand="hazard_ratio"))
    assert error.reason_code == "route_not_supported"
    assert "matched_case_control.estimand_not_supported" in str(error)


def test_an_interval_is_withheld_with_a_typed_refusal():
    error = refusal(lambda: fit(pairs(7, 3), level=0.95))
    assert error.reason_code == "cell_not_licensed"
    assert "matched_case_control.interval_withheld" in str(error)


def test_the_sampling_design_must_be_declared_and_be_the_matched_design():
    stratum, case, exposed = rows(pairs(7, 3))
    with pytest.raises(TypeError, match="sampling"):
        matched_case_control_odds_ratio(stratum=stratum, case=case, exposed=exposed)
    error = refusal(
        lambda: matched_case_control_odds_ratio(
            stratum=stratum, case=case, exposed=exposed, sampling="cohort"
        )
    )
    assert error.reason_code == "invalid_argument"
    assert "matched_case_control.sampling_design" in str(error)


def test_no_informative_set_and_a_monotone_likelihood_are_refused():
    error = refusal(lambda: fit(pairs(0, 0, both=3, neither=2) + [[(1, 1)]]))
    assert error.reason_code == "effect_not_identified"
    assert "matched_case_control.no_informative_sets" in str(error)
    for sets in (pairs(4, 0), pairs(0, 4)):
        error = refusal(lambda sets=sets: fit(sets))
        assert error.reason_code == "route_not_supported"
        assert "matched_case_control.estimate_not_finite" in str(error)


def test_malformed_rows_are_refused():
    base = {"stratum": ["a", "a"], "case": [1, 0], "exposed": [1, 0], "sampling": DECLARED}
    for override in (
        {"exposed": [1]},
        {"case": [1, 2]},
        {"exposed": [float("nan"), 0]},
        {"case": [1, 0.5]},
        {"stratum": [], "case": [], "exposed": []},
    ):
        error = refusal(
            lambda override=override: matched_case_control_odds_ratio(**{**base, **override})
        )
        assert error.reason_code == "invalid_argument"
        assert "matched_case_control.invalid_data" in str(error)
    with pytest.raises(CausalTypeError):
        matched_case_control_odds_ratio(**{**base, "case": [1, None]})


def test_a_cancelled_solve_is_a_budget_stop_never_a_verdict():
    token = CancellationToken()
    token.cancel()
    error = refusal(lambda: fit(pairs(7, 3), cancel=token))
    assert error.reason_code == "transport_budget_cancel"
    assert "matched_case_control.budget" in str(error)
    assert fit(pairs(7, 3)).odds_ratio == pytest.approx(7 / 3, abs=1e-9)


def sealed(body):
    """Re-seal an edited artifact body with a correct digest (a forged but consistent file)."""
    import hashlib

    body = {k: v for k, v in body.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return json.dumps({**body, "digest": hashlib.sha256(canonical.encode()).hexdigest()})


def test_an_exported_result_is_replayed_by_an_independent_consumer():
    sets = [*pairs(7, 3, both=4, neither=5), [(1, 1)], [(0, 1), (0, 0)]]
    sets += [[(1, 1), (0, 0), (0, 0)]] * 3 + [[(1, 0), (0, 1), (0, 0)]] * 2
    result = fit(sets)
    assert result.set_types == (
        (1, 1, 1, 1, 1),
        (2, 0, 1, 0, 1),
        (2, 1, 0, 0, 5),
        (2, 1, 1, 0, 3),
        (2, 1, 1, 1, 7),
        (2, 1, 2, 1, 4),
        (3, 1, 1, 0, 2),
        (3, 1, 1, 1, 3),
    )
    again = replay_matched_case_control(result.export())
    assert again.log_odds_ratio == result.log_odds_ratio
    assert again.counts == result.counts
    assert again.set_types == result.set_types
    assert replay_matched_case_control(again.export()).export() == result.export()
    body = json.loads(result.export())
    assert body["claim"] == "point_only" and "interval" not in body


def test_a_tampered_artifact_is_refused_for_the_right_reason():
    artifact = fit(pairs(7, 3)).export()
    body = json.loads(artifact)
    for edit in (
        {"log_odds_ratio": body["log_odds_ratio"] + 1e-9},
        {"counts": {**body["counts"], "informative": 11}},
        {"set_types": [[2, 1, 1, 1, 8], [2, 1, 1, 0, 2]]},
    ):
        with pytest.raises(CausalSerializationError, match="digest"):
            replay_matched_case_control(json.dumps({**body, **edit}))
        # Re-sealed edits keep a valid digest but no longer reproduce.
        with pytest.raises(CausalSerializationError, match="reproduce"):
            replay_matched_case_control(sealed({**body, **edit}))
    for forged in (
        {"format": "matched_case_control_v2"},
        {"claim": "calibrated"},
        {"interval": [0.5, 2.0]},
        {"set_types": [[2, 1, 1, 2, 1]]},
        {"set_types": "nope"},
        {"sampling": 3},
    ):
        with pytest.raises(CausalSerializationError):
            replay_matched_case_control(sealed({**body, **forged}))
    with pytest.raises(CausalSerializationError):
        replay_matched_case_control("not json")


def test_a_replay_that_would_expand_past_the_row_cap_is_refused_before_work():
    body = json.loads(fit(pairs(7, 3)).export())
    with pytest.raises(CausalResourceError):
        replay_matched_case_control(sealed({**body, "set_types": [[2, 1, 1, 1, 10**12]]}))
    with pytest.raises(CausalResourceError):
        replay_matched_case_control(fit(pairs(7, 3)).export(), max_rows=19)
    assert replay_matched_case_control(fit(pairs(7, 3)).export(), max_rows=20).counts.total == 10
