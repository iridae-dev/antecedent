"""Selective execution of the fixed-population temporal counterfactual.

Reference. The two-slice SCM of ``test_temporal_counterfactual`` is known in closed form, so every
unit's exogenous history and both counterfactual outcomes are hand-computable there (for example
unit 0 under ``(1, 1)`` versus ``(0, 0)`` is ``7.73`` versus ``2.83``, contrast ``4.9`` for every
unit). What this module adds is the reuse contract, stated as exact counts rather than timing: the
first request over four units abduces four units; a second request that changes only the named
action histories abduces none and reuses four, computes only the two worlds per unit (eight
predictions), and returns the very numbers and receipt a fresh one-shot call gives; a request that
changes the factual evidence, the mechanism fit or the snapshot is a different abduction identity
and abduces four more. Counts are integers and equalities are exact (bit-identical), so no
tolerance applies except where an outcome is compared with its closed form (``1e-9``).
"""

from __future__ import annotations

import pytest
from antecedent.temporal import TemporalRefusal
from antecedent.temporal_counterfactual import (
    ActionHistory,
    TemporalCounterfactualSession,
    temporal_fixed_population,
)

from test_temporal_counterfactual import (
    MINUS,
    NOISE,
    PLUS,
    histories,
    mechanisms,
    simulate,
)


def arguments(**overrides):
    base = {
        "plus": ActionHistory("always_treat", PLUS),
        "minus": ActionHistory("never_treat", MINUS),
        "mechanisms": mechanisms(),
        "snapshot": "snapshot-1",
    }
    base.update(overrides)
    return base


def status(result, stage):
    return next(s.status for s in result.stages if s.stage == stage)


def test_a_changed_action_history_reuses_the_abduction_and_changes_only_the_worlds():
    session = TemporalCounterfactualSession()
    first = session.evaluate(histories(), **arguments())
    assert (first.work.abductions, first.work.abductions_reused) == (4, 0)
    assert first.work.world_predictions == 8
    assert not first.abduction_reused
    assert status(first, "mechanism_fitting") == "declared"
    assert status(first, "factual_abduction") == "computed"
    assert session.retained == 1

    # Same evidence and fit, different named actions: the abduction is reused.
    other_plus = ActionHistory("treat_then_stop", (1.0, 0.0))
    second = session.evaluate(histories(), **arguments(plus=other_plus))
    assert (second.work.abductions, second.work.abductions_reused) == (0, 4)
    assert second.work.world_predictions == 8
    assert second.abduction_reused
    assert status(second, "factual_abduction") == "reused"
    assert status(second, "exogenous_draws") == "reused"
    assert status(second, "action_prediction") == "computed"
    assert session.retained == 1
    # The abduced draws are the same, the worlds are not.
    assert second.receipt.exogenous_digest == first.receipt.exogenous_digest
    assert second.receipt.plus_digest != first.receipt.plus_digest
    assert second.receipt.digest != first.receipt.digest
    for row, noise in zip(second.units, NOISE, strict=True):
        assert row.plus_outcome == pytest.approx(simulate(noise, (1.0, 0.0))[4], abs=1e-9)
        assert row.minus_outcome == pytest.approx(simulate(noise, MINUS)[4], abs=1e-9)

    # The session's cumulative work is the sum of its executions.
    total = session.work
    assert (total.abductions, total.abductions_reused, total.world_predictions) == (4, 4, 16)


def test_the_selective_answer_is_bit_identical_to_the_one_shot_call():
    session = TemporalCounterfactualSession()
    for _ in range(2):  # the second call is served from the retained abduction
        selective = session.evaluate(histories(), **arguments())
        one_shot = temporal_fixed_population(histories(), **arguments())
        assert selective.units == one_shot.units
        assert (selective.mean_plus, selective.mean_minus) == (one_shot.mean_plus, one_shot.mean_minus)
        assert selective.contrast == one_shot.contrast
        assert selective.receipt == one_shot.receipt
        assert selective.identity == one_shot.identity
        assert selective.inference_claim == "point_only"
        assert selective.contrast == pytest.approx(4.9, abs=1e-9)


def test_changed_evidence_fit_or_snapshot_is_a_different_abduction():
    session = TemporalCounterfactualSession()
    session.evaluate(histories(), **arguments())

    shifted = [[u + 0.01 for u in row] for row in NOISE]
    changed_evidence = session.evaluate(histories(shifted), **arguments())
    assert (changed_evidence.work.abductions, changed_evidence.work.abductions_reused) == (4, 0)

    changed_fit = session.evaluate(histories(), **arguments(mechanisms=mechanisms("fit-refit")))
    assert (changed_fit.work.abductions, changed_fit.work.abductions_reused) == (4, 0)

    changed_snapshot = session.evaluate(histories(), **arguments(snapshot="snapshot-2"))
    assert (changed_snapshot.work.abductions, changed_snapshot.work.abductions_reused) == (4, 0)
    assert session.retained == 4

    # The original evidence, fit and snapshot are still retained.
    again = session.evaluate(histories(), **arguments())
    assert (again.work.abductions, again.work.abductions_reused) == (0, 4)


def test_a_refused_request_leaves_the_retained_state_unchanged():
    session = TemporalCounterfactualSession()
    session.evaluate(histories(), **arguments())
    before = (session.retained, session.work)
    with pytest.raises(TemporalRefusal):
        session.evaluate(histories(), **arguments(latent_confounding=True))
    assert (session.retained, session.work) == before
    again = session.evaluate(histories(), **arguments())
    assert again.work.abductions_reused == 4
