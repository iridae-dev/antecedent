"""Caller learners on the compiled trial-to-target AIPW transport.

``analyze_trial_aipw`` declares two registered providers as the source-outcome and
source-membership learners and runs the ``TrialAipwLearnerTransport`` program. The route
evaluates the contrast from portable fold models.

Oracle. A noise-free randomized source trial with outcome ``1 + 2 t`` and known assignment
probability 1/2 has target contrast 2. A caller least-squares outcome learner is that
regression. The bit-for-bit twin of the compiled route and the factory kernel is in
``crates/antecedent/tests/compiled_twins_trial_joint_learners.rs``.
"""

from __future__ import annotations

import re

import antecedent as ac
import pytest
from antecedent import _reason_details as reason_details
from antecedent.errors import CausalUnsupportedError
from antecedent.external_learners import declare_portable_route
from antecedent.transport import advanced

from test_compiled_smoothed_dose_learners import _learners
from test_learned_transport import trial_fixture


def test_caller_learners_recover_the_known_trial_contrast():
    query, data = trial_fixture()
    result = advanced.analyze_trial_aipw(
        query, data, learners=_learners(), folds=3, execution=ac.Execution(seed=11)
    )
    assert result.estimate == pytest.approx(2.0, abs=1e-8)
    assert result.route == "TrialAipwLearnerTransport"
    assert re.fullmatch(r"[0-9a-f]{64}", result.program_identity)
    assert result.claim == "point_only"
    assert result.calibration == "not_measured"
    assert result.external_nuisances_unverified is True
    assert result.receipts
    again = advanced.analyze_trial_aipw(
        query, data, learners=_learners(), folds=3, execution=ac.Execution(seed=11)
    )
    assert again.program_identity == result.program_identity
    assert again.estimate == result.estimate
    other = advanced.analyze_trial_aipw(
        query,
        data,
        learners=_learners(configuration={"variant": 2}),
        folds=3,
        execution=ac.Execution(seed=11),
    )
    assert other.program_identity != result.program_identity


def test_a_non_portable_learner_is_refused_at_declaration():
    query, data = trial_fixture()
    learners = _learners(codec="json")
    with pytest.raises(
        CausalUnsupportedError, match=reason_details.dose_response.LEARNER_NOT_PORTABLE
    ) as caught:
        advanced.analyze_trial_aipw(query, data, learners=learners)
    assert caught.value.reason_code == "route_not_supported"
    with pytest.raises(
        CausalUnsupportedError, match=reason_details.dose_response.LEARNER_NOT_PORTABLE
    ):
        declare_portable_route("trial_aipw", learners)
