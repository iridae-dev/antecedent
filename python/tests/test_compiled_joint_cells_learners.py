"""Caller learners on the compiled factorized joint-cell fit.

``analyze_joint_cells`` declares two registered providers as the per-cell outcome and
conditional-propensity learners and runs the ``JointCellsLearnerFit`` program. The route
evaluates each cell from portable fold models.

Oracle. The fixture of ``test_derived_treatment`` draws ``Y = 0.4 cell + 1{cell = 3} + 0.5 z``
plus noise, so ``E[Y^do(cell)] = 0.4 cell + 1{cell = 3} + 0.25``. A caller least-squares
outcome learner on ``z`` and a caller logistic propensity are correctly specified inside each
cell. The bit-for-bit twin of the compiled route and the factory kernel is in
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
from test_derived_treatment import declaration, draw, truth


def test_caller_learners_recover_the_known_cell_means():
    result = advanced.analyze_joint_cells(
        draw(4000, 1, [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]]),
        declaration(),
        outcome="y",
        adjustment=["z"],
        learners=_learners(),
        all_orderings=False,
        folds=2,
        execution=ac.Execution(seed=7),
    )
    assert result.route == "JointCellsLearnerFit"
    assert re.fullmatch(r"[0-9a-f]{64}", result.program_identity)
    assert result.claim == "point_only"
    assert result.calibration == "not_measured"
    assert result.external_nuisances_unverified is True
    assert result.treatments == ("t0", "t1")
    assert len(result.supported) == 4
    assert result.receipts
    for cell in result.cells:
        assert cell.estimate == pytest.approx(truth(cell.cell), abs=0.05)
    again = advanced.analyze_joint_cells(
        draw(4000, 1, [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]]),
        declaration(),
        outcome="y",
        adjustment=["z"],
        learners=_learners(),
        all_orderings=False,
        folds=2,
        execution=ac.Execution(seed=7),
    )
    assert again.program_identity == result.program_identity


def test_a_non_portable_learner_is_refused_at_declaration():
    learners = _learners(codec="json")
    with pytest.raises(
        CausalUnsupportedError, match=reason_details.dose_response.LEARNER_NOT_PORTABLE
    ) as caught:
        advanced.analyze_joint_cells(
            draw(40, 1, [[0.4, 0.2, 0.2, 0.2], [0.1, 0.3, 0.2, 0.4]]),
            declaration(),
            outcome="y",
            adjustment=["z"],
            learners=learners,
        )
    assert caught.value.reason_code == "route_not_supported"
    with pytest.raises(
        CausalUnsupportedError, match=reason_details.dose_response.LEARNER_NOT_PORTABLE
    ):
        declare_portable_route("joint_cells", learners)
