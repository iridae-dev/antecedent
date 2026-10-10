"""Capability declarations select real adapters without granting missing state."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
from antecedent.prediction import FittedEffectModel
from antecedent.recalc import (
    Capabilities,
    RecalcRequest,
    RecalcUnavailable,
    Stage,
    TargetWeights,
    Utility,
)
from antecedent.recalc_capabilities import (
    Family,
    Operation,
    RetainedKind,
    capability,
    capability_matrix,
    require_adapter,
    retained_kind,
)
from antecedent.recalc_cell import CrossfitSession, resume_from_scores


def resolve(path):
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        try:
            value = import_module(".".join(parts[:end]))
        except ModuleNotFoundError:
            continue
        for part in parts[end:]:
            value = getattr(value, part)
        return value
    raise AssertionError(f"adapter path unavailable: {path}")


def test_matrix_covers_all_families_operations_and_complete_identity_categories():
    rows = capability_matrix()
    assert {(r.family, r.operation) for r in rows} == {
        (family, operation) for family in Family for operation in Operation
    }
    assert len(rows) == len(Family) * len(Operation)
    for row in rows:
        assert row.retained_objects
        assert row.compatible_operation
        assert row.refusal_code == "route_not_supported"
        assert {i.stage for i in row.identities} == {s for s in Stage if s.is_input}
        assert all(i.fields for i in row.identities)
        for adapter in row.adapters:
            assert callable(resolve(adapter.python_path))
            assert adapter.rust_path.startswith("antecedent")


@pytest.mark.parametrize("family", list(Family))
@pytest.mark.parametrize("operation", list(Operation))
def test_readable_results_or_flags_never_grant_execution(family, operation):
    for state in (object(), {"portable_scores": True}, Capabilities()):
        assert retained_kind(state) == RetainedKind.READABLE
        with pytest.raises(RecalcUnavailable) as error:
            require_adapter(family, operation, state)
        assert error.value.reason_code == "route_not_supported"
        assert error.value.detail == "recalc.capability_adapter_unavailable"
        assert error.value.stage == "score_artifact"


def test_empty_session_and_python_predictor_wrapper_do_not_prove_retention():
    class FalselyLiveSession(CrossfitSession):
        @property
        def is_live(self):
            return True

    assert retained_kind(CrossfitSession()) == RetainedKind.READABLE
    # Query the native retained state, not an overridable Python declaration.
    assert retained_kind(FalselyLiveSession()) == RetainedKind.READABLE
    # Public wrappers accept a native object; an arbitrary impostor is not one.
    assert retained_kind(FittedEffectModel(object())) == RetainedKind.READABLE


def test_live_score_and_resume_adapters_execute_at_the_declared_coordinate():
    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1, 1, 20), 30)
    t = rng.binomial(1, 0.5, len(z)).astype(float)
    request = RecalcRequest(
        data={"z": z, "t": t, "y": 2 * t + z},
        edges=(("z", "t"), ("z", "y"), ("t", "y")),
        treatment="t",
        outcome="y",
        utility=Utility(1, 0),
    )
    session = CrossfitSession()
    initial = session.execute(request, seed=3)
    assert initial.receipt.totals.fold_fits == 10
    assert retained_kind(session) == RetainedKind.LIVE_SCORES
    adapter = require_adapter(Family.DOUBLY_ROBUST, Operation.TARGET, session)
    assert adapter.retained == RetainedKind.LIVE_SCORES
    assert adapter.python_path == "antecedent.recalc_cell.CrossfitSession.execute"
    weights = TargetWeights(np.exp(0.2 * z), ("z",))
    retarget = session.execute(replace(request, target=weights), seed=3)
    assert retarget.receipt.totals.fold_fits == 0
    independent = CrossfitSession().execute(replace(request, target=weights), seed=3)
    assert retarget.law == independent.law
    scores = session.export_frozen_scores()
    resumed = resume_from_scores(
        scores,
        variables=("z", "t", "y"),
        edges=request.edges,
        utility=request.utility,
        quantity="average_effect",
        expected_identity=scores.identity,
    )
    selected = require_adapter(Family.DOUBLY_ROBUST, Operation.TARGET, resumed)
    assert selected.retained == RetainedKind.SCORES
    result = resumed.retarget(weights, row_ids=resumed.row_ids)
    assert result.receipt.totals.fold_fits == 0
    assert result.law == independent.law
    assert result.law.ate == pytest.approx(2.0, abs=1e-6)
    with pytest.raises(RecalcUnavailable) as error:
        require_adapter(Family.DOUBLY_ROBUST, Operation.DATA, resumed)
    assert error.value.reason_code == "route_not_supported"


def test_missing_adapters_and_inference_never_inherit_adjacent_routes():
    assert not capability(Family.DOUBLY_ROBUST, Operation.PROVIDER).adapters
    assert not capability(Family.DOUBLY_ROBUST, Operation.INFERENCE).adapters
    for family in (Family.STATIC, Family.DESIGN):
        assert not capability(family, Operation.INFERENCE).adapters
    assert not capability(Family.BAYESIAN, Operation.ACTION_GRID).adapters
    assert not capability(Family.BAYESIAN, Operation.TARGET).adapters
    assert not capability(Family.TEMPORAL, Operation.INFERENCE).adapters
    assert not capability(Family.TEMPORAL, Operation.PROVIDER).adapters


def test_verified_predictor_supports_point_prediction_without_becoming_score_state():
    import antecedent as ac
    from antecedent.estimators import DRLearner
    from antecedent.learners import Linear

    rng = np.random.default_rng(19)
    z = np.tile(np.linspace(-1, 1, 20), 30)
    t = rng.binomial(1, 0.5, len(z)).astype(float)
    result = ac.analyze(
        {"z": z, "t": t, "y": 2 * t + z},
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ac.AverageEffect(treatment="t", outcome="y"),
        # Correctly specified unpenalized outcome/final regressions reproduce
        # this deterministic linear SCM; default Ridge adds shrinkage bias.
        estimator=DRLearner(outcome=Linear(), final_learner=Linear()),
        bootstrap=0,
        refute=False,
        seed=3,
    )
    model = FittedEffectModel.load(result.fitted_model.export())
    assert retained_kind(model) == RetainedKind.PREDICTOR
    adapter = require_adapter(Family.DOUBLY_ROBUST, Operation.DATA, model)
    assert adapter.python_path == "antecedent.prediction.FittedEffectModel.predict"
    predicted = model.predict({"z": [-0.5, 0.0, 0.5]})
    assert predicted.values == pytest.approx([2, 2, 2], abs=1e-6)
    assert predicted.parent_claim == model.parent_claim
    assert predicted.to_dict()["uncertainty"]["status"] == "unavailable"
    with pytest.raises(RecalcUnavailable) as error:
        require_adapter(Family.DOUBLY_ROBUST, Operation.TARGET, model)
    assert error.value.reason_code == "route_not_supported"


def test_adjusted_capabilities_require_actual_native_fit_and_refuse_provider():
    from antecedent.recalc_adjusted import AdjustedRequest, AdjustedSession

    class FalselyLiveAdjusted(AdjustedSession):
        @property
        def is_live(self):
            return True

    assert retained_kind(FalselyLiveAdjusted()) == RetainedKind.READABLE
    request = AdjustedRequest(
        data={
            "t": np.tile([0.0, 1.0], 100),
            "y": np.tile([0.0, 2.0], 100) + np.linspace(-0.1, 0.1, 200),
        },
        edges=(("t", "y"),),
        treatments=("t",),
        outcome="y",
        adjustment=(),
        utility=Utility(1),
    )
    session = AdjustedSession()
    first = session.execute(request, seed=3)
    assert first.receipt.totals.model_fits == 1
    assert retained_kind(session) == RetainedKind.LIVE_FIT
    for operation in Operation:
        if operation == Operation.PROVIDER:
            with pytest.raises(RecalcUnavailable):
                require_adapter(Family.ADJUSTED, operation, session)
        else:
            adapter = require_adapter(Family.ADJUSTED, operation, session)
            assert adapter.retained == RetainedKind.LIVE_FIT
            assert callable(resolve(adapter.python_path))
    result = session.execute(replace(request, utility=Utility(2, 0.1)), seed=3)
    assert result.receipt.totals.model_fits == 0
    assert result.plan.recomputed_computations == (Stage.DECISION,)
