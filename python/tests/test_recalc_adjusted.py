"""Executed adjusted-family reuse, independent joint-fit oracles and process boundaries."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent.estimators import GlmAdjustment, LinearAdjustment
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success
from antecedent.recalc import (
    RecalcReceipt,
    RecalcRefusal,
    Stage,
    TargetWeights,
    Utility,
)
from antecedent.recalc_adjusted import (
    AdjustedRequest,
    AdjustedSession,
    CategoricalContrast,
    CategoricalModel,
    GlmModel,
    LinearModel,
    NumericContrast,
)


def linear_request(vector=False):
    rng = np.random.default_rng(61)
    n = 420
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    t2 = (rng.random(n) < 1 / (1 + np.exp(-0.5 * z - 0.8 * t))).astype(float)
    y = 2 * t - 0.7 * t2 + 1.5 * z + rng.normal(scale=0.2, size=n)
    data = {"z": z, "t": t, "y": y}
    edges = [("z", "t"), ("z", "y"), ("t", "y")]
    if vector:
        data["t2"] = t2
        edges += [("z", "t2"), ("t2", "y")]
    return AdjustedRequest(
        data=data,
        edges=edges,
        treatments=("t", "t2") if vector else ("t",),
        outcome="y",
        adjustment=("z",),
        utility=Utility(1),
        contrast=NumericContrast((1, 1), (0, 0)) if vector else NumericContrast(),
    )


def numpy_joint_oracle(request):
    x = np.column_stack(
        [np.ones(len(request.data[request.outcome]))]
        + [request.data[n] for n in (*request.adjustment, *request.treatments)]
    )
    y = request.data[request.outcome]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    covariance = np.linalg.inv(x.T @ x) * (residuals @ residuals) / (len(y) - x.shape[1])
    contrast = request.contrast
    weights = np.r_[
        np.zeros(1 + len(request.adjustment)), np.subtract(contrast.active, contrast.control)
    ]
    return weights @ beta, np.sqrt(weights @ covariance @ weights), beta, covariance


def assert_full_rerun(request, selective, *, seed=3):
    full = require_success(AdjustedSession().execute(request, execution=Execution(seed=seed)))
    assert full.receipt.totals.model_fits == 1
    assert full.receipt.totals.fold_fits == 0
    assert selective.law.ate == pytest.approx(full.law.ate, rel=1e-11, abs=1e-12)
    assert selective.law.std_error == pytest.approx(full.law.std_error, rel=1e-11, abs=1e-12)
    assert selective.decision == full.decision


@pytest.mark.parametrize("vector", [False, True])
def test_adjusted_native_fit_counts_and_joint_covariance_match_independent_ols(vector):
    request = linear_request(vector)
    session = AdjustedSession()
    first = require_success(session.execute(request, execution=Execution(seed=3)))
    expected, se, _, covariance = numpy_joint_oracle(request)
    assert first.law.ate == pytest.approx(expected, rel=1e-10)
    assert first.law.std_error == pytest.approx(se, rel=1e-10)
    assert first.receipt.totals.model_fits == 1
    assert first.receipt.totals.fold_fits == 0
    assert first.receipt.counts(Stage.SCORE_ARTIFACT).model_fits == 1
    if vector:
        assert abs(covariance[-1, -2]) > 1e-6
        naive = np.sqrt(covariance[-1, -1] + covariance[-2, -2])
        assert abs(se - naive) > 1e-4
    else:
        independent = require_success(
            ac.analyze(
                request.data,
                graph=request.edges,
                query=ac.AverageEffect(treatment="t", outcome="y"),
                estimator=LinearAdjustment(bootstrap=0),
                refute=False,
                execution=ac.Execution(seed=3),
            )
        )
        assert first.law.ate == pytest.approx(independent.effect, abs=1e-10)
    same = require_success(session.execute(request, execution=Execution(seed=3)))
    assert same.receipt.totals.total == 0
    assert same.plan.recomputed_computations == ()
    receipt = require_success(
        RecalcReceipt.consume(
            bytes(first.receipt.export()), expected_identity=first.receipt.identity
        )
    )
    assert receipt.totals.model_fits == 1
    assert "model_fits=1" in receipt.explain()


def test_adjusted_utility_and_numeric_contrasts_reuse_joint_fit():
    request = linear_request(True)
    session = AdjustedSession()
    require_success(session.execute(request, execution=Execution(seed=3)))
    utility = replace(request, utility=Utility(3, 0.5))
    changed = require_success(session.execute(utility, execution=Execution(seed=3)))
    assert changed.receipt.totals.model_fits == 0
    assert changed.plan.recomputed_computations == (Stage.DECISION,)
    assert_full_rerun(utility, changed)
    contrast = replace(utility, contrast=NumericContrast((1, 0), (0, 1)))
    changed = require_success(session.execute(contrast, execution=Execution(seed=3)))
    assert changed.receipt.totals.model_fits == 0
    assert changed.plan.status(Stage.SCORE_ARTIFACT).reused
    expected, se, _, _ = numpy_joint_oracle(contrast)
    assert changed.law.ate == pytest.approx(expected, rel=1e-10)
    assert changed.law.std_error == pytest.approx(se, rel=1e-10)
    assert_full_rerun(contrast, changed)


@pytest.mark.parametrize("change", ["outcome", "rows", "covariance", "seed", "graph"])
def test_adjusted_changed_fit_inputs_refit_and_match_full_rerun(change):
    request = linear_request(True)
    session = AdjustedSession()
    require_success(session.execute(request, execution=Execution(seed=3)))
    seed = 3
    if change == "outcome":
        data = dict(request.data)
        data["y"] = data["y"] + 0.3 * data["t"]
        request = replace(request, data=data)
    elif change == "rows":
        request = replace(request, data={k: np.asarray(v)[:-1] for k, v in request.data.items()})
    elif change == "covariance":
        request = replace(request, model=LinearModel("hc3"))
    elif change == "seed":
        seed = 4
    else:
        request = replace(request, edges=(*request.edges, ("t", "t2")))
    result = require_success(session.execute(request, execution=Execution(seed=seed)))
    assert result.receipt.totals.model_fits == 1
    assert result.plan.status(Stage.SCORE_ARTIFACT).recomputed
    assert_full_rerun(request, result, seed=seed)


def logistic_request():
    request = linear_request()
    rng = np.random.default_rng(62)
    data = dict(request.data)
    p = 1 / (1 + np.exp(-(-0.4 + 0.9 * data["t"] + 0.7 * data["z"])))
    data["y"] = (rng.random(len(p)) < p).astype(float)
    return replace(request, data=data, model=GlmModel())


def test_adjusted_glm_contrast_target_reuse_matches_existing_route_and_full_rerun():
    request = logistic_request()
    session = AdjustedSession()
    first = require_success(session.execute(request, execution=Execution(seed=3)))
    independent = require_success(
        ac.analyze(
            request.data,
            graph=request.edges,
            query=ac.AverageEffect(treatment="t", outcome="y"),
            estimator=GlmAdjustment(bootstrap=0),
            refute=False,
            execution=ac.Execution(seed=3),
        )
    )
    assert first.law.ate == pytest.approx(independent.effect, abs=1e-9)
    weights = TargetWeights(np.exp(0.4 * request.data["z"]), ("z",))
    weighted = replace(request, target=weights)
    result = require_success(session.execute(weighted, execution=Execution(seed=3)))
    assert result.receipt.totals.model_fits == 0
    assert result.plan.status(Stage.SCORE_ARTIFACT).reused
    assert result.plan.recomputed_computations == (Stage.LAW, Stage.DECISION)
    assert_full_rerun(weighted, result)
    # A second, independently coded IRLS solves the disclosed response-scale
    # model, then computes the weighted law and full-covariance delta variance.
    x = np.column_stack([np.ones(len(request.data["y"])), request.data["z"], request.data["t"]])
    beta = np.zeros(3)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(x @ beta)))
        information = x.T @ ((p * (1 - p))[:, None] * x)
        step = np.linalg.solve(information, x.T @ (request.data["y"] - p))
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break
    active, control = x.copy(), x.copy()
    active[:, -1], control[:, -1] = 1, 0
    pa, pc = 1 / (1 + np.exp(-(active @ beta))), 1 / (1 + np.exp(-(control @ beta)))
    w = np.asarray(weights.weights) / np.sum(weights.weights)
    gradient = np.sum(
        w[:, None]
        * (pa[:, None] * (1 - pa[:, None]) * active - pc[:, None] * (1 - pc[:, None]) * control),
        axis=0,
    )
    covariance = np.linalg.inv(information)
    assert result.law.ate == pytest.approx(w @ (pa - pc), abs=1e-9)
    assert result.law.std_error == pytest.approx(
        np.sqrt(gradient @ covariance @ gradient), rel=1e-7
    )
    changed_link = replace(weighted, model=GlmModel("binomial_probit"))
    linked = require_success(session.execute(changed_link, execution=Execution(seed=3)))
    assert linked.receipt.totals.model_fits == 1
    assert_full_rerun(changed_link, linked)


def categorical_request():
    rng = np.random.default_rng(79)
    n = 450
    z = rng.normal(size=n)
    index = rng.integers(0, 3, size=n)
    levels = ("control", "low", "high")
    labels = tuple(levels[i] for i in index)
    y = np.choose(index, [0, 0.8, 2.1]) + 0.5 * z + rng.normal(scale=0.2, size=n)
    return AdjustedRequest(
        data={"z": z, "t": index.astype(float), "y": y},
        edges=(("z", "t"), ("z", "y"), ("t", "y")),
        treatments=("t",),
        outcome="y",
        adjustment=("z",),
        utility=Utility(1),
        model=CategoricalModel(labels, levels, "control"),
        contrast=CategoricalContrast("control", "high"),
    )


def test_adjusted_categorical_contrasts_reuse_covariance_and_reference_changes_refit():
    request = categorical_request()
    session = AdjustedSession()
    first = require_success(session.execute(request, execution=Execution(seed=3)))
    contrast = replace(request, contrast=CategoricalContrast("low", "high"))
    result = require_success(session.execute(contrast, execution=Execution(seed=3)))
    assert first.law.ate == pytest.approx(2.1, abs=0.05)
    assert result.law.ate == pytest.approx(1.3, abs=0.05)
    assert result.receipt.totals.model_fits == 0
    assert_full_rerun(contrast, result)
    # Independent dummy-regression covariance includes the off-diagonal when
    # both queried levels differ from the fit's reference.
    index = request.data["t"]
    x = np.column_stack([np.ones(len(index)), request.data["z"], index == 1, index == 2])
    beta = np.linalg.lstsq(x, request.data["y"], rcond=None)[0]
    residual = request.data["y"] - x @ beta
    covariance = np.linalg.inv(x.T @ x) * (residual @ residual) / (len(index) - 4)
    w = np.array([0, 0, -1, 1])
    assert result.law.ate == pytest.approx(w @ beta, abs=1e-10)
    assert result.law.std_error == pytest.approx(np.sqrt(w @ covariance @ w), rel=1e-10)
    reference = replace(contrast, model=replace(request.model, reference="low"))
    changed = require_success(session.execute(reference, execution=Execution(seed=3)))
    assert changed.receipt.totals.model_fits == 1
    assert changed.law.ate == pytest.approx(result.law.ate, abs=1e-10)
    assert changed.law.std_error == pytest.approx(result.law.std_error, abs=1e-10)
    assert_full_rerun(reference, changed)
    recoded = replace(reference, data={**reference.data, "t": np.asarray(reference.data["t"]) + 1})
    recoding = require_success(session.execute(recoded, execution=Execution(seed=3)))
    assert recoding.receipt.totals.model_fits == 1
    assert recoding.law.ate == pytest.approx(changed.law.ate, abs=1e-10)
    assert_full_rerun(recoded, recoding)


@pytest.mark.parametrize(
    "kind",
    [
        "off_grid",
        "unknown_level",
        "sparse_level",
        "weight_dependence",
        "invalid_adjustment",
        "categorical_coding",
    ],
)
def test_adjusted_unsupported_coordinates_refuse_without_discarding_live_fit(kind):
    request = (
        categorical_request()
        if kind in ("unknown_level", "sparse_level", "categorical_coding")
        else linear_request()
    )
    session = AdjustedSession()
    require_success(session.execute(request, execution=Execution(seed=3)))
    identities = session.identities
    if kind == "off_grid":
        changed = replace(request, contrast=NumericContrast((2,), (0,)))
    elif kind == "unknown_level":
        changed = replace(request, contrast=CategoricalContrast("control", "missing"))
    elif kind == "sparse_level":
        changed = replace(request, model=replace(request.model, min_level_rows=1000))
    elif kind == "categorical_coding":
        changed = replace(request, data={**request.data, "t": np.zeros(len(request.data["t"]))})
    elif kind == "weight_dependence":
        changed = replace(request, target=TargetWeights(np.ones(len(request.data["y"])), ("y",)))
    else:
        changed = replace(request, adjustment=())
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(changed, execution=Execution(seed=3)))
    expected = {
        "off_grid": ("route_not_supported", "recalc.off_grid_request", "treatment_grid"),
        "unknown_level": ("route_not_supported", "recalc.off_grid_request", "treatment_grid"),
        "sparse_level": (
            "arm_not_populated",
            "recalc.adjusted_arm_not_populated",
            "score_artifact",
        ),
        "categorical_coding": ("invalid_argument", "recalc.categorical_role_mismatch", "query"),
        "weight_dependence": (
            "invalid_argument",
            "recalc.invalid_target_weights",
            "target_population",
        ),
        "invalid_adjustment": (
            "effect_not_identified",
            "recalc.adjustment_not_identified",
            "identification",
        ),
    }[kind]
    assert (error.value.reason_code, error.value.detail, error.value.stage) == expected
    assert session.identities == identities
    assert session.is_live


def test_adjusted_fresh_process_receipt_does_not_supply_fit(tmp_path):
    request = linear_request()
    session = AdjustedSession()
    first = require_success(session.execute(request, execution=Execution(seed=3)))
    path = tmp_path / "receipt.bin"
    path.write_bytes(bytes(first.receipt.export()))
    data = tmp_path / "data.json"
    data.write_text(json.dumps({k: np.asarray(v).tolist() for k, v in request.data.items()}))
    script = "from antecedent.outcomes import require_success\nfrom antecedent.execution import Execution\nimport json, sys\nfrom antecedent.recalc import RecalcReceipt, RecalcRefusal, ResumeContext, Utility\nfrom antecedent.recalc_adjusted import AdjustedRequest, AdjustedSession\nreceipt = require_success(RecalcReceipt.consume(open(sys.argv[1], 'rb').read()))\nrequest = AdjustedRequest(data=json.load(open(sys.argv[2])), edges=(('z', 't'), ('z', 'y'), ('t', 'y')), treatments=('t',), outcome='y', adjustment=('z',), utility=Utility(1))\nfor context in [None, ResumeContext(portable_fit=True), ResumeContext(portable_scores=True)]:\n    session = require_success(AdjustedSession.resume(receipt, context))\n    try:\n        require_success(session.execute(request, execution=Execution(seed=3)))\n        raise AssertionError('flags supplied a fit')\n    except RecalcRefusal:\n        assert not session.is_live\nsupplied = require_success(AdjustedSession.resume(receipt, ResumeContext(supplied_data=True)))\nresult = require_success(supplied.execute(request, execution=Execution(seed=3)))\nassert result.receipt.totals.model_fits == 1\nassert not result.receipt.claims_derived_reuse\nassert result.receipt.capabilities.boundary == 'fresh_process'\nchecked = require_success(RecalcReceipt.consume(result.receipt.export()))\nassert checked.identity == result.receipt.identity\nprint(json.dumps({'ate': result.law.ate, 'se': result.law.std_error}))\n"
    process = subprocess.run(
        [sys.executable, "-c", script, str(path), str(data)],
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=True,
    )
    actual = json.loads(process.stdout)
    assert actual["ate"] == pytest.approx(first.law.ate, abs=1e-10)
    assert actual["se"] == pytest.approx(first.law.std_error, abs=1e-10)


@pytest.mark.parametrize("kind", ["linear", "glm", "categorical"])
def test_adjusted_prediction_uses_named_native_schema_and_measures_zero_fits(kind):
    request = {
        "linear": linear_request,
        "glm": logistic_request,
        "categorical": categorical_request,
    }[kind]()
    session = AdjustedSession()
    require_success(session.execute(request, execution=Execution(seed=3)))
    schema = session.prediction_columns
    assert schema is not None
    if kind == "categorical":
        # Canonical levels excluding the retained reference; use the native
        # schema rather than assuming caller declaration/dictionary order.
        rows = [[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]]
    else:
        assert schema == ("z", "t")
        rows = [[0.0, 0.0], [0.5, 1.0], [-0.5, 0.0]]
    prediction = require_success(session.predict(rows, columns=schema))
    assert prediction.model_fits == 0
    independent = AdjustedSession()
    require_success(independent.execute(request, execution=Execution(seed=3)))
    assert prediction == require_success(
        independent.predict(rows, columns=independent.prediction_columns)
    )
    if kind == "linear":
        _, _, beta, _ = numpy_joint_oracle(request)
        assert prediction.values == pytest.approx(
            np.column_stack([np.ones(3), rows]) @ beta, abs=1e-10
        )
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.predict(rows, columns=tuple(reversed(schema))))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "recalc.adjusted_prediction_schema_mismatch",
        "score_artifact",
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.predict([[0.0]], columns=schema))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "recalc.adjusted_prediction_schema_mismatch",
        "score_artifact",
    )
    bad = [[0.0, 1.0, 1.0]] if kind == "categorical" else [[0.0, 2.0]]
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.predict(bad, columns=schema))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "route_not_supported",
        "recalc.adjusted_prediction_out_of_support",
        "treatment_grid",
    )
    assert session.is_live
    with pytest.raises(RecalcRefusal) as error:
        require_success(AdjustedSession().predict(rows, columns=schema))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "score_table_unavailable",
        "recalc.no_live_state",
        "score_artifact",
    )


def large_linear_request(rows, columns):
    """`columns` variables in all: treatment, outcome and `columns - 2` adjusters."""
    rng = np.random.default_rng(97)
    adjusters = {f"z{i}": rng.normal(size=rows) for i in range(columns - 2)}
    z_sum = sum(adjusters.values())
    t = (rng.random(rows) < 1 / (1 + np.exp(-0.2 * z_sum))).astype(float)
    y = 2 * t + 0.1 * z_sum + rng.normal(size=rows)
    edges = [(name, "t") for name in adjusters] + [(name, "y") for name in adjusters]
    return AdjustedRequest(
        data={**adjusters, "t": t, "y": y},
        edges=[*edges, ("t", "y")],
        treatments=("t",),
        outcome="y",
        adjustment=tuple(adjusters),
        utility=Utility(1),
        contrast=NumericContrast(),
    )


@pytest.mark.parametrize(("rows", "columns"), [(100_000, 20), (300_000, 5)])
def test_adjusted_accepts_inputs_beyond_the_former_row_and_value_ceilings(rows, columns):
    request = large_linear_request(rows, columns)
    result = require_success(AdjustedSession().execute(request, execution=Execution(seed=3)))
    expected, se, _, _ = numpy_joint_oracle(request)
    assert result.law.ate == pytest.approx(expected, rel=1e-8)
    assert result.law.std_error == pytest.approx(se, rel=1e-8)
    assert result.receipt.totals.model_fits == 1


def test_adjusted_over_budget_input_is_refused_before_it_is_copied():
    from antecedent import _native
    from antecedent._recalc_bounds import _columns
    from antecedent.errors import CausalValueError
    from antecedent.recalc_adjusted import _spec

    request = linear_request()
    values = sum(len(v) for v in request.data.values())

    # A column that only claims its size is refused before numpy could materialise it.
    class Claimed:
        size = 2**50

    with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded") as error:
        _columns({"a": Claimed()}, memory_limit_bytes=2**40)
    assert error.value.reason_code == "invalid_argument"

    session = AdjustedSession()
    for call in (session.plan, session.execute):
        with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded") as error:
            call(request, execution=Execution(limits=ResourceLimits(memory_bytes=values * 8 - 1)))
        assert error.value.reason_code == "invalid_argument"
    assert not session.is_live
    # The limit is exact: the same request runs when the copy fits.
    require_success(
        session.execute(
            request,
            execution=Execution(seed=3, limits=ResourceLimits(memory_bytes=1 << 30)),
        )
    )
    assert session.is_live

    # Native code applies the same budget to its own copy, independently of Python.
    native = _native.AdjustedSessionHandle()
    columns = [np.asarray(v) for v in request.data.values()]
    with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded") as error:
        require_success(
            native.execute(
                list(request.data),
                columns,
                json.dumps(_spec(request)),
                memory_limit_bytes=8,
            )
        )
    assert error.value.reason_code == "invalid_argument"
    assert not native.is_live()
    with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded"):
        require_success(
            native.predict(
                [[0.0, 0.0]] * 10,
                ["z", "t"],
                memory_limit_bytes=8,
            )
        )
    with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded"):
        require_success(
            AdjustedSession().predict(
                [[0.0, 0.0]] * 10,
                columns=["z", "t"],
                execution=Execution(limits=ResourceLimits(memory_bytes=8)),
            )
        )


def test_adjusted_cancelled_run_refuses_before_fitting():
    from antecedent import _native

    request = linear_request()
    token = _native.CancellationToken()
    token.cancel()
    session = AdjustedSession()
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(request, execution=Execution(cancel=token)))
    assert (error.value.reason_code, error.value.detail) == (
        "cancelled_no_claim",
        "recalc.cancelled",
    )
    assert not session.is_live
    # An unfired token changes nothing.
    assert require_success(
        session.execute(
            request,
            execution=Execution(seed=3, cancel=_native.CancellationToken()),
        )
    ).receipt


def test_adjusted_bridge_checks_glm_iterations_before_execution():
    from antecedent import _native
    from antecedent.errors import CausalValueError
    from antecedent.recalc_adjusted import _spec

    request = linear_request()
    native = _native.AdjustedSessionHandle()

    class NonIteratingList(list):
        def __iter__(self):
            raise AssertionError("native extraction invoked a custom iterator")

    require_success(
        native.execute(
            list(request.data),
            [np.asarray(v) for v in request.data.values()],
            json.dumps(_spec(request)),
            seed=3,
        )
    )
    report, refusal = require_success(
        native.predict(NonIteratingList([NonIteratingList([0.0, 0.0])]), ["z", "t"])
    )
    assert refusal is None
    assert json.loads(report)["model_fits"] == 0

    with pytest.raises(CausalValueError, match="recalc.adjusted_invalid_model") as error:
        require_success(
            AdjustedSession().execute(replace(logistic_request(), model=GlmModel(max_iter=0)))
        )
    assert error.value.reason_code == "invalid_argument"


@pytest.mark.parametrize("kind", ["invalid_outcome", "nonconvergence", "separation"])
def test_adjusted_glm_domain_and_solver_refusals_preserve_previous_fit(kind):
    request = logistic_request()
    session = AdjustedSession()
    first = require_success(session.execute(request, execution=Execution(seed=3)))
    identities = session.identities
    if kind == "invalid_outcome":
        changed = replace(request, data={**request.data, "y": np.full(len(request.data["y"]), 2.0)})
        expected = ("invalid_argument", "recalc.adjusted_model_data_invalid", "data_snapshot")
    elif kind == "nonconvergence":
        changed = replace(request, model=GlmModel(max_iter=1))
        expected = ("route_not_supported", "recalc.adjusted_glm_not_converged", "score_artifact")
    else:
        changed = replace(request, data={**request.data, "y": np.asarray(request.data["t"])})
        expected = ("route_not_supported", "recalc.adjusted_glm_separated", "score_artifact")
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(changed, execution=Execution(seed=3)))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == expected
    assert session.identities == identities
    assert session.is_live
    same = require_success(session.execute(request, execution=Execution(seed=3)))
    assert same.receipt.totals.total == 0
    assert same.law == first.law
