"""Caller learners on the compiled smoothed dose-response transport.

``analyze_smoothed_dose`` declares two registered providers (``Learners``) as the outcome-curve
and source-membership learners and runs the ``SmoothedDoseLearnerTransport`` program. The route
evaluates the grid from portable fold models, so each provider returns its fitted model as a
portable prediction map (``artifact_codec=PORTABLE_PREDICTION_MAP``).

Oracle. The fixture of ``test_smoothed_dose_transport`` draws ``y = a^2 + z (1 + a/2) + e`` with a
known dose density, so ``psi_h(a) = a^2 + h^2/5 + m_T (1 + a/2)``. A caller least-squares
outcome learner on the dose-by-covariate basis is correctly specified, so the doubly robust point
is within the direct route's tolerance of that truth whatever the membership model; the same
least-squares fit is the library's linear learner, so the plug-in equals the library route's.
The bit-for-bit twin of the compiled route and the factory kernel is in
``crates/antecedent/tests/compiled_twins_dose_feedback.rs``.
"""

from __future__ import annotations

import base64
import json
import re

import antecedent as ac
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.extensibility import (
    CausalProviderSpec,
    ProviderCapability,
    ProviderExecution,
    ProviderRegistry,
)
from antecedent.external_learners import ExternalLearner, Learners
from antecedent.outcomes import require_success
from antecedent.transport import advanced
from antecedent.transport._smoothed_dose_learners import (
    PORTABLE_PREDICTION_MAP,
    SmoothedDoseLearnersEstimate,
    analyze_smoothed_dose,
)

from test_smoothed_dose_transport import GRID, fixture, truth


def _spec(codec: str) -> CausalProviderSpec:
    return CausalProviderSpec(
        query_family="nuisance_learner",
        identification_requirements=("caller_identifies",),
        observed_distributions=("X,Y",),
        nuisance_functions=("mean",),
        support_conditions=("finite",),
        data_dependence=("training_rows", "sample_weights"),
        inference_claims=("point_only",),
        influence_function=None,
        fold_policy="caller_owned",
        output_shape="one per prediction row",
        uncertainty_semantics="point_only",
        artifact_codec=codec,
        deterministic=True,
        provenance={"package": "test", "version": "1"},
        capabilities=(ProviderCapability.MEAN,),
    )


def _sigmoid(eta):
    return 1.0 / (1.0 + np.exp(-eta))


class LinearMapProvider:
    """Least squares (or Newton-fitted logistic regression) returned as a portable map."""

    def __init__(self, logistic: bool = False, codec: str = PORTABLE_PREDICTION_MAP):
        self.logistic = logistic
        self.spec = _spec(codec)

    def _fit(self, x, y):
        if not self.logistic:
            return np.linalg.lstsq(x, y, rcond=None)[0]
        beta = np.zeros(x.shape[1])
        for _ in range(50):
            p = _sigmoid(x @ beta)
            hessian = x.T @ (x * (p * (1.0 - p))[:, None]) + 1e-10 * np.eye(x.shape[1])
            step = np.linalg.solve(hessian, x.T @ (y - p))
            beta = beta + step
            if np.max(np.abs(step)) < 1e-12:
                break
        return beta

    def execute(self, request):
        if request["operation"] == "fit":
            x = np.asarray(request["x"], dtype=float)
            y = np.asarray(request["y"], dtype=float)
            beta = self._fit(x, y)
            model = {
                "kind": "linear",
                "coefficients": [float(b) for b in beta],
                "logistic": self.logistic,
            }
            return ProviderExecution(
                [], None, (), "supported", {"version": "1"}, json.dumps(model).encode()
            )
        coefficients = json.loads(base64.b64decode(request["model_base64"]))["coefficients"]
        eta = np.asarray(request["x"], dtype=float) @ np.asarray(coefficients)
        return ProviderExecution(
            _sigmoid(eta) if self.logistic else eta, None, (), "supported", {"version": "1"}
        )


def _learners(*, codec: str = PORTABLE_PREDICTION_MAP, configuration=None) -> Learners:
    registry = ProviderRegistry()
    registry.register("outcome", LinearMapProvider(codec=codec))
    registry.register("membership", LinearMapProvider(logistic=True, codec=codec))
    return Learners(
        ExternalLearner(registry, "outcome", "regression", configuration=configuration),
        ExternalLearner(registry, "membership", "binary_probability"),
    )


def _options(**kwargs):
    return advanced.SmoothedDoseOptions(folds=3, **kwargs)


def _run(learners=None, **kwargs) -> SmoothedDoseLearnersEstimate:
    query, data = fixture()
    return analyze_smoothed_dose(
        query,
        data,
        learners=learners or _learners(),
        options=kwargs.pop("options", _options()),
        execution=ac.Execution(seed=5),
    )


@pytest.fixture(scope="module")
def result() -> SmoothedDoseLearnersEstimate:
    return _run()


def test_caller_learners_fit_the_known_truth_through_the_compiled_program(result):
    assert [p.dose for p in result.grid] == list(GRID)
    for point in result.grid:
        assert point.estimate == pytest.approx(truth(point.dose), abs=0.1)
        assert point.estimate == pytest.approx(point.plug_in + point.augmentation, abs=1e-12)
        assert point.quadrature["estimate_error"] < 1e-9
    assert result.route == "SmoothedDoseLearnerTransport"
    assert re.fullmatch(r"[0-9a-f]{64}", result.program_identity)
    assert result.claim == "point_only"
    assert result.calibration == "not_measured"
    assert result.interval is None
    assert result.uncertainty["status"] == "point_only"
    assert result.external_nuisances_unverified is True
    assert result.folds["count"] == 3
    # Three membership and three outcome folds, each fitted by a provider whose receipt is kept.
    assert len(result.provenance) == 6
    assert all(str(p["implementation"]).startswith("python_receipt:") for p in result.provenance)
    assert len(result.receipts) == 6
    assert result.variable_names == ("z", "a", "y")


def test_a_caller_least_squares_curve_is_the_library_linear_curve(result):
    query, data = fixture()
    library = require_success(
        advanced.prepare_smoothed_dose(
            query, data, options=_options(), execution=ac.Execution(seed=5)
        ).estimate()
    )
    assert [p.dose for p in library.grid] == [p.dose for p in result.grid]
    for mine, theirs in zip(result.grid, library.grid, strict=True):
        assert mine.plug_in == pytest.approx(theirs.plug_in, abs=1e-6)
        assert mine.smoothing_bias["local_quadratic_bias"] == pytest.approx(
            theirs.smoothing_bias["local_quadratic_bias"], abs=1e-6
        )


def test_the_program_identity_binds_the_learners_and_repeats_for_the_same_request(result):
    again = _run()
    assert again.program_identity == result.program_identity
    assert [p.estimate for p in again.grid] == [p.estimate for p in result.grid]
    other = _run(learners=_learners(configuration={"variant": 2}))
    assert other.program_identity != result.program_identity


def test_a_provider_without_a_portable_map_is_refused():
    with pytest.raises(CausalUnsupportedError, match="learner_not_portable"):
        _run(learners=_learners(codec="ascii"))


def test_bootstrap_replicates_are_refused_for_caller_learners():
    with pytest.raises(CausalError):
        _run(options=_options(bootstrap=10))


def test_learners_must_be_a_declaration():
    query, data = fixture()
    with pytest.raises(TypeError, match="Learners declaration"):
        analyze_smoothed_dose(query, data, learners=object())  # type: ignore[arg-type]
