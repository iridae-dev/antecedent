"""Typed objectives for the shared design ranker.

These declarations expose the existing native scoring models. Scores are point values or
Monte Carlo estimates, never calibrated confidence intervals or identification probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..errors import CausalTypeError, CausalValueError
from ._declarations import BinomialSignal, GaussianMeanSignal, StatePrior
from .plans import _count

_RealScalar = float | int | np.integer[Any] | np.floating[Any]
_NumericVector = Sequence[_RealScalar] | NDArray[np.integer[Any] | np.floating[Any]]


def _sequence(values: object, name: str, *, dimensions: int = 1) -> tuple[Any, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != dimensions:
            raise CausalValueError(f"{name} must be {dimensions}-dimensional")
    elif isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise CausalTypeError(f"{name} must be a sequence")
    return tuple(values)


def _numbers(values: object, name: str) -> tuple[float, ...]:
    entries = _sequence(values, name)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in entries):
        raise CausalTypeError(f"{name} must contain numbers")
    try:
        result = tuple(float(v) for v in entries)
    except OverflowError as error:
        raise CausalValueError(f"{name} must contain finite numbers") from error
    if not all(math.isfinite(v) for v in result):
        raise CausalValueError(f"{name} must contain finite numbers")
    return result


def _variance(value: object, name: str, *, positive: bool = False) -> float:
    result = _numbers((value,), name)[0]
    if result < 0 or (positive and result == 0):
        raise CausalValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


def _gram(values: _NumericVector, name: str) -> tuple[float, ...]:
    result = _numbers(values, name)
    side = math.isqrt(len(result))
    if not side or side * side != len(result):
        raise CausalValueError(f"{name} must be a non-empty flattened square matrix")
    return result


@dataclass(frozen=True, slots=True)
class GraphEntropy:
    """Heuristic graph-channel entropy reduction under ``prior=StructurePrior(...)``.

    This is the native soft-observation heuristic, not likelihood-based information gain.
    The prior's ``features`` optionally defines graph categories.
    """

    def _options(self) -> dict[str, Any]:
        return {"objective": "reduce_graph_entropy"}


@dataclass(frozen=True, slots=True)
class MeasurementColumn:
    """Gram cross-products for one additional measured covariate."""

    variable: int
    cross: _NumericVector
    self_dot: _RealScalar
    sigma2_after: _RealScalar | None = None

    def __post_init__(self) -> None:
        _count(self.variable, "variable")
        object.__setattr__(self, "cross", _numbers(self.cross, "cross"))
        object.__setattr__(self, "self_dot", _variance(self.self_dot, "self_dot"))
        if self.sigma2_after is not None:
            object.__setattr__(self, "sigma2_after", _variance(self.sigma2_after, "sigma2_after"))

    def _wire(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "cross": list(self.cross),
            "self_dot": self.self_dot,
            **({"sigma2_after": self.sigma2_after} if self.sigma2_after is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class DesignInformation:
    """A prospective design's flattened Gram matrix, noise variance and sample count."""

    xtx: _NumericVector
    sigma2: _RealScalar
    n: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "xtx", _gram(self.xtx, "xtx"))
        object.__setattr__(self, "sigma2", _variance(self.sigma2, "sigma2"))
        _count(self.n, "n", minimum=1)

    def _wire(self) -> dict[str, Any]:
        return {"xtx": list(self.xtx), "sigma2": self.sigma2, "n": self.n}


@dataclass(frozen=True, slots=True)
class EnvironmentInformation:
    """Additional Gram information from a named environment."""

    environment: int
    xtx: _NumericVector
    n: int
    sigma2: _RealScalar | None = None

    def __post_init__(self) -> None:
        _count(self.environment, "environment")
        _count(self.n, "n", minimum=1)
        object.__setattr__(self, "xtx", _gram(self.xtx, "xtx"))
        if self.sigma2 is not None:
            object.__setattr__(self, "sigma2", _variance(self.sigma2, "sigma2"))

    def _wire(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "xtx": list(self.xtx),
            "n": self.n,
            **({"sigma2": self.sigma2} if self.sigma2 is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class EffectWidth:
    """Signed treatment standard-error reduction from declared linear-model information.

    ``xtx`` is the current flattened row-major Gram matrix. Sampling and an environment
    without supplied Gram information use isotropic sample-size scaling. Measurement and
    intervention need their prospective information to model a change. The native score
    is ``se_before - se_after``, not a calibrated posterior or confidence interval width.
    """

    xtx: _NumericVector
    sigma2: _RealScalar
    treatment_col: int
    n: int
    measurement_columns: Sequence[MeasurementColumn] = ()
    intervention: DesignInformation | None = None
    environments: Sequence[EnvironmentInformation] = ()
    query_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "xtx", _gram(self.xtx, "xtx"))
        object.__setattr__(self, "sigma2", _variance(self.sigma2, "sigma2"))
        _count(self.n, "n", minimum=1)
        _count(self.query_id, "query_id")
        _count(self.treatment_col, "treatment_col")
        if self.treatment_col >= math.isqrt(len(self.xtx)):
            raise CausalValueError("treatment_col is outside the Gram matrix")
        columns = _sequence(self.measurement_columns, "measurement_columns")
        environments = _sequence(self.environments, "environments")
        if not all(isinstance(c, MeasurementColumn) for c in columns):
            raise CausalTypeError("measurement_columns must contain MeasurementColumn declarations")
        if not all(isinstance(e, EnvironmentInformation) for e in environments):
            raise CausalTypeError("environments must contain EnvironmentInformation declarations")
        if self.intervention is not None and not isinstance(self.intervention, DesignInformation):
            raise CausalTypeError("intervention must be DesignInformation")
        if len({c.variable for c in columns}) != len(columns):
            raise CausalValueError("measurement_columns has duplicate variable ids")
        if len({e.environment for e in environments}) != len(environments):
            raise CausalValueError("environments has duplicate environment ids")
        object.__setattr__(self, "measurement_columns", columns)
        object.__setattr__(self, "environments", environments)

    def _options(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "xtx": list(self.xtx),
            "sigma2": self.sigma2,
            "treatment_col": self.treatment_col,
            "n": self.n,
        }
        if self.measurement_columns:
            context["measure_columns"] = [c._wire() for c in self.measurement_columns]
        if self.environments:
            context["environment_grams"] = [e._wire() for e in self.environments]
        if self.intervention is not None:
            context["intervention_design"] = self.intervention._wire()
        return {
            "objective": "reduce_effect_posterior_width",
            "query_id": self.query_id,
            "effect_width": context,
        }


@dataclass(frozen=True, slots=True)
class ModelDistinction:
    """Native pairwise log-likelihood contrast heuristic over aligned model draws.

    Each row of ``log_likelihoods`` is the same set of draws evaluated by one model.
    This heuristic is not a posterior model probability or Bayes factor.
    """

    model_ids: Sequence[int] | NDArray[np.integer[Any]]
    log_likelihoods: Sequence[_NumericVector] | NDArray[np.integer[Any] | np.floating[Any]]

    def __post_init__(self) -> None:
        ids = tuple(
            _count(int(i) if isinstance(i, np.integer) else i, "model id")
            for i in _sequence(self.model_ids, "model_ids")
        )
        rows = tuple(
            _numbers(row, "log_likelihoods row")
            for row in _sequence(self.log_likelihoods, "log_likelihoods", dimensions=2)
        )
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise CausalValueError("model_ids needs at least two distinct model ids")
        if len(rows) != len(ids) or not rows or not rows[0]:
            raise CausalValueError("log_likelihoods needs one non-empty row per model")
        if any(len(row) != len(rows[0]) for row in rows):
            raise CausalValueError("log_likelihoods rows must have the same draw count")
        object.__setattr__(self, "model_ids", ids)
        object.__setattr__(self, "log_likelihoods", rows)

    def _options(self) -> dict[str, Any]:
        return {
            "objective": "distinguish_models",
            "model_ids": list(self.model_ids),
            "model_loglik": {
                "models": list(self.model_ids),
                "loglik": [v for row in self.log_likelihoods for v in row],
                "n_draws": len(self.log_likelihoods[0]),
            },
        }


@dataclass(frozen=True, slots=True)
class DecisionRegret:
    """Preposterior EVSI for a Python utility over numeric actions and one scalar state.

    ``utility(actions, outcomes)`` returns a flat row-major action-by-outcome table of
    utilities as a contiguous one-dimensional NumPy ``float64`` array. It must be
    deterministic for the declared inputs. Native callback refusals retain the original
    exception as their cause. This native plan objective has no portable utility artifact;
    use ``decision=DesignDecision(...)`` for replayable affine-utility study rankings.
    Costs remain separate and are not subtracted from this score.
    """

    actions: _NumericVector
    utility: Callable[[Any, Any], Any]
    prior: StatePrior
    signal: GaussianMeanSignal | BinomialSignal

    def __post_init__(self) -> None:
        actions = _numbers(self.actions, "actions")
        if not actions or len(set(actions)) != len(actions):
            raise CausalValueError("actions must be non-empty and distinct")
        if not callable(self.utility):
            raise CausalTypeError("utility must be callable")
        if not isinstance(self.prior, StatePrior):
            raise CausalTypeError("prior must be StatePrior")
        if not isinstance(self.signal, (GaussianMeanSignal, BinomialSignal)):
            raise CausalTypeError("signal must be GaussianMeanSignal or BinomialSignal")
        if self.prior.kind == "draws":
            if not _numbers(self.prior.states, "prior draws"):
                raise CausalValueError("prior draws must be non-empty")
        elif self.prior.kind == "normal":
            if self.prior.mean is None or self.prior.variance is None:
                raise CausalValueError("normal prior needs mean and variance")
            _numbers((self.prior.mean,), "prior mean")
            _variance(self.prior.variance, "prior variance", positive=True)
        else:
            raise CausalValueError("prior kind must be draws or normal")
        if isinstance(self.signal, GaussianMeanSignal):
            _variance(self.signal.noise_variance, "noise_variance", positive=True)
        object.__setattr__(self, "actions", actions)

    def _options(self) -> dict[str, Any]:
        prior = self.prior._wire()
        if "states" in prior:
            prior["draws"] = prior.pop("states")
        return {
            "objective": "reduce_decision_regret",
            "decision_id": 0,
            "decision": {
                "actions": list(self.actions),
                "utility": self.utility,
                "prior": prior,
                "signal": self.signal._wire(),
            },
        }


DesignObjective = GraphEntropy | EffectWidth | ModelDistinction | DecisionRegret
OBJECTIVE_TYPES = (GraphEntropy, EffectWidth, ModelDistinction, DecisionRegret)
