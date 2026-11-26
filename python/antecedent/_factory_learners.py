"""Adapt an ordinary Python model factory to a nuisance learner.

:func:`regressor` and :func:`learners.classifier` are re-exported from
:mod:`antecedent.learners`. They return an :class:`~antecedent.external_learners.ExternalLearner`.
They are not native :data:`~antecedent.learners.LearnerSpec` values.

Each fit calls ``factory()`` and pickles that instance into the artifact. Prediction
unpickles it and calls ``predict`` or ``predict_proba``. The artifact codec is
``live_python_estimator_v1``, so the learner is not a portable prediction map:
smoothed dose, trial AIPW and joint cells keep refusing it. The estimator class
must be importable by pickle, which means a class defined at module scope.
"""

from __future__ import annotations

import base64
import inspect
import json
import pickle
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from .errors import CausalTypeError, CausalValueError
from .extensibility import (
    CausalProviderSpec,
    FoldPolicy,
    ProviderCapability,
    ProviderExecution,
    ProviderRegistry,
)
from .external_learners import ExternalLearner

LIVE_PYTHON_ESTIMATOR = "live_python_estimator_v1"

Task = Literal["regression", "binary_probability"]


def regressor(
    factory: Any,
    *,
    name: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    seed: int | None = None,
    weights: bool | None = None,
    deterministic: bool = True,
) -> ExternalLearner:
    """A regression learner whose ``fit`` and ``predict`` come from ``factory()``.

    ``factory`` is called with no arguments once while the learner is declared,
    and again for every fit. A fold therefore gets a new model. ``configuration``
    is stored on the learner and enters the program identity. ``seed`` is applied
    as ``random_state`` before each fit, through ``set_params`` when the model
    declares that parameter, otherwise as an attribute. ``weights`` records
    whether ``fit`` accepts ``sample_weight``; the default reads the signature.
    """
    return _adapt(
        factory,
        "regression",
        name=name,
        configuration=configuration,
        seed=seed,
        weights=weights,
        deterministic=deterministic,
    )


def classifier(
    factory: Any,
    *,
    name: str | None = None,
    configuration: Mapping[str, Any] | None = None,
    seed: int | None = None,
    weights: bool | None = None,
    deterministic: bool = True,
) -> ExternalLearner:
    """A propensity learner whose probability is column ``classes_ == 1``.

    The returned model must implement ``fit`` and ``predict_proba``. A vector,
    or a single column, is that probability. Several columns require ``classes_``
    to contain class ``1`` once. The other arguments match :func:`regressor`.
    """
    return _adapt(
        factory,
        "binary_probability",
        name=name,
        configuration=configuration,
        seed=seed,
        weights=weights,
        deterministic=deterministic,
    )


class _FactoryProvider:
    def __init__(
        self,
        factory: Any,
        task: Task,
        *,
        seed: int | None,
        supports_weights: bool,
        deterministic: bool,
        provenance: Mapping[str, str],
    ) -> None:
        self._factory = factory
        self._task = task
        self._seed = seed
        self._supports_weights = supports_weights
        dependence = ["training_rows"]
        if supports_weights:
            dependence.append("sample_weights")
        self.spec = CausalProviderSpec(
            query_family="nuisance_learner",
            identification_requirements=("caller_identifies",),
            observed_distributions=("X,Y",),
            nuisance_functions=("mean",),
            support_conditions=("finite",),
            data_dependence=tuple(dependence),
            inference_claims=("point_only",),
            influence_function=None,
            fold_policy=FoldPolicy.CALLER_OWNED,
            output_shape="one per prediction row",
            uncertainty_semantics="point_only",
            artifact_codec=LIVE_PYTHON_ESTIMATOR,
            deterministic=deterministic,
            provenance=provenance,
            capabilities=(ProviderCapability.MEAN,),
        )

    def execute(self, request: Mapping[str, Any]) -> ProviderExecution:
        operation = request["operation"]
        if operation == "fit":
            model = self._factory()
            _apply_seed(model, self._seed)
            x = np.asarray(request["x"], dtype=np.float64)
            y = np.asarray(request["y"], dtype=np.float64)
            if request["weights"] is None:
                model.fit(x, y)
            else:
                model.fit(x, y, sample_weight=np.asarray(request["weights"], dtype=np.float64))
            artifact = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            return ProviderExecution([], None, (), "supported", {"version": "1"}, artifact)
        if operation == "predict":
            payload = base64.b64decode(request["model_base64"])
            model = pickle.loads(payload)  # noqa: S301 -- artifact this provider just pickled
            x = np.asarray(request["x"], dtype=np.float64)
            values = _predict(model, x, self._task)
            return ProviderExecution(values, None, (), "supported", {"version": "1"})
        raise CausalValueError(f"unsupported learner operation {operation!r}")


def _adapt(
    factory: Any,
    task: Task,
    *,
    name: str | None,
    configuration: Mapping[str, Any] | None,
    seed: int | None,
    weights: bool | None,
    deterministic: bool,
) -> ExternalLearner:
    if type(deterministic) is not bool:
        raise CausalTypeError("deterministic must be true or false")
    _require_seed(seed)
    declared = _configuration(configuration)
    model = _probe(factory, task)
    supports_weights = _weight_support(model, weights)
    _apply_seed(model, seed)
    provider_name = _provider_name(factory, name, task)
    provenance = {
        "package": getattr(factory, "__module__", None) or "python",
        "callable": getattr(factory, "__qualname__", None) or provider_name,
        "version": "1",
    }
    registry = ProviderRegistry()
    registry.register(
        provider_name,
        _FactoryProvider(
            factory,
            task,
            seed=seed,
            supports_weights=supports_weights,
            deterministic=deterministic,
            provenance=provenance,
        ),
    )
    return ExternalLearner(
        registry,
        provider_name,
        task,
        supports_weights=supports_weights,
        configuration=declared,
    )


def _probe(factory: Any, task: Task) -> Any:
    if not callable(factory):
        raise CausalTypeError("factory must be callable with no arguments")
    try:
        model = factory()
    except TypeError as error:
        raise CausalTypeError("factory must be callable with no arguments") from error
    if not callable(getattr(model, "fit", None)):
        raise CausalTypeError("the model must implement fit")
    if task == "regression" and not callable(getattr(model, "predict", None)):
        raise CausalTypeError("a regressor must implement predict")
    if task == "binary_probability" and not callable(getattr(model, "predict_proba", None)):
        raise CausalTypeError("a classifier must implement predict_proba")
    return model


def _weight_support(model: Any, weights: bool | None) -> bool:
    accepted = _accepts_sample_weight(model.fit)
    if weights is None:
        return accepted
    if type(weights) is not bool:
        raise CausalTypeError("weights must be true or false")
    if weights and not accepted:
        raise CausalValueError("weights were requested but fit does not accept sample_weight")
    return weights


def _accepts_sample_weight(fit: Any) -> bool:
    try:
        signature = inspect.signature(fit)
    except (TypeError, ValueError):
        return False
    return any(
        param.name == "sample_weight"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for param in signature.parameters.values()
    )


def _require_seed(seed: int | None) -> None:
    if seed is None:
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CausalTypeError("seed must be an integer")


def _has_random_state(model: Any) -> bool:
    if "random_state" in getattr(model, "__dict__", {}):
        return True
    for cls in type(model).__mro__:
        if "random_state" in getattr(cls, "__dict__", {}):
            return True
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if slots and "random_state" in slots:
            return True
    return False


def _apply_seed(model: Any, seed: int | None) -> None:
    if seed is None:
        return
    setter = getattr(model, "set_params", None)
    getter = getattr(model, "get_params", None)
    if callable(setter) and callable(getter):
        try:
            params = getter()
        except Exception:
            params = None
        if isinstance(params, Mapping) and "random_state" in params:
            setter(random_state=seed)
            return
    if not _has_random_state(model):
        raise CausalValueError("seed was given but the model has no random_state")
    model.random_state = seed


def _configuration(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CausalTypeError("configuration must be a mapping")
    declared = dict(value)
    try:
        json.dumps(declared, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CausalTypeError("configuration must be JSON") from error
    return declared


def _provider_name(factory: Any, name: str | None, task: Task) -> str:
    if name is None:
        name = getattr(factory, "__qualname__", None) or task
    if not isinstance(name, str) or not name.strip():
        raise CausalTypeError("name must be a nonempty string")
    return name


def _predict(model: Any, x: np.ndarray, task: Task) -> np.ndarray:
    if task == "regression":
        values = np.asarray(model.predict(x), dtype=np.float64)
    else:
        values = _positive_probability(model, x)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise CausalValueError(f"learner prediction must be {len(x)} finite values")
    return values


def _positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    if proba.ndim == 1:
        return proba
    if proba.ndim != 2 or proba.shape[1] == 0:
        raise CausalValueError("predict_proba must return one probability per row")
    if proba.shape[1] == 1:
        return proba[:, 0]
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise CausalValueError("predict_proba with several columns must set classes_")
    labels = np.asarray(classes).reshape(-1)
    if labels.shape[0] != proba.shape[1]:
        raise CausalValueError("classes_ length must match predict_proba columns")
    matches = np.flatnonzero(labels == 1)
    if matches.size != 1:
        raise CausalValueError("classifier classes_ must contain class 1 once")
    return proba[:, int(matches[0])]
