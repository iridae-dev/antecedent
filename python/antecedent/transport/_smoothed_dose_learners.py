"""Smoothed dose-response transport with the caller's own learners.

:func:`analyze_smoothed_dose` fits the same grid with the caller's own learners
(:class:`antecedent.external_learners.Learners`: a ``regression`` outcome-curve learner and a
``binary_probability`` membership learner) through the compiled ``SmoothedDoseLearnerTransport``
program. The grid is evaluated from portable fold models, so each provider declares
``artifact_codec=PORTABLE_PREDICTION_MAP`` and returns its fitted model as a portable
prediction map. The learners are external code the library does not verify; the result is
point-only with calibration not_measured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import _native
from ..errors import CausalTypeError, CausalUnsupportedError
from ..execution import Execution, native_controls
from ..external_learners import ExternalLearner, Learners, _learner_identity, _snapshot_learner
from ._learned_continuous import _frozen
from ._smoothed_dose import (
    SmoothedDoseData,
    SmoothedDoseGridPoint,
    SmoothedDoseOptions,
    SmoothedDoseQuery,
)

#: The ``artifact_codec`` of a learner provider whose fitted model artifact is a portable
#: prediction map: UTF-8 JSON ``{"kind": "linear", "coefficients": [...], "logistic": bool}``
#: with one coefficient per design column (the dose basis, intercept included), or a tree
#: ensemble. The caller-learner smoothed dose route evaluates its folds from these maps.
PORTABLE_PREDICTION_MAP = "portable_prediction_map_v1"


@dataclass(frozen=True, slots=True)
class SmoothedDoseLearnersEstimate:
    """The smoothed dose grid fitted with the caller's learners, from its compiled program.

    ``grid``, ``uncertainty``, ``overlap``, ``diagnostics``, ``provenance`` (membership folds,
    then outcome folds, each naming the provider fit it came from) and ``folds`` are the route's;
    ``external_nuisances_unverified`` is always true. ``route`` is the plan route
    (``SmoothedDoseLearnerTransport``), ``program_identity`` the lowered program identity,
    ``claim`` is ``point_only`` and ``calibration`` ``not_measured``. ``receipts`` holds one
    ``(fit id, fit receipt)`` per provider fit.
    """

    grid: tuple[SmoothedDoseGridPoint, ...]
    uncertainty: Mapping[str, Any]
    overlap: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    folds: Mapping[str, Any]
    external_nuisances_unverified: bool
    route: str
    program_identity: str
    claim: str
    calibration: str
    variable_names: tuple[str, ...]
    receipts: tuple[tuple[int, Any], ...] = field(repr=False, compare=False)

    @property
    def interval(self) -> None:
        """Always ``None``: caller learners run no bootstrap."""
        return None

    def as_mean_source(self, *, quantity: Any) -> Any:
        """One point-only mean per grid dose.

        ``quantity`` supplies the outcome, units, population and base regime. Each dose
        appends ``:dose=<value>`` to that regime. Influence standard errors stay off the source.
        """
        from ..results.sources import dose_grid_mean_source

        return dose_grid_mean_source(
            self.program_identity,
            self.grid,
            quantity,
            producer="SmoothedDoseLearnersEstimate",
            operation=self.route,
        )

    def __getitem__(self, dose: float) -> SmoothedDoseGridPoint:
        for point in self.grid:
            if point.dose == dose:
                return point
        raise KeyError(dose)


def _portable_learner(learner: ExternalLearner, role: str) -> ExternalLearner:
    if not isinstance(learner, ExternalLearner):
        raise CausalTypeError(f"the {role} learner must be an ExternalLearner")
    spec = learner.registry._entry(learner.name)[0].spec
    if spec.artifact_codec != PORTABLE_PREDICTION_MAP:
        raise CausalUnsupportedError(
            f"dose_response.learner_not_portable: the {role} learner's provider must declare "
            f"artifact_codec={PORTABLE_PREDICTION_MAP!r} and return its fitted model as a "
            "portable prediction map; the route evaluates its folds from those maps"
        )
    return _snapshot_learner(learner)


def analyze_smoothed_dose(
    query: SmoothedDoseQuery,
    data: SmoothedDoseData,
    *,
    learners: Learners,
    options: SmoothedDoseOptions | None = None,
    execution: Execution | None = None,
) -> SmoothedDoseLearnersEstimate:
    """Fit the smoothed dose-response transport grid with the caller's learners.

    The transport is decided on the query's selection diagram and the request validated before
    any learner is fitted; ``learners.outcome`` (a ``regression`` learner) then fits the outcome
    curve over the dose basis and ``learners.treatment`` (a ``binary_probability`` learner) the
    source membership, cross-fitted on the route's shared folds, in the compiled
    ``SmoothedDoseLearnerTransport`` program. Each provider declares
    ``artifact_codec=PORTABLE_PREDICTION_MAP`` (refused with
    ``dose_response.learner_not_portable`` otherwise). The basis, folds, quadrature and support
    floors come from ``options``; its built-in learner choices are not used, and
    ``options.bootstrap`` must be zero (caller learners run no bootstrap) and the rows must not be
    time-ordered (``route_not_supported``). The learners' names, specifications and
    configurations enter the program identity.
    """
    if not isinstance(query, SmoothedDoseQuery) or not isinstance(data, SmoothedDoseData):
        raise CausalTypeError(
            "analyze_smoothed_dose requires SmoothedDoseQuery and SmoothedDoseData"
        )
    if not isinstance(learners, Learners):
        raise CausalTypeError("learners must be a Learners declaration")
    options = options or SmoothedDoseOptions()
    if not isinstance(options, SmoothedDoseOptions):
        raise CausalTypeError("options must be SmoothedDoseOptions")
    outcome = _portable_learner(learners.outcome, "outcome")
    membership = _portable_learner(learners.treatment, "membership")
    identity = json.dumps(
        {"outcome": _learner_identity(outcome), "membership": _learner_identity(membership)},
        sort_keys=True,
        separators=(",", ":"),
    )
    payload, receipts = _native.smoothed_dose_learner_transport(
        query.graph,
        list(query.diagram.selections),
        query._json(),
        data,
        options._json(),
        outcome,
        membership,
        identity=identity,
        **native_controls(execution, seed="seed", cancel="cancel", memory="memory_limit_bytes"),
    )
    raw = json.loads(payload)
    estimate = raw["estimate"]
    return SmoothedDoseLearnersEstimate(
        grid=tuple(
            SmoothedDoseGridPoint(
                p["dose"],
                p["estimate"],
                p["plug_in"],
                p["augmentation"],
                _frozen(p["quadrature"]),
                _frozen(p["smoothing_bias"]),
                _frozen(p["support"]),
                p["influence_se_diagnostic"],
            )
            for p in estimate["grid"]
        ),
        uncertainty=_frozen(estimate["uncertainty"]),
        overlap=_frozen(estimate["overlap"]),
        diagnostics=_frozen(estimate["diagnostics"]),
        provenance=tuple(_frozen(p) for p in estimate["provenance"]),
        folds=_frozen(estimate["folds"]),
        external_nuisances_unverified=bool(estimate.get("external_nuisances_unverified", False)),
        route=str(raw["route"]),
        program_identity=str(raw["program_identity"]),
        claim=str(raw["claim"]),
        calibration=str(raw["calibration"]),
        variable_names=tuple(raw["variable_names"]),
        receipts=tuple((int(fit_id), receipt) for fit_id, receipt, _ in receipts),
    )


__all__ = ["PORTABLE_PREDICTION_MAP", "SmoothedDoseLearnersEstimate", "analyze_smoothed_dose"]
