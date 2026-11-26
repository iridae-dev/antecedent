"""Trial-to-target AIPW with the caller's own learners.

:func:`analyze_trial_aipw` fits the certified binary contrast with the caller's learners
(:class:`antecedent.external_learners.Learners`: a ``regression`` source-outcome learner and a
``binary_probability`` membership learner) through the compiled ``TrialAipwLearnerTransport``
program. Each provider declares ``artifact_codec=PORTABLE_PREDICTION_MAP`` and returns its
fitted model as a portable prediction map. The learners are external code the library does not
verify; the result is point-only with calibration not_measured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import _native
from ..errors import CausalTypeError
from ..execution import Execution, native_controls
from ..external_learners import (
    Learners,
    _learner_identity,
    _snapshot_learner,
    declare_portable_route,
)
from ..learners import Linear, Logistic, _learner_wire
from ._impl.trial import TrialAipwData, TrialAipwQuery
from ._learned_continuous import _frozen


@dataclass(frozen=True, slots=True)
class TrialAipwLearnersEstimate:
    """The target contrast fitted with the caller's learners, from its compiled program.

    ``estimate`` is the target nonparticipant contrast. ``overlap``, ``diagnostics`` and
    ``provenance`` are the route's. ``external_nuisances_unverified`` is always true. ``route``
    is ``TrialAipwLearnerTransport``, ``program_identity`` the lowered program identity,
    ``claim`` is ``point_only`` and ``calibration`` ``not_measured``. ``receipts`` holds one
    ``(fit id, fit receipt)`` per provider fit.
    """

    estimate: float
    overlap: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]
    external_nuisances_unverified: bool
    route: str
    program_identity: str
    claim: str
    calibration: str
    receipts: tuple[tuple[int, Any], ...] = field(repr=False, compare=False)

    def as_mean_source(self, *, quantity: Any) -> Any:
        """The target contrast as one point-only mean.

        ``quantity`` names the units, population and regime. The program identity is the
        contract id. Nuisance fits stay unverified, and no interval is carried.
        """
        from ..results.sources import point_mean_source

        return point_mean_source(
            self.program_identity,
            quantity,
            self.estimate,
            producer="TrialAipwLearnersEstimate",
            operation=self.route,
        )


def analyze_trial_aipw(
    query: TrialAipwQuery,
    data: TrialAipwData,
    *,
    learners: Learners,
    folds: int = 3,
    execution: Execution | None = None,
) -> TrialAipwLearnersEstimate:
    """Fit the certified binary trial contrast with the caller's learners.

    The transport is decided on the query's selection diagram and the request validated before
    any learner is fitted; ``learners.outcome`` (a ``regression`` learner) then fits the source
    outcome and ``learners.treatment`` (a ``binary_probability`` learner) the source membership,
    cross-fitted on ``folds`` shared folds, in the compiled ``TrialAipwLearnerTransport``
    program. A learner whose predict needs live process state is refused with
    ``dose_response.learner_not_portable``. Caller learners run no bootstrap. The learners'
    names, specifications and configurations enter the program identity.
    """
    if not isinstance(query, TrialAipwQuery) or not isinstance(data, TrialAipwData):
        raise CausalTypeError("analyze_trial_aipw requires TrialAipwQuery and TrialAipwData")
    if not isinstance(learners, Learners):
        raise CausalTypeError("learners must be a Learners declaration")
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise CausalTypeError("folds must be an integer")
    declare_portable_route("trial_aipw", learners)
    outcome = _snapshot_learner(learners.outcome)
    membership = _snapshot_learner(learners.treatment)
    identity = json.dumps(
        {"outcome": _learner_identity(outcome), "membership": _learner_identity(membership)},
        sort_keys=True,
        separators=(",", ":"),
    )
    options = json.dumps(
        {
            "outcome": _learner_wire(Linear()),
            "membership": _learner_wire(Logistic()),
            "folds": folds,
            "bootstrap": 0,
            "coverage_level": 0.95,
        }
    )
    payload, receipts = _native.trial_aipw_learner_transport(
        query.graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        query.treatment,
        query.outcome,
        data,
        options,
        outcome,
        membership,
        identity=identity,
        **native_controls(execution, seed="seed", cancel="cancel", memory="memory_limit_bytes"),
    )
    raw = json.loads(payload)
    estimate = raw["estimate"]
    return TrialAipwLearnersEstimate(
        estimate=float(estimate["estimate"]),
        overlap=_frozen(estimate["overlap"]),
        diagnostics=_frozen(estimate["diagnostics"]),
        provenance=tuple(_frozen(item) for item in estimate["provenance"]),
        external_nuisances_unverified=bool(estimate.get("external_nuisances_unverified", False)),
        route=str(raw["route"]),
        program_identity=str(raw["program_identity"]),
        claim=str(raw["claim"]),
        calibration=str(raw["calibration"]),
        receipts=tuple((int(fit_id), receipt) for fit_id, receipt, _ in receipts),
    )


__all__ = ["TrialAipwLearnersEstimate", "analyze_trial_aipw"]
