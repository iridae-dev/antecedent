"""Factorized joint cells with the caller's own learners.

:func:`analyze_joint_cells` fits each cell with the caller's learners
(:class:`antecedent.external_learners.Learners`: a ``regression`` outcome learner and a
``binary_probability`` propensity learner) through the compiled ``JointCellsLearnerFit``
program. Each provider declares ``artifact_codec=PORTABLE_PREDICTION_MAP`` and returns its
fitted model as a portable prediction map. The learners are external code the library does not
verify; the result is point-only with calibration not_measured. The adjustment set is the
caller's declaration: the route takes no graph and does not certify it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import _native
from .._data import ingest_columns
from ..derived import (
    CellRefusal,
    ContrastValue,
    DerivedTreatment,
    DerivedTreatmentPlan,
    JointCell,
    NormalizationCheck,
    OrderingSensitivity,
    OrderingSpread,
    _plan,
)
from ..errors import CausalTypeError, CausalValueError
from ..execution import Execution, native_controls, resolve_execution
from ..external_learners import (
    Learners,
    _learner_identity,
    _snapshot_learner,
    declare_portable_route,
)


def _cell(raw: Mapping[str, Any]) -> JointCell:
    supported = raw["status"] == "supported"
    return JointCell(
        cell=raw["cell"],
        levels=tuple(raw["levels"]),
        rows=raw["rows"],
        status=raw["status"],
        estimate=raw["estimate"] if supported else None,
        ess=raw["ess"] if supported else None,
        propensity_min=raw["propensity_min"] if supported else None,
        propensity_max=raw["propensity_max"] if supported else None,
        clipped_share=raw["clipped_share"] if supported else None,
        refusal=None
        if supported
        else CellRefusal(code=raw["code"], detail=raw["detail"], message=raw["message"]),
    )


@dataclass(frozen=True, slots=True)
class JointCellsLearnersEstimate:
    """The joint cells fitted with the caller's learners, from its compiled program.

    ``cells`` are the family under the declared ordering. ``contrasts`` are the family
    contrasts requested at the fit (empty when none were asked).     ``normalization``, ``sensitivity``, ``degenerate_conditionals`` and ``plan`` are the
    same records as :class:`~antecedent.derived.JointCellsResult`.
    ``external_nuisances_unverified`` is always true. ``route`` is ``JointCellsLearnerFit``,
    ``program_identity`` the lowered program identity, ``claim`` is ``point_only`` and
    ``calibration`` ``not_measured``. ``receipts`` holds one ``(fit id, fit receipt)`` per
    provider fit.
    """

    treatments: tuple[str, ...]
    n_rows: int
    folds: int
    cells: tuple[JointCell, ...]
    external_nuisances_unverified: bool
    route: str
    program_identity: str
    claim: str
    calibration: str
    provenance: str
    contrasts: tuple[ContrastValue, ...]
    normalization: tuple[NormalizationCheck, ...]
    sensitivity: OrderingSensitivity
    degenerate_conditionals: int
    plan: DerivedTreatmentPlan
    receipts: tuple[tuple[int, Any], ...] = field(repr=False, compare=False)

    @property
    def supported(self) -> tuple[JointCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "supported")

    def cell(self, *levels: int) -> JointCell:
        """The cell with the given component levels, in treatment order."""
        for cell in self.cells:
            if cell.levels == tuple(levels):
                return cell
        raise CausalValueError(f"no cell with levels {levels!r}")


def analyze_joint_cells(
    data: Mapping[str, Any] | Any,
    declaration: DerivedTreatment,
    *,
    outcome: str,
    adjustment: Sequence[str],
    learners: Learners,
    ordering: Sequence[str] | None = None,
    all_orderings: bool = True,
    folds: int = 5,
    clip: float = 0.01,
    min_cell_ess: float = 10.0,
    normalization_tolerance: float = 1e-9,
    ordering_tolerance_sd: float = 0.05,
    contrasts: Sequence[str] = (),
    execution: Execution | None = None,
) -> JointCellsLearnersEstimate:
    """Fit the joint cells of a declared joint-cell treatment with the caller's learners.

    The construction is checked before any learner is fitted. ``learners.outcome`` fits each
    cell's outcome regression and ``learners.treatment`` each binary conditional of the
    factorized propensity, in the compiled ``JointCellsLearnerFit`` program. A learner whose
    predict needs live process state is refused with ``dose_response.learner_not_portable``.
    ``ordering`` lists the construction columns in the declared factorization order (default:
    the declared source order); ``all_orderings`` re-estimates the family under every
    permutation. ``contrasts`` may hold ``"interaction"`` (two components, every cell
    supported) and ``"cell_minus_control:<cell>"``; a contrast that needs an unsupported cell
    comes back with its refusal code rather than a value. The learners' names, specifications
    and configurations enter the program identity. A contrast does not.
    """
    if not isinstance(declaration, DerivedTreatment):
        raise CausalTypeError("declaration must be a DerivedTreatment")
    if not isinstance(learners, Learners):
        raise CausalTypeError("learners must be a Learners declaration")
    if (
        isinstance(adjustment, str)
        or isinstance(contrasts, str)
        or (ordering is not None and isinstance(ordering, str))
    ):
        raise CausalTypeError(
            "adjustment, contrasts and ordering must be sequences of names, not a string"
        )
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise CausalTypeError("folds must be an integer")
    declare_portable_route("joint_cells", learners)
    outcome_learner = _snapshot_learner(learners.outcome)
    propensity = _snapshot_learner(learners.treatment)
    identity = json.dumps(
        {
            "outcome": _learner_identity(outcome_learner),
            "propensity": _learner_identity(propensity),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    names, columns = ingest_columns(data)
    policy = resolve_execution(execution)
    payload, receipts = policy._call(
        _native.joint_cells_learner_fit,
        names,
        columns,
        declaration.to_json(),
        outcome,
        list(adjustment),
        outcome_learner,
        propensity,
        identity=identity,
        ordering=None if ordering is None else list(ordering),
        all_orderings=all_orderings,
        folds=folds,
        clip=clip,
        min_cell_ess=min_cell_ess,
        normalization_tolerance=normalization_tolerance,
        ordering_tolerance_sd=ordering_tolerance_sd,
        contrasts=list(contrasts),
        **native_controls(policy, seed="seed", cancel="cancel", memory="memory_limit_bytes"),
    )
    raw = json.loads(payload)
    fit = raw["fit"]
    treatments = tuple(fit["treatments"])

    def named(ordering: Sequence[int]) -> tuple[str, ...]:
        return tuple(treatments[index] for index in ordering)

    sensitivity = fit["sensitivity"]
    return JointCellsLearnersEstimate(
        treatments=treatments,
        n_rows=int(fit["n_rows"]),
        folds=int(fit["folds"]),
        cells=tuple(_cell(cell) for cell in fit["cells"]),
        external_nuisances_unverified=bool(fit["external_nuisances_unverified"]),
        route=str(raw["route"]),
        program_identity=str(raw["program_identity"]),
        claim=str(raw["claim"]),
        calibration=str(raw["calibration"]),
        provenance=str(fit["provenance"]),
        contrasts=tuple(
            ContrastValue(
                name=item["name"],
                value=item["value"],
                refusal_code=item.get("code"),
                message=item.get("message"),
            )
            for item in fit["contrasts"]
        ),
        normalization=tuple(
            NormalizationCheck(
                ordering=named(item["ordering"]),
                cells_enumerated=item["cells_enumerated"],
                max_abs_error=item["max_abs_error"],
                max_row_sum=item["max_row_sum"],
            )
            for item in fit["normalization"]
        ),
        sensitivity=OrderingSensitivity(
            orderings=tuple(named(ordering) for ordering in sensitivity["orderings"]),
            tolerance=sensitivity["tolerance"],
            cells=tuple(
                OrderingSpread(
                    cell=item["cell"],
                    estimates=tuple(item["estimates"]),
                    spread=item["spread"],
                    flagged=item["flagged"],
                )
                for item in sensitivity["cells"]
            ),
            disagreement=sensitivity["disagreement"],
        ),
        degenerate_conditionals=int(fit["degenerate_conditionals"]),
        plan=_plan(fit["plan"]),
        receipts=tuple((int(fit_id), receipt) for fit_id, receipt, _ in receipts),
    )


__all__ = ["JointCellsLearnersEstimate", "analyze_joint_cells"]
