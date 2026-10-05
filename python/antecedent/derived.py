"""Derived treatments and factorized joint cells (point only).

A *derived treatment* is built from source columns (a joint cell of up to three binary
components, a product, a sum). :class:`DerivedTreatment` declares the construction: the source
columns with a role and a measurement time, the transformation, the legal derived values, what
an intervention on it means, and any exclusions from the adjustment set.

Each source is **treatment construction** (a component of the treatment), an **admissible
pre-treatment covariate** (may stay in the adjustment set) or a **forbidden descendant**
(measured after the treatment). :func:`check_derived_treatment` refuses, never repairs: a
constituent or descendant left in the adjustment set, an undeclared exact copy of a
constituent, duplicated constituents, a covariate that tracks a constituent, an illegal
observed value and a numerically rank-deficient adjustment design are each refused with the
columns named on ``error.refusal_fields.implicated_columns``. A column leaves the adjustment
set only through a :class:`DeclaredExclusion` that names its causal rule and a justification.

:func:`factorized_joint_cells` estimates the cells of a declared joint cell. Each cell
propensity is a product of binary conditionals under a declared ordering, every conditional
fit by cross-fitted ridge-logistic regression inside each observed prefix stratum. The result
carries, per cell, the AIPW point estimate with its weight effective sample size, propensity
range and clipped share; the normalization of the enumerated cell propensities; and the
estimates under every permutation of the ordering with a flag where they disagree. A cell
that cannot be evaluated is refused alone (``status == "unsupported"`` with its reason code)
and the rest of the family is kept. The claim is ``point_only``.

``nuisance="random_forest"`` or ``"gradient_boosted_trees"`` replaces the ridge-logistic
conditionals and the per-cell OLS outcome models with that learner, cross-fitted over the same
folds (a model never predicts a row it was fit on), with the learner identity, implementation
and seeds recorded on ``result.provenance``. A failed learner fit refuses the cells that need
it; nothing falls back to ridge. The learner must be available in the build. The result is
still ``point_only``.

Refused, with a registered reason code:

* an interval (``level=...``): ``penalized_interval_not_licensed``;
* an interval over a learner-supplied family (``nuisance="random_forest"`` with ``level=...``)
  and a machine-learning name that is not a declared learner (``"ml"``, ``"neural_network"``,
  ...): ``ml_nuisance_not_licensed``; ``"lasso"``: ``selection_inference_not_licensed``;
* an inconsistent declaration, leakage, illegal values: ``derived_treatment_invalid``;
* a rank-deficient adjustment design: ``design_rank_deficient``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._data import ingest_columns
from .errors import CausalValueError
from .preflight import PreflightReport
from .preflight import _report as _preflight_report

__all__ = [
    "CellRefusal",
    "ContrastValue",
    "DeclaredExclusion",
    "DerivedTreatment",
    "DerivedTreatmentPlan",
    "JointCell",
    "JointCellsResult",
    "NormalizationCheck",
    "OrderingSensitivity",
    "OrderingSpread",
    "SourceColumn",
    "check_derived_treatment",
    "factorized_joint_cells",
]


@dataclass(frozen=True, slots=True)
class SourceColumn:
    """One source column of a derived treatment.

    ``role`` is ``treatment_construction``, ``admissible_pre_treatment_covariate`` or
    ``forbidden_descendant``; ``when`` is ``pre_treatment``, ``at_treatment`` or
    ``post_treatment`` and must agree with the role.
    """

    name: str
    role: str
    when: str


@dataclass(frozen=True, slots=True)
class DeclaredExclusion:
    """A column removed from the adjustment set under a declared causal rule.

    ``rule`` is ``constituent_of_treatment`` (for a construction source) or
    ``post_treatment_descendant`` (for a forbidden descendant); a justification is required.
    """

    column: str
    rule: str
    justification: str


@dataclass(frozen=True, slots=True)
class DerivedTreatment:
    """An explicit derived-treatment declaration.

    ``transformation`` is ``joint_cell``, ``product`` or ``sum``; ``intervention`` is
    ``joint_components`` (valid only for a joint cell) or ``derived_level``. A joint cell's
    bit ``j`` is the ``j``-th construction source in the order listed.
    """

    name: str
    sources: Sequence[SourceColumn]
    legal_values: Sequence[float]
    transformation: str = "joint_cell"
    intervention: str = "joint_components"
    exclusions: Sequence[DeclaredExclusion] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "sources": [{"name": s.name, "role": s.role, "when": s.when} for s in self.sources],
                "transformation": self.transformation,
                "legal_values": [float(v) for v in self.legal_values],
                "intervention": self.intervention,
                "exclusions": [
                    {"column": e.column, "rule": e.rule, "justification": e.justification}
                    for e in self.exclusions
                ],
            }
        )

    @property
    def components(self) -> tuple[str, ...]:
        """The treatment-construction columns, in declared order."""
        return tuple(s.name for s in self.sources if s.role == "treatment_construction")


@dataclass(frozen=True, slots=True)
class DerivedTreatmentPlan:
    """The accepted construction: nothing was dropped except by a declared exclusion."""

    name: str
    treatment_columns: tuple[str, ...]
    adjustment: tuple[str, ...]
    exclusions: tuple[DeclaredExclusion, ...]
    retained_covariates: tuple[str, ...]
    observed_levels: tuple[float, ...]
    rows_complete: int
    report: PreflightReport


@dataclass(frozen=True, slots=True)
class CellRefusal:
    """Why one cell was refused, with its registered reason code."""

    code: str
    detail: str
    message: str


@dataclass(frozen=True, slots=True)
class JointCell:
    """One cell of the family under the declared ordering.

    ``levels`` is the level of each treatment component, in declared order. A supported cell
    carries its estimate and positivity diagnostics; an unsupported one carries ``refusal``
    and every numeric field is ``None``.
    """

    cell: int
    levels: tuple[int, ...]
    rows: int
    status: str
    estimate: float | None
    ess: float | None
    propensity_min: float | None
    propensity_max: float | None
    clipped_share: float | None
    refusal: CellRefusal | None


@dataclass(frozen=True, slots=True)
class NormalizationCheck:
    """Normalization of the enumerated cell propensities under one ordering.

    ``max_abs_error`` is ``None`` when some cell could not be enumerated.
    """

    ordering: tuple[str, ...]
    cells_enumerated: int
    max_abs_error: float | None
    max_row_sum: float


@dataclass(frozen=True, slots=True)
class OrderingSpread:
    """One cell's estimates under every ordering (``None`` where an ordering refused it)."""

    cell: int
    estimates: tuple[float | None, ...]
    spread: float | None
    flagged: bool


@dataclass(frozen=True, slots=True)
class OrderingSensitivity:
    """Sensitivity to the declared ordering; a flag is a receipt, never a verdict."""

    orderings: tuple[tuple[str, ...], ...]
    tolerance: float
    cells: tuple[OrderingSpread, ...]
    disagreement: bool


@dataclass(frozen=True, slots=True)
class ContrastValue:
    """A requested family contrast: a point value, or the refusal that withheld it."""

    name: str
    value: float | None
    refusal_code: str | None
    message: str | None


@dataclass(frozen=True, slots=True)
class JointCellsResult:
    """Factorized joint-cell estimates; ``claim`` is ``point_only`` and no interval exists."""

    treatments: tuple[str, ...]
    n_rows: int
    folds: int
    cells: tuple[JointCell, ...]
    normalization: tuple[NormalizationCheck, ...]
    sensitivity: OrderingSensitivity
    degenerate_conditionals: int
    contrasts: tuple[ContrastValue, ...]
    plan: DerivedTreatmentPlan
    provenance: str
    claim: str = "point_only"

    @property
    def supported(self) -> tuple[JointCell, ...]:
        return tuple(c for c in self.cells if c.status == "supported")

    @property
    def unsupported(self) -> tuple[JointCell, ...]:
        return tuple(c for c in self.cells if c.status != "supported")

    def cell(self, *levels: int) -> JointCell:
        """The cell with the given component levels, in treatment order."""
        for c in self.cells:
            if c.levels == tuple(levels):
                return c
        raise CausalValueError(f"no cell with levels {levels!r}")


def _plan(raw: Mapping[str, Any]) -> DerivedTreatmentPlan:
    return DerivedTreatmentPlan(
        name=raw["name"],
        treatment_columns=tuple(raw["treatment_columns"]),
        adjustment=tuple(raw["adjustment"]),
        exclusions=tuple(
            DeclaredExclusion(column=e["column"], rule=e["rule"], justification=e["justification"])
            for e in raw["exclusions"]
        ),
        retained_covariates=tuple(raw["retained_covariates"]),
        observed_levels=tuple(raw["observed_levels"]),
        rows_complete=raw["rows_complete"],
        report=_preflight_report(raw["report"]),
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


def _result(raw: Mapping[str, Any]) -> JointCellsResult:
    treatments = tuple(raw["treatments"])

    def named(ordering: Sequence[int]) -> tuple[str, ...]:
        return tuple(treatments[i] for i in ordering)

    sensitivity = raw["sensitivity"]
    return JointCellsResult(
        treatments=treatments,
        n_rows=raw["n_rows"],
        folds=raw["folds"],
        cells=tuple(_cell(c) for c in raw["cells"]),
        normalization=tuple(
            NormalizationCheck(
                ordering=named(n["ordering"]),
                cells_enumerated=n["cells_enumerated"],
                max_abs_error=n["max_abs_error"],
                max_row_sum=n["max_row_sum"],
            )
            for n in raw["normalization"]
        ),
        sensitivity=OrderingSensitivity(
            orderings=tuple(named(o) for o in sensitivity["orderings"]),
            tolerance=sensitivity["tolerance"],
            cells=tuple(
                OrderingSpread(
                    cell=c["cell"],
                    estimates=tuple(c["estimates"]),
                    spread=c["spread"],
                    flagged=c["flagged"],
                )
                for c in sensitivity["cells"]
            ),
            disagreement=sensitivity["disagreement"],
        ),
        degenerate_conditionals=raw["degenerate_conditionals"],
        contrasts=tuple(
            ContrastValue(
                name=c["name"],
                value=c["value"],
                refusal_code=c.get("code"),
                message=c.get("message"),
            )
            for c in raw["contrasts"]
        ),
        plan=_plan(raw["plan"]),
        provenance=raw["provenance"],
    )


def check_derived_treatment(
    data: Mapping[str, Any] | Any,
    declaration: DerivedTreatment,
    *,
    outcome: str,
    adjustment: Sequence[str],
    seed: int = 1,
    threads: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> DerivedTreatmentPlan:
    """Check a declaration against the table and the proposed adjustment set.

    Returns the accepted plan; raises ``CausalUnsupportedError`` (with
    ``refusal_fields.implicated_columns``) for every refusal listed in the module docstring.
    """
    names, columns = ingest_columns(data)
    raw = _native.derived_treatment_check_json(
        names,
        columns,
        declaration.to_json(),
        outcome,
        list(adjustment),
        seed=seed,
        threads=threads,
        cancel=cancel,
    )
    return _plan(json.loads(raw))


def factorized_joint_cells(
    data: Mapping[str, Any] | Any,
    declaration: DerivedTreatment,
    *,
    outcome: str,
    adjustment: Sequence[str],
    ordering: Sequence[str] | None = None,
    all_orderings: bool = True,
    nuisance: str = "ridge_logistic",
    penalties: Sequence[float] | None = None,
    inner_folds: int = 5,
    folds: int = 5,
    seed: int = 1,
    clip: float = 0.01,
    min_cell_ess: float = 10.0,
    normalization_tolerance: float = 1e-9,
    ordering_tolerance_sd: float = 0.05,
    contrasts: Sequence[str] = (),
    level: float | None = None,
    threads: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> JointCellsResult:
    """Estimate the cells of a declared joint-cell treatment (point only).

    ``ordering`` lists the construction columns in the declared factorization order (default:
    the declared source order); ``all_orderings`` re-estimates the family under every
    permutation. ``contrasts`` may hold ``"interaction"`` (two components, every cell
    supported) and ``"cell_minus_control:<cell>"``; a contrast that needs an unsupported cell
    comes back with its refusal code rather than a value. ``level`` requests an interval and
    is refused. ``nuisance`` is ``"ridge_logistic"`` (default), ``"random_forest"`` or
    ``"gradient_boosted_trees"``; ``penalties`` and ``inner_folds`` tune the ridge route only, so
    passing ``penalties`` with a learner is an error. At most three binary components are
    supported.
    """
    if not isinstance(nuisance, str):
        raise CausalValueError("nuisance must be a provider name string")
    if penalties is not None and nuisance != "ridge_logistic":
        raise CausalValueError(
            "penalties tune the ridge_logistic nuisance only; a declared learner takes none"
        )
    names, columns = ingest_columns(data)
    raw = _native.factorized_joint_cells_json(
        names,
        columns,
        declaration.to_json(),
        outcome,
        list(adjustment),
        ordering=None if ordering is None else list(ordering),
        all_orderings=all_orderings,
        nuisance=nuisance,
        penalties=None if penalties is None else [float(p) for p in penalties],
        inner_folds=inner_folds,
        folds=folds,
        seed=seed,
        clip=clip,
        min_cell_ess=min_cell_ess,
        normalization_tolerance=normalization_tolerance,
        ordering_tolerance_sd=ordering_tolerance_sd,
        contrasts=list(contrasts),
        interval=level is not None,
        threads=threads,
        cancel=cancel,
    )
    return _result(json.loads(raw))
