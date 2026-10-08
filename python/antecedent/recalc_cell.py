"""Selective recalculation on the cell-AIPW route, with portable score resume (2.3 C2).

This module extends :mod:`antecedent.recalc` without changing it. The plan, the receipt, the
stages and the structured refusals are that module's types; what is added is the route and the
way a fresh interpreter picks the work up.

**The cell-AIPW route.** :class:`CellSession` mirrors :class:`antecedent.recalc.RecalcSession`
over discrete joint binary treatments (the cell-saturated AIPW estimator)::

    session = recalc_cell.CellSession()
    first = session.execute(request, seed=7)        # (plan, receipt, law, decision)
    first.receipt.totals.fold_fits                  # 25 = 5 folds x (1 propensity + 4 cell fits)
    again = session.execute(replace(request, utility=Utility(3.0, 0.1)), seed=7)
    again.receipt.totals.fold_fits                  # 0: only the decision was recomputed

The fit count is read from the estimator's own instrument, which counts every multinomial
propensity and per-cell outcome regression that actually completed, so a zero is measured, not
asserted. A compatible target-weight change reweights the frozen cell scores; a changed outcome,
fold seed, graph, adjustment set or data refits.

**Portable resume.** :meth:`CellSession.export_frozen_scores` (and
:meth:`CrossfitSession.export_frozen_scores`) return :class:`FrozenScores`, a checksummed
``frozen_scores_v1`` artifact holding the scores and the producing workflow's stage digests but
no data and no model. :func:`resume_from_scores` builds a :class:`ScoreResumeSession` from those
bytes alone, in any interpreter. Its one licensed operation is :meth:`ScoreResumeSession.retarget`:
reweight the frozen scores by row weights and recompute the law and decision with zero fits.
Anything that would need the data or a fit (a changed outcome, folds, graph, row design, data
snapshot or new data) is refused before any work with the plan's ``Unavailable`` for the data
(:class:`ScoreResumeUnavailable`, ``detail == "recalc.unavailable_data"``) and leaves the session
unchanged. A retained ``expected_identity`` also refuses a consistently resealed artifact of
another run.

A resumed receipt shows ``score_artifact`` as ``reused`` with no work, and every other derived
stage recomputed (a fresh process never reuses a derived stage but the portable scores). The
``recalc_receipt_v1`` format refuses a derived stage reused in a fresh process, so a
:class:`ResumeReceipt` is a verified record, not an exportable artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from .errors import CausalSerializationError, CausalUnsupportedError, CausalValueError
from .recalc import (
    Decision,
    Law,
    RecalcNoLiveState,
    RecalcPlan,
    RecalcReceipt,
    RecalcRefusal,
    RecalcRequest,
    RecalcResult,
    RecalcUnavailable,
    Retarget,
    Stage,
    TargetWeights,
    Utility,
    stage_identity,
)

ARTIFACT_KIND = "frozen_scores_v1"

Quantity = Literal["interaction", "average_effect", "cell_minus_control"]
_QUANTITIES = ("interaction", "average_effect", "cell_minus_control")
_UNAVAILABLE = frozenset(
    {"recalc.unavailable_fit", "recalc.unavailable_data", "recalc.unavailable_provider"}
)


# -- refusals -------------------------------------------------------------------------------


class ScoreResumeRefusal(RecalcRefusal):
    """A request a score-resumed session cannot serve, or an artifact it refuses to resume.

    ``detail`` is the namespaced slot: ``recalc.unavailable_data`` for a request that needs the
    data or a fit, ``recalc.row_ids_mismatch`` / ``recalc.row_count_mismatch`` for weights over
    other rows, ``recalc.invalid_changed_input`` for a derived stage declared as a change and
    ``frozen_scores.identity_mismatch`` for an artifact whose identity is not the retained one.
    """


class ScoreResumeUnavailable(ScoreResumeRefusal, RecalcUnavailable):
    """The resumed session was given no data snapshot or fit for what the request needs.

    ``missing`` is ``"data"`` for every request a score-resumed session refuses: it holds the
    scores and the artifact's snapshot identity, never the data. ``plan`` is the refusing plan:
    its ``score_artifact`` is ``refused``, never claimed as reused.
    """


def _raise(refusal: str | None, *, resume: bool = False) -> None:
    if refusal is None:
        return
    wire = json.loads(refusal)
    detail = str(wire["detail"])
    unavailable = detail in _UNAVAILABLE
    if resume or detail.partition(".")[0] == "frozen_scores":
        raise (ScoreResumeUnavailable if unavailable else ScoreResumeRefusal)(wire)
    if unavailable:
        raise RecalcUnavailable(wire)
    if detail == "recalc.no_live_state":
        raise RecalcNoLiveState(wire)
    raise RecalcRefusal(wire)


# -- requests -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class CellRequest:
    """One cell-AIPW request over raw columns and declared edges.

    ``treatments`` are one to three binary columns intervened on jointly; ``adjustment`` the
    declared adjustment set (screened against the graph: no treatment, outcome or treatment
    descendant). ``quantity`` is the reported contrast: ``"interaction"`` (two treatments),
    ``"average_effect"`` (one treatment) or ``"cell_minus_control"`` with ``arm`` the cell
    mask. ``folds`` overrides the cross-fit fold count.
    """

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatments: Sequence[str]
    outcome: str
    utility: Utility
    adjustment: Sequence[str] = ()
    quantity: Quantity = "interaction"
    arm: int | None = None
    target: TargetWeights | None = None
    folds: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ResumeReceipt(RecalcReceipt):
    """The receipt of a resumed retarget: a verified record that cannot be exported.

    The table and counts are the Rust receipt's, checked against the plan before the call
    returned. ``recalc_receipt_v1`` refuses a derived stage reused in a fresh process, and a
    resumed run reuses the portable score artifact, so no artifact is sealed.
    """

    def export(self) -> bytes:
        """Always refuses: a resumed receipt is a record, not a ``recalc_receipt_v1`` artifact."""
        raise CausalUnsupportedError(
            "a resumed retarget receipt reuses the portable score artifact in a fresh process "
            "and is not exportable as recalc_receipt_v1",
            reason_code="route_not_supported",
            remedy="read the receipt's table and counts, or export it from the live session",
        )


def _column(name: str, values: ArrayLike) -> NDArray[np.float64]:
    try:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise CausalValueError(f"column {name!r} must be numeric") from error
    if array.ndim != 1:
        raise CausalValueError(f"column {name!r} must be one-dimensional")
    return array


def _seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise CausalValueError(f"seed must be a non-negative integer, got {seed!r}")
    return seed


def _quantity(quantity: str) -> str:
    if quantity not in _QUANTITIES:
        raise CausalValueError(f"quantity must be one of {_QUANTITIES}, got {quantity!r}")
    return quantity


def _frame(data: Mapping[str, ArrayLike]) -> tuple[list[str], list[NDArray[np.float64]]]:
    names = [str(name) for name in data]
    return names, [_column(name, data[name]) for name in data]


def _result(result: str | None, artifact: bytes | None, refusal: str | None) -> RecalcResult:
    _raise(refusal)
    if result is None or artifact is None:  # pragma: no cover - one of the two is set
        raise CausalSerializationError("recalc execution returned no result")
    wire = json.loads(result)
    receipt = RecalcReceipt._from_wire(  # noqa: SLF001
        wire["receipt"], wire["plan"], artifact, loaded=False
    )
    return RecalcResult(
        plan=RecalcPlan.from_wire(wire["plan"]),
        receipt=receipt,
        law=Law(**wire["law"]),
        decision=Decision(**wire["decision"]),
    )


# -- frozen scores --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class FrozenScores:
    """A portable ``frozen_scores_v1`` artifact: frozen scores, no data and no model.

    ``identity`` (64 hex characters) is the BLAKE3 identity of the scores, the estimator, the
    fit and snapshot identities and the producing workflow's stage digests. Retain it
    independently of the bytes: a consumer holding it refuses a consistently resealed artifact
    of another run.
    """

    identity: str
    _bytes: bytes

    def export(self) -> bytes:
        """The artifact bytes."""
        return self._bytes

    @classmethod
    def load(cls, data: bytes, *, expected_identity: str | None = None) -> FrozenScores:
        """Validate artifact bytes by recomputation and return them as :class:`FrozenScores`.

        Raises:
            ScoreResumeRefusal: ``expected_identity`` was given and is not the artifact's.
            CausalSerializationError: the bytes are corrupt, oversized, malformed or of an
                unsupported version.
        """
        handle = _resume_handle(bytes(data), expected_identity)
        return cls(identity=handle.artifact_identity(), _bytes=bytes(data))

    def resume(
        self,
        *,
        variables: Sequence[str],
        edges: Sequence[tuple[str, str]],
        utility: Utility,
        quantity: Quantity = "interaction",
        arm: int | None = None,
        expected_identity: str | None = None,
    ) -> ScoreResumeSession:
        """A :class:`ScoreResumeSession` built from these bytes alone."""
        return resume_from_scores(
            self._bytes,
            variables=variables,
            edges=edges,
            utility=utility,
            quantity=quantity,
            arm=arm,
            expected_identity=expected_identity,
        )


def _frozen(export: tuple[tuple[str, bytes] | None, str | None]) -> FrozenScores:
    value, refusal = export
    _raise(refusal)
    if value is None:  # pragma: no cover - one of the two is set
        raise CausalSerializationError("frozen score export returned nothing")
    return FrozenScores(identity=value[0], _bytes=value[1])


# -- cell session ---------------------------------------------------------------------------


class CellSession:
    """Selective recalculation of the cell-AIPW contrast over joint binary treatments.

    Holds the frozen cell scores, the last law and decision and the last identities. Each
    :meth:`execute` plans the request against them, runs only what the plan says must run,
    counts the work and returns ``(plan, receipt, law, decision)``. A refused plan raises
    :class:`~antecedent.recalc.RecalcRefusal` before any work and leaves the session unchanged.
    """

    __slots__ = ("_handle",)

    def __init__(self, *, retarget: Retarget = "licensed") -> None:
        self._handle = _native.CellSessionHandle(retarget)

    def set_retarget_support(self, retarget: Retarget) -> None:
        """Declare the estimator's retarget license for later plans (and for export)."""
        self._handle.set_retarget_support(retarget)

    @property
    def is_live(self) -> bool:
        """Whether the session holds live cell scores."""
        return self._handle.is_live()

    @property
    def identities(self) -> dict[Stage, str]:
        """Own-input digests of the last successful run."""
        return {Stage(r["stage"]): r["own"] for r in json.loads(self._handle.identities_json())}

    def score_columns(self) -> dict[int, NDArray[np.float64]] | None:
        """The live frozen scores by cell mask (``None`` without a run).

        The interaction is the weighted mean of ``s[0] - s[1] - s[2] + s[3]``, which is how a
        retargeted value is checked independently of the reweight that produced it.
        """
        columns = self._handle.score_columns()
        if columns is None:
            return None
        return {arm: np.asarray(values, dtype=np.float64) for arm, values in columns}

    def _call(
        self, request: CellRequest, seed: int, threads: int | None
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        names, columns = _frame(request.data)
        target = request.target
        positional = (
            names,
            columns,
            [(str(a), str(b)) for a, b in request.edges],
            [str(t) for t in request.treatments],
            request.outcome,
            [str(a) for a in request.adjustment],
            _quantity(request.quantity),
            float(request.utility.benefit_per_unit),
            float(request.utility.cost),
        )
        keywords: dict[str, Any] = {
            "arm": request.arm,
            "folds": request.folds,
            "target_weights": (
                None if target is None else _column("target weights", target.weights)
            ),
            "target_depends_on": None if target is None else [str(v) for v in target.depends_on],
            "seed": _seed(seed),
            "threads": threads,
        }
        return positional, keywords

    def plan(
        self, request: CellRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        """The plan ``request`` would run under, without running it."""
        positional, keywords = self._call(request, seed, threads)
        return RecalcPlan.from_wire(json.loads(self._handle.plan(*positional, **keywords)))

    def execute(
        self, request: CellRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcResult:
        """Plan, run only what the plan says must run, count the work and seal the receipt.

        Raises:
            RecalcRefusal: the plan holds a refused stage (an incompatible target), or the
                request is malformed at run time (``recalc.invalid_adjustment_set``,
                ``recalc.invalid_graph``); nothing was fitted.
            RecalcNoLiveState: the plan reuses a stage this session cannot back with live scores.
        """
        positional, keywords = self._call(request, seed, threads)
        return _result(*self._handle.execute(*positional, **keywords))

    def export_frozen_scores(self) -> FrozenScores:
        """Export the live cell scores as a portable :class:`FrozenScores` artifact.

        Raises:
            RecalcRefusal: ``recalc.retarget_not_licensed`` for a route whose retarget is not
                licensed (an unlicensed route cannot be resumed by retarget).
            RecalcNoLiveState: nothing has run in this session.
        """
        return _frozen(self._handle.export_frozen_scores())


# -- cross-fit session ----------------------------------------------------------------------


class CrossfitSession:
    """The cross-fitted AIPW average effect, so its frozen scores can be exported.

    Mirrors :class:`antecedent.recalc.RecalcSession.execute` (a first run is 10 fold fits, a
    compatible target change is 0) and adds :meth:`export_frozen_scores`.
    """

    __slots__ = ("_handle",)

    def __init__(self) -> None:
        self._handle = _native.CrossfitSessionHandle()

    @property
    def is_live(self) -> bool:
        """Whether the session holds a live prepared study."""
        return self._handle.is_live()

    def execute(
        self, request: RecalcRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcResult:
        """Plan, run only what the plan says must run, count the work and seal the receipt."""
        names, columns = _frame(request.data)
        target = request.target
        return _result(
            *self._handle.execute(
                names,
                columns,
                [(str(a), str(b)) for a, b in request.edges],
                request.treatment,
                request.outcome,
                float(request.utility.benefit_per_unit),
                float(request.utility.cost),
                target_weights=(
                    None if target is None else _column("target weights", target.weights)
                ),
                target_depends_on=(None if target is None else [str(v) for v in target.depends_on]),
                seed=_seed(seed),
                threads=threads,
            )
        )

    def export_frozen_scores(self) -> FrozenScores:
        """Export the live cross-fit scores as a portable :class:`FrozenScores` artifact."""
        return _frozen(self._handle.export_frozen_scores())


# -- portable resume ------------------------------------------------------------------------


def _resume_handle(data: bytes, expected_identity: str | None) -> _native.ScoreResumeHandle:
    handle, refusal = _native.ScoreResumeHandle.from_bytes(bytes(data), expected_identity)
    _raise(refusal, resume=True)
    if handle is None:  # pragma: no cover - one of the two is set
        raise CausalSerializationError("frozen scores artifact did not decode")
    return handle


class ScoreResumeSession:
    """A session built only from :class:`FrozenScores` bytes.

    It holds the frozen scores and the producing workflow's identities, and no data, fitted
    model or prepared study. Its one licensed operation is :meth:`retarget`; :meth:`execute`
    and every ``declared_changes`` entry that would need the data or a fit are refused with
    :class:`ScoreResumeUnavailable` before any work.
    """

    __slots__ = ("_arm", "_edges", "_handle", "_quantity", "_utility", "_variables")

    def __init__(
        self,
        handle: _native.ScoreResumeHandle,
        *,
        variables: Sequence[str],
        edges: Sequence[tuple[str, str]],
        utility: Utility,
        quantity: Quantity,
        arm: int | None,
    ) -> None:
        self._handle = handle
        self._variables = tuple(str(v) for v in variables)
        if len(set(self._variables)) != len(self._variables) or not self._variables:
            raise CausalValueError("variables must be unique, non-empty column names")
        self._edges = tuple((str(a), str(b)) for a, b in edges)
        self._utility = utility
        self._quantity = _quantity(quantity)
        self._arm = arm

    # -- state --

    @property
    def identity(self) -> str:
        """Identity (64 hex characters) of the artifact the session was built from."""
        return self._handle.artifact_identity()

    @property
    def n_rows(self) -> int:
        """Row count of the frozen scores."""
        return self._handle.n_rows()

    @property
    def row_ids(self) -> NDArray[np.uint32]:
        """The original row ids target weights are indexed by, in row order."""
        return np.asarray(self._handle.row_ids(), dtype=np.uint32)

    @property
    def has_run(self) -> bool:
        """Whether a resumed retarget has completed."""
        return self._handle.has_run()

    @property
    def identities(self) -> dict[Stage, str]:
        """Own-input digests of the last successful run (the artifact's before any run)."""
        return {Stage(r["stage"]): r["own"] for r in json.loads(self._handle.identities_json())}

    def score_columns(self) -> dict[int, NDArray[np.float64]]:
        """The frozen scores by cell mask (by arm for a cross-fit table)."""
        return {
            arm: np.asarray(values, dtype=np.float64)
            for arm, values in self._handle.score_columns()
        }

    # -- the licensed operation --

    def _position(self, name: str) -> int:
        try:
            return self._variables.index(name)
        except ValueError as error:
            raise CausalValueError(
                f"unknown variable {name!r}; variables: {self._variables}"
            ) from error

    def _arguments(
        self,
        target_weights: ArrayLike | TargetWeights | None,
        utility: Utility | None,
        depends_on: Sequence[str],
        row_ids: Sequence[int] | None,
        edges: Sequence[tuple[str, str]] | None,
        declared_changes: Mapping[Stage, str | bytes] | None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        weights: NDArray[np.float64] | None
        parents = tuple(depends_on)
        if isinstance(target_weights, TargetWeights):
            weights = _column("target weights", target_weights.weights)
            parents = parents or tuple(target_weights.depends_on)
        else:
            weights = None if target_weights is None else _column("target weights", target_weights)
        use = utility if utility is not None else self._utility
        chosen = self._edges if edges is None else tuple((str(a), str(b)) for a, b in edges)
        changes = [
            (Stage(stage).value, stage_identity(Stage(stage), part))
            for stage, part in (declared_changes or {}).items()
        ]
        positional = (
            len(self._variables),
            [(self._position(a), self._position(b)) for a, b in chosen],
            self._quantity,
            float(use.benefit_per_unit),
            float(use.cost),
        )
        keywords: dict[str, Any] = {
            "arm": self._arm,
            "target_weights": weights,
            "target_depends_on": [self._position(p) for p in parents],
            "target_row_ids": None if row_ids is None else [int(r) for r in row_ids],
            "changed_inputs": changes,
        }
        return positional, keywords

    def plan(
        self,
        target_weights: ArrayLike | TargetWeights | None = None,
        *,
        utility: Utility | None = None,
        depends_on: Sequence[str] = (),
        row_ids: Sequence[int] | None = None,
        edges: Sequence[tuple[str, str]] | None = None,
        declared_changes: Mapping[Stage, str | bytes] | None = None,
    ) -> RecalcPlan:
        """The plan a retarget would run under; refused (raising) when it needs data or a fit."""
        positional, keywords = self._arguments(
            target_weights, utility, depends_on, row_ids, edges, declared_changes
        )
        plan, refusal = self._handle.plan(*positional, **keywords)
        _raise(refusal, resume=True)
        if plan is None:  # pragma: no cover - one of the two is set
            raise CausalSerializationError("resume plan returned nothing")
        return RecalcPlan.from_wire(json.loads(plan))

    def retarget(
        self,
        target_weights: ArrayLike | TargetWeights | None = None,
        *,
        utility: Utility | None = None,
        depends_on: Sequence[str] = (),
        row_ids: Sequence[int] | None = None,
        edges: Sequence[tuple[str, str]] | None = None,
        declared_changes: Mapping[Stage, str | bytes] | None = None,
    ) -> RecalcResult:
        """Reweight the frozen scores and recompute the law and decision, with zero fits.

        ``target_weights`` are non-negative row weights in the frozen scores' row order (or a
        :class:`~antecedent.recalc.TargetWeights` carrying ``depends_on``); ``None`` is the
        observed population. ``depends_on`` names the variables the weights are a function of
        (they must lie in the adjustment set). ``utility`` defaults to the resume's.
        ``row_ids``, when given, must equal :attr:`row_ids`. ``declared_changes`` maps declared
        input stages to a new description (``Stage.QUERY: "outcome=y2"``); any change needs the
        data or a fit, so it is refused.

        Raises:
            ScoreResumeUnavailable: the request needs the data or a fit (a changed outcome,
                folds, graph, row design, data snapshot or treatment grid); nothing ran.
            ScoreResumeRefusal: weights over other row ids or of another length, or a derived
                stage declared as a change; nothing ran.
        """
        positional, keywords = self._arguments(
            target_weights, utility, depends_on, row_ids, edges, declared_changes
        )
        result, refusal = self._handle.execute_retarget(*positional, **keywords)
        _raise(refusal, resume=True)
        if result is None:  # pragma: no cover - one of the two is set
            raise CausalSerializationError("resume retarget returned no result")
        wire = json.loads(result)
        receipt = ResumeReceipt._from_wire(  # noqa: SLF001
            wire["receipt"], wire["plan"], b"", loaded=False
        )
        return RecalcResult(
            plan=RecalcPlan.from_wire(wire["plan"]),
            receipt=receipt,
            law=Law(**wire["law"]),
            decision=Decision(**wire["decision"]),
        )

    def execute(self, request: CellRequest | RecalcRequest, *, seed: int = 1) -> RecalcResult:
        """Refused: a resumed session holds no data, so it cannot run a request over data.

        The request's data is not bound to the artifact's snapshot, so the plan refuses with the
        data as the missing dependency. Use :meth:`retarget` for the licensed operation.

        Raises:
            ScoreResumeUnavailable: always, before any work.
        """
        _seed(seed)
        _frame(request.data)
        unbound = {Stage.DATA_SNAPSHOT: "unbound data supplied to a resumed session"}
        self.retarget(declared_changes=unbound)
        raise CausalSerializationError(  # pragma: no cover - the plan always refuses
            "a resumed session executed a request over data"
        )


def resume_from_scores(
    data: bytes | FrozenScores,
    *,
    variables: Sequence[str],
    edges: Sequence[tuple[str, str]],
    utility: Utility,
    quantity: Quantity = "interaction",
    arm: int | None = None,
    expected_identity: str | None = None,
) -> ScoreResumeSession:
    """A :class:`ScoreResumeSession` built from artifact bytes alone, with zero fits.

    ``variables`` are the original data's column names in column order (they fix the graph's
    variable count), ``edges`` the original graph and ``utility`` the original utility: the
    resumed workflow is the artifact's, and :meth:`ScoreResumeSession.retarget` changes the
    target population (and, if wanted, the utility). ``expected_identity`` is an identity the
    caller retained independently; an artifact with another identity is refused.

    Raises:
        ScoreResumeRefusal: ``frozen_scores.identity_mismatch`` for another identity.
        CausalSerializationError: corrupt, oversized, malformed or unsupported-version bytes.
    """
    raw = data.export() if isinstance(data, FrozenScores) else bytes(data)
    handle = _resume_handle(raw, expected_identity)
    return ScoreResumeSession(
        handle, variables=variables, edges=edges, utility=utility, quantity=quantity, arm=arm
    )


__all__ = [
    "ARTIFACT_KIND",
    "CellRequest",
    "CellSession",
    "CrossfitSession",
    "FrozenScores",
    "Quantity",
    "ResumeReceipt",
    "ScoreResumeRefusal",
    "ScoreResumeSession",
    "ScoreResumeUnavailable",
    "resume_from_scores",
]
