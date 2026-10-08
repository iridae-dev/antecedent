"""Selective recalculation with a visible receipt.

A causal workflow is a small directed graph of stages: declared inputs (graph, query, regime,
evidence, populations, data snapshot, row design, treatment grid, learner/folds/RNG, external
studies, utility) and derived stages (identification, provider request, prior, score or fit
artifact, law, decision). Every stage carries one identity: a digest of its declared inputs and
of the identities of the stages it depends on. A :class:`RecalcPlan` compares the identities of
a previous and a requested workflow and says, per stage, whether it is ``reused``,
``recomputed`` or ``refused``, and names the dependency that determined that status::

    plan = recalc.plan_recalculation(previous, requested)       # decide and explain, run nothing
    print(plan.explain())

    session = recalc.RecalcSession()
    first = session.execute(request, seed=7)                    # (plan, receipt, law, decision)
    again = session.execute(replace(request, utility=Utility(3.0, 0.1)), seed=7)
    again.plan.recomputed_computations                          # (Stage.DECISION,)
    again.receipt.totals.fold_fits                              # 0: nothing was refit
    blob = again.receipt.export()                               # portable recalc_receipt_v1
    recalc.RecalcReceipt.consume(blob, expected_identity=again.receipt.identity)

:meth:`RecalcSession.execute` runs one real route, the cross-fitted AIPW average effect over a
binary treatment, through the prepared-study architecture (``prepare``, ``refresh`` and
``retarget``), and counts what actually ran: identifications, nuisance fold fits read from the
estimator's own per-call fit cache, score builds, reweights and decisions. A receipt refuses to
exist when a count contradicts its status (a reused stage that did work, or a recomputed
computation that did none), so reuse is proven by counts and not by the plan alone. A
compatible target-weight change reuses the frozen scores; a utility-only change recomputes the
decision alone; changed folds, new rows or a changed graph, regime or query recompute what
depends on them.

What reuse means: "reused" never means "persistently cached". Within one process it names an
artifact the session still holds. Across a process boundary it needs a portable fit, score,
data snapshot or provider named in a :class:`ResumeContext`; an ordinary loaded result
supplies none of them, so every derived stage is then recomputed or refused with a specific
``recalc.unavailable_*`` result, never reused. A loaded :class:`RecalcReceipt` is a record: it
never claims reuse in a fresh process, and :meth:`RecalcReceipt.consume` refuses one that does.

Rust owns the stage model, the planner, the counts, the artifact identity, its independent
recomputation and every refusal; this module builds declarations and raises each structured
refusal as :class:`RecalcRefusal`, a :class:`~antecedent.errors.CausalUnsupportedError` with its
registered ``reason_code`` and a namespaced ``detail`` (``recalc.*`` or ``recalc_receipt.*``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native
from ._b4_refusal import StructuredRefusal
from .errors import CausalSerializationError, CausalValueError

ARTIFACT_KIND = "recalc_receipt_v1"
MAX_EXTERNAL_BRANCHES = 8

Retarget = Literal["licensed", "not_declared", "incompatible"]
"""Whether a target-population change is a licensed reweighting of retained scores
(``licensed``), has no declared reweighting (``not_declared``) or is incompatible with them."""
RequestKind = Literal["on_grid", "off_grid", "unsupported"]
"""How a request relates to the licensed grid of changes: ``on_grid`` (a licensed route),
``off_grid`` (a refit is needed) or ``unsupported`` (no route)."""
StatusKind = Literal["reused", "recomputed", "refused"]

_RETARGETS = ("licensed", "not_declared", "incompatible")
_REQUESTS = ("on_grid", "off_grid", "unsupported")


# -- stages ---------------------------------------------------------------------------------


class Stage(StrEnum):
    """One stage of the workflow, in topological order. The value is the stable label."""

    GRAPH = "graph"
    QUERY = "query"
    REGIME = "regime"
    EVIDENCE = "evidence"
    SOURCE_POPULATION = "source_population"
    TARGET_POPULATION = "target_population"
    DATA_SNAPSHOT = "data_snapshot"
    ROW_DESIGN = "row_design"
    TREATMENT_GRID = "treatment_grid"
    LEARNER_FOLDS_RNG = "learner_folds_rng"
    UTILITY = "utility"
    EXTERNAL_STUDY_0 = "external_study.0"
    EXTERNAL_STUDY_1 = "external_study.1"
    EXTERNAL_STUDY_2 = "external_study.2"
    EXTERNAL_STUDY_3 = "external_study.3"
    EXTERNAL_STUDY_4 = "external_study.4"
    EXTERNAL_STUDY_5 = "external_study.5"
    EXTERNAL_STUDY_6 = "external_study.6"
    EXTERNAL_STUDY_7 = "external_study.7"
    PROVIDER_REQUEST_0 = "provider_request.0"
    PROVIDER_REQUEST_1 = "provider_request.1"
    PROVIDER_REQUEST_2 = "provider_request.2"
    PROVIDER_REQUEST_3 = "provider_request.3"
    PROVIDER_REQUEST_4 = "provider_request.4"
    PROVIDER_REQUEST_5 = "provider_request.5"
    PROVIDER_REQUEST_6 = "provider_request.6"
    PROVIDER_REQUEST_7 = "provider_request.7"
    PRIOR_0 = "prior.0"
    PRIOR_1 = "prior.1"
    PRIOR_2 = "prior.2"
    PRIOR_3 = "prior.3"
    PRIOR_4 = "prior.4"
    PRIOR_5 = "prior.5"
    PRIOR_6 = "prior.6"
    PRIOR_7 = "prior.7"
    IDENTIFICATION = "identification"
    SCORE_ARTIFACT = "score_artifact"
    LAW = "law"
    DECISION = "decision"

    @property
    def is_input(self) -> bool:
        """Whether this stage is a declared input rather than a computation."""
        family = self.value.split(".", 1)[0]
        return family not in {
            "provider_request",
            "prior",
            "identification",
            "score_artifact",
            "law",
            "decision",
        }

    @classmethod
    def external_study(cls, branch: int) -> Stage:
        """The external-study input of ``branch`` (``0 <= branch < 8``)."""
        return cls(f"external_study.{_branch(branch)}")

    @classmethod
    def provider_request(cls, branch: int) -> Stage:
        """The provider request and version derived from external study ``branch``."""
        return cls(f"provider_request.{_branch(branch)}")

    @classmethod
    def prior(cls, branch: int) -> Stage:
        """The prior built from external study ``branch`` and the evidence."""
        return cls(f"prior.{_branch(branch)}")


def _branch(branch: int) -> int:
    if not isinstance(branch, int) or isinstance(branch, bool) or not 0 <= branch < 8:
        raise CausalValueError(
            f"external-study branch must be an integer in [0, {MAX_EXTERNAL_BRANCHES}), "
            f"got {branch!r}"
        )
    return branch


def stage_identity(stage: Stage | str, *parts: bytes | str) -> str:
    """The own-input digest (64 hex characters) of a stage declaration.

    The digest covers the stage label and each part, length-prefixed, so ``("ab", "c")`` and
    ``("a", "bc")`` differ. A caller declares a workflow as ``{stage: stage_identity(...)}``.
    """
    label = _stage(stage).value
    raw = [p.encode() if isinstance(p, str) else bytes(p) for p in parts]
    return _native.recalc_stage_identity(label, raw)


def _stage(value: Stage | str) -> Stage:
    try:
        return Stage(value)
    except ValueError as error:
        raise CausalValueError(f"unknown recalculation stage {value!r}") from error


Declaration = Mapping[Stage, str]


def _declared_json(declaration: Mapping[Stage, str] | Mapping[str, str]) -> str:
    rows = []
    for key, own in declaration.items():
        if not isinstance(own, str):
            raise CausalValueError("a stage declaration is a 64-character hex digest string")
        rows.append({"stage": _stage(key).value, "own": own})
    return json.dumps(rows)


# -- capabilities ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResumeContext:
    """What a fresh process was handed to resume with.

    An ordinary loaded result supplies none of these (all ``False``): ``ant.load`` does not
    recreate a prepared study. Only a portable fit or score, a supplied compatible data
    snapshot or a supplied compatible provider can resume its licensed operation.
    """

    portable_fit: bool = False
    portable_scores: bool = False
    supplied_data: bool = False
    supplied_provider: bool = False
    #: Portable scores bind the unchanged snapshot identity; raw data remain absent.
    scores_snapshot_bound: bool = False

    def to_wire(self) -> dict[str, bool]:
        """JSON-ready form."""
        wire = {
            "portable_fit": self.portable_fit,
            "portable_scores": self.portable_scores,
            "supplied_data": self.supplied_data,
            "supplied_provider": self.supplied_provider,
        }
        if self.scores_snapshot_bound:
            wire["scores_snapshot_bound"] = True
        return wire


@dataclass(frozen=True, slots=True)
class Capabilities:
    """The declared capabilities a plan is made under.

    ``retarget`` says whether the estimator's frozen-score row-weight retarget is licensed
    (``"licensed"``: a compatible target change reweights frozen scores without a refit;
    ``"not_declared"``: it recomputes the score artifact; ``"incompatible"``: the weights are
    not a licensed function of the adjustment set, so a target change is refused).
    ``request`` says whether the requested treatment grid or operation is inside the declared
    route, with the separately ``licensed_route`` that serves it when it is not. ``resume`` is
    ``None`` in process and a :class:`ResumeContext` across a process boundary.
    """

    retarget: Retarget = "licensed"
    request: RequestKind = "on_grid"
    licensed_route: str | None = None
    resume: ResumeContext | None = None

    def __post_init__(self) -> None:
        if self.retarget not in _RETARGETS:
            raise CausalValueError(f"retarget must be one of {_RETARGETS}, got {self.retarget!r}")
        if self.request not in _REQUESTS:
            raise CausalValueError(f"request must be one of {_REQUESTS}, got {self.request!r}")
        if self.request == "on_grid" and self.licensed_route is not None:
            raise CausalValueError("an on-grid request names no licensed route")

    @property
    def boundary(self) -> Literal["in_process", "fresh_process"]:
        """``"fresh_process"`` when a resume context is declared."""
        return "in_process" if self.resume is None else "fresh_process"

    def to_wire(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "retarget": self.retarget,
            "request": self.request,
            "licensed_route": self.licensed_route,
            "boundary": self.boundary,
            "resume": None if self.resume is None else self.resume.to_wire(),
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> Capabilities:
        """Rebuild from the native JSON form."""
        resume = wire.get("resume")
        return cls(
            retarget=wire["retarget"],
            request=wire["request"],
            licensed_route=wire.get("licensed_route"),
            resume=None if resume is None else ResumeContext(**resume),
        )


# -- plan -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageStatus:
    """Per-stage outcome: ``reused``, ``recomputed`` or ``refused``, and what determined it.

    ``detail`` is the determining dependency: for ``reused`` the dependency whose unchanged
    identity licensed reuse (the stage itself for an input); for ``recomputed`` the changed
    input (``own:<stage>:<change>``), the changed upstream stage
    (``upstream:<via><-<origin>:<change>``) or ``fresh_process``; for ``refused`` the
    ``recalc.*`` detail (with a ``[route]`` or ``[stage]`` suffix where one applies).
    """

    kind: StatusKind
    detail: str

    @property
    def text(self) -> str:
        """The canonical text, for example ``recomputed(own:utility:modified)``."""
        return f"{self.kind}({self.detail})"

    @property
    def reused(self) -> bool:
        """Whether the artifact is valid as it is."""
        return self.kind == "reused"

    @property
    def recomputed(self) -> bool:
        """Whether the stage must be recomputed."""
        return self.kind == "recomputed"

    @property
    def refused(self) -> bool:
        """Whether the stage cannot be served."""
        return self.kind == "refused"

    def __str__(self) -> str:
        return self.text

    def explain(self, stage: Stage) -> str:
        """One plain sentence naming the determining dependency."""
        if self.kind == "reused":
            if self.detail == stage.value:
                return "unchanged input"
            return f"reused: its dependency `{self.detail}` is unchanged"
        if self.kind == "refused":
            return f"refused: {self.detail}"
        if self.detail == "fresh_process":
            return "recomputed: this fresh process holds no artifact for it"
        scope, _, rest = self.detail.partition(":")
        if scope == "own":
            own, _, change = rest.rpartition(":")
            return f"recomputed: its own input `{own}` was {change}"
        via, _, tail = rest.partition("<-")
        origin, _, change = tail.rpartition(":")
        return f"recomputed: `{via}` changed because `{origin}` was {change}"


def _status_of(wire: Mapping[str, Any]) -> StageStatus:
    return StageStatus(kind=wire["tag"], detail=wire["detail"])


@dataclass(frozen=True, slots=True)
class PlanEntry:
    """One row of a plan: the stage, its status and its effective identity (hex)."""

    stage: Stage
    status: StageStatus
    identity: str


@dataclass(frozen=True, slots=True)
class RecalcPlan:
    """Per-stage reuse, recompute or refuse table for a requested workflow."""

    entries: tuple[PlanEntry, ...]
    identity: str
    is_executable: bool

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> RecalcPlan:
        """Rebuild from the native JSON form."""
        return cls(
            entries=tuple(
                PlanEntry(Stage(row["stage"]), _status_of(row), row["identity"])
                for row in wire["entries"]
            ),
            identity=wire["identity"],
            is_executable=bool(wire["executable"]),
        )

    @property
    def table(self) -> tuple[tuple[str, str], ...]:
        """``(stage label, status text)`` rows in topological order."""
        return tuple((e.stage.value, e.status.text) for e in self.entries)

    def _with(self, kind: StatusKind) -> tuple[Stage, ...]:
        return tuple(e.stage for e in self.entries if e.status.kind == kind)

    @property
    def reused(self) -> tuple[Stage, ...]:
        """Stages with status ``reused``."""
        return self._with("reused")

    @property
    def recomputed(self) -> tuple[Stage, ...]:
        """Stages with status ``recomputed``."""
        return self._with("recomputed")

    @property
    def refused(self) -> tuple[Stage, ...]:
        """Stages with status ``refused``."""
        return self._with("refused")

    @property
    def recomputed_computations(self) -> tuple[Stage, ...]:
        """Recomputed stages that are computations, not declared inputs."""
        return tuple(s for s in self.recomputed if not s.is_input)

    def status(self, stage: Stage) -> StageStatus:
        """The status of ``stage`` (it must be part of the requested workflow)."""
        for entry in self.entries:
            if entry.stage is stage:
                return entry.status
        raise CausalValueError(f"stage {stage.value!r} is not part of the requested workflow")

    def identity_of(self, stage: Stage) -> str:
        """The effective identity (hex) of ``stage`` in the requested workflow."""
        for entry in self.entries:
            if entry.stage is stage:
                return entry.identity
        raise CausalValueError(f"stage {stage.value!r} is not part of the requested workflow")

    @property
    def first_refusal(self) -> tuple[Stage, StageStatus] | None:
        """The first refused stage and its status, if any."""
        for entry in self.entries:
            if entry.status.refused:
                return entry.stage, entry.status
        return None

    def explain(self) -> str:
        """The whole table with, per stage, the dependency that determined its status."""
        width = max((len(e.stage.value) for e in self.entries), default=0)
        lines = [
            f"{e.stage.value:<{width}}  {e.status.text}  -- {e.status.explain(e.stage)}"
            for e in self.entries
        ]
        head = (
            f"plan {self.identity[:12]}: {len(self.reused)} reused, "
            f"{len(self.recomputed)} recomputed, {len(self.refused)} refused"
        )
        return "\n".join([head, *lines])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: per-stage status and identity."""
        return {
            "identity": self.identity,
            "executable": self.is_executable,
            "entries": [
                {"stage": e.stage.value, "status": e.status.text, "identity": e.identity}
                for e in self.entries
            ],
        }

    def __repr__(self) -> str:
        return (
            f"<RecalcPlan {self.identity[:12]} reused={len(self.reused)} "
            f"recomputed={len(self.recomputed)} refused={len(self.refused)} "
            f"executable={self.is_executable}>"
        )


def plan_recalculation(
    previous: Mapping[Stage, str] | Mapping[str, str],
    requested: Mapping[Stage, str] | Mapping[str, str],
    capabilities: Capabilities | None = None,
) -> RecalcPlan:
    """Plan ``requested`` against ``previous`` under ``capabilities``; run nothing.

    ``previous`` and ``requested`` map each stage in the workflow to its own-input digest
    (see :func:`stage_identity`); only stages declared in ``requested`` appear in the plan.
    """
    caps = capabilities if capabilities is not None else Capabilities()
    wire = _native.plan_recalculation(
        _declared_json(previous), _declared_json(requested), json.dumps(caps.to_wire())
    )
    return RecalcPlan.from_wire(json.loads(wire))


# -- refusals -------------------------------------------------------------------------------


class RecalcRefusal(StructuredRefusal):
    """A structured refusal of a recalculation or of a receipt artifact.

    ``reason_code`` is registered, ``detail`` is the namespaced ``recalc.*`` or
    ``recalc_receipt.*`` slot, ``stage`` the refusing stage label and ``plan`` the plan the
    refusal was part of (when there is one).
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        super().__init__(refusal)
        plan = refusal.get("plan")
        self.plan: RecalcPlan | None = None if plan is None else RecalcPlan.from_wire(plan)


class RecalcUnavailable(RecalcRefusal):
    """A dependency the fresh process was not given (``recalc.unavailable_*``).

    ``missing`` is ``"fit"``, ``"data"`` or ``"provider"``.
    """

    @property
    def missing(self) -> str:
        """What the fresh process lacks."""
        return (self.detail or "").rpartition("_")[2]


class RecalcNoLiveState(RecalcRefusal):
    """The plan reuses a derived stage but the session holds no live artifact for it."""


class RecalcReceiptRefusal(RecalcRefusal):
    """A receipt artifact whose table, counts or identity do not recompute."""


def _raise(refusal: str | None) -> None:
    if refusal is None:
        return
    wire = json.loads(refusal)
    detail = str(wire["detail"])
    if detail in {
        "recalc.unavailable_fit",
        "recalc.unavailable_data",
        "recalc.unavailable_provider",
    }:
        raise RecalcUnavailable(wire)
    if detail == "recalc.no_live_state":
        raise RecalcNoLiveState(wire)
    if detail.partition(".")[0] == "recalc_receipt":
        raise RecalcReceiptRefusal(wire)
    raise RecalcRefusal(wire)


# -- receipt --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageCounts:
    """Work counted for one stage (or the totals)."""

    identifications: int = 0
    fold_fits: int = 0
    score_computations: int = 0
    reweights: int = 0
    decisions: int = 0
    #: Successful full-model fits; distinct from nuisance fold fits.
    model_fits: int = 0
    posterior_draws: int = 0
    external_invocations: int = 0
    law_summaries: int = 0
    factor_builds: int = 0
    program_compilations: int = 0
    provider_bindings: int = 0
    factor_evaluations: int = 0
    integrations: int = 0
    provider_calls: int = 0

    @property
    def total(self) -> int:
        """Sum over every kind of work."""
        return (
            self.identifications
            + self.fold_fits
            + self.score_computations
            + self.reweights
            + self.decisions
            + self.model_fits
            + self.posterior_draws
            + self.external_invocations
            + self.law_summaries
            + self.factor_builds
            + self.program_compilations
            + self.provider_bindings
            + self.factor_evaluations
            + self.integrations
            + self.provider_calls
        )

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        """``(identifications, fold_fits, score_computations, reweights, decisions)``."""
        return (
            self.identifications,
            self.fold_fits,
            self.score_computations,
            self.reweights,
            self.decisions,
        )


@dataclass(frozen=True, slots=True)
class ReceiptEntry:
    """One stage of a receipt: status, effective identity and the work counted for it."""

    stage: Stage
    status: StageStatus
    identity: str
    counts: StageCounts


def _declared(rows: Sequence[Mapping[str, Any]]) -> dict[Stage, str]:
    return {Stage(row["stage"]): row["own"] for row in rows}


@dataclass(frozen=True, slots=True, eq=False)
class RecalcReceipt:
    """What a call reused, recomputed, refused and did, with its canonical identity.

    ``identity`` depends only on the per-stage status, stage identity and counts, not on the
    order they were supplied in. ``previous`` and ``requested`` are the declared own-input
    digests the plan compared; ``capabilities`` the declarations it was made under.
    ``loaded`` is ``True`` for a receipt rebuilt by :meth:`consume`: a loaded receipt is a
    record and recreates no prepared study, score table or fit.
    """

    entries: tuple[ReceiptEntry, ...]
    totals: StageCounts
    identity: str
    plan_identity: str
    capabilities: Capabilities
    previous: dict[Stage, str]
    requested: dict[Stage, str]
    plan: RecalcPlan
    loaded: bool
    _artifact: bytes

    @classmethod
    def _from_wire(
        cls, meta: Mapping[str, Any], plan: Mapping[str, Any], artifact: bytes, *, loaded: bool
    ) -> RecalcReceipt:
        return cls(
            entries=tuple(
                ReceiptEntry(
                    Stage(row["stage"]),
                    _status_of(row),
                    row["identity"],
                    StageCounts(**row["counts"]),
                )
                for row in meta["entries"]
            ),
            totals=StageCounts(**meta["totals"]),
            identity=meta["receipt_identity"],
            plan_identity=meta["plan_identity"],
            capabilities=Capabilities.from_wire(meta["capabilities"]),
            previous=_declared(meta["previous"]),
            requested=_declared(meta["requested"]),
            plan=RecalcPlan.from_wire(plan),
            loaded=loaded,
            _artifact=artifact,
        )

    def entry(self, stage: Stage) -> ReceiptEntry:
        """The entry of ``stage``."""
        for entry in self.entries:
            if entry.stage is stage:
                return entry
        raise CausalValueError(f"stage {stage.value!r} is not part of the receipt")

    def counts(self, stage: Stage) -> StageCounts:
        """The work counted against ``stage``."""
        return self.entry(stage).counts

    @property
    def status_table(self) -> tuple[tuple[str, str], ...]:
        """``(stage label, status text)`` rows in topological order."""
        return tuple((e.stage.value, e.status.text) for e in self.entries)

    @property
    def reused(self) -> tuple[Stage, ...]:
        """Stages with status ``reused``."""
        return tuple(e.stage for e in self.entries if e.status.reused)

    @property
    def recomputed(self) -> tuple[Stage, ...]:
        """Stages with status ``recomputed``."""
        return tuple(e.stage for e in self.entries if e.status.recomputed)

    @property
    def refused(self) -> tuple[Stage, ...]:
        """Stages with status ``refused``."""
        return tuple(e.stage for e in self.entries if e.status.refused)

    @property
    def claims_derived_reuse(self) -> bool:
        """Whether any derived stage is stored as reused (never true in a fresh process)."""
        return any(e.status.reused and not e.stage.is_input for e in self.entries)

    def export(self) -> bytes:
        """The portable ``recalc_receipt_v1`` artifact bytes."""
        return self._artifact

    @classmethod
    def consume(cls, data: bytes, *, expected_identity: str | None = None) -> RecalcReceipt:
        """Consume an artifact by recomputing its plan from the stored declarations.

        The stored table, counts, totals and identities must match the recomputed plan; a
        resealed edit is refused as :class:`RecalcReceiptRefusal`, a derived stage stored as
        reused under a fresh-process boundary is refused (``recalc_receipt.fresh_process_reuse``)
        and, when ``expected_identity`` was retained independently, a consistently resealed
        receipt of another run is refused too. Corruption raises ``CausalSerializationError``.
        """
        report, refusal = _native.consume_recalc_receipt(bytes(data), expected_identity)
        _raise(refusal)
        if report is None:  # pragma: no cover - the native call returns one or the other
            raise CausalSerializationError("recalc receipt artifact did not decode")
        wire = json.loads(report)
        return cls._from_wire(wire["receipt"], wire["plan"], bytes(data), loaded=True)

    def explain(self) -> str:
        """The table with the work behind each recomputed stage."""
        width = max((len(e.stage.value) for e in self.entries), default=0)
        kinds = ("identifications", "fold_fits", "score_computations", "reweights", "decisions")
        lines = []
        for e in self.entries:
            work = ", ".join(
                f"{name}={n}" for name, n in zip(kinds, e.counts.as_tuple(), strict=True) if n
            )
            for name in (
                "model_fits",
                "posterior_draws",
                "external_invocations",
                "law_summaries",
                "factor_builds",
                "program_compilations",
                "provider_bindings",
                "factor_evaluations",
                "integrations",
                "provider_calls",
            ):
                value = getattr(e.counts, name)
                if value:
                    work += (", " if work else "") + f"{name}={value}"
            suffix = f"  [{work}]" if work else ""
            lines.append(f"{e.stage.value:<{width}}  {e.status.text}{suffix}")
        head = (
            f"receipt {self.identity[:12]} ({self.capabilities.boundary}): "
            f"{len(self.reused)} reused, {len(self.recomputed)} recomputed, "
            f"{len(self.refused)} refused; total work {self.totals.total}"
        )
        return "\n".join([head, *lines])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: per-stage status, identity and counts, with the totals."""
        return {
            "identity": self.identity,
            "plan_identity": self.plan_identity,
            "boundary": self.capabilities.boundary,
            "loaded": self.loaded,
            "entries": [
                {
                    "stage": e.stage.value,
                    "status": e.status.text,
                    "identity": e.identity,
                    "counts": asdict(e.counts),
                }
                for e in self.entries
            ],
            "totals": asdict(self.totals),
            "previous": {stage.value: digest for stage, digest in self.previous.items()},
            "requested": {stage.value: digest for stage, digest in self.requested.items()},
        }

    def __repr__(self) -> str:
        return (
            f"<RecalcReceipt {self.identity[:12]} reused={len(self.reused)} "
            f"recomputed={len(self.recomputed)} refused={len(self.refused)} "
            f"loaded={self.loaded}>"
        )


# -- request and results --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Utility:
    """A net-benefit rule: ``benefit_per_unit * ate - cost``; treat when positive."""

    benefit_per_unit: float
    cost: float = 0.0


@dataclass(frozen=True, slots=True, eq=False)
class TargetWeights:
    """Row weights declaring a target population and the columns they depend on.

    ``depends_on`` must lie in the adjustment set for the route to be licensed.
    """

    weights: ArrayLike
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class RecalcRequest:
    """One cross-fitted AIPW average-effect request over raw columns and declared edges."""

    data: Mapping[str, ArrayLike]
    edges: Sequence[tuple[str, str]]
    treatment: str
    outcome: str
    utility: Utility
    target: TargetWeights | None = None


@dataclass(frozen=True, slots=True)
class Law:
    """The law of the requested quantity."""

    ate: float
    std_error: float | None


@dataclass(frozen=True, slots=True)
class Decision:
    """The decision under a utility."""

    net_benefit: float
    treat: bool


class RecalcResult(NamedTuple):
    """The result of one recalculation."""

    plan: RecalcPlan
    receipt: RecalcReceipt
    law: Law
    decision: Decision

    def explain(self) -> str:
        """The decision, the law behind it and what the recalculation reused."""
        verdict = "treat" if self.decision.treat else "do not treat"
        return (
            f"Decision: {verdict} (net benefit {self.decision.net_benefit:.4g}) on an average "
            f"effect of {self.law.ate:.4g} (standard error {self.law.std_error:.4g}). "
            f"{len(self.receipt.reused)} stage(s) reused, {len(self.receipt.recomputed)} "
            f"recomputed, {len(self.receipt.refused)} refused."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of the law, decision and receipt."""
        return {
            "law": {"ate": self.law.ate, "std_error": self.law.std_error},
            "decision": {"net_benefit": self.decision.net_benefit, "treat": self.decision.treat},
            "receipt": self.receipt.to_dict(),
        }


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


@dataclass(frozen=True, slots=True, eq=False)
class _Arguments:
    names: list[str]
    columns: list[NDArray[np.float64]]
    edges: list[tuple[str, str]]
    treatment: str
    outcome: str
    benefit_per_unit: float
    cost: float
    target_weights: NDArray[np.float64] | None
    target_depends_on: list[str] | None


def _arguments(request: RecalcRequest) -> _Arguments:
    names = [str(name) for name in request.data]
    columns = [_column(name, request.data[name]) for name in request.data]
    target = request.target
    return _Arguments(
        names=names,
        columns=columns,
        edges=[(str(a), str(b)) for a, b in request.edges],
        treatment=request.treatment,
        outcome=request.outcome,
        benefit_per_unit=float(request.utility.benefit_per_unit),
        cost=float(request.utility.cost),
        target_weights=None if target is None else _column("target weights", target.weights),
        target_depends_on=None if target is None else [str(v) for v in target.depends_on],
    )


# -- session --------------------------------------------------------------------------------


class RecalcSession:
    """Selective recalculation of the cross-fitted AIPW average effect.

    Holds the retained prepared study and the identities of the last successful run. Each
    :meth:`execute` plans the request against them, runs only what the plan says must run,
    counts the work and returns ``(plan, receipt, law, decision)``. A refused plan raises
    :class:`RecalcRefusal` before any work and leaves the session unchanged; a failure while
    running clears the session's identities so the next call recomputes every stage.

    A session built with :meth:`resume` models a fresh process: it knows the previous
    identities (what an ordinary loaded result keeps) but holds no live artifact, so it never
    reuses a derived stage. A missing data snapshot, fit or provider is a specific
    :class:`RecalcUnavailable`; supplied data resumes by recomputing, never by reuse.
    """

    __slots__ = ("_handle",)

    def __init__(self, *, retarget: Retarget = "licensed") -> None:
        self._handle = _native.RecalcSessionHandle(retarget)

    @classmethod
    def resume(
        cls,
        previous: Mapping[Stage, str] | RecalcReceipt,
        context: ResumeContext | None = None,
        *,
        retarget: Retarget = "licensed",
    ) -> RecalcSession:
        """A fresh-process session that knows ``previous`` and holds only what ``context`` names.

        ``previous`` is the declared identities of an earlier run (or a loaded
        :class:`RecalcReceipt`, whose requested workflow is what it kept).
        """
        declared = previous.requested if isinstance(previous, RecalcReceipt) else previous
        ctx = context if context is not None else ResumeContext()
        session = cls.__new__(cls)
        session._handle = _native.RecalcSessionHandle.resume(
            _declared_json(declared), json.dumps(ctx.to_wire()), retarget
        )
        return session

    def set_retarget_support(self, retarget: Retarget) -> None:
        """Declare the estimator's retarget license for later plans."""
        self._handle.set_retarget_support(retarget)

    def set_request_support(self, request: RequestKind, licensed_route: str | None = None) -> None:
        """Declare whether the next request is on the declared grid, off it or unsupported."""
        self._handle.set_request_support(request, licensed_route)

    @property
    def is_live(self) -> bool:
        """Whether the session holds a live prepared study."""
        return self._handle.is_live()

    @property
    def identities(self) -> dict[Stage, str]:
        """Own-input digests of the last successful run (or the resumed ones)."""
        return _declared(json.loads(self._handle.identities_json()))

    @property
    def capabilities(self) -> Capabilities:
        """The capabilities the next plan is made under."""
        return Capabilities.from_wire(json.loads(self._handle.capabilities_json()))

    def score_contrast(self) -> NDArray[np.float64] | None:
        """The frozen contrast scores ``phi_1 - phi_0`` of the live study, or ``None``.

        A weighted mean of these scores is the law, which is how a retargeted value is
        checked independently of the reweight that produced it.
        """
        scores = self._handle.score_contrast()
        return None if scores is None else np.asarray(scores, dtype=np.float64)

    def plan(
        self, request: RecalcRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        """The plan ``request`` would run under, without running it."""
        a = _arguments(request)
        wire = self._handle.plan(
            a.names,
            a.columns,
            a.edges,
            a.treatment,
            a.outcome,
            a.benefit_per_unit,
            a.cost,
            target_weights=a.target_weights,
            target_depends_on=a.target_depends_on,
            seed=_seed(seed),
            threads=threads,
        )
        return RecalcPlan.from_wire(json.loads(wire))

    def prepare(
        self, request: RecalcRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcPlan:
        """Name the plan ``request`` would run under; an alias of :meth:`plan`.

        Identification and the frozen score table are built by :meth:`execute` exactly when
        this plan says they must be, so there is no separate prepare-only run.
        """
        return self.plan(request, seed=seed, threads=threads)

    def execute(
        self, request: RecalcRequest, *, seed: int = 1, threads: int | None = None
    ) -> RecalcResult:
        """Plan, run only what the plan says must run, count the work and seal the receipt.

        Raises:
            RecalcRefusal: the plan holds a refused stage (an off-grid or unsupported request,
                an incompatible target, an unavailable dependency in a fresh process); nothing
                ran. :class:`RecalcUnavailable` names a missing fit, data snapshot or provider.
            RecalcNoLiveState: the plan reuses a derived stage this session cannot back with a
                live artifact (for example declared portable scores no loader supplies).
        """
        a = _arguments(request)
        result, artifact, refusal = self._handle.execute(
            a.names,
            a.columns,
            a.edges,
            a.treatment,
            a.outcome,
            a.benefit_per_unit,
            a.cost,
            target_weights=a.target_weights,
            target_depends_on=a.target_depends_on,
            seed=_seed(seed),
            threads=threads,
        )
        _raise(refusal)
        if result is None or artifact is None:  # pragma: no cover - one of the two is set
            raise CausalSerializationError("recalc execution returned no result")
        wire = json.loads(result)
        receipt = RecalcReceipt._from_wire(wire["receipt"], wire["plan"], artifact, loaded=False)
        return RecalcResult(
            plan=RecalcPlan.from_wire(wire["plan"]),
            receipt=receipt,
            law=Law(**wire["law"]),
            decision=Decision(**wire["decision"]),
        )


def consume(data: bytes, *, expected_identity: str | None = None) -> RecalcReceipt:
    """Consume a ``recalc_receipt_v1`` artifact; see :meth:`RecalcReceipt.consume`."""
    return RecalcReceipt.consume(data, expected_identity=expected_identity)


__all__ = [
    "ARTIFACT_KIND",
    "Capabilities",
    "Declaration",
    "Decision",
    "Law",
    "PlanEntry",
    "RecalcNoLiveState",
    "RecalcPlan",
    "RecalcReceipt",
    "RecalcReceiptRefusal",
    "RecalcRefusal",
    "RecalcRequest",
    "RecalcResult",
    "RecalcSession",
    "RecalcUnavailable",
    "ReceiptEntry",
    "RequestKind",
    "ResumeContext",
    "Retarget",
    "Stage",
    "StageCounts",
    "StageStatus",
    "TargetWeights",
    "Utility",
    "consume",
    "plan_recalculation",
    "stage_identity",
]
