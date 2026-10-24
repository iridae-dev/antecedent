"""One view over every layer of a compiled route.

:meth:`antecedent.estimation.PreparedAnalysis.plan_view` returns a
:class:`ProgramPlanView`. It is read from the lowered program the prepared handle
executes, before any estimate: building it does not identify, fit, scan rows,
enumerate graphs or simulate. Each layer is a typed attribute; ``to_dict()`` is
the JSON-safe whole. Variables are reported by name.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import PrivateAttr

from ._report import ResultModel

__all__ = [
    "PlanEvidenceLayer",
    "PlanIdentificationLayer",
    "PlanMixtureView",
    "PlanPhysicalLayer",
    "PlanScientificLayer",
    "PlanStatisticalLayer",
    "PlanSupportView",
    "PlanUncertaintyDemand",
    "ProgramPlanView",
]


class PlanScientificLayer(ResultModel):
    """Layer one: the scientific problem."""

    target: Mapping[str, Any]
    plan_id: str
    data: str
    graph_class: str
    query_variables: Sequence[str]
    source_digest: str | None = None


class PlanIdentificationLayer(ResultModel):
    """Layer two: how identification is carried and what it produced.

    ``kind`` is ``retained`` (identified at compile), ``deferred`` (refused at
    compile, with ``reason``), ``per_execution`` (with ``reason``),
    ``non_graphical`` (a feedback model) or ``not_claimed`` (a noncausal association, which
    claims no identification; ``refusal`` is the reason code of the refused causal query it
    follows). ``retained`` lists each product with its status, estimands (method and
    adjustment set), assumptions and masses.
    """

    kind: str
    status: str | None = None
    reason: str | None = None
    refusal: str | None = None
    non_graphical: str | None = None
    product: Mapping[str, Any] | None = None
    retained: Sequence[Mapping[str, Any]] = ()
    installed_at_prepare: bool = False

    @property
    def unresolved_obligations(self) -> list[Mapping[str, Any]]:
        """Obligations the checked identification product left unresolved."""
        if self.product is None:
            return []
        return list(self.product.get("unresolved_obligations", []))


class PlanStatisticalLayer(ResultModel):
    """Layer three: identifier, estimator, realization, models, folds, resampling, posterior."""

    identity: str
    identifier: str | None = None
    estimator: str | None = None
    realization: str
    models: Sequence[Mapping[str, Any]] = ()
    folds: Mapping[str, Any] | None = None
    resampling: Mapping[str, Any] | None = None
    posterior: Mapping[str, Any] | None = None
    design: Mapping[str, Any]


class PlanEvidenceLayer(ResultModel):
    """Layer four: sources, composition, and the decision or planning request."""

    identity: str
    sources: Sequence[Mapping[str, Any]] = ()
    composition: Mapping[str, Any]
    decision: Mapping[str, Any] | None = None
    planning: Mapping[str, Any] | None = None


class PlanPhysicalLayer(ResultModel):
    """Layer five: operators, the quoted design or why there is none, and byte estimates."""

    operators: Sequence[Mapping[str, Any]] = ()
    design: Mapping[str, Any]
    scratch_bytes: int | None = None
    retained_bytes: int | None = None
    output_bytes: int | None = None
    record: Mapping[str, Any] | None = None
    kernel_policy: Mapping[str, Any] | None = None

    @property
    def quoted_width(self) -> int | None:
        """Encoded design columns, when the route quotes a design."""
        return self.design.get("columns") if self.design.get("source") == "quoted" else None

    @property
    def not_design_based(self) -> str | None:
        """Why the route fits no single design, when it does not."""
        return self.design.get("reason")


class PlanUncertaintyDemand(ResultModel):
    """The uncertainty reading and claim the execution must publish."""

    reading: str
    claim_kind: str | None = None
    realization: str
    resampling: Mapping[str, Any] | None = None
    posterior: Mapping[str, Any] | None = None
    decision_request: str | None = None


class PlanSupportView(ResultModel):
    """Support-matrix cell and licensing status."""

    coordinate: str | None = None
    licensed: bool = False
    status: str | None = None


class PlanMixtureView(ResultModel):
    """The single logical plan of a graph-posterior analysis."""

    atom_kind: str | None = None
    atoms: Sequence[Mapping[str, Any]] = ()
    total_weight: float = 0.0
    identified_mass: float | None = None
    unidentified_mass: float | None = None
    composition: Mapping[str, Any]
    identification_retained: bool = False


class ProgramPlanView(ResultModel):
    """Every layer of the program a prepared handle executes."""

    route: str
    family: str
    stage: str
    identity: str
    depth: str
    cyclic_family: str | None = None
    planned_design: str | None = None
    scientific: PlanScientificLayer
    identification: PlanIdentificationLayer
    statistical: PlanStatisticalLayer
    evidence: PlanEvidenceLayer
    physical: PlanPhysicalLayer
    uncertainty: PlanUncertaintyDemand
    support: PlanSupportView
    mixture: PlanMixtureView | None = None
    _operation: dict[str, Any] | None = PrivateAttr(default=None)

    @property
    def operation(self) -> dict[str, Any] | None:
        """The sealed operation preparation retained: ``kind`` and its inspection record.

        A fresh plain ``dict`` with ``list`` sequences on every read, as the
        per-route inspection records it replaces returned.
        """
        return _plain(self._operation)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["operation"] = self.operation
        return payload

    @classmethod
    def from_json(cls, raw: str) -> ProgramPlanView:
        """Decode the native view, naming variables by the handle's columns."""
        payload = json.loads(raw)
        names: list[str] = list(payload.pop("names", []))
        detail = payload.pop("operation_detail", None)

        def name(index: Any) -> Any:
            if isinstance(index, int) and 0 <= index < len(names):
                return names[index]
            return index

        def named(values: Any) -> list[Any]:
            return [name(value) for value in values or []]

        scientific = dict(payload["scientific"])
        scientific["query_variables"] = named(scientific.get("query_variables"))
        target = dict(scientific.get("target") or {})
        if "outcome" in target:
            target["outcome"] = name(target["outcome"])
        scientific["target"] = target

        def estimand(raw_estimand: Mapping[str, Any]) -> dict[str, Any]:
            value = dict(raw_estimand)
            for key in ("adjustment_set", "instruments", "mediators"):
                value[key] = named(value.get(key))
            return value

        identification = dict(payload["identification"])
        retained = []
        for product in identification.get("retained") or []:
            product = dict(product)
            product["estimands"] = [estimand(item) for item in product.get("estimands") or []]
            retained.append(product)
        identification["retained"] = retained
        if identification.get("product") is not None:
            product = dict(identification["product"])
            product["estimands"] = [estimand(item) for item in product.get("estimands") or []]
            identification["product"] = product

        statistical = dict(payload["statistical"])
        models = []
        for model in statistical.get("models") or []:
            model = dict(model)
            model["inputs"] = named(model.get("inputs"))
            model["output"] = name(model.get("output"))
            models.append(model)
        statistical["models"] = models

        mixture = payload.get("mixture")
        if mixture is not None:
            mixture = dict(mixture)
            atoms = []
            for atom in mixture.get("atoms") or []:
                atom = dict(atom)
                if atom.get("estimand") is not None:
                    atom["estimand"] = estimand(atom["estimand"])
                atoms.append(atom)
            mixture["atoms"] = atoms

        payload.update(
            scientific=PlanScientificLayer(**scientific),
            identification=PlanIdentificationLayer(**identification),
            statistical=PlanStatisticalLayer(**statistical),
            evidence=PlanEvidenceLayer(**payload["evidence"]),
            physical=PlanPhysicalLayer(**payload["physical"]),
            uncertainty=PlanUncertaintyDemand(**payload["uncertainty"]),
            support=PlanSupportView(**payload["support"]),
            mixture=None if mixture is None else PlanMixtureView(**mixture),
        )
        payload.pop("operation", None)
        view = cls(**payload)
        view._operation = _plain(detail)
        return view

    def __repr__(self) -> str:
        return (
            f"<ProgramPlanView route={self.route!r} stage={self.stage!r} "
            f"identification={self.identification.kind!r} identity={self.identity[:12]!r}>"
        )


def _plain(value: Any) -> Any:
    """``value`` with every mapping a ``dict`` and every sequence a ``list``."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
