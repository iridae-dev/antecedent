"""Presentation and immutable read models; no scientific decisions are made here."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from ._freeze import freeze, thaw

if TYPE_CHECKING:
    from .joint_distribution import ScientificQuantity
    from .results.sources import MeanSource


class ScientificResult:
    """Shared conventions for specialized scientific result objects."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Dataclass decorators otherwise replace the inherited notebook summary.
        if "__repr__" not in cls.__dict__:
            setattr(cls, "__repr__", ScientificResult.__repr__)  # noqa: B010 — dataclass hook

    def __post_init__(self) -> None:
        from ._artifact_protocol import ExportedArtifact

        for field in fields(cast(Any, self)):
            if field.name not in ("_session", "_handle", "_native"):
                value = getattr(self, field.name)
                # An exported artifact is already immutable: retaining the very object keeps
                # its identity (and its producer-side expectation) through re-export.
                if not isinstance(value, ExportedArtifact):
                    object.__setattr__(self, field.name, freeze(value))

    def to_dict(self) -> dict[str, Any]:
        if is_dataclass(self):
            return thaw(self)
        report = getattr(self, "report", getattr(self, "raw", None))
        if report is not None:
            return report.to_dict() if hasattr(report, "to_dict") else thaw(report)
        return {name: thaw(value) for name, value in vars(self).items() if not name.startswith("_")}

    def _summary_fields(self) -> dict[str, Any]:
        """Read a bounded number of scalar fields; never serialize nested payloads."""
        names = (
            "kind",
            "method",
            "procedure",
            "label",
            "claim",
            "claim_label",
            "standing",
            "status",
            "point",
            "value",
            "estimate",
            "mean",
        )
        if is_dataclass(self):
            declared = {field.name for field in fields(cast(Any, self))}
            values = {name: getattr(self, name) for name in names if name in declared}
        else:
            state = vars(self)
            retained = state.get("_report")
            if not isinstance(retained, dict):
                from collections.abc import Mapping

                retained = retained if isinstance(retained, Mapping) else {}
            values = {
                name: state.get(name, retained.get(name))
                for name in names
                if name in state or name in retained
            }
        return {
            name: value
            for name, value in values.items()
            if type(value) in (str, int, float, bool, type(None))
        }

    def inspect(self) -> Any:
        from .results._report import InspectionReport, InspectionStage, SlotModel
        from .results.sources import is_mean_producer

        # Inspection reads retained presentation metadata. Full payload conversion is
        # explicit through to_dict(), and portable validation happens at export/load.
        family = type(self).__name__
        stage = getattr(self, "_inspection_stage", None)
        if stage is None:
            stage = {
                "ArtifactInspection": "exported",
                "ArtifactExpectation": "declared",
                "ArtifactAcceptance": "consumed",
            }.get(family)
        if stage is None:
            stage = (
                "consumed"
                if family.startswith("Consumed")
                else "prepared"
                if family.startswith("Prepared")
                else "exported"
                if family.startswith("Exported")
                else "declared"
                if family.endswith(("Law", "Contract", "Quantity", "Source"))
                else "executed"
            )
        data = self._summary_fields()
        unavailable = SlotModel(
            available=False, reason="not_established_by_this_object", summary="unavailable"
        )
        return InspectionReport(
            identification=unavailable,
            support=unavailable,
            uncertainty=SlotModel(
                available=False,
                reason="no_interval_claim",
                summary=str(data.get("claim", data.get("claim_label", "unavailable"))),
            ),
            assumptions=unavailable,
            family=family,
            details=data,
            stage=cast(InspectionStage, stage),
            answer=getattr(self, "answer", None),
            operations=("inspect", "to_dict")
            + (
                ("export",)
                if type(self).export is not ScientificResult.export
                or getattr(self, "_artifact_bytes", None) is not None
                else ()
            )
            + (("as_mean_source",) if is_mean_producer(self) else ()),
        )

    def to_rows(self) -> list[dict[str, Any]]:
        data = self.to_dict()
        payload = data.get("payload")
        records = payload if isinstance(payload, dict) else data
        labels: dict[str, Any] = {
            key: data[key] for key in ("claim", "label", "method", "paradigm") if key in data
        }
        for key in ("arms", "looks", "predictors", "differences", "sources", "results", "rows"):
            if isinstance(records.get(key), list):
                return [
                    {**labels, "record": key, **row}
                    for row in records[key]
                    if isinstance(row, dict)
                ]
        values = data.get("values")
        if isinstance(values, dict):
            return [
                {"name": key, "value": value, "label": data.get("label")}
                for key, value in values.items()
            ]
        return [data]

    def __repr__(self) -> str:
        # reprlib bounds each scalar string before building the display; do not
        # render an entire posterior/array only to truncate its final string.
        from reprlib import Repr

        renderer = Repr()
        renderer.maxstring = 96
        renderer.maxdict = 8
        return f"{type(self).__name__}({renderer.repr(self._summary_fields())})"

    def _repr_html_(self) -> str:
        return self.inspect()._repr_html_()

    def export(self, *args: Any, **kwargs: Any) -> Any:
        """Forward a consumed artifact unchanged, without restoring a live execution."""
        from .errors import CausalUnsupportedError

        data = getattr(self, "_artifact_bytes", None)
        if data is None:
            raise CausalUnsupportedError(
                "this object has no portable artifact", reason_code="route_not_supported"
            )
        from ._artifact_protocol import artifact_from_receipt

        return artifact_from_receipt(self, data)

    def as_mean_source(
        self,
        *,
        quantity: ScientificQuantity | None = None,
        field: str | None = None,
        contrast: Any = None,
    ) -> MeanSource:
        from .results.sources import adapt_mean_source

        return adapt_mean_source(self, quantity=quantity, field=field, contrast=contrast)
