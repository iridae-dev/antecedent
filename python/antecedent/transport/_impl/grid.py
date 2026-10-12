from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ..._columns import coerce_fields
from ..._defaults import OMITTED
from ..._transport_results import TransportContrast, TransportGridPoint

if TYPE_CHECKING:
    from ...estimation import PreparedAnalysis


from .catalog import EvidenceCatalog
from .exact import ClassicalTransportIdentification, ExactTransportData
from .statistical import EmpiricalTable, LearnedCategorical, StatisticalTransportData


@dataclass(frozen=True, slots=True)
class TransportResponseGridQuery:
    """Finite mean-response request; unsupported coordinates remain in the result."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Sequence[Mapping[str, float]]
    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]
    seed: int = 1
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin"

    def __post_init__(self) -> None:
        coerce_fields(self, at=tuple(MappingProxyType(dict(point)) for point in self.at))


@dataclass(frozen=True, slots=True)
class TransportResponseGrid:
    """Native-authorized response family with explicit point-local failures."""

    points: tuple[TransportGridPoint, ...]
    execution_id: str
    _execution: Any = field(repr=False, compare=False)

    def mean(self, point: int, outcome: str) -> float:
        row = self.points[point]
        if row["status"] != "available":
            raise ValueError(f"Grid point {point} is unavailable: {row['detail']}")
        return float(row["means"][outcome])

    def contrast(self, left: int, right: int, outcome: str) -> TransportContrast:
        """Transform two points using native paired draws and retained parent identity."""
        import json

        result = dict(json.loads(self._execution.contrast(left, right, outcome)))
        if result["interval"] is not None:
            result["interval"] = tuple(result["interval"])
        return TransportContrast(result)

    def scalar_projection(self, point: int, outcome: str) -> dict[str, Any]:
        """Return a scalar and loss receipt; this projection cannot impersonate the full grid."""
        import json

        return dict(json.loads(self._execution.scalar_projection(point, outcome)))

    def inspect(self) -> Any:
        import json

        from ...results._report import InspectionReport

        return InspectionReport(**json.loads(self._execution.inspection_json()))

    def export(self) -> bytes:
        return bytes(self._execution.export())


def _response_grid(native: Any, payload: str) -> TransportResponseGrid:
    import json

    parsed = json.loads(payload)
    return TransportResponseGrid(
        tuple(TransportGridPoint(point) for point in parsed["points"]),
        parsed["execution_id"],
        native.freeze(),
    )


def prepare_response_grid(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData | StatisticalTransportData,
    *,
    at: Sequence[Mapping[str, float]],
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin",
    bootstrap: int = OMITTED["transport_bootstrap"],
    coverage_level: float = OMITTED["transport_coverage_level"],
    seed: int = 1,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[TransportResponseGrid]:
    """Prepare exact or empirical finite responses on the common study lifecycle."""
    from ... import _native
    from ...estimation import PreparedAnalysis, _Controls

    if not isinstance(data, (ExactTransportData, StatisticalTransportData)):
        raise TypeError("Grid data must be exact laws or explicit statistical providers")
    native = _native.prepare_transport_grid(
        identification._native,
        catalog,
        data,
        [dict(point) for point in at],
        statistical=isinstance(data, StatisticalTransportData),
        estimator=estimator,
        bootstrap=bootstrap,
        coverage_level=coverage_level,
        seed=seed,
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="transport_grid", controls=_Controls(cancel=cancel))


def consume_response_grid(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[TransportResponseGrid]:
    """Verify a grid without fetching, fitting, or rebuilding raw samples."""
    from ... import _native
    from ...estimation import PreparedAnalysis, _Controls

    native = _native.consume_transport_grid(
        artifact,
        max_operations=max_operations,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="transport_grid", controls=_Controls(cancel=cancel))
