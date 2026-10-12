from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, overload

from ...errors import CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from ...estimation import PreparedAnalysis


from .catalog import EvidenceCatalog
from .exact import (
    ClassicalTransportIdentification,
    ExactTransportData,
    ExactTransportDistribution,
    ExactTransportQuery,
    prepare_exact,
)
from .grid import TransportResponseGrid, TransportResponseGridQuery, prepare_response_grid
from .statistical import (
    EmpiricalTable,
    LearnedCategorical,
    StatisticalTransportData,
    StatisticalTransportDistribution,
    StatisticalTransportQuery,
    prepare_statistical,
)
from .trial import (
    LearnedTrialEstimate,
    TransportControls,
    TransportInference,
    TrialAipw,
    TrialAipwData,
    TrialAipwQuery,
    prepare_trial,
)


@overload
def prepare(
    request: TrialAipwQuery,
    catalog: TrialAipwData,
    *,
    provider: TrialAipw | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[LearnedTrialEstimate]: ...


@overload
def prepare(
    request: ExactTransportQuery,
    catalog: ExactTransportData,
    *,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[ExactTransportDistribution]: ...


@overload
def prepare(
    request: StatisticalTransportQuery,
    catalog: StatisticalTransportData,
    *,
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]: ...


@overload
def prepare(
    request: TransportResponseGridQuery,
    catalog: ExactTransportData | StatisticalTransportData,
    *,
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[TransportResponseGrid]: ...


@overload
def prepare(
    request: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    controls: TransportControls | None = None,
) -> PreparedAnalysis[ExactTransportDistribution]: ...


@overload
def prepare(
    request: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Mapping[str, float],
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]: ...


def prepare(
    request: ClassicalTransportIdentification
    | ExactTransportQuery
    | StatisticalTransportQuery
    | TransportResponseGridQuery
    | TrialAipwQuery,
    catalog: EvidenceCatalog
    | ExactTransportData
    | StatisticalTransportData
    | TrialAipwData
    | None = None,
    data: ExactTransportData | StatisticalTransportData | TrialAipwData | None = None,
    *,
    at: Mapping[str, float] | None = None,
    provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[
    ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
    | LearnedTrialEstimate
]:
    """Prepare a typed scalar, grid, or binary trial request on one lifecycle.

    Pass ``prepare(request, data, provider=..., inference=..., controls=...)``.
    The legacy ``prepare(identification, catalog, data, at=...)`` form remains.
    """
    provider_requested = provider is not None
    inference_requested = inference is not None
    controls = controls or TransportControls()
    if data is None and isinstance(
        catalog, (ExactTransportData, StatisticalTransportData, TrialAipwData)
    ):
        data, catalog = catalog, None
    if isinstance(request, TrialAipwQuery):
        if not isinstance(data, TrialAipwData) or (
            provider is not None and not isinstance(provider, TrialAipw)
        ):
            raise CausalTypeError("TrialAipwQuery requires TrialAipwData and a TrialAipw provider")
        if catalog is not None or at is not None:
            raise CausalValueError(
                "The binary trial request already declares its populations and contrast"
            )
        return prepare_trial(
            request,
            data,
            provider=provider or TrialAipw(),
            inference=inference or TransportInference(),
            controls=controls,
        )
    if isinstance(
        request, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
    ):
        if catalog is not None or at is not None:
            raise CausalValueError("The typed request already owns its catalog and coordinates")
        catalog = request.catalog
        if isinstance(request, (StatisticalTransportQuery, TransportResponseGridQuery)):
            inference = inference or TransportInference(
                request.bootstrap, request.coverage_level, request.seed
            )
            provider = provider or (
                EmpiricalTable() if request.estimator == "plugin" else request.estimator
            )
        if isinstance(request, TransportResponseGridQuery):
            if isinstance(data, ExactTransportData) and (
                provider_requested
                or inference_requested
                or not (
                    isinstance(request.estimator, EmpiricalTable) or request.estimator == "plugin"
                )
            ):
                raise CausalValueError(
                    "Exact grid laws do not accept learner or sampling inference settings"
                )
            if not isinstance(data, (ExactTransportData, StatisticalTransportData)) or isinstance(
                provider, TrialAipw
            ):
                raise CausalTypeError(
                    "Grid requests require exact or categorical-law data/providers"
                )
            settings = inference or TransportInference()
            return prepare_response_grid(
                request.identification,
                catalog,
                data,
                at=request.at,
                estimator=provider or EmpiricalTable(),
                bootstrap=settings.bootstrap,
                coverage_level=settings.coverage_level,
                seed=settings.seed,
                max_operations=controls.max_operations,
                max_depth=controls.max_depth,
                max_support_rows=controls.max_support_rows,
                memory_bytes=controls.memory_bytes,
                cancel=controls.cancel,
            )
        at = request.at
        request = request.identification
    if not isinstance(catalog, EvidenceCatalog) or at is None:
        raise CausalTypeError("Scalar transport requires a catalog and intervention coordinates")
    limits = dict(
        max_operations=controls.max_operations,
        max_depth=controls.max_depth,
        max_support_rows=controls.max_support_rows,
        memory_bytes=controls.memory_bytes,
        cancel=controls.cancel,
    )
    if isinstance(data, ExactTransportData):
        if provider is not None or inference is not None:
            raise CausalValueError(
                "Exact laws do not accept learner or sampling inference settings"
            )
        return prepare_exact(request, catalog, data, at=at, **limits)
    if not isinstance(data, StatisticalTransportData) or isinstance(provider, TrialAipw):
        raise CausalTypeError("Statistical transport requires categorical-law data/providers")
    settings = inference or TransportInference()
    return prepare_statistical(
        request,
        catalog,
        data,
        at=at,
        estimator=provider or EmpiricalTable(),
        bootstrap=settings.bootstrap,
        coverage_level=settings.coverage_level,
        seed=settings.seed,
        **limits,
    )
