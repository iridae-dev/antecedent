from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...errors import CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from ...estimation import _PreparedEngine


from .catalog import EvidenceCatalog
from .exact import (
    ExactTransportData,
    ExactTransportDistribution,
    ExactTransportQuery,
    _prepare_exact,
)
from .grid import TransportResponseGrid, TransportResponseGridQuery, _prepare_response_grid
from .statistical import (
    EmpiricalTable,
    LearnedCategorical,
    StatisticalTransportData,
    StatisticalTransportDistribution,
    StatisticalTransportQuery,
    _prepare_statistical,
)
from .trial import (
    LearnedTrialEstimate,
    TransportInference,
    TrialAipw,
    TrialAipwData,
    TrialAipwQuery,
    _TransportLimits,
    prepare_trial,
)


def compile_modality(
    request: ExactTransportQuery
    | StatisticalTransportQuery
    | TransportResponseGridQuery
    | TrialAipwQuery,
    data: ExactTransportData | StatisticalTransportData | TrialAipwData,
    *,
    provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None = None,
    inference: TransportInference | None = None,
    controls: _TransportLimits | None = None,
    seed: int = 1,
) -> _PreparedEngine[
    ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
    | LearnedTrialEstimate
]:
    """Compile one typed scalar, grid or binary-trial question into its native study.

    Private lowering for ``antecedent.prepare``: the public lifecycle owns
    execution, so this takes the already-resolved physical limits.
    """
    provider_requested = provider is not None
    inference_requested = inference is not None
    if controls is None:
        if isinstance(
            request, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
        ):
            from .._day1 import _evaluation_limits

            controls = _TransportLimits(**_evaluation_limits(request.evaluation))
        else:
            controls = _TransportLimits()
    catalog: EvidenceCatalog | None = None
    at: Mapping[str, float] | None = None
    if isinstance(request, TrialAipwQuery):
        if not isinstance(data, TrialAipwData) or (
            provider is not None and not isinstance(provider, TrialAipw)
        ):
            raise CausalTypeError("TrialAipwQuery requires TrialAipwData and a TrialAipw provider")
        return prepare_trial(
            request,
            data,
            provider=provider or TrialAipw(),
            inference=inference or TransportInference(),
            controls=controls,
            seed=seed,
        )
    if isinstance(
        request, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
    ):
        catalog = request.catalog
        if isinstance(request, (StatisticalTransportQuery, TransportResponseGridQuery)):
            inference = inference or TransportInference(request.bootstrap, request.coverage_level)
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
            return _prepare_response_grid(
                request.identification,
                catalog,
                data,
                at=request.at,
                estimator=provider or EmpiricalTable(),
                bootstrap=settings.bootstrap,
                coverage_level=settings.coverage_level,
                seed=seed,
                max_operations=controls.max_operations,
                max_depth=controls.max_depth,
                max_support_rows=controls.max_support_rows,
                memory_bytes=controls.memory_bytes,
                cancel=controls.cancel,
            )
        at = request.at
        identification = request.identification
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
        return _prepare_exact(identification, catalog, data, at=at, **limits)
    if not isinstance(data, StatisticalTransportData) or isinstance(provider, TrialAipw):
        raise CausalTypeError("Statistical transport requires categorical-law data/providers")
    settings = inference or TransportInference()
    return _prepare_statistical(
        identification,
        catalog,
        data,
        at=at,
        estimator=provider or EmpiricalTable(),
        bootstrap=settings.bootstrap,
        coverage_level=settings.coverage_level,
        seed=seed,
        **limits,
    )
