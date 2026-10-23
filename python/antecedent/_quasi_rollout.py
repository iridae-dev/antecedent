"""Design binding and result mapping of the heterogeneity-robust rollout family.

The subject, cluster, period and cohort (or treated and post) columns stay frozen query
metadata; only the outcome and the baseline covariates reach the numeric ingest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._data import ingest_columns, read_columns
from .errors import CausalValueError
from .quasi import (
    CovariateAdjustedDiD,
    DesignEffect,
    QuasiRolloutEstimate,
    StaggeredRollout,
    _binary,
    _integer_column,
)


def _ids(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def staggered_rollout_payload(
    data: Any, query: StaggeredRollout
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Numeric outcome and covariates; subjects, clusters, periods and cohorts as metadata."""
    required: tuple[str, ...] = (
        query.outcome,
        query.subject,
        query.period,
        query.cohort,
        *query.covariates,
    )
    if query.cluster is not None:
        required += (query.cluster,)
    raw = read_columns(data, required, what="staggered rollout")
    subjects = _ids(raw[query.subject])
    clusters = subjects if query.cluster is None else _ids(raw[query.cluster])
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "periods": tuple(_integer_column(raw[query.period], "period", minimum=1)),
        "cohorts": tuple(_integer_column(raw[query.cohort], "cohort", minimum=0)),
    }
    numeric = (query.outcome, *query.covariates)
    names, columns = ingest_columns({name: raw[name] for name in numeric})
    return names, columns, design


def covariate_adjusted_payload(
    data: Any, query: CovariateAdjustedDiD
) -> tuple[list[str], list[Any], dict[str, Any]]:
    """Numeric outcome and covariates; subjects, clusters, treated and post as metadata."""
    required: tuple[str, ...] = (
        query.outcome,
        query.subject,
        query.treated,
        query.post,
        *query.covariates,
    )
    if query.cluster is not None:
        required += (query.cluster,)
    raw = read_columns(data, required, what="covariate-adjusted DiD")
    subjects = _ids(raw[query.subject])
    clusters = subjects if query.cluster is None else _ids(raw[query.cluster])
    if any(not value.strip() for value in (*subjects, *clusters)):
        raise CausalValueError("subject and cluster IDs must be non-empty")
    design = {
        "subjects": subjects,
        "clusters": clusters,
        "treated": tuple(_binary(raw[query.treated], query.treated)),
        "post": tuple(_binary(raw[query.post], query.post)),
    }
    numeric = (query.outcome, *query.covariates)
    names, columns = ingest_columns({name: raw[name] for name in numeric})
    return names, columns, design


def _pair(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


def quasi_rollout_from_wire(wire: Any) -> QuasiRolloutEstimate | None:
    """A loaded artifact's ``quasi_rollout`` body section as the live frozen result."""
    if not isinstance(wire, Mapping):
        return None

    def effect(row: Mapping[str, Any]) -> DesignEffect:
        optional_int = {
            key: None if row.get(key) is None else int(row[key])
            for key in ("group", "period", "event_time")
        }
        return DesignEffect(
            kind=str(row["kind"]),
            estimate=float(row["estimate"]),
            standard_error=None
            if row.get("standard_error") is None
            else float(row["standard_error"]),
            interval=_pair(row.get("interval")),
            band=_pair(row.get("band")),
            unit=None if row.get("unit") is None else str(row["unit"]),
            treated=int(row.get("treated", 0)),
            comparison=int(row.get("comparison", 0)),
            **optional_int,
        )

    critical = wire.get("critical_value")
    return QuasiRolloutEstimate(
        design=str(wire["design"]),
        method=str(wire["method"]),
        effects=tuple(effect(row) for row in wire.get("effects") or ()),
        uncertainty=str(wire["uncertainty"]),
        replicates_requested=int(wire.get("replicates_requested", 0)),
        replicates_ok=int(wire.get("replicates_ok", 0)),
        replicates_failed=int(wire.get("replicates_failed", 0)),
        critical_value=None if critical is None else float(critical),
        diagnostics=tuple(
            (str(name), float(value)) for name, value in wire.get("diagnostics") or ()
        ),
        calibration=str(wire.get("calibration", "unmeasured")),
        licensed=bool(wire.get("licensed", False)),
    )


def quasi_rollout_from_raw(raw: Any) -> QuasiRolloutEstimate | None:
    """The native ``quasi_rollout`` section as a frozen result, or ``None``."""
    section = getattr(raw, "quasi_rollout", None)
    if section is None:
        return None
    effects = tuple(
        DesignEffect(
            kind=str(effect.kind),
            estimate=float(effect.estimate),
            standard_error=effect.standard_error,
            interval=_pair(effect.interval),
            band=_pair(effect.band),
            group=effect.group,
            period=effect.period,
            event_time=effect.event_time,
            unit=effect.unit,
            treated=int(effect.treated),
            comparison=int(effect.comparison),
        )
        for effect in section.effects
    )
    return QuasiRolloutEstimate(
        design=str(section.design),
        method=str(section.method),
        effects=effects,
        uncertainty=str(section.uncertainty),
        replicates_requested=int(section.replicates_requested),
        replicates_ok=int(section.replicates_ok),
        replicates_failed=int(section.replicates_failed),
        critical_value=section.critical_value,
        diagnostics=tuple((str(name), float(value)) for name, value in section.diagnostics),
        calibration=str(section.calibration),
        licensed=bool(section.licensed),
    )
