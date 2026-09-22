"""Input coercion: the only module in the package allowed to accept union input types.

Every other public function takes one concrete type (a ``Mapping[str, NDArray]``,
a ``str``, …) and relies on the four functions declared here — ``coerce_data``,
``coerce_query``, ``coerce_refute``, ``coerce_latency`` — to normalize whatever a
caller passes (mapping, DataFrame, enum, string, bool, …) before it reaches
concrete-typed internals. Graph inputs are normalized by
``estimation._static_edges`` / ``_lagged_edges``.

``discovery_table`` is a narrower helper: the single owner of the panel /
multi-environment policy both ``coerce_data`` (a bare frame passed to
``Config.run()``) and ``accepted_graph.accept_discovery`` (the same frames, or a
bare sequence of per-unit tables, ahead of ``Config.accept()``) apply, so the two
spellings discover over identical data.

Wiring: ``coerce_data`` is used by the discovery config ``run()`` methods;
``coerce_refute`` / ``coerce_latency`` are used by ``estimation.py``
(``_resolve_latency_budget``, ``PreparedAnalysis.prepare``) and by
``_analyze.analyze`` itself. ``coerce_query`` is called once, at the top of
``_analyze.analyze``, as the single supported-query check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import EventFrame, MultiEnvFrame, PanelFrame
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError


def _pool_partitions(names: Sequence[str], partitions: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Row-concatenate matching-named columns across the units of a panel."""
    return {name: np.concatenate([part[i] for part in partitions]) for i, name in enumerate(names)}


def _refuse_environment_pooling() -> None:
    raise CausalUnsupportedError(
        "row-pooling environments into one table is refused: a distribution shift across "
        "environments makes any two variables whose means or scales shift dependent (a "
        "mixture), so a single-table algorithm would report spurious edges. Discover across "
        "environments with JPCMCIPlus (space dummies), or pass one environment's table",
        reason_code="data_modality_not_licensed",
    )


def _refuse_lagged_pooling() -> None:
    raise CausalUnsupportedError(
        "row-pooling units into one series is refused for this lagged algorithm: each unit "
        "boundary would give its first max_lag rows the previous unit's last observations "
        "as their lagged parents, biasing partial correlations, and it has no per-unit "
        "lagged design of its own (its regime labels run over one series' observations). "
        "PCMCI, PCMCIPlus, LPCMCI and DbnPosterior build every unit's lagged windows inside "
        "that unit; JPCMCIPlus does too, across environments. Or pass one unit's series",
        reason_code="data_modality_not_licensed",
    )


def coerce_data(
    value: Any, *, temporal: bool = False
) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Normalize tabular input to ``(names, float64 columns)``.

    Accepts, in order:

    - a ``(names, columns)`` pair (the packed form of the old two-positional-
      argument ``discover_*(names, columns)`` call shape);
    - an :class:`antecedent.data.EventFrame` (its ``names``/``columns`` fields
      taken directly — it has no ``to_numpy``, so ``as_columns`` alone would
      otherwise reject it as neither a mapping nor a DataFrame);
    - anything else :func:`antecedent._data.as_columns` already handles: a
      ``Mapping[str, array-like]``, a pandas DataFrame, or an equivalent
      frame-like object exposing ``columns`` + ``to_numpy``.

    A ``PanelFrame`` is stacked into one table of exchangeable unit rows for a
    static algorithm. ``temporal=True`` refuses it (an algorithm without a per-unit
    lagged design: :func:`coerce_temporal_data` serves the ones with it), and a
    ``MultiEnvFrame`` is refused either way: a pooled series would build lagged
    parents across unit boundaries, and pooled environments induce mixture
    dependence. :class:`PreparedAnalysis` dispatches those frames to
    ``prepare_panel`` / ``prepare_multi_env`` instead, which keep units and
    environments separate.
    """
    from ._data import as_columns, to_f64

    if isinstance(value, tuple) and len(value) == 2:
        names, columns = value
        return [str(n) for n in names], [to_f64(c) for c in columns]
    if isinstance(value, EventFrame):
        return list(value.names), [to_f64(c) for c in value.columns]
    if isinstance(value, PanelFrame):
        if temporal:
            _refuse_lagged_pooling()
        names = list(value.names)
        pooled = _pool_partitions(names, value.unit_columns)
        return names, [to_f64(pooled[n]) for n in names]
    if isinstance(value, MultiEnvFrame):
        _refuse_environment_pooling()
    return as_columns(value)


def coerce_temporal_data(
    value: Any,
) -> tuple[list[str], list[NDArray[np.float64]], list[int] | None]:
    """Normalize input for a lagged algorithm that builds a per-unit design.

    Returns ``(names, columns, unit_lengths)``. A :class:`antecedent.data.PanelFrame`
    yields its units' rows concatenated plus each unit's length, so the native
    algorithm builds every unit's lag windows inside that unit and pools the rows; any
    other input is one series (``unit_lengths is None``) under :func:`coerce_data`'s
    policy.
    """
    from ._data import to_f64

    if isinstance(value, PanelFrame):
        names = list(value.names)
        pooled = _pool_partitions(names, value.unit_columns)
        lengths = [len(unit[0]) for unit in value.unit_columns]
        return names, [to_f64(pooled[n]) for n in names], lengths
    names, columns = coerce_data(value, temporal=True)
    return names, columns, None


def discovery_table(value: Any, *, temporal: bool = False) -> Any:
    """One table for a single-table discovery config, under :func:`coerce_data`'s policy.

    A panel is stacked for a static algorithm; for a lagged one it stays a panel
    (:func:`coerce_temporal_data` builds the per-unit design from it). A
    multi-environment frame, or a bare sequence of per-environment tables, is
    refused. An event frame discovers on its recorded columns. Used ahead of
    :func:`accepted_graph.accept_discovery`'s single-table configs so that path
    and a caller-supplied ``.run(frame)`` see the identical table.
    """
    if isinstance(value, EventFrame):
        return dict(zip(value.names, value.columns, strict=True))
    if isinstance(value, PanelFrame):
        if temporal:
            return value
        names, columns = coerce_data(value)
        return dict(zip(names, columns, strict=True))
    if isinstance(value, MultiEnvFrame) or (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, Mapping))
        and not (isinstance(value, tuple) and len(value) == 2)
    ):
        _refuse_environment_pooling()
    return value


def coerce_query(value: Any) -> Any:
    """Validate a query input and return it unchanged.

    Every query dataclass in :mod:`antecedent.query` carries a ``kind``
    discriminator; anything without one is not a supported query type.
    """
    from .interference import InterferenceQuery
    from .query import (
        AnomalyAttribution,
        AverageDerivative,
        AverageEffect,
        ChangeAttribution,
        ConditionalEffect,
        Counterfactual,
        DirectionalDerivative,
        Elasticity,
        InterventionalDistribution,
        InterventionResponse,
        MediationEffect,
        PathSpecificEffect,
        PointDerivative,
        PulseEffect,
        ResponseCurve,
        ResponseJacobian,
        SemiElasticity,
        SustainedEffect,
        TemporalMediationEffect,
    )
    from .transport import Transport
    from .transport.advanced import TransportQuery

    valid = (
        AnomalyAttribution,
        AverageEffect,
        ChangeAttribution,
        ConditionalEffect,
        Counterfactual,
        InterventionalDistribution,
        MediationEffect,
        PathSpecificEffect,
        PulseEffect,
        SustainedEffect,
        TemporalMediationEffect,
        ResponseCurve,
        AverageDerivative,
        PointDerivative,
        Elasticity,
        SemiElasticity,
        DirectionalDerivative,
        ResponseJacobian,
        InterventionResponse,
        TransportQuery,
        Transport,
        InterferenceQuery,
    )
    if isinstance(value, valid):
        return value
    names = ", ".join(c.__name__ for c in valid)
    raise CausalTypeError(f"unsupported query type: {type(value)!r}; use one of {names}")


def coerce_refute(value: Any) -> str | bool:
    """Normalize a refute specification to a native-facing value.

    Accepts a ``Refute`` enum member (-> its wire string), a suite name
    string, or ``False`` (-> ``False``, meaning "no refutation").

    ``refute=True`` raises ``TypeError``. ``True`` carries no information
    about *which* suite to run; the code this consolidates
    (``estimation._resolve_latency_budget``) silently substitutes a
    mode-dependent default suite for it (``out_refute = mapped_refute if
    refute is True else refute``), which is undiscoverable from the call
    site. Pass ``refute="placebo"``, ``"cheap"``, ``"full"``, or a ``Refute``
    enum member instead.
    """
    from .ids import Refute

    if value is True:
        raise CausalTypeError(
            "refute=True is ambiguous: it does not say which refutation suite "
            'to run. Pass refute="placebo", "cheap", "full", or a Refute enum '
            "member instead (or refute=False for no refutation).",
            reason_code="invalid_argument",
        )
    if value is False:
        return False
    if isinstance(value, Refute):
        return str(value)
    if isinstance(value, str):
        return value
    raise CausalTypeError(
        f"unsupported refute type: {type(value)!r}; use a bool, a Refute enum "
        "member, or a suite name string"
    )


def coerce_latency(value: Any) -> str | None:
    """Normalize a latency specification to a native-facing string.

    Accepts a ``Latency`` enum member or a tier name string
    (``"interactive"`` / ``"standard"`` / ``"report"``) and returns the
    canonical lowercase tier string. ``None`` passes through unchanged — it
    means "no latency override" (``estimation._resolve_latency_budget`` skips
    tier mapping entirely in that case), which is not itself a tier.
    """
    from .ids import Latency

    if value is None:
        return None
    if isinstance(value, Latency):
        return str(value)
    if isinstance(value, str):
        key = value.strip().lower()
        valid = {str(m) for m in Latency}
        if key not in valid:
            raise CausalValueError(f"unknown latency={value!r}; use interactive|standard|report")
        return key
    raise CausalTypeError(
        f"unsupported latency type: {type(value)!r}; use a str or Latency enum member"
    )


__all__ = [
    "coerce_data",
    "coerce_latency",
    "coerce_query",
    "coerce_refute",
    "discovery_table",
]
