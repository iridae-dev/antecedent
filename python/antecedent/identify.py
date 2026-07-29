"""The staged identification surface: ``identify()`` → ``estimate()`` → ``validate()``.

:func:`identify` runs identification only (no estimate) and returns an
:class:`Identification` — enough state to continue into ``.estimate()`` /
``.validate()`` without recomputing identification from scratch. This is the
single class that replaces both ``estimation.IdentifyResult`` (identify-only
shape) and ``results.IdentificationView`` (the identification section of a
full :class:`antecedent.AnalysisResult`) — see :meth:`Identification.from_view`
and :meth:`Identification.to_identify_result` for the two conversions that
keep those existing shapes reachable rather than deleting them outright.

This module imports ``estimation``; ``estimation`` must never import this
module (that would be a cycle). Because of that direction, ``estimation.identify()``
keeps its own native-calling implementation unchanged — :func:`identify` here
wraps it rather than the reverse, so there remains exactly one code path that
calls the native identify-only entry point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .estimation import IdentifyResult
from .estimation import identify as _identify_native
from .graph import Dag
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, Frequentist
from .query import AverageEffect
from .results import IdentificationView


@dataclass(frozen=True)
class Identification:
    """A resolved identification strategy, staged for ``.estimate()`` / ``.validate()``.

    Produced by :func:`identify` (identify-only — ``assumption_count`` /
    ``derivation_step_count`` are unset, ``None``, since the identify-only
    native call doesn't compute them) or by :meth:`from_view` (from a
    completed analysis's identification section, where those counts are
    known). Conceptually immutable, like :class:`antecedent.AcceptedGraph`.
    """

    status: str
    method: str
    adjustment_set: list[str]
    graph: Dag | Sequence[tuple[str, str]]
    query: AverageEffect
    names: list[str] | None = None
    identifier: str | None = None
    assumption_count: int | None = None
    derivation_step_count: int | None = None

    def __bool__(self) -> bool:
        """``True`` when the estimand is identified.

        Delegates to :class:`antecedent.results.IdentificationView`'s status
        heuristic — the one place in this codebase that already makes this
        judgment call across the native status vocabulary
        (``NonparametricallyIdentified`` / ``PartiallyIdentified`` /
        ``NotIdentified`` / ``GraphDependent`` / the GCM path's
        ``"gcm.parametric"``) — rather than re-deriving it here.
        """
        return bool(
            IdentificationView(
                status=self.status,
                method=self.method,
                adjustment_set=list(self.adjustment_set),
                assumption_count=self.assumption_count or 0,
                derivation_step_count=self.derivation_step_count or 0,
            )
        )

    def to_identify_result(self) -> IdentifyResult:
        """Convert down to the legacy identify-only result shape.

        ``IdentifyResult`` stays exported for callers that only want that
        narrower shape; this keeps it a projection of ``Identification``
        rather than a second, independently-computed value.
        """
        return IdentifyResult(
            status=self.status,
            method=self.method,
            adjustment_set=list(self.adjustment_set),
        )

    @classmethod
    def from_view(
        cls,
        view: IdentificationView,
        *,
        graph: Dag | Sequence[tuple[str, str]],
        query: AverageEffect,
        names: Sequence[str] | None = None,
        identifier: str | None = None,
    ) -> Identification:
        """Build from a completed analysis's identification section.

        Unlike :func:`identify` alone, a view sourced from a full estimate
        carries ``assumption_count`` / ``derivation_step_count``.
        """
        return cls(
            status=view.status,
            method=view.method,
            adjustment_set=list(view.adjustment_set),
            graph=graph,
            query=query,
            names=list(names) if names is not None else None,
            identifier=identifier,
            assumption_count=view.assumption_count,
            derivation_step_count=view.derivation_step_count,
        )

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        inference: Frequentist | Bayesian | None = None,
        estimator: str | Estimator | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = False,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int = 1,
        latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    ) -> Any:
        """Estimate the effect on ``data`` using this identification's strategy.

        Dispatches through :func:`antecedent.analyze` with ``identifier=``
        fixed to the strategy this identification already resolved, so the
        estimate is consistent with what :func:`identify` reported rather
        than letting ``analyze`` pick a strategy on its own. There is no
        separate native "estimate against a precomputed identification" entry
        point — ``analyze`` recomputes identification internally as part of
        its own pipeline, deterministically arriving at the same strategy.
        """
        from ._analyze import analyze

        return analyze(
            data,
            query=self.query,
            graph=self.graph,
            inference=inference,
            identifier=self.identifier,
            estimator=estimator,
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            latency=latency,
        )

    def validate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
        seed: int = 1,
        threads: int = 1,
    ) -> Any:
        """Run the refutation/validation suite against this identification.

        ``refute`` defaults to ``None`` (the default suite) rather than a
        literal ``True`` — ``analyze()`` rejects an explicit ``True`` with
        ``TypeError`` (see ``_coerce.coerce_refute``), so validating "with the
        default suite" has to be spelled as "unset", not "True".
        """
        return self.estimate(data, refute=refute, seed=seed, threads=threads)


def identify(
    *,
    graph: Dag | Sequence[tuple[str, str]],
    query: AverageEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> Identification:
    """Identify without estimating; returns a stageable :class:`Identification`.

    Same parameters as the one-shot identify-only call
    (:func:`antecedent.estimation.identify`, still available unchanged for
    callers that only want the ``IdentifyResult`` shape) — the difference is
    the return type carries enough state to continue into ``.estimate()`` /
    ``.validate()`` without recomputing identification.

    Pass ``names`` when ``graph`` is an edge list (variable order). With a
    ``Dag``, names are taken from ``graph.nodes()``.
    """
    identifier_s = str(identifier) if isinstance(identifier, Identifier) else identifier
    result = _identify_native(graph=graph, query=query, names=names, identifier=identifier)
    return Identification(
        status=result.status,
        method=result.method,
        adjustment_set=list(result.adjustment_set),
        graph=graph,
        query=query,
        names=list(names) if names is not None else None,
        identifier=identifier_s,
    )


def estimate(
    identification: Identification,
    data: Mapping[str, Any] | Any,
    *,
    inference: Frequentist | Bayesian | None = None,
    estimator: str | Estimator | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = False,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
) -> Any:
    """Module-level mirror of :meth:`Identification.estimate`.

    The stub this replaces took ``(identification, *, graph, query, names,
    identifier)`` — all four of ``graph``/``query``/``names``/``identifier``
    are redundant once ``identification`` already carries them (that's the
    entire point of staging: continue without re-supplying what ``identify``
    already resolved), so they're dropped here. The stub also never accepted
    a ``data`` argument at all, despite obviously needing tabular data to run
    estimation — added as a required positional parameter.
    """
    return identification.estimate(
        data,
        inference=inference,
        estimator=estimator,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
    )


def validate(
    identification: Identification,
    data: Mapping[str, Any] | Any,
    *,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    threads: int = 1,
) -> Any:
    """Module-level mirror of :meth:`Identification.validate`. See :func:`estimate`."""
    return identification.validate(data, refute=refute, seed=seed, threads=threads)


__all__ = ["Identification", "IdentifyResult", "estimate", "identify", "validate"]
