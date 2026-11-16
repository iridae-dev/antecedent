"""Causal structure learning: algorithm configurations, graph posteriors and review.

Discovery proposes a structure from data; estimation never reruns it. The usual path is one
reviewed search, then estimation on the accepted structure::

    import antecedent
    from antecedent import discovery

    graph = discovery.discover(data, method=discovery.PC(alpha=0.01))
    outcome = antecedent.analyze(data, graph=graph, query=antecedent.AverageEffect("x", "y"))

:func:`discover` runs the configuration once and passes its result through the structure
review gate. Edges the data cannot orient stay unresolved, and pending review raises
:class:`~antecedent.errors.ReviewRequired` unless ``accept_discovered=True``.

The configurations, by the structure they return:

* Tabular, one graph: :class:`PC` and :class:`GES` (a CPDAG, assuming no unmeasured common
  cause), :class:`FCI` and :class:`RFCI` (a PAG, allowing unmeasured common causes),
  :class:`LiNGAM` and :class:`NOTEARS` (a DAG under linear-model assumptions).
* Time series, one graph: :class:`PCMCI` (lagged links), :class:`PCMCIPlus` (lagged and
  contemporaneous), :class:`LPCMCI` (with unmeasured common causes), :class:`JPCMCIPlus`
  (several environments) and :class:`RPCMCI` (labelled regimes).
* Graph posteriors (:class:`GraphPosterior`): :class:`ExactDagPosterior`, :class:`OrderMcmc`,
  :class:`StructureMcmc`, :class:`CiScreenedPosterior`, :class:`DbnPosterior`,
  :class:`ExactPagPosterior`, :class:`PagMcmc` and :class:`EquivalenceClassPosterior`.
* Stability of a frequentist search over resamples: :class:`BootstrapStability`.

Each single-graph configuration has ``run(data)`` (the raw :class:`DiscoveryResult`, no
review) and ``accept(data)`` (a reviewed :class:`~antecedent.AcceptedGraph`). Prior structural
claims are declared with :class:`BackgroundKnowledge`; a statement an algorithm cannot honour
is refused, never ignored.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

from ._coerce import coerce_data, coerce_temporal_data
from ._native import (
    DiscoveredLink,
    GraphEdge,
    GraphPosterior,
    PcmciDiscoveryResult,
    RegimeStabilityResult,
    RpcmciDiscoverySummary,
    StructureStabilityResult,
    two_regime_half_split,
)
from ._native import (
    discover_bootstrap_stability as _discover_bootstrap_stability,
)
from ._native import (
    discover_ci_screened_posterior as _discover_ci_screened_posterior,
)
from ._native import (
    discover_dbn_posterior as _discover_dbn_posterior,
)
from ._native import (
    discover_exact_dag_marginals as _discover_exact_dag_marginals,
)
from ._native import (
    discover_exact_dag_posterior as _discover_exact_dag_posterior,
)
from ._native import (
    discover_exact_pag_posterior as _discover_exact_pag_posterior,
)
from ._native import (
    discover_fci as _discover_fci,
)
from ._native import (
    discover_ges as _discover_ges,
)
from ._native import (
    discover_jpcmci_plus as _discover_jpcmci_plus,
)
from ._native import (
    discover_jpcmci_plus_stability as _discover_jpcmci_plus_stability,
)
from ._native import (
    discover_lingam as _discover_lingam,
)
from ._native import (
    discover_lpcmci as _discover_lpcmci,
)
from ._native import (
    discover_notears as _discover_notears,
)
from ._native import (
    discover_order_mcmc as _discover_order_mcmc,
)
from ._native import (
    discover_pag_mcmc as _discover_pag_mcmc,
)
from ._native import (
    discover_pc as _discover_pc,
)
from ._native import (
    discover_pcmci as _discover_pcmci,
)
from ._native import (
    discover_pcmci_plus as _discover_pcmci_plus,
)
from ._native import (
    discover_pooled_exact_dag_marginals as _discover_pooled_exact_dag_marginals,
)
from ._native import (
    discover_pooled_exact_dag_posterior as _discover_pooled_exact_dag_posterior,
)
from ._native import (
    discover_rfci as _discover_rfci,
)
from ._native import (
    discover_rpcmci as _discover_rpcmci,
)
from ._native import (
    discover_rpcmci_stability as _discover_rpcmci_stability,
)
from ._native import (
    discover_screened_exact_marginals as _discover_screened_exact_marginals,
)
from ._native import (
    discover_structure_mcmc as _discover_structure_mcmc,
)
from ._validate import count as _count
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution, native_controls

# Re-exported. The definitions live in `graph`, a leaf importing only `_native` and
# `errors`, so `_coerce` can reach `cpdag_oriented_edges` without importing this module.
from .graph import cpdag_oriented_edges, discovery_to_dag

if TYPE_CHECKING:
    from .accepted_graph import AcceptedGraph
    from .graph import Dag

CiSpec = str | Callable[..., Sequence[tuple[float, float]]]

# Natives that cannot observe a cancellation token directly; the execution's ambient
# policy still reaches them through ``Execution._call``.
_NATIVE_WITHOUT_CANCEL = frozenset(
    {
        "discover_exact_dag_posterior",
        "discover_pag_mcmc",
        "discover_exact_pag_posterior",
        "discover_exact_dag_marginals",
        "discover_pooled_exact_dag_posterior",
        "discover_pooled_exact_dag_marginals",
        "discover_order_mcmc",
        "discover_structure_mcmc",
        "discover_ci_screened_posterior",
        "discover_screened_exact_marginals",
        "discover_dbn_posterior",
    }
)


def _run_native(
    execution: Execution | None, native: Callable[..., Any], /, *args: Any, **kwargs: Any
) -> Any:
    """Run one native discovery under ``execution``'s seed, threads, cancellation and policy."""
    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an antecedent.Execution")
    controls = native_controls(
        policy,
        seed="seed",
        threads="threads",
        cancel=None if native.__name__ in _NATIVE_WITHOUT_CANCEL else "cancel",
    )
    if callable(kwargs.get("ci")):
        # A Python test callback is serialized to one worker by the native entry; the
        # ambient execution policy would replace that parallelism, so the controls are
        # passed directly and the run is not entered under the ambient policy.
        return native(*args, **controls, **kwargs)
    return policy._call(native, *args, **controls, **kwargs)


@dataclass(frozen=True, slots=True)
class Link:
    """A directed lagged link ``source[t - source_lag] -> target[t - target_lag]`` by name."""

    source: str
    target: str
    source_lag: int = 0
    target_lag: int = 0


def _variable_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _lag(value: object) -> int:
    return _count(value, "lag", minimum=0, maximum=2**32 - 1)


@dataclass(frozen=True, slots=True)
class BackgroundKnowledge:
    """What the analyst asserts about the graph before the data are consulted.

    ``required`` links must be in the graph, ``forbidden`` links must not, ``tiers`` order the
    variables (a variable in a later tier cannot cause one in an earlier tier; variables in no
    tier are unconstrained), and ``max_parents`` bounds the parents of each variable. Statements
    name variables, and are checked against the searched variables when the discovery runs.

    The same statement is read by the ordinary discovery classes (``PC``, ``FCI``, ``GES``,
    ``PCMCI``) and by the selective-discovery contracts, and has one digest
    (``DiscoveryResult.background_knowledge``) that changes exactly when the statements do.
    What each algorithm does with a statement, and which it refuses, is the table in the
    discovery guide: a statement an algorithm cannot honour is refused, never ignored.
    """

    required: tuple[Link, ...] = ()
    forbidden: tuple[Link, ...] = ()
    tiers: tuple[tuple[str, ...], ...] = ()
    max_parents: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("required", "forbidden"):
            links = tuple(getattr(self, field_name))
            for link in links:
                if not isinstance(link, Link):
                    raise CausalTypeError(f"{field_name} must hold Link objects")
            object.__setattr__(self, field_name, links)
        tiers = tuple(tuple(tier) for tier in self.tiers)
        for tier in tiers:
            for name in tier:
                _variable_name(name, "a tier member")
        object.__setattr__(self, "tiers", tiers)
        if self.max_parents is not None:
            object.__setattr__(
                self, "max_parents", _count(self.max_parents, "max_parents", minimum=0)
            )

    @property
    def is_empty(self) -> bool:
        """Whether nothing is asserted."""
        return not (self.required or self.forbidden or self.tiers or self.max_parents is not None)

    def as_dict(self) -> dict[str, Any]:
        """The statements by variable name, in a form that round-trips and prints."""

        def link(item: Link) -> list[Any]:
            return [item.source, item.source_lag, item.target, item.target_lag]

        out: dict[str, Any] = {
            "required": [link(item) for item in self.required],
            "forbidden": [link(item) for item in self.forbidden],
            "tiers": [list(tier) for tier in self.tiers],
        }
        if self.max_parents is not None:
            out["max_parents"] = self.max_parents
        return out

    def to_wire(self, names: Sequence[str]) -> dict[str, Any]:
        """The statements by position in ``names``, as the native layer reads them.

        The one translation both discovery paths use. A name outside ``names`` is refused with
        reason code ``unknown_variable``.
        """
        index = {name: i for i, name in enumerate(names)}

        def at(name: str) -> int:
            _variable_name(name, "variable")
            try:
                return index[name]
            except KeyError:
                raise CausalUnsupportedError(
                    f"background knowledge names {name!r}, which is not one of the searched "
                    f"variables {tuple(names)}",
                    reason_code="unknown_variable",
                ) from None

        def link(item: Link) -> list[int]:
            return [
                at(item.source),
                _lag(item.source_lag),
                at(item.target),
                _lag(item.target_lag),
            ]

        wire: dict[str, Any] = {
            "required": [link(item) for item in self.required],
            "forbidden": [link(item) for item in self.forbidden],
            "tiers": [[at(name) for name in tier] for tier in self.tiers],
        }
        if self.max_parents is not None:
            wire["max_parents"] = self.max_parents
        return wire


class _KnowledgeKwargs(TypedDict, total=False):
    background_knowledge: str


def _knowledge_kwargs(
    knowledge: BackgroundKnowledge | None, names: Sequence[str]
) -> _KnowledgeKwargs:
    """The native ``background_knowledge`` argument for a run over ``names``, if any."""
    if knowledge is None:
        return {}
    if not isinstance(knowledge, BackgroundKnowledge):
        raise CausalTypeError("knowledge must be a BackgroundKnowledge")
    if knowledge.is_empty:
        return {}
    return {"background_knowledge": json.dumps(knowledge.to_wire(names), sort_keys=True)}


@contextmanager
def _named_positions(names: Sequence[str]) -> Iterator[None]:
    """Name the variables of a knowledge refusal that the native layer reports by position.

    The native search sees columns, not names, so a contradiction reads "variable 0 -> variable
    1"; the caller wrote ``Link("age", "dose")``.
    """
    try:
        yield
    except CausalUnsupportedError as error:
        text = str(error.args[0]) if error.args else ""
        code = getattr(error, "reason_code", None)
        if code not in ("invalid_argument", "option_not_applicable") or "variable " not in text:
            raise
        message = text.removeprefix(f"reason={code}: ")
        named = re.sub(
            r"variable (\d+)",
            lambda m: f"variable {names[int(m[1])]!r}" if int(m[1]) < len(names) else m[0],
            message,
        )
        raise CausalUnsupportedError(named, reason_code=code) from error


def _wire_knowledge(wire: dict[str, Any], knowledge: BackgroundKnowledge | None) -> dict[str, Any]:
    """``wire`` with the by-name statements of non-empty ``knowledge`` added."""
    if knowledge is not None and not knowledge.is_empty:
        wire["background_knowledge"] = knowledge.as_dict()
    return wire


@dataclass(frozen=True)
class PC:
    """PC: a Markov equivalence class (CPDAG) learned from conditional-independence tests.

    Assumes acyclicity, causal sufficiency (no unmeasured common cause of two searched variables)
    and faithfulness. Edges whose direction the data cannot fix stay undirected and are
    reviewed before the structure is accepted. Use :class:`FCI` when a common cause may be
    unmeasured.

    Fields:

    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.

    Use :meth:`accept` (or :func:`discover`) for a reviewed :class:`~antecedent.AcceptedGraph`.
    """

    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["pc"] = "pc"

    algorithm_id: ClassVar[str] = "pc"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "pc",
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_pc,
                names,
                cols,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                max_cond_size=self.max_cond_size,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class PCMCI:
    """PCMCI: lagged causal links of a multivariate time series.

    Finds links ``X[t - lag] -> Y[t]`` for lags 1 to ``max_lag``; contemporaneous links are not
    searched (use :class:`PCMCIPlus`). Assumes stationarity of the link structure over the
    series, causal sufficiency and faithfulness.

    Fields:

    * ``max_lag`` -- the largest lag searched. Default ``1``.
    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["pcmci"] = "pcmci"

    algorithm_id: ClassVar[str] = "pcmci"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "pcmci",
                "max_lag": self.max_lag,
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        """Run the search once on a time series and return the raw result, without review.

        ``data`` is one series (a table whose rows are consecutive time points) or an
        :class:`antecedent.data.PanelFrame`, whose units each keep their own lag windows and are
        pooled. ``weights`` holds one observation weight per row and is accepted only with
        ``ci="weighted_parcorr"``. Read ``links`` and ``graph_edges`` first. To use the structure
        for estimation, call :meth:`accept` (or :func:`discover`) instead.
        """
        names, cols, unit_lengths = coerce_temporal_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_pcmci,
                names,
                cols,
                unit_lengths=unit_lengths,
                max_lag=self.max_lag,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                weights=weights,
                max_cond_size=self.max_cond_size,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class PCMCIPlus:
    """PCMCI+: lagged and contemporaneous links of a multivariate time series.

    Extends :class:`PCMCI` to lag 0. Lagged links are oriented by time; contemporaneous links
    the data cannot orient stay undirected (a temporal CPDAG) and are reviewed before
    acceptance. Same assumptions as :class:`PCMCI`.

    Fields:

    * ``max_lag`` -- the largest lag searched. Default ``1``.
    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["pcmci_plus"] = "pcmci_plus"

    # NB: asymmetric on purpose. ``run_temporal_discovery`` has always
    # returned the short form "pcmci+" as this config's algorithm id, while
    # ``discovery_algorithm``/``_wire()`` has always used the long form
    # "pcmci_plus" as the native "algorithm" wire value. Both are preserved.
    algorithm_id: ClassVar[str] = "pcmci+"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "pcmci_plus",
                "max_lag": self.max_lag,
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        """Run the search once on a time series and return the raw result, without review.

        ``data`` is one series (a table whose rows are consecutive time points) or an
        :class:`antecedent.data.PanelFrame`, whose units each keep their own lag windows and are
        pooled. ``weights`` holds one observation weight per row and is accepted only with
        ``ci="weighted_parcorr"``. Read ``links`` and ``graph_edges`` first. To use the structure
        for estimation, call :meth:`accept` (or :func:`discover`) instead.
        """
        names, cols, unit_lengths = coerce_temporal_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_pcmci_plus,
                names,
                cols,
                unit_lengths=unit_lengths,
                max_lag=self.max_lag,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                weights=weights,
                max_cond_size=self.max_cond_size,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class LPCMCI:
    """LPCMCI: lagged and contemporaneous links of a time series with unmeasured common causes.

    The time-series counterpart of :class:`FCI`: the result is a temporal PAG whose circle
    endpoints mark orientations the data leave open. Takes no background knowledge. Note the
    ``fdr`` default is ``False`` here.

    Fields:

    * ``max_lag`` -- the largest lag searched. Default ``1``.
    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``False``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = False
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["lpcmci"] = "lpcmci"

    algorithm_id: ClassVar[str] = "lpcmci"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "lpcmci",
            "max_lag": self.max_lag,
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        """Run the search once on a time series and return the raw result, without review.

        ``data`` is one series (a table whose rows are consecutive time points) or an
        :class:`antecedent.data.PanelFrame`, whose units each keep their own lag windows and are
        pooled. ``weights`` holds one observation weight per row and is accepted only with
        ``ci="weighted_parcorr"``. Read ``links`` and ``graph_edges`` first. To use the structure
        for estimation, call :meth:`accept` (or :func:`discover`) instead.
        """
        names, cols, unit_lengths = coerce_temporal_data(data)
        return _run_native(
            execution,
            _discover_lpcmci,
            names,
            cols,
            unit_lengths=unit_lengths,
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            ci=self.ci,
            weights=weights,
            max_cond_size=self.max_cond_size,
        )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class JPCMCIPlus:
    """J-PCMCI+: one temporal causal graph learned jointly from several environments or datasets.

    Pools the lagged tests across environments without concatenating them, and adds observed
    context variables and synthetic dummies so that differences between environments are
    modelled rather than read as links. The result is a temporal CPDAG over the system
    variables, with context nodes. Takes no background knowledge.

    Fields:

    * ``max_lag`` -- the largest lag searched. Default ``1``.
    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``context_names`` -- columns that are observed context variables, not system variables.
      Default ``()``.
    * ``include_space_dummy`` -- add a dummy identifying the environment (with two or more
      environments). Default ``True``.
    * ``include_time_dummy`` -- add a dummy for the time index. Default ``False``.
    * ``space_dummy_ci`` / ``time_dummy_ci`` -- whether a dummy enters a test as scalar
      columns (``"scalar"``, default) or as one multivariate block (``"multivariate"``).
    * ``time_dummy_encoding`` -- ``"integer"`` (default) or ``"one_hot"``.

    Pass a :class:`antecedent.data.MultiEnvFrame`, a :class:`antecedent.data.PanelFrame` or a
    sequence of environment tables to :meth:`accept`.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    context_names: tuple[str, ...] = ()
    include_space_dummy: bool = True
    include_time_dummy: bool = False
    space_dummy_ci: Literal["scalar", "multivariate"] = "scalar"
    time_dummy_encoding: Literal["integer", "one_hot"] = "integer"
    time_dummy_ci: Literal["scalar", "multivariate"] = "scalar"
    max_cond_size: int = 2
    kind: Literal["jpcmci_plus"] = "jpcmci_plus"

    algorithm_id: ClassVar[str] = "jpcmci_plus"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "jpcmci_plus",
            "max_lag": self.max_lag,
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "context_names": list(self.context_names),
            "include_space_dummy": self.include_space_dummy,
            "include_time_dummy": self.include_time_dummy,
            "space_dummy_ci": self.space_dummy_ci,
            "time_dummy_encoding": self.time_dummy_encoding,
            "time_dummy_ci": self.time_dummy_ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(
        self,
        names: list[str],
        env_columns: Sequence[Sequence[Any]],
        *,
        execution: Execution | None = None,
        weights: list[float] | None = None,
    ) -> PcmciDiscoveryResult:
        """Multi-environment discovery: takes ``names``/``env_columns`` directly.

        Unlike the single-table configs, J-PCMCI+ discovers across the
        environments themselves rather than a single pooled table:
        ``coerce_data`` refuses to row-concatenate a ``MultiEnvFrame`` into one
        table (see its docstring), which would destroy the per-environment
        structure J-PCMCI+ needs to build its space/time dummies from. Build
        ``env_columns`` with :func:`antecedent._data.as_multi_env_columns` or
        :func:`antecedent.data.multi_env`.
        """
        return _run_native(
            execution,
            _discover_jpcmci_plus,
            list(names),
            [list(cols) for cols in env_columns],
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            ci=self.ci,
            weights=weights,
            context_names=list(self.context_names) if self.context_names else None,
            include_space_dummy=self.include_space_dummy,
            include_time_dummy=self.include_time_dummy,
            space_dummy_ci=self.space_dummy_ci,
            time_dummy_encoding=self.time_dummy_encoding,
            time_dummy_ci=self.time_dummy_ci,
            max_cond_size=self.max_cond_size,
        )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class RPCMCI:
    """Regime-PCMCI. Pass ``regimes=`` to ``analyze`` / ``run()`` (required).

    Use ``two_regime_half_split(n)`` when a simple half-split label vector is enough.
    """

    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["rpcmci"] = "rpcmci"

    algorithm_id: ClassVar[str] = "rpcmci"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "rpcmci",
            "max_lag": self.max_lag,
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(
        self,
        data: Any,
        *,
        regimes: Sequence[int],
        execution: Execution | None = None,
        weights: list[float] | None = None,
    ) -> RpcmciDiscoverySummary:
        """``regimes`` is required (length = series length); no silent half-split.

        Call ``two_regime_half_split(len(series))`` for an explicit two-regime mid-point split.
        """
        names, cols = coerce_data(data, temporal=True)
        return _run_native(
            execution,
            _discover_rpcmci,
            names,
            cols,
            regimes=list(regimes),
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            ci=self.ci,
            weights=weights,
            max_cond_size=self.max_cond_size,
        )

    def accept(
        self,
        data: Any,
        *,
        regimes: Sequence[int] | None = None,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate.

        ``regimes`` (one label per observation) is required.
        """
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
            regimes=regimes,
        )


@dataclass(frozen=True)
class GES:
    """GES: greedy equivalence search for the CPDAG that maximises the Gaussian BIC score.

    A forward phase inserts edges and a backward phase deletes them, both over equivalence
    classes. Same assumptions as :class:`PC` (acyclicity, causal sufficiency), with a linear
    Gaussian score in place of tests. ``alpha``, ``fdr``, ``ci`` and ``max_cond_size`` are
    read only by the optional PC skeleton screen.

    Fields:

    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``screen_pc`` -- restrict the candidate insertions with a PC skeleton screen. Default
      ``False``.
    * ``max_subset`` -- the operator search per pair enumerates subsets of at most this many
      neighbours; ``None`` uses 12. When the cap binds, the result carries a
      ``ges.operator_search_pruned`` diagnostic.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.
    """

    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    screen_pc: bool = False
    max_subset: int | None = None
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["ges"] = "ges"

    algorithm_id: ClassVar[str] = "ges"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "ges",
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_ges,
                names,
                cols,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                max_cond_size=self.max_cond_size,
                screen_pc=self.screen_pc,
                max_subset=self.max_subset,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class LiNGAM:
    """DirectLiNGAM: a single DAG learned from linear non-Gaussian data.

    Assumes a linear acyclic model with independent non-Gaussian errors and no unmeasured
    common cause; under those assumptions the causal order, and so every edge direction, is
    identified. Takes no background knowledge.

    Fields:

    * ``prune_threshold`` -- edges whose absolute standardised coefficient
      ``|beta| * sd(parent) / sd(child)`` falls below this are pruned. Default ``0.05``.
    * ``max_cond_size`` -- the conditioning cap of the shared discovery constraints. Default
      ``8``.
    """

    prune_threshold: float = 0.05
    max_cond_size: int = 8
    kind: Literal["lingam"] = "lingam"

    algorithm_id: ClassVar[str] = "lingam"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "lingam",
            "prune_threshold": self.prune_threshold,
            "max_cond_size": self.max_cond_size,
            "alpha": 0.05,
            "fdr": True,
            "ci": "parcorr",
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_lingam,
            names,
            cols,
            prune_threshold=self.prune_threshold,
            max_cond_size=self.max_cond_size,
        )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class NOTEARS:
    """Continuous-SEM (NOTEARS) discovery over numeric columns.

    ``missing`` declares how rows with a missing value are handled. ``None`` (the default)
    refuses the table at the first missing cell. ``"complete_case"`` drops every row with a
    missing value in any discovery column and learns the graph on the complete rows.
    ``"pairwise_available"`` scores the least-squares loss on the pairwise-available
    covariance (each pair uses the rows observed on both), repaired to positive
    semidefinite by clipping negative eigenvalues. Both report ``result.row_accounting``
    (rows used and dropped or per-pair row counts, missing values per variable, a digest of
    the dropped rows) and speak for the full data only when missingness is completely at
    random.
    """

    l1: float = 0.1
    threshold: float = 0.3
    standardize: bool = True
    max_cond_size: int = 8
    missing: Literal["error", "complete_case", "pairwise_available"] | None = None
    kind: Literal["notears"] = "notears"

    algorithm_id: ClassVar[str] = "notears"

    def __post_init__(self) -> None:
        if self.missing not in (None, "error", "complete_case", "pairwise_available"):
            raise CausalValueError(
                f"NOTEARS missing must be None, 'error', 'complete_case' or 'pairwise_available'; got {self.missing!r}",
                reason_code="invalid_argument",
            )

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "algorithm": "notears",
            "lambda": self.l1,
            "threshold": self.threshold,
            "standardize": self.standardize,
            "max_cond_size": self.max_cond_size,
            "alpha": 0.05,
            "fdr": True,
            "ci": "parcorr",
        }
        if self.missing is not None:
            wire["missing"] = self.missing
        return wire

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_notears,
            names,
            cols,
            l1=self.l1,
            threshold=self.threshold,
            standardize=self.standardize,
            max_cond_size=self.max_cond_size,
            missing=self.missing,
        )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class FCI:
    """FCI: a partial ancestral graph (PAG) that allows unmeasured common causes.

    Drops the causal-sufficiency assumption of :class:`PC`: the PAG represents every maximal
    ancestral graph consistent with the tests, and a circle endpoint marks an orientation the
    data leave open. Assumes acyclicity and faithfulness. Class marks stay class information
    after acceptance.

    Fields:

    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.
    """

    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["fci"] = "fci"

    algorithm_id: ClassVar[str] = "fci"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "fci",
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_fci,
                names,
                cols,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                max_cond_size=self.max_cond_size,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


@dataclass(frozen=True)
class RFCI:
    """RFCI: a faster approximation of :class:`FCI` that runs fewer conditional-independence tests.

    Same assumptions and output type as :class:`FCI` (a PAG that allows unmeasured common
    causes); it skips the possible-d-separation phase, so its PAG can leave more endpoints
    unoriented than FCI's on the same data.

    Fields:

    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...) or a Python
      callable batch test, which runs on one worker. An unknown name raises with
      ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``knowledge`` -- a :class:`BackgroundKnowledge` of required and forbidden links, tiers and a
      parent limit; a statement the algorithm cannot honour is refused. Default ``None``.
    """

    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    knowledge: BackgroundKnowledge | None = None
    kind: Literal["rfci"] = "rfci"

    algorithm_id: ClassVar[str] = "rfci"

    def _wire(self) -> dict[str, Any]:
        return _wire_knowledge(
            {
                "algorithm": "rfci",
                "alpha": self.alpha,
                "fdr": self.fdr,
                "ci": self.ci,
                "max_cond_size": self.max_cond_size,
            },
            self.knowledge,
        )

    def run(self, data: Any, *, execution: Execution | None = None) -> DiscoveryResult:
        """Run the search once on a table and return the raw result, without review.

        ``data`` is a table in any form :func:`antecedent.data.table` accepts; ``execution``
        sets the seed, threads and cancellation. Read ``graph_edges`` and ``pending_edge_count``
        first. To use the structure for estimation, call :meth:`accept` (or :func:`discover`)
        instead.
        """
        names, cols = coerce_data(data)
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_rfci,
                names,
                cols,
                alpha=self.alpha,
                fdr=self.fdr,
                ci=self.ci,
                max_cond_size=self.max_cond_size,
                **_knowledge_kwargs(self.knowledge, names),
            )

    def accept(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        accept_discovered: bool = True,
    ) -> AcceptedGraph:
        """Run this configuration once and accept it through its review gate."""
        from .accepted_graph import accept_discovery

        return accept_discovery(
            self,
            data,
            execution=execution,
            accept_discovered=accept_discovered,
        )


#: Most variables the exact route lists graphs for; above it only edge marginals are exact.
EXACT_ENUMERATION_MAX_VARIABLES = 6
#: Most variables the exact route computes edge marginals for (the native work budget).
EXACT_MARGINALS_MAX_VARIABLES = 15

_ATOM_ONLY = frozenset(
    {"adjacency", "weights", "graph_keys", "n_graphs", "to_weighted_samples", "ess", "atoms"}
)


class _UnavailableGraphAtoms(CausalUnsupportedError, AttributeError):
    """Keep the scientific refusal compatible with Python attribute discovery."""


@dataclass(frozen=True)
class StructurePrior:
    """A prior over DAG structures beyond one global edge probability.

    ``edge_probabilities`` maps ``(from, to)`` variable names to the prior probability that the
    directed edge is present, each in the open interval (0, 1); ``default_edge_probability`` is
    that probability for every edge not listed (unset: the listed edges stand alone against a
    neutral prior). Naming an unknown variable, a self edge, an edge twice, or an edge the
    background knowledge forbids is refused.

    ``soft_tiers`` order variables like the tiers of :class:`BackgroundKnowledge`, but an edge
    from a later tier into an earlier one is penalised rather than forbidden: its prior odds
    are multiplied by ``exp(-soft_tier_log_penalty)``, or equivalently by
    ``soft_tier_odds_factor`` in (0, 1) (give one of the two). A hard tier always wins: an edge
    it forbids carries no prior at all.

    Both factors are modular, so the exact edge marginals and the order and structure samplers
    use the same prior and need no other change. Pass the prior as ``prior=`` to
    :class:`ExactDagPosterior`, :class:`OrderMcmc` or :class:`StructureMcmc`.
    """

    edge_probabilities: Mapping[tuple[str, str], float] | Sequence[tuple[str, str, float]] = ()
    default_edge_probability: float | None = None
    soft_tiers: tuple[tuple[str, ...], ...] = ()
    soft_tier_log_penalty: float | None = None
    soft_tier_odds_factor: float | None = None

    def __post_init__(self) -> None:
        raw = self.edge_probabilities
        triples = (
            [(a, b, p) for (a, b), p in raw.items()]
            if isinstance(raw, Mapping)
            else [tuple(item) for item in raw]
        )
        edges: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for item in triples:
            if len(item) != 3:
                raise CausalTypeError("an edge probability is (from, to, probability)")
            source = _variable_name(item[0], "an edge probability's source")
            target = _variable_name(item[1], "an edge probability's target")
            probability = _probability(item[2], f"the probability of {source} -> {target}")
            if source == target:
                raise CausalValueError(f"the edge probability names a self edge on {source!r}")
            if (source, target) in seen:
                raise CausalValueError(f"the edge {source} -> {target} is listed twice")
            seen.add((source, target))
            edges.append((source, target, probability))
        object.__setattr__(self, "edge_probabilities", tuple(edges))
        if self.default_edge_probability is not None:
            object.__setattr__(
                self,
                "default_edge_probability",
                _probability(self.default_edge_probability, "default_edge_probability"),
            )
        tiers = tuple(tuple(tier) for tier in self.soft_tiers)
        for tier in tiers:
            for name in tier:
                _variable_name(name, "a soft-tier member")
        object.__setattr__(self, "soft_tiers", tiers)
        if self.soft_tier_log_penalty is not None and self.soft_tier_odds_factor is not None:
            raise CausalValueError("give soft_tier_log_penalty or soft_tier_odds_factor, not both")
        if tiers and self.soft_tier_log_penalty is None and self.soft_tier_odds_factor is None:
            raise CausalValueError(
                "soft_tiers need a soft_tier_log_penalty or a soft_tier_odds_factor"
            )
        if not tiers and (
            self.soft_tier_log_penalty is not None or self.soft_tier_odds_factor is not None
        ):
            raise CausalValueError("a soft-tier penalty needs soft_tiers")
        if self.soft_tier_log_penalty is not None:
            penalty = float(self.soft_tier_log_penalty)
            if not math.isfinite(penalty) or penalty < 0.0:
                raise CausalValueError("soft_tier_log_penalty must be finite and non-negative")
            object.__setattr__(self, "soft_tier_log_penalty", penalty)
        if self.soft_tier_odds_factor is not None:
            object.__setattr__(
                self,
                "soft_tier_odds_factor",
                _probability(self.soft_tier_odds_factor, "soft_tier_odds_factor"),
            )

    def _native_kwargs(self) -> dict[str, Any]:
        """The native search's prior arguments (only those this prior sets)."""
        out: dict[str, Any] = {}
        if self.edge_probabilities:
            out["edge_priors"] = [list(edge) for edge in self.edge_probabilities]
        if self.default_edge_probability is not None:
            out["edge_prior_default"] = self.default_edge_probability
        if self.soft_tiers:
            out["soft_tiers"] = [list(tier) for tier in self.soft_tiers]
            out["soft_tier_penalty"] = (
                self.soft_tier_log_penalty
                if self.soft_tier_log_penalty is not None
                else -math.log(float(self.soft_tier_odds_factor or 1.0))
            )
        return out

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {}
        if self.edge_probabilities:
            wire["edge_probabilities"] = [list(edge) for edge in self.edge_probabilities]
        if self.default_edge_probability is not None:
            wire["default_edge_probability"] = self.default_edge_probability
        wire.update(
            {key: value for key, value in self._native_kwargs().items() if key.startswith("soft")}
        )
        return wire


def _probability(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{what} must be a number")
    probability = float(value)
    if not (0.0 < probability < 1.0):
        raise CausalValueError(f"{what} must lie in (0, 1)")
    return probability


def _prior_kwargs(prior: StructurePrior | None) -> dict[str, Any]:
    return {} if prior is None else prior._native_kwargs()


def _prior_wire(prior: StructurePrior | None) -> dict[str, Any]:
    return {} if prior is None else {"prior": prior._wire()}


def _imputed_tables(imputations: Any) -> list[tuple[list[str], list[Any]]]:
    """Coerce a sequence of completed tables, checking they state one set of variables."""
    if isinstance(imputations, (str, bytes, Mapping)) or hasattr(imputations, "to_numpy"):
        raise CausalTypeError(
            "imputations takes a sequence of completed tables, one per imputation, not one table"
        )
    try:
        tables = list(imputations)
    except TypeError as error:
        raise CausalTypeError(
            "imputations takes a sequence of completed tables, one per imputation"
        ) from error
    if not tables:
        raise CausalValueError("imputations needs at least one completed table")
    coerced = [coerce_data(table) for table in tables]
    names = coerced[0][0]
    for index, (other, _) in enumerate(coerced[1:], start=1):
        if other != names:
            raise CausalValueError(
                f"imputation {index} names variables {other} but imputation 0 names {names}; "
                "every imputation must state the same variables in the same order"
            )
    return coerced


@dataclass(frozen=True)
class ExactDagMarginals:
    """Exact edge marginals of the Gaussian-BIC DAG posterior of 7 to 15 variables.

    ``edge_marginal_matrix()[i][j]`` is ``P(i -> j)`` over ``names`` (the variables of the run, or
    the screened selection of :class:`ScreenedExactMarginals`, whose ``conditional_on_screen``
    states that every number is conditional on the CI screen), exact for the BIC pseudo-posterior with the
    edge prior of the run (uniform unless a :class:`StructurePrior` was given; no sampling
    error). ``log_evidence`` is the log pseudo-posterior
    normalizer ``log sum_G exp(score(G))``, with Gaussian score equal to -BIC/2 including
    the fitted residual variances; it is not an exact marginal likelihood. Above six variables the graphs themselves cannot be listed, so
    the graph-level outputs of :class:`GraphPosterior` (atoms, weights, the MAP graph, the
    mixture over graphs that ``analyze`` uses) do not exist; asking for one is refused rather than
    approximated. Use ``OrderMcmc`` or ``StructureMcmc`` for atoms.
    """

    names: list[str]
    n_vars: int
    log_evidence: float
    _matrix: tuple[tuple[float, ...], ...]
    conditional_on_screen: str | None = None
    pooled_over_imputations: int | None = None

    def edge_marginal_matrix(self) -> list[list[float]]:
        """The exact edge marginals as a list of rows: ``[i][j]`` is ``P(i -> j)``, in the order of
        ``names``.
        """
        return [list(row) for row in self._matrix]

    def __getattr__(self, name: str) -> Any:
        if name in _ATOM_ONLY:
            raise _UnavailableGraphAtoms(
                f"{name!r} needs the list of graphs, which the exact route does not produce above "
                f"{EXACT_ENUMERATION_MAX_VARIABLES} variables (it publishes exact edge marginals "
                "only); use OrderMcmc or StructureMcmc for graph atoms",
                reason_code="option_not_applicable",
            )
        raise AttributeError(name)


@dataclass(frozen=True)
class ExactDagPosterior:
    """Exact DAG posterior (Gaussian BIC).

    Up to 6 variables every DAG is enumerated and ``run`` returns a ``GraphPosterior`` (atoms,
    weights, marginals). From 7 to 15 variables ``run`` returns :class:`ExactDagMarginals`: the
    exact ``P(i -> j)`` and the log normalizing constant by dynamic programming over node subsets,
    with no graph atoms (and so no composition with ``analyze``). More than 15 variables are
    refused naming that size; use ``OrderMcmc``, ``StructureMcmc`` or ``CiScreenedPosterior``.

    Composed: pass the executed ``ExactDagPosterior().run(data)`` as ``discovery=`` with
    ``inference=Bayesian(...)`` to ``analyze`` to mix effect draws over the graph posterior
    (at most 6 variables).

    ``prior`` (a :class:`StructurePrior`) gives per-edge inclusion probabilities and soft tiers;
    unset, the edge prior is uniform.
    """

    prior: StructurePrior | None = None
    kind: Literal["exact_dag_posterior"] = "exact_dag_posterior"

    algorithm_id: ClassVar[str] = "exact_dag_posterior"

    def _wire(self) -> dict[str, Any]:
        return {"algorithm": "exact_dag_posterior", **_prior_wire(self.prior)}

    def run(
        self,
        data: Any = None,
        *,
        execution: Execution | None = None,
        imputations: Sequence[Any] | None = None,
    ) -> GraphPosterior | ExactDagMarginals:
        """Run the exact posterior on the data.

        ``imputations`` (instead of ``data``) is a sequence of completed tables, one per
        imputation of the measurement or missing-data model: the posterior is run on each and
        the result is the equal-weight mixture of the per-imputation posteriors (the mean of
        their edge marginals above six variables), so the data's own uncertainty widens the
        structure posterior.
        """
        if imputations is not None or data is None:
            return self._run_pooled(data, imputations, execution)
        names, cols = coerce_data(data)
        if len(names) <= EXACT_ENUMERATION_MAX_VARIABLES:
            return _run_native(
                execution, _discover_exact_dag_posterior, names, cols, **_prior_kwargs(self.prior)
            )
        flat, log_evidence = _run_native(
            execution, _discover_exact_dag_marginals, names, cols, **_prior_kwargs(self.prior)
        )
        n = len(names)
        matrix = tuple(tuple(flat[i * n : (i + 1) * n]) for i in range(n))
        return ExactDagMarginals(list(names), n, log_evidence, matrix)

    def _run_pooled(
        self, data: Any, imputations: Any, execution: Execution | None
    ) -> GraphPosterior | ExactDagMarginals:
        """The posterior pooled across imputations, computed by the engine."""
        if data is not None:
            raise CausalValueError("give data or imputations, not both")
        if imputations is None:
            raise CausalValueError(
                "run needs data, or imputations= (a sequence of completed tables)"
            )
        tables = _imputed_tables(imputations)
        names = tables[0][0]
        columns = [cols for _, cols in tables]
        if len(names) <= EXACT_ENUMERATION_MAX_VARIABLES:
            return _run_native(
                execution,
                _discover_pooled_exact_dag_posterior,
                names,
                columns,
                **_prior_kwargs(self.prior),
            )
        flat, log_evidence = _run_native(
            execution,
            _discover_pooled_exact_dag_marginals,
            names,
            columns,
            **_prior_kwargs(self.prior),
        )
        n = len(names)
        matrix = tuple(tuple(flat[i * n : (i + 1) * n]) for i in range(n))
        return ExactDagMarginals(
            list(names), n, log_evidence, matrix, pooled_over_imputations=len(columns)
        )


@dataclass(frozen=True)
class ExactPagPosterior:
    """Exact posterior over PAG classes of Gaussian maximal ancestral graphs (at most 4 variables).

    The Bayesian counterpart of :class:`FCI` and :class:`RFCI`: every maximal ancestral graph is
    enumerated, fitted by residual iterative conditional fitting and scored by BIC, and graphs
    with the same m-separation statements (Markov equivalent) are merged into their PAG. The
    result is a :class:`GraphPosterior` whose ``atom_kind`` is ``"Pag"`` and whose
    ``mark_masks`` flag the circle endpoints. Latent confounding is allowed; selection variables
    are not modelled. The weights are ``exp(BIC)`` under a uniform prior over the graphs, the BIC
    approximation to the marginal likelihood. More than four variables are refused (the
    candidate space is enumerated, not sampled).
    """

    kind: Literal["exact_pag_posterior"] = "exact_pag_posterior"

    algorithm_id: ClassVar[str] = "exact_pag_posterior"

    def _wire(self) -> dict[str, Any]:
        return {"algorithm": "exact_pag_posterior"}

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols = coerce_data(data)
        return _run_native(execution, _discover_exact_pag_posterior, names, cols)


@dataclass(frozen=True)
class PagMcmc:
    """Metropolis posterior over PAG classes of Gaussian maximal ancestral graphs.

    Two to 31 variables are representable; a run whose cost (moves times the m-separation
    queries that sign one graph) exceeds the sampler's work budget is refused before that width.

    The sampled counterpart of :class:`ExactPagPosterior`: each chain starts at the empty graph
    and changes the link of one pair at a time, scoring every maximal ancestral graph by BIC
    after residual iterative conditional fitting; a proposal that is not ancestral or not
    maximal is rejected. The pooled visit frequency of each Markov equivalence class is its
    weight and its PAG is completed from the class's graphs reachable by single-pair moves.
    ``result.converged`` records only that the chains' class frequencies agree to a total
    variation of 0.15; no finer statistic is computed, and the sampler reaches only the graphs
    connected to the empty graph by single-pair moves through maximal ancestral graphs.
    """

    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 2000
    thin: int = 1
    kind: Literal["pag_mcmc"] = "pag_mcmc"

    algorithm_id: ClassVar[str] = "pag_mcmc"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "pag_mcmc",
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            "thin": self.thin,
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_pag_mcmc,
            names,
            cols,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
        )


@dataclass(frozen=True)
class OrderMcmc:
    """Order-MCMC graph posterior over up to 16 variables.

    ``max_parents`` caps the parents of any node (``None``: no cap). Each step costs a
    ``2**n``-state dynamic program, so the run time doubles with every variable.

    ``noise="gaussian"`` scores parent sets by the Gaussian BIC, which cannot tell the DAGs of a
    Markov equivalence class apart. ``noise="laplace"`` scores them under independent Laplace
    noise (least-absolute-deviations fits, a heavy-tailed non-Gaussian family), so the posterior
    can orient an edge the Gaussian posterior leaves undirected, as LiNGAM does when the noise
    is truly non-Gaussian; it is a pseudo-posterior under that declared noise family and is
    misspecified under other non-Gaussian noise. It is the Bayesian counterpart of LiNGAM and
    NOTEARS orientation.

    ``prior`` (a :class:`StructurePrior`) gives per-edge inclusion probabilities and soft tiers.
    """

    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    require_diagnostics_gate: bool = True
    max_parents: int | None = None
    noise: Literal["gaussian", "laplace"] = "gaussian"
    prior: StructurePrior | None = None
    kind: Literal["order_mcmc"] = "order_mcmc"

    algorithm_id: ClassVar[str] = "order_mcmc"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "order_mcmc",
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            "thin": self.thin,
            "require_diagnostics_gate": self.require_diagnostics_gate,
            **({} if self.max_parents is None else {"max_parents": self.max_parents}),
            **({} if self.noise == "gaussian" else {"noise": self.noise}),
            **_prior_wire(self.prior),
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_order_mcmc,
            names,
            cols,
            noise=self.noise,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
            require_diagnostics_gate=self.require_diagnostics_gate,
            max_parents=self.max_parents,
            **_prior_kwargs(self.prior),
        )


@dataclass(frozen=True)
class StructureMcmc:
    """Structure-MCMC graph posterior over up to 16 variables.

    The multi-chain diagnostics gate is always on: the run is refused unless split R-hat,
    bulk ESS and every edge's Monte-Carlo error clear the publication bar, so there is no
    ``require_diagnostics_gate`` switch here (unlike :class:`OrderMcmc`). Per-edge
    ``edge_rhat`` / ``edge_ess`` / ``edge_mcse`` on the result report the chain diagnostics.
    ``max_parents`` caps the parents of any node (``None``: no cap). ``prior`` (a
    :class:`StructurePrior`) gives per-edge inclusion probabilities and soft tiers.
    """

    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    max_parents: int | None = None
    prior: StructurePrior | None = None
    kind: Literal["structure_mcmc"] = "structure_mcmc"

    algorithm_id: ClassVar[str] = "structure_mcmc"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "structure_mcmc",
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            "thin": self.thin,
            **({} if self.max_parents is None else {"max_parents": self.max_parents}),
            **_prior_wire(self.prior),
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_structure_mcmc,
            names,
            cols,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
            max_parents=self.max_parents,
            **_prior_kwargs(self.prior),
        )


@dataclass(frozen=True)
class CiScreenedPosterior:
    """Structure-MCMC graph posterior restricted to the pairs a PC skeleton screen keeps.

    A static PC search with the declared test first removes pairs it judges independent, then
    structure MCMC samples DAGs over the remaining pairs. The screen is a frequentist filter,
    not evidence against an edge: a weak true edge the test fails to detect is excluded from
    the proposal set, so the posterior cannot put mass on it. Optional soft weights are
    recorded as diagnostics, not used as truth.

    Fields:

    * ``alpha`` -- significance level of each conditional-independence test. Default ``0.05``.
    * ``fdr`` -- adjust the test p-values by Benjamini-Hochberg before comparing them with
      ``alpha``. Default ``True``.
    * ``ci`` -- the conditional-independence test: a registered name (``"parcorr"``, the default
      partial-correlation test for linear Gaussian data; also ``"robust_parcorr"``,
      ``"gsquared"`` for discrete data, ``"knn_dependence"``, ``"gpdc"``, ...). An unknown name
      raises with ``invalid_argument``.
    * ``max_cond_size`` -- the largest conditioning set a test uses. Default ``2``.
    * ``soft_weight`` -- ``"none"`` (default), ``"bayes_factor"`` or
      ``"posterior_dependence"``: proposal weights recorded as diagnostic notes.
    * ``n_chains``, ``n_warmup``, ``n_draws``, ``thin`` -- the MCMC schedule. Defaults ``2``,
      ``300``, ``600`` and ``1``.
    """

    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    max_cond_size: int = 2
    soft_weight: Literal["none", "bayes_factor", "posterior_dependence"] = "none"
    n_chains: int = 2
    n_warmup: int = 300
    n_draws: int = 600
    thin: int = 1
    prior: StructurePrior | None = None
    kind: Literal["ci_screened_posterior"] = "ci_screened_posterior"

    algorithm_id: ClassVar[str] = "ci_screened_posterior"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "ci_screened_posterior",
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
            "soft_weight": self.soft_weight,
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            "thin": self.thin,
            **_prior_wire(self.prior),
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols = coerce_data(data)
        return _run_native(
            execution,
            _discover_ci_screened_posterior,
            names,
            cols,
            alpha=self.alpha,
            fdr=self.fdr,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
            soft_weight=self.soft_weight,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
            **_prior_kwargs(self.prior),
        )


@dataclass(frozen=True)
class DbnPosterior:
    """Bounded-lag DBN posterior (Gaussian BIC).

    Exact enumeration only when ``p ≤ 4`` and ``max_lag ≤ 2``; larger templates
    automatically use MCMC (or set ``force_mcmc=True``).

    Standalone: ``DbnPosterior(...).run(...)`` returns a ``GraphPosterior``.
    Composed: pass ``discovery=DbnPosterior(...)`` with ``inference=Bayesian(...)``
    and ``PulseEffect``/``SustainedEffect`` to mix temporal effect draws, or
    with ``inference=Frequentist()`` to mix atom estimates under a shared
    circular-block SE whose replicate count follows ``bootstrap`` / the latency tier.

    ``prior`` (a :class:`StructurePrior`) prices each ``(from, to)`` pair alike at the
    contemporaneous edge and at every lag.
    """

    max_lag: int = 1
    force_mcmc: bool = False
    n_chains: int = 2
    n_warmup: int = 200
    n_draws: int = 400
    prior: StructurePrior | None = None
    kind: Literal["dbn_posterior"] = "dbn_posterior"

    algorithm_id: ClassVar[str] = "dbn_posterior"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "dbn_posterior",
            "max_lag": self.max_lag,
            "force_mcmc": self.force_mcmc,
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            **_prior_wire(self.prior),
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        names, cols, unit_lengths = coerce_temporal_data(data)
        return _run_native(
            execution,
            _discover_dbn_posterior,
            names,
            cols,
            unit_lengths=unit_lengths,
            max_lag=self.max_lag,
            force_mcmc=self.force_mcmc,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            **_prior_kwargs(self.prior),
        )


@dataclass(frozen=True)
class ScreenedExactMarginals:
    """Exact edge marginals of the neighbourhood a CI screen picks around variables of interest.

    The exact marginals stop at 15 variables, but they apply per question: the conditional-
    independence screen runs on every variable, ``focus`` and the variables nearest to it in the
    screen's skeleton (at most ``max_variables``, itself at most 15) are selected, and the exact
    dynamic programme runs on them with the screen as a per-edge prior: both directions of a pair
    the screen kept get ``in_probability``, every other pair ``out_probability``. Probabilities a
    ``prior`` (:class:`StructurePrior`) lists for an edge keep their value.

    ``run`` returns :class:`ExactDagMarginals` over the selected variables, with
    ``conditional_on_screen`` stating that every number is conditional on the screen: variables
    outside the set are not conditioned on, and no marginal is that of the full posterior.
    """

    focus: tuple[str, ...]
    max_variables: int = 12
    in_probability: float = 0.5
    out_probability: float = 0.02
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    max_cond_size: int = 2
    prior: StructurePrior | None = None
    kind: Literal["screened_exact_marginals"] = "screened_exact_marginals"

    algorithm_id: ClassVar[str] = "screened_exact_marginals"

    def __post_init__(self) -> None:
        focus = (self.focus,) if isinstance(self.focus, str) else tuple(self.focus)
        if not focus:
            raise CausalValueError("focus must name at least one variable")
        for name in focus:
            _variable_name(name, "a focus variable")
        object.__setattr__(self, "focus", focus)
        _probability(self.in_probability, "in_probability")
        _probability(self.out_probability, "out_probability")
        limit = EXACT_MARGINALS_MAX_VARIABLES
        _count(self.max_variables, "max_variables", minimum=len(focus), maximum=limit)

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "screened_exact_marginals",
            "focus": list(self.focus),
            "max_variables": self.max_variables,
            "in_probability": self.in_probability,
            "out_probability": self.out_probability,
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
            **_prior_wire(self.prior),
        }

    def run(self, data: Any, *, execution: Execution | None = None) -> ExactDagMarginals:
        names, cols = coerce_data(data)
        selected, flat, log_evidence, statement = _run_native(
            execution,
            _discover_screened_exact_marginals,
            names,
            cols,
            list(self.focus),
            max_variables=self.max_variables,
            in_probability=self.in_probability,
            out_probability=self.out_probability,
            alpha=self.alpha,
            fdr=self.fdr,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
            **_prior_kwargs(self.prior),
        )
        n = len(selected)
        matrix = tuple(tuple(flat[i * n : (i + 1) * n]) for i in range(n))
        return ExactDagMarginals(list(selected), n, log_evidence, matrix, statement)


@dataclass(frozen=True)
class EquivalenceClassPosterior:
    """Posterior over Markov equivalence classes (CPDAGs) of a DAG posterior.

    The wrapped search runs unchanged. Every DAG it lists is replaced by its CPDAG and the
    posterior mass of the DAGs that share a CPDAG is summed, so the answer is a
    :class:`GraphPosterior` whose ``atom_kind`` is ``"Cpdag"``: both directed bits of an
    unordered pair mark an undirected edge, and ``edge_marginals`` count an undirected edge in
    both directions. The class probabilities are exact functions of the DAG posterior. The
    per-edge Monte-Carlo errors of a sampled DAG posterior do not carry over to class bits and
    are dropped; the chain diagnostics of the DAG draws stay on ``converged``.

    ``base`` is an :class:`ExactDagPosterior` (at most six variables), :class:`OrderMcmc`,
    :class:`StructureMcmc`, :class:`CiScreenedPosterior` or the temporal :class:`DbnPosterior`.
    A template's class is that of its one-step window: lagged edges are oriented by time and
    compel what colliders and Meek's rules compel of the contemporaneous edges; lagged parents
    are treated as exogenous roots of the window. The exact marginals route lists no graphs and
    is refused.
    """

    base: ExactDagPosterior | OrderMcmc | StructureMcmc | CiScreenedPosterior | DbnPosterior
    kind: Literal["class_posterior"] = "class_posterior"

    algorithm_id: ClassVar[str] = "class_posterior"

    def __post_init__(self) -> None:
        if not isinstance(
            self.base,
            (ExactDagPosterior, OrderMcmc, StructureMcmc, CiScreenedPosterior, DbnPosterior),
        ):
            raise CausalUnsupportedError(
                "a class posterior collapses the graphs of ExactDagPosterior, OrderMcmc, "
                "StructureMcmc, CiScreenedPosterior or DbnPosterior; the exact edge marginals "
                "list no graphs. Next attempt: pass one of those as base",
                reason_code="option_not_applicable",
            )

    def _wire(self) -> dict[str, Any]:
        return {"algorithm": "class_posterior", "base": self.base._wire()}

    def run(self, data: Any, *, execution: Execution | None = None) -> GraphPosterior:
        """Run the posterior on the data and return a :class:`GraphPosterior`.

        Read ``weights`` and ``edge_marginals`` (or ``edge_marginal_matrix()``) first, then the
        convergence diagnostics where the method reports them. ``execution`` sets the seed and
        threads.
        """
        base = self.base
        if isinstance(base, DbnPosterior):
            names, cols, unit_lengths = coerce_temporal_data(data)
            return _run_native(
                execution,
                _discover_dbn_posterior,
                names,
                cols,
                unit_lengths=unit_lengths,
                max_lag=base.max_lag,
                force_mcmc=base.force_mcmc,
                n_chains=base.n_chains,
                n_warmup=base.n_warmup,
                n_draws=base.n_draws,
                equivalence_classes=True,
                **_prior_kwargs(base.prior),
            )
        names, cols = coerce_data(data)
        if isinstance(base, ExactDagPosterior):
            if len(names) > EXACT_ENUMERATION_MAX_VARIABLES:
                raise CausalUnsupportedError(
                    f"the exact posterior lists its graphs for at most "
                    f"{EXACT_ENUMERATION_MAX_VARIABLES} variables; {len(names)} were given. "
                    "Next attempt: use OrderMcmc or StructureMcmc as base",
                    reason_code="option_not_applicable",
                )
            return _run_native(
                execution,
                _discover_exact_dag_posterior,
                names,
                cols,
                equivalence_classes=True,
                **_prior_kwargs(base.prior),
            )
        if isinstance(base, OrderMcmc):
            return _run_native(
                execution,
                _discover_order_mcmc,
                names,
                cols,
                n_chains=base.n_chains,
                n_warmup=base.n_warmup,
                n_draws=base.n_draws,
                thin=base.thin,
                require_diagnostics_gate=base.require_diagnostics_gate,
                **({} if base.max_parents is None else {"max_parents": base.max_parents}),
                equivalence_classes=True,
                noise=base.noise,
                **_prior_kwargs(base.prior),
            )
        if isinstance(base, StructureMcmc):
            return _run_native(
                execution,
                _discover_structure_mcmc,
                names,
                cols,
                n_chains=base.n_chains,
                n_warmup=base.n_warmup,
                n_draws=base.n_draws,
                thin=base.thin,
                **({} if base.max_parents is None else {"max_parents": base.max_parents}),
                equivalence_classes=True,
                **_prior_kwargs(base.prior),
            )
        return _run_native(
            execution,
            _discover_ci_screened_posterior,
            names,
            cols,
            alpha=base.alpha,
            fdr=base.fdr,
            ci=base.ci,
            max_cond_size=base.max_cond_size,
            soft_weight=base.soft_weight,
            n_chains=base.n_chains,
            n_warmup=base.n_warmup,
            n_draws=base.n_draws,
            thin=base.thin,
            equivalence_classes=True,
            **_prior_kwargs(base.prior),
        )


@dataclass(frozen=True)
class BootstrapStability:
    """Sampling distribution of a frequentist structure search over resamples.

    The wrapped search is re-run on ``replicates`` resamples drawn under the data's sampling
    design (independent rows for a table, circular blocks for a series, whole units for a
    panel) and the result tabulates how often every node pair was adjacent
    (``result.edges``), carried every pair of endpoint marks, and how often the search
    returned every whole structure (``result.structures``). A resample on which the search
    fails is counted in ``replicates_failed`` and never redrawn; fewer than half succeeding
    is refused.

    The frequencies are a bootstrap distribution of the search, not a posterior. The
    structure the search returns on the full sample is ``result.reference_edges``. The
    conditional-independence test must be a named test, and a table with a missing searched
    value is refused (drop the rows first).

    ``base`` is :class:`PC`, :class:`FCI`, :class:`RFCI`, :class:`GES`, :class:`LiNGAM`,
    :class:`NOTEARS`, :class:`PCMCI` or :class:`PCMCIPlus`.
    """

    base: (
        PC | FCI | RFCI | GES | LiNGAM | NOTEARS | PCMCI | PCMCIPlus | LPCMCI | JPCMCIPlus | RPCMCI
    )
    replicates: int = 200
    kind: Literal["bootstrap_stability"] = "bootstrap_stability"

    algorithm_id: ClassVar[str] = "bootstrap_stability"

    def __post_init__(self) -> None:
        if not isinstance(
            self.base,
            (PC, FCI, RFCI, GES, LiNGAM, NOTEARS, PCMCI, PCMCIPlus, LPCMCI, JPCMCIPlus, RPCMCI),
        ):
            raise CausalUnsupportedError(
                "a stability search re-runs PC, FCI, RFCI, GES, LiNGAM, NOTEARS, PCMCI, "
                "PCMCIPlus, LPCMCI, JPCMCIPlus or RPCMCI; a posterior has its own sampling "
                "error. Next attempt: pass one of the supported searches as base",
                reason_code="option_not_applicable",
            )
        if isinstance(self.replicates, bool) or not (
            isinstance(self.replicates, int) and 2 <= self.replicates <= 10_000
        ):
            raise CausalValueError(
                f"replicates must be an integer in [2, 10000]; got {self.replicates!r}",
                reason_code="invalid_argument",
            )

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "bootstrap_stability",
            "base": self.base._wire(),
            "replicates": self.replicates,
        }

    def run(
        self,
        data: Any,
        *,
        execution: Execution | None = None,
        regimes: Sequence[int] | None = None,
    ) -> StructureStabilityResult | RegimeStabilityResult:
        """Re-run the search and tabulate what it returned.

        ``JPCMCIPlus`` reads ``data`` as the pair ``(names, env_columns)`` of its own ``run``,
        every environment resampled in circular blocks on its own. ``RPCMCI`` needs the
        ``regimes`` label per row and returns a :class:`RegimeStabilityResult`, one table per
        regime, from resamples that carry their labels.
        """
        base = self.base
        ci = getattr(base, "ci", "parcorr")
        if not isinstance(ci, str):
            raise CausalUnsupportedError(
                "a stability search takes a named conditional-independence test; a callable is "
                "re-entered once per resample. Next attempt: pass the test by name",
                reason_code="option_not_applicable",
            )
        common: dict[str, Any] = {"replicates": self.replicates}
        if isinstance(base, JPCMCIPlus):
            names, env_columns = data
            return _run_native(
                execution,
                _discover_jpcmci_plus_stability,
                list(names),
                [list(cols) for cols in env_columns],
                max_lag=base.max_lag,
                alpha=base.alpha,
                fdr=base.fdr,
                ci=ci,
                context_names=list(base.context_names) if base.context_names else None,
                max_cond_size=base.max_cond_size,
                **common,
            )
        if isinstance(base, RPCMCI):
            if regimes is None:
                raise CausalValueError(
                    "RPCMCI stability needs the regime label of every row",
                    reason_code="required_option_missing",
                )
            names, cols = coerce_data(data, temporal=True)
            return _run_native(
                execution,
                _discover_rpcmci_stability,
                names,
                cols,
                regimes=list(regimes),
                max_lag=base.max_lag,
                alpha=base.alpha,
                fdr=base.fdr,
                ci=ci,
                max_cond_size=base.max_cond_size,
                **common,
            )
        if isinstance(base, (PCMCI, PCMCIPlus, LPCMCI)):
            names, cols, unit_lengths = coerce_temporal_data(data)
            with _named_positions(names):
                return _run_native(
                    execution,
                    _discover_bootstrap_stability,
                    names,
                    cols,
                    base="lpcmci"
                    if isinstance(base, LPCMCI)
                    else ("pcmci" if isinstance(base, PCMCI) else "pcmci_plus"),
                    alpha=base.alpha,
                    fdr=base.fdr,
                    ci=ci,
                    max_cond_size=base.max_cond_size,
                    max_lag=base.max_lag,
                    unit_lengths=unit_lengths,
                    **(
                        {} if isinstance(base, LPCMCI) else _knowledge_kwargs(base.knowledge, names)
                    ),
                    **common,
                )
        names, cols = coerce_data(data)
        if isinstance(base, LiNGAM):
            return _run_native(
                execution,
                _discover_bootstrap_stability,
                names,
                cols,
                base="direct_lingam",
                prune_threshold=base.prune_threshold,
                max_cond_size=base.max_cond_size,
                **common,
            )
        if isinstance(base, NOTEARS):
            return _run_native(
                execution,
                _discover_bootstrap_stability,
                names,
                cols,
                base="notears",
                l1=base.l1,
                threshold=base.threshold,
                standardize=base.standardize,
                max_cond_size=base.max_cond_size,
                missing=base.missing,
                **common,
            )
        extra: dict[str, Any] = {}
        if isinstance(base, GES):
            extra = {"screen_pc": base.screen_pc, "max_subset": base.max_subset}
        with _named_positions(names):
            return _run_native(
                execution,
                _discover_bootstrap_stability,
                names,
                cols,
                base=base.kind,
                alpha=base.alpha,
                fdr=base.fdr,
                ci=ci,
                max_cond_size=base.max_cond_size,
                **extra,
                **_knowledge_kwargs(base.knowledge, names),
                **common,
            )


# Alias: DiscoveryResult is the preferred name; PcmciDiscoveryResult kept for compat.
DiscoveryResult = PcmciDiscoveryResult


StaticDiscovery = PC | GES | LiNGAM | NOTEARS | FCI | RFCI
TemporalDiscovery = PCMCI | PCMCIPlus | LPCMCI
#: Every configuration `accepted_graph.accept_discovery` runs through a review
#: gate: the single-table and single-series algorithms plus the two that read a
#: set of datasets (J-PCMCI+) or labelled regimes (RPCMCI).
ReviewedDiscovery = StaticDiscovery | TemporalDiscovery | JPCMCIPlus | RPCMCI

_STATIC_DISCOVERY_TYPES = (PC, GES, LiNGAM, NOTEARS, FCI, RFCI)
_TEMPORAL_DISCOVERY_TYPES = (PCMCI, PCMCIPlus, LPCMCI)
_ALL_DISCOVERY_TYPES = (
    PC,
    GES,
    LiNGAM,
    NOTEARS,
    FCI,
    RFCI,
    PCMCI,
    PCMCIPlus,
    LPCMCI,
    JPCMCIPlus,
    RPCMCI,
    ExactDagPosterior,
    OrderMcmc,
    StructureMcmc,
    CiScreenedPosterior,
    DbnPosterior,
)


def discover(
    data: Any,
    *,
    method: ReviewedDiscovery,
    execution: Execution | None = None,
    accept_discovered: bool = False,
    regimes: Sequence[int] | None = None,
) -> AcceptedGraph:
    """Execute discovery explicitly through its native structure review gate.

    Pending review raises ``ReviewRequired`` unless ``accept_discovered=True``
    explicitly permits the native gate to accept it. Equivalence-class marks
    remain unresolved class information. Pass the accepted structure as
    ``prepare(data, graph=structure, ...)``; estimation never reruns discovery.

    Args:
        data: the observations the configuration reads: a table for the tabular searches, a
            series or :class:`antecedent.data.PanelFrame` for the temporal ones, and
            environments for :class:`JPCMCIPlus`.
        method: a single-graph configuration (:class:`PC`, :class:`GES`, :class:`LiNGAM`,
            :class:`NOTEARS`, :class:`FCI`, :class:`RFCI`, :class:`PCMCI`, :class:`PCMCIPlus`,
            :class:`LPCMCI`, :class:`JPCMCIPlus` or :class:`RPCMCI`).
        execution: the :class:`~antecedent.Execution` (seed, threads, cancellation, callbacks).
            Default ``Execution()``.
        accept_discovered: let the gate accept pending directed edges. Default ``False``.
        regimes: one regime label per observation; :class:`RPCMCI` only.

    Returns:
        An :class:`~antecedent.AcceptedGraph`, ready for ``graph=``.

    Raises:
        ReviewRequired: edges are pending review and ``accept_discovered`` is ``False``.
        CausalUnsupportedError: ``option_not_applicable`` for ``regimes`` with another method.
        CausalTypeError: an ``execution``, ``method`` or ``accept_discovered`` of the wrong
            type (a graph posterior is not a ``method`` here).
    """
    from .estimation import _accept_live_discovery, _Controls

    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if not isinstance(
        method, (*_STATIC_DISCOVERY_TYPES, *_TEMPORAL_DISCOVERY_TYPES, JPCMCIPlus, RPCMCI)
    ):
        raise CausalTypeError("method must be a reviewed discovery configuration")
    if not isinstance(accept_discovered, bool):
        raise CausalTypeError("accept_discovered must be a boolean")
    if regimes is not None and not isinstance(method, RPCMCI):
        raise CausalUnsupportedError(
            "regimes labels RPCMCI discovery only", reason_code="option_not_applicable"
        )
    return policy._call(
        _accept_live_discovery,
        data,
        method,
        accept_discovered=accept_discovered,
        regimes=regimes,
        seed=policy.seed,
        threads=policy.threads,
        execution=policy,
        controls=_Controls(
            cancel=policy.cancel,
            on_progress=policy.on_progress,
            on_stage=policy.on_stage,
        ),
    )


def run_static_discovery(
    data: Any,
    discovery: StaticDiscovery,
    *,
    execution: Execution | None = None,
) -> tuple[DiscoveryResult, str]:
    """Run a tabular configuration (PC, GES, LiNGAM, NOTEARS, FCI or RFCI) without review.

    Returns ``(result, algorithm_id)``: the configuration's ``run(data)`` result and its
    algorithm identifier. The ``execution`` argument is accepted but not forwarded to ``run``.
    """
    if not isinstance(discovery, _STATIC_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported static discovery type: {type(discovery)!r}")
    return discovery.run(data, execution=execution), discovery.algorithm_id


def run_temporal_discovery(
    data: Any,
    discovery: TemporalDiscovery,
    *,
    execution: Execution | None = None,
) -> tuple[DiscoveryResult, str]:
    """Run a PCMCI, PCMCI+ or LPCMCI configuration without review.

    Returns ``(result, algorithm_id)``. The ``execution`` argument is accepted but not
    forwarded to ``run``.
    """
    if not isinstance(discovery, _TEMPORAL_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported temporal discovery type: {type(discovery)!r}")
    return discovery.run(data, execution=execution), discovery.algorithm_id


def discovery_algorithm(discovery: Any) -> dict[str, Any]:
    """The settings of a discovery configuration as a plain dictionary, with its algorithm name.

    Raises :class:`~antecedent.errors.CausalTypeError` for an object that is not one of this
    module's configurations.
    """
    if not isinstance(discovery, _ALL_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported discovery config: {type(discovery)!r}")
    return discovery._wire()


def graph_posterior_map_edges(post: GraphPosterior) -> list[tuple[str, str]]:
    """Oriented edges from the maximum-weight adjacency mask in a graph posterior.

    ``post.adjacency`` holds one Python ``int`` per atom; bit
    ``from * (n_vars - 1) + (to if to < from else to - 1)`` is the edge ``from -> to``.
    The integers are not limited to 64 bits, so graphs of up to 16 variables decode.
    """
    import numpy as np

    if post.n_graphs < 1 or not post.weights:
        raise CausalValueError("GraphPosterior has no graphs")
    i = int(np.argmax(np.asarray(post.weights, dtype=np.float64)))
    mask = int(post.adjacency[i])
    n = int(post.n_vars)
    names = list(post.names)
    if len(names) < n:
        names = [f"x{j}" for j in range(n)]
    edges: list[tuple[str, str]] = []
    for fr in range(n):
        for to in range(n):
            if fr == to:
                continue
            # edge_bit(n, from, to) = from*(n-1) + (to if to < from else to-1)
            bit = fr * (n - 1) + (to if to < fr else to - 1)
            if (mask >> bit) & 1:
                edges.append((names[fr], names[to]))
    return edges


def graph_posterior_map_dag(post: GraphPosterior) -> Dag:
    """MAP DAG from a graph posterior (maximum-weight atom)."""
    from .graph import Dag

    edges = graph_posterior_map_edges(post)
    names = list(post.names) if post.names else sorted({a for e in edges for a in e})
    return Dag.from_edges(names, edges)


__all__ = [
    "BackgroundKnowledge",
    "BootstrapStability",
    "CiScreenedPosterior",
    "DbnPosterior",
    "DiscoveredLink",
    "DiscoveryResult",
    "EquivalenceClassPosterior",
    "ExactDagMarginals",
    "ExactDagPosterior",
    "ExactPagPosterior",
    "FCI",
    "GES",
    "GraphEdge",
    "GraphPosterior",
    "JPCMCIPlus",
    "LPCMCI",
    "LiNGAM",
    "Link",
    "NOTEARS",
    "OrderMcmc",
    "PC",
    "PCMCI",
    "PCMCIPlus",
    "PagMcmc",
    "PcmciDiscoveryResult",
    "RFCI",
    "RPCMCI",
    "RpcmciDiscoverySummary",
    "StructureMcmc",
    "StructurePrior",
    "RegimeStabilityResult",
    "ScreenedExactMarginals",
    "StructureStabilityResult",
    "cpdag_oriented_edges",
    "discover",
    "discovery_algorithm",
    "discovery_to_dag",
    "graph_posterior_map_dag",
    "graph_posterior_map_edges",
    "run_static_discovery",
    "run_temporal_discovery",
    "two_regime_half_split",
]
