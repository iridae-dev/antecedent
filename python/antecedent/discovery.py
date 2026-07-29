"""Discovery algorithm configuration and helpers.

Each config dataclass below owns its own wire-serialization (``_wire``),
native dispatch (``run``), and — for the algorithms that produce a holdable
graph artifact — session acceptance (``accept``). ``run_static_discovery`` /
``run_temporal_discovery`` / ``discovery_algorithm`` are thin dispatchers kept
for the external callers (``estimation.py``, ``_analyze.py``, ``gcm.py``,
``accepted_graph.py``) that already depend on their exact signatures and
return types; the per-algorithm knowledge itself lives on the config classes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ._coerce import coerce_data
from ._native import (
    DiscoveredLink,
    GraphEdge,
    GraphPosterior,
    PcmciDiscoveryResult,
    RpcmciDiscoverySummary,
    two_regime_half_split,
)
from ._native import (
    discover_ci_screened_posterior as _discover_ci_screened_posterior,
)
from ._native import (
    discover_dbn_posterior as _discover_dbn_posterior,
)
from ._native import (
    discover_exact_dag_posterior as _discover_exact_dag_posterior,
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
    discover_pc as _discover_pc,
)
from ._native import (
    discover_pcmci as _discover_pcmci,
)
from ._native import (
    discover_pcmci_plus as _discover_pcmci_plus,
)
from ._native import (
    discover_rfci as _discover_rfci,
)
from ._native import (
    discover_rpcmci as _discover_rpcmci,
)
from ._native import (
    discover_structure_mcmc as _discover_structure_mcmc,
)
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

if TYPE_CHECKING:
    from .accepted_graph import AcceptedGraph
    from .graph import Cpdag, Dag

CiSpec = str | Callable[..., Sequence[tuple[float, float]]]


def _ci_str(ci: CiSpec) -> str:
    """Collapse a possibly-callable ``ci`` to a plain string.

    Only the ``run_static_discovery`` / ``run_temporal_discovery`` dispatchers
    apply this, because the ``analyze`` and ``AcceptedGraph.rediscover`` paths
    behind them have never carried a Python callable across. ``Config.run()``
    forwards ``ci`` raw — native supports a callable there (``ci_name`` comes
    back as ``python.callback``), and collapsing it would silently substitute
    a different independence test.
    """
    return ci if isinstance(ci, str) else "parcorr"


@dataclass(frozen=True)
class PC:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["pc"] = "pc"

    algorithm_id: ClassVar[str] = "pc"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "pc",
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_pc(
            names,
            cols,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            threads=threads,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class PCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["pcmci"] = "pcmci"

    algorithm_id: ClassVar[str] = "pcmci"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "pcmci",
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
        seed: int = 1,
        threads: int = 1,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_pcmci(
            names,
            cols,
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            ci=self.ci,
            weights=weights,
            threads=threads,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class PCMCIPlus:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["pcmci_plus"] = "pcmci_plus"

    # NB: asymmetric on purpose. ``run_temporal_discovery`` has always
    # returned the short form "pcmci+" as this config's algorithm id, while
    # ``discovery_algorithm``/``_wire()`` has always used the long form
    # "pcmci_plus" as the native "algorithm" wire value. Both are preserved.
    algorithm_id: ClassVar[str] = "pcmci+"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "pcmci_plus",
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
        seed: int = 1,
        threads: int = 1,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_pcmci_plus(
            names,
            cols,
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            ci=self.ci,
            weights=weights,
            threads=threads,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class LPCMCI:
    max_lag: int = 1
    alpha: float = 0.05
    fdr: bool = True
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
        seed: int = 1,
        threads: int = 1,
        weights: list[float] | None = None,
    ) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_lpcmci(
            names,
            cols,
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            ci=self.ci,
            weights=weights,
            threads=threads,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class JPCMCIPlus:
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
        seed: int = 1,
        threads: int = 1,
        weights: list[float] | None = None,
    ) -> PcmciDiscoveryResult:
        """Multi-environment discovery: takes ``names``/``env_columns`` directly.

        Unlike the single-table configs, J-PCMCI+ has no single-table ``data``
        shape to coerce (``coerce_data`` deliberately does not accept
        ``MultiEnvFrame`` — see its docstring); build ``env_columns`` with
        :func:`antecedent._data.as_multi_env_columns` or
        :func:`antecedent.data.multi_env`.
        """
        return _discover_jpcmci_plus(
            list(names),
            [list(cols) for cols in env_columns],
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            ci=self.ci,
            weights=weights,
            threads=threads,
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
        names: list[str],
        env_columns: Sequence[Sequence[Any]],
        *,
        seed: int = 1,
        threads: int = 1,
    ) -> AcceptedGraph:
        """Not supported yet — raises rather than silently building a wrong graph.

        Every sibling ``accept()`` runs discovery then calls
        ``AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)``.
        That dispatch only recognizes the short algorithm ids ``"pcmci"`` /
        ``"pcmci+"`` / ``"lpcmci"`` as temporal (lagged) results; this config's
        id (``"jpcmci_plus"``) would fall through to the *static* graph path,
        which would silently misinterpret a lagged, multi-environment result
        as an unlagged one — worse than an ``AttributeError``. Teaching
        ``AcceptedGraph.from_discovery`` about J-PCMCI+'s result shape is the
        right long-term fix; it is out of scope here. Call :meth:`run`
        directly and hold the result yourself in the meantime.
        """
        raise CausalUnsupportedError(
            "JPCMCIPlus.accept() is not supported yet: AcceptedGraph.from_discovery's "
            "dispatch only recognizes 'pcmci'/'pcmci+'/'lpcmci' as temporal (lagged) "
            "algorithm ids; this config's 'jpcmci_plus' id would fall through to the "
            "static graph path and silently misinterpret the lagged, multi-environment "
            "result. Call .run(...) directly and hold the result yourself."
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
        seed: int = 1,
        threads: int = 1,
        weights: list[float] | None = None,
    ) -> RpcmciDiscoverySummary:
        """``regimes`` is required (length = series length); no silent half-split.

        Call ``two_regime_half_split(len(series))`` for an explicit two-regime mid-point split.
        """
        names, cols = coerce_data(data)
        return _discover_rpcmci(
            names,
            cols,
            regimes=list(regimes),
            max_lag=self.max_lag,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            ci=self.ci,
            weights=weights,
            threads=threads,
            max_cond_size=self.max_cond_size,
        )

    def accept(
        self,
        data: Any,
        *,
        regimes: Sequence[int],
        seed: int = 1,
        threads: int = 1,
    ) -> AcceptedGraph:
        """Not supported — ``RpcmciDiscoverySummary`` carries no edge-level detail.

        Unlike every other discovery config's result, :meth:`run`'s
        ``RpcmciDiscoverySummary`` exposes only regime-level edge *counts*
        (``directed_edges`` / ``undirected_edges``: ``list[int]``, one count
        per regime) — no node names and no individual edge endpoints. There
        is nothing here from which :class:`antecedent.AcceptedGraph` could
        build a graph artifact, so this raises rather than silently returning
        something wrong (or a bare ``AttributeError``). Call :meth:`run`
        directly and consume the summary, or use ``PCMCI`` / ``PCMCIPlus`` /
        ``LPCMCI`` per-regime if a holdable structure is needed.
        """
        raise CausalUnsupportedError(
            "RPCMCI.accept() is not supported: RpcmciDiscoverySummary carries only "
            "regime-level edge counts (directed_edges/undirected_edges: list[int]), "
            "not node names or edge endpoints, so there is no edge detail to build "
            "an AcceptedGraph from. Call .run(...) directly and consume the summary, "
            "or use PCMCI/PCMCIPlus/LPCMCI for a holdable structure."
        )


@dataclass(frozen=True)
class GES:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    screen_pc: bool = False
    max_subset: int | None = None
    kind: Literal["ges"] = "ges"

    algorithm_id: ClassVar[str] = "ges"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "ges",
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_ges(
            names,
            cols,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            threads=threads,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
            screen_pc=self.screen_pc,
            max_subset=self.max_subset,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class LiNGAM:
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

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_lingam(
            names,
            cols,
            prune_threshold=self.prune_threshold,
            seed=seed,
            max_cond_size=self.max_cond_size,
            threads=threads,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class NOTEARS:
    l1: float = 0.1
    threshold: float = 0.3
    standardize: bool = True
    max_cond_size: int = 8
    kind: Literal["notears"] = "notears"

    algorithm_id: ClassVar[str] = "notears"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "notears",
            "lambda": self.l1,
            "threshold": self.threshold,
            "standardize": self.standardize,
            "max_cond_size": self.max_cond_size,
            "alpha": 0.05,
            "fdr": True,
            "ci": "parcorr",
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_notears(
            names,
            cols,
            l1=self.l1,
            threshold=self.threshold,
            standardize=self.standardize,
            seed=seed,
            max_cond_size=self.max_cond_size,
            threads=threads,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class FCI:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["fci"] = "fci"

    algorithm_id: ClassVar[str] = "fci"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "fci",
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_fci(
            names,
            cols,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            threads=threads,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class RFCI:
    alpha: float = 0.05
    fdr: bool = True
    ci: CiSpec = "parcorr"
    max_cond_size: int = 2
    kind: Literal["rfci"] = "rfci"

    algorithm_id: ClassVar[str] = "rfci"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "rfci",
            "alpha": self.alpha,
            "fdr": self.fdr,
            "ci": self.ci,
            "max_cond_size": self.max_cond_size,
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> DiscoveryResult:
        names, cols = coerce_data(data)
        return _discover_rfci(
            names,
            cols,
            alpha=self.alpha,
            fdr=self.fdr,
            seed=seed,
            threads=threads,
            ci=self.ci,
            max_cond_size=self.max_cond_size,
        )

    def accept(self, data: Any, *, seed: int = 1, threads: int = 1) -> AcceptedGraph:
        from .accepted_graph import AcceptedGraph

        result = self.run(data, seed=seed, threads=threads)
        return AcceptedGraph.from_discovery(result, algorithm_id=self.algorithm_id)


@dataclass(frozen=True)
class ExactDagPosterior:
    """Exact DAG posterior enumeration (hard limit: n ≤ 6, Gaussian BIC).

    For more variables use ``OrderMcmc``, ``StructureMcmc``, or ``CiScreenedPosterior``.

    Standalone: ``ExactDagPosterior().run(...)`` returns a ``GraphPosterior``.
    Composed: pass ``discovery=ExactDagPosterior()`` with ``inference=Bayesian(...)``
    to ``analyze`` to mix effect draws over the graph posterior (P1-D).
    """

    kind: Literal["exact_dag_posterior"] = "exact_dag_posterior"

    algorithm_id: ClassVar[str] = "exact_dag_posterior"

    def _wire(self) -> dict[str, Any]:
        return {"algorithm": "exact_dag_posterior"}

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> GraphPosterior:
        names, cols = coerce_data(data)
        return _discover_exact_dag_posterior(names, cols, seed=seed, threads=threads)


@dataclass(frozen=True)
class OrderMcmc:
    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    require_diagnostics_gate: bool = True
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
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> GraphPosterior:
        names, cols = coerce_data(data)
        return _discover_order_mcmc(
            names,
            cols,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
            require_diagnostics_gate=self.require_diagnostics_gate,
            seed=seed,
            threads=threads,
        )


@dataclass(frozen=True)
class StructureMcmc:
    """Structure-MCMC graph posterior.

    Unlike its sibling :class:`OrderMcmc`, this config has **no**
    ``require_diagnostics_gate`` field: the native ``discover_structure_mcmc``
    entry point takes no such parameter, so there is nothing here to plumb it
    through to. Concretely, this means R-hat / ESS convergence diagnostics are
    **never gated** on a ``StructureMcmc`` posterior the way they can be on
    ``OrderMcmc(require_diagnostics_gate=True)`` (the default there) — a
    non-converged chain's posterior is returned exactly the same as a
    converged one. This package cannot add the parameter from the Python side
    (it would require a change to the Rust ``discover_structure_mcmc``
    signature, which is out of scope for a ``python/antecedent`` change); the
    long-term fix is adding a ``require_diagnostics_gate`` parameter to
    ``discover_structure_mcmc`` in Rust to match ``discover_order_mcmc``, then
    threading it through here the same way :class:`OrderMcmc` already does.
    Callers that need a diagnostics gate today should use
    ``GraphPosterior.converged`` / ``.rejected_invalid`` /
    ``.ess`` on the returned posterior themselves, or prefer ``OrderMcmc``.
    """

    n_chains: int = 4
    n_warmup: int = 500
    n_draws: int = 1000
    thin: int = 1
    kind: Literal["structure_mcmc"] = "structure_mcmc"

    algorithm_id: ClassVar[str] = "structure_mcmc"

    def _wire(self) -> dict[str, Any]:
        return {
            "algorithm": "structure_mcmc",
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "mcmc_draws": self.n_draws,
            "thin": self.thin,
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> GraphPosterior:
        names, cols = coerce_data(data)
        return _discover_structure_mcmc(
            names,
            cols,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            thin=self.thin,
            seed=seed,
            threads=threads,
        )


@dataclass(frozen=True)
class CiScreenedPosterior:
    alpha: float = 0.05
    fdr: bool = True
    ci: str = "parcorr"
    max_cond_size: int = 2
    soft_weight: Literal["none", "bayes_factor", "posterior_dependence"] = "none"
    n_chains: int = 2
    n_warmup: int = 300
    n_draws: int = 600
    thin: int = 1
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
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> GraphPosterior:
        names, cols = coerce_data(data)
        return _discover_ci_screened_posterior(
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
            seed=seed,
            threads=threads,
        )


@dataclass(frozen=True)
class DbnPosterior:
    """Bounded-lag DBN posterior (Gaussian BIC).

    Exact enumeration only when ``p ≤ 4`` and ``max_lag ≤ 2``; larger templates
    automatically use MCMC (or set ``force_mcmc=True``).

    Standalone: ``DbnPosterior(...).run(...)`` returns a ``GraphPosterior``.
    Composed: pass ``discovery=DbnPosterior(...)`` with ``inference=Bayesian(...)``
    and ``PulseEffect``/``SustainedEffect`` to mix temporal effect draws (P1-D).
    """

    max_lag: int = 1
    force_mcmc: bool = False
    n_chains: int = 2
    n_warmup: int = 200
    n_draws: int = 400
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
        }

    def run(self, data: Any, *, seed: int = 1, threads: int = 1) -> GraphPosterior:
        names, cols = coerce_data(data)
        return _discover_dbn_posterior(
            names,
            cols,
            max_lag=self.max_lag,
            force_mcmc=self.force_mcmc,
            n_chains=self.n_chains,
            n_warmup=self.n_warmup,
            n_draws=self.n_draws,
            seed=seed,
            threads=threads,
        )


# Alias: DiscoveryResult is the preferred name; PcmciDiscoveryResult kept for compat.
DiscoveryResult = PcmciDiscoveryResult


def discovery_to_dag(result: DiscoveryResult) -> Dag:
    """Build a ``Dag`` from a discovery result's directed ``graph_edges``.

    Raises ``ValueError`` if any undirected/circle marks remain.
    """
    from .graph import Dag

    names: list[str] = []
    seen: set[str] = set()
    directed: list[tuple[str, str]] = []
    for e in result.graph_edges:
        for n in (e.source, e.target):
            if n not in seen:
                seen.add(n)
                names.append(n)
        if e.at_source == "tail" and e.at_target == "arrow":
            directed.append((e.source, e.target))
        elif e.at_source == "arrow" and e.at_target == "tail":
            directed.append((e.target, e.source))
        else:
            raise CausalValueError(
                f"cannot coerce edge {e.source}->{e.target} "
                f"({e.at_source}/{e.at_target}) into a DAG; "
                "use graph_edges or a CPDAG/PAG constructor"
            )
    return Dag.from_edges(names, directed)


StaticDiscovery = PC | GES | LiNGAM | NOTEARS | FCI | RFCI
TemporalDiscovery = PCMCI | PCMCIPlus | LPCMCI

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


def _without_callable_ci(discovery: Any) -> Any:
    """Config with a callable ``ci`` collapsed to its string form.

    Preserves the dispatcher paths' long-standing behavior; ``run()`` itself
    forwards a callable unchanged.
    """
    ci = getattr(discovery, "ci", None)
    if ci is None or isinstance(ci, str):
        return discovery
    return replace(discovery, ci=_ci_str(ci))


def run_static_discovery(
    data: Any,
    discovery: StaticDiscovery,
    *,
    seed: int = 1,
    threads: int = 1,
) -> tuple[DiscoveryResult, str]:
    """Dispatch a static discovery config to its ``run()``.

    Single source of truth for PC/GES/LiNGAM/NOTEARS/FCI/RFCI used by
    ``analyze``, ``AcceptedGraph``, and GCM compose helpers.
    """
    if not isinstance(discovery, _STATIC_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported static discovery type: {type(discovery)!r}")
    runnable = _without_callable_ci(discovery)
    return runnable.run(data, seed=seed, threads=threads), discovery.algorithm_id


def run_temporal_discovery(
    data: Any,
    discovery: TemporalDiscovery,
    *,
    seed: int = 1,
    threads: int = 1,
) -> tuple[DiscoveryResult, str]:
    """Dispatch a PCMCI-family discovery config to its ``run()``."""
    if not isinstance(discovery, _TEMPORAL_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported temporal discovery type: {type(discovery)!r}")
    runnable = _without_callable_ci(discovery)
    return runnable.run(data, seed=seed, threads=threads), discovery.algorithm_id


def discovery_algorithm(discovery: Any) -> dict[str, Any]:
    """Serialize a discovery config dataclass into kwargs for native analyze paths."""
    if not isinstance(discovery, _ALL_DISCOVERY_TYPES):
        raise CausalTypeError(f"unsupported discovery config: {type(discovery)!r}")
    return discovery._wire()


def graph_posterior_map_edges(post: GraphPosterior) -> list[tuple[str, str]]:
    """Oriented edges from the maximum-weight adjacency mask in a graph posterior."""
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


def cpdag_oriented_edges(cpdag: Cpdag, *, require_oriented: bool = True) -> list[tuple[str, str]]:
    """Return directed edges from a CPDAG; error if undirected remain when required."""
    from .graph import Cpdag

    if not isinstance(cpdag, Cpdag):
        raise CausalTypeError(f"expected Cpdag, got {type(cpdag)!r}")
    directed: list[tuple[str, str]] = []
    undirected = 0
    for src, tgt, kind in cpdag.edges():
        if kind == "directed":
            directed.append((src, tgt))
        elif kind == "undirected":
            undirected += 1
        else:
            undirected += 1
    if undirected and require_oriented:
        raise CausalValueError(
            f"CPDAG has {undirected} undirected/ambiguous edge(s); orient before "
            "PathSpecific/Interventional queries (require_oriented=True)"
        )
    if require_oriented:
        try:
            dag = cpdag.try_into_dag()
        except Exception as exc:  # noqa: BLE001
            raise CausalValueError(
                "CPDAG is not fully oriented; cannot coerce to DAG for path/distribution queries"
            ) from exc
        return list(dag.edges())
    return directed


__all__ = [
    "CiScreenedPosterior",
    "DbnPosterior",
    "DiscoveredLink",
    "DiscoveryResult",
    "ExactDagPosterior",
    "FCI",
    "GES",
    "GraphEdge",
    "GraphPosterior",
    "JPCMCIPlus",
    "LPCMCI",
    "LiNGAM",
    "NOTEARS",
    "OrderMcmc",
    "PC",
    "PCMCI",
    "PCMCIPlus",
    "PcmciDiscoveryResult",
    "RFCI",
    "RPCMCI",
    "RpcmciDiscoverySummary",
    "StructureMcmc",
    "cpdag_oriented_edges",
    "discovery_algorithm",
    "discovery_to_dag",
    "graph_posterior_map_dag",
    "graph_posterior_map_edges",
    "run_static_discovery",
    "run_temporal_discovery",
    "two_regime_half_split",
]
