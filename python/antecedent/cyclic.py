# SPDX-License-Identifier: MIT OR Apache-2.0
"""Nonlinear cyclic equilibrium SCMs with retained Rust execution and replay.

Equations have the form ``x[i] = intercept + activation(weights @ x) + u[i]``.
Noise covariance is joint and may be correlated. Fitting requires independent
observational/interventional equilibrium replicates with known hard clamps.
Answers are conditional on the specified/fitted model; no nonparametric causal
identification or nominal interval calibration is implied.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from . import _native


@dataclass(frozen=True)
class Equation:
    """Named additive-noise mechanism, optionally a compositional expression."""

    id: str
    intercept: float
    weights: tuple[float, ...]
    activation: Literal["identity", "tanh", "sin", "relu"] = "tanh"
    expression: dict[str, Any] | None = None
    parameters: tuple[float, ...] = ()

    @classmethod
    def compositional(
        cls,
        id: str,
        order: int,
        expression: dict[str, Any],
        *,
        parameters: tuple[float, ...] = (),
        intercept: float = 0.0,
    ) -> Equation:
        """Build a portable expression: add/multiply/tanh/sin/softplus and reads.

        Leaves are ``{"op": "variable", "variable": 0}``, ``parameter`` reads,
        or finite ``constant`` values. Add uses ``terms``; multiply uses
        ``factors``; unary operations use ``argument``. Rust checks bindings,
        expression limits and domain-wide derivative bounds before execution.
        """
        return cls(id, intercept, (0.0,) * order, "identity", expression, parameters)


@dataclass(frozen=True)
class SolvePolicy:
    """Numerical policy; initialization never supplies a selection law."""

    tolerance: float = 1e-10
    max_iterations: int = 1000
    require_contraction: bool = True


@dataclass(frozen=True)
class FitOptions:
    """Local coordinate likelihood fit with fixed joint noise covariance."""

    initial_step: float = 0.2
    tolerance: float = 1e-5
    max_sweeps: int = 500
    l1_penalty: float = 0.0


@dataclass(frozen=True)
class CyclicResult:
    """Model-conditional worlds, diagnostics, and independent replay artifact."""

    report: dict[str, Any]
    artifact: bytes

    @property
    def identity(self) -> str:
        """Request digest to retain separately from the artifact."""
        return str(self.report["identity"])

    def export(self) -> bytes:
        """Return the checksummed native artifact."""
        return self.artifact


def consume_artifact(
    artifact: bytes, *, expected_identity: str, max_bytes: int | None = None, cancel: Any = None
) -> CyclicResult:
    """Check identity/integrity and recompute in Rust, including any retained fit."""
    report = json.loads(
        _native.consume_cyclic_system(
            artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
        )
    )
    return CyclicResult(report, artifact)


@dataclass(frozen=True)
class EquilibriumModel:
    """Serializable nonlinear mechanisms and row-major disturbance covariance."""

    equations: tuple[Equation, ...]
    noise_covariance: tuple[float, ...]

    def isolate(
        self,
        domain: list[dict[str, float]],
        noise: list[float],
        *,
        clamps: list[dict[str, Any]] | None = None,
        max_boxes: int = 10_000,
        minimum_width: float = 1e-8,
    ) -> PreparedCyclicFamily:
        """Bounded root isolation for eligible polynomial mechanism compositions.

        The compiler preserves exact coefficients, including FMA product tails.
        Smooth non-polynomial operations and ReLU domains crossing a kink refuse
        this path. Unresolved boxes remain explicit; no root probabilities are
        inferred from their counts or solver initialization.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "equilibrium_roots",
                    "model": asdict(self),
                    "domain": domain,
                    "noise": noise,
                    "clamps": clamps or [],
                    "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                },
                allow_nan=False,
            )
        )

    def fit_adam(
        self,
        parameters: list[dict[str, Any]],
        training: list[dict[str, Any]],
        validation: list[dict[str, Any]],
        *,
        policy: AdamPolicy | None = None,
    ) -> PreparedCyclicFamily:
        """Projected nonlinear likelihood fitting with explicit validation stopping.

        Known intervention targets and all free-system Jacobians remain in the
        likelihood. Validation stopping, training-score stationarity and epoch
        exhaustion are separate facts. Final test regimes must stay outside this
        request; a minibatch seed never supplies an equilibrium selection law.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "adam_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "training": training,
                    "validation": validation,
                    "policy": asdict(policy or AdamPolicy()),
                },
                allow_nan=False,
            )
        )

    def fit_gradient(
        self,
        parameters: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        l1_penalty: float = 0.0,
    ) -> PreparedCyclicFamily:
        """Fit local expression/legacy parameters with the exact Jacobian score."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "gradient_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "rows": rows,
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                        "l1_penalty": l1_penalty,
                    },
                },
                allow_nan=False,
            )
        )

    def prepare(
        self,
        worlds: list[list[dict[str, Any]]],
        noise: list[list[float]],
        *,
        initial: list[float] | None = None,
        factual: list[float] | None = None,
        policy: SolvePolicy | None = None,
        parameters: list[dict[str, Any]] | None = None,
        rows: list[dict[str, Any]] | None = None,
        fit_options: FitOptions | None = None,
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> PreparedCyclicSystem:
        """Retain aligned intervention worlds and optional joint-likelihood fit.

        A clamp is ``{"variable": index, "value": number}``. A data row is
        ``{"values": [...], "clamps": [...]}``; parameter coordinates are
        ``{"kind": "weight", "equation": i, "parent": j}`` or
        ``{"kind": "intercept", "equation": i}``.
        """
        if (parameters is None) != (rows is None):
            raise ValueError("parameters and rows must be supplied together")
        fit = None
        if parameters is not None:
            fit = {
                "parameters": parameters,
                "rows": rows,
                "options": asdict(fit_options or FitOptions()),
            }
        request = {
            "model": asdict(self),
            "fit": fit,
            "worlds": worlds,
            "noise": noise,
            "initial": initial if initial is not None else [0.0] * len(self.equations),
            "policy": asdict(policy or SolvePolicy()),
            "factual": factual,
        }
        return PreparedCyclicSystem(
            json.dumps(request, allow_nan=False), max_bytes=max_bytes, cancel=cancel
        )


@dataclass(frozen=True)
class PreparedCyclicSystem:
    """Frozen retained request; Rust validates and executes its complete program."""

    request_json: str
    max_bytes: int | None = field(default=None, repr=False, compare=False)
    cancel: Any = field(default=None, repr=False, compare=False)
    _handle: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_handle",
            _native.prepare_cyclic_system(
                self.request_json, max_bytes=self.max_bytes, cancel=self.cancel
            ),
        )

    def inspect(self) -> dict[str, Any]:
        """Inspect the common native prepared handle's fit count and SCC schedules."""
        result: dict[str, Any] = json.loads(self._handle.inspect())
        return result

    def retarget(
        self,
        worlds: list[list[dict[str, Any]]],
        noise: list[list[float]],
        *,
        initial: list[float] | None = None,
    ) -> PreparedCyclicSystem:
        """Retarget worlds with zero causal refits; source and inference remain frozen."""
        request = json.loads(self.request_json)
        request.update(worlds=worlds, noise=noise)
        if initial is not None:
            request["initial"] = initial
        handle = self._handle.retarget(
            json.dumps(worlds, allow_nan=False),
            json.dumps(noise, allow_nan=False),
            json.dumps(request["initial"], allow_nan=False),
        )
        result = object.__new__(PreparedCyclicSystem)
        object.__setattr__(result, "request_json", json.dumps(request, allow_nan=False))
        object.__setattr__(result, "_handle", handle)
        object.__setattr__(result, "max_bytes", self.max_bytes)
        object.__setattr__(result, "cancel", self.cancel)
        return result

    def refresh(self, rows: list[dict[str, Any]]) -> PreparedCyclicSystem:
        """Replace training observations through a new preparation and refit."""
        request = json.loads(self.request_json)
        if request["fit"] is None:
            raise ValueError("refresh requires a fitted retained program")
        request["fit"]["rows"] = rows
        return PreparedCyclicSystem(
            json.dumps(request, allow_nan=False), max_bytes=self.max_bytes, cancel=self.cancel
        )

    def execute(self, *, max_bytes: int | None = None, cancel: Any = None) -> CyclicResult:
        """Execute the retained native product and produce its independent replay artifact."""
        report, artifact = self._handle.execute(
            max_bytes=self.max_bytes if max_bytes is None else max_bytes,
            cancel=self.cancel if cancel is None else cancel,
        )
        return CyclicResult(json.loads(report), artifact)

    def posterior(
        self,
        *,
        prior_sd: float,
        chains: int = 4,
        warmup: int = 500,
        draws: int = 1000,
        proposal_scale: float = 0.05,
        seed: int = 0,
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> dict[str, Any]:
        """Conditional-structure posterior with an explicit normal parameter prior.

        Prior support is restricted to the certified likelihood domain. Inspect
        ``diagnostics_pass`` before interpreting draws; structural-selection and
        covariance uncertainty are not included.
        """
        result: dict[str, Any] = json.loads(
            _native.cyclic_parameter_posterior(
                self.request_json,
                prior_sd,
                chains,
                warmup,
                draws,
                proposal_scale,
                seed,
                max_bytes=self.max_bytes if max_bytes is None else max_bytes,
                cancel=self.cancel if cancel is None else cancel,
            )
        )
        return result


@dataclass(frozen=True)
class OdePolicy:
    """Adaptive continuous-time local-error and work policy."""

    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-7
    initial_step: float = 0.01
    max_steps: int = 100_000


@dataclass(frozen=True)
class PreparedCyclicFamily:
    """Retained dynamics/constraint request; no static SCM coercion is implied."""

    request_json: str
    _handle: Any = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def inspect(self) -> dict[str, Any]:
        """Inspect retained work; this preflight never fits or executes a model."""
        if self._handle is None:
            return {"family": json.loads(self.request_json)["family"], "preparation_evaluations": 0}
        return dict(json.loads(self._handle.inspect()))

    def evidence(
        self,
        schema: dict[str, Any],
        selectors: list[dict[str, Any]],
        *,
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> CyclicEvidence:
        """Project retained native evidence without fitting/sampling again.

        Schema declares ``id``, ``population`` and ordered ``(name, units)``
        variables. Typed selectors bind the actual world, time and uncertainty
        port. Original diagnostics and failed slots remain part of the evidence.
        """
        self.execute(max_bytes=max_bytes, cancel=cancel)
        return CyclicEvidence(
            self._handle.evidence(
                json.dumps(schema, allow_nan=False),
                json.dumps(selectors, allow_nan=False),
                max_bytes=max_bytes,
                cancel=cancel,
            )
        )

    def identified_functional(
        self,
        *,
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> IdentifiedCyclicFunctional:
        """Retain the checked functional and its exact graph/assumption source."""
        source = json.loads(self.request_json)
        if source.get("family") != "graph" or source["request"].get("operation") not in (
            "identify",
            "adjust",
            "calculus",
        ):
            raise ValueError("expected a graph identification preparation")
        result = self.execute(max_bytes=max_bytes, cancel=cancel)
        answer = result.report["answer"]
        # Serde's tagged answer stores the operation report in `result`.
        graph = answer.get("result", answer.get("answer", answer))
        if (
            graph.get("status") not in ("identified", "verified_rule")
            or graph.get("program") is None
            or graph.get("fully_observational") is False
        ):
            raise ValueError("source did not identify a functional")
        return IdentifiedCyclicFunctional(source["request"], result.identity, graph["program"])

    def refresh(self, request_json: str) -> PreparedCyclicFamily:
        """Bind changed source/target/initial law/policy through a fresh preparation."""
        return PreparedCyclicFamily(request_json)

    def execute(self, *, max_bytes: int | None = None, cancel: Any = None) -> CyclicResult:
        """Prepare once, retain fitted/drawn products, and export without refitting."""
        with self._lock:
            if self._handle is None:
                object.__setattr__(
                    self,
                    "_handle",
                    _native.prepare_cyclic_family(
                        self.request_json, max_bytes=max_bytes, cancel=cancel
                    ),
                )
            report, artifact = self._handle.execute(max_bytes=max_bytes, cancel=cancel)
        return CyclicResult(json.loads(report), artifact)


def consume_family_artifact(
    artifact: bytes, *, expected_identity: str, max_bytes: int | None = None, cancel: Any = None
) -> CyclicResult:
    """Recompute a dynamic/constraint answer with independent identity checking."""
    report = json.loads(
        _native.consume_cyclic_family(
            artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
        )
    )
    return CyclicResult(report, artifact)


@dataclass(frozen=True)
class OdeSystem:
    """Drift ``intercept + activation(weights @ state) - decay * state``."""

    equations: tuple[Equation, ...]
    decay: tuple[float, ...]

    def prepare(
        self,
        initial: list[float],
        times: list[float],
        *,
        schedule: list[dict[str, Any]] | None = None,
        policy: OdePolicy | None = None,
    ) -> PreparedCyclicFamily:
        """Retain an initial-value problem and time-indexed sustained clamps."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ode",
                    "model": asdict(self),
                    "initial": initial,
                    "times": times,
                    "schedule": schedule or [],
                    "policy": asdict(policy or OdePolicy()),
                },
                allow_nan=False,
            )
        )

    def fit(
        self,
        parameters: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        *,
        integration: OdePolicy | None = None,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ) -> PreparedCyclicFamily:
        """Fit drift jointly using analytic sensitivities and known observation variances."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ode_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "observations": observations,
                    "integration": asdict(integration or OdePolicy()),
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                    },
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """Linear stochastic drift about a center, with diffusion covariance."""

    drift: tuple[float, ...]
    center: tuple[float, ...]
    diffusion: tuple[float, ...]

    def stationary(self) -> PreparedCyclicFamily:
        """Request stationary covariance with a checked Lyapunov stability witness."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_stationary",
                    "model": asdict(self),
                },
                allow_nan=False,
            )
        )

    def transient(
        self,
        initial_mean: list[float],
        initial_covariance: list[float],
        times: list[float],
        *,
        policy: OdePolicy | None = None,
    ) -> PreparedCyclicFamily:
        """Retain the explicit initial law and finite-horizon moment request."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_transient",
                    "model": asdict(self),
                    "initial_mean": initial_mean,
                    "initial_covariance": initial_covariance,
                    "times": times,
                    "policy": asdict(policy or OdePolicy()),
                },
                allow_nan=False,
            )
        )

    def fit(
        self,
        parameters: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        policy: FitOptions | None = None,
    ) -> PreparedCyclicFamily:
        """Fit stationary drift/center under known regimes and specified diffusion.

        A numerical fit does not establish that stationary observations identify
        every requested drift coordinate.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "rows": rows,
                    "policy": asdict(policy or FitOptions()),
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class PolynomialSystem:
    """Named polynomial residuals with explicit finite variable domains.

    Each equation is ``{"id": name, "terms": [{"coefficient": c,
    "powers": [p0, ...]}, ...]}``; each domain is ``{"lower": a, "upper": b}``.
    """

    variables: tuple[str, ...]
    equations: tuple[dict[str, Any], ...]
    domain: tuple[dict[str, float], ...]

    def replace_equation(self, equation: str, variable: int, value: float) -> PolynomialSystem:
        """Replace one named equation by ``variable=value``, retaining other equations."""
        if not 0 <= variable < len(self.variables):
            raise ValueError("variable index out of range")
        matches = [i for i, item in enumerate(self.equations) if item["id"] == equation]
        if len(matches) != 1:
            raise ValueError("equation identity must resolve uniquely")
        powers = [0] * len(self.variables)
        powers[variable] = 1
        equations = list(self.equations)
        equations[matches[0]] = {
            "id": equation,
            "terms": [
                {"coefficient": 1.0, "powers": powers},
                {"coefficient": -value, "powers": [0] * len(self.variables)},
            ],
        }
        return PolynomialSystem(self.variables, tuple(equations), self.domain)

    def prepare(
        self, *, max_boxes: int = 10_000, minimum_width: float = 1e-8
    ) -> PreparedCyclicFamily:
        """Retain bounded root isolation; unresolved boxes never become certified roots."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "polynomial",
                    "model": asdict(self),
                    "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class CyclicGraph:
    """Original directed feedback and distinct latent-confounding edges."""

    nodes: int
    directed: tuple[tuple[int, int], ...]
    bidirected: tuple[tuple[int, int], ...] = ()

    def identify(
        self,
        outcomes: list[int],
        treatments: list[int],
        *,
        io_scm: bool,
        scc_product_density: bool,
        continuous: bool = True,
        max_steps: int = 1000,
    ) -> PreparedCyclicFamily:
        """Generalized cyclic ID with explicit theorem premises and bounded search."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "identify",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "io_scm": io_scm,
                        "scc_product_density": scc_product_density,
                        "continuous": continuous,
                        "max_steps": max_steps,
                    },
                },
                allow_nan=False,
            )
        )

    def calculus(
        self,
        rule: Literal["observation", "exchange", "action"],
        x: list[int],
        y: list[int],
        z: list[int],
        w: list[int],
        *,
        io_scm: bool,
        standard_measurable_spaces: bool,
        max_checks: int = 1000,
    ) -> PreparedCyclicFamily:
        """Check Theorem 7.2 with indicators, fixed inputs and distinct bounded failure.

        The functional retains source and executable roots. A verified rewrite
        with remaining intervention factors does not become an identified effect.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "calculus",
                        "graph": asdict(self),
                        "rule": rule,
                        "x": x,
                        "y": y,
                        "z": z,
                        "w": w,
                        "io_scm": io_scm,
                        "standard_measurable_spaces": standard_measurable_spaces,
                        "max_checks": max_checks,
                    },
                },
                allow_nan=False,
            )
        )

    def adjust(
        self,
        outcomes: list[int],
        treatments: list[int],
        adjustment: list[int],
        *,
        io_scm: bool,
        standard_measurable_spaces: bool,
        continuous: bool = True,
    ) -> PreparedCyclicFamily:
        """Derive the executable cyclic backdoor adjustment functional when licensed."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "adjust",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "adjustment": adjustment,
                        "continuous": continuous,
                        "io_scm": io_scm,
                        "standard_measurable_spaces": standard_measurable_spaces,
                    },
                },
                allow_nan=False,
            )
        )

    def backdoor(
        self,
        outcomes: list[int],
        treatments: list[int],
        adjustment: list[int],
        *,
        io_scm: bool,
        scc_product_density: bool,
    ) -> PreparedCyclicFamily:
        """Check the sufficient sigma-backdoor criterion; failure is not impossibility."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "backdoor",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "adjustment": adjustment,
                        "io_scm": io_scm,
                        "scc_product_density": scc_product_density,
                    },
                },
                allow_nan=False,
            )
        )

    def oracle_discovery(self, *, simple_scm: bool, sigma_faithful: bool) -> PreparedCyclicFamily:
        """Exact sigma-CI reference reproduction; this does not learn from samples."""
        return _discovery_request(
            [[0.0] * 8 for _ in range(self.nodes)],
            simple_scm=simple_scm,
            sigma_faithful=sigma_faithful,
            independent_equilibrium_replicates=True,
            oracle=asdict(self),
            ci="parcorr",
            alpha=0.05,
            max_cond_size=self.nodes,
            pds_max_nodes=10_000,
        )


def _discovery_request(
    columns: list[list[float]],
    *,
    simple_scm: bool,
    sigma_faithful: bool,
    independent_equilibrium_replicates: bool,
    oracle: dict[str, Any] | None,
    ci: str,
    alpha: float,
    max_cond_size: int,
    pds_max_nodes: int,
    seed: int = 0,
) -> PreparedCyclicFamily:
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "graph",
                "request": {
                    "operation": "discover",
                    "seed": seed,
                    "columns": columns,
                    "oracle": oracle,
                    "ci": ci,
                    "alpha": alpha,
                    "max_cond_size": max_cond_size,
                    "pds_max_nodes": pds_max_nodes,
                    "simple_scm": simple_scm,
                    "sigma_faithful": sigma_faithful,
                    "independent_equilibrium_replicates": independent_equilibrium_replicates,
                },
            },
            allow_nan=False,
        )
    )


def discover_cyclic(
    columns: list[list[float]],
    *,
    simple_scm: bool,
    sigma_faithful: bool,
    independent_equilibrium_replicates: bool,
    ci: str = "parcorr",
    alpha: float = 0.05,
    max_cond_size: int | None = None,
    pds_max_nodes: int = 10_000,
    seed: int = 0,
) -> PreparedCyclicFamily:
    """FCI for cyclic equivalence classes with the chosen existing CI procedure.

    The CI procedure must suit the data-generating family. Partial correlation
    is not an independence oracle for arbitrary nonlinear systems. All engine
    truncation diagnostics and the PAG inducing-path interpretation are retained.
    """
    return _discovery_request(
        columns,
        simple_scm=simple_scm,
        sigma_faithful=sigma_faithful,
        independent_equilibrium_replicates=independent_equilibrium_replicates,
        oracle=None,
        ci=ci,
        alpha=alpha,
        max_cond_size=len(columns) if max_cond_size is None else max_cond_size,
        pds_max_nodes=pds_max_nodes,
        seed=seed,
    )


def learn_structure(
    model: EquilibriumModel,
    training: list[dict[str, Any]],
    *,
    heldout: list[dict[str, Any]] | None = None,
    threshold: float = 0.05,
    policy: FitOptions | None = None,
    seed: int = 0,
    bootstrap_replicates: int = 0,
    design: Literal["fixed_assignments", "randomized_within_targets"] = "fixed_assignments",
) -> PreparedCyclicFamily:
    """Select a sparse parametric equilibrium topology and refit it jointly.

    Supply a positive L1 penalty. Held-out intervention regimes evaluate the final
    fit and never tune selection. Uncertainty conditional on the selected model
    must not be read as uncertainty for the complete selection procedure.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "structure_fit",
                "seed": seed,
                "model": asdict(model),
                "training": training,
                "heldout": heldout or [],
                "threshold": threshold,
                "bootstrap_replicates": bootstrap_replicates,
                "design": design,
                "policy": asdict(policy or FitOptions(l1_penalty=1.0)),
            },
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class SamplingPolicy:
    """Retained independent chains, warmup, proposal scale, and replay seed."""

    chains: int = 4
    warmup: int = 500
    draws: int = 1000
    proposal_scale: float = 0.05
    seed: int = 0


def posterior_responses(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    worlds: list[list[dict[str, Any]]],
    noise: list[list[float]],
    *,
    prior_sd: float,
    initial: list[float] | None = None,
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Joint parameter/noise-aligned response program with a replayable posterior.

    Results retain world, parameter-draw, and supplied disturbance-row axes.
    Supplied disturbances are not automatically a population distribution.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "parameter_posterior",
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "prior_sd": prior_sd,
                "sampling": asdict(sampling or SamplingPolicy()),
                "worlds": worlds,
                "noise": noise,
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        )
    )


def partial_counterfactual(
    model: EquilibriumModel,
    observations: list[float | None],
    worlds: list[list[dict[str, Any]]],
    *,
    measurement_variances: list[float],
    initial: list[float],
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Abduce under an explicit exact/noisy observation model and reuse disturbances.

    This conditions on a specified mechanism/noise model. Parameter uncertainty
    is separate from latent-state/measurement uncertainty.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "partial_counterfactual",
                "model": asdict(model),
                "observations": observations,
                "measurement_variances": measurement_variances,
                "initial": initial,
                "sampling": asdict(sampling or SamplingPolicy()),
                "worlds": worlds,
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class BipartiteCausalGraph:
    """Equation-variable graph; numerical solvability is a separate question."""

    variables: int
    exogenous: tuple[int, ...]
    equations: tuple[tuple[int, ...], ...]

    def b_separation(
        self,
        x: int,
        y: int,
        conditioning: list[int],
        *,
        replacement: tuple[int, int] | None = None,
    ) -> PreparedCyclicFamily:
        """Compute causal ordering and determinism-aware graphical B-separation.

        Markov interpretation requires independent exogenous inputs and measurable
        unique cluster solutions; matching alone establishes neither.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "bipartite",
                        **asdict(self),
                        "x": x,
                        "y": y,
                        "conditioning": conditioning,
                        "replacement": replacement,
                    },
                },
                allow_nan=False,
            )
        )

    def domain_invariance(
        self,
        changed_equations: list[int],
        outcomes: list[int],
        conditioning: list[int],
    ) -> PreparedCyclicFamily:
        """Check the BGCM domain-selector sufficient invariance criterion.

        Compatible unique cluster solutions must hold in every domain; only named
        equations may change. A failed separation check does not disprove invariance.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "domain_invariance",
                        **asdict(self),
                        "changed_equations": changed_equations,
                        "outcomes": outcomes,
                        "conditioning": conditioning,
                    },
                },
                allow_nan=False,
            )
        )


def fit_polynomial_candidates(
    candidates: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    *,
    max_boxes: int = 10_000,
    minimum_width: float = 1e-8,
) -> PreparedCyclicFamily:
    """Fit a finite equation-parameter set under explicit root-selection/measurement laws.

    Each candidate names a polynomial model and geometric root regions with
    declared probabilities. A finite-set optimum certificate requires disjoint
    outward-rounded scores; unresolved roots/rankings retain a following attempt.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "polynomial_fit",
                "candidates": candidates,
                "measurements": measurements,
                "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
            },
            allow_nan=False,
        )
    )


def joint_counterfactual(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    observations: list[float | None],
    measurement_variances: list[float],
    worlds: list[list[dict[str, Any]]],
    *,
    prior_sd: float,
    initial: list[float],
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Update parameters and factual states jointly, then reuse each noise draw.

    Training and factual measurement evidence both enter the posterior. Structure
    and covariance are specified. Inspect joint chain diagnostics before use.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "joint_counterfactual",
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "prior_sd": prior_sd,
                "observations": observations,
                "measurement_variances": measurement_variances,
                "worlds": worlds,
                "initial": initial,
                "sampling": asdict(sampling or SamplingPolicy()),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        )
    )


def action_worlds(
    model: EquilibriumModel,
    worlds: list[dict[str, Any]],
    noise: list[list[float]],
    *,
    factual: list[float] | None = None,
    initial: list[float] | None = None,
    policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Retain named clamps, mechanism changes/replacements and additive shocks.

    Each world has ``id`` and ``actions``. Actions name stable ``mechanism`` IDs;
    kinds are ``clamp``, ``replace_mechanism``, ``change_mechanism``, and
    ``additive_shock``. Replacements/changes preserve the original disturbances;
    shocks add to them after factual abduction. Clamps remove the mechanism.
    Failed numerical/certificate attempts retain their original response slots.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "equilibrium_actions",
                "model": asdict(model),
                "worlds": worlds,
                "noise": noise,
                "factual": factual,
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "policy": asdict(policy or SolvePolicy()),
            },
            allow_nan=False,
        )
    )


def bootstrap_responses(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    worlds: list[list[dict[str, Any]]],
    noise: list[list[float]],
    *,
    design: Literal["fixed_assignments", "randomized_within_targets"] = "fixed_assignments",
    replicates: int = 100,
    seed: int = 0,
    policy: FitOptions | None = None,
    initial: list[float] | None = None,
    solve_policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Resample IID rows within known intervention regimes and propagate fits.

    Failed fits/response attempts retain their positions. This conditions on the
    declared structure and covariance; it does not claim calibrated coverage.
    """
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "frequentist_bootstrap",
                "design": design,
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "worlds": worlds,
                "noise": noise,
                "replicates": replicates,
                "seed": seed,
                "policy": asdict(policy or FitOptions()),
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        )
    )


@dataclass(frozen=True)
class PolynomialWorld:
    """Equation-specific actions; clamps remove the replaced disturbance."""

    id: str
    replacements: tuple[dict[str, Any], ...] = ()
    clamps: tuple[dict[str, Any], ...] = ()
    shocks: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PolynomialExogenousAtom:
    """Declared joint disturbances and geometric branch choices across all worlds.

    Relative weights define a law independently of root-search order. Measurement
    kernels map world IDs to category/relative-probability maps. Each training unit
    has its own atom; one query unit reuses its atom across every counterfactual.
    """

    id: str
    weight: float
    disturbances: tuple[float, ...]
    selections: dict[str, list[dict[str, float]]]
    measurements: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PolynomialCausalCandidate:
    """Finite parameter candidate with a declared prior and conditional exogenous law."""

    id: str
    model: PolynomialSystem
    atoms: tuple[PolynomialExogenousAtom, ...]
    prior_weight: float = 1.0


@dataclass(frozen=True)
class PolynomialObservation:
    """Partial state event at declared measurement resolution and/or noisy category.

    Bounds are observation events, separate from solver tolerance. A category
    requires an explicit measurement kernel in every exogenous atom.
    """

    world: str
    bounds: tuple[dict[str, float] | None, ...]
    label: str | None = None


@dataclass(frozen=True)
class PolynomialCausalModel:
    """Finite joint-selection SCM: fitting, abduction, action, prediction and replay.

    A candidate is shared across training units. Disturbances and branch selectors
    are independent between units conditional on that candidate, and shared across
    the potential worlds of one unit. Undefined/unresolved response mass is retained.
    """

    candidates: tuple[PolynomialCausalCandidate, ...]
    worlds: tuple[PolynomialWorld, ...]

    def prepare(
        self,
        factual: PolynomialObservation,
        *,
        training: tuple[PolynomialObservation, ...] = (),
        max_boxes: int = 10_000,
        minimum_width: float = 1e-8,
    ) -> PreparedCyclicFamily:
        """Condition the finite parameter/selection law and retain coupled world states."""
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "polynomial_causal",
                    "program": {
                        **asdict(self),
                        "training": [asdict(item) for item in training],
                        "factual": asdict(factual),
                        "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                    },
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class IdentifiedCyclicFunctional:
    """Checked cyclic functional bound to an independent identification source."""

    source: dict[str, Any]
    source_identity: str
    program: dict[str, Any]

    def evaluate_gaussian(
        self, law: dict[str, Any], assignments: list[list[tuple[int, float]]]
    ) -> PreparedCyclicFamily:
        """Evaluate continuous identified densities with analytic Gaussian marginalization.

        ``law`` is ``{kind: specified, mean: [...], covariance: [...]}`` or
        ``{kind: fit, rows: [...], independent_observational_replicates: True}``.
        Fitting is an unregularized IID observational Gaussian likelihood plug-in,
        conditional on the original graphical premises; no latent recovery or
        statistical interval claim is added. Discrete sources refuse this path.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "gaussian_functional",
                    "source": self.source,
                    "source_identity": self.source_identity,
                    "program": self.program,
                    "law": law,
                    "assignments": assignments,
                },
                allow_nan=False,
            )
        )

    def evaluate(
        self,
        support: list[list[float]],
        weights: list[float],
        assignments: list[list[tuple[int, float]]],
    ) -> PreparedCyclicFamily:
        """Consume a finite joint law or empirical table; retain positivity failures.

        Variable zero varies fastest in the joint weights. Known probabilities
        and empirical counts are distinct scientific inputs; this operation
        evaluates their declared law without adding a nominal uncertainty claim.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "functional",
                    "source": self.source,
                    "source_identity": self.source_identity,
                    "program": self.program,
                    "law": {"support": support, "weights": weights},
                    "assignments": assignments,
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class LocalMechanism:
    """Pure local callback returning ``(value, derivatives_in_read_order)``.

    The callback receives only its declared inputs. Purity and derivative
    correctness are caller obligations. Its code and external allocations are
    process-local; independent artifacts require serializable expressions.
    """

    id: str
    reads: tuple[int, ...]
    callback: Any


@dataclass(frozen=True)
class LocalCyclicModel:
    """Process-local additive-noise feedback model with explicit derivatives."""

    mechanisms: tuple[LocalMechanism, ...]

    def execute(
        self,
        noise: list[float],
        clamps: list[dict[str, Any]],
        *,
        initial: list[float] | None = None,
        factual: list[float] | None = None,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
        max_evaluations: int = 100_000,
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> dict[str, Any]:
        """Numerical SCC solve and optional shared-noise counterfactual.

        Initialization is numerical. It supplies neither uniqueness nor an
        equilibrium selection law, and no probabilistic root selection is made.
        Rust checks residuals, derivative shapes, cancellation and call budgets.
        """
        request = {
            "mechanisms": [(m.id, m.reads) for m in self.mechanisms],
            "noise": noise,
            "clamps": clamps,
            "initial": initial if initial is not None else [0.0] * len(self.mechanisms),
            "factual": factual,
            "policy": {
                "tolerance": tolerance,
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
            },
        }
        return json.loads(
            _native.execute_local_cyclic(
                json.dumps(request, allow_nan=False),
                [m.callback for m in self.mechanisms],
                max_bytes=max_bytes,
                cancel=cancel,
            )
        )

    def export(self) -> bytes:
        """Refuse to present arbitrary local callback code as a portable model."""
        raise ValueError(
            "local callback code is not portable; use serializable built-in expressions"
        )


@dataclass(frozen=True)
class CyclicEvidence:
    """Native source-bound contract envelopes, uncertainty ports and decision inputs."""

    _handle: Any

    @property
    def report(self) -> dict[str, Any]:
        """Return derived scientific coordinates, original diagnostics and failed slots."""
        return dict(json.loads(self._handle.report()))

    @property
    def identity(self) -> str:
        """Source/schema/world/time/selector identity for independent consumption."""
        return str(self.report["identity"])

    def export(self, *, max_bytes: int | None = None, cancel: Any = None) -> bytes:
        """Export original source and projection, preserving replay and uncertainty meaning."""
        return bytes(self._handle.export(max_bytes=max_bytes, cancel=cancel))

    def joint_law(self, *, max_bytes: int | None = None, cancel: Any = None) -> Any:
        """Use the existing joint-distribution/decision API for complete aligned laws.

        Bootstrap laws, numerical root sets, finite mass bounds, uncertified roots
        and failed posterior slots cannot be silently coerced to an outcome law.
        """
        from .joint_distribution import JointDistributionArtifact

        return JointDistributionArtifact._from_native(
            self._handle.joint_law(max_bytes=max_bytes, cancel=cancel)
        )

    def decide_enclosures(self, contract: Any) -> dict[str, Any]:
        """Conservative robust decisions on model solution/scenario outer enclosures.

        Unresolved root boxes remain in the set; undefined selected-state mass
        refuses. This introduces neither equilibrium probabilities nor a
        nonparametric identification or statistical coverage claim.
        """
        return dict(json.loads(self._handle.decide_enclosures(json.dumps(contract._wire()))))

    def decide_means(self, contract: Any) -> dict[str, Any]:
        """Evaluate an existing decision Contract on point/mean evidence only."""
        return dict(json.loads(self._handle.decide_means(json.dumps(contract._wire()))))


def consume_evidence(
    artifact: bytes,
    *,
    expected_identity: str,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> CyclicEvidence:
    """Independently replay a cyclic source and recheck its entire evidence projection."""
    return CyclicEvidence(
        _native.consume_cyclic_evidence(
            artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
        )
    )


@dataclass(frozen=True)
class OuMechanisms:
    """Affine stochastic feedback ``dX=(M X+b)dt+L dW`` with named equations.

    Forcing is distinct from the stationary mean. Replacements change exactly
    one equation's drift, forcing and diffusion-factor row, preserving all
    untargeted mechanisms. The stationary law is rechecked in every world.
    """

    equations: tuple[str, ...]
    drift: tuple[float, ...]
    forcing: tuple[float, ...]
    diffusion_factor: tuple[float, ...]

    def prepare(
        self,
        worlds: list[list[dict[str, Any]]],
        *,
        regimes: list[dict[str, Any]] | None = None,
        parameters: list[dict[str, Any]] | None = None,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ) -> PreparedCyclicFamily:
        """Evaluate or jointly fit independent stationary intervention regimes.

        A replacement supplies ``equation`` (stable id), ``drift`` (full row),
        ``forcing`` and ``diffusion_factor`` (full row). Known target mechanisms
        stay fixed during fitting; baseline coefficients are shared everywhere
        else. Parameters are ``drift`` (equation,parent), ``forcing`` (equation)
        or ``diffusion`` (equation,source). Identifiability is a separate question.
        """
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_mechanisms",
                    "model": asdict(self),
                    "worlds": worlds,
                    "regimes": regimes or [],
                    "parameters": parameters or [],
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                    },
                },
                allow_nan=False,
            )
        )


@dataclass(frozen=True)
class AdamPolicy:
    """Bounded projected Adam, with explicit independent validation stopping."""

    learning_rate: float = 0.01
    batch_size: int = 256
    max_epochs: int = 200
    minimum_epochs: int = 20
    patience: int = 10
    relative_tolerance: float = 1e-5
    score_tolerance: float = 1e-3
    l1_penalty: float = 0.001
    contraction_margin: float = 0.95
    seed: int = 0
