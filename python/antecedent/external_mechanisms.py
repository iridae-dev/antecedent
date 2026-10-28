# SPDX-License-Identifier: MIT OR Apache-2.0
"""Registered external cyclic mechanisms that take part in portable artifacts.

A process-local callback cannot be exported: its code does not travel. A *registered*
mechanism can. :func:`register_mechanism` binds a Python callable to a versioned
implementation id, the BLAKE3 digest of its content, its declared dependencies, its input
signature, its parameter schema and its derivative declaration (an analytic gradient, or a
finite difference with a declared step and scheme) and optional global Lipschitz bounds.
A fitted feedback system built from such mechanisms records those identities in its artifact
and never the code.

Independent replay needs an explicit *trusted resolver* supplied by the caller: a
:class:`MechanismRegistry`, or any callable ``resolver(spec: dict) -> implementation | None``.
The native layer verifies the resolved implementation against the record (implementation id,
version, content digest, dependencies and declarations) before it runs anything. Replay with
no resolver, with no implementation, or with a changed one is refused with a registered
reason code (``mechanism_resolver_missing``, ``mechanism_implementation_unavailable``,
``mechanism_implementation_changed``). A system with an external mechanism is
``verified_external``; it is never upgraded to ``native``.

The supplied callable runs with the host's authority. This library does not sandbox it; the
digest binds replay to the exact content the caller registered, it does not make the code safe.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native
from .errors import CausalValueError

__all__ = [
    "Dependency",
    "ExternalArtifact",
    "ExternalEquation",
    "ExternalFeedbackModel",
    "ExternalOdeModel",
    "FiniteDifference",
    "MechanismImplementation",
    "MechanismRegistry",
    "consume_artifact",
    "default_registry",
    "inspect_artifact",
    "load_artifact",
    "register_mechanism",
]

Resolver = Callable[[Mapping[str, Any]], "MechanismImplementation | None"]


@dataclass(frozen=True)
class Dependency:
    """A declared dependency of an implementation, bound by version and digest."""

    name: str
    version: str
    digest: str


@dataclass(frozen=True)
class FiniteDifference:
    """Derivative policy: ``h_j = step * max(1, |x_j|)`` with the named scheme."""

    step: float = 1e-6
    scheme: Literal["forward", "central"] = "central"


def _sources(*callables: Callable[..., Any] | None) -> bytes:
    parts: list[str] = []
    for fn in callables:
        if fn is None:
            continue
        try:
            parts.append(inspect.getsource(fn))
        except (OSError, TypeError) as error:
            raise CausalValueError(
                "cannot read the source of a mechanism callable; pass content= (the bytes that "
                "identify the implementation, for example its source or a build hash)"
            ) from error
    return "\n".join(parts).encode("utf-8")


class MechanismImplementation:
    """A registered, process-local implementation. Its callables are never serialized."""

    def __init__(
        self,
        implementation_id: str,
        version: str,
        value: Callable[[list[float], list[float]], float],
        reads: Sequence[int],
        parameters: Sequence[str] = (),
        *,
        content: bytes | str | None = None,
        dependencies: Sequence[Dependency] = (),
        input_gradient: Callable[[list[float], list[float]], Sequence[float]] | None = None,
        parameter_gradient: Callable[[list[float], list[float]], Sequence[float]] | None = None,
        mixed_gradient: Callable[[list[float], list[float]], Sequence[Sequence[float]]]
        | None = None,
        derivative: Literal["analytic"] | FiniteDifference | None = None,
        input_lipschitz: Sequence[float] | None = None,
    ) -> None:
        if not callable(value):
            raise CausalValueError("a mechanism needs a callable value(inputs, parameters)")
        if derivative is None:
            derivative = "analytic" if input_gradient is not None else FiniteDifference()
        if derivative == "analytic" and input_gradient is None:
            raise CausalValueError(
                "an analytic derivative declaration needs input_gradient(inputs, parameters)"
            )
        if content is None:
            content = _sources(value, input_gradient, parameter_gradient, mixed_gradient)
        elif isinstance(content, str):
            content = content.encode("utf-8")
        self._value = value
        self._input_gradient = input_gradient
        self._parameter_gradient = parameter_gradient
        self._mixed_gradient = mixed_gradient
        self.content: bytes = bytes(content)
        spec: dict[str, Any] = {
            "implementation_id": implementation_id,
            "version": version,
            "content_digest": _native.external_content_digest(self.content),
            "dependencies": [
                {"name": d.name, "version": d.version, "digest": d.digest} for d in dependencies
            ],
            "reads": [int(r) for r in reads],
            "parameter_names": list(parameters),
            "derivative": (
                {"kind": "analytic"}
                if derivative == "analytic"
                else {
                    "kind": "finite_difference",
                    "step": derivative.step,
                    "scheme": derivative.scheme,
                }
            ),
        }
        if input_lipschitz is not None:
            spec["input_lipschitz"] = [float(b) for b in input_lipschitz]
        self.spec: dict[str, Any] = spec
        self.spec_json: str = json.dumps(spec, allow_nan=False, sort_keys=True)
        # Validates the declarations and reports the identity every record will carry.
        self.identity: str = _native.external_spec_identity(self.spec_json)

    def value(self, inputs: list[float], parameters: list[float]) -> float:
        return float(self._value(inputs, parameters))

    def input_gradient(self, inputs: list[float], parameters: list[float]) -> list[float] | None:
        if self._input_gradient is None:
            return None
        return [float(g) for g in self._input_gradient(inputs, parameters)]

    def parameter_gradient(
        self, inputs: list[float], parameters: list[float]
    ) -> list[float] | None:
        if self._parameter_gradient is None:
            return None
        return [float(g) for g in self._parameter_gradient(inputs, parameters)]

    def input_gradient_parameter_derivative(
        self, inputs: list[float], parameters: list[float]
    ) -> list[list[float]] | None:
        if self._mixed_gradient is None:
            return None
        return [[float(g) for g in row] for row in self._mixed_gradient(inputs, parameters)]

    def __repr__(self) -> str:
        return (
            f"MechanismImplementation({self.spec['implementation_id']!r}, "
            f"version={self.spec['version']!r}, identity={self.identity[:12]})"
        )


class MechanismRegistry:
    """A trusted resolver: maps a recorded specification to a registered implementation.

    Lookup is by implementation id and version; the native layer then verifies content digest,
    dependencies and declarations, so a registry holding a different build of the same id and
    version is refused rather than trusted.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], MechanismImplementation] = {}

    def add(self, implementation: MechanismImplementation) -> MechanismImplementation:
        key = (implementation.spec["implementation_id"], implementation.spec["version"])
        if key in self._entries:
            raise CausalValueError(
                f"mechanism {key[0]!r} version {key[1]!r} is already registered in this registry"
            )
        self._entries[key] = implementation
        return implementation

    def register_mechanism(self, *args: Any, **kwargs: Any) -> MechanismImplementation:
        """Build and register an implementation; see :func:`register_mechanism`."""
        return self.add(MechanismImplementation(*args, **kwargs))

    def resolve(self, spec: Mapping[str, Any]) -> MechanismImplementation | None:
        return self._entries.get((spec["implementation_id"], spec["version"]))

    __call__ = resolve

    def __len__(self) -> int:
        return len(self._entries)


_DEFAULT = MechanismRegistry()


def default_registry() -> MechanismRegistry:
    """The process default registry used when ``register_mechanism`` gets no ``registry``."""
    return _DEFAULT


def register_mechanism(
    implementation_id: str,
    version: str,
    value: Callable[[list[float], list[float]], float],
    reads: Sequence[int],
    parameters: Sequence[str] = (),
    *,
    registry: MechanismRegistry | None = None,
    content: bytes | str | None = None,
    dependencies: Sequence[Dependency] = (),
    input_gradient: Callable[[list[float], list[float]], Sequence[float]] | None = None,
    parameter_gradient: Callable[[list[float], list[float]], Sequence[float]] | None = None,
    mixed_gradient: Callable[[list[float], list[float]], Sequence[Sequence[float]]] | None = None,
    derivative: Literal["analytic"] | FiniteDifference | None = None,
    input_lipschitz: Sequence[float] | None = None,
) -> MechanismImplementation:
    """Register a scalar mechanism ``value(inputs, parameters)`` for portable artifacts.

    ``reads`` are the endogenous coordinates passed as ``inputs`` in order; ``parameters``
    names the fitted parameter schema. Declare derivatives with ``input_gradient`` (analytic)
    or a :class:`FiniteDifference` policy; ``parameter_gradient`` is optional and makes the
    data-term score analytic, and ``mixed_gradient`` (``[parameter][input]`` second derivatives of the
    value) makes the log-determinant score analytic too. ``input_lipschitz`` are global bounds on ``|d value / d input|``;
    only with bounds can a solve be certified a contraction and a likelihood normalized.
    The bounds are declarations, carried in the record, that the caller is responsible for.
    ``content`` defaults to the source of the callables; pass bytes when source is unavailable.
    """
    target = registry if registry is not None else _DEFAULT
    return target.register_mechanism(
        implementation_id,
        version,
        value,
        reads,
        parameters,
        content=content,
        dependencies=dependencies,
        input_gradient=input_gradient,
        parameter_gradient=parameter_gradient,
        mixed_gradient=mixed_gradient,
        derivative=derivative,
        input_lipschitz=input_lipschitz,
    )


def _resolver(resolver: Any) -> Resolver | None:
    if resolver is None:
        return None
    if hasattr(resolver, "resolve"):
        return resolver.resolve
    if callable(resolver):
        return resolver
    raise CausalValueError("resolver must be a MechanismRegistry or a callable spec -> impl")


def _dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False)


def _clamps(clamps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"variable": int(c["variable"]), "value": float(c["value"])} for c in clamps]


@dataclass(frozen=True)
class ExternalEquation:
    """An equation whose mechanism is a registered external implementation.

    ``mechanism`` may be a :class:`MechanismImplementation` or a recorded spec mapping.
    """

    id: str
    mechanism: MechanismImplementation | Mapping[str, Any]
    parameters: tuple[float, ...] = ()
    intercept: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        spec = (
            self.mechanism.spec
            if isinstance(self.mechanism, MechanismImplementation)
            else dict(self.mechanism)
        )
        return {
            "origin": "external",
            "id": self.id,
            "intercept": float(self.intercept),
            "parameters": [float(p) for p in self.parameters],
            "mechanism": spec,
        }


def _equation_wire(equation: Any) -> dict[str, Any]:
    if isinstance(equation, ExternalEquation):
        return equation.to_wire()
    # A built-in `antecedent.cyclic.Equation`.
    body: dict[str, Any] = {
        "id": equation.id,
        "intercept": float(equation.intercept),
        "weights": [float(w) for w in equation.weights],
        "activation": equation.activation,
    }
    if getattr(equation, "expression", None) is not None:
        body["expression"] = equation.expression
    if getattr(equation, "parameters", ()):
        body["parameters"] = [float(p) for p in equation.parameters]
    return {"origin": "builtin", "equation": body}


@dataclass(frozen=True)
class ExternalArtifact:
    """A portable fitted system: bytes, the identity to retain, and the producing report."""

    data: bytes = field(repr=False)
    identity: str
    report: dict[str, Any]

    @property
    def trust(self) -> str:
        """``verified_external`` for a system with an external mechanism, never ``native``."""
        return str(self.report["trust"])


@dataclass(frozen=True)
class ExternalFeedbackModel:
    """An additive-noise feedback system with built-in and registered external equations."""

    equations: tuple[Any, ...]
    noise_covariance: tuple[tuple[float, ...], ...]

    def _system(self) -> dict[str, Any]:
        flat = [float(v) for row in self.noise_covariance for v in row]
        return {"equations": [_equation_wire(e) for e in self.equations], "noise_covariance": flat}

    def _request(
        self,
        worlds: Sequence[Sequence[Mapping[str, Any]]],
        noise: Sequence[Sequence[float]],
        *,
        initial: Sequence[float] | None,
        factual: Sequence[float] | None,
        fit: Mapping[str, Any] | None,
        tolerance: float,
        max_iterations: int,
        require_contraction: bool,
    ) -> str:
        n = len(self.equations)
        return _dumps(
            {
                "system": self._system(),
                "fit": fit,
                "worlds": [_clamps(w) for w in worlds],
                "noise": [[float(v) for v in row] for row in noise],
                "initial": [float(v) for v in (initial if initial is not None else [0.0] * n)],
                "policy": {
                    "tolerance": tolerance,
                    "max_iterations": max_iterations,
                    "require_contraction": require_contraction,
                },
                "factual": None if factual is None else [float(v) for v in factual],
            }
        )

    @staticmethod
    def fit_spec(
        parameters: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        *,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        difference_step: float = 1e-6,
    ) -> dict[str, Any]:
        """Joint likelihood fit request: targets, known-regime rows and a numerical policy.

        A target is ``{"kind": "parameter", "equation": i, "parameter": j}`` or
        ``{"kind": "intercept", "equation": i}``; a row is ``{"values": [...], "clamps":
        [{"variable": k, "value": v}]}``.
        """
        return {
            "parameters": [dict(p) for p in parameters],
            "rows": [
                {"values": [float(v) for v in r["values"]], "clamps": _clamps(r["clamps"])}
                for r in rows
            ],
            "policy": {
                "learning_rate": learning_rate,
                "score_tolerance": score_tolerance,
                "max_iterations": max_iterations,
                "difference_step": difference_step,
            },
        }

    def fit(
        self,
        parameters: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        *,
        resolver: Any,
        **policy: float,
    ) -> dict[str, Any]:
        """Fit declared scalars from known-regime rows. Needs the resolver, like replay."""
        spec = self.fit_spec(parameters, rows, **policy)
        request = {"system": self._system(), **spec}
        return json.loads(_native.external_fit(_dumps(request), _resolver(resolver)))

    def execute(
        self,
        worlds: Sequence[Sequence[Mapping[str, Any]]],
        noise: Sequence[Sequence[float]],
        *,
        resolver: Any,
        fit: Mapping[str, Any] | None = None,
        initial: Sequence[float] | None = None,
        factual: Sequence[float] | None = None,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
        require_contraction: bool = False,
    ) -> dict[str, Any]:
        """Verify, optionally fit, then solve every world for every disturbance row."""
        request = self._request(
            worlds,
            noise,
            initial=initial,
            factual=factual,
            fit=fit,
            tolerance=tolerance,
            max_iterations=max_iterations,
            require_contraction=require_contraction,
        )
        return json.loads(_native.external_execute(request, _resolver(resolver)))

    def export(
        self,
        artifact_id: str,
        worlds: Sequence[Sequence[Mapping[str, Any]]],
        noise: Sequence[Sequence[float]],
        *,
        resolver: Any,
        fit: Mapping[str, Any] | None = None,
        initial: Sequence[float] | None = None,
        factual: Sequence[float] | None = None,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
        require_contraction: bool = False,
    ) -> ExternalArtifact:
        """Fit and evaluate, then encode a portable artifact that carries no code."""
        request = self._request(
            worlds,
            noise,
            initial=initial,
            factual=factual,
            fit=fit,
            tolerance=tolerance,
            max_iterations=max_iterations,
            require_contraction=require_contraction,
        )
        report_json, data = _native.external_export(request, artifact_id, _resolver(resolver))
        report = json.loads(report_json)
        return ExternalArtifact(bytes(data), report["identity"], report)


def consume_artifact(data: bytes, identity: str, *, resolver: Any = None) -> dict[str, Any]:
    """Independently replay an artifact against the identity its producer retained.

    ``resolver`` is the caller's trusted resolver. Without it, without the recorded
    implementation, or with a changed one, replay is refused (see the module docstring).
    """
    return json.loads(_native.external_consume(bytes(data), identity, _resolver(resolver)))


def load_artifact(data: bytes, *, expected_identity: str, resolver: Any = None) -> dict[str, Any]:
    """Generic-loader entry point. A resolver is required: without one it is refused.

    ``antecedent.artifacts`` loads by container kind and passes no resolver, so loading an
    external-mechanism artifact that way raises ``mechanism_resolver_missing`` (when it names an
    external mechanism); pass the resolver to :func:`consume_artifact` instead.
    """
    return consume_artifact(data, expected_identity, resolver=resolver)


def inspect_artifact(data: bytes) -> dict[str, Any]:
    """The mechanism identities an artifact records. Runs nothing and needs no resolver."""
    return json.loads(_native.external_inspect(bytes(data)))


@dataclass(frozen=True)
class ExternalOdeModel:
    """Continuous-time drift ``dx/dt = f(x) - decay * x`` with registered external mechanisms."""

    equations: tuple[Any, ...]
    decay: tuple[float, ...]

    def trajectory(
        self,
        initial: Sequence[float],
        times: Sequence[float],
        *,
        resolver: Any,
        schedule: Sequence[Mapping[str, Any]] = (),
        absolute_tolerance: float = 1e-9,
        relative_tolerance: float = 1e-7,
        initial_step: float = 0.01,
        max_steps: int = 100_000,
    ) -> dict[str, Any]:
        """Dormand-Prince trajectory. ``drift_lipschitz`` is a declared bound, not measured."""
        request = {
            "system": {
                "equations": [_equation_wire(e) for e in self.equations],
                "decay": [float(d) for d in self.decay],
            },
            "initial": [float(v) for v in initial],
            "times": [float(t) for t in times],
            "schedule": [
                {"time": float(s["time"]), "clamps": _clamps(s["clamps"])} for s in schedule
            ],
            "policy": {
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "initial_step": initial_step,
                "max_steps": max_steps,
            },
        }
        return json.loads(_native.external_ode(_dumps(request), _resolver(resolver)))
