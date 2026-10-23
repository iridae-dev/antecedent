# SPDX-License-Identifier: MIT OR Apache-2.0
"""Typed cyclic worlds, laws recovered from clamp experiments, and dynamic-fit inference.

Public names are re-exported from :mod:`antecedent.cyclic`; import them from there.

Worlds name variables, never array positions. A world is a set of simultaneous
interventions (:class:`~antecedent.intervention.Set`, :class:`~antecedent.intervention.Shift`,
:class:`~antecedent.intervention.Soft`) plus initial-state replacements for continuous-time
families. Rust admits every world on its family: an action the family cannot take is refused
with its next attempt and is never rewritten into another action. Results are keyed by
scientific world id and keep the uniqueness certificate, the solve route, and the failed and
withheld mass; a certificate is never a claim of physical stability.

A recovered law keeps how its draws were produced. A sampling law (asymptotic normal or
bootstrap) is never read as a posterior, and supplied draws declare no meaning. Every answer is
conditional on the specified model or on the declared clamp-experiment recipe; no
nonparametric identification or nominal interval calibration is implied.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from . import _native
from .errors import CausalValueError
from .intervention import Set, Shift, Soft

if TYPE_CHECKING:
    from .cyclic import (
        CyclicRefusal,
        EquilibriumModel,
        FitOptions,
        OdePolicy,
        OdeSystem,
        OrnsteinUhlenbeck,
        OuMechanisms,
        PreparedCyclicFamily,
        SolvePolicy,
    )

SamplingDesign = Literal[
    "independent_subject_trajectories", "clustered_panel", "dependent_time_series"
]
ResamplingRule = Literal["row_subject_iid", "whole_cluster", "contiguous_block", "row_iid"]

# The resampling unit each declared design allows (the core sampling-design rules).
_DESIGN_RESAMPLING: dict[str, str] = {
    "independent_subject_trajectories": "row_subject_iid",
    "clustered_panel": "whole_cluster",
    "dependent_time_series": "contiguous_block",
}
_NEXT_ATTEMPT = re.compile(r"next attempt:\s*(?P<sentence>.+)", re.IGNORECASE | re.DOTALL)
_MAX_ID = 2**32 - 1


def _refusal(
    message: str, *, reason_code: str, detail: str, next_attempt: str | None
) -> CyclicRefusal:
    from .cyclic import CyclicRefusal

    return CyclicRefusal(message, reason_code=reason_code, detail=detail, next_attempt=next_attempt)


def _next_attempt_refusal(error: BaseException, detail: str) -> CyclicRefusal | None:
    """A native refusal that names its next attempt, as a :class:`CyclicRefusal`.

    A registered ``reason=<code>:`` the native layer wrote into the message is kept as the
    refusal's ``reason_code``; otherwise the route is ``route_not_supported``.
    """
    text = str(error)
    match = _NEXT_ATTEMPT.search(text)
    if match is None:
        return None
    refusal = _refusal(
        text,
        reason_code=getattr(error, "reason_code", None) or "route_not_supported",
        detail=detail,
        next_attempt=match.group("sentence").strip(),
    )
    facts = getattr(error, "execution_facts", None)
    if facts is not None:
        refusal.execution_facts = facts
    return refusal


def _identifier(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_ID:
        raise CausalValueError(f"{what} must be an integer id in [0, 2**32)")
    return value


def _label(value: Any) -> int:
    label = int(value)
    if isinstance(value, bool) or label != value or not 0 <= label < 2**64:
        raise CausalValueError("sampling-unit labels must be integers in [0, 2**64)")
    return label


def _count(value: Any, what: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CausalValueError(f"{what} must be an integer of at least {minimum}")
    return value


def _real(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CausalValueError(f"{what} must be a real number") from None
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _reals(values: Any, what: str) -> list[float]:
    if isinstance(values, str | bytes):
        raise CausalValueError(f"{what} must be a sequence of real numbers")
    return [_real(value, what) for value in values]


def _column(values: Any, name: str) -> list[float]:
    """One observed column as finite float64 values, converted once through NumPy."""
    import numpy as np

    try:
        column = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise CausalValueError(f"values of {name} must be real numbers") from None
    if column.ndim != 1 or not np.isfinite(column).all():
        raise CausalValueError(f"values of {name} must be a finite one-dimensional column")
    return column.tolist()


def _names(variables: Sequence[str], what: str = "variables") -> tuple[str, ...]:
    names = tuple(variables)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise CausalValueError(f"{what} must be a non-empty sequence of non-empty names")
    if len(set(names)) != len(names):
        raise CausalValueError(f"{what} must name each variable once")
    if len(names) > _MAX_ID:
        raise CausalValueError(f"too many {what}")
    return names


def _design(sampling_design: Any) -> str | None:
    if sampling_design is not None and sampling_design not in _DESIGN_RESAMPLING:
        raise CausalValueError(
            f"sampling_design={sampling_design!r} is not a declared design; use one of "
            + ", ".join(_DESIGN_RESAMPLING)
        )
    return sampling_design


def _resolve(name: Any, ids: Mapping[str, int], *, world: int | None = None) -> int:
    if name in ids:
        return ids[name]
    where = f"world {world}: " if world is not None else ""
    raise _refusal(
        f"{where}variable {name!r} is not declared",
        reason_code="unknown_variable",
        detail="cyclic_world.unknown_variable",
        next_attempt="name one of the declared variables " + ", ".join(map(repr, ids)),
    )


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InitialState:
    """Replacement of one declared initial-state law of a continuous-time family.

    ``law`` names a law declared by the family (by default each state's law carries the
    state's variable name); ``level`` is the replacement initial level, in that state's units.
    Only continuous-time families take initial-state changes; others refuse them.
    """

    law: str
    level: float

    def __post_init__(self) -> None:
        if not isinstance(self.law, str) or not self.law:
            raise CausalValueError("initial-state law must be a non-empty name")
        object.__setattr__(self, "level", _real(self.level, "initial level"))


@dataclass(frozen=True)
class World:
    """One scientific world: simultaneous interventions on named variables.

    ``id`` is the scientific world id results are keyed by; it is unique within a request.
    ``interventions`` holds :class:`~antecedent.intervention.Set` (hard clamp, the clamped
    equation is deleted), :class:`~antecedent.intervention.Shift` (additive shock on the
    variable's structural disturbance; feedback stays active) and
    :class:`~antecedent.intervention.Soft` (mechanism replacement of a named soft family:
    ``constant``, ``linear_mechanism`` or ``additive_shift``; the disturbance is kept). No
    interventions means the model as specified. A single intervention may be passed bare.
    """

    id: int
    interventions: tuple[Set | Shift | Soft, ...] = ()
    initial_states: tuple[InitialState, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.id, "world id")
        interventions = self.interventions
        if isinstance(interventions, Set | Shift | Soft):
            interventions = (interventions,)
        states = self.initial_states
        if isinstance(states, InitialState):
            states = (states,)
        object.__setattr__(self, "interventions", tuple(interventions))
        object.__setattr__(self, "initial_states", tuple(states))
        for state in self.initial_states:
            if not isinstance(state, InitialState):
                raise CausalValueError("initial_states must hold InitialState values")

    def _wire(self, ids: Mapping[str, int]) -> dict[str, Any]:
        actions = []
        for action in self.interventions:
            if isinstance(action, Set):
                actions.append(
                    {
                        "kind": "set",
                        "variable": _resolve(action.variable, ids, world=self.id),
                        "value": float(action.value),
                    }
                )
            elif isinstance(action, Shift):
                actions.append(
                    {
                        "kind": "shift",
                        "variable": _resolve(action.variable, ids, world=self.id),
                        "delta": float(action.delta),
                    }
                )
            elif isinstance(action, Soft):
                actions.append(
                    {
                        "kind": "soft",
                        "variable": _resolve(action.variable, ids, world=self.id),
                        "family": action.mechanism,
                        "parameters": [float(value) for value in action.parameters],
                    }
                )
            else:
                raise _refusal(
                    f"world {self.id}: {type(action).__name__} is not a cyclic world action",
                    reason_code="route_not_supported",
                    detail="cyclic_world.action_not_supported",
                    next_attempt="declare the action as Set, Shift or Soft on a named variable",
                )
        return {
            "world": self.id,
            "interventions": actions,
            "initial_states": [
                {"law": state.law, "level": state.level} for state in self.initial_states
            ],
        }


def _worlds(worlds: Sequence[World], ids: Mapping[str, int]) -> list[dict[str, Any]]:
    worlds = tuple(worlds)
    if not worlds or any(not isinstance(world, World) for world in worlds):
        raise CausalValueError("worlds must be a non-empty sequence of World values")
    return [world._wire(ids) for world in worlds]


def _compiled(
    request: dict[str, Any],
    names: tuple[str, ...] | None,
    *,
    detail: str,
    experimental: bool = False,
    max_bytes: int | None,
    cancel: Any,
) -> PreparedCyclicFamily:
    """Compile now, so a refused world surfaces at preparation with its next attempt."""
    from .cyclic import PreparedCyclicFamily

    request_json = json.dumps(request, allow_nan=False)
    prepare = (
        _native.prepare_cyclic_experimental_recovery
        if experimental
        else _native.prepare_cyclic_family
    )
    try:
        handle = prepare(request_json, max_bytes=max_bytes, cancel=cancel)
    except ValueError as error:
        refusal = _next_attempt_refusal(error, detail)
        if refusal is None:
            raise
        raise refusal from error
    family = PreparedCyclicFamily(request_json, variable_names=names)
    object.__setattr__(family, "_handle", handle)
    return family


@dataclass(frozen=True)
class LinearCyclicModel:
    """Specified linear Gaussian equilibrium ``x = K x + c + e``, ``e ~ N(mu, Sigma)``.

    ``coefficients`` is row-major ``K[into * n + from]`` with a zero diagonal; ``intercept``
    (``c``), ``noise_mean`` (``mu``) and the row-major ``noise_covariance`` (``Sigma``) are in
    variable order and in the variables' own units. Omitted intercept and noise mean are zero.
    The equilibrium of a world is unique when ``I - K`` restricted to its free block is
    invertible; the ``rho(|K|) < 1`` certificate is a uniqueness certificate of the declared
    model, not physical stability.
    """

    variables: tuple[str, ...]
    coefficients: tuple[float, ...]
    noise_covariance: tuple[float, ...]
    intercept: tuple[float, ...] | None = None
    noise_mean: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        names = _names(self.variables)
        n = len(names)
        coefficients = tuple(_reals(self.coefficients, "coefficients"))
        covariance = tuple(_reals(self.noise_covariance, "noise_covariance"))
        if len(coefficients) != n * n or len(covariance) != n * n:
            raise CausalValueError("coefficients and noise_covariance must be n*n row-major")
        intercept = tuple(_reals(self.intercept or (0.0,) * n, "intercept"))
        mean = tuple(_reals(self.noise_mean or (0.0,) * n, "noise_mean"))
        if len(intercept) != n or len(mean) != n:
            raise CausalValueError("intercept and noise_mean must have one value per variable")
        object.__setattr__(self, "variables", names)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "noise_covariance", covariance)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "noise_mean", mean)

    @classmethod
    def from_direct_effects(
        cls,
        variables: Sequence[str],
        effects: Mapping[tuple[str, str], float],
        *,
        noise_covariance: Sequence[float] | None = None,
        intercept: Sequence[float] | None = None,
        noise_mean: Sequence[float] | None = None,
    ) -> LinearCyclicModel:
        """Build ``K`` from named direct effects ``{(into, from): value}``.

        ``("B", "A"): 0.5`` declares ``B = 0.5 A + ...``. Unnamed entries are structural
        zeros. An omitted noise covariance is the identity (independent unit-variance
        disturbances).
        """
        names = _names(variables)
        ids = {name: index for index, name in enumerate(names)}
        n = len(names)
        coefficients = [0.0] * (n * n)
        for (into, source), value in effects.items():
            row, column = _resolve(into, ids), _resolve(source, ids)
            if row == column:
                raise CausalValueError("a direct effect of a variable on itself is not declared")
            coefficients[row * n + column] = _real(value, "direct effect")
        covariance = (
            tuple(noise_covariance)
            if noise_covariance is not None
            else tuple(1.0 if i == j else 0.0 for i in range(n) for j in range(n))
        )
        return cls(
            names,
            tuple(coefficients),
            covariance,
            None if intercept is None else tuple(intercept),
            None if noise_mean is None else tuple(noise_mean),
        )

    def worlds(
        self,
        worlds: Sequence[World],
        *,
        component_id: str,
        source_id: str,
        certifying_evidence: Sequence[str],
        certificate: Literal["required", "report_uncertified"] = "required",
        max_bytes: int | None = None,
        cancel: Any = None,
    ) -> PreparedCyclicFamily:
        """Prepare typed worlds of this model; compilation solves nothing.

        Each world evaluates to its Gaussian equilibrium law. ``certificate="required"``
        withholds a world whose ``rho(|K|) < 1`` certificate was not shown;
        ``"report_uncertified"`` answers a uniquely solvable world and marks it uncertified. A
        clamp that leaves no unique solution is a ``no_solution`` world with failed mass one,
        never a dropped world. ``source_id`` names where the model came from and does not
        certify the component; ``certifying_evidence`` must name other evidence.
        """
        if certificate not in ("required", "report_uncertified"):
            raise CausalValueError("certificate must be 'required' or 'report_uncertified'")
        ids = {name: index for index, name in enumerate(self.variables)}
        request = {
            "family": "linear_worlds",
            "variables": list(ids.values()),
            "model": {
                "coefficients": list(self.coefficients),
                "intercept": list(self.intercept or ()),
                "noise_mean": list(self.noise_mean or ()),
                "noise_covariance": list(self.noise_covariance),
            },
            "component": {
                "component_id": component_id,
                "source_id": source_id,
                "certifying_evidence": list(certifying_evidence),
            },
            "worlds": _worlds(worlds, ids),
            "certificate": certificate,
        }
        return _compiled(
            request,
            self.variables,
            detail="cyclic_world.linear",
            max_bytes=max_bytes,
            cancel=cancel,
        )


def equilibrium_worlds(
    model: EquilibriumModel,
    worlds: Sequence[World],
    noise: Sequence[Sequence[float]],
    *,
    variables: Sequence[str] | None = None,
    initial: Sequence[float] | None = None,
    policy: SolvePolicy | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """Typed worlds of a specified nonlinear equilibrium over shared disturbance draws."""
    from .cyclic import SolvePolicy

    names = _names(variables if variables is not None else [e.id for e in model.equations])
    if len(names) != len(model.equations):
        raise CausalValueError("variables must name one variable per equation")
    ids = {name: index for index, name in enumerate(names)}
    rows = [_reals(row, "noise") for row in noise]
    request = {
        "family": "equilibrium_worlds",
        "variables": list(ids.values()),
        "model": asdict(model),
        "worlds": _worlds(worlds, ids),
        "noise": rows,
        "initial": _reals(initial, "initial") if initial is not None else [0.0] * len(names),
        "policy": asdict(policy or SolvePolicy()),
    }
    return _compiled(
        request, names, detail="cyclic_world.equilibrium", max_bytes=max_bytes, cancel=cancel
    )


def ode_worlds(
    system: OdeSystem,
    initial: Sequence[float],
    times: Sequence[float],
    worlds: Sequence[World],
    *,
    variables: Sequence[str] | None = None,
    initial_laws: Sequence[str] | None = None,
    policy: OdePolicy | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """Typed worlds of a specified continuous-time system and its initial-state laws."""
    from .cyclic import OdePolicy

    names = _names(variables if variables is not None else [e.id for e in system.equations])
    if len(names) != len(system.equations):
        raise CausalValueError("variables must name one variable per state")
    laws = _names(initial_laws if initial_laws is not None else names, "initial_laws")
    if len(laws) != len(names):
        raise CausalValueError("initial_laws must name one law per state")
    ids = {name: index for index, name in enumerate(names)}
    request = {
        "family": "ode_worlds",
        "variables": list(ids.values()),
        "model": asdict(system),
        "initial": _reals(initial, "initial"),
        "initial_laws": list(laws),
        "times": _reals(times, "times"),
        "worlds": _worlds(worlds, ids),
        "policy": asdict(policy or OdePolicy()),
    }
    return _compiled(request, names, detail="cyclic_world.ode", max_bytes=max_bytes, cancel=cancel)


# ---------------------------------------------------------------------------
# Keyed results
# ---------------------------------------------------------------------------


def _law_meaning(origin: Mapping[str, Any]) -> str | None:
    kind = origin.get("origin", "supplied")
    if kind in ("asymptotic_normal", "bootstrap"):
        return "sampling_law"
    if kind == "posterior":
        return "parameter_posterior"
    return None


_READERS = {
    "gaussian_law": "mean() or covariance()",
    "equilibrium_draws": "draws()",
    "trajectory": "trajectory()",
}


@dataclass(frozen=True)
class WorldAnswer:
    """One evaluated world of a specified model, keyed by its scientific world id.

    ``outcome`` keeps the native status: ``gaussian_law`` (mean and row-major covariance in
    variable order and units), ``equilibrium_draws`` (one outcome per shared disturbance draw,
    failed draws in place), ``trajectory`` (states at the declared times), ``no_solution`` or
    ``withheld``. ``failed_mass`` and ``withheld_mass`` are never renormalized away.
    ``physical_stability`` is never claimed by these evaluators.
    """

    world: int
    family: str
    variables: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    outcome: dict[str, Any]
    certificate: dict[str, Any]
    solved_by: dict[str, Any]
    failed_mass: float
    withheld_mass: float
    physical_stability: str

    @property
    def status(self) -> str:
        """Native outcome status."""
        return str(self.outcome["status"])

    @property
    def certified(self) -> bool:
        """Whether the world's uniqueness certificate was shown for the declared model."""
        return bool(self.certificate["certified"])

    def _position(self, variable: int | str) -> int:
        if isinstance(variable, str):
            if variable not in self.variables:
                raise _refusal(
                    f"world {self.world}: variable {variable!r} is not in this world",
                    reason_code="unknown_variable",
                    detail="cyclic_world.unknown_variable",
                    next_attempt="name one of " + ", ".join(map(repr, self.variables)),
                )
            return self.variables.index(variable)
        if isinstance(variable, bool) or not 0 <= variable < len(self.variables):
            raise CausalValueError("variable position out of range")
        return variable

    def _require(self, status: str) -> dict[str, Any]:
        if self.status == status:
            return self.outcome
        if self.status in ("no_solution", "withheld"):
            attempt = (self.certificate.get("next_attempt") or {}).get(
                "sentence"
            ) or self.outcome.get("next_attempt")
            cause = self.outcome.get("cause", "a required certificate was not shown")
            raise _refusal(
                f"world {self.world} has no {status.replace('_', ' ')}: {cause}",
                reason_code="route_not_supported",
                detail=f"cyclic_world.{self.status}",
                next_attempt=attempt,
            )
        raise _refusal(
            f"world {self.world} evaluated to {self.status}, not {status}",
            reason_code="route_not_supported",
            detail="cyclic_world.outcome_kind",
            next_attempt=f"read this {self.status} world with {_READERS.get(self.status, 'its outcome')}",
        )

    def mean(self, variable: int | str) -> float:
        """Equilibrium mean of ``variable`` in a ``gaussian_law`` world, in its units.

        Raises :class:`CyclicRefusal` naming the next attempt for a world without a solution,
        a withheld world, or a world of another outcome kind.
        """
        position = self._position(variable)
        outcome = self._require("gaussian_law")
        return float(outcome["mean"][position])

    def covariance(self, variable: int | str, other: int | str) -> float:
        """Equilibrium covariance entry of a ``gaussian_law`` world."""
        row, column = self._position(variable), self._position(other)
        outcome = self._require("gaussian_law")
        return float(outcome["covariance"][row * len(self.variables) + column])

    def draws(self, variable: int | str) -> tuple[float | None, ...]:
        """Per-draw equilibrium value of an ``equilibrium_draws`` world; failed draws are None."""
        position = self._position(variable)
        outcome = self._require("equilibrium_draws")
        return tuple(
            float(item["values"][position]) if "values" in item else None
            for item in outcome["outcomes"]
        )

    def trajectory(self, variable: int | str) -> tuple[float, ...]:
        """State of ``variable`` at each declared observation time of a ``trajectory`` world."""
        position = self._position(variable)
        outcome = self._require("trajectory")
        return tuple(float(state[position]) for state in outcome["trajectory"]["states"])

    @classmethod
    def _from_wire(cls, answer: Mapping[str, Any], names: Sequence[str] | None) -> WorldAnswer:
        ids = list(answer["variables"])
        labels = tuple(
            names[i] if names is not None and 0 <= i < len(names) else str(i) for i in ids
        )
        actions = []
        for action in answer["actions"]:
            item = dict(action)
            variable = item["variable"]
            if names is not None and 0 <= variable < len(names):
                item["name"] = names[variable]
            actions.append(item)
        return cls(
            world=int(answer["world"]),
            family=str(answer["family"]),
            variables=labels,
            actions=tuple(actions),
            outcome=dict(answer["outcome"]),
            certificate=dict(answer["certificate"]),
            solved_by=dict(answer["solved_by"]),
            failed_mass=float(answer["failed_mass"]),
            withheld_mass=float(answer["withheld_mass"]),
            physical_stability=str(answer["physical_stability"]),
        )


@dataclass(frozen=True)
class RecoveredWorldAnswer:
    """One hard-clamp world of a recovered law, draw-aligned, keyed by world id.

    ``origin`` is how the draws were produced and ``law_meaning`` its evidence meaning:
    ``sampling_law`` (asymptotic normal or bootstrap), ``parameter_posterior``, or None for
    supplied draws. ``draws`` keep failed, inadmissible and uncertified draws at their weights;
    ``withheld_mass`` is the mass that did not enter the outcome mixture. ``outcome`` is the
    outcome's position among the law's retained units.
    """

    world: int
    origin: dict[str, Any]
    law_meaning: str | None
    outcome: int
    draws: tuple[dict[str, Any], ...]
    total_mass: float
    withheld_mass: float
    mixture_mean: float | None
    mixture_variance: float | None
    between_draw_variance: float | None

    def mean(self) -> float:
        """Weighted mean of certified draw means of the outcome, in its units.

        Raises :class:`CyclicRefusal` naming the next attempt when every draw was withheld
        (unresolved, inadmissible or uncertified).
        """
        if self.mixture_mean is None:
            raise _refusal(
                f"recovered world {self.world}: all {self.withheld_mass:g} of the law's mass "
                "was withheld from the outcome mixture",
                reason_code="route_not_supported",
                detail="cyclic_recovery.mass_withheld",
                next_attempt="add a regime that clamps each missing source while its target is "
                "passive, or read the draw statuses of the law",
            )
        return float(self.mixture_mean)

    @classmethod
    def _from_wire(
        cls, world: int, origin: Mapping[str, Any], answer: Mapping[str, Any]
    ) -> RecoveredWorldAnswer:
        def optional(key: str) -> float | None:
            value = answer.get(key)
            return None if value is None else float(value)

        return cls(
            world=int(world),
            origin=dict(origin),
            law_meaning=_law_meaning(origin),
            outcome=int(answer["outcome"]),
            draws=tuple(dict(draw) for draw in answer["draws"]),
            total_mass=float(answer["total_mass"]),
            withheld_mass=float(answer["withheld_mass"]),
            mixture_mean=optional("mixture_mean"),
            mixture_variance=optional("mixture_variance"),
            between_draw_variance=optional("between_draw_variance"),
        )


class CyclicInterventions(Mapping[int, "WorldAnswer | RecoveredWorldAnswer"]):
    """Evaluated intervention worlds keyed by scientific world id, in request order."""

    __slots__ = ("_worlds", "execution_facts")

    def __init__(
        self,
        worlds: Mapping[int, WorldAnswer | RecoveredWorldAnswer],
        execution_facts: Any = None,
    ) -> None:
        self._worlds = dict(worlds)
        #: Native execution facts of the evaluation.
        self.execution_facts = execution_facts

    def __getitem__(self, world: int) -> WorldAnswer | RecoveredWorldAnswer:
        try:
            return self._worlds[world]
        except KeyError:
            raise KeyError(f"no world with id {world!r}; ids are {list(self._worlds)}") from None

    def __iter__(self) -> Iterator[int]:
        return iter(self._worlds)

    def __len__(self) -> int:
        return len(self._worlds)

    def __repr__(self) -> str:
        return f"CyclicInterventions({list(self._worlds)})"

    @classmethod
    def _from_native(
        cls, payload: Sequence[Mapping[str, Any]], names: Sequence[str] | None, facts: Any
    ) -> CyclicInterventions:
        worlds: dict[int, WorldAnswer | RecoveredWorldAnswer] = {}
        for item in payload:
            if item["source"] == "specified":
                answer = WorldAnswer._from_wire(item["answer"], names)
                worlds[answer.world] = answer
            else:
                worlds[int(item["world"])] = RecoveredWorldAnswer._from_wire(
                    item["world"], item["origin"], item["answer"]
                )
        return cls(worlds, facts)


# ---------------------------------------------------------------------------
# Experimental recovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentRegime:
    """One declared clamp-experiment regime and its sample.

    ``data`` maps each observed unit (a declared variable name) to its values, one per row,
    in that unit's own units; a clamped unit's column holds its randomized clamp draws.
    ``clamps`` maps each surgically clamped unit to its declared level (or the mean of its
    randomized clamp draws); no clamps is the passive regime. ``source`` is the identity of
    the regime's sample; ``population`` the population its rows are drawn from.
    ``dependent=True`` declares that the regime shares its disturbance with another regime.
    ``sampling_units`` labels each row's sampling unit (subject or cluster) for whole-unit
    resampling; label every regime or none.
    """

    data: Mapping[str, Sequence[float]]
    source: int
    clamps: Mapping[str, float] = field(default_factory=dict)
    population: int = 0
    dependent: bool = False
    sampling_units: Sequence[int] | None = None

    def _wire(self, ids: Mapping[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
        source = self.source
        if isinstance(source, bool) or not isinstance(source, int) or not 0 <= source < 2**64:
            raise CausalValueError("regime source must be an integer id in [0, 2**64)")
        if not self.data:
            raise CausalValueError("a regime needs at least one observed unit")
        units = sorted(_resolve(name, ids) for name in self.data)
        regime = {
            "clamps": [
                {"unit": _resolve(name, ids), "value": _real(value, "clamp level")}
                for name, value in self.clamps.items()
            ],
            "units": units,
            "population": _identifier(self.population, "population"),
            "dependent": bool(self.dependent),
            "source": source,
        }
        sample = {
            "source": source,
            "columns": [
                {"unit": _resolve(name, ids), "values": _column(values, name)}
                for name, values in self.data.items()
            ],
            "sampling_units": (
                None
                if self.sampling_units is None
                else [_label(label) for label in self.sampling_units]
            ),
        }
        return regime, sample


@dataclass(frozen=True)
class ExperimentalFitOptions:
    """Row weighting, joint coefficient covariance and disturbance divisor of the fit.

    The byte budget of the fit comes from the execution context (``max_bytes``).
    ``covariance="not_computed"`` refuses asymptotic draws.
    """

    weighting: Literal["efficient", "unweighted"] = "efficient"
    covariance: Literal["sandwich", "model_based", "not_computed"] = "sandwich"
    divisor: Literal["row_count", "row_count_minus_one"] = "row_count"


@dataclass(frozen=True)
class AsymptoticNormal:
    """Draws from the joint asymptotic normal law of the fit: a sampling law, not a posterior.

    Rows are independent within each regime; under a clustered or time-series design use
    :class:`RecoveryBootstrap` instead.
    """

    draws: int
    seed: int = 0


@dataclass(frozen=True)
class RecoveryBootstrap:
    """Whole-recovery bootstrap of the experimental units: a sampling law.

    The resampling unit follows the study's declared ``sampling_design`` (whole subjects,
    whole clusters, or contiguous blocks of ``block_length`` rows); ``rule`` overrides it
    only to have Rust refuse a rule the design does not allow. Failed replicates keep their
    weight as failed mass.
    """

    replicates: int
    seed: int = 0
    rule: ResamplingRule | None = None
    block_length: int | None = None


@dataclass(frozen=True)
class FlatCoefficients:
    """Improper flat prior on the identified coefficient directions."""


@dataclass(frozen=True)
class NormalCoefficients:
    """Independent normal coefficient prior, projected onto the identified subspace only."""

    mean: float
    variance: float


@dataclass(frozen=True)
class KnownDisturbance:
    """Known disturbance variances, by retained unit name, in squared units."""

    variances: Mapping[str, float]


@dataclass(frozen=True)
class InverseWishartDisturbance:
    """Scaled inverse chi-square prior on each diagonal disturbance variance."""

    degrees_of_freedom: float
    scale: Mapping[str, float]


@dataclass(frozen=True)
class RecoveryPrior:
    """Prior of a recovered-law posterior; the intercept prior is flat."""

    coefficients: FlatCoefficients | NormalCoefficients
    disturbance: KnownDisturbance | InverseWishartDisturbance


@dataclass(frozen=True)
class RecoveryPosterior:
    """Gaussian posterior over the identified coefficient subspace: a parameter posterior.

    Unresolved directions stay unresolved in every draw; the prior never completes them.
    """

    prior: RecoveryPrior
    draws: int
    burn_in: int = 0
    seed: int = 0


RecoveryUncertainty = AsymptoticNormal | RecoveryBootstrap | RecoveryPosterior


def _by_unit(values: Mapping[str, float], retained: Sequence[str], what: str) -> list[float]:
    if set(values) != set(retained):
        raise CausalValueError(f"{what} must name exactly the retained units {list(retained)}")
    return [_real(values[name], what) for name in retained]


def _inference_wire(
    uncertainty: Any, design: str | None, retained: Sequence[str]
) -> dict[str, Any]:
    if isinstance(uncertainty, RecoveryBootstrap):
        if design is None:
            raise _refusal(
                "a bootstrap recovery resamples the unit the study's sampling design names, "
                "and no sampling design was declared",
                reason_code="required_option_missing",
                detail="cyclic_recovery.sampling_design_required",
                next_attempt="declare sampling_design= ('independent_subject_trajectories', "
                "'clustered_panel' or 'dependent_time_series')",
            )
        return {
            "method": "bootstrap",
            "replicates": _count(uncertainty.replicates, "replicates"),
            "seed": _count(uncertainty.seed, "seed", minimum=0),
            "design": design,
            "rule": uncertainty.rule or _DESIGN_RESAMPLING[design],
            "block_length": uncertainty.block_length,
        }
    if not isinstance(uncertainty, AsymptoticNormal | RecoveryPosterior):
        raise CausalValueError(
            "uncertainty must be AsymptoticNormal, RecoveryBootstrap or RecoveryPosterior"
        )
    # Whether a row-independent origin is allowed under the declared design is decided by
    # Rust request compilation, which reads the design from the fit options.
    if isinstance(uncertainty, AsymptoticNormal):
        return {
            "method": "asymptotic_normal",
            "draws": _count(uncertainty.draws, "draws"),
            "seed": _count(uncertainty.seed, "seed", minimum=0),
        }
    prior = uncertainty.prior
    if isinstance(prior.coefficients, NormalCoefficients):
        coefficients = {
            "kind": "normal",
            "mean": _real(prior.coefficients.mean, "prior mean"),
            "variance": _real(prior.coefficients.variance, "prior variance"),
        }
    elif isinstance(prior.coefficients, FlatCoefficients):
        coefficients = {"kind": "flat"}
    else:
        raise CausalValueError("coefficient prior must be FlatCoefficients or NormalCoefficients")
    if isinstance(prior.disturbance, KnownDisturbance):
        disturbance = {
            "kind": "known",
            "variances": _by_unit(prior.disturbance.variances, retained, "variances"),
        }
    elif isinstance(prior.disturbance, InverseWishartDisturbance):
        disturbance = {
            "kind": "inverse_wishart",
            "degrees_of_freedom": _real(prior.disturbance.degrees_of_freedom, "degrees"),
            "scale": _by_unit(prior.disturbance.scale, retained, "scale"),
        }
    else:
        raise CausalValueError(
            "disturbance prior must be KnownDisturbance or InverseWishartDisturbance"
        )
    return {
        "method": "posterior",
        "prior": {"coefficients": coefficients, "disturbance": disturbance},
        "draws": _count(uncertainty.draws, "draws"),
        "burn_in": _count(uncertainty.burn_in, "burn_in", minimum=0),
        "seed": _count(uncertainty.seed, "seed", minimum=0),
    }


def prepare_experimental_recovery(
    variables: Sequence[str],
    regimes: Sequence[ExperimentRegime],
    worlds: Sequence[World],
    *,
    outcome: str,
    uncertainty: RecoveryUncertainty,
    source_id: str,
    certifying_evidence: Sequence[str],
    sampling_design: SamplingDesign | None = None,
    dropped: Sequence[str] = (),
    fit: ExperimentalFitOptions | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """Prepare a linear equilibrium to be recovered from randomized clamp experiments.

    ``variables`` declares every unit by name; ``dropped`` names units the mask removes
    before recovery. Preparation admits the worlds on the retained units and does not fit.
    Execution fits the declared regimes, draws the declared ``uncertainty``
    (:class:`AsymptoticNormal`, :class:`RecoveryBootstrap` or :class:`RecoveryPosterior`)
    and evaluates each hard-clamp world from the portable law; failed and inadmissible draws
    stay at their weights. Directions the regimes do not pin down stay unresolved in every
    draw. A recovered law takes hard clamps only: a shift or mechanism replacement is refused
    with its next attempt. ``sampling_design`` is the study's declaration of how rows were
    sampled; the bootstrap resamples the unit it names, and Rust refuses a row-independent
    origin under a clustered or time-series design (``estimator_inference_mismatch``, naming a
    bootstrap under the declared design). The law stays conditional on the clamp-experiment
    recipe and grants no scientific license; ``certifying_evidence`` must name evidence
    other than ``source_id``.
    """
    names = _names(variables)
    ids = {name: index for index, name in enumerate(names)}
    design = _design(sampling_design)
    regimes = tuple(regimes)
    if not regimes or any(not isinstance(regime, ExperimentRegime) for regime in regimes):
        raise CausalValueError("regimes must be a non-empty sequence of ExperimentRegime")
    wires = [regime._wire(ids) for regime in regimes]
    masked = {_resolve(name, ids) for name in dropped}
    observed = {unit for regime, _ in wires for unit in regime["units"]}
    retained = [name for name in names if ids[name] in observed and ids[name] not in masked]
    options = asdict(fit or ExperimentalFitOptions())
    inference = _inference_wire(uncertainty, design, retained)
    if design is not None and inference["method"] != "bootstrap":
        # The bootstrap carries its design; a row-independent origin declares it here.
        options["sampling_design"] = design
    request = {
        "family": "experimental_recovery",
        "regimes": [regime for regime, _ in wires],
        "dropped": sorted(masked),
        "samples": [sample for _, sample in wires],
        "options": options,
        "inference": inference,
        "worlds": _worlds(worlds, ids),
        "outcome": _resolve(outcome, ids),
        "source_id": source_id,
        "certifying_evidence": list(certifying_evidence),
    }
    return _compiled(
        request,
        names,
        detail="cyclic_recovery.world",
        experimental=True,
        max_bytes=max_bytes,
        cancel=cancel,
    )


# ---------------------------------------------------------------------------
# Dynamic-fit inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OdeState:
    """ODE state ``variable`` (a state position) at ``time``, from ``initial`` at ``start``.

    ``schedule`` holds timed sustained clamps ``{"time": t, "clamps": [{"variable": i,
    "value": v}]}``. Time is in the system's own time units.
    """

    variable: int
    time: float
    initial: tuple[float, ...]
    start: float = 0.0
    schedule: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "ode_state",
            "initial": _reals(self.initial, "initial"),
            "start": _real(self.start, "start"),
            "time": _real(self.time, "time"),
            "schedule": list(self.schedule),
            "variable": self.variable,
        }


@dataclass(frozen=True)
class OuStationaryMean:
    """Stationary OU mean of ``variable`` after sustained hard ``clamps`` (empty: none)."""

    variable: int
    clamps: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "ou_stationary_mean",
            "clamps": list(self.clamps),
            "variable": self.variable,
        }


@dataclass(frozen=True)
class OuStationaryCovariance:
    """Stationary OU covariance entry ``(variable, other)`` after sustained hard clamps."""

    variable: int
    other: int
    clamps: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "ou_stationary_covariance",
            "clamps": list(self.clamps),
            "variable": self.variable,
            "other": self.other,
        }


@dataclass(frozen=True)
class OuMechanismMean:
    """Stationary mean of shared OU mechanisms after equation ``replacements``."""

    variable: int
    replacements: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "ou_mechanism_mean",
            "replacements": list(self.replacements),
            "variable": self.variable,
        }


@dataclass(frozen=True)
class OuMechanismCovariance:
    """Stationary covariance entry of shared OU mechanisms after equation replacements."""

    variable: int
    other: int
    replacements: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "ou_mechanism_covariance",
            "replacements": list(self.replacements),
            "variable": self.variable,
            "other": self.other,
        }


DynamicQuantity = (
    OdeState | OuStationaryMean | OuStationaryCovariance | OuMechanismMean | OuMechanismCovariance
)


@dataclass(frozen=True)
class DynamicBootstrap:
    """Whole-procedure bootstrap of a fitted dynamic family: a sampling law.

    ``kind="parametric"`` simulates every unit from the fit; ``"nonparametric"`` resamples
    whole units within ``stratification`` strata. The resampling unit follows the declared
    sampling design; ``resampling`` overrides it only to have Rust refuse a disallowed rule.
    Failed replicates keep their slots and weight.
    """

    replicates: int
    kind: Literal["parametric", "nonparametric"] = "nonparametric"
    seed: int = 0
    resampling: ResamplingRule | None = None
    stratification: Literal["fixed_assignments", "randomized_within_targets"] = "fixed_assignments"


@dataclass(frozen=True)
class ObservedInformation:
    """Observed information and its delta-method covariance, only when the likelihood is regular.

    An irregular likelihood (unconverged, flat or ill-conditioned) reports its reason and no
    covariance.
    """


@dataclass(frozen=True)
class DynamicPosterior:
    """Posterior under an independent normal prior over the estimated coordinates.

    ``mean`` and ``standard_deviation`` give one prior value per estimated coordinate, in
    estimate order. Inspect ``diagnostics_pass`` before interpreting draws.
    """

    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    seed: int = 0
    chains: int = 4
    warmup: int = 500
    draws: int = 1000
    proposal_scale: float | None = None
    start_spread: float = 1.0


DynamicMethod = DynamicBootstrap | ObservedInformation | DynamicPosterior


def _method_wire(method: Any, design: str) -> dict[str, Any]:
    if isinstance(method, DynamicBootstrap):
        return {
            "method": "bootstrap",
            "kind": method.kind,
            "replicates": _count(method.replicates, "replicates"),
            "design": design,
            "resampling": method.resampling or _DESIGN_RESAMPLING[design],
            "stratification": method.stratification,
            "seed": _count(method.seed, "seed", minimum=0),
        }
    if isinstance(method, ObservedInformation):
        return {"method": "information", "design": design}
    if isinstance(method, DynamicPosterior):
        return {
            "method": "posterior",
            "prior": {
                "mean": _reals(method.mean, "prior mean"),
                "standard_deviation": _reals(method.standard_deviation, "prior sd"),
            },
            "options": {
                "chains": method.chains,
                "warmup": method.warmup,
                "draws": method.draws,
                "proposal_scale": method.proposal_scale,
                "start_spread": method.start_spread,
            },
            "design": design,
            "seed": _count(method.seed, "seed", minimum=0),
        }
    raise CausalValueError(
        "method must be DynamicBootstrap, ObservedInformation or DynamicPosterior"
    )


def _dynamic_inference(
    family: str,
    problem: dict[str, Any],
    *,
    method: DynamicMethod,
    quantities: Sequence[DynamicQuantity],
    sampling_design: SamplingDesign,
    max_bytes: int | None,
    cancel: Any,
) -> PreparedCyclicFamily:
    design = _design(sampling_design)
    if design is None:
        raise _refusal(
            "dynamic inference follows the study's declared sampling design",
            reason_code="required_option_missing",
            detail="cyclic_dynamic_inference.sampling_design_required",
            next_attempt="declare sampling_design= ('independent_subject_trajectories', "
            "'clustered_panel' or 'dependent_time_series')",
        )
    request = {
        "family": "dynamic_inference",
        "problem": {"family": family, "problem": problem},
        "method": _method_wire(method, design),
        "quantities": [quantity._wire() for quantity in quantities],
    }
    return _compiled(
        request, None, detail="cyclic_dynamic_inference", max_bytes=max_bytes, cancel=cancel
    )


def infer_ode(
    system: OdeSystem,
    parameters: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    method: DynamicMethod,
    quantities: Sequence[DynamicQuantity] = (),
    sampling_design: SamplingDesign,
    integration: OdePolicy | None = None,
    learning_rate: float = 0.1,
    score_tolerance: float = 1e-6,
    max_iterations: int = 1000,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """See :meth:`antecedent.cyclic.OdeSystem.infer`."""
    from .cyclic import OdePolicy

    problem = {
        "template": asdict(system),
        "parameters": parameters,
        "data": observations,
        "integration": asdict(integration or OdePolicy()),
        "policy": {
            "learning_rate": learning_rate,
            "score_tolerance": score_tolerance,
            "max_iterations": max_iterations,
        },
    }
    return _dynamic_inference(
        "ode",
        problem,
        method=method,
        quantities=quantities,
        sampling_design=sampling_design,
        max_bytes=max_bytes,
        cancel=cancel,
    )


def infer_ou(
    process: OrnsteinUhlenbeck,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    method: DynamicMethod,
    quantities: Sequence[DynamicQuantity] = (),
    sampling_design: SamplingDesign,
    options: FitOptions | None = None,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """See :meth:`antecedent.cyclic.OrnsteinUhlenbeck.infer`."""
    from .cyclic import FitOptions

    problem = {
        "template": asdict(process),
        "parameters": parameters,
        "rows": rows,
        "options": asdict(options or FitOptions()),
    }
    return _dynamic_inference(
        "ou",
        problem,
        method=method,
        quantities=quantities,
        sampling_design=sampling_design,
        max_bytes=max_bytes,
        cancel=cancel,
    )


def infer_ou_mechanisms(
    mechanisms: OuMechanisms,
    parameters: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
    *,
    method: DynamicMethod,
    quantities: Sequence[DynamicQuantity] = (),
    sampling_design: SamplingDesign,
    learning_rate: float = 0.1,
    score_tolerance: float = 1e-6,
    max_iterations: int = 1000,
    max_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedCyclicFamily:
    """See :meth:`antecedent.cyclic.OuMechanisms.infer`."""
    problem = {
        "template": asdict(mechanisms),
        "parameters": parameters,
        "regimes": regimes,
        "policy": {
            "learning_rate": learning_rate,
            "score_tolerance": score_tolerance,
            "max_iterations": max_iterations,
        },
    }
    return _dynamic_inference(
        "ou_mechanisms",
        problem,
        method=method,
        quantities=quantities,
        sampling_design=sampling_design,
        max_bytes=max_bytes,
        cancel=cancel,
    )


# ---------------------------------------------------------------------------
# Evidence selectors
# ---------------------------------------------------------------------------


def _selector_position(variable: int | str, names: Sequence[str] | None) -> int:
    if isinstance(variable, str):
        if names is None or variable not in names:
            raise _refusal(
                f"selector variable {variable!r} is not a declared variable",
                reason_code="unknown_variable",
                detail="cyclic_evidence.unknown_variable",
                next_attempt="name a declared variable or give its position",
            )
        return list(names).index(variable)
    return _count(variable, "selector variable", minimum=0)


@dataclass(frozen=True)
class WorldSelector:
    """One typed world of a specified model, by scientific world id.

    A Gaussian-law world binds its mean to the interventional port (variance kept as a
    diagnostic); disturbance-draw worlds bind the scenario port; ``time`` selects an
    observation-time index of a trajectory. Certificate, solve route and failed or withheld
    mass travel with the column.
    """

    world: int
    variable: int | str
    time: int | None = None

    def _wire(self, names: Sequence[str] | None) -> dict[str, Any]:
        return {
            "kind": "world",
            "world": _identifier(self.world, "world id"),
            "variable": _selector_position(self.variable, names),
            "time": self.time,
        }


@dataclass(frozen=True)
class RecoveredResponseSelector:
    """Draw-aligned response of one recovered world (``world`` is its request position).

    The port follows the law's draw origin: supplied draws stay on the interventional port,
    a sampling law projects the per-draw interventional mean onto the sampling-law port, and
    a posterior onto the parameter-posterior port. ``variable`` is a retained-unit position.
    """

    world: int
    variable: int

    def _wire(self, names: Sequence[str] | None) -> dict[str, Any]:
        del names
        return {
            "kind": "response",
            "world": _count(self.world, "world position", minimum=0),
            "noise_row": 0,
            "variable": _count(self.variable, "variable position", minimum=0),
        }


@dataclass(frozen=True)
class RecoveredDrawSelector:
    """The interventional law within one recovered draw: its equilibrium mean.

    Its variance stays a diagnostic. This is outcome variation within a draw, not parameter
    uncertainty.
    """

    world: int
    draw: int
    variable: int

    def _wire(self, names: Sequence[str] | None) -> dict[str, Any]:
        del names
        return {
            "kind": "recovered_draw",
            "world": _count(self.world, "world position", minimum=0),
            "draw": _count(self.draw, "draw position", minimum=0),
            "variable": _count(self.variable, "variable position", minimum=0),
        }


@dataclass(frozen=True)
class DynamicQuantitySelector:
    """One declared quantity of a dynamic inference, by its position in ``quantities``.

    A bootstrap binds the sampling-law port, observed information the covariance-influence
    port, and a posterior the parameter-posterior port; failed replicates and draws keep
    their slots.
    """

    quantity: int

    def _wire(self, names: Sequence[str] | None) -> dict[str, Any]:
        del names
        return {
            "kind": "dynamic_quantity",
            "quantity": _count(self.quantity, "quantity position", minimum=0),
        }


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

# Request fields whose change invalidates a retained fit, per family (mirrors the native
# fit-dependency identity; every other field reuses the frozen statistical product).
_FIT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "frequentist_bootstrap": (
        "design",
        "model",
        "parameters",
        "rows",
        "policy",
        "replicates",
        "seed",
    ),
    "parameter_posterior": ("model", "parameters", "rows", "prior_sd", "sampling"),
    "joint_counterfactual": (
        "model",
        "parameters",
        "rows",
        "prior_sd",
        "observations",
        "measurement_variances",
        "initial",
        "sampling",
    ),
    "partial_counterfactual": (
        "model",
        "observations",
        "measurement_variances",
        "initial",
        "sampling",
    ),
    "ou_mechanisms": ("model", "regimes", "parameters", "policy"),
    "gaussian_functional": ("source", "source_identity", "program", "law"),
    "experimental_recovery": ("regimes", "dropped", "samples", "options", "inference"),
    "dynamic_inference": ("problem",),
    "system": ("request.model", "request.fit"),
}


def _origin_of(inference: Mapping[str, Any]) -> dict[str, Any]:
    method = inference["method"]
    if method == "asymptotic_normal":
        return {"origin": method, "draws": inference["draws"], "seed": inference["seed"]}
    if method == "bootstrap":
        return {
            "origin": method,
            "replicates": inference["replicates"],
            "seed": inference["seed"],
            "design": inference["design"],
            "rule": inference["rule"],
        }
    return {
        "origin": "posterior",
        "prior": inference["prior"],
        "draws": inference["draws"],
        "burn_in": inference.get("burn_in", 0),
        "seed": inference["seed"],
    }


def _distinct(directions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for direction in directions:
        seen.setdefault(json.dumps(direction, sort_keys=True), dict(direction))
    return list(seen.values())


def family_facts(
    request: Mapping[str, Any],
    law: Mapping[str, Any] | None,
    design_unresolved: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Draw origin, unresolved directions, fit dependencies and worlds of a family request.

    ``law`` is the recovered law of the last execution on the handle, when there was one.
    ``design_unresolved`` holds the rows the clamp design leaves unresolved, as the native
    compiler reads them before any fit; an experimental recovery reports them as its
    ``unresolved_directions`` until a fit on the handle reports the law's own.
    """
    family = request["family"]
    facts: dict[str, Any] = {
        "draw_origin": None,
        "law_meaning": None,
        "unresolved_directions": None,
        "fit_dependencies": list(_FIT_DEPENDENCIES.get(family, ("*",))),
    }
    if family == "recovered_linear":
        origin = dict(request.get("uncertainty") or {"origin": "supplied"})
        facts["draw_origin"] = origin
        facts["law_meaning"] = _law_meaning(origin)
        facts["unresolved_directions"] = _distinct(
            [item for draw in request["draws"] for item in draw.get("unresolved", [])]
        )
    elif family == "experimental_recovery":
        origin = _origin_of(request["inference"])
        facts["draw_origin"] = origin
        facts["law_meaning"] = _law_meaning(origin)
        if design_unresolved is not None:
            facts["unresolved_directions"] = [dict(row) for row in design_unresolved]
        if law is not None:
            facts["unresolved_directions"] = _distinct(
                [item for draw in law["draws"] for item in draw.get("unresolved", [])]
            )
            facts["draw_status"] = [status["status"] for status in law["draw_status"]]
            facts["failed_mass"] = law["failed_mass"]
    elif family == "dynamic_inference":
        method = request["method"]["method"]
        facts["uncertainty_method"] = method
        facts["law_meaning"] = {
            "bootstrap": "sampling_law",
            "information": "covariance_influence",
            "posterior": "parameter_posterior",
        }[method]
    if "worlds" in request and family in (
        "linear_worlds",
        "equilibrium_worlds",
        "ode_worlds",
        "experimental_recovery",
    ):
        facts["worlds"] = [world["world"] for world in request["worlds"]]
    return facts
