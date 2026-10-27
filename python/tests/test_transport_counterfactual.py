"""General finite counterfactual transport from Python, exact and sampled.

Every expected number is enumerated in this file from each population's
structural equations with exact rational arithmetic, independently of the
library: the evidence tables handed to the library are built here, and no native
code produces a truth. The same hand-checked values appear as literals so the
enumerator is itself checked.

Fixtures (finite structural models; ``U*`` are independent exogenous terms):

* mediation with a ternary mediator and a covariate shift in ``Z``::

      Z = Uz            X = Z xor Ux       M = (X + Z + Um) mod 3
      Y = [M + X*Z >= 2] xor Uy            Uz ~ Bern(1/4 source, 3/4 target)
      Ux ~ Bern(1/4)    Um ~ (1/4, 1/4, 1/2)    Uy ~ Bern(1/8)

  The mediation formula ``sum_{z,m} P(y|x,m,z) P(m|x',z) P*(z)`` gives
  ``P*(Y_{x,M_{x'}} = 1)``: 23/64 (x=0, x'=0), 5/16 (0, 1), 1/2 (1, 0), 19/32
  (1, 1); hence NDE = 1/2 - 23/64 = 9/64 and NIE = 5/16 - 23/64 = -3/64;
* a confounded covariate model (``X <-> Y``, selection on ``Z``) where
  ``P*(Y_{x=1} = 1) = 19/32``, ``P*(Y_{x=1} = 1, X = 0) = 47/128`` and
  ``P*(Y_{x=1} = 1 | X = 0) = 47/64`` need the source observational law, the
  source ``do(X)`` laws and the target marginal of ``Z`` together;
* a front-door graph with a latent ``X <-> Y`` district, selection on ``M``,
  where the target never measures ``Y``: ``P*(Y_{x=1} = 1) = 1/2``;
* a recanting-witness graph (Avin, Shpitser and Pearl 2005): the two models
  below have identical observational and interventional laws and
  ``P(Y_{x=1, Z_{x=0}} = 1)`` is 3/4 against 1/4;
* selection on the outcome mechanism: two target models agree on every supplied
  table and give ``P*(Y_{x=1} = 1)`` 7/8 against 7/16;
* a missing-experiment pair: with the source experiment withheld two models
  agree on every supplied table and give 19/32 against 23/64.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cache

import numpy as np
import pytest
from antecedent import Admg
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.state import CancellationToken
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

EXACT_TOLERANCE = 1e-12
#: Absolute tolerance of a plug-in at the large-sample size below, declared
#: before the run: about five standard deviations of the least precise estimand.
PLUG_IN_TOLERANCE = 0.01
LARGE_N = 200_000


# ---------------------------------------------------------------------------
# The independent enumerator.
# ---------------------------------------------------------------------------


class Scm:
    """A finite structural model: independent exogenous terms and deterministic
    mechanisms in causal order. Probabilities are exact rationals."""

    def __init__(self, exo, endo):
        self.exo = exo  # name -> list of Fraction
        self.endo = endo  # list of (name, number of levels, fn(values, exo) -> level)

    @property
    def names(self):
        return [name for name, _, _ in self.endo]

    def levels(self, name):
        return next(k for n, k, _ in self.endo if n == name)

    def states(self):
        keys = list(self.exo)
        for combo in itertools.product(*(range(len(self.exo[k])) for k in keys)):
            weight = F(1)
            for key, level in zip(keys, combo, strict=True):
                weight *= self.exo[key][level]
            if weight:
                yield dict(zip(keys, combo, strict=True)), weight

    def solve(self, u, do):
        values = {}
        for name, _, mechanism in self.endo:
            values[name] = do[name] if name in do else mechanism(values, u)
        return values

    def with_exo(self, name, law):
        return Scm({**self.exo, name: law}, self.endo)

    def blind_to_natural(self, outcome, treatment):
        """``outcome`` reads the natural value of ``treatment`` under any
        intervention: observationally equivalent, interventionally different."""
        mechanisms = {n: f for n, _, f in self.endo}
        m_t, m_y = mechanisms[treatment], mechanisms[outcome]

        def blind(values, u):
            seen = dict(values)
            seen[treatment] = m_t(values, u)
            return m_y(seen, u)

        return Scm(self.exo, [(n, k, blind if n == outcome else f) for n, k, f in self.endo])

    def value(self, term, u):
        """``term`` is a variable name or ``(variable, {intervened: level | term})``."""
        if isinstance(term, str):
            return self.solve(u, {})[term]
        variable, settings = term
        do = {
            name: (setting if isinstance(setting, int) else self.value(setting, u))
            for name, setting in settings.items()
        }
        return self.solve(u, do)[variable]

    def probability(self, event, given=()):
        """``P(event | given)`` over one shared exogenous state; ``None`` when
        the conditioning event has probability zero."""
        joint = evidence = F(0)
        for u, weight in self.states():
            if all(self.value(t, u) == level for t, level in given):
                evidence += weight
                if all(self.value(t, u) == level for t, level in event):
                    joint += weight
        return None if evidence == 0 else joint / evidence

    def law(self, do, measured):
        """``P(measured | do(do))``, row-major in ``measured`` order, last fastest."""
        sizes = [self.levels(m) for m in measured]
        table = [F(0)] * int(np.prod(sizes, dtype=np.int64))
        for u, weight in self.states():
            values = self.solve(u, do)
            index = 0
            for m, size in zip(measured, sizes, strict=True):
                index = index * size + values[m]
            table[index] += weight
        return table


def bern(p):
    return [1 - p, p]


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------


def _regime_id(population, do):
    return f"{population}|{'+'.join(sorted(do)) or 'obs'}"


@dataclass(frozen=True)
class Case:
    name: str
    directed: tuple
    bidirected: tuple
    models: dict  # population -> Scm
    target: str
    sources: tuple  # CounterfactualTransportSource
    specs: tuple  # (population, do mapping, measured-before-intervention variables)

    @property
    def names(self):
        return self.models[self.target].names

    def graph(self):
        return Admg.from_edges(self.names, list(self.directed), list(self.bidirected))

    def levels(self):
        scm = self.models[self.target]
        return {n: tuple(float(v) for v in range(scm.levels(n))) for n in self.names}

    def catalog(self, specs=None):
        regimes = {}
        for population, do, variables in specs if specs is not None else self.specs:
            key = (population, tuple(sorted(do)))
            if key in regimes:
                continue
            regimes[key] = transport.EvidenceRegime(
                _regime_id(population, do),
                population,
                kind="experimental" if do else "observational",
                interventions=sorted(do),
                measured=[v for v in variables if v not in do],
            )
        return transport.EvidenceCatalog(regimes=tuple(regimes.values()))

    def laws(self, specs=None, models=None):
        models = self.models if models is None else models
        out = []
        for population, do, variables in specs if specs is not None else self.specs:
            scm = models[population]
            measured = [v for v in variables if v not in do]
            table = scm.law(do, measured)
            world = ".".join(f"{k}{v}" for k, v in sorted(do.items()))
            out.append(
                transport.ExactDiscreteLaw(
                    population,
                    _regime_id(population, do),
                    tuple((m, tuple(float(v) for v in range(scm.levels(m)))) for m in measured),
                    tuple(float(p) for p in table),
                    f"snap|{population}|{_regime_id(population, do)}|{world}",
                    interventions=tuple((k, float(v)) for k, v in sorted(do.items())),
                )
            )
        return out

    def query(self, worlds, event, given=(), target=None, sources=None):
        return transport.CounterfactualTransportQuery(
            target=target or self.target,
            levels=self.levels(),
            worlds=worlds,
            event=event,
            given=given,
            sources=self.sources if sources is None else sources,
        )

    def decide(self, worlds, event, given=(), *, specs=None, **options):
        return transport.identify_counterfactual(
            graph=self.graph(),
            query=self.query(worlds, event, given),
            catalog=self.catalog(specs),
            **options,
        )


def _spec(population, do, variables):
    return (population, dict(do), list(variables))


@cache
def mediation_case(uz_target=F(3, 4)):
    def model(uz):
        return Scm(
            {
                "Uz": bern(uz),
                "Ux": bern(F(1, 4)),
                "Um": [F(1, 4), F(1, 4), F(1, 2)],
                "Uy": bern(F(1, 8)),
            },
            [
                ("Z", 2, lambda v, u: u["Uz"]),
                ("X", 2, lambda v, u: v["Z"] ^ u["Ux"]),
                ("M", 3, lambda v, u: (v["X"] + v["Z"] + u["Um"]) % 3),
                ("Y", 2, lambda v, u: int(v["M"] + v["X"] * v["Z"] >= 2) ^ u["Uy"]),
            ],
        )

    return Case(
        "mediation_covariate_shift",
        (("Z", "X"), ("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")),
        (),
        {"source": model(F(1, 4)), "target": model(uz_target)},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=[], selections=["Z"]),),
        (
            _spec("source", {}, ["Z", "X", "M", "Y"]),
            _spec("target", {}, ["Z"]),
        ),
    )


def _confounded(uz):
    return Scm(
        {"Uz": bern(uz), "U": bern(F(1, 2)), "Ux": bern(F(1, 4)), "Uy": bern(F(1, 8))},
        [
            ("Z", 2, lambda v, u: u["Uz"]),
            ("X", 2, lambda v, u: u["U"] ^ (v["Z"] & u["Ux"])),
            ("Y", 2, lambda v, u: v["X"] ^ (v["Z"] & u["U"]) ^ u["Uy"]),
        ],
    )


_CONFOUNDED_EDGES = (("Z", "X"), ("Z", "Y"), ("X", "Y"))


def two_source_case():
    return Case(
        "two_source_regimes",
        _CONFOUNDED_EDGES,
        (("X", "Y"),),
        {"source": _confounded(F(1, 4)), "target": _confounded(F(3, 4))},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=["X"], selections=["Z"]),),
        (
            _spec("source", {}, ["Z", "X", "Y"]),
            _spec("source", {"X": 0}, ["Z", "X", "Y"]),
            _spec("source", {"X": 1}, ["Z", "X", "Y"]),
            _spec("target", {}, ["Z"]),
        ),
    )


def missing_experiment_pair():
    """The two-source case with the source experiment withheld, and the second
    model family that agrees with it on every supplied table."""
    source, target = _confounded(F(1, 4)), _confounded(F(3, 4))
    case = Case(
        "missing_experiment",
        _CONFOUNDED_EDGES,
        (("X", "Y"),),
        {"source": source, "target": target},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=["X"], selections=["Z"]),),
        (
            _spec("source", {}, ["Z", "X", "Y"]),
            _spec("target", {}, ["Z"]),
        ),
    )
    alternative = {
        "source": source.blind_to_natural("Y", "X"),
        "target": target.blind_to_natural("Y", "X"),
    }
    return case, alternative


def front_door_case():
    def model(um):
        return Scm(
            {"U": bern(F(1, 2)), "Ux": bern(F(1, 8)), "Um": bern(um), "Uy": bern(F(1, 8))},
            [
                ("X", 2, lambda v, u: u["U"] ^ u["Ux"]),
                ("M", 2, lambda v, u: v["X"] ^ u["Um"]),
                ("Y", 2, lambda v, u: v["M"] ^ u["U"] ^ u["Uy"]),
            ],
        )

    return Case(
        "front_door_latent_district",
        (("X", "M"), ("M", "Y")),
        (("X", "Y"),),
        {"source": model(F(1, 4)), "target": model(F(1, 2))},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=[], selections=["M"]),),
        (
            _spec("source", {}, ["X", "M", "Y"]),
            _spec("target", {}, ["X", "M"]),
        ),
    )


def outcome_selection_pair():
    def model(uy):
        return Scm(
            {"U": bern(F(1, 2)), "Ux": bern(F(1, 4)), "Uy": bern(uy)},
            [
                ("X", 2, lambda v, u: u["U"] ^ u["Ux"]),
                ("Y", 2, lambda v, u: v["X"] ^ (u["U"] & u["Uy"])),
            ],
        )

    source, target = model(F(1, 2)), model(F(1, 4))
    case = Case(
        "selection_on_outcome",
        (("X", "Y"),),
        (("X", "Y"),),
        {"source": source, "target": target},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=["X"], selections=["Y"]),),
        (
            _spec("source", {}, ["X", "Y"]),
            _spec("source", {"X": 0}, ["X", "Y"]),
            _spec("source", {"X": 1}, ["X", "Y"]),
            _spec("target", {}, ["X", "Y"]),
        ),
    )
    alternative = {"source": source, "target": target.blind_to_natural("Y", "X")}
    return case, alternative


def recanting_pair():
    def model(blind):
        def w(v, u):
            return u["U"] if blind else v["X"] ^ u["U"]

        return Scm(
            {"U": bern(F(1, 2)), "Ux": bern(F(1, 2)), "Uy": bern(F(1, 4))},
            [
                ("X", 2, lambda v, u: u["Ux"]),
                ("W", 2, w),
                ("Z", 2, lambda v, u: v["W"]),
                ("Y", 2, lambda v, u: v["W"] ^ v["Z"] ^ u["Uy"]),
            ],
        )

    names = ["X", "W", "Z", "Y"]
    specs = []
    for size in range(len(names) + 1):
        for subset in itertools.combinations(names, size):
            for levels in itertools.product((0, 1), repeat=size):
                specs.append(_spec("pop", dict(zip(subset, levels, strict=True)), names))
    case = Case(
        "recanting_witness",
        (("X", "W"), ("W", "Z"), ("W", "Y"), ("Z", "Y")),
        (),
        {"pop": model(False)},
        "pop",
        (),
        tuple(specs),
    )
    return case, {"pop": model(True)}


def zero_support_case():
    def model(px):
        return Scm(
            {"Ux": bern(px), "Uy": bern(F(1, 4))},
            [
                ("X", 2, lambda v, u: u["Ux"]),
                ("Y", 2, lambda v, u: v["X"] ^ u["Uy"]),
            ],
        )

    return Case(
        "zero_support",
        (("X", "Y"),),
        (),
        {"source": model(F(1, 2)), "target": model(F(0))},
        "target",
        (transport.CounterfactualTransportSource("source", controllable=["X"], selections=["X"]),),
        (
            _spec("source", {}, ["X", "Y"]),
            _spec("source", {"X": 0}, ["X", "Y"]),
            _spec("target", {}, ["X"]),
        ),
    )


# Worlds and events -----------------------------------------------------------

World, Event = transport.CounterfactualWorld, transport.CounterfactualEvent


def nested(x, xp):
    """``Y_{x, M_{x'}}`` as worlds (the outer world reads M from the inner one)."""
    worlds = (
        World("inner", {"X": float(xp)}),
        World("outer", {"X": float(x)}, {("M", "Y"): "inner"}),
    )
    return worlds, (Event("Y", 1, "outer"),)


def nested_truth(x, xp):
    return [(("Y", {"X": x, "M": ("M", {"X": xp})}), 1)]


TREATED = (World("natural"), World("treated", {"X": 1.0}))


# ---------------------------------------------------------------------------
# The enumerator agrees with closed forms and with the hand-derived values.
# ---------------------------------------------------------------------------


MEDIATION_TRUTH = {(0, 0): F(23, 64), (0, 1): F(5, 16), (1, 0): F(1, 2), (1, 1): F(19, 32)}


def test_the_enumerator_matches_the_mediation_formula_and_the_hand_derived_values():
    case = mediation_case()
    target = case.models["target"]
    joint = target.law({}, ["Z", "X", "M", "Y"])

    def p(z=None, x=None, m=None, y=None):
        total = F(0)
        for index, mass in enumerate(joint):
            zz, rest = divmod(index, 12)
            xx, rest = divmod(rest, 6)
            mm, yy = divmod(rest, 2)
            if all(a is None or a == b for a, b in ((z, zz), (x, xx), (m, mm), (y, yy))):
                total += mass
        return total

    for (x, xp), expected in MEDIATION_TRUTH.items():
        formula = sum(
            p(z, x, m, 1) / p(z, x, m) * p(z, xp, m) / p(z, xp) * p(z)
            for z in (0, 1)
            for m in (0, 1, 2)
        )
        enumerated = target.probability(nested_truth(x, xp))
        assert formula == enumerated == expected, (x, xp)
    assert MEDIATION_TRUTH[(1, 0)] - MEDIATION_TRUTH[(0, 0)] == F(9, 64)
    assert MEDIATION_TRUTH[(0, 1)] - MEDIATION_TRUTH[(0, 0)] == F(-3, 64)


def test_the_enumerator_matches_the_front_door_and_two_source_closed_forms():
    front = front_door_case().models["target"]
    joint = front.law({}, ["X", "M", "Y"])

    def p(x=None, m=None, y=None):
        return sum(
            mass
            for index, mass in enumerate(joint)
            if all(
                a is None or a == b
                for a, b in zip((x, m, y), (index // 4, (index // 2) % 2, index % 2), strict=True)
            )
        )

    x = 1
    formula = sum(
        p(x, m) / p(x) * sum(p(xp, m, 1) / p(xp, m) * p(xp) for xp in (0, 1)) for m in (0, 1)
    )
    assert formula == front.probability([(("Y", {"X": 1}), 1)]) == F(1, 2)

    target = two_source_case().models["target"]
    do_one = target.probability([(("Y", {"X": 1}), 1)])
    both = target.probability([("X", 1), ("Y", 1)])
    # P(Y_1 = 1, X = 0) = P(Y_1 = 1) - P(Y = 1, X = 1) by consistency.
    assert do_one == F(19, 32)
    assert do_one - both == target.probability([(("Y", {"X": 1}), 1), ("X", 0)]) == F(47, 128)
    assert target.probability([(("Y", {"X": 1}), 1)], [("X", 0)]) == F(47, 64)


# ---------------------------------------------------------------------------
# Identified queries equal the enumerated truth.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    label: str
    case: Case
    worlds: tuple
    event: tuple
    given: tuple
    truth_event: list
    truth_given: list
    expected: F


def _targets():
    out = []
    case = mediation_case()
    for (x, xp), expected in MEDIATION_TRUTH.items():
        worlds, event = nested(x, xp)
        out.append(
            Target(f"Y_x{x}_Mx{xp}", case, worlds, event, (), nested_truth(x, xp), [], expected)
        )
    two = two_source_case()
    out.append(
        Target(
            "joint_Y1_and_X0",
            two,
            TREATED,
            (Event("Y", 1, "treated"), Event("X", 0, "natural")),
            (),
            [(("Y", {"X": 1}), 1), ("X", 0)],
            [],
            F(47, 128),
        )
    )
    out.append(
        Target(
            "effect_on_treated",
            two,
            TREATED,
            (Event("Y", 1, "treated"),),
            (Event("X", 0, "natural"),),
            [(("Y", {"X": 1}), 1)],
            [("X", 0)],
            F(47, 64),
        )
    )
    out.append(
        Target(
            "front_door",
            front_door_case(),
            (World("treated", {"X": 1.0}),),
            (Event("Y", 1, "treated"),),
            (),
            [(("Y", {"X": 1}), 1)],
            [],
            F(1, 2),
        )
    )
    return out


TARGETS = _targets()


def _identified(target):
    decision = target.case.decide(target.worlds, target.event, target.given)
    assert decision.status == "identified", decision.decision
    assert decision.identification_status == "identified"
    assert decision.reason_code is None and decision.detail is None
    return decision


@pytest.mark.parametrize("target", TARGETS, ids=[t.label for t in TARGETS])
def test_identified_targets_equal_the_enumerated_truth(target):
    decision = _identified(target)
    truth = target.case.models[target.case.target].probability(
        target.truth_event, target.truth_given
    )
    assert truth == target.expected
    derivation = decision.derivation
    result = derivation.evaluate(evidence=target.case.laws())
    assert abs(result.probability - float(truth)) <= EXACT_TOLERANCE
    assert result.conditioning_mass_checked
    assert result.cells_read > 0
    assert result.cited_laws, "the evaluation reports every law it read"
    assert result.uncertainty is None and result.scope.endswith("point_only")
    assert not derivation.bound, "evaluate does not bind"
    # The derivation names the regimes it reads, each a regime of the catalog.
    cited = {(c["population"], c["regime"]) for c in derivation.cited_regimes}
    declared = {(r.population, r.id) for r in target.case.catalog().regimes}
    assert cited and cited <= declared


def test_conditional_effect_is_the_ratio_of_two_derived_probabilities():
    target = next(t for t in TARGETS if t.label == "effect_on_treated")
    result = _identified(target).derivation.evaluate(evidence=target.case.laws())
    assert 0.0 < result.denominator < 1.0
    assert abs(result.numerator / result.denominator - 47 / 64) <= EXACT_TOLERANCE
    assert result.probability == result.numerator / result.denominator
    # The conditioning event has mass 1/2 in the target (enumerated).
    assert target.case.models["target"].probability([("X", 0)]) == F(1, 2)
    if result.conditioning_mass is not None:
        assert abs(result.conditioning_mass - 0.5) <= EXACT_TOLERANCE


def test_natural_direct_and_indirect_effects_are_the_enumerated_contrasts():
    case = mediation_case()
    value = {}
    for (x, xp), expected in MEDIATION_TRUTH.items():
        worlds, event = nested(x, xp)
        decision = case.decide(worlds, event)
        value[(x, xp)] = decision.derivation.evaluate(evidence=case.laws()).probability
        assert abs(value[(x, xp)] - float(expected)) <= EXACT_TOLERANCE
    nde = value[(1, 0)] - value[(0, 0)]
    nie = value[(0, 1)] - value[(0, 0)]
    assert abs(nde - 9 / 64) <= EXACT_TOLERANCE
    assert abs(nie - (-3 / 64)) <= EXACT_TOLERANCE


def test_two_source_regimes_are_cited_jointly_with_the_target_marginal():
    target = next(t for t in TARGETS if t.label == "joint_Y1_and_X0")
    result = _identified(target).derivation.evaluate(evidence=target.case.laws())
    cited = {(law["population"], tuple(sorted(law["interventions"]))) for law in result.cited_laws}
    assert ("source", ()) in cited
    assert ("source", ("X",)) in cited
    assert ("target", ()) in cited
    # The target supplies only its observational law (the marginal of Z).
    assert {law["regime"] for law in result.cited_laws if law["population"] == "target"} == {
        "target|obs"
    }


def test_the_target_need_not_measure_the_outcome():
    case = front_door_case()
    target = next(t for t in TARGETS if t.label == "front_door")
    laws = [law for law in case.laws() if law.population == "target"]
    assert laws and all("Y" not in dict(law.axes) for law in laws)
    result = _identified(target).derivation.evaluate(evidence=case.laws())
    assert abs(result.probability - 0.5) <= EXACT_TOLERANCE


# ---------------------------------------------------------------------------
# Lifecycle: decide once, evaluate, rebind, order invariance.
# ---------------------------------------------------------------------------


def test_bind_snapshot_rebuilds_the_numbers_and_leaves_the_decided_value_untouched():
    case = mediation_case()
    worlds, event = nested(0, 0)  # depends on the covariate law
    decision = case.decide(worlds, event)
    derivation = decision.derivation
    before = float(MEDIATION_TRUTH[(0, 0)])
    with pytest.raises(CausalUnsupportedError) as raised:
        require_success(derivation.estimate())
    assert raised.value.reason_code == "transport_missing_provider"

    first = require_success(derivation.bind_snapshot(case.laws()))
    assert first.bound and not derivation.bound
    assert abs(require_success(first.estimate()).probability - before) <= EXACT_TOLERANCE

    # Same shape, changed numbers: the covariate law of the target moves.
    shifted_case = mediation_case(F(1, 2))
    shifted_truth = shifted_case.models["target"].probability(nested_truth(0, 0))
    assert abs(float(shifted_truth) - before) > 1e-3
    second = require_success(first.bind_snapshot(shifted_case.laws()))
    assert (
        abs(require_success(second.estimate()).probability - float(shifted_truth))
        <= EXACT_TOLERANCE
    )
    # Nothing was searched again: the frozen derivation is the one decided once.
    assert second.derivation["canonical_text"] == derivation.derivation["canonical_text"]
    assert second.derivation["search_receipt"] == derivation.derivation["search_receipt"]
    # The earlier prepared value keeps its numbers.
    assert abs(require_success(first.estimate()).probability - before) <= EXACT_TOLERANCE

    # A refused rebind (a counted law) leaves the bound state alone.
    counted = [_counted(law) for law in case.laws()]
    with pytest.raises(CausalUnsupportedError) as refused:
        require_success(second.bind_snapshot(counted))
    assert refused.value.reason_code == "cell_not_licensed"
    assert (
        abs(require_success(second.estimate()).probability - float(shifted_truth))
        <= EXACT_TOLERANCE
    )


def test_evaluation_is_deterministic_and_invariant_to_law_and_regime_order():
    target = next(t for t in TARGETS if t.label == "effect_on_treated")
    case = target.case
    derivation = _identified(target).derivation
    laws = case.laws()
    first = derivation.evaluate(evidence=laws)
    again = derivation.evaluate(evidence=laws)
    reversed_ = derivation.evaluate(evidence=list(reversed(laws)))
    assert first.probability == again.probability == reversed_.probability
    assert bytes(first.export()) == bytes(reversed_.export()), "exports are canonical in law order"
    # Declaration order of the catalog's regimes does not change the decision.
    flipped = transport.EvidenceCatalog(regimes=tuple(reversed(case.catalog().regimes)))
    other = transport.identify_counterfactual(
        graph=case.graph(),
        query=case.query(target.worlds, target.event, target.given),
        catalog=flipped,
    )
    assert other.derivation.derivation["canonical_text"] == derivation.derivation["canonical_text"]


def test_numpy_tables_are_accepted_as_exact_laws():
    target = next(t for t in TARGETS if t.label == "front_door")
    derivation = _identified(target).derivation
    laws = [
        transport.ExactDiscreteLaw(
            law.population,
            law.regime,
            tuple((name, np.asarray(levels)) for name, levels in law.axes),
            np.asarray(law.probabilities, dtype=np.float64),
            law.snapshot_identity,
            interventions=law.interventions,
        )
        for law in target.case.laws()
    ]
    assert (
        abs(
            derivation.evaluate(evidence=transport.ExactTransportData(tuple(laws))).probability
            - 0.5
        )
        <= EXACT_TOLERANCE
    )


# ---------------------------------------------------------------------------
# Obstructions are proven by a replayed two-model witness.
# ---------------------------------------------------------------------------


def _counted(law, scale=4096):
    """``law`` with exact integer counts: its probabilities are dyadic with
    denominators dividing ``scale``, so the counts reproduce them exactly."""
    counts = tuple(int(p * scale) for p in law.probabilities)
    assert sum(counts) == scale and all(
        c / scale == p for c, p in zip(counts, law.probabilities, strict=True)
    )
    return transport.ExactDiscreteLaw(
        law.population,
        law.regime,
        law.axes,
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
        empirical_counts=counts,
    )


def _evidence_tables(case, models):
    return [law.probabilities for law in case.laws(models=models)]


def test_a_recanting_witness_is_a_proven_obstruction_and_replays_from_bytes():
    case, alternative = recanting_pair()
    # Independently: the two models give identical laws on every declared regime
    # and different values of the query, so no evidence determines it.
    assert _evidence_tables(case, case.models) == _evidence_tables(case, alternative)
    query = [(("Y", {"X": 1, "Z": ("Z", {"X": 0})}), 1)]
    assert case.models["pop"].probability(query) == F(3, 4)
    assert alternative["pop"].probability(query) == F(1, 4)

    worlds = (
        World("inner", {"X": 0.0}),
        World("outer", {"X": 1.0}, {("Z", "Y"): "inner"}),
    )
    decision = case.decide(worlds, (Event("Y", 1, "outer"),))
    assert decision.status == "proven_non_transportable"
    assert decision.reason_code == "transport_proven_non_transportable"
    assert decision.detail == "counterfactual_transport.checked_obstruction"
    first, second = decision.witness_values
    assert first != second
    for value in (first, second):
        numerator, denominator = (int(part) for part in value.split("/"))
        assert 0 <= numerator <= denominator
    assert decision.decision["check"]["cells_compared"] > 0
    assert decision.derivation is None
    with pytest.raises(CausalUnsupportedError) as raised:
        decision.prepare()
    assert raised.value.reason_code == "transport_proven_non_transportable"
    assert "counterfactual_transport.checked_obstruction" in str(raised.value)

    artifact = bytes(decision.export())
    consumed = transport.consume_counterfactual_transport_artifact(
        artifact,
        expected_premises_digest=decision.premises_digest,
        expected_data_digest=decision.data_digest,
    )
    assert consumed.status == "proven_non_transportable"
    assert consumed.independently_verified
    assert consumed.witness_values == decision.witness_values


def test_selection_on_the_outcome_mechanism_is_not_transportable():
    case, alternative = outcome_selection_pair()
    assert _evidence_tables(case, case.models) == _evidence_tables(case, alternative)
    query = [(("Y", {"X": 1}), 1)]
    assert case.models["target"].probability(query) == F(7, 8)
    assert alternative["target"].probability(query) == F(7, 16)
    decision = case.decide((World("treated", {"X": 1.0}),), (Event("Y", 1, "treated"),))
    assert decision.status == "proven_non_transportable"
    assert decision.witness_values[0] != decision.witness_values[1]
    # The same graph is transportable once Y's mechanism is not selected.
    unselected = transport.CounterfactualTransportQuery(
        target="target",
        levels=case.levels(),
        worlds=(World("treated", {"X": 1.0}),),
        event=(Event("Y", 1, "treated"),),
        sources=(
            transport.CounterfactualTransportSource("source", controllable=["X"], selections=[]),
        ),
    )
    identified = transport.identify_counterfactual(
        graph=case.graph(), query=unselected, catalog=case.catalog()
    )
    assert identified.status == "identified", identified.decision


def test_missing_evidence_names_the_regime_that_would_license_the_query():
    case, alternative = missing_experiment_pair()
    # Independently: with the experiment withheld two models agree on every
    # supplied table and differ on the target (19/32 against 23/64).
    assert _evidence_tables(case, case.models) == _evidence_tables(case, alternative)
    query = [(("Y", {"X": 1}), 1)]
    assert case.models["target"].probability(query) == F(19, 32)
    assert alternative["target"].probability(query) == F(23, 64)

    worlds, event = (World("treated", {"X": 1.0}),), (Event("Y", 1, "treated"),)
    decision = case.decide(worlds, event)
    assert decision.status == "missing_evidence"
    assert decision.reason_code == "transport_missing_evidence"
    assert decision.detail == "counterfactual_transport.missing_regime"
    assert any(
        m["population"] == "source" and tuple(m["interventions"]) == ("X",)
        for m in decision.missing_regimes
    ), decision.missing_regimes
    with pytest.raises(CausalUnsupportedError) as raised:
        decision.prepare()
    assert raised.value.reason_code == "transport_missing_evidence"
    message = str(raised.value)
    assert "counterfactual_transport.missing_regime" in message
    assert "experiment on" in message and "source" in message
    assert decision.derivation is None

    # Supplying the named regime (catalog and law) identifies it to the truth.
    full = two_source_case()
    supplied = full.decide(worlds, event)
    assert supplied.status == "identified"
    result = supplied.derivation.evaluate(evidence=full.laws())
    assert abs(result.probability - 19 / 32) <= EXACT_TOLERANCE


def test_zero_conditioning_mass_is_a_typed_refusal_not_a_number():
    case = zero_support_case()
    worlds = (World("natural"), World("treated", {"X": 0.0}))
    event, given = (Event("Y", 1, "treated"),), (Event("X", 1, "natural"),)
    # Enumerated: defined (1/4) in the source, undefined in the target.
    truth_event, truth_given = [(("Y", {"X": 0}), 1)], [("X", 1)]
    assert case.models["source"].probability(truth_event, truth_given) == F(1, 4)
    assert case.models["target"].probability(truth_event, truth_given) is None

    decision = case.decide(worlds, event, given)
    assert decision.status == "identified", decision.decision
    with pytest.raises(CausalUnsupportedError) as raised:
        decision.derivation.evaluate(evidence=case.laws())
    assert raised.value.reason_code == "transport_support_failure"
    assert "counterfactual_transport.zero_conditioning_mass" in str(raised.value)
    assert_registered_refusal(raised.value)

    # The same query is defined, and equals 1/4, in the source population.
    in_source = transport.identify_counterfactual(
        graph=case.graph(),
        query=case.query(
            worlds,
            event,
            given,
            target="source",
            sources=(
                transport.CounterfactualTransportSource(
                    "target", controllable=[], selections=["X"]
                ),
            ),
        ),
        catalog=case.catalog(),
    )
    assert in_source.status == "identified", in_source.decision
    value = in_source.derivation.evaluate(evidence=case.laws()).probability
    assert abs(value - 0.25) <= EXACT_TOLERANCE


# ---------------------------------------------------------------------------
# Refusals name what would license the request.
# ---------------------------------------------------------------------------


def test_budget_cancellation_and_bounds_are_coded_stops_never_verdicts():
    """A stop of the one SearchBudget is a SearchReceipt (a SearchStop), never a verdict."""
    case = mediation_case()
    worlds, event = nested(1, 0)
    starved = case.decide(worlds, event, max_operations=1, max_depth=8)
    assert starved.status == "budget_cancel"
    assert starved.reason_code == "transport_budget_cancel"
    assert starved.detail == "counterfactual_transport.budget"
    assert starved.derivation is None
    receipt = starved.decision["limits_receipt"]
    assert receipt["stop"] in ("search.operations", "search.depth")
    assert receipt["operations_limit"] == 1 and receipt["depth_limit"] == 8
    with pytest.raises(Exception, match="counterfactual_transport.budget") as raised:
        starved.prepare()
    assert raised.value.reason_code == "transport_budget_cancel"

    token = CancellationToken()
    token.cancel()
    cancelled = case.decide(worlds, event, cancel=token)
    assert cancelled.status == "budget_cancel"
    assert cancelled.reason_code == "transport_budget_cancel"

    # Search limits above the contract's maxima are refused by bound.
    with pytest.raises(CausalUnsupportedError) as raised:
        case.decide(worlds, event, max_operations=10**12)
    assert raised.value.reason_code == "route_not_supported"
    assert "counterfactual_transport.bounds_exceeded" in str(raised.value)


def test_malformed_requests_refuse_as_invalid_arguments():
    case = mediation_case()
    worlds, event = nested(1, 0)
    # Undeclared levels for a graph variable.
    levels = dict(case.levels())
    del levels["M"]
    query = transport.CounterfactualTransportQuery(
        target="target",
        levels=levels,
        worlds=worlds,
        event=event,
        sources=case.sources,
    )
    with pytest.raises(Exception, match="counterfactual_transport.invalid_query") as raised:
        transport.identify_counterfactual(graph=case.graph(), query=query, catalog=case.catalog())
    assert raised.value.reason_code == "invalid_argument"
    # An event on a variable that is not in the graph.
    with pytest.raises(CausalValueError):
        case.decide(worlds, (Event("Nope", 1, "outer"),))
    # Query value objects validate their own shape.
    with pytest.raises(CausalValueError):
        transport.CounterfactualTransportQuery(
            target="target",
            levels=case.levels(),
            worlds=worlds,
            event=(Event("Y", 1, "elsewhere"),),
        )
    with pytest.raises(CausalValueError):
        World("w", {"X": float("nan")})
    with pytest.raises(CausalValueError):
        World("w", reads={("M", "Y"): "w"})


def test_absent_mislabelled_counted_and_malformed_evidence_is_refused_before_a_number():
    target = next(t for t in TARGETS if t.label == "joint_Y1_and_X0")
    case = target.case
    derivation = _identified(target).derivation
    laws = case.laws()
    derivation.evaluate(evidence=laws)

    # The source experiment at one level is withheld: its factor has no law.
    thin = [
        law
        for law in laws
        if not (law.population == "source" and dict(law.interventions).get("X") == 1.0)
    ]
    assert len(thin) < len(laws)
    with pytest.raises(CausalUnsupportedError) as raised:
        derivation.evaluate(evidence=thin)
    assert raised.value.reason_code == "transport_missing_provider"
    assert "counterfactual_transport.missing_law" in str(raised.value)

    # A source law relabelled as the target's: the catalog binds no such regime.
    first = next(i for i, law in enumerate(laws) if law.population == "source")
    swapped = list(laws)
    law = swapped[first]
    swapped[first] = transport.ExactDiscreteLaw(
        "target",
        law.regime,
        law.axes,
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
    )
    with pytest.raises(CausalValueError):
        derivation.evaluate(evidence=swapped)

    # A counted (empirical) law is not licensed: sampled inference is another route.
    counted = list(laws)
    counted[0] = _counted(counted[0])
    with pytest.raises(CausalUnsupportedError) as raised:
        derivation.evaluate(evidence=counted)
    assert raised.value.reason_code == "cell_not_licensed"
    assert "counterfactual_transport.sampled_inference_closed" in str(raised.value)

    # Levels that are not the declared ones.
    wrong = list(laws)
    law = wrong[-1]
    wrong[-1] = transport.ExactDiscreteLaw(
        law.population,
        law.regime,
        tuple((n, (0.0, 2.0)) for n, _ in law.axes),
        law.probabilities,
        law.snapshot_identity,
        interventions=law.interventions,
    )
    with pytest.raises(CausalError) as raised:
        derivation.evaluate(evidence=wrong)
    assert isinstance(raised.value, CausalError)


# ---------------------------------------------------------------------------
# Artifacts: round trip, mutation, identity, limits, fresh process.
# ---------------------------------------------------------------------------


def _exported(label="Y_x0_Mx1"):
    target = next(t for t in TARGETS if t.label == label)
    result = _identified(target).derivation.evaluate(evidence=target.case.laws())
    return target, result, bytes(result.export())


def test_exported_result_is_recomputed_by_an_independent_consumer():
    for target in TARGETS:
        result = _identified(target).derivation.evaluate(evidence=target.case.laws())
        artifact = bytes(result.export())
        consumed = transport.consume_counterfactual_transport_artifact(
            artifact,
            expected_premises_digest=result.premises_digest,
            expected_data_digest=result.data_digest,
        )
        assert consumed.independently_verified
        assert consumed.probability == result.probability  # bit for bit
        assert abs(consumed.probability - float(target.expected)) <= EXACT_TOLERANCE
        assert consumed.cited_laws == result.cited_laws
        assert consumed.premises_digest == result.premises_digest
        assert consumed.data_digest == result.data_digest


def test_a_mutated_artifact_is_refused_or_returns_the_original_answer():
    target, result, artifact = _exported()
    identity = dict(
        expected_premises_digest=result.premises_digest, expected_data_digest=result.data_digest
    )
    # A text edit of the snapshot identity: the data digest no longer matches.
    edited = artifact.replace(b"snap|target|", b"snap|tarGet|")
    assert edited != artifact and len(edited) == len(artifact)
    with pytest.raises(CausalError) as raised:
        transport.consume_counterfactual_transport_artifact(edited)
    assert isinstance(raised.value, CausalError)
    assert raised.value.reason_code == "invalid_argument"
    assert "counterfactual_transport.invalid_artifact" in str(raised.value)

    # Every sampled single-bit edit either refuses or cannot change the answer.
    stride = max(1, len(artifact) // 80)
    for position in range(0, len(artifact), stride):
        mutated = bytearray(artifact)
        mutated[position] ^= 0x01
        try:
            consumed = transport.consume_counterfactual_transport_artifact(
                bytes(mutated), **identity
            )
        except CausalError:
            continue
        assert consumed.probability == result.probability, position

    # Truncation and a foreign payload are refused with typed errors.
    for broken in (artifact[: len(artifact) // 2], b"", b"not an artifact"):
        with pytest.raises(CausalError):
            transport.consume_counterfactual_transport_artifact(broken)


def test_expected_identity_and_consumer_limits_are_enforced_before_a_number():
    target, result, artifact = _exported()
    other = _exported("Y_x1_Mx1")[1]
    # The same query on a changed target covariate law: identical premises,
    # genuinely different law tables, hence a different data digest.
    shifted_case = mediation_case(F(1, 2))
    worlds, event = nested(0, 1)
    shifted = shifted_case.decide(worlds, event).derivation.evaluate(evidence=shifted_case.laws())
    assert shifted.data_digest != result.data_digest
    for expected in (
        dict(expected_premises_digest=other.premises_digest),
        dict(expected_data_digest=shifted.data_digest),
        dict(expected_premises_digest="not-the-identity"),
    ):
        with pytest.raises(CausalError) as raised:
            transport.consume_counterfactual_transport_artifact(artifact, **expected)
        assert raised.value.reason_code == "invalid_argument"
        assert "counterfactual_transport.invalid_artifact" in str(raised.value)
    # Either digest alone binds the expectation.
    transport.consume_counterfactual_transport_artifact(
        artifact, expected_data_digest=result.data_digest
    )
    # Consumer limits are checked before any work; nothing the artifact stores
    # raises them.
    with pytest.raises(CausalUnsupportedError) as raised:
        transport.consume_counterfactual_transport_artifact(artifact, max_search_operations=1)
    assert raised.value.reason_code == "route_not_supported"
    assert "counterfactual_transport.bounds_exceeded" in str(raised.value)
    with pytest.raises(CausalUnsupportedError) as raised:
        transport.consume_counterfactual_transport_artifact(artifact, max_laws=1)
    assert raised.value.reason_code == "route_not_supported"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalError) as raised:
        transport.consume_counterfactual_transport_artifact(artifact, cancel=token)
    assert isinstance(raised.value, CausalError)


_FRESH_PROCESS = textwrap.dedent(
    """
    import json, sys
    from antecedent.transport import advanced as t

    path, premises, data = sys.argv[1:4]
    artifact = open(path, "rb").read()
    result = t.consume_counterfactual_transport_artifact(
        artifact, expected_premises_digest=premises, expected_data_digest=data
    )
    print(json.dumps({"probability": result.probability, "verified": result.independently_verified}))
    """
)


def test_a_fresh_process_consumes_only_the_bytes_and_matches_the_enumerated_truth(tmp_path):
    target, result, artifact = _exported()
    path = tmp_path / "artifact.bin"
    path.write_bytes(artifact)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_PROCESS,
            str(path),
            result.premises_digest,
            result.data_digest,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout)
    assert report["verified"] is True
    assert abs(report["probability"] - float(target.expected)) <= EXACT_TOLERANCE
    # Another expected identity is refused in the fresh process too.
    refused = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, str(path), "not-the-identity", result.data_digest],
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode != 0
    assert "invalid_artifact" in refused.stderr


# ---------------------------------------------------------------------------
# Routed stages execute from the retained plan with no builder alive.
# ---------------------------------------------------------------------------


def _label(label):
    return next(t for t in TARGETS if t.label == label)


def test_identify_counterfactual_keeps_its_plan_after_the_builder_is_disposed():
    target = _label("Y_x0_Mx1")
    builder = target.case.decide(target.worlds, target.event, target.given)
    program = builder.decision
    derivation = builder.derivation
    del builder
    assert program["outcome"] == "identified" and program["identification_status"] == "identified"
    plan = derivation.derivation
    assert plan["canonical_text"] == program["derivation"]["canonical_text"]
    assert plan["rules"] and plan["cited"]
    result = require_success(
        require_success(derivation.bind_snapshot(target.case.laws())).estimate()
    )
    assert abs(result.probability - float(target.expected)) <= EXACT_TOLERANCE


def test_evaluate_runs_the_retained_plan_without_searching_again_after_builder_disposal():
    target = _label("Y_x1_Mx1")
    laws = target.case.laws()
    builder = target.case.decide(target.worlds, target.event, target.given)
    derivation = builder.derivation
    receipt = derivation.derivation["search_receipt"]
    del builder
    result = derivation.evaluate(evidence=laws)
    state = require_success(require_success(derivation.bind_snapshot(laws)).estimate())
    assert result.probability == state.probability
    assert abs(result.probability - float(target.expected)) <= EXACT_TOLERANCE
    plan = derivation.derivation
    assert plan["search_receipt"] == receipt, "nothing was searched again"


def test_estimate_returns_the_bound_state_of_the_retained_plan_after_builder_disposal():
    target = _label("effect_on_treated")
    builder = target.case.decide(target.worlds, target.event, target.given)
    derivation = builder.derivation
    bound = require_success(derivation.bind_snapshot(target.case.laws()))
    del builder
    state = require_success(bound.estimate())
    assert bound.bound and not derivation.bound
    assert abs(state.probability - float(target.expected)) <= EXACT_TOLERANCE
    plan = bound.derivation
    assert plan["canonical_text"] == derivation.derivation["canonical_text"]
    with pytest.raises(CausalUnsupportedError) as raised:
        require_success(derivation.estimate())
    assert raised.value.reason_code == "transport_missing_provider"


def test_bind_snapshot_rebinds_laws_to_the_retained_plan_after_builder_disposal():
    case = mediation_case()
    worlds, event = nested(0, 0)
    builder = case.decide(worlds, event)
    derivation = builder.derivation
    first = require_success(derivation.bind_snapshot(case.laws()))
    del builder
    shifted = mediation_case(F(1, 2))
    second = require_success(first.bind_snapshot(shifted.laws()))
    plan = second.derivation
    assert plan["canonical_text"] == derivation.derivation["canonical_text"]
    truth = float(shifted.models["target"].probability(nested_truth(0, 0)))
    assert abs(require_success(second.estimate()).probability - truth) <= EXACT_TOLERANCE
    assert (
        abs(require_success(first.estimate()).probability - float(MEDIATION_TRUTH[(0, 0)]))
        <= EXACT_TOLERANCE
    )


def test_result_export_binds_the_plan_and_the_laws_into_the_verified_identity():
    target = _label("joint_Y1_and_X0")
    builder = target.case.decide(target.worlds, target.event, target.given)
    derivation = builder.derivation
    del builder
    laws = target.case.laws()
    result = derivation.evaluate(evidence=laws)
    artifact = bytes(result.export())
    plan = result.derivation
    assert plan["canonical_text"] == derivation.derivation["canonical_text"]
    # The identity moves with the laws and not with the plan.
    shifted = mediation_case()  # a different model class: another premises digest
    other = _label("Y_x0_Mx0")
    other_result = require_success(
        require_success(
            shifted.decide(*nested(0, 0)).derivation.bind_snapshot(shifted.laws())
        ).estimate()
    )
    assert other_result.premises_digest != result.premises_digest
    again = require_success(require_success(derivation.bind_snapshot(laws)).estimate())
    assert bytes(again.export()) == artifact
    assert other.expected == MEDIATION_TRUTH[(0, 0)]


def test_consumer_recomputes_the_point_and_the_plan_after_builder_disposal():
    target = _label("Y_x0_Mx1")
    builder = target.case.decide(target.worlds, target.event, target.given)
    derivation = builder.derivation
    result = derivation.evaluate(evidence=target.case.laws())
    artifact = bytes(result.export())
    del builder
    consumed = transport.consume_counterfactual_transport_artifact(
        artifact,
        expected_premises_digest=result.premises_digest,
        expected_data_digest=result.data_digest,
    )
    plan = consumed.derivation
    assert plan["canonical_text"] == derivation.derivation["canonical_text"]
    assert (
        consumed.probability
        == require_success(
            require_success(derivation.bind_snapshot(target.case.laws())).estimate()
        ).probability
    )
    assert consumed.independently_verified


# ---------------------------------------------------------------------------
# Sampled evidence.
# ---------------------------------------------------------------------------


def sampled_tables(laws, n, seed, *, structural=True):
    """Multinomial draws of units from the enumerated observed laws, fixed seed."""
    rng = np.random.default_rng(seed)
    tables = []
    for law in laws:
        p = np.asarray(law.probabilities, dtype=np.float64)
        counts = rng.multinomial(n, p / p.sum())
        zeros = [i for i, mass in enumerate(law.probabilities) if mass == 0.0]
        tables.append(
            transport.SampledCountTable(
                law.population,
                law.regime,
                law.axes,
                counts,
                law.snapshot_identity,
                interventions=law.interventions,
                structural_zeros=zeros if structural else (),
            )
        )
    return tables


def cited_only(derivation, tables):
    cited = {(c["population"], c["regime"]) for c in derivation.cited_regimes}
    return [t for t in tables if (t.population, t.regime) in cited]


def _estimate(derivation, tables, **options):
    options.setdefault("overlap", transport.SamplingOverlap.all_independent())
    return derivation.estimate_sampled(tables, **options)


SAMPLED_CASES = ["Y_x1_Mx0", "Y_x0_Mx1", "joint_Y1_and_X0", "effect_on_treated", "front_door"]


@pytest.mark.parametrize("label", SAMPLED_CASES)
def test_large_samples_converge_to_the_enumerated_truth_with_four_distinct_labels(label):
    target = next(t for t in TARGETS if t.label == label)
    derivation = _identified(target).derivation
    laws = target.case.laws()
    tables = cited_only(
        derivation, sampled_tables(laws, LARGE_N, 1_000 + SAMPLED_CASES.index(label))
    )
    result = _estimate(derivation, tables)
    truth = float(target.expected)
    assert result.probability is not None
    assert abs(result.probability - truth) <= PLUG_IN_TOLERANCE, (result.probability, truth)
    # Four claims, four labels.
    assert result.labels["structural_identification"] == "identified_by_stored_derivation"
    assert result.labels["exact_finite_law_evaluation"] == (
        "exact_on_estimated_tables_not_population_laws"
    )
    assert result.labels["point_estimate"] == "plug_in_point_estimate"
    assert result.labels["statistical_inference"] == "withheld_cell_not_licensed"
    assert result.point["smoothed"] is False
    # The interval is never published; the standard error is the only uncertainty.
    assert result.interval["available"] is False
    assert result.interval["reason"] == "cell_not_licensed"
    assert result.interval["detail"] == "counterfactual_transport_sampled.interval_not_licensed"
    assert result.interval["calibration_reason"] == "estimator_grid_not_measured"
    se = result.standard_error
    assert se is not None
    assert se["label"] == "uncalibrated_standard_error"
    assert se["method"] == "delta_method_multinomial_central_difference"
    assert 0.0 < se["value"] < PLUG_IN_TOLERANCE
    assert abs(result.probability - truth) <= 6.0 * se["value"]
    assert result.envelope is None and result.point_withheld_reason is None
    assert result.cells_read > 0
    assert result.scope.endswith("point_only")


def test_sampled_natural_effects_converge_to_the_enumerated_contrasts():
    case = mediation_case()
    laws = case.laws()
    tables = sampled_tables(laws, LARGE_N, 77)
    value = {}
    for x, xp in MEDIATION_TRUTH:
        worlds, event = nested(x, xp)
        derivation = case.decide(worlds, event).derivation
        value[(x, xp)] = _estimate(derivation, cited_only(derivation, tables)).probability
    nde = value[(1, 0)] - value[(0, 0)]
    nie = value[(0, 1)] - value[(0, 0)]
    tolerance = 1.5 * PLUG_IN_TOLERANCE  # a contrast has up to twice the variance
    assert abs(nde - 9 / 64) <= tolerance
    assert abs(nie - (-3 / 64)) <= tolerance


def test_sampled_estimates_are_deterministic_and_invariant_to_table_order():
    target = next(t for t in TARGETS if t.label == "joint_Y1_and_X0")
    derivation = _identified(target).derivation
    tables = cited_only(derivation, sampled_tables(target.case.laws(), 5_000, 21))
    first = _estimate(derivation, tables)
    again = _estimate(derivation, tables)
    permuted = _estimate(derivation, list(reversed(tables)))
    assert first.probability == again.probability == permuted.probability
    assert first.standard_error["value"] == permuted.standard_error["value"]
    assert first.evidence_identity == permuted.evidence_identity
    assert first.derivation_identity == permuted.derivation_identity
    # A changed observation changes the identity and, for some table, the number.
    moved_number = False
    for index, table in enumerate(tables):
        counts = list(table.counts)
        source = next(i for i, c in enumerate(counts) if c > 10)
        counts[source] -= 10
        counts[(source + 1) % len(counts)] += 10
        edited = list(tables)
        edited[index] = transport.SampledCountTable(
            table.population,
            table.regime,
            table.axes,
            counts,
            table.snapshot_identity,
            interventions=table.interventions,
            structural_zeros=table.structural_zeros,
        )
        changed = _estimate(derivation, edited)
        assert changed.evidence_identity != first.evidence_identity
        moved_number |= changed.probability != first.probability
    assert moved_number, "no edited table moved the estimate"


def _joint_tables(seed=9, n=4_000):
    target = next(t for t in TARGETS if t.label == "joint_Y1_and_X0")
    derivation = _identified(target).derivation
    return derivation, cited_only(derivation, sampled_tables(target.case.laws(), n, seed))


def _emptied(table, cell):
    counts = list(table.counts)
    neighbour = cell + 1 if cell + 1 < len(counts) else cell - 1
    counts[neighbour] += counts[cell]
    counts[cell] = 0
    return transport.SampledCountTable(
        table.population,
        table.regime,
        table.axes,
        counts,
        table.snapshot_identity,
        interventions=table.interventions,
        structural_zeros=table.structural_zeros,
    )


def test_unseen_cells_are_refused_smoothed_or_enveloped_and_never_zero_filled():
    derivation, tables = _joint_tables()
    observational = next(
        i for i, t in enumerate(tables) if t.population == "source" and not t.interventions
    )
    counts = tables[observational].counts
    cell = min((c for c in range(len(counts)) if counts[c] > 0), key=lambda c: counts[c])
    tables[observational] = _emptied(tables[observational], cell)

    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, tables)
    assert raised.value.reason_code == "transport_support_failure"
    assert "counterfactual_transport_sampled.unseen_cell" in str(raised.value)

    smoothed = _estimate(
        derivation, tables, unseen_rule=transport.UnseenCellRule.additive_smoothing(0.5)
    )
    assert smoothed.labels["point_estimate"] == "smoothed_plug_in_point_estimate"
    assert smoothed.point["smoothed"] is True
    report = next(
        t for t in smoothed.tables if t["population"] == "source" and not t["interventions"]
    )
    assert report["smoothed"] is True and report["pseudo_count"] == 0.5
    assert report["unseen_cells"] == 1
    assert 0.0 <= smoothed.probability <= 1.0
    assert smoothed.interval["available"] is False

    # Declaring the cell a structural zero is a different, labelled claim.
    structural = list(tables)
    t = structural[observational]
    structural[observational] = transport.SampledCountTable(
        t.population,
        t.regime,
        t.axes,
        t.counts,
        t.snapshot_identity,
        interventions=t.interventions,
        structural_zeros=(*t.structural_zeros, cell),
    )
    declared = _estimate(derivation, structural)
    assert declared.probability != smoothed.probability

    enveloped = _estimate(
        derivation, tables, unseen_rule=transport.UnseenCellRule.unseen_mass_envelope(0.01)
    )
    assert enveloped.envelope is not None
    assert enveloped.envelope["label"] == "structural_envelope_vertex_family"
    assert enveloped.envelope["lower"] <= enveloped.envelope["upper"]
    assert enveloped.envelope["upper"] - enveloped.envelope["lower"] < 0.1
    assert enveloped.standard_error is None
    assert enveloped.standard_error_withheld_reason == (
        "partial_identification_has_no_single_standard_error"
    )
    assert enveloped.interval["available"] is False

    # A sparsity limit stricter than the table's unseen share refuses.
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(
            derivation,
            tables,
            unseen_rule=transport.UnseenCellRule.additive_smoothing(0.5),
            limits=transport.SampledTransportLimits(max_unseen_fraction=0.0),
        )
    assert "counterfactual_transport_sampled.sparsity_above_limit" in str(raised.value)


def test_overlap_must_be_declared_and_double_counting_is_refused():
    derivation, tables = _joint_tables(seed=5, n=3_000)
    assert len(tables) >= 3

    def key(table):
        return transport.SampledTableKey(table.population, table.regime, table.interventions)

    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, tables, overlap=transport.SamplingOverlap.undeclared())
    assert raised.value.reason_code == "sampling_dependence_unknown"
    assert "counterfactual_transport_sampled.overlap_unknown" in str(raised.value)

    one_unknown = transport.SamplingOverlap(
        default="independent", pairs=((key(tables[0]), key(tables[1]), "unknown"),)
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, tables, overlap=one_unknown)
    assert raised.value.reason_code == "sampling_dependence_unknown"

    # Shared units between tables that are not the same observations cannot be
    # given a covariance from counts: refused, not approximated.
    unlinked = transport.SamplingOverlap(
        default="independent", pairs=((key(tables[0]), key(tables[1]), "shared_units"),)
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, tables, overlap=unlinked)
    assert raised.value.reason_code == "route_not_supported"
    assert "counterfactual_transport_sampled.shared_units_unlinked" in str(raised.value)

    independent = _estimate(derivation, tables)
    assert len(independent.datasets) == len(tables)
    assert all(len(group) == 1 for group in independent.datasets)

    # Equal aggregate counts do not establish shared observations when the
    # physical snapshots differ.
    experiments = [i for i, t in enumerate(tables) if t.population == "source" and t.interventions]
    assert len(experiments) == 2
    original = tables[experiments[0]]
    twin = tables[experiments[1]]
    assert original.snapshot_identity != twin.snapshot_identity
    same_counts_distinct_snapshot = list(tables)
    same_counts_distinct_snapshot[experiments[1]] = transport.SampledCountTable(
        twin.population,
        twin.regime,
        twin.axes,
        original.counts,
        twin.snapshot_identity,
        interventions=twin.interventions,
        structural_zeros=original.structural_zeros,
    )
    coincidentally_equal = _estimate(derivation, same_counts_distinct_snapshot)
    assert len(coincidentally_equal.datasets) == len(same_counts_distinct_snapshot)

    # One physical dataset under two labels is double counting when declared
    # independent and one multinomial when declared shared.
    same_snapshot = list(same_counts_distinct_snapshot)
    same_snapshot[experiments[1]] = transport.SampledCountTable(
        twin.population,
        twin.regime,
        twin.axes,
        original.counts,
        original.snapshot_identity,
        interventions=twin.interventions,
        structural_zeros=original.structural_zeros,
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, same_snapshot)
    assert raised.value.reason_code == "evidence_double_counted"
    shared = transport.SamplingOverlap(
        default="independent",
        pairs=(
            (
                key(same_snapshot[experiments[0]]),
                key(same_snapshot[experiments[1]]),
                "shared_units",
            ),
        ),
    )
    grouped = _estimate(derivation, same_snapshot, overlap=shared)
    assert len(grouped.datasets) == len(same_snapshot) - 1


def test_uncited_missing_and_undersized_tables_are_refused_before_a_number():
    derivation, tables = _joint_tables(seed=11, n=3_000)
    withheld = [t for t in tables if not (t.population == "source" and t.interventions)]
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, withheld)
    assert raised.value.reason_code == "transport_missing_provider"
    assert "counterfactual_transport_sampled.missing_table" in str(raised.value)

    extra = transport.SampledCountTable(
        "target", "target|X", (("Z", (0.0, 1.0)),), [10, 10], "snap|elsewhere"
    )
    with pytest.raises(CausalError) as raised:
        _estimate(derivation, [*tables, extra])
    assert isinstance(raised.value, CausalError)  # a regime the catalog does not hold

    undersized = list(tables)
    t = undersized[0]
    undersized[0] = transport.SampledCountTable(
        t.population,
        t.regime,
        t.axes,
        [1] + [0] * (len(t.counts) - 1),
        t.snapshot_identity,
        interventions=t.interventions,
        structural_zeros=(),
    )
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, undersized)
    assert raised.value.reason_code == "transport_support_failure"
    assert "counterfactual_transport_sampled.sample_size_below_limit" in str(raised.value)


def test_sampled_estimate_zero_support_is_a_typed_refusal():
    case = zero_support_case()
    worlds = (World("natural"), World("treated", {"X": 0.0}))
    decision = case.decide(worlds, (Event("Y", 1, "treated"),), (Event("X", 1, "natural"),))
    derivation = decision.derivation
    tables = cited_only(derivation, sampled_tables(case.laws(), 2_000, 31))
    with pytest.raises(CausalUnsupportedError) as raised:
        _estimate(derivation, tables)
    assert raised.value.reason_code == "transport_support_failure"
    assert "counterfactual_transport.zero_conditioning_mass" in str(raised.value)


def test_sampled_artifact_round_trips_and_a_fresh_consumer_recomputes_from_the_counts(tmp_path):
    derivation, tables = _joint_tables(seed=21, n=10_000)
    result = _estimate(derivation, tables)
    artifact = bytes(result.export())
    consumed = transport.consume_sampled_counterfactual_transport_artifact(
        artifact,
        expected_premises_digest=result.premises_digest,
        expected_data_digest=result.data_digest,
    )
    assert consumed.independently_verified
    assert consumed.probability == result.probability  # bit for bit
    assert consumed.standard_error == result.standard_error
    assert consumed.labels == result.labels
    assert consumed.interval["available"] is False
    assert consumed.evidence_identity == result.evidence_identity

    # Edited counts (data identity), a wrong expectation and consumer limits.
    edited = artifact.replace(b"snap|source|", b"snap|sourcE|")
    assert edited != artifact
    with pytest.raises(CausalError) as raised:
        transport.consume_sampled_counterfactual_transport_artifact(edited)
    assert isinstance(raised.value, CausalError)
    assert raised.value.reason_code == "invalid_argument"
    assert "counterfactual_transport_sampled.invalid_artifact" in str(raised.value)
    with pytest.raises(CausalError) as raised:
        transport.consume_sampled_counterfactual_transport_artifact(
            artifact, expected_premises_digest="not-the-identity"
        )
    assert raised.value.reason_code == "invalid_argument"
    with pytest.raises(CausalUnsupportedError) as raised:
        transport.consume_sampled_counterfactual_transport_artifact(artifact, max_tables=1)
    assert raised.value.reason_code == "route_not_supported"

    # Sampled single-bit edits refuse or leave the recomputed answer unchanged.
    stride = max(1, len(artifact) // 60)
    for position in range(0, len(artifact), stride):
        mutated = bytearray(artifact)
        mutated[position] ^= 0x01
        try:
            again = transport.consume_sampled_counterfactual_transport_artifact(
                bytes(mutated),
                expected_premises_digest=result.premises_digest,
                expected_data_digest=result.data_digest,
            )
        except CausalError:
            continue
        assert again.probability == result.probability, position

    path = tmp_path / "sampled.bin"
    path.write_bytes(artifact)
    script = _FRESH_PROCESS.replace(
        "consume_counterfactual_transport_artifact",
        "consume_sampled_counterfactual_transport_artifact",
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), result.premises_digest, result.data_digest],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(done.stdout)
    assert report["verified"] is True
    assert report["probability"] == result.probability


def test_an_exact_artifact_is_not_a_sampled_artifact():
    _, _, exact = _exported()
    with pytest.raises(CausalError):
        transport.consume_sampled_counterfactual_transport_artifact(exact)
    derivation, tables = _joint_tables(seed=3, n=2_000)
    sampled = bytes(_estimate(derivation, tables).export())
    with pytest.raises(CausalError):
        transport.consume_counterfactual_transport_artifact(sampled)


def test_sampled_value_objects_validate_their_shape():
    with pytest.raises(CausalValueError):
        transport.UnseenCellRule("additive_smoothing")
    with pytest.raises(CausalValueError):
        transport.UnseenCellRule("nope")
    with pytest.raises(CausalValueError):
        transport.SamplingOverlap(default="maybe")
    with pytest.raises(CausalValueError):
        transport.SampledTransportLimits(max_unseen_fraction=1.5)
    with pytest.raises(CausalValueError):
        transport.SampledCountTable("p", "r", (("Z", (0.0, 1.0)),), [1, -1], "s")
    assert transport.UnseenCellRule.refuse().parameter is None


# ---------------------------------------------------------------------------
# Routed sampled stages execute from the retained plan with no builder alive.
# ---------------------------------------------------------------------------


def _sampled_builder(label="joint_Y1_and_X0", seed=4, n=20_000):
    target = _label(label)
    builder = target.case.decide(target.worlds, target.event, target.given)
    tables = sampled_tables(target.case.laws(), n, seed)
    return target, builder, tables


def test_estimate_sampled_runs_the_retained_plan_after_builder_disposal():
    target, builder, tables = _sampled_builder()
    derivation = builder.derivation
    cited = cited_only(derivation, tables)
    del builder
    plan = derivation.derivation
    assert plan["cited"] and plan["canonical_text"]
    result = derivation.estimate_sampled(cited, overlap=transport.SamplingOverlap.all_independent())
    assert abs(result.probability - float(target.expected)) <= PLUG_IN_TOLERANCE
    assert result.interval["available"] is False
    assert result.derivation_identity


def test_sampled_export_binds_the_counts_and_the_plan_after_builder_disposal():
    _, builder, tables = _sampled_builder(seed=6)
    derivation = builder.derivation
    cited = cited_only(derivation, tables)
    del builder
    independent = transport.SamplingOverlap.all_independent()
    result = derivation.estimate_sampled(cited, overlap=independent)
    artifact = bytes(result.export())
    plan = derivation.derivation
    assert plan["canonical_text"]
    again = derivation.estimate_sampled(list(reversed(cited)), overlap=independent)
    assert bytes(again.export()) == artifact, "the artifact is canonical in table order"
    assert again.premises_digest == result.premises_digest
    assert again.data_digest == result.data_digest


def test_sampled_consumer_recomputes_from_the_counts_after_builder_disposal():
    _, builder, tables = _sampled_builder(seed=8)
    derivation = builder.derivation
    cited = cited_only(derivation, tables)
    result = derivation.estimate_sampled(cited, overlap=transport.SamplingOverlap.all_independent())
    artifact = bytes(result.export())
    del builder
    consumed = transport.consume_sampled_counterfactual_transport_artifact(
        artifact,
        expected_premises_digest=result.premises_digest,
        expected_data_digest=result.data_digest,
    )
    plan = derivation.derivation
    assert plan["canonical_text"]
    assert consumed.independently_verified
    assert consumed.probability == result.probability
    assert consumed.labels == result.labels
