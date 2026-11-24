"""Decisions across preferences over graph-posterior class atoms under a declared member measure.

Fixture: a confounded table over ``t, y, z, w`` (``z`` follows ``w``, ``t = 1 + 0.4 z + noise``,
``y = 0.5 + 1.5 t + z + noise``) and a posterior with two CPDAG atoms: the path ``w - z - t``
with ``t -> y`` (three members, one per source of the path) at weight 0.6 and the fully
oriented ``w -> z -> t -> y`` at weight 0.4. Actions ``low`` and ``high`` read the mean of
``y`` under ``do(t = 0.5)`` and ``do(t = 1.5)`` and carry costs 0.5 and 1.5.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import _reason_details as reason_details
from antecedent import decision
from antecedent.inference import ClassPrior, _class_prior_kwargs
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

UNIFORM = "assumption.uniform_within_class"
NAMES = ["t", "y", "z", "w"]
DOSES = [0.5, 1.5]


# --------------------------------------------------------------------------- declarations


def test_a_uniform_measure_names_its_assumption():
    prior = ClassPrior.uniform(UNIFORM)
    assert prior.uniform_measure
    assert prior.assumption == UNIFORM
    with pytest.raises(ValueError, match="assumption must name"):
        ClassPrior.uniform("  ")
    with pytest.raises(ValueError, match="names its assumption"):
        ClassPrior(uniform_measure=True)
    with pytest.raises(ValueError, match="exactly one of"):
        ClassPrior(pairs=((1, 1.0),), uniform_measure=True, assumption=UNIFORM)


def test_exact_limit_and_sample_are_validated():
    prior = ClassPrior.uniform(UNIFORM).exact_limit(4).sample(25, seed=7)
    assert prior.completion_limit == 4
    assert prior.sampling == (25, 7)
    with pytest.raises(ValueError, match="exact limit"):
        ClassPrior.uniform(UNIFORM).exact_limit(0)
    with pytest.raises(ValueError, match="at least one member"):
        ClassPrior.uniform(UNIFORM).sample(0, seed=1)
    with pytest.raises(ValueError, match="seed must be"):
        ClassPrior.uniform(UNIFORM).sample(5, seed=-1)
    # Keyed masses need every member's key, so only a uniform measure is sampled.
    with pytest.raises(ValueError, match="only a uniform"):
        ClassPrior.from_pairs([(3, 1.0)]).sample(5, seed=1)


def test_a_temporal_class_refuses_a_member_measure():
    # A temporal class reads ordered or keyed masses (see the temporal class envelope tests);
    # a uniform member measure, an exact limit or a sample weighs graph-posterior class atoms.
    with pytest.raises(ValueError, match="temporal class takes ordered or keyed"):
        _class_prior_kwargs(ClassPrior.uniform(UNIFORM))


# --------------------------------------------------------------------------- compiled decisions


def _edge(source: str, target: str) -> int:
    n, i, j = len(NAMES), NAMES.index(source), NAMES.index(target)
    return 1 << (i * (n - 1) + (j if j < i else j - 1))


def _undirected(a: str, b: str) -> int:
    return _edge(a, b) | _edge(b, a)


PATH = _undirected("w", "z") | _undirected("z", "t") | _edge("t", "y")
ORIENTED = _edge("w", "z") | _edge("z", "t") | _edge("t", "y")


def _posterior():
    return ant.discovery.GraphPosterior.from_atoms(
        NAMES, [0.6, 0.4], [PATH, ORIENTED], atom_kind="Cpdag"
    )


def _table(rows: int = 300):
    i = np.arange(rows, dtype=np.float64)
    w = np.sin(0.37 * i)
    z = 0.6 * w + 0.5 * np.sin(0.91 * i)
    t = 1.0 + 0.4 * z + 0.5 * np.sin(2.3 * i)
    y = 0.5 + 1.5 * t + z + 0.3 * np.sin(1.3 * i)
    return {"t": t, "y": y, "z": z, "w": w}


def _prepared(prior: ClassPrior | None):
    return ant.prepare(
        _table(),
        discovery=_posterior(),
        query=ant.ResponseCurve("t", "y", grid=DOSES),
        refute="none",
        inference=ant.Bayesian(backend="conjugate", n_draws=64),
        class_prior=prior,
        execution=ant.Execution(seed=11),
    )


def _dose(dose: float) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="points",
        population_id="target",
        regime_id=f"do(t={dose})",
        horizon=0,
        functional_id="mean",
    )


def _contract(structural_policy: str) -> decision.Contract:
    benefit = decision.attribute("benefit", units="points")
    cost = decision.attribute("cost", units="usd")
    price = decision.tradeoff("price", units="points/usd", lower=0.1, upper=0.5, assumption="price")
    return decision.Contract(
        value=decision.Value(benefit - price * cost, units="points"),
        actions=(
            decision.Action("low", inputs={"benefit": _dose(0.5), "cost": 0.5}),
            decision.Action("high", inputs={"benefit": _dose(1.5), "cost": 1.5}),
        ),
        target_population="target",
        structural_policy=structural_policy,
    )


def _path_members(described):
    keys = {s["class_key"] for s in described["structures"]}
    path = max(keys, key=lambda key: sum(s["class_key"] == key for s in described["structures"]))
    return [s for s in described["structures"] if s["class_key"] == path]


def test_bayes_over_structures_without_a_measure_still_refuses():
    with pytest.raises(decision.PreferenceRefusal) as refused:
        require_success(_contract("bayes_over_structures").compile(_prepared(None)))
    assert refused.value.detail == reason_details.decision_adapters.COMPLETION_COUNT_NOT_PROBABILITY


def test_a_uniform_measure_makes_members_probabilities():
    compiled = require_success(
        _contract("bayes_over_structures").compile(_prepared(ClassPrior.uniform(UNIFORM)))
    )
    described = compiled.inspect()
    members = _path_members(described)
    assert len(members) == 3
    for member in members:
        assert member["weight"] == pytest.approx(0.2)
        assert member["fingerprint"] is not None
        assert member["multiplicity"] == 1
        assert member["sampled"] is False
    assert compiled.class_sampling == ()
    decided = require_success(compiled.execute())
    probabilities = sorted(s.probability for s in decided.structures)
    assert probabilities == pytest.approx([0.2, 0.2, 0.2, 0.4])
    assert decided.sampling == ()


def test_keyed_masses_are_read_by_member_fingerprint():
    uniform = require_success(
        _contract("require_invariant_best_action").compile(_prepared(ClassPrior.uniform(UNIFORM)))
    )
    described = uniform.inspect()
    prints = sorted(m["fingerprint"] for m in _path_members(described))
    (oriented,) = [
        s["fingerprint"] for s in described["structures"] if s not in _path_members(described)
    ]
    # The oriented DAG is also a member of the path class: a member is keyed by its content,
    # once, and its mass is read within each class that holds it.
    assert oriented in prints
    keyed = ClassPrior.from_pairs([(prints[0], 1.0), (prints[1], 2.0), (prints[2], 1.0)])
    compiled = require_success(_contract("require_invariant_best_action").compile(_prepared(keyed)))
    for member in _path_members(compiled.inspect()):
        expected = 0.3 if member["fingerprint"] == prints[1] else 0.15
        assert member["weight"] == pytest.approx(expected)

    missing = ClassPrior.from_pairs([(prints[0], 1.0)])
    with pytest.raises(decision.PreferenceRefusal) as refused:
        require_success(_contract("require_invariant_best_action").compile(_prepared(missing)))
    assert refused.value.reason_code == "invalid_argument"
    assert refused.value.detail == reason_details.preference_compile.CLASS_PRIOR_KEYS_MISMATCH


def test_a_sampled_class_is_never_called_best_for_every_member():
    prior = ClassPrior.uniform(UNIFORM).exact_limit(2).sample(40, seed=9)
    compiled = require_success(_contract("bayes_over_structures").compile(_prepared(prior)))
    (record,) = compiled.class_sampling
    assert record.scheme == "clique_picking"
    assert (record.class_size, record.class_size_exact) == (3, True)
    assert (record.drawn, record.seed, record.assumption) == (40, 9, UNIFORM)
    members = _path_members(compiled.inspect())
    assert sum(m["multiplicity"] for m in members) == 40
    assert sum(m["weight"] for m in members) == pytest.approx(0.6)

    decided = require_success(compiled.execute())
    assert decided.kind in ("best_on_sampled_measure", "undetermined_on_sampled_measure")
    assert decided.kind != "necessarily_best"
    assert all(a.necessarily_best is None for a in decided.actions)
    (mass,) = decided.sampled_masses
    total = mass.identified[0] + mass.unidentified[0] + mass.unevaluated[0]
    assert total == pytest.approx(0.6)
    assert decided.margins
    assert all(m.standard_error is not None for m in decided.margins)
    assert "sampled" in decided.statement

    invariant_contract = _contract("require_invariant_best_action")
    invariant = require_success(invariant_contract.compile(_prepared(prior)))
    per_member = require_success(invariant.execute())
    assert per_member.kind in ("best_on_every_sampled_member", "per_sampled_member")


def test_a_sampled_result_exposes_no_exact_verdict():
    # Every policy over a sampled class: no reading of the result names an action best for
    # every member, and the exported mixture states that its members were sampled.
    prior = ClassPrior.uniform(UNIFORM).exact_limit(2).sample(40, seed=9)
    for policy in ("bayes_over_structures", "maximin", "require_invariant_best_action"):
        compiled = require_success(_contract(policy).compile(_prepared(prior)))
        decided = require_success(compiled.execute())
        assert decided.kind not in ("necessarily_best", "uniquely_best", "tied")
        assert all(a.necessarily_best is None for a in decided.actions)
        assert all(a.uniquely_best is None for a in decided.actions)
        assert decided.draw_share_best_everywhere is None
        body = decided.to_dict()
        assert body["compiled"]["mixture"]["weight_basis"] == "sampled_class_measure"
        assert body["compiled"]["executed"] == body["compiled"]["programs"]
        assert body["compiled"]["options"]
        combined = body.get("combined") or {}
        assert combined.get("draw_share_best_everywhere") is None
        for action in combined.get("actions") or ():
            assert action.get("necessarily_best") is None
            assert action.get("uniquely_best") is None


def test_above_the_limit_without_a_sample_the_class_is_refused():
    prior = ClassPrior.uniform(UNIFORM).exact_limit(2)
    with pytest.raises(decision.PreferenceRefusal) as refused:
        require_success(_contract("bayes_over_structures").compile(_prepared(prior)))
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == reason_details.preference_compile.CLASS_COMPLETIONS_TRUNCATED


def test_ordered_masses_name_no_member_of_a_graph_posterior_class():
    with pytest.raises(ValueError, match="ordered masses name no member"):
        require_success(_prepared(ClassPrior.from_ordered([1.0, 2.0])).estimate())


def test_the_graph_posterior_estimate_refuses_a_class_prior_it_does_not_read():
    # The mixture estimates each class atom as a whole; only a decision compiled across the
    # posterior's structures reads a measure over members, so the estimate refuses it.
    prepared = _prepared(ClassPrior.uniform(UNIFORM))
    with pytest.raises(ant.errors.CausalError) as refused:
        require_success(prepared.estimate())
    assert reason_details.structure_posterior.CLASS_PRIOR_NOT_READ in str(refused.value)
    assert require_success(_prepared(None).estimate()) is not None
