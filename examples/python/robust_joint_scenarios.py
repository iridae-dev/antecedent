#!/usr/bin/env python3
"""Choose between actions that are robust to several explicit joint scenarios, then replay it.

Question: three actions are on the table: run campaign A, run campaign B, or run both and
count a success only when both campaigns succeed. Nobody can say which of three declared
scenarios holds, and the scenarios disagree about how the two campaigns' outcomes move together.
Which action has the best worst case, and what changes if the scenario search was not finished?

Data: declared, not estimated. Each scenario is a finite table of joint outcome rows, written
in the script, so the cross-action dependence is stated rather than built from marginals. A tiny
simulated table only anchors the prepared study.

The asserts check that the compiled maximin choice and each action's worst and best value match
a hand computation over the scenario table, that invariant-best names a disagreement instead of
choosing, that a non-exhaustive search gives an incomplete verdict that reports the unresolved
prior mass, and that the sealed result replays only under an independently compiled owner.

The result does not claim an expected utility or a posterior over the scenarios. It holds only
conditional on the declared scenarios, and replay cannot show that the scenario set is complete.
"""

from __future__ import annotations

# %%
from fractions import Fraction

import antecedent as ant
import numpy as np
from antecedent import decision
from antecedent import decision_robust as robust
from antecedent.artifacts import LoadContext
from antecedent.errors import CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

# Each scenario is a table of (outcome of campaign A, outcome of campaign B, number of rows)
# out of ten equally likely rows. The same row index is the same state of the world, so the
# table says how the two outcomes move together.
SCENARIOS = {
    "co_moving": [(1, 1, 4), (1, 0, 2), (0, 1, 1), (0, 0, 3)],
    "substitutes": [(1, 0, 3), (0, 1, 5), (0, 0, 2)],
    "b_stronger": [(1, 1, 2), (1, 0, 2), (0, 1, 4), (0, 0, 2)],
}
# The payoff of each action as a function of the two success indicators (a, b).
PAYOFFS = {
    "campaign_a": lambda a, b: a,
    "campaign_b": lambda a, b: b,
    "both": lambda a, b: a * b,
}


def quantity(name: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=name,
        variable_name=name,
        role="outcome",
        units="units",
        population_id="target",
        regime_id="do(joint)",
        horizon=0,
        functional_id="outcome",
    )


A, B = quantity("a"), quantity("b")


def scenario_law(name: str) -> JointDistributionArtifact:
    """The scenario's rows as one finite joint law over (a, b)."""
    rows = [[float(a), float(b)] for a, b, count in SCENARIOS[name] for _ in range(count)]
    assert len(rows) == 10
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=(A, B),
        alignment="joint",
        source_id=f"scenario-{name}",
        provider_id="declared-table",
        rng_id="none",
        snapshot_id="synthetic",
        causal_contract_id="conditional",
    )
    return JointDistributionArtifact(identity, np.array(rows), calibration="exact")


def contract(policy: str) -> robust.AdmissibleContract:
    return robust.admissible_contract(
        decision.Contract(
            actions=(
                decision.Action("campaign_a", inputs=(A,), utility=decision.x(0)),
                decision.Action("campaign_b", inputs=(B,), utility=decision.x(0)),
                decision.Action("both", inputs=(A, B), utility=decision.x(0) * decision.x(1)),
            ),
            utility_units="units",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
            structural_policy=policy,  # type: ignore[arg-type]
        )
    )


def claims(*, priors: dict[str, float] | None = None) -> list[robust.Claim]:
    supported = robust.Support("supported")
    return [
        robust.Claim.evaluated(
            name,
            scenario_law(name),
            support=supported,
            probability=None if priors is None else priors[name],
        )
        for name in SCENARIOS
    ]


# The study only anchors the causal binding; no model is fitted from it.
study = ant.prepare(
    {
        "t": np.tile([0.0, 1.0, 0.0, 1.0], 4),
        "z": np.tile([0.0, 0.0, 1.0, 1.0], 4),
        "y": np.arange(16, dtype=float),
    },
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.AverageEffect(treatment="t", outcome="y"),
    inference=ant.Frequentist(bootstrap=0),
    refute=False,
)

# %% [markdown]
# ## Compute the answer by hand
#
# For each scenario and action the value is the mean payoff over the ten rows. Maximin picks
# the action whose lowest value across the scenarios is highest. The three actions are chosen so
# that the dependence matters: the scenarios disagree about whether `both` pays off at all.

# %%
by_hand: dict[str, dict[str, Fraction]] = {action: {} for action in PAYOFFS}
for scenario, table in SCENARIOS.items():
    for action, payoff in PAYOFFS.items():
        total = sum(count * payoff(a, b) for a, b, count in table)
        by_hand[action][scenario] = Fraction(total, 10)

worst = {action: min(values.values()) for action, values in by_hand.items()}
best = {action: max(values.values()) for action, values in by_hand.items()}
hand_choice = max(worst, key=lambda action: worst[action])
for action in PAYOFFS:
    print(f"{action:11s} worst {float(worst[action]):.2f}  best {float(best[action]):.2f}")
print("hand-computed maximin:", hand_choice)

assert worst == {
    "campaign_a": Fraction(3, 10),
    "campaign_b": Fraction(5, 10),
    "both": Fraction(0),
}
assert hand_choice == "campaign_b"
assert sorted(worst.values())[-1] > sorted(worst.values())[-2]  # the choice is not a tie

# %% [markdown]
# ## Compile and evaluate the maximin decision
#
# `joint_scenario_carrier` checks that every scenario is a joint law with the same causal
# binding as the study and keeps the declaration that the search was exhaustive. The compiler
# binds that source to the decision and evaluates it without fitting the study.

# %%
maximin = contract("maximin")
source = require_success(
    robust.joint_scenario_carrier(
        study, maximin, claims(), joint_structure="rows are common states", exhaustive=True
    )
)
compiled = require_success(robust.compile_robust_decision(study, maximin, source))
result = require_success(compiled.evaluate())

print(f"verdict {result.verdict.kind}, chosen {result.verdict.selected}")
print(f"standing {result.standing}, complete {result.complete}")

assert result.complete
assert result.standing == "conditional_on_declared_joint_scenarios"
assert result.verdict.kind == "worst_case_choice"
assert result.verdict.selected == hand_choice
for row in result["robust"]["actions"]:
    low, high = row["range"]
    assert abs(low - float(worst[row["id"]])) < 1e-12, row["id"]
    assert abs(high - float(best[row["id"]])) < 1e-12, row["id"]

# %% [markdown]
# ## Ask for the invariant-best action instead
#
# Invariant-best chooses an action only if it is uniquely best in every scenario. Here campaign A
# leads in one scenario and campaign B in the others, so the readout names the disagreement and
# does not average it away.

# %%
invariant = contract("require_invariant_best_action")
invariant_source = require_success(
    robust.joint_scenario_carrier(
        study, invariant, claims(), joint_structure="rows are common states", exhaustive=True
    )
)
invariant_result = require_success(
    require_success(robust.compile_robust_decision(study, invariant, invariant_source)).evaluate()
)
print("invariant-best verdict:", invariant_result.verdict.kind)
print("scenario leaders:", invariant_result.verdict.leaders)
assert invariant_result.verdict.kind == "graph_dependent_choice"
assert invariant_result.verdict.selected is None

# %% [markdown]
# ## Declare the search incomplete
#
# Suppose the search over scenarios stopped early and a fourth scenario was never evaluated. The
# source says so (`exhaustive=False`, with the stop reason) and carries the prior mass the
# analyst gave that scenario. The decision is then incomplete: it reports no choice, keeps the
# per-action values of the scenarios that were evaluated, and reports the unresolved mass.
# A count of completed scenarios would not be a probability, so only declared priors count here.

# %%
priors = {"co_moving": 0.45, "substitutes": 0.25, "b_stronger": 0.15}
unfinished = [
    *claims(priors=priors),
    robust.Claim.unevaluated("unsearched", "search stopped first", probability=0.15),
]
partial_source = require_success(
    robust.joint_scenario_carrier(
        study,
        maximin,
        unfinished,
        joint_structure="rows are common states",
        exhaustive=False,
        search_stop="operations",
    )
)
partial = require_success(
    require_success(robust.compile_robust_decision(study, maximin, partial_source)).evaluate()
)
print(f"complete {partial.complete}, verdict {partial.verdict.kind}")
print(f"search stop {partial['search_stop']}")
print(f"evaluated mass {partial['robust']['evaluated_mass']}")
print(f"unevaluated mass {partial['robust']['unevaluated_mass']}")

assert not partial.complete
assert partial.verdict.kind == "insufficient_claims"
assert partial.verdict.selected is None
assert partial["search_stop"] == "search.operations"
assert abs(partial["robust"]["unevaluated_mass"] - 0.15) < 1e-12
assert abs(partial["robust"]["evaluated_mass"] - 0.85) < 1e-12

# %% [markdown]
# ## Export and replay under an independent owner
#
# The sealed result cannot be read from its bytes alone. The consumer compiles the same decision
# again from a separately retained source, whose identity it checks, and recomputes the readout.
# Loading without that owner refuses.

# %%
sealed = compiled.export()
retained_identity = source.identity
independent_source = require_success(ant.load(source.export(), expected=retained_identity))
independent = require_success(robust.compile_robust_decision(study, maximin, independent_source))
assert independent.program_identity == compiled.program_identity

replayed = require_success(ant.load(sealed, context=LoadContext(robust_source=independent)))
assert dict(replayed) == dict(result)
print(f"replayed choice {replayed.verdict.selected}, identical to the evaluated result")

try:
    ant.load(sealed)
except CausalValueError as error:
    print("without an independent owner:", type(error).__name__)
else:
    raise AssertionError("loading without an independent owner should refuse")

# A source whose identity differs from the retained one is not accepted.
assert ant.load(source.export(), expected="0" * 64).accepted is False
