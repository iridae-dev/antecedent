#!/usr/bin/env python3
"""Ask which doses keep an outcome above a floor, using the outcome law of a composed model.

Question: a dose X acts on a biomarker M, and M acts on a response Y. For each of three declared
doses, which ones give a response of at least 9 with probability at least one half? Which keep the
10th percentile of the response at 8 or more? An expected value cannot answer either question; the
answer needs the spread of Y as well as its mean.

Data: none. The three mechanisms are written down as a linear-Gaussian model with the declared
coefficients and noise levels in the first cell: M = 1 + 1.5 X + N(0, 0.8^2) and
Y = 2 + 0.5 X + 2 M + N(0, 0.6^2). Under do(X = v), Y is Gaussian with mean 4 + 3.5 v and variance
4 * 0.8^2 + 0.6^2.

What the asserts check: the probability and the quantile the library reports for each dose equal
a closed-form Gaussian calculation made here from those declared numbers (math.erf and
statistics.NormalDist); the expected-value question is answered from the mean; a quantile asked on
a mean coordinate is refused with a stated reason; and a sealed query replays to the same answer
and identity.

What the result does not claim: the mechanisms are assumptions the analyst wrote, not estimates.
The answer is conditional on them, and so is the replay, which checks that the numbers match the
declared model and not that the model is true. The outcome law has noise but no parameter
uncertainty, so it carries no standard error and no interval region.
"""

from __future__ import annotations

# %%
import math
from statistics import NormalDist

from antecedent import decision
from antecedent import inverse_query as iq
from antecedent import inverse_sources as sources
from antecedent import mechanism_composition as mc
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

DOSES = (0.0, 1.0, 2.0)
M_INTERCEPT, M_SLOPE, M_SIGMA = 1.0, 1.5, 0.8
Y_INTERCEPT, Y_X, Y_M, Y_SIGMA = 2.0, 0.5, 2.0, 0.6
ROLES = {"X": "treatment", "M": "mediator", "Y": "outcome"}
TOL = 1e-6


def quantity(variable: str) -> mc.Quantity:
    return mc.Quantity(
        variable, units="mg", population_id="pop", role=ROLES[variable], variable_name=variable
    )


def linear(variable, intercept, sigma, parents=()):
    names = [name for name, _ in parents]
    declaration = mc.MechanismDeclaration(
        output=quantity(variable),
        parents=tuple(quantity(name) for name in names),
        artifact_id=f"declared:{variable}",
        evidence_id=f"assumption:{variable}",
    )
    return mc.Mechanism.declared_linear(
        declaration, intercept=intercept, sigma=sigma, coefficients=dict(parents)
    )


def compose(m_slope: float = M_SLOPE):
    return mc.compose_mechanisms(
        [
            linear("X", 1.0, 2.0),
            linear("M", M_INTERCEPT, M_SIGMA, [("X", m_slope)]),
            linear("Y", Y_INTERCEPT, Y_SIGMA, [("X", Y_X), ("M", Y_M)]),
        ],
        target_population_id="pop",
        admit_declared_mechanisms=True,
    )


def response_under(dose: float, functional: str) -> ScientificQuantity:
    """``functional`` of Y (``mean`` or ``outcome``, the law) under ``do(X = dose)``."""
    return ScientificQuantity(
        variable_id="Y",
        variable_name="Y",
        role="outcome",
        units="mg",
        population_id="pop",
        regime_id=f"do(X={dose:g})",
        horizon=0,
        functional_id=functional,
    )


def action_id(dose: float) -> str:
    return f"dose_{dose:g}"


def actions(functional: str) -> list[sources.ComposedAction]:
    return [
        sources.ComposedAction(
            action_id(dose), response_under(dose, functional), {"X": dose}, outcome="Y"
        )
        for dose in DOSES
    ]


def inverse_query(functional: str, constraint: iq.Constraint) -> iq.InverseQuery:
    """Which doses meet ``constraint`` on Y itself (the utility is the identity)."""
    contract = decision.Contract(
        actions=tuple(
            decision.Action(
                action_id(dose), inputs=(response_under(dose, functional),), utility=decision.x(0)
            )
            for dose in DOSES
        ),
        utility_units="mg",
        criterion=decision.Criterion.expected_utility(),
        target_population="pop",
    )
    return iq.InverseQuery(contract, tuple(action_id(d) for d in DOSES), (constraint,))


# Closed-form truth from the declared numbers: Y | do(X = v) is Gaussian.
def true_mean(dose: float) -> float:
    return Y_INTERCEPT + Y_X * dose + Y_M * (M_INTERCEPT + M_SLOPE * dose)


TRUE_SD = math.sqrt(Y_M**2 * M_SIGMA**2 + Y_SIGMA**2)


def true_upper_probability(dose: float, threshold: float) -> float:
    z = (threshold - true_mean(dose)) / TRUE_SD
    return 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))


def true_quantile(dose: float, p: float) -> float:
    return NormalDist(true_mean(dose), TRUE_SD).inv_cdf(p)


# %% [markdown]
# ## Compose the mechanisms and declare the doses
#
# Each action is one hard intervention on X. Asking for the coordinate with
# `functional_id="outcome"` keeps the whole conditional law of Y (its mean and the noise that
# remains after the intervention). Asking for `mean` keeps only the expected value.

# %%
composition = compose()
law_source = require_success(sources.derive_composed_source(composition, actions("outcome")))
mean_source = require_success(sources.derive_composed_source(composition, actions("mean")))
print("outcome-law source:", law_source.origin.kind, law_source.origin.representation)
print("mean source:", mean_source.origin.kind, mean_source.origin.representation)
print("assumed mechanisms:", sorted(law_source.origin.assumed_mechanisms))

assert law_source.origin.kind == "composed_mechanism"
assert law_source.origin.representation == "law"
assert mean_source.origin.representation == "point_only"
assert set(law_source.origin.assumed_mechanisms) == {"X", "M", "Y"}
assert not law_source.origin.transported

# %% [markdown]
# ## A probability question
#
# Which doses give `P(Y >= 9) >= 0.5`? The reported probability for each dose is compared with the
# closed-form value from the declared coefficients and noise.

# %%
probability = require_success(
    law_source.evaluate(inverse_query("outcome", iq.probability_threshold(9.0, 0.5)))
)
for dose in DOSES:
    report = probability.action(action_id(dose))
    reported = report.point_values[0].value
    expected = true_upper_probability(dose, 9.0)
    print(f"dose {dose:g}: P(Y >= 9) = {reported:.4f} (closed form {expected:.4f}) {report.point}")
    assert abs(reported - expected) < TOL
    assert report.point_values[0].standard_error is None
    assert (report.point == "feasible") == (expected >= 0.5)
print("feasible doses:", probability.feasible_actions, "selected:", probability.selected)
assert probability.feasible_actions == ("dose_2",)

# %% [markdown]
# ## A quantile question, and what an expected value says instead
#
# The 10th percentile of Y must be at least 8. Separately, the expected value of Y must be at least
# 7. The mean question is answered from the mean coordinate alone and stays a point estimate; it
# accepts doses 1 and 2, while the probability question above accepts only dose 2. Two doses with
# a comfortable mean can still fail a floor on a tail.

# %%
quantile = require_success(
    law_source.evaluate(inverse_query("outcome", iq.target_quantile(0.1, 8.0)))
)
for dose in DOSES:
    reported = quantile.action(action_id(dose)).point_values[0].value
    expected = true_quantile(dose, 0.1)
    print(f"dose {dose:g}: 10th percentile of Y = {reported:.4f} (closed form {expected:.4f})")
    assert abs(reported - expected) < TOL
assert quantile.feasible_actions == ("dose_2",)

mean_answer = require_success(mean_source.evaluate(inverse_query("mean", iq.target_mean(7.0))))
for dose in DOSES:
    assert abs(mean_answer.action(action_id(dose)).point_values[0].value - true_mean(dose)) < TOL
print("expected value at least 7:", mean_answer.feasible_actions)
assert mean_answer.feasible_actions == ("dose_1", "dose_2")
assert mean_answer.origin.representation == "point_only"

# %% [markdown]
# ## A quantile on a mean coordinate refuses
#
# A mean has no spread, so no quantile or probability can be read from it. The refusal names the
# reason and the remedy; nothing is guessed from the mean.

# %%
try:
    require_success(mean_source.evaluate(inverse_query("mean", iq.target_quantile(0.1, 8.0))))
except iq.InverseQueryRefusal as refusal:
    print("refused:", refusal.reason_code, "/", refusal.detail)
    print("remedy:", refusal.remedy)
    assert refusal.reason_code == "decision_contract_unsatisfied"
    assert refusal.detail == "inverse_forward_source.composed_law_not_materialized"
    assert refusal.remedy
else:
    raise AssertionError("a quantile on a mean coordinate should be refused")

# %% [markdown]
# ## Seal the answer and replay it
#
# The sealed artifact carries the Gaussian parameters and the origin of the claim, so a consumer
# replays the answer without sampling. A refitted mechanism is a new source with a new claim
# digest; the sealed answer is never refreshed in place.

# %%
query = inverse_query("outcome", iq.probability_threshold(9.0, 0.5))
sealed = require_success(law_source.seal_per_action(query))
assert set(sealed) == {action_id(dose) for dose in DOSES}
for name, artifact in sealed.items():
    replay = require_success(
        iq.consume(bytes(artifact.export()), expected_identity=artifact.identity)
    )
    assert replay.origin == artifact.origin
    assert replay.identity == artifact.identity
    assert replay.feasible_actions == artifact.feasible_actions
print("replayed:", {name: art.feasible_actions for name, art in sealed.items()})

refit = require_success(sources.derive_composed_source(compose(m_slope=2.0), actions("outcome")))
print("same claim after refit:", refit.origin.claim_digest == law_source.origin.claim_digest)
assert refit.origin.claim_digest != law_source.origin.claim_digest
assert refit.origin.source_identity != law_source.origin.source_identity
