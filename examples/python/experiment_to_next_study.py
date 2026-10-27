"""Update a decision after an experiment, then rank the next studies.

A randomized comparison of a programme against a reference leaves a retained response law: 512
draws of the two arm means, from one fit. A later experiment measures the same contrast in the
same population. This example incorporates it through an explicit estimand mapping, shows the
changed response and the changed decision, ranks two candidate further studies by expected value
of sample information net of cost, and reads the status the update carries.

Every number comes from a simulation, so the known truth is a fixture truth of that stated model:
the reference arm averages 1.0 and the active arm 2.0, a true contrast of 1.0. The later
experiment's mean is declared as 0.3 above what the retained law expects; it is not drawn from the
world. The update is frozen importance conditioning of the retained draws. It does not refit the
model, and it does not turn the retained law into an identified causal result.
"""

from __future__ import annotations

import math
from dataclasses import replace

from antecedent import decision
from antecedent import joint_panel as jp
from antecedent import scientific_bridge as sb
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

# -- 1. The retained response law ------------------------------------------------------------
# 32 independent clusters, randomized with probability 1/2 to the reference (arm 0) or the active
# programme (arm 1). Reference clusters average 1.0 and active clusters 2.0.
units = [
    {
        "id": f"unit:{i}",
        "arm": i % 2,
        "probabilities": [0.5, 0.5],
        "outcome": (1.0 if i % 2 == 0 else 2.0) + (0.1 if i % 4 < 2 else -0.1),
        "population_weight": 1.0,
    }
    for i in range(32)
]
prepared = jp.prepare_native_joint_design(
    {
        "kind": "randomized",
        "design": {
            "worlds": ["reference", "active"],
            "sampling_unit": "independent_cluster",
            "population": "declared_cluster_population",
            "consistency": "keys denote exact complete assigned programs",
            "between_unit_isolation": "independent randomized clusters without spillover",
        },
        "units": units,
    }
)
result = require_success(
    prepared.execute(
        {
            "inference": "randomized_gaussian",
            "prior": {
                "mean": [0.0, 0.0],
                "variance": [4.0, 4.0],
                "outcome_variance": [1.0] * len(units),
                "draws": 512,
                "seed": 91,
            },
        }
    )
)
prior = result.report()
draws = prior["law"]["Moments"]["Bayesian"]["draws"]
COEFFICIENTS = [-1.0, 1.0]  # the contrast "active minus reference"
contrasts = [
    math.fsum(v * c for v, c in zip(row, COEFFICIENTS, strict=True)) for row in draws
]
prior_contrast = math.fsum(contrasts) / len(contrasts)
assert abs(prior_contrast - 1.0) < 0.1  # near the fixture's true contrast of 1.0

# -- 2. Map the experiment's estimand onto the retained response ------------------------------
# The retained target is the whole-program population mean contrast. The experiment reports a
# sample mean of the same contrast in the same population and regime. The bridge declares that
# identity mapping, its premise, and that it carries no extra mapping uncertainty.
target = result.experiment_quantity(
    coefficients=COEFFICIENTS, outcome_units="outcome_unit"
)
assert target.functional_id.startswith("whole_program_population_mean:linear:")
setting = sb.ExposureSetting(
    target.population_id,
    "native_summary:0",
    "native_summary:0",
    1.0,
    "native_projection",
    1.0,
    "native_summary_window",
    target.regime_id,
)
bridge = sb.BridgeContract(
    "parameter_intervention_response",
    replace(target, functional_id="experimental_sample_mean"),
    "model_parameter",
    target,
    "intervention_response",
    sb.Scope(target.population_id, target.regime_id),
    sb.AffineMapping(0.0, 1.0),
    assumptions=(
        sb.Assumption(
            "likelihood_model",
            "parameter_interpretation",
            "independent Gaussian sample mean maps to this exact native contrast",
            "declared_experiment",
        ),
    ),
    mapping_uncertainty=sb.MappingUncertainty("fixed_mapping", 0.0, 0.0, 0.0),
    source_dependence=sb.SourceDependence("slope_covariance", "retained_slots", 0.0),
    estimand_bridge=sb.EstimandBridge(
        "exact_translation",
        "declared_experiment",
        setting,
        setting,
        0.0,
        "native_window",
        0.0,
        "native_projection",
    ),
)

# The ledger names the three sources the update depends on, so authority can follow them.
ledger = EvidenceLedger()
dependency = prior["dependency"]
model = ledger.add_source(
    SourceDeclaration(dependency, "fitted_model", [("model", "rows")])
)
experiment = ledger.add_source(
    SourceDeclaration("new_experiment", "experiment", [("new", "rows")])
)
mapping = ledger.add_source(
    SourceDeclaration("mapping", "fitted_model", [("mapping", "rows")])
)

OBSERVED = prior_contrast + 0.3
OBSERVATION_VARIANCE = 0.2  # of one experimental observation
SAMPLE_SIZE = 4
updated = result.condition_experiment(
    coefficients=COEFFICIENTS,
    outcome_units="outcome_unit",
    bridge=bridge,
    ledger=ledger,
    observed_mean=OBSERVED,
    observation_variance=OBSERVATION_VARIANCE,
    sample_size=SAMPLE_SIZE,
    mapping_dependency=dependency,
    intercepts=[0.0] * len(contrasts),  # one mapping per retained draw: identity
    slopes=[1.0] * len(contrasts),
    model=model,
    experiment=experiment,
    mapping=mapping,
)

# -- 3. The changed response -----------------------------------------------------------------
report = updated.report(ledger)
weights = report["posterior_weights"]
# Independent arithmetic for the declared update: Gaussian likelihood ratios against the draw
# best supported by the experiment, normalised.
variance = OBSERVATION_VARIANCE / SAMPLE_SIZE
anchor = min(contrasts, key=lambda v: abs(OBSERVED - v))
raw = [
    math.exp((v - anchor) * (2 * OBSERVED - v - anchor) / (2 * variance))
    for v in contrasts
]
expected_weights = [w / math.fsum(raw) for w in raw]
assert all(abs(a - b) < 1e-12 for a, b in zip(weights, expected_weights, strict=True))
posterior_contrast = math.fsum(v * w for v, w in zip(contrasts, weights, strict=True))
assert posterior_contrast > prior_contrast + 0.1
print(
    f"contrast: {prior_contrast:.3f} before the experiment, {posterior_contrast:.3f} after"
)

# -- 4. The changed decision -----------------------------------------------------------------
# Act when the contrast beats a declared break-even of 1.1 outcome units; otherwise hold.
BREAK_EVEN = 1.1
assert prior_contrast < BREAK_EVEN < posterior_contrast
coordinates = tuple(ScientificQuantity.from_wire(q) for q in report["quantities"])
benefit = decision.const(-BREAK_EVEN)
for i, c in enumerate(COEFFICIENTS):
    benefit = benefit + decision.x(i) * c
contract = decision.Contract(
    actions=(
        decision.Action(
            "hold", inputs=coordinates, utility=0 * decision.x(0), kind="policy"
        ),
        decision.Action("act", inputs=coordinates, utility=benefit, kind="policy"),
    ),
    utility_units=target.units,
    criterion=decision.Criterion.expected_utility(),
    target_population=target.population_id,
)
outcome = updated.decide(contract, ledger)
by_id = {a["id"]: a for a in outcome["actions"]}
assert outcome["verdict"] == {"uniquely_optimal": "act"}
assert abs(by_id["act"]["expected_utility"] - (posterior_contrast - BREAK_EVEN)) < 1e-6
# The same rule on the retained law alone, which the experiment has not yet touched, holds.
assert prior_contrast - BREAK_EVEN < 0.0

# -- 5. The next study -----------------------------------------------------------------------
# Value each candidate by the decision it could change. The gross value is the same for two
# studies of the same sampling model, so the cheaper one ranks first.
FUTURE_VARIANCE = 1.0
actions = [
    {"id": "hold", "intercept": 0.0, "coefficients": [0.0, 0.0]},
    {"id": "act", "intercept": -BREAK_EVEN, "coefficients": COEFFICIENTS},
]
ranking = updated.rank_next_samples(
    actions=actions,
    candidates=[
        {
            "id": "cheap",
            "sample_size": 4,
            "observation_variance": FUTURE_VARIANCE,
            "cost_utility": 0.01,
        },
        {
            "id": "costly",
            "sample_size": 4,
            "observation_variance": FUTURE_VARIANCE,
            "cost_utility": 0.1,
        },
    ],
    ledger=ledger,
    replicates=64,
)["candidates"]
assert [c["id"] for c in ranking] == ["cheap", "costly"]
assert ranking[0]["gross_evsi"] == ranking[1]["gross_evsi"]
for candidate, cost in zip(ranking, (0.01, 0.1), strict=True):
    assert abs(candidate["net_value"] - (candidate["gross_evsi"] - cost)) < 1e-12


def study_value(sigma: float, cells: int = 1024) -> tuple[float, float]:
    """Value of one future observation with standard deviation ``sigma``, by quadrature.

    Each retained draw predicts the observation ``contrast`` (identity mapping). The value is the
    integral over the observation of ``max(0, sum_i w_i N(y; c_i, sigma^2) (c_i - break_even))``
    minus the value of deciding now. Returns the value and a bound on the quadrature error.
    """
    gains = [w * (c - BREAK_EVEN) for c, w in zip(contrasts, weights, strict=True)]
    lower, upper = min(contrasts) - 8 * sigma, max(contrasts) + 8 * sigma
    step = (upper - lower) / cells
    scale = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    future = 0.0
    for cell in range(cells):
        y = lower + (cell + 0.5) * step
        advantage = sum(
            g * scale * math.exp(-0.5 * ((y - c) / sigma) ** 2)
            for c, g in zip(contrasts, gains, strict=True)
        )
        future += step * max(advantage, 0.0)
    absolute = sum(abs(g) for g in gains)
    error = step / 2.0 * absolute * math.sqrt(2.0 / math.pi) / sigma
    error += absolute * 2.0 * math.exp(-32.0) / (8.0 * math.sqrt(2.0 * math.pi))
    return future - max(sum(gains), 0.0), error


# The library's Monte Carlo value agrees with independent numerical integration within the
# quadrature bound plus six Monte Carlo standard errors.
reference, quadrature_error = study_value(math.sqrt(FUTURE_VARIANCE / 4))
for candidate in ranking:
    tolerance = quadrature_error + 6.0 * candidate["standard_error"] + 1e-9
    assert abs(candidate["gross_evsi"] - reference) <= tolerance, candidate

# -- 6. The status of the update -------------------------------------------------------------
# The update reweights the retained draws. It re-derives no identification, so the result stays a
# model-conditional experimental update, with its calibration unmeasured.
assert report["scope"] == "model_conditional_experimental_update;calibration_unmeasured"
assert report["modular_policy"] == "frozen_joint_importance"
assert report["calibration"] == "unmeasured"
assert report["identification"]["target_kind"] == "whole_program_population_mean"
print(report["scope"])
print(report["identification"]["required_assumptions"])
assert ranking[0]["id"] == "cheap"
