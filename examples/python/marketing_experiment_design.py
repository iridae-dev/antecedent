#!/usr/bin/env python3
"""Which marketing experiment should we run next?

Ranking evidence-gathering actions by identification gain.

A B2B growth team wants to estimate the causal effect of personalised outbound
outreach on qualified sales pipeline.

Its current CRM data are observational. The main uncertainty is whether not_measured
account purchase intent drives both:

- which accounts receive personalised outreach; and
- which accounts create pipeline.

The team has several ways to spend its research budget:

1. purchase a better account-intent signal;
2. run a randomised outreach holdout;
3. collect data in a new territory;
4. simply wait for more accounts.

This example uses Antecedent to rank those actions by how much they increase the
probability that the outreach effect is identifiable. It then applies a £40,000
budget constraint and simulates the recommended action.

The scenario and data are synthetic, but they represent a realistic revenue-operations
decision. The causal truth is known so the result can be checked.
"""

from __future__ import annotations

# %%
from importlib.metadata import version

import antecedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from antecedent import AverageEffect, analyze
from antecedent.outcomes import require_success

print("antecedent", version("antecedent"))

# %% [markdown]
# ## 2. The causal question
#
# The target estimand is:
#
# $$
# \text{ATE}
# =
# E[
#   \text{pipeline}\mid do(\text{personalised outreach}=1)
# ]
# -
# E[
#   \text{pipeline}\mid do(\text{personalised outreach}=0)
# ]
# $$
#
# Pipeline is measured in **£ thousands per account**.
#
# The current structural posterior contains four material causal hypotheses. Only one
# currently identifies the effect using the variables already present in the CRM.

# %%
graph_hypotheses = pd.DataFrame(
    [
        {
            "graph_key": 101,
            "hypothesis": "Purchase intent confounds outreach and pipeline",
            "posterior_weight": 0.40,
            "effect_identified_now": False,
        },
        {
            "graph_key": 102,
            "hypothesis": "Intent and account size jointly drive assignment",
            "posterior_weight": 0.20,
            "effect_identified_now": False,
        },
        {
            "graph_key": 103,
            "hypothesis": "Observed account size is a sufficient adjustment set",
            "posterior_weight": 0.25,
            "effect_identified_now": True,
        },
        {
            "graph_key": 104,
            "hypothesis": "Buying-stage intent drives outreach and pipeline",
            "posterior_weight": 0.15,
            "effect_identified_now": False,
        },
    ]
)

print(graph_hypotheses)

# %%
baseline_identification_probability = float(
    graph_hypotheses.loc[
        graph_hypotheses["effect_identified_now"],
        "posterior_weight",
    ].sum()
)

print(
    "Current probability that the outreach effect is identifiable:",
    f"{baseline_identification_probability:.0%}",
)

# %%
plt.figure(figsize=(8.5, 4.8))
plt.barh(
    graph_hypotheses["hypothesis"],
    graph_hypotheses["posterior_weight"],
)
plt.xlabel("Posterior weight")
plt.title("Current uncertainty over the marketing causal structure")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Candidate research actions
#
# Antecedent's design API uses stable integer IDs for registered variables, queries,
# environments and graph hypotheses.
#
# In this planning model:
#
# - variable `0` is personalised outreach;
# - variable `3` is the currently unavailable purchase-intent measure;
# - query `0` is the outreach-to-pipeline ATE;
# - environment `7` is a proposed new sales territory.
#
# The declared cost is a caller-owned scalar. Here it is expressed directly in pounds.

# %%
design = antecedent.design

candidates = [
    design.Experiment([0], cost=60_000.0, sample_budget=800, tag=100),
    design.Measurement([3], cost=25_000.0, sample_budget=0, tag=200),
    design.Environment(7, additional_rows=300, cost=35_000.0, sample_budget=300, tag=300),
    design.Sampling(500, cost=15_000.0, sample_budget=500, tag=400),
]

candidate_catalogue = pd.DataFrame(
    [
        {
            "candidate_index": 0,
            "action": "Randomised outreach holdout",
            "kind": "intervene",
            "cost_gbp": 60_000,
            "business interpretation": (
                "Randomise personalised outreach so assignment no longer depends on account intent"
            ),
        },
        {
            "candidate_index": 1,
            "action": "Purchase account-intent measurement",
            "kind": "measure",
            "cost_gbp": 25_000,
            "business interpretation": (
                "Measure the missing pre-treatment variable required by "
                "the current identification analysis"
            ),
        },
        {
            "candidate_index": 2,
            "action": "Add a new sales territory",
            "kind": "observe_environment",
            "cost_gbp": 35_000,
            "business interpretation": ("Observe 300 more accounts in a different territory"),
        },
        {
            "candidate_index": 3,
            "action": "Wait for 500 more CRM accounts",
            "kind": "increase_sampling_rate",
            "cost_gbp": 15_000,
            "business interpretation": (
                "Increase sample size without changing what is measured or how outreach is assigned"
            ),
        },
    ]
)

print(candidate_catalogue)

# %% [markdown]
# ## 4. Encode what each action can resolve
#
# The planning assumptions are explicit:
#
# - measuring purchase intent unlocks the registered query;
# - randomising outreach identifies the effect under every current graph hypothesis;
# - another territory does not, by itself, remove the missing-confounder problem;
# - more rows of the same observational data improve precision but do not create
#   identification.
#
# These are design assumptions supplied to the ranker. Antecedent does not infer them
# from the action labels.

# %%
structure_prior = design.StructurePrior(
    weights=tuple(graph_hypotheses["posterior_weight"]),
    identified=tuple(graph_hypotheses["effect_identified_now"]),
    keys=tuple(int(k) for k in graph_hypotheses["graph_key"]),
    # A randomised outreach intervention identifies the effect under every
    # currently weighted graph hypothesis.
    identified_under_intervention=(True, True, True, True),
)

query_id = 0
purchase_intent_variable_id = 3

# A measured purchase-intent variable is sufficient to unlock query 0.
variable_unlocks = {query_id: [purchase_intent_variable_id]}

# %% [markdown]
# ## 5. Rank actions by identification gain
#
# With no decision declared, `rank_designs` ranks plans on the `"identification"` basis: the
# score is the increase over the current identified posterior mass. A score of `0.75`, for
# example, moves the analysis from 25% identified mass to 100%.
#
# Higher scores are better.

# %%
monte_carlo = design.MonteCarlo(
    min_batches=2, max_batches=8, batch_size=8, rank_uncertainty_threshold=0.01
)

unconstrained_ranking = design.rank_designs(
    candidates,
    prior=structure_prior,
    query_id=query_id,
    variable_unlocks=variable_unlocks,
    monte_carlo=monte_carlo,
    rng_seed=11,
)

print("Ranking basis:", unconstrained_ranking.basis)
print(
    "Monte Carlo samples:",
    unconstrained_ranking.mc_samples,
    "| early stopped:",
    unconstrained_ranking.early_stopped,
)

# %%
unconstrained_rows = []

for ranked in unconstrained_ranking.candidates:
    details = candidate_catalogue.loc[candidate_catalogue["candidate_index"] == ranked.index].iloc[
        0
    ]

    post_identification_probability = min(
        1.0,
        baseline_identification_probability + ranked.score,
    )

    unconstrained_rows.append(
        {
            "rank": ranked.rank + 1,
            "candidate_index": ranked.index,
            "action": details["action"],
            "kind": ranked.kind,
            "identification_gain": ranked.score,
            "identification_probability_after": (post_identification_probability),
            "cost_gbp": details["cost_gbp"],
            "score_stderr": ranked.stderr,
            "rank_uncertain": ranked.rank_uncertain,
        }
    )

unconstrained_table = pd.DataFrame(unconstrained_rows)
print(unconstrained_table)

# %%
plt.figure(figsize=(8.5, 4.8))
plt.barh(
    unconstrained_table["action"],
    unconstrained_table["identification_gain"],
)
plt.xlabel("Increase in identification probability")
plt.title("Which actions address the identification problem?")
plt.tight_layout()
plt.show()

# %% [markdown]
# The randomised holdout and the intent measurement can both resolve the current
# identification problem under the declared planning assumptions.
#
# Collecting more of the same observational data receives no identification gain.
# This does **not** mean more data are worthless: they can improve precision after the
# estimand is identified. It means sample size alone cannot repair a missing causal
# assumption or an unmeasured confounder.

# %% [markdown]
# ## 6. Add a transparent cost-efficiency view
#
# The design objective above optimises identification gain, not pounds spent. We can
# combine the engine's score with business-owned costs after ranking.
#
# This ratio is deliberately simple: identification-probability gain per £1,000.
# It is a reporting calculation, not a hidden modification of Antecedent's objective.

# %%
unconstrained_table["gain_per_£1k"] = np.where(
    unconstrained_table["cost_gbp"] > 0,
    unconstrained_table["identification_gain"] / (unconstrained_table["cost_gbp"] / 1_000),
    np.nan,
)

cost_efficiency = unconstrained_table.sort_values(
    ["gain_per_£1k", "identification_gain"],
    ascending=False,
).reset_index(drop=True)

print(
    cost_efficiency[
        [
            "action",
            "identification_gain",
            "cost_gbp",
            "gain_per_£1k",
        ]
    ]
)

# %%
plt.figure(figsize=(8.5, 4.8))
plt.scatter(
    unconstrained_table["cost_gbp"],
    unconstrained_table["identification_gain"],
    s=90,
)

for _, row in unconstrained_table.iterrows():
    plt.annotate(
        row["action"],
        (row["cost_gbp"], row["identification_gain"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("Declared cost (£)")
plt.ylabel("Increase in identification probability")
plt.title("Identification gain versus research cost")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Apply the £40,000 budget
#
# Hard constraints are applied before scoring. Candidates that violate them are recorded
# explicitly rather than silently disappearing.

# %%
budget_gbp = 40_000.0

constrained_ranking = design.rank_designs(
    candidates,
    prior=structure_prior,
    query_id=query_id,
    variable_unlocks=variable_unlocks,
    max_cost=budget_gbp,
    monte_carlo=monte_carlo,
    rng_seed=11,
)

constrained_rows = []

for ranked in constrained_ranking.candidates:
    details = candidate_catalogue.loc[candidate_catalogue["candidate_index"] == ranked.index].iloc[
        0
    ]

    constrained_rows.append(
        {
            "rank": ranked.rank + 1,
            "candidate_index": ranked.index,
            "action": details["action"],
            "cost_gbp": details["cost_gbp"],
            "identification_gain": ranked.score,
            "identification_probability_after": min(
                1.0,
                baseline_identification_probability + ranked.score,
            ),
        }
    )

constrained_table = pd.DataFrame(constrained_rows)
print(constrained_table)

# %%
violations = pd.DataFrame(
    [
        {
            "candidate_index": violation.index,
            "action": candidate_catalogue.loc[
                candidate_catalogue["candidate_index"] == violation.index,
                "action",
            ].iloc[0],
            "constraint": violation.constraint,
            "detail": violation.detail,
        }
        for violation in constrained_ranking.violations
    ]
)

print(violations)

# %%
selected_index = constrained_ranking.best.index
selected_action = candidate_catalogue.loc[
    candidate_catalogue["candidate_index"] == selected_index,
    "action",
].iloc[0]

print("Recommended action under the budget:", selected_action)

# %% [markdown]
# Under the £40,000 constraint, the randomised holdout is infeasible. Measuring account
# intent is the highest-ranked feasible action because it directly addresses the current
# identification gap.
#
# The recommendation is advisory. The library does not purchase data, launch an
# experiment, or automatically rerun an analysis.

# %% [markdown]
# ## 8. Simulate the recommended action
#
# We now simulate account-level CRM data.
#
# The true data-generating process is:
#
# $$
# \text{account size} \rightarrow \text{outreach}
# $$
#
# $$
# \text{account size} \rightarrow \text{pipeline}
# $$
#
# $$
# \text{purchase intent} \rightarrow \text{outreach}
# $$
#
# $$
# \text{purchase intent} \rightarrow \text{pipeline}
# $$
#
# $$
# \text{outreach} \rightarrow \text{pipeline}
# $$
#
# The true incremental effect of personalised outreach is **£12,000 of qualified
# pipeline per account**.

# %%
DATA_SEED = 31
N_ACCOUNTS = 1_200
TRUE_EFFECT_KGBP = 12.0

rng = np.random.default_rng(DATA_SEED)

account_size_score = rng.normal(size=N_ACCOUNTS)
purchase_intent_score = rng.normal(size=N_ACCOUNTS)

assignment_score = (
    0.8 * account_size_score + 1.1 * purchase_intent_score + rng.normal(size=N_ACCOUNTS)
)
personalised_outreach = (assignment_score > 0).astype(float)

qualified_pipeline_kgbp = (
    50.0
    + TRUE_EFFECT_KGBP * personalised_outreach
    + 9.0 * account_size_score
    + 14.0 * purchase_intent_score
    + 5.0 * rng.normal(size=N_ACCOUNTS)
)

crm = pd.DataFrame(
    {
        "account_size_score": account_size_score,
        "purchase_intent_score": purchase_intent_score,
        "personalised_outreach": personalised_outreach,
        "qualified_pipeline_kgbp": qualified_pipeline_kgbp,
    }
)

print(crm.head())

# %% [markdown]
# Before purchasing the intent signal, the CRM contains account size, outreach and
# pipeline. The raw difference between contacted and uncontacted accounts is biased
# because high-intent accounts are more likely to receive the campaign.

# %%
naive_difference = (
    crm.loc[
        crm["personalised_outreach"] == 1,
        "qualified_pipeline_kgbp",
    ].mean()
    - crm.loc[
        crm["personalised_outreach"] == 0,
        "qualified_pipeline_kgbp",
    ].mean()
)

print(
    "Naive treated-minus-untreated difference:",
    f"£{naive_difference:,.2f}k per account",
)
print(
    "True causal effect:",
    f"£{TRUE_EFFECT_KGBP:,.2f}k per account",
)

# %% [markdown]
# After the intent measurement is available, the reviewed graph contains an observed
# backdoor adjustment set: account size and purchase intent.

# %%
measured_data = {
    "account_size_score": crm["account_size_score"].to_numpy(),
    "purchase_intent_score": crm["purchase_intent_score"].to_numpy(),
    "personalised_outreach": crm["personalised_outreach"].to_numpy(),
    "qualified_pipeline_kgbp": crm["qualified_pipeline_kgbp"].to_numpy(),
}

reviewed_graph = [
    ("account_size_score", "personalised_outreach"),
    ("account_size_score", "qualified_pipeline_kgbp"),
    ("purchase_intent_score", "personalised_outreach"),
    ("purchase_intent_score", "qualified_pipeline_kgbp"),
    ("personalised_outreach", "qualified_pipeline_kgbp"),
]

measured_result = require_success(
    analyze(
        data=measured_data,
        graph=reviewed_graph,
        query=AverageEffect(
            treatment="personalised_outreach",
            outcome="qualified_pipeline_kgbp",
        ),
        execution=antecedent.Execution(seed=DATA_SEED),
    )
)

print(measured_result)

# %% [markdown]
# ### Reuse the analysis and read its report
#
# Keep `result.study` to run the same analysis again. `ant.estimate(study)` runs
# the prepared branch; `ant.rebind(study, data=new_data)` returns a new branch on the
# new data without changing the original. Earlier results stay unchanged.
#
# The report explains the answer and its limitations. Check calibration too:
# `not_measured` means no coverage record measured this construction; the interval is
# reported without a calibration claim.
# A passing diagnostic does not fill that gap.

# %%
study = measured_result.study
report = measured_result.inspect().to_dict()
print("Answer:", measured_result.answer)
print("Calibration:", measured_result.calibration.status, measured_result.calibration.reason)
print(report)

# A fresh preparation is optional when estimation should happen later.
prepared = antecedent.prepare(
    measured_data,
    graph=reviewed_graph,
    query=AverageEffect("personalised_outreach", "qualified_pipeline_kgbp"),
    execution=antecedent.Execution(seed=DATA_SEED),
)
assert prepared.inspect().identification.available
assert np.isclose(antecedent.estimate(prepared).answer.value, measured_result.answer.value)

# %%
outcome_comparison = pd.DataFrame(
    [
        {
            "quantity": "Naive observational contrast",
            "effect_kgbp_per_account": naive_difference,
        },
        {
            "quantity": "Adjusted effect after measuring intent",
            "effect_kgbp_per_account": measured_result.answer.value,
        },
        {
            "quantity": "True simulated causal effect",
            "effect_kgbp_per_account": TRUE_EFFECT_KGBP,
        },
    ]
)

print(outcome_comparison.round(2))

# %%
plt.figure(figsize=(8.5, 4.6))
plt.bar(
    outcome_comparison["quantity"],
    outcome_comparison["effect_kgbp_per_account"],
)
plt.ylabel("Qualified pipeline (£k per account)")
plt.title("The recommended measurement enables an adjusted causal estimate")
plt.xticks(rotation=12, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. What the example demonstrates
#
# The design ranker distinguishes four questions that are often conflated:
#
# - **Will this action change identification?**
# - **Will it only improve precision?**
# - **Is it feasible under the declared resource constraints?**
# - **Which feasible action should the analyst recommend?**
#
# In this scenario:
#
# - a randomised holdout and a missing-confounder measurement both resolve identification;
# - simply adding more observational accounts does not;
# - the holdout exceeds the available budget;
# - measuring account intent is therefore the best feasible next step;
# - after that measurement, an adjusted analysis recovers the simulated campaign effect.
#
# The planning assumptions should be reviewed just as carefully as the causal graph.
# A variable should only be registered as an identification unlock when the scientific
# argument supports that claim.

# %% [markdown]
# ## 10. Extensions
#
# Useful next versions of this analysis could:
#
# - rank by expected graph-entropy reduction rather than identification probability;
# - rank by expected effect-width reduction after identification is established;
# - compare decision utility from deploying, pausing or targeting the campaign;
# - use a richer graph posterior produced by a discovery workflow;
# - add time-to-result, implementation risk and vendor lock-in to the business layer.

# %%
import platform

print(
    {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "antecedent": version("antecedent"),
        "design_seed": 11,
        "data_seed": DATA_SEED,
        "accounts": N_ACCOUNTS,
        "budget_gbp": budget_gbp,
        "true_effect_kgbp_per_account": TRUE_EFFECT_KGBP,
    }
)
