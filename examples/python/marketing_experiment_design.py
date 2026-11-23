#!/usr/bin/env python3
"""Which evidence-gathering action most raises the chance that an outreach effect is identified?

A sales team wants the effect of personalised outreach on qualified pipeline. Its CRM data are
observational, and unmeasured purchase intent may drive both outreach and pipeline. Four
candidate actions (buy an intent signal, run a randomised holdout, add a territory, wait for
more accounts) are ranked by gain in identified posterior mass over four declared graph
hypotheses, then again under a £40,000 budget.

Data: the graph weights and what each action resolves are declared planning assumptions, not
inferred. The chosen action is then simulated on 1,200 accounts with a true effect of £12,000
of pipeline per account. The script prints the ranking and the adjusted estimate beside that
truth and asserts only that a separately prepared study gives the same estimate. The ranking
recommends and runs nothing, and it is only as good as the declared assumptions.

Run: ``python examples/python/marketing_experiment_design.py``.
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
# ## The causal question
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
# The team's current structural belief is a weighted set of four causal hypotheses,
# declared here by hand. Only one identifies the effect using the variables already present
# in the CRM.

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
# ## Candidate research actions
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
# ## Encode what each action can resolve
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
# ## Rank actions by identification gain
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
# ## Add a cost-efficiency view
#
# The ranking above scores identification gain, not pounds spent. The caller can combine
# that score with its own costs after ranking.
#
# The ratio below is identification-probability gain per £1,000. It is computed here in
# pandas for reporting; it does not change the ranking's objective.

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
# ## Apply the £40,000 budget
#
# Hard constraints are applied before scoring. A candidate that violates one is listed
# under `violations` with the constraint it broke, not dropped.

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
# experiment or rerun an analysis.

# %% [markdown]
# ## Simulate the recommended action
#
# Suppose the intent signal is bought. The account-level CRM data are simulated from this
# process:
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
# Keep `measured_result.study` to run the same analysis again. `antecedent.estimate(study)`
# runs the prepared branch; `antecedent.rebind(study, data=new_data)` returns a new branch
# on the new data and leaves the original and its earlier results unchanged.
#
# The report states the answer and its limitations. Read the calibration status too:
# `not_measured` means no coverage record measured this construction, so the interval is
# reported without a calibration claim. A passing diagnostic does not fill that gap.

# %%
study = measured_result.study
report = measured_result.inspect().to_dict()
print("Answer:", measured_result.answer)
print("Calibration:", measured_result.calibration.status, measured_result.calibration.reason)
print(report)

# `prepare` checks identification without estimating, so the estimate can run later; it
# gives the same answer as the one-call analysis.
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
# ## What the example demonstrates
#
# The ranking separates four questions that are often conflated:
#
# - **Will this action change identification?**
# - **Will it only improve precision?**
# - **Is it feasible under the declared resource constraints?**
# - **Which feasible action should the analyst recommend?**
#
# In this scenario:
#
# - a randomised holdout and a missing-confounder measurement both resolve identification;
# - adding more observational accounts does not;
# - the holdout exceeds the available budget;
# - measuring account intent is therefore the best feasible next step;
# - after that measurement, the adjusted estimate is close to the simulated campaign effect
#   and the naive contrast is not (compare the printed table; neither is asserted).
#
# Review the planning assumptions as carefully as the causal graph. Register a variable as
# an identification unlock only when the scientific argument supports that claim.

# %% [markdown]
# ## Extensions
#
# Extensions of this analysis could:
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
