#!/usr/bin/env python3
"""How much qualified pipeline does extra paid-search spend cause, when the data do not fix
the causal directions?

Data: 2,000 simulated weeks. Market demand raises both paid-search spend and qualified
pipeline, and each extra £1,000 of spend adds £1,500 of pipeline (true effect 1.5), so the raw
spend-pipeline slope is confounded.

The script discovers the equivalence class (PC orients no edge here), estimates the average
effect under a posterior over compatible DAGs, reports the posterior mass where the effect is
not identified, and compares that with a reviewed DAG and the naive slope. It prints the
estimates beside the simulated 1.5 and does not assert them. Each answer is conditional on the
declared or discovered structure; the posterior over graphs is not evidence that any one graph
is true.

Run: ``python examples/python/marketing_channel_structural_uncertainty.py``.
"""

from __future__ import annotations

# %%
from importlib.metadata import version

import antecedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from antecedent import AverageEffect, Bayesian, analyze
from antecedent.discovery import ExactDagPosterior
from antecedent.outcomes import require_success

print("antecedent", version("antecedent"))

# %% [markdown]
# ## Simulate weekly marketing operations
#
# The simulated business process is:
#
# $$
# \text{market demand}
# \rightarrow
# \text{paid-search spend}
# $$
#
# $$
# \text{market demand}
# \rightarrow
# \text{qualified pipeline}
# $$
#
# $$
# \text{paid-search spend}
# \rightarrow
# \text{qualified pipeline}
# $$
#
# Variables are expressed in practical units:
#
# - `market_demand_index`: a standardised demand indicator;
# - `paid_search_spend_kgbp`: weekly paid-search spend in **£ thousands**;
# - `qualified_pipeline_kgbp`: qualified pipeline created in **£ thousands**.
#
# The true incremental effect is **1.5**, meaning that an additional **£1,000**
# of paid-search spend generates **£1,500** of qualified pipeline on average.

# %%
SEED = 7
N_WEEKS = 2_000
TRUE_PIPELINE_PER_SPEND = 1.5

rng = np.random.default_rng(SEED)

market_demand_index = rng.normal(size=N_WEEKS)

paid_search_spend_kgbp = 20.0 + 4.0 * market_demand_index + 5.0 * rng.normal(size=N_WEEKS)

qualified_pipeline_kgbp = (
    80.0
    + TRUE_PIPELINE_PER_SPEND * paid_search_spend_kgbp
    + 12.0 * market_demand_index
    + 8.0 * rng.normal(size=N_WEEKS)
)

data = {
    "market_demand_index": market_demand_index,
    "paid_search_spend_kgbp": paid_search_spend_kgbp,
    "qualified_pipeline_kgbp": qualified_pipeline_kgbp,
}
frame = pd.DataFrame(data)

print(frame.head())

# %%
summary = frame.agg(["mean", "std", "min", "max"]).T
print(summary.round(2))

# %% [markdown]
# ## Why the raw marketing correlation is not the answer
#
# High-demand weeks receive more budget and would have produced more pipeline anyway.
# A regression or chart that ignores demand mixes the causal return on spend with this
# budget-allocation policy.

# %%
naive_slope = np.polyfit(
    frame["paid_search_spend_kgbp"],
    frame["qualified_pipeline_kgbp"],
    deg=1,
)[0]

print(f"Naive observational slope: £{naive_slope:,.2f}k pipeline per £1k spend")
print(f"True causal effect:         £{TRUE_PIPELINE_PER_SPEND:,.2f}k pipeline per £1k spend")

plt.figure(figsize=(7.5, 4.8))
scatter = plt.scatter(
    frame["paid_search_spend_kgbp"],
    frame["qualified_pipeline_kgbp"],
    c=frame["market_demand_index"],
    alpha=0.35,
    s=16,
)
plt.colorbar(scatter, label="Market demand index")
plt.xlabel("Paid-search spend (£k per week)")
plt.ylabel("Qualified pipeline (£k per week)")
plt.title("Spend and pipeline rise together in high-demand weeks")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Discover what the observational data support
#
# PC discovery tests conditional independences. In this linear-Gaussian example every pair
# of variables is dependent given the third, so it recovers the complete three-variable
# skeleton and cannot orient any edge.
#
# That is the correct result, not a failure: every DAG on a complete skeleton implies the
# same independences, so the observational distribution alone cannot select one causal story
# from the equivalence class.

# %%
pc = antecedent.discovery.PC(alpha=0.05).run(data)

print(
    "Edges oriented by PC:",
    f"{pc.cpdag_directed_edges} directed,",
    f"{pc.cpdag_undirected_edges} undirected",
)

# %% [markdown]
# With this seed, the expected result is **0 directed edges and
# 3 undirected edges**.

# %%
node_positions = {
    "Market demand": (0.0, 1.0),
    "Paid-search spend": (-1.0, 0.0),
    "Qualified pipeline": (1.0, 0.0),
}
undirected_edges = [
    ("Market demand", "Paid-search spend"),
    ("Market demand", "Qualified pipeline"),
    ("Paid-search spend", "Qualified pipeline"),
]

plt.figure(figsize=(7, 4.8))
for left, right in undirected_edges:
    x = [node_positions[left][0], node_positions[right][0]]
    y = [node_positions[left][1], node_positions[right][1]]
    plt.plot(x, y, linewidth=2)

for node, (x, y) in node_positions.items():
    plt.scatter([x], [y], s=2400)
    plt.text(x, y, node, ha="center", va="center", fontsize=10)

plt.title("The discovered CPDAG does not justify arrow directions")
plt.xlim(-1.6, 1.6)
plt.ylim(-0.4, 1.4)
plt.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ask the business question across compatible structures
#
# The query is the average effect of increasing weekly paid-search spend on qualified
# pipeline. `ExactDagPosterior` weights the compatible DAGs, and Antecedent checks
# identification in each one before estimating the effect there. Graphs in which the effect
# is not identified contribute no estimate; their weight is reported as unidentified mass.

# %%
result = require_success(
    analyze(
        data=data,
        discovery=ExactDagPosterior().run(data, execution=antecedent.Execution(seed=SEED)),
        query=AverageEffect(
            treatment="paid_search_spend_kgbp",
            outcome="qualified_pipeline_kgbp",
        ),
        inference=Bayesian(n_draws=200, backend="conjugate"),
        execution=antecedent.Execution(seed=SEED),
    )
)

posterior = result.posterior
assert posterior is not None

print(result)

# %% [markdown]
# ### Reuse the analysis and read its report
#
# Keep `result.study` to run the same analysis again. `antecedent.estimate(study)`
# runs the prepared branch; `antecedent.rebind(study, data=new_data)` returns a new branch
# on the new data and leaves the original and its earlier results unchanged.
#
# The report states the answer and its limitations. Read the calibration status too:
# `not_measured` means no coverage record measured this construction, so the interval is
# reported without a calibration claim. A passing diagnostic does not fill that gap.

# %%
study = result.study
report = result.inspect().to_dict()
print("Answer:", result.answer)
print("Calibration:", result.calibration.status, result.calibration.reason)
print(report)

# The study retains the graph posterior, so estimating again does not rerun discovery.
repeated = require_success(antecedent.estimate(study))
assert repeated.data_snapshot_id == result.data_snapshot_id

# %%
identification_summary = pd.DataFrame(
    {
        "structural region": [
            "Effect identified",
            "Effect unidentified",
        ],
        "posterior mass": [
            1.0 - posterior.unidentified_mass,
            posterior.unidentified_mass,
        ],
    }
)

print(identification_summary)

# %%
plt.figure(figsize=(7.5, 3.8))
plt.barh(
    identification_summary["structural region"],
    identification_summary["posterior mass"],
)
plt.xlim(0, 1)
plt.xlabel("Posterior mass")
plt.title("How much structural belief supports an identifiable ROI?")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare with the causal story the marketing team believes
#
# Subject-matter knowledge may justify the intended DAG: current weekly pipeline does
# not cause the budget that was already set, and an external demand index precedes both.
#
# Supplying that reviewed graph produces an estimate conditional on it. That is a different
# claim from saying that the graph was learned uniquely from the data.

# %%
reviewed_marketing_graph = [
    ("market_demand_index", "paid_search_spend_kgbp"),
    ("market_demand_index", "qualified_pipeline_kgbp"),
    ("paid_search_spend_kgbp", "qualified_pipeline_kgbp"),
]

reviewed_result = require_success(
    analyze(
        data=data,
        graph=reviewed_marketing_graph,
        query=AverageEffect(
            treatment="paid_search_spend_kgbp",
            outcome="qualified_pipeline_kgbp",
        ),
        execution=antecedent.Execution(seed=SEED),
    )
)

print(reviewed_result)

# %%
business_view = pd.DataFrame(
    [
        {
            "analysis": "Naive spend-pipeline slope",
            "effect_per_£1k_spend": naive_slope,
            "structural caveat": "Does not adjust for market demand",
        },
        {
            "analysis": "Reviewed marketing DAG",
            "effect_per_£1k_spend": reviewed_result.answer.value,
            "structural caveat": "Conditional on the supplied causal graph",
        },
        {
            "analysis": "Compatible-graph posterior",
            "effect_per_£1k_spend": posterior.effect_mean,
            "structural caveat": (
                f"{posterior.unidentified_mass:.0%} of structural posterior "
                "does not identify the effect"
            ),
        },
    ]
)

print(business_view.round(2))

# %% [markdown]
# ## Decision interpretation
#
# A marketing leader should not read the graph-posterior result as one conventional ROI
# estimate with a slightly wider error bar.
#
# There are two distinct uncertainties:
#
# - **effect uncertainty** among causal structures where the effect can be identified;
# - **identification uncertainty** from plausible structures where observational data
#   do not identify the effect at all.
#
# Steps that would reduce the second kind:
#
# 1. document time ordering and budget-setting policy to rule out implausible directions;
# 2. add pre-treatment demand measurements;
# 3. run a geo experiment or budget holdout;
# 4. update the analysis after those interventions reduce structural ambiguity.

# %% [markdown]
# ## Reproducibility information

# %%
import platform

print(
    {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "antecedent": version("antecedent"),
        "seed": SEED,
        "weeks": N_WEEKS,
        "true_pipeline_per_£1k_spend": TRUE_PIPELINE_PER_SPEND,
    }
)
