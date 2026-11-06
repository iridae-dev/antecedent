#!/usr/bin/env python3
"""Pricing, availability, and latent demand.

When inventory is limited, recorded sales are ``min(demand, inventory)``. A price response for
sales is therefore not automatically a price response for demand. This example makes the
observation mechanism and its identifying assumption explicit, estimates the supported Cox IPCW
demand curve, and keeps a true remaining refusal rather than substituting the observed proxy.
"""

from __future__ import annotations

# %%
import antecedent
import numpy as np
from antecedent import PointDerivative, ResponseCurve, analyze
from antecedent.observation import IndependentGiven, RightCensored
from antecedent.outcomes import require_success

SEED = 505
N = 2_000

# %% [markdown]
# ## Simulate demand and the sales recording process
#
# Season affects both price and latent demand. Conditional on price and season in this synthetic
# design, inventory is generated independently of the remaining demand shock. Inventory binds
# often enough to pull the recorded-sales slope away from demand, without driving fitted
# censoring survival through the positivity floor. That design detail motivates—but does not
# prove in real data—the declared independent-censoring assumption below.

# %%
rng = np.random.default_rng(SEED)
season_index = rng.normal(size=N)
price = np.clip(10.0 + 0.9 * season_index + rng.normal(scale=0.8, size=N), 7.0, 13.0)
latent_demand = np.maximum(
    0.0,
    150.0 - 8.0 * price + 18.0 * season_index + rng.normal(scale=8.0, size=N),
)
inventory = np.maximum(
    5.0,
    100.0 - 2.0 * price + 8.0 * season_index + rng.normal(scale=13.0, size=N),
)
observed_sales = np.minimum(latent_demand, inventory)
demand_observed = (latent_demand <= inventory).astype(float)

observed_data = {
    "season_index": season_index,
    "price": price,
    "inventory": inventory,
    "observed_sales": observed_sales,
    "demand_observed": demand_observed,
}
# fraction of inventory-censored rows
print(float(1.0 - demand_observed.mean()))

# %% [markdown]
# ## The tempting analysis answers the sales question
#
# This query can run through the complete-observation response estimator, but its outcome is
# explicitly `observed_sales`. Its slope mixes the demand response with inventory constraints.
# It must not be labelled a demand elasticity or demand derivative.

# %%
sales_graph = [
    ("season_index", "price"),
    ("season_index", "inventory"),
    ("season_index", "observed_sales"),
    ("price", "observed_sales"),
    ("inventory", "observed_sales"),
]
sales_query = PointDerivative("price", "observed_sales", at=10.0)
sales_response = require_success(
    analyze(
        observed_data,
        graph=sales_graph,
        query=sales_query,
        estimator_config={"bandwidth": 0.8},
    )
)

print(
    {
        "estimand": "local causal derivative of recorded sales",
        "estimate": sales_response.estimate,
        "support": sales_response.support.status,
        "uncertainty": sales_response.uncertainty.kind,
    }
)

# %% [markdown]
# ## State the scientific demand query
#
# `RightCensored` describes the recorded columns. `IndependentGiven` is a separate scientific
# assumption: knowing price and season, the censoring process carries no additional information
# about latent demand. Merely having `inventory` and `demand_observed` columns would not justify
# that claim.

# %%
sales_as_censored_demand = RightCensored(
    "latent_demand",
    "observed_sales",
    "inventory",
    "demand_observed",
)
independent_censoring = IndependentGiven(["price", "season_index"])

demand_query = ResponseCurve(
    "price",
    "latent_demand",
    grid=[9.0, 10.0, 11.0],
    observation=sales_as_censored_demand,
    observation_assumptions=[independent_censoring],
)
print(demand_query)

# %% [markdown]
# ## Estimate demand under the censoring assumption
#
# We assume that, after accounting for price and season, censoring provides no
# additional information about demand. `IndependentGiven(["price", "season_index"])`
# states that assumption and includes treatment and all adjustment variables.
#
# The curve uses Cox inverse-probability-of-censoring weights (IPCW). The Cox
# model assumes proportional hazards; fitting it does not test the independent
# censoring assumption. This path returns a point curve without a band that
# jointly accounts for censoring correction and curve estimation.
#
# The same setup cannot estimate a demand derivative: observation-adjusted
# `PointDerivative` queries remain unsupported.

# %%
demand_graph = [
    ("season_index", "price"),
    ("season_index", "latent_demand"),
    ("price", "latent_demand"),
]
demand = require_success(analyze(observed_data, graph=demand_graph, query=demand_query))
print(
    {
        "estimand": "price response of latent demand under Cox IPCW",
        "values": demand.response.values,
        "support": demand.support.status,
        "uncertainty": demand.uncertainty.kind,
    }
)

demand_slope = PointDerivative(
    "price",
    "latent_demand",
    at=10.0,
    observation=sales_as_censored_demand,
    observation_assumptions=[independent_censoring],
)
try:
    require_success(analyze(observed_data, graph=demand_graph, query=demand_slope))
except antecedent.CausalError as error:
    refusal = error.report.to_dict()
    assert refusal["message"]
    print("Expected refusal:", refusal["code"], refusal["message"])
else:
    raise AssertionError("observation-adjusted derivative unexpectedly bypassed its guard")

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
study = demand.study
report = demand.inspect().to_dict()
print("Answer:", demand.answer)
print("Calibration:", demand.calibration.status, demand.calibration.reason)
print(report)

# %% [markdown]
# ## Read the two answers separately
#
# The sales derivative describes **recorded sales**, which are limited by stock.
# The demand curve describes **demand under the stated censoring assumption**.
# It has no joint band for censoring correction and curve estimation.
#
# The demand derivative remains unsupported. The sales slope cannot stand in
# for it, and this analysis does not fill in missing demand values.
