#!/usr/bin/env python3
"""Synthetic control, synthetic difference-in-differences and staggered adoption on one panel.

The question: what did a rollout do to units that adopted it, when nothing randomized who
adopted and when? The data is one simulated panel of 72 units over 14 periods. Outcomes follow
unit levels, a common period path and a unit-specific loading on a smooth trend, plus noise.
24 units never adopt, 24 adopt in period 5 and 24 adopt in period 9 (one of those, `tr`, is the
single treated unit used for the synthetic-control designs). The effect of adoption is known
and grows with time since adoption: base 2 (cohort 5) or 3 (cohort 9) plus 0.25 per period.

The assertions compare each estimate with the effect the simulation applied: the synthetic
control and synthetic difference-in-differences effects for `tr` against its mean applied
effect, and every cohort-by-event-time contrast of the staggered event study against its applied
effect (and the pre-adoption leads against zero), each within a stated tolerance. Two designs
are refused with their reasons.

Every result here is an unsupported point utility; identification rests on assumptions the
data cannot confirm (no anticipation, comparable donors or parallel trends, no interference).
A close pre-adoption fit or flat leads do not verify them, and nothing here claims coverage of
any interval."""

from __future__ import annotations

# %%
import numpy as np
from antecedent import analyze
from antecedent.errors import CausalError
from antecedent.quasi import StaggeredAdoption, SyntheticControl, SyntheticDifferenceInDifferences

SEED = 20260611
PERIODS = list(range(1, 15))
COHORTS = {0: 24, 5: 24, 9: 24}  # cohort -> number of units; cohort 0 never adopts
BASE = {5: 2.0, 9: 3.0}
SLOPE = 0.25
NOISE = 0.3
TOL_SINGLE = 0.7  # tolerance for the single-treated-unit designs
TOL_CELL = 0.7  # tolerance for each staggered cohort-by-event-time contrast


def applied_effect(cohort: int, period: int) -> float:
    """The effect the simulation adds to a unit of `cohort` in `period`."""
    if cohort == 0 or period < cohort:
        return 0.0
    return BASE[cohort] + SLOPE * (period - cohort)


def refused(call) -> CausalError:
    """Run `call` and return the refusal it raises; anything else is a failure."""
    try:
        call()
    except CausalError as error:
        return error
    raise AssertionError("the call should have been refused")


def show_refusal(label: str, error: CausalError) -> None:
    code = getattr(error, "reason_code", None)
    first = (str(error).strip().splitlines() or [""])[0][:160]
    print(f"{label}: {type(error).__name__} " + (f"reason_code={code}" if code else f"- {first}"))


# %% [markdown]
# ## Simulate the panel
#
# Outcome = unit level + common period path + loading x trend + applied effect + noise. The
# single treated unit `tr` has level 0 and loading 0.5, inside the range of the 24 never-treated
# donors, so a weighted mix of donors can imitate it before adoption.

# %%
rng = np.random.default_rng(SEED)
rows: dict[str, list] = {"y": [], "unit": [], "period": [], "cohort": []}
applied: dict[tuple[str, int], float] = {}
for cohort, count in COHORTS.items():
    for index in range(count):
        unit = "tr" if (cohort, index) == (9, 0) else f"u{cohort}_{index}"
        level, loading = (0.0, 0.5) if unit == "tr" else (rng.normal(scale=2.0), rng.uniform())
        for period in PERIODS:
            effect = applied_effect(cohort, period)
            common = 1.5 * np.sin(period / 2.0)
            rows["y"].append(
                level + common + loading * 0.15 * period + effect + NOISE * rng.normal()
            )
            rows["unit"].append(unit)
            rows["period"].append(period)
            rows["cohort"].append(cohort)
            applied[(unit, period)] = effect

n_units = sum(COHORTS.values())
print(f"{n_units} units x {len(PERIODS)} periods = {len(rows['y'])} rows")

# The single-treated-unit designs need one treated unit and untreated donors: keep `tr` and the
# never-treated units, so no donor is itself treated.
keep = [i for i, (u, c) in enumerate(zip(rows["unit"], rows["cohort"])) if c == 0 or u == "tr"]
single = {key: [values[i] for i in keep] for key, values in rows.items()}
true_single = float(np.mean([applied[("tr", p)] for p in range(9, 15)]))
print(f"applied mean post-adoption effect on tr (periods 9-14): {true_single:.3f}")

# %% [markdown]
# ## Synthetic control
#
# One treated unit is compared with a convex mix of donors whose weights are fitted to its
# pre-adoption path. The estimate is the treated unit's mean post-period outcome minus the
# weighted donors'. Check the pre-fit error and how many donors carry weight.

# %%
sc_result = analyze(single, query=SyntheticControl("y", "unit", "period", "tr", 9))
sc = sc_result.value.synthetic_control
print(f"effect {sc.estimate:.3f}  (applied {true_single:.3f})")
print(f"pre-adoption RMSE {sc.pre_treatment_rmse:.3f}, effective donors {sc.effective_donors:.1f}")
print("uncertainty:", sc.uncertainty, "| support:", sc.support_status)
print("calibration:", sc_result.inspect().calibration.status)
print(
    f"placebo tail fraction {sc.placebo_rank_p_value:.3f} (exchangeability assumed; "
    "not an interval or a calibrated test)"
)

weights = dict(sc.donor_weights)
assert abs(sum(weights.values()) - 1.0) < 1e-6 and min(weights.values()) >= -1e-9
assert sc.n_donors == COHORTS[0] and sc.n_pre_periods == 8 and sc.n_post_periods == 6
assert abs(sc.estimate - true_single) < TOL_SINGLE, sc.estimate

# %% [markdown]
# ## Synthetic difference-in-differences
#
# This design weights both donor units and pre-periods, then takes the weighted
# difference-in-differences contrast. It reports a point estimate and both weight vectors, and
# no interval.

# %%
sdid_result = analyze(
    single, query=SyntheticDifferenceInDifferences("y", "unit", "period", "tr", 9)
)
sdid = sdid_result.value.synthetic_did
print(f"effect {sdid.estimate:.3f}  (applied {true_single:.3f})")
print("uncertainty:", sdid.uncertainty, "| calibration:", sdid_result.inspect().calibration.status)
print("heaviest time weights:", sorted(sdid.time_weights, key=lambda w: -w[1])[:3])

assert abs(sum(w for _, w in sdid.donor_weights) - 1.0) < 1e-6
assert abs(sum(w for _, w in sdid.time_weights) - 1.0) < 1e-6
assert abs(sdid.estimate - true_single) < TOL_SINGLE, sdid.estimate

# %% [markdown]
# ## Staggered adoption event study
#
# With units adopting in different periods, an event study reports one contrast per cohort and
# event time against the never-treated units, with period `g - 1` as each cohort's reference.
# Pre-adoption contrasts are descriptive leads, not a test of parallel trends.

# %%
event_result = analyze(
    rows, query=StaggeredAdoption("y", "unit", "period", "cohort", event_study=True)
)
event = event_result.value.panel_did
print("calibration:", event_result.inspect().calibration.status, "|", event.uncertainty)

post_errors, lead_errors = [], []
for cell in event.effects:
    truth = applied_effect(cell.cohort, cell.period)
    (post_errors if cell.event_time >= 0 else lead_errors).append(abs(cell.estimate - truth))
for cohort in (5, 9):
    shown = [c for c in event.effects if c.cohort == cohort and c.event_time in (0, 2, 4)]
    for cell in shown:
        print(
            f"cohort {cohort} event time {cell.event_time}: "
            f"{cell.estimate:.2f} (applied {applied_effect(cohort, cell.period):.2f})"
        )
print(
    f"max |error|: post-adoption {max(post_errors):.2f} over {len(post_errors)} cells, "
    f"leads {max(lead_errors):.2f} over {len(lead_errors)} cells"
)

assert len(post_errors) == 10 + 6 and lead_errors
assert max(post_errors) < TOL_CELL and max(lead_errors) < TOL_CELL
# Timing matters: the cohort-5 effect keeps growing, so late cells differ a lot from early ones.
late = next(c for c in event.effects if (c.cohort, c.event_time) == (5, 9))
early = next(c for c in event.effects if (c.cohort, c.event_time) == (5, 0))
assert late.estimate - early.estimate > 1.5

# %% [markdown]
# ## Where the designs refuse
#
# A synthetic control needs at least two pre-adoption periods, and the staggered design needs
# never-treated units as its comparison group. Each refusal is raised with its reason.

# %%
too_short = refused(lambda: analyze(single, query=SyntheticControl("y", "unit", "period", "tr", 1)))
show_refusal("adoption in the first period", too_short)

no_controls = {
    key: [v for v, c in zip(values, rows["cohort"]) if c != 0] for key, values in rows.items()
}
no_never = refused(
    lambda: analyze(
        no_controls, query=StaggeredAdoption("y", "unit", "period", "cohort", event_study=True)
    )
)
show_refusal("no never-treated units", no_never)
assert "pre-period" in str(too_short).lower()
assert "never-treated" in str(no_never).lower()
