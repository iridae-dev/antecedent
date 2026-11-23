# When outside events interrupt a launch

You’re launching a website. When a news outlet mentions it, more people visit. Others may pick up the story while traffic is still high. Should you pay to limit traffic during the first four hours?

The script simulates both choices against the same sequence of announcements. You’ll compare their peak loads, then ask whether checking for announcements before you decide is worth the cost.

Save the [complete example](#the-complete-example) as `event_driven_launch.py` and run `python event_driven_launch.py` with Antecedent installed. It generates its own data.

## Forecast the next four hours

`synthetic_history()` records announcements from hour 0 through hour 600. Give the model both ends of that window as well as the timestamps. If the last announcement came an hour before recording stopped, that quiet hour counts as an observation too.

The script fits a Hawkes model. It estimates how often announcements arrive on their own and how many follow-ups a single announcement generates on average. You specify how quickly its influence fades. In this run, the fitted background rate is about 0.42 announcements per hour, and each announcement generates about 0.38 direct follow-ups on average. The forecast starts at hour 600 and simulates 256 possible sequences of announcements over the next four hours.

## Compare actions under the same announcements

For each sequence, `worlds()` simulates the launch twice. In the baseline simulation, each announcement briefly increases attention, which drives load up. In the `hold` simulation, the service keeps load fixed at 0.25. Pairing the runs gives both actions the same announcements and background fluctuations.

`contract()` penalizes each action for its highest recorded load. In the 256 baseline runs, peak load averages about 3.16 points. Holding load incurs a penalty of just 0.75 points, including the 0.5-point fee. The simulation therefore favors limiting traffic.

You can also ask how often load stays at or below 0.8. The `value` for each action in the peak-limit report is the fraction of runs that meet that threshold. With this seed, every run with a hold meets the threshold and every unrestricted run exceeds it.

The simulation records load every 15 minutes and repeats the calculation at half that interval. Compare the results under `Paired half-step diagnostics`. If they differ substantially, the coarser grid may be missing peaks between readings.

## Decide what another hour of observation would buy

Now suppose you must choose at hour 601. You can pay 0.02 points to learn whether any announcement came in during the preceding hour. Is that information worth buying? Both options leave you with the same four-hour launch ahead of you.

An announcement just before hour 601 makes follow-ups more likely during the launch. The simulation carries that history forward. Read `baseline_action` for the choice you would make without checking. Compare it with the two `branch_actions`, one for a quiet hour and one for an hour with at least one announcement. `net_value` is the expected gain from choosing with that information after paying the 0.02-point fee.

With the supplied seed, the best action is `hold` before observing and after either possible observation. Learning whether an announcement arrived would leave the decision unchanged, so the gross gain is zero. After the observation fee, the net value is −0.02 points.

## Interpret the result

The timestamps help estimate when announcements will arrive. The simulated benefit of holding load depends on a separate assumption in `worlds()` about what the service can do. To use the comparison for a real launch, you would need evidence for that assumption and a cost that reflects what holding load would mean for your service.

## Where to go next

For rolling forecasts, alternative event sources, and more detail on numerical accuracy, see [event forecasts and stochastic decisions](../guides/event-processes.md).

## The complete example

<!-- example: examples/python/event_driven_launch.py -->

[Open the Python script](../../examples/python/event_driven_launch.py)

```python
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare launch actions under the same arrivals, then value an extra hour of data."""

from __future__ import annotations

import random
from collections.abc import Mapping
from math import log1p
from typing import Any

from antecedent import decision, events
from antecedent.joint_distribution import ScientificQuantity

CUTOFF = 600.0
PATHS = 256
BASE = "baseline"
HOLD = "hold"


def synthetic_history() -> dict[str, Any]:
    """Generate arrivals from independent clusters with known settings."""
    rng = random.Random(220)
    arrivals: list[float] = []
    time = rng.expovariate(0.4)
    while time < CUTOFF:
        arrivals.append(time)
        time += rng.expovariate(0.4)
    for parent in arrivals:
        mass = rng.expovariate(1.0)
        while mass < 0.35:
            child = parent - log1p(-mass / 0.35)
            if child < CUTOFF:
                arrivals.append(child)
            mass += rng.expovariate(1.0)
    return {
        "catalog": ["announcement"],
        "time_unit": "hour",
        "origin": "launch clock",
        "source": "synthetic announcements",
        "empty_origin": 0.0,
        "window": {"start": 0.0, "end": CUTOFF},
        "left": [],
        "events": [{"time": time, "kind": 0} for time in sorted(arrivals)],
    }


def worlds(start: float) -> dict[str, Any]:
    return {
        "system": {
            "variables": ["load"],
            "inputs": [
                {
                    "name": "attention",
                    "times": [start],
                    "values": [0.0],
                    "envelope": [0.0, 64.0],
                },
            ],
            "drift": [{"op": "variable", "variable": 1}],
            "diffusion": [{"expression": {"op": "constant", "value": 0.1}}],
        },
        "population": "launch",
        "horizon": 4,
        "time_unit": "hour",
        "origin": "launch clock",
        "policy": {
            "start": start,
            "step": 0.25,
            "steps": 16,
            "paths": PATHS,
            "scheme": "euler_maruyama",
            "seed": 108,
        },
        "initial": [0.0],
        "worlds": [
            {"id": HOLD, "actions": [{"variable": 0, "times": [start], "values": [0.25]}]},
        ],
        "bindings": [{"kind": 0, "input": "attention", "magnitude": 0.75, "decay": 1.0}],
        "outputs": [
            {
                "state": "load",
                "world": world,
                "variable_id": "launch.load",
                "units": "score",
                "functional": {"kind": "maximum"},
            }
            for world in (BASE, HOLD)
        ],
    }


def contract(*, peak_limit: float | None = None) -> decision.Contract:
    actions = []
    for world in (BASE, HOLD):
        peak = ScientificQuantity(
            variable_id="launch.load",
            variable_name="load",
            role="outcome",
            units="score",
            population_id="launch",
            regime_id=world,
            horizon=4,
            functional_id="grid_maximum",
        )
        # Holding load also incurs a business cost in utility units.
        utility = -decision.x(0)
        if peak_limit is None and world == HOLD:
            utility = utility - 0.5
        actions.append(decision.Action(world, inputs=(peak,), utility=utility))
    return decision.Contract(
        actions=actions,
        utility_units="score",
        criterion=(
            decision.Criterion.expected_utility()
            if peak_limit is None
            else decision.Criterion.threshold_probability(-peak_limit)
        ),
        target_population="launch",
        horizon=4,
    )


def main() -> None:
    history = synthetic_history()
    forecast_request = {
        "operation": "forecast",
        "request": {
            "history": history,
            "forecast": {"start": CUTOFF, "end": CUTOFF + 4.0},
            "source": {
                "kind": "fitted",
                "fit": {"decay": [1.0], "column_cap": 0.9, "iterations": 1600},
            },
            "paths": PATHS,
            "max_events": 2048,
            "seed": 403,
        },
    }
    prepared = events.prepare(forecast_request)
    plan = prepared.inspect()
    assert plan["route"] == "EventProcess"

    forecast = prepared.execute()
    fitted: Mapping[str, Any] = forecast.report()["fit"]
    print("Fitted event model", fitted["parameters"])
    response = events.prepare(
        {"operation": "worlds", "worlds": worlds(CUTOFF), "contract": contract()},
        source=forecast,
    ).execute()
    print("Decision under paired arrivals", response.report()["decision"])
    print("Paired half-step diagnostics", response.report()["grid"])
    peak_limit = 0.8
    risk_request = {
        "operation": "worlds",
        "worlds": worlds(CUTOFF),
        "contract": contract(peak_limit=peak_limit),
    }
    risk = events.prepare(risk_request, source=forecast).execute()
    print("Probability of staying at or below the peak limit", risk.report()["decision"])

    decision_time = CUTOFF + 1.0
    study = (
        events.prepare(
            {
                "operation": "study",
                "prior": {
                    "history": history,
                    "hypotheses": [{"weight": 1.0, "parameters": fitted["parameters"]}],
                },
                "request": {
                    "decision_time": decision_time,
                    "end": decision_time + 4.0,
                    "kind": 0,
                    "minimum_count": 1,
                    "paths": PATHS,
                    "max_events": 128,
                    "seed": 129,
                },
                "worlds": worlds(decision_time),
                "contract": contract(),
                "cost": {
                    "amount": 0.02,
                    "unit": "score",
                    "utility_unit": "score",
                    "utility_per_cost": 1.0,
                },
            }
        )
        .execute()
        .report()
    )
    no_notice, notice = study["signal_probabilities"]
    assert abs(no_notice + notice - 1.0) < 1e-12
    assert study["gross_value"] >= -1e-12
    print("Action without the signal", study["baseline_action"])
    print("Actions after no notice / a notice", study["branch_actions"])
    print("Gross and net value of observing", study["gross_value"], study["net_value"])


if __name__ == "__main__":
    main()
```
