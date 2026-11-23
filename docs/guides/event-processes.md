# Event forecasts and stochastic decisions

An event forecast represents arrivals by type on a named continuous clock. A compiler plan may obtain arrivals from a finite fixture, a verified external object, or a fixed-decay linear Hawkes model. The same retained path law can drive several intervention worlds. The SDE runs the worlds for the same arrival path and Brownian increments, so its joint output can answer decisions driven by pathwise risk or nonlinear utility.

Choose the observation window and record the complete history from an explicit empty origin. Native likelihood treats the forecast cutoff as the first instant outside the observed window. Gaps and ties in native timestamps need a separate observation model. Branching and decay arrays use target rows and source columns. Excitation describes predictive association. World effects come from the declared causal equations and bindings.

A dense catalog is limited by the likelihood build. One build charges `(events + 1) * types²` against the operation budget, and the catalog cap is the largest type count for which a build at the event cap uses at most half that budget. A forecast can still be refused when its fit or simulation exceeds the budget. The composer accepts as many intervention worlds as the SDE retains, 63 plus the baseline, and refuses when the grid's cell product exceeds its execution budget.

## Horizon and lifetime offspring

The branching entry `α` is the expected number of direct children over the whole future. On a finite horizon `H` the direct contribution of an exponential kernel is `α (1 - exp(-β H))`. A release at time `t` scored through a window that ends at `T` uses the remaining horizon `T - t`, so a later release excites less of that window.

Lifetime descendants, excluding the initiating event, are the matching column of `(I - A)^{-1} - I`. The resolvent itself counts the initiator. Descendants inside a finite horizon come from the linear filter ODE. They are not the resolvent of the horizon-discounted matrix `α (1 - exp(-β H))`, which would give every generation the full horizon.

`HawkesParameters::excitation_offspring` reports all four quantities as functionals of the conditional intensity.

For a complete fit, action comparison, and future-study example, start with [When outside events interrupt a launch](../tutorials/event-driven-launch.md).

## Python workflow

```python
from antecedent import events

history = {
    "catalog": ["announcement"],
    "time_unit": "hour",
    "origin": "launch clock",
    "source": "announcement log",
    "empty_origin": 0.0,
    "window": {"start": 0.0, "end": 12.0},
    "left": [],
    "events": [{"time": 11.5, "kind": 0}],
}
request = {
    "operation": "forecast",
    "request": {
        "history": history,
        "forecast": {"start": 12.0, "end": 16.0},
        "source": {
            "kind": "declared",
            "parameters": {"baseline": [0.2], "branching": [0.4], "decay": [0.5]},
        },
        "paths": 256,
        "max_events": 512,
        "seed": 403,
    },
}
plan = events.prepare(request)
print(plan.inspect())
forecast = plan.execute()
```

Pass that result as `source` to `events.prepare({"operation": "worlds", "worlds": world_request, "contract": contract_json}, source=forecast)`. The world request carries the existing SDE declaration, clock, grid, initial state and interventions, plus input pulses and output coordinates. Construct the decision inputs using those exact output coordinates. The returned joint distribution uses the existing distribution artifact codec. Rust exposes the same workflow as `Study::prepare_event_forecast` followed by `Study::prepare_event_worlds`.

On each solver evaluation the pulse from a previous arrival is its magnitude multiplied by its exponential decay since arrival. A path's maximum, trapezoidal integral and threshold indicator use all points on the declared grid. Check the paired half-step diagnostics before using a threshold near a sensitive boundary. The result describes grid-observed risk.

## Fit and future observations

Select `source.kind = "fitted"` with the fixed `decay`, `column_cap`, `iterations`, and optional `refits`. Inspect the fit and its conditioning before relying on it. A recursive-refit forecast repeats the fit on complete simulated training windows and propagates the resulting parameters. Count intervals are published only for a matching measured construction. A count interval keeps every count whose tie-randomized rank among the forecast paths lies in the central 90% band; one draw from the forecast seed settles the two boundary counts, so under a correct forecast law the interval covers 90% of future counts rather than at least 90%, and the same seed reproduces it. The time-ordered assessment reports held-out scores and reliability separately from release support.

A future study takes a finite weighted parameter belief and estimates the value of a specified count threshold in a new segment. Both alternatives choose at the same time. The observation segment changes the arrival process's state at the decision cutoff. For an actual observation, the compiled update instead reads the complete new segment and returns an updated belief and continuation law. Pass that law through the same world and decision contract.

## Compare two actions on the same arrival paths

This example uses the forecast above to compare an open launch with a policy that holds the load coordinate at 0.25. A smaller peak has higher utility. The initial load and the effect of the policy are model premises.

```python
from antecedent import decision
from antecedent.joint_distribution import ScientificQuantity

outputs = [
    {
        "state": "load",
        "world": name,
        "variable_id": "launch.load",
        "units": "score",
        "functional": {"kind": "maximum"},
    }
    for name in ("baseline", "hold")
]
world_request = {
    "system": {
        "variables": ["load"],
        "inputs": [{
            "name": "attention", "times": [12.0], "values": [0.0],
            "interpolation": "piecewise_constant", "envelope": [0.0, 64.0],
        }],
        "drift": [{"op": "variable", "variable": 1}],
        "diffusion": [{"expression": {"op": "constant", "value": 0.1}}],
    },
    "population": "launch",
    "horizon": 4,
    "time_unit": "hour",
    "origin": "launch clock",
    "policy": {
        "start": 12.0, "step": 0.25, "steps": 16, "paths": 256,
        "scheme": "euler_maruyama", "seed": 108,
    },
    "initial": [0.0],
    "worlds": [{
        "id": "hold",
        "actions": [{"variable": 0, "times": [12.0], "values": [0.25]}],
    }],
    "bindings": [{
        "kind": 0, "input": "attention", "magnitude": 0.75, "decay": 0.5,
    }],
    "outputs": outputs,
}
actions = [
    decision.Action(
        name,
        inputs=(ScientificQuantity(
            variable_id="launch.load", variable_name="load", role="outcome",
            units="score", population_id="launch", regime_id=name, horizon=4,
            functional_id="grid_maximum", transform_id="identity",
        ),),
        utility=-decision.x(0),
    )
    for name in ("baseline", "hold")
]
contract = decision.Contract(
    actions=actions,
    utility_units="score",
    criterion=decision.Criterion.expected_utility(),
    target_population="launch",
    horizon=4,
)
response = events.prepare(
    {"operation": "worlds", "worlds": world_request, "contract": contract},
    source=forecast,
).execute()
print(response.report()["decision"])
```
