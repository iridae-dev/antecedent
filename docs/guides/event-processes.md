# Event forecasts and stochastic decisions

An event forecast represents arrivals by type on a named continuous clock. A compiler plan may obtain arrivals from a finite fixture, a verified external object, or a fixed-decay linear Hawkes model. The same retained path law can drive several intervention worlds. The SDE runs the worlds for the same arrival path and Brownian increments, so its joint output can answer decisions driven by pathwise risk or nonlinear utility.

Choose the observation window and record the complete history from an explicit empty origin. Native likelihood treats the forecast cutoff as the first instant outside the observed window. Gaps and ties in native timestamps need a separate observation model. Branching and decay arrays use target rows and source columns. Excitation describes predictive association. World effects come from the declared causal equations and bindings.

A dense catalog is limited by the likelihood build. One build charges `(events + 1) * types²` against the operation budget, and the catalog cap is the largest type count for which a build at the event cap uses at most half that budget. A forecast can still be refused when its fit or simulation exceeds the budget. The composer accepts as many intervention worlds as the SDE retains, 63 plus the baseline, and refuses when the grid's cell product exceeds its execution budget.

## Horizon and lifetime offspring

The branching entry `α` is the expected number of direct children over the whole future. On a finite horizon `H` the direct contribution of an exponential kernel is `α (1 - exp(-β H))`. A release at time `t` scored through a window that ends at `T` uses the remaining horizon `T - t`, so a later release excites less of that window.

Lifetime descendants, excluding the initiating event, are the matching column of `(I - A)^{-1} - I`. The resolvent itself counts the initiator. Descendants inside a finite horizon come from the linear filter ODE. They are not the resolvent of the horizon-discounted matrix `α (1 - exp(-β H))`, which would give every generation the full horizon.

`Study::prepare_excitation_offspring` (and `prepare_marked_offspring` for a marked kernel) compiles the query and reports all four quantities as functionals of the conditional intensity. In Python the operation is `offspring` (or `marked_offspring`) of `events.prepare`.

## Finite-rank marks

A marked history replaces the catalog with prototypes. Each prototype carries nonnegative source features, target features and a reference mass. With fixed decays the ground intensity is linear in the baseline and in the excitation coefficients. The fit does not project onto a spectral-radius constraint: that constraint is not a simple function of the coefficients. It projects onto a reproduction cap, the largest lifetime direct-child count allowed for any declared prototype, strictly below one. The cap is sufficient for stability and is generally stricter than `ρ < 1`, as the catalog column cap is. Declared parameters are checked with the spectral witness of the prototype branching matrix.

The rank and the basis are capped by the same dense-feature allowance as the catalog. The prototype count is capped by the cost of that spectral witness.

## Immigrant diagnostic

An excitation mass is a conditional intensity. `prepare_excitation_challenge` fits the constant-baseline model and a second model whose immigrant intensity may follow a declared nonnegative feature, such as a trend that is zero early and positive late. The diagnostic agrees when the horizon-specific direct masses differ by at most the caller's tolerance and neither fit is weakly identified or pinned to its stability cap. Otherwise the result names the reason and `agreed_direct` withholds the mass. The challenge runs as a compiled program of the `HawkesExcitation` route.

Agreement is not an identification result. A latent trend can enter both fits, and the tolerance is declared rather than derived from an uncertainty test. A causal reading still needs an identification strategy, or explicit structural assumptions, and the evidence that supports them.

`Study::prepare_release_comparison` (and `prepare_marked_release_comparison`) fits the challenge and compares the two releases only when the diagnostic agrees; a withheld reading reports no comparison. It scores the constant-baseline kernel's offspring of one added event over each remaining horizon. That is a conditional scenario under the agreed diagnostic, not an identified response to the release. A utility over path outcomes is still a decision contract on the event worlds.

## Parameter bridge and an intervened law

An agreed direct mass can be offered to the parameter-intervention bridge. The source and target functionals are `direct_offspring` under an identity or affine map, or `total_offspring` (the lifetime resolvent count: direct mass plus feedback descendants) under an identity map only. The fitted regime stays `observational` and the response names a different regime. A singular Gram or an active stability cap refuses before the bridge is asked. `derive_bridged_claim` then carries that one mean. Horizon and lifetime descendants are reported beside it and are not the bridged value; when excitation runs both ways (a two-way kernel or a feedback member) an affine bridge carries the direct mass and reports the resolvent value beside it. A withheld diagnostic produces no law.

The reading is taken against a declared structure, in any form the shared member carrier (`antecedent::analysis::structure_members`) takes: one DAG, ADMG or temporal graph; a CPDAG, PAG or temporal class through its completions; an explicit candidate set with optional weights; a DAG or ADMG graph posterior; or a cyclic system, whose edges an event process reads as lags. Each member is identified as `identify_structure` identifies it, by the complete identification algorithm for a static member. A member that orients the exposure into the outcome reads the bridged kernel value, a member where the edge is absent or reversed reads the bridged zero, and an unidentified member reads nothing. `AcceptedExcitation::structure` reports each member, the envelope over identified members, their weight and, for a weighted form, the mixture conditional on them. Only a structure none of whose members identifies the effect refuses (`events.reading_not_identified`). The bridge evidence is the provenance digest of the search the structure came from, or the declared structure's canonical identity; a single accepted DAG keeps its earlier result and program identity.

The release action is one added event: now, at a later time, or under a different mark. `Study::prepare_release_law` (or `AcceptedExcitation::release_law`, which compiles and runs it) draws the offspring of that event as a program composed on the accepted question. `Study::prepare_release_question` compiles the challenge, the bridge and one release action as one program; Python's `excitation_question` operation runs it. A marked history becomes a catalog with one name per prototype, so the existing paired worlds and decision contract can read the paths. When an identified member reads zero, the law is one weighted mixture: members that orient the edge draw with the added event, null members draw the same path index on the same random stream without it, so the difference between them is the structure alone. Those paths are a conditional scenario under the kernel the bridge kept. They are not, by themselves, the utility of releasing.

A shock that moves the exposure and is excluded from the outcome leaves the Hawkes model. The result is a lag-count query — the exposure, the outcome, the horizon, and the instrument — and it does not return an excitation coefficient. An ordinary identified query can take that lag. `Study::prepare_release_study` composes the observation study on the accepted question only when every hypothesis is the catalog kernel the bridge kept, and `PreparedEventStudy::release_information` then reports that the signal prices the release. A marked kernel is not priced that way. When a member reads zero, `Study::prepare_member_release_study` prices the release under each member kernel (the kept kernel, or that kernel without the exposure's edge into the outcome), reports the range of gross values over identified members and, with weights, the study under the belief that mixes the two kernels. Python's `member_release_study` operation runs it.

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
