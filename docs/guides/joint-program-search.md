# Search and certify intervention programs

Choose among coordinated intervention programs under budgets and restrictions. Use a fitted joint response. The fitted response supplies the objective. You declare which controls may move. You declare their bounds, their costs and any hard restrictions. Antecedent then ranks a finite candidate set, or searches a continuous schedule space. The result reports what the search establishes.

Start from a retained `PanelResult` (see [fit and compare coordinated intervention programs](joint-intervention-responses.md)). Every call below evaluates that frozen fit without refitting. In the snippets, `result` is that `PanelResult` over three channels `a`, `b` and `c` scheduled for 24 weeks. `Execution` is `antecedent.Execution`. A runnable ranking under a budget is in [`beyond_marketing_mix.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/beyond_marketing_mix.py).

The objective is the declared affine utility `utility_offset + utility_scale * R(w)`. `R(w)` is the fitted response functional of program `w`. By default it is `"terminal_mean"`, the fitted mean outcome in the last scheduled period. The cost of a program is the sum over its controls of `unit_cost` times the control's level.

## Two different claims

A search result makes one of two kinds of statement. They are not interchangeable.

- **A numerical certificate** is a bound on the *fitted scalar objective*. With interval arithmetic and branch-and-bound, no feasible schedule has a fitted value higher than the bound. It says nothing about whether the fitted mechanism is right.
- **A real-world optimum** would require the fitted response to describe what actually happens under each program. That is an identification and model question. A certified numerical optimum of a confounded or misspecified response is still a best schedule of that response. The result stays model-conditional unless the design identifies the programs.

Every report carries `scope`. It reads `model_conditional` unless the retained design identifies the worlds. Every report also carries `selected_value_reliability`, which reads `not_measured`. A search does not measure how optimistic a selection from the same data is. The [held-out procedures](#select-from-data-and-evaluate-on-held-out-units) address that. Finite search certifies only the candidate set you supplied. Continuous search without a certificate reports a best-found point and its stopping reason.

## Rank a finite set of programs

Call `search_declared_programs`. It ranks the panel's worlds as candidate programs. Each control binds one scheduled channel level to a lower bound, an upper bound and a unit cost. A linear `budget` excludes a world, with a concrete reason, before any value is attempted. Hard `restrictions` do the same.

```python
controls = [
    {
        "channel": name,
        "offset": offset,
        "lower": 2.0 if name == "c" else 0.0,
        "upper": 2.0 if name == "c" else 4.0,
        "unit_cost": 0.0 if name == "c" else 1.0,
    }
    for offset in range(24)
    for name in ("a", "b", "c")
]
ranked = result.search_declared_programs(
    controls=controls, utility_offset=0.5, utility_scale=2.0, execution=Execution(seed=20261021)
)
rows = {row["id"]: row for row in ranked["rows"]}
assert ranked["exhaustive"] is True
assert ranked["global_certificate"] is False
assert ranked["continuous_gap"] is None
```

Each row has the world's `value`, its `cost`, its `controls` and a `feasibility` entry. `value` is the declared affine utility of its fitted terminal mean. With `budget=100.0`, a program costing more is reported as `{"kind": "operational", "reason": "budget"}` with no value. A restriction names itself as the reason. Controls must bind every channel at every schedule offset. Otherwise the call refuses.

Exhaustion is over the finite set of worlds on the result. It is not a continuous optimum. `continuous_gap` is absent. `global_certificate` is false.

## Certify a continuous schedule

Call `certify_program`. It builds an arithmetic objective from the retained fitted response. That objective includes histories, lags, saturation, interactions, population and time weights. It optimizes the objective with outward-rounded interval bounds and deterministic branch-and-bound. Declare every scheduled control. Use equal bounds to fix a coordinate.

Here only the last week's level of channel `a` may move (the other coordinates are fixed by equal bounds), which keeps the search small enough to certify:

```python
late_a = [
    {
        "channel": name,
        "offset": offset,
        "lower": 0.0,
        "upper": 2.0 if offset == 23 and name == "a" else 0.0,
        "unit_cost": 1.0 if offset == 23 and name == "a" else 0.0,
    }
    for offset in range(24)
    for name in ("a", "b", "c")
]
certificate = result.certify_program(
    base_world=0,
    controls=late_a,
    initial=[0.0] * len(late_a),
    absolute_gap=1e-3,
    max_splits=2000,
    execution=Execution(seed=20261021),
)
report = certificate.report()
assert report["stop"] == "Certified"
assert report["gap"] <= 1e-3
assert report["scope"] == "model_conditional"
assert report["selected_value_reliability"] == "not_measured"
```

Derivative signs and additive factorization permit reductions. Structurally proven concavity permits supporting-plane dual bounds. A resource stop (`SplitLimit`) keeps a valid partial bound. It does not claim the requested gap. Only a gap within your tolerance certifies optimality. The report lists the feasible controls, the pruning counts and the frontier. It also lists the shape proofs and the fitted source identity.

Certificates export and replay. `certificate.export()` freezes the response, the control bindings and the bound proof. `certificate.verify_current(artifact)` replays against this exact fitted response and search domain. `joint_panel.consume_program_certificate(artifact)` returns the historical numbers. `grants_current_authority` is false. Changing the utility, the fit or the domain changes the identities. `verify_current` refuses.

## Search when no certificate is available

Call `search_program`. It attempts the certificate. When the response has an expression without a supported interval evaluator, it runs a bounded best-found search instead:

```python
searched = result.search_program(
    base_world=0,
    controls=controls,
    initial=[0.0] * len(controls),
    max_evaluations=32,
    max_iterations=4,
    execution=Execution(seed=20261021),
)
assert searched["route"] in {"certified", "best_found"}
```

On a best-found result, the report keeps the concrete certificate refusal (`certificate_refusal`) and the stopping reason. It omits a global gap. Cancellation, invalid scopes and inadequate admission stay typed errors. Set `uncertainty="aligned_law"` to evaluate on the complete joint law. You also get the selected action's law separately. Its posterior meaning stays distinct from frequentist selected-value coverage.

## Select from data and evaluate on held-out units

Choosing the best program from the same data that fitted the response makes the selected value optimistic. Two procedures repeat the whole pipeline and evaluate the selected program on reserved original units.

- `prepare_randomized_selection` applies to randomized experiments over whole programs. Declare the design. Declare `training` and `validation` units: distinct original identities from an independent sampling frame, with the arm, assignment probabilities and outcome of each. Declare a `policy` with hard per-program resource costs. `execute(repetitions=...)` repeats generation, fitting, feasibility, search and selection. It then evaluates the fixed selected action on the reserved units. Hard costs are enforced before a winner is chosen. `simultaneous_intervals` gives joint bootstrap max-t intervals over a fixed feasible set.
- Call `select_instrumented_program`. It fits the structural-feature instrumental-variable model. It generates a prespecified or training-dependent dose domain. It derives and searches that polynomial response under hard restrictions. It refits the held-out units independently. The target is the declared homogeneous structural moment population. Local IV effects are not turned into population effects.

Both keep every resampling failure, tie, instability and the training-minus-held-out optimism. Both replay from the original observations. Call `report()`. Call `export()`. Call `verify_current`. Call `consume_randomized_selection` or `consume_instrumented_selection`. Bootstrap diagnostics assert no coverage at ties. The held-out evaluation's justified sampling law supplies the uncertainty of the fixed action. Independent random seeds cannot replace reserved observations.

## Related

- [Fit and compare coordinated intervention programs](joint-intervention-responses.md)
- [Update a joint response with experiments and designs](joint-program-designs.md)
- [Decision contracts](../decisions.md)
