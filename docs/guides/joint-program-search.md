# Search and certify intervention programs

Choose among coordinated intervention programs under budgets and restrictions, using a fitted joint response. The fitted response supplies the objective; you declare which controls may move, their bounds, their costs and any hard restrictions. Antecedent then ranks a finite candidate set, or searches a continuous schedule space, and reports what the search establishes.

Start from a retained `PanelResult` (see [fit and compare coordinated intervention programs](joint-intervention-responses.md)). Every call below evaluates that frozen fit without refitting.

## Two different claims

A search result makes one of two kinds of statement, and they are not interchangeable.

- **A numerical certificate** is a bound on the *fitted scalar objective*: with interval arithmetic and branch-and-bound, no feasible schedule has a fitted value higher than the bound. It says nothing about whether the fitted mechanism is right.
- **A real-world optimum** would require the fitted response to describe what actually happens under each program. That is an identification and model question. A certified numerical optimum of a confounded or misspecified response is still a best schedule of that response, and the result stays model-conditional unless the design identifies the programs.

Every report therefore carries `scope` (`model_conditional` unless the retained design identifies the worlds) and `selected_value_reliability`, which reads `unmeasured` until a procedure that measures the reliability of selecting from data has been run. Finite search certifies only the candidate set you supplied; continuous search without a certificate reports a best-found point and its stopping reason.

## Rank a finite set of programs

`search_declared_programs` ranks the panel's compiled worlds as candidate programs. Each control binds one scheduled channel level to a lower and upper bound and a unit cost; a linear `budget` and hard `restrictions` exclude a world with a concrete reason before any value is attempted.

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

Each row has the world's `value` (the declared affine utility of its fitted terminal mean), its `cost`, its `controls` and a `feasibility` entry. With `budget=100.0`, a program costing more is reported as `{"kind": "operational", "reason": "budget"}` with no value; a restriction names itself as the reason. Controls must bind every channel at every schedule offset, or the call refuses.

Exhaustion is over the finite set of compiled worlds. It is not a continuous optimum, so `continuous_gap` is absent and `global_certificate` is false.

## Certify a continuous schedule

`certify_program` derives an arithmetic objective from the retained fitted response, including histories, lags, saturation, interactions, population and time weights, and optimizes it with outward-rounded interval bounds and deterministic branch-and-bound. Declare every scheduled control; use equal bounds to fix a coordinate.

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
assert report["selected_value_reliability"] == "unmeasured"
```

Compiler-derived derivative signs and additive factorization permit reductions, and structurally proven concavity permits supporting-plane dual bounds. A resource stop (`SplitLimit`) keeps a valid partial bound and does not claim the requested gap; only a gap within your tolerance certifies optimality. The report lists the feasible controls, pruning counts and frontier, shape proofs and the fitted source identity.

Certificates export and replay: `certificate.export()` freezes the response, control bindings and bound proof; `certificate.verify_current(artifact)` replays against this exact fitted response and search domain, and `joint_panel.consume_program_certificate(artifact)` returns the historical numbers with `grants_current_authority` false. Changing the utility, the fit or the domain changes the identities and `verify_current` refuses.

## Search when no certificate is available

`search_program` attempts the certificate and, when the response has an expression without a supported interval evaluator, runs a bounded best-found search instead:

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

On the best-found route the report keeps the concrete certificate refusal (`certificate_refusal`) and the stopping reason, and omits a global gap. Cancellation, invalid scopes and inadequate admission stay typed errors. Set `uncertainty="aligned_law"` to evaluate on the complete joint law and get the selected action's law separately; its posterior meaning stays distinct from frequentist selected-value coverage.

## Select from data and evaluate on held-out units

Choosing the best program from the same data that fitted the response makes the selected value optimistic. Two procedures repeat the whole pipeline and evaluate the selected program on reserved original units.

- `prepare_randomized_selection` applies to randomized experiments over whole programs. Declare the design, `training` and `validation` units (distinct original identities from an independent sampling frame, with the arm, assignment probabilities and outcome of each) and a `policy` with hard per-program resource costs. `execute(repetitions=...)` repeats generation, fitting, feasibility, search and selection, then evaluates the fixed selected action on the reserved units. Hard costs are enforced before a winner is chosen. `simultaneous_intervals` gives joint bootstrap max-t intervals over a fixed feasible set.
- `select_instrumented_program` fits the structural-feature instrumental-variable model, generates a prespecified or training-dependent dose domain, derives and searches its polynomial response under hard restrictions, and refits the held-out units independently. The target is the declared homogeneous structural moment population; local IV effects are not turned into population effects.

Both keep every resampling failure, tie, instability and the training-minus-held-out optimism, and both replay from the original observations (`report()`, `export()`, `verify_current`, and `consume_randomized_selection` / `consume_instrumented_selection`). Bootstrap diagnostics assert no coverage at ties. The held-out evaluation's justified sampling law supplies the uncertainty of the fixed action; independent random seeds cannot replace reserved observations.

## Related

- [Fit and compare coordinated intervention programs](joint-intervention-responses.md)
- [Update a joint response with experiments and designs](joint-program-designs.md)
- [Decision contracts](../decisions.md)
