# Compare a coordinated intervention program with its baseline

A marketing team is considering 24 weeks of advertising in two channels. How much higher would the average outcome be in the final week if the team spent four units on each channel every week, compared with spending nothing?

You’ll fit both spending schedules to data from four simulated markets. Because the same model predicts both outcomes, its errors can cancel when you subtract one prediction from the other.

Run `python examples/python/joint_intervention_program.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/joint_intervention_program.ipynb). See the [complete example](#the-complete-example).

The uncertainty calculation uses the variance and covariance of the two predicted outcomes.

## The task

- **Units and population.** Four geographic groups, each observed for 24 weeks after two weeks of recorded history. The target is the mean over exactly these four groups with their own histories and covariates: a finite population, not a draw from a larger one.
- **Intervention.** A **program** assigns a level to each channel in each week of a declared **horizon**. The baseline holds both channels at 0 for all 24 weeks. The coordinated program holds both at 4.
- **Outcome and time.** The mean outcome in the last week of the horizon.
- **Contrast.** The coordinated program minus the baseline.

The example reads simulated observations from `conformance/response/joint_panel/`. You specify the form of the response model, and Antecedent estimates its parameters from those observations.

| Declared | Learned from the panel |
| --- | --- |
| Channels, horizon, programs, the contrast | Lag weights over the current and two previous weeks |
| Lag-weighted exposure, saturated as `x / (x + h)` | The half-saturation point `h` |
| Additive group, season and observed-covariate terms | Channel coefficients, interaction, intercepts |
| Gaussian errors, autocorrelated within a group | The error autocorrelation |
| One interaction between the channels | Its coefficient |

Because the generating mechanism is known, so is the answer. The lag weights sum to one, so in the last week of a constant program the exposure equals the level, 4. Saturation with half-point 2 gives `s = 4 / (4 + 2) = 2/3`. The channel coefficients are 3 and 2 and the interaction coefficient is 1.5, so the baseline-to-joint change in the mean is

```text
3 × (2/3) + 2 × (2/3) + 1.5 × (2/3)² = 2 + 1.333… + 0.667… = 4
```

Season, market and covariate effects cancel because the two schedules share them. That leaves a difference of four outcome units, a value we can use to check the fitted result.

## Prepare, then inspect

Each world pairs a name with a full schedule: a level for every channel in every week, and the history and future context of each group. Preparation compiles the question without fitting anything.

```python
prepared = prepare(rows, draws=100, fitted=True)
assert prepared.inspect()["fits_performed"] == 0
```

`prepare(..., fitted=True)` compares two worlds. The weights in `functional` add the coordinated program’s outcome (world 1) and subtract the baseline (world 0). Every declared world must appear in the comparison. Later, when you check a design with four assignment arms, you’ll declare the two single-channel schedules as well.

## Run: one fit, one shared uncertainty

```python
result = require_success(prepared.estimate(execution=Execution(seed=SEED)))
report = result.report()
assert report["fits_performed"] == 1
contrast = result.scheduled_contrast("terminal_mean", level=0.95)
```

Look for `fits_performed` in the report. Antecedent fits the model once, draws 100 parameter vectors, and uses each vector to predict both programs. `EstimatorSampling` means the draws approximate sampling uncertainty using a group-cluster sandwich estimator. A failed draw keeps its slot.

`contrast["point"]` is the fitted joint-minus-baseline change in the last week, and `contrast["standard_error"]` and `contrast["interval"]` come from the contrast computed draw by draw. The interval is nominal. The example prints `report["diagnostics"]["calibration"]`; unless it reports a matching measurement, do not read the interval as demonstrated coverage.

The example checks that the optimizer converged and that the point estimate is within three standard errors of the known 4. This is a check on the fixture and the fit, not a coverage claim.

### Why the covariance matters

Both programs are evaluated on one fitted law, so their estimated responses are not independent: the intercept, the season terms and the covariate effect are shared, and an error in them moves both programs together. The variance of the contrast is

```text
Var(joint − baseline) = Var(joint) + Var(baseline) − 2 Cov(joint, baseline)
```

If you instead take each program's own uncertainty and combine them as if they came from independent studies, you drop the covariance term. The example computes both from the same draws:

```python
covariance = statistics.covariance(program, baseline)
shared_se = statistics.stdev(differences)
independent_se = math.sqrt(statistics.variance(program) + statistics.variance(baseline))
assert shared_se < independent_se
```

The independent combination reports a larger standard error here, because the shared terms cancel in the difference. With a negative covariance it would understate the uncertainty instead. In either case only the number that keeps the dependence between estimated responses describes the contrast. The example also asserts the identity `shared_se² = Var(joint) + Var(baseline) − 2 Cov` and that `scheduled_contrast` agrees with the per-draw differences.

## What the assignment design establishes

The fit above answers "what does the fitted mechanism say?" It does not say whether the data identify the effect of the whole program. The report states that separately, as `identified`:

```python
assert report["identified"] is False
```

Antecedent keeps the difference between a model that can compute a whole program and a design that identifies its causal effect. A fitted curve can evaluate any schedule. Whether that evaluation is the causal effect of holding both channels high for 24 weeks depends on how the programs were assigned.

In the fixture, assignment was randomized every week within each group, with known probabilities that depend on the observed covariate `z`. Declare that design and ask again:

```python
per_week = prepare(
    rows, assignment=weekly_assignment(rows), exogenous_context=True
).inspect()
weekly = whole_program_attempt(per_week)
assert per_week["identified"] is False
assert weekly["outcome"] == "Rejected"
assert weekly["next_attempt"] is not None
```

In this panel each market drew a new assignment every week. The whole-program check needs the assignment probability for a full 24-week schedule. It rejects the weekly design and tells you which evidence is missing.

A design that does identify the programs assigns them whole. The fixture also contains one independent group per program, each assigned its whole program with probability one quarter:

```python
experiment = read_rows("whole_program_interaction.csv")
randomized = prepare(
    experiment, assignment=whole_program_assignment(), exogenous_context=True
).inspect()
assert randomized["identified"] is True
```

With the whole-program assignment declared, the programs are identified, provided `exogenous_context=True` declares the covariate and season paths exogenous to the programs (in this fixture they are generated independently of the whole-program assignment). Leave that declaration out and the attempt is rejected again. This inspection fits nothing: identification comes from the design and the declarations, not from the fit.

What identification does not do is fix the fit's uncertainty. With one group per program there is no replication, so the example does not fit that dataset, and numerical results from any fit stay model-conditional; read their calibration status rather than assuming coverage.

## Continue with the result

A fitted result can be exported and replayed by a consumer that did no fitting:

```python
replay = require_success(
    joint_panel.consume(
        bytes(result.export(execution=Execution(seed=SEED))), execution=Execution(seed=SEED)
    )
)
assert replay["point"] == report["point"]
assert replay["grants_current_authority"] is False
```

The replay reproduces the numbers. `grants_current_authority` is false: replaying a historical result is not a live study and does not authorize a decision.

The example ends by printing a summary as JSON. Read it against the steps above:

| Key | What it reports |
| --- | --- |
| `known_effect` | The final-week difference in the simulation, 4 |
| `estimated_effect` | `contrast["point"]` |
| `shared_fit_standard_error` | The standard error of the per-draw differences |
| `independent_combination_standard_error` | The larger value that drops the covariance |
| `response_covariance` | The covariance between the two programs' estimated responses |
| `failed_draw_slots` | Draws that failed and were kept as failed slots |
| `weekly_assignment_identifies_program` | `false` |
| `weekly_assignment_detail` | The rejected attempt's stated reason |
| `whole_program_assignment_identifies_program` | `true` |

For choosing among candidate programs under budgets, see the [joint intervention responses guide](../guides/joint-intervention-responses.md); for cell-level contrasts see the [joint cells guide](../guides/joint-cells.md).

## What this established

The shared draws let you estimate the final-week difference and its standard error. The simulated difference is four units. To support a causal claim about an entire schedule using the randomization check, you also need a design that assigns complete schedules and specifies exogenous future context.

## What this did not establish

The fitted panel has `report["identified"] = False`. Its estimate depends on the response model, and its interval has nominal coverage. The separate whole-program experiment illustrates an identifying design. With just one group per program, it provides no replication for estimating between-group variation.

## Where to go next

The next question is which data would earn the identified status for the program you care about: assigning whole programs, with replication, rather than assigning each week. That decision and its cost belong in the study design before the fit.

For one connected marketing investigation, [beyond the marketing mix](beyond-marketing-mix.md) follows a three-channel panel through spending proposals, experimental evidence, a business decision and the next study.

## The complete example

<!-- example: examples/python/joint_intervention_program.py -->

[Open the Python script](../../examples/python/joint_intervention_program.py)

The script reads the fixture from a repository checkout and, without one, downloads it from the matching release tag. A Rust version of the panel workflow is in [`examples/rust/joint_panel_workflow.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/joint_panel_workflow.rs).
