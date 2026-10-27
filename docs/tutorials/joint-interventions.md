# Compare a coordinated intervention program with its baseline

A marketing team sets two channels, A and B, in every market every week. The question is not what one channel does in one week. It is what happens at the end of a 24-week horizon if both channels are held at a high level the whole time, compared with spending nothing on either. That is a question about a whole coordinated program, and answering it well needs three things to describe the same target: the intervention, the evidence for it and the uncertainty reported with it.

This tutorial builds that comparison on simulated data, so the true answer is known and the result can be checked against it.

## The task

- **Units and population.** Four geographic groups, each observed for 24 weeks after two weeks of recorded history. The target is the mean over exactly these four groups with their own histories and covariates: a finite population, not a draw from a larger one.
- **Intervention.** A **program** assigns a level to each channel in each week of a declared **horizon**. The baseline holds both channels at 0 for all 24 weeks. The coordinated program holds both at 4.
- **Outcome and time.** The mean outcome in the last week of the horizon.
- **Contrast.** The coordinated program minus the baseline.

The data are a synthetic fixture shipped in the repository (`conformance/response/joint_panel/`), generated from a stated mechanism. The mechanism is declared to Antecedent only in family form; its numbers are learned.

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

Group, season and covariate terms are the same in both programs and cancel. This 4 is a fixture truth of the stated mechanism, never a claim about real marketing data.

## Prepare, then inspect

Each world pairs a name with a full schedule: a level for every channel in every week, and the history and future context of each group. Preparation compiles the question without fitting anything.

```python
prepared = prepare(rows, draws=100)
assert prepared.inspect()["fits_performed"] == 0
```

In the example, `prepare` passes the two interaction channels, the two worlds the question compares (the baseline and the joint program) and the contrast as `functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 0}`: world 1 minus world 0. A contrast names exactly the worlds it compares; Antecedent refuses a contrast whose formula leaves a declared world out. The design checks below declare all four programs (the baseline, A alone, B alone and the joint program), because the whole-program design has one arm per program.

## Run: one fit, one shared uncertainty

```python
result = require_success(prepared.estimate(execution=Execution(seed=SEED)))
contrast = result.scheduled_contrast("terminal_mean", level=0.95)
```

The fit happens once. Each of the 100 retained draws is one complete parameter vector from the fit's sampling law, and every world is evaluated on the same draw. For a frequentist run the draws approximate the sampling law of the fitted parameters with a group-cluster sandwich; `contrast["uncertainty"]` says so with `EstimatorSampling`. A draw that failed stays a failed slot and is never replaced by zero.

`contrast["point"]` is the fitted joint-minus-baseline change in the last week, and `contrast["standard_error"]` and `contrast["interval"]` come from the contrast computed draw by draw. The interval is nominal. The example prints `report["diagnostics"]["calibration"]`; unless it reports a matching measurement, do not read the interval as demonstrated coverage.

The example checks that the point estimate is within three standard errors of the known 4. This is a check on the fixture and the fit, not a coverage claim.

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

The fit above answers "what does the fitted mechanism say?" It does not say whether the data identify the effect of the whole program. Inspection reports that separately, as `identified`:

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

The attempt to identify the programs by known whole-program randomization is rejected, and the rejection names its next attempt. Randomizing each week does not randomize a 24-week trajectory. No group in these data was ever assigned the whole program, and what the weekly design supports is week-level contrasts, not the path that follows from holding both channels at 4 for the whole horizon.

A design that does identify the programs assigns them whole. The fixture also contains one independent group per program, each assigned its whole program with probability one quarter:

```python
experiment = read_rows("whole_program_interaction.csv")
randomized = prepare(
    experiment, assignment=whole_program_assignment(), exogenous_context=True
).inspect()
assert randomized["identified"] is True
```

With the whole-program assignment declared, the programs are identified, provided `exogenous_context=True` declares the covariate and season paths exogenous to the programs (in this fixture they are generated independently of the whole-program assignment). Leave that declaration out and the attempt is rejected again. This inspection fits nothing: identification comes from the design and the declarations, not from the fit.

What identification does not do is fix the fit's uncertainty. With one group per program there is no replication, so the example does not fit that dataset, and numerical results from any fit stay model-conditional with calibration unmeasured.

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

The replay reproduces the numbers. `grants_current_authority` is false: replaying a historical result is not a live study and does not authorize a decision. For choosing among candidate programs under budgets, see the [joint intervention responses guide](../guides/joint-intervention-responses.md); for cell-level contrasts see the [joint cells guide](../guides/joint-cells.md).

## The complete example

<!-- example: examples/python/joint_intervention_program.py -->

The script reads the fixture from a repository checkout. Run it from the repository root. A Rust version of the panel workflow is in `examples/rust/joint_panel_workflow.rs`.

## What this established, and what comes next

Under the declared mechanism and one shared fit, the joint program's 24-week effect over the baseline was estimated with uncertainty that carries the dependence between the programs' estimated responses, and the example checks the known 4 against it. That estimate is conditional on the fitted mechanism. Weekly randomization did not identify the whole program, and whole-program randomization with a declared exogenous context would.

The next question is which data would earn the identified status for the program you care about: assigning whole programs, with replication, rather than assigning each week. That decision and its cost belong in the study design before the fit.
