# Beyond the marketing mix: decide what to change and what to learn next

A retailer can spend six units a week on advertising. Should it split the money evenly across search, social and video, or concentrate on two channels? The retailer cares about revenue in week 24 and has the option to spend nothing if the expected gain is too small.

You’ll use weekly data from four simulated markets to choose a proposal, then use a geographic experiment to revise its predicted gain. The same estimate goes into two decisions. Is the proposal worth its cost? Is another experiment worth paying for?

Run `python examples/python/beyond_marketing_mix.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/beyond_marketing_mix.ipynb). See the [complete example](#the-complete-example).

This example builds on two shorter tutorials about [spending schedules](joint-interventions.md) and [updating a fit with an experiment](experiment-to-next-study.md).

## The task

The data cover 72 weeks in four markets, plus two earlier weeks of advertising history. Managers spend more when they expect demand to rise. Advertising in any one week can affect revenue that week and the next two. Returns diminish as exposure grows, and the return on search depends partly on video spending.

For each proposal, predict the **average week-24 revenue index across the four markets**. All proposals start with the same two weeks of spending history and use the same assumed paths for future demand and seasonality. Only the spending schedule changes.

| Declared by the researcher | Learned or computed |
| --- | --- |
| Four markets, observed data, future context, 24-week schedules and terminal outcome | Fitted response for each complete schedule |
| Three-period lag, Michaelis–Menten saturation, Gaussian group intercepts and serial errors | Channel coefficients, lag weights, shared half-saturation and geographic intercepts |
| Search–video interaction and seasonal terms | Interaction, season and demand coefficients |
| Three candidate allocations, bounds and budget | Best fitted terminal outcome in that finite set |
| Exact experimental estimand, identity bridge, likelihood and evidence origins | Reweighted response draws |
| Contribution margin, implementation price and a risk-neutral criterion | Expected business value and preferred action |
| Two further studies of that same contrast, sampling variance and costs | Expected value of sample information, net of cost |

Geographic variation here means different fitted market intercepts. The example does not estimate a separate channel coefficient for every market. The demand indicator is observed; no hidden confounder is simulated. Even with that convenient setup, this panel declaration supplies no accepted identification proof for a complete spending program.

The example generates its own observations, needs no additional dependencies or credentials, and runs offline. It prints the compiled question, the fit diagnostics, the fitted curves and, at the end, one JSON summary with the selected program, the contrast before and after the experiment, the decision, the study ranking and the two refusals it met.

## 1. Fit a marketing response

For channel `j`, the mechanism first forms lag-weighted exposure and then saturates it:

```text
exposure[j, t] = w[j, 0] spend[j, t]
               + w[j, 1] spend[j, t−1]
               + w[j, 2] spend[j, t−2]
s[j, t] = exposure[j, t] / (exposure[j, t] + half_saturation)

revenue = intercept + market + season + demand
        + sum_j coefficient[j] s[j, t]
        + interaction s[search, t] s[video, t] + error
```

You supply this family, rather than its generating parameter values. `TRUTH` stays outside the estimator's options and initialization. The fit learns the lag weights, saturation and coefficients from the observations.

Preparation compiles the population, histories, worlds and contrast without fitting:

```python
inspection = prepared.inspect()
assert inspection["fits_performed"] == 0
print(inspection)
```

Read the attempts and their `next_attempt` fields before executing. They state what would be needed for an identified program effect. Then estimate and read both optimizer and posterior diagnostics. The example requires convergence under its stated sampling policy: four chains, 8,000 warmup iterations and 12,000 retained draws per chain, with maximum R-hat 1.05 and minimum effective sample size 100. The diagnostics check whether the numerical run meets that policy.

The simulation declares a future demand path for each market; this is a scenario input, not a fitted demand forecast.

The script prints fitted isolated response curves at exposures 0, 1, 2, 3 and 4, as well as fitted lag weights and geographic intercepts. These curves are the fitted mean response to lag-weighted exposure, with the other channels at zero. Read them alongside the search–video interaction: search's incremental contribution with video running includes an additional interaction term. Adding the two isolated curves misses that term.

A pulse of spending contributes exposure in its own week and the next two weeks. A sustained program accumulates those contributions; after two weeks its exposure equals its constant spending level because the lag weights sum to one. Saturation then makes a second unit of exposure worth less than the first. Neither that shape nor the geographic intercepts establish causal identification.

## 2. Choose among spending programs

All three proposals spend six units per week, for a total budget of 144:

| Program | Search | Social | Video |
| --- | ---: | ---: | ---: |
| `balanced` | 2 | 2 | 2 |
| `search_video` | 3 | 0 | 3 |
| `search_social` | 3 | 3 | 0 |

Each scheduled channel is bounded between zero and three. The zero-spend `hold` program remains a reference and an admissible action; spending the entire allowance is optional.

Retarget the prepared question to each proposal and its baseline, keeping the source data and fitting policy fixed:

```python
branch = prepared.retarget(worlds(scenarios, name, levels), execution=EXECUTION)
result = require_success(branch.estimate(execution=EXECUTION))
ranked = result.search_declared_programs(
    controls=controls, budget=144.0, execution=EXECUTION
)
assert ranked["exhaustive"] and not ranked["global_certificate"]
assert prepared.inspect()["fits_performed"] == 1
```

The cache lets all three proposals reuse the same fit. Choose the proposal with the highest predicted week-24 revenue among those in the table. A different split of the budget might do better. For searches over a continuous range of allocations, see the [program search guide](../guides/joint-program-search.md).

Every proposed program and its baseline share each complete parameter draw. The script checks that `scheduled_contrast` equals the draw-by-draw difference and verifies

```text
Var(program − baseline) = Var(program) + Var(baseline) − 2 Cov(program, baseline)
```

The simulation also provides an independent truth. For `balanced`, terminal exposure is 2 and saturation is `2 / (2 + 2) = 1/2`. Its true increment is

```text
3 × 1/2 + 2 × 1/2 + 1 × 1/2 + 1.5 × (1/2)² = 3.375
```

The script asserts that arithmetic to `1e-12`, then checks the selected program's fitted point against its own generating truth within 0.6 revenue-index units. The absolute tolerance checks recovery of this fixture. Report the fitted point and posterior spread separately. The example prints a posterior standard deviation, which falls from about 0.129 to 0.109 after the experiment; these describe uncertainty about the fitted mean rather than future outcome noise.

## 3. Ask whether the allocation has a causal interpretation

We have a fitted allocation winner. What would make its incremental effect causal?

Managers spend more when they expect higher demand. The model accounts for the demand indicator, but the panel supplies no assignment probabilities for complete spending schedules. Look at `identified` in the selected result. Here it is `False`. The plan lists the evidence each identification attempt would require.

The next step is concrete: obtain known, positive-probability assignments of complete programs to independent units with matching histories and pretreatment context, or supply an accepted unrolled graph and an identification proof whose functional the computation reproduces. Weekly randomization alone does not identify a sustained trajectory. The smaller [joint-intervention tutorial](joint-interventions.md) executes that distinction and shows the corresponding design declarations.

At this stage you have a proposal worth testing and a prediction to test it against.

## 4. Incorporate a geographic experiment

Suppose a later geographic holdout reports the selected program's terminal contrast against `hold`. The example simulates **16 independent Gaussian contrast summaries**, centered on that exact generating truth, with known observation variance `0.8²`. It does not simulate or analyze the holdout's raw randomized assignments. This keeps the demonstration on the supported experimental-summary conditioning route, and makes the likelihood premise explicit.

The target remains the same four-market average, terminal week and exact spending schedule. A fixed identity `BridgeContract` declares that the experiment's sample mean measures that contrast, with no population transport, window extrapolation or extra mapping uncertainty. A real holdout in other markets or over another window would need a justified mapping and its uncertainty; giving it this identity bridge would change the scientific assumptions.

The ledger registers three origins: the fitted panel, the new experiment and the mapping declaration. The mapping is bound to the fit's dependency identity and paired with its draw slots. These inputs are retained by the update:

```python
updated = result.condition_experiment(
    bridge=bridge_for(target), ledger=ledger,
    observed_mean=observed, observation_variance=0.8**2,
    sample_size=16, mapping_dependency=dependency,
    intercepts=[0.0] * len(draws), slopes=[1.0] * len(draws),
    model=model, experiment=experiment, mapping=mapping,
    execution=EXECUTION,
)
```

Each original response draw keeps its values and receives a Gaussian likelihood weight. No fit is repeated. The script independently computes those normalized weights and requires agreement to `1e-12`. It also attempts to use the terminal-week bridge for a different time window and asserts that the update refuses. Resolving that refusal requires a bridge for the changed estimand, rather than relabeling the terminal experiment.

Read the result's standing:

```text
scope: model_conditional_experimental_update
modular_policy: frozen_joint_importance
```

With the fixed seed, the fitted point contrast is about 3.266, against the generating truth of 3.375. The retained posterior mean moves from about 3.260 to 3.269 after the experimental mean of 3.294. These are different summaries: the optimizer point need not equal the posterior mean.

The experiment gives more weight to draws that predict its result well. If all the draws disagree sharply with the experiment, check the fitted model and the range of draws it produced. Reweighting can only redistribute weight among those draws. See the [experimental-update guide](../guides/joint-program-designs.md).

## 5. Decide whether to implement it

A revenue winner is an input to a decision. Declare a contribution margin of 0.4 and an implementation price of 0.009 contribution-index units per spend unit. The action's utility is

```text
implement = 0.4 × terminal revenue increment − 144 × 0.009
hold      = 0
```

These prices convert a horizon's spending into the units of this terminal-outcome decision. They do not turn terminal revenue into cumulative profit. A quarterly profit question would need an outcome accumulated over the quarter and an experimental bridge for that outcome.

The risk policy is posterior expected utility: this example is risk-neutral about the uncertain mean response. The parameter posterior does not describe future realized revenue shocks, and the example makes no probability-of-loss claim about them.

The available actions here are implementing the selected proposal or holding; the example does not rerank all allocations after conditioning.

The decision contract reads the update's exact scientific coordinates and declares their population **and horizon**. The script checks the reported expected utility against independently weighted contrasts, and prints business value before and after the experiment. A decision change is not forced by choosing a break-even after seeing the evidence.

To see why revenue maximization is insufficient, increase the implementation price to 0.02 using the same updated law. The decision now chooses `hold`; the script asserts that result. Changing the utility has performed no additional fit. See [decision contracts](../decisions.md) for other supported objectives and [decide under uncertainty](decide-under-uncertainty.md) for a smaller business example.

## 6. Decide what to learn next

Now we have a decision under today's evidence. Which additional evidence would improve it?

Both candidate studies would measure the week-24 contrast using 16 independent summaries with observation variance 0.64. One costs 0.001 contribution-index units and the other costs 0.05. Because the two would collect the same information, you can compare the price of that information directly.

```python
ranking = updated.rank_next_samples(
    actions=actions, candidates=candidates, ledger=ledger,
    replicates=128, execution=EXECUTION,
)["candidates"]
```

For each possible study result, update the weights and choose between spending and holding. Averaging the gain from that choice gives a gross information value of about 0.0031 contribution-index units. The script checks this value by independent integration.

Subtract each study’s price. The cheaper study has a net value of about 0.0021, compared with −0.0469 for the more expensive one. Compare each value with zero, the gain from proceeding without another study. The Monte Carlo standard error of the gross value is about 0.00062.

An interaction study or a study in other markets would need its own supported estimand bridge. `rank_next_studies` supports those declared regimes; the [experimental-update guide](../guides/joint-program-designs.md#decide-then-rank-further-studies) explains that route. Greater precision for this decision would still be a different achievement from obtaining a design that identifies the program effect.

## Epilogue: preserve the record as evidence changes

The example exports the fitted panel and experimental update, checks the update against its current ledger, then withdraws the experimental source. Current use of the dependent decision refuses. The original panel artifact still replays its stored numerical result with `grants_current_authority=False`.

That is the scope of the Python demonstration. It checks the exported update against the current ledger (`verify_current`) and replays the original panel fit historically. It does not claim standalone historical replay of the exported panel update or an exported decision through a Python helper for a different design family. The [replay tutorial](replay-and-current-decisions.md) demonstrates a separate mixed-evidence workflow with a decision artifact.

The final reuse receipt is `fits_performed == 1`, printed as `reuse_receipt` in the summary beside `bridge_refusal` and `withdrawal_refusal`, the messages of the two refusals. Program changes, experimental conditioning, utility changes and further-study ranking have built on the retained fit. Historical reproducibility and current permission to use evidence remain separate questions.

## What this established

A marketing response model supported a spending proposal with carryover, saturation, interaction and geographic variation. Inspecting its identification requirements exposed what the observational fit did not establish. A mapped experimental summary changed the retained law, a stated business utility determined an action, and that same decision supplied the objective for ranking further evidence.

## What this did not establish

None of this identified the effect of a whole spending program: the selected allocation remains a model-conditional proposal until a design such as whole-program randomization supplies that evidence. The posterior describes uncertainty about the fitted mean response, not future revenue shocks, and the experiment was simulated as summaries centered on the generating truth.

## Where to go next

To design an experiment that assigns complete spending schedules, see the [experimental-update guide](../guides/joint-program-designs.md). To compare studies by the decisions they would inform, see [design ranking](../design-ranking.md).

## The complete example

<!-- example: examples/python/beyond_marketing_mix.py -->

[Open the Python script](../../examples/python/beyond_marketing_mix.py)
