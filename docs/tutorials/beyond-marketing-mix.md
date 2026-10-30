# Beyond the marketing mix: decide what to change and what to learn next

A retailer wants to allocate advertising across search, social and video. A marketing mix model can estimate delayed response, diminishing returns and channel interactions. Causal inference asks a further question: what evidence supports treating those responses as the consequences of changing spending? This tutorial follows that question through an experiment, a business decision and the value of another study.

The investigation uses one synthetic geographic panel and one retained response model. You declare the scientific question, inspect its evidence requirements before fitting, and carry its quantities into later computations. The known generating mechanism provides an independent numerical check. Its numbers describe this simulation, not a retailer in the world.

## The task

Four geographic markets have 72 observed weeks and two weeks of recorded advertising history. Managers spend more when an observed demand indicator is high. Search, social and video have three-period carryover and a saturating response; search and video also interact. Markets have different intercepts, and revenue has seasonal variation and Gaussian noise.

Compare constant spending programs over the next 24 weeks. The outcome is the **mean revenue index across these same four markets in the terminal week**, conditional on their last two observed advertising weeks and declared future demand and season paths. It is not cumulative quarterly revenue. The example uses a 24-week planning horizon throughout, so changing it to a calendar quarter would require changing the target and experimental mapping together.

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

Run the maintained example from a checkout with the 3.0 Python extension installed:

```bash
python examples/python/beyond_marketing_mix.py
```

It generates its own observations, needs no additional dependencies or credentials, and runs offline. It prints the compiled question, fitted diagnostics and final decision and study ranking. Sampling and artifact verification make this a longer example than the introductory joint-intervention tutorial; the catalog gives the measured local runtime and a planning allowance (about 20 seconds locally; allow up to one minute).

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
assert ranked["exhaustive"]
assert not ranked["global_certificate"]
assert prepared.inspect()["fits_performed"] == 1
```

The explicit cache budget retains the fitted mechanism. Each call exhausts its pair of declared worlds; comparing the three proposal values exhausts the three proposals above. This establishes a finite candidate winner under the fitted terminal-mean objective. It supplies no continuous-allocation optimality bound. The [program search guide](../guides/joint-program-search.md) covers continuous certificates and held-out evaluation.

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

Spending follows anticipated demand. The declared model adjusts for the observed demand indicator, but an adjustment term does not itself prove that the fitted temporal mechanism evaluates an identified whole-program functional. Inspection reports `identified=False`. The whole-program randomization attempt reports that no known assignment contract was supplied, and names the missing evidence.

The next step is concrete: obtain known, positive-probability assignments of complete programs to independent units with matching histories and pretreatment context, or supply an accepted unrolled graph and an identification proof whose functional the computation reproduces. Weekly randomization alone does not identify a sustained trajectory. The smaller [joint-intervention tutorial](joint-interventions.md) executes that distinction and shows the corresponding design declarations.

Keep the fitted winner as a model-conditional proposal while deciding what evidence to obtain. A good fit and a numerical search result have answered useful questions, but neither has discharged this identification obligation.

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

The evidence changes the law. It supplies no blanket validation of every channel curve or allocation. Importance conditioning also cannot create response draws outside the retained law; a sharply conflicting experiment calls for checking the model and sampling support. The [experimental-update guide](../guides/joint-program-designs.md) explains the bridge, dependence and design contracts in more detail.

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

The example compares two ways to repeat the same terminal-contrast study. Both supply 16 independent summaries with observation variance 0.64. Their costs differ: 0.001 versus 0.05 contribution-index units. Holding the sampling model fixed isolates the effect of cost and avoids inventing an unsupported mapping to a search–video interaction experiment.

```python
ranking = updated.rank_next_samples(
    actions=actions, candidates=candidates, ledger=ledger,
    replicates=128, execution=EXECUTION,
)["candidates"]
```

The terminal actions use exactly the contribution utilities above. Each possible future observation reweights the retained law; the study's gross value is the expected gain from choosing after that observation rather than now. Its net value subtracts the study cost. The script asserts equal gross values for equal sampling models, the cost arithmetic and a shared admissible-action identity. It also checks gross information value against independent Gaussian-mixture quadrature, within the quadrature error bound plus six Monte Carlo standard errors. In the seeded run, gross information value is about 0.0031 contribution-index units, with a Monte Carlo standard error of 0.00062. Net values are about 0.0021 and −0.0469. The cheaper study ranks first. That ranking does not itself say to run it: compare its net value with zero, the value of proceeding without another study, and read the Monte Carlo standard error.

An interaction study or a study in other markets would need its own licensed estimand bridge. `rank_next_studies` supports those declared regimes; the [experimental-update guide](../guides/joint-program-designs.md#decide-then-rank-further-studies) explains that route. Greater precision for this decision would still be a different achievement from obtaining a design that identifies the program effect.

## Epilogue: preserve the record as evidence changes

The example exports the fitted panel and experimental update, checks the update against its current ledger, then withdraws the experimental source. Current use of the dependent decision refuses. The original panel artifact still replays its stored numerical result with `grants_current_authority=False`.

That is the scope of the Python demonstration. It verifies the exported update while evidence is current and historically replays the original panel fit. It does not claim standalone historical replay of the exported panel update or an exported decision through a Python helper for a different design family. The [replay tutorial](replay-and-current-decisions.md) demonstrates a separate mixed-evidence workflow with a decision artifact.

The final reuse receipt is `fits_performed == 1`. Program changes, experimental conditioning, utility changes and further-study ranking have built on the retained fit. Historical reproducibility and current permission to use evidence remain separate questions.

<!-- example: examples/python/beyond_marketing_mix.py -->

## What this established

A marketing response model supported a spending proposal with carryover, saturation, interaction and geographic variation. Inspecting its identification requirements exposed what the observational fit did not establish. A mapped experimental summary changed the retained law, a stated business utility determined an action, and that same decision supplied the objective for ranking further evidence.

The investigation continued from marketing mix analysis into causal evidence and scientific decisions while retaining the qualifications of each result. The next question is: *what evidence would most improve tomorrow's decision?*
