# The smallest defensible test: from synthetic users to a launch decision

A team has redesigned a website and wants as few people as possible to see it before launch. Thousands of software agents have tried both versions. Customers have used only the old one. How many people should try the redesign before the team decides whether to launch?

The team requires at least 82% of human users to complete checkout on the new site. You’ll use the agents’ results to predict how people will do, then update that prediction as the first human checkout results arrive.

The script simulates two pilots. In one, 32 people provide enough evidence to launch under the team’s rule. In the other, just seven of the first 16 people complete checkout. The team holds the launch and considers a smaller experiment to find out whether changing authentication would help.

All sessions in this example are simulated, including those labeled human. The two pilots illustrate how different observations can change the decision under the same rule.

Run `python examples/python/synthetic_users_launch.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/synthetic_users_launch.ipynb). See the [complete example](#run-the-complete-investigation).

The pilot analysis uses posterior probabilities and a binomial model for the number of successful checkouts.

## The task and the launch contract

Each user or agent attempts navigation, product discovery and checkout separately. Someone who fails navigation still attempts checkout, so the checkout rate counts everyone who tried the site. Historical sessions include an additional task that tests recovery from an authentication problem.

Before collecting any new human observations, declare:

| Item | Declaration |
| --- | --- |
| Target population | Humans represented by the recent old-site cohort, at launch |
| Outcome | Probability of completing the redesigned checkout task |
| Business requirement | Completion probability at least 0.82 |
| Risk rule | At least 0.95 posterior probability that the requirement holds |
| Terminal actions | Launch this version or withhold it |
| Pilot protocol | Plan an initial cohort from 8, 16 and 32; inspect at 8, 16, 32 and, if needed, 64 |
| Confidential exposure cost | 0.009 planning-utility units per full-site human session |
| Restricted authentication study | 0.001 utility units per session, exposing only that flow |

The 0.82 floor is a business requirement fixed before seeing the pilot. Recent old-site completion is about 0.80, but the contract asks whether the new site meets its floor; it does not silently convert a historical comparison into a randomized redesign effect.

The launch utility makes the risk rule executable. Launch earns `+1` when the requirement holds and `−19` when it fails; withholding earns zero. Consequently:

```text
E[launch utility] = P(requirement holds) − 19 P(requirement fails)
                 = 20 P(requirement holds) − 19
```

Launch beats withholding exactly when that probability exceeds 0.95. The native decision contract reads a posterior over the indicator `requirements_hold`. Its uncertainty is uncertainty about a population completion probability, not the probability that one particular visitor completes checkout.

The contract compares launching with withholding. When it withholds, the application chooses between **investigate** and **do not launch** according to whether a further study could change the decision.

## 1. Learn what the old site can tell us

The example generates three independent datasets:

| Dataset | Sessions | Confidential human exposure |
| --- | ---: | ---: |
| Humans on the old site | 600 | 0 |
| Agents on the old site | 2,000 | 0 |
| Agents on the redesign | 2,000 | 0 |

Antecedent fits Bernoulli mechanisms for the old-site tasks and compares source-agent parameters with target-human parameters. Navigation and discovery are consistent with the declared invariance in this fixture. Checkout and authentication recovery differ, so those mechanisms require human-specific fits. A non-rejection for the first two tasks does not establish equivalence or show that their redesign effects transfer.

The declared causal picture is that population and website version can change mechanisms. The redesign changes task behavior; human error recovery may change the response to authentication. For the later authentication question, an unobserved error propensity may influence both the recovery path and checkout success. These are researcher-supplied causal assumptions, not a graph discovered from the clickstream summaries.

The old-site transport executes with human target fits for the shifted mechanisms. Its receipt names the carried and refitted components and reports `support="extrapolative"` and `inference="point_only"`. Then try to carry redesigned checkout into humans while explicitly declaring that it may shift:

```python
tmt.transport_typed_mechanisms(
    tmt.TypedTransport(
        source_population="agents_new",
        target_population=POPULATION,
        selection_nodes=(
            tmt.SelectionNode("unresolved_human_checkout", ("checkout",)),
        ),
    ),
    new_source,
    [],  # no human evidence on the redesign yet
)
```

Antecedent refuses with `transport_missing_evidence` and `typed_mechanism_transport.shifted_without_target_evidence`. The missing human mechanism stays missing. The next section constructs a **prior prediction** under a further assumption; it does not turn that refused causal transport into an identified effect.

## 2. Construct an informative, revisable prior

Use the agents' *change* between website versions to inform the human prediction, rather than treating agent completion as human completion:

```text
predicted human redesign performance
    = recent human old-site performance
    + agent redesign performance − agent old-site performance
```

For checkout, the seeded data give about `0.800 + 0.976 − 0.900 = 0.876`. This mapping assumes that the old-site human–agent gap remains applicable to the redesign's effect. It also assumes comparable tasks and a stable recent human population, traffic mix and measurement process. A version-by-population interaction can break it even when the old-site comparison looked satisfactory.

Keep two uncertainties separate. Sampling variance comes from the three datasets. Transfer variance describes uncertainty about the human–agent mapping. The example adds a declared transfer standard deviation of 0.025 for checkout, and 0.04 for navigation and discovery. These are assumptions supplied by the researcher. Increasing the number of agent sessions cannot remove them.

`priors.compose_external_priors` combines the transferred Gaussian summary with a broad human-centered baseline, at weights 0.8 and 0.2. Omitting the cross-population `TransportPolicy` refuses; supplying `invariant_effect_modifiers()` records the assumption. The example independently checks the mixture mean and variance to `1e-12`.

The pilot observes binary outcomes, so it needs an explicit probability model. Application code retains the declared mixture components and evaluates them on a fine grid of completion probabilities from 0.0005 to 0.9995. It does not pass a Gaussian coefficient prior off as a native Bernoulli posterior. The transferred component describes the agent-informed prediction. The broad component preserves three candidate explanations for a discrepancy:

- A human-specific problem with authentication.
- A mismatch between the agent and human populations.
- A changed relationship between navigation and checkout.

Sampling variation remains possible under every component. The grid includes mildly harmful and borderline responses as well as large failures; it does not force every website to be either clearly safe or catastrophically bad. Navigation and discovery are descriptive diagnostics in this worked decision. The launch gate models checkout; a joint requirement over all three stages would need a joint uncertainty model for those requirements.

The evidence ledger records the three source datasets and their row identities. The derived prior names all three in its lineage. Each new human block is recorded once, and each checkpoint posterior names the prior and the human blocks it used. The 4,000 agent sessions never become 4,000 independent human observations.

## 3. Let the decision guide the human pilot

Before seeing the pilot, rank initial cohorts of 8, 16 and 32 people by expected value of sample information, net of exposure cost. Each candidate supplies a predictive distribution for its checkout count. Antecedent updates the finite signal law, integrates the best terminal action after the observation, and subtracts the declared cost.

Of the three initial cohorts, 16 people offer the greatest expected gain after exposure costs. The team commits to 16 and checks after eight for an early conflict that would warrant a pause. After 16 it can decide whether to recruit the next batch.

Here “smallest” refers to the cohorts and stopping rules the team actually considered. Change the cost of exposing the redesign and a different cohort may be worth recruiting.

At each checkpoint, update the **original** mixture with all observations collected so far:

```text
posterior weight(component, hypothesis)
    ∝ original prior weight(component, hypothesis)
      × product of the observed Bernoulli likelihoods
```

Recomputing from the original prior avoids accidentally applying an earlier human block twice. The script checks the log-count calculation against an independent product over individual observations to `1e-12`.

The checkpoint controller is Bayesian and model-based. Its stopping decisions use the retained posterior and the value of another candidate study. It does not repeatedly inspect a fixed-sample p-value and treat it as an optional-stopping procedure. The small conflict-tail diagnostic below has a separate role: deciding when to investigate the model.

## 4A. Agreement: a small test can be enough for the decision

The hidden generating mechanism has human checkout completion of 0.88, close to the transferred prediction. The example's observations and native decision results are:

| New humans exposed | Checkout successes | Posterior probability of meeting 0.82 | Result |
| --- | ---: | ---: | --- |
| 8 | 7 | 0.903 | Complete the initial pilot |
| 16 | 14 | 0.927 | Continue to the next checkpoint |
| 32 | 29 | 0.974 | Launch; stop further testing |

After 16 people the team still has too little evidence to launch. At 32 the pilot crosses the risk threshold. Recruiting 32 more people would cost about 0.142 utility units more than the expected gain from their results.

Now make the comparison that motivates the tutorial:

| Analysis | New human exposure | Evidence behind the belief | Probability of meeting the floor | Terminal choice |
| --- | ---: | --- | ---: | --- |
| Small human sample alone | 32 | Broad probability prior + the same 29 successes | 0.869 | Withhold |
| Guided small test | 32 | Historical humans + agents + those 29 successes | 0.974 | Launch |
| Fixed larger human-only comparator | 128 | Broad probability prior + 113 successes | 0.967 | Launch |

The larger comparator is a separate, predeclared reference study. It shares the first 32 observations and adds independent humans from the same synthetic population. It is not a universal sample-size recommendation. The human-only grid calculation is independently checked against the exact Beta-posterior tail, using its binomial-CDF identity.

With the historical and agent evidence, the team reaches the same launch decision after exposing 32 people to the redesign as the human-only comparison reaches after 128. The two analyses still give different estimates of uncertainty.

### Test how much that conclusion relies on transfer

Repeat the calculation while shifting the transferred human checkout prediction downward, leaving the broad alternative component available:

| Additional human-specific shift | Probability of meeting the floor after the same small pilot | Terminal choice |
| --- | ---: | --- |
| 0 | 0.974 | Launch |
| −0.04 | 0.845 | Withhold |
| −0.08 | 0.514 | Withhold |

The old-site observations cannot rule out every such redesign-specific shift. The launch conclusion therefore depends on the transfer model and its uncertainty, cohort stability and representative recruitment. The pilot supports a decision within those premises; agreement does not prove agent-to-human equivalence.

## 4B. Conflict: stop trusting the prediction, preserve the question

In the second hidden mechanism, humans complete checkout with probability 0.55 even though the agents predict improvement. The prior, costs, protocol and launch contract stay the same.

After 16 human sessions, only seven checkouts succeed. The probability of meeting the business floor is about 0.013, so Antecedent withholds launch. The posterior weight on the transferred component falls from 0.8 to about 0.018. This is a coherent mixture update: the contradictory observations reduce reliance on the transferred prediction without deleting their record.

The lower-tail probability of this checkout count under the original transferred prediction is about `0.00012`. `ConflictPolicy` recommends dropping external strength at the predeclared diagnostic threshold. The example reports that recommendation alongside the original mixture posterior; it does **not** use the same pilot to choose a new prior and then silently update that new prior with the pilot again.

All three alternative explanations retain equal posterior mass because they predict the same distribution of the observed task summaries. That equality is checked. The analysis has located a checkout discrepancy, but it has not identified its cause. More observations of the same summaries can sharpen the completion rate while leaving that causal ambiguity unresolved.

The application reports **investigate**. The pilot gives the team a reason to withhold launch, and an authentication change could still improve checkout. It would report `do_not_launch` if the evidence pointed to a regression with little prospect of a useful repair.

## 5. Repair identification, then choose a targeted study

The next question is specific: **would changing the authentication flow improve human checkout completion on the redesigned site?**

Use a causal graph with `auth → checkout` and an unobserved common cause, represented by `auth ↔ checkout`. Observing more naturally selected recovery paths cannot identify the authentication intervention in this graph. A `repair.HedgeContract` exposes the missing experiment.

Compare three candidate studies:

| Study | Human exposure | What it observes or assigns | Identification result |
| --- | ---: | --- | --- |
| `generic_32` | 32 full-site sessions | More ordinary task summaries | Insufficient |
| `auth_8` | 8 restricted-flow sessions | Randomized authentication flow, four people per arm | Verified sufficient |
| `auth_32` | 32 restricted-flow sessions | The same randomized intervention, 16 per arm | Verified sufficient |

The theorem check says what those designs **would identify if they deliver their declared evidence**. It does not say eight people estimate the effect precisely, prove the bypass is safe, or validate the entire website. The proposed randomization requires well-defined flows, positive assignment probabilities, representative recruitment, consistency and no interference between people.

The decision being informed now is whether to fund an authentication repair or abandon this redesign. This is an explicit change of decision, with the same human population and checkout outcome. Its utility is the modeled completion gain minus a repair cost of 0.15 units. A patched website would be a new version requiring its own launch assessment.

In the authentication explanation, the bypass restores checkout toward the agent-informed human prediction. In the population and navigation explanations, it does not. The candidate likelihoods describe the difference in success counts between randomized arms. Under the hidden conflict mechanism, an independent uniform-threshold calculation gives the known intervention gain `0.88 − 0.55 = 0.33`; that truth is kept outside the estimator and checked separately.

Antecedent ranks the declared signals and costs:

| Study | Gross information value | Net value | Recommendation |
| --- | ---: | ---: | --- |
| `auth_8` | 0.0098 | **0.0018** | Run this targeted study |
| `auth_32` | 0.0284 | −0.0036 | Extra exposure costs more than its additional value |
| `generic_32` | 0.00009 | −0.2879 | Neither identifies the missing effect nor helps this decision enough |

The narrow study has a lower exposure price because participants see only the authentication flow, rather than the confidential site. Its eight people are **additional** to the 16-person conflict pilot. No study is run by the ranker; it returns a recommendation.

The signal likelihoods and finite hypothesis model are supplied by the application through `ExternalSignal`. Antecedent owns the checked request binding, native signal update, integration, cost comparison and ranking. The receipt says `externally_attested`: the library has not independently established that a proposed study has the declared likelihood.

The native ranker accepts equally weighted state draws. Application code therefore approximates the weighted belief by 4,096 deterministic quantile draws and records the atom-mass error. The script independently enumerates all possible count summaries, checks the resulting information values to `1e-9`, and verifies that the unrounded belief gives the same candidate order and cost signs. Export and consumption also preserve the ranking's external-signal standing.

## What this established

When the human pilot agreed with the agent prediction, 32 new human sessions were enough to meet the team’s launch threshold. Under its stated costs, testing another batch would have been more expensive than the information it provided.

When the prediction failed, the pilot ruled out launching on current evidence. Testing an authentication bypass with eight additional people offered more decision value than another sample of ordinary checkout sessions.

The application supplies the finite hypothesis family, binomial likelihoods, numerical prior projection and checkpoint controller. Antecedent supplies the prior-composition, mechanism-transport, decision, identification-repair and study-ranking operations they call. The complete example makes that boundary executable.

## What this did not establish

The investigation did not establish agent-to-human equivalence, the cause of the checkout discrepancy or the effect of an authentication repair. The targeted study is a recommendation under declared likelihoods, not a result.

The next empirical question is whether people assigned to the bypass complete checkout more often.

## Where to go next

For API details, see [priors](../priors.md), [mechanism transport](../guides/mechanism-and-synthesis-transport.md), [decision contracts](../decisions.md), [study ranking](../design-ranking.md), with [external signals](../design-ranking.md#external-signals) for the `ExternalSignal` route used in step 5, and [from a failed identification to a ranked evidence plan](../design-ranking.md#from-a-failed-identification-to-a-ranked-evidence-plan) for the `antecedent.repair` interface. [Beyond the marketing mix](beyond-marketing-mix.md) follows a complementary investigation from a fitted response through experimental evidence and the next study.

If the launch also faces bursts of outside activity, [When outside events interrupt a launch](event-driven-launch.md) shows how to compare actions under the same event forecast.

## Run the complete investigation

The example needs no additional dependencies, credentials, network access or confidential content. It generates all observations, keeps the generating truth separate, and runs both paths in about two seconds locally. The pilot counts are illustrative seeded realizations; other observations can lead to more testing or to withholding launch.

<!-- example: examples/python/synthetic_users_launch.py -->

[Open the Python script](../../examples/python/synthetic_users_launch.py)
