# The smallest defensible test: from synthetic users to a launch decision

A product team has recent human clickstreams from its existing website and thousands of agent sessions on both the existing and redesigned sites. The redesign is confidential. The team wants to learn enough from a small human pilot to decide whether to launch. Causal inference helps separate what the agents predict from what would justify transferring that prediction to humans—and helps choose the next experiment when the prediction fails.

The useful proposition is **historical human evidence + a large synthetic dataset + a small human test can support the decision that otherwise needs a much larger new human test**. This tutorial executes that comparison. Antecedent carries the declared prior, checks transport requirements, evaluates the launch contract, values further observations and checks which follow-up designs could identify the unresolved effect.

In the successful branch, 32 newly exposed humans support the same launch choice as a separate 128-person human-only comparator. In the conflicting branch, 16 humans are enough to withhold launch and change the scientific question: an eight-person authentication experiment is worth more than another undifferentiated sample.

These are decisions under explicit models, not interchangeable datasets. The two analyses need not have the same posterior, precision or power. The small test earns its role by informing this decision with the earlier evidence, rather than reproducing every result a large experiment could establish.

## The task and the launch contract

The dataset consists of synthetic clickstream summaries, one independent user or agent per row. Each session attempts navigation, product discovery and checkout microtasks. Each task produces a binary success indicator. These are separately attempted tasks, **not conditional funnel conversion rates**: a failed navigation task does not remove a person from the checkout denominator. Historical sessions also contain a separate authentication-recovery probe.

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

The application exposes three workflow outcomes: **launch**, **investigate**, and **do not launch**. Antecedent evaluates the terminal launch-versus-withhold comparison. The predeclared application controller distinguishes an unresolved question worth investigating from a version the evidence says to abandon. That controller is application code, visible in the example.

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

In this fixture, 16 is the best initial cohort in that catalog and has positive net value. Eight is too small to be worth its cost for this launch decision; 32 costs more than its initial information is worth. The initial cohort has an interim safety look at eight. It is a committed 16-person study unless a conflict requires an earlier pause; the eight-person look is not presented as an independently sufficient launch study. After the initial cohort, the controller can request the next declared batch or stop.

This is the sense in which the example seeks the smallest defensible test: it compares declared studies for a declared decision, under exposure costs and a predeclared protocol. It does not prove a globally smallest sample size across every possible experiment or stopping policy.

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

At 16, a favorable observed rate alone is insufficient: the contract still withholds launch. At 32, the risk rule is met, and another 32-person batch has net information value about `−0.142` utility units. Further full-site exposure is not justified by that candidate's value for this decision.

Now make the comparison that motivates the tutorial:

| Analysis | New human exposure | Evidence behind the belief | Probability of meeting the floor | Terminal choice |
| --- | ---: | --- | ---: | --- |
| Small human sample alone | 32 | Broad probability prior + the same 29 successes | 0.869 | Withhold |
| Guided small test | 32 | Historical humans + agents + those 29 successes | 0.974 | Launch |
| Fixed larger human-only comparator | 128 | Broad probability prior + 113 successes | 0.967 | Launch |

The larger comparator is a separate, predeclared reference study. It shares the first 32 observations and adds independent humans from the same synthetic population. It is not a universal sample-size recommendation. The human-only grid calculation is independently checked against the exact Beta-posterior tail, using its binomial-CDF identity.

The earlier evidence changes what the small sample can establish **for the decision**. It supports the same launch action as the larger comparator while exposing one quarter as many humans to confidential content. It does not make the 32-person and 128-person datasets statistically equivalent.

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

The application reports **investigate**: launch is withheld, and an actionable causal question remains. The `do_not_launch` branch is reserved for a clear regression with little remaining support for a useful authentication remedy. This example executes agreement and investigation; it does not manufacture a third observed dataset solely to exercise every workflow label.

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

## What the example establishes

The successful path uses a large agent dataset and historical human observations to make a much smaller new human study decision-relevant. Antecedent guides the initial pilot choice, updates the decision as evidence arrives, and helps determine when further confidential exposure is not worth its cost. The comparison with the larger human-only study makes the benefit visible in the action taken, without equating the underlying evidence.

The failed-transfer path uses the small pilot to ask a better next question. A prior conflict reduces trust in the synthetic prediction; an identification check distinguishes a targeted intervention from more of the same observations; information value and exposure cost select the next study.

The application supplies the finite hypothesis family, binomial likelihoods, numerical prior projection and checkpoint controller. Antecedent supplies the prior-composition, mechanism-transport, decision, identification-repair and study-ranking operations they call. The complete example makes that boundary executable.

**Synthetic data helps determine which real-world evidence is necessary. It does not replace the need for that evidence.**

## Run the complete investigation

Run from a checkout with the 3.0 Python extension installed:

```bash
python examples/python/synthetic_users_launch.py
```

The example needs no additional dependencies, credentials, network access or confidential content. It generates all observations, keeps the generating truth separate, and runs both paths in about two seconds locally. The pilot counts are illustrative seeded realizations; other observations can lead to more testing or to withholding launch.

<!-- example: examples/python/synthetic_users_launch.py -->

For API details, see [priors](../priors.md), [mechanism transport](../guides/mechanism-and-synthesis-transport.md), [decision contracts](../decisions.md), [study ranking](../design-ranking.md), and the public `antecedent.repair` interface. [Beyond the marketing mix](beyond-marketing-mix.md) follows a complementary investigation from a fitted response through experimental evidence and the next study.
