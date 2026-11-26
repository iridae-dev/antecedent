# From day zero to publication: plan the evidence, then earn the claim

When an AI gives bad advice, can an explanation help people catch the mistake? A researcher designs an interface that shows an explanation and prompts people to review the recommendation. If people make better decisions, either change might deserve the credit.

You’ll plan two experiments. The first compares the redesigned interface with the original. The follow-up varies the explanation and the prompt independently, so you can estimate how much the explanation contributes.

Both experiments use simulated participants. At the end you’ll write up three results from their task records, including how much longer people took and how uncertain the explanation’s effect remains.

Run `python examples/python/research_from_day_zero.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/research_from_day_zero.ipynb). See the [complete example](#run-the-complete-example).

The sample-size calculation uses statistical power for a difference in means.

## The research question

The proposed explanation interface adds two things: explanatory information and a prompt encouraging users to review the recommendation. The initial study compares that complete interface with a baseline. Participants complete 20 tasks containing incorrect AI recommendations and 20 containing correct recommendations.

The primary outcome is each participant's proportion of correct decisions on the incorrect-AI tasks. Mean time per incorrect-AI task and accuracy when the AI is correct are secondary outcomes. Voluntary verification behavior is also recorded.

| Stage | Question | Concrete research output |
| --- | --- | --- |
| Before the first participant | What evidence would answer the primary question? | A protocol, sample-size plan and simulation |
| After the first study | What remains unresolved, and which study would resolve it? | An identification check, study ranking and amended protocol |
| Before submission | Which proposed claims are supported by which results, and which methods produced them? | A claim audit, bounded abstract, methods citations and reproducible record |

The population is the recruited cohort of eligible English-speaking desktop volunteers, performing the specified tasks. Randomizing the interface supports an assignment effect in that cohort. It does not automatically extend the result to every AI user, task or device.

## Beginning: design the study your claim needs

### Replace an attractive claim with an answerable question

“Explanations improve appropriate trust” is an appealing aim, but accepting recommendations, answering correctly, inspecting an explanation and trusting a system are different quantities.

Declare the primary target more narrowly:

> What is the mean effect of assignment to the complete explanation interface on participants' accuracy when AI advice is incorrect?

Randomize **participants**, with fixed arm sizes. Keep their individual task records, then pass one accuracy summary per participant to the primary estimator. Twenty repeated task responses do not make one person into 20 independent participants.

The example records the outcome, target, population, randomization, secondary outcomes, analysis and fixed enrollment in a protocol object. These declarations determine what the first study is intended to establish.

### Plan, then inspect the assumptions behind the plan

Use the experimental planner before collecting responses:

```python
from antecedent.design import planning

plan = planning.plan_experiment(
    planning.Complete(),
    planning.Assumptions.continuous(sd=0.16, effect=0.08),
    planning.Power(0.8),
    planning.SampleSize(),
    simulation=planning.Simulation(replicates=512, seed=17),
)
```

Here 0.08 is the effect size the researcher wants to detect, and 0.16 is the assumed standard deviation of participant-level accuracy. They are planning assumptions, not estimates from data that have not been collected. The continuous-outcome planning model approximates those participant proportions; the worked task generator later produces bounded binary task responses.

The example computes:

| Planning assumptions | Participants to recruit |
| --- | ---: |
| SD 0.16, no attrition | 128 |
| SD 0.20, no attrition | 199 |
| SD 0.16, 20% attrition | 160 |

The baseline plan assigns 64 participants per arm. Its analytic power is about 0.801; the simulation gives 0.803 with Monte Carlo standard error 0.018. The script checks their agreement and verifies that one participant fewer falls below the requested analytic target.

The power calculation tells you how often this design would detect the assumed effect. Look at the sensitivity table before setting a recruitment target. If participant accuracy varies more than you expect, the same number of participants gives you less power.

### Preserve the original research commitment

The protocol includes the native planning-request identity. Application code serializes and hashes the protocol before generating any observed study data. A later amendment receives a new identity and names its parent; it does not overwrite the first protocol.

The hash lets another researcher verify which protocol you used. To document when you committed to that protocol, deposit it with a registration service before collecting data.

The [planning guide](../guides/power-and-sample-size.md) covers precision, assurance, attrition and other supported designs.

## Middle: use the findings to choose the next question

### Analyze the intervention you actually randomized

The complete interface is assigned at participant level:

```python
assignment = CompleteRandomization(treated=64)
design = ExperimentDesign(
    assignment,
    realized_assignment=assigned.tolist(),
    assignment_units=ids,
    outcome_units=ids,
    treatment_arms=("baseline", "explanation_interface"),
)
prepared = ant.prepare(
    rows,  # one accuracy summary per participant
    query=RandomizedEffect("accuracy", design),
    execution=ant.Execution(seed=SEED),
)
assert prepared.inspect().stage == "prepared"
result = require_success(ant.estimate(prepared))
```

The example independently checks the native effect against the difference of participant means, and its variance against the participant-level Neyman construction. All analyses of the first study refer to the same participant source in the evidence ledger. Reanalyzing its outcomes does not create additional independent participants.

The simulated first study produces these pointwise results:

| Outcome | Estimated assignment effect | 95% interval |
| --- | ---: | --- |
| Accuracy on incorrect-AI tasks | +8.52 percentage points | +4.71 to +12.32 |
| Mean time per incorrect-AI task | +6.37 seconds | +5.64 to +7.09 |
| Accuracy on correct-AI tasks | −1.17 percentage points | −3.10 to +0.76 |

The interface improves the primary accuracy outcome in this fixture and takes longer. The correct-AI result does not establish an absence of harm. The secondary intervals are pointwise, not a simultaneous guarantee across every endpoint.

### Separate an interface effect from a mechanism claim

Two explanations remain plausible: the information helps people reason, or the review prompt makes them pause. The original comparison changed both together. Increasing its sample size can sharpen the bundle effect without separating the components.

For the component question, the researcher declares a projected graph with `content → accuracy` and a shared cause represented by `content ↔ accuracy`. The latter captures the other changes bundled with content in the original design. This is a declared causal model, not a graph Antecedent discovers from the outcomes.

`repair.HedgeContract` exposes the need for an intervention that separates content. The example checks proposed evidence against that contract:

| Proposed work | What it adds for the content question | Native classification |
| --- | --- | --- |
| More participants in the original study | More observations of the same bundled regimes | Insufficient |
| Add a perceived-understanding questionnaire | A further measured variable | Insufficient |
| Independently randomize content and review prompt | A content intervention across balanced prompt conditions | Verified sufficient |

The factorial study randomizes participants to four interfaces. They see the original interface, the explanation alone, the prompt alone, or both. These comparisons can isolate the contribution of each change.

“Verified sufficient” means the declared evidence would identify the specified contrast under the graph and design assumptions. It does not promise a narrow interval, a positive effect or a valid psychological explanation. The study must still implement its declared assignment, recruitment, measurement, consistency and no-interference assumptions.

### Rank useful follow-ups by what they would inform

Suppose the researchers also need to decide whether to retain information in the next prototype. Both alternatives keep the review-prompt regime fixed. Retaining information has utility `content_effect − 0.03`; omitting it has utility zero. The 0.03 threshold converts the additional component's cost and burden into accuracy-equivalent utility.

For planning, declare a normal belief about the content effect with mean 0.04 and SD 0.04. This is a researcher-supplied judgment spanning small, absent and larger effects. It is not an identified component posterior extracted from the bundle comparison.

The ranker evaluates 128- and 256-person factorial studies under a Gaussian working signal for the estimated content contrast. Its observation variance is `4 × 0.16² / N`, corresponding to a balanced marginal contrast. A session costs 0.00003 accuracy-utility units.

```python
ranking = design.rank_designs(
    candidates,
    decision=declared_component_decision,
    signal=component_signal,
    cost_map=design.CostMap("accuracy_utility", "accuracy_utility", 1.0),
)
```

The actual ranking is:

| Study | Gross information value | Net value |
| --- | ---: | ---: |
| Factorial, 128 participants | 0.00864 | **0.00480** |
| Factorial, 256 participants | 0.00983 | 0.00215 |

The larger study supplies more information, but the smaller has higher net value under this decision and cost map. The script checks both information values against an independent normal-theory calculation, checks cost subtraction, and replays the exported ranking with its retained identity.

This objective is explicit. Researchers seeking a particular precision or a well-powered interaction test should plan for that objective instead. The ranking is not a recommendation to maximize the chance of a significant paper, and the main-effect plan does not establish adequate power for every secondary contrast.

### Amend the protocol before running the follow-up

The new protocol names the first protocol as its parent and declares a new primary target: the content main effect averaged over balanced prompt conditions. It retains the planning prior, selected sample size, assignment and primary contrast before generating the new cohort.

The follow-up recruits 128 new participants, 32 per factorial cell. It estimates a content effect of **+3.83 percentage points**, with a 95% interval of **−0.25 to +7.90**. The example independently reconstructs the content effect, prompt effect and interaction from the four cell means.

The 95% interval for content alone runs from a slight loss to a gain of nearly eight percentage points. The study separates the explanation from the prompt, but that range still leaves room for both no benefit and a practically important one.

## End: write the paper your evidence supports

### Give every proposed claim an evidence requirement

The example ends with an application-level claim manifest. Each proposed statement names the relevant outcome, result, protocol or identification check. Antecedent supplies the statistical results and identification verdicts; application code connects them to the claims the researcher has explicitly listed.

| Proposed statement | Audit result | Why |
| --- | --- | --- |
| The assigned interface improved accuracy in the studied tasks | Supported | The randomized primary contrast and its interval support that direction |
| The factorial study estimates a separate content effect | Supported | Content and prompt were randomized independently |
| Content independently improved accuracy | Inconclusive | The content interval includes zero |
| The benefit occurred because people inspected and understood the AI | Unsupported | Randomizing components does not identify voluntary inspection's effect; understanding was not measured |
| The interface did not slow participants down | Contradicted | The observed time increase and interval indicate a cost |
| There was no accuracy loss when AI advice was correct | Not established | The secondary interval includes losses, and no noninferiority margin was declared |
| The results establish improved trust or a benefit for all AI users | Unsupported | Construct and population-generalization arguments are missing |

The mechanism check is executable. A second `HedgeContract` models an unobserved common cause of voluntary inspection and accuracy, while allowing component effects through other paths. The factorial study does not randomize inspection, so the native repair check remains insufficient for that mediator claim. Randomized components can answer a component question while leaving a proposed behavioral mechanism unresolved.

The researcher lists the claims to check. As the manuscript changes, update that list and rerun the audit so that each statement can be traced to the evidence behind it.

### Produce a bounded abstract

The script assembles a numerical abstract from the retained results. Its substance is:

> In the recruited cohort and tested tasks, assignment to the bundled interface changed accuracy by 8.52 percentage points, with a 95% interval from 4.71 to 12.32, and added 6.37 seconds per task. The independent factorial follow-up estimated a content effect of 3.83 percentage points, with an interval from −0.25 to 7.90.

The accompanying record retains the unresolved mechanism and generalization questions. It preserves the positive primary finding, the time cost and the inconclusive component result rather than selecting only the favorable parts of the investigation.

### Cite the methods behind every number

For each number in the paper, cite the method that produced it. Read `outcome.method_provenance` for the randomized-design identification record and the Neyman difference-in-means estimator record. Each includes the published sources, documented deviations, limits of the method review, and the record’s ID and digest.

The citations travel with the results. Every audit row that rests on a result stores the record ids and digests of that result next to its effect and interval, so the claim table and the methods section cannot drift apart. One call then collects the bibliography of the whole investigation:

```python
from antecedent.citations import bibliography, capability_provenance

references = bibliography(
    [pilot_outcome, time_outcome, correct_outcome, follow_outcome],
    capability_provenance("design.evsi"),  # the ranking that chose the follow-up
)
for citation in references:
    print(citation.reference)
    for caveat in citation.caveats:
        print(f"  - {caveat.text} ({caveat.record})")
```

The four randomized analyses share one estimator source, so it is listed once with every record that cites it. The study ranking is a design capability rather than an analysis route, so it is cited by its capability id. The bibliography names Splawa-Neyman, Dabrowska and Speed (1990) for the participant-level contrasts, and Raiffa and Schlaifer (1961) and Berger (1985) for the expected value of sample information.

Each citation comes with the caveats its records document, and they belong in the paper. For this study three matter: the interval uses the conservative Neyman variance bound, which omits the unobservable variance of individual effects; the pointwise normal interval needs at least 30 assigned participants per arm, which both studies meet; and the value of the follow-up is an exact closed-form point value under the declared design prior, with no interval. A methods paragraph built from that record reads:

> Effects of interface assignment were estimated as differences in participant-level means under complete randomization, with the conservative Neyman variance bound (Splawa-Neyman, Dabrowska and Speed, 1990), as implemented in Antecedent (provenance record `estimate.randomized.neyman_itt`). The factorial follow-up used the fixed-cell 2×2 contrasts of the same estimator. Its size was chosen by the expected value of sample information under a declared Gaussian design prior (Raiffa and Schlaifer, 1961; Berger, 1985; record `design.evsi`).

Report each record's digest beside its id; the script prints both. The digest changes when a record's sources or deviations change, so a reader can tell which version of the method record the paper relied on.

Provenance is descriptive: it changes no number, support status or calibration annotation. It cites the methods, not the researcher's choices; the protocol snapshots, the design prior and the claim audit are still the researcher's to report. See [cite the methods behind a study](../guides/citing-a-study.md).

### Retain the scientific record

The output includes both protocol identities, the native planning identity, disjoint cohort sources, the claim audit with each row's method records, the supported abstract and the methods bibliography. The ledger resolves **256 distinct participants** across the two studies. Multiple outcomes and repeated tasks do not inflate that count.

Both randomized results export and reload against their own independent expectations; attempting to load the component result under the original bundle result's expectation refuses. The repair and ranking artifacts also replay. Each exported result also carries its provenance records, so whoever receives a result receives its citations with it: `ant.load(...).citations()` on the receiving side lists the same sources and caveats. These artifacts preserve their supported scientific objects, not a complete manuscript or a replacement for retaining the study data and code. The ledger's combined provenance is not a pooled estimate: the bundle effect and component effect remain different targets.

## What this established

The first experiment supports a claim about the complete interface. Participants caught more incorrect recommendations and spent longer on each task. The follow-up lets the paper report a separate estimate for explanatory content, with an interval wide enough to include zero.

The protocol, participant records and method citations show readers how you reached each conclusion.

## What this did not establish

The first study did not separate explanatory content from the review prompt. The follow-up identified the content contrast, and its interval includes zero. Nothing here identified the effect of voluntary inspection, measured trust or understanding, or extended the results beyond the recruited cohort and tasks. The protocol identities show which specification was retained, not that it was registered with an outside party, and every number comes from a synthetic generator.

## Where to go next

For details, see [experimental planning](../guides/power-and-sample-size.md), [randomized and factorial designs](../guides/designs/randomized-experiments.md), [study ranking](../design-ranking.md), [the evidence ledger](../evidence-ledger.md), [citing the methods behind a study](../guides/citing-a-study.md) and [results as claims](../concepts/results-are-claims.md). [The smallest defensible test](synthetic-users-launch.md) follows a complementary investigation centered on a launch decision.

## Run the complete example

The example runs offline with Antecedent and NumPy, needs no credentials or real participant data, and completes in under a second locally. It keeps the hidden generating mechanism separate from the estimator inputs, checks its arithmetic independently and prints the research record as JSON.

<!-- example: examples/python/research_from_day_zero.py -->

[Open the Python script](../../examples/python/research_from_day_zero.py)
