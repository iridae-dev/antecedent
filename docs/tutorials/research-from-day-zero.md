# From day zero to publication: plan the evidence, then earn the claim

An HCI researcher wants to know whether explanations help people make better decisions when an AI recommendation is wrong. Causal inference matters before recruitment: what should change between interfaces, which outcomes would answer the question, and which study could distinguish a useful interface from a convincing explanation of why it works?

This tutorial follows one investigation through three stages. Antecedent helps plan the first study, choose a follow-up when its results leave an important question unresolved, and check the claims proposed for the paper. The researcher supplies the scientific aims and assumptions; the computation makes their consequences inspectable.

The worked example is fully synthetic. It preserves a protocol before generating observations, analyzes a randomized interface comparison, selects and analyzes an independent factorial follow-up, and produces an evidence-linked claim table and a supported abstract. The final results include an inconclusive component effect: a useful investigation need not end with every hypothesis confirmed.

## The research question

The proposed explanation interface adds two things: explanatory information and a prompt encouraging users to review the recommendation. The initial study compares that complete interface with a baseline. Participants complete 20 tasks containing incorrect AI recommendations and 20 containing correct recommendations.

The primary outcome is each participant's proportion of correct decisions on the incorrect-AI tasks. Mean time per incorrect-AI task and accuracy when the AI is correct are secondary outcomes. Voluntary verification behavior is also recorded.

| Stage | Question | Concrete research output |
| --- | --- | --- |
| Before the first participant | What evidence would answer the primary question? | A protocol, sample-size plan and simulation |
| After the first study | What remains unresolved, and which study would resolve it? | An identification check, study ranking and amended protocol |
| Before submission | Which proposed claims are supported by which results? | A claim audit, bounded abstract and reproducible record |

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

The simulation generates studies under the declared planning model and applies the planned analysis. It is not evidence that the assumed effect or variability describes future participants. The sensitivity table makes the recruitment consequences of those assumptions visible before they become an expensive surprise.

### Preserve the original research commitment

The protocol includes the native planning-request identity. Application code serializes and hashes the protocol before generating any observed study data. A later amendment receives a new identity and names its parent; it does not overwrite the first protocol.

This is a useful preregistration input, not proof of external preregistration. A real investigation must deposit its protocol through the researcher's chosen registration process. A content identity establishes which specification was retained; it does not establish when an outside party first received it.

The [planning guide](../guides/power-and-sample-size.md) covers precision, assurance, attrition and other supported designs.

## Middle: use the findings to choose the next question

### Analyze the intervention you actually randomized

The complete interface is assigned at participant level:

```python
assignment = CompleteRandomization(treated=64)
design = ExperimentDesign(
    assignment,
    realized_assignment=assigned.tolist(),
    assignment_units=participant_ids,
    outcome_units=participant_ids,
)
prepared = ant.prepare(
    participant_data,
    query=RandomizedEffect("accuracy", design),
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

The questionnaire is not assigned a causal or construct-validity role merely because its name sounds relevant. The factorial study supplies four randomized cells: neither component, content alone, prompt alone and both.

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

This study identifies the component contrast, but the interval includes zero. The correct scientific outcome is an estimate with consequential uncertainty, not a forced confirmation. A future study can build on that result; it cannot turn an exploratory question from the first study into an originally preregistered primary hypothesis.

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

The audit does not read arbitrary manuscript prose or certify a paper automatically. Its value is that a researcher can make the evidence requirements explicit and repeat the checks as the manuscript, evidence or proposed wording changes.

### Produce a bounded abstract

The script assembles a numerical abstract from the retained results. Its substance is:

> In the recruited cohort and tested tasks, assignment to the bundled interface changed accuracy by 8.52 percentage points, with a 95% interval from 4.71 to 12.32, and added 6.37 seconds per task. The independent factorial follow-up estimated a content effect of 3.83 percentage points, with an interval from −0.25 to 7.90.

The accompanying record retains the unresolved mechanism and generalization questions. It preserves the positive primary finding, the time cost and the inconclusive component result rather than selecting only the favorable parts of the investigation.

### Retain the scientific record

The output includes both protocol identities, the native planning identity, disjoint cohort sources, the claim audit and the supported abstract. The ledger resolves **256 distinct participants** across the two studies. Multiple outcomes and repeated tasks do not inflate that count.

Both randomized results export and reload against their own independent expectations; attempting to load the component result under the original bundle result's expectation refuses. The repair and ranking artifacts also replay. These artifacts preserve their supported scientific objects, not a complete manuscript or a replacement for retaining the study data and code. The ledger's combined provenance is not a pooled estimate: the bundle effect and component effect remain different targets.

## What this established

Before recruitment, Antecedent made the study's assumptions and resource requirements inspectable. After the first results, it helped distinguish an unresolved causal question from a need for more precision, checked which designs could answer it, and valued the useful candidates. Before publication, its results supported an audit that separated supported, inconclusive, contradicted and unsupported statements.

The investigator still chooses the questions, models, measures and interpretation. The tutorial's application code supplies protocol snapshots and the explicit claim manifest. Together, those pieces keep the research connected from the first plan to the claims in the paper.

## Run the complete example

```bash
python examples/python/research_from_day_zero.py
```

The example runs offline with Antecedent and NumPy, needs no credentials or real participant data, and completes in under a second locally. It keeps the hidden generating mechanism separate from the estimator inputs, checks its arithmetic independently and prints the research record as JSON.

<!-- example: examples/python/research_from_day_zero.py -->

For details, see [experimental planning](../guides/power-and-sample-size.md), [randomized and factorial designs](../guides/designs/randomized-experiments.md), [study ranking](../design-ranking.md), [the evidence ledger](../evidence-ledger.md), and [results as claims](../concepts/results-are-claims.md). [The smallest defensible test](synthetic-users-launch.md) follows a complementary investigation centered on a launch decision.
