# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

**Antecedent is a causal inference compiler for research and for building causally aware software and agents.** Discover causal structure, estimate intervention effects, validate analyses, answer counterfactual questions and transport evidence across populations.

Declare a question, structure or study design, and evidence. Antecedent checks identification and compiles an executable plan with compatible methods, explicit assumptions and evidence requirements. Inspect before execution; reuse with compatible data. Unidentified effects, missing evidence and unsupported methods remain explicit.

Explore results in a notebook or embed causal analysis in applications and agents. Readable reports and machine-readable contracts carry assumptions, diagnostics, uncertainty and limits through reuse, decisions and export. Applications and agents can distinguish supported claims, partial answers and refusals before using a result.

## When statistical significance isn't the decision

Example: A retailer runs a geographic experiment with an advertising platform to measure incremental online and in-store purchases. The result suggests **1,000 additional orders per region per day**, a **200% return on advertising spend**, and positive net profit. But the p-value is **0.07**.

Should the retailer stop advertising because the effect isn't significant at 0.05? Continue because the estimated return is positive? Or run the experiment longer?

Antecedent connects these questions through a checked causal result, a decision contract and expected-value-of-information methods. The result carries its assumptions and uncertainty into the decision; the decision provides the objective for evaluating whether another study is worth its cost.

1. **Establish the causal effect.** Antecedent checks the experimental design, assignment units, estimand, assumptions, and available evidence. It estimates the incremental purchases attributable to advertising, with uncertainty appropriate to the supported design. Geographic dependence and interference cannot simply be ignored; unsupported claims are explicitly refused.
2. **Translate the result into business value.** A decision contract consumes the causal result and defines the available actions—such as continuing or stopping the campaign—and their utilities: incremental contribution margin, advertising costs, and other relevant consequences. Antecedent checks that the result's scientific quantities and uncertainty support the requested comparison of expected net value, losses, or regret. A business decision need not use the same threshold as a hypothesis test.
3. **Make uncertainty explicit.** A positive point estimate is not a guarantee of profit, and p = 0.07 is not a 7% probability that the campaign has no effect. Where justified uncertainty distributions are available, Antecedent can evaluate downside-risk constraints and probabilities of economically important outcomes. Where they are not, it does not invent them.
4. **Ask whether more evidence is worth buying.** Antecedent's expected-value-of-information methods compare the potential improvement in the decision with the cost of another study. If monitoring or early stopping is part of the experimental design, the inference must account for that procedure rather than repeatedly testing until a conventional threshold is crossed.

The right answer might be to continue advertising, stop, or invest in another experiment. **It depends on the evidence, the cost of being wrong, and the value of learning more—not whether a p-value happens to cross 0.05.**

Antecedent preserves the assumptions, uncertainty, and limits of each answer as it moves from causal analysis to decision evaluation and study ranking.

## What backs a supported capability

Antecedent‘s own development applies **high-assurance engineering principles**. Machine-readable evidence and enforced limits govern capability claims:

- **Provenance** names source methods, implementation exposure and deviations that limit published guarantees.
- **Independent checks** use analytic truths, independent fixtures or pinned external implementations. Internal consistency and frozen outputs are labelled separately; component evidence does not establish whole-analysis parity.
- **Licensed combinations** name the question, structure, evidence, inference and estimator supported by executing tests. “Licensed” means evidenced support; unlicensed combinations refuse. 
- **Calibration** records coverage, sample sizes and Monte Carlo error within a method's scope. CI requires measurements to match the code through remeasurement or validated replay.

See [provenance](provenance/README.md), [evidence rules](parity/README.md) and the [support matrix](docs/support-matrix.md).

## Install and use Antecedent

Python **3.11+**:

```bash
python -m pip install --upgrade antecedent
```

Rust **1.85+**:

```bash
cargo add antecedent
```

Python includes NumPy and the CPU-native neural nuisance learner. See [capabilities](docs/capabilities.md) for optional Rust learners and [contributing](CONTRIBUTING.md) for source builds.

Simulate a randomized treatment with an effect of two outcome units:

```python
import numpy as np
import antecedent as ant

rng = np.random.default_rng(17)
z = rng.normal(size=160)
treatment = rng.binomial(1, 0.5, size=160).astype(float)
data = {
    "treatment": treatment,
    "outcome": 2 * treatment + z + rng.normal(size=160),
    "z": z,
}
graph = [("treatment", "outcome"), ("z", "outcome")]
query = ant.AverageEffect("treatment", "outcome")

# Compile and inspect the plan before estimation.
study = ant.prepare(data, graph=graph, query=query, seed=19, bootstrap=25)
print(study.inspect_plan())
result = study.estimate()
print("Answer:", result.answer.kind, result.answer.value)
print("Calibration:", result.calibration.status)
report = result.inspect().to_dict()
print("Identification:", report["identification"]["summary"])

# Reuse the analysis with new data.
new_data = {**data, "outcome": data["outcome"] + treatment}
updated = result.study.refresh(new_data)
print("Updated effect:", updated.answer.value)

# Export and check artifact acceptance.
loaded = ant.load(result.export())
print("Artifact acceptance:", loaded.acceptance.status)
```

Expect an estimate near two; the update adds one. The bootstrap budget is small: check calibration before relying on coverage. Execution checks assumption compatibility, not truth. Loading restores the result and acceptance report, not a live study or dataset.

Continue with the [Python workflow](docs/python-workflow.md), [Rust quickstart](docs/rust-quickstart.md) or [examples](examples/README.md). 

### Antecedent capabilities

| Task | Where to start |
| --- | --- |
| Identify and estimate causal effects | [Supported analyses](docs/supported-analyses.md), [Python options](docs/python-options.md) |
| Analyze experiments, policies, survival and longitudinal regimes | [Examples](docs/examples.md), [design-family support](docs/graphless-support-matrix.md) |
| Estimate effects with cross-fitted nuisance learners | [Capabilities](docs/capabilities.md), [cluster-owned cross-fitting](docs/guides/clustered-dml.md) |
| Transport effects; answer temporal and counterfactual questions | [Population, time and uncertainty](docs/population-time-uncertainty.md), [transport and counterfactuals](docs/transport-counterfactuals.md) |
| Carry a transported value into a decision, inverse query or handoff; revalidate it | [Transported decisions](docs/guides/transported-decisions-and-measurement.md), [inverse and decision reach](docs/guides/inverse-and-decision-reach.md), [assumption range and revalidation](docs/guides/assumption-range-and-revalidation.md) |
| Derive uncertainty of composed, adaptive and decision-reliability results | [Composed uncertainty](docs/guides/composed-uncertainty.md), [adaptive procedures](docs/guides/adaptive-procedure-uncertainty.md), [what these routes do not do](docs/user-facing-limits.md) |
| Make decisions from causal results or external evidence | [Result lifecycle](docs/lifecycle.md), [decisions](docs/decisions.md), [external evidence](docs/external-science.md) |
| Rank studies by objective, information value and cost | [Study ranking](docs/design-ranking.md) |
| Recalculate an analysis and inspect reuse | [Selective recalculation](docs/recalculation-capabilities.md) |
| Explore sensitivity, structural scenarios and composition | [Sensitivity and robustness](docs/sensitivity-and-robustness.md), [composition](docs/composition.md) |

The [support matrix](docs/support-matrix.md) licenses combinations; the [capability inventory](docs/capabilities.md) lists components.

### Read the claim, not just the number

Identification, empirical support and uncertainty stay separate:

- **Identification** depends on the graph, design and evidence. More observations cannot repair a missing causal assumption.
- **Diagnostics** assess data and estimator fit; they do not prove the causal model.
- **Calibration:** `result.calibration.status == "calibrated"` means a record matches the execution and scope, and <!-- generated:claim:calibrated_slot_means_a_current_matching_record:begin -->the record still attests the current code.<!-- generated:claim:calibrated_slot_means_a_current_matching_record:end --> Implemented intervals are not automatically calibrated.
- **Structural and sensitivity ranges** describe variation over assumptions, not sampling confidence intervals. CATE predictions do not imply pointwise confidence intervals.
- **Artifact acceptance** separates independent verification from declarations and operations that cannot be replayed.

The [support matrix](docs/support-matrix.md) separates licenses from coverage measurements. See [guarantees](docs/guarantees.md) and [refusals](docs/refusal-and-partial-knowledge.md) for partial or unavailable answers.

## What's new in 2.4

Scientific composition connects models while preserving meaning, evidence and uncertainty. These goals include implemented and unfinished work:

- **Inputs and mechanisms.** Propagate measurement and latent uncertainty; compose fitted mechanisms with explicit support and dependence.
- **Evidence synthesis and revision.** Prevent double-counting; revalidate claims when evidence changes. Distinguish pooling, borrowing and transport.
- **Scientific bridges.** Connect predictions, proxies and outcomes to causal questions and decisions under declared assumptions, including cohort decisions and broader transported counterfactuals.
- **Sequential decisions.** Evaluate supplied action–observation policies with belief updates; keep observation and intervention distinct.
- **Adaptive experiments.** Represent assignment, delayed outcomes and stopping rules; calibrate the complete procedure.
- **Composition guarantees.** Check compatibility, propagate dependent uncertainty and inspect or replay pipeline evidence.

See the [roadmap](ROADMAP.md#24-direction--broader-scientific-composition) for open work and the [changelog](CHANGELOG.md) for release history.

## Documentation and contribution

The [documentation index](docs/index.md) covers concepts, workflows and APIs. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute; commits require Developer Certificate of Origin sign-off.

## License

MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
