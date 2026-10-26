# Antecedent

[![CI](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml/badge.svg)](https://github.com/iridae-dev/antecedent/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/antecedent)](https://crates.io/crates/antecedent) [![PyPI](https://img.shields.io/pypi/v/antecedent)](https://pypi.org/project/antecedent/) [![GitHub Release](https://img.shields.io/github/v/release/iridae-dev/antecedent)](https://github.com/iridae-dev/antecedent/releases/latest) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21556247-blue)](https://doi.org/10.5281/zenodo.21556247)

**Antecedent is a causal inference compiler: a library for research and for building causally aware software and agents.** It ships no agent; a host supplies planning and tool use. Discover causal structure, estimate intervention effects, validate analyses, answer counterfactual questions and transport evidence across populations.

Declare a question, structure or study design, and evidence. Antecedent checks identification and compiles an executable plan with compatible methods, explicit assumptions and evidence requirements. Inspect before execution; reuse with compatible data. Unidentified effects, missing evidence and unsupported methods remain explicit.

Explore results in a notebook or embed causal analysis in applications and agents you build. Readable reports and machine-readable contracts carry assumptions, diagnostics, uncertainty and limits through reuse, decisions and export. A versioned JSON Schema and stable refusal codes let a host distinguish supported claims, partial answers and refusals before using a result; see [embedding Antecedent](docs/embedding.md) and the minimal [reference host](examples/rust/reference_host.rs) ([Python](examples/python/reference_host.py)), which contains no agent logic.

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
study = ant.prepare(
    ant.data.table(data), graph=graph, query=query,
    inference=ant.Frequentist(bootstrap=25),
    execution=ant.Execution(seed=19),
)
print(study.inspect())
result = ant.estimate(study)
print("Answer:", result.answer.kind, result.answer.value)
print("Uncertainty:", result.uncertainty.summary)
report = result.inspect().to_dict()
print("Identification:", report["identification"]["summary"])
print(study.report("markdown"))  # the prepared contract and plan

# Reuse the analysis with new data.
new_data = {**data, "outcome": data["outcome"] + treatment}
updated_study = ant.rebind(study, data=new_data)
updated = ant.estimate(updated_study)
print("Updated effect:", updated.answer.value)

# Export and check artifact acceptance.
artifact = result.export()
loaded = ant.load(artifact, expected=artifact.expectation)
print("Artifact acceptance:", loaded.acceptance.status)
```

Expect an estimate near two; the update adds one. The bootstrap budget is small: check calibration before relying on coverage. Execution checks assumption compatibility, not truth. Loading restores the result and acceptance report, not a live study or dataset.

When the declared structure does not identify a valid question, `analyze` returns an `Outcome` with a typed `refusal`. Invalid arguments and operational failures raise errors. Opt in with `on_refusal="describe"` to retain the refusal beside a separately labelled `associational` outcome and the evidence that would resolve it. See [the fallback guide](docs/guides/fallback.md) and [the Python migration guide](docs/migrations/2.4-to-3.0-python.md).

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

## What's new in 2.4.1

More complete workflows for weighted studies, effects after discovery and combining independent evidence:

- **Weighted effects and uncertainty.** Separate population targets, corrected matching and responder variances, finite-cluster references, and explicit arm-linear population interval methods. Each construction keeps its own assumptions and measured scope. See [sampling weights](docs/guides/sampling-weights.md).
- **Pool studies with unequal, unknown variances.** Equal-study Gaussian pooling uses empirical study-level variation and a Student reference for a common marginal mean. Rust provides `pool_gaussian_studies`; Python selects `random_effects(..., method="equal_study_gaussian")`. Request the measured interval through `pooled_interval`. See [evidence synthesis](docs/evidence-ledger.md).
- **Estimate an effect after discovering its parents.** A versioned procedure devotes three quarters of rows to discovery and the remaining quarter to independent analysis. Unresolved edges elsewhere need not prevent an effect estimate when the treatment's parent set is invariant. See [selective discovery](docs/selective-discovery.md).
- **Transport through uncertain structure.** PAG and temporal class transport retain member-level evidence, actual available experiments and replayable proofs. A declared randomized binary trial can refine a static class. With known outcome bounds, Rust also returns a labelled confidence set when the effect remains unidentified; dependent temporal rows use block bootstrap. See [discovered-structure transport](docs/guides/discovered-structure-transport.md).
- **Program interventions and choose fitted mechanisms.** Typed policies support arithmetic, conditions, clipping and history reads across simulation, counterfactuals and temporal rollouts. Boosted and forest mechanism families support declared candidates and validation splits. See [intervention programs](docs/guides/intervention-programs.md) and [mechanism selection](docs/guides/mechanism-selection.md).
- **Check models and work at larger scales.** Graph falsification tests local Markov implications; noise replay and intrinsic influence expose mechanism-level variation. Budgeted searches, wider graph posteriors, matching, sensitivity and participant trials extend earlier size limits with explicit completion diagnostics. See [graph falsification](docs/guides/graph-falsification.md), [noise replay](docs/guides/noise-replay.md) and [intrinsic influence](docs/guides/intrinsic-influence.md).

Read the [2.4.1 release notes](docs/release-notes/v2.4.1.md) for method boundaries and compatibility, the [changelog](CHANGELOG.md) for the full delta, or the [roadmap](ROADMAP.md) for future work.

## Documentation and contribution

The [documentation index](docs/index.md) covers concepts, workflows and APIs.

Task guides for the newer surfaces:

- [Cyclic systems and feedback loops](docs/guides/cyclic-systems.md) and [joint intervention responses](docs/guides/joint-intervention-responses.md)
- [Study ranking and power](docs/design-ranking.md), [staleness and refresh](docs/guides/staleness.md) and the [readable report](docs/guides/report.md)
- [Embedding in applications and agents](docs/embedding.md), [bringing your own table](docs/guides/your-data.md) and [Parquet, Arrow IPC and large tables](docs/guides/large-tables.md)
- [Migration from the previous release](docs/migration/from_previous_release.md) and the [Rust/Python naming dictionary](docs/api_naming.md) See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute; commits require Developer Certificate of Origin sign-off.

## License

MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
