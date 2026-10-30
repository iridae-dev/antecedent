# Antecedent

Antecedent is a compiler and runtime for causal science.

Model causal systems, evaluate interventions, and connect evidence to decisions, with assumptions, uncertainty and limits made explicit. Declare a question, a model or study design, and the evidence. Antecedent checks what the question needs, compiles an analysis you can inspect, and executes a supported route. The answer carries its assumptions, uncertainty, provenance and limits into the comparison, decision or software that uses it.

Antecedent's development applies high-assurance engineering principles, with explicit evidence governing capability claims. [Assurance](concepts/assurance.md) describes what that covers.

## A first answer

```bash
python -m pip install --upgrade antecedent
```

Then prepare, inspect and estimate an effect in simulated data where the true answer is known. [Your first analysis](start/first-analysis.md) runs it in Python and in Rust, branches it onto changed data, and exports it. [Installation](start/installation.md) covers platforms, optional features and source builds.

## Choose your path

**Research and scientific decision-making.** Ask a causal question, inspect the assumptions and evidence it needs, and carry the answer into the next comparison, decision or study. Begin with [how to read a result](start/reading-a-result.md), then the [observational study tutorial](tutorials/observational-study.md).

**Building software and agents.** Embed causal workflows with inspectable plans and machine-readable assumptions, uncertainty and limits. Your application supplies planning, permissions and the policy for using an answer. Begin with the [application component tutorial](tutorials/application-component.md) and [embedding](guides/build/embedding.md).

## Tutorials

Each tutorial follows one scientific task and pairs with a maintained example.

| Task | Tutorial |
| --- | --- |
| Observe, fix and shock a variable in a feedback system | [One feedback system, three questions](tutorials/feedback-three-questions.md) |
| Compare a coordinated intervention program with a baseline | [Joint interventions](tutorials/joint-interventions.md) |
| Start with a marketing mix model, compare spending programs, incorporate experimental evidence, make a decision and rank the next studies | [Beyond the marketing mix](tutorials/beyond-marketing-mix.md) |
| Combine historical humans, synthetic users and a small pilot to decide whether to launch or investigate | [The smallest defensible test](tutorials/synthetic-users-launch.md) |
| Write conditional and history-dependent pricing and inventory policies, then replay the same stores | [Programmable causal policies](tutorials/programmable-policies.md) |
| Plan an HCI study, choose follow-up research and check the claims in the paper | [From day zero to publication](tutorials/research-from-day-zero.md) |
| Update a response from an experiment and choose the next study | [From an experiment to the next study](tutorials/experiment-to-next-study.md) |
| Change a utility, export a decision, withdraw a source | [Replay and current decisions](tutorials/replay-and-current-decisions.md) |
| Decide when a result is suggestive but not significant | [Decide under uncertainty](tutorials/decide-under-uncertainty.md) |
| Analyze a real public dataset end to end | [An observational study](tutorials/observational-study.md) |
| Use Antecedent as a component of an application | [Application component](tutorials/application-component.md) |

## Guides

Task guides cover bringing [your own data](guides/your-data.md), [notebook to system](guides/notebook-to-system.md), [estimating effects](supported-analyses.md), [study ranking](design-ranking.md), [sensitivity and robustness](sensitivity-and-robustness.md), [transport and counterfactuals](transport-counterfactuals.md), [decisions](decisions.md), [composition](composition.md) and [staleness and rebinding](guides/staleness.md). The navigation lists the guides by task.

## Concepts

[How Antecedent works](concepts/how-antecedent-works.md), [results are claims](concepts/results-are-claims.md), [interventions and worlds](concepts/interventions-and-worlds.md), [uncertainty](concepts/uncertainty.md), [refusals](concepts/refusals.md) and [assurance](concepts/assurance.md) explain what a result means and where its limits come from.

## Look things up

- [Support lookup](support-lookup.md): search whether a question, structure, evidence, inference and estimator combination has executing support evidence, and read the [support matrix](support-matrix.md) behind it.
- [Python API](python-api.md), [Python options](python-options.md) and the [Rust API](https://docs.rs/antecedent/).
- [Glossary](concepts/glossary.md): estimand, world, claim, licensed and the other standing labels.
- [Capability inventory](capabilities.md), [naming dictionary](api_naming.md) and [examples](examples.md).

## Release and migration

[Release notes for 3.0.0](release-notes/v3.0.0.md) describe what is new. The [migration guide](migration.md) gives the exact old and new forms.
