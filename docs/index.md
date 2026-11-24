# Antecedent

Antecedent is a compiler and runtime for causal science, for Python and Rust.

Declare a causal question, a model or study design, and the evidence. Antecedent checks what the question needs, compiles an analysis you can inspect, and executes a supported route. The answer carries its assumptions, uncertainty, provenance and limits into the comparison, decision or software that uses it. A question the declarations cannot answer returns a typed [refusal](concepts/refusals.md) that names what is missing.

Antecedent's development applies high-assurance engineering principles, with explicit evidence governing capability claims. [Assurance](concepts/assurance.md) describes what that covers.

## A first answer

```bash
python -m pip install --upgrade antecedent   # Python 3.11 or later
cargo add antecedent                         # Rust 1.97 or later
```

Then prepare, inspect and estimate an effect in simulated data where the true answer is known. [Your first analysis](start/first-analysis.md) runs it in Python, branches it onto changed data, exports it, and shows the Rust form. [Installation](start/installation.md) covers platforms, optional features and source builds.

## Choose your path

**Research and scientific decision-making.** Ask a causal question, inspect the assumptions and evidence it needs, and carry the answer into the next comparison, decision or study. Begin with [how to read a result](start/reading-a-result.md), then the [observational study tutorial](tutorials/observational-study.md). When you write up a study, [cite the methods behind it](guides/citing-a-study.md): every result names the published sources its route implemented and where the implementation departs from them.

**Building software and agents.** Embed causal workflows with inspectable plans and machine-readable assumptions, uncertainty and limits. Your application supplies planning, permissions and the policy for using an answer. Begin with the [application component tutorial](tutorials/application-component.md) and [embedding](guides/build/embedding.md).

## Tutorials

Each tutorial follows one scientific task and pairs with a maintained [example](examples.md).

| Tutorial | What you do |
| --- | --- |
| [One feedback system, three questions](tutorials/feedback-three-questions.md) | Observe, fix and shock a variable in a feedback system |
| [Compare a coordinated intervention program](tutorials/joint-interventions.md) | Compare a coordinated intervention program with a baseline |
| [Beyond the marketing mix](tutorials/beyond-marketing-mix.md) | Start with a marketing mix model, compare spending programs, incorporate experimental evidence, make a decision and rank the next studies |
| [The smallest defensible test](tutorials/synthetic-users-launch.md) | Combine historical humans, synthetic users and a small pilot to decide whether to launch or investigate |
| [Write and replay causal policies](tutorials/programmable-policies.md) | Write conditional and history-dependent pricing and inventory policies, then replay the same stores |
| [From day zero to publication](tutorials/research-from-day-zero.md) | Plan an HCI study, choose follow-up research, check the claims in the paper and cite the methods behind them |
| [From a new experiment to the next study](tutorials/experiment-to-next-study.md) | Update a response from an experiment and choose the next study |
| [Stop, continue or run longer](tutorials/decide-under-uncertainty.md) | Decide when a result is suggestive but not significant |
| [Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md) | Decide with a trade-off stated as a range, read where the choice switches, and optionally narrow the range from past decisions |
| [Replay and current decisions](tutorials/replay-and-current-decisions.md) | Change a utility, export a decision, withdraw a source |
| [Estimate an effect from observational data](tutorials/observational-study.md) | Analyze a real public dataset end to end |
| [When the comparison group does not overlap](tutorials/when-overlap-fails.md) | Follow a refusal when the comparison group does not overlap, and see how far the answer moves |
| [When the causal assumptions stop holding](tutorials/when-assumptions-change.md) | Follow a claim when new data stop fitting its graph: check, rediscover, compare readings and plan the study that would settle it |
| [Use an analysis inside an application](tutorials/application-component.md) | Use Antecedent as a component of an application |
| [When outside events interrupt a launch](tutorials/event-driven-launch.md) | Forecast clustered announcements, compare launch actions under the same arrivals and value another hour of observation |

## Guides

Guides are grouped in the navigation by the question you have. Each row names the group and a page to start from.

| Your question | Guide group | Start with |
| --- | --- | --- |
| How do I bring my table, files, weights or missing values? | Data and inputs | [Bring your own table](guides/your-data.md), [missing values](guides/missing-data.md) |
| Which question and estimator fit my data, and how do I check the answer? | Estimate effects | [Choose a question and an estimator](supported-analyses.md), [probe sensitivity to assumptions](sensitivity-and-robustness.md) |
| How do I analyze an experiment, a panel, a discontinuity or a survival outcome? | Experiments and study designs | [Randomized and factorial experiments](guides/designs/randomized-experiments.md), [panel designs](guides/designs/panel-designs.md) |
| I do not know the graph, or I am not sure of it. | Structure and discovery | [Discover, then estimate](selective-discovery.md), [check a graph against data](guides/graph-falsification.md) |
| How do I fit mechanisms, ask counterfactuals or model feedback? | Structural models and feedback | [Fit and query a structural causal model](guides/gcm-estimands.md), [feedback loops and equilibria](guides/cyclic-systems.md) |
| What happens when several things change together? | Joint interventions and search | [Fit and compare intervention programs](guides/joint-intervention-responses.md) |
| Does this effect hold in another population? | Transport | [Carry an effect to another population](guides/transport/carry-an-effect.md), [when transport fails](guides/transport-failure.md) |
| Which action should I take, and which study should I run next? | Decisions and study planning | [Evaluate actions with a decision contract](decisions.md), [rank further studies](design-ranking.md) |
| How do I combine external evidence, measurement models and several sources? | Evidence and composition | [External evidence and study synthesis](evidence-ledger.md), [compose sources into a decision](composition.md) |
| How do I inspect a plan, reuse a study, or recover from a refusal? | Prepare, branch and reuse | [Inspect a plan and read a report](guides/plan-view.md), [detect stale results](guides/staleness.md), [recover from refusals](guides/refusal-recipes.md) |
| How do I embed Antecedent in software or exchange results with other tools? | Build with Antecedent | [Embed Antecedent in an application](guides/build/embedding.md), [from a notebook to a system](guides/notebook-to-system.md), [coming from DoWhy](interop_dowhy.md) |

## Concepts

The concept pages explain what a result means and where its limits come from.

- [How Antecedent works](concepts/how-antecedent-works.md): how declarations compile into a program you can inspect, execute and read as a claim.
- [Results are claims](concepts/results-are-claims.md): what an answer asserts and the vocabulary used for it.
- [Interventions and worlds](concepts/interventions-and-worlds.md), [uncertainty](concepts/uncertainty.md) and [refusals](concepts/refusals.md).
- [Assurance](concepts/assurance.md) and [how methods are checked](guides/scientific-verification.md): the evidence behind a supported capability.
- [Glossary](concepts/glossary.md): estimand, world, claim, supported and the other standing labels.

## Look things up

- [Support lookup](support-lookup.md): search whether a question, structure, evidence, inference and estimator combination has executing support evidence, and read the [support matrix](support-matrix.md) behind it.
- [Python API](python-api.md), [Python options](python-options.md) and the [Rust API](https://docs.rs/antecedent/).
- [Refusal codes](reference/refusal-codes.md): refusal families, codes and remedies.
- [Capability inventory](capabilities.md), [naming dictionary](api_naming.md) and [examples](examples.md).

## Release and migration

[Release notes for 3.0.0](release-notes/v3.0.0.md) describe what is new. The [migration guide](migration.md) gives the exact old and new forms.
