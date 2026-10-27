# Assurance

Antecedent's development applies high-assurance engineering principles. Method records name sources, implementation choices and deviations that limit published guarantees. Capability claims are tied to executing tests, with analytic checks, external comparisons, internal consistency and replay evidence labelled for what each establishes. Licensed support and calibration have explicit scopes; CI checks whether their evidence still applies to the code. In supported workflows, results carry their assumptions, uncertainty and limits through reuse, export and downstream decisions.

This describes an engineering approach. It is not formal verification of the whole system, not a certification, and not proof that your causal model is true.

## Four kinds of evidence

These are different kinds of evidence. They do not combine into a single "validated" badge.

**Provenance.** Each method record names its source, its exposure to other implementations and the material deviations from it, and says which published conclusions transfer and which do not. For example, a route without cross-fitting does not inherit the guarantees of cross-fitted double machine learning, and a recorded comparison with another library's weighting run does not supply an external check of an augmented-weighting estimator. Records are in [provenance](https://github.com/iridae-dev/antecedent/blob/main/provenance/README.md).

**Independent checks.** A check states its proposition and its reference: an analytic truth, an independently constructed fixture, or a pinned external implementation. Internal consistency and frozen Antecedent outputs are labelled separately, and a comparison of one component establishes that component's behavior. See [parity](https://github.com/iridae-dev/antecedent/blob/main/parity/README.md).

**Licensed combinations.** A combination of question, structure or design, evidence, inference and estimator is licensed when it has executing support evidence. The [support matrix](../support-matrix.md) lists them, and [support lookup](../support-lookup.md) finds the row for your request. The support status of the route that actually ran stays on the result.

**Calibration.** A calibration record states the uncertainty construction, the tested setting, sample sizes, coverage and Monte Carlo error. A result's calibration status depends on a matching record that still applies to the executing code. CI checks that recorded measurements still match the code; it does not rerun every simulation on every change. See [uncertainty](uncertainty.md#reading-calibration-status).

## Claim vocabulary

The same words are used across the documentation, with the meanings in [results are claims](results-are-claims.md#the-standing-vocabulary): implemented, executes, identified, model conditional, licensed, independently checked, calibrated, replayed, accepted artifact, current authority, numerically certified. Avoid reading "verified" or "validated" without asking what was checked, by what, and under which conditions.

## Capability, licensed support and scientific validity

Three propositions stay separate.

| Term | What it means | What it does not mean |
| --- | --- | --- |
| Capability | An implementation or type exists | Every combination using it is permitted |
| Licensed support | A specific question, structure, inference and estimator route has a recorded execution and evidence contract | The real-world causal assumptions are true |
| Scientific validity | The assumptions are appropriate for the real situation | Something software can establish from a successful call |

Licensed does not mean measured. For each licensed row the support matrix states whether a coverage record is cited, whether the estimator has no coverage measurement, or whether the row reports no interval.

## What assurance does not cover

- Whether your assumptions hold in the real system. Software cannot establish that from a successful call.
- Combinations without executing evidence. Capabilities pages inventory what exists; the support matrix says what is licensed.
- Statistical coverage where no matching record exists.
- Resource admission is not a measurement of process-wide memory.

## Where it shows up for you

Results report their identification, support, uncertainty, assumptions and calibration status in `result.inspect()`. Exported artifacts carry identities and acceptance information. Refusals state which premise was not met. Those fields are the interface to the assurance machinery; you rarely need to read the records themselves.
