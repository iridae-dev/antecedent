# Glossary

Terms that name what an answer is, what it rests on and what has been checked. Several describe different checks; read them together in `result.inspect()` rather than treating any one as a general guarantee.

## The question

| Term | Meaning |
| --- | --- |
| Estimand | The quantity the question asks for, for a stated population and time. It is the target of identification and estimation |
| Intervention | A declared change to the system: observing, fixing (`do`), shifting a disturbance, or replacing a mechanism. See [interventions and worlds](interventions-and-worlds.md) |
| World | A named set of simultaneous interventions on one model. The unintervened system is a world |
| Cross-world (counterfactual) | A question about the same unit under several worlds that share exogenous state |
| Claim | An answer together with its target, assumptions, evidence, uncertainty and limits |
| Answer kind | The shape of the answer: `point`, `bounds`, `partial`, `response`, `structured`, `policy_value`, `distribution`, `associational`, `refused` or `unavailable` |

## Standing of an answer

| Term | Meaning | Example |
| --- | --- | --- |
| Identified | Declared causal assumptions and evidence determine the requested quantity through an implemented identification route | A backdoor adjustment identifies an average effect under the supplied DAG |
| Model conditional | The result follows from the specified or fitted model, without the design identifying the real-world effect | A fitted joint response, or a cyclic equilibrium under the specified equations |
| Supported | The exact configured route has provenance and an independent scientific check that currently passes, with its recorded scope and limits, so it runs. A route that is not supported refuses in release builds with `not_supported` and names the missing evidence | An estimator supported for the all-observed mean effect refuses another functional |
| Scientific refusal | A typed refusal because the question, structure or evidence is outside what the route can support (`not_supported`, or a more specific reason code). A refusal is not an error in the data | A cell with no supported route refuses with its reason on file |
| Development build | A build that runs a route that is not supported and reports `would_refuse` beside the result, so the result is never mistaken for a supported one | A route still under review runs in a development build and reports why a release build would refuse it |
| Independently checked | A named calculation or proposition was compared with an appropriate independent reference | A closed-form value, or a pinned external implementation |
| Method provenance | The provenance records behind the route a result executed: published sources, documented deviations from the published method, review scope limits, record id and digest. Descriptive; it changes no number or status | `result.method_provenance` and `result.citations()`; see [citing a study](../guides/citing-a-study.md) |
| Calibration status | An annotation on an interval, never a condition for running: `calibrated` (a matching coverage record covers this execution), `outside_measured_scope`, `measured_below_nominal` or `not_measured` | An interval outside the record's sample-size or configuration scope reports `outside_measured_scope` |
| Point-only | A numerical estimate without an interval claim | A finite-law decision value can be exact under the supplied law and still lack a sampling interval |
| Structural envelope | A range across declared structural possibilities | Effects across the completions of a CPDAG; not a confidence interval |
| Assumption range | A range across a declared sensitivity set | Varying an invariance premise changes the transported answer |
| Numerical uncertainty | Error from finite computation or simulation | A Monte Carlo band is not a sampling confidence interval |
| Numerically certified | A named numerical proposition was established within its domain | An optimality bound on a numerical objective under a fitted law; not a statement about the real-world optimum |
| Empirical support | Supplied observations or laws cover the requested coordinates | A response grid keeps an unsupported dose visible rather than deleting it |
| Unidentified mass | Belief (or class weight) on structures where the effect is not identified, kept visible rather than renormalized away | A posterior with 0.2 mass on graphs that do not identify the effect |
| Failed mass | Weight on worlds or draws that could not be solved or evaluated, retained as failed rather than dropped | A clamp with no unique equilibrium |

## Reuse and consumption

| Term | Meaning | Example |
| --- | --- | --- |
| Prepared study | An immutable study that has been validated and identified; nothing is fitted until it is executed | `ant.prepare(...)` |
| Branch | A new immutable study derived from an earlier one by changed data or declarations; the source is unchanged | `ant.rebind(study, data=...)` |
| Retained execution | The fitted state and intermediate work kept so compatible changes can reuse it | Changing a utility without refitting |
| Loaded | A saved result is intact: its contract is present, understood and agrees with its digests, so its recorded answer is kept. Nothing is recomputed, and nothing about the real-world assumptions is checked | `ant.load(bytes(artifact), expected=artifact.expectation)` reports `acceptance.status == "loaded"` |
| Replayed | Opt-in replay recomputed every published part of a loaded result and each agreed. Otherwise replay reports what was not intact, not replayable or a mismatch; a part that is not replayable can still be used | `ant.load(bytes(artifact), replay=True)` reports `acceptance.status == "replayed"` |
| Identity-bound | Consumption matches an expected identity retained independently of the artifact | The consumer uses an expectation the producer saved separately |
| Resumable | An explicit route permits future execution from retained state and required inputs | Loading a result does not restore a live study |
| Current authority | A supported workflow checked the present standing of its sources for current use | A decision whose contributing source has been withdrawn can still be replayed historically but is refused for current use |
| Attested | A caller or provider declares a claim or its provenance, without Antecedent having checked it | An external callback's reported mean |
| Verified | A particular object passed its specified checker. Ask what the receipt verifies | Artifact identity verification does not verify a real-world causal assumption |

## Refusals

| Term | Meaning |
| --- | --- |
| Refusal | A typed scientific outcome stating what was asked, what is missing and the next option. See [refusals](refusals.md) |
| Operational failure | An invalid argument, cancellation or resource limit. It raises an error and carries no scientific meaning |
| Outcome | The returned object holding either a result or a typed refusal |

For the exact combinations that run, use [supported analyses](../supported-analyses.md) and [support lookup](../support-lookup.md). For scope-specific coverage, see [population, time and uncertainty](../population-time-uncertainty.md). For how the evidence behind these terms is governed, see [assurance](assurance.md).
