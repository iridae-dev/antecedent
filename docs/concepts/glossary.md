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
| Answer kind | The shape of the answer: `point`, `bounds`, `partial`, `response`, `structured`, `associational`, `refused` or `unavailable` |

## Standing of an answer

| Term | Meaning | Example |
| --- | --- | --- |
| Identified | Declared causal assumptions and evidence determine the requested quantity through an implemented identification route | A backdoor adjustment identifies an average effect under the supplied DAG |
| Model conditional | The result follows from the specified or fitted model, without the design identifying the real-world effect | A fitted joint response, or a cyclic equilibrium under the specified equations |
| Licensed | The exact configured route has executing support evidence, with its recorded scope and limits. Unrelated to a software license | An estimator licensed for the all-observed mean effect refuses another functional |
| Independently checked | A named calculation or proposition was compared with an appropriate independent reference | A closed-form value, or a pinned external implementation |
| Calibrated | A matching coverage record attests the executing code and covers this execution's scope | An interval outside the record's sample-size or configuration scope is not calibrated by that record |
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
| Replayed | A specified calculation was reproduced under the consumer's procedure | A finite composition re-derived from its artifact |
| Accepted artifact | Identity, interpretation or replay checks on a saved result passed. This is not a check on the real-world assumptions | Loading an artifact with its expected identity |
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
