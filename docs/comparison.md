# Antecedent and other causal libraries

This page is a selection aid, not a competitor benchmark. It describes
Antecedent from contracts verified in this repository and links to other
projects for capabilities that may fit a different workflow. Claims about
another project should be checked against that project's current documentation.

Antecedent is an identification-first Rust engine with Python bindings. Its
distinctive workflow is to keep the query, graph class, identification status,
empirical support, inference mode, and validation contract explicit through
estimation.

## What Antecedent is licensed to run

The [support matrix](support-matrix.md) is authoritative. A capability present
in the codebase is not necessarily a licensed `analyze()` combination.

Antecedent 1.11 composes **463 licensed combinations** of 1403 meaningful
cells into an inspectable, reusable execution workflow. Each combination
fixes the question, graph class, structure source, inference method, and
validation level. A successful licensed Python analysis retains a study and
exports its result.

Use [supported analyses](supported-analyses.md) to find a starting point and
[capabilities](capabilities.md) for the methods behind each path. The
[1.11 release notes](release-notes/v1.11.0.md) describe this cut; the
[1.10 release notes](release-notes/v1.10.0.md) remain the composition-contract
baseline. Older release notes record when individual methods were added.

Some important distinctions when choosing a workflow:

- Static DAG graph-posterior paths include average effects, conditional effects,
  response curves, and one-coordinate intervention responses. Each has its own
  inference and validation restrictions.
- Temporal DAG graph-posterior paths include Pulse and Sustained effects under
  both inference modes. Temporal mediation supports Bayesian validation
  `none`/`cheap`/`full` and Frequentist validation `none`.
- Bayesian envelopes on incomplete `TemporalCpdag` and `TemporalPag` graphs
  are supported for licensed queries. A caller-supplied `ClassPrior` can provide
  mixture weights; enumerating completions alone does not assign probabilities.
- Derivatives on explicit or accepted DAGs support Frequentist and Bayesian
  inference at validation `none`. Partial-graph derivatives remain refused.
- ADMG interventional distributions run through Python `analyze`/`prepare` and
  Rust `Study` at validation `none`.
- Licensed anomaly/change attribution paths run through Python `analyze`/`prepare`
  and Rust `Study` at validation `none`.
- Licensed transport and interference queries run through both Python `analyze`
  and Rust `Study`, with their explicit design assumptions.

Graph-posterior response surfaces on temporal DAGs and posterior mixing over
`TemporalCpdag`/`TemporalPag` atoms remain refused. ADMG graph-posterior
AverageEffect at validation `none` identifies each atom with `general.id` and
estimates `functional.effect`; cheap/full stay closed. These are different
requests from incomplete-class envelopes and from CPDAG/PAG graph-posterior
AverageEffect, which evaluates each atom with the class ATE envelope and
combines only under a shared estimand. Unidentified structural mass stays
visible; priors do not establish identification.

## What the repository compares externally

Antecedent is independently implemented. Selected conformance fixtures record
black-box outputs from pinned external packages:

- DoWhy 0.14 supplies scoped identification and estimation reference fixtures.
  Executing tests compare only the fields named by each evidence contract.
  Some fixtures carry a recorded DoWhy value that is intentionally not asserted;
  their limitations say so.
- Tigramite 5.2.1.30 supplies the core temporal discovery and conditional-
  independence fixtures. J-PCMCI+ and fixed-regime RPCMCI-related fixtures use
  5.2.9.7. Evidence ranges from frozen examples to behavioral comparisons and
  is not a claim of whole-library parity.
- causal-learn 0.1.4.3 supplies the frozen FCI and GES reference cases, while
  lingam 1.9.1 supplies the DirectLiNGAM order/coefficient reference in the
  same static-discovery fixture.
- statsmodels 0.14.4 supplies the conditional-effect and multiplicity
  reference outputs claimed by the capability manifests.
- bpbounds 0.1.8 supplies the canonical binary-IV Balke–Pearl bounds fixture;
  causaleffect 1.3.15 supplies formula-class references only for Antecedent's
  explicitly scoped sound sID subset.

Each external claim has an immutable record under `parity/baselines`, a frozen
fixture, and an executing conformance test. See
[ADR 0009](https://github.com/iridae-dev/antecedent/blob/main/adr/0009-parity-baselines.md), the parity manifests, and each
licensed row's evidence kind and limitations. A shared algorithm name is not
evidence of equivalent behavior.

## Choosing a workflow

### DoWhy

DoWhy and Antecedent overlap in model–identify–estimate–refute workflows and
graphical causal-model tooling. Evaluate DoWhy when its Python ecosystem,
examples, estimator integrations, or GCM workflows are the primary need.

Antecedent differs by making its support matrix and typed refusals part of the
runtime contract. It also carries accepted partial graphs and selected graph
posteriors into licensed downstream analyses. That does not imply broad feature
parity with DoWhy.

### EconML

EconML focuses on machine-learning estimators for heterogeneous treatment
effects and policy-oriented workflows. Antecedent's licensed
`ConditionalEffect` path is a linear interaction model; it does not provide
causal forests, meta-learners, or a general ML CATE surface.

`antecedent.handoff.econml(result)` emits the adjustment set and identification
status for point-identified backdoor / generalized-adjustment estimands,
including lag-aligned temporal columns when the certificate carries offsets.
`spec.attach(...)` records a caller-fitted estimate as attested, uncalibrated
evidence; Antecedent does not wrap the learner or inherit native calibration.
Front-door, IV, general-ID, partial-identification, and graph-posterior
results refuse rather than pretending they are a set. The adapter does not
wrap EconML learners or absorb ML CATE.

### Tigramite

Tigramite is the upstream reference used for selected PCMCI-family and
conditional-independence conformance fixtures. Evaluate it directly for
time-series discovery research, its current algorithm set, and its native
visualization workflow.

Antecedent's temporal discovery implementations are useful when discovery must
feed a licensed effect, intervention, prepared-analysis, or artifact workflow in
the same engine. External comparisons are fixture-specific: for example,
fixed-label per-regime tests are not parity evidence for unsupervised RPCMCI
regime learning.

### causal-learn

causal-learn provides a broad Python causal-discovery surface. Evaluate it when
discovery itself is the endpoint or when the required discovery algorithm is
outside Antecedent's implemented set.

Graphs can be imported through supported interchange formats, but importing a
graph does not bypass Antecedent's matrix. Its graph class, query, structure
source, inference mode, and validation suite still determine whether the
downstream analysis is licensed, not applicable, or refused.

## Current Antecedent boundaries

The following are current product boundaries or explicit matrix refusals:

- no ML-based CATE estimators (the EconML adapter emits a set, not a learner);
- no plotting module;
- no R, Julia, or JavaScript bindings;
- no complete PAG-native ID/IDC;
- no complete general sID recursion;
- no temporal graph-posterior response surface; for incomplete-class response
  uncertainty, use the query-specific contracts in [causal responses](causal-responses.md);
- no partial-graph derivative cells;
- no exact DAG pseudo-posterior enumeration beyond six nodes;
- no automatic estimator choice and no prior that can rescue identification.

These statements are not interchangeable:

- **not applicable** means a matrix coordinate does not denote;
- **refused** means the coordinate is meaningful but this release does not
  license it;
- **outside product scope** names a broader capability Antecedent does not aim
  to provide in this release line.

For exact current behavior, use the [support matrix](support-matrix.md). For the
implemented building blocks behind it, use [Capabilities](capabilities.md).
