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

The 0.9 matrix licenses these main families:

- static `AverageEffect` on explicit or accepted DAGs, ADMGs, and PAGs under
  the inference and validation combinations listed in the matrix;
- `ConditionalEffect`, `PathSpecificEffect`, and
  `InterventionalDistribution` on their narrower DAG-only cells;
- static and temporal `ResponseCurve` / `InterventionResponse` under
  Frequentist inference, explicit or accepted structure, and validation
  `none`;
- pulse and single-step sustained temporal effects on `TemporalDag`, including
  the listed Frequentist, Bayesian, and DBN-posterior cells;
- temporal mediation on explicit or accepted `TemporalDag` under Frequentist
  inference and validation `none`;
- Bayesian graph-posterior `AverageEffect` envelopes whose atoms are DAGs.

Graph-posterior support is deliberately narrow. The static envelope is
`AverageEffect × Dag × graph_posterior × Bayesian` with validation
`none`/`cheap`/`full`. Temporal graph-posterior support is pulse and
single-step sustained effect on `TemporalDag` with validation `none`.
Frequentist graph-posterior combinations, response mixtures, and
ADMG/CPDAG/PAG posterior atoms are refused. Unidentified atom mass is retained;
priors do not upgrade identification.

Derivative query types remain importable at the Python root so unsupported
requests fail as typed matrix refusals. They are not licensed analysis cells.

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

See [ADR 0009](../adr/0009-parity-baselines.md), the parity manifests, and each
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

There is no built-in EconML handoff. Graph interchange or a backdoor adjustment
set can be moved manually when that estimand actually is an adjustment estimand.
Front-door, IV, general-ID, partial-identification, and graph-posterior results
cannot be reduced to "pass this adjustment set to another estimator."

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

- no ML-based CATE estimators or built-in EconML integration;
- no plotting module;
- no R, Julia, or JavaScript bindings through 1.0;
- no complete PAG-native ID/IDC;
- no complete general sID recursion;
- no Bayesian response curves or response mixtures over graph posteriors;
- no licensed derivative analysis cells;
- no class-aware temporal effect identification on incomplete
  `TemporalCpdag`/`TemporalPag`;
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
