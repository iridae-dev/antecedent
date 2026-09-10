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

The 1.5 matrix keeps the 1.4 licensed cells and adds no new query kinds.
Prepared iid AllObserved `AverageEffect` plans with explicit AIPW and discrete joint
`InterventionResponse` plans with cell-AIPW export cross-fitted scores and retarget to a declared covariate population.
`analyze()` does not always return scores. Linear, ATT/trim/clustered AIPW, matching, IV, and Bayesian plans do not export scores. `outcome_functional`
covers mean and exceedance, including class-aware ConditionalEffect grids mixed
across envelope atoms. Discrete joint interventions may use `cell.aipw`.
`TieredBackground` certifies a tier-closure adjustment set as a fast path.
CoDetermined joint cells are ADMG generalized adjustment on the known closure
plus `cell.aipw`; Unknown-tier joint stays refused.
Static Cpdag / Pag Frequentist effect and response aggregates publish joint-IF
standard errors mixed by frozen completion weights. Scalar Dag
`InterventionResponse` licenses cheap/full: `cell.aipw` on the cell-versus-control
contrast; plugin g-comp cheap is overlap only and full is overlap plus sampling-stability of the g-comp level.
`ResponseCurve` cheap/full stay n/a. Temporal / DBN multi-atom Frequentist
uncertainty stays unavailable until 1.9.

The 1.4 matrix licenses the 1.3 families plus:

- `AverageEffect` on explicit or accepted `Cpdag` under Frequentist or
  Bayesian inference with validation `none` / `cheap` / `full`, as a MEC
  envelope whose runtime class stays `Cpdag`;
- `ResponseCurve` / `InterventionResponse` on explicit or accepted `Cpdag` /
  `Pag` under Frequentist or Bayesian inference with validation `none`, via
  the same generalized-adjustment envelope (not a licensed MAG/PAG
  response-identification theory);
- `ConditionalEffect` on explicit or accepted `Cpdag` / `Pag` under
  Frequentist or Bayesian inference with all three validation values, after a
  pre-treatment and augmented backdoor check;
- Frequentist Pulse / single-step Sustained on explicit or accepted
  `TemporalCpdag` / `TemporalPag` with all three validation values
  (`TemporalPag` retains directed/bidirected MAG completions; finite-window
  equivalence audits cannot confer class-wide point identification);
- Frequentist graph-posterior `AverageEffect` on DAG atoms, the sibling of
  the 1.1 Bayesian envelope.

The 1.3 matrix added:

- Frequentist `PointDerivative` / `Elasticity` / `SemiElasticity` /
  `AverageDerivative` / `DirectionalDerivative` / `ResponseJacobian` on
  explicit or accepted DAGs at validation `none`;
- static `MediationEffect` on explicit or accepted DAGs with validation
  `none` / `cheap` / `full`;
- `Counterfactual` on an explicit DAG at validation `none`;
- selected AIPW, marginal KM, and conditional Cox IPCW observation pairs on
  licensed Frequentist `ResponseCurve` cells, as published in the
  [observation pair contract](observation-contract.md).

The 1.2 matrix licensed these families (structure and validation qualifiers are
part of the claim, not implementation detail):

- Frequentist `AverageEffect` on explicit or accepted DAGs, ADMGs, and PAGs,
  and Bayesian `AverageEffect` on explicit or accepted DAGs and PAGs; all
  three validation values are licensed for those cells;
- Bayesian graph-posterior `AverageEffect` over DAG atoms, with validation
  `none`, `cheap`, or `full`;
- Frequentist and Bayesian `ConditionalEffect` on explicit or accepted DAGs with all three
  validation values;
- Frequentist `PathSpecificEffect` and `InterventionalDistribution` on an
  explicit or accepted DAG with validation `none`, `cheap`, or `full`;
- static and temporal `ResponseCurve` / `InterventionResponse` under
  Frequentist or Bayesian inference, explicit or accepted DAG or TemporalDag
  structure, and validation
  `none`;
- pulse and single-step sustained temporal effects on explicit or accepted
  `TemporalDag` under Frequentist or Bayesian inference with all three
  validation values, including Bayesian DBN-posterior cells;
- multi-step sustained effects on explicit or accepted `TemporalDag` under
  Frequentist or Bayesian inference, with validation `none` and no split;
- temporal mediation on explicit or accepted `TemporalDag` under Frequentist
  or Bayesian inference and validation `none`, `cheap`, or `full`.

The Bayesian additions use the documented Gaussian linear forms. Temporal
mediation requires one mediator with treatment at lag one and mediator/outcome
contemporaneous, adjusting for observed baseline parents. Multi-step sustained
uses sequential g-computation; Bayesian time copies share stationary mechanism
draws. These restrictions are part of each licensed form; see the
[1.2 evidence ledger](v1.2-evidence.md),
[1.3 evidence ledger](v1.3-evidence.md), and
[1.4 evidence ledger](v1.4-evidence.md) and
[1.5 evidence ledger](v1.5-evidence.md).

Graph-posterior support is deliberately narrow. The static envelope is
`AverageEffect × Dag × graph_posterior` under Bayesian or Frequentist
inference with validation `none`/`cheap`/`full`. Temporal graph-posterior
support is pulse and single-step sustained effect on `TemporalDag` with
Bayesian inference and validation `none`, `cheap`, or `full`.
Frequentist DBN-posterior mixing, response mixtures, and
ADMG/CPDAG/PAG posterior atoms are refused. Unidentified atom mass is retained;
priors do not upgrade identification.

Derivative query types remain importable at the Python root so unsupported
requests fail as typed matrix refusals. They are licensed on Frequentist
explicit or accepted DAGs at validation `none`; Bayesian, partial-graph, and
validation cheap/full coordinates remain refused.

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
- no Frequentist DBN-posterior mixing (1.7) or response mixtures over graph posteriors;
- no Bayesian or partial-graph derivative cells;
- no Bayesian envelope on incomplete `TemporalCpdag`/`TemporalPag` (1.7);
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
