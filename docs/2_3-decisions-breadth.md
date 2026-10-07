# Decisions, design, repair and estimator breadth (2.3, milestone B)

This page lists each milestone B route: the question it answers, its scope
bounds, its claim label, the Python module that owns it and its refusal
namespace. Every claim here is point-only or makes no claim; the registry in
`parity/promotion_2_3.toml` is the authority, and this page restates it.

## Reading the labels

- **Point-only** means the value is a point value (or an enumerated finite-law
  quantity). No interval, coverage or probability claim comes with it.
- **No claim** means the route states a request or a classification and
  computes no estimate.
- **Closed pending calibration** means a route or field exists in the registry
  but is refused until its calibration is measured at the 2.3 release cut.
- Every standard error, p-value, interval and Monte Carlo error that appears in
  B is a **diagnostic with calibration unmeasured**. It is reported so it can be
  inspected. It is not a confidence statement, its nominal level is not a
  coverage claim, and no coverage record is allocated for it.

A refusal carries a registered reason code plus a namespaced detail
(`<namespace>.<snake_case>`); the namespace named per route below is the stable
thing to switch on.

## Decisions

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Decision contract | Which action has the best expected utility under a durable, replayable declaration? | At least two actions with stable semantic ids; closed utility expressions; inputs inside the target population and horizon; a hard constraint excludes an action and never becomes a penalty. | point-only | `antecedent.decision` (`Contract`, `Action`, `Criterion`) | `decision_contract` |
| Typed functionals | What is the expectation, probability or quantile of a utility on an aligned finite law? | A mean answers only the expectation of an affine utility; a probability or quantile needs a quantile function, CDF or draws; a nonlinear function of several inputs needs aligned joint draws, and independent marginals refuse. | point-only | `antecedent.decision` | `decision_evaluation` |
| Admissibility and support | Can declared exclusions, per-input support rules and an uncertainty requirement remove actions? | Rules remove actions and never penalize; support statuses are ordered by severity; an uncertainty requirement is met only by a claim of exactly its own kind. | point-only | `antecedent.decision` | `decision_admissibility` |
| Structural policy | What does a declared policy choose when the graph structure is uncertain? | Maximin, report-only and Bayes-over-structures; Bayes refuses without genuine atom probabilities, and unevaluated mass is never normalized away. | point-only | `antecedent.decision_robust` | `decision_structural` |
| Claim adapters and identified sets | Can scenario, weighted-scenario and CPDAG-completion reports, and identified-set intervals, enter a decision? | Supplied sets only; an identified set has no probability law so Bayes over it refuses; completion counts are never probabilities. | point-only | `antecedent.scenario_decision` | `decision_adapters`, `decision_claims` |
| Robust verdicts and result artifact | Is the choice robust across structures, support-robust, graph-dependent or insufficiently claimed, and can the result be replayed? | Verdicts over supplied structures; an external-callback value keeps its receipt through export and replay and is never labelled native. | point-only | `antecedent.decision_robust` | `decision_robustness` |

Two limits are worth stating because they are easy to misread. A typed
functional does not by itself license a native support row, and a decision over
an external mean claim is labelled point-only attested, not native.

## Signals, EVSI and design ranking

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Study candidate | What evidence would a proposed study collect, with sample, timing, unit and cost identities? | A candidate observing separate regimes cannot claim a joint factor; missing unit or cost semantics refuse. | no claim | `antecedent.repair` (`StudyCandidate`, `ExpectedEvidence`) | `study_candidate` |
| External signal | Is a signal's observation law and posterior update coherent, with external trust kept external? | A predictive law without a coherent update, or a wrong candidate or sample id, refuses; trust is never native. | point-only | `antecedent.design` | `signal_provider` |
| Prior and signal planning | Can a prior bank and a candidate signal be combined into a checked planning input? | Pooling is declared; an observation shared between a prior source and the candidate, or between two sources, refuses; a changed population needs a declared transport policy. | point-only | `antecedent.priors` | `prior_signal` |
| EVSI | What are the expected value of sample information, of perfect information and the net value under a declared cost map? | Exact integration under a declared utility-unit cost map; currency without a cost-to-utility map, and a changed terminal action set, refuse a net-value comparison. | point-only | `antecedent.design` (`rank_designs`) | `evsi` |
| Design ranking artifact | Can a ranking of candidate studies be exported and replayed by a fresh consumer? | Native values are recomputed on replay; external values are retained as attested; an altered signal fingerprint, overlapping source data or an incompatible cost unit refuses. | point-only | `antecedent.design_ranking` | `design_ranking` |

The Monte Carlo error of a sampled EVSI and any rank uncertainty are
diagnostics with calibration unmeasured, not licensed claims.

## Repair

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Evidence obligations | What evidence does a failed identification contract owe, by exact population, regime, variables and proof step? | A request, never a verdict; separate marginal studies cannot satisfy one required joint-regime factor. | no claim | `antecedent.repair` (`EvidenceObligation`) | `evidence_obligations` |
| Identification repair | Which candidate study repairs a contract, as classified by each family's theorem checker? | Subset search under a `SearchBudget`; the subset size is at most 4; exhaustion means not found within the budget, never impossible. | no claim | `antecedent.repair` | `identification_repair` |
| Repair search receipt | Can a stored search receipt be replayed by a fresh consumer? | The consumer trusts nothing stored: it rebuilds the family and re-runs the checker on each stored delta; truncated work is stored as unevaluated. | no claim | `antecedent.repair` | `repair_artifact` |
| Proposal receipt | Can the evidence a failed contract owes be linked, per proposed study, to the ranking that values it? | Identities are bound and the family checker is re-run on arrival; a hypothetical derivation is never treated as available evidence. | point-only | `antecedent.proposals` | `proposal_receipt` |

## Sensitivity composition and inverse queries

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Sensitivity-decision composition | Over a declared assumption grid, where does the leading action switch? | Grid points only; a tie coordinate is reported exactly and an interpolated crossing is labelled an interpolation. A declared assumption range is not a sampling interval and never a probability; composing the two is refused. | point-only | `antecedent.sensitivity_decision` | `sensitivity_decision_composition` |
| Functional inverse query | What is the least action on a declared grid whose functional meets a target? | A finite declared grid; point, interval-region, identified-set, all-scenario and posterior-probability feasibility are separate fields, and a found grid point is never a global feasibility claim. A mean cannot satisfy a probability or quantile target. | point-only | `antecedent.inverse_query` | `functional_inverse_query` |

## Dose grid and chain recovery

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Dose-grid functional | What are the level, derivative or contrast of a randomized-dose response curve at named doses, and where is the data too thin to say? | A randomized continuous dose, a Gaussian-kernel local quadratic at a caller-declared fixed bandwidth, one named functional per call, per-dose support reported. | point-only | `antecedent.dose_grid` (`dose_functional`, `dose_support_table`) | `dose_grid` |
| Chain observation recovery | Is the law of two binary variables recoverable from the observed pattern law when one response indicator causes the other? | An m-graph with exactly one response edge; a self-censoring edge is refused with a verified two-model witness, and a graph outside the class is `route_not_supported`, never nonrecoverable. | point-only | `antecedent.recovery_chain` (`recover_chain`) | `recovery_chain` |

For the dose grid the pointwise normal intervals and standard errors are
diagnostics with calibration unmeasured, the smoothing bias is not included, and
the simultaneous band is closed (`dose_grid.simultaneous_band_closed`) until
its own calibration exists. A sampled provider composed on chain recovery, and
its intervals, stay closed.

## B4 estimators and compact export

All four estimators report point values. Every standard error, covariance
entry, Wald statistic, p-value (raw and Holm) and bootstrap standard error they
report is a diagnostic with calibration unmeasured; Type I error, power and
coverage are not collected in 2.3.

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Vector treatment coefficients | What are the joint coefficients of several treatments estimated together, with their full covariance? | One shared adjustment block and one row snapshot; a different adjustment set, snapshot or row count, a constant or collinear treatment, or a rank-deficient block refuses. | point-only | `antecedent.vector_treatment` | `vector_treatment` |
| Categorical treatment contrasts | What are level-versus-reference and pairwise contrasts, with an omnibus test and a declared monotonicity test? | A declared level with too few rows refuses by name; nothing is dropped or merged. The monotonicity p-value is a conservative bound. | point-only | `antecedent.categorical_treatment` | `categorical_treatment` |
| Nonlinear mediation | What are the natural direct, indirect and total effects of a binary treatment through a continuous mediator with a nonlinear outcome? | Requires sequential ignorability and declared cross-world independence; a Gauss-Hermite quadrature check refuses on disagreement; the interventional estimand is refused. | point-only | `antecedent.nonlinear_mediation` | `nonlinear_mediation` |
| Latent-class effects | What are the per-class treatment effects of a K-class mixture of linear outcome models? | At most four classes; treatment as-if randomized within class; EM finds a local maximum from seeded restarts, not a certified global one; a class effect is not an individual effect. | point-only | `antecedent.latent_class` | `latent_class` |
| Compact runtime export | Can a fitted linear-in-coefficients effect model be exported as an identity-bound runtime object that predicts only inside its declared support? | A finite basis, a coefficient vector and a covariance; a query outside the support or inside a mask region refuses. The model-based standard error carries no extrapolation or misspecification uncertainty. | point-only | `antecedent.compact_export` (`CompactExport`) | `compact_export` |

For nonlinear mediation the bootstrap standard error is reported with interval
status `closed_calibration_unmeasured`: no interval is produced.

## What B does not claim

- No route here produces a calibrated interval, credible interval or coverage
  statement.
- A structural or assumption range is a range over declared cases. It is never
  a probability.
- A route that composes an external value keeps that value's trust label
  (externally attested unless an exact-request receipt is retained) and never
  upgrades it to native.
- Registration is not execution. A record proves what its cited tests execute;
  see the milestone C page for how B routes are composed in
  [composition](2_3-composition.md).

Related pages: [decisions](2_3-decisions.md),
[external scientific objects](2_3-external-science.md) and
[joint distributions](2_3-joint-distributions.md).
