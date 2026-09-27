# Capabilities

This is a readable inventory of what exists in Antecedent, not the product's
mental model or a promise that every combination can run. Read
[what “supported” means](guarantees.md) for the distinction between capability,
licensed execution, and real-world scientific validity. The parity manifests
are the maintained implementation inventory; the [support matrix](support-matrix.md)
is the public **license** for analysis cells. Inspect / claim / reuse /
handoff compositions live in [`parity/compiler.toml`](https://github.com/iridae-dev/antecedent/blob/2.0/parity/compiler.toml)
and are not analysis-matrix coordinates. Presence here does not mean every
query × graph class × structure × inference × validation combination runs.
For selection guidance and product boundaries, see [Comparison](comparison.md).

## How to read capability claims

The matrix has three active runtime states:

* **licensed** — the staged path runs under the row's recorded evidence
  contract;
* **n/a** — the coordinate does not denote and is a typed impossibility;
* **refused** — the coordinate is meaningful, but this release does not
  license it.

The historical `allowed_unlicensed` wire value remains decodable for
compatibility, but this release has no active allowlist entries and the release
gate rejects new ones. Evidence kinds are scoped: a known-truth fixture may pin
only identification or an effect point, while an internal cross-check may establish
prepared-vs-fresh consistency without pinning the scientific target. Read each
row's `limitations`; a shared method name is not a parity claim.

A licensed row means the staged runtime path, refusal boundary, and recorded evidence
contract are exercised for that coordinate. It does not mean causal assumptions were
verified from the data, intervals are universally calibrated, identification is complete
beyond the named subset, or parametric restrictions disappeared. In particular, priors
cannot convert a nonidentified estimand into an identified one.

At analysis level, the support matrix is the license. The licensed matrix includes
temporal policy cells: per-horizon
`TemporalMediationEffect`, multi-step and joint `Sequence` overlays,
observation-adjusted temporal curves (Frequentist IPCW pairs and the
parametric Bayesian observed-data CAR route), DBN-posterior mixtures on
the contrasts the handle already runs, and bounded prior transfer on
named Pulse / Sustained / ResponseCurve cells. It also includes retargetable
prepared AIPW scores (AllObserved iid AIPW and cell-AIPW only; `analyze()`
does not always return scores), exceedance functionals, cell-saturated joint AIPW,
`TieredBackground` as a fast path over ADMG / PAG adjustment, and joint
influence-function standard errors on static Cpdag / Pag effect and response
aggregates. Unknown tiers retain distinct canonical scenario effects.
Completions stay envelope atoms; the runtime class is not collapsed.
Static Cpdag / Pag Frequentist aggregates publish joint-IF standard errors.
A static class effect whose completions disagree — Cpdag / Pag average and
conditional effects, Frequentist and Bayesian, and an Unknown tier's canonical
scenarios — publishes the identified set over its identified completions
alongside that aggregate, with every completion's value and enumeration weight
and with unidentified, unevaluable and incomplete-search mass kept apart. A
point-identified class effect still publishes a point.
Frequentist DBN Pulse/Sustained mixtures publish shared outer-block
bootstrap uncertainty. TemporalCpdag/TemporalPag class envelopes publish a
shared circular-block mixture SE (frozen-weight aggregate over identified
atoms; unidentified mass is retained, not mixed) and Imbens–Manski
identified-set intervals with per-completion endpoints; Bayesian class envelopes
publish the product-posterior envelope quantile (completion posteriors from
distinct `StreamDomain` RNG streams) at Imbens–Manski tails; both are flagged
`truncated` over a capped
completion enumeration. Temporal PAG results retain MAG
completions and disclose finite-window audit caps. Those families are not licensed on every
coordinate: TemporalPag
mediation, partial-graph derivatives, graph-posterior / nested counterfactuals,
and cheap/full counterfactual validation remain refused. Importability is not
a license. The [support matrix](support-matrix.md) is the public license.

The two-arm randomized survival utilities cover Kaplan–Meier curves/RMST,
Aalen–Johansen cumulative incidence, delayed entry under the marginal
observation contract, and a separate caller-supplied IPCW curve/RMST route.
The randomized survival and competing-risk curves run through retained
`prepare` / `analyze` as well as the direct utilities. The unweighted route
requires marginal `IndependentGiven(())`; the retained IPCW route accepts
caller-supplied censoring survival columns and an explicit conditional
`IndependentGiven` claim. They remain point-only and unlicensed; supplied
probabilities are not fit or independently verified. See [Survival outcomes](survival-outcomes.md).

## Graph primitives

Implemented graph representations:

* DAG;
* ADMG;
* CPDAG;
* PAG;
* temporal DAG;
* temporal CPDAG;
* temporal PAG.

Graph operations:

* d-separation;
* m-separation;
* districts;
* latent projection;
* Markov-equivalence completions;
* PAG m-separation as a statement about every MAG in the class: `separated`,
  `connected`, or `undetermined` (definite-status paths first, then the
  enumerated completions; never "separated" because a path was of indefinite
  status);
* temporal unfolding;
* intervention overlays.

Static and temporal graphs have separate semantics. A static graph is not
interpreted as temporal by default. `AverageEffect` on a supplied `Cpdag` is
licensed (explicit/accepted, Frequentist and Bayesian) via a MEC envelope;
runtime class stays `Cpdag`. Completing the graph yourself is still the `Dag`
cell. Pulse and single-step Sustained on incomplete `TemporalCpdag` /
`TemporalPag` are licensed (explicit/accepted, Frequentist) via a completion
envelope, and the Bayesian counterparts are licensed as well. A fully oriented
supplied class stays that class. Completing those graphs yourself is still the
`TemporalDag` coordinate.

Graph interchange is available through NetworkX, DOT, JSON, GML, and versioned
CBOR artifacts.

## Discovery

### Static

* PC
* FCI
* RFCI
* GES
* DirectLiNGAM
* NOTEARS

### Temporal and multi-context

* PCMCI
* PCMCI+
* LPCMCI
* J-PCMCI+
* regime-specific RPCMCI workflows

### Bayesian structure learning

* exact DAG posterior;
* order MCMC;
* structure MCMC;
* CI-screened graph posterior;
* DBN posterior.

Selected posterior graph samples can be propagated into licensed Bayesian
or Frequentist effect envelopes. Static graph-posterior analysis covers
`AverageEffect` on DAG, CPDAG, and PAG atoms (CPDAG/PAG at validation none)
and `ResponseCurve` / one-coordinate `InterventionResponse` with DAG atoms.
CPDAG/PAG posterior atoms are evaluated with the existing class ATE envelope
and combined by `StructuralAggregationPolicy`: a weighted mean only when
estimands agree, otherwise an identified set or `GraphDependent` result.
Completion enumeration is not posterior probability; unidentified mass is
retained. A Frequentist multi-atom aggregate publishes a joint-IF SE only for
a scalar (one-coordinate `InterventionResponse`) when atom influences align on
the shared rows; a multi-atom curve, or unaligned influences, withholds
uncertainty with `estimate.response.graph_posterior.uncertainty_withheld`. The
interval is not coverage-calibrated. Failed estimation mass is unevaluable, not
unidentified, and makes the result `GraphDependent`. Temporal
graph-posterior analysis covers pulse, single- or multi-step sustained
effects, single-horizon Frequentist or Bayesian temporal mediation, and
licensed TemporalDag / TemporalCpdag / TemporalPag `ResponseCurve` /
`InterventionResponse` cells (see the [support matrix](support-matrix.md)).
ADMG graph-posterior AverageEffect identifies each atom with `general.id` and
estimates `functional.effect`. TemporalPag graph-posterior mediation stays
refused.

Panel Pulse/Sustained on an explicit or accepted `TemporalDag` can prepare and
refresh. Every panel route requires one time-index regularity across units, so
a horizon step means one duration everywhere, and refuses a unit with fewer
lag-aligned rows than a per-unit fit needs.

Panel Pulse / single-step Sustained fits one pooled common-coefficient
regression over the stacked unit rows: its analytic SE is the Arellano
cluster-by-unit variance with `G − 1` degrees of freedom (the SE is scaled by
`t_{G−1}/z` so `estimate ± 1.96·SE` is the t interval), and its bootstrap SE
resamples whole units, each draw its own cluster, refitting the pooled
regression. The same pooled fit answers a multi-environment Pulse /
single-step Sustained, clustered by environment. On a supplied
`TemporalCpdag` / `TemporalPag` each identified completion is fit that way and
mixed by completion mass.

Panel `ResponseCurve` / `InterventionResponse` and panel multi-step Sustained
instead average per-unit fits with equal weight: the response publishes the
between-unit pointwise band `mean ± t_{N−1}·sd/√N` over the unit surfaces (the
series simultaneous band is withheld, requested bootstrap replicates are
unused), and multi-step Sustained reports the between-unit SE on the same `t`
scale. Which estimand a result reports — pooled common coefficient or
equal-weight unit average — is named in its diagnostics and in an assumption
record. The panel support report, its per-cell status and its assumptions are
merged over every unit.

Bayesian panel response uses per-unit posterior means under the caller's
resolved prior. Bayesian panel class Pulse and multi-step panel Sustained
follow the class-prior contract: with a caller `class_prior` over an uncapped
class the completion posteriors' draws are mixed, and without one the effect
is NaN with the completion posteriors kept as atoms. Panel class response
publishes the pointwise envelope over the completions' unit-average surfaces
at every requested horizon, as the series class response does. A class-aware
multi-step Sustained refuses a discovery/estimation split and a transferred
prior. Units are not stacked onto the series class owner; a temporal class is
not completed onto the DAG executor.

Unconditional finite-discrete `InterventionalDistribution` on an explicit or
accepted ADMG is licensed at validation `none` via general ID (bidirected
edges stay). Python `analyze`/`prepare` reach these cells. Cheap/full and
IDC conditionals remain refused.

### Conditional independence tests

* partial correlation;
* weighted and robust partial correlation;
* regression CI;
* k-nearest-neighbour CI;
* mixed k-nearest-neighbour CI;
* symbolic conditional mutual information;
* GPDC;
* G²;
* oracle tests;
* Bayesian CI tests.

Multiplicity corrections include BH, BY, Bonferroni, and Holm.

Discovery stability tools include block bootstrap, lag and threshold
sensitivity, orientation stability, environment holdout, synthetic-null
checks, and permutation or phase-randomized surrogates.

## Identification

Antecedent reports whether a query is:

* nonparametrically identified;
* partially identified;
* graph-dependent;
* not identified.

Implemented identification strategies:

* backdoor adjustment;
* efficient backdoor adjustment;
* front-door identification;
* instrumental variables;
* sharp regression discontinuity, for the effect at the cutoff only (see below);
* Shpitser–Pearl ID/IDC for DAGs and ADMGs with hard `Set` interventions on
  finite domains. Every valid ID query ends either in a functional or in a
  hedge; this is checked exhaustively against enumerated SCMs on all ADMGs of
  up to four nodes (single treatment and outcome) and on a sample of larger
  joint queries, not proved. Napkin-type functionals keep a free pre-treatment
  variable: the identity holds at every supported value of it;
* line-5 hedge node-set certificates, checkable against the hedge definition
  with `HedgeCertificate::verify` (existential over the C-forest edge subsets,
  which the certificate does not store);
* bounded path-specific identification by selected-edge graph reduction;
* generalized adjustment for partial graphs;
* unfolded temporal backdoor;
* temporal mediation;
* pairwise backdoor identification for continuous-response functionals;
* sharp binary-IV Balke–Pearl ATE bounds by response-type enumeration;
* classical single-source sID (Bareinboim–Pearl Figure-5 recursion, access to
  all source experiments): an independently checked symbolic distribution, an
  independently verified s-hedge `ProvenNonTransportable` certificate, or
  `NotCertified` where the recursion meets an obstruction it cannot certify;
  meta-transport does the same across sources. Completeness holds only in
  those experimental-information families;
* catalog-bound single-source transport on a sound but incomplete subset
  (direct, S-admissible / exogenous standardization, singleton c-components),
  with `NotCertified` outside that subset. Failing to bind a formula to a finite
  supplied catalog is not a non-transportability proof.

`AutoIdentifier` reports applicable strategies. It does not silently choose an
estimator.

For PAGs, Antecedent enumerates the valid MAG completions (maximal, ancestral,
same unshielded colliders, one Markov class) and identifies each; the envelope
keeps unidentified mass. `AverageEffect` and `ConditionalEffect` use generalized
adjustment after a MAG visibility check. Single-treatment `ResponseCurve` /
`InterventionResponse` try that adjustment first and then Shpitser–Pearl ID on
the completion with every *invisible* directed edge `A -> B` (Zhang 2008) also
read as `A <-> B`. Every DAG a MAG represents projects to a subgraph of that
ADMG, so a functional found there holds for all of them; a directed MAG edge is
never read as unconfounded unless it is visible. This is sound and reaches
effects no adjustment set identifies, but it is not complete: a refusal
(`identify.response.mag_id_refused`) is not a proof of non-identifiability. The
complete algorithm for PAGs, IDP (Jaber, Zhang & Bareinboim 2019), is not
implemented, and there is no PAG-native IDC. A circle-free graph that is not a
maximal ancestral graph (for example the front-door ADMG `T -> M -> Y`,
`T <-> Y`) has no completion and identifies nothing as a `Pag`; hold it as an
`Admg`. This is not a path-specific, distribution, or mediation surface.
Transport completeness is per theorem family (see
[transport scope](guides/transport-scope.md)): classical sID and meta-transport
are complete only in their experimental-information families, and the finite
catalog search is sound and incomplete.

## Estimation

### Frequentist

* linear and generalized-linear outcome regression;
* g-computation;
* inverse probability weighting;
* propensity matching;
* covariate-distance matching;
* stratification;
* AIPW;
* front-door functional plug-in estimation (`frontdoor.functional`);
* linear front-door two-stage estimation (`frontdoor.linear_two_stage`);
* Wald estimation;
* 2SLS;
* sharp local-linear regression discontinuity (effect at the cutoff);
* linear conditional effect models;
* temporal adjustment;
* temporal mediation;
* functional plug-in estimation;
* continuous causal-response curves (Kennedy-style cross-fitted doubly robust
  local polynomial);
* observed-law Riesz average derivatives;
* additive-GAM plug-in Jacobians and directional derivatives (at most two
  treatment dimensions);
* additive-GAM g-computation for numeric hard, shift, and stochastic
  intervention responses;
* selected-outcome IPW and cross-fitted AIPW, plus marginal right/left Kaplan–Meier
  IPCW and conditional right/left Cox IPCW, composed into point-only response
  curves under explicit observation assumptions. The same selected / KM /
  Cox pairs ride Frequentist `TemporalDag` curves at validation `none`;
  unlicensed non-Complete pairs refuse at compile.

Response results keep structural identification, empirical support, and
uncertainty kind as separate axes. Pointwise and simultaneous bands are not aliases. Frequentist temporal observation curves use nuisance-refitting outer block-bootstrap bands. Bayesian temporal observations use the Gaussian observed-data SEM with latent-trajectory Gibbs sampling under declared ignorable trajectory coarsening and distinct priors. Neither substitutes complete-data intervals. Interval censoring
and truncation remain Gaussian-likelihood stages, not a causal-response MLE.
One-shot `discovery=` on response queries fails closed; discover and accept
the structure before estimating a response.
The list above is inventory. Derivative cells are licensed on explicit or
accepted DAGs under Frequentist and Bayesian inference at validation `none`;
partial-graph derivatives remain refused. `ResponseCurve` and `InterventionResponse` are
licensed on `Dag`, `Admg`, `Cpdag`, `Pag`, `TemporalDag`, `TemporalCpdag`, and
`TemporalPag` under Frequentist
and Bayesian inference with validation `none` (class graphs via the completion
envelope: generalized adjustment per completion, then, for `Pag`, the sound but
incomplete visibility-aware ID described under identification; see the
[support matrix](support-matrix.md)). That is not PAG-native (IDP) response
identification. Graph-posterior
`ResponseCurve` cheap/full on `Admg` / `Cpdag` / `Pag`, and Bayesian
graph-posterior `InterventionResponse` cheap/full on `Cpdag` / `Pag`, stay
refused. Bayesian responses require the documented Gaussian additive models and
AllObserved population, with pointwise posterior intervals. Static Bayesian
response uses complete observations; temporal Bayesian response also supports
the five licensed observation pairs through its observed-data SEM backend.
Licensed Bayesian `Cpdag` / `Pag` cells keep per-completion posteriors unmixed
and publish the completion identified set when completions disagree. Frequentist
TemporalCpdag/Pag cells retain completion identified sets; DAG-posterior cells
retain atom probabilities and unidentified mass.
`ConditionalEffect` is licensed on `Dag`, `Cpdag`, and `Pag`. The public
license is that matrix, not this page.

Three of these carry parametric scope conditions that the estimator cannot check
at runtime:

* **Linear front-door two-stage estimation** (`frontdoor.linear_two_stage`) is the
  linear-SEM product-of-coefficients estimator. The front-door criterion licenses
  the functional `E[Y|do(t)] = sum_m P(m|t) sum_t' E[Y|m,t'] P(t')`; the product of
  coefficients equals it only when `E[M|T]` is linear and `E[Y|M,T]` has no
  treatment-mediator interaction. A latent treatment-outcome confounder that
  modifies the mediator's effect breaks this (a binary example converges to 0.363
  against a true 0.315), so a result produced by this estimator is reported as
  identified under parametric restrictions, with `frontdoor.linear_path_product`
  in its identification assumptions, exactly as a Wald estimate carries the IV
  restriction; this holds under the `frontdoor` and the `auto` identifier alike.
  A `frontdoor.functional` result keeps the nonparametric claim: it evaluates the
  identified functional, and its per-arm linear outcome regression is a model of
  an observable regression, recorded at estimation scope. For a discrete treatment use
  `frontdoor.functional`, which estimates the functional itself: saturated cell
  means for discrete mediators (nothing assumed; refused when a mediator value is
  missing from an arm), or a per-arm linear outcome regression for continuous
  mediators (treatment-mediator interaction free, within-arm linearity recorded),
  with an influence-function standard error. It refuses a continuous treatment.
* **Sharp regression discontinuity identifies and estimates one effect: the
  average effect for units at the cutoff,**
  `lim_{r↓c} E[Y | R = r] − lim_{r↑c} E[Y | R = r]`. It is not the population
  average effect and not the average over the bandwidth window; with an effect
  that varies in the running variable these are different numbers. The result's
  target population is `LocalAtCutoff { running, cutoff }`: a study that selects
  `rd.sharp` and leaves the population at its default is retargeted to it, the
  `identify.rd.local_estimand` diagnostic says the population-wide effect is not
  identified, and any other target population is refused. `AutoIdentifier` uses
  a supplied design only for a query that asks for that population, and never
  infers a running variable or cutoff. The design is checked where it can be:
  the graph must make the running variable the treatment's only parent, and the
  estimator compares the treatment column with `T = 1{R ≥ c}` on every complete
  row and refuses (`rd_assignment_not_sharp`) on any violation, so imperfect
  compliance is not reported as a treatment effect (fuzzy RD is not
  implemented). Continuity of the potential-outcome regressions at the cutoff
  and no manipulation of the running variable are recorded as assumptions and
  are not tested. The estimator uses a caller-supplied bandwidth with a
  uniform kernel and reports a conventional, not bias-corrected, interval; its
  default SE is the HC1 residual sandwich, and the homoskedastic SE is an
  explicit opt-in (`RdConfig::with_se_kind`). There
  is no data-driven bandwidth selector and no Calonico–Cattaneo–Titiunik robust
  correction, so the estimate is only as defensible as the chosen bandwidth.
* **Propensity-matching standard errors** use a pooled homoskedastic variance
  proxy rather than the full Abadie–Imbens conditional variance estimator. Under
  heteroskedastic outcome variance the reported standard error is biased.

Applying the first two outside their assumed regime produces a biased estimate
with no runtime signal.

`response.kennedy_dr` is also a least-squares construction (additive GAMs plus
a local-quadratic of the doubly robust pseudo-outcome) and needs finite
outcome moments. Unlike the three cases above, it reports
`response.outcome_tail_ratio` at runtime and warns
`response.heavy_tailed_outcome` when the ratio exceeds 20. That warning does
not demote `evidence_status` or `support.status`. See
[causal-responses.md](causal-responses.md#least-squares-kennedy-dr-regularity).

Frequentist interventional distributions (`functional.distribution`) publish
every atom probability with a 95% interval formed on the logit scale from that
atom's bootstrap SE, `expit(logit p̂ ± z·se / (p̂(1 − p̂)))`, so both bounds lie
in `[0, 1]`. A binary `{0, 1}` outcome's interventional mean is `P(Y = 1 | do(x))`
and carries the same interval. Rust reads them from
`InterventionalDistributionEstimate::atom_uncertainty` and `mean_interval`;
Python from `EstimateView.distribution` and `EstimateView.mean_interval`. The
mean's `se_bootstrap` is unchanged, but `mean ± z·se` is not the published
interval and can leave `[0, 1]` near the boundary. A plug-in probability of
exactly 0 or 1 has no sampling spread, so it publishes no interval and the
`estimate.distribution.interval_unavailable` warning names the atom.

#### Serially dependent rows: circular-block uncertainty

Lag-aligned rows of one time series are not independent, so temporal
Frequentist intervals do not use the iid OLS standard error or an iid row
bootstrap. Every temporal circular-block bootstrap starts from one block rule
(`antecedent_data::circular_block_length`):

```text
block = min(n, max(span, ceil(n^(1/3))))
```

`span` keeps each estimating row's lag window inside one block. It is the
unfolded `history + horizon` window for temporal-backdoor designs, the deepest
design lag + 1 for temporal mediation and multi-step sequential g-computation,
and the DBN max lag + 1 for DBN-posterior mixtures. When an estimating score
(the target's influence, a mixture's score, or any normal-equation score of any
fitted regression, residuals included) is persistently dependent, the block is
lengthened to `ceil(b_PW·n^(1/6))` (the Politis–White length at the fixed-b
testing rate), capped at `n/3` (`antecedent_estimate::dependence_block_length`).

* **TemporalDag Pulse / single-step Sustained, multi-step Sustained, and
  temporal mediation.** These resample circular blocks of consecutive
  lag-aligned rows, so every replicate row keeps its own lag window, and refit
  the design (every mechanism of the sequential g-computation) on each
  replicate. Mediation refits Total, Direct, and Mediated on the same
  replicate. No analytic SE is published (`se_analytic` is NaN). With zero
  replicates there is no SE.
* **Multi-atom temporal mixtures** (class envelopes and DBN posteriors) resample
  blocks of consecutive series times over the window every atom can evaluate;
  each atom refits its own lag-aligned rows at those times, so every atom sees
  the same replicate.

In both, the replicate SD is multiplied by the circular-Bartlett fixed-b factor
`cv(block/n) / 1.96` with `cv(b) = 1.96 + 2.4389b + 3.7072b² − 2.1055b³`
(`antecedent_estimate::circular_fixed_b_scale`), the simulated 95% fixed-b
critical value for a mean studentized by the circular Bartlett variance that the
circular-block bootstrap estimates. The Kiefer–Vogelsang polynomial is for the
non-circular Bartlett estimator and is too small at long blocks (it covers 0.942
at `b = 1/3` for a nominal 95% interval). The published SE interval is the 95%
`estimate ± 1.96·SE`; the short-series sweeps measure a nominal-0.90 interval
built from the same SE (the 95% ratio around a normal 0.90 critical value, not
the 90% fixed-b value), which covers 0.901–0.913 for `b ≤ 1/3` in the same
simulation. The replicate SD is also multiplied by the Bartlett kernel-bias
factor `1/sqrt(f)` of the interval's target scores
(`antecedent_estimate::kernel_bias_scale`, the largest over the scores of
`antecedent_estimate::kernel_bias_factor`, the same per-score factor the response
bands apply per cell): `f` is the share of a fitted autoregression's long-run
variance that the Bartlett kernel at the block length keeps, under a
Kendall-corrected AR(1) and a BIC-selected AR(q ≤ 4) (larger factor kept). A
circular block of length `ℓ` reproduces the Bartlett variance at bandwidth `ℓ`,
which the fixed-b critical value does not correct for. The
`estimate.temporal.circular_block_se` diagnostic (the shared-block diagnostic
for mixtures) records the block length, row count, fixed-b factor, and the
estimating score's effective rows: over every score of the interval (every
atom's and the mixture's for mixtures, the three contrasts for mediation), the
smaller of the lag-1 reading `n(1 − r₁)/(1 + r₁)` and the block-length reading
`n·γ̂₀ / ĝ_b` (Bartlett long-run variance at the block length).
`estimate.temporal.circular_block_se.short_series` warns below a threshold set
per SE family from a coverage sweep over AR(1) persistence and series length
([short-series thresholds](short-series-thresholds.md)): 45 for Pulse /
single-step Sustained, 40 for mediation, 40 for multi-step Sustained, and 155
for mixtures.

The sweep separates designs whose estimating score forgets within a few lags
from designs where treatment and residual are both persistent. With a
short-memory score (an MA(3) treatment, or the confounded lag DGP for
multi-step Sustained) and AR(1) residuals up to ρ = 0.95, one-series intervals
covered 0.878–0.932 at nominal 0.90 for every n from 40 to 400 (2000
replicates per cell), and the warning is quiet from n = 160. With an AR(1)
treatment as well, Pulse, mediation Total/Direct and multi-step Sustained
covered 0.778–0.896 wherever the series was short for that memory (n ≤ 60 at
ρ ≥ 0.8, up to n = 160 at ρ ≥ 0.9, n = 400 at ρ = 0.95; 19 of those 44 cells
below 0.855); the warning fires on at least 91% of the replicates of every
cell below 0.855. The mixture threshold is higher because a non-causal
completion that omits a persistent confounder is biased in finite samples
(TemporalCpdag Pulse covered 0.685–0.905 at ρ ≥ 0.9 up to n = 400,
0.685–0.878 at n ≤ 160); no block length removes a bias, and the warning is
the boundary.

Temporal response surfaces and observation / Sequence tuple bands use their own
block rule, `max(max(span, ceil(sqrt(n))), min(ceil(b_PW·n^(1/6)), n/3))` with
`b_PW` read from the level's normal-equation scores and centered covariate
columns (recorded as `response.temporal.block_length`), the same
circular-Bartlett factor, and a per-cell kernel-bias factor `1/sqrt(f)`
(`response.temporal.kernel_bias_factor`): `f` is the share of the long-run
variance of the autoregression fitted to that cell's influence (Kendall AR(1),
or the BIC-selected AR(q ≤ 4) when larger) that the block's Bartlett kernel
keeps. `response.temporal.effective_rows` carries each cell's effective rows and
`response.temporal.block.short_series` warns below 15. Their 400-replicate
coverage at nominal 0.95 is gated for iid and AR(1) ρ = 0.5 residuals and, under
an AR(1) φ = 0.9 treatment, for both the dose curve (0.945–0.948 / 0.948) and
the shift response (0.943 / 0.948; 0.910 / 0.912 before the factor). AR(1)
ρ = 0.9 residuals (0.887–0.943 pointwise, 0.877 simultaneous at n = 160;
0.900–0.922 / 0.895 at n = 400; 0.902–0.943 / 0.905 at n = 1000) remain a
boundary record, disclosed on every band as
`response.temporal.block.persistence_boundary`: the persistent part is 15% of
the residual (lag-1 autocorrelation 0.13) and its long-run ratio is not
estimable at these n by any block length or fitted autoregression.

### Bayesian

* Bayesian g-computation;
* temporal Bayesian g-computation;
* Gaussian conditional effects with linear treatment–modifier interactions;
* Gaussian temporal mediation with observed baseline-parent adjustment and
  direct/mediated/total posterior decomposition;
* static and temporal Gaussian response estimation;
* multi-step sustained sequential g-computation with shared stationary
  mechanism draws;
* conjugate Gaussian models;
* Laplace GLM approximation;
* HMC GLMs;
* graph-by-effect posterior envelopes on the exact licensed DAG and
  `TemporalDag` query families described above;
* same-design prior transfer (including licensed Bayesian Pulse,
  single-step Sustained, and temporal `ResponseCurve` on explicit
  `TemporalDag`, when a fixture names source cell, target cell, and
  `PriorCatalog.filter_compatible`);
* effect-level and mapped prior transfer;
* prior catalogs and compatibility filtering;
* power-prior mixtures;
* conflict-sensitive prior weighting;
* transport policies across compatible designs.

Unidentified graph-posterior mass is retained rather than silently
renormalized away. Static DAG-posterior ATE (Bayesian and Frequentist) and
temporal DBN-posterior pulse / single- and multi-step sustained paths consume frozen
known-truth mixture fixtures: the identified atoms pin the conditional effect,
unidentified mass stays visible, and priors do not upgrade structural
identification. Prepared-vs-fresh equality remains an additional execution
invariant rather than the license.

Conditional effects, temporal mediation and DBN-posterior pulse/sustained effects license query-native `cheap` and `full` validation. Multi-step Sustained refuters re-estimate the complete sequential model. Bayesian checks retain each child mechanism for PPC; full prior sensitivity refits the composed effect. Single-regression sensitivity formulas are inapplicable to composed effects, while sequential unobserved-confounder perturbations remain available. Composed mediation and
multi-step sustained posteriors support conjugate and Laplace backends; HMC
composition remains refused. Per-cell evidence is in the [support matrix](support-matrix.md).

## Observation, transport, and interference

They change what identifies the estimand and are not hidden behind an
ordinary `target_population` flag: their design facts are explicit fields of
their queries. `TransportQuery` × `Admg` × explicit × Frequentist (Direct /
S-admissible sID plus binary trial-to-target IPW) and `InterferenceQuery` ×
`Dag` × explicit × Frequentist (NeighborCount under Bernoulli assignment,
HT/Hájek, Young variance) run on `analyze` and the Rust `Study` API at
validation `none`, and retain a study like every other licensed cell. On the
GCM path, `AnomalyAttribution` / `ChangeAttribution` × `Dag` × explicit ×
Frequentist run on `analyze` and the Rust `Study` API at validation `none`.

* **Observation** (`antecedent.observation`): complete, right/left/interval-
  censored, truncated, and selected mechanisms. Assumptions are declared
  separately from the recorded columns; MAR / independent censoring is never
  inferred from column presence.
* **Structural transport** (`antecedent.transport`, theorem-stage types in
  `antecedent.transport.advanced`): single-source selection
  diagrams, the `identify` stage, and `advanced.TransportQuery`, whose trial-to-target
  IPW reports separate selection and treatment overlap diagnostics
  (`result.transport_overlap`). `transport.advanced.estimate_trial_effect` remains an
  unlicensed IPW/AIPW utility that returns bare numbers. Distinct from Bayesian prior/evidence transfer in
  `antecedent.priors`.
* **Randomized interference** (`antecedent.interference`): assignment design,
  exposure mapping, and exposure-contrast estimands with Horvitz–Thompson and
  Hájek estimates (`result.interference`). The network and realized assignment
  are fixed and supplied by the caller on `InterferenceQuery`.
  A second retained frequentist construction accepts completely randomized
  clusters with an explicit matching `PartialInterference` partition,
  `NeighborFraction`, and the total exposure contrast `(0, 0)` to `(1, 1)`.
  It requires at least two clusters per arm and reports a conservative
  cluster-level variance without an interval or a Bernoulli calibration claim.
  `SaturationDesign` is a third retained construction: complete allocation of
  clusters to low/high treatment probabilities followed by independent
  Bernoulli assignment of units. With a matching `PartialInterference` partition,
  `InterferenceQuery` accepts `NeighborCount`, `NeighborFraction`, or
  `WeightedNeighborExposure` and estimates direct, spillover, or total contrasts
  from the requested exposure levels. Exact cluster-allocation and neighbor
  assignment probabilities are computed in Rust. The variance is a descriptive
  covariance-free proxy; no interval or support-matrix license is claimed.
  Direct and retained Python routes use the same native estimator.
  `ObservedExposureDesign` is a retained observational construction on the
  same query. It requires an explicit `PartialInterference` partition, a fixed
  within-cluster network, both exposure propensities and their known or
  externally estimated provenance, plus a declared no-unmeasured-network-
  confounding assumption. It supports neighbor count, fraction, and weighted
  exposure mappings, and reports exposed-unit and cluster counts, probability
  bounds, and a descriptive cluster-robust variance. The direct utility and
  retained analysis share the native estimator. Neither the propensities nor
  network exchangeability are verified by the library; the result is off-axis
  and has no licensed interval.
  `interference.estimate` remains an unlicensed utility over every design and
  exposure mapping; it returns bare numbers.
* **Randomized experiment ITT** (`antecedent.experiment`): `ExperimentDesign`
  carries assignment/outcome unit IDs and known assignment for a binary
  intention-to-treat contrast into graphless `analyze` / `prepare`. The native
  Rust route retains the original `RandomizedEffect` query and experiment
  identity rather than treating it as an interference query. Bernoulli,
  complete, stratified, and cluster assignment are retained. Bernoulli uses
  a Horvitz–Thompson contrast and its assignment-design variance; complete
  randomization uses difference in means with a Neyman conservative variance;
  stratified assignment uses block-weighted differences and blockwise Neyman
  variance. Cluster assignment uses unit-weighted cluster totals and a
  conservative cluster-level Neyman variance, requiring at least two clusters
  in each arm. The result retains design type, arm counts, block labels and unit
  IDs, and labels uncertainty without claiming a confidence interval.
  `SwitchbackEffect` also runs through retained `analyze` with row-aligned
  sequence and period labels, known marginal assignment probabilities, and
  a sequence-clustered sandwich variance. It requires two independent
  sequences with both arms observed, assumes no carryover and no interference
  between sequences, and reports no interval. These design queries are off
  the support-matrix axis; results report `evidence_status="off_axis"`.
  `experiment.ComplierEffect` uses the same retained randomized study path
  for independent Bernoulli encouragement with row-aligned treatment receipt.
  It reports the outcome ITT, positive receipt first stage, Wald CACE/LATE,
  and independent-unit influence variance under exclusion and monotonicity;
  it refuses zero or negative first stages and publishes no interval.
  `experiment.FactorialRandomization` carries a fixed four-cell 2×2 design
  through the same retained `RandomizedEffect` route. It requires at least two
  units per cell and reports both marginal main effects and their interaction
  with cellwise Neyman conservative variance estimates and no interval. The
  query and artifact retain the second-factor assignment, cell counts, and
  labels. `MultiArmExperimentDesign` carries three or more named actions,
  observed assignments, distinct assignment and outcome units, and each row's
  known action-probability vector through the same retained
  `RandomizedEffect` path. It requires observed support for every action and
  independent unit-level assignment. The result retains Horvitz–Thompson arm
  means and every contrast to the first action, with covariance-free variance
  bounds and no calibrated interval. The direct multi-arm utility and retained
  route share the native estimator; multi-arm combinations with CUPED, ANCOVA,
  receipt adjustment, or exact Fisher inference refuse.
  `experiment.estimate_complier_effect` also provides a direct randomized
  noncompliance ITT and Wald CACE/LATE point estimate with an
  influence-function standard error;
  `experiment.estimate_cuped_effect` provides one-covariate CUPED precision
  adjustment with a standard error. `RandomizedEffect(...,
  ancova_covariates=("baseline_a", "baseline_b"))` now carries multi-covariate
  ANCOVA through the retained Rust Study and Python `prepare` / `analyze`
  path for independent Bernoulli assignment with a common probability. It
  fits the treatment and pre-assignment covariate coefficients jointly and
  labels its independent-row HC0 variance; the result has no calibrated
  interval and is off the support-matrix axis. ANCOVA refuses non-Bernoulli
  designs, fixed CUPED or receipt adjustment on the same query, duplicate or
  collinear covariates, and non-finite values. The direct utilities likewise
  have no new support-matrix licenses or calibrated interval claims. The native
  `factorial.estimate` utility handles independent-Bernoulli 2×2 factors and
  returns cell means, main effects, interaction, and variance upper bounds;
  block and multi-arm factorial designs remain gaps. The direct
  `experiment.SwitchbackEffect` utility estimates unit-period ITT with a
  sequence-clustered standard error, requires at least two independent
  sequences with both arms observed in each, and assumes no carryover. It is
  point-only and unlicensed. The retained `SwitchbackEffect` route uses the
  randomized study path and preserves period identity in result artifacts;
  it adds no licensed coordinate.

```python
design = ant.ExperimentDesign(
    assignment=ant.interference.BernoulliAssignment(0.5),
    realized_assignment=assigned,
    assignment_units=account_ids,
    outcome_units=account_ids,
)
result = ant.analyze(
    {"outcome": outcome},
    query=ant.RandomizedEffect("outcome", design),
)

multi_arm = ant.MultiArmExperimentDesign(
    realized_assignment=actions,
    action_labels=("control", "low", "high"),
    assignment_probabilities=probability_rows,
    assignment_units=account_ids,
    outcome_units=account_ids,
)
multi_result = ant.analyze(
    {"outcome": outcome},
    query=ant.RandomizedEffect("outcome", multi_arm),
)
contrasts = multi_result.randomized_effect.multi_arm_contrasts
```

Multi-source meta-transport and cyclic/equilibrium models remain outside the
current contract. Observational network exposure has the explicit, point-only
retained route described above.

## Treatment policy evaluation

`antecedent.policy.BinaryPolicy` represents fixed binary recommendations or a
deterministic top-k ranking. `evaluate_policy` uses known randomized assignment
propensities to estimate value and value relative to a reference, while applying
declared treatment costs, availability, capacity, and budget limits. The caller
must supply evaluation rows held out from policy selection; the API cannot
verify that separation. It publishes point estimates only, so policy intervals,
learned-policy guarantees, and regret are not claimed. `uplift_by_score` also
reports held-out HT contrasts and standard errors over caller-supplied ranked
score bins; it does not fit or verify cross-fitted CATE scores. These are
point utilities and do not add licensed support-matrix cells. The same module
provides fixed multi-action recommendations with randomized
Horvitz–Thompson evaluation, per-action capacity, availability, cost, and
  budget checks. It requires known positive action probabilities and does not
  validate the held-out split.

`antecedent.policy.PolicyValue` runs a fixed binary policy through retained
`prepare` / `analyze` using each evaluation subject's outcome, known
assignment chance, and supplied outcome predictions. Its
recommendations, costs, capacity, budget, action availability, nuisance
predictions, and subject ownership are checked on the prepared evaluation
rows. A new evaluation sample requires a new prepare; `refresh(new_data)` is
refused because those row-bound inputs cannot be verified against replacement
rows. The result reports paired row-score standard errors under independent
evaluation subjects but no interval or licensed support-matrix cell.
`MultiActionPolicyValue(..., baseline_groups=...)` also retains pre-treatment
group labels and reports each action-versus-control contrast within each group
as a point-only multi-action CATE. Every group needs observed control and each
action, with known positive assignment probabilities. These conditional
contrasts are frozen in the result artifact; they do not fit a CATE model or
claim an interval or licensed support-matrix cell.

`policy.ConditionalDoseResponse(...)` carries a fixed target-dose grid through
`prepare` / `analyze`. Baseline group labels remain bound to the prepared row
order; the outcome, observed dose, and caller-supplied density travel as native
table columns. The direct `policy.estimate_continuous_dose_response` utility
and retained Study route use the same triangular-kernel inverse-density
estimator. For each group and target, the result reports the local response,
row count, effective sample size, minimum density, maximum normalized weight,
and descriptive local outcome SD. The result and artifact declare point-only
uncertainty. This evaluates a response curve under conditional exchangeability,
correct supplied density, consistency, no interference, and local positivity;
it does not learn or evaluate a continuous-dose policy or license an interval.

## Quasi-experimental point utilities

`antecedent.quasi.DifferenceInDifferences` estimates a repeated-cross-section
2×2 difference in differences with a native Rust kernel. The
`PanelDifferenceInDifferences` utility accepts a balanced two-period panel,
requires one pre and one post observation per subject and stable treatment,
then computes the treated-control mean difference in subject-level changes.
Both results assume parallel untreated trends, no anticipation, and no
interference; the panel utility additionally requires stable treatment within
subject. The original repeated-cross-section utility publishes a point only;
`PanelDifferenceInDifferences.repeated_cross_section(...)` also runs through
retained `prepare` / `analyze` and reports a cluster-robust standard error.
The balanced-panel route reports the same pointwise uncertainty with an
optional higher-level cluster column. Neither retained design reports a
p-value or interval. Both are marked `unlicensed_point_utility` and add no
support-matrix license.
Staggered adoption and synthetic-panel designs also run through the retained
`prepare` / `analyze` flow. Fuzzy regression discontinuity and regression kink
now use that flow as fixed-bandwidth, graphless local ratio queries, alongside
their existing direct utilities. Their retained result reports local counts,
first-stage strength, and a descriptive HC0 standard error, but no interval.
Staggered event studies also run through retained `prepare` / `analyze` with
`StaggeredAdoption(..., event_study=True)`; the direct utility shares its native
estimator.

`antecedent.quasi.StaggeredAdoption` adds a balanced-panel group-time ATT
utility. Cohort 0 is explicitly never treated; for each adoption cohort and
post period, the native estimator compares outcome changes from that cohort's
immediately prior period with changes among never-treated units. It reports
treated/control counts for every comparison. These counts are descriptive
support diagnostics, not overlap tests. Parallel untreated trends by cohort,
no anticipation, absorbing treatment, valid never-treated controls, independent
clusters, and no interference are assumptions; pretrends are not tested. Both
group-time ATT and the retained event-study route report pointwise
cluster-robust standard errors by default, using subjects as clusters or an
optional higher-level cluster column. The CR1-style multiplier is `G/(G - 1)`;
at least two distinct clusters must contribute to each cohort/control
comparison. Event-study pre-adoption contrasts are descriptive only. Neither
route reports p-values or intervals, validates assumptions, or adds a
support-matrix license; both remain `unlicensed_point_utility`.

`antecedent.quasi.SyntheticControl` and
`SyntheticDifferenceInDifferences` use the same graphless, row-aligned
synthetic-panel preparation path. The first fits convex donor weights and
reports pre-fit error, effective donor count, and descriptive, uncalibrated
leave-one-donor-out placebos. The second fits convex donor and pre-period
weights for a difference-in-differences contrast. Both freeze unit and period
identity, execute in native Rust, and export separate result sections in
portable artifacts. They require a balanced panel, explicit treatment timing,
no anticipation or interference, and a defensible donor counterfactual or
untreated trend. Results are point-only, remain `unlicensed_point_utility`,
and add no support-matrix license or calibrated interval claim.
For synthetic control only, a declared uniform one-treated-unit assignment
enables an exact Fisher sharp-null test that refits each possible treated unit.
It reports the full assignment distribution and a p-value, without converting
the effect's point estimate into an interval or a licensed matrix cell.
An optional positive `augmentation_ridge` on `SyntheticControl` fits a
donor-trained pre-to-post outcome model and corrects the simplex gap. The
retained result and artifact preserve the original gap, correction, and adjusted
point estimate. This declares outcome-model transport to the treated unit;
the adjusted effect remains point-only and off the support-matrix axis.

`RandomizedEffect.estimate` also exposes native direct utilities for other
assignment kernels. Those direct results are unlicensed and publish no
interval. The retained `analyze` route supports Bernoulli, complete,
stratified, cluster, fixed-cell 2×2 factorial, and independent multi-arm
designs; unsupported
combinations refuse there.

## Longitudinal regime value

`antecedent.regimes.LongitudinalRegimeQuery` retains one outcome row per
subject and the subject's complete binary treatment history in graphless
`prepare` / `analyze`. It accepts a static action sequence or a prespecified
action matrix, known sequential randomization probabilities, optional
censoring probabilities and observed-outcome flags, unique subject IDs, and
one fold ID per subject. The study freezes these design facts, reports the
regime value and weight diagnostics, and refreshes only the aligned outcome
rows. The answer is point-only with no effect or interval placeholder.
Caller-estimated probabilities are refused on this retained path, and no
support-matrix license or cross-fitting claim is made.

Set `method="g_formula"` and provide subject-by-period conditional reward
predictions to evaluate a prespecified regime through the same retained
query, native estimator, result, and artifact path. The predictions and
subject/fold ownership are frozen with the study. The caller owns the outcome
model and must justify its predictions; Antecedent does not fit or verify it.
This route reports a point value without an interval or support-matrix license.
Set `method="sequential_dr"` and provide cross-fitted Q predictions, aligned
subject and prediction fold IDs, monotone observation histories, treatment
probabilities, and conditional censoring probabilities to run the backward
recursive doubly robust score through the same retained Study path. The
caller owns nuisance fitting and the sequential exchangeability claim. The
result and artifact preserve the fold and dropout contract and weight
diagnostics; the value remains point-only without a calibrated interval or
support-matrix license.

`antecedent.regimes.evaluate_regime_value` evaluates a prespecified static or
history-adaptive binary regime from subject-level treatment histories. It uses
caller-supplied sequential probabilities for treatment and remaining observed
through each period, applying a trajectory-level inverse-probability score in
native Rust. Dynamic callbacks see the subject's past treatments and
covariates through the current pre-treatment period. The result reports a
point value, matched observed fraction, effective sample size, and maximum
weight, but no standard error or interval. The caller must justify consistency,
sequential exchangeability, and treatment/censoring positivity; probability
floors only refuse near-zero supplied support and do not diagnose a causal
design. This utility has no support-matrix license, does not fit nuisance
models or cross-fit. `evaluate_sequential_gformula` adds a plug-in value path
from caller-supplied period-specific conditional reward predictions. It checks
probability positivity and preserves one fold ID per subject in result
provenance, but does not fit models or verify that predictions are out of fold.
`evaluate_sequential_doubly_robust` adds backward-recursive augmentation from
caller-supplied Q predictions, treatment/censoring probabilities, and aligned
subject/prediction fold IDs. It checks dropout monotonicity and ownership
alignment but cannot verify actual out-of-fold fitting. These three paths remain
unlicensed point utilities without standard errors or intervals.
`fit_marginal_structural_model` estimates
an additive terminal-outcome MSM with one treatment coefficient per period.
Callers must provide conditional treatment probabilities and per-period
stabilizing numerator probabilities; numerator probabilities are never fitted
from the evaluation sample. Optional conditional censoring survival enters as
an unstabilized inverse probability. The native fit refuses positivity-floor
violations, overflow, insufficient observed subjects, or a rank-deficient
weighted design. Its CR1 subject-clustered sandwich standard errors are
pointwise only; it makes no confidence-interval, simultaneous-coverage, or
calibration claim. Fold IDs are preserved as subject ownership metadata when
provided, but propensity fitting and out-of-fold status are not verified. The
same native MSM kernel now also runs through the retained Study route:
`LongitudinalRegimeQuery.marginal_structural_model(...)` can be passed to
`prepare` or `analyze`, with subject-level fold ownership, probability and
numerator positivity checks, assumptions, coefficients, pointwise CR1
standard errors, and query/result artifact identity preserved. Its concise
factory does not ask for a regime action because the MSM fits period
coefficients rather than evaluating one policy. The direct and retained paths
remain off the support-matrix axis, report no interval, and do not verify
whether caller-supplied probabilities were trained out of fold. Each row is
one subject, so repeated periods cannot be split across analysis units.

```python
summary = ant.regimes.evaluate_regime_value(
    outcomes=terminal_outcome,
    treatment_history=treatment_by_subject_and_period,
    regime=[False, True, True],
    treatment_probabilities=conditional_probability_of_treatment,
    outcome_observed=terminal_outcome_observed,
    censoring_survival=conditional_probability_of_remaining_observed,
)
```

```python
msm = ant.regimes.fit_marginal_structural_model(
    terminal_outcome,
    treatment_by_subject_and_period,
    conditional_treatment_probability,
    stabilizing_numerator_probabilities=[0.4, 0.3, 0.25],
    subject_ids=subject_ids,
    fold_ids=subject_fold_ids,
    outcome_observed=terminal_outcome_observed,
    censoring_survival=conditional_remaining_observed_probability,
)

query = ant.regimes.LongitudinalRegimeQuery.marginal_structural_model(
    outcome="terminal_outcome",
    treatment_history=treatment_by_subject_and_period,
    treatment_probabilities=conditional_treatment_probability,
    stabilizing_numerator_probabilities=[0.4, 0.3, 0.25],
    subject_ids=subject_ids,
    fold_ids=subject_fold_ids,
    outcome_observed=terminal_outcome_observed,
    censoring_probabilities=conditional_remaining_observed_probability,
)
retained = ant.analyze({"terminal_outcome": terminal_outcome}, query=query)
period_effects = retained.longitudinal_regime.period_effects
```

## External estimator handoff and provider contracts

`antecedent.extensibility.CausalProviderSpec` declares identification,
distributions, nuisance functions, support, inference, fold ownership, output,
artifact codec, and provenance for integrations. It is descriptive and does
not execute provider code. A separately installed Python package can expose a
zero-argument provider factory under the `antecedent.providers` entry-point
group. The caller explicitly loads it with
`providers.load_entry_point("package_provider_name")`, then passes
`ProviderQuery("package_provider_name", {...})` to `analyze`. Importing
Antecedent never discovers or executes installed providers. Loaded providers
use the same externally attested result validation as direct registration;
entry-point provenance is retained in the result. No Antecedent or Rust rebuild
is needed to install the external package. The EconML-compatible handoff is the implemented
adapter: Antecedent emits certified `Y`, `T`, `W`, optional `X`, and aligned
weights; for temporal estimands it trims boundaries and reports row origins.
The handoff does not assign or verify cross-fitting folds—the external learner
owns them, and this is recorded as unknown to the receipt.

The separately distributed package declares its factory in `pyproject.toml`:

```toml
[project.entry-points."antecedent.providers"]
my_provider = "my_package.provider:create_provider"
```

`create_provider()` returns an object with a `CausalProviderSpec` at `.spec`
and an `execute(request)` method returning `ProviderExecution`. The provider
package can be installed or upgraded independently. The caller loads it by
name before submitting its `ProviderQuery`; an absent or ambiguous entry point
is refused, and no plugin can declare itself native licensed through this path.

A host can promote an externally attested provider to `verified_extension` by
calling `ProviderRegistry.verify(name, fixtures, evidence_origin=...)`. Each
`ProviderVerificationFixture` supplies an independent known-truth request,
expected estimate and uncertainty, assumptions, support label, provenance,
and an artifact SHA-256 digest. A caller-supplied decoder can check the artifact
round trip. The host runs every fixture through the ordinary provider runtime,
checks declared output shape and uncertainty semantics, and replays each
fixture when the provider declares deterministic behavior. Only after every
check passes does the registry record a `ProviderVerificationReport` and
promote that provider. Subsequent results carry the evidence digest and origin;
changing the provider's declared spec invalidates execution. An installed
entry-point provider can use this route without rebuilding Python or Rust.
The scientific truth and independence of the reference cases remain the
caller's responsibility. Verification covers those cases and does not grant
native licensing or interval calibration.

```python
from antecedent.extensibility import ProviderVerificationFixture, providers

providers.load_entry_point("my_provider")
report = providers.verify(
    "my_provider",
    [ProviderVerificationFixture(
        name="known-effect-1",
        request={"case": "known-effect-1"},
        expected_estimate=[1.0],
        expected_uncertainty=None,
        expected_assumptions=("randomized assignment",),
        expected_support_status="both_arms",
        expected_provenance={"package": "my-provider", "version": "1.0"},
        artifact_digest="<SHA-256 of expected provider artifact>",
    )],
    evidence_origin="independent-reference-suite/v1",
)
```

`EconMLSpec.attach` accepts a caller-fitted estimate and creates an artifact
binding its payload digest to the declared learner/configuration and available
identification and data-snapshot identities. It does not rerun the learner or
verify its provenance, fold ownership, or uncertainty calibration. Loaded
external results start externally attested with calibration unavailable. They
can become verified extensions only after the host runs independent fixtures;
that bounded verification still does not calibrate intervals.

## Interventions and counterfactuals

Antecedent includes a structural causal model layer.

Antecedent includes a structural causal model layer.

Supported mechanisms:

* linear-Gaussian models;
* constant mechanisms;
* discrete mechanisms;
* hierarchical linear and generalized-linear models;
* Minnesota BVAR;
* linear Gaussian state-space models;
* Gaussian-process mechanisms.

Supported interventions:

* hard interventions;
* soft interventions;
* stochastic interventions;
* sequenced interventions;
* temporal policies;
* dynamic policies;
* mechanism overrides.

Do-sampling methods include weighting, KDE, and MCMC.

Counterfactual primitives exist:

* abduction–action–prediction;
* nested counterfactuals;
* temporal trajectories;
* unit-level counterfactual analysis.

`analyze` licenses `Counterfactual` on an explicit or accepted DAG at
validation `none`, Frequentist and Bayesian, as a two-world GCM ITE. Unit ITEs
are exact only for invertible additive-noise mechanisms; downstream of a
discrete or state-space mechanism a unit effect is one sampled counterfactual
that varies with the seed. Nested counterfactuals, temporal trajectories,
graph-posterior structure, and cheap/full validation remain refused. The public license is the
[support matrix](support-matrix.md).

## Attribution and diagnostics

Antecedent can analyze:

* anomalous outcomes;
* distribution shifts;
* structural changes;
* mechanism changes;
* change points;
* unit-level change;
* path contributions;
* arrow strength;
* feature relevance;
* root-cause rankings.

`AnomalyAttribution` and `ChangeAttribution` on an explicit Dag
at Frequentist validation `none` run on `analyze` and the Rust `Study` API.
Mechanism-change,
unit-change, cheap/full, Bayesian, accepted, and graph-posterior stay
refused. The public license is the [support matrix](support-matrix.md).

Implemented techniques:

* likelihood-ratio tests;
* mean-difference tests;
* classifier-based tests;
* MMD;
* Gaussian KL divergence;
* CUSUM-style scans;
* Shapley attribution;
* coalition caching.

## Validation and sensitivity

Estimate validation:

* placebo refuters;
* random common-cause refuters;
* unobserved common-cause refuters;
* bootstrap refuters;
* data-subset refuters;
* dummy-outcome refuters;
* overlap diagnostics;
* E-values;
* graph refutation.

The 1.2 functional suites refit path-specific effects on row subsets and compare
entire interventional-distribution tables, including conditional strata.
Temporal mediation uses mediator-placebo and contrast-specific stability checks.
Static `MediationEffect` cheap/full uses a mediation-native suite: placebo
mediator (indirect-effect target), random common cause on the requested contrast,
binary mediator-range overlap when applicable, and an 80% subset on `full`.
Continuous-treatment conditional overlap reports unsupported mass under a
descriptive residual-support model; lower `comparison` values mean better
support. Passing any of these checks does not establish causal identification
or validate the structural assumptions.

Sensitivity methods:

* linear sensitivity;
* partial-linear sensitivity;
* nonparametric sensitivity;
* Riesz sensitivity.

Bayesian validation:

* prior predictive checks;
* posterior predictive checks;
* prior sensitivity;
* MCMC diagnostics;
* simulation-based calibration hooks.

Resampling support:

* IID bootstrap;
* Bayesian bootstrap;
* moving-block bootstrap;
* circular-block bootstrap;
* column permutation;
* phase-randomized surrogates.

### "Not applicable" means three different things

The words "not applicable" surface in three unrelated places. A caller who
only sees the bare phrase cannot tell which claim is being made — each is a
different strength of statement, and only one of them is permanent:

* **The support matrix's `not_applicable`** (`SupportRefusal::NotApplicable`,
  wire id `not_applicable`). This is the strongest claim in the system: the
  coordinate — a fixed (query, graph class, structure, inference, validation)
  cell — does not denote, permanently, independent of any run's data. See the
  [support matrix](support-matrix.md).
* **`antecedent-validate`'s `NotApplicable`** (`ValidationOutcome::NotApplicable`
  / `ValidationError::NotApplicable`). This is a per-run, data-dependent skip:
  a requested validator is incompatible with *this run's* problem — an
  E-value on a non-binary treatment, an MCMC diagnostic on a non-MCMC
  posterior, a refuter outside its applicable regime. The same validator can
  run cleanly on a different dataset against the same licensed cell. Callers
  that only read `result.refutations` cannot see this skip — the produced
  `RefutationReport`s and the skips are two disjoint outcomes, and
  `result.refutations` carries only the former. Every execute path now emits
  one `refute.validator.not_applicable` diagnostic per skipped validator into
  `result.diagnostics`, naming the validator and the reason, so the skip is
  visible instead of silently dropped. Its message states explicitly that
  the skip is per-run and data-dependent, not a permanent support-matrix
  refusal, so the two senses of "not applicable" are never mistaken for each
  other at the point a caller actually reads them.
* **The response path's NaN scalar summary**
  (`estimate.response.no_scalar_summary`). A function-valued or
  not-point-identified response has no single-number effect summary;
  `result.effect` is `NaN` and a diagnostic states the scalar reading is "not
  applicable" — the caller must read `result.response` instead. This is not
  an error and not a refusal; it says the wrong field was checked, not that
  anything failed.

None of the three imply each other. A licensed cell can still emit a
per-run validator skip or a NaN scalar summary; a matrix `not_applicable`
cell never reaches either of the other two because `analyze` refuses it
before validation or estimation runs.

## Experimental design

Antecedent can rank candidate actions such as:

* measuring a variable;
* intervening on a variable;
* observing an environment;
* changing a sampling plan.

Ranking criteria:

* expected information gain;
* probability of identification;
* expected effect-interval width;
* decision utility.

The design layer supports batched Monte Carlo evaluation, common random
numbers, and early stopping.

## Incremental state

`CausalState` supports stateful and online workflows.

Available components:

* explicit invalidation;
* incremental OLS;
* streaming covariance;
* particle-filter state-space models;
* local score caches;
* rolling mechanism diagnostics;
* configurable cache budgets;
* prepared analyses;
* progressive and cancellable execution;
* opt-in adaptive resampling bounded by Monte Carlo error (production evaluates the full request).

Invalidation does not automatically rerun an analysis.

`PreparedStudy` (`Study::prepare`) caches identification across the licensed
prepared paths. For `AverageEffect`, an estimate click reuses prepare-time
identification (`exec.identify.cached`) on `Dag`, `Cpdag`, `Pag`, and `Admg`,
including the CPDAG MEC envelope, the generalized-adjustment PAG envelope, and
the general-ID bidirected ADMG functional. Static graph
posteriors freeze each weighted atom's
identified/unidentified status, result, and estimand; temporal DBN posteriors
also freeze each identified atom's finite-unfolding indexer. Unidentified
atoms remain unidentified and keep their original weight. Temporal response
and mediation paths retain their existing query-native caches. This is an
execution property, not a license: refused coordinates remain refused.

The sharp-RD estimator remains the deliberate identify-per-click exception.
Progress sinks receive an `identify.compute` label exactly when identification
is computed, so a prepared click's reuse is observable rather than merely
flagged. Prepared graph and query identity are immutable; a changed graph or query
requires a new prepare cycle, so no cache can cross coordinate boundaries. A
same-schema data refresh may reuse structural identification but still
re-estimates from the replacement data.

## Data support

Antecedent supports:

* tabular data;
* time series;
* panel data;
* multi-environment data;
* event data converted into temporal frames.

Python interfaces support NumPy, pandas, and Arrow CDI. Rust uses `TableView`.

## Artifacts

Durable artifact format **0.5** is current; it includes the optional
identified-set interval on structural-mixture analysis results. Format 0.4 was
the 1.0 wire freeze, and 0.4 artifacts migrate unchanged. Versioned artifacts
include:

* graphs;
* graph posteriors;
* model bundles;
* analysis traces;
* causal state.

Artifacts use schema-versioned CBOR containers with optional
Zstandard-compressed sections, selective reads, and memory-mapped access.
