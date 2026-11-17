# Compiler routes for composition

This is the implementation ledger for composition across Antecedent's existing
families. Every executable addition starts with a declared scientific question,
passes the identification and carrier checks, lowers to a six-layer program,
executes that retained program, and publishes a versioned artifact with an
independent consumer. A Python adapter may construct a declaration or read an
answer; it cannot establish a new scientific claim.

The **claim-change** column measures the existing scientific claims or
calibration that would have to be withdrawn if the route were implemented by
changing old semantics. An additive, separately versioned route should keep
that number at zero. This is not an estimate of engineering effort. New routes
still need their own identification proof and calibration where they assert
coverage or error control.

| Priority | Limitation impact | Useful flow and existing support | Remaining compiler obligation | Existing claim change |
| --- | --- | --- | --- | --- |
| 1 | Critical | A new joint cyclic posterior route samples fitted mechanisms from one likelihood, draws population disturbances from the specified covariance, solves each hard-clamp world on the same parameter and disturbance pair, then compiles nonlinear multi-action utility | Prespecify and earn final population-predictive reliability or coverage at the exact decision coordinate; the model and covariance remain assumptions | None for an additive route; high if parameter draws are reinterpreted as outcome risk |
| 2 | Critical | Uncertainty survives discovery, imputation, estimation, transport and decision; individual and some composite routes exist | Lower the full selection, fitting and transport schedule with dependence; measure coverage at the final coordinate | None for a new route; high if component intervals are relabelled as whole-chain coverage |
| 3 | High | A declared transported law can enter multi-action utility; a new joint Bayesian transport-effect decision program binds the raw selection diagram, source and target rows, shared Gaussian fit, overlap rule and effect utility, then independently refits and replays them | Extend producer-fitted decisions to outcome-predictive and other transport families; earn final decision reliability at each exact coordinate | None for the additive route; medium if current point transport claims are promoted to laws |
| 4 | High | A new compiler cell replays the original and whole-chain bootstrap assumption envelopes at one terminal coordinate and forms a Bonferroni-split basic-bootstrap union; robust decisions remain conditional on the supplied draw table | Prespecify and earn coverage of the whole bootstrap schedule before calling the candidate interval calibrated or feeding it into a confidence decision | None for the additive route; high if an assumption range is treated as a confidence interval |
| 5 | High | The graph-checked policy route separates a narrower policy information set from the full nuisance adjustment set in a version-3 artifact, while retaining held-out or cross-fit scoring | Calibrate the selection-aware standard error for the exact learner and evaluation schedule | None for the additive route; high if same-data scores acquire an unbiasedness claim |
| 6 | High | Graph uncertainty and missingness compose for the bounded binary average effect, including graph search rerun per imputation; the retained mixture draws now export an unverified typed effect law | Lower richer treatment targets and the joint graph, missing-data and treatment-world draw schedule | None for a new route; high if separate marginal bounds are multiplied into a posterior |
| 7 | Medium | A scientific surrogate bridge already composes with transport and nonlinear utility over joint draws; the new predictive bridge-transport-decision cell binds all three declarations in one program and replays both the bridge and utility | Estimate or validate the transport premise itself, rather than only carrying its declared evidence; measure any final decision reliability claim | None for a new route; high if a surrogate score is substituted for an endpoint |
| 8 | Medium | Scalar and vector feedback compile one cyclic program; a new joint likelihood posterior carries fitted parameter dependence into connected predictive equilibria and paired decisions | Calibrate the connected equilibrium and generalize fitted component evidence beyond one specified model and covariance | None for a new route; medium if the acyclic interface contract is reinterpreted as cyclic |
| 9 | Medium | A new time-ordered finite path-specific transport cell identifies each edge-intervention world under a selection diagram, evaluates the target contrast on exact laws, and replays both derivations | Develop sampled and continuous longitudinal path estimators, with uncertainty at the transported contrast | None for the additive point route; high if ordinary transport is relabelled path-specific |
| 10 | Medium | Independent indicators retain the version-3 route; a separate fitted joint-error compiler cell propagates labelled validation uncertainty; an ignored fixed-truth grid harness and exact model-coordinate calibration lookup are source-ready | Register and earn whole-route coverage records before the new equal-tail interval can report a measured status | None for the additive route; medium if independent indicators are silently treated as a joint model |
| 11 | Lower | `compile_chain_bundle_recomputation` lowers a fully embedded bundle and executable propagation-chain terminal together; a version-2 portable receipt reconstructs the graph and terminal compiler from bytes, reruns every dependency consumer and compares the live result | Generalize live recomputation to other executable node kinds | None for the additive route; low for numerical claims, but identity-only plans cannot become execution receipts |

## Current source slice

`propagation_chain_law_v2` retains the aligned output column and row weights
for posterior and predictive readings. Its predictive mapping distinguishes
parameter posterior draws from an outcome innovation; a predictive output is
required to declare an outcome or utility coordinate and is labelled noncausal and
uncalibrated. The chain is still compiled by
`compile_propagation_chain`, and the independent consumer replays the stored
law. A single fixed-reference decision utility stage is represented in the
lowered evidence and decision layer; its verdict is already replayed with the
stage result. The version-2 decision must be the output utility stage and
declare its utility coordinate. Version-1 records remain consumable. The
additive `chain_decision_v1` route compiles an existing `DecisionContract` over
one retained law coordinate or a version-4 joint output law, including nonlinear utility, multiple actions,
constraints and regret. Its consumer independently replays the chain and
decision; posterior parameter draws refuse predictive outcome risk. A joint
output law retains multiple stage columns with one common row-weight vector;
the consumer replays every column. Action-specific worlds and their coupling
can be declared as paired joint rows through a stricter compiler entry point.
It refuses independent or undisclosed source columns, requires a common model,
parameter or disturbance factor, and requires each action to name a distinct
retained regime. The existing artifact replays the paired regret. This checks
the coupling declaration but does not derive it from an intervention model.

The additive `cyclic_derived_action_world_decision_v1` program binds a checked
cyclic structural model, distinct hard-clamp interventions, their exact outcome
coordinates and a multi-action decision contract before numerical execution.
It solves every world on the same explicit disturbance rows, requires a unique
certified equilibrium for every row, then sends the derived paired columns
through the chain and decision compilers. Its separate six-layer identity
includes the source compiler identity, output mapping and contract. Its
portable artifact independently replays the cyclic source, recompiles the
model-to-decision program and consumes the chain decision. This method is a
finite empirical disturbance-scenario comparison conditional on specified
mechanisms. It does not fit uncertain mechanisms, infer the disturbance law
from a population or claim measured coverage. The source test is unearned.

`cyclic_posterior_action_decision_v1` jointly fits the selected cyclic
parameters from intervention-labelled rows under the specified likelihood and
normal prior. It draws one parameter vector from that joint posterior and one
Gaussian disturbance from the declared covariance per retained row, then
solves every hard-clamp action world on that same pair. All worlds must have a
certified unique equilibrium; MCMC convergence diagnostics must pass. The
joint predictive outcomes enter the action-world chain and decision compiler,
and a portable consumer reruns the posterior, worlds and decision under the
outer six-layer identity. Model structure, covariance, prior, training rows,
sampler policy, seed, action worlds and utility all enter that identity. This
is a model-conditional population predictive method; the covariance and
structural equations are assumptions. Its ignored reference assertion is
unearned, and no calibrated population coverage or decision reliability is
claimed.

`propagation_chain_assumption_v3` adds a second route for aligned sampling or
posterior draws combined with interval-valued assumptions through rowwise
transformations. The compiler carries a separate `assumption_envelope` reading
and realization, including the draw-law meaning; the retained record stores one lower and upper enclosure per
draw row and replays them independently. Its reported `lower` and `upper` are
the weighted means of those endpoint columns, bounding the mean under every
fixed admissible assumption. They have **no measured coverage**. The route
refuses pooling and preposterior updates, which require additional law methods.
A terminal fixed-reference utility stage lowers with an identified-set carrier
and reports whether every admissible assumption favors acting, favors withholding,
or leaves the comparison unresolved, conditional on the supplied draw table.
The additive robust-decision compiler evaluates a `DecisionContract` with
maximin-over-structures criterion over those rowwise enclosures. Interval
arithmetic bounds nonlinear utilities and chance constraints for every action;
it reports a winner only when one certainly admissible action strictly
dominates every possibly admissible competitor. The consumer replays the
chain, contract and bounds. Neither route by itself establishes a calibrated
sampling confidence region.

`sampling_assumption_union_v1` binds the original sampling-plus-assumption
chain and each full-chain bootstrap rerun to one six-layer program. It requires
the same terminal quantity and stage schedule and fixes every interval-valued
assumption while allowing sampling rows to change. The weighted lower and
upper means are recomputed through each chain compiler. A basic bootstrap
one-sided bound for each endpoint uses half the nominal error budget, giving
one conservative union candidate for the final identified set. The portable
consumer recompiles all nested chains and compares the terminal result. The
replicate generator and its sampling-unit design remain declared provenance;
the route reports `not_measured` and cannot confer coverage on a robust
decision until an exact-coordinate whole-schedule record is earned. Its
ignored reference assertion is unearned.

The scientific bridge's exported chain stage now carries its source-law
meaning, output reading and target quantity. A predictive source remains
predictive when uncertain mapping coefficients are applied; a new outcome
residual is separately labelled an innovation. The chain rule accepts that
declared combination, and the checked result's `compile_chain_stage` lowers the handoff
through the central chain compiler as a predictive law.
`bridge_transport_decision_v1` now lowers a predictive source table, its
checked bridge contract and explicit target-scope transport premise, and the
nonlinear decision contract together. It requires the exact terminal-outcome
coordinate and predictive value kind before mapping, then derives the bridge
law and compiles its stage and decision. The portable consumer replays both
the scientific bridge artifact and the chain decision under the outer program
identity. The premise is still an assumption, not an estimated transport
functional. It earns no new coverage or decision-reliability claim; its source
reference assertion has not been run.

An already declared transported joint law can enter `chain_decision_v1` with
its receipt identity bound to the compiled chain input. The ingress checks
the response's support label against the caller's decision policy and refuses
any unsupported law column. It requires equal weights because this chain
adapter does not yet retain the transport artifact's weight vector. Independent
consumption replays the chain and utility; it does not refit the original
transport estimator or recheck its source population's evidence. The transport
receipt remains a declared premise, with no new coverage claim.

`joint_transport_effect_decision_v1` is a separate fitted producer route for a
target **average treatment effect**. Its outer six-layer program binds the raw
selection diagram and transport query, source trial units and target covariate
sample, the shared and source-varying Gaussian model, priors and their data
ownership, known outcome variances, overlap tolerance, posterior seed, exact
effect coordinate and utility contract before fitting. The existing joint
Bayesian producer verifies support and source independence, fits one conjugate
posterior, and retains aligned source and target effect draws. The target effect
column then enters the posterior chain and decision compiler. A combined
portable receipt reconstructs the raw graph and query, reidentifies the
transport certificate, refits the complete source-target producer, recompiles
the effect decision and compares both results. This is a decision about the
posterior of a causal functional, conditional on the supplied model and
selection premises. It is not a predictive outcome-risk law. The composed
decision reports `not_measured`; its source assertion and calibration have not
been run or earned.

The cyclic component adapter checks each scientific port and joins scalar
mechanisms before preparing the complete connected model through the existing
cyclic compiler. It requires the solver's global contraction certificate and
retains explicit aligned disturbances and hard-clamp worlds. Existing acyclic
interface composition continues to refuse feedback. The adapter does not
promote any local calibration or support label to the connected equilibrium;
the multi-output entry point names each produced equation and its consumed
ports, then compiles the expanded system through the same cyclic compiler.
One input may feed several output equations with separate coefficients.
The additive `prepare_component_feedback_with_ledger` and
`prepare_vector_component_feedback_with_ledger` entry points check every pair
of inputs marked `DisjointEvidence` against active row-level ledger sources.
The vector check runs before output expansion, so inputs read by separate
equations of one component are still compared. An unknown or shared overlap
refuses; the ledger state digest enters the structural equation identities and
therefore the central cyclic compiler program. This is an evidence identity
and overlap check, not a calibrated inferential claim for the equilibrium.

The observational policy compiler retains the existing held-out and cross-fit
learning estimators under a source digest of all scored rows, covariates,
evaluation settings and the split seed. It refuses a policy declared as
learned on its own scoring rows before fitting, and its replay consumer
recompiles and re-executes the estimator. The existing estimate remains
conditional on declared adjustment sufficiency; its standard error still has
`not_measured` calibration.
An additive graph declaration includes the DAG, treatment and outcome nodes,
and the columns available at decision time. The compiler checks DAG validity,
forbids treatment descendants among the adjustment columns, and tests
backdoor separation after deleting treatment out-edges. This is a graphical
check conditional on the supplied DAG's
truth and causal sufficiency, not an empirical test of exchangeability. The
graph route has a separate identification tag and version-2 replay feature;
requests without a graph keep their version-1 artifact and identity.
An additive version-3 artifact handles a strict policy-information subset. Its
learner fits action predictions only on the named information columns, while
the propensity and outcome nuisance models still use the full backdoor
adjustment set. The feature mask enters the nuisance-cache key, so predictions
from different policy schedules cannot be reused. The graph declaration and
all rows remain bound to the compiler identity, and independent replay refits
both model families. The held-out and cross-fit score standard errors still
report `not_measured`; no selection-aware coverage record is earned.

The joint measured-signal route multiplies each checked `P(m_i | C)` only under
the explicit declaration that indicator errors are independent given the same
construct. It then conditions the complete finite model law, optionally after
an intervention, through a new compiler cell and a version-3 observation
artifact. The producer and consumer evaluate the same joint likelihood;
measurement-rate uncertainty requires a separate fitted-model method and
calibration records. An additive dependent-error route
requires the complete normalized `P(m_1,...,m_k | C)` table for every construct
level, with indicator domains, population, regime and evidence identity. It
looks up the recorded indicator cell once and conditions the finite law using
that joint likelihood. It has a separate compiler cell and version-4 replay
artifact; fitting uncertainty and coverage remain unmeasured on that declared-law route.
The additive fitted route accepts complete labelled joint-cell counts, one
positive symmetric Dirichlet concentration per cell and a declared validation
evidence identity. It compiles fitting, posterior draws and conditioning as one
new six-layer program, samples each construct row jointly, then runs each
sampled table through the dependent-measurement compiler. The reported plug-in
law uses the posterior mean table; its aligned draw rows represent uncertainty
in the fitted error table only, conditional on the composed mechanism model.
An equal-tail posterior interval of any outcome-level probability is available
with `not_measured` calibration. The retained-plan consumer recomputes every
draw and compares the report. New whole-route coverage records are owed;
the older independent and declared dependent routes retain their meanings.
`fitted_dependent_measurement_calibration.rs` prespecifies the joint validation
table, a labelled multinomial sampling design, a 150/300/600-per-construct
grid, the exact outcome-probability truth and the route's equal-tail 95% rule.
It is ignored and has not been added to the frozen calibration family roster;
no record is earned by that source-only fixture. A prospective group, its
source-fixed-family review and the exact 150/300/600 design are staged in
`parity/pending_composability_calibration.toml`. The active gate and its
measured-family manifest remain unchanged until a permitted refresh and run.
The compiled plan can now construct the exact calibration basis from its
mechanism-plan identity, measured cell, scoped joint-error table shape, prior,
outcome level, validation size and posterior draw count. Its interval lookup
uses that basis and currently returns `not_measured`. The ignored harness binds
the construction it receives from this runtime lookup, so a future earned
record is constrained to the same model and query coordinate rather than
generalizing one fixed-truth simulation to unrelated mechanisms.

`temporal_path_transport_v1` takes a finite selection-diagram problem, labels
every node with a time, and requires every directed edge to advance time. It
constructs the baseline and active edge-intervention worlds through the
cross-world query contract, identifies each target-population outcome event
against the same catalog, and evaluates their contrast on the same exact
regime laws. Time, path edges, selection premises, laws and evaluator limits
enter its six-layer compiler identity. A portable artifact independently
consumes both finite counterfactual transport proofs and reconstitutes the
outer program and contrast. A source known-truth test compares the result to
an independent target SCM oracle; it is not earned. This route makes an exact
point claim conditional on the supplied finite laws. It has no sampled
transport interval or general continuous longitudinal estimator.

The graph-imputation point consumer retains the actual pooled effect draws
from its source-bound execution. An additive exporter binds those draws to the
exact binary average-effect quantity and the replay receipt. It labels them a
model-conditional causal-functional posterior with unverified trust and
point-only calibration. This is a narrow downstream law handoff, not a
whole-schedule calibration or a treatment-world predictive law.

Bundle recomputation has a first executable slice: a fully embedded bundle
whose terminal is a propagation chain. A new six-layer program binds the
bundle identity, terminal node and retained chain declaration. Execution
requires every embedded artifact's consumer to replay and then compares the
live compiler result with the terminal record; its receipt carries every
dependency's artifact identity and Merkle digest. Reference nodes and a
historical recalculation receipt alone do not qualify as live execution.
The portable version-2 receipt additionally embeds the bundle so a new
consumer can reconstruct its Merkle graph under an independently retained
identity, compile the terminal from the embedded chain declaration and replay
the same live receipt without the original in-memory plan. Other executable
terminal kinds still need their own compiler adapters.
