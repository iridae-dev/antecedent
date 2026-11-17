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
| 1 | Critical | A chain's posterior or predictive law reaches nonlinear multi-action utility; the stricter action-world entry point checks distinct regime outputs and shared factors before compiling paired regret; a new cyclic-model route derives paired rows from shared disturbances and hard-clamp worlds | Fit mechanism uncertainty jointly with the intervention worlds and calibrate any population predictive interpretation; the new route is conditional on its specified model and disturbance table | None for a new route; high if parameter draws are reinterpreted as outcome risk |
| 2 | Critical | Uncertainty survives discovery, imputation, estimation, transport and decision; individual and some composite routes exist | Lower the full selection, fitting and transport schedule with dependence; measure coverage at the final coordinate | None for a new route; high if component intervals are relabelled as whole-chain coverage |
| 3 | High | Transported or synthesized evidence feeds a distributional decision; `compile_transported_law_chain_decision` now checks the transport support policy and compiles an aligned, equally weighted declared law with its receipt into multi-action utility | Lower producer fitting, target population, overlap and pooling dependence into that same program | None for a new route; medium if current point transport claims are promoted to laws |
| 4 | High | Sampling error composes with an assumption range; the chain envelope keeps the axes distinct, and `chain_robust_decision_v1` compiles conservative multi-action dominance with nonlinear utilities and chance constraints | Compile a calibrated sampling union at the exact output coordinate | None for a new route; high if an assumption range is treated as a confidence interval |
| 5 | High | A policy learned on logged data is evaluated on that data; the version-2 graph route checks a declared causally sufficient DAG for backdoor separation, pre-treatment adjustment and an explicit policy information set before compiling the same held-out or cross-fit estimator | Permit an information set narrower than the estimator's adjustment set by separating policy-learning features from nuisance features; calibrate the selection-aware standard error | None for the additive route; high if same-data scores acquire an unbiasedness claim |
| 6 | High | Graph uncertainty and missingness compose for the bounded binary average effect, including graph search rerun per imputation; the retained mixture draws now export an unverified typed effect law | Lower richer treatment targets and the joint graph, missing-data and treatment-world draw schedule | None for a new route; high if separate marginal bounds are multiplied into a posterior |
| 7 | Medium | A scientific surrogate bridge already composes with transport and nonlinear utility over joint draws | Lower the bridge declaration, transport and utility together with checked value kind and coordinate | None for a new route; high if a surrogate score is substituted for an endpoint |
| 8 | Medium | Scalar and vector feedback compile one cyclic program with aligned disturbances and named hard clamps; optional ledger entry points check disjointness across all inputs of the original component and bind the ledger state into the program | Establish statistical dependence of fitted mechanism errors and calibrate the connected equilibrium before promoting component evidence to an inferential claim | None for a new route; medium if the acyclic interface contract is reinterpreted as cyclic |
| 9 | Medium | Static path-specific transport and fixed-population temporal counterfactuals exist separately | Prove the target temporal path functional under selection and mediator assumptions, then lower its estimator and claim | None for a new route; high if ordinary transport is relabelled path-specific |
| 10 | Medium | Independent indicators retain the version-3 route; a separate fitted joint-error compiler cell now uses labelled validation counts and a Dirichlet posterior to propagate dependent measurement uncertainty through the finite query; an ignored fixed-truth grid harness is source-ready | Register and earn whole-route coverage records, then add a calibration lookup for the equal-tail outcome-probability interval; its present status is `not_measured` | None for the additive route; medium if independent indicators are silently treated as a joint model |
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
chain, contract and bounds. Neither route establishes a calibrated sampling
union or confidence region; that compiler obligation remains open.

The scientific bridge's exported chain stage now carries its source-law
meaning, output reading and target quantity. A predictive source remains
predictive when uncertain mapping coefficients are applied; a new outcome
residual is separately labelled an innovation. The chain rule accepts that
declared combination, and the checked result's `compile_chain_stage` lowers the handoff
through the central chain compiler as a predictive law.
The bridge preparation and transport declarations are not yet lowered in one
compiler program, so the remaining row-7 obligation is still open.

An already declared transported joint law can enter `chain_decision_v1` with
its receipt identity bound to the compiled chain input. The ingress checks
the response's support label against the caller's decision policy and refuses
any unsupported law column. It requires equal weights because this chain
adapter does not yet retain the transport artifact's weight vector. Independent
consumption replays the chain and utility; it does not refit the original
transport estimator or recheck its source population's evidence. The transport
receipt remains a declared premise, with no new coverage claim.

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
backdoor separation after deleting treatment out-edges. The current policy
learner reads every adjustment column, so the declared information set must
equal that set. This is a graphical check conditional on the supplied DAG's
truth and causal sufficiency, not an empirical test of exchangeability. The
graph route has a separate identification tag and version-2 replay feature;
requests without a graph keep their version-1 artifact and identity.

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
no record is earned or runtime lookup enabled by that source-only fixture.

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
