# Carry uncertainty through a composed analysis

## Run a small example

Compute the product of two columns from one aligned synthetic law; check the hand-derived mean and variance before requesting an interval.

<!-- example: examples/python/composed_uncertainty.py -->

Continue below for the exact contracts and refused adjacent requests.

A quantity computed from several uncertain inputs has the spread of the
composition, not of its components. Two routes read that spread at the composed
coordinate. Neither produces an interval without a measured record.

| Route | Label |
| --- | --- |
| Composed spread, interval candidate and exceedance decision | `uncalibrated_spread`; candidates `unmeasured` |
| Whole path of the finite Markovian cross-world row | `uncalibrated_spread` |

The whole-path account is a sealed sibling of the cross-world execution:
`CrossWorldExecution.whole_path_uncertainty("fixed_units" | "independent_sample")`
in Python (`CrossWorldExecution::whole_path_uncertainty(UnitSampling, ctx)` in Rust)
returns a `WholePathAccount` with `inspect()` and `verify_against(execution)`.

## What each kind of dependence information licenses

- A joint moment (a covariance matrix) gives the exact variance of a linear
  combination, and a mixture with known weights gives its mean and variance.
- An aligned joint law (rows that are realizations of one joint law, proven by a
  shared alignment identity) licenses nonlinear functionals by row pairing.
- Only the marginals give a structural envelope (Frechet/Hoeffding, Makarov): an
  envelope is neither a point nor an interval and does not substitute for a
  joint law. A declared correlation range is an assumption range, labelled so.

Pairing the rows of unrelated marginal draws by index silently assumes
independence and is refused. A transported claim's uncertainty is three
pieces: source sampling, target sampling and the transport functional; naming
some and omitting others is refused with the missing pieces listed.

## Composed interval and exceedance

An interval construction at the composed coordinate (normal, draw percentile or
marginal) is a candidate with calibration status `unmeasured`. The interval
reading stays closed until a `CoverageRecord` measured at exactly that
`ComposedCoordinate` (operation, route, transported or not, input roles) and
construction exists, and then only at its recorded nominal level. A
component's record is never a composed coordinate's. An exceedance decision
carries a Monte Carlo band, which is numerical error and not coverage.

## Whole path

The path is abduction, action, per-world prediction and summary. The route
accounts for four sources: structural (declared, not propagated), posterior (the
exact predictive variance of the unit's counterfactual quantity under its
exogenous posterior, all worlds reading one state), sampling (fixed units or an
independent sample, which you declare) and numerical (exact enumeration, no
draws). The two spreads, `realized_mean_predictive_variance` and
`unit_mean_sampling_variance`, are for different estimands and are never added.
The derivation re-abduces every unit and refuses a result computed from other
mechanisms, evidence, worlds or quantity. The account is not a field of the
execution's wire: a consumer derives it again from the consumed execution and
the declared unit sampling, and `verify_against` refuses an account that does
not re-derive from that execution (`cross_world_uncertainty.account_mismatch`).
Units times joint exogenous states are bounded; sampled-path and transported
cross-world executions have no whole-path derivation and are refused.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Interval of a composition with no record at its coordinate | closed reading (`cell_not_licensed`) | Use the spread, or measure the coordinate |
| Row pairing of unaligned or marginal inputs | `joint_law_required` (`alignment_mismatch`, `index_pairing_refused`) | Supply draws from one joint law |
| Independence declared for inputs of unknown evidence overlap | `sampling_dependence_unknown` (`joint_law_propagation.evidence_overlap_unknown`) | Register the sources in the evidence ledger |
| Independence declared for evidence the ledger shows is shared | `evidence_double_counted` (`joint_law_propagation.shared_evidence_declared_independent`) | Treat the inputs as dependent and supply a covariance or an aligned joint law |
| Some transport pieces named, some omitted | missing-pieces refusal | Name source sampling, target sampling and the transport functional |
| Unit sampling not declared | `sampling_dependence_unknown` (`cross_world_uncertainty.unit_sampling_undeclared`) | Declare fixed units or a sample from a population |
| A sampled-path or transported cross-world path | `cell_not_licensed` (`cross_world_uncertainty.sampled_path_not_licensed`, `transported_path_closed`) | Use exact enumeration in the population the mechanisms were declared for |
| Fitted or sampled mechanisms in a cross-world row | closed | Use the sampled counterfactual route with its posterior input |
| Interval or posterior from the whole path | closed | Use the labelled spread |

## Uncertainty of an adaptive procedure

Enumerate the declared two-unit procedure, preserving its undefined contrast mass. The report is conditional on the supplied truths and design.

<!-- example: examples/python/adaptive_procedure.py -->

An adaptive experiment's data are shaped by the assignment policy, outcome delay
and loss, exposure failure, the stopping rule and the final selection. This
route derives the uncertainty of the complete declared procedure and keeps
inferential paradigms apart: a report of one paradigm is never a statement of
another. Every result is labelled derived and unmeasured: the law is exact under the
stated model but no coverage of any method has been measured.

In Python, `AdaptiveInferenceSession.derive_procedure(elements)` returns the `AdaptiveProcedure` whose `report` derives the uncertainty (`antecedent.adaptive_inference`). Every premise failure of admission carries the registered code `cell_not_licensed` with its own detail.

### Declaring the procedure

A `ProcedureDeclaration` holds the assignment rule, enrolment schedule,
outcome observation (lag and loss), exposure model, stopping rule and final
selection. Nothing is defaulted; a missing element is refused with its own
detail. The validated `ProcedureSpec` is the only input of the routes. The
design is batched sequential with Bernoulli outcomes, at most 4 arms, 4
contexts, 8 cells, 24 looks and 65,535 units. Declaration capacity is separate
from the cost of deriving the uncertainty: the exact law is bounded by its live
states (1,000,000) and unit expansions (60,000,000), so a procedure of a few
hundred units or more is refused there with `exact_law_states_exceeded` or
`exact_law_work_exceeded` and is derived by the whole-procedure simulation
instead, which is admitted against a work budget (replications x units x
measurements, 4,000,000,000) and the memory budget of the execution context
(`simulation_work_exceeded`, `simulation_memory_exceeded`) and returns a receipt of
the work done. Only an empty or unordered schedule is `looks_not_ordered`.

A Thompson rule may carry a minimum exploration probability
(`AssignmentRule.thompson_beta_floored`), and a stopping rule may stop for
futility (`ProcedureStopping.posterior_futility`): when every other arm has
posterior probability below a threshold of exceeding the control (arm 0) by a
margin, deciding for the control. The participant-level family, with
several binary responses per participant, historical priors and
decisions at completed-participant batch boundaries, is declared with
`ParticipantProcedure` and described in
[inference from adaptive experiments](../adaptive-inference.md).

### Paradigms

| Paradigm | What is computed | Admitted when |
| --- | --- | --- |
| Bayesian posterior | Posterior-mean martingale gap under the prior predictive; a nonzero gap witnesses a failed premise | Loss of outcomes does not depend on their values |
| Decision-theoretic stopping | Stopping time, decision and regret of the declared rule under a hypothesized truth | Posterior expected-loss or futility rule, loss independent of values |
| Frequentist fixed sample | Sampling law of the design as declared | Fixed assignment, no data-dependent stopping |
| Frequentist group sequential | Law and stagewise-ordered p-value | Fixed randomization with declared boundaries |
| Adaptive weighting | Estimator law by whole-procedure simulation; plug-in variance is a diagnostic | Loss independent of values |
| Anytime-valid | Time-uniform bounds at the stopping time and the exact probability the terminal bound excludes the truth | One context, loss independent of values |
| Whole-procedure inversion | Confidence set by inverting the stagewise-ordered test | Declared contrast grid |

The exact law is of the declared procedure under a hypothesized truth, nothing
more: it is not an interval and not a calibration.

### What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Procedure element missing | `adaptive_contract_incomplete` (for example `adaptive_inference.stopping_rule_missing`) | Declare it |
| Posterior from value-dependent loss | `cell_not_licensed` (`adaptive_inference.posterior_premise_outcome_dependent_delay`) | Model the loss mechanism or restrict to ignorable loss |
| Fixed-sample or group-sequential on an adaptive design | `cell_not_licensed` (`adaptive_inference.fixed_sample_design_adaptive`, `group_sequential_assignment_adaptive`) | Use the adaptive-weighting or anytime-valid paradigm |
| Rule that does not match the paradigm | `cell_not_licensed` (`adaptive_inference.rule_mismatch`) | Declare the matching stopping rule |
| Anytime-valid with several contexts | `cell_not_licensed` (an `adaptive_inference.*` premise failure) | Run per context |
| Any interval or coverage claim | not licensed | Treat as derived and unmeasured |
