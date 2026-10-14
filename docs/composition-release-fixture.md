# Release-boundary composition fixture

One scenario, small enough to check by hand, that runs the composition routes
together through their public Rust and Python surfaces: an uncertain measured
quantity enters a causal model, several components contribute mechanisms or
evidence, shared evidence and dependence are preserved, a downstream decision
depends on the resulting claim, new observations revalidate that claim, and a
consumer that holds nothing but bytes recomputes the decision.

Everything is point-only. No interval, posterior decision or ranking guarantee
is produced, and each request for one is refused with `cell_not_licensed`.
Calibration is unmeasured.

Files: the declared inputs, hand-derived values and expected refusals are
`conformance/composition/release_boundary/expected.json` (read by both test
suites); the fresh-process consumer is `consumer.py` beside it; the tests are
`crates/antecedent/tests/composition_release_fixture.rs` and
`python/tests/test_composition_release_fixture.py`; the exit-gate story is
`story_7_*` in `composition_exit_gate.rs` and `test_composition_exit_gate.py`.

## The scenario

| Step | Declared | Hand value |
|---|---|---|
| Measured exposure | sensitivity 0.9, specificity 0.8, 410 of 1000 recorded exposed | prevalence `(0.41 + 0.8 - 1) / 0.7 = 0.3` |
| Study evidence | A: 0.10, B: 0.20, variance 0.0004 each, disjoint rows | pool 0.15, variance 0.0002 |
| Composed model | `X ~ Bern(0.3)`, `Y | X ~ Bern(0.10 + 0.15 X)` | `P(Y=1|do(X=1)) = 0.25`, `P(Y=1) = 0.145`, excess `0.045` |
| Decision | `act = 1000 * excess - 30`, `hold = 0` | act 15, hold 0, `act` uniquely optimal |
| Revision | B grows to 600 rows, summary 0.07 variance 0.0002 | pool `600/7500 = 0.08`, excess 0.024, act -6, `hold` |

The excess risk is also `0.3 * 0.15`: the composed model, the aligned
product of the two inputs' enumerated joint law and the consumer's closed form
agree.

## What each requirement is checked by

| Requirement | Public routes | Test |
|---|---|---|
| A measured quantity enters a causal model | `measurement_model` (binary misclassification), `mechanism_composition` (finite tables) | `a_measured_exposure_enters_the_causal_model_through_the_declared_confusion_model` |
| Several components contribute | the corrected prevalence, the pooled effect, a declared baseline | the same |
| Shared evidence and dependence | `evidence_ledger` overlap, declared covariance, `joint_law_propagation` | `shared_evidence_is_counted_once_and_dependence_is_declared_or_refused`, `unpaired_marginals_are_never_paired_by_index_and_an_enumerated_joint_law_is` |
| A decision depends on the claim | `decision` contract on a caller-declared mean source | `the_decision_reads_the_licensed_mean_and_matches_the_hand_computed_utilities` |
| New observations revalidate | ledger `update`, `composition_verification` | `new_observations_revalidate_the_claim_graph_and_the_decision_is_reevaluated` |
| No double counting | ledger lineage and row overlap | `duplicated_evidence_cannot_be_counted_twice` |
| Unsupported conversions fail closed | `measurement_model`, `interface_contract`, `decision` | `unsupported_semantic_conversions_fail_closed` |
| Independent consumer and mutations | every artifact consumer, closed-form oracle | `a_fresh_consumer_recomputes_the_decision_and_every_mutation_is_refused_as_expected` |
| Withdrawal and dependence gap refuse | ledger withdrawal, unknown overlap | `an_upstream_withdrawal_makes_the_downstream_decision_refuse_end_to_end`, `a_dependence_gap_makes_the_downstream_decision_refuse_end_to_end` |

## The refusals

| Situation | Code | Detail |
|---|---|---|
| Studies share rows and are declared independent | `evidence_double_counted` | `evidence_ledger.shared_observations` |
| A duplicate re-enters through lineage | `evidence_double_counted` | `evidence_ledger.reentry_duplicate` |
| A contribution with undeclared observations | `evidence_overlap_unknown` | `evidence_ledger.overlap_unknown` |
| Shared evidence declared independent in the joint-law rule | `evidence_double_counted` | `joint_law_propagation.shared_evidence_declared_independent` |
| Unknown dependence in the joint-law rule | `sampling_dependence_unknown` | `joint_law_propagation.dependence_unknown` |
| Unpaired marginals paired by index | `joint_law_required` | `joint_law_propagation.index_pairing_refused` |
| The measured column named as the construct | `proxy_offered_as_construct` | `measurement_model.indicator_as_construct` |
| A value used without its measurement model | `interface_contract_assumption_mismatch` | `interface_contract.measurement_assumption_mismatch` |
| A mean where a law is required | `interface_contract_operation_unavailable` | `interface_contract.operation_unavailable` |
| Declared support not inside the consumer's | `interface_contract_quantity_mismatch` | `interface_contract.support_mismatch` |
| Mean source in other units than the contract's input | `quantity_semantics_mismatch` | `decision_evaluation.quantity_not_found` |
| A criterion that needs a law, on a mean source | `decision_contract_unsatisfied` | `decision_evaluation.mean_source_insufficient` |
| Ledger bytes of another history under the retained identity | `route_not_supported` | `evidence_ledger.identity_mismatch` |
| Claim graph verified at another ledger state | `invalid_argument` | `composition_verification.ledger_mismatch` |
| Decision while its claim awaits recomputation | `claim_revalidation_required` | `composition_verification.recompute_required` |
| Decision whose claim rests on a withdrawn pool | `claim_license_withdrawn` | `composition_verification.input_retired` |
| Interval or posterior of any stage | `cell_not_licensed` | the owning route's detail |

## Gaps in the public surface

The fixture reports these instead of reaching into internals.

* A mean source is not bound to a ledger result or a claim-graph node. Nothing
  in `Contract.evaluate` or `evaluate_contract_on_means` stops a decision from
  being evaluated on a stale claim; the fixture asks the claim graph
  (`require_valid("decision")`) first and records the evidence state in the
  mean source's `snapshot_id` and the plan in `causal_contract_id`.
* A decision on a mean source is not exportable or replayable
  (`decision_evaluation.mean_source_not_replayable`), so the consumer
  re-evaluates the contract from the consumed numbers rather than replaying a
  stored decision.
* The corrected prevalence reaches the mechanism table by a number the caller
  writes down. The table is therefore a declared mechanism (admitted
  explicitly), not a fitted one. No route hands a measurement estimate to a
  mechanism declaration with its provenance.
* The ledger's overlap reaches the joint-law rule as evidence identities the
  caller chooses. No route derives joint-law inputs or the covariance from a
  ledger result.
* The Python `verify_ledger` needs a live ledger session; a consumed ledger
  artifact has no `account_observations`, so a Python consumer compares state
  digests instead of calling it. Rust accepts the consumed ledger.
* The consumer composes the excess risk as `pi * (q1 - q0)` from the corrected
  prevalence and two interventional artifacts; the observational law is not
  exported by the query kinds used here.

## Running

    cargo test -p antecedent --test composition_release_fixture
    pytest python/tests/test_composition_release_fixture.py
    bash scripts/gate_composition.sh
    bash scripts/gate_composition_exit.sh

## Closed

No interval for the corrected prevalence, the pooled effect, the composed
claim or the decision is produced. The calibration design for any such
interval is the whole method (measurement correction, pooling, composition,
decision) at the composed coordinate; none is measured here. A request is
refused with `cell_not_licensed` and the remedy of the owning route.
