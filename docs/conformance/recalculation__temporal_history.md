# Checked whole-history recalculation

**Suite path:** `conformance/recalculation/temporal_history`

The point oracle enumerates 640 independent exogenous configurations of the stated binary structural model. Each repeated unit contributes two identical histories. The target differs only in the distribution of root baseline S0; all other declared DAG mechanisms are invariant. Integrating the structural probabilities gives the response formula and numeric pins in `expected.json`, independently of identification, fitted counts, and the recalculation adapter.

The native adapter identifies the actual observational source joint `(S0,Y)|do(A1,A2)` with general ID, checks positive source S0 support, conditions that executing joint on S0, and standardizes with the caller's target S0 law. No observation is relabeled as an experiment. The separate empirical artifact reexecutes this source identification and numerical program from the embedded linked-unit histories.

The additive candidate fixture checks dependence preservation. Its shared unit disturbance cancels from the paired effect on every whole-unit draw, giving exactly 5, while it remains in the response 7.165 and its replicate variation. Twenty replicates test deterministic execution and replay only. This twenty-replicate candidate fixture remains unmeasured. The normal measured temporal route uses its separately attested B500 protocol and exact complete-history sample/design scope; this fixture supplies no interval license.

Executing consumers: `crates/antecedent/tests/recalc_temporal.rs` and the `temporal_history_fit` unit tests in `crates/antecedent-estimate/src/temporal_history_fit.rs`. The source theorem is the explicit bounded DAG/root initial-state shift composition; it does not inherit a broader transport or interval guarantee.

## Expected summary

Top-level keys: `case, control_first_unit_y_to_zero, coordinate_order, identical_histories_per_unit, paired_additive_candidate, rule, scope, source_units, structural_model, target_p1_0_25, target_p1_0_8` (11 fields).
