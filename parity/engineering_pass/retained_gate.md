# Retained-route gate regression repair

Reviewer handoff record. [Integrated verification](validation.md) records the executed results; references below to pending root validation describe the handoff stage.

Owner: foundations reviewer; scope limited to `scripts/gate_retained_routes.py` and `scripts/test_gate_retained_routes.py`. No Rust/manifest edits, no Cargo commands.

The failing negative selftest reported `dispatch entry no longer requires the lowered program`. Investigation showed a **gate false positive in its baseline**, rather than an ineffective mutation: `execute/dispatch.rs::execute_on` gained an early functional-validation refusal before its genuine `require_lowered` / `enter_lowered_dispatch` statements. The gate searched only the first 900 masked characters after the signature. The valid guard fell outside that arbitrary window, so baseline already reported it missing; deleting it produced no *new* finding. The selftest correctly required a new finding, exposing this gate defect.

Repair: `_dispatch_entry_body` finds the exact entry's brace-balanced body in Rust-masked source (shared lexer masks comments/literals), scans to its matching closing brace without any source-length cap, and blanks nested blocks. Guard tokens must therefore belong to top-level code of that entry. A neighboring helper or unexecuted nested closure cannot satisfy the entry's own requirement. Existing required function markers, lowered requirement, dispatch-scope requirement, refusal registration, reentry/bypass checks and measurement/route records are unchanged.

Regression evidence:

- Original negative mutation remains unchanged: deleting the actual execute_on lowered/scoping statements must introduce a new finding.
- Added negative mutation deletes real guard but appends an unrelated guard helper: function-boundary enforcement must still refuse.
- Added negative mutation moves guard tokens into an unexecuted nested closure: nested code must not satisfy entry obligation.
- Added positive fixture inserts 100 harmless top-level statements before the actual guard (well beyond 900 characters): findings must exactly equal baseline, with no fabricated guard failure.

Verification command: `python3 scripts/test_gate_retained_routes.py`; captured output `/private/tmp/engineering-retained-gate.log`. Outcome pending at report creation. Root retains integrated check/commit responsibility.

## Production-baseline follow-through (no differential-only closure)

The first repaired selftest caught all 31 negative mutations but reported **75 preexisting production findings**. That is not gate closure. Actual findings were captured separately in `/private/tmp/engineering-retained-baseline.log`: 71 `closed.*` execution-only matrix rows had no gate mapping, plus four real registered cells without a family-owned sealer (`CounterfactualClass`, `PathSpecificCpdag`, `PathSpecificMixed`, `PathSpecificPag`). The balanced guard repair itself introduced none of those findings.

Reviewed every affected coordinate through:

- `support/axis.rs` (public matrix target spelling; including distinction between static `ConditionalEffect` and temporal conditional pulse/sustained, and TemporalAdmg's projection/template meanings).
- `execute/execute_helpers_route.rs::classify_route` (query + data modality -> execution route).
- `execute/route_input.rs` (route + graph -> extracted family cell).
- `lowering/registry.rs` (actual registered lowering family).
- `prepared/sealers/*.rs` (real serving operation, excluding estimator/fresh generic sealers).
- `builder/finish.rs::lag_zero_average_effect` (AverageEffect on a temporal DAG/class becomes explicit lag-0 pulse, retaining the matrix label).

Added a finite `CLOSED_ROUTE_CELLS` review table for those exact target/graph coordinate families, not a catch-all `closed.*` acceptance. A closed id must agree exactly with its recorded target and graph. Unknown targets/graphs remain errors. For TemporalAdmg coordinates spanning both contemporaneous table and lagged-series shapes, the manifest check now validates **both** actual cells (derivatives: MixedAdmgResponse + TemporalResponseAdmg; attribution: GcmClass + GcmTemporalClass; MediationEffect: PathSpecificMixed + TemporalMediationAdmg). Static counterfactual/path-specific/nested/conditional/interference on TemporalAdmg retain their contemporaneous-projection cells; merely seeing the graph prefix never incorrectly routes them to a temporal estimator. Every mapped registered cell must have a family-owned serving operation; program-only sealers remain excluded. Global computation checks still inspect each operation's own module/calls.

Added negative selftests for mismatched closed id/metadata and a new unreviewed closed target. The selftest now requires **zero repository baseline findings before mutations**. A future baseline false positive cannot hide another missed negative case behind differential testing. Diagnostic wording was updated to describe the actual current program-only gap, rather than claiming a removed OrdinaryDispatch enum variant still exists.

Mapping the rows revealed real family-wrapper debts that had previously been hidden by missing mappings: twelve anomaly/change-attribution coordinates whose ClassAttribution payload was CheckedRouteOperation, plus five temporal-ADMG contrast/mediation coordinates whose TemporalAdmgEffect payload was CheckedRouteOperation. Captured in `/private/tmp/engineering-retained-baseline-after-map.log`. Reported these to root and facade owner; root explicitly assigned facade owner concrete family-owned operations, in addition to the four cross-world cells. No blanket wrapper exemption or generic-sealer counting was introduced to clear the gate.

Final zero-baseline + negative-selftest verification is pending facade implementation and root's integrated validation. This review does not treat the earlier differential selftest pass as completion.

## Final-source validation and strengthened secondary-cell ownership

Facade owner implemented dedicated StaticCrossWorld, ClassAttribution and TemporalAdmg operations, each freezing scientific query/graph/lowered identity and invoking real family executors. A pure-Python actual repository scan now returns **Findings: 0** (`/private/tmp/engineering-retained-final-baseline.log`). The strengthened selftest initially passed all **33** injected negatives at **zero** repository baseline, plus the long-valid-entry positive.

One final audit improvement prevents an ambiguous TemporalAdmg coordinate's *secondary* cell from silently reverting to CheckedRouteOperation while its primary cell stays native. Per-cell manifest ownership now inspects sealed payload types and rejects program-only wrappers for every mapped cell. The preexisting explicit WRAPPER_FAMILY_EXECUTORS mechanism remains available only under its existing separate executor/own-guard/computation verification; that map is empty, and this pass adds no exemptions. Added a 34th negative mutation replacing the temporal mixed family's concrete payload with CheckedRouteOperation. Final 34-case selftest is running; final result will be appended. This changes no Rust source and no scientific bytes.
