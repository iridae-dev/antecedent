# Reviewed public API additions

Owner: integration. Fresh `scripts/gate_public_api.py` built rustdoc JSON for all
published library crates on the integrated source. Review found exactly 24 added
snapshot lines across five crates, with no removed or changed existing lines.
Only those five snapshots were deliberately updated from that fresh output; the
complete follow-up comparison passes. No stale rustdoc directory was used as the
initial verification.

| Owner | Additions and concrete purpose |
| --- | --- |
| `antecedent` | `analysis::RetainedOperation::StaticCrossWorld` names the actual retained class/mixed operation; `analysis::discovered_transport::transport_discovered_structure_with_context` preserves the discovery caller’s context through subsequent structural transport and lets external retained-outcome callers fund/cancel that stage without repeating discovery. |
| `antecedent-graph` | `TemporalPagCompletionSampler::with_budget` supplies the existing default-window construction with actual caller admission, cancellation and search limits. |
| `antecedent-identify` | Configured `GeneralizedAdjustmentIdentifier::identify_{cpdag,pag}_{response_general,derivative_pairs}` methods prevent default-config resets in response dispatch; `TransportIdentifier::identify_with_context` governs actual per-member standardizer search; `discovered_transport::transport_across_{class,pag}_with_context` governs completion and member work. |
| `antecedent-design` | `evsi::EvaluatedEvsi` privately owns the exact original prepared laws through publication, with `report`/`into_report`; `evsi::evaluate_evsi_with_context` retains caller policy while keeping request RNG identity; `design_ranking_artifact::seal_with_context` seals that same evaluation without global recall or provider re-preparation. Generated auto-traits/Debug appear with this new type; no manual unwind-safety claim is introduced. |
| `antecedent-validate` | Non-exhaustive `ValidationError::Resource { message }` preserves a physical resource refusal through validation instead of reporting scientific/discovery failure. |

Existing standalone/default functions and request DTOs retain their signatures.
The enum additions are to non-exhaustive enums whose external exhaustive matching
already requires a wildcard. These additive APIs need no source migration or
artifact version bump. New consuming helpers are private unless a real cross-crate
caller requires the capability. This pass does not claim the separate 3.0 Python
practitioner/API pass or final release acceptance complete.

The existing Python discovery-and-transport binding calls the contextual stage
through `discover_and_transport`; its facade registry now records that actual
route. Existing Python signatures, reply tuple fields, structured refusal JSON,
artifact layouts and canonical writers are unchanged. Intentional refusal and
physical-accounting corrections, plus their measurement limitations, are recorded
in the [main review](README.md), [context audit](context_ownership.md) and
[validation results](validation.md).
