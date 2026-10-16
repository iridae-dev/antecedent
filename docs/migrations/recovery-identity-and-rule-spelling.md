# Recovery derivations and rule-version spellings

## What changed

1. The recovery derivation record builds its m-graph signature, derivation identity and scientific identity from canonical length-delimited digests instead of formatted text. The rule version text is unchanged by this part.
2. Four rule and ranking versions are spelled by behavior and the earlier spellings are no longer accepted by any consumer:

| Earlier spelling | Current spelling |
| --- | --- |
| `x10.recovery.v1` | `binary_observation_recovery.v1` |
| `b2.recovery_chain.v1` | `ordered_response_recovery_chain.v1` |
| `x9.rules.v1` | `mixed_source_proof_rules.v1` |
| `x6.ranking.v1` | `study_plan_ranking.v1` |

## Class

Requires regeneration; the stored wire remains inspectable. Affected artifact families: observation recovery point, sampled recovery, ordered-response recovery chain, mixed-source point and study plan. A consumer re-decides the derivation and compares it with the stored record, so any record carrying an earlier signature, identity or spelling is refused as a derivation, premises or proof mismatch.

Coverage records keyed by the earlier derivation identity are preserved. They no longer match a current derivation and the whole-method calibration stays unmeasured until it is measured again.

## Who is affected

Anyone holding recovery, mixed-source or study-plan artifacts exported by an earlier release, and anyone who keyed their own records by the earlier derivation identity.

## How to regenerate

Run the same recovery, mixed-source or study-planning request with the current producer (same graph, catalog, data and seed) and export it again. The statistics are the current producer's; only the identities and the digests that bind them differ. Do not edit a stored spelling and reseal it: the consumer re-decides and refuses.
