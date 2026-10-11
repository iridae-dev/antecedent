# Artifacts sealed by an earlier release

These bytes were produced by the shipped 2.3.0 code (tag `v2.3.0`), not by the
current producers. They carry the original spelling of four version strings that
were later renamed to describe their behavior:

| Original spelling     | Current spelling                     |
| --------------------- | ------------------------------------ |
| `x10.recovery.v1`     | `binary_observation_recovery.v1`     |
| `x9.rules.v1`         | `mixed_source_proof_rules.v1`        |
| `x6.ranking.v1`       | `study_plan_ranking.v1`              |
| `b2.recovery_chain.v1`| `ordered_response_recovery_chain.v1` |

Consumers accept either spelling of each, verify the digests of the bytes as they
were carried, and re-decide the derivation under the same rule. The tests that
read these files assert that they are consumed, that the carried identity is kept,
and that a tampered or other-version artifact is refused.

| File | Producer (at `v2.3.0`) |
| --- | --- |
| `observation_recovery_point.cbor` | `exported(&model())` of `observation_recovery_lifecycle` |
| `mixed_source_point.cbor` | `result.export(&prepared)` of `mixed_source_lifecycle` over `evidence()`, `do(X = true)` |
| `study_plan_mz.json` | `mz_plan().to_artifact()` of `study_planning_lifecycle` |
| `study_plan_mixed.json` | the surrogate-trial plan of `a_declared_margin_repairs_the_named_missing_joint` |
| `recovery_chain_recovered.cbor` | `recovered_wire(0, true).export()` of the io `recovery_chain_artifact` tests |
| `sampled_recovery.cbor` | `wire_of(&fixture(seeded_rows(4000, 17), &loose(99))).export()` of the io `sampled_recovery_artifact` tests |

They are immutable evidence of what shipped. Do not regenerate them with current
code; regenerate only from a checkout of the `v2.3.0` tag.
