Reviewer handoff record. [Integrated verification](validation.md) records executed results; pending root language below describes the handoff stage.

# Bounded bridge ownership cleanup

Owner: methods_review agent; root owns serialized formatter/Clippy/tests and commits. Source frozen after these three-file edits; no Cargo, rustfmt, tests or commits run by reviewer.

crates/antecedent-contract/src/bridge/claim.rs: extracted private source_point_value. It owns source admission as one coherent boundary: same scientific coordinate, claimed units consistent with that coordinate, Point claim kind, finite present value. derive_bridged_claim_propagating still checks contract licensing first, invokes that source-admission boundary second, then applies mapping, propagates uncertainty, builds identities/evidence/claim, records provenance and receipt in their original order. Refusal constructors and arguments are unchanged. Numerical formulas, canonical writers, encoding/version words, evidence order, source/contract/claim identity inputs, provenance construction and receipts are untouched. No new public API or abstraction.

crates/antecedent-contract/src/bridge/propagation.rs: quoted SplitMix64 in the stream documentation; no code change.

crates/antecedent-contract/src/bridge/refusal.rs: added #[must_use] to existing public pure draws_unusable, draws_outside_domain and chain_stage_not_affine constructors. This is an intentional caller diagnostic for discarding constructed refusal values, with no runtime behavior, encoded field, reason/detail, remedy or return-type change.

No blanket lint allowances or golden updates. Root should rerun strict Clippy and the existing bridge claim/uncertainty/refusal/identity suites in its serialized validation queue.
