# Decision and claim identities under new domains

## What changed

The identities and digests that bind decision, external-claim and composition artifacts are derived with the canonical length-delimited writer under new domain tags instead of formatted text. The container layouts and the version constants are unchanged; the derived values are not.

| Artifact | Changed values | Class |
| --- | --- | --- |
| `decision_contract_v1` | contract `identity` | requires regeneration |
| `decision_result_v1` | `contract_identity`, `source_digest`, every lineage digest | requires regeneration |
| `external_response_claim_v1` | lineage digests, `provider_fingerprint` | requires regeneration |
| `composition_bundle_v1` | node identities, edge digests, bundle identity | requires regeneration |
| `inverse_functional_query_v1` | whole-artifact identity (now framed by the canonical writer) | requires regeneration |
| `admissible_decision_contract_v1`, `robust_decision_result_v1` | bound contract identity (now framed by the canonical writer) | requires regeneration |
| `joint_distribution_v1` (decision sources, joint laws) | none | byte-compatible |

## Who is affected

Anyone holding a decision contract, decision result, external claim or composition bundle exported by the previous release. A consumer that retains the identity it computes today refuses them; a consumer that retained the earlier identity passes the binding check and is refused at the lineage or node check. Nothing is silently accepted.

Joint-law and decision-source artifacts load unchanged. The inverse-query artifact also gained an optional `forward_origin` field that is omitted when absent, so its layout is otherwise unchanged.

## How to regenerate

Re-declare the object from its retained inputs and export it again:

1. Contract: rebuild the `DecisionContract` (or load its JSON declaration; a stored `identity` field is ignored) and call `DecisionContractArtifact::new(..).to_bytes(..)`.
2. Result: load the unchanged source distribution, evaluate the contract against it and export with `DecisionResultArtifact::new(result, &source).to_bytes(..)`.
3. External claim: bind the provider response against the checked contract again and export with `ExternalClaimArtifact::from_bound_claim(..).to_bytes(..)`.
4. Bundle: assemble it again from the regenerated nodes and retain the new bundle identity.

Retain the new identities out of band; the previous ones identify different values.
