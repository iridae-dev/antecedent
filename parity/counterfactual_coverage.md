# Counterfactual coverage for the 2.1 development branch

| Query | Graph and evidence | Execution | Uncertainty or refusal |
| --- | --- | --- | --- |
| Existing unit-level counterfactual | Licensed fixed Markovian DAG and compatible SCM evidence | Shared-exogenous abduction, action, prediction | Existing support registry governs inference and validation |
| Natural direct effect `Y₁,M₀ − Y₀,M₀` | Explicit three-node Markovian `X→M`, `X→Y`, `M→Y` DAG; tabular linear Gaussian SCM; Frequentist; no refutation suite | Shared abduced exogenous draw across both worlds; known linear SCM target truth; Rust and Python staged execution and artifact replay | Point-only; no sampling interval reported |
| Incompatible nested worlds | Counterfactual query outside the named compatible mediation contract | Refused | Stable `cross_world_not_identified` reason |
| Nested effect with accepted, uncertain, latent-confounded, or temporal graph | Graph/evidence outside the fixed explicit DAG contract | Refused | No counterfactual-ID or transported claim |
| Nested effect with Bayesian inference or cheap/full validation | No joint posterior SCM or licensed refutation route | Refused | No posterior or calibration claim |
| Other path-specific, temporal, or transported counterfactuals | Contracts not established by this slice | Refused | No inferred cross-world composition |

The named cell has deterministic evidence against its known SCM target truth, frozen shared-world
semantics, contract/body inconsistency rejection, fitted-snapshot tampering, and
artifact replay/consumption in
`crates/antecedent/tests/nested_counterfactual_route_evidence.rs`. The support
registry remains the coordinate-level license; a missing implementation is not
a zero-probability counterfactual event.
