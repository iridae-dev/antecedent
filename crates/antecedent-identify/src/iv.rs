//! Instrumental-variable identification for DAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{BitSet, DSeparationWorkspace, Dag, DenseNodeId, GraphWorkspace};

use crate::backdoor::{PreparedIdentificationGraph, dense_to_var, remove_outgoing, var_to_dense};
use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, EstimandClaim, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand,
};

/// Assumption id: the restriction under which the Wald ratio is the average effect
/// (constant linear effect), or else the complier local effect (monotonicity).
pub const IV_EFFECT_RESTRICTION_ID: &str = "iv.constant_linear_effect_or_monotonicity";
/// Assumption id: non-zero first stage for one instrument.
pub const IV_RELEVANCE_ID: &str = "iv.relevance";

/// Configuration for instrument search.
#[derive(Clone, Debug)]
pub struct InstrumentSearchConfig {
    /// Maximum number of instruments to return.
    pub max_results: usize,
}

impl Default for InstrumentSearchConfig {
    fn default() -> Self {
        Self { max_results: 64 }
    }
}

/// Instrumental-variable identifier for static DAGs.
#[derive(Clone, Debug, Default)]
pub struct InstrumentalVariableIdentifier {
    /// Search configuration.
    pub config: InstrumentSearchConfig,
}

impl InstrumentalVariableIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph for IV identification.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::with_assumptions(graph.clone(), assumptions))
    }

    /// Identify an average-effect query via a valid instrument.
    ///
    /// # Errors
    ///
    /// Unsupported query or unknown variables.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "IV identification only supports AverageEffect",
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone(), workspace)
    }

    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let dag = prepared.dag();
        let t = var_to_dense(ate.treatment, dag)?;
        let y = var_to_dense(ate.outcome, dag)?;

        // Candidates: all nodes except T,Y, with parents of T checked first.
        let parents_of_t: Vec<DenseNodeId> =
            dag.parents(t).iter().copied().filter(|&p| p != y).collect();
        let mut candidates: Vec<DenseNodeId> = parents_of_t.clone();
        for i in 0..dag.node_count() {
            let v = DenseNodeId::from_raw(u32::try_from(i).expect("node id fits u32"));
            if v == t || v == y || candidates.contains(&v) {
                continue;
            }
            candidates.push(v);
        }

        let mut valid: Vec<DenseNodeId> = Vec::new();
        let mut examined = 0u64;

        for &z in &candidates {
            examined += 1;
            if is_valid_instrument(dag, z, t, y, &mut workspace.graph, &mut workspace.dsep)? {
                valid.push(z);
                if valid.len() >= self.config.max_results {
                    break;
                }
            }
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("iv"));
        for record in &prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "iv.criterion",
            "Z is not a descendant of T, is relevant to T given ∅, and is d-separated \
             from Y in G with T's out-edges cut; Wald identifies ATE under linearity \
             (or LATE under monotonicity)",
        );

        if valid.is_empty() {
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }

        // What the graph cannot certify: d-connection licenses a first stage but not a
        // non-zero one, and the Wald ratio is the average effect only under a constant
        // linear structural effect (with heterogeneous effects it is the complier local
        // average effect, and only under monotonicity).
        assumptions.push(wald_effect_restriction());
        let shared = assumptions.clone();

        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        let mut claims = Vec::with_capacity(valid.len());
        for &z in &valid {
            let z_var = dense_to_var(z, dag)?;
            // Each instrument's ratio relies on that instrument only.
            let own = instrument_assumptions(z_var);
            let mut claim_assumptions = shared.clone();
            claim_assumptions.extend_unique(&own);
            assumptions.extend_unique(&own);
            claims.push(EstimandClaim {
                status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
                required_assumptions: claim_assumptions,
            });
            let (active, control) = match (&ate.active, &ate.control) {
                (
                    antecedent_core::Intervention::Set { value: active, .. },
                    antecedent_core::Intervention::Set { value: control, .. },
                ) => (active.clone(), control.clone()),
                _ => {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: "IV ATE requires Set interventions",
                    });
                }
            };
            let functional = arena
                .iv_wald(ate.treatment, ate.outcome, &[z_var], &active, &control)
                .map_err(|_| IdentificationError::UnsupportedQuery {
                    message: "IV requires a single instrument distinct from the treatment",
                })?;
            estimands.push(IdentifiedEstimand::instrumental("iv", Arc::from([z_var]), functional));
            derivation.push("iv.instrument", format!("Z={}", z_var.raw()));
        }

        Ok(IdentificationResult::identified_under_parametric_restrictions(
            query,
            estimands,
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: u64::try_from(valid.len()).unwrap_or(u64::MAX),
            },
        )
        .with_estimand_claims(claims))
    }
}

fn iv_default(assumption: Assumption) -> AssumptionRecord {
    AssumptionRecord {
        assumption,
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("iv") },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
}

/// The restriction that decides which effect the Wald ratio is.
fn wald_effect_restriction() -> AssumptionRecord {
    iv_default(Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
        id: Arc::from(IV_EFFECT_RESTRICTION_ID),
        description: Arc::from(
            "the Wald ratio equals the average effect only under a constant linear structural \
             effect of the treatment on the outcome; with heterogeneous effects it is the \
             complier local average effect, and only under monotonicity (no defiers)",
        ),
    }))
}

/// Records one instrument's ratio relies on: its exclusion restriction and relevance.
fn instrument_assumptions(instrument: antecedent_core::VariableId) -> [AssumptionRecord; 2] {
    [
        iv_default(Assumption::ExclusionRestriction { instrument }),
        iv_default(Assumption::Custom {
            id: Arc::from(IV_RELEVANCE_ID),
            description: Arc::from(format!(
                "instrument {} has a non-zero first-stage association with the treatment (the \
                 graph shows a connecting path, not its strength)",
                instrument.raw()
            )),
        }),
    ]
}

/// Whether `z` is a valid instrument for `t` -> `y`.
fn is_valid_instrument(
    dag: &Dag,
    z: DenseNodeId,
    t: DenseNodeId,
    y: DenseNodeId,
    graph_ws: &mut GraphWorkspace,
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    if z == t || z == y {
        return Ok(false);
    }

    // Treatment descendants are not valid unadjusted instruments: cutting
    // T's outgoing edges would hide the Z–Y dependence through T.
    let mut desc = BitSet::with_len(dag.node_count());
    dag.descendants_of(&[t], &mut desc, graph_ws);
    if desc.contains(z) {
        return Ok(false);
    }

    // 1. Relevance: Z is not d-separated from T given ∅ (association, not
    // necessarily a directed path Z → … → T — e.g. Z ← C → T is relevant).
    let independent_of_t = dag.is_d_separated(z, t, &[], ws).map_err(IdentificationError::from)?;
    if independent_of_t {
        return Ok(false);
    }

    // 2. Exclusion + no Z-Y confounding: with T's outgoing edges cut (so the
    // legitimate T -> Y channel is removed), Z must be d-separated from Y
    // given ∅. Conditioning on T directly would instead open the T-collider
    // between Z and any T-Y confounder, so we mutilate rather than condition.
    let t_mutilated = remove_outgoing(dag, t)?;
    t_mutilated.is_d_separated(z, y, &[], ws).map_err(IdentificationError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use antecedent_core::{AverageEffectQuery, VariableId};

    #[test]
    fn confounded_treatment_with_valid_instrument() {
        // Z -> T -> Y, U -> T, U -> Y (U unmeasured confounder).
        let mut g = Dag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        assert!(res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(0)]));
        // The confounder U itself must never be reported as a valid instrument.
        assert!(!res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(3)]));
    }

    #[test]
    fn each_instrument_claims_its_own_exclusion_and_the_wald_restriction() {
        // Z1 -> T <- Z2, T -> Y, U -> T, U -> Y: two valid instruments, one estimand each.
        let mut g = Dag::with_variables(5);
        let z1 = DenseNodeId::from_raw(0);
        let z2 = DenseNodeId::from_raw(1);
        let t = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        let u = DenseNodeId::from_raw(4);
        g.insert_directed(z1, t).unwrap();
        g.insert_directed(z2, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        // U is a parent of T but fails exogeneity, so exactly Z1 and Z2 qualify.
        let instruments: Vec<u32> = res.estimands.iter().map(|e| e.instruments[0].raw()).collect();
        assert_eq!(instruments, vec![0, 1]);
        assert_eq!(res.estimand_claims.len(), 2);

        let exclusions = |set: &AssumptionSet| -> Vec<u32> {
            set.entries
                .iter()
                .filter_map(|r| match &r.assumption {
                    Assumption::ExclusionRestriction { instrument } => Some(instrument.raw()),
                    _ => None,
                })
                .collect()
        };
        for (index, instrument) in [0u32, 1].into_iter().enumerate() {
            let claim = res.narrowed_to(index).unwrap();
            assert_eq!(claim.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
            assert_eq!(exclusions(&claim.required_assumptions), vec![instrument]);
            assert!(claim.required_assumptions.entries.iter().any(|r| matches!(
                &r.assumption,
                Assumption::ParametricRestriction(p) if p.id.as_ref() == IV_EFFECT_RESTRICTION_ID
            )));
            let relevance: Vec<&str> = claim
                .required_assumptions
                .entries
                .iter()
                .filter_map(|r| match &r.assumption {
                    Assumption::Custom { id, description } if id.as_ref() == IV_RELEVANCE_ID => {
                        Some(description.as_ref())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(relevance.len(), 1);
            assert!(relevance[0].starts_with(&format!("instrument {instrument} ")));
        }
        // The listing covers both instruments.
        assert_eq!(exclusions(&res.required_assumptions), vec![0, 1]);
    }

    #[test]
    fn relevance_via_common_cause_not_directed_path() {
        // Z ← C → T → Y: Z does not reach T, but is associated with T and is a valid IV.
        let mut g = Dag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let c = DenseNodeId::from_raw(1);
        let t = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        g.insert_directed(c, z).unwrap();
        g.insert_directed(c, t).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::IdentifiedUnderParametricRestrictions);
        assert!(res.estimands.iter().any(|e| e.instruments.as_ref() == [VariableId::from_raw(0)]));
    }

    #[test]
    fn direct_edge_to_outcome_rejects_instrument() {
        // Z -> T -> Y, Z -> Y (direct edge violates exclusion), U -> T, U -> Y.
        let mut g = Dag::with_variables(4);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }

    #[test]
    fn treatment_descendant_rejects_instrument() {
        // U → T → Y, U → Y, T → Z: Z is a treatment descendant, not an IV.
        let mut g = Dag::with_variables(4);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(2);
        let z = DenseNodeId::from_raw(3);
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(t, z).unwrap();

        let id = InstrumentalVariableIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }
}
