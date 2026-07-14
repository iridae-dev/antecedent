//! Generalized adjustment for CPDAG/PAG classes (DESIGN.md §10.2).
//!
//! Identification over a PAG streams bounded completions and aggregates an
//! [`IdentificationEnvelope`], preserving unidentified mass.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, Value, VariableId};
use causal_expr::CausalExprArena;
use causal_graph::{Admg, CompletionSampler, DSeparationWorkspace, DenseNodeId, Endpoint, Pag};

use crate::envelope::{GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Config for PAG generalized-adjustment envelopes.
#[derive(Clone, Debug)]
pub struct GeneralizedAdjustmentConfig {
    /// Max completions to stream (hard bound; never retain unbounded).
    pub max_completions: usize,
    /// Uniform weight per streamed completion.
    pub per_completion_weight: f64,
}

impl Default for GeneralizedAdjustmentConfig {
    fn default() -> Self {
        Self { max_completions: 32, per_completion_weight: 1.0 }
    }
}

/// Class-aware identifier for PAGs via completion envelopes.
#[derive(Clone, Debug, Default)]
pub struct GeneralizedAdjustmentIdentifier {
    /// Config.
    pub config: GeneralizedAdjustmentConfig,
}

impl GeneralizedAdjustmentIdentifier {
    /// Default identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Identify an average effect over a PAG by streaming completions.
    ///
    /// # Errors
    ///
    /// Query type unsupported or graph errors.
    pub fn identify_pag_envelope(
        &self,
        pag: &Pag,
        query: &AverageEffectQuery,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        let t = query.treatment;
        let y = query.outcome;
        let t_d = DenseNodeId::from_raw(t.raw());
        let y_d = DenseNodeId::from_raw(y.raw());
        if t_d.as_usize() >= pag.node_count() || y_d.as_usize() >= pag.node_count() {
            return Err(IdentificationError::msg("treatment/outcome not in PAG"));
        }

        let sampler = CompletionSampler::new(pag.clone(), self.config.max_completions)
            .map_err(IdentificationError::from)?;
        let mut cases = Vec::new();
        let w = ProbabilityMass(self.config.per_completion_weight);
        for completion in sampler {
            let result = identify_on_mag_completion(&completion.graph, t, y, t_d, y_d)?;
            cases.push(GraphIdentificationCase { graph: completion.graph, result, weight: w });
        }
        Ok(IdentificationEnvelope::from_cases(cases))
    }
}

fn identify_on_mag_completion(
    mag: &Pag,
    t: VariableId,
    y: VariableId,
    t_d: DenseNodeId,
    y_d: DenseNodeId,
) -> Result<IdentificationResult, IdentificationError> {
    let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(t, y));
    let Some(admg) = mag_to_admg(mag) else {
        // Residual undirected marks → not a MAG; count as unidentified case.
        return Ok(not_identified(query, "completion is not a MAG (undirected marks remain)"));
    };

    // Backdoor-style: empty Z m-separates T from Y in G_{\underline{T}} (remove outgoing from T).
    let mutilated = mutilate_outgoing(&admg, t_d);
    let mut ws = DSeparationWorkspace::default();
    let sep =
        mutilated.is_m_separated(t_d, y_d, &[], &mut ws).map_err(IdentificationError::from)?;
    if sep {
        let mut arena = CausalExprArena::new();
        let functional = arena.backdoor_ate(t, y, &[], Value::f64(1.0), Value::f64(0.0));
        let estimand =
            IdentifiedEstimand::backdoor("generalized.adjustment.empty", Arc::from([]), functional);
        return Ok(IdentificationResult::identified(
            query,
            vec![estimand],
            arena,
            {
                let mut d = DerivationTrace::default();
                d.push("generalized.adjustment", "empty Z m-separates in G_underbar{T}");
                d
            },
            AssumptionSet::default(),
            IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 1 },
        ));
    }
    Ok(not_identified(query, "no empty-set generalized adjustment on completion"))
}

fn not_identified(query: CausalQuery, detail: &str) -> IdentificationResult {
    let mut derivation = DerivationTrace::default();
    derivation.push("generalized.adjustment", detail);
    IdentificationResult::not_identified(
        query,
        derivation,
        AssumptionSet::default(),
        IdentificationPerformanceRecord::default(),
    )
}

fn mag_to_admg(mag: &Pag) -> Option<Admg> {
    let n = mag.node_count() as u32;
    let mut admg = Admg::with_variables(n);
    for i in 0..mag.node_count() {
        let a = DenseNodeId::from_raw(i as u32);
        for (b, at_a, at_b) in mag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(at_a, Endpoint::Circle) || matches!(at_b, Endpoint::Circle) {
                return None;
            }
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    admg.insert_directed(a, b).ok()?;
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    admg.insert_directed(b, a).ok()?;
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    admg.insert_bidirected(a, b).ok()?;
                }
                (Endpoint::Tail, Endpoint::Tail) => {
                    // Undirected — not a MAG.
                    return None;
                }
                _ => return None,
            }
        }
    }
    Some(admg)
}

fn mutilate_outgoing(admg: &Admg, t: DenseNodeId) -> Admg {
    let n = admg.node_count() as u32;
    let mut out = Admg::with_variables(n);
    for i in 0..admg.node_count() {
        let u = DenseNodeId::from_raw(i as u32);
        for &v in admg.children(u) {
            if u == t {
                continue; // remove outgoing from T
            }
            let _ = out.insert_directed(u, v);
        }
        for &v in admg.bidirected_neighbors(u) {
            if v.raw() > u.raw() {
                let _ = out.insert_bidirected(u, v);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use causal_graph::Pag;

    #[test]
    fn envelope_preserves_mass_on_mixed_pag() {
        // T o→ Y : some completions identify, some may not depending on marks.
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier {
            config: GeneralizedAdjustmentConfig { max_completions: 8, per_completion_weight: 1.0 },
        };
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        let total = env.identified_weight.0 + env.unidentified_weight.0;
        assert!(total > 0.0);
        // Mass is accounted (no silent drop).
        assert!((total - env.cases.len() as f64).abs() < 1e-9);
    }

    #[test]
    fn directed_edge_identifies_with_empty_z() {
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(env.unidentified_weight.0 == 0.0);
    }
}
