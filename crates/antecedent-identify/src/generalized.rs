//! Generalized adjustment for CPDAG/PAG classes.
//!
//! Identification over a PAG streams bounded completions and aggregates an
//! [`IdentificationEnvelope`], preserving unidentified mass.
//!
//! # Search completeness
//!
//! Per MAG completion, adjustment sets are searched among subsets of
//! `An({T,Y}) \ (Desc(T) ∪ {T,Y})` in the completion ADMG, tested for m-separation of
//! `T` and `Y` in `G_{\underline{T}}` (outgoing edges from `T` removed). Enumeration is
//! by increasing set size and stops at the first valid set (minimal-first). Completions
//! that are not MAGs, or MAGs with no qualifying set in this candidate family, contribute
//! unidentified mass. A completion whose candidate family exceeds `max_candidates` is
//! *also* folded into unidentified mass (enumeration was never attempted, so it can't be
//! proven identified or not) but is additionally counted in
//! [`IdentificationEnvelope::truncated_completions`] and tagged with
//! [`CAPPED_COMPLETION_DIAGNOSTIC_CODE`], so callers can tell "the search was cut short"
//! apart from "the search completed and proved non-identifiability".
//!
//! This is **generalized adjustment**, not the full ID/IDC algorithm (see roadmap P5.3).
//! Sets outside the ancestor candidate family are not searched.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    Value, VariableId,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{
    Admg, BitSet, CompletionSampler, DSeparationWorkspace, DenseNodeId, Endpoint, Pag,
};

use crate::envelope::{
    GraphFeature, GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass,
};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand,
};

/// Diagnostic code attached to a per-completion [`IdentificationResult`] when adjustment-set
/// enumeration was capped by `max_candidates` before it could search — as opposed to
/// searching exhaustively and finding no valid set. Both cases surface as
/// [`IdentificationStatus::NotIdentified`] and fold into
/// [`IdentificationEnvelope::unidentified_weight`] identically (unidentified mass is
/// preserved either way), but [`IdentificationEnvelope::truncated_completions`] counts only
/// this diagnostic so a caller can tell "the search was cut short" apart from "the search
/// completed and proved non-identifiability".
pub const CAPPED_COMPLETION_DIAGNOSTIC_CODE: &str =
    "identify.generalized_adjustment.completion_capped";

/// Config for PAG generalized-adjustment envelopes.
#[derive(Clone, Debug)]
pub struct GeneralizedAdjustmentConfig {
    /// Max completions to stream (hard bound; never retain unbounded).
    pub max_completions: usize,
    /// Uniform weight per streamed completion.
    pub per_completion_weight: f64,
    /// Max candidate covariates to enumerate (bitmask width).
    pub max_candidates: usize,
}

impl Default for GeneralizedAdjustmentConfig {
    fn default() -> Self {
        Self { max_completions: 32, per_completion_weight: 1.0, max_candidates: 16 }
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
        let t_d = pag_var_to_dense(pag, t)?;
        let y_d = pag_var_to_dense(pag, y)?;
        let (active, control) = match (&query.active, &query.control) {
            (
                antecedent_core::Intervention::Set { value: active, .. },
                antecedent_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "generalized adjustment ATE requires Set interventions",
                });
            }
        };

        let mut sampler = CompletionSampler::new(pag.clone(), self.config.max_completions)
            .map_err(IdentificationError::from)?;
        let mut cases = Vec::new();
        let w = ProbabilityMass(self.config.per_completion_weight);
        for completion in sampler.by_ref() {
            let result = identify_on_mag_completion(
                &completion.graph,
                t,
                y,
                t_d,
                y_d,
                active.clone(),
                control.clone(),
                self.config.max_candidates,
            )?;
            cases.push(GraphIdentificationCase { graph: completion.graph, result, weight: w });
        }
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        envelope.push_features(pag_circle_features(pag));
        let validation = sampler.validation_report();
        envelope.push_features([GraphFeature {
            kind: Arc::from("pag_completion_validation"),
            detail: Arc::from(format!(
                "exhaustively audited {} endpoint assignment(s): represented={}, \
                 rejected_non_ancestral={}, rejected_nonmaximal={}, \
                 rejected_local_incompatible={}, ambiguous_global_class={}, \
                 equivalence_audit_skipped={}",
                validation.assignments_examined,
                validation.represented_completions,
                validation.rejected_non_ancestral,
                validation.rejected_nonmaximal,
                validation.rejected_local_incompatible,
                validation.ambiguous_global_class,
                validation.equivalence_audit_skipped,
            )),
        }]);
        // The global class audit is exponential in node count and is skipped on larger
        // graphs. Local validity still holds for every case, but "identified in every
        // member of the equivalence class" is exactly the claim the skipped audit would
        // have supported, so it is downgraded here rather than asserted.
        if sampler.class_audit_incomplete() {
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_equivalence_audit_skipped"),
                detail: Arc::from(
                    "the global m-separation equivalence audit was too expensive for this graph; \
                     completions are locally valid but not certified to form one Markov class",
                ),
            }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        // `NonparametricallyIdentified` on this path asserts identification for *every* member
        // of the equivalence class. When the sampler hit its retention cap, identification
        // ran only on a deterministic low-mask prefix, so the assertion is unearned — a
        // verified but unretained completion may well be unidentified. Downgrade rather than
        // overclaim.
        if sampler.hit_cap() {
            envelope.push_features([GraphFeature {
                kind: Arc::from("completion_enumeration_capped"),
                detail: Arc::from(format!(
                    "retained {} of {} verified MAG completion(s) under max_completions={}; \
                     identification is established only over the deterministic retained subset",
                    envelope.cases.len(),
                    validation.represented_completions,
                    self.config.max_completions
                )),
            }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        Ok(envelope)
    }
}

fn pag_circle_features(pag: &Pag) -> Vec<GraphFeature> {
    let review = antecedent_graph::PagReview::from_pag(pag.clone(), "generalized.adjustment");
    if review.pending_circles.is_empty() {
        return Vec::new();
    }
    vec![GraphFeature {
        kind: Arc::from("pag_circle_marks"),
        detail: Arc::from(format!(
            "{} edge(s) with circle endpoints in source PAG",
            review.pending_circles.len()
        )),
    }]
}

fn pag_var_to_dense(pag: &Pag, id: VariableId) -> Result<DenseNodeId, IdentificationError> {
    for (i, node) in pag.nodes().iter().enumerate() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == id {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    Err(IdentificationError::UnknownVariable { id })
}

fn identify_on_mag_completion(
    mag: &Pag,
    t: VariableId,
    y: VariableId,
    t_d: DenseNodeId,
    y_d: DenseNodeId,
    active: Value,
    control: Value,
    max_candidates: usize,
) -> Result<IdentificationResult, IdentificationError> {
    let query = CausalQuery::AverageEffect(AverageEffectQuery::new(
        t,
        y,
        Arc::from([]),
        antecedent_core::Intervention::set(t, control.clone()),
        antecedent_core::Intervention::set(t, active.clone()),
        antecedent_core::TargetPopulation::AllObserved,
    ));
    let Some(admg) = mag_to_admg(mag) else {
        return Ok(not_identified(query, "completion is not a MAG (undirected marks remain)"));
    };

    let mutilated = mutilate_outgoing(&admg, t_d);
    let candidates = adjustment_candidates(&admg, t_d, y_d);
    if candidates.len() > max_candidates {
        return Ok(capped_completion_result(query, candidates.len(), max_candidates));
    }

    let mut ws = DSeparationWorkspace::default();
    let mut examined = 0u64;
    let mut found: Option<Vec<DenseNodeId>> = None;
    'sizes: for size in 0..=candidates.len() {
        let mut early = false;
        let mut enum_err: Option<IdentificationError> = None;
        crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
            if enum_err.is_some() {
                return true;
            }
            examined += 1;
            match mutilated.is_m_separated(t_d, y_d, z, &mut ws) {
                Ok(true) => {
                    found = Some(z.to_vec());
                    early = true;
                    true
                }
                Ok(false) => false,
                Err(e) => {
                    enum_err = Some(IdentificationError::from(e));
                    true
                }
            }
        });
        if let Some(e) = enum_err {
            return Err(e);
        }
        if early {
            break 'sizes;
        }
    }

    let Some(z_dense) = found else {
        return Ok(IdentificationResult::not_identified(
            query,
            {
                let mut d = DerivationTrace::default();
                d.push(
                    "generalized.adjustment",
                    "no generalized adjustment set among ancestor candidates on completion",
                );
                d
            },
            AssumptionSet::default(),
            IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
        ));
    };

    let z_vars: Arc<[VariableId]> =
        z_dense.iter().map(|&d| mag_dense_to_var(mag, d)).collect::<Result<Vec<_>, _>>()?.into();
    let mut arena = CausalExprArena::new();
    let functional = arena.backdoor_ate(t, y, &z_vars, active, control);
    let label =
        if z_vars.is_empty() { "generalized.adjustment.empty" } else { "generalized.adjustment" };
    let estimand = IdentifiedEstimand::backdoor(label, Arc::clone(&z_vars), functional);
    Ok(IdentificationResult::identified(
        query,
        vec![estimand],
        arena,
        {
            let mut d = DerivationTrace::default();
            d.push(
                "generalized.adjustment",
                format!(
                    "Z (size {}) m-separates T from Y in G_underbar{{T}} among ancestor candidates",
                    z_vars.len()
                ),
            );
            d
        },
        AssumptionSet::default(),
        IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
    ))
}

fn mag_dense_to_var(mag: &Pag, id: DenseNodeId) -> Result<VariableId, IdentificationError> {
    match mag.nodes().get(id.as_usize()) {
        Some(antecedent_graph::NodeRef::Static(v)) => Ok(*v),
        _ => Err(IdentificationError::UnknownVariable { id: VariableId::from_raw(id.raw()) }),
    }
}

/// Candidates = An({T,Y}) \ (Desc(T) ∪ {T,Y}) in the ADMG.
fn adjustment_candidates(admg: &Admg, t: DenseNodeId, y: DenseNodeId) -> Vec<DenseNodeId> {
    let an = directed_closure(admg, &[t, y], true);
    let desc_t = directed_closure(admg, &[t], false);
    let mut out = Vec::new();
    for i in 0..admg.node_count() {
        let id = DenseNodeId::from_raw(i as u32);
        if id == t || id == y {
            continue;
        }
        if !an.contains(id) || desc_t.contains(id) {
            continue;
        }
        out.push(id);
    }
    out
}

fn directed_closure(admg: &Admg, seeds: &[DenseNodeId], ancestors: bool) -> BitSet {
    let mut out = BitSet::with_len(admg.node_count());
    let mut stack: Vec<DenseNodeId> = seeds.to_vec();
    for &s in seeds {
        out.insert(s);
    }
    while let Some(u) = stack.pop() {
        let nbrs = if ancestors { admg.parents(u) } else { admg.children(u) };
        for &v in nbrs {
            if !out.contains(v) {
                out.insert(v);
                stack.push(v);
            }
        }
    }
    out
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

/// Not-identified result for a completion whose candidate set exceeded `max_candidates`
/// before enumeration could even start. Distinguished from a genuinely-searched, genuinely-
/// blocked completion via [`CAPPED_COMPLETION_DIAGNOSTIC_CODE`] (see its doc comment); the
/// derivation text is kept human-readable and consistent with the other not-identified
/// branches in this function.
fn capped_completion_result(
    query: CausalQuery,
    n_candidates: usize,
    max_candidates: usize,
) -> IdentificationResult {
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "generalized.adjustment",
        "generalized adjustment candidate set exceeds enumeration limit",
    );
    let diagnostic = Diagnostic::new(
        CAPPED_COMPLETION_DIAGNOSTIC_CODE,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Warning,
        format!(
            "generalized adjustment candidate set ({n_candidates}) exceeds max_candidates \
             ({max_candidates}); enumeration was not attempted on this completion, so \
             identifiability could not be determined (not the same as a proven non-\
             identifiable completion)"
        ),
    );
    IdentificationResult::from_parts(
        IdentificationStatus::NotIdentified,
        query,
        Vec::new(),
        CausalExprArena::new(),
        derivation,
        AssumptionSet::default(),
        vec![diagnostic],
        IdentificationPerformanceRecord::default(),
        None,
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
                continue;
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
    use antecedent_graph::Pag;

    #[test]
    fn envelope_preserves_mass_on_mixed_pag() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier {
            config: GeneralizedAdjustmentConfig {
                max_completions: 8,
                per_completion_weight: 1.0,
                max_candidates: 16,
            },
        };
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        let total = env.identified_weight.0 + env.unidentified_weight.0;
        assert!(total > 0.0);
        assert!((total - env.cases.len() as f64).abs() < 1e-9);
    }

    #[test]
    fn directed_edge_identifies_with_empty_z() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/generalized_adjustment/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["adjustment_set"].as_array().unwrap().len(), 0);
        let mut pag = Pag::with_variables(2);
        pag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(env.unidentified_weight.0 == 0.0);
    }

    #[test]
    fn confounder_identifies_with_nonempty_z() {
        // Z → T, Z → Y, T → Y  (backdoor {Z}).
        let mut pag = Pag::with_variables(3);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        pag.insert_directed(z, t).unwrap();
        pag.insert_directed(z, y).unwrap();
        pag.insert_directed(t, y).unwrap();
        let id = GeneralizedAdjustmentIdentifier::new();
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let env = id.identify_pag_envelope(&pag, &q).unwrap();
        assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(env.cases[0].result.status, IdentificationStatus::NonparametricallyIdentified);
        let z_set = &env.cases[0].result.estimands[0].adjustment_set;
        assert!(z_set.iter().any(|v| v.raw() == 0), "expected Z in adjustment, got {z_set:?}");
    }

    #[test]
    fn capped_enumeration_is_distinguishable_from_genuine_nonidentification() {
        // Case A: Z → T, Z → Y, T → Y, but max_candidates=0 forces the single-candidate set
        // {Z} over the cap before enumeration can even start — the search was truncated, not
        // exhausted.
        let mut capped_pag = Pag::with_variables(3);
        let z = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        capped_pag.insert_directed(z, t).unwrap();
        capped_pag.insert_directed(z, y).unwrap();
        capped_pag.insert_directed(t, y).unwrap();
        let capped_id = GeneralizedAdjustmentIdentifier {
            config: GeneralizedAdjustmentConfig {
                max_completions: 8,
                per_completion_weight: 1.0,
                max_candidates: 0,
            },
        };
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
        let capped_env = capped_id.identify_pag_envelope(&capped_pag, &q).unwrap();
        assert!(capped_env.unidentified_weight.0 > 0.0);
        assert_eq!(
            capped_env.truncated_completions,
            capped_env.cases.len(),
            "every case in this fixture should be capped"
        );
        assert!(
            capped_env.cases[0]
                .result
                .diagnostics
                .iter()
                .any(|d| d.code.as_ref() == CAPPED_COMPLETION_DIAGNOSTIC_CODE),
            "capped case should carry the capped-completion diagnostic"
        );

        // Case B: bare T ↔ Y (unmeasured confounding, no other variables). The search is
        // never truncated (candidates.len() == 0 never exceeds the default max_candidates),
        // but no adjustment set can ever separate T from Y, so this is a genuine
        // non-identification — proved, not merely "could not tell".
        let mut confounded_pag = Pag::with_variables(2);
        confounded_pag
            .insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            .unwrap();
        let genuine_id = GeneralizedAdjustmentIdentifier::new();
        let q2 = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let genuine_env = genuine_id.identify_pag_envelope(&confounded_pag, &q2).unwrap();
        assert!(genuine_env.unidentified_weight.0 > 0.0);
        assert_eq!(
            genuine_env.truncated_completions, 0,
            "genuinely-blocked completions must not count as truncated"
        );

        // Both envelopes report positive unidentified mass (neither is silently dropped), but
        // only the capped one is flagged as truncated -- proving the two are distinguishable
        // rather than both being folded indistinguishably into `unidentified_weight`.
        assert!(capped_env.truncated_completions > genuine_env.truncated_completions);
    }
}
