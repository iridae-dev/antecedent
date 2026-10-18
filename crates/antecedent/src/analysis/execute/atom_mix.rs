//! Shared graph-posterior atom mixing, identification lookup, and refuter reduction.
//!
//! The scalar posterior routes (class, temporal class, ADMG, temporal-class
//! mediation) accumulate the same weighted mixture. Route-specific bounds,
//! standard errors, and refusal text stay on [`AtomEvalView`] and the call.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{HashMap, HashSet};

use super::{
    CausalError, CausalPosterior, Diagnostic, EffectEstimate, GraphIdentFlag, IdentificationResult,
    IdentificationStatus, IdentifiedEstimand, IdentifiedKeyWeights, OverlapPolicy,
    QueryRefutationPlan, StructuralAggregationPolicy, WeightedGraphSamples,
    mix_atom_refutation_reports, mix_weighted_analytic_se, mixed_static_influence, mixture_se,
    resolve_structural_aggregation, scalar_identified_set,
};
use crate::result::{StructuralResponseAtom, StructuralResponseMixture};

/// One evaluated posterior atom, as the shared mixer reads it.
pub(super) trait AtomEvalView {
    /// Posterior graph key.
    fn key(&self) -> u64;
    /// Identification status recorded for this atom.
    fn status(&self) -> IdentificationStatus;
    /// Estimand that can enter the structural aggregate, when this atom has one.
    fn estimand(&self) -> Option<&IdentifiedEstimand>;
    /// Finite scalar contribution. `None` is unevaluable when the weight is positive.
    fn scalar(&self) -> Option<f64>;
    /// Completion-conditional posterior, when this atom retained one.
    fn posterior(&self) -> Option<&CausalPosterior>;
    /// Assumptions of the scalar estimate, when that estimate exists.
    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet>;
    /// Identified-set bounds published by this atom.
    fn identified_bounds(&self) -> Option<(f64, f64)>;
    /// This atom is only partially identified, or its completions disagree.
    fn partial(&self) -> bool;
    /// Analytic SE to mix. `None` leaves the aggregate SE unset (`NaN`).
    fn analytic_se(&self) -> Option<f64>;
    /// Row-aligned influence, when the route mixes a joint SE.
    fn influence(&self) -> Option<&[f64]>;
    /// The standard errors this atom's own fit reported, which the aggregate adopts when the
    /// atom covers the entire target. `None` leaves the aggregate's as mixed.
    fn reported_uncertainty(&self) -> Option<antecedent_estimate::EstimateUncertainty> {
        None
    }
}

/// Identification proof for one posterior atom.
///
/// The mixer builds a key-to-index map from these proofs and clones the
/// matching [`IdentificationResult`]. A missing key still resolves to the
/// first proof, which is the previous `find` / `[0]` fallback.
pub(super) trait AtomProof {
    /// Key shared with [`AtomEvalView::key`].
    fn proof_key(&self) -> u64;
    /// Identification cloned into the mixture when this atom is primary.
    fn identification(&self) -> &IdentificationResult;
}

impl AtomProof for crate::analysis::prepared::CachedClassPosteriorAtomIdentification {
    fn proof_key(&self) -> u64 {
        self.key
    }

    fn identification(&self) -> &IdentificationResult {
        &self.identification
    }
}

impl AtomProof for crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification {
    fn proof_key(&self) -> u64 {
        self.key
    }

    fn identification(&self) -> &IdentificationResult {
        &self.identification
    }
}

impl AtomProof for crate::analysis::prepared::CachedTemporalClassPosteriorAtomIdentification {
    fn proof_key(&self) -> u64 {
        self.key
    }

    fn identification(&self) -> &IdentificationResult {
        &self.identification
    }
}

/// First-seen key to proof index.
pub(super) struct AtomProofIndex<'a, A> {
    atoms: &'a [A],
    by_key: HashMap<u64, usize>,
}

impl<'a, A: AtomProof> AtomProofIndex<'a, A> {
    /// Index `atoms` in order. A repeated key keeps the earliest proof.
    #[must_use]
    pub(super) fn build(atoms: &'a [A]) -> Self {
        let mut by_key = HashMap::with_capacity(atoms.len());
        for (index, atom) in atoms.iter().enumerate() {
            by_key.entry(atom.proof_key()).or_insert(index);
        }
        Self { atoms, by_key }
    }

    /// Identification for `key`, or the first proof when `key` is absent.
    ///
    /// # Panics
    ///
    /// Panics when `atoms` is empty. That is the previous `[0]` fallback, not
    /// a new refusal.
    #[must_use]
    pub(super) fn identification_for(&self, key: u64) -> IdentificationResult {
        let index = self.by_key.get(&key).copied().unwrap_or(0);
        self.atoms[index].identification().clone()
    }
}

/// How an atom contributes to the mixture's identified-set bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AtomBoundRule {
    /// Finite scalars only: the atom's identified set, otherwise its point.
    SetOrPoint,
    /// Finite scalars only: the point.
    Point,
    /// Every atom: its identified set, including atoms with no finite scalar.
    SetAlways,
}

/// Inputs that are not on the per-atom view.
pub(super) struct AtomMixInput<'a, A> {
    /// Frozen posterior samples. Weights and unidentified atoms are read from here.
    pub graphs: &'a WeightedGraphSamples,
    /// Proofs aligned with the route's identified atoms.
    pub proofs: &'a [A],
    /// Unidentified mass already computed by the caller. Stored, not recomputed.
    pub unidentified_mass: f64,
    /// True when no class atom truncated its completion search.
    pub full_mass_scope: bool,
    /// Truncated completions the mixture reports.
    pub truncated_atoms: usize,
    /// Bound rule for this route.
    pub bounds: AtomBoundRule,
    /// Class, temporal class, and ADMG fall back to `identification.estimands`.
    /// Mediation does not.
    pub estimand_fallback: bool,
}

// Manual impls: the input only borrows `A`, so it is `Copy` whatever `A` is.
impl<A> Clone for AtomMixInput<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A> Copy for AtomMixInput<'_, A> {}

/// Scalar mixture shared by the posterior-atom routes.
pub(super) struct MixedAtomEvals {
    /// Aggregation policy over the contributing estimands.
    pub policy: StructuralAggregationPolicy,
    /// Identification of the primary contributing atom.
    pub identification: IdentificationResult,
    /// Primary estimand, or the identification's first estimand when fallback is on.
    pub estimand: IdentifiedEstimand,
    /// Mixed scalar. The SE is `NaN` when the route does not publish one.
    pub estimate: EffectEstimate,
    /// Component posterior, only when it covers the whole target.
    pub posterior: Option<CausalPosterior>,
    /// Structural mixture, including unidentified atoms appended from `graphs`.
    pub mixture: StructuralResponseMixture,
    /// True when the policy licensed a weighted mean and some mass was mixable.
    pub mixable_scalar: bool,
    /// Running lower bound before the finite check. `INFINITY` when unset.
    pub bound_lo: f64,
    /// Running upper bound before the finite check. `NEG_INFINITY` when unset.
    pub bound_hi: f64,
}

/// Mix evaluated posterior atoms.
///
/// Unidentified samples absent from `evals` are appended in sample order. A
/// repeated unidentified key keeps the first sample's weight.
///
/// # Errors
///
/// Returns the caller's refusal when no atom supplies an identification, and
/// the caller's missing-estimand refusal when the primary identification has
/// no estimand. Those strings are the existing route refusals.
///
/// # Panics
///
/// Panics when a contributing atom needs a proof and `proofs` is empty. That
/// is the previous `class_atoms[0]` / `atoms[0]` fallback.
#[allow(
    clippy::too_many_lines,
    reason = "one accumulation replaces four copied mixers; the route differences are the bound rule and the refusal text"
)]
pub(super) fn mix_atom_evals<E, A, M, N>(
    evals: &[E],
    input: AtomMixInput<'_, A>,
    missing_values: M,
    missing_estimand: N,
) -> Result<MixedAtomEvals, CausalError>
where
    E: AtomEvalView,
    A: AtomProof,
    M: FnOnce(usize, f64) -> String,
    N: FnOnce() -> String,
{
    let proofs = AtomProofIndex::build(input.proofs);
    let mut atoms = Vec::new();
    let mut identified_weight = 0.0;
    let mut unevaluable_weight = 0.0;
    let mut mixable = 0.0;
    let mut weighted = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut contributing = Vec::new();
    let mut any_partial = false;
    let mut primary_identification = None;
    let mut primary_estimand = None;
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut atom_posteriors = Vec::new();
    let mut outer_se_items = Vec::new();
    let mut outer_atom_ifs: Vec<&[f64]> = Vec::new();
    let mut outer_atom_weights = Vec::new();
    let mut sole_uncertainty = None;

    let key_weights = IdentifiedKeyWeights::new(input.graphs);
    for eval in evals {
        let w = key_weights.get(eval.key());
        if input.bounds == AtomBoundRule::SetAlways {
            if let Some((lower, upper)) = eval.identified_bounds() {
                lo = lo.min(lower);
                hi = hi.max(upper);
            }
        }
        let value = eval.scalar();
        if let Some(v) = value {
            identified_weight += w;
            match input.bounds {
                AtomBoundRule::SetOrPoint => {
                    if let Some((lower, upper)) = eval.identified_bounds() {
                        lo = lo.min(lower);
                        hi = hi.max(upper);
                    } else {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                AtomBoundRule::Point => {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                AtomBoundRule::SetAlways => {}
            }
            if let Some(estimand) = eval.estimand() {
                contributing.push(estimand.clone());
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(proofs.identification_for(eval.key()));
                    if let Some(recorded) = eval.assumptions() {
                        assumptions = recorded.clone();
                    }
                }
            }
            any_partial |= eval.partial();
            if let Some(posterior) = eval.posterior() {
                atom_posteriors.push((w, posterior.clone()));
            }
            if let Some(se) = eval.analytic_se() {
                outer_se_items.push((w, se));
            }
            sole_uncertainty = eval.reported_uncertainty();
            if let Some(influence) = eval.influence() {
                outer_atom_ifs.push(influence);
                outer_atom_weights.push(w);
            }
            mixable += w;
            weighted += w * v;
        } else if w > 0.0 {
            unevaluable_weight += w;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: eval.key(),
            weight: w,
            status: eval.status(),
            value: value.map(antecedent_core::ResponseValue::Scalar),
            posterior: eval.posterior().cloned(),
            response: None,
        });
    }

    append_unidentified_atoms(&mut atoms, input.graphs);

    let total = input.graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let unevaluable_mass = if total > 0.0 { unevaluable_weight / total } else { 0.0 };
    let refs: Vec<&IdentifiedEstimand> = contributing.iter().collect();
    let policy = if refs.is_empty() && any_partial {
        StructuralAggregationPolicy::IdentifiedSetEnvelope
    } else {
        resolve_structural_aggregation(&refs, any_partial)
    };
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let n_contributing = outer_se_items.len();
    let joint = mixable_scalar && n_contributing > 1 && outer_atom_ifs.len() == n_contributing;
    let mixed_influence = if joint && mixable_scalar {
        mixed_static_influence(&outer_atom_ifs, &outer_atom_weights)
    } else {
        None
    };
    let se = if mixable_scalar {
        if joint {
            mixed_influence.as_deref().map_or(f64::NAN, mixture_se)
        } else {
            mix_weighted_analytic_se(outer_se_items)
        }
    } else {
        f64::NAN
    };
    let conditional_on_identified =
        mixable_scalar.then_some(antecedent_core::ResponseValue::Scalar(ate));
    let mixture = StructuralResponseMixture {
        weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
        atoms,
        identified_mass,
        unidentified_mass: input.unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass: 0.0,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified,
        full_mass_scope: input.full_mass_scope,
        truncated_atoms: input.truncated_atoms,
    };
    let identification = primary_identification
        .ok_or_else(|| CausalError::Compile { message: missing_values(evals.len(), mixable) })?;
    let estimand = if input.estimand_fallback {
        primary_estimand.or_else(|| identification.estimands.first().cloned())
    } else {
        primary_estimand
    };
    let estimand = estimand.ok_or_else(|| CausalError::Compile { message: missing_estimand() })?;
    let mut estimate = EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride);
    if joint {
        if let Some(influence) = mixed_influence {
            estimate.scores.influence = Some(influence);
        }
    }
    // A component posterior is not the posterior of a weighted aggregate.
    // Retain it, and the standard errors of its fit, at the top level only when it covers the
    // entire target.
    let covers_target = mixable_scalar
        && contributing.len() == 1
        && input.unidentified_mass == 0.0
        && unevaluable_mass == 0.0
        && mixture.full_mass_scope;
    if let Some(uncertainty) = sole_uncertainty.filter(|_| covers_target) {
        estimate.uncertainty = uncertainty;
    }
    let posterior = if covers_target && atom_posteriors.len() == 1 {
        atom_posteriors.into_iter().next().map(|(_, posterior)| posterior)
    } else {
        None
    };
    Ok(MixedAtomEvals {
        policy,
        identification,
        estimand,
        estimate,
        posterior,
        mixture,
        mixable_scalar,
        bound_lo: lo,
        bound_hi: hi,
    })
}

/// Append unidentified posterior samples that are not already in `atoms`.
///
/// Sample order is kept. A key that is already present, including one appended
/// earlier in this walk, is skipped, so the stored weight is the first sample's
/// weight rather than a sum.
pub(super) fn append_unidentified_atoms(
    atoms: &mut Vec<StructuralResponseAtom>,
    graphs: &WeightedGraphSamples,
) {
    let mut present: HashSet<u64> = atoms.iter().map(|atom| atom.graph_key).collect();
    for ((key, weight), flag) in
        graphs.graph_keys.iter().zip(graphs.weights.iter()).zip(graphs.identified.iter())
    {
        if *flag != GraphIdentFlag::Unidentified {
            continue;
        }
        if !present.insert(*key) {
            continue;
        }
        atoms.push(StructuralResponseAtom {
            graph_key: *key,
            weight: *weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        });
    }
}

/// Reports produced for one posterior atom (or one plugin level inside it).
pub(super) struct AtomRefuterBatch {
    /// Posterior weight of this batch. Plugin levels pass `atom_weight * case_weight`.
    pub weight: f64,
    /// Refuter reports for this batch.
    pub reports: Vec<antecedent_validate::RefutationReport>,
    /// Diagnostics from the refuter run, already filtered by the caller.
    pub notes: Vec<Diagnostic>,
}

/// How positive-weight refuter reports are reduced when a scalar mix is licensed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ScalarRefuterMix {
    /// [`mix_atom_refutation_reports`], including partial-coverage warnings.
    Effect,
    /// [`QueryRefutationPlan::mix_weighted`], with no coverage warning.
    Mediation,
}

/// Reduce per-atom refuter reports and attach the route's summary diagnostic.
///
/// `coverage_before_summary` is the DBN mediation order. The posterior routes
/// push the summary first, then coverage warnings.
#[must_use]
pub(super) fn finish_atom_refutations(
    per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)>,
    mut diagnostics: Vec<Diagnostic>,
    mix_scalar: bool,
    scalar_mix: ScalarRefuterMix,
    summary: Option<Diagnostic>,
    coverage_before_summary: bool,
) -> (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>) {
    let reports = if mix_scalar {
        match scalar_mix {
            ScalarRefuterMix::Effect => {
                let (mixed, coverage) = mix_atom_refutation_reports(&per_atom);
                if coverage_before_summary {
                    diagnostics.extend(coverage);
                    if let Some(summary) = summary {
                        diagnostics.push(summary);
                    }
                } else {
                    if let Some(summary) = summary {
                        diagnostics.push(summary);
                    }
                    diagnostics.extend(coverage);
                }
                mixed
            }
            ScalarRefuterMix::Mediation => {
                if let Some(summary) = summary {
                    diagnostics.push(summary);
                }
                QueryRefutationPlan::mix_weighted(per_atom)
            }
        }
    } else {
        if let Some(summary) = summary {
            diagnostics.push(summary);
        }
        per_atom.into_iter().flat_map(|(_, reports)| reports).collect()
    };
    (reports, diagnostics)
}

/// Run one refuter batch per evaluated atom, then [`finish_atom_refutations`].
///
/// `listed` chooses the atoms named in the summary. That set includes atoms the
/// runner skips for zero weight, matching the previous key lists.
///
/// # Errors
///
/// Forwards the first error from `run`.
pub(super) fn drive_posterior_refuters<E, L, K, R, S>(
    evals: &[E],
    mix_scalar: bool,
    scalar_mix: ScalarRefuterMix,
    listed: L,
    key_of: K,
    mut run: R,
    summary: S,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError>
where
    L: Fn(&E) -> bool,
    K: Fn(&E) -> u64,
    R: FnMut(&E) -> Result<Vec<AtomRefuterBatch>, CausalError>,
    S: FnOnce(&str, bool) -> Diagnostic,
{
    let inner_keys = evals
        .iter()
        .filter(|eval| listed(eval))
        .map(|eval| format!("{:x}", key_of(eval)))
        .collect::<Vec<_>>()
        .join(",");
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    for eval in evals {
        for batch in run(eval)? {
            diagnostics.extend(batch.notes);
            per_atom.push((batch.weight, batch.reports));
        }
    }
    Ok(finish_atom_refutations(
        per_atom,
        diagnostics,
        mix_scalar,
        scalar_mix,
        Some(summary(&inner_keys, mix_scalar)),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_table::{
        DEFAULT_IDENTIFIER_ID, EstimatorId, identify_static, select_estimand,
    };
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_graph::{Dag, DenseNodeId};

    struct Eval {
        key: u64,
        estimand: IdentifiedEstimand,
        influence: Vec<f64>,
    }

    impl AtomEvalView for Eval {
        fn key(&self) -> u64 {
            self.key
        }
        fn status(&self) -> IdentificationStatus {
            IdentificationStatus::NonparametricallyIdentified
        }
        fn estimand(&self) -> Option<&IdentifiedEstimand> {
            Some(&self.estimand)
        }
        fn scalar(&self) -> Option<f64> {
            Some(0.5)
        }
        fn posterior(&self) -> Option<&CausalPosterior> {
            None
        }
        fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
            None
        }
        fn identified_bounds(&self) -> Option<(f64, f64)> {
            None
        }
        fn partial(&self) -> bool {
            false
        }
        fn analytic_se(&self) -> Option<f64> {
            Some(0.1)
        }
        fn influence(&self) -> Option<&[f64]> {
            Some(&self.influence)
        }
    }

    struct Proof(u64, IdentificationResult);

    impl AtomProof for Proof {
        fn proof_key(&self) -> u64 {
            self.0
        }
        fn identification(&self) -> &IdentificationResult {
            &self.1
        }
    }

    /// The mixer's joint SE and influence are the helpers' results on the same columns,
    /// computed from one mixture instead of two.
    #[test]
    fn joint_mixture_matches_the_standalone_helpers() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identification = identify_static(DEFAULT_IDENTIFIER_ID, &dag, &query).unwrap();
        let estimand = select_estimand(&identification, EstimatorId::LinearAdjustmentAte).unwrap();
        let keys = [3_u64, 5, 9];
        let weights = vec![0.5, 0.3, 0.2];
        let graphs = WeightedGraphSamples::new(
            weights.clone(),
            vec![GraphIdentFlag::Identified; 3],
            keys.to_vec(),
        )
        .unwrap();
        let evals: Vec<Eval> = keys
            .iter()
            .map(|&key| Eval {
                key,
                estimand: estimand.clone(),
                influence: (0..40)
                    .map(|r| ((r * 7 + usize::try_from(key).unwrap()) % 11) as f64 - 5.0)
                    .collect(),
            })
            .collect();
        let proofs: Vec<Proof> =
            keys.iter().map(|&key| Proof(key, identification.clone())).collect();
        let mixed = mix_atom_evals(
            &evals,
            AtomMixInput {
                graphs: &graphs,
                proofs: &proofs,
                unidentified_mass: 0.0,
                full_mass_scope: true,
                truncated_atoms: 0,
                bounds: AtomBoundRule::Point,
                estimand_fallback: true,
            },
            |_, _| String::new(),
            String::new,
        )
        .unwrap();
        let columns: Vec<&[f64]> = evals.iter().map(|eval| eval.influence.as_slice()).collect();
        assert_eq!(
            mixed.estimate.se_analytic().to_bits(),
            super::super::mix_static_envelope_se(&columns, &weights).to_bits()
        );
        assert_eq!(
            mixed.estimate.scores.influence.as_deref(),
            super::super::mixed_static_influence(&columns, &weights).as_deref()
        );
    }

    /// Timing harness for the shared atom mixer over long influence columns
    /// (`cargo test -p antecedent --lib atom_mix::tests::mixer_scales -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing harness"]
    fn mixer_scales_with_atoms_and_rows() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identification = identify_static(DEFAULT_IDENTIFIER_ID, &dag, &query).unwrap();
        let estimand = select_estimand(&identification, EstimatorId::LinearAdjustmentAte).unwrap();
        for (atoms, rows) in [(8_usize, 100_000_usize), (32, 100_000), (32, 400_000)] {
            let weights = vec![1.0 / atoms as f64; atoms];
            let keys: Vec<u64> = (0..atoms as u64).collect();
            let graphs = WeightedGraphSamples::new(
                weights,
                vec![GraphIdentFlag::Identified; atoms],
                keys.clone(),
            )
            .unwrap();
            let evals: Vec<Eval> = keys
                .iter()
                .map(|&key| Eval {
                    key,
                    estimand: estimand.clone(),
                    influence: (0..rows)
                        .map(|r| ((r as u64 * 31 + key) % 97) as f64 - 48.0)
                        .collect(),
                })
                .collect();
            let proofs: Vec<Proof> =
                keys.iter().map(|&key| Proof(key, identification.clone())).collect();
            let started = std::time::Instant::now();
            for _ in 0..3 {
                let mixed = mix_atom_evals(
                    &evals,
                    AtomMixInput {
                        graphs: &graphs,
                        proofs: &proofs,
                        unidentified_mass: 0.0,
                        full_mass_scope: true,
                        truncated_atoms: 0,
                        bounds: AtomBoundRule::Point,
                        estimand_fallback: true,
                    },
                    |_, _| String::new(),
                    String::new,
                )
                .unwrap();
                std::hint::black_box(&mixed);
            }
            eprintln!(
                "atoms={atoms:>3} rows={rows:>7} mix={:.1} ms",
                started.elapsed().as_secs_f64() * 1e3 / 3.0
            );
        }
    }
}
