//! Helpers shared by the exact, statistical, learned-trial and grid transport modalities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::{
    EvidenceCatalog, ExecutionContext, IdentityDomain, NodeRef, RegimeId, VariableId,
};
use antecedent_estimate::ReplicatePolicy;
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundTransportFunctional, ClassicalTransportDerivation, ClassicalTransportQuery, SidLimits,
};
use antecedent_io::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_proof::TransportProofWire,
};
use serde::Serialize;
use std::sync::Arc;

use super::exact::ExactFactorRequirement;

/// The facade's licensing floor for a nominal-0.95 percentile or posterior interval edge.
pub(super) const PERCENTILE_FLOOR: ReplicatePolicy =
    ReplicatePolicy::percentile_floor(crate::result::PERCENTILE_95_MIN_REPLICATES);

/// Largest departure from one of a probability vector's mass that still counts as normalised.
pub(super) const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Whether `probabilities` is a probability vector: every entry finite in `[0, 1]` and the
/// (Neumaier-compensated) mass within [`PROBABILITY_SUM_TOLERANCE`] of one.
pub(super) fn is_probability_vector(probabilities: &[f64]) -> bool {
    let (mut total, mut correction) = (0.0_f64, 0.0_f64);
    for &p in probabilities {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            return false;
        }
        let next = total + p;
        correction += if total.abs() >= p.abs() { (total - next) + p } else { (p - next) + total };
        total = next;
    }
    ((total + correction) - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE
}

/// Whether an exact distribution is well formed: aligned atoms and probabilities, atoms as wide
/// as the outcome list, and a probability vector.
pub(super) fn is_valid_exact_law(law: &antecedent_expr::ExactDistribution) -> bool {
    law.atoms.len() == law.probabilities.len()
        && law.atoms.iter().all(|atom| atom.len() == law.outcomes.len())
        && is_probability_vector(&law.probabilities)
}

/// Withhold a Bayesian transport posterior's intervals when its draws fall under
/// [`PERCENTILE_FLOOR`], keeping the point law and recording why.
pub(super) fn apply_posterior_draw_floor(
    estimate: &mut antecedent_estimate::BayesianStatisticalTransportEstimate,
) {
    if estimate.interval_reason.as_deref()
        != Some(antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED)
    {
        return;
    }
    if let Err(reason) =
        PERCENTILE_FLOOR.decide(estimate.draws_requested, estimate.draws_ok, estimate.draws_failed)
    {
        estimate.atom_intervals = Arc::from([]);
        estimate.mean_intervals = Arc::from([]);
        estimate.interval_reason = Some(Arc::from(reason));
    }
}

/// The evidence catalog a scenario or sequence decision binds and the one shared search
/// budget it charges.
pub struct EvidenceSearch {
    /// Evidence catalog the decision binds.
    pub catalog: EvidenceCatalog,
    /// Shared search budget of the decision, its verification replays and its compilation.
    pub budget: antecedent_core::SearchLimits,
}

/// Wrap any conversion failure as the one transport IO error.
pub(super) fn err(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(error.to_string())
}

/// Carry an estimation failure across the prepared-study boundary, keeping a
/// registered refusal code as [`IoError::Refused`] rather than flattening it.
pub(super) fn estimate_err(error: antecedent_estimate::EstimationError) -> IoError {
    IoError::from(error)
}

/// Refuse an exact-evaluation failure with its registered transport code.
pub(super) fn eval_err(error: &antecedent_expr::EvalError) -> IoError {
    estimate_err(antecedent_estimate::refuse_eval(error))
}

/// One compiled plan per request, in request order, through `prepare`.
pub(super) fn compile_plans<P>(
    requests: &[Assignment],
    prepare: impl FnMut(&Assignment) -> Result<P, IoError>,
) -> Result<Vec<P>, IoError> {
    requests.iter().map(prepare).collect()
}

/// Evaluate every retained plan after refusing a cancelled context at `stage`.
pub(super) fn evaluate_plans<P, D>(
    stage: &str,
    plans: &[P],
    ctx: &ExecutionContext,
    evaluate: impl FnMut(&P) -> Result<D, IoError>,
) -> Result<Vec<D>, IoError> {
    antecedent_estimate::refuse_cancelled(ctx, stage).map_err(estimate_err)?;
    plans.iter().map(evaluate).collect()
}

/// Law wires in the canonical order: by CBOR of `(population, regime, interventions)`.
pub(super) fn ordered_laws(
    laws: impl Iterator<Item = ExactLawWire>,
) -> Result<Vec<ExactLawWire>, IoError> {
    let mut laws = laws
        .map(|law| {
            Ok((
                antecedent_io::to_cbor(&(law.population.clone(), law.regime, &law.interventions))?,
                law,
            ))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    laws.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(laws.into_iter().map(|(_, law)| law).collect())
}

/// Every provider law of `data` as a wire, in canonical order.
pub(super) fn law_wires(data: &ExactTransportData) -> Result<Vec<ExactLawWire>, IoError> {
    ordered_laws(data.laws().iter().map(ExactLawWire::from_law))
}

/// A request assignment as wire values sorted by variable.
pub(super) fn assignments(request: &Assignment) -> Vec<(u32, ValueWire)> {
    let mut values: Vec<_> =
        request.entries().iter().map(|(v, x)| (v.raw(), ValueWire::from_value(x))).collect();
    values.sort_by_key(|(v, _)| *v);
    values
}

/// The distinct distribution leaves of a bound functional, in expression order.
pub(super) fn requirements(functional: &BoundTransportFunctional) -> Vec<ExactFactorRequirement> {
    let mut out = functional
        .leaf_factors()
        .into_iter()
        .map(|(expression, leaf)| ExactFactorRequirement {
            expression,
            binding: leaf.binding,
            variables: leaf.variables.to_vec(),
            conditioned_on: leaf.conditioned_on.to_vec(),
            interventions: leaf.intervention.iter().map(|a| a.variable).collect(),
        })
        .collect::<Vec<_>>();
    out.sort_by_key(|factor| factor.expression.raw());
    out
}

/// Sort axis values numerically (total order), then by CBOR bytes. A label has no numeric
/// value and follows every number; equal numbers of different wire kinds and distinct
/// labels are ordered by their encoding rather than by input order.
pub(super) fn sort_axis_values(values: &mut Vec<ValueWire>) -> Result<(), IoError> {
    let mut keyed = std::mem::take(values)
        .into_iter()
        .map(|value| Ok((value.to_value().as_f64(), antecedent_io::to_cbor(&value)?, value)))
        .collect::<Result<Vec<_>, IoError>>()?;
    keyed.sort_by(|a, b| {
        match (a.0, b.0) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
        .then_with(|| a.1.cmp(&b.1))
    });
    *values = keyed.into_iter().map(|(_, _, value)| value).collect();
    Ok(())
}

/// Hex digest of a wire value under `domain`.
pub(super) fn digest(domain: IdentityDomain, value: &impl Serialize) -> Result<String, IoError> {
    antecedent_io::digest_hex(domain, value)
}

/// The graph fields every transport artifact carries.
pub(super) struct GraphFields<'a> {
    pub(super) nodes: &'a [u32],
    pub(super) directed: &'a [(u32, u32)],
    pub(super) bidirected: &'a [(u32, u32)],
    pub(super) selections: &'a [u32],
}

/// Rebuild the selection diagram an artifact declares and check its proof against it,
/// under the consumer's own limits. Nothing in the artifact is trusted before the check.
pub(super) fn rebuild_checked_proof(
    fields: &GraphFields<'_>,
    proof: &TransportProofWire,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<(SelectionDiagram, ClassicalTransportDerivation), IoError> {
    let mut graph = Admg::empty();
    for node in fields.nodes {
        graph.add_node(NodeRef::Static(VariableId::from_raw(*node))).map_err(err)?;
    }
    for &(a, b) in fields.directed {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).map_err(err)?;
    }
    for &(a, b) in fields.bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).map_err(err)?;
    }
    graph.validate().map_err(err)?;
    let diagram = SelectionDiagram::try_new(
        graph,
        fields.selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
    )
    .map_err(err)?;
    let query = ClassicalTransportQuery {
        outcomes: proof.proof.outcomes.iter().copied().map(VariableId::from_raw).collect(),
        treatments: proof.proof.treatments.iter().copied().map(VariableId::from_raw).collect(),
        source: Arc::from(proof.proof.source.as_str()),
        target: Arc::from(proof.proof.target.as_str()),
    };
    let derivation = proof.check(
        &diagram,
        &query,
        SidLimits { steps: limits.operations, depth: limits.depth },
        ctx,
    )?;
    Ok((diagram, derivation))
}

/// Point each catalog binding at the snapshot its replacement providers carry.
///
/// One policy for every modality: a regime whose providers vanish, or that now spans more
/// than one snapshot, refuses. Keeping the old snapshot id for an unbound regime would
/// leave a binding that names data the handle no longer holds.
pub(super) fn rebind_snapshots<'a>(
    catalog: &EvidenceCatalog,
    snapshots_of: impl Fn(RegimeId) -> Vec<&'a str>,
) -> Result<EvidenceCatalog, IoError> {
    let mut rebound = catalog.clone();
    let mut bindings = rebound.bindings.to_vec();
    for binding in &mut bindings {
        let mut snapshots = snapshots_of(binding.regime);
        snapshots.sort_unstable();
        snapshots.dedup();
        match snapshots.as_slice() {
            [] => return Err(err("missing bound snapshot")),
            [snapshot] => binding.snapshot_identity = Arc::from(*snapshot),
            _ => return Err(err("regime has inconsistent snapshot identities")),
        }
    }
    rebound.bindings = bindings.into();
    Ok(rebound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceKind, EvidenceRegime,
        RegimeBinding, RegimeKind, SamplingDesign, TargetSampling, VariableCoordinate,
        VariableDomain,
    };

    fn catalog_bound_to(snapshot: &str) -> EvidenceCatalog {
        let coordinate = |n| VariableCoordinate {
            variable: VariableId::from_raw(n),
            domain: VariableDomain::Binary,
            unit: None,
        };
        EvidenceCatalog::try_new(
            [Environment::try_new("target", [coordinate(0), coordinate(1)], []).unwrap()],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [VariableId::from_raw(0), VariableId::from_raw(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [RegimeBinding {
                dataset_identity: None,
                regime: RegimeId::from_raw(0),
                snapshot_identity: Arc::from(snapshot),
                schema_names: Arc::from([]),
                sampling: SamplingDesign::Independent,
                weights: None,
                dependence: DependenceGroup::IndependentStudies,
            }],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap()
    }

    #[test]
    fn rebinding_follows_the_replacement_snapshot() {
        let catalog = catalog_bound_to("old");
        let rebound = rebind_snapshots(&catalog, |_| vec!["new", "new"]).unwrap();
        assert_eq!(rebound.bindings[0].snapshot_identity.as_ref(), "new");
        assert_eq!(catalog.bindings[0].snapshot_identity.as_ref(), "old");
    }

    #[test]
    fn rebinding_refuses_an_unbound_or_ambiguous_regime() {
        let catalog = catalog_bound_to("old");
        let missing = rebind_snapshots(&catalog, |_| Vec::new()).unwrap_err();
        assert!(missing.to_string().contains("missing bound snapshot"), "{missing}");
        let split = rebind_snapshots(&catalog, |_| vec!["a", "b"]).unwrap_err();
        assert!(split.to_string().contains("inconsistent snapshot identities"), "{split}");
    }
}

#[cfg(test)]
mod probability_vector_tests {
    use super::is_probability_vector;

    #[test]
    fn accepts_normalised_and_refuses_the_rest() {
        assert!(is_probability_vector(&[0.25, 0.75]));
        assert!(!is_probability_vector(&[0.25, 0.76]));
        assert!(!is_probability_vector(&[1.5, -0.5]));
        assert!(!is_probability_vector(&[f64::NAN, 1.0]));
        assert!(!is_probability_vector(&[]));
    }
}
