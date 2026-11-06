//! Dependence and structural-atom checks for composed inputs.

use antecedent_contract::ExternalCapability;

use super::{BoundaryError, DecisionInput, InputSource};
use crate::decision_eval::DecisionEvalError;
use crate::decision_structural::StructuralAtom;

/// How two inputs' evidence relates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceRelation {
    /// Declared independent: no shared data, prior or fitted model.
    IndependentSources,
    /// Both use these data.
    SharedData {
        /// Shared data identities.
        ids: Vec<String>,
    },
    /// Both use this prior.
    SharedPrior {
        /// Shared prior identity.
        id: String,
    },
    /// Both use this fitted model.
    SharedFittedModel {
        /// Shared model identity.
        id: String,
    },
    /// Dependence was not assessed; this is never independence.
    UnknownDependence,
}

impl EvidenceRelation {
    fn shared_ids(&self) -> Vec<String> {
        match self {
            Self::SharedData { ids } => ids.clone(),
            Self::SharedPrior { id } | Self::SharedFittedModel { id } => vec![id.clone()],
            Self::IndependentSources | Self::UnknownDependence => Vec::new(),
        }
    }
}

/// Kind of supported dependence route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKind {
    /// A covariance supplied by an aligned joint law.
    Covariance,
    /// An aligned joint law over both inputs' coordinates.
    JointLaw,
}

/// A declared route that accounts for dependence between two inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependenceRoute {
    /// Route kind.
    pub kind: RouteKind,
    /// Route identity.
    pub id: String,
    /// Input that supplies the joint law; it must be one of the pair.
    pub source_input: String,
}

/// The declared relation between two inputs, with an optional dependence route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairRelation {
    /// First input identity.
    pub left: String,
    /// Second input identity.
    pub right: String,
    /// Declared evidence relation.
    pub relation: EvidenceRelation,
    /// Covariance or joint-law route, when one is declared.
    pub route: Option<DependenceRoute>,
}

/// The declared way inputs are combined. The operations are distinct and none
/// implies another.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompositionOperation {
    /// Combine estimates as independent evidence, or pair their draws.
    StatisticalPooling,
    /// Use one source as a prior for another.
    BayesianBorrowing,
    /// Carry a result to a target population.
    CausalTransport,
    /// Reuse evidence in a second analysis, counted once.
    EvidenceReuse,
}

/// What a passed composition check records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionReceipt {
    /// The declared operation.
    pub operation: CompositionOperation,
    /// Whether the combination assumes independent sources.
    pub independence_assumed: bool,
    /// Identities of the dependence routes used.
    pub routes: Vec<String>,
    /// Shared data, prior and model identities, sorted and distinct.
    pub shared_evidence: Vec<String>,
}

enum PairIssue {
    Shared,
    Unknown,
}

fn pair_policy(
    operation: CompositionOperation,
    relation: &EvidenceRelation,
) -> Result<(), PairIssue> {
    use CompositionOperation as Op;
    use EvidenceRelation as Rel;
    match (operation, relation) {
        (_, Rel::UnknownDependence) => Err(PairIssue::Unknown),
        (Op::StatisticalPooling, Rel::SharedData { .. } | Rel::SharedPrior { .. })
        | (Op::StatisticalPooling | Op::BayesianBorrowing, Rel::SharedFittedModel { .. })
        | (Op::BayesianBorrowing, Rel::SharedData { .. }) => Err(PairIssue::Shared),
        _ => Ok(()),
    }
}

fn validate_route(
    route: &DependenceRoute,
    left: &DecisionInput,
    right: &DecisionInput,
    row_pairing: bool,
) -> Result<(), BoundaryError> {
    let unsupported = || BoundaryError::DependenceRouteNotSupported {
        left: left.id.clone(),
        right: right.id.clone(),
    };
    let source = [left, right].into_iter().find(|input| input.id == route.source_input);
    let Some(input) = source else {
        return Err(unsupported());
    };
    let aligned = rows_pairable(input);
    let capable = match route.kind {
        // A covariance cannot pair draws row by row (`covariance_insufficient_for_nonlinear`).
        RouteKind::Covariance if row_pairing => false,
        RouteKind::Covariance => {
            input.provenance.capabilities.contains(&ExternalCapability::Covariance)
        }
        RouteKind::JointLaw => input.provenance.capabilities.contains(&ExternalCapability::Sample),
    };
    if aligned && capable && !route.id.trim().is_empty() { Ok(()) } else { Err(unsupported()) }
}

fn find_relation<'a>(
    relations: &'a [PairRelation],
    left: &str,
    right: &str,
) -> Option<&'a PairRelation> {
    relations
        .iter()
        .find(|r| (r.left == left && r.right == right) || (r.left == right && r.right == left))
}

fn validate_relations(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
) -> Result<(), BoundaryError> {
    for (position, relation) in relations.iter().enumerate() {
        for id in [&relation.left, &relation.right] {
            if !inputs.iter().any(|input| &input.id == id) {
                return Err(BoundaryError::UnknownInput(id.clone()));
            }
        }
        if relation.left == relation.right {
            return Err(BoundaryError::InvalidInput("a relation names two different inputs"));
        }
        if find_relation(&relations[..position], &relation.left, &relation.right).is_some() {
            return Err(BoundaryError::InvalidInput("duplicate relation for one pair"));
        }
    }
    Ok(())
}

/// Check that `inputs` may be combined under the declared `operation`.
///
/// Every pair of inputs needs a declared [`PairRelation`]; a pair with none is
/// [`EvidenceRelation::UnknownDependence`]. A relation declared independent is
/// overridden to shared data when the two inputs carry the same snapshot
/// identity. Unknown dependence refuses every operation unless a route is
/// declared. Statistical pooling also refuses shared data, priors and fitted
/// models; Bayesian borrowing allows a shared prior but refuses shared data and
/// fitted models; transport and evidence reuse record shared evidence without
/// assuming independence. A declared covariance or joint-law route must name one
/// of the pair as an aligned joint law with the matching capability.
///
/// # Errors
/// No declared operation, fewer than two inputs, malformed relations, shared
/// evidence or unknown dependence without a supported route.
pub fn check_composition(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
    operation: Option<CompositionOperation>,
) -> Result<CompositionReceipt, BoundaryError> {
    compose_checked(inputs, relations, operation, false)
}

/// Shared body of [`check_composition`] and [`check_paired_draws`].
///
/// `row_pairing` is set when the caller pairs draws row by row; a covariance
/// then supports no dependence, as the joint-law rule refuses a covariance for
/// a nonlinear row-paired operation.
fn compose_checked(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
    operation: Option<CompositionOperation>,
    row_pairing: bool,
) -> Result<CompositionReceipt, BoundaryError> {
    let Some(operation) = operation else {
        return Err(BoundaryError::OperationNotDeclared);
    };
    if inputs.len() < 2 {
        return Err(BoundaryError::InvalidInput("composition needs at least two inputs"));
    }
    for (position, input) in inputs.iter().enumerate() {
        if inputs[..position].iter().any(|earlier| earlier.id == input.id) {
            return Err(BoundaryError::InvalidInput("duplicate input id"));
        }
    }
    validate_relations(inputs, relations)?;
    let mut routes = Vec::new();
    let mut shared = Vec::new();
    for (i, left) in inputs.iter().enumerate() {
        for right in &inputs[i + 1..] {
            let declared = find_relation(relations, &left.id, &right.id);
            let mut relation =
                declared.map_or(EvidenceRelation::UnknownDependence, |r| r.relation.clone());
            if relation == EvidenceRelation::IndependentSources
                && left.provenance.snapshot_id == right.provenance.snapshot_id
            {
                relation =
                    EvidenceRelation::SharedData { ids: vec![left.provenance.snapshot_id.clone()] };
            }
            let route = declared.and_then(|r| r.route.as_ref());
            if let Some(route) = route {
                validate_route(route, left, right, row_pairing)?;
            }
            if let Err(issue) = pair_policy(operation, &relation) {
                if let Some(route) = route {
                    routes.push(route.id.clone());
                } else {
                    let (l, r) = (left.id.clone(), right.id.clone());
                    return Err(match issue {
                        PairIssue::Shared => {
                            BoundaryError::SharedEvidenceNotIndependent { left: l, right: r }
                        }
                        PairIssue::Unknown => {
                            BoundaryError::UnknownDependence { left: l, right: r }
                        }
                    });
                }
            } else if let Some(route) = route {
                routes.push(route.id.clone());
            }
            shared.extend(relation.shared_ids());
        }
    }
    shared.sort();
    shared.dedup();
    routes.sort();
    routes.dedup();
    Ok(CompositionReceipt {
        operation,
        independence_assumed: operation == CompositionOperation::StatisticalPooling
            && routes.is_empty(),
        routes,
        shared_evidence: shared,
    })
}

/// The alignment of an input's rows as the joint-law propagation rule reads
/// it: the artifact's real pairing identity for one joint law, independent
/// marginals for separately sampled draws, moments only otherwise.
fn input_alignment(input: &DecisionInput) -> antecedent_contract::InputAlignment {
    match &input.source {
        InputSource::JointLaw(artifact) => artifact
            .input_alignment()
            .unwrap_or(antecedent_contract::InputAlignment::IndependentMarginals),
        _ => antecedent_contract::InputAlignment::MomentsOnly,
    }
}

/// Whether an input's rows are realizations of one joint law, decided by the
/// joint-law propagation rule (`antecedent_contract::check_rows_pairable`).
fn rows_pairable(input: &DecisionInput) -> bool {
    // One input is checked against a partner of the same law, so only its own
    // flag is decided here; distinct artifacts are compared by their real
    // identities through `check_rows_pairable_across`.
    let alignment = match input_alignment(input) {
        antecedent_contract::InputAlignment::Aligned { .. } => {
            antecedent_contract::InputAlignment::Aligned {
                alignment_id: "declared_pairing".to_owned(),
            }
        }
        other => other,
    };
    antecedent_contract::check_rows_pairable(&alignment).is_ok()
}

/// Check that the draws of `inputs` may be paired row by row.
///
/// Pairing draws from separate sources treats them as independent, so this is
/// statistical pooling plus a requirement that every input is an aligned joint
/// law with the same draw count.
///
/// # Errors
/// Everything [`check_composition`] refuses under statistical pooling, and any
/// input that is not an aligned joint law, or a draw-count mismatch.
pub fn check_paired_draws(
    inputs: &[DecisionInput],
    relations: &[PairRelation],
) -> Result<CompositionReceipt, BoundaryError> {
    let mut draw_count = None;
    let alignments: Vec<antecedent_contract::InputAlignment> =
        inputs.iter().map(input_alignment).collect();
    for input in inputs {
        // The propagation rule decides whether the input's rows may be paired;
        // the refusal keeps this module's existing error and detail.
        let InputSource::JointLaw(artifact) = &input.source else {
            return Err(BoundaryError::Eval(DecisionEvalError::JointLawRequired {
                action: None,
                supplied_alignment: "not_a_joint_law",
            }));
        };
        if !rows_pairable(input) {
            return Err(BoundaryError::Eval(DecisionEvalError::JointLawRequired {
                action: None,
                supplied_alignment: "not_a_joint_law",
            }));
        }
        let n = artifact.metadata().shape[0];
        if draw_count.is_some_and(|previous| previous != n) {
            return Err(BoundaryError::InvalidInput("paired draw count mismatch"));
        }
        draw_count = Some(n);
    }
    let receipt =
        compose_checked(inputs, relations, Some(CompositionOperation::StatisticalPooling), true)?;
    // Separate artifacts are paired only when they carry the same alignment
    // identity; a flag on each input alone cannot detect two different laws.
    // A declared joint-law route is the pair's own pairing declaration.
    for (i, left) in inputs.iter().enumerate() {
        for (j, right) in inputs.iter().enumerate().skip(i + 1) {
            let routed = find_relation(relations, &left.id, &right.id)
                .and_then(|relation| relation.route.as_ref())
                .is_some_and(|route| route.kind == RouteKind::JointLaw);
            if routed {
                continue;
            }
            let pair = [(left.id.as_str(), &alignments[i]), (right.id.as_str(), &alignments[j])];
            if antecedent_contract::check_rows_pairable_across(&pair).is_err() {
                return Err(BoundaryError::Eval(DecisionEvalError::JointLawRequired {
                    action: None,
                    supplied_alignment: "alignment_mismatch",
                }));
            }
        }
    }
    Ok(receipt)
}

/// How structural atoms are to be combined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AtomCombination {
    /// Report each atom's answer without combining.
    ReportEach,
    /// Rank by the worst case over atoms.
    WorstCase,
    /// Weight atoms by their declared probabilities.
    WeightedByDeclaredProbabilities,
}

/// A permitted way to combine structural atoms.
#[derive(Clone, Debug, PartialEq)]
pub enum AtomPlan {
    /// Report each atom.
    ReportEach,
    /// Worst case over atoms.
    WorstCase,
    /// Declared `(atom id, probability)` weights, not renormalized.
    Weighted(Vec<(String, f64)>),
}

/// Check that structural `atoms` may be combined as requested.
///
/// Two or more atoms are alternative structures; averaging them is supported only
/// when every atom carries a declared probability. A completion count is not a
/// probability, and missing mass is retained, not renormalized.
///
/// # Errors
/// Blank or repeated atom identities, weighting without a probability on every
/// atom (`ConflictingAtomsNotAveraged`), and invalid probabilities.
pub fn check_atom_combination(
    atoms: &[StructuralAtom],
    combination: AtomCombination,
) -> Result<AtomPlan, BoundaryError> {
    if atoms.is_empty() {
        return Err(BoundaryError::InvalidInput("structural atoms"));
    }
    for (position, atom) in atoms.iter().enumerate() {
        if atom.id.trim().is_empty()
            || atoms[..position].iter().any(|earlier| earlier.id == atom.id)
        {
            return Err(BoundaryError::InvalidInput("structural atom identity"));
        }
    }
    match combination {
        AtomCombination::ReportEach => Ok(AtomPlan::ReportEach),
        AtomCombination::WorstCase => Ok(AtomPlan::WorstCase),
        AtomCombination::WeightedByDeclaredProbabilities => {
            if atoms.len() == 1 && atoms[0].probability.is_none() {
                return Ok(AtomPlan::Weighted(vec![(atoms[0].id.clone(), 1.0)]));
            }
            if atoms.iter().any(|atom| atom.probability.is_none()) {
                return Err(BoundaryError::ConflictingAtomsNotAveraged);
            }
            let weights: Vec<(String, f64)> = atoms
                .iter()
                .filter_map(|atom| atom.probability.map(|p| (atom.id.clone(), p)))
                .collect();
            let total: f64 = weights.iter().map(|(_, p)| *p).sum();
            if weights.iter().any(|(_, p)| !p.is_finite() || *p < 0.0) || total > 1.0 + 1e-9 {
                return Err(BoundaryError::InvalidAtomProbabilities);
            }
            Ok(AtomPlan::Weighted(weights))
        }
    }
}
