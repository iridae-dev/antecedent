//! The joint-law propagation rule: which dependence information a composed
//! uncertain quantity needs, and nothing computed until it is present.
//!
//! This is the one place that decides whether the draws or moments of several
//! inputs may be combined. Unrelated posterior or bootstrap draws are never
//! paired by index; a row pairing is a statement that the rows are
//! realizations of one joint law, and it is licensed only by an alignment
//! identity that every input shares.
//!
//! The rule, by operation:
//!
//! * **Linear** operations (a linear transform, a contrast, a sum, a weighted
//!   sum) need a covariance when the inputs share evidence or an explicit joint
//!   moment exists. A declared independence is accepted only for inputs with
//!   distinct evidence. The delta method is exact for a linear map, so a
//!   covariance suffices.
//! * **Nonlinear** operations (a product, any nonlinear functional, a chain of
//!   Monte Carlo transformations) need an *aligned joint law*: draws that carry
//!   one shared alignment identity. A covariance is not enough and independent
//!   marginals cannot be paired.
//! * **Mixtures** need known weights; the components need no joint law.
//! * **Transported compositions** (an interval transported to a target
//!   population composed with another claim, or the source sampling, target
//!   sampling and transport functional of one transported claim) need an
//!   aligned joint law or a supplied covariance: the pieces are dependent when
//!   evidence is shared, and independence is never assumed for them.
//! * **A transported single claim** (one input) is a linear transform of the
//!   transported interval and needs no cross term; a mixture of transported
//!   claims with known weights needs no joint law; any other composition of a
//!   transported interval with another claim needs the aligned draws or the
//!   covariance, and a spread of the pieces of one transported claim needs
//!   all three pieces (source sampling, target sampling, transport functional).
//! * **A dependence envelope** is the one route for a functional of unpaired
//!   marginals. It is a structural envelope (or, for a declared correlation
//!   range, an assumption range), labelled as such, and is never a point, an
//!   interval or a posterior.
//!
//! A composed quantity has three readings besides the envelope:
//!
//! * the **point** value under a licensed dependence route;
//! * the **spread** (variance or standard error, and the draw-based summaries
//!   built on aligned draws) at the composed coordinate, labelled
//!   `uncalibrated_spread`: it is what the licensed dependence route
//!   determines and carries no coverage statement;
//! * an **interval** only through [`check_composed_interval`], which needs a
//!   measured coverage record for the composed coordinate (operation, route,
//!   input roles and construction). A component interval does not cover the
//!   composition; without a record the interval reading is closed. A
//!   posterior statement about a composed quantity is closed in the same way.
//!
//! Every refusal carries a registered reason code, a namespaced
//! `joint_law_propagation.*` detail and a remedy that names the dependence
//! information that would license the request.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use crate::ExternalRefusal;
use crate::evidence_ledger::{EvidenceKey, EvidenceLedger, Overlap, evidence_key};
use antecedent_core::named_variants;

/// Stage reported by every refusal of this module.
pub const JOINT_LAW_STAGE: &str = "propagate";
/// Label of a structural (Fréchet-Hoeffding) envelope.
pub use antecedent_core::STRUCTURAL_ENVELOPE_LABEL;
/// Label of a range implied by a declared correlation range.
pub const ASSUMPTION_RANGE_LABEL: &str = "assumption_range";
/// Label of a value computed under a licensed dependence route.
pub const POINT_ONLY_LABEL: &str = "point_only";
/// Label of a spread computed under a licensed dependence route, with no
/// coverage statement.
pub const UNCALIBRATED_SPREAD_LABEL: &str = "uncalibrated_spread";
/// Label of an interval backed by a measured coverage record at its composed
/// coordinate.
pub const MEASURED_INTERVAL_LABEL: &str = "measured_interval";

/// The operation that combines uncertain inputs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PropagationOperation {
    /// A linear transform `a X + c` (one input).
    LinearTransform,
    /// A contrast `X - Y` or a linear combination with coefficients summing to
    /// zero.
    Contrast,
    /// A sum `X + Y`.
    Sum,
    /// A weighted sum `sum a_i X_i`.
    WeightedSum,
    /// A product `X Y`.
    Product,
    /// A nonlinear functional of the inputs.
    NonlinearFunctional,
    /// A chain of Monte Carlo transformations, each stage fed by the last.
    ChainedSampling,
    /// A finite mixture of the inputs.
    Mixture,
    /// A transported interval composed with another claim, or the pieces of one
    /// transported claim (source sampling, target sampling, transport
    /// functional).
    TransportedComposition,
}

/// The class of dependence information an operation needs.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PropagationClass {
    /// A covariance or explicit joint moment suffices.
    Linear,
    /// An aligned joint law is required.
    Nonlinear,
    /// Known mixture weights are required.
    Mixture,
    /// An aligned joint law or a supplied covariance, never independence.
    Transported,
}

named_variants! {
    PropagationOperation {
        LinearTransform => "linear_transform",
        Contrast => "contrast",
        Sum => "sum",
        WeightedSum => "weighted_sum",
        Product => "product",
        NonlinearFunctional => "nonlinear_functional",
        ChainedSampling => "chained_sampling",
        Mixture => "mixture",
        TransportedComposition => "transported_composition",
    }
}

impl PropagationOperation {
    /// The dependence class the operation needs.
    #[must_use]
    pub const fn class(self) -> PropagationClass {
        match self {
            Self::LinearTransform | Self::Contrast | Self::Sum | Self::WeightedSum => {
                PropagationClass::Linear
            }
            Self::Product | Self::NonlinearFunctional | Self::ChainedSampling => {
                PropagationClass::Nonlinear
            }
            Self::Mixture => PropagationClass::Mixture,
            Self::TransportedComposition => PropagationClass::Transported,
        }
    }

    const fn min_inputs(self) -> usize {
        match self {
            Self::LinearTransform | Self::Mixture => 1,
            _ => 2,
        }
    }
}

/// What an input is in the composition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InputRole {
    /// An ordinary estimate or draw set.
    Estimate,
    /// An interval (or draw set) transported to another population.
    TransportedInterval,
    /// The source-population sampling component of a transported claim.
    SourceSampling,
    /// The target-population sampling component of a transported claim.
    TargetSampling,
    /// The transport functional (the map from source to target quantity,
    /// including any estimated weights or nuisance functions).
    TransportFunctional,
}

named_variants! {
    InputRole {
        Estimate => "estimate",
        TransportedInterval => "transported_interval",
        SourceSampling => "source_sampling",
        TargetSampling => "target_sampling",
        TransportFunctional => "transport_functional",
    }
}

impl InputRole {
    const fn is_transported(self) -> bool {
        !matches!(self, Self::Estimate)
    }

    /// One of the three pieces a transported claim's uncertainty depends on.
    const fn is_piece(self) -> bool {
        matches!(self, Self::SourceSampling | Self::TargetSampling | Self::TransportFunctional)
    }
}

/// How an input's draws relate to other inputs' draws.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputAlignment {
    /// Rows are realizations of the joint law with this alignment identity;
    /// two inputs pair row by row only when the identities are equal.
    Aligned {
        /// The shared alignment identity.
        alignment_id: String,
    },
    /// Independently produced draws: row pairing is invalid.
    IndependentMarginals,
    /// A summary (moments or a marginal law) with no draws to pair.
    MomentsOnly,
}

/// One input of a composition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropagationInput {
    /// Stable input identity.
    pub id: String,
    /// Identity of the evidence (data snapshot, prior or fitted model) the
    /// input rests on. Two inputs with the same evidence identity are not
    /// independent.
    pub evidence_id: String,
    /// What the input is in the composition.
    pub role: InputRole,
    /// How the input's draws relate to the others'.
    pub alignment: InputAlignment,
}

/// The dependence information the caller supplies.
#[derive(Clone, Debug, PartialEq)]
pub enum PropagationDependence {
    /// Nothing declared. This is never independence.
    Unknown,
    /// The inputs are declared independent; only accepted for inputs with
    /// distinct evidence.
    DeclaredIndependent,
    /// An explicit joint moment: a cross-covariance with this identity.
    Covariance {
        /// Identity of the covariance declaration or of the shared evidence
        /// that supplies it.
        id: String,
    },
    /// The inputs' draws are an aligned joint law.
    AlignedJointLaw,
    /// Known mixture weights.
    KnownWeights,
    /// A declared range of one common correlation; an assumption range.
    CorrelationRange {
        /// Lower correlation.
        rho_lo: f64,
        /// Upper correlation.
        rho_hi: f64,
    },
    /// No dependence information beyond the marginals: Fréchet-Hoeffding.
    MarginalsOnly,
}

/// How the composed quantity is to be read.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RequestedReading {
    /// A point value.
    Point,
    /// The spread of the composed quantity under the licensed dependence
    /// route: a variance, a standard error or a draw-based summary, with no
    /// coverage statement.
    Spread,
    /// An interval. Closed unless a measured coverage record for the composed
    /// coordinate is supplied to [`check_composed_interval`].
    Interval,
    /// A posterior or posterior-probability statement. Closed.
    Posterior,
    /// A labelled envelope.
    Envelope,
}

named_variants! {
    RequestedReading {
        Point => "point",
        Spread => "spread",
        Interval => "interval",
        Posterior => "posterior",
        Envelope => "envelope",
    }
}

/// A composition to be checked before any number is formed.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagationRequest {
    /// The operation.
    pub operation: PropagationOperation,
    /// The inputs.
    pub inputs: Vec<PropagationInput>,
    /// The dependence information supplied.
    pub dependence: PropagationDependence,
    /// How the result is to be read.
    pub reading: RequestedReading,
}

/// The computation a licensed request runs. Each route is its own support row.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PropagationRoute {
    /// Delta method for a linear map under a covariance (or no cross term for
    /// a single input or a distinct-evidence independence declaration).
    LinearMoments,
    /// An expectation under an aligned joint law.
    AlignedJointLaw,
    /// Law of total variance for known weights.
    KnownWeightMixture,
    /// Fréchet-Hoeffding envelope over every coupling of the marginals.
    FrechetEnvelope,
    /// Extremes over a declared correlation range.
    CorrelationRangeEnvelope,
}

named_variants! {
    PropagationRoute {
        LinearMoments => "linear_moments",
        AlignedJointLaw => "aligned_joint_law",
        KnownWeightMixture => "known_weight_mixture",
        FrechetEnvelope => "frechet_envelope",
        CorrelationRangeEnvelope => "correlation_range_envelope",
    }
}

impl PropagationRoute {
    /// The label the route's result carries.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LinearMoments | Self::AlignedJointLaw | Self::KnownWeightMixture => {
                POINT_ONLY_LABEL
            }
            Self::FrechetEnvelope => STRUCTURAL_ENVELOPE_LABEL,
            Self::CorrelationRangeEnvelope => ASSUMPTION_RANGE_LABEL,
        }
    }
}

/// What a passed check records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropagationPlan {
    /// The checked operation.
    pub operation: PropagationOperation,
    /// The route to run.
    pub route: PropagationRoute,
    /// The alignment identity the inputs share, for an aligned route.
    pub alignment_id: Option<String>,
    /// Identity of the covariance declaration, for a covariance route.
    pub covariance_id: Option<String>,
    /// Evidence identities shared by two or more inputs, sorted and distinct.
    pub shared_evidence: Vec<String>,
    /// The label of the result.
    pub label: &'static str,
    /// The pieces of a transported claim (`source_sampling`,
    /// `target_sampling`, `transport_functional`) that were not named, when at
    /// least one piece was. Empty when no piece was named or all were.
    pub missing_transport_pieces: Vec<&'static str>,
}

fn refuse(
    code: &'static str,
    detail: &'static str,
    offending: impl Into<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal::new(code, JOINT_LAW_STAGE, detail)
        .with_offending(offending)
        .with_remedy(remedy)
}

fn blank(text: &str) -> bool {
    text.trim().is_empty()
}

fn invalid(offending: impl Into<String>, remedy: &'static str) -> ExternalRefusal {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "joint_law_propagation.invalid_request",
        offending,
        remedy,
    )
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn validate(request: &PropagationRequest) -> Result<(), ExternalRefusal> {
    let needed = request.operation.min_inputs();
    if request.inputs.len() < needed {
        return Err(invalid(
            request.operation.as_str(),
            "name at least two inputs for a combination (one for a linear transform or a mixture)",
        ));
    }
    let mut seen = BTreeSet::new();
    for input in &request.inputs {
        if blank(&input.id) || blank(&input.evidence_id) {
            return Err(invalid(
                "inputs",
                "give every input an identity and the identity of the evidence it rests on",
            ));
        }
        if !seen.insert(input.id.as_str()) {
            return Err(invalid(input.id.clone(), "list each input once"));
        }
        if let InputAlignment::Aligned { alignment_id } = &input.alignment {
            if blank(alignment_id) {
                return Err(invalid(
                    input.id.clone(),
                    "an aligned input names its alignment identity",
                ));
            }
        }
    }
    if let PropagationDependence::Covariance { id } = &request.dependence {
        if blank(id) {
            return Err(invalid(
                "covariance",
                "name the covariance declaration or the shared evidence that supplies it",
            ));
        }
    }
    Ok(())
}

fn remember_shared<'a>(
    shared: &mut std::collections::BTreeMap<EvidenceKey<'a>, &'a str>,
    label: &'a str,
) {
    shared
        .entry(evidence_key(label))
        .and_modify(|first| *first = (*first).min(label))
        .or_insert(label);
}
fn shared_evidence(inputs: &[PropagationInput]) -> Vec<String> {
    let mut groups = std::collections::BTreeMap::<EvidenceKey<'_>, (&str, bool)>::new();
    for input in inputs {
        let label = input.evidence_id.as_str();
        groups
            .entry(evidence_key(label))
            .and_modify(|(first, duplicate)| {
                *first = (*first).min(label);
                *duplicate = true;
            })
            .or_insert((label, false));
    }
    let mut shared: Vec<_> = groups
        .into_values()
        .filter(|(_, duplicate)| *duplicate)
        .map(|(label, _)| label.to_owned())
        .collect();
    shared.sort();
    shared
}

/// The alignment identity every input shares, or the refusal that says why
/// the rows may not be paired.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn common_alignment(inputs: &[PropagationInput]) -> Result<String, ExternalRefusal> {
    let joint = antecedent_core::reason_code!("joint_law_required");
    let mut identity: Option<&str> = None;
    for input in inputs {
        match &input.alignment {
            InputAlignment::Aligned { alignment_id } => match identity {
                None => identity = Some(alignment_id.as_str()),
                Some(first) if first == alignment_id => {}
                Some(first) => {
                    return Err(refuse(
                        joint,
                        "joint_law_propagation.alignment_mismatch",
                        input.id.clone(),
                    "the inputs are aligned joint laws with different alignment identities and their rows are not paired",
                    )
                    .with_expected(first)
                    .with_supplied(alignment_id.clone()));
                }
            },
            InputAlignment::IndependentMarginals | InputAlignment::MomentsOnly => {
                return Err(refuse(
                    joint,
                    "joint_law_propagation.index_pairing_refused",
                    input.id.clone(),
                    "draws of unrelated marginals are never paired by index; supply the draws of one joint law (one alignment identity for every input), or a covariance for a linear operation, or ask for the dependence envelope",
                ));
            }
        }
    }
    identity
        .map(str::to_owned)
        .ok_or_else(|| invalid("inputs", "an aligned route needs at least one aligned input"))
}

fn plan(
    request: &PropagationRequest,
    route: PropagationRoute,
    alignment_id: Option<String>,
    covariance_id: Option<String>,
    shared: Vec<String>,
) -> PropagationPlan {
    PropagationPlan {
        operation: request.operation,
        route,
        alignment_id,
        covariance_id,
        shared_evidence: shared,
        label: route.label(),
        missing_transport_pieces: Vec::new(),
    }
}

fn closed_reading(reading: RequestedReading) -> Option<ExternalRefusal> {
    let code = antecedent_core::reason_code!("cell_not_licensed");
    match reading {
        RequestedReading::Interval => Some(refuse(
            code,
            "joint_law_propagation.interval_route_closed",
            "interval",
            "no interval for a composed quantity has a measured coverage record at its composed coordinate, and a component interval does not cover the composition; the labelled point value under a licensed dependence route, or the dependence envelope, is the result",
        )),
        RequestedReading::Posterior => Some(refuse(
            code,
            "joint_law_propagation.posterior_route_closed",
            "posterior",
            "no posterior statement for a composed quantity has a measured calibration at its composed coordinate; the labelled point value under a licensed dependence route is the result",
        )),
        RequestedReading::Point | RequestedReading::Spread | RequestedReading::Envelope => None,
    }
}

fn dependence_unknown(operation: PropagationOperation) -> ExternalRefusal {
    refuse(
        antecedent_core::reason_code!("sampling_dependence_unknown"),
        "joint_law_propagation.dependence_unknown",
        operation.as_str(),
        "declare the dependence between the inputs: a covariance when they share evidence, an aligned joint law (draws with one alignment identity), or distinct evidence with a declared independence; independence is never assumed",
    )
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_transported(
    request: &PropagationRequest,
    shared: Vec<String>,
) -> Result<PropagationPlan, ExternalRefusal> {
    let needs = |offending: &str| {
        refuse(
            antecedent_core::reason_code!("joint_law_required"),
            "joint_law_propagation.transported_dependence_required",
            offending.to_owned(),
            "a transported interval depends on source sampling, target sampling and the transport functional, and the pieces are dependent when evidence is shared; supply the aligned draws of one joint law (one alignment identity for every piece) or the covariance between the pieces before composing it with another claim",
        )
    };
    let no_pieces = request.inputs.iter().all(|input| !input.role.is_piece());
    // One transported claim transformed on its own has no second claim to be
    // dependent on; its internal pieces are already inside it.
    if request.operation == PropagationOperation::LinearTransform
        && request.inputs.len() == 1
        && request.inputs[0].role == InputRole::TransportedInterval
    {
        return Ok(plan(request, PropagationRoute::LinearMoments, None, None, shared));
    }
    // A mixture selects one component; transported components with known
    // weights need no joint law.
    if request.operation == PropagationOperation::Mixture
        && no_pieces
        && request.dependence == PropagationDependence::KnownWeights
    {
        return Ok(plan(request, PropagationRoute::KnownWeightMixture, None, None, shared));
    }
    match &request.dependence {
        PropagationDependence::AlignedJointLaw => {
            let alignment = common_alignment(&request.inputs)?;
            Ok(plan(request, PropagationRoute::AlignedJointLaw, Some(alignment), None, shared))
        }
        PropagationDependence::Covariance { id } => {
            if request.operation.class() == PropagationClass::Nonlinear {
                return Err(needs(request.operation.as_str()));
            }
            Ok(plan(request, PropagationRoute::LinearMoments, None, Some(id.clone()), shared))
        }
        _ => Err(needs(request.operation.as_str())),
    }
}

/// The pieces of a transported claim that were not named, when any piece was.
fn missing_pieces(inputs: &[PropagationInput]) -> Vec<&'static str> {
    if inputs.iter().all(|input| !input.role.is_piece()) {
        return Vec::new();
    }
    [InputRole::SourceSampling, InputRole::TargetSampling, InputRole::TransportFunctional]
        .into_iter()
        .filter(|piece| inputs.iter().all(|input| input.role != *piece))
        .map(InputRole::as_str)
        .collect()
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_envelope(
    request: &PropagationRequest,
    shared: Vec<String>,
) -> Result<PropagationPlan, ExternalRefusal> {
    use PropagationOperation as Op;
    if let PropagationDependence::CorrelationRange { rho_lo, rho_hi } = request.dependence {
        if !(rho_lo.is_finite()
            && rho_hi.is_finite()
            && -1.0 <= rho_lo
            && rho_lo < rho_hi
            && rho_hi <= 1.0)
        {
            return Err(invalid(
                "correlation_range",
                "declare finite correlation bounds with -1 <= lower < upper <= 1",
            ));
        }
    }
    let route = match request.dependence {
        PropagationDependence::CorrelationRange { .. } => {
            PropagationRoute::CorrelationRangeEnvelope
        }
        _ => PropagationRoute::FrechetEnvelope,
    };
    let supported = match route {
        PropagationRoute::CorrelationRangeEnvelope => {
            matches!(request.operation, Op::Sum | Op::Contrast | Op::WeightedSum | Op::Product)
        }
        _ => matches!(
            request.operation,
            Op::Sum | Op::Contrast | Op::WeightedSum | Op::Product | Op::NonlinearFunctional
        ),
    };
    if !supported || request.inputs.len() != 2 {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "joint_law_propagation.envelope_not_available",
            request.operation.as_str(),
            "the dependence envelope covers a sum, contrast, weighted sum, product or monotone functional of exactly two marginals (a correlation range covers a sum, contrast, weighted sum or product); a mixture, a chain or more inputs need their own dependence information",
        ));
    }
    if request.reading != RequestedReading::Envelope {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "joint_law_propagation.envelope_is_not_a_point",
            request.reading.as_str(),
            "the dependence envelope is the set of values no dependence could exceed; it is not a point, an interval or a posterior and does not stand in for an aligned joint law; request the envelope reading, or supply the aligned joint law for a point value",
        ));
    }
    Ok(plan(request, route, None, None, shared))
}

/// Check a composition against the propagation rule.
///
/// # Errors
/// The first failed check as a coded structured refusal: an invalid request, a
/// closed interval or posterior reading, an envelope read as a point,
/// unknown dependence, shared evidence declared independent, a nonlinear
/// operation without an aligned joint law (index pairing of unrelated
/// marginals, mismatched alignment identities, a covariance where a joint law
/// is needed), a mixture without known weights, or a transported interval
/// without an aligned law or covariance.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_propagation(request: &PropagationRequest) -> Result<PropagationPlan, ExternalRefusal> {
    finish_propagation(request, check_core(request, None, None))
}

/// Check a composition against the propagation rule, deciding whether inputs
/// share evidence from the evidence ledger's row-level overlap rather than from
/// the equality of identity strings.
///
/// An input's `evidence_id` is read as a ledger source identity (its hex
/// spelling). Two inputs with the same identity share evidence (the forwarded
/// duplicate); two distinct ledger sources share evidence when the ledger finds
/// common observations (a subset, a resample, a derived source); the ledger
/// reporting the overlap as not computable, or an identity the ledger does not
/// hold, is unknown dependence and is never independence. A declared
/// independence of inputs whose overlap is unknown is refused rather than
/// accepted on distinct strings.
///
/// # Errors
/// As [`check_propagation`], and `sampling_dependence_unknown` with
/// `joint_law_propagation.evidence_overlap_unknown` when independence is
/// declared for inputs whose evidence overlap the ledger cannot decide.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_propagation_with_ledger(
    request: &PropagationRequest,
    ledger: &EvidenceLedger,
) -> Result<PropagationPlan, ExternalRefusal> {
    validate(request)?;
    let (shared, unresolved) = ledger_shared_evidence(&request.inputs, ledger);
    finish_propagation(request, check_core(request, Some(shared), unresolved))
}

/// Evidence identities that share observations, and the first pair whose
/// overlap cannot be decided, per the ledger.
fn ledger_shared_evidence(
    inputs: &[PropagationInput],
    ledger: &EvidenceLedger,
) -> (Vec<String>, Option<String>) {
    let mut shared = std::collections::BTreeMap::new();
    let mut unresolved: Option<String> = None;
    for (position, input) in inputs.iter().enumerate() {
        for other in &inputs[position + 1..] {
            let a = evidence_key(&input.evidence_id);
            let b = evidence_key(&other.evidence_id);
            if a == b {
                remember_shared(&mut shared, &input.evidence_id);
                remember_shared(&mut shared, &other.evidence_id);
                continue;
            }
            let ids = a.source().zip(b.source());
            let overlap = ids.and_then(|(a, b)| ledger.overlap(a, b).ok());
            match overlap {
                Some(Overlap::Independent) => {}
                Some(Overlap::SharedObservations { .. }) => {
                    remember_shared(&mut shared, &input.evidence_id);
                    remember_shared(&mut shared, &other.evidence_id);
                }
                Some(Overlap::Unknown { .. }) | None => {
                    unresolved.get_or_insert_with(|| format!("{}|{}", input.id, other.id));
                }
            }
        }
    }
    let mut labels: Vec<_> = shared.into_values().map(str::to_owned).collect();
    labels.sort();
    (labels, unresolved)
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn finish_propagation(
    request: &PropagationRequest,
    core: Result<PropagationPlan, ExternalRefusal>,
) -> Result<PropagationPlan, ExternalRefusal> {
    let mut plan = core?;
    plan.missing_transport_pieces = missing_pieces(&request.inputs);
    if request.reading == RequestedReading::Spread {
        if let Some(first) = plan.missing_transport_pieces.first() {
            return Err(refuse(
                antecedent_core::reason_code!("joint_law_required"),
                "joint_law_propagation.transport_pieces_incomplete",
                *first,
                "the spread of a transported claim depends on source sampling, target sampling and the transport functional together; name all three pieces with their dependence (one alignment identity or a covariance), or ask for the point value",
            ));
        }
        plan.label = UNCALIBRATED_SPREAD_LABEL;
    }
    Ok(plan)
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_core(
    request: &PropagationRequest,
    ledger_shared: Option<Vec<String>>,
    unresolved: Option<String>,
) -> Result<PropagationPlan, ExternalRefusal> {
    validate(request)?;
    if let Some(refusal) = closed_reading(request.reading) {
        return Err(refusal);
    }
    let shared = ledger_shared.unwrap_or_else(|| shared_evidence(&request.inputs));
    let envelope_dependence = matches!(
        request.dependence,
        PropagationDependence::CorrelationRange { .. } | PropagationDependence::MarginalsOnly
    );
    if request.reading == RequestedReading::Spread && envelope_dependence {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "joint_law_propagation.spread_needs_licensed_dependence",
            "spread",
            "a spread of a composed quantity is determined by a covariance, an aligned joint law, known mixture weights or a declared independence of distinct evidence; a correlation range or the marginals alone give only the labelled envelope, so request the envelope reading",
        ));
    }
    if request.reading == RequestedReading::Envelope && !envelope_dependence {
        return Err(invalid(
            "reading",
            "an envelope reading needs a declared correlation range or the marginals-only declaration; with a licensed dependence route ask for the point value",
        ));
    }
    let transported = request.operation == PropagationOperation::TransportedComposition
        || request.inputs.iter().any(|input| input.role.is_transported());
    if transported {
        return check_transported(request, shared);
    }
    if envelope_dependence {
        return check_envelope(request, shared);
    }
    match request.operation.class() {
        PropagationClass::Mixture => match request.dependence {
            PropagationDependence::KnownWeights => {
                Ok(plan(request, PropagationRoute::KnownWeightMixture, None, None, shared))
            }
            _ => Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "joint_law_propagation.mixture_weights_unknown",
                request.operation.as_str(),
                "declare the mixture weights: a mixture's mean and variance need known weights and the component moments, and no joint law of the components",
            )),
        },
        PropagationClass::Linear => check_linear(request, shared, unresolved),
        PropagationClass::Nonlinear => check_nonlinear(request, shared),
        PropagationClass::Transported => check_transported(request, shared),
    }
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_linear(
    request: &PropagationRequest,
    shared: Vec<String>,
    unresolved: Option<String>,
) -> Result<PropagationPlan, ExternalRefusal> {
    if request.inputs.len() == 1 {
        return Ok(plan(request, PropagationRoute::LinearMoments, None, None, shared));
    }
    match &request.dependence {
        PropagationDependence::DeclaredIndependent => {
            if let Some(evidence) = shared.first() {
                return Err(refuse(
                    antecedent_core::reason_code!("evidence_double_counted"),
                    "joint_law_propagation.shared_evidence_declared_independent",
                    evidence.clone(),
                    "inputs that rest on the same evidence are not independent; supply the covariance between them (the shared evidence supplies it) or an aligned joint law",
                ));
            }
            if let Some(pair) = unresolved {
                return Err(refuse(
                    antecedent_core::reason_code!("sampling_dependence_unknown"),
                    "joint_law_propagation.evidence_overlap_unknown",
                    pair,
                    "the evidence ledger cannot decide whether these inputs share observations (an unregistered source, or a source with undeclared observations); register both sources with their observations, or supply the covariance or an aligned joint law",
                ));
            }
            Ok(plan(request, PropagationRoute::LinearMoments, None, None, shared))
        }
        PropagationDependence::Covariance { id } => {
            Ok(plan(request, PropagationRoute::LinearMoments, None, Some(id.clone()), shared))
        }
        PropagationDependence::AlignedJointLaw => {
            let alignment = common_alignment(&request.inputs)?;
            Ok(plan(request, PropagationRoute::AlignedJointLaw, Some(alignment), None, shared))
        }
        PropagationDependence::Unknown
        | PropagationDependence::KnownWeights
        | PropagationDependence::CorrelationRange { .. }
        | PropagationDependence::MarginalsOnly => Err(dependence_unknown(request.operation)),
    }
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
fn check_nonlinear(
    request: &PropagationRequest,
    shared: Vec<String>,
) -> Result<PropagationPlan, ExternalRefusal> {
    let joint = antecedent_core::reason_code!("joint_law_required");
    match &request.dependence {
        PropagationDependence::AlignedJointLaw => {
            let alignment = common_alignment(&request.inputs)?;
            Ok(plan(request, PropagationRoute::AlignedJointLaw, Some(alignment), None, shared))
        }
        PropagationDependence::Covariance { .. } => Err(refuse(
            joint,
            "joint_law_propagation.covariance_insufficient_for_nonlinear",
            request.operation.as_str(),
            "a product, a nonlinear functional or a chained transformation is an expectation under the joint law, which a covariance does not determine; supply the aligned draws of one joint law, or ask for the dependence envelope",
        )),
        PropagationDependence::DeclaredIndependent => Err(refuse(
            joint,
            "joint_law_propagation.nonlinear_needs_aligned_joint_law",
            request.operation.as_str(),
            "a nonlinear operation needs the joint law as aligned draws (one alignment identity for every input); a declared independence is not a joint law of draws, and the draws of independent marginals are never paired by index",
        )),
        PropagationDependence::Unknown | PropagationDependence::KnownWeights => {
            if request
                .inputs
                .iter()
                .any(|input| matches!(input.alignment, InputAlignment::IndependentMarginals))
            {
                // The inputs are independently produced draws: say so rather
                // than report an undeclared dependence.
                common_alignment(&request.inputs)?;
            }
            Err(dependence_unknown(request.operation))
        }
        PropagationDependence::CorrelationRange { .. } | PropagationDependence::MarginalsOnly => {
            Err(dependence_unknown(request.operation))
        }
    }
}

/// The coordinate at which a composed interval's coverage is a property: the
/// operation, the route, whether any input is transported and the sorted
/// multiset of input roles. A coverage record is about one coordinate and says
/// nothing about another; a component interval's record is never a composed
/// coordinate's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposedCoordinate {
    /// The operation.
    pub operation: PropagationOperation,
    /// The licensed route.
    pub route: PropagationRoute,
    /// Sorted role names of the inputs (one entry per input).
    pub roles: Vec<&'static str>,
    /// Sorted distinct horizons of the one temporal claim whose horizons the inputs are; empty
    /// when the inputs are not horizons of a single claim. A record measured over one set of
    /// horizons says nothing about another.
    pub horizons: Vec<u32>,
}

impl ComposedCoordinate {
    /// The coordinate of a checked composition.
    #[must_use]
    pub fn of(request: &PropagationRequest, plan: &PropagationPlan) -> Self {
        let mut roles: Vec<&'static str> =
            request.inputs.iter().map(|input| input.role.as_str()).collect();
        roles.sort_unstable();
        Self { operation: plan.operation, route: plan.route, roles, horizons: Vec::new() }
    }

    /// The coordinate over the horizons of one temporal claim.
    #[must_use]
    pub fn over_horizons(mut self, horizons: &[u32]) -> Self {
        let mut horizons = horizons.to_vec();
        horizons.sort_unstable();
        horizons.dedup();
        self.horizons = horizons;
        self
    }

    /// Stable identity string: `operation|route|role,role,...`, followed by
    /// `|horizons=h,h,...` when the inputs are horizons of one temporal claim. A coordinate
    /// without horizons keeps the identity it always had.
    #[must_use]
    pub fn id(&self) -> String {
        let base =
            format!("{}|{}|{}", self.operation.as_str(), self.route.as_str(), self.roles.join(","));
        if self.horizons.is_empty() {
            return base;
        }
        let horizons: Vec<String> = self.horizons.iter().map(u32::to_string).collect();
        format!("{base}|horizons={}", horizons.join(","))
    }
}

/// The inputs of a composition across the horizons of one temporal claim: one per horizon,
/// all resting on the same evidence and all rows of the one joint law `alignment_id` names (a
/// replicate or a draw index that refit every horizon together).
///
/// Horizons fit one by one are independent marginals, not these: their draws carry no common
/// alignment identity, and [`check_horizon_composition`] refuses them.
#[must_use]
pub fn horizon_inputs(
    horizons: &[u32],
    evidence_id: &str,
    alignment_id: &str,
) -> Vec<PropagationInput> {
    horizons
        .iter()
        .map(|horizon| PropagationInput {
            id: format!("horizon:{horizon}"),
            evidence_id: evidence_id.to_owned(),
            role: InputRole::Estimate,
            alignment: InputAlignment::Aligned { alignment_id: alignment_id.to_owned() },
        })
        .collect()
}

/// Check a composition across the horizons of one temporal claim and name its coordinate.
///
/// The inputs are the claim's horizons, coupled by the joint law `alignment_id` (see
/// [`horizon_inputs`]); the dependence is that aligned joint law. The result is the checked
/// plan and the coordinate, carrying the horizon axis.
///
/// # Errors
/// No horizons, repeated horizons, a blank evidence or alignment identity, or a request the
/// propagation rule refuses (for example an operation that needs more inputs than horizons).
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_horizon_composition(
    operation: PropagationOperation,
    horizons: &[u32],
    evidence_id: &str,
    alignment_id: &str,
    reading: RequestedReading,
) -> Result<(PropagationPlan, ComposedCoordinate), ExternalRefusal> {
    let mut sorted = horizons.to_vec();
    sorted.sort_unstable();
    let before = sorted.len();
    sorted.dedup();
    if sorted.is_empty() || sorted.len() != before {
        return Err(invalid("horizons", "name each horizon of the claim once, and at least one"));
    }
    let request = PropagationRequest {
        operation,
        inputs: horizon_inputs(&sorted, evidence_id, alignment_id),
        dependence: PropagationDependence::AlignedJointLaw,
        reading,
    };
    let plan = check_propagation(&request)?;
    let coordinate = ComposedCoordinate::of(&request, &plan).over_horizons(&sorted);
    Ok((plan, coordinate))
}

/// A coverage record offered for a composed coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct CoverageRecord {
    /// Identity of the record (the attested measurement).
    pub record_id: String,
    /// The [`ComposedCoordinate::id`] the record was measured at.
    pub coordinate_id: String,
    /// The interval construction the record was measured for (a percentile
    /// interval of aligned draws, a normal approximation to the propagated
    /// variance, ...). A record for one construction is not another's.
    pub construction: String,
    /// The nominal level the record was measured at.
    pub nominal_level: f64,
    /// Whether the record is a completed, attested measurement (as opposed to
    /// a wired but unmeasured placeholder).
    pub measured: bool,
}

/// What a passed composed-interval check records.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedIntervalPlan {
    /// The propagation plan of the composition (with its dependence route).
    pub propagation: PropagationPlan,
    /// The coordinate the coverage record was matched against.
    pub coordinate: ComposedCoordinate,
    /// The record that licensed the interval.
    pub record_id: String,
    /// The construction the interval must be built with.
    pub construction: String,
    /// The nominal level.
    pub nominal_level: f64,
    /// The label of the result.
    pub label: &'static str,
}

/// Check a composed interval: the composition's dependence route must license
/// its spread, and a measured coverage record must exist for exactly this
/// composed coordinate and construction.
///
/// Without a record the interval reading is closed (the component intervals do
/// not cover the composition). With one, the interval is licensed only at the
/// recorded nominal level; the record, not this function, carries the coverage
/// evidence.
///
/// # Errors
/// An invalid request, an unlicensed dependence route (the refusals of
/// [`check_propagation`] under the spread reading), a request whose reading is
/// not an interval, a missing, unmeasured or mismatched coverage record, or a
/// nominal level outside `(0, 1)`.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_composed_interval(
    request: &PropagationRequest,
    record: Option<&CoverageRecord>,
    construction: &str,
) -> Result<ComposedIntervalPlan, ExternalRefusal> {
    if request.reading != RequestedReading::Interval {
        return Err(invalid(
            request.reading.as_str(),
            "ask for the interval reading, or use check_propagation for the point, spread or envelope",
        ));
    }
    if blank(construction) {
        return Err(invalid("construction", "name the interval construction the record covers"));
    }
    let spread = PropagationRequest { reading: RequestedReading::Spread, ..request.clone() };
    let propagation = check_propagation(&spread)?;
    let coordinate = ComposedCoordinate::of(request, &propagation);
    let closed = || {
        closed_reading(RequestedReading::Interval).unwrap_or_else(|| {
            invalid("interval", "ask for the point or spread reading of the composition")
        })
    };
    let Some(record) = record else {
        return Err(closed());
    };
    let code = antecedent_core::reason_code!("cell_not_licensed");
    if !record.measured {
        return Err(refuse(
            code,
            "joint_law_propagation.coverage_record_unmeasured",
            record.record_id.clone(),
            "the coverage record is wired but not measured; run the whole-method coverage measurement at the composed coordinate and attest it, or report the labelled spread",
        ));
    }
    if record.coordinate_id != coordinate.id() {
        return Err(refuse(
            code,
            "joint_law_propagation.coverage_record_coordinate_mismatch",
            record.record_id.clone(),
            "a coverage record is about one composed coordinate (operation, route and input roles); measure coverage at this coordinate, or report the labelled spread",
        )
        .with_expected(coordinate.id())
        .with_supplied(record.coordinate_id.clone()));
    }
    if record.construction != construction {
        return Err(refuse(
            code,
            "joint_law_propagation.coverage_record_construction_mismatch",
            record.record_id.clone(),
            "a coverage record is about one interval construction; build the interval with the construction the record was measured for",
        )
        .with_expected(record.construction.clone())
        .with_supplied(construction.to_owned()));
    }
    if !(record.nominal_level.is_finite()
        && 0.0 < record.nominal_level
        && record.nominal_level < 1.0)
    {
        return Err(invalid(
            "nominal_level",
            "the record's nominal level is a probability strictly between zero and one",
        ));
    }
    Ok(ComposedIntervalPlan {
        propagation,
        coordinate,
        record_id: record.record_id.clone(),
        construction: construction.to_owned(),
        nominal_level: record.nominal_level,
        label: MEASURED_INTERVAL_LABEL,
    })
}

/// Whether one input's rows may enter a row pairing at all.
///
/// This is the single-input form of the rule for a nonlinear operation under an
/// aligned joint law: the input is checked against a partner that is aligned
/// with it by declaration, so only the input's own alignment decides. It cannot
/// detect two joint artifacts whose alignment identities differ; where several
/// inputs are paired use [`check_rows_pairable_across`] with their real
/// alignment identities. Existing
/// callers that validate pairing relations separately (shared evidence, declared
/// dependence routes) use this so that no code outside this module decides that
/// draws are row-paired.
///
/// # Errors
/// `joint_law_required` with `joint_law_propagation.index_pairing_refused` when
/// the input's draws are independent marginals or a summary.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_rows_pairable(alignment: &InputAlignment) -> Result<(), ExternalRefusal> {
    let shared = InputAlignment::Aligned { alignment_id: "declared_pairing".to_owned() };
    let request = PropagationRequest {
        operation: PropagationOperation::NonlinearFunctional,
        inputs: vec![
            PropagationInput {
                id: "pairing_input".to_owned(),
                evidence_id: "pairing_evidence_input".to_owned(),
                role: InputRole::Estimate,
                alignment: alignment.clone(),
            },
            PropagationInput {
                id: "pairing_partner".to_owned(),
                evidence_id: "pairing_evidence_partner".to_owned(),
                role: InputRole::Estimate,
                alignment: shared,
            },
        ],
        dependence: PropagationDependence::AlignedJointLaw,
        reading: RequestedReading::Point,
    };
    check_propagation(&request).map(|_| ())
}

/// Whether the rows of several inputs may be paired row by row.
///
/// The multi-input form of the rule for a nonlinear operation under an aligned
/// joint law: the inputs' own alignments decide, so two inputs flagged joint
/// with different alignment identities (separate artifacts, other draws) refuse
/// as `alignment_mismatch`, and an independent-marginal or summary input
/// refuses as `index_pairing_refused`. A covariance or a declared independence
/// is never a route to row pairing. Each entry is an input identity and its
/// alignment; the evidence identities are distinct placeholders because only
/// the alignment is decided here.
///
/// With fewer than two inputs it is the single-input rule,
/// [`check_rows_pairable`].
///
/// # Errors
/// `joint_law_required` with `joint_law_propagation.alignment_mismatch` for
/// different alignment identities, or
/// `joint_law_propagation.index_pairing_refused` for an input with no aligned
/// draws.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn check_rows_pairable_across(
    inputs: &[(&str, &InputAlignment)],
) -> Result<(), ExternalRefusal> {
    match inputs {
        [] => Ok(()),
        [(_, only)] => check_rows_pairable(only),
        many => {
            let request = PropagationRequest {
                operation: PropagationOperation::NonlinearFunctional,
                inputs: many
                    .iter()
                    .map(|(id, alignment)| PropagationInput {
                        id: (*id).to_owned(),
                        evidence_id: format!("pairing_evidence:{id}"),
                        role: InputRole::Estimate,
                        alignment: (*alignment).clone(),
                    })
                    .collect(),
                dependence: PropagationDependence::AlignedJointLaw,
                reading: RequestedReading::Point,
            };
            check_propagation(&request).map(|_| ())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(id: &str, evidence: &str, alignment: InputAlignment) -> PropagationInput {
        PropagationInput {
            id: id.to_owned(),
            evidence_id: evidence.to_owned(),
            role: InputRole::Estimate,
            alignment,
        }
    }

    fn aligned(id: &str, evidence: &str, alignment: &str) -> PropagationInput {
        input(id, evidence, InputAlignment::Aligned { alignment_id: alignment.to_owned() })
    }

    fn request(
        operation: PropagationOperation,
        inputs: Vec<PropagationInput>,
        dependence: PropagationDependence,
        reading: RequestedReading,
    ) -> PropagationRequest {
        PropagationRequest { operation, inputs, dependence, reading }
    }

    fn detail(request: &PropagationRequest) -> (&'static str, String) {
        let refusal = check_propagation(request).expect_err("refusal");
        refusal.validate().expect("well-formed refusal");
        assert!(refusal.remedy.is_some(), "every refusal states what would license it");
        (refusal.code, refusal.detail)
    }

    #[test]
    fn horizons_of_one_joint_law_compose_and_independent_marginals_do_not() {
        let (plan, coordinate) = check_horizon_composition(
            PropagationOperation::NonlinearFunctional,
            &[3, 1, 2],
            "series_snapshot",
            "replicates:7",
            RequestedReading::Spread,
        )
        .expect("horizons coupled by one joint law compose");
        assert_eq!(plan.route, PropagationRoute::AlignedJointLaw);
        assert_eq!(plan.alignment_id.as_deref(), Some("replicates:7"));
        assert_eq!(coordinate.horizons, vec![1, 2, 3]);
        assert!(coordinate.id().ends_with("|horizons=1,2,3"), "{}", coordinate.id());
        // A coordinate without the axis keeps its identity, and a different horizon set is a
        // different coordinate.
        let plain = ComposedCoordinate { horizons: Vec::new(), ..coordinate.clone() };
        assert!(!plain.id().contains("horizons"));
        assert_ne!(coordinate.id(), coordinate.clone().over_horizons(&[1, 2]).id());

        // Horizons fit one by one carry no common alignment: pairing them is refused.
        let marginals = request(
            PropagationOperation::NonlinearFunctional,
            (1..=3)
                .map(|h| {
                    input(
                        &format!("horizon:{h}"),
                        "series_snapshot",
                        InputAlignment::IndependentMarginals,
                    )
                })
                .collect(),
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Spread,
        );
        assert!(check_propagation(&marginals).is_err());
        let declared_independent = request(
            PropagationOperation::Sum,
            (1..=3)
                .map(|h| {
                    input(
                        &format!("horizon:{h}"),
                        "series_snapshot",
                        InputAlignment::IndependentMarginals,
                    )
                })
                .collect(),
            PropagationDependence::DeclaredIndependent,
            RequestedReading::Spread,
        );
        let (code, _) = detail(&declared_independent);
        assert_eq!(code, "evidence_double_counted", "one series' horizons are not independent");

        for bad in [&[][..], &[1, 1][..]] {
            let error = check_horizon_composition(
                PropagationOperation::Sum,
                bad,
                "e",
                "a",
                RequestedReading::Point,
            )
            .expect_err("a horizon set names each horizon once");
            error.validate().expect("well-formed refusal");
        }
    }

    #[test]
    fn linear_operations_take_a_covariance_or_distinct_evidence() {
        let shared = vec![
            input("effect_a", "snapshot_1", InputAlignment::MomentsOnly),
            input("effect_b", "snapshot_1", InputAlignment::MomentsOnly),
        ];
        let with_covariance = request(
            PropagationOperation::Contrast,
            shared.clone(),
            PropagationDependence::Covariance { id: "cov_ab".into() },
            RequestedReading::Point,
        );
        let plan = check_propagation(&with_covariance).expect("covariance licenses a contrast");
        assert_eq!(plan.route, PropagationRoute::LinearMoments);
        assert_eq!(plan.shared_evidence, vec!["snapshot_1".to_owned()]);
        assert_eq!(plan.label, "point_only");

        let independent_but_shared = request(
            PropagationOperation::Sum,
            shared,
            PropagationDependence::DeclaredIndependent,
            RequestedReading::Point,
        );
        let (code, reason) = detail(&independent_but_shared);
        assert_eq!(code, "evidence_double_counted");
        assert_eq!(reason, "joint_law_propagation.shared_evidence_declared_independent");

        let distinct = request(
            PropagationOperation::WeightedSum,
            vec![
                input("a", "snapshot_1", InputAlignment::MomentsOnly),
                input("b", "snapshot_2", InputAlignment::MomentsOnly),
            ],
            PropagationDependence::DeclaredIndependent,
            RequestedReading::Point,
        );
        assert!(check_propagation(&distinct).is_ok());

        let unknown = request(
            PropagationOperation::Sum,
            distinct.inputs.clone(),
            PropagationDependence::Unknown,
            RequestedReading::Point,
        );
        assert_eq!(detail(&unknown).0, "sampling_dependence_unknown");
    }

    #[test]
    fn evidence_aliases_cannot_license_independence_and_reports_are_stable() {
        let lower = "ab".repeat(32);
        let upper = lower.to_uppercase();
        for aliases in [
            vec!["snapshot_1".to_owned(), " snapshot_1 ".to_owned()],
            vec![lower.clone(), upper.clone(), format!(" {upper} ")],
        ] {
            let inputs: Vec<_> = aliases
                .iter()
                .enumerate()
                .map(|(i, label)| input(&format!("input:{i}"), label, InputAlignment::MomentsOnly))
                .collect();
            let independent = request(
                PropagationOperation::Sum,
                inputs.clone(),
                PropagationDependence::DeclaredIndependent,
                RequestedReading::Point,
            );
            assert_eq!(
                detail(&independent).1,
                "joint_law_propagation.shared_evidence_declared_independent"
            );
            let ledger = EvidenceLedger::new();
            let refusal = check_propagation_with_ledger(&independent, &ledger).unwrap_err();
            assert_eq!(
                refusal.detail,
                "joint_law_propagation.shared_evidence_declared_independent"
            );
            let mut covariance = request(
                PropagationOperation::Sum,
                inputs,
                PropagationDependence::Covariance { id: "cov".into() },
                RequestedReading::Point,
            );
            let expected = vec![aliases.iter().min().unwrap().clone()];
            assert_eq!(check_propagation(&covariance).unwrap().shared_evidence, expected);
            covariance.inputs.reverse();
            assert_eq!(check_propagation(&covariance).unwrap().shared_evidence, expected);
            assert_eq!(
                check_propagation_with_ledger(&covariance, &ledger).unwrap().shared_evidence,
                expected
            );
        }
        for (left, right) in [("Site", "site"), ("site x", "site  x"), ("deadbeef", "DEADBEEF")] {
            let distinct = request(
                PropagationOperation::Sum,
                vec![
                    input("a", left, InputAlignment::MomentsOnly),
                    input("b", right, InputAlignment::MomentsOnly),
                ],
                PropagationDependence::DeclaredIndependent,
                RequestedReading::Point,
            );
            assert!(
                check_propagation(&distinct).is_ok(),
                "opaque names remain case and internal-space sensitive"
            );
            assert!(
                check_propagation_with_ledger(&distinct, &EvidenceLedger::new()).is_err(),
                "unregistered sources remain unknown to a ledger"
            );
        }
    }

    #[test]
    fn nonlinear_operations_need_one_alignment_identity() {
        let paired = request(
            PropagationOperation::Product,
            vec![aligned("x", "s", "joint_17"), aligned("y", "s", "joint_17")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        let plan = check_propagation(&paired).expect("aligned draws license a product");
        assert_eq!(plan.route, PropagationRoute::AlignedJointLaw);
        assert_eq!(plan.alignment_id.as_deref(), Some("joint_17"));

        // Two joint laws with different identities are not row-paired.
        let mismatched = request(
            PropagationOperation::NonlinearFunctional,
            vec![aligned("x", "s1", "joint_17"), aligned("y", "s2", "joint_99")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        assert_eq!(detail(&mismatched).1, "joint_law_propagation.alignment_mismatch");

        // Unrelated marginal draws asked to be paired by index.
        let index_paired = request(
            PropagationOperation::Product,
            vec![
                input("x", "s1", InputAlignment::IndependentMarginals),
                input("y", "s2", InputAlignment::IndependentMarginals),
            ],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        let (code, name) = detail(&index_paired);
        assert_eq!(code, "joint_law_required");
        assert_eq!(name, "joint_law_propagation.index_pairing_refused");

        // The same inputs with nothing declared name the index pairing too.
        let undeclared = request(
            PropagationOperation::ChainedSampling,
            index_paired.inputs.clone(),
            PropagationDependence::Unknown,
            RequestedReading::Point,
        );
        assert_eq!(detail(&undeclared).1, "joint_law_propagation.index_pairing_refused");

        // A covariance does not determine a nonlinear functional.
        let covariance_only = request(
            PropagationOperation::Product,
            index_paired.inputs.clone(),
            PropagationDependence::Covariance { id: "cov".into() },
            RequestedReading::Point,
        );
        assert_eq!(
            detail(&covariance_only).1,
            "joint_law_propagation.covariance_insufficient_for_nonlinear"
        );

        let summaries = request(
            PropagationOperation::Product,
            vec![
                input("x", "s1", InputAlignment::MomentsOnly),
                input("y", "s2", InputAlignment::MomentsOnly),
            ],
            PropagationDependence::Unknown,
            RequestedReading::Point,
        );
        assert_eq!(detail(&summaries).0, "sampling_dependence_unknown");
    }

    #[test]
    fn mixtures_need_known_weights_and_no_joint_law() {
        let comps = vec![
            input("a", "s1", InputAlignment::IndependentMarginals),
            input("b", "s1", InputAlignment::IndependentMarginals),
        ];
        let known = request(
            PropagationOperation::Mixture,
            comps.clone(),
            PropagationDependence::KnownWeights,
            RequestedReading::Point,
        );
        assert_eq!(
            check_propagation(&known).expect("mixture").route,
            PropagationRoute::KnownWeightMixture
        );
        let unknown = request(
            PropagationOperation::Mixture,
            comps,
            PropagationDependence::Unknown,
            RequestedReading::Point,
        );
        let (code, name) = detail(&unknown);
        assert_eq!(code, "invalid_argument");
        assert_eq!(name, "joint_law_propagation.mixture_weights_unknown");
    }

    #[test]
    fn the_envelope_is_labelled_and_never_a_point() {
        let marginals = vec![
            input("x", "s1", InputAlignment::MomentsOnly),
            input("y", "s2", InputAlignment::MomentsOnly),
        ];
        let ok = request(
            PropagationOperation::Sum,
            marginals.clone(),
            PropagationDependence::MarginalsOnly,
            RequestedReading::Envelope,
        );
        let plan = check_propagation(&ok).expect("envelope");
        assert_eq!(plan.route, PropagationRoute::FrechetEnvelope);
        assert_eq!(plan.label, "structural_envelope");
        let range = request(
            PropagationOperation::Product,
            marginals.clone(),
            PropagationDependence::CorrelationRange { rho_lo: -0.2, rho_hi: 0.6 },
            RequestedReading::Envelope,
        );
        assert_eq!(check_propagation(&range).expect("range").label, "assumption_range");
        for (rho_lo, rho_hi) in [(-1.1, 0.5), (-0.5, 1.1), (0.5, 0.5), (f64::NAN, 0.5)] {
            let malformed = request(
                PropagationOperation::Product,
                marginals.clone(),
                PropagationDependence::CorrelationRange { rho_lo, rho_hi },
                RequestedReading::Envelope,
            );
            let (code, reason) = detail(&malformed);
            assert_eq!(code, "invalid_argument");
            assert_eq!(reason, "joint_law_propagation.invalid_request");
        }
        let as_point = request(
            PropagationOperation::Sum,
            marginals.clone(),
            PropagationDependence::MarginalsOnly,
            RequestedReading::Point,
        );
        let (code, name) = detail(&as_point);
        assert_eq!(code, "cell_not_licensed");
        assert_eq!(name, "joint_law_propagation.envelope_is_not_a_point");
        let mixture = request(
            PropagationOperation::Mixture,
            marginals,
            PropagationDependence::MarginalsOnly,
            RequestedReading::Envelope,
        );
        let (code, name) = detail(&mixture);
        assert_eq!(code, "route_not_supported");
        assert_eq!(name, "joint_law_propagation.envelope_not_available");
    }

    #[test]
    fn interval_and_posterior_readings_are_closed() {
        for (reading, expected) in [
            (RequestedReading::Interval, "joint_law_propagation.interval_route_closed"),
            (RequestedReading::Posterior, "joint_law_propagation.posterior_route_closed"),
        ] {
            let closed = request(
                PropagationOperation::Sum,
                vec![aligned("x", "s", "j"), aligned("y", "s", "j")],
                PropagationDependence::AlignedJointLaw,
                reading,
            );
            let (code, name) = detail(&closed);
            assert_eq!(code, "cell_not_licensed");
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn a_transported_interval_needs_an_aligned_law_or_a_covariance() {
        let mut transported = input("transported", "s_source", InputAlignment::MomentsOnly);
        transported.role = InputRole::TransportedInterval;
        let other = input("other_claim", "s_target", InputAlignment::MomentsOnly);
        for dependence in [
            PropagationDependence::Unknown,
            PropagationDependence::DeclaredIndependent,
            PropagationDependence::MarginalsOnly,
        ] {
            let composed = request(
                PropagationOperation::Sum,
                vec![transported.clone(), other.clone()],
                dependence,
                RequestedReading::Point,
            );
            let reading = if composed.dependence == PropagationDependence::MarginalsOnly {
                RequestedReading::Envelope
            } else {
                RequestedReading::Point
            };
            let composed = PropagationRequest { reading, ..composed };
            let (code, name) = detail(&composed);
            assert_eq!(code, "joint_law_required");
            assert_eq!(name, "joint_law_propagation.transported_dependence_required");
        }
        let with_covariance = request(
            PropagationOperation::Sum,
            vec![transported.clone(), other.clone()],
            PropagationDependence::Covariance { id: "transport_cov".into() },
            RequestedReading::Point,
        );
        assert!(check_propagation(&with_covariance).is_ok());
        let mut aligned_t = aligned("transported", "s_source", "joint_3");
        aligned_t.role = InputRole::SourceSampling;
        let pieces = request(
            PropagationOperation::TransportedComposition,
            vec![aligned_t, aligned("target", "s_source", "joint_3")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        let plan = check_propagation(&pieces).expect("aligned pieces");
        assert_eq!(plan.shared_evidence, vec!["s_source".to_owned()]);
    }

    #[test]
    fn malformed_requests_are_refused_before_the_rule() {
        let single = request(
            PropagationOperation::Sum,
            vec![aligned("x", "s", "j")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        assert_eq!(detail(&single).1, "joint_law_propagation.invalid_request");
        let duplicate = request(
            PropagationOperation::Sum,
            vec![aligned("x", "s", "j"), aligned("x", "t", "j")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        );
        assert_eq!(detail(&duplicate).1, "joint_law_propagation.invalid_request");
        let blank_cov = request(
            PropagationOperation::Sum,
            vec![aligned("x", "s", "j"), aligned("y", "t", "j")],
            PropagationDependence::Covariance { id: " ".into() },
            RequestedReading::Point,
        );
        assert_eq!(detail(&blank_cov).1, "joint_law_propagation.invalid_request");
    }

    #[test]
    fn a_spread_is_labelled_and_needs_the_same_dependence_as_the_point() {
        let shared = vec![
            input("a", "snapshot_1", InputAlignment::MomentsOnly),
            input("b", "snapshot_1", InputAlignment::MomentsOnly),
        ];
        let with_covariance = request(
            PropagationOperation::Contrast,
            shared.clone(),
            PropagationDependence::Covariance { id: "cov_ab".into() },
            RequestedReading::Spread,
        );
        let plan = check_propagation(&with_covariance).expect("spread under a covariance");
        assert_eq!(plan.route, PropagationRoute::LinearMoments);
        assert_eq!(plan.label, "uncalibrated_spread");
        assert!(plan.missing_transport_pieces.is_empty());

        // The point reading of the same composition keeps its own label.
        let point = PropagationRequest { reading: RequestedReading::Point, ..with_covariance };
        assert_eq!(check_propagation(&point).expect("point").label, "point_only");

        // Shared evidence declared independent: no spread either.
        let independent_but_shared = request(
            PropagationOperation::Sum,
            shared,
            PropagationDependence::DeclaredIndependent,
            RequestedReading::Spread,
        );
        assert_eq!(detail(&independent_but_shared).0, "evidence_double_counted");

        // A nonlinear spread needs the aligned joint law; the unpaired marginals
        // are refused, never paired by index.
        let unpaired = request(
            PropagationOperation::Product,
            vec![
                input("x", "s1", InputAlignment::IndependentMarginals),
                input("y", "s2", InputAlignment::IndependentMarginals),
            ],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Spread,
        );
        assert_eq!(detail(&unpaired).1, "joint_law_propagation.index_pairing_refused");

        // An envelope dependence is not a spread.
        let envelope = request(
            PropagationOperation::Sum,
            vec![
                input("x", "s1", InputAlignment::MomentsOnly),
                input("y", "s2", InputAlignment::MomentsOnly),
            ],
            PropagationDependence::MarginalsOnly,
            RequestedReading::Spread,
        );
        let (code, name) = detail(&envelope);
        assert_eq!(code, "cell_not_licensed");
        assert_eq!(name, "joint_law_propagation.spread_needs_licensed_dependence");
    }

    #[test]
    fn transported_pieces_are_dependent_and_a_spread_needs_all_three() {
        let piece = |id: &str, role: InputRole, alignment: &str| PropagationInput {
            id: id.to_owned(),
            evidence_id: "s_source".to_owned(),
            role,
            alignment: InputAlignment::Aligned { alignment_id: alignment.to_owned() },
        };
        let two = vec![
            piece("source", InputRole::SourceSampling, "joint_9"),
            piece("target", InputRole::TargetSampling, "joint_9"),
        ];
        let incomplete = request(
            PropagationOperation::TransportedComposition,
            two.clone(),
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Spread,
        );
        let refusal = check_propagation(&incomplete).expect_err("incomplete");
        refusal.validate().expect("well-formed refusal");
        assert_eq!(refusal.code, "joint_law_required");
        assert_eq!(refusal.detail, "joint_law_propagation.transport_pieces_incomplete");
        assert_eq!(refusal.offending.as_deref(), Some("transport_functional"));
        // The point value of the same pieces is not a spread claim.
        let point = PropagationRequest { reading: RequestedReading::Point, ..incomplete.clone() };
        assert_eq!(
            check_propagation(&point).expect("point").missing_transport_pieces,
            vec!["transport_functional"]
        );

        let mut three = two;
        three.push(piece("functional", InputRole::TransportFunctional, "joint_9"));
        let complete = PropagationRequest { inputs: three.clone(), ..incomplete };
        let plan = check_propagation(&complete).expect("complete pieces");
        assert_eq!(plan.label, "uncalibrated_spread");
        assert_eq!(plan.alignment_id.as_deref(), Some("joint_9"));

        // Pieces whose draws carry different alignment identities are not paired.
        three[2] = piece("functional", InputRole::TransportFunctional, "joint_10");
        let unpaired = PropagationRequest { inputs: three, ..complete.clone() };
        assert_eq!(detail(&unpaired).1, "joint_law_propagation.alignment_mismatch");

        // Independence is never assumed for the pieces.
        let independent = PropagationRequest {
            dependence: PropagationDependence::DeclaredIndependent,
            ..complete
        };
        assert_eq!(detail(&independent).1, "joint_law_propagation.transported_dependence_required");
    }

    #[test]
    fn a_single_transported_claim_and_a_known_weight_mixture_need_no_cross_term() {
        let mut transported = input("transported", "s_source", InputAlignment::MomentsOnly);
        transported.role = InputRole::TransportedInterval;
        let transform = request(
            PropagationOperation::LinearTransform,
            vec![transported.clone()],
            PropagationDependence::Unknown,
            RequestedReading::Spread,
        );
        let plan = check_propagation(&transform).expect("one claim, scaled");
        assert_eq!(plan.route, PropagationRoute::LinearMoments);

        let mut second = input("other", "s_other", InputAlignment::MomentsOnly);
        second.role = InputRole::TransportedInterval;
        let mixture = request(
            PropagationOperation::Mixture,
            vec![transported.clone(), second],
            PropagationDependence::KnownWeights,
            RequestedReading::Spread,
        );
        assert_eq!(
            check_propagation(&mixture).expect("mixture").route,
            PropagationRoute::KnownWeightMixture
        );

        // Composition with another claim is still refused without dependence.
        let other = input("claim", "s_target", InputAlignment::MomentsOnly);
        let sum = request(
            PropagationOperation::Sum,
            vec![transported, other],
            PropagationDependence::Unknown,
            RequestedReading::Spread,
        );
        assert_eq!(detail(&sum).1, "joint_law_propagation.transported_dependence_required");
    }

    #[test]
    fn a_composed_interval_needs_a_measured_record_at_its_own_coordinate() {
        let inputs = vec![aligned("x", "s", "joint_1"), aligned("y", "s", "joint_1")];
        let composed = request(
            PropagationOperation::Product,
            inputs,
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Interval,
        );
        let construction = "aligned_draw_percentile";
        let (code, name) = {
            let refusal = check_composed_interval(&composed, None, construction)
                .expect_err("closed without a record");
            refusal.validate().expect("well-formed refusal");
            (refusal.code, refusal.detail)
        };
        assert_eq!(code, "cell_not_licensed");
        assert_eq!(name, "joint_law_propagation.interval_route_closed");

        let spread = PropagationRequest { reading: RequestedReading::Spread, ..composed.clone() };
        let coordinate =
            ComposedCoordinate::of(&spread, &check_propagation(&spread).expect("spread"));
        assert_eq!(coordinate.id(), "product|aligned_joint_law|estimate,estimate");
        let record = CoverageRecord {
            record_id: "record_1".into(),
            coordinate_id: coordinate.id(),
            construction: construction.into(),
            nominal_level: 0.9,
            measured: true,
        };
        let plan = check_composed_interval(&composed, Some(&record), construction).expect("plan");
        assert_eq!(plan.label, "measured_interval");
        assert!((plan.nominal_level - 0.9).abs() < 1e-12);
        assert_eq!(plan.propagation.route, PropagationRoute::AlignedJointLaw);

        let reason = |record: &CoverageRecord, construction: &str| {
            check_composed_interval(&composed, Some(record), construction)
                .expect_err("refusal")
                .detail
        };
        let unmeasured = CoverageRecord { measured: false, ..record.clone() };
        assert_eq!(
            reason(&unmeasured, construction),
            "joint_law_propagation.coverage_record_unmeasured"
        );
        let wrong_coordinate = CoverageRecord {
            coordinate_id: "sum|linear_moments|estimate,estimate".into(),
            ..record.clone()
        };
        assert_eq!(
            reason(&wrong_coordinate, construction),
            "joint_law_propagation.coverage_record_coordinate_mismatch"
        );
        assert_eq!(
            reason(&record, "normal_approximation"),
            "joint_law_propagation.coverage_record_construction_mismatch"
        );
        let bad_level = CoverageRecord { nominal_level: 1.0, ..record };
        assert_eq!(reason(&bad_level, construction), "joint_law_propagation.invalid_request");

        // The dependence route is checked before any record is consulted.
        let unpaired = request(
            PropagationOperation::Product,
            vec![
                input("x", "s1", InputAlignment::IndependentMarginals),
                input("y", "s2", InputAlignment::IndependentMarginals),
            ],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Interval,
        );
        let refusal = check_composed_interval(&unpaired, None, construction).expect_err("refusal");
        assert_eq!(refusal.detail, "joint_law_propagation.index_pairing_refused");
        // A non-interval reading is not this function's.
        let point = PropagationRequest { reading: RequestedReading::Point, ..composed };
        assert_eq!(
            check_composed_interval(&point, None, construction).expect_err("reading").detail,
            "joint_law_propagation.invalid_request"
        );
    }

    #[test]
    fn names_round_trip() {
        for op in [
            PropagationOperation::LinearTransform,
            PropagationOperation::Contrast,
            PropagationOperation::Sum,
            PropagationOperation::WeightedSum,
            PropagationOperation::Product,
            PropagationOperation::NonlinearFunctional,
            PropagationOperation::ChainedSampling,
            PropagationOperation::Mixture,
            PropagationOperation::TransportedComposition,
        ] {
            assert_eq!(PropagationOperation::from_name(op.as_str()), Some(op));
        }
        for route in [
            PropagationRoute::LinearMoments,
            PropagationRoute::AlignedJointLaw,
            PropagationRoute::KnownWeightMixture,
            PropagationRoute::FrechetEnvelope,
            PropagationRoute::CorrelationRangeEnvelope,
        ] {
            assert_eq!(PropagationRoute::from_name(route.as_str()), Some(route));
        }
        for role in [
            InputRole::Estimate,
            InputRole::TransportedInterval,
            InputRole::SourceSampling,
            InputRole::TargetSampling,
            InputRole::TransportFunctional,
        ] {
            assert_eq!(InputRole::from_name(role.as_str()), Some(role));
        }
        for reading in [
            RequestedReading::Point,
            RequestedReading::Spread,
            RequestedReading::Interval,
            RequestedReading::Posterior,
            RequestedReading::Envelope,
        ] {
            assert_eq!(RequestedReading::from_name(reading.as_str()), Some(reading));
        }
    }
}
