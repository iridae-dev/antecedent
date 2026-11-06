//! Replayable artifact for a composed uncertain quantity.
//!
//! Format version 2, kind `joint_law_propagation_v1`. The artifact binds, in
//! one sealed identity, the identities of the inputs (the evidence each rests
//! on, its role in the composition and how its draws are aligned), the
//! alignment identity the rows share, the propagation rule that supported the
//! computation (operation, dependence declaration and reading), the numbers
//! the computation consumed as IEEE bit patterns, the stored result and the
//! label the result carries (`point_only`, `uncalibrated_spread`,
//! `structural_envelope` or `assumption_range`).
//!
//! Version 2 stores the result as a typed body ([`PropagationResultWire`]).
//! Version 1 stored it as a list of named `(name, u64)` entries; the reader
//! still accepts that layout and maps it into the typed body, so stored
//! artifacts keep replaying. The identity excludes the result and is derived
//! under the identity version of the first layout, so an artifact's identity
//! is the same in both layouts.
//!
//! # What a consumer does
//!
//! A consumer trusts none of it. It recomputes the identity over the rule,
//! inputs, numbers and plan; re-runs the propagation rule on the stored
//! declaration and requires the stored route, label and alignment identity to
//! be the ones the rule yields; requires the stored numbers to be the ones the
//! operation names (unit coefficients for a sum, coefficients summing to zero
//! for a contrast, the product functional for a product, the matching envelope
//! query); recomputes the result from the stored numbers
//! and accepts only identical result bits. What this protects against:
//! corruption, edits that are not resealed, and resealed edits that change the
//! operation, the dependence declaration, the alignment identity, a number or
//! the label. It does not protect against a producer that seals wrong but
//! coherent inputs, or a bug shared by producer and consumer; the independent
//! oracles are in the test suites.
//!
//! # What an envelope is
//!
//! A `structural_envelope` is the set of values a functional of two marginals
//! can take over every coupling of them. It carries no point, interval or
//! posterior and cannot be replayed into one: its label is part of the
//! identity and a consumer refuses a point reading of it.
//!
//! # What a spread is
//!
//! A `uncalibrated_spread` result (a variance, a Monte Carlo standard error, an
//! exceedance probability with its Monte Carlo band and a verdict against a
//! cutoff) is computed at the composed coordinate under the supported
//! dependence route. The verdict is about the Monte Carlo band of the stated
//! probability only. The artifact carries no interval and no coverage
//! statement: an interval at a composed coordinate needs a measured coverage
//! record, which the artifact does not hold and the rule's interval reading
//! refuses without.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod wire;

pub use wire::{
    BoundsWire, ConsumedJointLawPropagation, EnvelopeQuantityWire, EnvelopeQueryWire,
    JointLawPropagationArtifactWire, MarginalWire, PropagationInputWire, PropagationRequestWire,
    PropagationResultWire, PropagationRuleWire, VerdictWire,
};

use antecedent_contract::{
    ExternalRefusal, InputAlignment, InputRole, JOINT_LAW_STAGE, PropagationDependence,
    PropagationInput, PropagationOperation, PropagationPlan, PropagationRequest, PropagationRoute,
    RequestedReading, check_propagation,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_stats::{
    DecisionVerdict, JointLawError, MomentSummary, ValueFunctional, aligned_functional_draws,
    aligned_functional_mean, composed_exceedance_decision, correlation_range_envelope,
    covariance_envelope, difference_cdf_envelope, draw_spread, functional_envelope,
    linear_combination_moments, linear_variance_envelope, mixture_exceedance, mixture_moments,
    paired_moments, pooled_dependence_envelope, product_moment_envelope, sum_cdf_envelope,
};

use crate::IoError;
use crate::distribution_artifact::MAX_DISTRIBUTION_DRAWS;

/// The artifact format this reader writes.
pub const JOINT_LAW_PROPAGATION_ARTIFACT_VERSION: u32 = 2;
/// The earlier format, whose result is a list of named entries; the reader
/// accepts it and maps it into the typed result.
pub const JOINT_LAW_PROPAGATION_LEGACY_ARTIFACT_VERSION: u32 = 1;
/// The version the identity digest is derived under. The identity covers no
/// result, so the result layout revision does not change it.
const JOINT_LAW_PROPAGATION_IDENTITY_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE: &str = "joint_law_propagation_v1";
/// Most inputs one composition may name.
pub const JOINT_LAW_PROPAGATION_MAX_INPUTS: usize = 256;
/// Most bytes an encoded propagation artifact may occupy; larger input is
/// refused before any CBOR is parsed.
pub const MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Longest identity, label or name in the artifact, in bytes.
pub const MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES: usize = 4096;
/// Most shared-evidence identities the artifact may record.
pub const MAX_JOINT_LAW_PROPAGATION_SHARED_EVIDENCE: usize = 4096;

/// Intake bounds for a joint-law propagation artifact. The defaults are the
/// `JOINT_LAW_PROPAGATION_MAX_*` and `MAX_JOINT_LAW_PROPAGATION_*` constants and
/// [`MAX_DISTRIBUTION_DRAWS`]; a consumer with a larger budget passes its own to
/// [`JointLawPropagationArtifactWire::decode_within`] and
/// [`JointLawPropagationArtifactWire::consume_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointLawPropagationLimits {
    /// Most encoded bytes.
    pub max_bytes: usize,
    /// Most inputs one composition may name, and the most entries of a coefficient,
    /// mean, variance, weight, estimate or component list.
    pub max_inputs: usize,
    /// Most draws of an aligned or enumerated marginal.
    pub max_draws: usize,
    /// Longest identity, label or name, in bytes.
    pub max_text_bytes: usize,
    /// Most shared-evidence identities.
    pub max_shared_evidence: usize,
}

impl Default for JointLawPropagationLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES,
            max_inputs: JOINT_LAW_PROPAGATION_MAX_INPUTS,
            max_draws: MAX_DISTRIBUTION_DRAWS,
            max_text_bytes: MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES,
            max_shared_evidence: MAX_JOINT_LAW_PROPAGATION_SHARED_EVIDENCE,
        }
    }
}

/// Why a joint-law propagation artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum JointLawPropagationArtifactError {
    /// The bytes do not decode as this format.
    #[error("joint-law propagation artifact does not decode: {0}")]
    Decode(String),
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The identity does not match the stored content, or the caller's
    /// expected identity.
    #[error("identity mismatch")]
    IdentityMismatch,
    /// The artifact could not be encoded.
    #[error("joint-law propagation artifact does not encode: {0}")]
    Encode(String),
    /// A stored field is not a well formed declaration.
    #[error("malformed declaration: {0}")]
    Malformed(String),
    /// The request is refused: each refusal has a registered code, a
    /// namespaced `joint_law_propagation.*` detail and a remedy.
    #[error("joint-law propagation refused: {} refusal(s)", .refusals.len())]
    Refused {
        /// The refusals (never empty).
        refusals: Vec<ExternalRefusal>,
    },
    /// The replayed result differs from the stored one.
    #[error("result does not replay")]
    ResultMismatch,
    /// Cancellation was observed at a work boundary.
    #[error("joint-law propagation cancelled")]
    Cancelled,
}

impl From<IoError> for JointLawPropagationArtifactError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

impl JointLawPropagationArtifactError {
    /// The artifact refusals as structured refusals. A malformed or edited
    /// artifact is `artifact_invalid`/`artifact_refused`; a refused request
    /// keeps its own refusals.
    #[must_use]
    pub fn to_refusals(&self) -> Vec<ExternalRefusal> {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let cancelled = antecedent_core::reason_code!("cancelled_no_claim");
        let make = |code: &'static str, detail: &'static str, remedy: &'static str| {
            vec![
                ExternalRefusal::new(code, JOINT_LAW_STAGE, detail)
                    .with_offending(self.to_string())
                    .with_remedy(remedy),
            ]
        };
        match self {
            Self::Refused { refusals } => refusals.clone(),
            Self::IdentityMismatch | Self::ResultMismatch => make(
                invalid,
                antecedent_core::reason_details::joint_law_propagation::ARTIFACT_REFUSED,
                "supply the artifact whose identity equals the retained one and whose result recomputes from its stored inputs and rule",
            ),
            Self::Cancelled => make(
                cancelled,
                antecedent_core::reason_details::joint_law_propagation::CANCELLED,
                "rerun without cancellation; the cancellation is not a verdict on the composition",
            ),
            Self::Decode(_)
            | Self::UnsupportedSemantics(_)
            | Self::UnsupportedVersion { .. }
            | Self::LimitsExceeded(_)
            | Self::Encode(_)
            | Self::Malformed(_) => make(
                invalid,
                antecedent_core::reason_details::joint_law_propagation::ARTIFACT_INVALID,
                "supply an intact joint_law_propagation_v1 artifact of the supported layout",
            ),
        }
    }
}

type Error = JointLawPropagationArtifactError;

fn route_refusal(detail: &'static str, offending: &str, remedy: &'static str) -> Error {
    Error::Refused {
        refusals: vec![
            ExternalRefusal::new(
                antecedent_core::reason_code!("route_not_supported"),
                JOINT_LAW_STAGE,
                detail,
            )
            .with_offending(offending.to_owned())
            .with_remedy(remedy),
        ],
    }
}

fn invalid_request(offending: &str, remedy: &'static str) -> Error {
    Error::Refused {
        refusals: vec![
            ExternalRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                JOINT_LAW_STAGE,
                antecedent_core::reason_details::joint_law_propagation::INVALID_REQUEST,
            )
            .with_offending(offending.to_owned())
            .with_remedy(remedy),
        ],
    }
}

fn math_error(error: JointLawError) -> Error {
    match error {
        JointLawError::StructureNotVerified { structure } => route_refusal(
            antecedent_core::reason_details::joint_law_propagation::STRUCTURE_NOT_VERIFIED,
            structure,
            "the value function must have a verified mixed-difference sign on the marginals' support for its coupling extremes to be bounds; use one of the closed functionals on a support where it holds, or supply the aligned joint law",
        ),
        JointLawError::BudgetExceeded(what) => route_refusal(
            antecedent_core::reason_details::joint_law_propagation::BOUNDS_EXCEEDED,
            what,
            "coarsen the marginals or the draw set to within the declared bounds",
        ),
        JointLawError::InvalidLevel(what) => invalid_request(
            what,
            "give a level or cutoff strictly between zero and one, and rows at least as many as the effective size",
        ),
        JointLawError::Cancelled => Error::Cancelled,
        other => Error::Malformed(other.to_string()),
    }
}

// ----------------------------------------------------------- conversions

/// Stable integer code of a decision verdict: `0` act, `1` withhold, `2`
/// unresolved.
#[must_use]
pub fn verdict_code(verdict: DecisionVerdict) -> u64 {
    match verdict {
        DecisionVerdict::Act => 0,
        DecisionVerdict::Withhold => 1,
        DecisionVerdict::Unresolved => 2,
    }
}

fn floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|b| f64::from_bits(*b)).collect()
}

fn bits_of(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// The dependence declaration as stable wire fields.
fn dependence_wire(
    dependence: &PropagationDependence,
) -> (&'static str, Option<String>, Option<u64>, Option<u64>) {
    match dependence {
        PropagationDependence::Unknown => ("unknown", None, None, None),
        PropagationDependence::DeclaredIndependent => ("declared_independent", None, None, None),
        PropagationDependence::Covariance { id } => ("covariance", Some(id.clone()), None, None),
        PropagationDependence::AlignedJointLaw => ("aligned_joint_law", None, None, None),
        PropagationDependence::KnownWeights => ("known_weights", None, None, None),
        PropagationDependence::CorrelationRange { rho_lo, rho_hi } => {
            ("correlation_range", None, Some(rho_lo.to_bits()), Some(rho_hi.to_bits()))
        }
        PropagationDependence::MarginalsOnly => ("marginals_only", None, None, None),
    }
}

fn dependence_from_wire(rule: &PropagationRuleWire) -> Result<PropagationDependence, Error> {
    let malformed = |what: &str| Error::Malformed(what.to_owned());
    Ok(match rule.dependence.as_str() {
        "unknown" => PropagationDependence::Unknown,
        "declared_independent" => PropagationDependence::DeclaredIndependent,
        "covariance" => PropagationDependence::Covariance {
            id: rule.covariance_id.clone().ok_or_else(|| malformed("covariance id missing"))?,
        },
        "aligned_joint_law" => PropagationDependence::AlignedJointLaw,
        "known_weights" => PropagationDependence::KnownWeights,
        "correlation_range" => PropagationDependence::CorrelationRange {
            rho_lo: f64::from_bits(
                rule.rho_lo.ok_or_else(|| malformed("lower correlation missing"))?,
            ),
            rho_hi: f64::from_bits(
                rule.rho_hi.ok_or_else(|| malformed("upper correlation missing"))?,
            ),
        },
        "marginals_only" => PropagationDependence::MarginalsOnly,
        _ => return Err(malformed("unknown dependence declaration")),
    })
}

fn input_to_wire(input: &PropagationInput) -> PropagationInputWire {
    let (alignment, alignment_id) = match &input.alignment {
        InputAlignment::Aligned { alignment_id } => ("aligned", Some(alignment_id.clone())),
        InputAlignment::IndependentMarginals => ("independent_marginals", None),
        InputAlignment::MomentsOnly => ("moments_only", None),
    };
    PropagationInputWire {
        input_id: input.id.clone(),
        evidence_id: input.evidence_id.clone(),
        role: input.role.as_str().to_owned(),
        alignment: alignment.to_owned(),
        alignment_id,
    }
}

fn input_from_wire(wire: &PropagationInputWire) -> Result<PropagationInput, Error> {
    let role = InputRole::from_name(&wire.role)
        .ok_or_else(|| Error::Malformed("unknown input role".to_owned()))?;
    let alignment = match (wire.alignment.as_str(), &wire.alignment_id) {
        ("aligned", Some(id)) => InputAlignment::Aligned { alignment_id: id.clone() },
        ("independent_marginals", None) => InputAlignment::IndependentMarginals,
        ("moments_only", None) => InputAlignment::MomentsOnly,
        _ => return Err(Error::Malformed("input alignment is not well formed".to_owned())),
    };
    Ok(PropagationInput {
        id: wire.input_id.clone(),
        evidence_id: wire.evidence_id.clone(),
        role,
        alignment,
    })
}

fn request_from_wire(
    rule: &PropagationRuleWire,
    inputs: &[PropagationInputWire],
) -> Result<PropagationRequest, Error> {
    let operation = PropagationOperation::from_name(&rule.operation)
        .ok_or_else(|| Error::Malformed("unknown operation".to_owned()))?;
    let reading = RequestedReading::from_name(&rule.reading)
        .ok_or_else(|| Error::Malformed("unknown reading".to_owned()))?;
    Ok(PropagationRequest {
        operation,
        inputs: inputs.iter().map(input_from_wire).collect::<Result<_, _>>()?,
        dependence: dependence_from_wire(rule)?,
        reading,
    })
}

// ----------------------------------------------------------------- bounds

fn check_bounds(
    wire: &JointLawPropagationArtifactWire,
    limits: &JointLawPropagationLimits,
) -> Result<(), Error> {
    let too_large = Error::LimitsExceeded;
    if wire.inputs.len() > limits.max_inputs {
        return Err(too_large("inputs"));
    }
    let long = |text: &str| text.len() > limits.max_text_bytes;
    let long_opt = |text: &Option<String>| text.as_deref().is_some_and(long);
    if wire.shared_evidence.len() > limits.max_shared_evidence || wire.required_features.len() > 16
    {
        return Err(too_large("shared evidence"));
    }
    if wire.shared_evidence.iter().any(|id| long(id))
        || wire.required_features.iter().any(|f| long(f))
        || long(&wire.route)
        || long(&wire.label)
        || long(&wire.identity)
        || long_opt(&wire.alignment_id)
        || long_opt(&wire.covariance_id)
        || long(&wire.rule.operation)
        || long(&wire.rule.dependence)
        || long(&wire.rule.reading)
        || long_opt(&wire.rule.covariance_id)
        || wire.inputs.iter().any(|input| {
            long(&input.input_id)
                || long(&input.evidence_id)
                || long(&input.role)
                || long(&input.alignment)
                || long_opt(&input.alignment_id)
        })
    {
        return Err(too_large("text"));
    }
    let draws = |n: usize| {
        if n > limits.max_draws { Err(too_large("draws")) } else { Ok(()) }
    };
    match &wire.request {
        PropagationRequestWire::LinearCombination { coefficients, means, covariance } => {
            if coefficients.len() > limits.max_inputs
                || means.len() > limits.max_inputs
                || covariance.len() > limits.max_inputs.saturating_mul(limits.max_inputs)
            {
                return Err(too_large("linear combination"));
            }
        }
        PropagationRequestWire::AlignedLinearCombination { x, y, .. }
        | PropagationRequestWire::AlignedFunctional { x, y, .. } => {
            draws(x.len())?;
            draws(y.len())?;
        }
        PropagationRequestWire::AlignedFunctionalSpread { x, y, .. }
        | PropagationRequestWire::AlignedExceedance { x, y, .. } => {
            draws(x.len())?;
            draws(y.len())?;
        }
        PropagationRequestWire::MixtureExceedance { weights, components, .. } => {
            if weights.len() > limits.max_inputs || components.len() > limits.max_inputs {
                return Err(too_large("mixture components"));
            }
            let total: usize = components.iter().map(Vec::len).sum();
            draws(total)?;
        }
        PropagationRequestWire::Mixture { weights, means, variances } => {
            if weights.len() > limits.max_inputs
                || means.len() > limits.max_inputs
                || variances.len() > limits.max_inputs
            {
                return Err(too_large("mixture components"));
            }
        }
        PropagationRequestWire::FrechetEnvelope { x, y, .. } => {
            for marginal in [x, y] {
                draws(marginal.values.len())?;
                draws(marginal.masses.len())?;
            }
        }
        PropagationRequestWire::CorrelationRange { .. } => {}
        PropagationRequestWire::PooledCorrelationRange { estimates, standard_errors } => {
            if estimates.len() > limits.max_inputs || standard_errors.len() > limits.max_inputs {
                return Err(too_large("pooled studies"));
            }
        }
    }
    Ok(())
}

// --------------------------------------------------------------- identity

/// Identity of everything the computation rests on, excluding its result.
///
/// The digest is over the canonical wire encoding under a versioned tag; the
/// stored result is excluded because it is an output replayed from the rest.
///
/// # Errors
/// Encoding failure.
#[allow(
    clippy::too_many_arguments,
    reason = "each argument is one separately named part of the identity"
)]
pub fn joint_law_propagation_identity(
    rule: &PropagationRuleWire,
    inputs: &[PropagationInputWire],
    request: &PropagationRequestWire,
    route: &str,
    label: &str,
    alignment_id: Option<&str>,
    covariance_id: Option<&str>,
    shared_evidence: &[String],
) -> Result<String, IoError> {
    crate::digest_hex(
        IdentityDomain::Program,
        &(
            JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE,
            JOINT_LAW_PROPAGATION_IDENTITY_VERSION,
            rule,
            inputs,
            request,
            route,
            label,
            alignment_id,
            covariance_id,
            shared_evidence,
        ),
    )
}

// -------------------------------------------------------------- execution

fn require_len(actual: usize, expected: usize, what: &str) -> Result<(), Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Malformed(format!("{what} count differs from the declared inputs")))
    }
}

/// Relative tolerance under which contrast coefficients sum to zero.
const CONTRAST_SUM_TOLERANCE: f64 = 1e-12;

/// Whether the stored linear coefficients are the ones `operation` names: a sum
/// has unit coefficients, a contrast coefficients summing to zero, a transform
/// one coefficient. A non-finite coefficient is left to the computation, which
/// refuses it with its own typed error.
#[allow(
    clippy::float_cmp,
    reason = "a sum's coefficients are the exact literal 1.0, not a computed quantity"
)]
fn coefficients_fit(operation: PropagationOperation, coefficients: &[f64]) -> bool {
    use PropagationOperation as Op;
    if coefficients.iter().any(|c| !c.is_finite()) {
        return true;
    }
    match operation {
        Op::Sum => coefficients.iter().all(|c| *c == 1.0),
        Op::Contrast => {
            let total: f64 = coefficients.iter().sum();
            let scale: f64 = coefficients.iter().map(|c| c.abs()).sum();
            total.abs() <= CONTRAST_SUM_TOLERANCE * scale
        }
        Op::LinearTransform => coefficients.len() == 1,
        _ => true,
    }
}

/// Whether the stored functional is the one `operation` names: a product is
/// the `product` functional, any other nonlinear operation names its own.
fn functional_fits(operation: PropagationOperation, functional: &str) -> bool {
    operation != PropagationOperation::Product || functional == ValueFunctional::Product.as_str()
}

/// Whether the numbers fit the route and operation the rule yielded.
fn variant_fits(
    operation: PropagationOperation,
    route: PropagationRoute,
    request: &PropagationRequestWire,
    spread: bool,
) -> bool {
    use PropagationOperation as Op;
    use PropagationRequestWire as W;
    let linear = matches!(
        operation,
        Op::LinearTransform | Op::Contrast | Op::Sum | Op::WeightedSum | Op::TransportedComposition
    );
    let nonlinear =
        matches!(operation, Op::Product | Op::NonlinearFunctional | Op::TransportedComposition);
    match (route, request) {
        (PropagationRoute::LinearMoments, W::LinearCombination { coefficients, .. }) => {
            linear && coefficients_fit(operation, &floats(coefficients))
        }
        (PropagationRoute::AlignedJointLaw, W::AlignedLinearCombination { a, b, .. }) => {
            linear && coefficients_fit(operation, &[f64::from_bits(*a), f64::from_bits(*b)])
        }
        (PropagationRoute::AlignedJointLaw, W::AlignedFunctional { functional, .. }) => {
            nonlinear && functional_fits(operation, functional)
        }
        // The draw-based spread and decision numbers are a spread reading's
        // own: a point reading is not given a spread under its own label.
        (
            PropagationRoute::AlignedJointLaw,
            W::AlignedFunctionalSpread { functional, .. } | W::AlignedExceedance { functional, .. },
        ) => spread && nonlinear && functional_fits(operation, functional),
        (PropagationRoute::KnownWeightMixture, W::MixtureExceedance { .. }) => {
            spread && operation == Op::Mixture
        }
        (PropagationRoute::KnownWeightMixture, W::Mixture { .. }) => operation == Op::Mixture,
        (PropagationRoute::FrechetEnvelope, W::FrechetEnvelope { query, .. }) => match query {
            EnvelopeQueryWire::SumCdf { .. } => operation == Op::Sum,
            EnvelopeQueryWire::DifferenceCdf { .. } => operation == Op::Contrast,
            EnvelopeQueryWire::LinearVariance { a, b } => {
                matches!(operation, Op::Sum | Op::Contrast | Op::WeightedSum)
                    && coefficients_fit(operation, &[f64::from_bits(*a), f64::from_bits(*b)])
            }
            EnvelopeQueryWire::ProductMoment | EnvelopeQueryWire::Covariance => {
                operation == Op::Product
            }
            EnvelopeQueryWire::Functional { functional } => {
                matches!(operation, Op::Product | Op::NonlinearFunctional)
                    && functional_fits(operation, functional)
            }
        },
        (PropagationRoute::CorrelationRangeEnvelope, W::CorrelationRange { a, b, .. }) => {
            !linear || coefficients_fit(operation, &[f64::from_bits(*a), f64::from_bits(*b)])
        }
        (PropagationRoute::CorrelationRangeEnvelope, W::PooledCorrelationRange { .. }) => true,
        _ => false,
    }
}

fn evaluate_envelope(
    query: &EnvelopeQueryWire,
    x: &MarginalWire,
    y: &MarginalWire,
    ctx: &ExecutionContext,
) -> Result<PropagationResultWire, Error> {
    use EnvelopeQuantityWire as Quantity;
    let (mx, my) = (x.marginal()?, y.marginal()?);
    let cancel = &ctx.cancellation;
    let (quantity, bounds) = match query {
        EnvelopeQueryWire::SumCdf { threshold } => {
            let t = f64::from_bits(*threshold);
            (Quantity::SumCdf, sum_cdf_envelope(&mx, &my, t).map_err(math_error)?)
        }
        EnvelopeQueryWire::DifferenceCdf { threshold } => {
            let t = f64::from_bits(*threshold);
            (Quantity::DifferenceCdf, difference_cdf_envelope(&mx, &my, t).map_err(math_error)?)
        }
        EnvelopeQueryWire::ProductMoment => (
            Quantity::ProductMoment,
            product_moment_envelope(&mx, &my, cancel).map_err(math_error)?,
        ),
        EnvelopeQueryWire::Covariance => {
            (Quantity::Covariance, covariance_envelope(&mx, &my, cancel).map_err(math_error)?)
        }
        EnvelopeQueryWire::LinearVariance { a, b } => (
            Quantity::LinearVariance,
            linear_variance_envelope(f64::from_bits(*a), f64::from_bits(*b), &mx, &my, cancel)
                .map_err(math_error)?,
        ),
        EnvelopeQueryWire::Functional { functional } => {
            let functional = ValueFunctional::from_name(functional)
                .ok_or_else(|| Error::Malformed("unknown value functional".to_owned()))?;
            (
                Quantity::Functional,
                functional_envelope(&mx, &my, functional, cancel).map_err(math_error)?,
            )
        }
    };
    Ok(PropagationResultWire::Envelope { quantity, bounds: bounds.into() })
}

fn evaluate_correlation_range(
    rule: &PropagationRuleWire,
    request: &PropagationRequestWire,
    n_inputs: usize,
) -> Result<PropagationResultWire, Error> {
    let (Some(lo), Some(hi)) = (rule.rho_lo, rule.rho_hi) else {
        return Err(Error::Malformed("correlation range missing".to_owned()));
    };
    let (rho_lo, rho_hi) = (f64::from_bits(lo), f64::from_bits(hi));
    match request {
        PropagationRequestWire::CorrelationRange { x_mean, x_sd, y_mean, y_sd, a, b } => {
            let x = MomentSummary {
                mean: f64::from_bits(*x_mean),
                standard_deviation: f64::from_bits(*x_sd),
            };
            let y = MomentSummary {
                mean: f64::from_bits(*y_mean),
                standard_deviation: f64::from_bits(*y_sd),
            };
            let range = correlation_range_envelope(
                x,
                y,
                f64::from_bits(*a),
                f64::from_bits(*b),
                rho_lo,
                rho_hi,
            )
            .map_err(math_error)?;
            Ok(PropagationResultWire::CorrelationRange {
                product_moment: range.product_moment.into(),
                covariance: range.covariance.into(),
                linear_variance: range.linear_variance.into(),
            })
        }
        PropagationRequestWire::PooledCorrelationRange { estimates, standard_errors } => {
            require_len(estimates.len(), n_inputs, "study")?;
            let envelope = pooled_dependence_envelope(
                &floats(estimates),
                &floats(standard_errors),
                rho_lo,
                rho_hi,
            )
            .map_err(|e| Error::Malformed(e.to_string()))?;
            Ok(PropagationResultWire::PooledRange {
                estimate: BoundsWire {
                    lower: envelope.estimate_min.to_bits(),
                    upper: envelope.estimate_max.to_bits(),
                },
                variance: BoundsWire {
                    lower: envelope.variance_min.to_bits(),
                    upper: envelope.variance_max.to_bits(),
                },
            })
        }
        _ => Err(Error::Malformed("request is not a correlation range".to_owned())),
    }
}

/// The mean and variance of a linear combination under the declared covariance.
fn evaluate_linear_combination(
    rule: &PropagationRuleWire,
    n_inputs: usize,
    coefficients: &[u64],
    means: &[u64],
    covariance: &[u64],
) -> Result<PropagationResultWire, Error> {
    require_len(coefficients.len(), n_inputs, "coefficient")?;
    if rule.dependence == "declared_independent"
        && covariance.len() == n_inputs * n_inputs
        && (0..n_inputs).any(|i| {
            (0..n_inputs).any(|j| i != j && f64::from_bits(covariance[i * n_inputs + j]) != 0.0)
        })
    {
        return Err(invalid_request(
            "covariance",
            "declared independent inputs require zero cross-covariances; declare the covariance dependence when cross terms are present",
        ));
    }
    let m = linear_combination_moments(&floats(coefficients), &floats(means), &floats(covariance))
        .map_err(math_error)?;
    Ok(PropagationResultWire::Moments { mean: m.mean.to_bits(), variance: m.variance.to_bits() })
}

/// The moments of `a x + b y` over aligned draws.
fn evaluate_aligned_linear_combination(
    n_inputs: usize,
    a: u64,
    b: u64,
    x: &[u64],
    y: &[u64],
) -> Result<PropagationResultWire, Error> {
    require_len(2, n_inputs, "aligned column")?;
    let (a, b) = (f64::from_bits(a), f64::from_bits(b));
    if !a.is_finite() || !b.is_finite() {
        return Err(math_error(JointLawError::NonFinite("coefficient")));
    }
    let pm = paired_moments(&floats(x), &floats(y)).map_err(math_error)?;
    let mean = a * pm.mean_x + b * pm.mean_y;
    let variance = a * a * pm.var_x + b * b * pm.var_y + 2.0 * a * b * pm.covariance;
    if !mean.is_finite() || !variance.is_finite() {
        return Err(math_error(JointLawError::NonFinite("combination moment")));
    }
    let variance = variance.max(0.0);
    Ok(PropagationResultWire::AlignedMoments {
        mean: mean.to_bits(),
        variance: variance.to_bits(),
        covariance: pm.covariance.to_bits(),
        n_rows: pm.n as u64,
    })
}

/// The exceedance decision of an aligned functional; `levels` are the threshold, the
/// cutoff and the band level as float bits.
fn evaluate_aligned_exceedance(
    n_inputs: usize,
    functional: &str,
    x: &[u64],
    y: &[u64],
    levels: [u64; 3],
    ctx: &ExecutionContext,
) -> Result<PropagationResultWire, Error> {
    let [threshold, cutoff, band_level] = levels;
    require_len(2, n_inputs, "aligned column")?;
    let functional = ValueFunctional::from_name(functional)
        .ok_or_else(|| Error::Malformed("unknown value functional".to_owned()))?;
    let rows = aligned_functional_draws(&floats(x), &floats(y), functional, &ctx.cancellation)
        .map_err(math_error)?;
    let decision = composed_exceedance_decision(
        &rows,
        f64::from_bits(threshold),
        f64::from_bits(cutoff),
        f64::from_bits(band_level),
        None,
    )
    .map_err(math_error)?;
    Ok(PropagationResultWire::AlignedExceedance {
        probability: decision.probability.to_bits(),
        standard_error: decision.standard_error.to_bits(),
        band: decision.band.to_bits(),
        verdict: decision.verdict.into(),
        n_rows: decision.n as u64,
    })
}

/// Compute the named result values of `request`.
fn evaluate(
    rule: &PropagationRuleWire,
    request: &PropagationRequestWire,
    n_inputs: usize,
    ctx: &ExecutionContext,
) -> Result<PropagationResultWire, Error> {
    if ctx.cancellation.is_cancelled() {
        return Err(Error::Cancelled);
    }
    let result = match request {
        PropagationRequestWire::LinearCombination { coefficients, means, covariance } => {
            evaluate_linear_combination(rule, n_inputs, coefficients, means, covariance)
        }
        PropagationRequestWire::AlignedLinearCombination { a, b, x, y } => {
            evaluate_aligned_linear_combination(n_inputs, *a, *b, x, y)
        }
        PropagationRequestWire::AlignedFunctional { functional, x, y } => {
            require_len(2, n_inputs, "aligned column")?;
            let functional = ValueFunctional::from_name(functional)
                .ok_or_else(|| Error::Malformed("unknown value functional".to_owned()))?;
            let (xs, ys) = (floats(x), floats(y));
            let mean = aligned_functional_mean(&xs, &ys, functional, &ctx.cancellation)
                .map_err(math_error)?;
            Ok(PropagationResultWire::AlignedMean { mean: mean.to_bits(), n_rows: xs.len() as u64 })
        }
        PropagationRequestWire::AlignedFunctionalSpread { functional, x, y } => {
            require_len(2, n_inputs, "aligned column")?;
            let functional = ValueFunctional::from_name(functional)
                .ok_or_else(|| Error::Malformed("unknown value functional".to_owned()))?;
            let rows =
                aligned_functional_draws(&floats(x), &floats(y), functional, &ctx.cancellation)
                    .map_err(math_error)?;
            let spread = draw_spread(&rows, None).map_err(math_error)?;
            Ok(PropagationResultWire::AlignedSpread {
                mean: spread.mean.to_bits(),
                variance: spread.variance.to_bits(),
                mean_standard_error: spread.mean_standard_error.to_bits(),
                n_rows: spread.n as u64,
            })
        }
        PropagationRequestWire::AlignedExceedance {
            functional,
            x,
            y,
            threshold,
            cutoff,
            band_level,
        } => {
            let levels = [*threshold, *cutoff, *band_level];
            evaluate_aligned_exceedance(n_inputs, functional, x, y, levels, ctx)
        }
        PropagationRequestWire::MixtureExceedance { weights, components, threshold } => {
            require_len(components.len(), n_inputs, "mixture component")?;
            let columns: Vec<Vec<f64>> = components.iter().map(|c| floats(c)).collect();
            let borrowed: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
            let (probability, standard_error) =
                mixture_exceedance(&floats(weights), &borrowed, f64::from_bits(*threshold))
                    .map_err(math_error)?;
            Ok(PropagationResultWire::MixtureExceedance {
                probability: probability.to_bits(),
                standard_error: standard_error.to_bits(),
            })
        }
        PropagationRequestWire::Mixture { weights, means, variances } => {
            require_len(weights.len(), n_inputs, "mixture component")?;
            let m = mixture_moments(&floats(weights), &floats(means), &floats(variances))
                .map_err(math_error)?;
            Ok(PropagationResultWire::Mixture {
                mean: m.mean.to_bits(),
                variance: m.variance.to_bits(),
                within: m.within.to_bits(),
                between: m.between.to_bits(),
            })
        }
        PropagationRequestWire::FrechetEnvelope { query, x, y } => {
            require_len(2, n_inputs, "marginal")?;
            evaluate_envelope(query, x, y, ctx)
        }
        PropagationRequestWire::CorrelationRange { .. }
        | PropagationRequestWire::PooledCorrelationRange { .. } => {
            evaluate_correlation_range(rule, request, n_inputs)
        }
    }?;
    if result.float_entries().iter().any(|(_, bits)| !f64::from_bits(*bits).is_finite()) {
        return Err(math_error(JointLawError::NonFinite("propagation result")));
    }
    Ok(result)
}

fn plan_for(
    rule: &PropagationRuleWire,
    inputs: &[PropagationInputWire],
    request: &PropagationRequestWire,
) -> Result<PropagationPlan, Error> {
    let declared = request_from_wire(rule, inputs)?;
    let plan = check_propagation(&declared)
        .map_err(|refusal| Error::Refused { refusals: vec![refusal] })?;
    let spread = declared.reading == RequestedReading::Spread;
    if !variant_fits(plan.operation, plan.route, request, spread) {
        return Err(invalid_request(
            plan.route.as_str(),
            "supply the numbers the supported route consumes (a covariance for the linear route, aligned draws for the aligned route, weights for a mixture, two marginals for an envelope); a chained transformation is checked by the rule but its stages run in the caller",
        ));
    }
    Ok(plan)
}

impl JointLawPropagationArtifactWire {
    /// Check `declared` against the propagation rule, execute `request` and
    /// seal an artifact.
    ///
    /// # Errors
    /// A refused request (each refusal carries its registered code and
    /// remedy), numbers that do not fit the supported route, a failed
    /// computation, cancellation or an encoding failure.
    pub fn seal(
        declared: &PropagationRequest,
        request: PropagationRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        Self::seal_within(declared, request, ctx, &JointLawPropagationLimits::default())
    }

    /// [`Self::seal`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::seal`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn seal_within(
        declared: &PropagationRequest,
        request: PropagationRequestWire,
        ctx: &ExecutionContext,
        limits: &JointLawPropagationLimits,
    ) -> Result<Self, Error> {
        let (dependence, covariance_id, rho_lo, rho_hi) = dependence_wire(&declared.dependence);
        let rule = PropagationRuleWire {
            operation: declared.operation.as_str().to_owned(),
            dependence: dependence.to_owned(),
            covariance_id,
            rho_lo,
            rho_hi,
            reading: declared.reading.as_str().to_owned(),
        };
        let inputs: Vec<PropagationInputWire> = declared.inputs.iter().map(input_to_wire).collect();
        let plan = plan_for(&rule, &inputs, &request)?;
        let result = evaluate(&rule, &request, inputs.len(), ctx)?;
        let identity = joint_law_propagation_identity(
            &rule,
            &inputs,
            &request,
            plan.route.as_str(),
            plan.label,
            plan.alignment_id.as_deref(),
            plan.covariance_id.as_deref(),
            &plan.shared_evidence,
        )
        .map_err(|e| Error::Encode(e.to_string()))?;
        let wire = Self {
            version: JOINT_LAW_PROPAGATION_ARTIFACT_VERSION,
            required_features: vec![JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE.to_owned()],
            rule,
            inputs,
            request,
            route: plan.route.as_str().to_owned(),
            label: plan.label.to_owned(),
            alignment_id: plan.alignment_id,
            covariance_id: plan.covariance_id,
            shared_evidence: plan.shared_evidence,
            result,
            identity,
        };
        check_bounds(&wire, limits)?;
        Ok(wire)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, Error> {
        self.export_within(&JointLawPropagationLimits::default())
    }

    /// [`Self::export`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::export`].
    pub fn export_within(&self, limits: &JointLawPropagationLimits) -> Result<Vec<u8>, Error> {
        if self.version != JOINT_LAW_PROPAGATION_ARTIFACT_VERSION {
            return Err(Error::UnsupportedVersion { version: self.version });
        }
        check_bounds(self, limits)?;
        crate::to_cbor(self).map_err(|e| Error::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        Self::decode_within(bytes, &JointLawPropagationLimits::default())
    }

    /// [`Self::decode`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::decode`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn decode_within(bytes: &[u8], limits: &JointLawPropagationLimits) -> Result<Self, Error> {
        if bytes.len() > limits.max_bytes {
            return Err(Error::LimitsExceeded("artifact bytes"));
        }
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if is_supported_version(version) {
                Ok(())
            } else {
                Err(Error::UnsupportedVersion { version })
            }
        })?;
        if wire.required_features != [JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE] {
            return Err(Error::UnsupportedSemantics("required features"));
        }
        check_bounds(&wire, limits)?;
        Ok(wire)
    }

    /// Verify and replay.
    ///
    /// `expected_identity`, when given, is the identity the caller holds
    /// independently of the artifact.
    ///
    /// # Errors
    /// Any of the refusals of [`JointLawPropagationArtifactError`].
    pub fn consume(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedJointLawPropagation, Error> {
        self.consume_within(expected_identity, ctx, &JointLawPropagationLimits::default())
    }

    /// [`Self::consume`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::consume`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn consume_within(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
        limits: &JointLawPropagationLimits,
    ) -> Result<ConsumedJointLawPropagation, Error> {
        if !is_supported_version(self.version) {
            return Err(Error::UnsupportedVersion { version: self.version });
        }
        if self.required_features != [JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE] {
            return Err(Error::UnsupportedSemantics("required features"));
        }
        check_bounds(self, limits)?;
        let recomputed = joint_law_propagation_identity(
            &self.rule,
            &self.inputs,
            &self.request,
            &self.route,
            &self.label,
            self.alignment_id.as_deref(),
            self.covariance_id.as_deref(),
            &self.shared_evidence,
        )
        .map_err(|e| Error::Encode(e.to_string()))?;
        if recomputed != self.identity {
            return Err(Error::IdentityMismatch);
        }
        if let Some(expected) = expected_identity {
            if expected != recomputed {
                return Err(Error::IdentityMismatch);
            }
        }
        // The rule decides again from the stored declaration: a resealed edit
        // that changes the operation, the dependence declaration, an
        // alignment identity or the label no longer matches what it yields.
        let plan = plan_for(&self.rule, &self.inputs, &self.request)?;
        let same_plan = plan.route.as_str() == self.route
            && plan.label == self.label
            && plan.alignment_id == self.alignment_id
            && plan.covariance_id == self.covariance_id
            && plan.shared_evidence == self.shared_evidence;
        if !same_plan {
            return Err(Error::IdentityMismatch);
        }
        let result = evaluate(&self.rule, &self.request, self.inputs.len(), ctx)?;
        if result != self.result {
            return Err(Error::ResultMismatch);
        }
        Ok(ConsumedJointLawPropagation {
            identity: recomputed,
            operation: self.rule.operation.clone(),
            route: self.route.clone(),
            label: self.label.clone(),
            alignment_id: self.alignment_id.clone(),
            result,
        })
    }
}

/// Whether this reader accepts the stored format version: the current one and
/// the earlier named-list layout.
const fn is_supported_version(version: u32) -> bool {
    version == JOINT_LAW_PROPAGATION_ARTIFACT_VERSION
        || version == JOINT_LAW_PROPAGATION_LEGACY_ARTIFACT_VERSION
}

/// Bit patterns of a float slice, for building request wires.
#[must_use]
pub fn float_bits(values: &[f64]) -> Vec<u64> {
    bits_of(values)
}
