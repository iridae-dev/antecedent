//! Replayable artifact for a composed uncertain quantity.
//!
//! Format version 2, kind `joint_law_propagation_v1`. The artifact binds, in
//! one sealed identity, the identities of the inputs (the evidence each rests
//! on, its role in the composition and how its draws are aligned), the
//! alignment identity the rows share, the propagation rule that licensed the
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
//! be the ones the rule yields; recomputes the result from the stored numbers
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
//! cutoff) is computed at the composed coordinate under the licensed
//! dependence route. The verdict is about the Monte Carlo band of the stated
//! probability only. The artifact carries no interval and no coverage
//! statement: an interval at a composed coordinate needs a measured coverage
//! record, which the artifact does not hold and the rule's interval reading
//! refuses without.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, InputAlignment, InputRole, JOINT_LAW_STAGE, PropagationDependence,
    PropagationInput, PropagationOperation, PropagationPlan, PropagationRequest, PropagationRoute,
    RequestedReading, check_propagation,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_stats::{
    DecisionVerdict, JointLawError, Marginal, MomentSummary, ValueBounds, ValueFunctional,
    aligned_functional_draws, aligned_functional_mean, composed_exceedance_decision,
    correlation_range_envelope, covariance_envelope, difference_cdf_envelope, draw_spread,
    functional_envelope, linear_combination_moments, linear_variance_envelope, mixture_exceedance,
    mixture_moments, paired_moments, pooled_dependence_envelope, product_moment_envelope,
    sum_cdf_envelope,
};
use serde::{Deserialize, Serialize};

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
                "joint_law_propagation.artifact_refused",
                "supply the artifact whose identity equals the retained one and whose result recomputes from its stored inputs and rule",
            ),
            Self::Cancelled => make(
                cancelled,
                "joint_law_propagation.cancelled",
                "rerun without cancellation; the cancellation is not a verdict on the composition",
            ),
            Self::Decode(_)
            | Self::UnsupportedSemantics(_)
            | Self::UnsupportedVersion { .. }
            | Self::LimitsExceeded(_)
            | Self::Encode(_)
            | Self::Malformed(_) => make(
                invalid,
                "joint_law_propagation.artifact_invalid",
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
                "joint_law_propagation.invalid_request",
            )
            .with_offending(offending.to_owned())
            .with_remedy(remedy),
        ],
    }
}

fn math_error(error: JointLawError) -> Error {
    match error {
        JointLawError::StructureNotVerified { structure } => route_refusal(
            "joint_law_propagation.structure_not_verified",
            structure,
            "the value function must have a verified mixed-difference sign on the marginals' support for its coupling extremes to be bounds; use one of the closed functionals on a support where it holds, or supply the aligned joint law",
        ),
        JointLawError::BudgetExceeded(what) => route_refusal(
            "joint_law_propagation.bounds_exceeded",
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

// ----------------------------------------------------------------- the wire

/// One input of the composition.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationInputWire {
    /// Stable input identity.
    pub input_id: String,
    /// Identity of the evidence the input rests on.
    pub evidence_id: String,
    /// The input's role (`estimate`, `transported_interval`, ...).
    pub role: String,
    /// `aligned`, `independent_marginals` or `moments_only`.
    pub alignment: String,
    /// The alignment identity, present exactly when `alignment` is `aligned`.
    pub alignment_id: Option<String>,
}

/// The propagation rule's declaration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PropagationRuleWire {
    /// The operation's stable name.
    pub operation: String,
    /// `unknown`, `declared_independent`, `covariance`, `aligned_joint_law`,
    /// `known_weights`, `correlation_range` or `marginals_only`.
    pub dependence: String,
    /// The covariance declaration identity, for `covariance`.
    pub covariance_id: Option<String>,
    /// Lower correlation bits, for `correlation_range`.
    pub rho_lo: Option<u64>,
    /// Upper correlation bits, for `correlation_range`.
    pub rho_hi: Option<u64>,
    /// The reading (`point`, `spread` or `envelope`).
    pub reading: String,
}

/// A finite marginal, as bit patterns.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarginalWire {
    /// Atom values.
    pub values: Vec<u64>,
    /// Atom masses.
    pub masses: Vec<u64>,
}

impl MarginalWire {
    /// A marginal of equal-mass draws.
    #[must_use]
    pub fn from_draws(draws: &[f64]) -> Self {
        let mass = 1.0 / draws.len().max(1) as f64;
        Self {
            values: draws.iter().map(|v| v.to_bits()).collect(),
            masses: vec![mass.to_bits(); draws.len()],
        }
    }

    fn marginal(&self) -> Result<Marginal, Error> {
        Marginal::from_weighted(&floats(&self.values), &floats(&self.masses)).map_err(math_error)
    }
}

/// What an envelope over every coupling is asked for.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum EnvelopeQueryWire {
    /// `P(X + Y <= t)`.
    SumCdf {
        /// Threshold bits.
        threshold: u64,
    },
    /// `P(X - Y <= t)`.
    DifferenceCdf {
        /// Threshold bits.
        threshold: u64,
    },
    /// `E[X Y]`.
    ProductMoment,
    /// `Cov(X, Y)`.
    Covariance,
    /// `Var(a X + b Y)`.
    LinearVariance {
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
    },
    /// `E[g(X, Y)]` for a closed value functional.
    Functional {
        /// The functional's stable name.
        functional: String,
    },
}

/// The numbers a computation consumed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum PropagationRequestWire {
    /// `sum a_i X_i` from means and a covariance (delta method).
    LinearCombination {
        /// Coefficient bits.
        coefficients: Vec<u64>,
        /// Mean bits.
        means: Vec<u64>,
        /// Row-major covariance bits.
        covariance: Vec<u64>,
    },
    /// `a X + b Y` from aligned draws.
    AlignedLinearCombination {
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// `E[g(X, Y)]` from aligned draws.
    AlignedFunctional {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// The spread of the rows `g(x_i, y_i)` of aligned draws (spread reading).
    AlignedFunctionalSpread {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
    },
    /// `P(g(X, Y) > threshold)` from aligned draws with its Monte Carlo band
    /// and a verdict against a probability cutoff (spread reading).
    AlignedExceedance {
        /// The functional's stable name.
        functional: String,
        /// Draws of `X`, bits.
        x: Vec<u64>,
        /// Draws of `Y`, bits, row-aligned with `x`.
        y: Vec<u64>,
        /// Threshold of the composed value, bits.
        threshold: u64,
        /// Probability cutoff of the verdict, bits.
        cutoff: u64,
        /// Level of the Monte Carlo band, bits.
        band_level: u64,
    },
    /// `P(X > threshold)` of a finite mixture of draw sets with known weights
    /// (spread reading).
    MixtureExceedance {
        /// Weight bits.
        weights: Vec<u64>,
        /// Draws of each component, bits.
        components: Vec<Vec<u64>>,
        /// Threshold, bits.
        threshold: u64,
    },
    /// A finite mixture with known weights.
    Mixture {
        /// Weight bits.
        weights: Vec<u64>,
        /// Component mean bits.
        means: Vec<u64>,
        /// Component variance bits.
        variances: Vec<u64>,
    },
    /// A Fréchet-Hoeffding envelope of two marginals.
    FrechetEnvelope {
        /// The query.
        query: EnvelopeQueryWire,
        /// Marginal of `X`.
        x: MarginalWire,
        /// Marginal of `Y`.
        y: MarginalWire,
    },
    /// Ranges over the declared correlation range (in the rule).
    CorrelationRange {
        /// Mean of `X`, bits.
        x_mean: u64,
        /// Standard deviation of `X`, bits.
        x_sd: u64,
        /// Mean of `Y`, bits.
        y_mean: u64,
        /// Standard deviation of `Y`, bits.
        y_sd: u64,
        /// Coefficient of `X`, bits.
        a: u64,
        /// Coefficient of `Y`, bits.
        b: u64,
    },
    /// The pooled estimate and variance over the declared correlation range.
    PooledCorrelationRange {
        /// Study estimate bits.
        estimates: Vec<u64>,
        /// Study standard error bits.
        standard_errors: Vec<u64>,
    },
}

/// The versioned artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointLawPropagationArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The rule that licensed the computation.
    pub rule: PropagationRuleWire,
    /// The inputs.
    pub inputs: Vec<PropagationInputWire>,
    /// The numbers consumed.
    pub request: PropagationRequestWire,
    /// The route the rule yielded.
    pub route: String,
    /// The label of the result.
    pub label: String,
    /// The alignment identity the rows share, for an aligned route.
    pub alignment_id: Option<String>,
    /// The covariance declaration identity, for a covariance route.
    pub covariance_id: Option<String>,
    /// Evidence identities shared by two or more inputs.
    pub shared_evidence: Vec<String>,
    /// The stored result (floats as bit patterns, counts as integers). A
    /// version 1 artifact stored a list of named entries; it is mapped into
    /// this body on decode.
    #[serde(deserialize_with = "deserialize_result")]
    pub result: PropagationResultWire,
    /// Digest of everything above except the result.
    pub identity: String,
}

/// A consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedJointLawPropagation {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The operation.
    pub operation: String,
    /// The route.
    pub route: String,
    /// The label of the result.
    pub label: String,
    /// The alignment identity, for an aligned route.
    pub alignment_id: Option<String>,
    /// The replayed result, bit-identical to the stored one.
    pub result: PropagationResultWire,
}

impl ConsumedJointLawPropagation {
    /// A named float value.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.result.value(name)
    }

    /// Every named float value in stored order.
    #[must_use]
    pub fn values(&self) -> Vec<(&'static str, f64)> {
        self.result.float_entries().into_iter().map(|(n, b)| (n, f64::from_bits(b))).collect()
    }

    /// The number of aligned rows, for an aligned route.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        self.result.n_rows()
    }

    /// The decision verdict of a composed exceedance decision.
    #[must_use]
    pub fn verdict(&self) -> Option<DecisionVerdict> {
        self.result.verdict()
    }

    /// Whether the result is an envelope (a structural envelope or an
    /// assumption range) and so is not a point.
    #[must_use]
    pub fn is_envelope(&self) -> bool {
        self.label != antecedent_contract::POINT_ONLY_LABEL
            && self.label != antecedent_contract::UNCALIBRATED_SPREAD_LABEL
    }
}

// ----------------------------------------------------------- result

/// A lower and an upper bound, as bit patterns.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundsWire {
    /// Lower bound bits.
    pub lower: u64,
    /// Upper bound bits.
    pub upper: u64,
}

impl From<ValueBounds> for BoundsWire {
    fn from(bounds: ValueBounds) -> Self {
        Self { lower: bounds.lower.to_bits(), upper: bounds.upper.to_bits() }
    }
}

/// The quantity a single envelope bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvelopeQuantityWire {
    /// `P(X + Y <= t)`.
    SumCdf,
    /// `P(X - Y <= t)`.
    DifferenceCdf,
    /// `E[X Y]`.
    ProductMoment,
    /// `Cov(X, Y)`.
    Covariance,
    /// `Var(a X + b Y)`.
    LinearVariance,
    /// `E[g(X, Y)]` for a closed value functional.
    Functional,
}

impl EnvelopeQuantityWire {
    const ALL: [Self; 6] = [
        Self::SumCdf,
        Self::DifferenceCdf,
        Self::ProductMoment,
        Self::Covariance,
        Self::LinearVariance,
        Self::Functional,
    ];

    /// The stable name of the lower bound.
    #[must_use]
    pub const fn lower_name(self) -> &'static str {
        match self {
            Self::SumCdf => "sum_cdf_lower",
            Self::DifferenceCdf => "difference_cdf_lower",
            Self::ProductMoment => "product_moment_lower",
            Self::Covariance => "covariance_lower",
            Self::LinearVariance => "linear_variance_lower",
            Self::Functional => "functional_lower",
        }
    }

    /// The stable name of the upper bound.
    #[must_use]
    pub const fn upper_name(self) -> &'static str {
        match self {
            Self::SumCdf => "sum_cdf_upper",
            Self::DifferenceCdf => "difference_cdf_upper",
            Self::ProductMoment => "product_moment_upper",
            Self::Covariance => "covariance_upper",
            Self::LinearVariance => "linear_variance_upper",
            Self::Functional => "functional_upper",
        }
    }
}

/// The stored verdict of a composed exceedance decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictWire {
    /// Act.
    Act,
    /// Withhold.
    Withhold,
    /// Unresolved.
    Unresolved,
}

impl From<DecisionVerdict> for VerdictWire {
    fn from(verdict: DecisionVerdict) -> Self {
        match verdict {
            DecisionVerdict::Act => Self::Act,
            DecisionVerdict::Withhold => Self::Withhold,
            DecisionVerdict::Unresolved => Self::Unresolved,
        }
    }
}

impl From<VerdictWire> for DecisionVerdict {
    fn from(verdict: VerdictWire) -> Self {
        match verdict {
            VerdictWire::Act => Self::Act,
            VerdictWire::Withhold => Self::Withhold,
            VerdictWire::Unresolved => Self::Unresolved,
        }
    }
}

/// The stored result of a propagation: floats as IEEE bit patterns, counts as
/// integers, the verdict as its own enum.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum PropagationResultWire {
    /// Mean and variance of a linear combination under a covariance.
    Moments {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
    },
    /// Moments of a linear combination of aligned draws.
    AlignedMoments {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Covariance bits.
        covariance: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// Mean of a functional over aligned draws.
    AlignedMean {
        /// Mean bits.
        mean: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// Spread of a functional over aligned draws.
    AlignedSpread {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Monte Carlo standard error of the mean, bits.
        mean_standard_error: u64,
        /// Aligned row count.
        n_rows: u64,
    },
    /// An exceedance probability over aligned draws with its verdict.
    AlignedExceedance {
        /// Probability bits.
        probability: u64,
        /// Standard error bits.
        standard_error: u64,
        /// Monte Carlo band bits.
        band: u64,
        /// The verdict against the cutoff.
        verdict: VerdictWire,
        /// Aligned row count.
        n_rows: u64,
    },
    /// An exceedance probability of a finite mixture.
    MixtureExceedance {
        /// Probability bits.
        probability: u64,
        /// Standard error bits.
        standard_error: u64,
    },
    /// Moments of a finite mixture.
    Mixture {
        /// Mean bits.
        mean: u64,
        /// Variance bits.
        variance: u64,
        /// Within-component variance bits.
        within: u64,
        /// Between-component variance bits.
        between: u64,
    },
    /// A Fréchet-Hoeffding envelope of one quantity.
    Envelope {
        /// The bounded quantity.
        quantity: EnvelopeQuantityWire,
        /// Its bounds.
        bounds: BoundsWire,
    },
    /// Ranges over a declared correlation range.
    CorrelationRange {
        /// Bounds of `E[X Y]`.
        product_moment: BoundsWire,
        /// Bounds of `Cov(X, Y)`.
        covariance: BoundsWire,
        /// Bounds of `Var(a X + b Y)`.
        linear_variance: BoundsWire,
    },
    /// The pooled estimate and variance over a declared correlation range.
    PooledRange {
        /// Bounds of the pooled estimate.
        estimate: BoundsWire,
        /// Bounds of the pooled variance.
        variance: BoundsWire,
    },
}

impl PropagationResultWire {
    /// Every float entry by its stable name, in stored order.
    #[must_use]
    pub fn float_entries(&self) -> Vec<(&'static str, u64)> {
        match *self {
            Self::Moments { mean, variance } => vec![("mean", mean), ("variance", variance)],
            Self::AlignedMoments { mean, variance, covariance, .. } => {
                vec![("mean", mean), ("variance", variance), ("covariance", covariance)]
            }
            Self::AlignedMean { mean, .. } => vec![("mean", mean)],
            Self::AlignedSpread { mean, variance, mean_standard_error, .. } => {
                vec![
                    ("mean", mean),
                    ("variance", variance),
                    ("mean_standard_error", mean_standard_error),
                ]
            }
            Self::AlignedExceedance { probability, standard_error, band, .. } => {
                vec![
                    ("probability", probability),
                    ("standard_error", standard_error),
                    ("band", band),
                ]
            }
            Self::MixtureExceedance { probability, standard_error } => {
                vec![("probability", probability), ("standard_error", standard_error)]
            }
            Self::Mixture { mean, variance, within, between } => vec![
                ("mean", mean),
                ("variance", variance),
                ("within", within),
                ("between", between),
            ],
            Self::Envelope { quantity, bounds } => {
                vec![(quantity.lower_name(), bounds.lower), (quantity.upper_name(), bounds.upper)]
            }
            Self::CorrelationRange { product_moment, covariance, linear_variance } => {
                let mut out = Vec::with_capacity(6);
                for (quantity, bounds) in [
                    (EnvelopeQuantityWire::ProductMoment, product_moment),
                    (EnvelopeQuantityWire::Covariance, covariance),
                    (EnvelopeQuantityWire::LinearVariance, linear_variance),
                ] {
                    out.push((quantity.lower_name(), bounds.lower));
                    out.push((quantity.upper_name(), bounds.upper));
                }
                out
            }
            Self::PooledRange { estimate, variance } => vec![
                ("pooled_estimate_lower", estimate.lower),
                ("pooled_estimate_upper", estimate.upper),
                ("pooled_variance_lower", variance.lower),
                ("pooled_variance_upper", variance.upper),
            ],
        }
    }

    /// A named float value.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.float_entries().into_iter().find(|(n, _)| *n == name).map(|(_, b)| f64::from_bits(b))
    }

    /// The aligned row count, for a result over aligned draws.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        match *self {
            Self::AlignedMoments { n_rows, .. }
            | Self::AlignedMean { n_rows, .. }
            | Self::AlignedSpread { n_rows, .. }
            | Self::AlignedExceedance { n_rows, .. } => Some(n_rows),
            _ => None,
        }
    }

    /// The decision verdict, for an exceedance decision.
    #[must_use]
    pub fn verdict(&self) -> Option<DecisionVerdict> {
        match self {
            Self::AlignedExceedance { verdict, .. } => Some((*verdict).into()),
            _ => None,
        }
    }

    /// The entries in the earlier named-list layout: floats as bit patterns,
    /// `n_rows` and `verdict_code` as integers.
    #[must_use]
    pub fn named_entries(&self) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> =
            self.float_entries().into_iter().map(|(n, b)| (n.to_owned(), b)).collect();
        if let Some(rows) = self.n_rows() {
            out.push(("n_rows".to_owned(), rows));
        }
        if let Some(verdict) = self.verdict() {
            // The earlier layout placed the verdict code before the row count.
            let at = out.len() - usize::from(self.n_rows().is_some());
            out.insert(at, ("verdict_code".to_owned(), verdict_code(verdict)));
        }
        out
    }

    /// Map the earlier named-list layout into the typed result. The entry
    /// names and their order identify the shape exactly as the earlier writer
    /// produced them.
    fn from_named_entries(entries: &[(String, u64)]) -> Result<Self, String> {
        let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
        let bits: Vec<u64> = entries.iter().map(|(_, b)| *b).collect();
        let bounds = |at: usize| BoundsWire { lower: bits[at], upper: bits[at + 1] };
        let shape = |expected: &[&str]| names == expected;
        if shape(&["mean", "variance"]) {
            return Ok(Self::Moments { mean: bits[0], variance: bits[1] });
        }
        if shape(&["mean", "variance", "covariance", "n_rows"]) {
            return Ok(Self::AlignedMoments {
                mean: bits[0],
                variance: bits[1],
                covariance: bits[2],
                n_rows: bits[3],
            });
        }
        if shape(&["mean", "n_rows"]) {
            return Ok(Self::AlignedMean { mean: bits[0], n_rows: bits[1] });
        }
        if shape(&["mean", "variance", "mean_standard_error", "n_rows"]) {
            return Ok(Self::AlignedSpread {
                mean: bits[0],
                variance: bits[1],
                mean_standard_error: bits[2],
                n_rows: bits[3],
            });
        }
        if shape(&["probability", "standard_error", "band", "verdict_code", "n_rows"]) {
            let verdict = [
                DecisionVerdict::Act,
                DecisionVerdict::Withhold,
                DecisionVerdict::Unresolved,
            ]
            .into_iter()
            .find(|verdict| verdict_code(*verdict) == bits[3])
            .ok_or_else(|| "unknown stored verdict code".to_owned())?;
            return Ok(Self::AlignedExceedance {
                probability: bits[0],
                standard_error: bits[1],
                band: bits[2],
                verdict: verdict.into(),
                n_rows: bits[4],
            });
        }
        if shape(&["probability", "standard_error"]) {
            return Ok(Self::MixtureExceedance { probability: bits[0], standard_error: bits[1] });
        }
        if shape(&["mean", "variance", "within", "between"]) {
            return Ok(Self::Mixture {
                mean: bits[0],
                variance: bits[1],
                within: bits[2],
                between: bits[3],
            });
        }
        if let Some(quantity) = EnvelopeQuantityWire::ALL
            .into_iter()
            .find(|q| shape(&[q.lower_name(), q.upper_name()]))
        {
            return Ok(Self::Envelope { quantity, bounds: bounds(0) });
        }
        if shape(&[
            "product_moment_lower",
            "product_moment_upper",
            "covariance_lower",
            "covariance_upper",
            "linear_variance_lower",
            "linear_variance_upper",
        ]) {
            return Ok(Self::CorrelationRange {
                product_moment: bounds(0),
                covariance: bounds(2),
                linear_variance: bounds(4),
            });
        }
        if shape(&[
            "pooled_estimate_lower",
            "pooled_estimate_upper",
            "pooled_variance_lower",
            "pooled_variance_upper",
        ]) {
            return Ok(Self::PooledRange { estimate: bounds(0), variance: bounds(2) });
        }
        Err("stored result entries do not form a known result".to_owned())
    }
}

/// Read the result of either layout: the typed body, or the earlier list of
/// named entries mapped into it.
fn deserialize_result<'de, D>(deserializer: D) -> Result<PropagationResultWire, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stored {
        Typed(PropagationResultWire),
        Named(Vec<(String, u64)>),
    }
    match Stored::deserialize(deserializer)? {
        Stored::Typed(result) => Ok(result),
        Stored::Named(entries) => {
            if entries.len() > MAX_NAMED_RESULT_ENTRIES {
                return Err(serde::de::Error::custom("too many stored result entries"));
            }
            PropagationResultWire::from_named_entries(&entries).map_err(serde::de::Error::custom)
        }
    }
}

/// Most entries the earlier named-list layout may carry.
const MAX_NAMED_RESULT_ENTRIES: usize = 16;

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

fn check_bounds(wire: &JointLawPropagationArtifactWire) -> Result<(), Error> {
    let too_large = Error::LimitsExceeded;
    if wire.inputs.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS {
        return Err(too_large("inputs"));
    }
    let long = |text: &str| text.len() > MAX_JOINT_LAW_PROPAGATION_TEXT_BYTES;
    let long_opt = |text: &Option<String>| text.as_deref().is_some_and(long);
    if wire.shared_evidence.len() > MAX_JOINT_LAW_PROPAGATION_SHARED_EVIDENCE
        || wire.required_features.len() > 16
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
        if n > MAX_DISTRIBUTION_DRAWS { Err(too_large("draws")) } else { Ok(()) }
    };
    match &wire.request {
        PropagationRequestWire::LinearCombination { coefficients, means, covariance } => {
            if coefficients.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || means.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || covariance.len()
                    > JOINT_LAW_PROPAGATION_MAX_INPUTS * JOINT_LAW_PROPAGATION_MAX_INPUTS
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
            if weights.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || components.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
            {
                return Err(too_large("mixture components"));
            }
            let total: usize = components.iter().map(Vec::len).sum();
            draws(total)?;
        }
        PropagationRequestWire::Mixture { weights, means, variances } => {
            if weights.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || means.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || variances.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
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
            if estimates.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
                || standard_errors.len() > JOINT_LAW_PROPAGATION_MAX_INPUTS
            {
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
        (PropagationRoute::LinearMoments, W::LinearCombination { .. })
        | (PropagationRoute::AlignedJointLaw, W::AlignedLinearCombination { .. }) => linear,
        (PropagationRoute::AlignedJointLaw, W::AlignedFunctional { .. }) => nonlinear,
        // The draw-based spread and decision numbers are a spread reading's
        // own: a point reading is not given a spread under its own label.
        (
            PropagationRoute::AlignedJointLaw,
            W::AlignedFunctionalSpread { .. } | W::AlignedExceedance { .. },
        ) => spread && nonlinear,
        (PropagationRoute::KnownWeightMixture, W::MixtureExceedance { .. }) => {
            spread && operation == Op::Mixture
        }
        (PropagationRoute::KnownWeightMixture, W::Mixture { .. }) => operation == Op::Mixture,
        (PropagationRoute::FrechetEnvelope, W::FrechetEnvelope { query, .. }) => match query {
            EnvelopeQueryWire::SumCdf { .. }
            | EnvelopeQueryWire::DifferenceCdf { .. }
            | EnvelopeQueryWire::LinearVariance { .. } => {
                matches!(operation, Op::Sum | Op::Contrast | Op::WeightedSum)
            }
            EnvelopeQueryWire::ProductMoment | EnvelopeQueryWire::Covariance => {
                operation == Op::Product
            }
            EnvelopeQueryWire::Functional { .. } => {
                matches!(operation, Op::Product | Op::NonlinearFunctional)
            }
        },
        (
            PropagationRoute::CorrelationRangeEnvelope,
            W::CorrelationRange { .. } | W::PooledCorrelationRange { .. },
        ) => true,
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
        EnvelopeQueryWire::ProductMoment => {
            (Quantity::ProductMoment, product_moment_envelope(&mx, &my, cancel).map_err(math_error)?)
        }
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
            (0..n_inputs)
                .any(|j| i != j && f64::from_bits(covariance[i * n_inputs + j]).abs() > 0.0)
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
            "supply the numbers the licensed route consumes (a covariance for the linear route, aligned draws for the aligned route, weights for a mixture, two marginals for an envelope); a chained transformation is checked by the rule but its stages run in the caller",
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
    /// remedy), numbers that do not fit the licensed route, a failed
    /// computation, cancellation or an encoding failure.
    pub fn seal(
        declared: &PropagationRequest,
        request: PropagationRequestWire,
        ctx: &ExecutionContext,
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
        check_bounds(&wire)?;
        Ok(wire)
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, Error> {
        if self.version != JOINT_LAW_PROPAGATION_ARTIFACT_VERSION {
            return Err(Error::UnsupportedVersion { version: self.version });
        }
        check_bounds(self)?;
        crate::to_cbor(self).map_err(|e| Error::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES {
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
        check_bounds(&wire)?;
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
        if !is_supported_version(self.version) {
            return Err(Error::UnsupportedVersion { version: self.version });
        }
        if self.required_features != [JOINT_LAW_PROPAGATION_ARTIFACT_FEATURE] {
            return Err(Error::UnsupportedSemantics("required features"));
        }
        check_bounds(self)?;
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
