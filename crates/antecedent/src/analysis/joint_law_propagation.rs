//! Composition-aware uncertainty: propagate uncertain inputs through a
//! combination only under the dependence information the combination needs.
//!
//! One rule decides every composition (`antecedent_contract::check_propagation`)
//! and this module routes a request through it and adds none of its own:
//!
//! 1. **Check** the declared operation, inputs and dependence. Linear
//!    operations (a transform, contrast, sum, weighted sum) take a covariance
//!    or distinct evidence; products, nonlinear functionals and chained
//!    transformations need an aligned joint law (draws that carry one
//!    alignment identity); mixtures need known weights; a transported interval
//!    composed with another claim needs an aligned law or a covariance.
//!    Unrelated marginal draws are never paired by index.
//! 2. **Execute** the licensed route in `antecedent_stats::joint_law`: the
//!    delta method for a linear map, the aligned expectation, the law of total
//!    variance for a mixture, or a dependence envelope.
//! 3. **Export** seals the declaration, the numbers and the result into
//!    `joint_law_propagation_v1`; **consume** re-runs the rule and replays the
//!    result to identical bits.
//!
//! # What a value means
//!
//! A value computed under a licensed dependence route is a labelled point
//! value (`point_only`): no interval and no posterior is produced. A
//! dependence envelope is a `structural_envelope` (the set of values no
//! coupling of the marginals could exceed) or, for a declared correlation
//! range, an `assumption_range`. It is never a point, interval or posterior,
//! and it never stands in for an aligned joint law.
//! [`PropagationClaim::interval`] and [`PropagationClaim::posterior`] refuse
//! with the stated remedy.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    ExternalRefusal, InputAlignment, InputRole, JOINT_LAW_STAGE, PropagationDependence,
    PropagationInput, PropagationOperation, PropagationRequest, RequestedReading,
    check_propagation,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_io::distribution_artifact::{DistributionArtifact, DrawAlignment};
use antecedent_io::joint_law_propagation_artifact::{
    ConsumedJointLawPropagation, JointLawPropagationArtifactError, JointLawPropagationArtifactWire,
    PropagationRequestWire, float_bits,
};

pub use antecedent_io::joint_law_propagation_artifact::{EnvelopeQueryWire, MarginalWire};
pub use antecedent_stats::ValueFunctional;

/// The refusals of a propagation request: each with a registered reason code,
/// a namespaced `joint_law_propagation.*` detail, the offending subject and a
/// remedy that names the dependence information that would license it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointLawPropagationRefusal {
    /// Every refusal. Never empty.
    pub refusals: Vec<ExternalRefusal>,
}

impl JointLawPropagationRefusal {
    fn one(refusal: ExternalRefusal) -> Self {
        Self { refusals: vec![refusal] }
    }

    fn invalid(offending: &str, remedy: &'static str) -> Self {
        Self::one(
            ExternalRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                JOINT_LAW_STAGE,
                "joint_law_propagation.invalid_request",
            )
            .with_offending(offending.to_owned())
            .with_remedy(remedy),
        )
    }

    /// The code of the first refusal.
    #[must_use]
    pub fn code(&self) -> &'static str {
        self.refusals.first().map_or(antecedent_core::reason_code!("invalid_argument"), |r| r.code)
    }

    /// The namespaced detail of every refusal.
    #[must_use]
    pub fn details(&self) -> Vec<&str> {
        self.refusals.iter().map(|r| r.detail.as_str()).collect()
    }
}

impl std::fmt::Display for JointLawPropagationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusals.first() {
            Some(first) => {
                write!(
                    f,
                    "{}{}: {}",
                    antecedent_core::reason_code::PREFIX,
                    first.code,
                    first.detail
                )?;
                if let Some(subject) = &first.offending {
                    write!(f, " ({subject})")?;
                }
                if let Some(remedy) = first.remedy {
                    write!(f, "; {remedy}")?;
                }
                Ok(())
            }
            None => write!(f, "joint-law propagation refused"),
        }
    }
}

impl std::error::Error for JointLawPropagationRefusal {}

impl From<JointLawPropagationArtifactError> for JointLawPropagationRefusal {
    fn from(error: JointLawPropagationArtifactError) -> Self {
        Self { refusals: error.to_refusals() }
    }
}

type Refusal = JointLawPropagationRefusal;

// ------------------------------------------------------------------ inputs

/// The draws of one uncertain input with the identities the rule reads: the
/// evidence it rests on, its role in the composition and how its rows are
/// aligned with other inputs'.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagationDraws {
    id: String,
    evidence_id: String,
    role: InputRole,
    alignment: InputAlignment,
    values: Vec<f64>,
}

/// The alignment identity of a joint-law artifact: a digest of the study,
/// provider, random stream, snapshot, checked contract and draw count. Two
/// columns of one artifact share it; two artifacts never do, so rows of
/// different artifacts are never paired.
///
/// # Errors
/// An artifact whose draws are independent marginals has no alignment
/// identity: its rows are not realizations of one joint law.
pub fn alignment_identity(artifact: &DistributionArtifact) -> Result<String, Refusal> {
    let identity = &artifact.metadata().identity;
    if identity.alignment != DrawAlignment::Joint {
        return Err(Refusal::one(
            ExternalRefusal::new(
                antecedent_core::reason_code!("joint_law_required"),
                JOINT_LAW_STAGE,
                "joint_law_propagation.index_pairing_refused",
            )
            .with_offending(identity.source_id.clone())
            .with_remedy(
                "the artifact holds independently sampled marginals whose rows are not paired; supply the draws of one joint law, or ask for the dependence envelope",
            ),
        ));
    }
    antecedent_io::digest_hex(
        IdentityDomain::Program,
        &(
            "joint_law_alignment_v1",
            &identity.source_id,
            &identity.provider_id,
            &identity.rng_id,
            &identity.snapshot_id,
            &identity.causal_contract_id,
            artifact.n_draws() as u64,
        ),
    )
    .map_err(|e| Refusal::invalid(&e.to_string(), "the alignment identity could not be encoded"))
}

/// The alignment of an artifact's draws as the propagation rule reads it: the
/// artifact's pairing identity ([`alignment_identity`]) for draws of one joint
/// law, and independent marginals otherwise.
///
/// This is the one mapping from an artifact's declared alignment to the rule's
/// input alignment; callers that pair draws of several artifacts compare these
/// values (through `check_rows_pairable_across`) rather than a constant.
///
/// # Errors
/// A joint artifact whose pairing identity cannot be encoded.
pub fn input_alignment(artifact: &DistributionArtifact) -> Result<InputAlignment, Refusal> {
    match artifact.metadata().identity.alignment {
        DrawAlignment::Joint => {
            Ok(InputAlignment::Aligned { alignment_id: alignment_identity(artifact)? })
        }
        DrawAlignment::IndependentMarginals => Ok(InputAlignment::IndependentMarginals),
    }
}

impl PropagationDraws {
    /// One column of a distribution artifact. The evidence identity is the
    /// artifact's data snapshot; the alignment follows the artifact's declared
    /// alignment.
    ///
    /// # Errors
    /// A column outside the artifact, a weighted artifact or a masked
    /// (unsupported) coordinate.
    pub fn from_artifact_column(
        id: impl Into<String>,
        artifact: &DistributionArtifact,
        column: usize,
    ) -> Result<Self, Refusal> {
        let id = id.into();
        let [rows, width] = artifact.shape();
        if column >= width {
            return Err(Refusal::invalid(&id, "name a column of the artifact"));
        }
        let metadata = artifact.metadata();
        if metadata.weights.is_some() {
            return Err(Refusal::invalid(
                &id,
                "weighted draws have no equal-row alignment in this route; supply equal-weight draws",
            ));
        }
        if metadata
            .supported
            .as_ref()
            .is_some_and(|mask| !mask.get(column).copied().unwrap_or(false))
        {
            return Err(Refusal::invalid(&id, "the coordinate is unsupported in the artifact"));
        }
        let alignment = input_alignment(artifact)?;
        let values = (0..rows).map(|row| artifact.draws()[row * width + column]).collect();
        Ok(Self {
            id,
            evidence_id: metadata.identity.snapshot_id.clone(),
            role: InputRole::Estimate,
            alignment,
            values,
        })
    }

    /// Independently produced draws of one marginal with no joint law.
    #[must_use]
    pub fn independent_marginal(
        id: impl Into<String>,
        evidence_id: impl Into<String>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_id: evidence_id.into(),
            role: InputRole::Estimate,
            alignment: InputAlignment::IndependentMarginals,
            values,
        }
    }

    /// Draws that the caller states are rows of the joint law with this
    /// alignment identity (obtained from [`alignment_identity`] or supplied by
    /// the provider of the joint law).
    #[must_use]
    pub fn aligned(
        id: impl Into<String>,
        evidence_id: impl Into<String>,
        alignment_id: impl Into<String>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            id: id.into(),
            evidence_id: evidence_id.into(),
            role: InputRole::Estimate,
            alignment: InputAlignment::Aligned { alignment_id: alignment_id.into() },
            values,
        }
    }

    /// The same input resting on the evidence-ledger source `source`, so a
    /// check that has the ledger (`check_propagation_with_ledger`) decides its
    /// overlap with other inputs from their observations rather than from the
    /// equality of snapshot identities.
    #[must_use]
    pub fn with_evidence_source(
        mut self,
        source: antecedent_contract::evidence_ledger::SourceId,
    ) -> Self {
        self.evidence_id = source.to_hex();
        self
    }

    /// The same input with another role in the composition (for the pieces of
    /// a transported claim).
    #[must_use]
    pub fn with_role(mut self, role: InputRole) -> Self {
        self.role = role;
        self
    }

    /// The input's identity.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The draws.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn as_input(&self) -> PropagationInput {
        PropagationInput {
            id: self.id.clone(),
            evidence_id: self.evidence_id.clone(),
            role: self.role,
            alignment: self.alignment.clone(),
        }
    }
}

/// A summary-only input: a mean (its variance is on the covariance diagonal).
#[derive(Clone, Debug, PartialEq)]
pub struct MomentInput {
    /// Input identity.
    pub id: String,
    /// Identity of the evidence the input rests on.
    pub evidence_id: String,
    /// Role in the composition.
    pub role: InputRole,
    /// The mean.
    pub mean: f64,
}

/// A component of a finite mixture.
#[derive(Clone, Debug, PartialEq)]
pub struct MixtureComponent {
    /// Component identity.
    pub id: String,
    /// Identity of the evidence the component rests on.
    pub evidence_id: String,
    /// The component's known weight.
    pub weight: f64,
    /// The component's mean.
    pub mean: f64,
    /// The component's variance.
    pub variance: f64,
}

/// The dependence information of a linear combination of summaries.
#[derive(Clone, Debug, PartialEq)]
pub enum LinearDependence {
    /// Declared independent: the supplied covariance must be diagonal and the
    /// inputs' evidence distinct.
    DeclaredIndependent,
    /// An explicit joint moment: the supplied covariance includes the cross
    /// terms; `id` names the declaration or the shared evidence behind it.
    Covariance {
        /// Identity of the covariance declaration.
        id: String,
    },
}

// ------------------------------------------------------------------ claims

/// A composed uncertain quantity: the licensed route, the labelled value and
/// the sealed artifact that replays it.
#[derive(Clone, Debug, PartialEq)]
pub struct PropagationClaim {
    declared: PropagationRequest,
    artifact: JointLawPropagationArtifactWire,
}

impl PropagationClaim {
    /// The identity of the declaration, numbers and plan.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.artifact.identity
    }

    /// The operation's stable name.
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.artifact.rule.operation
    }

    /// The route that ran.
    #[must_use]
    pub fn route(&self) -> &str {
        &self.artifact.route
    }

    /// The label of the result: `point_only`, `structural_envelope` or
    /// `assumption_range`.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.artifact.label
    }

    /// Whether the result is an envelope and therefore not a point.
    #[must_use]
    pub fn is_envelope(&self) -> bool {
        self.artifact.label != antecedent_contract::POINT_ONLY_LABEL
    }

    /// The alignment identity the rows share, for an aligned route.
    #[must_use]
    pub fn alignment_id(&self) -> Option<&str> {
        self.artifact.alignment_id.as_deref()
    }

    /// A named float value (`mean`, `variance`, `lower`/`upper` pairs ...).
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        if name == "n_rows" {
            return None;
        }
        self.artifact.result.iter().find(|(n, _)| n == name).map(|(_, b)| f64::from_bits(*b))
    }

    /// The number of aligned rows, for an aligned route.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        self.artifact.result.iter().find(|(n, _)| n == "n_rows").map(|(_, v)| *v)
    }

    /// Every named float value in stored order.
    #[must_use]
    pub fn values(&self) -> Vec<(String, f64)> {
        self.artifact
            .result
            .iter()
            .filter(|(n, _)| n != "n_rows")
            .map(|(n, b)| (n.clone(), f64::from_bits(*b)))
            .collect()
    }

    /// Encode the sealed artifact.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, Refusal> {
        self.artifact.export().map_err(Refusal::from)
    }

    /// An interval for the composed quantity. Always refused: no composed
    /// coordinate has a measured coverage record, and a component interval
    /// does not cover the composition.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with the stated remedy.
    pub fn interval(&self) -> Result<(), Refusal> {
        self.closed(RequestedReading::Interval)
    }

    /// A posterior statement for the composed quantity. Always refused.
    ///
    /// # Errors
    /// Always `cell_not_licensed` with the stated remedy.
    pub fn posterior(&self) -> Result<(), Refusal> {
        self.closed(RequestedReading::Posterior)
    }

    fn closed(&self, reading: RequestedReading) -> Result<(), Refusal> {
        let request = PropagationRequest { reading, ..self.declared.clone() };
        match check_propagation(&request) {
            Err(refusal) => Err(Refusal::one(refusal)),
            Ok(_) => Err(Refusal::invalid(
                reading.as_str(),
                "the interval and posterior readings of a composed quantity are closed",
            )),
        }
    }
}

// ------------------------------------------------------------- execution

/// Check `declared` against the propagation rule, execute `numbers` and seal
/// the claim.
///
/// # Errors
/// A refusal of the rule (with its registered code and remedy), numbers that do
/// not fit the licensed route, a failed computation or cancellation.
pub fn propagate(
    declared: &PropagationRequest,
    numbers: PropagationRequestWire,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let artifact = JointLawPropagationArtifactWire::seal(declared, numbers, ctx)?;
    Ok(PropagationClaim { declared: declared.clone(), artifact })
}

fn paired(x: &PropagationDraws, y: &PropagationDraws) -> Result<(), Refusal> {
    if x.values.len() == y.values.len() {
        Ok(())
    } else {
        Err(Refusal::invalid(
            &x.id,
            "aligned columns have one value per joint realization; the draw counts differ",
        ))
    }
}

/// `E[g(X, Y)]` of a closed value functional under the aligned joint law of
/// two inputs. A product or a nonlinear functional needs the aligned law; the
/// draws of independent or unrelated marginals are refused.
///
/// # Errors
/// As [`propagate`].
pub fn propagate_aligned_functional(
    operation: PropagationOperation,
    x: &PropagationDraws,
    y: &PropagationDraws,
    functional: ValueFunctional,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = PropagationRequest {
        operation,
        inputs: vec![x.as_input(), y.as_input()],
        dependence: PropagationDependence::AlignedJointLaw,
        reading: RequestedReading::Point,
    };
    check_propagation(&declared).map_err(Refusal::one)?;
    paired(x, y)?;
    propagate(
        &declared,
        PropagationRequestWire::AlignedFunctional {
            functional: functional.as_str().to_owned(),
            x: float_bits(&x.values),
            y: float_bits(&y.values),
        },
        ctx,
    )
}

/// Mean and variance of `a X + b Y` from aligned draws (covariance from the
/// joint law).
///
/// # Errors
/// As [`propagate`].
pub fn propagate_aligned_linear(
    operation: PropagationOperation,
    coefficients: (f64, f64),
    x: &PropagationDraws,
    y: &PropagationDraws,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = PropagationRequest {
        operation,
        inputs: vec![x.as_input(), y.as_input()],
        dependence: PropagationDependence::AlignedJointLaw,
        reading: RequestedReading::Point,
    };
    check_propagation(&declared).map_err(Refusal::one)?;
    paired(x, y)?;
    propagate(
        &declared,
        PropagationRequestWire::AlignedLinearCombination {
            a: coefficients.0.to_bits(),
            b: coefficients.1.to_bits(),
            x: float_bits(&x.values),
            y: float_bits(&y.values),
        },
        ctx,
    )
}

/// Mean and variance of `sum a_i X_i` from means and a covariance (the delta
/// method, exact for a linear map).
///
/// `covariance` is the row-major covariance including cross terms.
///
/// # Errors
/// As [`propagate`]; also a declared independence over a covariance with
/// nonzero cross terms.
pub fn propagate_linear_moments(
    operation: PropagationOperation,
    inputs: &[MomentInput],
    coefficients: &[f64],
    covariance: &[f64],
    dependence: &LinearDependence,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let k = inputs.len();
    if coefficients.len() != k || covariance.len() != k * k {
        return Err(Refusal::invalid(
            operation.as_str(),
            "give one coefficient per input and a k by k covariance",
        ));
    }
    if matches!(dependence, LinearDependence::DeclaredIndependent) {
        let cross = (0..k).any(|i| (0..k).any(|j| i != j && covariance[i * k + j].abs() > 0.0));
        if cross {
            return Err(Refusal::invalid(
                "covariance",
                "the covariance has cross terms but the inputs are declared independent; declare the covariance",
            ));
        }
    }
    let declared = PropagationRequest {
        operation,
        inputs: inputs
            .iter()
            .map(|m| PropagationInput {
                id: m.id.clone(),
                evidence_id: m.evidence_id.clone(),
                role: m.role,
                alignment: InputAlignment::MomentsOnly,
            })
            .collect(),
        dependence: match dependence {
            LinearDependence::DeclaredIndependent => PropagationDependence::DeclaredIndependent,
            LinearDependence::Covariance { id } => {
                PropagationDependence::Covariance { id: id.clone() }
            }
        },
        reading: RequestedReading::Point,
    };
    let means: Vec<f64> = inputs.iter().map(|m| m.mean).collect();
    propagate(
        &declared,
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(coefficients),
            means: float_bits(&means),
            covariance: float_bits(covariance),
        },
        ctx,
    )
}

/// Mean and variance of a finite mixture with known weights, by the law of
/// total variance. The components need no joint law.
///
/// # Errors
/// As [`propagate`].
pub fn propagate_mixture(
    components: &[MixtureComponent],
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = PropagationRequest {
        operation: PropagationOperation::Mixture,
        inputs: components
            .iter()
            .map(|c| PropagationInput {
                id: c.id.clone(),
                evidence_id: c.evidence_id.clone(),
                role: InputRole::Estimate,
                alignment: InputAlignment::MomentsOnly,
            })
            .collect(),
        dependence: PropagationDependence::KnownWeights,
        reading: RequestedReading::Point,
    };
    let weights: Vec<f64> = components.iter().map(|c| c.weight).collect();
    let means: Vec<f64> = components.iter().map(|c| c.mean).collect();
    let variances: Vec<f64> = components.iter().map(|c| c.variance).collect();
    propagate(
        &declared,
        PropagationRequestWire::Mixture {
            weights: float_bits(&weights),
            means: float_bits(&means),
            variances: float_bits(&variances),
        },
        ctx,
    )
}

/// The structural envelope of a functional of two unpaired marginals: the
/// extremes over every coupling (Fréchet-Hoeffding). Labelled
/// `structural_envelope`; never a point, an interval or a posterior.
///
/// # Errors
/// As [`propagate`]; also a value function whose mixed-difference sign is not
/// verified on the marginals' support.
pub fn propagate_dependence_envelope(
    operation: PropagationOperation,
    query: EnvelopeQueryWire,
    x: &PropagationDraws,
    y: &PropagationDraws,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = PropagationRequest {
        operation,
        inputs: vec![x.as_input(), y.as_input()],
        dependence: PropagationDependence::MarginalsOnly,
        reading: RequestedReading::Envelope,
    };
    propagate(
        &declared,
        PropagationRequestWire::FrechetEnvelope {
            query,
            x: MarginalWire::from_draws(&x.values),
            y: MarginalWire::from_draws(&y.values),
        },
        ctx,
    )
}

/// Moment summaries and a coefficient pair for a declared correlation range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationRangeSummary {
    /// Mean of `X`.
    pub x_mean: f64,
    /// Standard deviation of `X`.
    pub x_sd: f64,
    /// Mean of `Y`.
    pub y_mean: f64,
    /// Standard deviation of `Y`.
    pub y_sd: f64,
    /// Coefficient of `X`.
    pub a: f64,
    /// Coefficient of `Y`.
    pub b: f64,
}

/// Ranges of `E[XY]`, `Cov(X, Y)` and `Var(aX + bY)` over a declared
/// correlation range. Labelled `assumption_range`.
///
/// # Errors
/// As [`propagate`]; also a degenerate or out-of-range correlation range.
pub fn propagate_correlation_range(
    operation: PropagationOperation,
    x: (&str, &str),
    y: (&str, &str),
    summary: CorrelationRangeSummary,
    rho_lo: f64,
    rho_hi: f64,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let input = |(id, evidence): (&str, &str)| PropagationInput {
        id: id.to_owned(),
        evidence_id: evidence.to_owned(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    };
    let declared = PropagationRequest {
        operation,
        inputs: vec![input(x), input(y)],
        dependence: PropagationDependence::CorrelationRange { rho_lo, rho_hi },
        reading: RequestedReading::Envelope,
    };
    propagate(
        &declared,
        PropagationRequestWire::CorrelationRange {
            x_mean: summary.x_mean.to_bits(),
            x_sd: summary.x_sd.to_bits(),
            y_mean: summary.y_mean.to_bits(),
            y_sd: summary.y_sd.to_bits(),
            a: summary.a.to_bits(),
            b: summary.b.to_bits(),
        },
        ctx,
    )
}

/// Verify a sealed artifact and replay it to identical result bits.
///
/// `expected_identity`, when given, is the identity the caller retained
/// independently of the artifact.
///
/// # Errors
/// A malformed or edited artifact, a retained identity that differs, a result
/// that does not replay, or a stored declaration the rule no longer licenses.
pub fn consume_joint_law_propagation_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedJointLawPropagation, Refusal> {
    let wire = JointLawPropagationArtifactWire::decode(bytes)?;
    wire.consume(expected_identity, ctx).map_err(Refusal::from)
}
