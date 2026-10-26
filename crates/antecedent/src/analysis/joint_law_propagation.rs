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
//! A **spread** (a variance, a Monte Carlo standard error, or an exceedance
//! probability with its Monte Carlo band and verdict) is requested through the
//! `*_spread` and `*_exceedance` functions; it is labelled
//! `uncalibrated_spread` and carries no coverage statement. An **interval** at
//! a composed coordinate is licensed only by
//! [`PropagationClaim::check_composed_interval`] with a measured coverage
//! record for exactly that coordinate and construction.
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
    ComposedCoordinate, CoverageRecord, ExternalRefusal, InputAlignment, InputRole,
    JOINT_LAW_STAGE, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading, check_propagation,
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_io::distribution_artifact::{DistributionArtifact, DrawAlignment};
use antecedent_io::joint_law_propagation_artifact::{
    ConsumedJointLawPropagation, JointLawPropagationArtifactError, JointLawPropagationArtifactWire,
    PropagationRequestWire, float_bits, verdict_code,
};

pub use antecedent_io::joint_law_propagation_artifact::{EnvelopeQueryWire, MarginalWire};
pub use antecedent_stats::ValueFunctional;
use antecedent_stats::{
    DecisionVerdict, NORMAL_APPROXIMATION_CONSTRUCTION, normal_interval_candidate,
};

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
        let values = artifact.draws().iter().skip(column).step_by(width).copied().collect();
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
            && self.artifact.label != antecedent_contract::UNCALIBRATED_SPREAD_LABEL
    }

    /// The alignment identity the rows share, for an aligned route.
    #[must_use]
    pub fn alignment_id(&self) -> Option<&str> {
        self.artifact.alignment_id.as_deref()
    }

    /// A named float value (`mean`, `variance`, `lower`/`upper` pairs ...).
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.entries().find_map(|entry| match entry {
            ResultEntry::Value(entry_name, value) if entry_name == name => Some(value),
            _ => None,
        })
    }

    /// The number of aligned rows, for an aligned route.
    #[must_use]
    pub fn n_rows(&self) -> Option<u64> {
        self.entries().find_map(|entry| match entry {
            ResultEntry::Rows(rows) => Some(rows),
            _ => None,
        })
    }

    /// Every named float value in stored order.
    pub fn values(&self) -> impl Iterator<Item = (&str, f64)> + '_ {
        self.entries().filter_map(|entry| match entry {
            ResultEntry::Value(name, value) => Some((name, value)),
            _ => None,
        })
    }

    /// The decision verdict (`act`, `withhold` or `unresolved`) of a composed
    /// exceedance decision. The verdict is about the Monte Carlo band of the
    /// stated probability only; it is not a coverage statement.
    #[must_use]
    pub fn verdict(&self) -> Option<&'static str> {
        self.decision_verdict().map(DecisionVerdict::as_str)
    }

    /// The typed decision verdict of a composed exceedance decision.
    #[must_use]
    pub fn decision_verdict(&self) -> Option<DecisionVerdict> {
        self.entries().find_map(|entry| match entry {
            ResultEntry::Verdict(verdict) => verdict,
            _ => None,
        })
    }

    /// The stored result, each entry decoded by the kind its name declares.
    fn entries(&self) -> impl Iterator<Item = ResultEntry<'_>> + '_ {
        self.artifact.result.iter().map(|(name, stored)| match name.as_str() {
            ROWS_ENTRY => ResultEntry::Rows(*stored),
            VERDICT_ENTRY => ResultEntry::Verdict(verdict_of_code(*stored)),
            _ => ResultEntry::Value(name, f64::from_bits(*stored)),
        })
    }

    /// The composed coordinate a coverage record for this claim must be
    /// measured at: `operation|route|role,role,...`.
    ///
    /// # Errors
    /// A declaration whose dependence route does not license a spread (an
    /// envelope has no composed coordinate with a spread).
    pub fn coordinate_id(&self) -> Result<String, Refusal> {
        let request =
            PropagationRequest { reading: RequestedReading::Spread, ..self.declared.clone() };
        let plan = check_propagation(&request).map_err(Refusal::one)?;
        Ok(ComposedCoordinate::of(&request, &plan).id())
    }

    /// The calibration key of this claim's composed interval: the composed
    /// coordinate, the route, the dependence declaration, the construction and
    /// the declared sample size of the least-supported input's evidence.
    ///
    /// # Errors
    /// A declaration whose dependence route licenses no spread, or a sample
    /// size of zero.
    pub fn composed_interval_basis(
        &self,
        construction: &str,
        evidence_rows: u64,
    ) -> Result<antecedent_core::CalibrationBasis, Refusal> {
        if evidence_rows == 0 {
            return Err(Refusal::invalid(
                "evidence_rows",
                "declare the number of observations behind the least-supported input",
            ));
        }
        let request =
            PropagationRequest { reading: RequestedReading::Spread, ..self.declared.clone() };
        let plan = check_propagation(&request).map_err(Refusal::one)?;
        let coordinate = ComposedCoordinate::of(&request, &plan);
        Ok(super::joint_law_composed_basis::composed_basis(
            &coordinate.id(),
            plan.route.as_str(),
            &self.declared.dependence,
            construction,
            evidence_rows,
        ))
    }

    /// The interval reading of the composed quantity, licensed only by a
    /// currently attesting coverage record the registry holds for exactly this
    /// composed coordinate, construction, nominal level and evidence size.
    /// The caller supplies no record. Without one it is closed with the typed
    /// refusal; a component interval does not cover the composition.
    ///
    /// A licensed `normal_approximation` interval is built from this claim's
    /// mean and variance at the record's nominal level. For another
    /// construction the licence is returned without bounds: the caller builds
    /// the interval from the aligned draws with the construction the record
    /// was measured for.
    ///
    /// `evidence_rows` is declared, not authenticated: the number of
    /// observations behind the least-supported input.
    ///
    /// # Errors
    /// The refusals of `antecedent_contract::check_composed_interval`: an
    /// unlicensed dependence route, a missing or non-attesting record, or a
    /// nominal level outside `(0, 1)`.
    pub fn check_composed_interval(
        &self,
        construction: &str,
        evidence_rows: u64,
    ) -> Result<ComposedInterval, Refusal> {
        let request =
            PropagationRequest { reading: RequestedReading::Interval, ..self.declared.clone() };
        let basis = self.composed_interval_basis(construction, evidence_rows)?;
        let coordinate_id = basis.functional.to_string();
        let record = match super::joint_law_composed_basis::resolve(&basis) {
            super::joint_law_composed_basis::Resolution::NoRecord => None,
            super::joint_law_composed_basis::Resolution::NotAttesting { record_id, .. } => {
                Some(CoverageRecord {
                    record_id,
                    coordinate_id,
                    construction: construction.to_owned(),
                    nominal_level: super::joint_law_composed_basis::COMPOSED_LEVEL,
                    measured: false,
                })
            }
            super::joint_law_composed_basis::Resolution::Calibrated { record_id } => {
                Some(CoverageRecord {
                    record_id,
                    coordinate_id,
                    construction: construction.to_owned(),
                    nominal_level: super::joint_law_composed_basis::COMPOSED_LEVEL,
                    measured: true,
                })
            }
        };
        let plan =
            antecedent_contract::check_composed_interval(&request, record.as_ref(), construction)
                .map_err(Refusal::one)?;
        let bounds = if construction == NORMAL_APPROXIMATION_CONSTRUCTION {
            let (Some(mean), Some(variance)) = (self.value("mean"), self.value("variance")) else {
                return Err(Refusal::invalid(
                    construction,
                    "the claim carries no mean and variance to build a normal approximation from; build the interval from the aligned draws",
                ));
            };
            let candidate = normal_interval_candidate(mean, variance, plan.nominal_level)
                .map_err(|e| {
                    Refusal::invalid(
                        &e.to_string(),
                        "the interval needs a finite mean, a non-negative variance and a level in (0, 1)",
                    )
                })?;
            Some((candidate.lower, candidate.upper))
        } else {
            None
        };
        Ok(ComposedInterval {
            coordinate_id: plan.coordinate.id(),
            record_id: plan.record_id,
            construction: plan.construction,
            nominal_level: plan.nominal_level,
            label: plan.label,
            bounds,
        })
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

/// A composed interval licensed by a measured coverage record at its composed
/// coordinate. The record, not this value, carries the coverage evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedInterval {
    coordinate_id: String,
    record_id: String,
    construction: String,
    nominal_level: f64,
    label: &'static str,
    bounds: Option<(f64, f64)>,
}

impl ComposedInterval {
    /// The composed coordinate the record was matched against.
    #[must_use]
    pub fn coordinate_id(&self) -> &str {
        &self.coordinate_id
    }

    /// The identity of the record that licensed the interval.
    #[must_use]
    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    /// The construction the interval is built with.
    #[must_use]
    pub fn construction(&self) -> &str {
        &self.construction
    }

    /// The nominal level, the record's.
    #[must_use]
    pub fn nominal_level(&self) -> f64 {
        self.nominal_level
    }

    /// The label of the result (`measured_interval`).
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// The interval ends, when the facade builds the construction from the
    /// claim's own mean and variance.
    #[must_use]
    pub fn bounds(&self) -> Option<(f64, f64)> {
        self.bounds
    }
}

/// Stored result name of the aligned row count (an integer, not float bits).
const ROWS_ENTRY: &str = "n_rows";
/// Stored result name of the decision verdict code (an integer, not float bits).
const VERDICT_ENTRY: &str = "verdict_code";

/// One stored result entry, decoded by the kind its name declares. The stored form is a list
/// of `(name, u64)` pairs because it is part of the sealed artifact identity; this is the typed
/// reading of it.
enum ResultEntry<'a> {
    /// The aligned row count.
    Rows(u64),
    /// The decision verdict; `None` for a code the artifact format does not define.
    Verdict(Option<DecisionVerdict>),
    /// A named float value.
    Value(&'a str, f64),
}

/// The verdict a stored code names, by the artifact format's own coding.
fn verdict_of_code(code: u64) -> Option<DecisionVerdict> {
    [DecisionVerdict::Act, DecisionVerdict::Withhold, DecisionVerdict::Unresolved]
        .into_iter()
        .find(|verdict| verdict_code(*verdict) == code)
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
    linear_moments_reading(
        RequestedReading::Point,
        operation,
        inputs,
        coefficients,
        covariance,
        dependence,
        ctx,
    )
}

/// The spread (a variance under the licensed dependence route) of
/// `sum a_i X_i` from means and a covariance, labelled `uncalibrated_spread`.
/// The variance is what the route determines; it carries no coverage statement.
///
/// # Errors
/// As [`propagate_linear_moments`].
pub fn propagate_linear_moments_spread(
    operation: PropagationOperation,
    inputs: &[MomentInput],
    coefficients: &[f64],
    covariance: &[f64],
    dependence: &LinearDependence,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    linear_moments_reading(
        RequestedReading::Spread,
        operation,
        inputs,
        coefficients,
        covariance,
        dependence,
        ctx,
    )
}

fn linear_moments_reading(
    reading: RequestedReading,
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
        reading,
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

// ------------------------------------------------------- spread readings

/// The spread of `g(X, Y)` under the aligned joint law of two inputs: the
/// mean, the variance of the composed rows and the Monte Carlo standard error
/// of the mean, labelled `uncalibrated_spread`. It carries no coverage
/// statement; an interval needs [`PropagationClaim::check_composed_interval`].
///
/// # Errors
/// As [`propagate_aligned_functional`].
pub fn propagate_spread(
    operation: PropagationOperation,
    x: &PropagationDraws,
    y: &PropagationDraws,
    functional: ValueFunctional,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = aligned_declared(operation, x, y, RequestedReading::Spread);
    check_propagation(&declared).map_err(Refusal::one)?;
    paired(x, y)?;
    propagate(
        &declared,
        PropagationRequestWire::AlignedFunctionalSpread {
            functional: functional.as_str().to_owned(),
            x: float_bits(&x.values),
            y: float_bits(&y.values),
        },
        ctx,
    )
}

/// The spread (variance and covariance) of `a X + b Y` from aligned draws,
/// labelled `uncalibrated_spread`.
///
/// # Errors
/// As [`propagate_aligned_linear`].
pub fn propagate_aligned_linear_spread(
    operation: PropagationOperation,
    coefficients: (f64, f64),
    x: &PropagationDraws,
    y: &PropagationDraws,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = aligned_declared(operation, x, y, RequestedReading::Spread);
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

fn aligned_declared(
    operation: PropagationOperation,
    x: &PropagationDraws,
    y: &PropagationDraws,
    reading: RequestedReading,
) -> PropagationRequest {
    PropagationRequest {
        operation,
        inputs: vec![x.as_input(), y.as_input()],
        dependence: PropagationDependence::AlignedJointLaw,
        reading,
    }
}

/// The threshold, probability cutoff and Monte Carlo band level of an
/// exceedance decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExceedanceQuery {
    /// The composed value the probability is of exceeding.
    pub threshold: f64,
    /// The probability cutoff the verdict compares against, in `(0, 1)`.
    pub cutoff: f64,
    /// The level of the Monte Carlo band, in `(0, 1)`.
    pub band_level: f64,
}

/// `P(g(X, Y) > threshold)` from aligned draws with its Monte Carlo band and
/// the verdict (`act`, `withhold` or `unresolved`) against the cutoff, labelled
/// `uncalibrated_spread`. The verdict is about the Monte Carlo error of the
/// stated probability only; it says nothing about the law's calibration.
///
/// # Errors
/// As [`propagate_aligned_functional`]; also a cutoff or band level outside
/// `(0, 1)` or a non-finite threshold.
pub fn propagate_composed_exceedance(
    operation: PropagationOperation,
    x: &PropagationDraws,
    y: &PropagationDraws,
    functional: ValueFunctional,
    query: ExceedanceQuery,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let declared = aligned_declared(operation, x, y, RequestedReading::Spread);
    check_propagation(&declared).map_err(Refusal::one)?;
    paired(x, y)?;
    propagate(
        &declared,
        PropagationRequestWire::AlignedExceedance {
            functional: functional.as_str().to_owned(),
            x: float_bits(&x.values),
            y: float_bits(&y.values),
            threshold: query.threshold.to_bits(),
            cutoff: query.cutoff.to_bits(),
            band_level: query.band_level.to_bits(),
        },
        ctx,
    )
}

/// A component of a finite mixture given as draws, with its known weight.
#[derive(Clone, Debug, PartialEq)]
pub struct MixtureDraws {
    /// Component identity.
    pub id: String,
    /// Identity of the evidence the component rests on.
    pub evidence_id: String,
    /// The component's known weight.
    pub weight: f64,
    /// The component's draws.
    pub draws: Vec<f64>,
}

/// `P(X > threshold)` of a finite mixture of draw sets with known weights, with
/// a Monte Carlo standard error that treats each component's draws as
/// independently produced, labelled `uncalibrated_spread`. The components need
/// no joint law and are never paired.
///
/// # Errors
/// As [`propagate`]; also weights that are not a probability vector or an empty
/// or non-finite component.
pub fn propagate_mixture_exceedance(
    components: &[MixtureDraws],
    threshold: f64,
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
        reading: RequestedReading::Spread,
    };
    let weights: Vec<f64> = components.iter().map(|c| c.weight).collect();
    propagate(
        &declared,
        PropagationRequestWire::MixtureExceedance {
            weights: float_bits(&weights),
            components: components.iter().map(|c| float_bits(&c.draws)).collect(),
            threshold: threshold.to_bits(),
        },
        ctx,
    )
}

// ------------------------------------------------- transported uncertainty

/// The spread of a transported claim from its three pieces (source sampling,
/// target sampling, transport functional) and their covariance, labelled
/// `uncalibrated_spread`.
///
/// `pieces` carry their roles (`InputRole::SourceSampling`,
/// `InputRole::TargetSampling`, `InputRole::TransportFunctional`); the rule
/// refuses a spread whose pieces are incomplete, and a declared independence of
/// the pieces, because the pieces are dependent when evidence is shared.
/// `covariance_id` names the declaration or the shared evidence that supplies
/// the covariance.
///
/// # Errors
/// As [`propagate_linear_moments`], plus the transported-composition refusals
/// (`joint_law_propagation.transport_pieces_incomplete`,
/// `joint_law_propagation.transported_dependence_required`).
pub fn propagate_transported_spread(
    pieces: &[MomentInput],
    coefficients: &[f64],
    covariance: &[f64],
    covariance_id: &str,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    linear_moments_reading(
        RequestedReading::Spread,
        PropagationOperation::TransportedComposition,
        pieces,
        coefficients,
        covariance,
        &LinearDependence::Covariance { id: covariance_id.to_owned() },
        ctx,
    )
}

/// The spread of a transported claim from the aligned draws of its three
/// pieces: the rows must share one alignment identity (columns of one joint
/// law). The covariance is the population covariance of the aligned rows, and
/// its declaration names the alignment identity, so a refusal for a mismatched
/// or unpaired piece comes from the rule before any number is formed.
///
/// `coefficients` weight (source sampling, target sampling, transport
/// functional) in that order.
///
/// # Errors
/// As [`propagate_transported_spread`]; also `joint_law_required` when the
/// pieces' alignment identities differ or a piece has no aligned draws.
pub fn propagate_transported_spread_aligned(
    source_sampling: &PropagationDraws,
    target_sampling: &PropagationDraws,
    transport_functional: &PropagationDraws,
    coefficients: [f64; 3],
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let pieces = [
        source_sampling.clone().with_role(InputRole::SourceSampling),
        target_sampling.clone().with_role(InputRole::TargetSampling),
        transport_functional.clone().with_role(InputRole::TransportFunctional),
    ];
    let declared = PropagationRequest {
        operation: PropagationOperation::TransportedComposition,
        inputs: pieces.iter().map(PropagationDraws::as_input).collect(),
        dependence: PropagationDependence::AlignedJointLaw,
        reading: RequestedReading::Spread,
    };
    let plan = check_propagation(&declared).map_err(Refusal::one)?;
    let n = pieces[0].values.len();
    if pieces.iter().any(|p| p.values.len() != n) {
        return Err(Refusal::invalid(
            &pieces[0].id,
            "aligned columns have one value per joint realization; the draw counts differ",
        ));
    }
    if n < 2 || pieces.iter().any(|p| p.values.iter().any(|v| !v.is_finite())) {
        return Err(Refusal::invalid(
            &pieces[0].id,
            "a spread needs at least two finite aligned rows",
        ));
    }
    let nf = n as f64;
    let means: Vec<f64> = pieces.iter().map(|p| p.values.iter().sum::<f64>() / nf).collect();
    let mut covariance = vec![0.0; 9];
    for i in 0..3 {
        for j in i..3 {
            let total: f64 = pieces[i]
                .values
                .iter()
                .zip(&pieces[j].values)
                .map(|(a, b)| (a - means[i]) * (b - means[j]))
                .sum();
            covariance[i * 3 + j] = total / nf;
            covariance[j * 3 + i] = total / nf;
        }
    }
    let alignment = plan.alignment_id.unwrap_or_default();
    let moment_inputs: Vec<MomentInput> = pieces
        .iter()
        .zip(&means)
        .map(|(p, mean)| MomentInput {
            id: p.id.clone(),
            evidence_id: p.evidence_id.clone(),
            role: p.role,
            mean: *mean,
        })
        .collect();
    linear_moments_reading(
        RequestedReading::Spread,
        PropagationOperation::TransportedComposition,
        &moment_inputs,
        &coefficients,
        &covariance,
        &LinearDependence::Covariance { id: format!("aligned_draw_covariance:{alignment}") },
        ctx,
    )
}

/// Rebuild a claim from a sealed artifact for a further reading (the interval
/// check). The artifact is verified and replayed first (a consumer re-derives,
/// it does not trust), so an edited artifact is refused.
///
/// # Errors
/// As [`consume_joint_law_propagation_artifact`].
pub fn claim_from_artifact(
    bytes: &[u8],
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<PropagationClaim, Refusal> {
    let wire = JointLawPropagationArtifactWire::decode(bytes)?;
    wire.consume(expected_identity, ctx)?;
    let malformed =
        |what: &str| Refusal::invalid(what, "supply an intact joint_law_propagation_v1 artifact");
    let operation = PropagationOperation::from_name(&wire.rule.operation)
        .ok_or_else(|| malformed("operation"))?;
    let reading =
        RequestedReading::from_name(&wire.rule.reading).ok_or_else(|| malformed("reading"))?;
    let dependence = match wire.rule.dependence.as_str() {
        "unknown" => PropagationDependence::Unknown,
        "declared_independent" => PropagationDependence::DeclaredIndependent,
        "covariance" => PropagationDependence::Covariance {
            id: wire.rule.covariance_id.clone().ok_or_else(|| malformed("covariance id"))?,
        },
        "aligned_joint_law" => PropagationDependence::AlignedJointLaw,
        "known_weights" => PropagationDependence::KnownWeights,
        "correlation_range" => {
            let (Some(lo), Some(hi)) = (wire.rule.rho_lo, wire.rule.rho_hi) else {
                return Err(malformed("correlation range"));
            };
            PropagationDependence::CorrelationRange {
                rho_lo: f64::from_bits(lo),
                rho_hi: f64::from_bits(hi),
            }
        }
        "marginals_only" => PropagationDependence::MarginalsOnly,
        _ => return Err(malformed("dependence")),
    };
    let mut inputs = Vec::with_capacity(wire.inputs.len());
    for input in &wire.inputs {
        let alignment = match (input.alignment.as_str(), &input.alignment_id) {
            ("aligned", Some(id)) => InputAlignment::Aligned { alignment_id: id.clone() },
            ("independent_marginals", None) => InputAlignment::IndependentMarginals,
            ("moments_only", None) => InputAlignment::MomentsOnly,
            _ => return Err(malformed("input alignment")),
        };
        inputs.push(PropagationInput {
            id: input.input_id.clone(),
            evidence_id: input.evidence_id.clone(),
            role: InputRole::from_name(&input.role).ok_or_else(|| malformed("input role"))?,
            alignment,
        });
    }
    let declared = PropagationRequest { operation, inputs, dependence, reading };
    Ok(PropagationClaim { declared, artifact: wire })
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
