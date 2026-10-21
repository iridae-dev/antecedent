//! Retained rows of the unlicensed Bayesian randomized-design posteriors, and the independent
//! artifact of the randomized-dose grid posterior.
//!
//! The renderers [`dose_grid_row`], [`policy_row`], [`quantile_row`] and [`survival_row`] turn a
//! posterior into the [`PosteriorRowResultWire`] that results and analysis-result artifacts
//! retain: the posterior mean and equal-tailed 95% credible interval of every coordinate, the
//! covariance of the joint draws, the exact Gaussian moments when the residual variance is
//! known, and the named scalar and vector summaries. The survival and policy rows are produced
//! by the `Study` routes of `bayesian.piecewise_survival` and `bayesian.continuous_dose_policy`.
//!
//! The dose grid has no `Study` query, exactly like its frequentist row
//! (`dose_grid_functional_artifact`), so its posterior (`bayesian.dose_grid`: the joint
//! posterior of a level, derivative or contrast of a randomized-dose response curve under a
//! declared polynomial coefficient prior) is sealed here as format version 1 together with
//! every premise and the compact data table it was computed from. Every result is a posterior
//! whose calibration is `unmeasured`: no coverage claim is made and the estimator is not
//! licensed.
//!
//! A consumer trusts nothing in the result. It re-validates the stored request, recomputes
//! the posterior from the embedded table with the stored seed and accepts only a result that
//! is identical, bit for bit, to the stored one. Two digests guard the stored request: the
//! premises digest (every declared model quantity) and a separate data digest (the table,
//! bit for bit). A changed premise, table entry or stored value is refused even when both
//! digests are re-sealed, because the replayed result then differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::IdentityDomain;
use antecedent_estimate::EstimationError;
use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::dose_grid_functional::{DoseDesign, DoseFunctional};
use antecedent_estimate::dose_grid_posterior::{
    CompiledDoseGridPosterior, DoseGridPosteriorRequest,
};
use antecedent_estimate::survival_posterior::IntervalHazardPosterior;
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::dose_grid_functional_artifact::DoseFunctionalWire;

/// The artifact format this reader writes and accepts.
pub const POSTERIOR_ROW_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const POSTERIOR_ROW_ARTIFACT_FEATURE: &str = "bayesian_posterior_row_v1";
pub use antecedent_io_core::{POSTERIOR_ROW_CALIBRATION, POSTERIOR_ROW_UNCERTAINTY};

/// Estimator id of the randomized-dose grid posterior.
pub const DOSE_GRID_POSTERIOR_ESTIMATOR: &str = "bayesian.dose_grid";
/// Estimator id of the fixed group-dose policy value posterior.
pub const CONTINUOUS_DOSE_POLICY_POSTERIOR_ESTIMATOR: &str = "bayesian.continuous_dose_policy";
/// Estimator id of the randomized quantile posterior.
pub const QUANTILE_POSTERIOR_ESTIMATOR: &str = "bayesian.randomized_quantile";
/// Estimator id of the piecewise-exponential survival posterior.
pub const SURVIVAL_POSTERIOR_ESTIMATOR: &str = "bayesian.piecewise_survival";

const DESIGN_RANDOMIZED: &str = "randomized_dose";
const DESIGN_OBSERVATIONAL: &str = "observational";
const KIND_LEVEL: &str = "level";
const KIND_DERIVATIVE: &str = "derivative";
const KIND_CONTRAST: &str = "contrast";

/// Why a posterior-row artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PosteriorRowArtifactError {
    /// The feature marker, a field spelling or the result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The row refused the stored request with its own typed reason.
    #[error("{detail}: {message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// Namespaced detail, such as `survival_posterior.competing_causes_unsupported`.
        detail: String,
        /// What was refused and the next attempt.
        message: String,
    },
    /// The recomputed result differs from the stored one.
    #[error("the stored result does not replay")]
    ResultMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored table.
    #[error("data digest mismatch")]
    DataMismatch,
    /// The bytes do not decode.
    #[error("artifact does not decode: {0}")]
    Undecodable(String),
}

impl PosteriorRowArtifactError {
    /// `(reason code, detail)` of this refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, String) {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::UnsupportedSemantics(_) => (
                antecedent_core::reason_code!("route_not_supported"),
                "bayesian_row.unsupported_semantics".to_owned(),
            ),
            Self::LimitsExceeded(_) => (invalid, "bayesian_row.consumer_limit_exceeded".to_owned()),
            Self::Refused { code, detail, .. } => (*code, detail.clone()),
            Self::ResultMismatch => (invalid, "bayesian_row.result_replay_mismatch".to_owned()),
            Self::PremisesMismatch => (invalid, "bayesian_row.premises_mismatch".to_owned()),
            Self::DataMismatch => (invalid, "bayesian_row.data_identity_mismatch".to_owned()),
            Self::Undecodable(_) => (invalid, "bayesian_row.undecodable".to_owned()),
        }
    }
}

impl From<PosteriorRowArtifactError> for IoError {
    fn from(error: PosteriorRowArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: bayesian posterior row: {error}") }
    }
}

/// Keep the typed refusal of an estimator; name any other failure under `prefix`.
pub(crate) fn from_estimation(error: EstimationError, prefix: &str) -> PosteriorRowArtifactError {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => {
            let split = message
                .split_once(": ")
                .filter(|(detail, _)| detail.contains('.'))
                .map(|(detail, text)| (detail.to_owned(), text.to_owned()));
            match split {
                Some((detail, text)) => {
                    PosteriorRowArtifactError::Refused { code, detail, message: text }
                }
                None => PosteriorRowArtifactError::Refused {
                    code,
                    detail: format!("{prefix}.refused"),
                    message,
                },
            }
        }
        EstimationError::Unsupported { message } => {
            let detail = if message.contains("threshold grid") {
                format!("{prefix}.unresolved_draws")
            } else {
                format!("{prefix}.unsupported")
            };
            PosteriorRowArtifactError::Refused {
                code: antecedent_core::reason_code!("route_not_supported"),
                detail,
                message: message.to_owned(),
            }
        }
        other => PosteriorRowArtifactError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: format!("{prefix}.invalid_request"),
            message: other.to_string(),
        },
    }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PosteriorRowConsumeLimits {
    /// Most rows of the stored table.
    pub max_rows: usize,
    /// Most grid doses, thresholds or hazard intervals.
    pub max_points: usize,
}

impl Default for PosteriorRowConsumeLimits {
    fn default() -> Self {
        Self { max_rows: 100_000, max_points: 1_024 }
    }
}

/// The residual-variance prior of a polynomial dose-response posterior.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum ResidualPriorWire {
    /// The residual variance is known.
    KnownVariance(f64),
    /// `sigma^2 ~ InvGamma(shape, scale)`.
    InverseGamma {
        /// Shape.
        shape: f64,
        /// Scale.
        scale: f64,
    },
}

impl From<ResidualPriorWire> for ResidualPrior {
    fn from(wire: ResidualPriorWire) -> Self {
        match wire {
            ResidualPriorWire::KnownVariance(variance) => Self::KnownVariance(variance),
            ResidualPriorWire::InverseGamma { shape, scale } => Self::InverseGamma { shape, scale },
        }
    }
}

/// Request of the randomized-dose grid posterior.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DoseGridPosteriorWire {
    /// `randomized_dose` or `observational` (refused).
    pub design: String,
    /// The one named functional.
    pub functional: DoseFunctionalWire,
    /// Grid doses of a level or derivative; empty for a contrast.
    pub grid: Vec<f64>,
    /// Declared dose support the polynomial is rescaled on.
    pub support: (f64, f64),
    /// Prior scale of every polynomial coefficient.
    pub coefficient_scales: Vec<f64>,
    /// Residual-variance prior.
    pub residual: ResidualPriorWire,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draws.
    pub seed: u64,
    /// Observed doses.
    pub dose: Vec<f64>,
    /// Observed outcomes, aligned with `dose`.
    pub outcome: Vec<f64>,
}

/// One posterior request of a randomized design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum PosteriorRowRequestWire {
    /// Randomized-dose grid posterior.
    DoseGrid(DoseGridPosteriorWire),
}

/// The stored result of a posterior row: the wire the analysis result artifact also carries.
pub use antecedent_io_core::PosteriorRowWire as PosteriorRowResultWire;

/// Versioned posterior row with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PosteriorRowArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Premises and the compact data table.
    pub request: PosteriorRowRequestWire,
    /// The stored result.
    pub result: PosteriorRowResultWire,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the data table.
    pub data_digest: String,
}

/// Canonical bit-level view of a request, split into premises and data.
#[derive(Default, Serialize)]
struct Canonical {
    tag: &'static str,
    premises: Vec<(&'static str, Vec<u64>)>,
    labels: Vec<(&'static str, Vec<String>)>,
    data: Vec<(&'static str, Vec<u64>)>,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn tally(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn residual_bits(residual: ResidualPriorWire) -> Vec<u64> {
    match residual {
        ResidualPriorWire::KnownVariance(variance) => vec![0, variance.to_bits()],
        ResidualPriorWire::InverseGamma { shape, scale } => {
            vec![1, shape.to_bits(), scale.to_bits()]
        }
    }
}

impl PosteriorRowRequestWire {
    /// Estimator id of the row this request asks for.
    #[must_use]
    pub const fn estimator(&self) -> &'static str {
        match self {
            Self::DoseGrid(_) => DOSE_GRID_POSTERIOR_ESTIMATOR,
        }
    }

    fn rows(&self) -> usize {
        match self {
            Self::DoseGrid(w) => w.dose.len().max(w.outcome.len()),
        }
    }

    fn points(&self) -> usize {
        match self {
            Self::DoseGrid(w) => w.grid.len().max(w.coefficient_scales.len()),
        }
    }

    fn canonical(&self) -> Canonical {
        match self {
            Self::DoseGrid(w) => Canonical {
                tag: "bayesian_dose_grid_v1",
                premises: vec![
                    ("from", w.functional.from.map(f64::to_bits).into_iter().collect()),
                    ("to", w.functional.to.map(f64::to_bits).into_iter().collect()),
                    ("grid", bits(&w.grid)),
                    ("support", bits(&[w.support.0, w.support.1])),
                    ("scales", bits(&w.coefficient_scales)),
                    ("residual", residual_bits(w.residual)),
                    ("draws", vec![count(w.draws)]),
                    ("seed", vec![w.seed]),
                    ("rows", vec![count(w.dose.len())]),
                ],
                labels: vec![
                    ("design", vec![w.design.clone()]),
                    ("kind", vec![w.functional.kind.clone()]),
                ],
                data: vec![("dose", bits(&w.dose)), ("outcome", bits(&w.outcome))],
            },
        }
    }

    fn premises_digest(&self) -> Result<String, IoError> {
        let canonical = self.canonical();
        crate::digest_hex(
            IdentityDomain::Claim,
            &(canonical.tag, &canonical.premises, &canonical.labels),
        )
    }

    fn data_digest(&self) -> Result<String, IoError> {
        let canonical = self.canonical();
        crate::digest_hex(IdentityDomain::DataSnapshot, &(canonical.tag, &canonical.data))
    }

    fn check_limits(
        &self,
        limits: &PosteriorRowConsumeLimits,
    ) -> Result<(), PosteriorRowArtifactError> {
        if self.rows() > limits.max_rows {
            return Err(PosteriorRowArtifactError::LimitsExceeded("rows"));
        }
        if self.points() > limits.max_points {
            return Err(PosteriorRowArtifactError::LimitsExceeded(
                "grid doses, thresholds or intervals",
            ));
        }
        Ok(())
    }
}

fn dose_functional(w: &DoseGridPosteriorWire) -> Result<DoseFunctional, PosteriorRowArtifactError> {
    let spec = &w.functional;
    let unsupported = PosteriorRowArtifactError::UnsupportedSemantics;
    match (spec.kind.as_str(), spec.from, spec.to) {
        (KIND_LEVEL, None, None) if !w.grid.is_empty() => Ok(DoseFunctional::Level),
        (KIND_DERIVATIVE, None, None) if !w.grid.is_empty() => Ok(DoseFunctional::Derivative),
        (KIND_CONTRAST, Some(from), Some(to)) if w.grid.is_empty() => {
            Ok(DoseFunctional::Contrast { from, to })
        }
        (KIND_LEVEL | KIND_DERIVATIVE, _, _) => {
            Err(unsupported("a level or derivative needs a grid and no contrast doses"))
        }
        (KIND_CONTRAST, _, _) => {
            Err(unsupported("a contrast names its two doses and takes no grid"))
        }
        _ => Err(unsupported("functional kind")),
    }
}

/// The retained row of a dose-grid posterior: the joint draw covariance across the evaluation
/// doses, the exact Gaussian moments when the residual variance is known, and the doses.
#[must_use]
pub fn dose_grid_row(
    kind: &str,
    functional: DoseFunctional,
    posterior: &antecedent_estimate::dose_grid_posterior::DoseGridPosterior,
) -> PosteriorRowResultWire {
    let coordinates = match functional {
        DoseFunctional::Contrast { from, to } => vec![format!("contrast({from}->{to})")],
        _ => posterior.points.iter().map(|dose| format!("{kind}({dose})")).collect(),
    };
    let mut result = PosteriorRowResultWire::new(
        DOSE_GRID_POSTERIOR_ESTIMATOR,
        format!("randomized_dose_response_{kind}"),
        coordinates,
        posterior.n_draws,
    )
    .with_intervals(posterior.mean.clone(), &posterior.interval_95);
    result.covariance = Some(posterior.covariance.clone());
    result.exact_mean.clone_from(&posterior.exact_mean);
    result.exact_covariance.clone_from(&posterior.exact_covariance);
    result.vectors.insert("points".to_owned(), posterior.points.clone());
    result
}

fn checked_dose_request(
    w: &DoseGridPosteriorWire,
) -> Result<DoseGridPosteriorRequest<'_>, PosteriorRowArtifactError> {
    let design = match w.design.as_str() {
        DESIGN_RANDOMIZED => DoseDesign::RandomizedDose,
        DESIGN_OBSERVATIONAL => DoseDesign::Observational,
        _ => return Err(PosteriorRowArtifactError::UnsupportedSemantics("design spelling")),
    };
    let functional = dose_functional(w)?;
    Ok(DoseGridPosteriorRequest {
        dose: &w.dose,
        outcome: &w.outcome,
        grid: &w.grid,
        support: w.support,
        coefficient_scales: &w.coefficient_scales,
        residual: w.residual.into(),
        functional,
        design,
        draws: w.draws,
        seed: w.seed,
    })
}

/// The retained row of a fixed group-dose policy posterior: policy, reference and their
/// paired difference, with the joint covariance of the paired draws.
#[must_use]
pub fn policy_row(
    posterior: &antecedent_estimate::continuous_dose_posterior::DosePolicyPosterior,
) -> PosteriorRowResultWire {
    let coordinates =
        ["policy_value", "reference_value", "incremental_value"].map(str::to_owned).to_vec();
    let incremental: Vec<f64> =
        posterior.policy_draws.iter().zip(&posterior.reference_draws).map(|(p, r)| p - r).collect();
    let mut result = PosteriorRowResultWire::new(
        CONTINUOUS_DOSE_POLICY_POSTERIOR_ESTIMATOR,
        "kernel_smoothed_group_dose_policy_value".to_owned(),
        coordinates,
        posterior.draws,
    )
    .with_intervals(
        vec![posterior.policy_mean, posterior.reference_mean, posterior.incremental_mean],
        &[
            posterior.policy_interval_95,
            posterior.reference_interval_95,
            posterior.incremental_interval_95,
        ],
    );
    result.covariance =
        Some(draw_covariance(&[&posterior.policy_draws, &posterior.reference_draws, &incremental]));
    result.exact_mean = posterior.exact_means.map(|means| means.to_vec());
    result.extras.insert(
        "probability_incremental_positive".to_owned(),
        posterior.probability_incremental_positive,
    );
    result.extras.insert("minimum_local_rows".to_owned(), tally(posterior.minimum_local_rows));
    if let Some(variances) = posterior.exact_variances {
        result.vectors.insert("exact_variances".to_owned(), variances.to_vec());
    }
    result.vectors.insert(
        "group_fractions".to_owned(),
        posterior.group_fractions.iter().map(|(_, fraction)| *fraction).collect(),
    );
    result
}

/// Covariance of the columns of draw vectors of equal length, row-major.
fn draw_covariance(columns: &[&[f64]]) -> Vec<f64> {
    let means: Vec<f64> = columns
        .iter()
        .map(|column| column.iter().sum::<f64>() / tally(column.len().max(1)))
        .collect();
    let mut out = Vec::with_capacity(columns.len() * columns.len());
    for (left, left_mean) in columns.iter().zip(&means) {
        for (right, right_mean) in columns.iter().zip(&means) {
            let total: f64 =
                left.iter().zip(*right).map(|(a, b)| (a - left_mean) * (b - right_mean)).sum();
            out.push(total / tally(left.len().max(1)));
        }
    }
    out
}

/// The retained row of a quantile posterior: control, treated and their difference, with the
/// joint covariance of the resolved draws.
#[must_use]
pub fn quantile_row(
    posterior: &antecedent_estimate::quantile_posterior::QuantilePosterior,
) -> PosteriorRowResultWire {
    let difference: Vec<f64> =
        posterior.treated_draws.iter().zip(&posterior.control_draws).map(|(a, c)| a - c).collect();
    let coordinates =
        ["control_quantile", "treated_quantile", "quantile_difference"].map(str::to_owned).to_vec();
    let mut result = PosteriorRowResultWire::new(
        QUANTILE_POSTERIOR_ESTIMATOR,
        "randomized_potential_outcome_quantile".to_owned(),
        coordinates,
        posterior.draws_resolved,
    )
    .with_intervals(
        vec![
            posterior.control_quantile_mean,
            posterior.treated_quantile_mean,
            posterior.difference_mean,
        ],
        &[
            posterior.control_interval_95,
            posterior.treated_interval_95,
            posterior.difference_interval_95,
        ],
    );
    result.covariance =
        Some(draw_covariance(&[&posterior.control_draws, &posterior.treated_draws, &difference]));
    result.extras.insert(
        "probability_treated_quantile_higher".to_owned(),
        posterior.probability_treated_quantile_higher,
    );
    result.extras.insert("tau".to_owned(), posterior.tau);
    result.extras.insert("draws_requested".to_owned(), tally(posterior.draws_requested));
    result.extras.insert("draws_resolved".to_owned(), tally(posterior.draws_resolved));
    result.vectors.insert("control_cdf_mean".to_owned(), posterior.control_cdf_mean.clone());
    result.vectors.insert("treated_cdf_mean".to_owned(), posterior.treated_cdf_mean.clone());
    result
}

fn hazard_vectors(prefix: &str, arm: &[IntervalHazardPosterior], out: &mut PosteriorRowResultWire) {
    let column = |select: fn(&IntervalHazardPosterior) -> f64| -> Vec<f64> {
        arm.iter().map(select).collect()
    };
    out.vectors.insert(format!("{prefix}_length"), column(|p| p.length));
    out.vectors.insert(format!("{prefix}_events"), column(|p| tally(p.events)));
    out.vectors.insert(format!("{prefix}_exposure"), column(|p| p.exposure));
    out.vectors.insert(format!("{prefix}_shape"), column(|p| p.shape));
    out.vectors.insert(format!("{prefix}_rate"), column(|p| p.rate));
}

/// The retained row of a survival posterior: the survival and restricted-mean contrasts with
/// the covariance of their paired draws, and every interval's gamma posterior.
#[must_use]
pub fn survival_row(
    posterior: &antecedent_estimate::survival_posterior::PiecewiseSurvivalPosterior,
) -> PosteriorRowResultWire {
    let mut result = PosteriorRowResultWire::new(
        SURVIVAL_POSTERIOR_ESTIMATOR,
        "randomized_survival_and_restricted_mean_contrast".to_owned(),
        ["survival_difference", "rmst_difference"].map(str::to_owned).to_vec(),
        posterior.draws,
    )
    .with_intervals(
        vec![posterior.survival_difference_mean, posterior.rmst_difference_mean],
        &[posterior.survival_difference_interval_95, posterior.rmst_difference_interval_95],
    );
    result.covariance = Some(draw_covariance(&[
        &posterior.survival_difference_draws,
        &posterior.rmst_difference_draws,
    ]));
    for (name, value) in [
        ("control_survival_mean", posterior.control_survival_mean),
        ("treated_survival_mean", posterior.treated_survival_mean),
        ("control_rmst_mean", posterior.control_rmst_mean),
        ("treated_rmst_mean", posterior.treated_rmst_mean),
        ("probability_treated_survives_longer", posterior.probability_treated_survives_longer),
    ] {
        result.extras.insert(name.to_owned(), value);
    }
    result.vectors.insert("cuts".to_owned(), posterior.cuts.clone());
    hazard_vectors("control", &posterior.control, &mut result);
    hazard_vectors("treated", &posterior.treated, &mut result);
    result
}

impl PosteriorRowArtifactWire {
    /// Evaluate the row on `request` and seal the result with both digests.
    ///
    /// # Errors
    /// The row's typed refusal, a stored collection above the default consumer bounds, or an
    /// unsupported spelling.
    pub fn seal(request: PosteriorRowRequestWire) -> Result<Self, PosteriorRowArtifactError> {
        Self::seal_with_context(request, &antecedent_core::ExecutionContext::production_default(0))
    }

    /// Compile and execute under caller resource and cancellation policy.
    pub fn seal_with_context(
        request: PosteriorRowRequestWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        CompiledPosteriorRow::compile(request, ctx)?.artifact(ctx)
    }

    fn from_executed(
        request: PosteriorRowRequestWire,
        result: PosteriorRowResultWire,
    ) -> Result<Self, PosteriorRowArtifactError> {
        let undecodable = |e: IoError| PosteriorRowArtifactError::Undecodable(e.to_string());
        let wire = Self {
            version: POSTERIOR_ROW_ARTIFACT_VERSION,
            required_features: vec![POSTERIOR_ROW_ARTIFACT_FEATURE.to_owned()],
            premises_digest: request.premises_digest().map_err(undecodable)?,
            data_digest: request.data_digest().map_err(undecodable)?,
            request,
            result,
        };
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still recomputes the whole result.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        self.request.premises_digest()
    }

    /// The data digest the stored table should carry.
    ///
    /// # Errors
    /// The table does not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        self.request.data_digest()
    }

    fn validate_shape(&self) -> Result<(), PosteriorRowArtifactError> {
        let unsupported = PosteriorRowArtifactError::UnsupportedSemantics;
        if self.required_features != [POSTERIOR_ROW_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        let result = &self.result;
        if result.estimator != self.request.estimator() {
            return Err(unsupported("the result is not of the requested estimator"));
        }
        if result.uncertainty != POSTERIOR_ROW_UNCERTAINTY
            || result.calibration != POSTERIOR_ROW_CALIBRATION
            || result.licensed
        {
            return Err(unsupported(
                "these rows publish unlicensed posteriors with unmeasured calibration",
            ));
        }
        let width = result.coordinates.len();
        if result.mean.len() != width || result.lower.len() != width || result.upper.len() != width
        {
            return Err(unsupported("coordinate arrays must align"));
        }
        let square =
            |values: &Option<Vec<f64>>| values.as_ref().is_none_or(|v| v.len() == width * width);
        if !square(&result.covariance) || !square(&result.exact_covariance) {
            return Err(unsupported("covariance must be coordinates by coordinates"));
        }
        if result.exact_mean.as_ref().is_some_and(|v| v.len() != width) {
            return Err(unsupported("exact mean must have one entry per coordinate"));
        }
        Ok(())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if version == POSTERIOR_ROW_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Decode and recompute everything under the consumer's limits; accept only a result
    /// identical to the stored one.
    ///
    /// # Errors
    /// Any reconstruction failure, with the reason code and `bayesian_row.*` detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
    ) -> Result<Self, IoError> {
        Self::consume_typed(bytes, limits).map_err(IoError::from)
    }

    /// [`Self::consume_with_limits`] with the typed refusal kind.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume_typed(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
    ) -> Result<Self, PosteriorRowArtifactError> {
        Self::consume_typed_with_context(
            bytes,
            limits,
            &antecedent_core::ExecutionContext::production_default(0),
        )
    }

    /// Independently compile/replay under caller-supplied execution authority.
    pub fn consume_typed_with_context(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        Self::consume_prepared_with_context(bytes, limits, ctx).map(|(wire, _program)| wire)
    }

    /// Replay once and retain the exact checked program used by the consumer.
    pub fn consume_prepared_with_context(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(Self, CompiledPosteriorRow), PosteriorRowArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal("dose_grid.cancelled", "artifact consumption cancelled"));
        }
        let bound = (bytes.len() as u64).checked_mul(8).ok_or_else(|| {
            execution_refusal("dose_grid.resource_refused", "artifact byte size overflow")
        })?;
        if ctx.memory.hard_limit_bytes.is_some_and(|limit| bound > limit) {
            return Err(execution_refusal(
                "dose_grid.resource_refused",
                "artifact decoding exceeds memory limit",
            ));
        }
        let _decode = ctx
            .acquire_admission(0, bound)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => PosteriorRowArtifactError::UnsupportedSemantics("shape"),
            other => PosteriorRowArtifactError::Undecodable(other.to_string()),
        })?;
        wire.request.check_limits(&limits)?;
        let undecodable = |e: IoError| PosteriorRowArtifactError::Undecodable(e.to_string());
        crate::require_digest(
            &wire.premises_digest,
            wire.request.premises_digest(),
            undecodable,
            || PosteriorRowArtifactError::PremisesMismatch,
        )?;
        crate::require_digest(&wire.data_digest, wire.request.data_digest(), undecodable, || {
            PosteriorRowArtifactError::DataMismatch
        })?;
        let program = CompiledPosteriorRow::compile(wire.request.clone(), ctx)?;
        let fresh = program.execute(ctx)?;
        let stored = crate::to_cbor(&wire.result).map_err(undecodable)?;
        let recomputed = crate::to_cbor(&fresh).map_err(undecodable)?;
        if stored != recomputed {
            return Err(PosteriorRowArtifactError::ResultMismatch);
        }
        Ok((wire, program))
    }
}

fn execution_refusal(detail: &str, message: &str) -> PosteriorRowArtifactError {
    PosteriorRowArtifactError::Refused {
        code: antecedent_core::reason_code!("transport_budget_cancel"),
        detail: detail.into(),
        message: message.into(),
    }
}
/// Retained scientific request and checked statistical program. Encoding never performs inference.
#[derive(Clone, Debug)]
pub struct CompiledPosteriorRow {
    request: std::sync::Arc<PosteriorRowRequestWire>,
    program: CompiledDoseGridPosterior,
    premises_digest: String,
    data_digest: String,
    _source_retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}
impl CompiledPosteriorRow {
    /// Validate premises/support/prior and retain a typed numerical program without fitting.
    pub fn compile(
        request: PosteriorRowRequestWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        let PosteriorRowRequestWire::DoseGrid(wire) = &request;
        let program = CompiledDoseGridPosterior::compile(&checked_dose_request(wire)?, ctx)
            .map_err(|e| from_estimation(e, "dose_grid"))?;
        let premises_digest = request
            .premises_digest()
            .map_err(|e| PosteriorRowArtifactError::Undecodable(e.to_string()))?;
        let data_digest = request
            .data_digest()
            .map_err(|e| PosteriorRowArtifactError::Undecodable(e.to_string()))?;
        let bytes = program.retained_source_bytes().checked_mul(2).ok_or_else(|| {
            execution_refusal("dose_grid.resource_refused", "retained wire size overflow")
        })?;
        let retention = ctx
            .acquire_admission(0, bytes)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        Ok(Self {
            request: std::sync::Arc::new(request),
            program,
            premises_digest,
            data_digest,
            _source_retention: std::sync::Arc::new(retention),
        })
    }
    /// Frozen scientific/data identity pair; no fitting or simulation.
    #[must_use]
    pub fn identities(&self) -> (&str, &str) {
        (&self.premises_digest, &self.data_digest)
    }
    /// Borrow the exact frozen request.
    #[must_use]
    pub fn request(&self) -> &PosteriorRowRequestWire {
        &self.request
    }
    /// Execute the retained original numerical engine; compatible retargeting reuses its coefficient posterior.
    pub fn execute(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<PosteriorRowResultWire, PosteriorRowArtifactError> {
        let posterior = self.program.execute(ctx).map_err(|e| from_estimation(e, "dose_grid"))?;
        let PosteriorRowRequestWire::DoseGrid(wire) = self.request.as_ref();
        let result = dose_grid_row(&wire.functional.kind, posterior.functional, &posterior);
        if !result.all_finite() {
            return Err(PosteriorRowArtifactError::Refused {
                code: antecedent_core::reason_code!("transport_numerical_failure"),
                detail: "bayesian_row.non_finite_result".into(),
                message: "posterior produced non-finite values".into(),
            });
        }
        Ok(result)
    }
    /// Retarget evaluation coordinates with unchanged data/prior/seed/draws.
    pub fn retarget(
        &self,
        grid: Vec<f64>,
        functional: DoseFunctionalWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(execution_refusal("dose_grid.cancelled", "retargeting cancelled"));
        }
        let bytes = self
            .program
            .retained_source_bytes()
            .checked_mul(2)
            .and_then(|n| n.checked_add((grid.len() as u64).checked_mul(16)?))
            .ok_or_else(|| {
                execution_refusal("dose_grid.resource_refused", "retargeted wire size overflow")
            })?;
        let retention = ctx
            .acquire_admission(0, bytes)
            .map_err(|e| execution_refusal("dose_grid.resource_refused", &e.to_string()))?;
        let mut request = self.request.as_ref().clone();
        let PosteriorRowRequestWire::DoseGrid(wire) = &mut request;
        wire.grid = grid;
        wire.functional = functional;
        let numerical = checked_dose_request(wire)?;
        let program = self
            .program
            .retarget(numerical.grid, numerical.functional, ctx)
            .map_err(|e| from_estimation(e, "dose_grid"))?;
        let premises_digest = request
            .premises_digest()
            .map_err(|e| PosteriorRowArtifactError::Undecodable(e.to_string()))?;
        Ok(Self {
            request: std::sync::Arc::new(request),
            program,
            premises_digest,
            data_digest: self.data_digest.clone(),
            _source_retention: std::sync::Arc::new(retention),
        })
    }
    /// Execute and separately encode the original posterior artifact format.
    pub fn artifact(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<PosteriorRowArtifactWire, PosteriorRowArtifactError> {
        PosteriorRowArtifactWire::from_executed(self.request.as_ref().clone(), self.execute(ctx)?)
    }
}
