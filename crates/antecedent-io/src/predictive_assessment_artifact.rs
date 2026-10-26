//! Predictive assessment artifact (`predictive_assessment_v1`).
//!
//! A bounded, checksummed, sectioned container holding one predictive
//! assessment: the predictive laws, the realized observations and the declared
//! design, with the scores and diagnostics computed from them.
//!
//! * `predictive_assessment_meta` (CBOR): version, feature marker, the
//!   point-only claim, the **unmeasured** calibration status, the closed interval
//!   status, the target, the validation design and dependence declarations, the
//!   scoring and diagnostic configuration, each predictor's source and fit
//!   declaration, the identity digests, the stored result labels and the standing
//!   caveats;
//! * `predictive_assessment_numbers` (little-endian `f64`): the realized
//!   outcomes, then every predictor's laws case by case (a law code followed by
//!   its parameters), then the stored result values;
//! * `predictive_assessment_keys` (bytes): the 32-byte observation identities,
//!   then each declared fit set.
//!
//! A consumer trusts none of the stored results. It rebuilds the assessment from
//! the stored laws and observations, re-scores every case (the same code the
//! producer ran), recomputes the identity digests and every score, difference
//! and diagnostic, and refuses unless each stored value is reproduced (bit for
//! bit on one platform; within `1e-12` relative across platforms, where `exp`,
//! `ln` and `erfc` may differ in the last bits). A resealed edit of a law, an
//! outcome, a declaration or a stored value is refused, and a consumer that
//! retained the identity independently refuses a consistently resealed
//! assessment of other laws or observations.
//!
//! Refusals carry the `predictive_assessment.wrong_contract` detail for a changed identity and
//! `predictive_assessment.report_replay_mismatch` for a stored value that does not recompute.
//!
//! What this does **not** say: no interval or significance level for a score or
//! a score difference (closed: `cell_not_licensed`), no calibration guarantee
//! (diagnostics are labelled and `unmeasured`), no causal validity (a score is
//! not a causal claim) and no probability that a claim holds.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_io_core::byte_cursor::RealCursor;
use antecedent_io_core::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use antecedent_validate::predictive_assessment::{
    AssessmentIdentity, DependenceDeclaration, EnsembleCrpsForm, FitDeclaration, ForecastLaw,
    Horizon, LawSource, MAX_CASES, MAX_JOINT_DIMENSION, MAX_LAW_ATOMS, MAX_PREDICTORS,
    ObservationKey, Outcome, OverlapPolicy, OverlapStatus, PredictiveAssessmentError,
    PredictiveAssessmentReport, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
    ProviderIdentity, ProviderTrustLabel, RealizedObservations, ScoreKind, ValidationDesign,
    assess_predictive_laws,
};
use serde::{Deserialize, Serialize};


/// The artifact major version this reader writes and accepts.
pub const PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const PREDICTIVE_ASSESSMENT_ARTIFACT_FEATURE: &str = "predictive_assessment_v1";
/// Calibration status: diagnostics of this route have no measured calibration.
pub const PREDICTIVE_ASSESSMENT_CALIBRATION: &str = "unmeasured";
/// Interval status: a score or score-difference interval is closed.
pub const PREDICTIVE_ASSESSMENT_INTERVAL: &str = "not_reported";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_PREDICTIVE_ASSESSMENT_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;
/// Most numbers (`f64` words) an artifact may carry.
pub const MAX_PREDICTIVE_ASSESSMENT_NUMBERS: usize = 32 * 1024 * 1024;
/// The standing caveats every artifact carries.
pub const PREDICTIVE_ASSESSMENT_CAVEATS: [&str; 3] = [
    "a predictive score is not causal validity and not the probability that a claim holds",
    "calibration, PIT, coverage and drift diagnostics are labelled and unmeasured: no \
     calibration guarantee, interval or significance level is claimed",
    "no interval is reported for a score or a score difference; exact finite-law scoring \
     supplies no sampling interval",
];

const ARTIFACT_KIND: &str = crate::artifact_kinds::PREDICTIVE_ASSESSMENT;
const META_SECTION: &str = "predictive_assessment_meta";
const NUMBERS_SECTION: &str = "predictive_assessment_numbers";
const KEYS_SECTION: &str = "predictive_assessment_keys";
const KEY_BYTES: usize = 32;

crate::sectioned_artifact_error!(
    /// Why a predictive assessment artifact was refused or could not be produced.
    PredictiveAssessmentArtifactError,
    "predictive assessment",
    "predictive_assessment",
    "result",
    true
);

type Error = PredictiveAssessmentArtifactError;

fn refused(error: &PredictiveAssessmentError) -> Error {
    Error::refused(error.reason_code(), error.detail(), error.message.clone())
}

fn malformed(text: &str) -> Error {
    Error::Malformed(text.to_owned())
}

// -- wire types ---------------------------------------------------------------------------

/// The predictive target on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetWire {
    /// The predicted quantity.
    pub quantity: String,
    /// Steps ahead; absent for a contemporaneous target.
    pub horizon_steps: Option<u32>,
    /// The population.
    pub population: String,
    /// The units.
    pub units: String,
}

impl TargetWire {
    fn from_target(target: &PredictiveTarget) -> Self {
        Self {
            quantity: target.quantity.clone(),
            horizon_steps: match target.horizon {
                Horizon::Contemporaneous => None,
                Horizon::Steps(steps) => Some(steps),
            },
            population: target.population.clone(),
            units: target.units.clone(),
        }
    }

    fn to_target(&self) -> PredictiveTarget {
        PredictiveTarget::new(
            self.quantity.clone(),
            self.horizon_steps.map_or(Horizon::Contemporaneous, Horizon::Steps),
            self.population.clone(),
            self.units.clone(),
        )
    }
}

/// The validation design on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesignWire {
    /// `held_out_split`, `rolling_origin`, `blocked` or `simulation_stream`.
    pub kind: String,
    /// Forecast steps ahead (rolling origin).
    pub horizon_steps: Option<u32>,
    /// Stream identity (simulation stream).
    pub stream: Option<String>,
    /// Origin (rolling origin) or block (blocked) of each case, canonical order.
    pub fold_labels: Vec<u64>,
}

/// The declared dependence on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceWire {
    /// `undeclared`, `iid`, `serial` or `cluster`.
    pub kind: String,
    /// Truncation lag (serial).
    pub truncation_lag: Option<u32>,
    /// Time index of each case (serial), canonical order.
    pub time_index: Vec<u64>,
    /// Cluster label of each case (cluster), canonical order.
    pub labels: Vec<u32>,
}

/// A provider object identity on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWire {
    /// Provider service or model identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider implementation or model version.
    pub version_id: String,
    /// Input data or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
}

/// Where a predictor's laws came from, on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceWire {
    /// `native`, `provider` or `callback`.
    pub class: String,
    /// Model label (native) or callback description (callback).
    pub label: Option<String>,
    /// Provider identity (provider).
    pub provider: Option<ProviderWire>,
    /// Retained provider trust (provider).
    pub trust: Option<String>,
}

/// What a predictor was fitted on, on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitWire {
    /// `fitted_on`, `not_fitted` or `undeclared`.
    pub kind: String,
    /// Number of fit keys stored in the keys section.
    pub keys: u64,
    /// Why undeclared.
    pub reason: Option<String>,
}

/// One predictor's declaration (its laws live in the numbers section).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorMeta {
    /// Unique name.
    pub name: String,
    /// Source of the laws.
    pub source: SourceWire,
    /// What the laws are about.
    pub target: TargetWire,
    /// What the laws were fitted on.
    pub fit: FitWire,
}

/// Identity digests on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityWire {
    /// Observations digest.
    pub observations: String,
    /// `(name, digest)` per predictor, by name.
    pub predictors: Vec<(String, String)>,
    /// Design and dependence digest.
    pub design: String,
    /// Configuration digest.
    pub configuration: String,
    /// Digest over all of the above.
    pub digest: String,
}

impl IdentityWire {
    fn from_identity(identity: &AssessmentIdentity) -> Self {
        Self {
            observations: identity.observations.clone(),
            predictors: identity.predictors.clone(),
            design: identity.design.clone(),
            configuration: identity.configuration.clone(),
            digest: identity.digest.clone(),
        }
    }

    fn to_identity(&self) -> AssessmentIdentity {
        AssessmentIdentity {
            observations: self.observations.clone(),
            predictors: self.predictors.clone(),
            design: self.design.clone(),
            configuration: self.configuration.clone(),
            digest: self.digest.clone(),
        }
    }
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictiveAssessmentMeta {
    /// Major version ([`PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION`]).
    pub version: u16,
    /// Feature marker ([`PREDICTIVE_ASSESSMENT_ARTIFACT_FEATURE`]).
    pub feature: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Always `not_established`.
    pub causal_validity: String,
    /// Always `not_reported`.
    pub interval: String,
    /// Number of cases.
    pub cases: u64,
    /// The outcome family: `real`, `category` or `vector`.
    pub outcome_family: String,
    /// The observation target.
    pub target: TargetWire,
    /// The validation design.
    pub design: DesignWire,
    /// The declared dependence.
    pub dependence: DependenceWire,
    /// Requested score names, in request order.
    pub scores: Vec<String>,
    /// `empirical_law` or `fair`.
    pub ensemble_crps: String,
    /// Number of PIT bins.
    pub pit_bins: u32,
    /// Number of reliability bins.
    pub reliability_bins: u32,
    /// Declared interval levels.
    pub interval_levels: Vec<f64>,
    /// `(candidate, reference)` comparisons.
    pub comparisons: Vec<(String, String)>,
    /// `label_only` or `require_held_out`.
    pub overlap_policy: String,
    /// The predictors, in declaration order.
    pub predictors: Vec<PredictorMeta>,
    /// Identity digests.
    pub identity: IdentityWire,
    /// Labels of the stored result numbers, in order.
    pub result_labels: Vec<String>,
    /// Stored result texts (overlap statuses, presence flags).
    pub result_texts: Vec<(String, String)>,
    /// Standing caveats.
    pub caveats: Vec<String>,
}

// -- flattened results --------------------------------------------------------------------

struct Flat {
    texts: Vec<(String, String)>,
    numbers: Vec<(String, f64)>,
}

fn flatten(report: &PredictiveAssessmentReport) -> Flat {
    let mut texts = Vec::new();
    let mut numbers = Vec::new();
    for predictor in &report.predictors {
        let name = &predictor.name;
        let overlap = match &predictor.overlap {
            OverlapStatus::HeldOutVerified { fitted_on } => {
                format!("held_out_verified:{fitted_on}")
            }
            OverlapStatus::InSample { shared, assessed } => {
                format!("in_sample:{shared}:{assessed}")
            }
            OverlapStatus::Unverified { reason } => format!("unverified:{reason}"),
            OverlapStatus::NotFitted => "not_fitted".to_owned(),
        };
        texts.push((format!("overlap.{name}"), overlap));
        for score in &predictor.scores {
            let kind = score.kind.name();
            numbers.push((format!("scores.{name}.{kind}"), score.mean));
            for fold in &score.folds {
                numbers.push((format!("folds.{name}.{kind}.{}.mean", fold.fold), fold.mean));
                numbers
                    .push((format!("folds.{name}.{kind}.{}.cases", fold.fold), fold.cases as f64));
            }
        }
        let diagnostics = &predictor.diagnostics;
        if let Some(pit) = &diagnostics.pit {
            for (b, count) in pit.counts.iter().enumerate() {
                numbers.push((format!("diagnostics.{name}.pit.{b}"), *count));
            }
            numbers.push((format!("diagnostics.{name}.pit.ks"), pit.ks_distance));
            numbers.push((format!("diagnostics.{name}.pit.chi_square"), pit.chi_square));
        }
        if let Some(reliability) = &diagnostics.reliability {
            numbers.push((format!("diagnostics.{name}.reliability.gap"), reliability.weighted_gap));
            for (b, bin) in reliability.bins.iter().enumerate() {
                numbers
                    .push((format!("diagnostics.{name}.reliability.{b}.count"), bin.count as f64));
                numbers.push((
                    format!("diagnostics.{name}.reliability.{b}.observed"),
                    bin.observed_frequency,
                ));
            }
        }
        if let Some(coverage) = &diagnostics.coverage {
            for (b, level) in coverage.iter().enumerate() {
                numbers.push((
                    format!("diagnostics.{name}.coverage.{b}.empirical"),
                    level.empirical_coverage,
                ));
                numbers.push((format!("diagnostics.{name}.coverage.{b}.width"), level.mean_width));
            }
        }
        if let Some(drift) = &diagnostics.drift {
            numbers.push((format!("diagnostics.{name}.drift.ks"), drift.ks_distance));
            numbers
                .push((format!("diagnostics.{name}.drift.psi"), drift.population_stability_index));
        }
        if let Some(sd) = diagnostics.mean_predictive_sd {
            numbers.push((format!("diagnostics.{name}.sharpness"), sd));
        }
    }
    for (i, difference) in report.differences.iter().enumerate() {
        let prefix = format!(
            "differences.{i}.{}.{}.{}",
            difference.candidate,
            difference.reference,
            difference.score.name()
        );
        numbers.push((format!("{prefix}.mean"), difference.mean_difference));
        texts.push((
            format!("{prefix}.dependence"),
            format!(
                "{}:{}:{}:{}",
                difference.dependence,
                difference.standard_error.is_some(),
                difference.dm_statistic.is_some(),
                difference.fit_overlap.map_or_else(|| "none".to_owned(), |n| n.to_string()),
            ),
        ));
        if let Some(se) = difference.standard_error {
            numbers.push((format!("{prefix}.se"), se));
        }
        if let Some(dm) = difference.dm_statistic {
            numbers.push((format!("{prefix}.dm"), dm));
        }
    }
    Flat { texts, numbers }
}

fn family_of(label: &str) -> &'static str {
    if label.starts_with("overlap.") {
        "overlap"
    } else if label.starts_with("scores.") || label.starts_with("folds.") {
        "scores"
    } else if label.starts_with("differences.") {
        "differences"
    } else {
        "diagnostics"
    }
}

/// Stored forecasts and scores reproduce the recomputed ones to the assessment's replay
/// tolerance, relative to the larger value (absolute below one). Identical bit patterns match,
/// so an undefined or infinite score that replays to the same value passes.
const REPLAY: ReplayTolerance = ReplayTolerance::new(
    antecedent_validate::predictive_assessment::REPLAY_TOLERANCE,
    ReplayScale::LargerOrOne,
    ReplayNonFinite::SameBits,
);

fn close(a: f64, b: f64) -> bool {
    REPLAY.close(a, b)
}

// -- numbers section ----------------------------------------------------------------------

struct Writer {
    values: Vec<f64>,
}

impl Writer {
    fn count(&mut self, value: usize) {
        self.values.push(value as f64);
    }

    fn law(&mut self, law: &ForecastLaw) {
        match law {
            ForecastLaw::Gaussian { mean, sd } => self.values.extend([0.0, *mean, *sd]),
            ForecastLaw::Bernoulli { probability } => self.values.extend([1.0, *probability]),
            ForecastLaw::Categorical { probabilities } => {
                self.values.push(2.0);
                self.count(probabilities.len());
                self.values.extend_from_slice(probabilities);
            }
            ForecastLaw::FiniteReal { support, probabilities } => {
                self.values.push(3.0);
                self.count(support.len());
                self.values.extend_from_slice(support);
                self.values.extend_from_slice(probabilities);
            }
            ForecastLaw::EmpiricalSample { draws } => {
                self.values.push(4.0);
                self.count(draws.len());
                self.values.extend_from_slice(draws);
            }
            ForecastLaw::PointEvaluated { log_density, cdf_below, cdf_at } => {
                self.values.extend([
                    5.0,
                    if log_density.is_some() { 1.0 } else { 0.0 },
                    log_density.unwrap_or(0.0),
                    *cdf_below,
                    *cdf_at,
                ]);
            }
            ForecastLaw::JointFinite { dimension, atoms, probabilities } => {
                self.values.push(6.0);
                self.count(*dimension);
                self.count(probabilities.len());
                self.values.extend_from_slice(atoms);
                self.values.extend_from_slice(probabilities);
            }
            ForecastLaw::JointSample { dimension, draws } => {
                self.values.push(7.0);
                self.count(*dimension);
                self.count(draws.len() / dimension.max(&1));
                self.values.extend_from_slice(draws);
            }
        }
    }

    fn outcome(&mut self, outcome: &Outcome) {
        match outcome {
            Outcome::Real(value) => self.values.push(*value),
            Outcome::Category(category) => self.values.push(f64::from(*category)),
            Outcome::Vector(values) => {
                self.count(values.len());
                self.values.extend_from_slice(values);
            }
        }
    }
}

struct Reader<'a> {
    values: RealCursor<'a, Error>,
}

fn numbers_end_early() -> Error {
    malformed("numbers section ends early")
}

impl<'a> Reader<'a> {
    fn new(values: &'a [f64]) -> Self {
        Self { values: RealCursor::new(values, numbers_end_early) }
    }

    fn take(&mut self) -> Result<f64, Error> {
        self.values.take()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is checked to be a non-negative integer no larger than `max`"
    )]
    fn count(&mut self, max: usize) -> Result<usize, Error> {
        let value = self.take()?;
        if !value.is_finite() || value < 0.0 || value.fract() > 0.0 || value > max as f64 {
            return Err(malformed("a stored count is not an integer within its bound"));
        }
        Ok(value as usize)
    }

    fn slice(&mut self, n: usize) -> Result<Vec<f64>, Error> {
        Ok(self.values.take_slice(n)?.to_vec())
    }

    fn law(&mut self) -> Result<ForecastLaw, Error> {
        let code = self.count(7)?;
        match code {
            0 => Ok(ForecastLaw::Gaussian { mean: self.take()?, sd: self.take()? }),
            1 => Ok(ForecastLaw::Bernoulli { probability: self.take()? }),
            2 => {
                let k = self.count(MAX_LAW_ATOMS)?;
                Ok(ForecastLaw::Categorical { probabilities: self.slice(k)? })
            }
            3 => {
                let n = self.count(MAX_LAW_ATOMS)?;
                let support = self.slice(n)?;
                Ok(ForecastLaw::FiniteReal { support, probabilities: self.slice(n)? })
            }
            4 => {
                let m = self.count(MAX_LAW_ATOMS)?;
                Ok(ForecastLaw::EmpiricalSample { draws: self.slice(m)? })
            }
            5 => {
                let has = self.count(1)?;
                let density = self.take()?;
                let cdf_below = self.take()?;
                let cdf_at = self.take()?;
                Ok(ForecastLaw::PointEvaluated {
                    log_density: (has == 1).then_some(density),
                    cdf_below,
                    cdf_at,
                })
            }
            6 => {
                let dimension = self.count(MAX_JOINT_DIMENSION)?;
                let n = self.count(MAX_LAW_ATOMS)?;
                let atoms = self.slice(n.saturating_mul(dimension))?;
                Ok(ForecastLaw::JointFinite { dimension, atoms, probabilities: self.slice(n)? })
            }
            _ => {
                let dimension = self.count(MAX_JOINT_DIMENSION)?;
                let m = self.count(MAX_LAW_ATOMS)?;
                Ok(ForecastLaw::JointSample {
                    dimension,
                    draws: self.slice(m.saturating_mul(dimension))?,
                })
            }
        }
    }

    fn outcome(&mut self, family: &str) -> Result<Outcome, Error> {
        match family {
            "real" => Ok(Outcome::Real(self.take()?)),
            "category" => {
                let category = self.count(u32::MAX as usize)?;
                Ok(Outcome::Category(u32::try_from(category).map_err(|_| malformed("category"))?))
            }
            "vector" => {
                let d = self.count(MAX_JOINT_DIMENSION)?;
                Ok(Outcome::Vector(self.slice(d)?))
            }
            _ => Err(malformed("outcome family")),
        }
    }
}

// -- declarations -------------------------------------------------------------------------

fn design_wire(design: &ValidationDesign) -> DesignWire {
    match design {
        ValidationDesign::HeldOutSplit => DesignWire {
            kind: "held_out_split".into(),
            horizon_steps: None,
            stream: None,
            fold_labels: Vec::new(),
        },
        ValidationDesign::RollingOrigin { origins, horizon_steps } => DesignWire {
            kind: "rolling_origin".into(),
            horizon_steps: Some(*horizon_steps),
            stream: None,
            fold_labels: origins.clone(),
        },
        ValidationDesign::Blocked { blocks } => DesignWire {
            kind: "blocked".into(),
            horizon_steps: None,
            stream: None,
            fold_labels: blocks.iter().map(|b| u64::from(*b)).collect(),
        },
        ValidationDesign::SimulationStream { stream } => DesignWire {
            kind: "simulation_stream".into(),
            horizon_steps: None,
            stream: Some(stream.clone()),
            fold_labels: Vec::new(),
        },
    }
}

fn design_of(wire: &DesignWire) -> Result<ValidationDesign, Error> {
    match wire.kind.as_str() {
        "held_out_split" => Ok(ValidationDesign::HeldOutSplit),
        "rolling_origin" => Ok(ValidationDesign::RollingOrigin {
            origins: wire.fold_labels.clone(),
            horizon_steps: wire.horizon_steps.ok_or_else(|| malformed("rolling origin horizon"))?,
        }),
        "blocked" => Ok(ValidationDesign::Blocked {
            blocks: wire
                .fold_labels
                .iter()
                .map(|b| u32::try_from(*b).map_err(|_| malformed("block label")))
                .collect::<Result<_, _>>()?,
        }),
        "simulation_stream" => Ok(ValidationDesign::SimulationStream {
            stream: wire.stream.clone().ok_or_else(|| malformed("simulation stream"))?,
        }),
        _ => Err(Error::UnsupportedSemantics("validation design")),
    }
}

fn dependence_wire(dependence: Option<&DependenceDeclaration>) -> DependenceWire {
    let mut wire = DependenceWire {
        kind: antecedent_validate::predictive_assessment::dependence_label(dependence).into(),
        truncation_lag: None,
        time_index: Vec::new(),
        labels: Vec::new(),
    };
    match dependence {
        Some(DependenceDeclaration::Serial { time_index, truncation_lag }) => {
            wire.truncation_lag = Some(*truncation_lag);
            time_index.clone_into(&mut wire.time_index);
        }
        Some(DependenceDeclaration::Cluster { labels }) => labels.clone_into(&mut wire.labels),
        Some(DependenceDeclaration::Iid) | None => {}
    }
    wire
}

fn dependence_of(wire: &DependenceWire) -> Result<Option<DependenceDeclaration>, Error> {
    match wire.kind.as_str() {
        "undeclared" => Ok(None),
        "iid" => Ok(Some(DependenceDeclaration::Iid)),
        "serial" => Ok(Some(DependenceDeclaration::Serial {
            time_index: wire.time_index.clone(),
            truncation_lag: wire.truncation_lag.ok_or_else(|| malformed("truncation lag"))?,
        })),
        "cluster" => Ok(Some(DependenceDeclaration::Cluster { labels: wire.labels.clone() })),
        _ => Err(Error::UnsupportedSemantics("dependence")),
    }
}

fn source_wire(source: &LawSource) -> SourceWire {
    match source {
        LawSource::Native { model } => SourceWire {
            class: "native".into(),
            label: Some(model.clone()),
            provider: None,
            trust: None,
        },
        LawSource::Provider { identity, trust } => SourceWire {
            class: "provider".into(),
            label: None,
            provider: Some(ProviderWire {
                provider_id: identity.provider_id.clone(),
                object_id: identity.object_id.clone(),
                version_id: identity.version_id.clone(),
                snapshot_id: identity.snapshot_id.clone(),
                request_id: identity.request_id.clone(),
            }),
            trust: Some(trust.name().to_owned()),
        },
        LawSource::Callback { description } => SourceWire {
            class: "callback".into(),
            label: Some(description.clone()),
            provider: None,
            trust: None,
        },
    }
}

fn source_of(wire: &SourceWire) -> Result<LawSource, Error> {
    match (wire.class.as_str(), &wire.label, &wire.provider, &wire.trust) {
        ("native", Some(label), None, None) => Ok(LawSource::Native { model: label.clone() }),
        ("callback", Some(label), None, None) => {
            Ok(LawSource::Callback { description: label.clone() })
        }
        ("provider", None, Some(provider), Some(trust)) => {
            let trust = match trust.as_str() {
                "unverified" => ProviderTrustLabel::Unverified,
                "externally_attested" => ProviderTrustLabel::ExternallyAttested,
                "verified_extension" => ProviderTrustLabel::ExactRequestVerified,
                _ => return Err(Error::UnsupportedSemantics("provider trust")),
            };
            Ok(LawSource::Provider {
                identity: ProviderIdentity {
                    provider_id: provider.provider_id.clone(),
                    object_id: provider.object_id.clone(),
                    version_id: provider.version_id.clone(),
                    snapshot_id: provider.snapshot_id.clone(),
                    request_id: provider.request_id.clone(),
                },
                trust,
            })
        }
        _ => Err(malformed("law source declaration")),
    }
}

fn outcome_family(outcomes: &[Outcome]) -> &'static str {
    match outcomes.first() {
        Some(Outcome::Category(_)) => "category",
        Some(Outcome::Vector(_)) => "vector",
        _ => "real",
    }
}

// -- encode / decode ----------------------------------------------------------------------

fn encode_parts(
    spec: &PredictiveAssessmentSpec,
    report: &PredictiveAssessmentReport,
) -> Result<(PredictiveAssessmentMeta, Vec<u8>, Vec<u8>), Error> {
    let flat = flatten(report);
    let mut writer = Writer { values: Vec::new() };
    for outcome in &spec.observations.outcomes {
        writer.outcome(outcome);
    }
    let mut keys = Vec::with_capacity(spec.observations.ids.len() * KEY_BYTES);
    for key in &spec.observations.ids {
        keys.extend_from_slice(key.as_bytes());
    }
    let mut predictors = Vec::with_capacity(spec.predictors.len());
    for predictor in &spec.predictors {
        for law in &predictor.laws {
            writer.law(law);
        }
        let fit = match &predictor.fit {
            FitDeclaration::FittedOn(fit_keys) => {
                let mut sorted = fit_keys.clone();
                sorted.sort_unstable();
                sorted.dedup();
                for key in &sorted {
                    keys.extend_from_slice(key.as_bytes());
                }
                FitWire { kind: "fitted_on".into(), keys: sorted.len() as u64, reason: None }
            }
            FitDeclaration::NotFitted => {
                FitWire { kind: "not_fitted".into(), keys: 0, reason: None }
            }
            FitDeclaration::Undeclared { reason } => {
                FitWire { kind: "undeclared".into(), keys: 0, reason: Some(reason.clone()) }
            }
        };
        predictors.push(PredictorMeta {
            name: predictor.name.clone(),
            source: source_wire(&predictor.source),
            target: TargetWire::from_target(&predictor.target),
            fit,
        });
    }
    let result_labels: Vec<String> = flat.numbers.iter().map(|(label, _)| label.clone()).collect();
    writer.values.extend(flat.numbers.iter().map(|(_, value)| *value));
    if writer.values.len() > MAX_PREDICTIVE_ASSESSMENT_NUMBERS {
        return Err(Error::LimitsExceeded("numbers"));
    }
    let meta = PredictiveAssessmentMeta {
        version: PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION,
        feature: PREDICTIVE_ASSESSMENT_ARTIFACT_FEATURE.to_owned(),
        inference_claim: report.inference_claim.to_owned(),
        calibration: PREDICTIVE_ASSESSMENT_CALIBRATION.to_owned(),
        causal_validity: report.causal_validity.to_owned(),
        interval: PREDICTIVE_ASSESSMENT_INTERVAL.to_owned(),
        cases: spec.observations.ids.len() as u64,
        outcome_family: outcome_family(&spec.observations.outcomes).to_owned(),
        target: TargetWire::from_target(&spec.observations.target),
        design: design_wire(&spec.design),
        dependence: dependence_wire(spec.dependence.as_ref()),
        scores: spec.scores.iter().map(|kind| kind.name().to_owned()).collect(),
        ensemble_crps: spec.ensemble_crps.name().to_owned(),
        pit_bins: u32::try_from(spec.pit_bins).map_err(|_| Error::LimitsExceeded("pit bins"))?,
        reliability_bins: u32::try_from(spec.reliability_bins)
            .map_err(|_| Error::LimitsExceeded("reliability bins"))?,
        interval_levels: spec.interval_levels.clone(),
        comparisons: spec.comparisons.clone(),
        overlap_policy: match spec.overlap_policy {
            OverlapPolicy::LabelOnly => "label_only".to_owned(),
            OverlapPolicy::RequireHeldOut => "require_held_out".to_owned(),
        },
        predictors,
        identity: IdentityWire::from_identity(&report.identity),
        result_labels,
        result_texts: flat.texts,
        caveats: PREDICTIVE_ASSESSMENT_CAVEATS.iter().map(|c| (*c).to_owned()).collect(),
    };
    let numbers = writer.values.iter().flat_map(|value| value.to_le_bytes()).collect();
    Ok((meta, numbers, keys))
}

fn encode_container(
    meta: &PredictiveAssessmentMeta,
    numbers: &[u8],
    keys: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, Error> {
    crate::artifact_codec::encode_sectioned(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "predictive_assessment_point_only_unmeasured",
            max_bytes: MAX_PREDICTIVE_ASSESSMENT_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[
            crate::OutSection {
                id: NUMBERS_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: numbers,
            },
            crate::OutSection { id: KEYS_SECTION, content_type: crate::CONTENT_BYTES, bytes: keys },
        ],
    )
}

fn decode_container(bytes: &[u8]) -> Result<(PredictiveAssessmentMeta, Vec<u8>, Vec<u8>), Error> {
    let (meta, parts) = crate::artifact_codec::decode_sectioned::<Error, PredictiveAssessmentMeta>(
        ARTIFACT_KIND,
        MAX_PREDICTIVE_ASSESSMENT_ARTIFACT_BYTES,
        META_SECTION,
        &[NUMBERS_SECTION, KEYS_SECTION],
        PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION,
        bytes,
    )?;
    let Ok([numbers, keys]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(malformed("unsupported container layout"));
    };
    Ok((meta, numbers, keys))
}

/// Encode a metadata, numbers and keys section as a checksummed container.
///
/// Hidden: the producer path is [`PredictiveAssessmentArtifact::to_bytes`]; tests use
/// this to build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts_for_test(
    meta: &PredictiveAssessmentMeta,
    numbers: &[u8],
    keys: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, Error> {
    encode_container(meta, numbers, keys, artifact_id)
}

/// Decode the three sections of a container, refusing another version before the
/// metadata is interpreted.
///
/// Hidden: see [`encode_parts_for_test`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts_for_test(
    bytes: &[u8],
) -> Result<(PredictiveAssessmentMeta, Vec<u8>, Vec<u8>), Error> {
    decode_container(bytes)
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct PredictiveAssessmentConsumeLimits {
    /// Most cases an artifact may carry.
    pub max_cases: usize,
    /// Most predictors an artifact may carry.
    pub max_predictors: usize,
    /// Most `f64` words the numbers section may carry.
    pub max_numbers: usize,
}

impl Default for PredictiveAssessmentConsumeLimits {
    fn default() -> Self {
        Self {
            max_cases: MAX_CASES,
            max_predictors: MAX_PREDICTORS,
            max_numbers: MAX_PREDICTIVE_ASSESSMENT_NUMBERS,
        }
    }
}

fn key_at(keys: &[u8], index: usize) -> ObservationKey {
    let mut bytes = [0_u8; KEY_BYTES];
    bytes.copy_from_slice(&keys[index * KEY_BYTES..(index + 1) * KEY_BYTES]);
    ObservationKey::from_bytes(bytes)
}

fn check_claims(meta: &PredictiveAssessmentMeta) -> Result<(), Error> {
    crate::artifact_codec::require_semantics::<Error>(
        meta.feature == PREDICTIVE_ASSESSMENT_ARTIFACT_FEATURE,
        "feature marker",
    )?;
    if meta.inference_claim != "point_only" {
        return Err(Error::UnsupportedSemantics("inference claim"));
    }
    if meta.calibration != PREDICTIVE_ASSESSMENT_CALIBRATION {
        return Err(Error::UnsupportedSemantics("calibration"));
    }
    if meta.causal_validity != "not_established" {
        return Err(Error::UnsupportedSemantics("causal validity"));
    }
    if meta.interval != PREDICTIVE_ASSESSMENT_INTERVAL {
        return Err(Error::UnsupportedSemantics("interval"));
    }
    if meta.caveats != PREDICTIVE_ASSESSMENT_CAVEATS {
        return Err(Error::UnsupportedSemantics("caveats"));
    }
    Ok(())
}

fn spec_of(
    meta: &PredictiveAssessmentMeta,
    numbers: &[f64],
    keys: &[u8],
    limits: &PredictiveAssessmentConsumeLimits,
) -> Result<PredictiveAssessmentSpec, Error> {
    let cases = usize::try_from(meta.cases).map_err(|_| Error::LimitsExceeded("cases"))?;
    if cases == 0 || cases > limits.max_cases {
        return Err(Error::LimitsExceeded("cases"));
    }
    if meta.predictors.is_empty() || meta.predictors.len() > limits.max_predictors {
        return Err(Error::LimitsExceeded("predictors"));
    }
    let fit_total: u64 =
        meta.predictors.iter().map(|p| p.fit.keys).fold(0_u64, u64::saturating_add);
    let expected_keys = u64::try_from(cases).unwrap_or(u64::MAX).saturating_add(fit_total);
    if keys.len() as u64 != expected_keys.saturating_mul(KEY_BYTES as u64) {
        return Err(malformed("keys section length"));
    }
    let result_count = meta.result_labels.len();
    if numbers.len() < result_count {
        return Err(malformed("numbers section length"));
    }
    let mut reader = Reader::new(&numbers[..numbers.len() - result_count]);
    let ids: Vec<ObservationKey> = (0..cases).map(|i| key_at(keys, i)).collect();
    let outcomes =
        (0..cases).map(|_| reader.outcome(&meta.outcome_family)).collect::<Result<Vec<_>, _>>()?;
    let mut fit_cursor = cases;
    let mut predictors = Vec::with_capacity(meta.predictors.len());
    for declared in &meta.predictors {
        let laws = (0..cases).map(|_| reader.law()).collect::<Result<Vec<_>, _>>()?;
        let fit = match declared.fit.kind.as_str() {
            "fitted_on" => {
                let count = usize::try_from(declared.fit.keys)
                    .map_err(|_| Error::LimitsExceeded("fit keys"))?;
                let fitted = (fit_cursor..fit_cursor + count).map(|i| key_at(keys, i)).collect();
                fit_cursor += count;
                FitDeclaration::FittedOn(fitted)
            }
            "not_fitted" => FitDeclaration::NotFitted,
            "undeclared" => FitDeclaration::Undeclared {
                reason: declared.fit.reason.clone().unwrap_or_default(),
            },
            _ => return Err(Error::UnsupportedSemantics("fit declaration")),
        };
        predictors.push(PredictorLaws {
            name: declared.name.clone(),
            source: source_of(&declared.source)?,
            target: declared.target.to_target(),
            laws,
            fit,
        });
    }
    if reader.values.remaining() != 0 {
        return Err(malformed("numbers section carries unread words"));
    }
    let scores = meta
        .scores
        .iter()
        .map(|name| ScoreKind::from_name(name).ok_or(Error::UnsupportedSemantics("score")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PredictiveAssessmentSpec {
        observations: RealizedObservations { target: meta.target.to_target(), ids, outcomes },
        predictors,
        design: design_of(&meta.design)?,
        dependence: dependence_of(&meta.dependence)?,
        scores,
        ensemble_crps: match meta.ensemble_crps.as_str() {
            "empirical_law" => EnsembleCrpsForm::EmpiricalLaw,
            "fair" => EnsembleCrpsForm::Fair,
            _ => return Err(Error::UnsupportedSemantics("ensemble crps form")),
        },
        pit_bins: meta.pit_bins as usize,
        reliability_bins: meta.reliability_bins as usize,
        interval_levels: meta.interval_levels.clone(),
        comparisons: meta.comparisons.clone(),
        overlap_policy: match meta.overlap_policy.as_str() {
            "label_only" => OverlapPolicy::LabelOnly,
            "require_held_out" => OverlapPolicy::RequireHeldOut,
            _ => return Err(Error::UnsupportedSemantics("overlap policy")),
        },
        score_interval: None,
    })
}

/// A produced or consumed predictive assessment: the declared laws and
/// observations, the recomputed report and the identity digests.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveAssessmentArtifact {
    meta: PredictiveAssessmentMeta,
    spec: PredictiveAssessmentSpec,
    report: PredictiveAssessmentReport,
}

impl PredictiveAssessmentArtifact {
    /// Assess `spec` (in its canonical case order) and seal the answer with its
    /// identity.
    ///
    /// # Errors
    /// Any refusal of the assessment contract (a typed [`Error::Refused`] with its
    /// registered reason code) and the format's bounds.
    pub fn seal(spec: &PredictiveAssessmentSpec, ctx: &ExecutionContext) -> Result<Self, Error> {
        let canonical = spec.canonicalized().map_err(|error| refused(&error))?;
        let report = assess_predictive_laws(&canonical, ctx).map_err(|error| refused(&error))?;
        let (meta, _, _) = encode_parts(&canonical, &report)?;
        Ok(Self { meta, spec: canonical, report })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &PredictiveAssessmentMeta {
        &self.meta
    }

    /// The assessment as declared, in canonical case order.
    #[must_use]
    pub fn spec(&self) -> &PredictiveAssessmentSpec {
        &self.spec
    }

    /// The recomputed report.
    #[must_use]
    pub fn report(&self) -> &PredictiveAssessmentReport {
        &self.report
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &AssessmentIdentity {
        &self.report.identity
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, Error> {
        let (meta, numbers, keys) = encode_parts(&self.spec, &self.report)?;
        encode_container(&meta, &numbers, &keys, artifact_id)
    }

    /// Consume an artifact by recomputation under the default limits.
    ///
    /// # Errors
    /// See [`Self::from_bytes_with_limits`].
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&AssessmentIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        Self::from_bytes_with_limits(
            bytes,
            expected,
            PredictiveAssessmentConsumeLimits::default(),
            ctx,
        )
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected` is an identity the consumer retained independently (for example
    /// from the producer's result); when given, every identity field must match
    /// it, so a resealed change of laws, observations, design or configuration is
    /// refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a bound
    /// exceeded, a changed identity ([`Error::IdentityMismatch`]), a stored value
    /// that does not replay ([`Error::ResultMismatch`]), or a refusal of the
    /// assessment contract on the stored declarations.
    pub fn from_bytes_with_limits(
        bytes: &[u8],
        expected: Option<&AssessmentIdentity>,
        limits: PredictiveAssessmentConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        let (meta, number_bytes, keys) = decode_container(bytes)?;
        check_claims(&meta)?;
        if number_bytes.len() % 8 != 0 {
            return Err(malformed("numbers section length"));
        }
        if number_bytes.len() / 8 > limits.max_numbers {
            return Err(Error::LimitsExceeded("numbers"));
        }
        let numbers = crate::width::f64s_from_le_bytes(&number_bytes);
        let spec = spec_of(&meta, &numbers, &keys, &limits)?;
        let report = assess_predictive_laws(&spec, ctx).map_err(|error| refused(&error))?;
        if let Some(field) = meta.identity.to_identity().first_difference(&report.identity) {
            return Err(Error::IdentityMismatch { field });
        }
        if let Some(expected) = expected {
            if let Some(field) = expected.first_difference(&report.identity) {
                return Err(Error::IdentityMismatch { field });
            }
        }
        let replayed = flatten(&report);
        let stored_values = &numbers[numbers.len() - meta.result_labels.len()..];
        if meta.result_texts != replayed.texts {
            let label = meta
                .result_texts
                .iter()
                .zip(&replayed.texts)
                .find(|(stored, fresh)| stored != fresh)
                .map_or("overlap", |(stored, _)| family_of(&stored.0));
            return Err(Error::ResultMismatch(label));
        }
        if meta.result_labels.len() != replayed.numbers.len() {
            return Err(Error::ResultMismatch("diagnostics"));
        }
        for ((label, fresh), (stored_label, stored)) in
            replayed.numbers.iter().zip(meta.result_labels.iter().zip(stored_values))
        {
            if label != stored_label || !close(*fresh, *stored) {
                return Err(Error::ResultMismatch(family_of(label)));
            }
        }
        Ok(Self { meta, spec, report })
    }
}
