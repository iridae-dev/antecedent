//! The assessment: declared inputs in, scores, differences and diagnostics out.
//!
//! [`assess_predictive_laws`] scores every predictor's laws against the realized
//! observations under a declared validation design and dependence, compares
//! predictors pairwise on their shared observations, and attaches the
//! calibration diagnostics. All sums are formed in a canonical case order, so
//! the result does not depend on how the cases were listed.
//!
//! What the report is not: an interval (see [`super::expected_score_interval`] for the
//! interval of an expected score), a calibration guarantee or a causal claim. See [`PredictiveAssessmentReport::request_score_interval`],
//! [`PredictiveAssessmentReport::request_causal_validity`] and
//! [`PredictiveAssessmentReport::request_score_as_probability`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::convert::Infallible;

use antecedent_core::{ExecutionContext, reason_code};

use super::calibration::{
    CalibrationDiagnostics, check_configuration, diagnose, distribution_drift,
};
use super::compare::{ScoreDifference, difference};
use super::design::{
    DependenceDeclaration, ResolvedDesign, ValidationDesign, dependence_label, resolve_design,
};
use super::identity::{
    AssessmentIdentity, ConfigurationRef, configuration_digest, design_digest, observations_digest,
    predictor_digest, whole_digest,
};
use super::law::{
    ForecastLaw, Horizon, MAX_PREDICTORS, ObservationKey, Outcome, PredictiveTarget, PredictorLaws,
    RealizedObservations,
};
use super::overlap::{OverlapPolicy, OverlapStatus, decide_overlap, fit_overlap};
use super::refusal::{
    PredictiveAssessmentError, bounds, cancelled, detail, invalid, refuse, unsupported,
};
use super::score::{EnsembleCrpsForm, ScoreKind, score_case_validated};

/// Largest total scoring work an assessment accepts.
pub const MAX_ASSESSMENT_WORK: u64 = 4_000_000_000;
/// The family name every report carries.
pub const ASSESSMENT_FAMILY: &str = "predictive_assessment";
/// The inference claim of every report: scores are point values.
pub const ASSESSMENT_INFERENCE_CLAIM: &str = "point_only";
/// What the report says about causal validity: nothing is established.
pub const ASSESSMENT_CAUSAL_VALIDITY: &str = "not_established";
/// Cases between cancellation checks.
const CANCELLATION_STRIDE: usize = 256;

/// A request for a score or score-difference interval. Always refused.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreIntervalRequest {
    /// Requested nominal level.
    pub level: f64,
    /// The sampling design or method the interval would rest on.
    pub method: String,
}

/// A predictive assessment as declared.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveAssessmentSpec {
    /// The realized observations, with their identities and target.
    pub observations: RealizedObservations,
    /// The predictors to assess (at most [`MAX_PREDICTORS`]).
    pub predictors: Vec<PredictorLaws>,
    /// How the held-out cases were formed.
    pub design: ValidationDesign,
    /// How the cases depend on each other; `None` withholds every standard error.
    pub dependence: Option<DependenceDeclaration>,
    /// The scores to compute, each for every predictor.
    pub scores: Vec<ScoreKind>,
    /// How an empirical sample's CRPS is formed.
    pub ensemble_crps: EnsembleCrpsForm,
    /// Number of PIT histogram bins.
    pub pit_bins: usize,
    /// Number of reliability bins.
    pub reliability_bins: usize,
    /// Central prediction-interval levels whose empirical coverage is reported.
    pub interval_levels: Vec<f64>,
    /// `(candidate, reference)` pairs to compare; empty compares every other
    /// predictor with the first.
    pub comparisons: Vec<(String, String)>,
    /// What to do with in-sample or unverified predictors.
    pub overlap_policy: OverlapPolicy,
    /// A score or score-difference interval request; always refused.
    pub score_interval: Option<ScoreIntervalRequest>,
}

impl PredictiveAssessmentSpec {
    /// A specification with the default diagnostics (ten PIT and reliability
    /// bins, 50% and 90% central intervals), no declared dependence and no
    /// held-out requirement.
    #[must_use]
    pub fn new(
        observations: RealizedObservations,
        predictors: Vec<PredictorLaws>,
        design: ValidationDesign,
        scores: Vec<ScoreKind>,
    ) -> Self {
        Self {
            observations,
            predictors,
            design,
            dependence: None,
            scores,
            ensemble_crps: EnsembleCrpsForm::EmpiricalLaw,
            pit_bins: 10,
            reliability_bins: 10,
            interval_levels: vec![0.5, 0.9],
            comparisons: Vec::new(),
            overlap_policy: OverlapPolicy::LabelOnly,
            score_interval: None,
        }
    }
}

impl PredictiveAssessmentSpec {
    /// The same assessment with every per-case array (observation identities,
    /// outcomes, every predictor's laws, and the design and dependence
    /// declarations) permuted into the canonical case order the assessment sums
    /// in. Assessing the canonical form gives the identical result, and it is
    /// the form an artifact stores.
    ///
    /// # Errors
    /// A design or dependence declaration whose arrays do not have one entry per
    /// case, or an iid declaration for a temporal or clustered design.
    pub fn canonicalized(&self) -> Result<Self, PredictiveAssessmentError> {
        let resolved =
            resolve_design(&self.design, self.dependence.as_ref(), &self.observations.ids)?;
        let order = &resolved.order;
        let mut out = self.clone();
        out.observations.ids = canonical(self.observations.ids.as_slice(), order);
        out.observations.outcomes = canonical(self.observations.outcomes.as_slice(), order);
        for predictor in &mut out.predictors {
            if predictor.laws.len() == order.len() {
                predictor.laws = canonical(predictor.laws.as_slice(), order);
            }
        }
        match &mut out.design {
            ValidationDesign::RollingOrigin { origins, .. } => {
                *origins = canonical(origins.as_slice(), order)
            }
            ValidationDesign::Blocked { blocks } => *blocks = canonical(blocks.as_slice(), order),
            ValidationDesign::HeldOutSplit | ValidationDesign::SimulationStream { .. } => {}
        }
        match &mut out.dependence {
            Some(DependenceDeclaration::Serial { time_index, .. }) => {
                *time_index = canonical(time_index.as_slice(), order);
            }
            Some(DependenceDeclaration::Cluster { labels }) => {
                *labels = canonical(labels.as_slice(), order)
            }
            Some(DependenceDeclaration::Iid) | None => {}
        }
        Ok(out)
    }
}

/// Mean score over the cases of one fold.
#[derive(Clone, Debug, PartialEq)]
pub struct FoldScore {
    /// The fold label (forecast origin or block).
    pub fold: u64,
    /// Cases in the fold.
    pub cases: usize,
    /// Mean score over the fold's cases.
    pub mean: f64,
}

/// One score of one predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSummary {
    /// The score.
    pub kind: ScoreKind,
    /// Mean over the cases.
    pub mean: f64,
    /// Number of cases.
    pub cases: usize,
    /// Per-case scores in canonical case order.
    pub per_case: Vec<f64>,
    /// Mean score per fold, ascending; empty for designs without folds.
    pub folds: Vec<FoldScore>,
}

/// The assessment of one predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictorAssessment {
    /// The predictor's name.
    pub name: String,
    /// Class of the law source: `native`, `provider`.
    pub source: &'static str,
    /// Overlap of the assessed observations with what the law was fitted on.
    pub overlap: OverlapStatus,
    /// The requested scores.
    pub scores: Vec<ScoreSummary>,
    /// Calibration and sharpness diagnostics.
    pub diagnostics: CalibrationDiagnostics,
}

/// The withheld uncertainty statement of every assessment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AssessmentUncertainty {
    /// `not reported`: no sampling interval for a score or a score difference.
    pub sampling_interval: &'static str,
    /// Registered reason code of the withheld interval.
    pub reason_code: &'static str,
    /// Namespaced detail.
    pub detail: &'static str,
}

impl AssessmentUncertainty {
    /// The status every report carries.
    #[must_use]
    pub const fn withheld() -> Self {
        Self {
            sampling_interval: "not reported",
            reason_code: reason_code!("cell_not_licensed"),
            detail: detail::INTERVAL_NOT_LICENSED,
        }
    }
}

/// The result of assessing predictive laws.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveAssessmentReport {
    /// [`ASSESSMENT_FAMILY`].
    pub family: &'static str,
    /// [`ASSESSMENT_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
    /// [`ASSESSMENT_CAUSAL_VALIDITY`].
    pub causal_validity: &'static str,
    /// The quantity, horizon, population and units assessed.
    pub target: PredictiveTarget,
    /// Name of the validation design.
    pub design: &'static str,
    /// Name of the declared dependence (`undeclared` when none).
    pub dependence: &'static str,
    /// Number of cases.
    pub cases: usize,
    /// Observation keys in canonical case order.
    pub case_keys: Vec<ObservationKey>,
    /// Per-predictor assessments, in declaration order.
    pub predictors: Vec<PredictorAssessment>,
    /// Score differences, in comparison order, each score in request order.
    pub differences: Vec<ScoreDifference>,
    /// Withheld sampling-interval status.
    pub uncertainty: AssessmentUncertainty,
    /// Identities of the inputs.
    pub identity: AssessmentIdentity,
}

impl PredictiveAssessmentReport {
    /// Refuse an unscoped score or score-difference interval request: an interval is
    /// for one predictor's score (or one pair's difference) at one level, asked with
    /// [`super::expected_score_interval`].
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `predictive_assessment.interval_not_licensed`.
    pub fn request_score_interval(&self) -> Result<Infallible, PredictiveAssessmentError> {
        Err(interval_refusal())
    }

    /// Refuse reading a predictive score as causal validity.
    ///
    /// # Errors
    /// Always `route_not_supported` with `predictive_assessment.not_a_causal_claim`.
    pub fn request_causal_validity(&self) -> Result<Infallible, PredictiveAssessmentError> {
        Err(unsupported(
            detail::NOT_A_CAUSAL_CLAIM,
            "a predictive score measures how well a law predicts the assessed observations; it \
             does not establish that the law, or any model behind it, is causally valid, \
             identifies an effect, or predicts under intervention. Causal validity needs an \
             identification argument and refutation, not a score",
        ))
    }

    /// Refuse reading a model score as the probability that a claim holds.
    ///
    /// # Errors
    /// Always `route_not_supported` with `predictive_assessment.score_not_probability`.
    pub fn request_score_as_probability(&self) -> Result<Infallible, PredictiveAssessmentError> {
        Err(unsupported(
            detail::SCORE_NOT_PROBABILITY,
            "a model score is a loss of a predictive law on the assessed observations, not a \
             probability that a scientific claim or model is true. A probability of a claim \
             needs a declared prior over claims and a likelihood, which this assessment does not \
             supply",
        ))
    }
}

pub(crate) fn interval_refusal() -> PredictiveAssessmentError {
    refuse(
        reason_code!("cell_not_licensed"),
        detail::INTERVAL_NOT_LICENSED,
        "the report states point scores, paired differences and the Diebold-Mariano style \
         statistic as diagnostics and carries no interval. An interval is a statement about \
         the expected score of one predictor (or the expected difference of two) under the \
         declared sampling design: name the predictor, the score and the level and ask \
         for the expected-score interval, which publishes it only for a held-out or unfitted predictor under \
         a declared dependence where a current coverage record measured the construction. An \
         exact finite-law expected score has no sampling interval to report",
    )
}

fn sorted_keys(keys: &[ObservationKey]) -> Vec<ObservationKey> {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    sorted
}

/// Validate everything that does not need the design resolved.
fn validate_spec(spec: &PredictiveAssessmentSpec) -> Result<(), PredictiveAssessmentError> {
    if spec.score_interval.is_some() {
        return Err(interval_refusal());
    }
    spec.observations.validate()?;
    let cases = spec.observations.ids.len();
    if let ValidationDesign::RollingOrigin { horizon_steps, .. } = &spec.design {
        if spec.observations.target.horizon != Horizon::Steps(*horizon_steps) {
            return Err(refuse(
                reason_code!("quantity_semantics_mismatch"),
                detail::HORIZON_MISMATCH,
                format!(
                    "the rolling-origin design forecasts {horizon_steps} steps ahead but the \
                     observations are {}; declare the target horizon the design evaluates",
                    spec.observations.target.horizon.describe()
                ),
            ));
        }
    }
    if spec.predictors.is_empty() || spec.predictors.len() > MAX_PREDICTORS {
        return Err(bounds(format!(
            "{} predictors supplied; 1 to {MAX_PREDICTORS} are admitted",
            spec.predictors.len()
        )));
    }
    let mut names: Vec<&str> = spec.predictors.iter().map(|p| p.name.as_str()).collect();
    names.sort_unstable();
    if names.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(detail::INVALID_LAW, "predictor names must be distinct"));
    }
    for predictor in &spec.predictors {
        predictor.validate(cases)?;
        predictor.target.check_matches(&spec.observations.target, &predictor.name)?;
    }
    if spec.scores.is_empty() {
        return Err(invalid(
            detail::SCORE_NOT_DEFINED,
            "request at least one score: the log score, the CRPS, the Brier score or the \
             energy score",
        ));
    }
    let mut kinds = spec.scores.clone();
    kinds.sort_unstable();
    if kinds.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid(detail::SCORE_NOT_DEFINED, "a score is requested more than once"));
    }
    check_configuration(spec.pit_bins, spec.reliability_bins, &spec.interval_levels)?;
    let work: u64 = spec
        .predictors
        .iter()
        .flat_map(|p| p.laws.iter().map(ForecastLaw::work))
        .fold(0_u64, u64::saturating_add)
        .saturating_mul(spec.scores.len() as u64);
    if work > MAX_ASSESSMENT_WORK {
        return Err(bounds(format!(
            "the assessment needs about {work} scoring operations; at most \
             {MAX_ASSESSMENT_WORK} are admitted. Assess fewer cases or smaller samples"
        )));
    }
    Ok(())
}

fn resolve_comparisons(
    spec: &PredictiveAssessmentSpec,
) -> Result<Vec<(usize, usize)>, PredictiveAssessmentError> {
    let index = |name: &str| {
        spec.predictors.iter().position(|p| p.name == name).ok_or_else(|| {
            invalid(
                detail::INVALID_LAW,
                format!("comparison names predictor `{name}`, which is not assessed"),
            )
        })
    };
    if spec.comparisons.is_empty() {
        return Ok((1..spec.predictors.len()).map(|i| (i, 0)).collect());
    }
    let mut pairs = Vec::with_capacity(spec.comparisons.len());
    for (candidate, reference) in &spec.comparisons {
        let (c, r) = (index(candidate)?, index(reference)?);
        if c == r {
            return Err(invalid(
                detail::INVALID_LAW,
                format!("`{candidate}` is compared with itself"),
            ));
        }
        pairs.push((c, r));
    }
    Ok(pairs)
}

fn score_predictor(
    spec: &PredictiveAssessmentSpec,
    predictor: &PredictorLaws,
    resolved: &ResolvedDesign,
    ctx: &ExecutionContext,
) -> Result<Vec<ScoreSummary>, PredictiveAssessmentError> {
    let mut summaries = Vec::with_capacity(spec.scores.len());
    for &kind in &spec.scores {
        let mut per_case = Vec::with_capacity(resolved.order.len());
        for (position, &i) in resolved.order.iter().enumerate() {
            if position % CANCELLATION_STRIDE == 0 && ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            per_case.push(score_case_validated(
                kind,
                spec.ensemble_crps,
                &predictor.laws[i],
                &spec.observations.outcomes[i],
            )?);
        }
        let mean = per_case.iter().sum::<f64>() / per_case.len() as f64;
        let mut by_fold: BTreeMap<u64, (usize, f64)> = BTreeMap::new();
        for (score, fold) in per_case.iter().zip(&resolved.fold_of) {
            if let Some(label) = fold {
                let entry = by_fold.entry(*label).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += score;
            }
        }
        let folds = by_fold
            .into_iter()
            .map(|(fold, (cases, sum))| FoldScore { fold, cases, mean: sum / cases as f64 })
            .collect();
        summaries.push(ScoreSummary { kind, mean, cases: per_case.len(), per_case, folds });
    }
    Ok(summaries)
}

fn canonical<T: Clone>(values: &[T], order: &[usize]) -> Vec<T> {
    order.iter().map(|&i| values[i].clone()).collect()
}

/// Assess predictive laws against realized observations.
///
/// # Errors
/// A typed [`PredictiveAssessmentError`]: a target or horizon mismatch, a
/// malformed law or observation set, a callback law, a score a law does not
/// support, an iid declaration for a dependent design, a declared held-out
/// requirement that is not met, a requested interval, an exceeded bound or a
/// cancellation.
pub fn assess_predictive_laws(
    spec: &PredictiveAssessmentSpec,
    ctx: &ExecutionContext,
) -> Result<PredictiveAssessmentReport, PredictiveAssessmentError> {
    validate_spec(spec)?;
    let keys = &spec.observations.ids;
    let resolved = resolve_design(&spec.design, spec.dependence.as_ref(), keys)?;
    let assessed = sorted_keys(keys);
    let pairs = resolve_comparisons(spec)?;
    let outcomes_c: Vec<Outcome> =
        canonical(spec.observations.outcomes.as_slice(), &resolved.order);
    let mut assessments = Vec::with_capacity(spec.predictors.len());
    for predictor in &spec.predictors {
        let overlap =
            decide_overlap(&predictor.name, &predictor.fit, &assessed, spec.overlap_policy)?;
        let scores = score_predictor(spec, predictor, &resolved, ctx)?;
        let laws_c: Vec<ForecastLaw> = canonical(predictor.laws.as_slice(), &resolved.order);
        let mut diagnostics = diagnose(
            &laws_c,
            &outcomes_c,
            spec.pit_bins,
            spec.reliability_bins,
            &spec.interval_levels,
        );
        if resolved.time.is_some() || matches!(spec.design, ValidationDesign::RollingOrigin { .. })
        {
            // Case order is a time order here: compare the first half with the second.
            diagnostics.drift = diagnostics.pit.as_ref().and_then(|pit| {
                let (first, second) = pit.mid_pit.split_at(pit.mid_pit.len() / 2);
                distribution_drift(first, second, spec.pit_bins).ok()
            });
        }
        assessments.push(PredictorAssessment {
            name: predictor.name.clone(),
            source: predictor.source.class(),
            overlap,
            scores,
            diagnostics,
        });
    }
    let differences = compare_predictors(spec, &assessments, &pairs, &resolved)?;
    let identity = assessment_identity(spec, &resolved);
    Ok(PredictiveAssessmentReport {
        family: ASSESSMENT_FAMILY,
        inference_claim: ASSESSMENT_INFERENCE_CLAIM,
        causal_validity: ASSESSMENT_CAUSAL_VALIDITY,
        target: spec.observations.target.clone(),
        design: spec.design.name(),
        dependence: dependence_label(spec.dependence.as_ref()),
        cases: keys.len(),
        case_keys: canonical(keys.as_slice(), &resolved.order),
        predictors: assessments,
        differences,
        uncertainty: AssessmentUncertainty::withheld(),
        identity,
    })
}

fn compare_predictors(
    spec: &PredictiveAssessmentSpec,
    assessments: &[PredictorAssessment],
    pairs: &[(usize, usize)],
    resolved: &ResolvedDesign,
) -> Result<Vec<ScoreDifference>, PredictiveAssessmentError> {
    let mut out = Vec::new();
    for &(c, r) in pairs {
        let overlap = fit_overlap(&spec.predictors[c].fit, &spec.predictors[r].fit);
        for (candidate, reference) in assessments[c].scores.iter().zip(&assessments[r].scores) {
            out.push(difference(
                candidate.kind,
                (&assessments[c].name, &candidate.per_case),
                (&assessments[r].name, &reference.per_case),
                spec.dependence.as_ref(),
                resolved.cluster.as_deref(),
                overlap,
            )?);
        }
    }
    Ok(out)
}

/// The identities of a validated specification.
pub(crate) fn assessment_identity(
    spec: &PredictiveAssessmentSpec,
    resolved: &ResolvedDesign,
) -> AssessmentIdentity {
    let observations = observations_digest(
        &spec.observations.target,
        &spec.observations.ids,
        &spec.observations.outcomes,
        &resolved.order,
    );
    let mut predictors: Vec<(String, String)> = spec
        .predictors
        .iter()
        .map(|p| (p.name.clone(), predictor_digest(p, &resolved.order)))
        .collect();
    predictors.sort();
    let design = design_digest(&spec.design, spec.dependence.as_ref(), resolved);
    let configuration = configuration_digest(&ConfigurationRef {
        scores: &spec.scores,
        ensemble_crps: spec.ensemble_crps,
        pit_bins: spec.pit_bins,
        reliability_bins: spec.reliability_bins,
        interval_levels: &spec.interval_levels,
        comparisons: &spec.comparisons,
        overlap_policy: spec.overlap_policy,
    });
    let digest = whole_digest(&observations, &predictors, &design, &configuration);
    AssessmentIdentity { observations, predictors, design, configuration, digest }
}

/// Relative tolerance of a replay comparison: transcendental functions
/// (`exp`, `ln`, `erfc`) may differ in the last bits across platforms.
pub const REPLAY_TOLERANCE: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
        || (a - b).abs() <= REPLAY_TOLERANCE * 1.0_f64.max(a.abs()).max(b.abs())
}

fn close_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => close(x, y),
        _ => false,
    }
}

fn close_all(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

fn diagnostics_difference(
    a: &CalibrationDiagnostics,
    b: &CalibrationDiagnostics,
) -> Option<&'static str> {
    let pit_same = match (&a.pit, &b.pit) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.bins == y.bins
                && close_all(&x.counts, &y.counts)
                && close(x.max_bin_deviation, y.max_bin_deviation)
                && close(x.chi_square, y.chi_square)
                && close(x.ks_distance, y.ks_distance)
                && close_all(&x.mid_pit, &y.mid_pit)
        }
        _ => false,
    };
    if !pit_same {
        return Some("pit");
    }
    let reliability_same = match (&a.reliability, &b.reliability) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            close(x.weighted_gap, y.weighted_gap)
                && x.bins.len() == y.bins.len()
                && x.bins.iter().zip(&y.bins).all(|(p, q)| {
                    p.count == q.count
                        && close(p.mean_forecast, q.mean_forecast)
                        && close(p.observed_frequency, q.observed_frequency)
                })
        }
        _ => false,
    };
    if !reliability_same {
        return Some("reliability");
    }
    let coverage_same = match (&a.coverage, &b.coverage) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.len() == y.len()
                && x.iter().zip(y).all(|(p, q)| {
                    close(p.level, q.level)
                        && close(p.empirical_coverage, q.empirical_coverage)
                        && close(p.mean_width, q.mean_width)
                })
        }
        _ => false,
    };
    if !coverage_same {
        return Some("coverage");
    }
    let drift_same = match (&a.drift, &b.drift) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.reference_cases == y.reference_cases
                && x.comparison_cases == y.comparison_cases
                && close(x.ks_distance, y.ks_distance)
                && close(x.population_stability_index, y.population_stability_index)
        }
        _ => false,
    };
    if !drift_same {
        return Some("drift");
    }
    if close_opt(a.mean_predictive_sd, b.mean_predictive_sd) { None } else { Some("sharpness") }
}

impl PredictiveAssessmentReport {
    /// The first quantity that differs from `other`, comparing numbers within
    /// [`REPLAY_TOLERANCE`] and everything else exactly; `None` when `other` is
    /// the same assessment.
    #[must_use]
    pub fn first_replay_difference(&self, other: &Self) -> Option<&'static str> {
        if self.identity != other.identity || self.cases != other.cases {
            return Some("identity");
        }
        if self.case_keys != other.case_keys
            || self.design != other.design
            || self.dependence != other.dependence
            || self.target != other.target
        {
            return Some("design");
        }
        if self.predictors.len() != other.predictors.len() {
            return Some("predictors");
        }
        for (a, b) in self.predictors.iter().zip(&other.predictors) {
            if a.name != b.name || a.overlap != b.overlap || a.source != b.source {
                return Some("overlap");
            }
            if a.scores.len() != b.scores.len() {
                return Some("scores");
            }
            for (x, y) in a.scores.iter().zip(&b.scores) {
                let folds_same = x.folds.len() == y.folds.len()
                    && x.folds.iter().zip(&y.folds).all(|(p, q)| {
                        p.fold == q.fold && p.cases == q.cases && close(p.mean, q.mean)
                    });
                if x.kind != y.kind
                    || x.cases != y.cases
                    || !close(x.mean, y.mean)
                    || !close_all(&x.per_case, &y.per_case)
                    || !folds_same
                {
                    return Some("scores");
                }
            }
            if let Some(what) = diagnostics_difference(&a.diagnostics, &b.diagnostics) {
                return Some(what);
            }
        }
        if self.differences.len() != other.differences.len() {
            return Some("differences");
        }
        for (a, b) in self.differences.iter().zip(&other.differences) {
            let same = a.score == b.score
                && a.candidate == b.candidate
                && a.reference == b.reference
                && a.shared_observations == b.shared_observations
                && a.dependence == b.dependence
                && a.fit_overlap == b.fit_overlap
                && close(a.mean_difference, b.mean_difference)
                && close_opt(a.standard_error, b.standard_error)
                && close_opt(a.dm_statistic, b.dm_statistic);
            if !same {
                return Some("differences");
            }
        }
        None
    }
}
