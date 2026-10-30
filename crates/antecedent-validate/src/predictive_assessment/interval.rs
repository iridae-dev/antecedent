//! Sampling intervals for a mean score and for a mean score difference.
//!
//! A score of a fixed law on held-out observations is a mean of case scores; under a
//! declared sampling design that mean estimates the *expected* score of the law over
//! the design's population of cases. Its interval is the Student-t interval of the
//! mean with the variance the declared dependence implies:
//!
//! * `iid`: `s^2 / n`, `n - 1` degrees of freedom;
//! * `cluster`: the cluster-robust variance over `G` clusters, `G - 1` degrees of
//!   freedom, at least [`SCORE_INTERVAL_MIN_CLUSTERS`] clusters;
//! * `serial`: the Bartlett-kernel long-run variance with the declared truncation lag
//!   `L` (Newey and West 1987), with `n / (L + 1) - 1` degrees of freedom (the fixed-b
//!   correction in the form of a t reference), at most
//!   [`SCORE_INTERVAL_MAX_BANDWIDTH_FRACTION`] of the cases inside the bandwidth.
//!
//! The interval is published only for a predictor whose laws were verified held out of
//! the cases (or were never fitted), and only where a current attesting coverage record
//! measured this construction at this design, score and law family, with the number of
//! cases inside its measured range. The record is resolved from the shipped coverage
//! table; nothing a caller declares opens a route. An exact finite-law expected score is
//! not a sampling quantity and has no interval.
//!
//! Assumptions the interval rests on, stated on every result: cases are independent
//! (or dependent as declared), their score distribution has a finite variance, the law
//! was fixed before the cases were observed, and for a rolling-origin design the
//! estimand is the average expected score across the forecast origins.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use antecedent_io_core::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};
use antecedent_io_core::contract_section::CalibrationStatus;
use antecedent_stats::student_t_ppf;

use super::assess::{PredictiveAssessmentReport, PredictiveAssessmentSpec, PredictorAssessment};
use super::compare::variance_of_mean;
use super::design::{DependenceDeclaration, resolve_design};
use super::law::LawKind;
use super::overlap::OverlapStatus;
use super::refusal::{PredictiveAssessmentError, detail, refuse};
use super::score::ScoreKind;

/// Query axis name of every score-interval coverage record.
pub const SCORE_INTERVAL_QUERY: &str = "PredictiveScoreInterval";
/// Largest share of the cases the Bartlett bandwidth `L + 1` may span.
pub const SCORE_INTERVAL_MAX_BANDWIDTH_FRACTION: f64 = 0.1;
/// Fewest clusters a cluster-robust interval is published for.
pub const SCORE_INTERVAL_MIN_CLUSTERS: usize = 10;
/// Smallest nominal level a request may ask for.
pub const SCORE_INTERVAL_MIN_LEVEL: f64 = 0.5;
/// Largest nominal level a request may ask for.
pub const SCORE_INTERVAL_MAX_LEVEL: f64 = 0.9999;
/// Assumptions every interval states.
pub const SCORE_INTERVAL_ASSUMPTIONS: &str = "cases are independent or dependent as declared; \
    the case scores have a finite variance; the law was fixed before the cases were observed; \
    for a rolling-origin design the estimand is the average expected score over the forecast \
    origins";

/// What interval to build.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreIntervalQuery {
    /// The predictor whose mean score (or whose difference to `reference`) is estimated.
    pub predictor: String,
    /// A reference predictor: the interval is for the mean of `score(predictor) -
    /// score(reference)` over the shared cases.
    pub reference: Option<String>,
    /// The score.
    pub score: ScoreKind,
    /// Nominal two-sided level, in `[0.5, 0.9999]`.
    pub level: f64,
}

/// A published interval for an expected score or an expected score difference.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreInterval {
    /// The predictor.
    pub predictor: String,
    /// The reference predictor, for a difference.
    pub reference: Option<String>,
    /// The score.
    pub score: ScoreKind,
    /// Mean of the case scores (or of the paired differences).
    pub mean: f64,
    /// Standard error of the mean under the declared dependence.
    pub standard_error: f64,
    /// Degrees of freedom of the Student-t reference.
    pub degrees_of_freedom: f64,
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Nominal two-sided level.
    pub level: f64,
    /// Cases the mean is over.
    pub cases: usize,
    /// `iid`, `cluster` or `serial`.
    pub dependence: &'static str,
    /// Interval construction label.
    pub construction: &'static str,
    /// Governing coverage record (empty on an unlicensed construction).
    pub coverage_record: String,
    /// Commit the governing record was measured at (empty on an unlicensed construction).
    pub calibration_sha: String,
    /// Stated assumptions, [`SCORE_INTERVAL_ASSUMPTIONS`].
    pub assumptions: &'static str,
}

/// The facts an interval's coverage record is keyed on.
#[derive(Clone, Debug, PartialEq)]
pub struct IntervalFacts<'a> {
    /// Whether the interval is for a paired difference.
    pub difference: bool,
    /// `iid`, `serial` or `cluster`.
    pub dependence: &'a str,
    /// The score.
    pub score: ScoreKind,
    /// Law family name, e.g. `gaussian`.
    pub law: &'a str,
    /// Validation design name, e.g. `held_out_split`.
    pub design: &'a str,
    /// Overlap label of the predictor(s), e.g. `held_out_verified`.
    pub overlap: &'a str,
    /// Nominal level.
    pub level: f64,
    /// Number of cases.
    pub cases: usize,
}

/// An interval built from the declared design, with the calibration basis a coverage
/// record would have to match. Unlicensed: `coverage_record` is empty, and the value is
/// not a published claim. It exists so that a measurement can score the construction
/// itself; [`expected_score_interval`] is the only route that publishes.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstructedInterval {
    /// The interval, with no governing record.
    pub interval: ScoreInterval,
    /// The key and scope a coverage record must equal and cover.
    pub basis: CalibrationBasisWire,
}

fn interval_scope_refusal(message: String) -> PredictiveAssessmentError {
    refuse(reason_code!("cell_not_licensed"), detail::INTERVAL_SCOPE_OUTSIDE_MEASURED, message)
}

fn interval_overlap_label(
    name: &str,
    overlap: &OverlapStatus,
) -> Result<&'static str, PredictiveAssessmentError> {
    match overlap {
        OverlapStatus::HeldOutVerified { .. } => Ok("held_out_verified"),
        OverlapStatus::NotFitted => Ok("not_fitted"),
        OverlapStatus::InSample { shared, assessed } => Err(refuse(
            reason_code!("evidence_double_counted"),
            detail::IN_SAMPLE,
            format!(
                "predictor `{name}` was fitted on {shared} of the {assessed} assessed \
                 observations: its mean score is not an estimate of an expected score on new \
                 cases, so no sampling interval is built"
            ),
        )),
        OverlapStatus::Unverified { reason } => Err(refuse(
            reason_code!("evidence_overlap_unknown"),
            detail::OVERLAP_UNKNOWN,
            format!(
                "what predictor `{name}` was fitted on is not declared ({reason}); declare its \
                 fit set so that held-out can be verified before an interval is built"
            ),
        )),
    }
}

fn interval_law_family(
    spec: &PredictiveAssessmentSpec,
    names: &[&str],
) -> Result<String, PredictiveAssessmentError> {
    let mut families: Vec<LawKind> = Vec::new();
    for name in names {
        let predictor = spec.predictors.iter().find(|p| p.name == *name).ok_or_else(|| {
            interval_scope_refusal(format!("predictor `{name}` is not part of this assessment"))
        })?;
        for law in &predictor.laws {
            if !families.contains(&law.kind()) {
                families.push(law.kind());
            }
        }
    }
    if families.len() != 1 {
        return Err(interval_scope_refusal(
            "the laws are of more than one family; an interval is measured for one law family \
             at a time"
                .to_owned(),
        ));
    }
    Ok(families[0].name().to_owned())
}

/// `(interval method, dependence label)` of a dependence name.
#[must_use]
pub fn score_interval_construction(dependence: &str) -> (&'static str, &'static str) {
    match dependence {
        "cluster" => ("student_t_cluster_robust", "cluster"),
        "serial" => ("student_t_bartlett_hac", "serial"),
        _ => ("student_t_iid", "iid"),
    }
}

/// The calibration basis of one interval: the key a coverage record must equal and the
/// scope (the number of cases) it must cover.
#[must_use]
pub fn score_interval_basis(facts: &IntervalFacts<'_>) -> CalibrationBasisWire {
    let (construction, dependence) = score_interval_construction(facts.dependence);
    CalibrationBasisWire {
        key: CalibrationKeyWire {
            validation: "not_applicable".into(),
            query: SCORE_INTERVAL_QUERY.to_owned(),
            graph_class: "NotApplicable".to_owned(),
            structure: "fixed".to_owned(),
            modality: "case_scores".to_owned(),
            inference: "Frequentist".to_owned(),
            estimator: if facts.difference {
                "predictive_assessment_score_difference".to_owned()
            } else {
                "predictive_assessment_score_mean".to_owned()
            },
            interval_method: construction.to_owned(),
            se_kind: "analytic_mean_variance".to_owned(),
            dependence: dependence.to_owned(),
            posterior: String::new(),
            functional: format!(
                "score={};law={};design={};overlap={}",
                facts.score.name(),
                facts.law,
                facts.design,
                facts.overlap
            ),
            level: facts.level,
            identification: "point".to_owned(),
        },
        scope: CalibrationScopeWire {
            row_count: facts.cases as u64,
            replicates_ok: None,
            posterior_draws: None,
            unidentified_mass: 0.0,
        },
    }
}

/// Construct the interval for `query` from an assessment and the specification it was
/// computed from, without resolving a coverage record. A measurement scores this; a
/// caller publishes through [`expected_score_interval`].
///
/// # Errors
/// An unknown predictor or score, a level outside `[0.5, 0.9999]`, an undeclared
/// dependence, a predictor that is in sample or whose fit is undeclared, clusters fewer
/// than [`SCORE_INTERVAL_MIN_CLUSTERS`] or a bandwidth wider than
/// [`SCORE_INTERVAL_MAX_BANDWIDTH_FRACTION`] of the cases, or a degenerate sample
/// variance.
#[allow(clippy::too_many_lines, reason = "one construction per dependence declaration")]
pub fn construct_expected_score_interval(
    spec: &PredictiveAssessmentSpec,
    report: &PredictiveAssessmentReport,
    query: &ScoreIntervalQuery,
) -> Result<ConstructedInterval, PredictiveAssessmentError> {
    if !(query.level.is_finite()
        && query.level >= SCORE_INTERVAL_MIN_LEVEL
        && query.level <= SCORE_INTERVAL_MAX_LEVEL)
    {
        return Err(interval_scope_refusal(format!(
            "level {} is outside [{SCORE_INTERVAL_MIN_LEVEL}, {SCORE_INTERVAL_MAX_LEVEL}]",
            query.level
        )));
    }
    let find = |name: &str| {
        report.predictors.iter().find(|p| p.name == name).ok_or_else(|| {
            interval_scope_refusal(format!("predictor `{name}` is not part of this assessment"))
        })
    };
    let candidate = find(&query.predictor)?;
    let candidate_overlap = interval_overlap_label(&candidate.name, &candidate.overlap)?;
    let scores_of = |assessment: &PredictorAssessment| {
        assessment
            .scores
            .iter()
            .find(|s| s.kind == query.score)
            .map(|s| s.per_case.clone())
            .ok_or_else(|| {
                interval_scope_refusal(format!(
                    "score `{}` was not computed for predictor `{}`",
                    query.score.name(),
                    assessment.name
                ))
            })
    };
    let mut values = scores_of(candidate)?;
    let mut names = vec![query.predictor.as_str()];
    let mut overlap_label = candidate_overlap.to_owned();
    if let Some(reference) = &query.reference {
        let other = find(reference)?;
        let other_overlap = interval_overlap_label(&other.name, &other.overlap)?;
        let reference_scores = scores_of(other)?;
        values = values.iter().zip(&reference_scores).map(|(a, b)| a - b).collect();
        names.push(reference.as_str());
        if other_overlap != candidate_overlap {
            overlap_label = format!("{candidate_overlap}+{other_overlap}");
        }
    }
    let n = values.len();
    let dependence = spec.dependence.as_ref().ok_or_else(|| {
        refuse(
            reason_code!("sampling_dependence_unknown"),
            detail::INTERVAL_DEPENDENCE_UNDECLARED,
            "no dependence of the cases is declared and independence is never assumed; declare \
             iid, serial dependence with its truncation lag, or cluster dependence with the \
             cluster labels",
        )
    })?;
    let resolved = resolve_design(&spec.design, spec.dependence.as_ref(), &spec.observations.ids)?;
    let variance = variance_of_mean(&values, dependence, resolved.cluster.as_deref())?;
    let nf = n as f64;
    let (degrees_of_freedom, label) = match dependence {
        DependenceDeclaration::Iid => (nf - 1.0, "iid"),
        DependenceDeclaration::Serial { truncation_lag, .. } => {
            let bandwidth = f64::from(*truncation_lag) + 1.0;
            if bandwidth / nf > SCORE_INTERVAL_MAX_BANDWIDTH_FRACTION {
                return Err(interval_scope_refusal(format!(
                    "the Bartlett bandwidth {bandwidth} spans more than \
                     {SCORE_INTERVAL_MAX_BANDWIDTH_FRACTION} of the {n} cases; the interval is \
                     measured only for a bandwidth inside that share"
                )));
            }
            ((nf / bandwidth - 1.0).max(1.0), "serial")
        }
        DependenceDeclaration::Cluster { .. } => {
            let mut labels: Vec<u32> = resolved.cluster.clone().unwrap_or_default();
            labels.sort_unstable();
            labels.dedup();
            if labels.len() < SCORE_INTERVAL_MIN_CLUSTERS {
                return Err(interval_scope_refusal(format!(
                    "{} clusters are declared; the cluster-robust interval is measured for at \
                     least {SCORE_INTERVAL_MIN_CLUSTERS}",
                    labels.len()
                )));
            }
            (labels.len() as f64 - 1.0, "cluster")
        }
    };
    let standard_error = variance.sqrt();
    if !(standard_error.is_finite() && standard_error > 0.0) {
        return Err(interval_scope_refusal(
            "the case scores have no sampling variance (all equal, or one cluster dominates); \
             no interval is built"
                .to_owned(),
        ));
    }
    let law = interval_law_family(spec, &names)?;
    let basis = score_interval_basis(&IntervalFacts {
        difference: query.reference.is_some(),
        dependence: label,
        score: query.score,
        law: &law,
        design: report.design,
        overlap: &overlap_label,
        level: query.level,
        cases: n,
    });
    let tail = 1.0 - (1.0 - query.level) / 2.0;
    let critical = student_t_ppf(tail, degrees_of_freedom);
    let mean = values.iter().sum::<f64>() / nf;
    let half = critical * standard_error;
    Ok(ConstructedInterval {
        interval: ScoreInterval {
            predictor: query.predictor.clone(),
            reference: query.reference.clone(),
            score: query.score,
            mean,
            standard_error,
            degrees_of_freedom,
            lower: mean - half,
            upper: mean + half,
            level: query.level,
            cases: n,
            dependence: label,
            construction: score_interval_construction(label).0,
            coverage_record: String::new(),
            calibration_sha: String::new(),
            assumptions: SCORE_INTERVAL_ASSUMPTIONS,
        },
        basis,
    })
}

/// Build the interval for `query` and publish it.
///
/// The record is resolved from the shipped coverage table
/// ([`antecedent_io_core::calibration::calibration_slot`]): the interval is returned only
/// when a current attesting non-boundary record measured exactly this construction.
///
/// # Errors
/// As [`construct_expected_score_interval`], and `cell_not_licensed` with
/// `predictive_assessment.interval_scope_outside_measured` when no current attesting
/// coverage record measured this construction at this design, score, law family, level
/// and number of cases.
pub fn expected_score_interval(
    spec: &PredictiveAssessmentSpec,
    report: &PredictiveAssessmentReport,
    query: &ScoreIntervalQuery,
) -> Result<ScoreInterval, PredictiveAssessmentError> {
    let ConstructedInterval { mut interval, basis } =
        construct_expected_score_interval(spec, report, query)?;
    let slot = calibration_slot(&basis);
    match (&slot.status, &slot.record_id, &slot.calibration_sha) {
        (CalibrationStatus::Calibrated, Some(id), Some(sha)) => {
            interval.coverage_record.clone_from(id);
            interval.calibration_sha.clone_from(sha);
            Ok(interval)
        }
        _ => Err(interval_scope_refusal(format!(
            "no current attesting coverage record measured the {} interval ({}; {} dependence) \
             with {} cases at level {} ({})",
            if query.reference.is_some() { "difference" } else { "mean" },
            basis.key.functional,
            interval.dependence,
            interval.cases,
            query.level,
            slot.reason.as_deref().unwrap_or("construction_not_measured")
        ))),
    }
}
