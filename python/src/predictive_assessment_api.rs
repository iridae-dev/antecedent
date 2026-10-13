//! Python bridge for predictive assessment.
//!
//! Python declares predictive laws, realized observations (with identities), the
//! validation design and the dependence of the cases; the target and horizon
//! checks, the proper scores, the paired differences, the diagnostics, the
//! overlap check, the artifact and every refusal are Rust's. A refusal comes
//! back as structured JSON for the Python layer to raise as a typed exception.
//!
//! Nothing here estimates dependence, reports an interval for a score, or reads
//! a score as causal validity: those requests come back as the typed refusals of
//! the contract.

use antecedent::analysis::predictive_assessment::{
    PredictiveAssessmentConsumeLimits, PredictiveAssessmentRefusal, PreparedPredictiveAssessment,
    consume_predictive_assessment_artifact, fit_declaration_from_ledger,
    observation_keys_from_ledger,
};
use antecedent_contract::evidence_ledger::SourceId;
use antecedent_validate::predictive_assessment::{
    AssessmentIdentity, DependenceDeclaration, EnsembleCrpsForm, FiniteTruth, FitDeclaration,
    ForecastLaw, Horizon, LawSource, ObservationKey, Outcome, OverlapPolicy, OverlapStatus,
    PredictiveAssessmentReport, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
    ProviderIdentity, ProviderTrustLabel, RealizedObservations, ScoreIntervalRequest, ScoreKind,
    ValidationDesign, distribution_drift, exact_expected_score, score_case,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::measurement_model_api::{
    list, member, object, opt_text, parse, present, real, reals, small, text, texts, unsigned,
};
use crate::transport_common::execution_context;

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

type Refused<T> = (Option<T>, Option<String>);

fn refusal_json(refusal: &PredictiveAssessmentRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "predictive_assessment",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
    .to_string()
}

fn contract_refusal(
    error: &antecedent_validate::predictive_assessment::PredictiveAssessmentError,
) -> String {
    json!({
        "code": error.code,
        "stage": "predictive_assessment",
        "detail": error.detail,
        "message": error.message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn unhex32(value: &str) -> Option<[u8; 32]> {
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    };
    let mut out = [0_u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(out)
}

fn key_of(value: &Json) -> PyResult<ObservationKey> {
    value
        .as_str()
        .and_then(unhex32)
        .map(ObservationKey::from_bytes)
        .ok_or_else(|| crate::value_err("an observation key is 64 hexadecimal characters"))
}

fn keys(map: &Map<String, Json>, name: &str) -> PyResult<Vec<ObservationKey>> {
    list(map, name)?.iter().map(key_of).collect()
}

fn u64s(map: &Map<String, Json>, name: &str) -> PyResult<Vec<u64>> {
    list(map, name)?
        .iter()
        .map(|v| {
            v.as_u64().ok_or_else(|| {
                crate::value_err(format!("`{name}` holds a value that is not a count"))
            })
        })
        .collect()
}

fn u32s(map: &Map<String, Json>, name: &str) -> PyResult<Vec<u32>> {
    u64s(map, name)?
        .into_iter()
        .map(|n| {
            u32::try_from(n)
                .map_err(|_| crate::value_err(format!("`{name}` holds a value above 32 bits")))
        })
        .collect()
}

fn parse_target(value: &Json) -> PyResult<PredictiveTarget> {
    let map = object(value, "predictive target")?;
    let horizon = match present(map, "horizon_steps") {
        None => Horizon::Contemporaneous,
        Some(steps) => Horizon::Steps(
            steps
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| crate::value_err("`horizon_steps` must be a 32-bit count"))?,
        ),
    };
    Ok(PredictiveTarget::new(
        text(map, "quantity")?,
        horizon,
        text(map, "population")?,
        text(map, "units")?,
    ))
}

fn parse_law(value: &Json) -> PyResult<ForecastLaw> {
    let map = object(value, "predictive law")?;
    let dimension = |map: &Map<String, Json>| -> PyResult<usize> {
        usize::try_from(unsigned(map, "dimension")?)
            .map_err(|_| crate::value_err("`dimension` does not fit"))
    };
    Ok(match text(map, "kind")?.as_str() {
        "gaussian" => ForecastLaw::Gaussian { mean: real(map, "mean")?, sd: real(map, "sd")? },
        "bernoulli" => ForecastLaw::Bernoulli { probability: real(map, "probability")? },
        "categorical" => ForecastLaw::Categorical { probabilities: reals(map, "probabilities")? },
        "finite_real" => ForecastLaw::FiniteReal {
            support: reals(map, "support")?,
            probabilities: reals(map, "probabilities")?,
        },
        "empirical_sample" => ForecastLaw::EmpiricalSample { draws: reals(map, "draws")? },
        "point_evaluated" => ForecastLaw::PointEvaluated {
            log_density: present(map, "log_density")
                .map(|v| {
                    v.as_f64().ok_or_else(|| crate::value_err("`log_density` must be a number"))
                })
                .transpose()?,
            cdf_below: real(map, "cdf_below")?,
            cdf_at: real(map, "cdf_at")?,
        },
        "joint_finite" => ForecastLaw::JointFinite {
            dimension: dimension(map)?,
            atoms: reals(map, "atoms")?,
            probabilities: reals(map, "probabilities")?,
        },
        "joint_sample" => {
            ForecastLaw::JointSample { dimension: dimension(map)?, draws: reals(map, "draws")? }
        }
        other => return Err(crate::value_err(format!("unknown law kind `{other}`"))),
    })
}

fn parse_outcomes(value: &Json) -> PyResult<Vec<Outcome>> {
    let map = object(value, "outcomes")?;
    let values = list(map, "values")?;
    match text(map, "family")?.as_str() {
        "real" => values
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(Outcome::Real)
                    .ok_or_else(|| crate::value_err("a real outcome must be a number"))
            })
            .collect(),
        "category" => values
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .map(Outcome::Category)
                    .ok_or_else(|| crate::value_err("a category outcome must be a count"))
            })
            .collect(),
        "vector" => values
            .iter()
            .map(|row| {
                row.as_array()
                    .ok_or_else(|| crate::value_err("a vector outcome must be a list"))?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| crate::value_err("a vector entry must be a number"))
                    })
                    .collect::<PyResult<Vec<f64>>>()
                    .map(Outcome::Vector)
            })
            .collect(),
        other => Err(crate::value_err(format!("unknown outcome family `{other}`"))),
    }
}

fn parse_source(value: &Json) -> PyResult<LawSource> {
    let map = object(value, "law source")?;
    Ok(match text(map, "class")?.as_str() {
        "native" => LawSource::Native { model: text(map, "model")? },
        "callback" => LawSource::Callback { description: text(map, "description")? },
        "provider" => {
            let id = object(member(map, "identity")?, "provider identity")?;
            let trust = match text(map, "trust")?.as_str() {
                "unverified" => ProviderTrustLabel::Unverified,
                "externally_attested" => ProviderTrustLabel::ExternallyAttested,
                "verified_extension" => ProviderTrustLabel::ExactRequestVerified,
                other => {
                    return Err(crate::value_err(format!("unknown provider trust `{other}`")));
                }
            };
            LawSource::Provider {
                identity: ProviderIdentity {
                    provider_id: text(id, "provider_id")?,
                    object_id: text(id, "object_id")?,
                    version_id: text(id, "version_id")?,
                    snapshot_id: text(id, "snapshot_id")?,
                    request_id: text(id, "request_id")?,
                },
                trust,
            }
        }
        other => return Err(crate::value_err(format!("unknown law source `{other}`"))),
    })
}

fn parse_fit(value: &Json) -> PyResult<FitDeclaration> {
    let map = object(value, "fit declaration")?;
    Ok(match text(map, "kind")?.as_str() {
        "fitted_on" => FitDeclaration::FittedOn(keys(map, "keys")?),
        "not_fitted" => FitDeclaration::NotFitted,
        "undeclared" => FitDeclaration::Undeclared {
            reason: opt_text(map, "reason")?.unwrap_or_else(|| "not declared".to_owned()),
        },
        other => return Err(crate::value_err(format!("unknown fit declaration `{other}`"))),
    })
}

fn parse_design(value: &Json) -> PyResult<ValidationDesign> {
    let map = object(value, "validation design")?;
    Ok(match text(map, "kind")?.as_str() {
        "held_out_split" => ValidationDesign::HeldOutSplit,
        "rolling_origin" => ValidationDesign::RollingOrigin {
            origins: u64s(map, "origins")?,
            horizon_steps: small(map, "horizon_steps")?,
        },
        "blocked" => ValidationDesign::Blocked { blocks: u32s(map, "blocks")? },
        "simulation_stream" => ValidationDesign::SimulationStream { stream: text(map, "stream")? },
        other => return Err(crate::value_err(format!("unknown validation design `{other}`"))),
    })
}

fn parse_dependence(value: &Json) -> PyResult<Option<DependenceDeclaration>> {
    if value.is_null() {
        return Ok(None);
    }
    let map = object(value, "dependence declaration")?;
    Ok(Some(match text(map, "kind")?.as_str() {
        "iid" => DependenceDeclaration::Iid,
        "serial" => DependenceDeclaration::Serial {
            time_index: u64s(map, "time_index")?,
            truncation_lag: small(map, "truncation_lag")?,
        },
        "cluster" => DependenceDeclaration::Cluster { labels: u32s(map, "labels")? },
        other => return Err(crate::value_err(format!("unknown dependence `{other}`"))),
    }))
}

fn parse_score(name: &str) -> PyResult<ScoreKind> {
    ScoreKind::from_name(name).ok_or_else(|| {
        crate::value_err(format!(
            "unknown score `{name}`; use log_score, crps, brier or energy_score"
        ))
    })
}

fn parse_form(name: &str) -> PyResult<EnsembleCrpsForm> {
    match name {
        "empirical_law" => Ok(EnsembleCrpsForm::EmpiricalLaw),
        "fair" => Ok(EnsembleCrpsForm::Fair),
        other => Err(crate::value_err(format!(
            "unknown ensemble CRPS form `{other}`; use empirical_law or fair"
        ))),
    }
}

fn parse_spec(value: &Json) -> PyResult<PredictiveAssessmentSpec> {
    let map = object(value, "predictive assessment")?;
    let observed = object(member(map, "observations")?, "observations")?;
    let observations = RealizedObservations {
        target: parse_target(member(observed, "target")?)?,
        ids: keys(observed, "ids")?,
        outcomes: parse_outcomes(member(observed, "outcomes")?)?,
    };
    let mut predictors = Vec::new();
    for entry in list(map, "predictors")? {
        let p = object(entry, "predictor")?;
        predictors.push(PredictorLaws {
            name: text(p, "name")?,
            source: parse_source(member(p, "source")?)?,
            target: parse_target(member(p, "target")?)?,
            laws: list(p, "laws")?.iter().map(parse_law).collect::<PyResult<Vec<_>>>()?,
            fit: parse_fit(member(p, "fit")?)?,
        });
    }
    let scores =
        texts(map, "scores")?.iter().map(|name| parse_score(name)).collect::<PyResult<Vec<_>>>()?;
    let mut spec = PredictiveAssessmentSpec::new(
        observations,
        predictors,
        parse_design(member(map, "design")?)?,
        scores,
    );
    spec.dependence = parse_dependence(member(map, "dependence")?)?;
    spec.ensemble_crps = parse_form(&text(map, "ensemble_crps")?)?;
    spec.pit_bins = usize::try_from(unsigned(map, "pit_bins")?)
        .map_err(|_| crate::value_err("`pit_bins` does not fit"))?;
    spec.reliability_bins = usize::try_from(unsigned(map, "reliability_bins")?)
        .map_err(|_| crate::value_err("`reliability_bins` does not fit"))?;
    spec.interval_levels = reals(map, "interval_levels")?;
    spec.comparisons = list(map, "comparisons")?
        .iter()
        .map(|pair| {
            let pair = pair
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| crate::value_err("a comparison is a [candidate, reference] pair"))?;
            match (pair[0].as_str(), pair[1].as_str()) {
                (Some(a), Some(b)) => Ok((a.to_owned(), b.to_owned())),
                _ => Err(crate::value_err("a comparison names two predictors")),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    spec.overlap_policy = match text(map, "overlap_policy")?.as_str() {
        "label_only" => OverlapPolicy::LabelOnly,
        "require_held_out" => OverlapPolicy::RequireHeldOut,
        other => return Err(crate::value_err(format!("unknown overlap policy `{other}`"))),
    };
    if let Some(request) = present(map, "score_interval") {
        let request = object(request, "score interval request")?;
        spec.score_interval = Some(ScoreIntervalRequest {
            level: real(request, "level")?,
            method: text(request, "method")?,
        });
    }
    Ok(spec)
}

fn parse_identity(value: &Json) -> PyResult<AssessmentIdentity> {
    let map = object(value, "assessment identity")?;
    let predictors = list(map, "predictors")?
        .iter()
        .map(|pair| {
            let pair = pair
                .as_array()
                .filter(|p| p.len() == 2)
                .ok_or_else(|| crate::value_err("a predictor identity is a [name, digest] pair"))?;
            match (pair[0].as_str(), pair[1].as_str()) {
                (Some(a), Some(b)) => Ok((a.to_owned(), b.to_owned())),
                _ => Err(crate::value_err("a predictor identity is two strings")),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(AssessmentIdentity {
        observations: text(map, "observations")?,
        predictors,
        design: text(map, "design")?,
        configuration: text(map, "configuration")?,
        digest: text(map, "digest")?,
    })
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

fn identity_json(identity: &AssessmentIdentity) -> Json {
    json!({
        "observations": identity.observations,
        "predictors": identity.predictors.iter().map(|(n, d)| json!([n, d])).collect::<Vec<_>>(),
        "design": identity.design,
        "configuration": identity.configuration,
        "digest": identity.digest,
    })
}

fn target_json(target: &PredictiveTarget) -> Json {
    json!({
        "quantity": target.quantity,
        "horizon_steps": match target.horizon {
            Horizon::Contemporaneous => Json::Null,
            Horizon::Steps(steps) => json!(steps),
        },
        "population": target.population,
        "units": target.units,
    })
}

fn overlap_json(status: &OverlapStatus) -> Json {
    match status {
        OverlapStatus::HeldOutVerified { fitted_on } => {
            json!({"label": status.label(), "fitted_on": fitted_on, "shared": 0, "reason": Json::Null})
        }
        OverlapStatus::InSample { shared, assessed } => json!({
            "label": status.label(), "shared": shared, "assessed": assessed, "reason": Json::Null
        }),
        OverlapStatus::Unverified { reason } => {
            json!({"label": status.label(), "shared": 0, "reason": reason})
        }
        OverlapStatus::NotFitted => {
            json!({"label": status.label(), "shared": 0, "reason": Json::Null})
        }
    }
}

fn drift_json(drift: &antecedent_validate::predictive_assessment::DistributionDrift) -> Json {
    json!({
        "label": drift.label,
        "calibration": drift.calibration,
        "route": drift.route,
        "reference_cases": drift.reference_cases,
        "comparison_cases": drift.comparison_cases,
        "ks_distance": drift.ks_distance,
        "population_stability_index": drift.population_stability_index,
        "bins": drift.bins,
    })
}

fn report_json(report: &PredictiveAssessmentReport) -> Json {
    let predictors: Vec<Json> = report
        .predictors
        .iter()
        .map(|p| {
            let d = &p.diagnostics;
            json!({
                "name": p.name,
                "source": p.source,
                "overlap": overlap_json(&p.overlap),
                "scores": p.scores.iter().map(|s| json!({
                    "kind": s.kind.name(),
                    "mean": s.mean,
                    "cases": s.cases,
                    "per_case": s.per_case,
                    "folds": s.folds.iter().map(|f| json!({
                        "fold": f.fold, "cases": f.cases, "mean": f.mean
                    })).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "diagnostics": {
                    "label": d.label,
                    "calibration": d.calibration,
                    "pit": d.pit.as_ref().map(|pit| json!({
                        "bins": pit.bins,
                        "counts": pit.counts,
                        "max_bin_deviation": pit.max_bin_deviation,
                        "chi_square": pit.chi_square,
                        "ks_distance": pit.ks_distance,
                        "mid_pit": pit.mid_pit,
                    })),
                    "reliability": d.reliability.as_ref().map(|r| json!({
                        "weighted_gap": r.weighted_gap,
                        "bins": r.bins.iter().map(|b| json!({
                            "lower": b.lower, "upper": b.upper, "count": b.count,
                            "mean_forecast": b.mean_forecast,
                            "observed_frequency": b.observed_frequency,
                        })).collect::<Vec<_>>(),
                    })),
                    "coverage": d.coverage.as_ref().map(|levels| levels.iter().map(|c| json!({
                        "level": c.level,
                        "empirical_coverage": c.empirical_coverage,
                        "mean_width": c.mean_width,
                    })).collect::<Vec<_>>()),
                    "mean_predictive_sd": d.mean_predictive_sd,
                    "drift": d.drift.as_ref().map(drift_json),
                },
            })
        })
        .collect();
    let differences: Vec<Json> = report
        .differences
        .iter()
        .map(|d| {
            json!({
                "score": d.score.name(),
                "candidate": d.candidate,
                "reference": d.reference,
                "shared_observations": d.shared_observations,
                "mean_difference": d.mean_difference,
                "standard_error": d.standard_error,
                "dm_statistic": d.dm_statistic,
                "dependence": d.dependence,
                "pairing": d.pairing,
                "fit_overlap": d.fit_overlap,
                "label": d.label,
                "calibration": d.calibration,
                "interval": d.interval,
            })
        })
        .collect();
    json!({
        "family": report.family,
        "inference_claim": report.inference_claim,
        "causal_validity": report.causal_validity,
        "target": target_json(&report.target),
        "design": report.design,
        "dependence": report.dependence,
        "cases": report.cases,
        "case_keys": report.case_keys.iter().map(ObservationKey::to_hex).collect::<Vec<_>>(),
        "predictors": predictors,
        "differences": differences,
        "uncertainty": {
            "sampling_interval": report.uncertainty.sampling_interval,
            "reason_code": report.uncertainty.reason_code,
            "detail": report.uncertainty.detail,
        },
        "identity": identity_json(&report.identity),
    })
}

// ---------------------------------------------------------------------------
// Handles.
// ---------------------------------------------------------------------------

/// A sealed predictive assessment: the report, its identity and its artifact.
#[pyclass(name = "PredictiveAssessmentHandle", skip_from_py_object)]
pub(crate) struct PredictiveAssessmentHandle {
    inner: PreparedPredictiveAssessment,
}

#[pymethods]
impl PredictiveAssessmentHandle {
    /// The report as JSON.
    fn report_json(&self) -> String {
        report_json(self.inner.report()).to_string()
    }

    /// The identity digests as JSON.
    fn identity_json(&self) -> String {
        identity_json(self.inner.identity()).to_string()
    }

    /// Write the `predictive_assessment_v1` artifact.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Refused<Py<PyBytes>> {
        match self.inner.export(artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// Always the typed refusal: no interval for a score or a score difference.
    fn request_score_interval(&self) -> String {
        match self.inner.request_score_interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }

    /// Always the typed refusal: a score is not causal validity.
    fn request_causal_validity(&self) -> String {
        match self.inner.request_causal_validity() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }

    /// Always the typed refusal: a model score is not the probability of a claim.
    fn request_score_as_probability(&self) -> String {
        match self.inner.request_score_as_probability() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Assess the declared laws against the realized observations.
#[pyfunction]
#[pyo3(signature = (spec_json, *, seed=0, cancel=None))]
fn predictive_assessment_assess(
    spec_json: &str,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<PredictiveAssessmentHandle>> {
    let spec = parse_spec(&parse(spec_json, "predictive assessment")?)?;
    let ctx = execution_context(seed, None, cancel);
    Ok(match PreparedPredictiveAssessment::assess(&spec, &ctx) {
        Ok(inner) => (Some(PredictiveAssessmentHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// Independently consume an artifact by re-scoring every case, against an
/// identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None, max_cases=None, *, cancel=None))]
fn predictive_assessment_consume(
    artifact: &[u8],
    expected_identity_json: Option<&str>,
    max_cases: Option<usize>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<PredictiveAssessmentHandle>> {
    let expected = expected_identity_json
        .map(|text| parse(text, "assessment identity").and_then(|v| parse_identity(&v)))
        .transpose()?;
    let mut limits = PredictiveAssessmentConsumeLimits::default();
    if let Some(max_cases) = max_cases {
        limits.max_cases = max_cases;
    }
    let ctx = execution_context(0, None, cancel);
    Ok(match consume_predictive_assessment_artifact(artifact, expected.as_ref(), limits, &ctx) {
        Ok(inner) => (Some(PredictiveAssessmentHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// One proper score of one law at one realized outcome.
#[pyfunction]
fn predictive_assessment_score_case(
    kind: &str,
    form: &str,
    law_json: &str,
    outcome_json: &str,
) -> PyResult<Refused<f64>> {
    let kind = parse_score(kind)?;
    let form = parse_form(form)?;
    let law = parse_law(&parse(law_json, "predictive law")?)?;
    let outcomes = parse_outcomes(&parse(outcome_json, "outcome")?)?;
    let [outcome] = outcomes.as_slice() else {
        return Err(crate::value_err("score_case takes exactly one outcome"));
    };
    Ok(match law.validate().and_then(|()| score_case(kind, form, &law, outcome)) {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(contract_refusal(&error))),
    })
}

/// The exact expected score of a forecast under a finite true law.
#[pyfunction]
fn predictive_assessment_exact_expected_score(
    kind: &str,
    law_json: &str,
    truth_json: &str,
) -> PyResult<Refused<f64>> {
    let kind = parse_score(kind)?;
    let law = parse_law(&parse(law_json, "predictive law")?)?;
    let truth = parse(truth_json, "finite truth")?;
    let map = object(&truth, "finite truth")?;
    let truth = match text(map, "kind")?.as_str() {
        "real" => FiniteTruth::Real {
            support: reals(map, "support")?,
            probabilities: reals(map, "probabilities")?,
        },
        "category" => FiniteTruth::Category { probabilities: reals(map, "probabilities")? },
        "vector" => FiniteTruth::Vector {
            dimension: usize::try_from(unsigned(map, "dimension")?)
                .map_err(|_| crate::value_err("`dimension` does not fit"))?,
            atoms: reals(map, "atoms")?,
            probabilities: reals(map, "probabilities")?,
        },
        other => return Err(crate::value_err(format!("unknown finite truth `{other}`"))),
    };
    Ok(match exact_expected_score(kind, &law, &truth) {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(contract_refusal(&error))),
    })
}

/// The separately labelled drift of the PIT between two windows.
#[pyfunction]
fn predictive_assessment_drift(
    reference: Vec<f64>,
    comparison: Vec<f64>,
    bins: usize,
) -> Refused<String> {
    match distribution_drift(&reference, &comparison, bins) {
        Ok(drift) => (Some(drift_json(&drift).to_string()), None),
        Err(error) => (None, Some(contract_refusal(&error))),
    }
}

/// The observation key of a stable key within a domain.
#[pyfunction]
fn predictive_assessment_observation_key(domain: &str, key: &[u8]) -> String {
    ObservationKey::from_key(domain, key).to_hex()
}

fn source_ids(sources_json: &str) -> PyResult<Vec<SourceId>> {
    let parsed = parse(sources_json, "sources")?;
    parsed
        .as_array()
        .ok_or_else(|| crate::value_err("sources must be a list"))?
        .iter()
        .map(|v| {
            v.as_str()
                .and_then(SourceId::from_hex)
                .ok_or_else(|| crate::value_err("a source identity is 64 hexadecimal characters"))
        })
        .collect()
}

/// What a law was fitted on, from the evidence ledger's resolved observations.
#[pyfunction]
fn predictive_assessment_ledger_fit(
    ledger: PyRef<'_, EvidenceLedgerHandle>,
    sources_json: &str,
) -> PyResult<Refused<String>> {
    let sources = source_ids(sources_json)?;
    Ok(match fit_declaration_from_ledger(ledger.ledger(), &sources) {
        Ok(FitDeclaration::FittedOn(found)) => (
            Some(
                json!({
                    "kind": "fitted_on",
                    "keys": found.iter().map(ObservationKey::to_hex).collect::<Vec<_>>(),
                    "reason": Json::Null,
                })
                .to_string(),
            ),
            None,
        ),
        Ok(FitDeclaration::Undeclared { reason }) => {
            (Some(json!({"kind": "undeclared", "keys": [], "reason": reason}).to_string()), None)
        }
        Ok(FitDeclaration::NotFitted) => (
            Some(json!({"kind": "not_fitted", "keys": [], "reason": Json::Null}).to_string()),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// The assessed observations of one ledger observation source.
#[pyfunction]
fn predictive_assessment_ledger_observations(
    ledger: PyRef<'_, EvidenceLedgerHandle>,
    source: &str,
) -> PyResult<Refused<Vec<String>>> {
    let id = SourceId::from_hex(source)
        .ok_or_else(|| crate::value_err("a source identity is 64 hexadecimal characters"))?;
    Ok(match observation_keys_from_ledger(ledger.ledger(), id) {
        Ok(found) => (Some(found.iter().map(ObservationKey::to_hex).collect()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PredictiveAssessmentHandle>()?;
    module.add_function(wrap_pyfunction!(predictive_assessment_assess, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_consume, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_score_case, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_exact_expected_score, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_drift, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_observation_key, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_ledger_fit, module)?)?;
    module.add_function(wrap_pyfunction!(predictive_assessment_ledger_observations, module)?)?;
    Ok(())
}
