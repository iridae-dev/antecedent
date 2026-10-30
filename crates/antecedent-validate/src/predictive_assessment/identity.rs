//! Canonical identities of an assessment's inputs.
//!
//! Every digest is a typed, length-delimited canonical encoding (BLAKE3 under a
//! domain tag); no Debug, JSON or repr text is hashed. Floats enter as their
//! IEEE-754 bit pattern under the format version [`IDENTITY_VERSION`]. Cases are
//! encoded in the canonical case order, and a sample's draws sorted, so listing
//! the cases or the draws in another order leaves every digest unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

use super::design::{DependenceDeclaration, ResolvedDesign, ValidationDesign};
use super::law::{
    ForecastLaw, Horizon, LawSource, Outcome, PredictiveTarget, PredictorLaws, ProviderIdentity,
};
use super::overlap::{FitDeclaration, OverlapPolicy};
use super::score::{EnsembleCrpsForm, ScoreKind};

/// Base version word written into every assessment identity. Caller-minted
/// provenance is an appended domain-tagged extension; all-ledger identities
/// retain their original base encoding.
pub const IDENTITY_VERSION: u64 = 1;

/// The identities of one assessment: what a consumer holding only the bytes and
/// an independently retained identity compares.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssessmentIdentity {
    /// Digest of the observation target, identities and outcomes.
    pub observations: String,
    /// Digest of each predictor (name, source, target, fit declaration, laws), by name.
    pub predictors: Vec<(String, String)>,
    /// Digest of the validation design and the declared dependence.
    pub design: String,
    /// Digest of the scoring and diagnostic configuration.
    pub configuration: String,
    /// Digest over all of the above.
    pub digest: String,
}

impl AssessmentIdentity {
    /// The first field that differs from `other`, in a fixed order.
    #[must_use]
    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.observations != other.observations {
            Some("observations")
        } else if self.predictors != other.predictors {
            Some("predictor_laws")
        } else if self.design != other.design {
            Some("validation_design")
        } else if self.configuration != other.configuration {
            Some("configuration")
        } else if self.digest != other.digest {
            Some("identity")
        } else {
            None
        }
    }
}

fn target(writer: &mut CanonicalWriter, value: &PredictiveTarget) {
    writer.text(&value.quantity).text(&value.population).text(&value.units);
    match value.horizon {
        Horizon::Contemporaneous => writer.word(0),
        Horizon::Steps(steps) => writer.word(1).word(u64::from(steps)),
    };
}

fn provider(writer: &mut CanonicalWriter, identity: &ProviderIdentity) {
    writer
        .text(&identity.provider_id)
        .text(&identity.object_id)
        .text(&identity.version_id)
        .text(&identity.snapshot_id)
        .text(&identity.request_id);
}

fn law(writer: &mut CanonicalWriter, value: &ForecastLaw) {
    writer.text(value.kind().name());
    match value {
        ForecastLaw::Gaussian { mean, sd } => {
            writer.real(*mean).real(*sd);
        }
        ForecastLaw::Bernoulli { probability } => {
            writer.real(*probability);
        }
        ForecastLaw::Categorical { probabilities } => {
            writer.reals(probabilities);
        }
        ForecastLaw::FiniteReal { support, probabilities } => {
            writer.reals(support).reals(probabilities);
        }
        ForecastLaw::EmpiricalSample { draws } => {
            let mut sorted = draws.clone();
            sorted.sort_by(f64::total_cmp);
            writer.reals(&sorted);
        }
        ForecastLaw::PointEvaluated { log_density, cdf_below, cdf_at } => {
            writer.opt_real(*log_density).real(*cdf_below).real(*cdf_at);
        }
        ForecastLaw::JointFinite { dimension, atoms, probabilities } => {
            writer.word(*dimension as u64).reals(atoms).reals(probabilities);
        }
        ForecastLaw::JointSample { dimension, draws } => {
            // Rows are sorted lexicographically so the draw order is immaterial.
            let mut rows: Vec<&[f64]> = draws.chunks_exact(*dimension).collect();
            rows.sort_by(|a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|ordering| ordering.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            writer.word(*dimension as u64).word(rows.len() as u64);
            for row in rows {
                for value in row {
                    writer.real(*value);
                }
            }
        }
    }
}

/// Digest of the observations in canonical order.
pub(crate) fn observations_digest(
    observed: &PredictiveTarget,
    keys: &[super::law::ObservationKey],
    outcomes: &[Outcome],
    order: &[usize],
    bind_provenance: bool,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.predictive_assessment.observations");
    writer.word(IDENTITY_VERSION);
    target(&mut writer, observed);
    writer.word(order.len() as u64);
    for &i in order {
        writer.bytes(keys[i].as_bytes());
        match &outcomes[i] {
            Outcome::Real(y) => {
                writer.word(0).real(*y);
            }
            Outcome::Category(c) => {
                writer.word(1).word(u64::from(*c));
            }
            Outcome::Vector(values) => {
                writer.word(2).reals(values);
            }
        }
    }
    if bind_provenance && keys.iter().any(super::law::ObservationKey::is_caller_minted) {
        writer.text("caller_minted_provenance_v2").word(order.len() as u64);
        for &i in order { writer.word(keys[i].is_caller_minted() as u64); }
    }
    writer.finish_hex()
}

/// Digest of one predictor with its cases in canonical order.
pub(crate) fn predictor_digest(predictor: &PredictorLaws, order: &[usize], bind_provenance: bool) -> String {
    let mut writer = CanonicalWriter::new("antecedent.predictive_assessment.predictor");
    writer.word(IDENTITY_VERSION).text(&predictor.name).text(predictor.source.class());
    match &predictor.source {
        LawSource::Native { model } => {
            writer.text(model);
        }
        LawSource::Provider { identity, trust } => {
            provider(&mut writer, identity);
            writer.text(trust.name());
        }
        LawSource::Callback { description } => {
            writer.text(description);
        }
    }
    target(&mut writer, &predictor.target);
    match &predictor.fit {
        FitDeclaration::FittedOn(keys) => {
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            sorted.dedup();
            writer.word(1).word(sorted.len() as u64);
            for key in sorted {
                writer.bytes(key.as_bytes());
            }
        }
        FitDeclaration::NotFitted => {
            writer.word(2);
        }
        FitDeclaration::Undeclared { reason } => {
            writer.word(3).text(reason);
        }
    }
    writer.word(order.len() as u64);
    for &i in order {
        law(&mut writer, &predictor.laws[i]);
    }
    if bind_provenance {
        if let FitDeclaration::FittedOn(keys) = &predictor.fit {
            let sorted = super::overlap::sorted_distinct(keys);
            if sorted.iter().any(super::law::ObservationKey::is_caller_minted) {
                writer.text("caller_minted_fit_provenance_v2").word(sorted.len() as u64);
                for key in sorted { writer.word(key.is_caller_minted() as u64); }
            }
        }
    }
    writer.finish_hex()
}

/// Digest of the design and the declared dependence in canonical order.
pub(crate) fn design_digest(
    design: &ValidationDesign,
    dependence: Option<&DependenceDeclaration>,
    resolved: &ResolvedDesign,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.predictive_assessment.design");
    writer.word(IDENTITY_VERSION).text(design.name());
    match design {
        ValidationDesign::HeldOutSplit | ValidationDesign::Blocked { .. } => {}
        ValidationDesign::RollingOrigin { horizon_steps, .. } => {
            writer.word(u64::from(*horizon_steps));
        }
        ValidationDesign::SimulationStream { stream } => {
            writer.text(stream);
        }
    }
    for fold in &resolved.fold_of {
        writer.opt_word(*fold);
    }
    match dependence {
        None => {
            writer.text("undeclared");
        }
        Some(declared) => {
            writer.text(declared.name());
            match declared {
                DependenceDeclaration::Iid => {}
                DependenceDeclaration::Serial { truncation_lag, .. } => {
                    writer.word(u64::from(*truncation_lag));
                    for t in resolved.time.iter().flatten() {
                        writer.word(*t);
                    }
                }
                DependenceDeclaration::Cluster { .. } => {
                    for label in resolved.cluster.iter().flatten() {
                        writer.word(u64::from(*label));
                    }
                }
            }
        }
    }
    writer.finish_hex()
}

/// The scoring and diagnostic configuration of an assessment.
pub(crate) struct ConfigurationRef<'a> {
    pub scores: &'a [ScoreKind],
    pub ensemble_crps: EnsembleCrpsForm,
    pub pit_bins: usize,
    pub reliability_bins: usize,
    pub interval_levels: &'a [f64],
    pub comparisons: &'a [(String, String)],
    pub overlap_policy: OverlapPolicy,
}

/// Digest of the configuration.
pub(crate) fn configuration_digest(configuration: &ConfigurationRef<'_>) -> String {
    let mut writer = CanonicalWriter::new("antecedent.predictive_assessment.configuration");
    writer.word(IDENTITY_VERSION);
    writer.word(configuration.scores.len() as u64);
    for score in configuration.scores {
        writer.text(score.name());
    }
    writer.text(configuration.ensemble_crps.name());
    writer.word(configuration.pit_bins as u64);
    writer.word(configuration.reliability_bins as u64);
    writer.reals(configuration.interval_levels);
    writer.word(configuration.comparisons.len() as u64);
    for (candidate, reference) in configuration.comparisons {
        writer.text(candidate).text(reference);
    }
    writer.text(match configuration.overlap_policy {
        OverlapPolicy::LabelOnly => "label_only",
        OverlapPolicy::RequireHeldOut => "require_held_out",
    });
    writer.finish_hex()
}

/// The whole-assessment digest over the component digests.
pub(crate) fn whole_digest(
    observations: &str,
    predictors: &[(String, String)],
    design: &str,
    configuration: &str,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.predictive_assessment.identity");
    writer.word(IDENTITY_VERSION).text(observations);
    writer.word(predictors.len() as u64);
    for (name, digest) in predictors {
        writer.text(name).text(digest);
    }
    writer.text(design).text(configuration);
    writer.finish_hex()
}
