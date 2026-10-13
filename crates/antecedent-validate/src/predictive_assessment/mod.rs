//! Assessment of predictive laws: proper scores, calibration diagnostics and
//! score differences, under a declared validation design.
//!
//! A model-independent contract. It is neither a model zoo nor a forecasting
//! application: it takes predictive laws (from native finite and closed-form
//! families, or supplied by an external provider and declared through the
//! provider contract) and realized observations, and reports how the laws
//! performed on those observations.
//!
//! # What the contract fixes
//!
//! * **The target.** Every law and the observations name the same quantity,
//!   horizon, population and units; a mismatch is refused, never converted.
//! * **The design.** Held-out split, rolling origin, blocked or clustered, or
//!   simulation stream, with the dependence of the cases declared. An iid
//!   assumption is refused for a temporal or clustered design, and with no
//!   declaration every standard error is withheld.
//! * **The overlap.** Observations that were also used to fit or tune a law are
//!   detected from row-level identities (the evidence ledger's overlap model)
//!   and the assessment is labelled `in_sample`, or refused when held-out
//!   observations are required. A provider law with no overlap declaration is
//!   `unverified`.
//! * **The scores.** Log score, CRPS (closed form for Gaussian, exact for finite
//!   laws, the energy form for samples), Brier score and the energy score of a
//!   joint law, all strictly proper losses.
//! * **The differences.** Paired on the shared observations, with a
//!   Diebold-Mariano style standard error under the declared dependence.
//! * **The diagnostics.** PIT histogram, reliability, interval coverage and
//!   sharpness, and a separate drift route; all labelled `diagnostic` with
//!   `calibration = "unmeasured"`.
//!
//! # What it does not claim
//!
//! * No interval for a score or a score difference: that needs a declared
//!   sampling design with measured coverage and none exists (`cell_not_licensed`).
//!   An exact finite-law expected score has no sampling interval to report.
//! * No calibration guarantee for the assessed law and no causal validity: a good
//!   score says the law predicted these observations well, not that any model
//!   behind it is causally valid or that a claim has a probability.
//! * No law callbacks: a law that exists only as a callback is refused by name
//!   (`predictive_assessment.law_callback_not_built`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod assess;
mod bayesian;
mod calibration;
mod compare;
mod design;
mod exact;
mod identity;
mod law;
mod overlap;
mod refusal;
mod score;

pub use assess::{
    ASSESSMENT_CAUSAL_VALIDITY, ASSESSMENT_FAMILY, ASSESSMENT_INFERENCE_CLAIM,
    AssessmentUncertainty, FoldScore, MAX_ASSESSMENT_WORK, PredictiveAssessmentReport,
    PredictiveAssessmentSpec, PredictorAssessment, REPLAY_TOLERANCE, ScoreIntervalRequest,
    ScoreSummary, assess_predictive_laws,
};
pub use bayesian::PredictiveReplicates;
pub use calibration::{
    CalibrationDiagnostics, CoverageDiagnostic, DIAGNOSTIC_CALIBRATION, DIAGNOSTIC_LABEL,
    DistributionDrift, MAX_BINS, MAX_INTERVAL_LEVELS, PSI_PSEUDOCOUNT, PitDiagnostic,
    ReliabilityBin, ReliabilityDiagnostic, distribution_drift, pit_available_for,
};
pub use compare::{ScoreDifference, variance_of_mean};
pub use design::{DependenceDeclaration, ValidationDesign, dependence_label};
pub use exact::{FiniteTruth, exact_expected_score};
pub use identity::{AssessmentIdentity, IDENTITY_VERSION};
pub use law::{
    ForecastLaw, Horizon, LawKind, LawSource, MAX_CASES, MAX_JOINT_DIMENSION, MAX_LAW_ATOMS,
    MAX_PREDICTORS, ObservationKey, Outcome, PROBABILITY_SUM_TOLERANCE, PredictiveTarget,
    PredictorLaws, ProviderIdentity, ProviderTrustLabel, RealizedObservations,
};
pub use overlap::{FitDeclaration, OverlapPolicy, OverlapStatus};
pub use refusal::{PredictiveAssessmentError, detail};
pub use score::{
    EnsembleCrpsForm, ScoreKind, brier_binary, brier_categorical, crps_ensemble_energy_sorted,
    crps_gaussian, crps_weighted_atoms, energy_score_weighted, log_score_gaussian, score_case,
};
