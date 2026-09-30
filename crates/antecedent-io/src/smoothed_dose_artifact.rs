//! Independent point-result artifacts for the smoothed dose-response transport grid.
//!
//! Format version 1 of the 2.2B cell X4 artifact (`checked_smoothed_dose_transport_v1`).
//! A consumer never fits a learner or resamples. It refuses stored bounds above its own
//! compiled bounds and a request above its own row and covariate limits before any
//! work; re-derives the transport certificate from the stored graph and smoothed-dose
//! query and requires the stored certificate record to match; re-validates the request
//! (bounds, known density, dose support, local support) exactly as the producer did;
//! recomputes the shared fold assignment from the stored rows; checks every stored fold
//! model's shape and provenance against the learner specs; and then **re-predicts every
//! nuisance from the stored portable fold models, re-integrates every quadrature and
//! replays the point, the quadrature errors, the smoothing-bias diagnostic, the local
//! support, the held-out diagnostics, the overlap and the post-fit refusals bit for bit**.
//!
//! # What replay does and does not establish
//!
//! Replay establishes that the stored point is exactly what the stored fold models,
//! rows and premises produce, including the quadrature. It uses the same evaluator as
//! the producer, so it is an integrity replay, not an independent check of the
//! evaluator; the evaluator's correctness is established by the known-truth,
//! exact-integration and misspecification fixtures. It does **not** replay the fits: a
//! stored model is producer evidence that a model of that spec was fitted on the other
//! folds, which cannot be verified without refitting. A forger who replaces a fold
//! model, recomputes everything it implies and re-digests produces an artifact that
//! consumes. The digests are integrity checks against accidental or partial edits, not
//! authentication. Provider versions are not compared with the consuming build's.
//!
//! The verified identity is three digests: premises (graph, selections, smoothed-dose
//! query, certificate, options, seed, fold scheme, frozen bounds, feature schema,
//! sampling design, variable names), data (rows, doses, known densities) and evidence
//! (the whole executed result, fold models included). The interval route is closed
//! (`cell_not_licensed`), so the format has no interval field; the stored status is
//! re-derived.
use crate::{IoError, admg_from_wire, admg_to_wire, wire::AdmgWire};
use antecedent_core::{ExecutionContext, IdentityDomain, SmoothedDoseTransportQuery, VariableId};
use antecedent_estimate::smoothed_dose::{
    SMOOTHED_DOSE_BOUNDS, SMOOTHED_DOSE_FOLD_SCHEME, SmoothedDoseBounds,
    evaluate_smoothed_dose_models, parse_smoothing_kernel, smoothed_dose_fold_assignment,
};
use antecedent_estimate::{
    EstimationError, SmoothedDoseEstimate, SmoothedDoseInput, SmoothedDoseOptions,
    SmoothedDoseUncertainty, validate_smoothed_dose,
};
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The artifact format this reader writes and accepts.
pub const SMOOTHED_DOSE_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const SMOOTHED_DOSE_ARTIFACT_FEATURE: &str = "checked_smoothed_dose_transport_v1";

use crate::learned_continuous_artifact::CertificateRecord;

/// Why a smoothed-dose artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SmoothedDoseArtifactError {
    /// The feature marker or shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored size or bound exceeds the consumer's own limits or compiled bounds.
    #[error("limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored request no longer validates, or replay refused it; carries the
    /// registered reason code and the namespaced detail of the refusal.
    #[error("replay refused ({code}): {message}")]
    ReplayRefused {
        /// Registered runtime reason code.
        code: &'static str,
        /// The refusal message, starting with its namespaced detail.
        message: String,
    },
    /// The request is malformed (not a coded refusal).
    #[error("request does not validate: {0}")]
    InvalidRequest(String),
    /// The certificate re-derived from the stored graph and query is not the stored one.
    #[error("certificate does not verify: {0}")]
    ProofMismatch(String),
    /// The stored fold assignment or scheme is not the recomputed one.
    #[error("fold provenance does not match the stored rows")]
    FoldMismatch,
    /// The stored fold models or provenance do not match the specs, folds or design.
    #[error("fold models or provenance contradict the learner specs")]
    ProvenanceMismatch,
    /// A replayed grid estimate differs from the stored one.
    #[error("point estimate does not replay")]
    PointMismatch,
    /// A replayed quadrature record differs from the stored one.
    #[error("quadrature record does not replay")]
    QuadratureMismatch,
    /// A replayed smoothing-bias diagnostic differs from the stored one.
    #[error("smoothing-bias diagnostic does not replay")]
    SmoothingBiasMismatch,
    /// A replayed local-support record or influence diagnostic differs.
    #[error("local support does not replay")]
    SupportMismatch,
    /// The replayed held-out diagnostics differ from the stored ones.
    #[error("nuisance diagnostics do not replay")]
    DiagnosticsMismatch,
    /// The replayed membership overlap differs from the stored one.
    #[error("overlap report does not replay")]
    OverlapMismatch,
    /// The stored interval status contradicts the closed interval route.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored rows.
    #[error("data digest mismatch")]
    DataIdentityMismatch,
    /// The evidence digest does not match the stored result.
    #[error("evidence digest mismatch")]
    EvidenceMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

impl SmoothedDoseArtifactError {
    /// The registered reason code a replay refusal carries.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::ReplayRefused { code, .. } => Some(code),
            _ => None,
        }
    }
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct SmoothedDoseConsumeLimits {
    /// Most rows (trial plus target) an artifact may carry.
    pub max_rows: usize,
    /// Most covariates an artifact may carry.
    pub max_features: usize,
}

impl Default for SmoothedDoseConsumeLimits {
    fn default() -> Self {
        Self {
            max_rows: SMOOTHED_DOSE_BOUNDS.max_rows,
            max_features: SMOOTHED_DOSE_BOUNDS.max_features,
        }
    }
}

/// The smoothed-dose query in stable wire form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseQueryWire {
    /// Outcome coordinate.
    pub outcome: u32,
    /// Dose coordinate.
    pub dose: u32,
    /// Source population key.
    pub source_population: String,
    /// Target population key.
    pub target_population: String,
    /// Grid doses in declared order.
    pub grid: Vec<f64>,
    /// Bandwidth.
    pub bandwidth: f64,
    /// Kernel name.
    pub kernel: String,
    /// Declared dose support `(lo, hi)`.
    pub dose_support: (f64, f64),
    /// Density provenance.
    pub density_provenance: String,
}

impl SmoothedDoseQueryWire {
    /// The wire form of a query.
    #[must_use]
    pub fn from_query(query: &SmoothedDoseTransportQuery) -> Self {
        Self {
            outcome: query.outcome.raw(),
            dose: query.dose.raw(),
            source_population: query.source_population.to_string(),
            target_population: query.target_population.to_string(),
            grid: query.grid.to_vec(),
            bandwidth: query.bandwidth,
            kernel: query.kernel.name().into(),
            dose_support: query.dose_support,
            density_provenance: query.density_provenance.to_string(),
        }
    }

    /// The query this wire encodes.
    ///
    /// # Errors
    /// An unsupported kernel (`dose_response.kernel_not_supported`).
    pub fn to_query(&self) -> Result<SmoothedDoseTransportQuery, EstimationError> {
        Ok(SmoothedDoseTransportQuery {
            outcome: VariableId::from_raw(self.outcome),
            dose: VariableId::from_raw(self.dose),
            source_population: Arc::from(self.source_population.as_str()),
            target_population: Arc::from(self.target_population.as_str()),
            grid: Arc::from(self.grid.as_slice()),
            bandwidth: self.bandwidth,
            kernel: parse_smoothing_kernel(&self.kernel)?,
            dose_support: self.dose_support,
            density_provenance: Arc::from(self.density_provenance.as_str()),
        })
    }
}

/// Everything a public producer stores: premises, rows, point, models and bookkeeping.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDoseArtifactWire {
    /// [`SMOOTHED_DOSE_ARTIFACT_VERSION`].
    pub version: u32,
    /// Required semantics; unknown entries prevent scientific acceptance.
    pub required_features: Vec<String>,
    /// Original static graph coordinates.
    pub graph: AdmgWire,
    /// Variables whose mechanisms differ.
    pub selections: Vec<u32>,
    /// The smoothed-dose query (grid, bandwidth, kernel, support, provenance).
    pub query: SmoothedDoseQueryWire,
    /// Name of every graph coordinate, bound into the verified identity.
    pub variable_names: Vec<String>,
    /// The certificate this execution ran under.
    pub certificate: CertificateRecord,
    /// Rows and the sampling design.
    pub input: SmoothedDoseInput,
    /// Learners, basis, folds, quadrature, thresholds and the bootstrap request.
    pub options: SmoothedDoseOptions,
    /// The frozen bounds the producer ran under.
    pub bounds: SmoothedDoseBounds,
    /// Master seed of the execution.
    pub seed: u64,
    /// Grid, fold models, provenance, diagnostics and interval status.
    pub result: SmoothedDoseEstimate,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the rows.
    pub data_digest: String,
    /// Digest of the executed result.
    pub evidence_digest: String,
}

/// Everything a public producer hands to [`SmoothedDoseArtifactWire::checked`].
pub struct SmoothedDoseArtifactInput<'a> {
    /// The static graph.
    pub graph: &'a antecedent_graph::Admg,
    /// Selection targets of the diagram.
    pub selections: Vec<u32>,
    /// The smoothed-dose query.
    pub query: &'a SmoothedDoseTransportQuery,
    /// Name of every graph coordinate; empty when the caller binds none.
    pub variable_names: &'a [String],
    /// The identification the estimate ran under.
    pub identification: &'a TransportIdentification,
    /// Rows and sampling design.
    pub input: &'a SmoothedDoseInput,
    /// Request options.
    pub options: &'a SmoothedDoseOptions,
    /// Master seed.
    pub seed: u64,
    /// The executed estimate.
    pub result: &'a SmoothedDoseEstimate,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    selections: &'a [u32],
    query: &'a SmoothedDoseQueryWire,
    certificate: &'a CertificateRecord,
    options: &'a SmoothedDoseOptions,
    bounds: &'a SmoothedDoseBounds,
    seed: u64,
    fold_scheme: &'a str,
    features: &'a [u32],
    sampling: antecedent_estimate::TrialSampling,
    variable_names: &'a [String],
}

#[derive(Serialize)]
struct DataView<'a> {
    tag: &'static str,
    covariates: &'a [Vec<f64>],
    outcome: &'a [f64],
    dose: &'a [f64],
    dose_density: &'a [f64],
    source: &'a [bool],
}

#[derive(Serialize)]
struct EvidenceView<'a> {
    tag: &'static str,
    result: &'a SmoothedDoseEstimate,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// A verified replay: the wire plus the independently rebuilt identification.
pub struct ConsumedSmoothedDose {
    /// The decoded, verified artifact.
    pub wire: SmoothedDoseArtifactWire,
    /// The identification re-derived from the stored graph and query.
    pub identification: TransportIdentification,
}

type Refusal = SmoothedDoseArtifactError;

fn replay_refusal(error: EstimationError) -> Refusal {
    match error {
        EstimationError::Refused { code, message } => Refusal::ReplayRefused { code, message },
        other => Refusal::InvalidRequest(other.to_string()),
    }
}

impl SmoothedDoseArtifactWire {
    /// Build an artifact from checked premises and an executed estimate, then verify it.
    ///
    /// # Errors
    /// The premises do not encode, or the estimate does not verify against them.
    pub fn checked(input: SmoothedDoseArtifactInput<'_>) -> Result<Self, IoError> {
        let certificate = CertificateRecord::from_identification(input.identification)
            .map_err(|message| IoError::from(Refusal::ProofMismatch(message)))?;
        let mut wire = Self {
            version: SMOOTHED_DOSE_ARTIFACT_VERSION,
            required_features: vec![SMOOTHED_DOSE_ARTIFACT_FEATURE.into()],
            graph: admg_to_wire(input.graph)?,
            selections: input.selections,
            query: SmoothedDoseQueryWire::from_query(input.query),
            variable_names: input.variable_names.to_vec(),
            certificate,
            input: input.input.clone(),
            options: input.options.clone(),
            bounds: SMOOTHED_DOSE_BOUNDS,
            seed: input.seed,
            result: input.result.clone(),
            premises_digest: String::new(),
            data_digest: String::new(),
            evidence_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        wire.evidence_digest = wire.expected_evidence_digest()?;
        wire.verify(SmoothedDoseConsumeLimits::default())?;
        Ok(wire)
    }

    /// The premises digest this artifact's stored premises hash to.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let mut graph = self.graph.clone();
        graph.directed.sort_unstable();
        graph.bidirected.sort_unstable();
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &PremisesView {
                tag: "smoothed_dose_transport_premises_v1",
                graph,
                selections: &self.selections,
                query: &self.query,
                certificate: &self.certificate,
                options: &self.options,
                bounds: &self.bounds,
                seed: self.seed,
                fold_scheme: SMOOTHED_DOSE_FOLD_SCHEME,
                features: &self.input.features,
                sampling: self.input.sampling,
                variable_names: &self.variable_names,
            },
        )?
        .to_hex())
    }

    /// The data digest the stored rows hash to.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &DataView {
                tag: "smoothed_dose_transport_data_v1",
                covariates: &self.input.covariates,
                outcome: &self.input.outcome,
                dose: &self.input.dose,
                dose_density: &self.input.dose_density,
                source: &self.input.source,
            },
        )?
        .to_hex())
    }

    /// The evidence digest the stored result hashes to.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn expected_evidence_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &EvidenceView { tag: "smoothed_dose_transport_evidence_v1", result: &self.result },
        )?
        .to_hex())
    }

    /// Scientific execution identity.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn identity(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(IdentityDomain::LearnedTrial, self)?.to_hex())
    }

    /// [`Refusal::NamesMismatch`] unless `names` equals the stored variable names.
    ///
    /// # Errors
    /// The caller's names are not the names the artifact was built under.
    pub fn check_variable_names(&self, names: &[String]) -> Result<(), Refusal> {
        if self.variable_names == names { Ok(()) } else { Err(Refusal::NamesMismatch) }
    }

    /// Encode without verifying; a consumer verifies on decode.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding failure (an artifact carrying an
    /// interval field fails here).
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != SMOOTHED_DOSE_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        crate::from_cbor(bytes)
    }

    /// Decode and independently verify without fitting or resampling.
    ///
    /// # Errors
    /// Any decoding or verification failure.
    pub fn consume(
        bytes: &[u8],
        limits: SmoothedDoseConsumeLimits,
    ) -> Result<ConsumedSmoothedDose, IoError> {
        let wire = Self::decode(bytes)?;
        let identification = wire.verify(limits)?;
        Ok(ConsumedSmoothedDose { wire, identification })
    }

    /// Everything that is checked before any replay work: the format, the consumer's
    /// own limits, the stored bounds against the compiled ones and the stored shapes.
    fn check_shape(&self, limits: &SmoothedDoseConsumeLimits) -> Result<(), Refusal> {
        if self.required_features != [SMOOTHED_DOSE_ARTIFACT_FEATURE] {
            return Err(Refusal::UnsupportedSemantics("required feature set"));
        }
        let n = self.input.source.len();
        if n > limits.max_rows {
            return Err(Refusal::LimitsExceeded("row count"));
        }
        if self.input.features.len() > limits.max_features {
            return Err(Refusal::LimitsExceeded("feature count"));
        }
        if self.bounds != SMOOTHED_DOSE_BOUNDS {
            return Err(Refusal::LimitsExceeded(
                "stored bounds are not this build's frozen bounds",
            ));
        }
        let o = &self.options;
        if self.query.grid.len() > self.bounds.max_grid
            || o.folds > self.bounds.max_folds
            || o.bootstrap > self.bounds.max_bootstrap
            || !self.bounds.quadrature_nodes.contains(&o.quadrature_nodes)
        {
            return Err(Refusal::LimitsExceeded("request above the stored frozen bounds"));
        }
        let r = &self.result;
        if r.folds.assignment.len() != n
            || r.grid.len() != self.query.grid.len()
            || r.models.outcome.len() != o.folds
            || r.models.membership.len() != o.folds
            || r.provenance.len() != 2 * o.folds
        {
            return Err(Refusal::UnsupportedSemantics("result shape"));
        }
        Ok(())
    }

    /// Every fold model names its stored spec, matches its role's design width and is
    /// consistent within its role; the recorded provenance is the models' own.
    fn check_provenance(&self) -> Result<(), Refusal> {
        let o = &self.options;
        let r = &self.result;
        let widths = [self.input.features.len() + 1, o.basis.width(self.input.features.len())];
        let roles = [
            (&r.models.membership, o.membership, widths[0]),
            (&r.models.outcome, o.outcome, widths[1]),
        ];
        let recorded: Vec<_> = r
            .models
            .membership
            .iter()
            .chain(&r.models.outcome)
            .map(|m| m.provenance.clone())
            .collect();
        if recorded != r.provenance {
            return Err(Refusal::ProvenanceMismatch);
        }
        for (models, spec, width) in roles {
            let first = &models[0].provenance;
            for model in models {
                let p = &model.provenance;
                if model.columns != width
                    || model.validate().is_err()
                    || p.implementation.is_empty()
                    || p.version.is_empty()
                    || p != first
                {
                    return Err(Refusal::ProvenanceMismatch);
                }
            }
            // `auto` resolves to a data-dependent learner; every other spec names itself.
            if !matches!(spec, antecedent_estimate::LearnerSpec::Auto) && first.spec != spec.name()
            {
                return Err(Refusal::ProvenanceMismatch);
            }
        }
        Ok(())
    }

    /// Verify every stored claim against a replay from the stored models.
    ///
    /// # Errors
    /// The first failing check, as a typed [`SmoothedDoseArtifactError`].
    pub fn verify(
        &self,
        limits: SmoothedDoseConsumeLimits,
    ) -> Result<TransportIdentification, IoError> {
        if self.version != SMOOTHED_DOSE_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: self.version });
        }
        self.check_shape(&limits)?;
        if self.premises_digest != self.expected_premises_digest()? {
            return Err(Refusal::PremisesMismatch.into());
        }
        if self.data_digest != self.expected_data_digest()? {
            return Err(Refusal::DataIdentityMismatch.into());
        }
        if self.evidence_digest != self.expected_evidence_digest()? {
            return Err(Refusal::EvidenceMismatch.into());
        }
        let graph = admg_from_wire(&self.graph)?;
        let diagram = antecedent_graph::SelectionDiagram::try_new(
            graph,
            self.selections.iter().map(|v| VariableId::from_raw(*v)).collect::<Vec<_>>(),
        )
        .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        let query = self.query.to_query().map_err(replay_refusal)?;
        query.validate().map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        let id = TransportIdentifier::new()
            .identify(&diagram, &query.transport_query())
            .map_err(|e| Refusal::ProofMismatch(e.to_string()))?;
        match CertificateRecord::from_identification(&id) {
            Ok(record) if record == self.certificate => {}
            Ok(_) => return Err(Refusal::ProofMismatch("stored certificate differs".into()).into()),
            Err(message) => return Err(Refusal::ProofMismatch(message).into()),
        }
        validate_smoothed_dose(&id, &query, &self.input, &self.options).map_err(replay_refusal)?;
        let r = &self.result;
        if r.folds.count != self.options.folds
            || r.folds.scheme != SMOOTHED_DOSE_FOLD_SCHEME
            || r.folds.assignment != smoothed_dose_fold_assignment(&self.input, self.options.folds)
        {
            return Err(Refusal::FoldMismatch.into());
        }
        self.check_provenance()?;
        let ctx = ExecutionContext::production_default(self.seed);
        let replay = evaluate_smoothed_dose_models(
            &query,
            &self.input,
            &self.options,
            &r.folds.assignment,
            &r.models,
            &ctx,
        )
        .map_err(replay_refusal)?;
        for (stored, replayed) in r.grid.iter().zip(&replay.grid) {
            if stored.dose.to_bits() != replayed.dose.to_bits()
                || stored.estimate.to_bits() != replayed.estimate.to_bits()
                || stored.plug_in.to_bits() != replayed.plug_in.to_bits()
                || stored.augmentation.to_bits() != replayed.augmentation.to_bits()
                || !stored.estimate.is_finite()
            {
                return Err(Refusal::PointMismatch.into());
            }
            if stored.quadrature != replayed.quadrature {
                return Err(Refusal::QuadratureMismatch.into());
            }
            if stored.smoothing_bias != replayed.smoothing_bias {
                return Err(Refusal::SmoothingBiasMismatch.into());
            }
            if stored.support != replayed.support
                || stored.influence_se_diagnostic.to_bits()
                    != replayed.influence_se_diagnostic.to_bits()
            {
                return Err(Refusal::SupportMismatch.into());
            }
        }
        if replay.diagnostics != r.diagnostics {
            return Err(Refusal::DiagnosticsMismatch.into());
        }
        if replay.overlap != r.overlap {
            return Err(Refusal::OverlapMismatch.into());
        }
        if r.uncertainty != SmoothedDoseUncertainty::for_request(self.options.bootstrap) {
            return Err(
                Refusal::UncertaintyMismatch("status is not the closed-route status").into()
            );
        }
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frozen_bounds_are_the_recorded_ones() {
        let b = SMOOTHED_DOSE_BOUNDS;
        assert_eq!(
            (b.max_grid, b.quadrature_nodes, b.max_folds, b.min_bootstrap, b.max_bootstrap),
            (16, [16, 32], 20, 199, 2000)
        );
        assert_eq!(
            (b.max_rows, b.max_features, b.max_basis_degree, b.max_knots),
            (200_000, 256, 3, 8)
        );
    }
}
