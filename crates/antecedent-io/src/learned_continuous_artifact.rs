//! Independent point and analytic-interval artifacts for learned continuous-outcome trial transport.
//!
//! Format versions 1 (point) and 2 (analytic interval) of the 2.2A cell X4 artifact
//! (separate from the `learned_trial` v1 wire, which stays consumable). A consumer
//! never fits a learner or resamples. It
//! re-derives the transport certificate from the stored graph and query and requires the
//! stored certificate record to match; re-validates the request against the frozen
//! bounds; recomputes the shared fold assignment from the stored rows; replays the
//! augmented inverse-odds **score** bit for bit from the stored out-of-fold nuisance
//! predictions; recomputes the held-out diagnostics, the separate membership and
//! treatment overlap, and re-applies the declared overlap thresholds; checks the recorded
//! provider provenance against the stored learner specs; and re-derives the interval
//! status.
//!
//! # What "independent consumption" does and does not establish
//!
//! It replays the certificate, the folds and the score. It does **not** replay the
//! nuisance fits: the stored out-of-fold predictions (`membership`, `mu0`, `mu1`) and the
//! provider versions are producer-recorded evidence that cannot be verified without
//! refitting the learners on the stored rows. A forger who edits the predictions *and*
//! recomputes the point, diagnostics and overlap consistently, then re-digests, produces
//! an artifact that consumes; only a consumer that refits detects that. The digests below
//! are integrity checks against accidental or partial edits, not authentication. The
//! recorded provider versions are not compared with the consuming build's (the consumer
//! never fits, so a different build is not a disagreement); only the spec names, the
//! per-role consistency and the digest binding are checked.
//!
//! The verified identity is three digests: the premises digest (graph, selections, query,
//! certificate, learner specs, fold count and scheme, seed, overlap thresholds, requested
//! bootstrap, frozen bounds, feature schema, sampling design and the variable-name
//! mapping), the data digest (the rows, which a refresh may replace) and the evidence
//! digest (the whole executed result: point, interval status, out-of-fold nuisance
//! predictions, per-nuisance provider provenance, the fold assignment, diagnostics and
//! overlap). The evidence digest is kept apart from the premises digest because the fold
//! assignment is a function of the rows, and the premises must stay row-independent.
//!
//! Version 2 recomputes the design-specific influence standard error and Wald interval
//! from the stored rows, point and out-of-fold predictions. Version 1 cannot carry an
//! interval; an added field fails decoding or verification.
use crate::{
    IoError, admg_from_wire, admg_to_wire, query_wire::TransportQueryWire, wire::AdmgWire,
};
use antecedent_core::IdentityDomain;
use antecedent_estimate::{
    LEARNED_CONTINUOUS_BOUNDS, LearnedContinuousEstimate, LearnedContinuousOptions,
    LearnedContinuousUncertainty, TrialAipwInput, check_membership_overlap,
    learned_trial::{trial_fold_assignment, trial_nuisance_diagnostics},
    validate_learned_continuous,
};
use antecedent_identify::{TransportFormula, TransportIdentification, TransportIdentifier};
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const LEARNED_CONTINUOUS_ARTIFACT_VERSION: u32 = 1;
/// Interval-bearing format; version 1 point artifacts remain consumable.
pub const LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION: u32 = 2;
/// The feature marker of the accepted format.
pub const LEARNED_CONTINUOUS_ARTIFACT_FEATURE: &str = "checked_learned_continuous_point_v1";
/// Required semantics of an interval-bearing artifact.
pub const LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_FEATURE: &str =
    "checked_learned_continuous_analytic_interval_v2";

/// Why a learned-continuous artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LearnedContinuousArtifactError {
    /// The feature marker or shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored size or bound exceeds the consumer's or the frozen cell's bound.
    #[error("limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The request no longer validates against its stored graph, query and rows.
    #[error("request does not validate: {0}")]
    InvalidRequest(String),
    /// The certificate re-derived from the stored graph and query is not the stored one.
    #[error("certificate does not verify: {0}")]
    ProofMismatch(String),
    /// The stored fold assignment or scheme is not the recomputed one.
    #[error("fold provenance does not match the stored rows")]
    FoldMismatch,
    /// The stored provider provenance does not cover every fitted nuisance, names a
    /// learner other than the stored spec, or is inconsistent within a nuisance role.
    #[error("provider provenance is incomplete or contradicts the learner specs")]
    ProvenanceMismatch,
    /// The recomputed point differs from the stored point.
    #[error("point estimate does not replay")]
    PointMismatch,
    /// The analytic standard error or interval does not replay.
    #[error("analytic interval does not replay")]
    IntervalMismatch,
    /// The recomputed held-out diagnostics differ from the stored ones.
    #[error("nuisance diagnostics do not replay")]
    DiagnosticsMismatch,
    /// The recomputed membership or treatment overlap differs from the stored one.
    #[error("overlap report does not replay")]
    OverlapMismatch,
    /// The stored membership predictions leave the declared overlap threshold.
    #[error("membership overlap is below the declared threshold")]
    OverlapRefused,
    /// The stored interval status contradicts its format and request.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored rows.
    #[error("data digest mismatch")]
    DataIdentityMismatch,
    /// The evidence digest does not match the stored result (point, nuisance predictions,
    /// provenance, folds, diagnostics, overlap, interval status).
    #[error("evidence digest mismatch")]
    EvidenceMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct LearnedContinuousConsumeLimits {
    /// Most rows (trial plus target) an artifact may carry.
    pub max_rows: usize,
    /// Most covariates an artifact may carry.
    pub max_features: usize,
}

impl Default for LearnedContinuousConsumeLimits {
    fn default() -> Self {
        Self { max_rows: 1_000_000, max_features: 256 }
    }
}

/// The certificate the identification produced, in stable form.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificateRecord {
    /// Stable rule id.
    pub rule: String,
    /// `direct` or `standardize`.
    pub formula: String,
    /// Standardizers, in the formula's order (empty for `direct`).
    pub over: Vec<u32>,
    /// Selection variables the rule used.
    pub selection_targets: Vec<u32>,
    /// Premises the implementation checked.
    pub premises: Vec<String>,
}

impl CertificateRecord {
    /// The record of a direct or standardization certificate.
    ///
    /// # Errors
    /// Any other identification outcome.
    pub fn from_identification(id: &TransportIdentification) -> Result<Self, String> {
        let TransportIdentification::Transportable { formula, certificate } = id else {
            return Err("the graph and query are not transportable".into());
        };
        let (kind, over) = match formula {
            TransportFormula::Direct(_) => ("direct", Vec::new()),
            TransportFormula::Standardize { over, .. } => {
                ("standardize", over.iter().map(|v| v.raw()).collect())
            }
            TransportFormula::RecursiveFactorization { .. } => {
                return Err("recursive factorization is identify-only".into());
            }
        };
        Ok(Self {
            rule: certificate.rule.to_string(),
            formula: kind.into(),
            over,
            selection_targets: certificate.selection_targets.iter().map(|v| v.raw()).collect(),
            premises: certificate.premises.iter().map(ToString::to_string).collect(),
        })
    }
}

/// Everything a public producer stores: premises, rows, point and bookkeeping.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedContinuousArtifactWire {
    /// [`LEARNED_CONTINUOUS_ARTIFACT_VERSION`].
    pub version: u32,
    /// Required semantics; unknown entries prevent scientific acceptance.
    pub required_features: Vec<String>,
    /// Original static graph coordinates.
    pub graph: AdmgWire,
    /// Variables whose mechanisms differ.
    pub selections: Vec<u32>,
    /// Target, populations and binary contrast.
    pub query: TransportQueryWire,
    /// Name of every graph coordinate, bound into the verified identity.
    pub variable_names: Vec<String>,
    /// The certificate this execution ran under.
    pub certificate: CertificateRecord,
    /// Rows and the sampling design.
    pub input: TrialAipwInput,
    /// Learner specs, folds, overlap thresholds and the bootstrap request.
    pub options: LearnedContinuousOptions,
    /// Master seed of the execution.
    pub seed: u64,
    /// Point, provenance, diagnostics and interval status.
    pub result: LearnedContinuousEstimate,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the rows.
    pub data_digest: String,
    /// Digest of the executed result, including provenance and the fold assignment.
    pub evidence_digest: String,
}

/// Everything a public producer hands to [`LearnedContinuousArtifactWire::checked`].
pub struct LearnedContinuousArtifactInput<'a> {
    /// The static graph.
    pub graph: &'a antecedent_graph::Admg,
    /// Selection targets of the diagram.
    pub selections: Vec<u32>,
    /// The binary contrast query.
    pub query: &'a antecedent_core::TransportQuery,
    /// Name of every graph coordinate; empty when the caller binds none.
    pub variable_names: &'a [String],
    /// The identification the estimate ran under.
    pub identification: &'a TransportIdentification,
    /// Rows and sampling design.
    pub input: &'a TrialAipwInput,
    /// Request options.
    pub options: &'a LearnedContinuousOptions,
    /// Master seed.
    pub seed: u64,
    /// The executed estimate.
    pub result: &'a LearnedContinuousEstimate,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    selections: &'a [u32],
    query: &'a TransportQueryWire,
    certificate: &'a CertificateRecord,
    options: &'a LearnedContinuousOptions,
    seed: u64,
    fold_scheme: &'a str,
    bounds: (usize, u32, u32),
    features: &'a [u32],
    sampling: antecedent_estimate::TrialSampling,
    variable_names: &'a [String],
}

#[derive(Serialize)]
struct DataView<'a> {
    tag: &'static str,
    covariates: &'a [Vec<f64>],
    outcome: &'a [f64],
    treatment: &'a [bool],
    source: &'a [bool],
    randomization: &'a [f64],
}

#[derive(Serialize)]
struct EvidenceView<'a> {
    tag: &'static str,
    result: &'a LearnedContinuousEstimate,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// A verified replay: the wire plus the independently rebuilt identification.
pub struct ConsumedLearnedContinuous {
    /// The decoded, verified artifact.
    pub wire: LearnedContinuousArtifactWire,
    /// The identification re-derived from the stored graph and query.
    pub identification: TransportIdentification,
}

type Refusal = LearnedContinuousArtifactError;

impl LearnedContinuousArtifactWire {
    /// Build an artifact from checked premises and an executed estimate, then verify it.
    ///
    /// # Errors
    /// The premises do not encode, or the estimate does not verify against them.
    pub fn checked(input: LearnedContinuousArtifactInput<'_>) -> Result<Self, IoError> {
        let certificate = CertificateRecord::from_identification(input.identification)
            .map_err(|message| IoError::from(Refusal::ProofMismatch(message)))?;
        let mut wire = Self {
            version: if input.result.interval.is_some() {
                LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION
            } else {
                LEARNED_CONTINUOUS_ARTIFACT_VERSION
            },
            required_features: vec![if input.result.interval.is_some() {
                LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_FEATURE.into()
            } else {
                LEARNED_CONTINUOUS_ARTIFACT_FEATURE.into()
            }],
            graph: admg_to_wire(input.graph)?,
            selections: input.selections,
            query: crate::query_wire::transport_query_to_wire(input.query)?,
            variable_names: input.variable_names.to_vec(),
            certificate,
            input: input.input.clone(),
            options: *input.options,
            seed: input.seed,
            result: input.result.clone(),
            premises_digest: String::new(),
            data_digest: String::new(),
            evidence_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        wire.evidence_digest = wire.expected_evidence_digest()?;
        wire.verify(LearnedContinuousConsumeLimits::default())?;
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
                tag: if self.version == LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION {
                    "learned_continuous_analytic_interval_v2"
                } else {
                    "learned_continuous_point_v1"
                },
                graph,
                selections: &self.selections,
                query: &self.query,
                certificate: &self.certificate,
                options: &self.options,
                seed: self.seed,
                fold_scheme:
                    antecedent_estimate::learned_continuous::LEARNED_CONTINUOUS_FOLD_SCHEME,
                bounds: LEARNED_CONTINUOUS_BOUNDS,
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
                tag: "learned_continuous_data_v1",
                covariates: &self.input.covariates,
                outcome: &self.input.outcome,
                treatment: &self.input.treatment,
                source: &self.input.source,
                randomization: &self.input.randomization,
            },
        )?
        .to_hex())
    }

    /// The evidence digest the stored result hashes to: point, interval status,
    /// out-of-fold nuisance predictions, per-nuisance provenance, fold assignment,
    /// diagnostics and overlap.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn expected_evidence_digest(&self) -> Result<String, IoError> {
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &EvidenceView {
                tag: if self.version == LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION {
                    "learned_continuous_evidence_v2"
                } else {
                    "learned_continuous_evidence_v1"
                },
                result: &self.result,
            },
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

    /// [`Refusal::NamesMismatch`] unless the stored source and target
    /// population keys are `source` and `target`.
    ///
    /// The rows carry no population name, so a relabelled population is a
    /// premise only: the premises digest binds it against unsealed edits, but
    /// an artifact re-sealed under another label replays. A consumer that knows
    /// which populations it expects checks them here, exactly as it checks the
    /// variable names with [`Self::check_variable_names`].
    ///
    /// # Errors
    /// The stored populations are not the caller's.
    pub fn check_population_names(&self, source: &str, target: &str) -> Result<(), Refusal> {
        if self.query.source_population == source && self.query.target_population == target {
            Ok(())
        } else {
            Err(Refusal::NamesMismatch)
        }
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
        if peek.version != LEARNED_CONTINUOUS_ARTIFACT_VERSION
            && peek.version != LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION
        {
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
        limits: LearnedContinuousConsumeLimits,
    ) -> Result<ConsumedLearnedContinuous, IoError> {
        let wire = Self::decode(bytes)?;
        let identification = wire.verify(limits)?;
        Ok(ConsumedLearnedContinuous { wire, identification })
    }

    fn check_shape(&self, limits: &LearnedContinuousConsumeLimits) -> Result<(), Refusal> {
        let feature = if self.version == LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION {
            LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_FEATURE
        } else {
            LEARNED_CONTINUOUS_ARTIFACT_FEATURE
        };
        if self.required_features != [feature] {
            return Err(Refusal::UnsupportedSemantics("required feature set"));
        }
        let n = self.input.source.len();
        if n > limits.max_rows {
            return Err(Refusal::LimitsExceeded("row count"));
        }
        if self.input.features.len() > limits.max_features {
            return Err(Refusal::LimitsExceeded("feature count"));
        }
        if self.options.folds
            > antecedent_estimate::learned_continuous::LEARNED_CONTINUOUS_MAX_FOLDS
            || self.options.bootstrap
                > antecedent_estimate::learned_continuous::LEARNED_CONTINUOUS_MAX_BOOTSTRAP
        {
            return Err(Refusal::LimitsExceeded("frozen fold or bootstrap bound"));
        }
        let r = &self.result;
        if [r.membership.len(), r.mu0.len(), r.mu1.len(), r.folds.assignment.len()]
            .iter()
            .any(|len| *len != n)
        {
            return Err(Refusal::UnsupportedSemantics("nuisance vector length"));
        }
        Ok(())
    }

    /// The provenance covers every fitted nuisance (membership, then each arm's outcome,
    /// one entry per fold), names the stored learner spec and is consistent within a role.
    /// Provider versions are producer evidence and are not compared with this build's.
    fn check_provenance(&self) -> Result<(), Refusal> {
        let folds = self.options.folds;
        let provenance = &self.result.provenance;
        if provenance.len() != 3 * folds
            || provenance.iter().any(|p| p.implementation.is_empty() || p.version.is_empty())
        {
            return Err(Refusal::ProvenanceMismatch);
        }
        let roles = [self.options.membership, self.options.outcome, self.options.outcome];
        for (block, spec) in provenance.chunks(folds).zip(roles) {
            let first = &block[0];
            let consistent = block.iter().all(|p| {
                p.spec == first.spec
                    && p.implementation == first.implementation
                    && p.version == first.version
            });
            // `auto` resolves to a data-dependent learner; every other spec names itself.
            let named =
                matches!(spec, antecedent_estimate::LearnerSpec::Auto) || first.spec == spec.name();
            if !consistent || !named {
                return Err(Refusal::ProvenanceMismatch);
            }
        }
        Ok(())
    }

    /// Verify every stored claim against an independent recomputation.
    ///
    /// # Errors
    /// The first failing check, as a typed [`LearnedContinuousArtifactError`].
    pub fn verify(
        &self,
        limits: LearnedContinuousConsumeLimits,
    ) -> Result<TransportIdentification, IoError> {
        if self.version != LEARNED_CONTINUOUS_ARTIFACT_VERSION
            && self.version != LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION
        {
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
            self.selections
                .iter()
                .map(|v| antecedent_core::VariableId::from_raw(*v))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        let query = crate::query_wire::transport_query_from_wire(&self.query)?;
        antecedent_estimate::validate_trial_query(&query)
            .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        let id = TransportIdentifier::new()
            .identify(&diagram, &query)
            .map_err(|e| Refusal::ProofMismatch(e.to_string()))?;
        match CertificateRecord::from_identification(&id) {
            Ok(record) if record == self.certificate => {}
            Ok(_) => return Err(Refusal::ProofMismatch("stored certificate differs".into()).into()),
            Err(message) => return Err(Refusal::ProofMismatch(message).into()),
        }
        validate_learned_continuous(&id, &self.input, &self.options)
            .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        let r = &self.result;
        if r.folds.count != self.options.folds
            || r.folds.scheme
                != antecedent_estimate::learned_continuous::LEARNED_CONTINUOUS_FOLD_SCHEME
            || r.folds.assignment != trial_fold_assignment(&self.input, self.options.folds)
        {
            return Err(Refusal::FoldMismatch.into());
        }
        self.check_provenance()?;
        let replay = antecedent_estimate::trial_to_target_effect(
            &id,
            &self.input.outcome,
            &self.input.treatment,
            &self.input.source,
            &r.membership,
            &self.input.randomization,
            Some((&r.mu0, &r.mu1)),
        )
        .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        if replay.aipw != Some(r.estimate) || !r.estimate.is_finite() {
            return Err(Refusal::PointMismatch.into());
        }
        if replay.overlap != r.overlap {
            return Err(Refusal::OverlapMismatch.into());
        }
        let diagnostics = trial_nuisance_diagnostics(&self.input, &r.membership, &r.mu0, &r.mu1)
            .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
        if diagnostics != r.diagnostics {
            return Err(Refusal::DiagnosticsMismatch.into());
        }
        check_membership_overlap(&r.membership, &self.options)
            .map_err(|_| IoError::from(Refusal::OverlapRefused))?;
        let expected = if self.version == LEARNED_CONTINUOUS_INTERVAL_ARTIFACT_VERSION {
            if self.options.bootstrap != 0 {
                return Err(Refusal::UncertaintyMismatch(
                    "analytic interval with bootstrap request",
                )
                .into());
            }
            let (se, interval) =
                antecedent_estimate::learned_continuous::learned_continuous_analytic_interval(
                    &self.input,
                    r,
                    self.options.coverage_level,
                )
                .map_err(|e| Refusal::InvalidRequest(e.to_string()))?;
            if r.standard_error != Some(se) || r.interval != Some(interval) {
                return Err(Refusal::IntervalMismatch.into());
            }
            LearnedContinuousUncertainty::analytic()
        } else {
            if r.interval.is_some() || r.standard_error.is_some() {
                return Err(Refusal::IntervalMismatch.into());
            }
            LearnedContinuousUncertainty::for_request(self.options.bootstrap)
        };
        if r.uncertainty != expected {
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
        assert_eq!(LEARNED_CONTINUOUS_BOUNDS, (20, 199, 2000));
    }
}
