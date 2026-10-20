//! Independent artifact for the unlicensed Bayesian smoothed dose-response transport grid.
//!
//! Format version 1 (`bayesian_smoothed_dose_v1`). The artifact stores the selection diagram,
//! the smoothed-dose query, the transport certificate the posterior ran under, the rows, and
//! the declared outcome model (dose-by-covariate basis, coefficient prior scales, residual
//! prior, window floor, draws and seed). The stored result is the joint posterior of
//! `psi_h(a)` over the grid: posterior mean and equal-tailed 95% credible interval of every
//! grid dose, the joint covariance across the grid, and the exact Gaussian moments when the
//! residual variance is known. It is a posterior for an outcome-model likelihood with
//! calibration `unmeasured`; it is not licensed.
//!
//! A consumer trusts nothing in the result. It re-derives the transport certificate from the
//! stored graph and query and requires the stored certificate record to match, then
//! recomputes the posterior from the stored rows and seed and accepts only a result that is
//! identical, bit for bit, to the stored one. Premises and data digests guard the stored
//! values against unsealed edits; a re-sealed edit is caught by the replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bayesian_posterior_row_artifact::{
    POSTERIOR_ROW_ARTIFACT_VERSION, POSTERIOR_ROW_CALIBRATION, POSTERIOR_ROW_UNCERTAINTY,
    PosteriorRowArtifactError, PosteriorRowConsumeLimits, PosteriorRowResultWire,
    ResidualPriorWire, from_estimation,
};
use crate::learned_continuous_artifact::CertificateRecord;
use crate::smoothed_dose_artifact::SmoothedDoseQueryWire;
use crate::{IoError, admg_from_wire, admg_to_wire, wire::AdmgWire};
use antecedent_core::{IdentityDomain, SmoothedDoseTransportQuery, VariableId};
use antecedent_estimate::SmoothedDoseInput;
use antecedent_estimate::smoothed_dose::DoseBasis;
use antecedent_estimate::smoothed_dose_posterior::{
    SmoothedDosePosteriorModel, smoothed_dose_posterior,
};
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const SMOOTHED_DOSE_POSTERIOR_ARTIFACT_VERSION: u32 = POSTERIOR_ROW_ARTIFACT_VERSION;
/// The feature marker of the accepted format.
pub const SMOOTHED_DOSE_POSTERIOR_ARTIFACT_FEATURE: &str = "bayesian_smoothed_dose_v1";
/// Estimator id of the smoothed dose-response posterior.
pub const SMOOTHED_DOSE_POSTERIOR_ESTIMATOR: &str = "bayesian.smoothed_dose";

/// The declared outcome model and posterior sample of the smoothed dose posterior.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDosePosteriorModelWire {
    /// Dose-by-covariate basis of `mu(t, x)`.
    pub basis: DoseBasis,
    /// Prior scale of the intercept.
    pub intercept_scale: f64,
    /// Prior scale of every other coefficient.
    pub coefficient_scale: f64,
    /// Residual-variance prior.
    pub residual: ResidualPriorWire,
    /// Fewest source rows inside every grid window.
    pub min_local_rows: usize,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draws.
    pub seed: u64,
}

/// Everything a public producer stores: premises, rows, the model and the posterior.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothedDosePosteriorArtifactWire {
    /// [`SMOOTHED_DOSE_POSTERIOR_ARTIFACT_VERSION`].
    pub version: u32,
    /// Required semantics.
    pub required_features: Vec<String>,
    /// Original static graph coordinates.
    pub graph: AdmgWire,
    /// Variables whose mechanisms differ.
    pub selections: Vec<u32>,
    /// The smoothed-dose query.
    pub query: SmoothedDoseQueryWire,
    /// Name of every graph coordinate, bound into the digests.
    pub variable_names: Vec<String>,
    /// The certificate the posterior ran under.
    pub certificate: CertificateRecord,
    /// Rows and the sampling design.
    pub input: SmoothedDoseInput,
    /// The declared outcome model.
    pub model: SmoothedDosePosteriorModelWire,
    /// The joint posterior over the grid.
    pub result: PosteriorRowResultWire,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the rows.
    pub data_digest: String,
}

/// Everything a public producer hands to [`SmoothedDosePosteriorArtifactWire::checked`].
pub struct SmoothedDosePosteriorArtifactInput<'a> {
    /// The static graph.
    pub graph: &'a antecedent_graph::Admg,
    /// Selection targets of the diagram.
    pub selections: Vec<u32>,
    /// The smoothed-dose query.
    pub query: &'a SmoothedDoseTransportQuery,
    /// Name of every graph coordinate; empty when the caller binds none.
    pub variable_names: &'a [String],
    /// The identification the posterior runs under.
    pub identification: &'a TransportIdentification,
    /// Rows and sampling design.
    pub input: &'a SmoothedDoseInput,
    /// The declared outcome model.
    pub model: SmoothedDosePosteriorModelWire,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: AdmgWire,
    selections: &'a [u32],
    query: &'a SmoothedDoseQueryWire,
    certificate: &'a CertificateRecord,
    model: &'a SmoothedDosePosteriorModelWire,
    features: &'a [u32],
    variable_names: &'a [String],
}

#[derive(Serialize)]
struct DataView<'a> {
    tag: &'static str,
    covariates: &'a [Vec<f64>],
    outcome: &'a [f64],
    dose: &'a [f64],
    source: &'a [bool],
}

type Refusal = PosteriorRowArtifactError;

fn invalid(detail: &str, message: impl Into<String>) -> Refusal {
    Refusal::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: detail.to_owned(),
        message: message.into(),
    }
}

fn render(
    query: &SmoothedDoseTransportQuery,
    model: &SmoothedDosePosteriorModelWire,
    posterior: &antecedent_estimate::smoothed_dose_posterior::SmoothedDosePosterior,
) -> PosteriorRowResultWire {
    let coordinates = query.grid.iter().map(|dose| format!("psi_h({dose})")).collect();
    let mut result = PosteriorRowResultWire::new(
        SMOOTHED_DOSE_POSTERIOR_ESTIMATOR,
        "smoothed_dose_response_transport_grid".to_owned(),
        coordinates,
        posterior.n_draws,
    )
    .with_intervals(posterior.mean.clone(), &posterior.interval_95);
    result.covariance = Some(posterior.covariance.clone());
    result.exact_mean.clone_from(&posterior.exact_mean);
    result.exact_covariance.clone_from(&posterior.exact_covariance);
    result.vectors.insert("grid".to_owned(), posterior.grid.clone());
    for (name, value) in [
        ("source_rows", posterior.source_rows),
        ("target_rows", posterior.target_rows),
        ("minimum_local_rows", posterior.minimum_local_rows),
    ] {
        result.extras.insert(name.to_owned(), f64::from(u32::try_from(value).unwrap_or(u32::MAX)));
    }
    result.extras.insert("bandwidth".to_owned(), query.bandwidth);
    result.extras.insert(
        "draws_requested".to_owned(),
        f64::from(u32::try_from(model.draws).unwrap_or(u32::MAX)),
    );
    result
}

/// Run the posterior on checked premises and render its result.
fn evaluate(
    identification: &TransportIdentification,
    query: &SmoothedDoseTransportQuery,
    input: &SmoothedDoseInput,
    model: &SmoothedDosePosteriorModelWire,
) -> Result<PosteriorRowResultWire, Refusal> {
    let declared = SmoothedDosePosteriorModel {
        basis: &model.basis,
        intercept_scale: model.intercept_scale,
        coefficient_scale: model.coefficient_scale,
        residual: model.residual.into(),
        min_local_rows: model.min_local_rows,
        draws: model.draws,
        seed: model.seed,
    };
    let posterior = smoothed_dose_posterior(identification, query, input, &declared)
        .map_err(|error| from_estimation(error, "smoothed_dose_posterior"))?;
    let result = render(query, model, &posterior);
    if !result.all_finite() {
        return Err(Refusal::Refused {
            code: antecedent_core::reason_code!("transport_numerical_failure"),
            detail: "bayesian_row.non_finite_result".to_owned(),
            message: "the posterior produced a non-finite value".to_owned(),
        });
    }
    Ok(result)
}

impl SmoothedDosePosteriorArtifactWire {
    /// Build an artifact from checked premises, run the posterior and verify the result.
    ///
    /// # Errors
    /// The premises do not encode, or the posterior refuses its inputs (an uncertified
    /// identification, a window outside the declared support, thin local support, an
    /// invalid prior or draw count).
    pub fn checked(input: SmoothedDosePosteriorArtifactInput<'_>) -> Result<Self, Refusal> {
        let undecodable = |e: IoError| Refusal::Undecodable(e.to_string());
        let certificate = CertificateRecord::from_identification(input.identification)
            .map_err(|message| invalid("smoothed_dose_posterior.not_certified", message))?;
        let result = evaluate(input.identification, input.query, input.input, &input.model)?;
        let mut wire = Self {
            version: SMOOTHED_DOSE_POSTERIOR_ARTIFACT_VERSION,
            required_features: vec![SMOOTHED_DOSE_POSTERIOR_ARTIFACT_FEATURE.to_owned()],
            graph: admg_to_wire(input.graph).map_err(undecodable)?,
            selections: input.selections,
            query: SmoothedDoseQueryWire::from_query(input.query),
            variable_names: input.variable_names.to_vec(),
            certificate,
            input: input.input.clone(),
            model: input.model,
            result,
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest().map_err(undecodable)?;
        wire.data_digest = wire.expected_data_digest().map_err(undecodable)?;
        wire.validate_shape()?;
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
        crate::digest_hex(
            IdentityDomain::TransportCertificate,
            &PremisesView {
                tag: "bayesian_smoothed_dose_premises_v1",
                graph,
                selections: &self.selections,
                query: &self.query,
                certificate: &self.certificate,
                model: &self.model,
                features: &self.input.features,
                variable_names: &self.variable_names,
            },
        )
    }

    /// The data digest the stored rows hash to.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        crate::digest_hex(
            IdentityDomain::TransportCertificate,
            &DataView {
                tag: "bayesian_smoothed_dose_data_v1",
                covariates: &self.input.covariates,
                outcome: &self.input.outcome,
                dose: &self.input.dose,
                source: &self.input.source,
            },
        )
    }

    fn validate_shape(&self) -> Result<(), Refusal> {
        let unsupported = Refusal::UnsupportedSemantics;
        if self.required_features != [SMOOTHED_DOSE_POSTERIOR_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        let result = &self.result;
        let width = result.coordinates.len();
        if result.estimator != SMOOTHED_DOSE_POSTERIOR_ESTIMATOR
            || result.uncertainty != POSTERIOR_ROW_UNCERTAINTY
            || result.calibration != POSTERIOR_ROW_CALIBRATION
            || result.licensed
        {
            return Err(unsupported("this row publishes an unlicensed posterior"));
        }
        let square =
            |values: &Option<Vec<f64>>| values.as_ref().is_none_or(|v| v.len() == width * width);
        if width != self.query.grid.len()
            || result.mean.len() != width
            || result.lower.len() != width
            || result.upper.len() != width
            || !square(&result.covariance)
            || !square(&result.exact_covariance)
            || result.exact_mean.as_ref().is_some_and(|v| v.len() != width)
        {
            return Err(unsupported("result shape"));
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
            if version == SMOOTHED_DOSE_POSTERIOR_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })?;
        wire.validate_shape()?;
        Ok(wire)
    }

    fn check_limits(&self, limits: &PosteriorRowConsumeLimits) -> Result<(), Refusal> {
        if self.input.source.len() > limits.max_rows {
            return Err(Refusal::LimitsExceeded("rows"));
        }
        if self.query.grid.len() > limits.max_points
            || self.input.features.len() > limits.max_points
        {
            return Err(Refusal::LimitsExceeded("grid doses or covariates"));
        }
        Ok(())
    }

    /// Re-derive the certificate from the stored graph and query, then recompute the
    /// posterior from the stored rows.
    fn replay(&self) -> Result<(PosteriorRowResultWire, TransportIdentification), Refusal> {
        let undecodable = |e: IoError| Refusal::Undecodable(e.to_string());
        let graph = admg_from_wire(&self.graph).map_err(undecodable)?;
        let diagram = antecedent_graph::SelectionDiagram::try_new(
            graph,
            self.selections.iter().map(|v| VariableId::from_raw(*v)).collect::<Vec<_>>(),
        )
        .map_err(|e| invalid("smoothed_dose_posterior.invalid_graph", e.to_string()))?;
        let query = self
            .query
            .to_query()
            .map_err(|error| from_estimation(error, "smoothed_dose_posterior"))?;
        let id = TransportIdentifier::new()
            .identify(&diagram, &query.transport_query())
            .map_err(|e| invalid("smoothed_dose_posterior.not_certified", e.to_string()))?;
        match CertificateRecord::from_identification(&id) {
            Ok(record) if record == self.certificate => {}
            Ok(_) => {
                return Err(invalid(
                    "smoothed_dose_posterior.certificate_mismatch",
                    "the stored certificate differs from the one the graph and query derive",
                ));
            }
            Err(message) => return Err(invalid("smoothed_dose_posterior.not_certified", message)),
        }
        let result = evaluate(&id, &query, &self.input, &self.model)?;
        Ok((result, id))
    }

    /// Decode and recompute everything under the consumer's limits; accept only a result
    /// identical to the stored one.
    ///
    /// # Errors
    /// Any reconstruction failure, with the reason code and `bayesian_row.*` detail.
    pub fn consume_typed(bytes: &[u8], limits: PosteriorRowConsumeLimits) -> Result<Self, Refusal> {
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => Refusal::UnsupportedSemantics("shape"),
            other => Refusal::Undecodable(other.to_string()),
        })?;
        wire.check_limits(&limits)?;
        let undecodable = |e: IoError| Refusal::Undecodable(e.to_string());
        crate::require_digest(
            &wire.premises_digest,
            wire.expected_premises_digest(),
            undecodable,
            || Refusal::PremisesMismatch,
        )?;
        crate::require_digest(&wire.data_digest, wire.expected_data_digest(), undecodable, || {
            Refusal::DataMismatch
        })?;
        let (fresh, _) = wire.replay()?;
        let stored = crate::to_cbor(&wire.result).map_err(undecodable)?;
        let recomputed = crate::to_cbor(&fresh).map_err(undecodable)?;
        if stored != recomputed {
            return Err(Refusal::ResultMismatch);
        }
        Ok(wire)
    }
}
