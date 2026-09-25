//! Durable transport grid and statistical-provider records.
#![allow(missing_docs)]
use crate::error::convert_err as err;
use crate::identity::digest_wire_hex as digest;
use crate::{
    IoError, exact_law_wire::ExactLawWire, query_wire::ValueWire,
    transport_catalog_wire::EvidenceCatalogWire, transport_proof::TransportProofWire,
};
use antecedent_core::IdentityDomain;
use antecedent_estimate::EmpiricalTableOptions;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StatisticalOptionsWire {
    pub estimator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learner: Option<antecedent_estimate::LearnerSpec>,
    pub bootstrap_replicates: u32,
    #[serde(default = "default_posterior_draws")]
    pub posterior_draws: u32,
    pub coverage_level: f64,
    pub max_joint_cells: usize,
}

/// Posterior draw payload for Bayesian finite structural transport.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BayesianTransportPosteriorWire {
    pub estimator: String,
    pub interval_method: String,
    #[serde(default)]
    pub interval_reason: Option<String>,
    pub draws_requested: u32,
    pub draws_ok: u32,
    pub draws_failed: u32,
    pub probabilities: Vec<Vec<f64>>,
    pub atom_intervals: Vec<(f64, f64)>,
    pub mean_intervals: Vec<(u32, f64, f64)>,
}

impl StatisticalOptionsWire {
    #[must_use]
    pub fn from_options(options: &EmpiricalTableOptions) -> Self {
        Self {
            estimator: options.estimator.as_str().into(),
            learner: match options.estimator {
                antecedent_estimate::EmpiricalTableEstimator::Learned(spec) => Some(spec),
                _ => None,
            },
            bootstrap_replicates: options.bootstrap_replicates,
            posterior_draws: options.posterior_draws,
            coverage_level: options.coverage_level,
            max_joint_cells: options.max_joint_cells,
        }
    }
    pub fn to_options(&self) -> Result<EmpiricalTableOptions, IoError> {
        let estimator = match (self.estimator.as_str(), self.learner) {
            (antecedent_estimate::EMPIRICAL_TABLE_PLUGIN, None) => {
                antecedent_estimate::EmpiricalTableEstimator::Plugin
            }
            (antecedent_estimate::EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP, None) => {
                antecedent_estimate::EmpiricalTableEstimator::EmpiricalSupportBayesianBootstrap
            }
            (antecedent_estimate::STATE_SPACE_DIRICHLET, None) => {
                antecedent_estimate::EmpiricalTableEstimator::StateSpaceDirichlet
            }
            ("transport.learned_categorical_plugin", Some(spec)) => {
                spec.validate().map_err(err)?;
                antecedent_estimate::EmpiricalTableEstimator::Learned(spec)
            }
            _ => return Err(err("unknown or inconsistent statistical provider")),
        };
        Ok(EmpiricalTableOptions {
            estimator,
            bootstrap_replicates: self.bootstrap_replicates,
            posterior_draws: self.posterior_draws,
            coverage_level: self.coverage_level,
            max_joint_cells: self.max_joint_cells,
        })
    }
}

fn default_posterior_draws() -> u32 {
    199
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SampleSummary {
    pub population: String,
    pub regime: u32,
    pub snapshot: String,
    pub interventions: Vec<(u32, ValueWire)>,
    pub n: u32,
    pub content_digest: String,
}
impl SampleSummary {
    pub fn from_sample(sample: &antecedent_estimate::RegimeSample) -> Result<Self, IoError> {
        let columns: Vec<_> = sample.columns.iter().map(|(v, xs)| (v.raw(), xs)).collect();
        let mut interventions: Vec<_> = sample
            .interventions
            .iter()
            .map(|a| (a.variable.raw(), ValueWire::from_value(&a.value)))
            .collect();
        interventions.sort_by_key(|a| a.0);
        Ok(Self {
            population: sample.population.to_string(),
            regime: sample.regime.raw(),
            snapshot: sample.snapshot_identity.to_string(),
            interventions,
            n: u32::try_from(sample.n()).map_err(err)?,
            content_digest: digest(IdentityDomain::DataSnapshot, &columns)?,
        })
    }
}

/// Located point-local missing evidence or support outcome.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TransportGridFailureWire {
    /// Stable kind: `missing_evidence` or `support_failure`.
    pub kind: String,
    /// Located provider/denominator explanation.
    pub detail: String,
    /// Stable provider/support failure code.
    pub code: String,
    /// Original variables whose provider support is required.
    pub variables: Vec<u32>,
    /// Original expression coordinate for a failed ratio.
    pub expression: Option<u32>,
    /// Located original variable assignments.
    pub assignment: Vec<(u32, ValueWire)>,
    /// Population/regime dependencies of the failing factor.
    pub bindings: Vec<(String, Option<u32>)>,
    /// Concrete intervention world required by this factor.
    pub interventions: Vec<(u32, ValueWire)>,
}

impl TransportGridFailureWire {
    #[must_use]
    pub fn from_failure(f: &antecedent_core::TransportGridFailure) -> Self {
        Self {
            kind: f.kind.clone(),
            detail: f.detail.clone(),
            code: f.code.clone(),
            variables: f.variables.clone(),
            expression: f.expression,
            assignment: f.assignment.iter().map(|(v, x)| (*v, ValueWire::from_value(x))).collect(),
            bindings: f.bindings.clone(),
            interventions: f
                .interventions
                .iter()
                .map(|(v, x)| (*v, ValueWire::from_value(x)))
                .collect(),
        }
    }
    #[must_use]
    pub fn to_failure(&self) -> antecedent_core::TransportGridFailure {
        antecedent_core::TransportGridFailure {
            kind: self.kind.clone(),
            detail: self.detail.clone(),
            code: self.code.clone(),
            variables: self.variables.clone(),
            expression: self.expression,
            assignment: self.assignment.iter().map(|(v, x)| (*v, x.to_value())).collect(),
            bindings: self.bindings.clone(),
            interventions: self.interventions.iter().map(|(v, x)| (*v, x.to_value())).collect(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridWire {
    pub version: u32,
    pub required_features: Vec<String>,
    pub nodes: Vec<u32>,
    pub directed: Vec<(u32, u32)>,
    pub bidirected: Vec<(u32, u32)>,
    pub selections: Vec<u32>,
    pub proof: TransportProofWire,
    pub catalog: EvidenceCatalogWire,
    pub laws: Vec<ExactLawWire>,
    pub at: Vec<Vec<(u32, ValueWire)>>,
    pub operations: usize,
    pub depth: usize,
    pub seed: u64,
    pub statistical: bool,
    pub samples: Vec<SampleSummary>,
    pub options: Option<StatisticalOptionsWire>,
    pub points: Vec<GridPointWire>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub point_evidence: Vec<String>,
    pub reasoning: crate::contract_section::ReasoningSectionWire,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum GridPointWire {
    Exact(Vec<u8>),
    Statistical(Vec<u8>),
    Unavailable(TransportGridFailureWire),
}
impl GridWire {
    pub fn export(&self, identity: &str) -> Result<Vec<u8>, IoError> {
        encode_transport_section("transport_grid_v1", "transport_grid", identity, self)
    }
}

/// Shared sectioned container writer for retained transport products.
pub fn encode_transport_section(
    section_id: &str,
    kind: &str,
    identity: &str,
    payload: &impl Serialize,
) -> Result<Vec<u8>, IoError> {
    use crate::container::{ArtifactManifest, CompressPolicy, EncodedArtifact, pack_section};
    use crate::wire::{ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion};
    let (descriptor, section) = pack_section(
        section_id,
        "application/cbor",
        crate::to_cbor(payload)?,
        CompressPolicy::Never,
    );
    let artifact = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: FormatVersion { major: 1, minor: 0 },
            minimum_reader_version: FormatVersion { major: 1, minor: 0 },
            artifact_kind: ArtifactKind::Other(kind.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(err)?,
            artifact_id: identity.into(),
            sections: vec![descriptor],
            provenance: ProvenanceWire {
                note: "checked transport; pointwise uncertainty; no new calibration claim".into(),
            },
        },
        sections: vec![section],
    };
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes)?;
    Ok(bytes)
}

/// Bind retained scientific point evidence without constructing a child artifact.
/// # Errors
/// Canonical serialization failure.
pub fn point_evidence_identity(
    execution: &str,
    distribution: &antecedent_expr::ExactDistribution,
    estimate: Option<&antecedent_estimate::StatisticalTransportEstimate>,
) -> Result<String, IoError> {
    let atoms: Vec<Vec<ValueWire>> = distribution
        .atoms
        .iter()
        .map(|row| row.iter().map(ValueWire::from_value).collect())
        .collect();
    let support: Vec<_> = distribution
        .support
        .iter()
        .map(|r| {
            (
                r.expression.raw(),
                r.status,
                r.denominator,
                r.assignment
                    .iter()
                    .map(|(v, x)| (v.raw(), ValueWire::from_value(x)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let uncertainty = estimate.and_then(|e| e.uncertainty.as_ref()).map(|r| {
        (
            r.estimator.as_ref(),
            r.method.as_ref(),
            r.coverage_target,
            r.interval_scope.as_ref(),
            r.sample_sizes.iter().map(|(s, n)| (s.as_ref(), *n)).collect::<Vec<_>>(),
            r.support_regime.as_ref(),
            r.replicates_requested,
            r.replicates_ok,
            r.replicates_failed,
            r.seed,
            r.calibration_binding.as_deref(),
        )
    });
    let mean_intervals = estimate
        .and_then(|e| e.mean_intervals.as_ref())
        .map(|rows| rows.iter().map(|(v, lo, hi)| (v.raw(), lo, hi)).collect::<Vec<_>>());
    digest(
        IdentityDomain::Execution,
        &(
            execution,
            atoms,
            distribution.probabilities.as_ref(),
            support,
            uncertainty,
            estimate.and_then(|e| e.atom_intervals.as_deref()),
            mean_intervals,
            estimate
                .and_then(|e| e.atom_replicates.as_deref())
                .map(|rows| rows.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>()),
            estimate.and_then(|e| e.replicate_ids.as_deref()),
            estimate.and_then(|e| e.uncertainty_reason.as_deref()),
        ),
    )
}

/// Versioned grid identity; v2 binds evidence digests instead of encoded child bytes.
/// # Errors
/// Unsupported version or serialization failure.
pub fn grid_identity(wire: &GridWire) -> Result<String, IoError> {
    match wire.version {
        1 => digest(IdentityDomain::Execution, wire),
        2 => {
            let mut metadata = wire.clone();
            for point in &mut metadata.points {
                match point {
                    GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => {
                        bytes.clear()
                    }
                    GridPointWire::Unavailable(_) => {}
                }
            }
            digest(IdentityDomain::Execution, &metadata)
        }
        _ => Err(err("unsupported grid identity version")),
    }
}
