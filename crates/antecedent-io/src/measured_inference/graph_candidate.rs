//! Source-bound measured adapter for the graph-selection imputation composite.
//!
//! A decoded source is data, not authority. The consumer reconstructs the original table,
//! re-derives the operation of every supported graph (identification status and adjustment set)
//! from the graph itself and refuses unless each equals the operation the compiler retained and
//! sealed into its program. The support rule that drops a graph with several minimal back-door
//! adjustment sets is applied here, from the identifier's own output, and the compiler asks this
//! module for it, so both sides keep the same graphs. It then executes the single native
//! pipeline under that program identity. The measured factory is reached only through [`authorize`], so every interval is
//! reported, `calibrated` exactly when a current attesting coverage record covers the execution.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::assumption::AssumptionSource;
use antecedent_core::execution::{AdmissionError, AdmissionLease, RngFactory, mix64};
use antecedent_core::graph_imputation_contract::{
    AtomOperation, GraphImputationParts, GraphImputationSource, GraphSupport, SUPPORT_RULE,
};
use antecedent_core::search::{SearchBudget, SearchLimits};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CalibrationBasis, CanonicalEncode, CanonicalWriter,
    CausalQuery, DistributionRef, ExecutionContext, IdentificationStatus, IdentityDomain,
    PopulationRegistry, SemanticDigest, TargetPopulation, VariableId,
};
use antecedent_data::{
    ImputationDomain, ImputationSpec, TableView, TabularData, complete_float_columns,
};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, dag_from_adjacency_mask};
use antecedent_discovery::graph_imputation_enumeration::{
    BoundedExactDagError, graph_imputation_exact_dags, graph_imputation_supported_masks,
};
use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};
use antecedent_estimate::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, EnvelopeOptions,
    GraphEffectDraws, OverlapPolicy, aggregate_effect_envelope,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{BackdoorIdentifier, IdentificationWorkspace};
use antecedent_prob::{
    BayesLikelihood, GraphIdentFlag, InferenceDiagnostics, PosteriorQuantityKind, WeightedGraphSamples,
};
use serde::{Deserialize, Serialize};

use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;

/// Family of data-generating processes a prior-predictive study of this procedure draws from.
/// The route reports this coordinate, so only a record measured under it can cover a run.
const PRIOR_PREDICTIVE_FAMILY: &str = "prior_predictive.v2:effect_uniform_minus1_plus1:linear_gaussian_outcome:logistic_binary_treatment:outcome_missing_at_random_by_treatment:";
const MEMORY_LIMIT: u64 = 25 * 1024 * 1024;

/// Target covariate sample for the separate graph-imputation transport route.
/// Every column names a source schema index and contains complete binary cells.
/// The target is a representative sample; selection targets name pretreatment
/// covariate mechanisms allowed to differ. The transport fit includes every
/// selected covariate even when a graph's minimal backdoor set omits it.
/// No target outcome or treatment is accepted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportStandardization {
    /// Target covariate columns in ascending source-schema index order.
    pub columns: Vec<(u32, Vec<f64>)>,
    /// Variables whose population mechanisms may differ, in ascending order.
    pub selection_targets: Vec<u32>,
    /// Population represented by the original graph-imputation sample.
    pub source_population: String,
    /// Empirical target covariate frame. Its sampling uncertainty is not
    /// propagated by this route.
    pub target_population: String,
    /// Independent source sample identity declared by the caller.
    pub source_sample: String,
    /// Independent target sample identity declared by the caller.
    pub target_sample: String,
    /// Source unit identities in original row order.
    pub source_unit_ids: Vec<String>,
    /// Target unit identities in representative row order.
    pub target_unit_ids: Vec<String>,
}

impl TransportStandardization {
    /// Validate the target's binary sampling frame and its graph-scope premise.
    ///
    /// # Errors
    /// Missing, duplicate, nonbinary, or unsupported target rows or selections.
    pub fn validate(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        self.validate_for_support(parts, GraphSupport::TreatmentRootPreTreatmentCovariates)
    }

    /// Validate a separate confounded-pretreatment target, including the
    /// augmented adjustment set in every retained graph.
    ///
    /// # Errors
    /// Unsupported rows, graph support or an opened backdoor path.
    pub fn validate_confounded(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        self.validate_for_support(parts, GraphSupport::PreTreatmentCovariates)?;
        let n = parts.names.len();
        for atom in &parts.atoms {
            let dag = dag_from_adjacency_mask(AdjacencyMask::from_word(atom.mask), n)
                .map_err(invalid)?;
            let treatment = antecedent_graph::DenseNodeId::from_raw(parts.treatment);
            let outcome = antecedent_graph::DenseNodeId::from_raw(parts.outcome);
            let mut adjustment = atom.adjustment.clone();
            for &id in &self.selection_targets {
                if !adjustment.contains(&id) { adjustment.push(id); }
            }
            let adjustment: Vec<_> = adjustment.into_iter()
                .map(antecedent_graph::DenseNodeId::from_raw).collect();
            let overlay = antecedent_graph::overlay::GraphOverlay::remove_outgoing(n, &[treatment]);
            let separated = dag.view(&overlay).is_d_separated(treatment, outcome, &adjustment,
                &mut antecedent_graph::DSeparationWorkspace::default()).map_err(invalid)?;
            if !separated {
                return Err(invalid("selected covariates open a backdoor path in a retained graph"));
            }
        }
        Ok(())
    }

    fn validate_for_support(&self, parts: &GraphImputationParts, support: GraphSupport) -> Result<(), IoError> {
        if self.source_population.trim().is_empty() || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
            || self.source_sample.trim().is_empty() || self.target_sample.trim().is_empty()
            || self.source_sample == self.target_sample
            || self.columns.is_empty() || self.selection_targets.is_empty()
        {
            return Err(invalid("transport needs distinct source and target samples and selected covariates"));
        }
        if parts.support != support {
            return Err(invalid(if support == GraphSupport::TreatmentRootPreTreatmentCovariates {
                "target standardization requires the root-treatment pretreatment graph support"
            } else { "target standardization requires the confounded pretreatment graph support" }));
        }
        let rows = self.columns[0].1.len();
        if rows < 2 || self.columns.iter().any(|(_, values)| values.len() != rows) {
            return Err(invalid("target covariate columns need aligned representative rows"));
        }
        if self.source_unit_ids.len() != parts.rows() || self.target_unit_ids.len() != rows {
            return Err(invalid("source and target unit identities must align with their rows"));
        }
        let mut units = BTreeSet::new();
        if self.source_unit_ids.iter().chain(&self.target_unit_ids).any(|unit| {
            unit.trim().is_empty() || !units.insert(unit.as_str())
        }) {
            return Err(invalid("source and target units must have distinct nonblank identities"));
        }
        let mut previous = None;
        for (id, values) in &self.columns {
            if previous.is_some_and(|last| last >= *id)
                || *id as usize >= parts.names.len()
                || *id == parts.treatment || *id == parts.outcome
                || parts.domains[*id as usize] != 1
                || values.iter().any(|value| *value != 0.0 && *value != 1.0)
            {
                return Err(invalid("target columns must be unique binary pretreatment covariates"));
            }
            previous = Some(*id);
        }
        let mut last = None;
        for &id in &self.selection_targets {
            if last.is_some_and(|previous| previous >= id)
                || !self.columns.iter().any(|(column, _)| *column == id)
            {
                return Err(invalid("selection targets must be unique target covariates"));
            }
            last = Some(id);
        }
        if parts.atoms.iter().any(|atom| {
            atom.adjustment.iter().any(|id| {
                    !self.columns.iter().any(|(column, _)| column == id)
                        || parts.domains[*id as usize] != 1
                })
        }) {
            return Err(invalid("every retained graph needs binary target data for its adjustment set"));
        }
        Ok(())
    }

    /// Canonical identity of the target rows, populations, sample identities
    /// and structural selection premise.
    ///
    /// # Errors
    /// Encoding refusal.
    pub fn digest(&self) -> Result<[u8; 32], IoError> {
        Ok(CanonicalWriter::digest_bytes(
            "antecedent.graph_imputation_transport.target.v1",
            &crate::to_cbor(self)?,
        ))
    }

    /// Number of representative target rows after validation.
    #[must_use]
    pub fn rows(&self) -> usize { self.columns.first().map_or(0, |column| column.1.len()) }
}

/// Independent target frame for mixed binary and continuous pretreatment
/// covariates. Continuous columns use declared Gaussian kernel bandwidths for
/// source-to-target density-ratio standardization on every completed table.
/// The empirical target frame and bandwidths remain model assumptions; this
/// route earns no interval coverage or decision reliability from the binary
/// stratum route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelTransportStandardization {
    /// Complete target covariate columns in ascending source-schema order.
    pub columns: Vec<(u32, Vec<f64>)>,
    /// Pretreatment mechanisms allowed to differ, in ascending order.
    pub selection_targets: Vec<u32>,
    /// Source population represented by the graph-imputation sample.
    pub source_population: String,
    /// Population represented by the independent target frame.
    pub target_population: String,
    /// Source-sample identity bound to the upstream compiler.
    pub source_sample: String,
    /// Independent target-sample identity.
    pub target_sample: String,
    /// Source unit identities in original row order.
    pub source_unit_ids: Vec<String>,
    /// Target unit identities in representative row order.
    pub target_unit_ids: Vec<String>,
    /// One positive finite bandwidth for every continuous target column.
    pub bandwidths: Vec<(u32, f64)>,
    /// Minimum mean source kernel support at each target row in (0, 1).
    pub min_kernel_overlap: f64,
}

impl KernelTransportStandardization {
    /// Validate source and target independence, mixed domains, bandwidths and
    /// the graph's pretreatment adjustment support before numerical execution.
    ///
    /// # Errors
    /// Missing columns, unsafe selections, invalid data or kernel policy.
    pub fn validate(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        if self.source_population.trim().is_empty() || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
            || self.source_sample.trim().is_empty() || self.target_sample.trim().is_empty()
            || self.source_sample == self.target_sample
            || self.columns.is_empty() || self.selection_targets.is_empty()
            || parts.support != GraphSupport::TreatmentRootPreTreatmentCovariates
            || !self.min_kernel_overlap.is_finite()
            || !(0.0..1.0).contains(&self.min_kernel_overlap)
        {
            return Err(invalid("kernel transport needs distinct samples, pretreatment support and a bounded overlap floor"));
        }
        let rows = self.rows();
        if rows < 2 || self.columns.iter().any(|(_, values)| values.len() != rows)
            || self.source_unit_ids.len() != parts.rows()
            || self.target_unit_ids.len() != rows
            || parts.rows().checked_mul(rows.saturating_add(parts.rows()))
                .and_then(|pairs| pairs.checked_mul(parts.atoms.len()))
                .and_then(|pairs| pairs.checked_mul(parts.imputations as usize))
                .is_none_or(|pairs| pairs > 50_000_000)
        {
            return Err(invalid("kernel target columns and unit identities must align"));
        }
        let mut units = BTreeSet::new();
        if self.source_unit_ids.iter().chain(&self.target_unit_ids).any(|unit| {
            unit.trim().is_empty() || !units.insert(unit.as_str())
        }) {
            return Err(invalid("source and target units must have distinct nonblank identities"));
        }
        let mut previous = None;
        for (id, values) in &self.columns {
            if previous.is_some_and(|last| last >= *id)
                || *id as usize >= parts.names.len()
                || *id == parts.treatment || *id == parts.outcome
                || values.iter().any(|value| !value.is_finite())
                || (parts.domains[*id as usize] == 1
                    && values.iter().any(|value| *value != 0.0 && *value != 1.0))
            {
                return Err(invalid("kernel target columns must be unique complete pretreatment covariates in their domains"));
            }
            previous = Some(*id);
        }
        let mut last = None;
        for &id in &self.selection_targets {
            if last.is_some_and(|previous| previous >= id)
                || !self.columns.iter().any(|(column, _)| *column == id)
            {
                return Err(invalid("selection targets must be unique target covariates"));
            }
            last = Some(id);
        }
        if parts.atoms.iter().any(|atom| atom.adjustment.iter().any(|id| {
            !self.columns.iter().any(|(column, _)| column == id)
        })) {
            return Err(invalid("every graph adjustment needs a complete target column"));
        }
        let continuous: Vec<u32> = self.columns.iter().filter_map(|(id, _)| {
            (parts.domains[*id as usize] == 0).then_some(*id)
        }).collect();
        if continuous.is_empty()
            || !self.selection_targets.iter().any(|id| continuous.contains(id))
            || self.bandwidths.len() != continuous.len()
            || self.bandwidths.iter().zip(continuous).any(|(&(id, bandwidth), expected)| {
                id != expected || !bandwidth.is_finite() || bandwidth <= 0.0
            })
        {
            return Err(invalid("each continuous target column needs one positive bandwidth"));
        }
        Ok(())
    }

    /// Source-target frame, kernel policy and selection-premise digest.
    ///
    /// # Errors
    /// Encoding refusal.
    pub fn digest(&self) -> Result<[u8; 32], IoError> {
        Ok(CanonicalWriter::digest_bytes(
            "antecedent.graph_imputation_kernel_transport.target.v1",
            &crate::to_cbor(self)?,
        ))
    }

    /// Number of aligned target rows.
    #[must_use]
    pub fn rows(&self) -> usize { self.columns.first().map_or(0, |column| column.1.len()) }
}

fn invalid(message: impl std::fmt::Display) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("graph_imputation.invalid_source: {message}"),
    }
}

fn execution_refused(message: impl std::fmt::Display) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("graph_imputation.execution_refused: {message}"),
    }
}

fn cancelled() -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cancelled_no_claim"),
        message: "graph_imputation.cancelled: candidate execution cancelled".to_owned(),
    }
}

fn admission(error: AdmissionError) -> IoError {
    match error {
        AdmissionError::Cancelled => cancelled(),
        other => IoError::Refused {
            code: antecedent_core::reason_code!("execution_resource_limit"),
            message: format!("graph_imputation.resource_limit: {other}"),
        },
    }
}

/// One supported graph with the operation re-derived for it.
struct DerivedAtom {
    operation: AtomOperation,
    estimand: IdentifiedEstimand,
}

/// The retained graphs of a support and the number the support rule excluded.
struct DerivedSupport {
    atoms: Vec<DerivedAtom>,
    excluded: u32,
}

/// Re-derive every structurally supported graph's effect proof and apply the support rule: a
/// graph whose identifier lists more than one adjustment-shaped estimand has no single minimal
/// back-door set, names no single operation and is excluded. This is the consumer's own
/// identification of the declared graphs; it never reads the carried operations.
fn derive_atoms(
    n: usize,
    treatment: u32,
    outcome: u32,
    support: GraphSupport,
    query: &AverageEffectQuery,
) -> Result<DerivedSupport, IoError> {
    let masks = graph_imputation_supported_masks(n, treatment as usize, outcome as usize, support);
    let mut derived = Vec::with_capacity(masks.len());
    let mut excluded = 0u32;
    for mask in masks {
        let dag = dag_from_adjacency_mask(AdjacencyMask::from_word(mask), n)
            .map_err(|error| invalid(format!("graph mask {mask}: {error}")))?;
        let identifier = BackdoorIdentifier::new();
        let prepared = identifier
            .prepare(&dag)
            .map_err(|error| invalid(format!("graph mask {mask}: {error}")))?;
        let result = identifier
            .identify(
                &prepared,
                &CausalQuery::AverageEffect(query.clone()),
                &mut IdentificationWorkspace::default(),
            )
            .map_err(|error| invalid(format!("graph mask {mask}: {error}")))?;
        if result.estimands.iter().filter(|estimand| estimand.is_adjustment_shaped()).count() > 1 {
            excluded += 1;
            continue;
        }
        let index = result
            .select_estimand_index(IdentifiedEstimand::is_adjustment_shaped)
            .map_err(|error| invalid(format!("graph mask {mask}: {error:?}")))?;
        let estimand = result.estimands[index].clone();
        let status = if result.estimand_claims.is_empty() {
            result.status
        } else {
            result.narrowed_to(index).map_or(result.status, |claim| claim.status)
        };
        if !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(invalid(format!("supported graph mask {mask} is not point identified")));
        }
        let adjustment = estimand.adjustment_set.iter().map(|variable| variable.raw()).collect();
        derived
            .push(DerivedAtom { operation: AtomOperation { mask, status, adjustment }, estimand });
    }
    Ok(DerivedSupport { atoms: derived, excluded })
}

/// The graphs a declared support retains and the operation of each, with the number of
/// structurally supported graphs the support rule excluded.
#[derive(Debug, Eq, PartialEq)]
pub struct RetainedSupport {
    /// Operation of every retained graph, in increasing mask order.
    pub operations: Vec<AtomOperation>,
    /// Structurally supported graphs with several minimal back-door sets.
    pub excluded: u32,
}

/// Derive the retained graphs of a declared support from the graphs themselves. The compiler and
/// the consumer both obtain their graph set here.
///
/// # Errors
/// A structurally supported graph that has no adjustment set cannot be identified.
pub fn retained_support(
    n: usize,
    treatment: u32,
    outcome: u32,
    support: GraphSupport,
) -> Result<RetainedSupport, IoError> {
    let query = AverageEffectQuery::binary_ate(
        VariableId::from_raw(treatment),
        VariableId::from_raw(outcome),
    );
    let derived = derive_atoms(n, treatment, outcome, support, &query)?;
    Ok(RetainedSupport {
        operations: derived.atoms.into_iter().map(|atom| atom.operation).collect(),
        excluded: derived.excluded,
    })
}

/// The retained support a compiler would declare for these parts, derived from the graphs.
///
/// # Errors
/// A supported graph cannot be identified.
pub fn derive_support(parts: &GraphImputationParts) -> Result<RetainedSupport, IoError> {
    retained_support(parts.names.len(), parts.treatment, parts.outcome, parts.support)
}

struct Reconstructed {
    source: GraphImputationSource,
    table: TabularData,
    variables: Arc<[VariableId]>,
    imputer: Imputer,
    query: AverageEffectQuery,
    atoms: Vec<DerivedAtom>,
    _lease: AdmissionLease,
    _graph_lease: AdmissionLease,
}

fn reconstruct(bytes: &[u8], ctx: &ExecutionContext) -> Result<Reconstructed, IoError> {
    let lease = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(16).saturating_add(1 << 20))
        .map_err(admission)?;
    let source = GraphImputationSource::decode(bytes).map_err(invalid)?;
    let parts = source.parts();
    let n = parts.names.len();
    let graph_lease = ctx
        .acquire_admission(
            0,
            antecedent_core::graph_imputation_contract::admission_bytes(parts.rows(), n),
        )
        .map_err(admission)?;
    let table = TabularData::from_f64_columns(
        parts
            .names
            .iter()
            .zip(&parts.columns)
            .map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .map_err(invalid)?;
    let digest = table.storage().content_digest_with_context(ctx).map_err(admission)?;
    if digest != parts.data_digest {
        return Err(invalid("the reconstructed table differs from the compiler's source identity"));
    }
    let variables: Arc<[VariableId]> = Arc::from(
        (0..n)
            .map(|index| u32::try_from(index).map(VariableId::from_raw))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| invalid("variable index exceeds platform"))?,
    );
    let mut columns = Vec::with_capacity(n);
    let mut domains = Vec::with_capacity(n);
    for (index, &id) in variables.iter().enumerate() {
        let variable = table.schema().get(id).map_err(invalid)?;
        columns.push(ImputationColumn::new(
            Arc::clone(&variable.name),
            table.float64_values(id).map_err(invalid)?,
        ));
        domains.push((
            Arc::clone(&variable.name),
            if parts.domains[index] == 1 {
                ImputationDomain::Binary
            } else {
                ImputationDomain::Continuous
            },
        ));
    }
    let spec = ImputationSpec::new()
        .with_imputations(parts.imputations)
        .with_iterations(parts.iterations)
        .with_seed(parts.imputation_seed);
    let imputer = Imputer::new_with_domains(columns, spec, ImputationUnits::Rows, &domains)
        .map_err(invalid)?;
    let query = AverageEffectQuery::binary_ate(
        VariableId::from_raw(parts.treatment),
        VariableId::from_raw(parts.outcome),
    );
    // The compiler's retained child operations must equal those derived here, atom for atom.
    let DerivedSupport { atoms, excluded } =
        derive_atoms(n, parts.treatment, parts.outcome, parts.support, &query)?;
    if atoms.len() != parts.atoms.len()
        || atoms.iter().zip(&parts.atoms).any(|(derived, carried)| derived.operation != *carried)
    {
        return Err(invalid(
            "a retained child operation differs from the one derived from its graph",
        ));
    }
    if excluded != parts.excluded {
        return Err(invalid(
            "the excluded graph count differs from the one the support rule derives",
        ));
    }
    Ok(Reconstructed {
        source,
        table,
        variables,
        imputer,
        query,
        atoms,
        _lease: lease,
        _graph_lease: graph_lease,
    })
}

struct Execution {
    draws: Vec<f64>,
    paired_worlds: Vec<[f64; 2]>,
    mean: f64,
    sd: f64,
    receipt: [u8; 32],
    unidentified_by_table: Vec<f64>,
    lease: AdmissionLease,
    paired_leases: Vec<AdmissionLease>,
}

#[derive(Clone, Copy)]
enum TransportTarget<'a> {
    Binary(&'a TransportStandardization),
    ConfoundedBinary(&'a TransportStandardization),
    Kernel(&'a KernelTransportStandardization),
}

impl<'a> TransportTarget<'a> {
    fn columns(self) -> &'a [(u32, Vec<f64>)] {
        match self { Self::Binary(value) | Self::ConfoundedBinary(value) => &value.columns, Self::Kernel(value) => &value.columns }
    }

    fn selections(self) -> &'a [u32] {
        match self {
            Self::Binary(value) | Self::ConfoundedBinary(value) => &value.selection_targets,
            Self::Kernel(value) => &value.selection_targets,
        }
    }

    fn rows(self) -> usize {
        match self { Self::Binary(value) | Self::ConfoundedBinary(value) => value.rows(), Self::Kernel(value) => value.rows() }
    }

    fn digest(self) -> Result<[u8; 32], IoError> {
        match self { Self::Binary(value) | Self::ConfoundedBinary(value) => value.digest(), Self::Kernel(value) => value.digest() }
    }
}

fn kernel_transport_weights(
    source_columns: &[Vec<f64>], target_columns: &[&[f64]],
    adjustment: &[u32], domains: &[u8], target: &KernelTransportStandardization,
) -> Result<Vec<f64>, IoError> {
    let source_rows = source_columns.first().map_or(0, Vec::len);
    let target_rows = target.rows();
    let kernel = |source: usize, other: usize, other_target: bool| -> f64 {
        let mut log_weight = 0.0;
        for (position, &id) in adjustment.iter().enumerate() {
            let left = source_columns[position][source];
            let right = if other_target {
                target_columns[position][other]
            } else {
                source_columns[position][other]
            };
            if domains[id as usize] == 1 {
                if left != right { return 0.0; }
            } else {
                let bandwidth = target.bandwidths.iter()
                    .find(|(column, _)| *column == id)
                    .map(|(_, bandwidth)| *bandwidth)
                    .unwrap_or(f64::NAN);
                let distance = (left - right) / bandwidth;
                log_weight -= 0.5 * distance * distance;
            }
        }
        log_weight.exp()
    };
    for target_row in 0..target_rows {
        let support = (0..source_rows).map(|source_row| {
            kernel(source_row, target_row, true)
        }).sum::<f64>() / source_rows as f64;
        if !support.is_finite() || support < target.min_kernel_overlap {
            return Err(execution_refused("target covariate row lacks declared kernel overlap in the completed source"));
        }
    }
    let mut weights = Vec::with_capacity(source_rows);
    for source_row in 0..source_rows {
        let numerator = (0..target_rows).map(|target_row| {
            kernel(source_row, target_row, true)
        }).sum::<f64>() / target_rows as f64;
        let denominator = (0..source_rows).map(|other| {
            kernel(source_row, other, false)
        }).sum::<f64>() / source_rows as f64;
        let weight = numerator / denominator;
        if !weight.is_finite() || weight < 0.0 {
            return Err(execution_refused("kernel target-to-source density ratio is invalid"));
        }
        weights.push(weight);
    }
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(execution_refused("kernel transport has no positive source mass"));
    }
    let scale = source_rows as f64 / total;
    for weight in &mut weights { *weight *= scale; }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(execution_refused("normalized kernel transport weights are nonfinite"));
    }
    Ok(weights)
}

fn transported_population(
    data: &TabularData,
    query: &AverageEffectQuery,
    atom: &DerivedAtom,
    target: TransportTarget<'_>, domains: &[u8], retain_weights: bool,
) -> Result<(AverageEffectQuery, PopulationRegistry, IdentifiedEstimand, Option<Vec<f64>>), IoError> {
    let mut adjustment = atom.operation.adjustment.clone();
    for &selected in target.selections() {
        if !adjustment.contains(&selected) { adjustment.push(selected); }
    }
    adjustment.sort_unstable();
    let mut estimand = atom.estimand.clone();
    estimand.adjustment_set = Arc::from(adjustment.iter().copied()
        .map(VariableId::from_raw).collect::<Vec<_>>());
    let mut source_columns = Vec::with_capacity(adjustment.len());
    let mut target_columns = Vec::with_capacity(adjustment.len());
    for &id in &adjustment {
        source_columns.push(data.float64_values(VariableId::from_raw(id)).map_err(invalid)?);
        target_columns.push(target.columns().iter().find(|(column, _)| *column == id)
            .ok_or_else(|| invalid("target is missing an identified adjustment covariate"))?.1.as_slice());
    }
    let binary_key = |columns: &[&[f64]], row: usize| -> Result<Vec<u8>, IoError> {
        columns.iter().map(|column| match column[row] {
            0.0 => Ok(0), 1.0 => Ok(1),
            _ => Err(execution_refused("completed source or target covariate is not binary")),
        }).collect()
    };
    let weights = match target {
        TransportTarget::Binary(binary) | TransportTarget::ConfoundedBinary(binary) => {
            let source_slices: Vec<&[f64]> = source_columns.iter().map(Vec::as_slice).collect();
            let source_rows = data.row_count();
            let mut source_count = BTreeMap::<Vec<u8>, usize>::new();
            let mut target_count = BTreeMap::<Vec<u8>, usize>::new();
            let mut source_keys = Vec::with_capacity(source_rows);
            for row in 0..source_rows {
                let key = binary_key(&source_slices, row)?;
                *source_count.entry(key.clone()).or_default() += 1;
                source_keys.push(key);
            }
            for row in 0..binary.rows() {
                *target_count.entry(binary_key(&target_columns, row)?).or_default() += 1;
            }
            if target_count.iter().any(|(key, count)| *count > 0 && !source_count.contains_key(key)) {
                return Err(execution_refused("target adjustment stratum has no completed source support"));
            }
            if matches!(target, TransportTarget::ConfoundedBinary(_)) {
                let treatment = data.float64_values(query.treatment).map_err(invalid)?;
                let mut arms = BTreeMap::<Vec<u8>, (bool, bool)>::new();
                for (key, &value) in source_keys.iter().zip(&treatment) {
                    let entry = arms.entry(key.clone()).or_default();
                    match value { 0.0 => entry.0 = true, 1.0 => entry.1 = true,
                        _ => return Err(execution_refused("completed treatment must be binary")) }
                }
                if target_count.keys().any(|key| !arms.get(key).is_some_and(|&(control, treated)| control && treated)) {
                    return Err(execution_refused("target stratum lacks both source treatment arms"));
                }
            }
            let source_n = source_rows as f64;
            let target_n = binary.rows() as f64;
            source_keys.iter().map(|key| {
                let target_count = *target_count.get(key).unwrap_or(&0) as f64;
                let source_count = source_count[key] as f64;
                target_count * source_n / (source_count * target_n)
            }).collect()
        }
        TransportTarget::Kernel(kernel) => kernel_transport_weights(
            &source_columns, &target_columns, &adjustment, domains, kernel,
        )?,
    };
    let retained_weights = retain_weights.then(|| weights.clone());
    let handle = DistributionRef::from_raw(1);
    let mut registry = PopulationRegistry::new();
    registry.insert_distribution_with_dependence(
        handle, Arc::<[f64]>::from(weights),
        Arc::<[VariableId]>::from(adjustment.iter().copied()
            .map(VariableId::from_raw).collect::<Vec<_>>()),
    );
    Ok((
        query.clone().with_target_population(TargetPopulation::CustomDistribution(handle)),
        registry, estimand, retained_weights,
    ))
}

#[allow(
    clippy::too_many_lines,
    reason = "the composite seals each completed table, graph fit and mixture together"
)]
fn execute(
    rec: &Reconstructed, target: Option<TransportTarget<'_>>, paired: bool,
    ctx: &ExecutionContext,
) -> Result<Execution, IoError> {
    let parts = rec.source.parts();
    let (treatment, outcome) =
        (VariableId::from_raw(parts.treatment), VariableId::from_raw(parts.outcome));
    let program = SemanticDigest::from_bytes(rec.source.program());
    let contract = parts.contract_digest();
    let scoped = ctx.enter_lowered_dispatch(program);
    let quote = u64::from(parts.posterior_draws)
        .checked_mul(u64::from(parts.imputations))
        .and_then(|v| v.checked_mul(32))
        .ok_or_else(|| invalid("posterior admission overflow"))?;
    let lease = scoped.acquire_admission(0, quote).map_err(admission)?;
    let operations = (parts.imputations as usize)
        .checked_mul(rec.atoms.len().saturating_add(1))
        .and_then(|v| v.checked_add(64))
        .ok_or_else(|| invalid("search size overflow"))?;
    let mut budget =
        SearchBudget::with_memory(SearchLimits { operations, depth: 1 }, MEMORY_LIMIT, &scoped)
            .map_err(|receipt| execution_refused(receipt.summary()))?;
    let retained: Vec<u64> = rec.atoms.iter().map(|atom| atom.operation.mask).collect();
    let mut pooled = Vec::new();
    let mut pooled_worlds = Vec::new();
    let mut unidentified_by_table = Vec::new();
    let mut table_leases = Vec::new();
    let mut paired_leases = Vec::new();
    if paired {
        let bytes = u64::from(parts.imputations).checked_mul(u64::from(parts.posterior_draws))
            .and_then(|n| n.checked_mul(16)).ok_or_else(|| invalid("paired world admission overflow"))?;
        paired_leases.push(scoped.acquire_admission(0, bytes).map_err(admission)?);
        if target.is_some() {
            paired_leases.push(scoped.acquire_admission(0, (parts.rows() as u64).saturating_mul(8))
                .map_err(admission)?);
        }
    }
    let mut receipt = CanonicalWriter::new(match target {
        Some(TransportTarget::Binary(_)) => "antecedent.graph_imputation_transport.execution.v1",
        Some(TransportTarget::ConfoundedBinary(_)) => "antecedent.graph_imputation_confounded_transport.execution.v1",
        Some(TransportTarget::Kernel(_)) => "antecedent.graph_imputation_kernel_transport.execution.v1",
        None => "antecedent.graph_candidate.execution.v1",
    });
    receipt.bytes(program.as_bytes()).bytes(&contract).word(u64::from(parts.imputations));
    if let Some(target) = target { receipt.bytes(&target.digest()?); }
    for index in 0..parts.imputations {
        let (data, completed_digest) = complete_table(rec, index, &scoped)?;
        let graphs = graph_imputation_exact_dags(
            &data,
            &rec.variables,
            treatment,
            outcome,
            parts.support,
            &retained,
            &mut budget,
        )
        .map_err(|error| match error {
            BoundedExactDagError::Search(receipt) => execution_refused(receipt.summary()),
            other => execution_refused(other),
        })?;
        let mut weights = BTreeMap::<u64, f64>::new();
        for (mask, &weight) in graphs.atoms().iter().zip(graphs.weights().iter()) {
            let mask =
                mask.to_word().ok_or_else(|| execution_refused("graph exceeds bounded mask"))?;
            *weights.entry(mask).or_default() += weight;
        }
        receipt.bytes(&completed_digest);
        let mut kept = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut effects = Vec::new();
        let mut world_atoms: Vec<(u64, f64, Vec<[f64; 2]>)> = Vec::new();
        let mut assumptions = AssumptionSet::new();
        let mut total = 0.0;
        let mut value_count: u64 = 0;
        for (mask, weight) in weights {
            if weight == 0.0 {
                continue;
            }
            let atom = rec
                .atoms
                .iter()
                .find(|atom| atom.operation.mask == mask)
                .ok_or_else(|| execution_refused("discovery returned an unsupported graph"))?;
            if scoped.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let mut child_ctx = scoped.clone();
            child_ctx.rng =
                RngFactory::from_seed(mix64(parts.execution_seed ^ (u64::from(index) << 8) ^ mask));
            let fitter = BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                likelihood: BayesLikelihood::GaussianIdentity,
                n_draws: parts.posterior_draws as usize,
                seed: child_ctx.rng.master_seed(),
                overlap: OverlapPolicy::ExplicitOverride,
                prior_scale: parts.prior_scale,
                prior: None,
            };
            let fit_error = |error| {
                    if child_ctx.cancellation.is_cancelled() {
                        cancelled()
                    } else {
                        execution_refused(format!("graph mask {mask}: {error}"))
                    }
                };
            let (prepared, target_weights) = if let Some(target) = target {
                let (query, registry, estimand, weights) =
                    transported_population(&data, &rec.query, atom, target, &parts.domains, paired)?;
                (fitter.prepare_for_population(&data, &estimand, &query, Some(&registry))
                    .map_err(fit_error)?, weights)
            } else {
                (fitter.prepare(&data, &atom.estimand, &rec.query).map_err(fit_error)?, None)
            };
            let fit = fitter.fit(&prepared, atom.operation.status,
                &mut BayesianGCompWorkspace::default(), &child_ctx).map_err(fit_error)?;
            if paired {
                if target.is_some_and(|target| !matches!(target, TransportTarget::Binary(_))) {
                    return Err(invalid("paired target worlds require the root-treatment binary route"));
                }
                let design = &prepared.design;
                let t = design.treatment_column().ok_or_else(|| invalid("paired worlds lack treatment column"))?;
                if target_weights.is_some()
                    && (t != 1 || design.ncols != 2 + 2 * prepared.adjustment_set.len())
                {
                    return Err(invalid("paired target coefficient design differs from centered adjustment model"));
                }
                let mut columns = Vec::with_capacity(design.ncols);
                for index in 0..design.ncols {
                    let column = fit.draws.schema.quantities.iter().position(|quantity|
                        matches!(quantity, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index))
                        .ok_or_else(|| invalid("paired worlds lack coefficient draws"))?;
                    columns.push(fit.draws.column(column).map_err(execution_refused)?);
                }
                let bytes = (fit.draws.n_draws as u64).checked_mul(16)
                    .ok_or_else(|| invalid("paired atom admission overflow"))?;
                paired_leases.push(scoped.acquire_admission(0, bytes).map_err(admission)?);
                let mut world_values = Vec::with_capacity(fit.draws.n_draws);
                for draw in 0..fit.draws.n_draws {
                    let mut mean = 0.0;
                    for col in 0..design.ncols {
                        let values = &design.matrix[col * design.nrows..(col + 1) * design.nrows];
                        let value = if col == t { 0.0 } else if let Some(weights) = &target_weights {
                            let names = prepared.coef_names.as_ref()
                                .ok_or_else(|| invalid("target worlds lack coefficient names"))?;
                            // Weighted g-computation appends treatment interactions centered at
                            // these very target weights. Their target mean is zero in both worlds.
                            let name = names.get(col).ok_or_else(|| invalid("target world coefficient name is missing"))?;
                            if col >= 2 + prepared.adjustment_set.len() {
                                if !name.starts_with("treatment_x_centered_") {
                                    return Err(invalid("paired target interaction column is not centered"));
                                }
                                0.0
                            } else {
                                let mass = weights.iter().sum::<f64>();
                                if weights.len() != design.nrows || !mass.is_finite() || mass <= 0.0 {
                                    return Err(invalid("paired target weights differ from design rows"));
                                }
                                values.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() / mass
                            }
                        } else { values.iter().sum::<f64>() / design.nrows as f64 };
                        mean += value * columns[col][draw];
                    }
                    let control = mean + prepared.control * columns[t][draw];
                    let active = mean + prepared.active * columns[t][draw];
                    if !control.is_finite() || !active.is_finite() { return Err(invalid("paired worlds are nonfinite")); }
                    world_values.push([control, active]);
                }
                world_atoms.push((mask, weight, world_values));
            }
            if !matches!(
                fit.identification,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            ) || fit.unidentified_mass != 0.0
                || fit.unevaluable_mass != 0.0
                || fit.subsampled_out_mass != 0.0
            {
                return Err(execution_refused(
                    "child returned unsupported identification or incomplete mass",
                ));
            }
            let column = fit
                .effect_column()
                .ok_or_else(|| execution_refused("atom returned no effect column"))?;
            let values = fit.draws.column(column).map_err(execution_refused)?;
            if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
                return Err(execution_refused("missing or nonfinite effect draws"));
            }
            let digest = child_receipt(mask, &completed_digest, values, &fit.assumptions);
            kept.push((mask, weight, digest));
            total += weight;
            value_count += values.len() as u64;
            flags.push(GraphIdentFlag::Identified);
            keys.push(mask);
            push_effect(&mut effects, mask, values);
            assumptions.extend_unique(fit.assumptions.entries.iter());
        }
        if (total - 1.0).abs() > 1e-12 {
            return Err(execution_refused("graph weights are not normalized"));
        }
        let mix_weights: Vec<f64> = kept.iter().map(|(_, weight, _)| *weight).collect();
        let bytes = value_count
            .checked_mul(32)
            .and_then(|v| v.checked_add((kept.len() as u64).saturating_mul(1024)))
            .and_then(|v| v.checked_add(4096))
            .ok_or_else(|| invalid("mixture admission overflow"))?;
        table_leases.push(scoped.acquire_admission(0, bytes).map_err(admission)?);
        let samples =
            WeightedGraphSamples::new(mix_weights, flags, keys).map_err(execution_refused)?;
        let mut mixed = aggregate_effect_envelope(
            &samples,
            &effects,
            InferenceDiagnostics::analytic("graph_imputation_label.bic_weighted_graph_mixture"),
            EnvelopeOptions::default(),
        )
        .map_err(execution_refused)?;
        mixed.assumptions.extend_unique(assumptions.entries.iter());
        if mixed.unidentified_mass > 0.0 {
            return Err(execution_refused("a completed table retained unidentified graph mass"));
        }
        unidentified_by_table.push(mixed.unidentified_mass);
        let column = mixed
            .effect_column()
            .ok_or_else(|| execution_refused("mixture returned no effect column"))?;
        let values = mixed.draws.column(column).map_err(execution_refused)?;
        if pooled.len() != values.len() * index as usize {
            return Err(execution_refused("completed tables returned different draw counts"));
        }
        pooled.extend_from_slice(values);
        if paired {
            // Draw one graph and one coefficient row jointly for both worlds.
            // Do not apply the scalar mixture's moment adjustment to either world.
            let mut rng = antecedent_core::CausalRng::from_seed(mix64(
                parts.execution_seed ^ (u64::from(index) << 32) ^ 0x574F_524C_4450_4149,
            ));
            for _ in 0..values.len() {
                let u = rng.next_f64();
                let mut cumulative = 0.0;
                let mut selected = world_atoms.len() - 1;
                for (i, (_, weight, _)) in world_atoms.iter().enumerate() {
                    cumulative += weight / total;
                    if u < cumulative { selected = i; break; }
                }
                let draws = &world_atoms[selected].2;
                let row = ((rng.next_f64() * draws.len() as f64) as usize).min(draws.len() - 1);
                pooled_worlds.push(draws[row]);
            }
        }
        receipt.word(kept.len() as u64);
        for (mask, weight, digest) in &kept {
            receipt.word(*mask).word(weight.to_bits()).bytes(digest);
        }
    }
    if pooled.len() < 2 || pooled.iter().any(|value| !value.is_finite()) {
        return Err(execution_refused("pooled draws must be finite"));
    }
    let count = pooled.len() as f64;
    let mean = pooled.iter().sum::<f64>() / count;
    let sd =
        (pooled.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (count - 1.0)).sqrt();
    receipt.word(pooled.len() as u64).word(mean.to_bits()).word(sd.to_bits());
    Ok(Execution {
        draws: pooled,
        paired_worlds: pooled_worlds,
        mean,
        sd,
        receipt: receipt.finish(),
        unidentified_by_table,
        lease,
        paired_leases,
    })
}

fn complete_table(
    rec: &Reconstructed,
    index: u32,
    ctx: &ExecutionContext,
) -> Result<(TabularData, [u8; 32]), IoError> {
    let completed = rec.imputer.impute(index, ctx).map_err(|error| {
        if ctx.cancellation.is_cancelled() { cancelled() } else { execution_refused(error) }
    })?;
    let replacements: Vec<_> = rec
        .variables
        .iter()
        .zip(completed.columns())
        .map(|(&id, column)| (id, Arc::clone(column)))
        .collect();
    let data = complete_float_columns(&rec.table, &replacements).map_err(execution_refused)?;
    let completed_digest = data.storage().content_digest_with_context(ctx).map_err(admission)?;
    Ok((data, completed_digest))
}

fn child_receipt(
    mask: u64,
    completed_digest: &[u8; 32],
    values: &[f64],
    assumptions: &AssumptionSet,
) -> [u8; 32] {
    let mut child = CanonicalWriter::new("antecedent.graph_candidate.child_execution.v1");
    child.word(mask).bytes(completed_digest).word(values.len() as u64);
    for value in values {
        child.word(value.to_bits());
    }
    child.word(assumptions.entries.len() as u64);
    for record in &assumptions.entries {
        record.assumption.encode(&mut child);
        record.scope.encode(&mut child);
        child.text(record.status.as_str());
        match &record.source {
            AssumptionSource::UserDeclared => {
                child.word(0);
            }
            AssumptionSource::AlgorithmDefault { algorithm } => {
                child.word(1).text(algorithm);
            }
            AssumptionSource::Artifact => {
                child.word(2);
            }
            AssumptionSource::Derived { from } => {
                child.word(3).text(from);
            }
        }
    }
    child.finish()
}

fn push_effect(effects: &mut Vec<GraphEffectDraws>, mask: u64, values: &[f64]) {
    effects.push(GraphEffectDraws { graph_key: mask, effect_draws: Arc::from(values) });
}

/// Which declared study a calibration basis describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateBasis {
    /// Effect drawn from a prior per replicate. The coordinate the measured route reports.
    PriorPredictive,
    /// One fixed effect, a pointwise study that no run of the route can match.
    FixedEffect,
}

/// The facts of the execution that make up the functional coordinate. Seeds are not part of it:
/// they do not change the procedure a record measured. The variable count and graph count are.
fn core_functional(parts: &GraphImputationParts) -> String {
    format!(
        "support={}:rule={}:variables={}:graphs={}:domains={}:m={}:sweeps={}:draws={}:prior_bits={:016x}",
        parts.support.as_str(),
        SUPPORT_RULE,
        parts.names.len(),
        parts.atoms.len(),
        parts.domains.iter().map(u8::to_string).collect::<String>(),
        parts.imputations,
        parts.iterations,
        parts.posterior_draws,
        parts.prior_scale.to_bits(),
    )
}

/// The functional coordinate the measured route reports for these parts.
#[must_use]
pub fn route_functional(parts: &GraphImputationParts) -> String {
    format!("{PRIOR_PREDICTIVE_FAMILY}graph_imputation_label.v3:{}", core_functional(parts))
}

fn basis_for(parts: &GraphImputationParts, kind: CandidateBasis) -> CalibrationBasis {
    let a = |s: &str| Arc::<str>::from(s);
    let (inference, functional) = match kind {
        CandidateBasis::PriorPredictive => ("GraphImputationCompositeV3", route_functional(parts)),
        CandidateBasis::FixedEffect => (
            "GraphImputationCompositeV2",
            format!("graph_imputation_label.v2:{}", core_functional(parts)),
        ),
    };
    CalibrationBasis::new(
        [
            a("AverageEffect"),
            a("Dag"),
            a("graph_posterior"),
            a("tabular"),
            a(inference),
            a("graph_imputation_label.bic_conjugate"),
            a("nearest_rank_95"),
            a(""),
            a("iid"),
            a("mar_bic_graph_mixture_conjugate"),
            Arc::from(functional),
        ],
        0.95,
        a("point"),
        parts.rows() as u64,
        None,
        Some(parts.posterior_draws * parts.imputations),
        0.0,
    )
    .with_validation("none")
}

struct Reading {
    source: GraphImputationSource,
    draws: Vec<f64>,
    point: f64,
    standard_deviation: f64,
    lower: f64,
    upper: f64,
    receipt: [u8; 32],
    unidentified_by_table: Vec<f64>,
    lease: AdmissionLease,
}

// Single native implementation, shared by the point replay, the calibration harness and the
// measured route.
fn execute_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<Reading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    let run = execute(&rec, None, false, ctx)?;
    let mut sorted = run.draws.clone();
    sorted.sort_by(f64::total_cmp);
    // Fixed nearest-rank 95% interval; a record must measure exactly this algorithm.
    let rank = |numerator: usize| (sorted.len() * numerator).div_ceil(1000).saturating_sub(1);
    let (lower, upper) = (sorted[rank(25)], sorted[rank(975)]);
    Ok(Reading {
        source: rec.source,
        draws: run.draws,
        point: run.mean,
        standard_deviation: run.sd,
        lower,
        upper,
        receipt: run.receipt,
        unidentified_by_table: run.unidentified_by_table,
        lease: run.lease,
    })
}

/// Joint source-population potential-outcome means from one graph, imputation,
/// and coefficient draw per row. Model-conditional; not predictive or calibrated.
pub struct GraphPairedWorldReading {
    /// `[control, active]` coordinates from the same fitted draw.
    pub worlds: Vec<[f64; 2]>,
    /// Source-bound paired execution receipt.
    pub receipt: [u8; 32],
    /// Retained producer program and source identities.
    pub program: [u8; 32],
    /// Original source data identity.
    pub source: [u8; 32],
    _lease: AdmissionLease,
    _paired_leases: Vec<AdmissionLease>,
}

/// Reconstruct and refit the entire graph/imputation schedule and sample paired
/// Gaussian mean worlds with a shared graph and coefficient selector.
///
/// # Errors
/// Changed source, unsupported child, numerical failure or incomplete schedule.
pub fn replay_paired_worlds(bytes: &[u8], ctx: &ExecutionContext) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    let run = execute(&rec, None, true, ctx)?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation.paired_worlds.execution.v1");
    writer.bytes(&run.receipt).word(run.paired_worlds.len() as u64);
    for pair in &run.paired_worlds { writer.word(pair[0].to_bits()).word(pair[1].to_bits()); }
    Ok(GraphPairedWorldReading {
        worlds: run.paired_worlds, receipt: writer.finish(), program: rec.source.program(),
        source: rec.source.parts().data_digest, _lease: run.lease, _paired_leases: run.paired_leases,
    })
}

/// A replayed point with no interval. Its dispersion is a model-conditional diagnostic.
pub struct GraphPointReading {
    /// Pooled posterior mean of the graph-mixture effect.
    pub mean: f64,
    /// Pooled posterior standard deviation; a diagnostic, not a standard error.
    pub diagnostic_sd: f64,
    /// Aligned mixture draws across every completed imputation and retained
    /// graph. They carry model-conditional posterior spread, not coverage.
    pub draws: Vec<f64>,
    /// Identity of the complete execution.
    pub receipt: [u8; 32],
    /// The compiler's lowered program identity the source carried.
    pub program: [u8; 32],
    /// Content identity of the original table.
    pub source: [u8; 32],
    /// Unidentified graph mass of each completed table.
    pub unidentified_by_table: Vec<f64>,
    _lease: AdmissionLease,
}

/// Reconstruct the source, check every retained child operation against its graph and execute.
///
/// # Errors
/// A malformed source, a retained operation that differs from the derived one, a failed child
/// fit, cancellation or a resource refusal.
pub fn replay_point(bytes: &[u8], ctx: &ExecutionContext) -> Result<GraphPointReading, IoError> {
    let reading = execute_source(bytes, ctx)?;
    let parts = reading.source.parts();
    Ok(GraphPointReading {
        mean: reading.point,
        diagnostic_sd: reading.standard_deviation,
        draws: reading.draws,
        receipt: reading.receipt,
        program: reading.source.program(),
        source: parts.data_digest,
        unidentified_by_table: reading.unidentified_by_table,
        _lease: reading.lease,
    })
}

/// Replay paired model-conditional potential-outcome mean worlds on a
/// representative binary target frame. Both worlds share each sampled graph,
/// imputed table and fitted coefficient draw; this is not outcome prediction.
///
/// # Errors
/// Invalid target, target support, imputation, model or resource refusal.
pub fn replay_transport_paired_worlds(bytes: &[u8], target: &TransportStandardization,
    ctx: &ExecutionContext) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes = (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Binary(target)), true, ctx)?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation_transport.paired_worlds.execution.v1");
    writer.bytes(&run.receipt).bytes(&target.digest()?).word(run.paired_worlds.len() as u64);
    for pair in &run.paired_worlds { writer.word(pair[0].to_bits()).word(pair[1].to_bits()); }
    Ok(GraphPairedWorldReading { worlds: run.paired_worlds, receipt: writer.finish(),
        program: rec.source.program(), source: rec.source.parts().data_digest,
        _lease: run.lease, _paired_leases: run.paired_leases })
}

/// Re-derive graph operations, impute the source, reselect graphs for every
/// completed table, fit each graph-specific outcome law and standardize every
/// child effect draw to the independent target covariate sample. Every
/// selected covariate enters the graph-specific outcome fit even if absent
/// from its minimal backdoor adjustment set. This is a new
/// model-conditional posterior route with no inherited interval calibration.
///
/// # Errors
/// Invalid target evidence, missing target support, graph or fit refusal,
/// cancellation or admission refusal.
pub fn replay_transport_point(
    bytes: &[u8], target: &TransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes = (target.rows() as u64)
        .saturating_mul(target.columns.len() as u64)
        .saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Binary(target)), false, ctx)?;
    Ok(GraphPointReading {
        mean: run.mean,
        diagnostic_sd: run.sd,
        draws: run.draws,
        receipt: run.receipt,
        program: rec.source.program(),
        source: rec.source.parts().data_digest,
        unidentified_by_table: run.unidentified_by_table,
        _lease: run.lease,
    })
}

/// Reidentify every confounded-pretreatment atom and verify that adding the
/// selection targets preserves its backdoor criterion. The target is an
/// empirical binary frame; both source treatment arms must occur in each
/// target-supported adjustment stratum on every completed table.
///
/// # Errors
/// Unsupported graph, unsafe augmentation, missing overlap, fitting or admission.
pub fn replay_confounded_transport_point(
    bytes: &[u8], target: &TransportStandardization, ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate_confounded(rec.source.parts())?;
    let target_bytes = (target.rows() as u64)
        .saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::ConfoundedBinary(target)), false, ctx)?;
    Ok(GraphPointReading { mean: run.mean, diagnostic_sd: run.sd, draws: run.draws,
        receipt: run.receipt, program: rec.source.program(),
        source: rec.source.parts().data_digest, unidentified_by_table: run.unidentified_by_table,
        _lease: run.lease })
}

/// Refit the same graph and imputation schedule while standardizing each
/// retained child to a mixed binary-continuous target frame with prespecified
/// kernel density-ratio weights. The target frame is empirical and intervals
/// remain unmeasured.
///
/// # Errors
/// Invalid target evidence, insufficient kernel overlap, graph, fit,
/// cancellation or admission refusal.
pub fn replay_kernel_transport_point(
    bytes: &[u8], target: &KernelTransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes = (target.rows() as u64)
        .saturating_mul(target.columns.len() as u64)
        .saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Kernel(target)), false, ctx)?;
    Ok(GraphPointReading {
        mean: run.mean, diagnostic_sd: run.sd, draws: run.draws,
        receipt: run.receipt, program: rec.source.program(),
        source: rec.source.parts().data_digest,
        unidentified_by_table: run.unidentified_by_table,
        _lease: run.lease,
    })
}

/// Source-bound, independently replayed measurement. It is calibrated only when a current
/// attesting record covers the reported construction and the execution's row count.
///
/// # Errors
/// The source is invalid, execution is cancelled or over budget, or no attesting record covers
/// this execution.
pub fn measure_source(bytes: &[u8], ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
    let reading = execute_source(bytes, ctx)?;
    let parts = reading.source.parts();
    let program = SemanticDigest::from_bytes(reading.source.program());
    let contract = SemanticDigest::from_bytes(parts.contract_digest());
    let premises = crate::digest_hex(
        IdentityDomain::Claim,
        &(
            "graph_imputation_interval.v1",
            contract.to_hex(),
            program.to_hex(),
            "one_row_per_declared_independent_subject",
        ),
    )?;
    let data = SemanticDigest::from_bytes(parts.data_digest).to_hex();
    let scope = serde_json::json!({
        "evidence_interpretation": "prior-predictive interval of a graph-weighted model mixture under the stated declarations; not pointwise frequentist coverage",
        "declared_model_assumptions": [
            "independent rows",
            "linear Gaussian outcome model",
            "missing at random given the observed variables",
            "declared graph support, not learned from data",
            "graphs with several minimal back-door adjustment sets are outside the support",
            "BIC graph weights and conjugate child posteriors",
        ],
        "graph_support": parts.support.as_str(),
        "support_rule": SUPPORT_RULE,
        "excluded_graphs": parts.excluded,
        "variables": parts.names.len(),
        "graphs": parts.atoms.len(),
        "rows": parts.rows(),
        "reported_scalar": "treatment_effect",
        "contract": contract.to_hex(),
        "compiled_program": program.to_hex(),
    });
    authorize(
        "graph_imputation_interval",
        bytes.to_vec(),
        &premises,
        &data,
        scope,
        vec![ScalarExecution {
            name: "treatment_effect".to_owned(),
            point: reading.point,
            interval: (reading.lower, reading.upper),
            basis: basis_for(parts, CandidateBasis::PriorPredictive),
        }],
    )
}

/// Reconsume a measured envelope by replaying its original source and retained operations. The
/// expected identity is separately retained; it is never evidence or authority by itself.
///
/// # Errors
/// Any source, operation, result, evidence or expected identity mismatch.
pub(crate) fn consume_measured(
    bytes: &[u8],
    expected: &MeasuredExpectation,
    ctx: &ExecutionContext,
) -> Result<MeasuredInference, IoError> {
    let wire = decode(bytes, expected)?;
    let measured = measure_source(&wire.source, ctx)?;
    compare_replay(&wire, &measured)?;
    Ok(measured)
}

/// Feature-gated study-only observation of the same private execution. It carries no
/// authority and cannot be compiled into normal builds.
///
/// # Errors
/// Invalid source, cancellation, or resource refusal.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn calibration_candidate(
    bytes: &[u8],
    kind: CandidateBasis,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, CalibrationBasis), IoError> {
    let reading = execute_source(bytes, ctx)?;
    let basis = basis_for(reading.source.parts(), kind);
    Ok((reading.lower, reading.upper, basis))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_io_core::coverage_records_data::CoverageRecord;

    struct Lcg(u64);
    impl Lcg {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (a, b) = (self.uniform().max(1e-12), self.uniform());
            (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
        }
    }

    // Treatment is driven by the covariate, so the covariate confounds.
    fn parts(support: GraphSupport, variables: usize, rows: usize) -> GraphImputationParts {
        let mut rng = Lcg(11);
        let mut t = Vec::new();
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut z = Vec::new();
        for _ in 0..rows {
            let covariate = rng.normal();
            let treated = f64::from(rng.uniform() < 1.0 / (1.0 + (-covariate).exp()));
            let outcome = 0.5 * treated + 0.8 * covariate + rng.normal();
            t.push(treated);
            x.push(covariate);
            z.push(rng.normal());
            y.push(if rng.uniform() < 0.8 { outcome } else { f64::NAN });
        }
        let mut names: Vec<String> = vec!["t".into(), "y".into(), "x".into(), "z".into()];
        let mut columns = vec![t, y, x, z];
        let mut domains = vec![1u8, 0, 0, 0];
        names.truncate(variables);
        columns.truncate(variables);
        domains.truncate(variables);
        let table = TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .unwrap();
        let data_digest =
            table.storage().content_digest_with_context(&ExecutionContext::for_tests(3)).unwrap();
        let mut parts = GraphImputationParts {
            names,
            domains,
            columns,
            treatment: 0,
            outcome: 1,
            support,
            imputations: 2,
            iterations: 2,
            imputation_seed: 17,
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
            data_digest,
            atoms: Vec::new(),
            excluded: 0,
        };
        let retained = derive_support(&parts).unwrap();
        parts.atoms = retained.operations;
        parts.excluded = retained.excluded;
        parts
    }

    fn encoded(parts: GraphImputationParts) -> Vec<u8> {
        GraphImputationSource::new(parts, [9; 32]).unwrap().encode()
    }

    #[test]
    #[ignore = "source-only paired graph and imputation worlds; calibration not earned"]
    fn paired_worlds_share_each_graph_and_coefficient_draw() {
        let bytes = encoded(parts(GraphSupport::TreatmentRootPreTreatmentCovariates, 3, 100));
        let ctx = ExecutionContext::for_tests(19);
        let first = replay_paired_worlds(&bytes, &ctx).unwrap();
        let second = replay_paired_worlds(&bytes, &ctx).unwrap();
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.worlds, second.worlds);
        assert_eq!(first.worlds.len(), 200);
        assert!(first.worlds.iter().all(|pair| pair[0].is_finite() && pair[1].is_finite()));
        assert!(first.worlds.windows(2).any(|pair| pair[0] != pair[1]));
    }

    // An attesting record keyed exactly as the route's own basis is, covering a row range.
    fn record(functional: String, n_min: u64, n_max: u64) -> CoverageRecord {
        CoverageRecord {
            id: "cov.average_effect.dag.graphimputationcompositev3.nearest_rank_95.l95.fixture.v2_validation_none.v2_structure_graph_posterior",
            validation_schema: 2,
            validation: "none",
            query: "AverageEffect",
            graph_class: "Dag",
            structure: "graph_posterior",
            modality: "tabular",
            inference: "GraphImputationCompositeV3",
            estimator: "graph_imputation_label.bic_conjugate",
            interval_method: "nearest_rank_95",
            se_kind: "",
            dependence: "iid",
            posterior: "mar_bic_graph_mixture_conjugate",
            functional: Box::leak(functional.into_boxed_str()),
            identification: "point",
            nominal: 0.95,
            n_min,
            n_max,
            replicates_min: 0,
            posterior_draws_min: 0,
            unidentified_mass_max: 0.0,
            observed: 0.95,
            mcse: 0.011,
            replicates: 400,
            boundary: false,
            grid: &[],
            confirmation: &[],
            confirmed_passing_range: None,
            dgp: "crates/antecedent/tests/x.rs::dgp",
            test: "crates/antecedent/tests/x.rs::test",
            calibration_sha: "0123456789abcdef0123456789abcdef01234567",
        }
    }

    fn refusal(error: IoError) -> (String, String) {
        match error {
            IoError::Refused { code, message } => (code.to_string(), message),
            other => panic!("unexpected refusal: {other}"),
        }
    }

    #[test]
    fn confounded_support_has_differing_adjustment_sets_and_root_support_has_one() {
        let sets = |parts: &GraphImputationParts| -> std::collections::BTreeSet<Vec<u32>> {
            parts.atoms.iter().map(|atom| atom.adjustment.clone()).collect()
        };
        let root = parts(GraphSupport::TreatmentRootDirectEffect, 3, 30);
        assert_eq!(root.atoms.len(), 6);
        assert_eq!(sets(&root), std::collections::BTreeSet::from([vec![]]));
        let direct = parts(GraphSupport::TreatmentDirectEffect, 3, 30);
        assert_eq!(direct.atoms.len(), 8);
        assert_eq!(sets(&direct), std::collections::BTreeSet::from([vec![], vec![2]]));
        let four = parts(GraphSupport::TreatmentDirectEffect, 4, 30);
        assert!(four.atoms.len() > direct.atoms.len());
        assert!(sets(&four).len() > 2);
    }

    #[test]
    fn a_graph_with_two_minimal_sets_is_excluded_and_the_count_is_exact() {
        let direct = GraphSupport::TreatmentDirectEffect;
        let four = parts(direct, 4, 30);
        let structural = graph_imputation_supported_masks(4, 0, 1, direct);
        assert!(four.excluded > 0, "four variables admit graphs with two minimal sets");
        assert_eq!(four.atoms.len() + four.excluded as usize, structural.len());
        // Mask 897 (T=0, Y=1) has {X2} and {X3} as minimal sets and is not retained.
        assert!(structural.contains(&897));
        assert!(four.atoms.iter().all(|atom| atom.mask != 897));
        // Unique sets exclude nothing: the three-variable counts and the root support are whole.
        assert_eq!(parts(direct, 3, 30).excluded, 0);
        assert_eq!(parts(GraphSupport::TreatmentRootDirectEffect, 4, 30).excluded, 0);
        assert_eq!(parts(GraphSupport::TreatmentRootDirectEffect, 4, 30).atoms.len(), 100);
    }

    #[test]
    fn a_forged_exclusion_count_or_graph_list_is_refused() {
        let ctx = ExecutionContext::for_tests(7);
        let source = parts(GraphSupport::TreatmentDirectEffect, 4, 40);
        let reading = replay_point(&encoded(source.clone()), &ctx).unwrap();
        assert!(reading.mean.is_finite());
        let mut forged = source.clone();
        forged.excluded += 1;
        let (code, message) = refusal(replay_point_error(&encoded(forged), &ctx));
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("excluded graph count differs"), "{message}");
        let mut shortened = source;
        shortened.atoms.pop();
        let (_, message) = refusal(replay_point_error(&encoded(shortened), &ctx));
        assert!(message.contains("retained child operation differs"), "{message}");
    }

    #[test]
    fn untrusted_source_is_refused_before_any_execution() {
        let (code, message) =
            refusal(measure_source(b"untrusted", &ExecutionContext::for_tests(7)).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(
            message.contains(antecedent_core::reason_details::graph_imputation::INVALID_SOURCE)
        );
    }

    #[test]
    fn a_retained_operation_that_differs_from_its_graph_is_refused() {
        let ctx = ExecutionContext::for_tests(7);
        let mut tampered = parts(GraphSupport::TreatmentDirectEffect, 3, 40);
        let confounded =
            tampered.atoms.iter().position(|atom| !atom.adjustment.is_empty()).unwrap();
        tampered.atoms[confounded].adjustment.clear();
        let (code, message) = refusal(replay_point_error(&encoded(tampered), &ctx));
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("retained child operation differs"));
        let mut changed_data = parts(GraphSupport::TreatmentDirectEffect, 3, 40);
        changed_data.columns[2][0] += 1.0;
        assert!(replay_point(&encoded(changed_data), &ctx).is_err());
    }

    fn replay_point_error(bytes: &[u8], ctx: &ExecutionContext) -> IoError {
        match replay_point(bytes, ctx) {
            Ok(_) => panic!("expected a refusal"),
            Err(error) => error,
        }
    }

    #[test]
    fn point_replay_is_deterministic_and_finite() {
        let ctx = ExecutionContext::for_tests(7);
        let bytes = encoded(parts(GraphSupport::TreatmentDirectEffect, 3, 60));
        let first = replay_point(&bytes, &ctx).unwrap();
        let second = replay_point(&bytes, &ctx).unwrap();
        assert_eq!(first.receipt, second.receipt);
        assert!(first.mean.is_finite() && first.diagnostic_sd > 0.0);
        assert!(first.unidentified_by_table.iter().all(|mass| *mass == 0.0));
    }

    #[test]
    fn without_a_matching_record_the_interval_is_reported_not_calibrated() {
        let ctx = ExecutionContext::for_tests(7);
        let source = parts(GraphSupport::TreatmentDirectEffect, 3, 60);
        let functional = route_functional(&source);
        let bytes = encoded(source);
        let measured =
            super::super::with_test_records(vec![], || measure_source(&bytes, &ctx)).unwrap();
        let scalar = &measured.report().scalars[0];
        assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
        assert_ne!(scalar.calibration.status, "calibrated");
        // Another graph size, another support or a row range that excludes this run does not cover it.
        let other_size = route_functional(&parts(GraphSupport::TreatmentDirectEffect, 4, 60));
        let other_support =
            route_functional(&parts(GraphSupport::TreatmentRootDirectEffect, 3, 60));
        for wrong in [
            record(other_size, 30, 4096),
            record(other_support, 30, 4096),
            record(functional.clone(), 61, 4096),
            record(functional, 30, 59),
        ] {
            let measured =
                super::super::with_test_records(vec![wrong], || measure_source(&bytes, &ctx))
                    .unwrap();
            assert_ne!(measured.report().scalars[0].calibration.status, "calibrated");
        }
    }

    #[test]
    fn a_covering_attesting_record_supports_the_route_and_it_replays() {
        let ctx = ExecutionContext::for_tests(7);
        for (support, variables) in [
            (GraphSupport::TreatmentRootDirectEffect, 3),
            (GraphSupport::TreatmentDirectEffect, 3),
            (GraphSupport::TreatmentDirectEffect, 4),
        ] {
            let source = parts(support, variables, 60);
            let functional = route_functional(&source);
            let bytes = encoded(source);
            let measured =
                super::super::with_test_records(vec![record(functional.clone(), 30, 4096)], || {
                    measure_source(&bytes, &ctx)
                })
                .unwrap();
            let scalar = &measured.report().scalars[0];
            assert_eq!(scalar.name, "treatment_effect");
            assert_eq!(scalar.calibration.status, "calibrated");
            assert!(scalar.lower < scalar.upper);
            let basis = scalar.calibration.basis.as_ref().unwrap();
            assert_eq!(basis.scope.row_count, 60);
            assert_eq!(basis.key.functional, functional);
            let again = super::super::with_test_records(vec![record(functional, 30, 4096)], || {
                consume_measured(measured.export(), &measured.report().identity, &ctx)
            })
            .unwrap();
            assert_eq!(again.report().identity.seal, measured.report().identity.seal);
        }
    }

    #[test]
    fn a_record_never_extrapolates_beyond_its_measured_rows() {
        let ctx = ExecutionContext::for_tests(7);
        let source = parts(GraphSupport::TreatmentDirectEffect, 3, 60);
        let functional = route_functional(&source);
        let bytes = encoded(source);
        let outside =
            super::super::with_test_records(vec![record(functional.clone(), 30, 59)], || {
                measure_source(&bytes, &ctx)
            });
        assert_ne!(outside.unwrap().report().scalars[0].calibration.status, "calibrated");
        let inside = super::super::with_test_records(vec![record(functional, 30, 60)], || {
            measure_source(&bytes, &ctx)
        });
        assert_eq!(inside.unwrap().report().scalars[0].calibration.status, "calibrated");
    }
}
