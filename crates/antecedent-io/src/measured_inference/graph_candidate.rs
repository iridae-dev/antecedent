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
    BayesLikelihood, GraphIdentFlag, InferenceDiagnostics, PosteriorQuantityKind,
    WeightedGraphSamples,
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

/// Raw independent target sampling frame for a future weighted transport route.
/// Missing cells are represented by `None`; observation probabilities describe
/// the probability that the entire covariate vector is observed, conditional on
/// the declared always-observed columns. They are assumptions, not estimates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawTargetFrame {
    /// Covariate columns in ascending source-schema order.
    pub columns: Vec<(u32, Vec<Option<f64>>)>,
    /// Unique identities in original row order.
    pub row_ids: Vec<String>,
    /// Independent cluster identity for each row; repeated within a cluster.
    pub cluster_ids: Vec<String>,
    /// Sampling weights in original row order.
    pub weights: Vec<f64>,
    /// Positive probabilities of observing the complete vector, one per row.
    pub observation_probabilities: Vec<f64>,
    /// Columns on which the declared missing-at-random mechanism is conditional.
    pub always_observed: Vec<u32>,
}

impl RawTargetFrame {
    /// Check row alignment, distinct source and target units, declared domains,
    /// positive weights and complete-vector observation probabilities.
    ///
    /// # Errors
    /// Invalid identity, alignment, missingness declaration or covariate value.
    #[allow(clippy::float_cmp, reason = "coded binary levels are compared exactly")]
    pub fn validate(
        &self,
        parts: &GraphImputationParts,
        source_ids: &[String],
    ) -> Result<(), IoError> {
        let rows = self.row_ids.len();
        if rows < 2
            || self.columns.is_empty()
            || source_ids.len() != parts.rows()
            || self.cluster_ids.len() != rows
            || self.weights.len() != rows
            || self.observation_probabilities.len() != rows
            || self.columns.iter().any(|(_, values)| values.len() != rows)
        {
            return Err(invalid("raw target frame columns and identities must align"));
        }
        let mut units = BTreeSet::new();
        if source_ids
            .iter()
            .chain(&self.row_ids)
            .any(|id| id.trim().is_empty() || !units.insert(id.as_str()))
            || self.cluster_ids.iter().any(|id| id.trim().is_empty())
            || self.weights.iter().any(|weight| !weight.is_finite() || *weight <= 0.0)
            || self
                .observation_probabilities
                .iter()
                .any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
            || self
                .weights
                .iter()
                .zip(&self.observation_probabilities)
                .any(|(weight, probability)| !(weight / probability).is_finite())
        {
            return Err(invalid(
                "raw target identities, weights or observation probabilities are invalid",
            ));
        }
        let mut previous = None;
        for (id, values) in &self.columns {
            if previous.is_some_and(|last| last >= *id)
                || *id as usize >= parts.names.len()
                || *id == parts.treatment
                || *id == parts.outcome
                || values.iter().flatten().any(|value| {
                    !value.is_finite()
                        || (parts.domains[*id as usize] == 1 && *value != 0.0 && *value != 1.0)
                })
            {
                return Err(invalid(
                    "raw target columns must be ordered pretreatment covariates in their domains",
                ));
            }
            previous = Some(*id);
        }
        let mut previous = None;
        for id in &self.always_observed {
            if previous.is_some_and(|last| last >= *id)
                || !self
                    .columns
                    .iter()
                    .any(|(column, values)| column == id && values.iter().all(Option::is_some))
            {
                return Err(invalid("MAR conditioning columns must be unique and always observed"));
            }
            previous = Some(*id);
        }
        // Equal observed conditioning values must imply equal declared
        // probabilities; this checks the declared functional dependence, not MAR itself.
        let mut probabilities = BTreeMap::new();
        for row in 0..rows {
            let key: Vec<u64> = self
                .always_observed
                .iter()
                .map(|id| {
                    let value = self
                        .columns
                        .iter()
                        .find(|(column, _)| column == id)
                        .and_then(|(_, values)| values[row])
                        .expect("validated always-observed column");
                    if value == 0.0 { 0 } else { value.to_bits() }
                })
                .collect();
            let probability = self.observation_probabilities[row].to_bits();
            if probabilities.insert(key, probability).is_some_and(|prior| prior != probability) {
                return Err(invalid(
                    "observation probabilities must depend only on always-observed columns",
                ));
            }
        }
        if (0..rows)
            .filter(|&row| self.columns.iter().all(|(_, values)| values[row].is_some()))
            .count()
            < 2
        {
            return Err(invalid("raw target frame needs at least two complete rows"));
        }
        Ok(())
    }

    /// Bind every raw row, identity, sampling weight and missingness premise.
    ///
    /// # Errors
    /// Encoding refusal.
    pub fn digest(&self) -> Result<[u8; 32], IoError> {
        Ok(CanonicalWriter::digest_bytes(
            "antecedent.graph_imputation.raw_target_frame.v1",
            &crate::to_cbor(self)?,
        ))
    }

    /// Return complete row indices and their unnormalized inverse-observation
    /// weights. Call only after validating the frame; consumers must normalize
    /// these weights in their target integration, including in each replicate.
    #[must_use]
    pub fn complete_rows(&self) -> Vec<(usize, f64)> {
        (0..self.row_ids.len())
            .filter(|&row| self.columns.iter().all(|(_, values)| values[row].is_some()))
            .map(|row| (row, self.weights[row] / self.observation_probabilities[row]))
            .collect()
    }
}

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
        self.validate_for_support(parts, GraphSupport::TreatmentRootPreTreatmentCovariates)?;
        validate_target_selection_admissibility(parts, &self.selection_targets)
    }

    /// Validate a separate confounded-pretreatment target, including the
    /// augmented adjustment set in every retained graph.
    ///
    /// # Errors
    /// Unsupported rows, graph support or an opened backdoor path.
    pub fn validate_confounded(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        self.validate_for_support(parts, GraphSupport::PreTreatmentCovariates)?;
        validate_confounded_adjustment(parts, &self.selection_targets)
    }

    #[allow(clippy::float_cmp, reason = "coded binary levels are compared exactly")]
    fn validate_for_support(
        &self,
        parts: &GraphImputationParts,
        support: GraphSupport,
    ) -> Result<(), IoError> {
        if self.source_population.trim().is_empty()
            || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
            || self.source_sample.trim().is_empty()
            || self.target_sample.trim().is_empty()
            || self.source_sample == self.target_sample
            || self.columns.is_empty()
            || self.selection_targets.is_empty()
        {
            return Err(invalid(
                "transport needs distinct source and target samples and selected covariates",
            ));
        }
        if parts.support != support {
            return Err(invalid(if support == GraphSupport::TreatmentRootPreTreatmentCovariates {
                "target standardization requires the root-treatment pretreatment graph support"
            } else {
                "target standardization requires the confounded pretreatment graph support"
            }));
        }
        let rows = self.columns[0].1.len();
        if rows < 2 || self.columns.iter().any(|(_, values)| values.len() != rows) {
            return Err(invalid("target covariate columns need aligned representative rows"));
        }
        if self.source_unit_ids.len() != parts.rows() || self.target_unit_ids.len() != rows {
            return Err(invalid("source and target unit identities must align with their rows"));
        }
        let mut units = BTreeSet::new();
        if self
            .source_unit_ids
            .iter()
            .chain(&self.target_unit_ids)
            .any(|unit| unit.trim().is_empty() || !units.insert(unit.as_str()))
        {
            return Err(invalid("source and target units must have distinct nonblank identities"));
        }
        let mut previous = None;
        for (id, values) in &self.columns {
            if previous.is_some_and(|last| last >= *id)
                || *id as usize >= parts.names.len()
                || *id == parts.treatment
                || *id == parts.outcome
                || parts.domains[*id as usize] != 1
                || values.iter().any(|value| *value != 0.0 && *value != 1.0)
            {
                return Err(invalid(
                    "target columns must be unique binary pretreatment covariates",
                ));
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
            return Err(invalid(
                "every retained graph needs binary target data for its adjustment set",
            ));
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
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |column| column.1.len())
    }
}

fn validate_confounded_adjustment(
    parts: &GraphImputationParts,
    selected: &[u32],
) -> Result<(), IoError> {
    let n = parts.names.len();
    for atom in &parts.atoms {
        let dag =
            dag_from_adjacency_mask(AdjacencyMask::from_word(atom.mask), n).map_err(invalid)?;
        let treatment = antecedent_graph::DenseNodeId::from_raw(parts.treatment);
        let outcome = antecedent_graph::DenseNodeId::from_raw(parts.outcome);
        let mut adjustment = atom.adjustment.clone();
        for &id in selected {
            if !adjustment.contains(&id) {
                adjustment.push(id);
            }
        }
        let adjustment: Vec<_> =
            adjustment.into_iter().map(antecedent_graph::DenseNodeId::from_raw).collect();
        let overlay = antecedent_graph::overlay::GraphOverlay::remove_outgoing(n, &[treatment]);
        let separated = dag
            .view(&overlay)
            .is_d_separated(
                treatment,
                outcome,
                &adjustment,
                &mut antecedent_graph::DSeparationWorkspace::default(),
            )
            .map_err(invalid)?;
        if !separated {
            return Err(invalid("selected covariates open a backdoor path in a retained graph"));
        }
        validate_selection_admissibility(&dag, treatment, outcome, &adjustment, selected)?;
    }
    Ok(())
}

fn validate_target_selection_admissibility(
    parts: &GraphImputationParts,
    selected: &[u32],
) -> Result<(), IoError> {
    let n = parts.names.len();
    for atom in &parts.atoms {
        let dag =
            dag_from_adjacency_mask(AdjacencyMask::from_word(atom.mask), n).map_err(invalid)?;
        let mut adjustment = atom.adjustment.clone();
        for &id in selected {
            if !adjustment.contains(&id) {
                adjustment.push(id);
            }
        }
        let adjustment: Vec<_> =
            adjustment.into_iter().map(antecedent_graph::DenseNodeId::from_raw).collect();
        validate_selection_admissibility(
            &dag,
            antecedent_graph::DenseNodeId::from_raw(parts.treatment),
            antecedent_graph::DenseNodeId::from_raw(parts.outcome),
            &adjustment,
            selected,
        )?;
    }
    Ok(())
}

// Add one exogenous selection node per checked mechanism, intervene on
// treatment, then check whether the selection node and outcome are separated
// conditional on treatment and the exact augmented standardization set.
fn validate_selection_admissibility(
    dag: &antecedent_graph::Dag,
    treatment: antecedent_graph::DenseNodeId,
    outcome: antecedent_graph::DenseNodeId,
    adjustment: &[antecedent_graph::DenseNodeId],
    selected: &[u32],
) -> Result<(), IoError> {
    let n = dag.node_count();
    let expanded_len = n
        .checked_add(selected.len())
        .ok_or_else(|| invalid("selection diagram node count overflow"))?;
    let count = u32::try_from(expanded_len).map_err(invalid)?;
    let mut expanded = antecedent_graph::Dag::with_variables(count);
    for (parent, child) in dag.directed_edge_pairs_sorted() {
        expanded
            .insert_directed(
                antecedent_graph::DenseNodeId::from_raw(parent),
                antecedent_graph::DenseNodeId::from_raw(child),
            )
            .map_err(invalid)?;
    }
    let mut selections = Vec::with_capacity(selected.len());
    for (offset, &id) in selected.iter().enumerate() {
        let selection =
            antecedent_graph::DenseNodeId::from_raw(u32::try_from(n + offset).map_err(invalid)?);
        expanded
            .insert_directed(selection, antecedent_graph::DenseNodeId::from_raw(id))
            .map_err(invalid)?;
        selections.push(selection);
    }
    let overlay =
        antecedent_graph::overlay::GraphOverlay::do_intervention(expanded_len, &[treatment]);
    let mut conditioned = adjustment.to_vec();
    conditioned.push(treatment);
    for selection in selections {
        if !expanded
            .view(&overlay)
            .is_d_separated(
                selection,
                outcome,
                &conditioned,
                &mut antecedent_graph::DSeparationWorkspace::default(),
            )
            .map_err(invalid)?
        {
            return Err(invalid("selected mechanisms are not S-admissible in a retained graph"));
        }
    }
    Ok(())
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
        self.validate_for_support(parts, GraphSupport::TreatmentRootPreTreatmentCovariates)
    }

    /// Validate the mixed target under pretreatment confounding, including the
    /// complete augmented backdoor and selection-diagram checks for every atom.
    ///
    /// # Errors
    /// Invalid target, confounding adjustment or selection admissibility.
    pub fn validate_confounded(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        self.validate_for_support(parts, GraphSupport::PreTreatmentCovariates)?;
        if self.min_kernel_overlap <= 0.0 {
            return Err(invalid("confounded kernel overlap floor must be strictly positive"));
        }
        validate_confounded_adjustment(parts, &self.selection_targets)
    }

    #[allow(clippy::float_cmp, reason = "coded binary levels are compared exactly")]
    fn validate_for_support(
        &self,
        parts: &GraphImputationParts,
        support: GraphSupport,
    ) -> Result<(), IoError> {
        if self.source_population.trim().is_empty()
            || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
            || self.source_sample.trim().is_empty()
            || self.target_sample.trim().is_empty()
            || self.source_sample == self.target_sample
            || self.columns.is_empty()
            || self.selection_targets.is_empty()
            || parts.support != support
            || !self.min_kernel_overlap.is_finite()
            || !(0.0..1.0).contains(&self.min_kernel_overlap)
        {
            return Err(invalid(
                "kernel transport needs distinct samples, pretreatment support and a bounded overlap floor",
            ));
        }
        let rows = self.rows();
        if rows < 2
            || self.columns.iter().any(|(_, values)| values.len() != rows)
            || self.source_unit_ids.len() != parts.rows()
            || self.target_unit_ids.len() != rows
            || parts
                .rows()
                .checked_mul(rows.saturating_add(parts.rows()))
                .and_then(|pairs| pairs.checked_mul(parts.atoms.len()))
                .and_then(|pairs| pairs.checked_mul(parts.imputations as usize))
                .is_none_or(|pairs| pairs > 50_000_000)
        {
            return Err(invalid("kernel target columns and unit identities must align"));
        }
        let mut units = BTreeSet::new();
        if self
            .source_unit_ids
            .iter()
            .chain(&self.target_unit_ids)
            .any(|unit| unit.trim().is_empty() || !units.insert(unit.as_str()))
        {
            return Err(invalid("source and target units must have distinct nonblank identities"));
        }
        let mut previous = None;
        for (id, values) in &self.columns {
            if previous.is_some_and(|last| last >= *id)
                || *id as usize >= parts.names.len()
                || *id == parts.treatment
                || *id == parts.outcome
                || values.iter().any(|value| !value.is_finite())
                || (parts.domains[*id as usize] == 1
                    && values.iter().any(|value| *value != 0.0 && *value != 1.0))
            {
                return Err(invalid(
                    "kernel target columns must be unique complete pretreatment covariates in their domains",
                ));
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
            atom.adjustment.iter().any(|id| !self.columns.iter().any(|(column, _)| column == id))
        }) {
            return Err(invalid("every graph adjustment needs a complete target column"));
        }
        let continuous: Vec<u32> = self
            .columns
            .iter()
            .filter_map(|(id, _)| (parts.domains[*id as usize] == 0).then_some(*id))
            .collect();
        if continuous.is_empty()
            || !self.selection_targets.iter().any(|id| continuous.contains(id))
            || self.bandwidths.len() != continuous.len()
            || self.bandwidths.iter().zip(continuous).any(|(&(id, bandwidth), expected)| {
                id != expected || !bandwidth.is_finite() || bandwidth <= 0.0
            })
        {
            return Err(invalid("each continuous target column needs one positive bandwidth"));
        }
        validate_target_selection_admissibility(parts, &self.selection_targets)?;
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
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |column| column.1.len())
    }
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
    /// Bernoulli predictive routes only: each paired world's target success
    /// probabilities, aligned to `paired_worlds`.
    paired_probabilities: Vec<[f64; 2]>,
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
    ConfoundedKernel(&'a KernelTransportStandardization),
    MarBinary(&'a TransportStandardization, &'a [f64], [u8; 32], bool),
    MarKernel(&'a KernelTransportStandardization, &'a [f64], [u8; 32], bool),
}

impl<'a> TransportTarget<'a> {
    fn columns(self) -> &'a [(u32, Vec<f64>)] {
        match self {
            Self::Binary(value)
            | Self::ConfoundedBinary(value)
            | Self::MarBinary(value, _, _, _) => &value.columns,
            Self::Kernel(value)
            | Self::ConfoundedKernel(value)
            | Self::MarKernel(value, _, _, _) => &value.columns,
        }
    }

    fn selections(self) -> &'a [u32] {
        match self {
            Self::Binary(value)
            | Self::ConfoundedBinary(value)
            | Self::MarBinary(value, _, _, _) => &value.selection_targets,
            Self::Kernel(value)
            | Self::ConfoundedKernel(value)
            | Self::MarKernel(value, _, _, _) => &value.selection_targets,
        }
    }

    fn digest(self) -> Result<[u8; 32], IoError> {
        match self {
            Self::Binary(value) | Self::ConfoundedBinary(value) => value.digest(),
            Self::Kernel(value) | Self::ConfoundedKernel(value) => value.digest(),
            Self::MarBinary(_, _, digest, _) | Self::MarKernel(_, _, digest, _) => Ok(digest),
        }
    }
}

#[allow(clippy::float_cmp, reason = "coded binary levels and treatment arms are compared exactly")]
fn kernel_transport_weights(
    source_columns: &[Vec<f64>],
    target_columns: &[&[f64]],
    adjustment: &[u32],
    domains: &[u8],
    target: &KernelTransportStandardization,
    treatment: Option<&[f64]>,
    target_weights: Option<&[f64]>,
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
                if left != right {
                    return 0.0;
                }
            } else {
                let bandwidth = target
                    .bandwidths
                    .iter()
                    .find(|(column, _)| *column == id)
                    .map_or(f64::NAN, |(_, bandwidth)| *bandwidth);
                let distance = (left - right) / bandwidth;
                log_weight -= 0.5 * distance * distance;
            }
        }
        log_weight.exp()
    };
    if treatment.is_some_and(|values| {
        values.len() != source_rows || values.iter().any(|value| *value != 0.0 && *value != 1.0)
    }) {
        return Err(execution_refused("completed treatment must be binary"));
    }
    for target_row in 0..target_rows {
        if let Some(treatment) = treatment {
            for arm in [0.0, 1.0] {
                let count = treatment.iter().filter(|&&value| value == arm).count();
                if count == 0 {
                    return Err(execution_refused("source treatment arm is absent"));
                }
                let support = (0..source_rows)
                    .filter(|&row| treatment[row] == arm)
                    .map(|row| kernel(row, target_row, true))
                    .sum::<f64>()
                    / count as f64;
                if !support.is_finite() || support <= 0.0 || support < target.min_kernel_overlap {
                    return Err(execution_refused(
                        "target row lacks kernel overlap in a source treatment arm",
                    ));
                }
            }
        }
        let support =
            (0..source_rows).map(|source_row| kernel(source_row, target_row, true)).sum::<f64>()
                / source_rows as f64;
        if !support.is_finite() || support < target.min_kernel_overlap {
            return Err(execution_refused(
                "target covariate row lacks declared kernel overlap in the completed source",
            ));
        }
    }
    let target_total = target_weights.map_or(target_rows as f64, |weights| weights.iter().sum());
    if target_weights.is_some_and(|weights| {
        weights.len() != target_rows || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
    }) || !target_total.is_finite()
        || target_total <= 0.0
    {
        return Err(invalid("invalid MAR target weights"));
    }
    let mut weights = Vec::with_capacity(source_rows);
    for source_row in 0..source_rows {
        let numerator = (0..target_rows)
            .map(|target_row| {
                kernel(source_row, target_row, true)
                    * target_weights.map_or(1.0, |weights| weights[target_row])
            })
            .sum::<f64>()
            / target_total;
        let denominator =
            (0..source_rows).map(|other| kernel(source_row, other, false)).sum::<f64>()
                / source_rows as f64;
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
    for weight in &mut weights {
        *weight *= scale;
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(execution_refused("normalized kernel transport weights are nonfinite"));
    }
    Ok(weights)
}

fn binary_target_weights(
    source_keys: &[Vec<u8>],
    source_count: &BTreeMap<Vec<u8>, usize>,
    target_mass: &BTreeMap<Vec<u8>, f64>,
) -> Result<Vec<f64>, IoError> {
    let total: f64 = target_mass.values().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(execution_refused("invalid target mass"));
    }
    let source_n = source_keys.len() as f64;
    let weights: Vec<f64> = source_keys
        .iter()
        .map(|key| {
            let mass = *target_mass.get(key).unwrap_or(&0.0);
            mass * source_n / (source_count[key] as f64 * total)
        })
        .collect();
    if weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0) {
        return Err(execution_refused("invalid normalized binary transport weights"));
    }
    Ok(weights)
}

/// The query, population registry, identified estimand and optional target weights of a transport.
type TransportedPopulation =
    (AverageEffectQuery, PopulationRegistry, IdentifiedEstimand, Option<Vec<f64>>);

#[allow(clippy::too_many_lines, reason = "one construction per transport target form")]
fn transported_population(
    data: &TabularData,
    query: &AverageEffectQuery,
    atom: &DerivedAtom,
    target: TransportTarget<'_>,
    domains: &[u8],
    retain_weights: bool,
) -> Result<TransportedPopulation, IoError> {
    let mut adjustment = atom.operation.adjustment.clone();
    for &selected in target.selections() {
        if !adjustment.contains(&selected) {
            adjustment.push(selected);
        }
    }
    adjustment.sort_unstable();
    let mut estimand = atom.estimand.clone();
    estimand.adjustment_set =
        Arc::from(adjustment.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>());
    let mut source_columns = Vec::with_capacity(adjustment.len());
    let mut target_columns = Vec::with_capacity(adjustment.len());
    for &id in &adjustment {
        source_columns.push(data.float64_values(VariableId::from_raw(id)).map_err(invalid)?);
        target_columns.push(
            target
                .columns()
                .iter()
                .find(|(column, _)| *column == id)
                .ok_or_else(|| invalid("target is missing an identified adjustment covariate"))?
                .1
                .as_slice(),
        );
    }
    let binary_key = |columns: &[&[f64]], row: usize| -> Result<Vec<u8>, IoError> {
        columns
            .iter()
            .map(|column| match column[row] {
                0.0 => Ok(0),
                1.0 => Ok(1),
                _ => Err(execution_refused("completed source or target covariate is not binary")),
            })
            .collect()
    };
    let weights = match target {
        TransportTarget::Binary(binary)
        | TransportTarget::ConfoundedBinary(binary)
        | TransportTarget::MarBinary(binary, _, _, _) => {
            let source_slices: Vec<&[f64]> = source_columns.iter().map(Vec::as_slice).collect();
            let source_rows = data.row_count();
            let mut source_count = BTreeMap::<Vec<u8>, usize>::new();
            let mut target_count = BTreeMap::<Vec<u8>, f64>::new();
            let mut source_keys = Vec::with_capacity(source_rows);
            for row in 0..source_rows {
                let key = binary_key(&source_slices, row)?;
                *source_count.entry(key.clone()).or_default() += 1;
                source_keys.push(key);
            }
            for row in 0..binary.rows() {
                let weight = match target {
                    TransportTarget::MarBinary(_, weights, _, _) => weights[row],
                    _ => 1.0,
                };
                if !weight.is_finite() || weight <= 0.0 {
                    return Err(execution_refused("invalid target weight"));
                }
                *target_count.entry(binary_key(&target_columns, row)?).or_default() += weight;
            }
            if target_count
                .iter()
                .any(|(key, count)| *count > 0.0 && !source_count.contains_key(key))
            {
                return Err(execution_refused(
                    "target adjustment stratum has no completed source support",
                ));
            }
            if matches!(
                target,
                TransportTarget::ConfoundedBinary(_) | TransportTarget::MarBinary(_, _, _, true)
            ) {
                let treatment = data.float64_values(query.treatment).map_err(invalid)?;
                let mut arms = BTreeMap::<Vec<u8>, (bool, bool)>::new();
                for (key, &value) in source_keys.iter().zip(&treatment) {
                    let entry = arms.entry(key.clone()).or_default();
                    match value {
                        0.0 => entry.0 = true,
                        1.0 => entry.1 = true,
                        _ => return Err(execution_refused("completed treatment must be binary")),
                    }
                }
                if target_count
                    .keys()
                    .any(|key| !arms.get(key).is_some_and(|&(control, treated)| control && treated))
                {
                    return Err(execution_refused(
                        "target stratum lacks both source treatment arms",
                    ));
                }
            }
            binary_target_weights(&source_keys, &source_count, &target_count)?
        }
        TransportTarget::Kernel(kernel)
        | TransportTarget::ConfoundedKernel(kernel)
        | TransportTarget::MarKernel(kernel, _, _, _) => {
            let treatment = if matches!(
                target,
                TransportTarget::ConfoundedKernel(_) | TransportTarget::MarKernel(_, _, _, true)
            ) {
                Some(data.float64_values(query.treatment).map_err(invalid)?)
            } else {
                None
            };
            kernel_transport_weights(
                &source_columns,
                &target_columns,
                &adjustment,
                domains,
                kernel,
                treatment.as_deref(),
                match target {
                    TransportTarget::MarKernel(_, weights, _, _) => Some(weights),
                    _ => None,
                },
            )?
        }
    };
    let retained_weights = retain_weights.then(|| weights.clone());
    let handle = DistributionRef::from_raw(1);
    let mut registry = PopulationRegistry::new();
    registry.insert_distribution_with_dependence(
        handle,
        Arc::<[f64]>::from(weights),
        Arc::<[VariableId]>::from(
            adjustment.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        ),
    );
    Ok((
        query.clone().with_target_population(TargetPopulation::CustomDistribution(handle)),
        registry,
        estimand,
        retained_weights,
    ))
}

fn logistic_probability(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "the composite seals each completed table, graph fit and mixture together"
)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sampled row indices are a unit-interval draw scaled by a row count, clamped below it"
)]
fn execute(
    rec: &Reconstructed,
    target: Option<TransportTarget<'_>>,
    paired: bool,
    logit_predictive: bool,
    gaussian_predictive: Option<(f64, f64, u64)>,
    predictive_seed: Option<u64>,
    ctx: &ExecutionContext,
) -> Result<Execution, IoError> {
    let parts = rec.source.parts();
    if logit_predictive
        && (!paired
            || !matches!(target, Some(TransportTarget::Binary(_)))
            || parts.domains[parts.outcome as usize] != 1)
    {
        return Err(invalid(
            "Bernoulli predictive worlds require a binary outcome and binary target frame",
        ));
    }
    if let Some((variance, correlation, _)) = gaussian_predictive
        && (!paired
            || logit_predictive
            || !matches!(target, Some(TransportTarget::Binary(_)))
            || parts.domains[parts.outcome as usize] != 0
            || !variance.is_finite()
            || variance <= 0.0
            || !correlation.is_finite()
            || !(-1.0..=1.0).contains(&correlation))
    {
        return Err(invalid("invalid Gaussian target predictive declaration"));
    }
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
    let mut pooled_probabilities = Vec::new();
    let mut unidentified_by_table = Vec::new();
    let mut table_leases = Vec::new();
    let mut paired_leases = Vec::new();
    if paired {
        let bytes = u64::from(parts.imputations)
            .checked_mul(u64::from(parts.posterior_draws))
            .and_then(|n| n.checked_mul(16))
            .ok_or_else(|| invalid("paired world admission overflow"))?;
        paired_leases.push(scoped.acquire_admission(0, bytes).map_err(admission)?);
        if target.is_some() {
            paired_leases.push(
                scoped
                    .acquire_admission(0, (parts.rows() as u64).saturating_mul(8))
                    .map_err(admission)?,
            );
        }
    }
    let mut receipt = CanonicalWriter::new(match target {
        Some(TransportTarget::Binary(_)) => "antecedent.graph_imputation_transport.execution.v1",
        Some(TransportTarget::ConfoundedBinary(_)) => {
            "antecedent.graph_imputation_confounded_transport.execution.v1"
        }
        Some(TransportTarget::Kernel(_)) => {
            "antecedent.graph_imputation_kernel_transport.execution.v1"
        }
        Some(TransportTarget::ConfoundedKernel(_)) => {
            "antecedent.graph_imputation_confounded_kernel_transport.execution.v1"
        }
        Some(TransportTarget::MarBinary(_, _, _, _)) => {
            "antecedent.graph_imputation_mar_binary_transport.execution.v1"
        }
        Some(TransportTarget::MarKernel(_, _, _, _)) => {
            "antecedent.graph_imputation_mar_kernel_transport.execution.v1"
        }
        None => "antecedent.graph_candidate.execution.v1",
    });
    receipt.bytes(program.as_bytes()).bytes(&contract).word(u64::from(parts.imputations));
    let transported_imputer = if let Some(target) = target {
        receipt.bytes(&target.digest()?);
        let (imputer, interacted) = congenial_imputer(rec, target)?;
        receipt.word(interacted.len() as u64);
        for id in interacted {
            receipt.word(u64::from(id));
        }
        Some(imputer)
    } else {
        None
    };
    let imputer = transported_imputer.as_ref().unwrap_or(&rec.imputer);
    for index in 0..parts.imputations {
        let (data, completed_digest) = complete_table(rec, imputer, index, &scoped)?;
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
        let mut world_atoms: Vec<(u64, f64, Vec<[f64; 2]>, Vec<[f64; 2]>)> = Vec::new();
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
                backend: if logit_predictive {
                    BayesianBackendKind::Laplace
                } else {
                    BayesianBackendKind::ConjugateGaussian
                },
                likelihood: if logit_predictive {
                    BayesLikelihood::BernoulliLogit
                } else {
                    BayesLikelihood::GaussianIdentity
                },
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
                let (query, registry, estimand, weights) = transported_population(
                    &data,
                    &rec.query,
                    atom,
                    target,
                    &parts.domains,
                    paired,
                )?;
                // The logit child is a source conditional model. Target weights
                // select covariates for prediction; they do not weight its likelihood.
                let prepared = if logit_predictive {
                    fitter
                        .prepare(&data, &estimand, &rec.query)
                        .map_err(fit_error)?
                        .with_target_interactions(
                            weights
                                .as_ref()
                                .ok_or_else(|| invalid("predictive target weights missing"))?,
                        )
                        .map_err(fit_error)?
                } else {
                    fitter
                        .prepare_for_population(&data, &estimand, &query, Some(&registry))
                        .map_err(fit_error)?
                };
                (prepared, weights)
            } else {
                (fitter.prepare(&data, &atom.estimand, &rec.query).map_err(fit_error)?, None)
            };
            let fit = fitter
                .fit(
                    &prepared,
                    atom.operation.status,
                    &mut BayesianGCompWorkspace::default(),
                    &child_ctx,
                )
                .map_err(fit_error)?;
            if paired {
                if target.is_some_and(|target| !matches!(target, TransportTarget::Binary(_))) {
                    return Err(invalid(
                        "paired target worlds require the root-treatment binary route",
                    ));
                }
                let design = &prepared.design;
                let t = design
                    .treatment_column()
                    .ok_or_else(|| invalid("paired worlds lack treatment column"))?;
                if target_weights.is_some()
                    && (t != 1 || design.ncols != 2 + 2 * prepared.adjustment_set.len())
                {
                    return Err(invalid(
                        "paired target coefficient design differs from centered adjustment model",
                    ));
                }
                let mut columns = Vec::with_capacity(design.ncols);
                for index in 0..design.ncols {
                    let column = fit.draws.schema.quantities.iter().position(|quantity|
                        matches!(quantity, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index))
                        .ok_or_else(|| invalid("paired worlds lack coefficient draws"))?;
                    columns.push(fit.draws.column(column).map_err(execution_refused)?);
                }
                let bytes = (fit.draws.n_draws as u64)
                    .checked_mul(if logit_predictive { 32 } else { 16 })
                    .ok_or_else(|| invalid("paired atom admission overflow"))?;
                paired_leases.push(scoped.acquire_admission(0, bytes).map_err(admission)?);
                let mut world_values = Vec::with_capacity(fit.draws.n_draws);
                let mut success_values = Vec::new();
                // A declared residual or replicate seed selects a stream inside
                // the producer's execution stream, as the coefficient draws it
                // pairs with do. Keyed by the declared seed alone, every analysis
                // reused one fixed residual sample, whose Monte Carlo quantile
                // error then repeated in each interval instead of averaging out.
                let stream = match predictive_seed.or(gaussian_predictive.map(|(_, _, seed)| seed))
                {
                    Some(seed) => mix64(seed ^ mix64(parts.execution_seed ^ 0x5052_4544_5354_524d)),
                    None => parts.execution_seed,
                };
                let mut predictive_rng = antecedent_core::CausalRng::from_seed(mix64(
                    stream ^ (u64::from(index) << 32) ^ mask ^ 0x4245_524e_5052_4544,
                ));
                // A 0/1 world is one realization of a success probability, and a
                // 95% interval of that realization is its support. What a
                // Bernoulli predictive law asserts is the probability that a
                // future unit drawn from the target frame succeeds under each
                // action: the frame average of the unit probabilities under this
                // graph, completed table and coefficient draw.
                let success_frame = if logit_predictive {
                    let weights = target_weights
                        .as_ref()
                        .ok_or_else(|| invalid("predictive target weights missing"))?;
                    let mass = weights.iter().sum::<f64>();
                    let Some(TransportTarget::Binary(frame)) = target else {
                        return Err(invalid("predictive worlds need a complete target frame"));
                    };
                    if weights.len() != design.nrows
                        || !mass.is_finite()
                        || mass <= 0.0
                        || frame.rows() == 0
                    {
                        return Err(invalid("predictive target weights differ from design rows"));
                    }
                    let mut centers = Vec::with_capacity(prepared.adjustment_set.len());
                    let mut frame_columns = Vec::with_capacity(prepared.adjustment_set.len());
                    for (offset, variable) in prepared.adjustment_set.iter().enumerate() {
                        let col = 2 + offset;
                        let values = &design.matrix[col * design.nrows..(col + 1) * design.nrows];
                        centers.push(
                            values
                                .iter()
                                .zip(weights)
                                .map(|(value, weight)| value * weight)
                                .sum::<f64>()
                                / mass,
                        );
                        frame_columns.push(
                            frame
                                .columns
                                .iter()
                                .find(|(id, _)| *id == variable.raw())
                                .map(|(_, values)| values.as_slice())
                                .filter(|values| values.len() == frame.rows())
                                .ok_or_else(|| {
                                    invalid("predictive graph needs an absent target covariate")
                                })?,
                        );
                    }
                    // Binary target covariates repeat: average over their
                    // distinct rows, each weighted by its frame share.
                    let mut counts = BTreeMap::<Vec<u64>, usize>::new();
                    for row in 0..frame.rows() {
                        *counts
                            .entry(
                                frame_columns.iter().map(|column| column[row].to_bits()).collect(),
                            )
                            .or_default() += 1;
                    }
                    paired_leases.push(
                        scoped
                            .acquire_admission(
                                0,
                                (counts.len() as u64)
                                    .saturating_mul((centers.len() as u64).saturating_add(1))
                                    .saturating_mul(16),
                            )
                            .map_err(admission)?,
                    );
                    let rows = frame.rows() as f64;
                    let patterns: Vec<(Vec<f64>, f64)> = counts
                        .into_iter()
                        .map(|(key, count)| {
                            (key.into_iter().map(f64::from_bits).collect(), count as f64 / rows)
                        })
                        .collect();
                    Some((centers, patterns))
                } else {
                    None
                };
                for draw in 0..fit.draws.n_draws {
                    if logit_predictive || gaussian_predictive.is_some() {
                        // One observed target-frame covariate row is shared by
                        // both worlds. Source weights center the interaction
                        // parameterization but do not select a source proxy row.
                        let weights = target_weights
                            .as_ref()
                            .ok_or_else(|| invalid("predictive target weights missing"))?;
                        let mass = weights.iter().sum::<f64>();
                        if weights.len() != design.nrows || !mass.is_finite() || mass <= 0.0 {
                            return Err(invalid(
                                "predictive target weights differ from design rows",
                            ));
                        }
                        let Some(TransportTarget::Binary(frame)) = target else {
                            return Err(invalid("predictive worlds need a complete target frame"));
                        };
                        let row = ((predictive_rng.next_f64() * frame.rows() as f64) as usize)
                            .min(frame.rows() - 1);
                        let covariates = prepared.adjustment_set.len();
                        let mut linear = columns[0][draw];
                        let mut interaction = 0.0;
                        for (offset, variable) in prepared.adjustment_set.iter().enumerate() {
                            let col = 2 + offset;
                            let values =
                                &design.matrix[col * design.nrows..(col + 1) * design.nrows];
                            let center = values
                                .iter()
                                .zip(weights)
                                .map(|(value, weight)| value * weight)
                                .sum::<f64>()
                                / mass;
                            let value = frame
                                .columns
                                .iter()
                                .find(|(id, _)| *id == variable.raw())
                                .and_then(|(_, values)| values.get(row))
                                .copied()
                                .ok_or_else(|| {
                                    invalid("predictive graph needs an absent target covariate")
                                })?;
                            linear += value * columns[col][draw];
                            interaction +=
                                (value - center) * columns[2 + covariates + offset][draw];
                        }
                        let treatment_effect = columns[t][draw] + interaction;
                        let mean0 = linear + prepared.control * treatment_effect;
                        let mean1 = linear + prepared.active * treatment_effect;
                        if let Some((variance, correlation, _)) = gaussian_predictive {
                            let u = predictive_rng.next_f64().max(f64::MIN_POSITIVE);
                            let angle = 2.0 * std::f64::consts::PI * predictive_rng.next_f64();
                            let radius = (-2.0 * u.ln()).sqrt();
                            let z0 = radius * angle.cos();
                            let z1 = radius * angle.sin();
                            let scale = variance.sqrt();
                            let pair = [
                                mean0 + scale * z0,
                                mean1
                                    + scale
                                        * (correlation * z0
                                            + (1.0 - correlation * correlation).max(0.0).sqrt()
                                                * z1),
                            ];
                            if pair.iter().any(|value| !value.is_finite()) {
                                return Err(invalid("Gaussian predictive worlds are nonfinite"));
                            }
                            world_values.push(pair);
                        } else {
                            let p0 = logistic_probability(mean0);
                            let p1 = logistic_probability(mean1);
                            if !p0.is_finite() || !p1.is_finite() {
                                return Err(invalid(
                                    "Bernoulli predictive probabilities are nonfinite",
                                ));
                            }
                            let u0 = predictive_rng.next_f64();
                            let u1 = predictive_rng.next_f64();
                            world_values.push([
                                if u0 < p0 { 1.0 } else { 0.0 },
                                if u1 < p1 { 1.0 } else { 0.0 },
                            ]);
                            if let Some((centers, patterns)) = &success_frame {
                                let mut success = [0.0; 2];
                                for (values, share) in patterns {
                                    let mut base = columns[0][draw];
                                    let mut modifier = 0.0;
                                    for (offset, value) in values.iter().enumerate() {
                                        base += value * columns[2 + offset][draw];
                                        modifier += (value - centers[offset])
                                            * columns[2 + covariates + offset][draw];
                                    }
                                    let effect = columns[t][draw] + modifier;
                                    success[0] += share
                                        * logistic_probability(base + prepared.control * effect);
                                    success[1] += share
                                        * logistic_probability(base + prepared.active * effect);
                                }
                                if success.iter().any(|value| !value.is_finite()) {
                                    return Err(invalid(
                                        "Bernoulli success probabilities are nonfinite",
                                    ));
                                }
                                success_values.push(success);
                            }
                        }
                        continue;
                    }
                    let mut mean = 0.0;
                    for col in 0..design.ncols {
                        let values = &design.matrix[col * design.nrows..(col + 1) * design.nrows];
                        let value = if col == t {
                            0.0
                        } else if let Some(weights) = &target_weights {
                            let names = prepared
                                .coef_names
                                .as_ref()
                                .ok_or_else(|| invalid("target worlds lack coefficient names"))?;
                            // Weighted g-computation appends treatment interactions centered at
                            // these very target weights. Their target mean is zero in both worlds.
                            let name = names.get(col).ok_or_else(|| {
                                invalid("target world coefficient name is missing")
                            })?;
                            if col >= 2 + prepared.adjustment_set.len() {
                                if !name.starts_with("treatment_x_centered_") {
                                    return Err(invalid(
                                        "paired target interaction column is not centered",
                                    ));
                                }
                                0.0
                            } else {
                                let mass = weights.iter().sum::<f64>();
                                if weights.len() != design.nrows || !mass.is_finite() || mass <= 0.0
                                {
                                    return Err(invalid(
                                        "paired target weights differ from design rows",
                                    ));
                                }
                                values.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() / mass
                            }
                        } else {
                            values.iter().sum::<f64>() / design.nrows as f64
                        };
                        mean += value * columns[col][draw];
                    }
                    let control = mean + prepared.control * columns[t][draw];
                    let active = mean + prepared.active * columns[t][draw];
                    if !control.is_finite() || !active.is_finite() {
                        return Err(invalid("paired worlds are nonfinite"));
                    }
                    world_values.push([control, active]);
                }
                world_atoms.push((mask, weight, world_values, success_values));
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
                for (i, (_, weight, _, _)) in world_atoms.iter().enumerate() {
                    cumulative += weight / total;
                    if u < cumulative {
                        selected = i;
                        break;
                    }
                }
                let draws = &world_atoms[selected].2;
                let row = ((rng.next_f64() * draws.len() as f64) as usize).min(draws.len() - 1);
                pooled_worlds.push(draws[row]);
                if logit_predictive {
                    pooled_probabilities.push(
                        *world_atoms[selected]
                            .3
                            .get(row)
                            .ok_or_else(|| invalid("Bernoulli success probabilities misaligned"))?,
                    );
                }
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
        paired_probabilities: pooled_probabilities,
        mean,
        sd,
        receipt: receipt.finish(),
        unidentified_by_table,
        lease,
        paired_leases,
    })
}

/// The source imputer made congenial with a transported child fit, and the covariates it
/// interacts with treatment.
///
/// A transported child centres one treatment interaction on the target weights for every
/// covariate it adjusts for or selects, so its target effect reads how the source effect varies
/// with those covariates. An additive conditional model fills each missing cell as if the effect
/// did not vary, which pulls the transported effect toward the source-average effect. Each
/// retained graph's adjustment set and each selection target therefore enters the imputation
/// model as a product with treatment (passive imputation).
fn congenial_imputer(
    rec: &Reconstructed,
    target: TransportTarget<'_>,
) -> Result<(Imputer, Vec<u32>), IoError> {
    let parts = rec.source.parts();
    let mut covariates: Vec<u32> = rec
        .atoms
        .iter()
        .flat_map(|atom| atom.operation.adjustment.iter().copied())
        .chain(target.selections().iter().copied())
        .filter(|&id| id != parts.treatment && id != parts.outcome)
        .collect();
    covariates.sort_unstable();
    covariates.dedup();
    let name = |id: u32| -> Result<Arc<str>, IoError> {
        parts
            .names
            .get(id as usize)
            .map(|text| Arc::<str>::from(text.as_str()))
            .ok_or_else(|| invalid("an interacted covariate is outside the source schema"))
    };
    let treatment = name(parts.treatment)?;
    let mut pairs = Vec::with_capacity(covariates.len());
    for &id in &covariates {
        pairs.push((Arc::clone(&treatment), name(id)?));
    }
    let imputer = rec.imputer.clone().with_interactions(&pairs).map_err(execution_refused)?;
    Ok((imputer, covariates))
}

fn complete_table(
    rec: &Reconstructed,
    imputer: &Imputer,
    index: u32,
    ctx: &ExecutionContext,
) -> Result<(TabularData, [u8; 32]), IoError> {
    let completed = imputer.impute(index, ctx).map_err(|error| {
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
    let run = execute(&rec, None, false, false, None, None, ctx)?;
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
    /// Bernoulli predictive routes only, aligned to `worlds`: the probability
    /// that a future target-frame unit succeeds under `[control, active]`
    /// given that row's graph, completed table and coefficients. Empty on
    /// every other route.
    pub probabilities: Vec<[f64; 2]>,
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
pub fn replay_paired_worlds(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    let run = execute(&rec, None, true, false, None, None, ctx)?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation.paired_worlds.execution.v1");
    writer.bytes(&run.receipt).word(run.paired_worlds.len() as u64);
    for pair in &run.paired_worlds {
        writer.word(pair[0].to_bits()).word(pair[1].to_bits());
    }
    Ok(GraphPairedWorldReading {
        worlds: run.paired_worlds,
        probabilities: run.paired_probabilities,
        receipt: writer.finish(),
        program: rec.source.program(),
        source: rec.source.parts().data_digest,
        _lease: run.lease,
        _paired_leases: run.paired_leases,
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

/// Replay target-population predictive Bernoulli action worlds using a
/// Laplace-approximated logit child for each imputation and retained graph.
/// A target covariate row and coefficient state are shared; Bernoulli
/// innovations are independent conditional on that state. Each row also
/// carries the target-frame success probability of both actions under its
/// state. No calibration is implied.
///
/// # Errors
/// Invalid source, target, outcome, model, cancellation or resource refusal.
pub fn replay_bernoulli_logit_predictive_worlds(
    bytes: &[u8],
    target: &TransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Binary(target)), true, true, None, None, ctx)?;
    if run.paired_probabilities.len() != run.paired_worlds.len() {
        return Err(invalid("Bernoulli success probabilities misaligned"));
    }
    let mut writer = CanonicalWriter::new(
        "antecedent.graph_imputation_bernoulli_logit.target_predictive_worlds.execution.v1",
    );
    writer.bytes(&run.receipt).bytes(&target.digest()?).word(run.paired_worlds.len() as u64);
    for (pair, success) in run.paired_worlds.iter().zip(&run.paired_probabilities) {
        writer
            .word(pair[0].to_bits())
            .word(pair[1].to_bits())
            .word(success[0].to_bits())
            .word(success[1].to_bits());
    }
    Ok(GraphPairedWorldReading {
        worlds: run.paired_worlds,
        probabilities: run.paired_probabilities,
        receipt: writer.finish(),
        program: rec.source.program(),
        source: rec.source.parts().data_digest,
        _lease: run.lease,
        _paired_leases: run.paired_leases,
    })
}

/// Replay individual Gaussian target action worlds with a shared graph,
/// completed table, coefficient draw and target covariate row. Residual
/// innovations have the declared variance and cross-world correlation.
///
/// # Errors
/// Invalid source, frame, residual law, fit, cancellation or resource refusal.
pub fn replay_gaussian_target_predictive_worlds(
    bytes: &[u8],
    target: &TransportStandardization,
    residual_variance: f64,
    cross_world_correlation: f64,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(
        &rec,
        Some(TransportTarget::Binary(target)),
        true,
        false,
        Some((residual_variance, cross_world_correlation, seed)),
        None,
        ctx,
    )?;
    let mut writer = CanonicalWriter::new(
        "antecedent.graph_imputation_gaussian.target_predictive_worlds.execution.v1",
    );
    writer
        .bytes(&run.receipt)
        .bytes(&target.digest()?)
        .word(residual_variance.to_bits())
        .word(cross_world_correlation.to_bits())
        .word(seed)
        .word(run.paired_worlds.len() as u64);
    for pair in &run.paired_worlds {
        writer.word(pair[0].to_bits()).word(pair[1].to_bits());
    }
    Ok(GraphPairedWorldReading {
        worlds: run.paired_worlds,
        probabilities: run.paired_probabilities,
        receipt: writer.finish(),
        program: rec.source.program(),
        source: rec.source.parts().data_digest,
        _lease: run.lease,
        _paired_leases: run.paired_leases,
    })
}

/// The declared source with its imputation and child-posterior streams moved
/// to replicate `replicate` of a resampling schedule seeded by `seed`. A
/// replicate that reran the declared streams would re-read the completed tables
/// and posterior draws every other replicate holds, so the draws retained across
/// replicates would repeat one another instead of sampling the imputation and
/// posterior law independently. The moved streams stay reproducible from the
/// declaration and the schedule seed.
///
/// # Errors
/// A malformed source.
pub fn replicate_source(bytes: &[u8], seed: u64, replicate: u32) -> Result<Vec<u8>, IoError> {
    let source = GraphImputationSource::decode(bytes).map_err(invalid)?;
    let mut parts = source.parts().clone();
    let stream = mix64(seed ^ (u64::from(replicate) << 32) ^ 0x5245_504c_4943_4154);
    parts.imputation_seed = mix64(parts.imputation_seed ^ stream);
    parts.execution_seed = mix64(parts.execution_seed ^ stream.rotate_left(29));
    Ok(GraphImputationSource::new(parts, source.program()).map_err(invalid)?.encode())
}

/// Resample complete independent target clusters and rerun the entire graph,
/// imputation and individual predictive-world fit for each replicate, each on
/// its own imputation and posterior streams ([`replicate_source`]). Both
/// actions retain the same graph, coefficient draw and target row. One paired
/// outcome row, with its success probabilities on a Bernoulli route, is
/// sampled from each full replay; no calibration is implied.
///
/// # Errors
/// Invalid target or clusters, changed support, fit, resource or replay failure.
#[allow(
    clippy::too_many_lines,
    reason = "one replay per resampled cluster set with its full refit"
)]
#[allow(
    clippy::used_underscore_binding,
    reason = "the paired leases are held in the underscore field until the reading is dropped"
)]
pub fn replay_target_predictive_cluster_bootstrap_worlds(
    bytes: &[u8],
    target: &TransportStandardization,
    cluster_ids: &[String],
    replicates: u32,
    seed: u64,
    gaussian: Option<(f64, f64, u64)>,
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    if !(40..=256).contains(&replicates)
        || cluster_ids.len() != target.rows()
        || cluster_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(invalid("invalid predictive target cluster declaration"));
    }
    let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
    for (row, id) in cluster_ids.iter().enumerate() {
        grouped.entry(id.as_str()).or_default().push(row);
    }
    let groups: Vec<Vec<usize>> = grouped.into_values().collect();
    let max_rows = groups
        .len()
        .checked_mul(groups.iter().map(Vec::len).max().unwrap_or(0))
        .filter(|&n| groups.len() >= 2 && n <= 1_000_000)
        .ok_or_else(|| invalid("unsupported predictive target cluster size"))?;
    let mut reading = if let Some((variance, correlation, residual_seed)) = gaussian {
        replay_gaussian_target_predictive_worlds(
            bytes,
            target,
            variance,
            correlation,
            residual_seed,
            ctx,
        )?
    } else {
        replay_bernoulli_logit_predictive_worlds(bytes, target, ctx)?
    };
    let quote = (max_rows as u64)
        .saturating_mul((target.columns.len() as u64).saturating_mul(16) + 128)
        .saturating_add(u64::from(replicates).saturating_mul(16))
        .saturating_add(
            target
                .source_unit_ids
                .iter()
                .chain(&target.target_unit_ids)
                .map(|id| (id.len() as u64).saturating_add(24))
                .fold(0_u64, u64::saturating_add),
        );
    let lease = ctx.acquire_admission(0, quote).map_err(admission)?;
    let mut state = seed;
    let mut draw_state = mix64(seed ^ 0x4452_4157_5f56_3201);
    let mut pairs = Vec::with_capacity(replicates as usize);
    let mut successes =
        Vec::with_capacity(if gaussian.is_none() { replicates as usize } else { 0 });
    let mut receipt = CanonicalWriter::new(
        "antecedent.graph_imputation.target_predictive_cluster_bootstrap.execution.v1",
    );
    receipt
        .bytes(&reading.program)
        .bytes(&target.digest()?)
        .word(u64::from(replicates))
        .word(seed)
        .bytes(&crate::to_cbor(&cluster_ids)?);
    if let Some((variance, correlation, residual_seed)) = gaussian {
        receipt
            .text("gaussian")
            .word(variance.to_bits())
            .word(correlation.to_bits())
            .word(residual_seed);
    } else {
        receipt.text("bernoulli_logit");
    }
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let original = reconstruct(&replicate_source(bytes, seed, replicate)?, ctx)?;
        let mut indices = Vec::new();
        for _ in 0..groups.len() {
            let chosen = paired_bootstrap_pool_index(&mut state, groups.len());
            indices.extend_from_slice(&groups[chosen]);
        }
        let mut sampled = target.clone();
        for ((_, values), (_, original)) in sampled.columns.iter_mut().zip(&target.columns) {
            *values = indices.iter().map(|&row| original[row]).collect();
        }
        let source_ids: BTreeSet<&str> =
            sampled.source_unit_ids.iter().map(String::as_str).collect();
        let mut nonce = 0_u64;
        let prefix = loop {
            let prefix = format!("bootstrap-slot:{nonce}:");
            if !source_ids.iter().any(|id| id.starts_with(&prefix)) {
                break prefix;
            }
            nonce = nonce.checked_add(1).ok_or_else(|| invalid("bootstrap identity overflow"))?;
        };
        sampled.target_unit_ids = (0..indices.len()).map(|i| format!("{prefix}{i}")).collect();
        sampled.validate(original.source.parts())?;
        let noise_seed = mix64(
            seed ^ u64::from(replicate)
                ^ 0x5052_4544_4943_5401
                ^ gaussian.map_or(0, |(_, _, residual_seed)| residual_seed),
        );
        let run = execute(
            &original,
            Some(TransportTarget::Binary(&sampled)),
            true,
            gaussian.is_none(),
            gaussian,
            Some(noise_seed),
            ctx,
        )?;
        if run.paired_worlds.is_empty() {
            return Err(invalid("predictive target bootstrap has no paired worlds"));
        }
        let draw_index = paired_bootstrap_pool_index(&mut draw_state, run.paired_worlds.len());
        let pair = run.paired_worlds[draw_index];
        receipt.word(u64::from(replicate));
        for index in indices {
            receipt.word(index as u64);
        }
        receipt
            .bytes(&run.receipt)
            .word(noise_seed)
            .word(draw_index as u64)
            .word(pair[0].to_bits())
            .word(pair[1].to_bits());
        if gaussian.is_none() {
            let success = *run
                .paired_probabilities
                .get(draw_index)
                .ok_or_else(|| invalid("Bernoulli success probabilities misaligned"))?;
            receipt.word(success[0].to_bits()).word(success[1].to_bits());
            successes.push(success);
        }
        pairs.push(pair);
    }
    reading.worlds = pairs;
    reading.probabilities = successes;
    reading.receipt = receipt.finish();
    reading._paired_leases.push(lease);
    Ok(reading)
}

/// Replay paired model-conditional potential-outcome mean worlds on a
/// representative binary target frame. Both worlds share each sampled graph,
/// imputed table and fitted coefficient draw; this is not outcome prediction.
///
/// # Errors
/// Invalid target, target support, imputation, model or resource refusal.
pub fn replay_transport_paired_worlds(
    bytes: &[u8],
    target: &TransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Binary(target)), true, false, None, None, ctx)?;
    let mut writer =
        CanonicalWriter::new("antecedent.graph_imputation_transport.paired_worlds.execution.v2");
    writer.bytes(&run.receipt).bytes(&target.digest()?).word(run.paired_worlds.len() as u64);
    for pair in &run.paired_worlds {
        writer.word(pair[0].to_bits()).word(pair[1].to_bits());
    }
    Ok(GraphPairedWorldReading {
        worlds: run.paired_worlds,
        probabilities: run.paired_probabilities,
        receipt: writer.finish(),
        program: rec.source.program(),
        source: rec.source.parts().data_digest,
        _lease: run.lease,
        _paired_leases: run.paired_leases,
    })
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the remainder is below a bound that came from a usize"
)]
fn paired_bootstrap_pool_index(state: &mut u64, len: usize) -> usize {
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let word = z ^ (z >> 31);
        if word >= threshold {
            return (word % bound) as usize;
        }
    }
}

/// IID-resample aligned target rows and rerun the full paired-world route.
/// One intact graph/imputation/coefficient world pair is retained per replicate.
/// This is a model-conditional candidate, with no earned coverage.
///
/// # Errors
/// Invalid resampling policy, target support, replay or admission refusal.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the remainder is below a bound that came from a usize"
)]
#[allow(
    clippy::used_underscore_binding,
    reason = "the paired leases are held in the underscore field until the reading is dropped"
)]
pub fn replay_transport_paired_worlds_bootstrap(
    bytes: &[u8],
    target: &TransportStandardization,
    replicates: u32,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<GraphPairedWorldReading, IoError> {
    if !(40..=256).contains(&replicates) {
        return Err(invalid("paired target bootstrap needs 40 to 256 replicates"));
    }
    let mut reading = replay_transport_paired_worlds(bytes, target, ctx)?;
    let rows = target.rows();
    let bootstrap_bytes = (rows as u64)
        .saturating_mul((target.columns.len() as u64).saturating_mul(2) + 1)
        .saturating_mul(8)
        .saturating_add(u64::from(replicates).saturating_mul(16))
        .saturating_add(
            target
                .source_unit_ids
                .iter()
                .chain(&target.target_unit_ids)
                .map(|id| (id.len() as u64).saturating_add(24))
                .fold(0_u64, u64::saturating_add),
        );
    let lease = ctx.acquire_admission(0, bootstrap_bytes).map_err(admission)?;
    let mut state = seed;
    let mut draw_state = mix64(seed ^ 0x4452_4157_5f56_3201);
    let mut pairs = Vec::with_capacity(replicates as usize);
    let mut receipt = CanonicalWriter::new(
        "antecedent.graph_imputation_transport.paired_world_bootstrap.execution.v2",
    );
    receipt.bytes(&reading.program).bytes(&target.digest()?).word(u64::from(replicates)).word(seed);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let mut sampled = target.clone();
        let indices: Vec<usize> = (0..rows)
            .map(|_| {
                let bound = rows as u64;
                let threshold = bound.wrapping_neg() % bound;
                loop {
                    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    let word = z ^ (z >> 31);
                    if word >= threshold {
                        break (word % bound) as usize;
                    }
                }
            })
            .collect();
        for ((_, values), (_, original)) in sampled.columns.iter_mut().zip(&target.columns) {
            *values = indices.iter().map(|&index| original[index]).collect();
        }
        // Unit identities denote slots; the selected original indices are sealed.
        let run = replay_transport_paired_worlds(bytes, &sampled, ctx)?;
        if run.worlds.is_empty() {
            return Err(invalid("paired bootstrap produced no worlds"));
        }
        let index = paired_bootstrap_pool_index(&mut draw_state, run.worlds.len());
        let pair = run.worlds[index];
        receipt.word(u64::from(replicate));
        for index in indices {
            receipt.word(index as u64);
        }
        receipt
            .bytes(&run.receipt)
            .word(index as u64)
            .word(pair[0].to_bits())
            .word(pair[1].to_bits());
        pairs.push(pair);
    }
    reading.worlds = pairs;
    reading.receipt = receipt.finish();
    reading._paired_leases.push(lease);
    Ok(reading)
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
    bytes: &[u8],
    target: &TransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Binary(target)), false, false, None, None, ctx)?;
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
    bytes: &[u8],
    target: &TransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate_confounded(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(
        &rec,
        Some(TransportTarget::ConfoundedBinary(target)),
        false,
        false,
        None,
        None,
        ctx,
    )?;
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

/// Refit the same graph and imputation schedule while standardizing each
/// retained child to a mixed binary-continuous target frame with prespecified
/// kernel density-ratio weights. The target frame is empirical and intervals
/// remain unmeasured.
///
/// # Errors
/// Invalid target evidence, insufficient kernel overlap, graph, fit,
/// cancellation or admission refusal.
pub fn replay_kernel_transport_point(
    bytes: &[u8],
    target: &KernelTransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(&rec, Some(TransportTarget::Kernel(target)), false, false, None, None, ctx)?;
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

/// Reidentify and refit a mixed kernel target under pretreatment confounding.
/// Each target row must meet the declared kernel-overlap floor separately in
/// both source treatment arms on every completed table.
///
/// # Errors
/// Invalid graph, target, adjustment, selection, overlap, fit or resources.
pub fn replay_confounded_kernel_transport_point(
    bytes: &[u8],
    target: &KernelTransportStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    target.validate_confounded(rec.source.parts())?;
    let target_bytes =
        (target.rows() as u64).saturating_mul(target.columns.len() as u64).saturating_mul(16);
    let _target_lease = ctx.acquire_admission(0, target_bytes).map_err(admission)?;
    let run = execute(
        &rec,
        Some(TransportTarget::ConfoundedKernel(target)),
        false,
        false,
        None,
        None,
        ctx,
    )?;
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

    #[test]
    fn paired_selector_reaches_later_imputations() {
        let mut state = mix64(7 ^ 0x4452_4157_5f56_3201);
        let indices: Vec<_> =
            (0..40).map(|_| paired_bootstrap_pool_index(&mut state, 200)).collect();
        assert!(indices.iter().any(|&index| index < 100));
        assert!(indices.iter().any(|&index| index >= 100));
    }

    #[test]
    fn confounded_kernel_requires_both_arm_supports() {
        let target = KernelTransportStandardization {
            columns: vec![(2, vec![0.5, 0.6])],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: "source-sample".into(),
            target_sample: "target-sample".into(),
            source_unit_ids: (0..4).map(|i| format!("s-{i}")).collect(),
            target_unit_ids: (0..2).map(|i| format!("t-{i}")).collect(),
            bandwidths: vec![(2, 1.0)],
            min_kernel_overlap: 0.01,
        };
        let source = vec![vec![0.0, 1.0, 0.0, 1.0]];
        let target_columns = vec![target.columns[0].1.as_slice()];
        assert!(
            kernel_transport_weights(
                &source,
                &target_columns,
                &[2],
                &[1, 0, 0],
                &target,
                Some(&[0.0, 1.0, 0.0, 1.0]),
                None
            )
            .is_ok()
        );
        assert!(
            kernel_transport_weights(
                &source,
                &target_columns,
                &[2],
                &[1, 0, 0],
                &target,
                Some(&[0.0, 0.0, 0.0, 0.0]),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn selection_collider_is_not_s_admissible() {
        use antecedent_graph::{Dag, DenseNodeId};
        let node = DenseNodeId::from_raw;
        // T -> Y, U -> Y, U -> Z, S -> Z: conditioning on Z opens S--Y.
        let mut collider = Dag::with_variables(5);
        collider.insert_directed(node(0), node(1)).unwrap();
        collider.insert_directed(node(3), node(1)).unwrap();
        collider.insert_directed(node(3), node(2)).unwrap();
        assert!(
            validate_selection_admissibility(&collider, node(0), node(1), &[node(2)], &[2])
                .is_err()
        );
        // With Z exogenous, conditioning on Z blocks the selected mechanism.
        let mut root = Dag::with_variables(5);
        root.insert_directed(node(0), node(1)).unwrap();
        root.insert_directed(node(2), node(1)).unwrap();
        assert!(
            validate_selection_admissibility(&root, node(0), node(1), &[node(2)], &[2]).is_ok()
        );
        // Check all selection nodes against one common expanded diagram.
        root.insert_directed(node(4), node(1)).unwrap();
        assert!(
            validate_selection_admissibility(&root, node(0), node(1), &[node(2), node(4)], &[2, 4])
                .is_ok()
        );
        assert!(
            validate_selection_admissibility(
                &collider,
                node(0),
                node(1),
                &[node(2), node(4)],
                &[2, 4]
            )
            .is_err()
        );
    }
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
    #[allow(clippy::float_cmp, reason = "replayed worlds must match bit for bit")]
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

    #[test]
    fn replicate_sources_keep_the_declaration_and_move_only_the_streams() {
        let bytes = encoded(parts(GraphSupport::TreatmentRootPreTreatmentCovariates, 3, 100));
        let declared = GraphImputationSource::decode(&bytes).unwrap();
        let first = replicate_source(&bytes, 7, 0).unwrap();
        assert_eq!(first, replicate_source(&bytes, 7, 0).unwrap());
        let mut seen = BTreeSet::new();
        for replicate in 0..8 {
            let moved =
                GraphImputationSource::decode(&replicate_source(&bytes, 7, replicate).unwrap())
                    .unwrap();
            let (moved_parts, declared_parts) = (moved.parts(), declared.parts());
            assert_eq!(moved.program(), declared.program());
            assert_eq!(moved_parts.data_digest, declared_parts.data_digest);
            assert_eq!(moved_parts.source_digest(), declared_parts.source_digest());
            assert_eq!(moved_parts.atoms, declared_parts.atoms);
            assert_eq!(moved_parts.posterior_draws, declared_parts.posterior_draws);
            assert!(seen.insert((moved_parts.imputation_seed, moved_parts.execution_seed)));
            assert_ne!(moved_parts.imputation_seed, declared_parts.imputation_seed);
            assert_ne!(moved_parts.execution_seed, declared_parts.execution_seed);
        }
        assert_ne!(first, replicate_source(&bytes, 8, 0).unwrap());
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

#[cfg(test)]
mod raw_target_frame_tests {
    use super::RawTargetFrame;

    #[test]
    fn complete_rows_keep_inverse_observation_weights_and_raw_identity() {
        let frame = RawTargetFrame {
            columns: vec![
                (2, vec![Some(0.0), None, Some(1.0)]),
                (3, vec![Some(1.0), Some(0.0), Some(1.0)]),
            ],
            row_ids: vec!["a".into(), "b".into(), "c".into()],
            cluster_ids: vec!["one".into(), "one".into(), "two".into()],
            weights: vec![1.0, 2.0, 3.0],
            observation_probabilities: vec![0.5, 0.5, 0.75],
            always_observed: vec![3],
        };
        assert_eq!(frame.complete_rows(), vec![(0, 2.0), (2, 4.0)]);
        let mut changed = frame.clone();
        changed.cluster_ids[0] = "other".into();
        assert_ne!(frame.digest().unwrap(), changed.digest().unwrap());
        changed = frame.clone();
        changed.observation_probabilities[1] = 0.25;
        assert_ne!(frame.digest().unwrap(), changed.digest().unwrap());
    }
}

/// Raw independent target frame with declared complete-vector MAR probabilities.
/// Weights are sampling weights divided by those probabilities on complete rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarTargetStandardization {
    /// Prespecified identity of the complete-vector observation probability mechanism.
    pub observation_mechanism_id: String,
    /// Prespecified identity of sampling weights and the independent-cluster design.
    pub sampling_design_id: String,
    /// Raw target observations, missingness and design weights.
    pub frame: RawTargetFrame,
    /// Selected mechanisms allowed to vary across populations.
    pub selection_targets: Vec<u32>,
    /// Source and target population labels.
    pub source_population: String,
    /// Target population label.
    pub target_population: String,
    /// Identity of the retained source sample.
    pub source_sample: String,
    /// Independent target sample identity.
    pub target_sample: String,
    /// Source row identities.
    pub source_unit_ids: Vec<String>,
    /// Continuous-column bandwidths; empty for a binary target.
    pub bandwidths: Vec<(u32, f64)>,
    /// Kernel overlap floor (ignored for the binary mode).
    pub min_kernel_overlap: f64,
    /// Use mixed-domain kernel transport rather than binary strata.
    pub kernel: bool,
    /// Use pretreatment-confounding support and both-arm overlap.
    pub confounded: bool,
}

impl MarTargetStandardization {
    /// Canonical digest binds all raw rows, missingness, design and modelling assumptions.
    pub fn digest(&self) -> Result<[u8; 32], IoError> {
        Ok(CanonicalWriter::digest_bytes(
            "antecedent.graph_imputation.mar_target.v2",
            &crate::to_cbor(self)?,
        ))
    }

    /// Validate raw MAR declarations and the complete-case transport scope.
    pub fn validate(&self, parts: &GraphImputationParts) -> Result<(), IoError> {
        if self.observation_mechanism_id.trim().is_empty()
            || self.sampling_design_id.trim().is_empty()
        {
            return Err(invalid(
                "MAR observation mechanism and target sampling design need prespecified identities",
            ));
        }
        self.frame.validate(parts, &self.source_unit_ids)?;
        let rows = self.frame.complete_rows();
        let columns: Vec<_> = self
            .frame
            .columns
            .iter()
            .map(|(id, values)| {
                (*id, rows.iter().map(|(row, _)| values[*row].expect("complete row")).collect())
            })
            .collect();
        let ids = rows.iter().map(|(row, _)| self.frame.row_ids[*row].clone()).collect();
        if self.kernel {
            let target = KernelTransportStandardization {
                columns,
                selection_targets: self.selection_targets.clone(),
                source_population: self.source_population.clone(),
                target_population: self.target_population.clone(),
                source_sample: self.source_sample.clone(),
                target_sample: self.target_sample.clone(),
                source_unit_ids: self.source_unit_ids.clone(),
                target_unit_ids: ids,
                bandwidths: self.bandwidths.clone(),
                min_kernel_overlap: self.min_kernel_overlap,
            };
            if self.confounded { target.validate_confounded(parts) } else { target.validate(parts) }
        } else {
            if !self.bandwidths.is_empty() {
                return Err(invalid("binary MAR target does not accept kernel bandwidths"));
            }
            let target = TransportStandardization {
                columns,
                selection_targets: self.selection_targets.clone(),
                source_population: self.source_population.clone(),
                target_population: self.target_population.clone(),
                source_sample: self.source_sample.clone(),
                target_sample: self.target_sample.clone(),
                source_unit_ids: self.source_unit_ids.clone(),
                target_unit_ids: ids,
            };
            if self.confounded { target.validate_confounded(parts) } else { target.validate(parts) }
        }
    }
}

/// Replay source imputation and graph selection with inverse-observation weighted
/// complete target cases. MAR and the supplied observation probabilities are assumptions.
///
/// # Errors
/// Invalid declarations, support, overlap, inference, cancellation or admission.
pub fn replay_mar_transport_point(
    bytes: &[u8],
    target: &MarTargetStandardization,
    ctx: &ExecutionContext,
) -> Result<GraphPointReading, IoError> {
    let rec = reconstruct(bytes, ctx)?;
    let lease_bytes = (target.frame.row_ids.len() as u64)
        .saturating_mul(target.frame.columns.len() as u64 + 4)
        .saturating_mul(32);
    let _target_lease = ctx.acquire_admission(0, lease_bytes).map_err(admission)?;
    target.validate(rec.source.parts())?;
    let rows = target.frame.complete_rows();
    let columns: Vec<_> = target
        .frame
        .columns
        .iter()
        .map(|(id, values)| {
            (
                *id,
                rows.iter().map(|(row, _)| values[*row].expect("validated complete row")).collect(),
            )
        })
        .collect();
    let ids = rows.iter().map(|(row, _)| target.frame.row_ids[*row].clone()).collect();
    let weights: Vec<f64> = rows.iter().map(|(_, weight)| *weight).collect();
    let digest = target.digest()?;
    let run = if target.kernel {
        let standard = KernelTransportStandardization {
            columns,
            selection_targets: target.selection_targets.clone(),
            source_population: target.source_population.clone(),
            target_population: target.target_population.clone(),
            source_sample: target.source_sample.clone(),
            target_sample: target.target_sample.clone(),
            source_unit_ids: target.source_unit_ids.clone(),
            target_unit_ids: ids,
            bandwidths: target.bandwidths.clone(),
            min_kernel_overlap: target.min_kernel_overlap,
        };
        execute(
            &rec,
            Some(TransportTarget::MarKernel(&standard, &weights, digest, target.confounded)),
            false,
            false,
            None,
            None,
            ctx,
        )?
    } else {
        let standard = TransportStandardization {
            columns,
            selection_targets: target.selection_targets.clone(),
            source_population: target.source_population.clone(),
            target_population: target.target_population.clone(),
            source_sample: target.source_sample.clone(),
            target_sample: target.target_sample.clone(),
            source_unit_ids: target.source_unit_ids.clone(),
            target_unit_ids: ids,
        };
        execute(
            &rec,
            Some(TransportTarget::MarBinary(&standard, &weights, digest, target.confounded)),
            false,
            false,
            None,
            None,
            ctx,
        )?
    };
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

#[cfg(test)]
mod mar_weighted_transport_tests {
    use super::{KernelTransportStandardization, binary_target_weights, kernel_transport_weights};
    use std::collections::BTreeMap;

    #[test]
    fn binary_target_strata_use_normalized_weighted_mass() {
        let keys = vec![vec![0], vec![0], vec![1], vec![1]];
        let source = BTreeMap::from([(vec![0], 2), (vec![1], 2)]);
        let target = BTreeMap::from([(vec![0], 1.0), (vec![1], 3.0)]);
        assert_eq!(
            binary_target_weights(&keys, &source, &target).unwrap(),
            vec![0.5, 0.5, 1.5, 1.5]
        );
    }

    #[test]
    fn kernel_target_mass_uses_inverse_observation_weights_and_checks_both_arms() {
        let target = KernelTransportStandardization {
            columns: vec![(2, vec![0.0, 10.0])],
            selection_targets: vec![2],
            source_population: "source".into(),
            target_population: "target".into(),
            source_sample: "source".into(),
            target_sample: "target".into(),
            source_unit_ids: (0..4).map(|i| format!("s{i}")).collect(),
            target_unit_ids: vec!["a".into(), "b".into()],
            bandwidths: vec![(2, 1.0)],
            min_kernel_overlap: 0.01,
        };
        let source = vec![vec![0.0, 0.0, 10.0, 10.0]];
        let columns = vec![target.columns[0].1.as_slice()];
        let treatment = [0.0, 1.0, 0.0, 1.0];
        let weighted = kernel_transport_weights(
            &source,
            &columns,
            &[2],
            &[1, 1, 0],
            &target,
            Some(&treatment),
            Some(&[1.0, 3.0]),
        )
        .unwrap();
        assert!(weighted[2] > weighted[0] * 2.9);
        assert!((weighted.iter().sum::<f64>() - 4.0).abs() < 1e-10);
        assert!(
            kernel_transport_weights(
                &source,
                &columns,
                &[2],
                &[1, 1, 0],
                &target,
                Some(&[0.0, 0.0, 1.0, 1.0]),
                Some(&[1.0, 3.0])
            )
            .is_err()
        );
        assert!(
            kernel_transport_weights(
                &source,
                &columns,
                &[2],
                &[1, 1, 0],
                &target,
                None,
                Some(&[1.0, 0.0])
            )
            .is_err()
        );
    }
}
