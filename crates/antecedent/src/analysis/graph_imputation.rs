//! Bounded compiled graph-selection imputation composite.
//!
//! The mixture weights the models of one treatment-to-outcome contrast. Under the root support
//! the empty adjustment set identifies it in every graph, so the graphs differ only in how the
//! other variables are modeled. Under the direct-effect support the treatment may have parents,
//! adjustment sets differ between graphs, and the mixture carries uncertainty about which
//! variables confound.
//! The support is further restricted to graphs whose treatment-to-outcome effect has a unique
//! minimal back-door adjustment set: a graph with several names no single operation. The
//! identifier's own output decides, through the same derivation the consumer applies, the
//! mixture weights are over the retained graphs only and the number excluded is sealed.
//! The compiler prepares every retained graph's own Bayesian g-computation operation and seals
//! each retained operation into the lowered program. Execution, replay and the measured
//! interval all run through the single native implementation in the I/O layer, which
//! re-derives each operation from its graph and refuses unless it equals the compiler's. The
//! interval is reported only through the measured route, which is calibrated exactly when a
//! current attesting coverage record covers the execution.
//! No external callback, likelihood or posterior is accepted.
use std::sync::Arc;

use antecedent_core::graph_imputation_contract::{
    AtomOperation, GraphImputationParts, GraphImputationSource, MAX_VARIABLES, MIN_IMPUTATIONS,
    MIN_ITERATIONS, MIN_POSTERIOR_DRAWS, MIN_ROWS, admission_bytes, labeled_dag_count,
};
use antecedent_core::{AverageEffectQuery, CanonicalWriter, ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TableView, TabularData};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, dag_from_adjacency_mask};
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::graph_candidate::{
    GraphPointReading, replay_point, retained_support,
};

use super::RetainedOperation;
use super::lowering::graph_imputation::{LoweringInput, RetainedAtomBinding, lower};
use antecedent_core::search::{SearchBudget, SearchLimits};
mod wire;
pub use wire::{GraphImputationArtifact, GraphImputationExpectation, consume};

use crate::planner::lowered::LoweredProgram;
use crate::{BayesianConfig, CausalError, EstimatorId, InferenceMode, RefuteSuite, Study};

pub use antecedent_core::graph_imputation_contract::GraphSupport as GraphImputationSupport;

fn invalid(message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile { message: format!("graph_imputation.candidate_refused: {message}") }
}

/// A refusal of the native replay, in the engine's own terms. Calibration never refuses: an
/// execution no record covers is reported with its calibration status.
fn replay_error(error: IoError) -> GraphImputationError {
    match error {
        IoError::Refused { code: "cancelled_no_claim", .. } => {
            CausalError::Cancelled { stage: "graph imputation replay" }.into()
        }
        IoError::Refused { code: "execution_resource_limit", message } => {
            CausalError::Resource { message }.into()
        }
        other => CausalError::from(other).into(),
    }
}

/// Native declaration only. Graph score, prior, inference, and mixture construction are fixed.
#[derive(Clone, Debug)]
pub struct GraphImputationSpec {
    /// Binary intervention variable.
    pub treatment: VariableId,
    /// Numeric outcome variable.
    pub outcome: VariableId,
    /// Predeclared graph support, a substantive assumption not inferred from data.
    pub support: GraphImputationSupport,
    /// Native chained-equations configuration.
    pub imputation: ImputationSpec,
    /// Column domains in schema order: zero continuous, one binary.
    pub domains: Arc<[u8]>,
    /// Number of child effect draws per graph.
    pub posterior_draws: u32,
    /// Positive conjugate-model prior scale.
    pub prior_scale: f64,
    /// Seed of the compiled child execution streams.
    pub execution_seed: u64,
}

/// The native compiler retains all graph atoms and their own proofs before running any imputation.
pub struct CompiledGraphImputation {
    source: TabularData,
    source_digest: [u8; 32],
    _lease: antecedent_core::execution::AdmissionLease,
    #[cfg_attr(not(test), allow(dead_code, reason = "inspected by the conformance tests"))]
    atoms: Vec<RetainedAtomBinding>,
    lowered: Arc<LoweredProgram>,
    source_bytes: Arc<[u8]>,
}

/// A numerical child failure remains distinct from graph non-identification.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationError {
    /// An existing engine refusal or execution error.
    #[error(transparent)]
    Engine(#[from] CausalError),
    /// No scientific point has been supported for this candidate.
    #[error("graph_imputation.point_unsupported: the compiled candidate is not supported")]
    PointUnavailable,
    /// The replayed execution reported no interval scalar.
    #[error("graph_imputation.interval_unsupported: the replay reported no interval")]
    IntervalUnavailable,
    /// A graph inside declared support did not have an identified effect proof.
    #[error("graph_imputation.supported_graph_unidentified: graph mask {mask}")]
    SupportedGraphNotIdentified {
        /// Exact graph mask for the refused proof.
        mask: u64,
    },
    /// The shared bounded search stopped with its resource receipt.
    #[error("graph_imputation.search_stopped: {}", .0.summary())]
    SearchStopped(antecedent_core::search::SearchReceipt),
    /// Some graphs had no finite score, so their mass is unknown.
    #[error("graph_imputation.unscorable: table {table}, {count} graphs")]
    Unscorable {
        /// Completed table index.
        table: u32,
        /// Graph count with no finite score.
        count: u64,
    },
    /// A completed table retained graph mass without an identified effect.
    #[error("graph_imputation.unidentified: table {table}, mass {mass}")]
    Unidentified {
        /// Completed table index.
        table: u32,
        /// Graph weight that was not identified.
        mass: f64,
    },
    /// One or more graph atoms failed or were skipped.
    #[error(
        "graph_imputation.atom_failure: table {table}, failed mass {failed_mass}, skipped mass {skipped_mass}: {reasons:?}"
    )]
    AtomFailures {
        /// Completed table index.
        table: u32,
        /// Graph weight whose execution failed.
        failed_mass: f64,
        /// Graph weight that was not executed.
        skipped_mass: f64,
        /// Graph masks and bounded reasons.
        reasons: Vec<(u64, String)>,
    },
}

/// Point and diagnostic SD are model-conditional; neither is an interval or calibrated authority.
pub struct GraphImputationPoint {
    /// Model-conditional point diagnostic.
    pub mean: f64,
    /// Model-conditional dispersion diagnostic, not a standard error.
    pub diagnostic_sd: f64,
    /// Identity of the retained execution.
    pub receipt: [u8; 32],
    program: [u8; 32],
    source: [u8; 32],
    /// Unidentified graph mass for each completed table.
    pub unidentified_by_table: Vec<f64>,
    bytes: Arc<[u8]>,
    ctx: ExecutionContext,
    _reading: GraphPointReading,
}

impl GraphImputationPoint {
    fn replay(bytes: &[u8], ctx: &ExecutionContext) -> Result<Self, GraphImputationError> {
        let reading = replay_point(bytes, ctx).map_err(replay_error)?;
        Ok(Self {
            mean: reading.mean,
            diagnostic_sd: reading.diagnostic_sd,
            receipt: reading.receipt,
            program: reading.program,
            source: reading.source,
            unidentified_by_table: reading.unidentified_by_table.clone(),
            bytes: Arc::from(bytes),
            ctx: ctx.clone(),
            _reading: reading,
        })
    }

    /// The measured report of this execution's original source, issued by the same native
    /// replay. The interval is always reported; its calibration slot says whether a current
    /// attesting record covers this construction and row count.
    ///
    /// # Errors
    /// A typed engine refusal when the replay itself fails.
    pub fn measured(&self) -> Result<MeasuredInference, GraphImputationError> {
        antecedent_io::measured_inference::graph_candidate::measure_source(&self.bytes, &self.ctx)
            .map_err(replay_error)
    }

    /// The interval of the measured route; read its calibration with [`Self::calibration`].
    ///
    /// # Errors
    /// As [`Self::measured`].
    pub fn interval(&self) -> Result<(f64, f64), GraphImputationError> {
        let measured = self.measured()?;
        let scalar =
            measured.report().scalars.first().ok_or(GraphImputationError::IntervalUnavailable)?;
        Ok((scalar.lower, scalar.upper))
    }
}

impl GraphImputationPoint {
    /// The calibration slot of the measured interval.
    ///
    /// # Errors
    /// As [`Self::measured`].
    pub fn calibration(
        &self,
    ) -> Result<antecedent_io_core::contract_section::CalibrationSlotWire, GraphImputationError>
    {
        let measured = self.measured()?;
        let scalar =
            measured.report().scalars.first().ok_or(GraphImputationError::IntervalUnavailable)?;
        Ok(scalar.calibration.clone())
    }
}

impl CompiledGraphImputation {
    /// Compile on the outer context. This does no imputation, graph scoring or estimation.
    ///
    /// # Errors
    /// Refuses invalid declarations, insufficient resources, and unsupported child proofs.
    #[allow(
        clippy::too_many_lines,
        reason = "compilation validates and seals the complete bounded declaration"
    )]
    pub fn compile(
        source: TabularData,
        spec: &GraphImputationSpec,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphImputationError> {
        if ctx.lowered_dispatch().is_some() {
            return Err(crate::unsupported_reason!(
                "public_builder_reentry",
                "graph_imputation.reentry: compile is not permitted inside another lowered dispatch"
            )
            .into());
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "graph imputation compile" }.into());
        }
        let n = source.schema().variables().len();
        let rows = source.row_count();
        if !(2..=MAX_VARIABLES).contains(&n)
            || rows < MIN_ROWS
            || spec.domains.len() != n
            || spec.domains.iter().any(|&domain| domain > 1)
            || spec.treatment == spec.outcome
            || spec.treatment.raw() as usize >= n
            || spec.outcome.raw() as usize >= n
            || spec.domains.get(spec.treatment.raw() as usize) != Some(&1)
            || spec.imputation.imputations < MIN_IMPUTATIONS
            || spec.imputation.iterations < MIN_ITERATIONS
            || spec.posterior_draws < MIN_POSTERIOR_DRAWS
            || !spec.prior_scale.is_finite()
            || spec.prior_scale <= 0.0
            || source.storage().analysis_mask().is_some()
            || source.storage().weights().is_some()
        {
            return Err(invalid(
                "declaration, data shape or sampling design lies outside the bounded cell",
            )
            .into());
        }
        let quote = admission_bytes(rows, n);
        let lease = ctx.acquire_admission(0, quote).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation compile" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let columns = wire::canonical_source(&source, ctx)?;
        let mut schema = CanonicalWriter::new("antecedent.graph_imputation.schema.v1");
        schema.word(n as u64);
        for variable in source.schema().variables() {
            schema.word(u64::from(variable.id.raw())).text(&variable.name);
        }
        let schema_digest = schema.finish();
        let variables: Arc<[VariableId]> = Arc::from(
            (0..n)
                .map(|i| u32::try_from(i).map(VariableId::from_raw))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| invalid("variable index exceeds platform"))?,
        );
        let source_digest =
            source.storage().content_digest_with_context(ctx).map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "graph source identity" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
        let query = AverageEffectQuery::binary_ate(spec.treatment, spec.outcome);
        let config = BayesianConfig::conjugate()
            .n_draws(spec.posterior_draws as usize)
            .prior_scale(spec.prior_scale);
        let most_atoms = usize::try_from(labeled_dag_count(n)).unwrap_or(usize::MAX);
        let mut budget = SearchBudget::with_memory(
            SearchLimits { operations: most_atoms.saturating_add(64), depth: 1 },
            25 * 1024 * 1024,
            ctx,
        )
        .map_err(GraphImputationError::SearchStopped)?;
        // The same derivation the consumer applies decides which supported graphs are retained.
        let retained = retained_support(n, spec.treatment.raw(), spec.outcome.raw(), spec.support)
            .map_err(CausalError::from)?;
        let mut atoms = Vec::new();
        let total = retained.operations.len();
        for (done, operation) in retained.operations.iter().enumerate() {
            let mask = operation.mask;
            if let Err(stop) = budget.charge(1, 25 * 1024 * 1024) {
                return Err(GraphImputationError::SearchStopped(budget.receipt(
                    stop,
                    vec![format!("compiled:0..{done}")],
                    vec![format!("remaining:{done}..{total}")],
                )));
            }
            let adjacency = AdjacencyMask::from_word(mask);
            let Ok(graph) = dag_from_adjacency_mask(adjacency, n) else {
                continue;
            };
            let study = Study::tabular(source.clone())
                .graph(graph)
                .query(query.clone())
                .estimator(EstimatorId::BayesianGcomp)
                .inference(InferenceMode::Bayesian(config.clone()))
                .refute(RefuteSuite::None)
                .build()?;
            let prepared = match study.prepare_deferred(ctx) {
                Ok(prepared) => prepared,
                Err(CausalError::NotIdentified { .. }) => {
                    return Err(GraphImputationError::SupportedGraphNotIdentified { mask });
                }
                Err(error) => return Err(error.into()),
            };
            atoms.push(RetainedAtomBinding { mask, prepared });
        }
        // The consumer re-derives these from each graph; they are the retained operations, not a
        // restatement of the graph.
        let mut operations = Vec::with_capacity(atoms.len());
        for atom in &atoms {
            let RetainedOperation::BayesianGcomp { operation, .. } =
                atom.prepared.retained_operation()
            else {
                return Err(invalid("atom has no retained Bayesian g-computation operation").into());
            };
            operations.push(AtomOperation {
                mask: atom.mask,
                status: operation.identification().status,
                adjustment: operation.estimand().adjustment_set.iter().map(|v| v.raw()).collect(),
            });
        }
        if operations != retained.operations {
            return Err(invalid(
                "a prepared child operation differs from the one the consumer derives",
            )
            .into());
        }
        let parts = GraphImputationParts {
            names: columns.iter().map(|(name, _)| name.to_string()).collect(),
            domains: spec.domains.to_vec(),
            columns: columns.into_iter().map(|(_, values)| values).collect(),
            treatment: spec.treatment.raw(),
            outcome: spec.outcome.raw(),
            support: spec.support,
            imputations: spec.imputation.imputations,
            iterations: spec.imputation.iterations,
            imputation_seed: spec.imputation.seed,
            posterior_draws: spec.posterior_draws,
            prior_scale: spec.prior_scale,
            execution_seed: spec.execution_seed,
            data_digest: source_digest,
            atoms: operations,
            excluded: retained.excluded,
        };
        parts.validate().map_err(invalid)?;
        let lowered = lower(
            LoweringInput {
                source_digest,
                schema_digest,
                rows: source.row_count() as u64,
                variables: Arc::clone(&variables),
                treatment: spec.treatment,
                outcome: spec.outcome,
                support: spec.support,
                contract: parts.contract_digest(),
                excluded: u64::from(parts.excluded),
                imputation: spec.imputation,
                domains: Arc::clone(&spec.domains),
                posterior_draws: spec.posterior_draws,
                prior_scale: spec.prior_scale,
                execution_seed: spec.execution_seed,
                atoms: atoms.clone(),
            },
            &mut budget,
        )?;
        let program = *lowered.identity().as_bytes();
        let source_bytes: Arc<[u8]> =
            Arc::from(GraphImputationSource::new(parts, program).map_err(invalid)?.encode());
        Ok(Self {
            source,
            source_digest,
            _lease: lease,
            atoms,
            lowered: Arc::new(lowered),
            source_bytes,
        })
    }

    /// Execute the retained source and child operations. Each operation is re-derived from its
    /// graph and must equal the one this compiler sealed; the point is model-conditional.
    ///
    /// # Errors
    /// Refuses a changed source, an operation that differs from its graph, failed children,
    /// cancellation, or insufficient resources.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationPoint, GraphImputationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "graph imputation execute" }.into());
        }
        let checked =
            self.source.storage().content_digest_with_context(ctx).map_err(
                |error| match error {
                    antecedent_core::execution::AdmissionError::Cancelled => {
                        CausalError::Cancelled { stage: "graph source identity" }
                    }
                    other => CausalError::Resource { message: other.to_string() },
                },
            )?;
        if checked != self.source_digest {
            return Err(invalid("source snapshot changed").into());
        }
        let point = GraphImputationPoint::replay(&self.source_bytes, ctx)?;
        if point.program != *self.lowered.identity().as_bytes()
            || point.source != self.source_digest
        {
            return Err(invalid("replay belongs to another retained program or source").into());
        }
        Ok(point)
    }

    /// Re-execute from the retained source and require the independently held identity.
    ///
    /// # Errors
    /// Refuses any execution failure or identity mismatch.
    pub fn replay(
        &self,
        expected: [u8; 32],
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationPoint, GraphImputationError> {
        let result = self.execute(ctx)?;
        if result.receipt != expected {
            return Err(invalid("composite receipt does not match").into());
        }
        Ok(result)
    }

    /// The canonical original-source bytes the replay consumes.
    #[cfg(feature = "calibration-internal")]
    fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }
}

/// Measure the whole-method candidate from an original native source under the fixed-effect
/// coordinate.
///
/// This feature-gated harness entry is not an interval support or a public claim.
/// Its bounds and basis are derived from the same compiled source the measured route replays.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn graph_imputation_calibration_candidate(
    source: TabularData,
    spec: &GraphImputationSpec,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, antecedent_core::CalibrationBasis), GraphImputationError> {
    candidate(
        source,
        spec,
        antecedent_io::measured_inference::graph_candidate::CandidateBasis::FixedEffect,
        ctx,
    )
}

/// Measure the prior-predictive candidate the measured route reports.
///
/// The harness draws the data-generating process from the family the basis names. The basis
/// carries the real row count, the variable and graph counts and the declaration, so a record
/// measured over a grid of those covers executions inside it. It is not public inference
/// authority and cannot support a claim without independent scientific review.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn graph_imputation_prior_predictive_calibration_candidate(
    source: TabularData,
    spec: &GraphImputationSpec,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, antecedent_core::CalibrationBasis), GraphImputationError> {
    candidate(
        source,
        spec,
        antecedent_io::measured_inference::graph_candidate::CandidateBasis::PriorPredictive,
        ctx,
    )
}

#[cfg(feature = "calibration-internal")]
fn candidate(
    source: TabularData,
    spec: &GraphImputationSpec,
    kind: antecedent_io::measured_inference::graph_candidate::CandidateBasis,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, antecedent_core::CalibrationBasis), GraphImputationError> {
    let plan = CompiledGraphImputation::compile(source, spec, ctx)?;
    antecedent_io::measured_inference::graph_candidate::calibration_candidate(
        plan.source_bytes(),
        kind,
        ctx,
    )
    .map_err(replay_error)
}

#[cfg(test)]
mod compiled_tests {
    use super::*;
    use antecedent_core::graph_imputation_contract::labeled_dag_count;
    use antecedent_io::measured_inference::graph_candidate::derive_support;

    // Deterministic data. With `confounded`, the covariate drives treatment and the outcome.
    fn table(rows: usize, variables: usize, confounded: bool) -> TabularData {
        let mut state = 41u64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 4];
        for row in 0..rows {
            let covariate = uniform() * 2.0 - 1.0;
            let treated = if confounded {
                f64::from(uniform() < 0.5 + 0.4 * covariate)
            } else {
                f64::from(row % 2 == 1)
            };
            let outcome = 0.5 * treated
                + if confounded { 0.9 * covariate } else { 0.2 * covariate }
                + 0.3 * (uniform() - 0.5);
            columns[0].push(treated);
            columns[1].push(if row % 7 == 3 { f64::NAN } else { outcome });
            columns[2].push(covariate);
            columns[3].push(uniform());
        }
        let names = ["t", "y", "x", "z"];
        TabularData::from_f64_columns(
            (0..variables).map(|index| (names[index], columns[index].as_slice())),
        )
        .unwrap()
    }

    fn spec(support: GraphImputationSupport, variables: usize) -> GraphImputationSpec {
        GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from(&[1u8, 0, 0, 0][..variables]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
        }
    }

    fn compiled(
        support: GraphImputationSupport,
        variables: usize,
        rows: usize,
        confounded: bool,
    ) -> CompiledGraphImputation {
        let ctx = ExecutionContext::for_tests(19);
        CompiledGraphImputation::compile(
            table(rows, variables, confounded),
            &spec(support, variables),
            &ctx,
        )
        .unwrap()
    }

    #[test]
    fn exact_dag_enumeration_matches_known_counts_and_rejects_cycles() {
        for (n, expected) in [(2, 3), (3, 25), (4, 543)] {
            let dags = (0..(1u64 << (n * (n - 1))))
                .filter(|&mask| dag_from_adjacency_mask(AdjacencyMask::from_word(mask), n).is_ok())
                .count();
            assert_eq!(dags, expected);
        }
        assert_eq!(labeled_dag_count(4), 543);
        assert_eq!(labeled_dag_count(5), 29_281);
    }

    #[test]
    fn declared_support_has_exact_expected_graphs() {
        use antecedent_discovery::graph_imputation_enumeration::graph_imputation_supported_masks;
        let count = |n, support| graph_imputation_supported_masks(n, 0, 1, support).len();
        let root = GraphImputationSupport::TreatmentRootDirectEffect;
        let direct = GraphImputationSupport::TreatmentDirectEffect;
        assert_eq!([count(2, root), count(3, root), count(4, root)], [1, 6, 100]);
        assert_eq!([count(2, direct), count(3, direct)], [1, 8]);
        assert!(!graph_imputation_supported_masks(3, 0, 1, root).contains(&0));
    }

    #[test]
    fn supported_three_variable_mixture_runs_from_retained_atoms() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compiled(GraphImputationSupport::TreatmentRootDirectEffect, 3, 12, false);
        assert_eq!(plan.atoms.len(), 6);
        let result = plan.execute(&ctx).unwrap();
        assert!(result.mean.is_finite() && result.diagnostic_sd > 0.0);
        assert!(result.unidentified_by_table.iter().all(|mass| mass.abs() < f64::EPSILON));
    }

    #[test]
    fn root_treatment_atoms_identify_the_same_contrast_with_the_empty_adjustment_set() {
        use super::super::prepared::IdentificationCache;
        let plan = compiled(GraphImputationSupport::TreatmentRootDirectEffect, 3, 12, false);
        assert_eq!(plan.atoms.len(), 6);
        let masks: std::collections::BTreeSet<u64> = plan.atoms.iter().map(|a| a.mask).collect();
        assert_eq!(masks.len(), 6, "atoms are six distinct graphs");
        let mut functionals = Vec::new();
        for atom in &plan.atoms {
            let cache = atom.prepared.study().identification.as_ref().unwrap();
            let IdentificationCache::Static(cache) = cache else {
                panic!("mask {} is not a static identification", atom.mask);
            };
            assert!(
                cache.estimand.adjustment_set.is_empty(),
                "a root treatment needs no adjustment, mask {} adds {:?}",
                atom.mask,
                cache.estimand.adjustment_set
            );
            assert_eq!(
                cache.identification.status,
                antecedent_identify::IdentificationStatus::NonparametricallyIdentified,
                "mask {}",
                atom.mask
            );
            functionals.push(format!("{:?}", cache.estimand.method));
        }
        assert!(functionals.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn a_confounded_treatment_gives_atoms_with_different_adjustment_sets() {
        for variables in [3, 4] {
            let plan = compiled(GraphImputationSupport::TreatmentDirectEffect, variables, 40, true);
            let sets: std::collections::BTreeSet<Vec<u32>> = plan
                .atoms
                .iter()
                .map(|atom| {
                    let RetainedOperation::BayesianGcomp { operation, .. } =
                        atom.prepared.retained_operation()
                    else {
                        panic!("mask {} retains no Bayesian g-computation", atom.mask);
                    };
                    operation.estimand().adjustment_set.iter().map(|v| v.raw()).collect()
                })
                .collect();
            assert!(sets.contains(&Vec::new()), "some graph has an unconfounded treatment");
            assert!(sets.len() > 1, "graph uncertainty must change what is adjusted for: {sets:?}");
            if variables == 4 {
                assert!(plan.atoms.len() > 25, "the four-variable cell exceeds the first cell");
            }
        }
    }

    #[test]
    fn the_consumer_rederives_exactly_the_operations_the_compiler_retained() {
        for (support, variables, confounded) in [
            (GraphImputationSupport::TreatmentRootDirectEffect, 3, false),
            (GraphImputationSupport::TreatmentDirectEffect, 3, true),
            (GraphImputationSupport::TreatmentDirectEffect, 4, true),
        ] {
            let plan = compiled(support, variables, 40, confounded);
            let source = GraphImputationSource::decode(&plan.source_bytes).unwrap();
            assert_eq!(source.program(), *plan.lowered.identity().as_bytes());
            assert_eq!(source.parts().atoms.len(), plan.atoms.len());
            let derived = derive_support(source.parts()).unwrap();
            assert_eq!(derived.operations, source.parts().atoms);
            assert_eq!(derived.excluded, source.parts().excluded);
        }
    }

    #[test]
    fn the_lowered_program_commits_to_the_support_and_its_operations() {
        let root = compiled(GraphImputationSupport::TreatmentRootDirectEffect, 3, 40, true);
        let direct = compiled(GraphImputationSupport::TreatmentDirectEffect, 3, 40, true);
        assert_ne!(root.lowered.identity(), direct.lowered.identity());
        let again = compiled(GraphImputationSupport::TreatmentDirectEffect, 3, 40, true);
        assert_eq!(direct.lowered.identity(), again.lowered.identity());
    }

    #[test]
    fn a_graph_with_two_minimal_sets_is_excluded_by_compiler_and_consumer_alike() {
        use antecedent_discovery::graph_imputation_enumeration::graph_imputation_supported_masks;
        let direct = GraphImputationSupport::TreatmentDirectEffect;
        let plan = compiled(direct, 4, 40, true);
        let structural = graph_imputation_supported_masks(4, 0, 1, direct);
        let source = GraphImputationSource::decode(&plan.source_bytes).unwrap();
        let compiled_masks: Vec<u64> = plan.atoms.iter().map(|atom| atom.mask).collect();
        assert!(structural.contains(&897) && !compiled_masks.contains(&897));
        let excluded = source.parts().excluded as usize;
        assert!(excluded > 0);
        assert_eq!(compiled_masks.len() + excluded, structural.len());
        // The consumer derives the identical retained list and count from the graphs.
        let derived = derive_support(source.parts()).unwrap();
        let consumer_masks: Vec<u64> = derived.operations.iter().map(|op| op.mask).collect();
        assert_eq!(consumer_masks, compiled_masks);
        assert_eq!(derived.excluded, source.parts().excluded);
        // The sealed contract and program commit to the retained list and the count.
        let mut other_count = source.parts().clone();
        other_count.excluded += 1;
        assert_ne!(other_count.contract_digest(), source.parts().contract_digest());
        let mut other_list = source.parts().clone();
        other_list.atoms.pop();
        assert_ne!(other_list.contract_digest(), source.parts().contract_digest());
        // Graphs with a unique minimal set exclude nothing.
        let three = compiled(direct, 3, 40, true);
        assert_eq!(three.atoms.len(), 8);
        assert_eq!(GraphImputationSource::decode(&three.source_bytes).unwrap().parts().excluded, 0);
    }

    #[test]
    fn four_variable_cell_executes_every_retained_graph() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compiled(GraphImputationSupport::TreatmentDirectEffect, 4, 40, true);
        let point = plan.execute(&ctx).unwrap();
        assert!(point.mean.is_finite());
        assert_eq!(point.unidentified_by_table.len(), 2);
    }

    #[test]
    fn retained_execution_replay_and_source_mutation() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compiled(GraphImputationSupport::TreatmentDirectEffect, 3, 40, true);
        let point = plan.execute(&ctx).unwrap();
        assert_eq!(plan.replay(point.receipt, &ctx).unwrap().receipt, point.receipt);
        assert!(plan.replay([0; 32], &ctx).is_err());
        let artifact = plan.export(&point, &ctx).unwrap();
        let consumed = consume(artifact.bytes(), artifact.expectation(), &ctx).unwrap();
        assert_eq!(consumed.receipt, point.receipt);
        assert_eq!(consumed.mean.to_bits(), point.mean.to_bits());
        let mut wrong = artifact.expectation();
        wrong.receipt = [1; 32];
        assert!(consume(artifact.bytes(), wrong, &ctx).is_err());
        let mut changed = artifact.bytes().to_vec();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        assert!(consume(&changed, artifact.expectation(), &ctx).is_err());
        let mut changed_support = artifact.bytes().to_vec();
        let support_offset = b"ANTECEDENT-GRAPH-CANDIDATE\x01".len() + 1 + 4 + 4 + 4;
        changed_support[support_offset] = 0;
        assert!(consume(&changed_support, artifact.expectation(), &ctx).is_err());
    }

    #[test]
    fn a_forged_child_operation_is_refused_at_consumption() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compiled(GraphImputationSupport::TreatmentDirectEffect, 3, 40, true);
        let point = plan.execute(&ctx).unwrap();
        let artifact = plan.export(&point, &ctx).unwrap();
        let source = GraphImputationSource::decode(artifact.bytes()).unwrap();
        let mut parts = source.parts().clone();
        let adjusted = parts.atoms.iter().position(|atom| !atom.adjustment.is_empty()).unwrap();
        parts.atoms[adjusted].adjustment.clear();
        let forged = GraphImputationSource::new(parts, source.program()).unwrap().encode();
        let refusal = consume(&forged, artifact.expectation(), &ctx).err().unwrap();
        assert!(refusal.to_string().contains("retained child operation differs"), "{refusal}");
    }

    #[test]
    fn interval_is_reported_with_the_slot_of_an_not_measured_construction() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compiled(GraphImputationSupport::TreatmentRootDirectEffect, 3, 12, false);
        let point = plan.execute(&ctx).unwrap();
        let (lower, upper) = point.interval().expect("the interval is reported");
        assert!(lower <= point.mean && point.mean <= upper, "{lower} {} {upper}", point.mean);
        let slot = point.calibration().expect("the calibration slot");
        assert_eq!(slot.status.as_str(), "not_measured");
        assert_eq!(slot.reason.as_deref(), Some("estimator_grid_not_measured"));
    }
}
