//! Bounded compiled graph-selection imputation point operation.
//!
//! Unlicensed composite: replay and a calibration candidate are wired; no coverage record exists.
//! No external callback, likelihood or posterior is accepted. An interval must remain closed.
use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::assumption::AssumptionSource;
use antecedent_core::execution::{RngFactory, mix64};
use antecedent_core::{
    AverageEffectQuery, CanonicalEncode, CanonicalWriter, ExecutionContext, VariableId,
};
use antecedent_data::{
    ImputationDomain, ImputationSpec, TableView, TabularData, complete_float_columns,
};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, dag_from_adjacency_mask};
use antecedent_estimate::multiple_imputation::{ImputationColumn, ImputationUnits, Imputer};

use super::graph_imputation_mix::{AtomOutcome, AtomOutcomeRecord, mix_completed_table};
use super::graph_imputation_receipt::{AtomReceipt, AtomStatus, TableReceipt};
use super::lowering::graph_imputation::{LoweringInput, RetainedAtomBinding, lower};
use antecedent_core::search::{SearchBudget, SearchLimits};
use antecedent_discovery::graph_imputation_enumeration::{
    BoundedExactDagError, graph_imputation_exact_dags,
};
mod wire;
pub use wire::{GraphImputationArtifact, GraphImputationExpectation, consume};

use super::missing_imputation::stacked_posterior;
use crate::planner::lowered::LoweredProgram;
use crate::{BayesianConfig, CausalError, EstimatorId, InferenceMode, RefuteSuite, Study};

fn invalid(message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile { message: format!("graph_imputation.candidate_refused: {message}") }
}

#[cfg_attr(
    not(any(test, feature = "calibration-internal")),
    allow(dead_code, reason = "candidate execution is closed outside calibration and tests")
)]
fn imputation_admission_refused(
    error: &antecedent_estimate::multiple_imputation::MultipleImputationError,
) -> bool {
    error.detail().rsplit('.').next() == Some("admission_refused")
}

/// Explicit structural support stated before any data are scored.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphImputationSupport {
    /// Treatment has no parents and a direct arrow to the outcome.
    TreatmentRootDirectEffect,
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
#[cfg_attr(
    not(any(test, feature = "calibration-internal")),
    allow(dead_code, reason = "candidate execution is closed outside calibration and tests")
)]
pub struct CompiledGraphImputation {
    source: TabularData,
    source_digest: [u8; 32],
    spec: GraphImputationSpec,
    variables: Arc<[VariableId]>,
    imputer: Imputer,
    _lease: antecedent_core::execution::AdmissionLease,
    atoms: Vec<RetainedAtomBinding>,
    lowered: Arc<LoweredProgram>,
}

/// A numerical child failure remains distinct from graph non-identification.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationError {
    /// An existing engine refusal or execution error.
    #[error(transparent)]
    Engine(#[from] CausalError),
    /// No scientific point has been licensed for this candidate.
    #[error("graph_imputation.point_unlicensed: the compiled candidate is not licensed")]
    PointUnavailable,
    /// No calibrated interval has been licensed for this composite.
    #[error("graph_imputation.interval_unlicensed: no calibrated interval is available")]
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
    // Never exposed as an interval: retained only for later feature-gated calibration.
    #[cfg(feature = "calibration-internal")]
    posterior: antecedent_estimate::bayesian::CausalPosterior,
    _lease: antecedent_core::execution::AdmissionLease,
}

impl GraphImputationPoint {
    /// Refuse interval requests until a matching coverage license exists.
    ///
    /// # Errors
    /// Always returns the typed unlicensed-interval refusal.
    pub fn interval(&self) -> Result<(f64, f64), GraphImputationError> {
        Err(GraphImputationError::IntervalUnavailable)
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
        spec: GraphImputationSpec,
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
        if !(2..=3).contains(&n)
            || !(4..=10_000).contains(&rows)
            || spec.domains.len() != n
            || spec.domains.iter().any(|&domain| domain > 1)
            || spec.treatment == spec.outcome
            || spec.treatment.raw() as usize >= n
            || spec.outcome.raw() as usize >= n
            || spec.domains.get(spec.treatment.raw() as usize) != Some(&1)
            || !(2..=64).contains(&spec.imputation.imputations)
            || !(1..=64).contains(&spec.imputation.iterations)
            || !(100..=4096).contains(&spec.posterior_draws)
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
        let quote = (rows as u64)
            .checked_mul(n as u64)
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_add(25 * 1_048_576))
            .ok_or_else(|| invalid("plan admission overflow"))?;
        let lease = ctx.acquire_admission(0, quote).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation compile" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        wire::canonical_source(&source, ctx)?;
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
        let mut columns = Vec::with_capacity(n);
        let mut domains = Vec::with_capacity(n);
        for (index, &id) in variables.iter().enumerate() {
            let variable = source.schema().get(id).map_err(CausalError::from)?;
            columns.push(ImputationColumn::new(
                Arc::clone(&variable.name),
                source.float64_values(id).map_err(CausalError::from)?,
            ));
            let domain = match spec.domains[index] {
                0 => ImputationDomain::Continuous,
                1 => ImputationDomain::Binary,
                _ => return Err(invalid("unknown domain").into()),
            };
            domains.push((Arc::clone(&variable.name), domain));
        }
        let imputer =
            Imputer::new_with_domains(columns, spec.imputation, ImputationUnits::Rows, &domains)
                .map_err(|e| CausalError::Estimate(e.into()))?;
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
        let mut budget = SearchBudget::with_memory(
            SearchLimits { operations: 128, depth: 1 },
            25 * 1024 * 1024,
            ctx,
        )
        .map_err(GraphImputationError::SearchStopped)?;
        let mut atoms = Vec::new();
        for mask in 0..(1u64 << (n * (n - 1))) {
            if let Err(stop) = budget.charge(1, 25 * 1024 * 1024) {
                return Err(GraphImputationError::SearchStopped(budget.receipt(
                    stop,
                    vec![format!("compiled:0..{mask}")],
                    vec![format!("remaining:{mask}..{}", 1u64 << (n * (n - 1)))],
                )));
            }
            let adjacency = AdjacencyMask::from_word(mask);
            if !antecedent_discovery::graph_imputation_enumeration::graph_imputation_supported_mask(
                &adjacency,
                n,
                spec.treatment.raw() as usize,
                spec.outcome.raw() as usize,
            ) {
                continue;
            }
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
        let lowered = lower(
            LoweringInput {
                source_digest,
                schema_digest,
                rows: source.row_count() as u64,
                variables: Arc::clone(&variables),
                treatment: spec.treatment,
                outcome: spec.outcome,
                support: spec.support,
                imputation: spec.imputation,
                domains: Arc::clone(&spec.domains),
                posterior_draws: spec.posterior_draws,
                prior_scale: spec.prior_scale,
                execution_seed: spec.execution_seed,
                atoms: atoms.clone(),
            },
            &mut budget,
        )?;
        Ok(Self {
            source,
            source_digest,
            spec,
            variables,
            imputer,
            _lease: lease,
            atoms,
            lowered: Arc::new(lowered),
        })
    }

    /// The point executor uses only the already retained composite and child programs.
    ///
    /// # Errors
    /// Refuses failed atoms, invalid native results, cancellation, or insufficient resources.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationPoint, GraphImputationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "graph imputation execute" }.into());
        }
        Err(GraphImputationError::PointUnavailable)
    }

    // Internal measurement and conformance only; never promoted to public authority.
    #[cfg_attr(
        not(any(test, feature = "calibration-internal")),
        allow(dead_code, reason = "candidate execution is closed outside calibration and tests")
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "the composite seals each step and its resource and provenance receipts"
    )]
    fn execute_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationPoint, GraphImputationError> {
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
        let parent = self.lowered.identity();
        let scoped = ctx.enter_lowered_dispatch(parent);
        let quote = u64::from(self.spec.posterior_draws)
            .checked_mul(u64::from(self.spec.imputation.imputations))
            .and_then(|n| n.checked_mul(32))
            .ok_or_else(|| invalid("posterior admission overflow"))?;
        let lease = scoped.acquire_admission(0, quote).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation execution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut budget = SearchBudget::with_memory(
            SearchLimits { operations: 4096, depth: 1 },
            25 * 1024 * 1024,
            &scoped,
        )
        .map_err(GraphImputationError::SearchStopped)?;
        let mut completed_posteriors = Vec::new();
        let mut table_leases = Vec::new();
        let mut tables = Vec::new();
        for index in 0..self.spec.imputation.imputations {
            let completed = self.imputer.impute(index, &scoped).map_err(|e| {
                if scoped.cancellation.is_cancelled() {
                    CausalError::Cancelled { stage: "graph imputation" }
                } else if imputation_admission_refused(&e) {
                    CausalError::Resource { message: e.to_string() }
                } else {
                    CausalError::Estimate(e.into())
                }
            })?;
            let replacements: Vec<_> = self
                .variables
                .iter()
                .zip(completed.columns())
                .map(|(&id, column)| (id, Arc::clone(column)))
                .collect();
            let data =
                complete_float_columns(&self.source, &replacements).map_err(CausalError::from)?;
            let completed_digest = data.storage().content_digest_with_context(&scoped).map_err(
                |error| match error {
                    antecedent_core::execution::AdmissionError::Cancelled => {
                        CausalError::Cancelled { stage: "completed input identity" }
                    }
                    other => CausalError::Resource { message: other.to_string() },
                },
            )?;
            let graphs = graph_imputation_exact_dags(
                &data,
                &self.variables,
                self.spec.treatment,
                self.spec.outcome,
                &mut budget,
            )
            .map_err(|error| match error {
                BoundedExactDagError::Search(receipt) => {
                    GraphImputationError::SearchStopped(receipt)
                }
                BoundedExactDagError::Unscorable { count } => {
                    GraphImputationError::Unscorable { table: index, count }
                }
                BoundedExactDagError::Discovery(error) => CausalError::Discovery(error).into(),
            })?;
            let mut weights = BTreeMap::<u64, f64>::new();
            for (mask, &weight) in graphs.atoms().iter().zip(graphs.weights().iter()) {
                let mask = mask.to_word().ok_or_else(|| invalid("graph exceeds bounded mask"))?;
                *weights.entry(mask).or_default() += weight;
            }
            let mut outcomes = Vec::new();
            let mut receipts = Vec::new();
            for (mask, weight) in weights {
                if weight == 0.0 {
                    continue;
                }
                let atom = self
                    .atoms
                    .iter()
                    .find(|a| a.mask == mask)
                    .ok_or_else(|| invalid("discovery returned an uncompiled graph"))?;
                let child_program = atom
                    .prepared
                    .plan()
                    .lowered()
                    .ok_or_else(|| invalid("atom program is absent"))?
                    .identity();
                let cache = atom
                    .prepared
                    .study()
                    .identification
                    .as_ref()
                    .ok_or_else(|| invalid("atom identification is absent"))?;
                let proof = crate::analysis::identification_seal::seal_cache(cache)?
                    .ok_or_else(|| invalid("atom identification is unsealable"))?;
                let static_cache = atom
                    .prepared
                    .study()
                    .static_cache()
                    .ok_or_else(|| invalid("atom lacks static identification"))?;
                if !crate::analysis::execute::identification_status_ok_for_case(
                    static_cache.identification.status,
                ) {
                    receipts.push(AtomReceipt {
                        graph_mask: mask,
                        program: None,
                        identification: proof,
                        weight,
                        status: AtomStatus::Unidentified,
                        execution: None,
                    });
                    outcomes.push(AtomOutcomeRecord {
                        graph_mask: mask,
                        weight,
                        outcome: AtomOutcome::Unidentified,
                    });
                    continue;
                }
                let mut child_ctx = scoped.clone();
                child_ctx.rng = RngFactory::from_seed(mix64(
                    self.spec.execution_seed ^ (u64::from(index) << 8) ^ mask,
                ));
                let child = match atom
                    .prepared
                    .execute_compiled_bayesian_atom(&data, parent, &child_ctx)
                {
                    Ok(result) => result,
                    Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. })) => {
                        return Err(error.into());
                    }
                    Err(_error) if child_ctx.cancellation.is_cancelled() => {
                        return Err(CausalError::Cancelled { stage: "graph atom execution" }.into());
                    }
                    Err(error) => {
                        outcomes.push(AtomOutcomeRecord {
                            graph_mask: mask,
                            weight,
                            outcome: AtomOutcome::Failed {
                                reason: error.to_string().chars().take(128).collect(),
                            },
                        });
                        continue;
                    }
                };
                let posterior = child
                    .posterior
                    .as_ref()
                    .ok_or_else(|| invalid("atom returned no effect posterior"))?;
                if !crate::analysis::execute::identification_status_ok_for_case(
                    posterior.identification,
                ) || posterior.unidentified_mass != 0.0
                    || posterior.unevaluable_mass != 0.0
                    || posterior.subsampled_out_mass != 0.0
                {
                    return Err(invalid(
                        "child returned unsupported identification or incomplete mass",
                    )
                    .into());
                }
                let column = posterior
                    .effect_column()
                    .ok_or_else(|| invalid("atom returned no effect column"))?;
                let values = posterior.draws.column(column).map_err(invalid)?;
                let mut execution =
                    CanonicalWriter::new("antecedent.graph_imputation.child_execution.v1");
                execution
                    .bytes(child_program.as_bytes())
                    .bytes(&completed_digest)
                    .word(values.len() as u64);
                for value in values {
                    execution.word(value.to_bits());
                }
                execution.word(posterior.assumptions.entries.len() as u64);
                for record in &posterior.assumptions.entries {
                    record.assumption.encode(&mut execution);
                    record.scope.encode(&mut execution);
                    execution.text(record.status.as_str());
                    match &record.source {
                        AssumptionSource::UserDeclared => {
                            execution.word(0);
                        }
                        AssumptionSource::AlgorithmDefault { algorithm } => {
                            execution.word(1).text(algorithm);
                        }
                        AssumptionSource::Artifact => {
                            execution.word(2);
                        }
                        AssumptionSource::Derived { from } => {
                            execution.word(3).text(from);
                        }
                    }
                }
                receipts.push(AtomReceipt {
                    graph_mask: mask,
                    program: Some(*child_program.as_bytes()),
                    identification: proof,
                    weight,
                    status: AtomStatus::Identified,
                    execution: Some(execution.finish()),
                });
                outcomes.push(AtomOutcomeRecord {
                    graph_mask: mask,
                    weight,
                    outcome: AtomOutcome::Identified {
                        effect_draws: Arc::from(values),
                        assumptions: posterior.assumptions.clone(),
                    },
                });
            }
            let mixed = mix_completed_table(&outcomes, &scoped).map_err(|error| match error {
                super::graph_imputation_mix::MixError::Cancelled => {
                    CausalError::Cancelled { stage: "graph mixture" }.into()
                }
                super::graph_imputation_mix::MixError::Resource(message) => {
                    CausalError::Resource { message }.into()
                }
                super::graph_imputation_mix::MixError::Unavailable {
                    failed_mass,
                    skipped_mass,
                    reasons,
                } => GraphImputationError::AtomFailures {
                    table: index,
                    failed_mass,
                    skipped_mass,
                    reasons,
                },
                super::graph_imputation_mix::MixError::Invalid(message) => invalid(message).into(),
                super::graph_imputation_mix::MixError::Estimation(message) => {
                    invalid(message).into()
                }
            })?;
            if mixed.posterior.unidentified_mass > 0.0 {
                return Err(GraphImputationError::Unidentified {
                    table: index,
                    mass: mixed.posterior.unidentified_mass,
                });
            }
            completed_posteriors.push(mixed.posterior.clone());
            table_leases.push(mixed);
            tables.push(TableReceipt { index, completed_input: completed_digest, atoms: receipts });
        }
        let receipt = super::graph_imputation_receipt::seal(
            *parent.as_bytes(),
            self.source_digest,
            self.spec.imputation.imputations,
            &tables,
        )
        .map_err(invalid)?;
        let pooled = stacked_posterior(&completed_posteriors)?;
        let mean = *pooled.summaries.mean.first().ok_or_else(|| invalid("missing pooled mean"))?;
        let diagnostic_sd =
            *pooled.summaries.sd.first().ok_or_else(|| invalid("missing pooled SD"))?;
        Ok(GraphImputationPoint {
            mean,
            diagnostic_sd,
            receipt: receipt.identity,
            program: *parent.as_bytes(),
            source: self.source_digest,
            unidentified_by_table: receipt.masses.iter().map(|m| m[0]).collect(),
            #[cfg(feature = "calibration-internal")]
            posterior: pooled,
            _lease: lease,
        })
    }

    /// Calibration-only candidate, derived solely from this retained execution. This is
    /// not a public interval or a coverage claim; no record is created here.
    #[cfg(feature = "calibration-internal")]
    fn calibration_candidate(
        &self,
        point: &GraphImputationPoint,
        ctx: &ExecutionContext,
    ) -> Result<(f64, f64, antecedent_core::CalibrationBasis), CausalError> {
        if point.program != *self.lowered.identity().as_bytes()
            || point.source != self.source_digest
            || point.unidentified_by_table.iter().any(|&mass| mass != 0.0)
        {
            return Err(invalid(
                "calibration candidate requires this producer and fully identified mass",
            ));
        }
        let column =
            point.posterior.effect_column().ok_or_else(|| invalid("missing effect draws"))?;
        let draws = point.posterior.draws.column(column).map_err(invalid)?;
        let bytes = (draws.len() as u64)
            .checked_mul(8)
            .ok_or_else(|| invalid("candidate size overflow"))?;
        let _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation calibration candidate" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        if draws.len() < 2 || draws.iter().any(|x| !x.is_finite()) {
            return Err(invalid("candidate draws must be finite"));
        }
        let mut sorted = draws.to_vec();
        sorted.sort_by(f64::total_cmp);
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "graph imputation calibration candidate" });
        }
        // Fixed nearest-rank 95% interval; the harness must measure this exact algorithm.
        let rank = |numerator: usize| ((sorted.len() * numerator + 999) / 1000).saturating_sub(1);
        let functional = format!(
            "graph_imputation.v2:support=treatment_root_direct_effect:t={}:y={}:domains={:?}:m={}:sweeps={}:draws={}:prior_bits={:016x}:imputation_seed={}:execution_seed={}",
            self.spec.treatment.raw(),
            self.spec.outcome.raw(),
            self.spec.domains,
            self.spec.imputation.imputations,
            self.spec.imputation.iterations,
            self.spec.posterior_draws,
            self.spec.prior_scale.to_bits(),
            self.spec.imputation.seed,
            self.spec.execution_seed
        );
        let a = |s: &str| Arc::<str>::from(s);
        let basis = antecedent_core::CalibrationBasis::new(
            [
                a("AverageEffect"),
                a("Dag"),
                a("graph_posterior"),
                a("tabular"),
                a("GraphImputationCompositeV2"),
                a("graph_imputation.bic_conjugate"),
                a("nearest_rank_95"),
                a(""),
                a("iid"),
                a("mar_bic_graph_mixture_conjugate"),
                Arc::from(functional),
            ],
            0.95,
            a("point"),
            self.source.row_count() as u64,
            None,
            Some(self.spec.posterior_draws * self.spec.imputation.imputations),
            0.0,
        )
        .with_validation("none");
        Ok((sorted[rank(25)], sorted[rank(975)], basis))
    }

    /// Re-execute from the retained native plan and require the independently held identity.
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
}

/// Measure the closed whole-method candidate from an original native source.
///
/// This feature-gated harness entry is not an interval license or a public claim.
/// Its bounds and basis are derived from the same retained compiled execution.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
pub fn graph_imputation_calibration_candidate(
    source: TabularData,
    spec: GraphImputationSpec,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, antecedent_core::CalibrationBasis), GraphImputationError> {
    let plan = CompiledGraphImputation::compile(source, spec, ctx)?;
    let point = plan.execute_candidate(ctx)?;
    Ok(plan.calibration_candidate(&point, ctx)?)
}

#[cfg(test)]
mod compiled_tests {
    use super::*;

    #[test]
    fn exact_dag_enumeration_matches_known_counts_and_rejects_cycles() {
        for (n, expected) in [(2, 3), (3, 25)] {
            let dags: Vec<_> = (0..(1u64 << (n * (n - 1))))
                .filter(|&mask| dag_from_adjacency_mask(AdjacencyMask::from_word(mask), n).is_ok())
                .collect();
            assert_eq!(dags.len(), expected);
            assert!(dags.contains(&0));
            assert!(dags.len() < (1 << (n * (n - 1))));
        }
    }

    #[test]
    fn declared_support_has_exact_expected_graphs() {
        use antecedent_discovery::graph_imputation_enumeration::graph_imputation_supported_mask;
        for (n, expected) in [(2, 1), (3, 6)] {
            let masks: Vec<_> = (0..(1u64 << (n * (n - 1))))
                .filter(|&mask| {
                    let adjacency = AdjacencyMask::from_word(mask);
                    graph_imputation_supported_mask(&adjacency, n, 0, 1)
                        && dag_from_adjacency_mask(adjacency, n).is_ok()
                })
                .collect();
            assert_eq!(masks.len(), expected);
            assert!(!masks.contains(&0));
        }
    }

    #[test]
    fn supported_three_variable_mixture_runs_from_retained_atoms() {
        let t = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.];
        let x = [-0.6, 0.2, 1.4, -0.4, 0.8, -1.3, 1.7, -0.8, 0.5, 1.1, -0.2, 0.7];
        let y = [0.2, 1.3, -0.1, 0.8, 0.3, 1.0, f64::NAN, 1.4, -0.3, 0.9, 0.1, 1.1];
        let source = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let spec = GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphImputationSupport::TreatmentRootDirectEffect,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from([1, 0, 0]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
        };
        let ctx = ExecutionContext::for_tests(19);
        let plan = CompiledGraphImputation::compile(source, spec, &ctx).unwrap();
        assert_eq!(plan.atoms.len(), 6);
        let result = plan.execute_candidate(&ctx).unwrap();
        assert!(result.mean.is_finite());
        assert!(result.unidentified_by_table.iter().all(|mass| mass.abs() < f64::EPSILON));
    }

    #[test]
    fn retained_execution_replay_and_source_mutation() {
        let t = [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.];
        let y = [0.2, 1.3, -0.1, 0.8, 0.3, 1.0, f64::NAN, 1.4, -0.3, 0.9, 0.1, 1.1];
        let source =
            TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
        let spec = GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphImputationSupport::TreatmentRootDirectEffect,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from([1, 0]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
        };
        let ctx = ExecutionContext::for_tests(19);
        let plan = CompiledGraphImputation::compile(source, spec, &ctx).unwrap();
        assert_eq!(plan.atoms.len(), 1);
        let point = plan.execute_candidate(&ctx).unwrap();
        assert!(matches!(point.interval(), Err(GraphImputationError::IntervalUnavailable)));
        assert!(matches!(plan.execute(&ctx), Err(GraphImputationError::PointUnavailable)));
        assert!(matches!(
            plan.replay(point.receipt, &ctx),
            Err(GraphImputationError::PointUnavailable)
        ));
        let artifact = plan.export(&point, &ctx).unwrap();
        assert_eq!(
            wire::consume_candidate(artifact.bytes(), artifact.expectation(), &ctx)
                .unwrap()
                .receipt,
            point.receipt
        );
        let mut changed = artifact.bytes().to_vec();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        assert!(wire::consume_candidate(&changed, artifact.expectation(), &ctx).is_err());
        let mut changed_support = artifact.bytes().to_vec();
        let support_offset = b"ANTECEDENT-GRAPH-IMPUTATION\x02".len() + 1 + 4 + 4 + 4;
        changed_support[support_offset] = 0;
        assert!(wire::consume_candidate(&changed_support, artifact.expectation(), &ctx).is_err());
    }
}
