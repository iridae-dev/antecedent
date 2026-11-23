use super::{
    Arc, Assignment, AssumptionSet, CausalExprArena, CausalSchema, CompiledEvaluator, DomainRef,
    EmpiricalDistributionFactorSnapshot, EmpiricalProviderProvenance, EmpiricalTableProvider,
    EncodedColumns, EstimandMethod, EstimationError, EvalContext, EvalError, ExecutionContext,
    FactorKey, FactorRequirement, FactorSignature, FunctionalDistributionWorkspace,
    FunctionalProgram, HashSet, IdentifiedEstimand, OverlapPolicy, ProgramEvaluator, RowResampling,
    StreamDomain, TableView, TabularData, VariableId, average_over_free_variables,
    build_empirical_provider, collect_factor_signatures, collect_observational_factors, eval_err,
    functional_free_variables, functional_posterior, functional_program,
    provider_from_bayesian_bootstrap, provider_from_columns, require_method, resample_rows,
};

/// Prepared scalar functional (ATE / path-specific NE contrast).
#[derive(Clone, Debug)]
pub struct PreparedFunctionalEffect {
    semantic_schema: CausalSchema,
    vars_needed: HashSet<VariableId>,
    /// Identified estimand.
    pub estimand: IdentifiedEstimand,
    /// Arena owning the functional. Shared (never mutated after `prepare`);
    /// see [`PreparedFunctionalDistribution::arena`](super::PreparedFunctionalDistribution::arena).
    pub arena: Arc<CausalExprArena>,
    /// Compiled evaluator.
    pub compiled: CompiledEvaluator,
    /// Checked owner used by execution; the legacy public arena/evaluator fields
    /// above are retained for low-level API compatibility only.
    program: FunctionalProgram,
    program_evaluator: ProgramEvaluator,
    /// Empirical CPT provider.
    pub provider: EmpiricalTableProvider,
    /// Assumptions from identification.
    pub assumptions: AssumptionSet,
    /// Free variables of the scalar functional (see the module docs).
    pub free_variables: Arc<[VariableId]>,
    pub(super) bootstrap_columns: EncodedColumns,
    pub(super) bootstrap_factors: Vec<FactorKey>,
    pub(super) bootstrap_signatures: Vec<FactorSignature>,
    /// Input row count for the currently bound empirical law.
    source_rows: usize,
}

impl PreparedFunctionalEffect {
    /// Rebind empirical factors without reconstructing the checked causal program.
    /// # Errors
    /// Changed semantic schema or unavailable complete-case factors.
    pub fn rebind_checked(&self, data: &TabularData) -> Result<Self, EstimationError> {
        if data.schema() != &self.semantic_schema {
            return Err(EstimationError::data_msg(
                "functional effect refresh changed semantic schema",
            ));
        }
        let (provider, columns) = build_empirical_provider(
            data,
            &self.vars_needed,
            &self.bootstrap_factors,
            &self.bootstrap_signatures,
        )?;
        Ok(Self {
            semantic_schema: self.semantic_schema.clone(),
            vars_needed: self.vars_needed.clone(),
            estimand: self.estimand.clone(),
            arena: Arc::clone(&self.arena),
            compiled: self.compiled.clone(),
            program: self.program.clone(),
            program_evaluator: self.program_evaluator.clone(),
            provider,
            assumptions: self.assumptions.clone(),
            free_variables: Arc::clone(&self.free_variables),
            bootstrap_columns: columns,
            bootstrap_factors: self.bootstrap_factors.clone(),
            bootstrap_signatures: self.bootstrap_signatures.clone(),
            source_rows: data.row_count(),
        })
    }
    /// Checked expression owner used for every numerical evaluation.
    #[must_use]
    pub fn program(&self) -> &FunctionalProgram {
        &self.program
    }

    /// Variables whose empirical laws the functional reads (the factors' variables and
    /// conditioning variables and the extra variables the caller declared), ascending.
    #[must_use]
    pub fn variables_read(&self) -> Vec<VariableId> {
        let mut variables: Vec<VariableId> = self.vars_needed.iter().copied().collect();
        variables.sort_unstable();
        variables
    }

    /// Export every empirical law needed to independently evaluate this
    /// checked scalar functional, including its free-variable weighting law.
    pub fn factor_snapshot(&self) -> Result<EmpiricalDistributionFactorSnapshot, EstimationError> {
        let mut requirements = self.program.factor_requirements().to_vec();
        if !self.free_variables.is_empty() {
            requirements.push(FactorRequirement {
                variables: Arc::clone(&self.free_variables),
                conditioned_on: Arc::from([]),
                intervention: Arc::from([]),
                domain: DomainRef::Observational,
                population: Arc::from(""),
                regime: None,
            });
        }
        let provider = self.provider.snapshot_factors(&requirements)?;
        Ok(EmpiricalDistributionFactorSnapshot {
            requirements: Arc::from(requirements),
            provider,
            provenance: EmpiricalProviderProvenance {
                provider: "empirical_table",
                source_rows: self.source_rows,
                complete_case_rows: self.bootstrap_columns.n(),
                missing_row_policy: "joint_complete_case",
            },
        })
    }

    /// Value of the scalar functional on `provider`'s row law, with the functional's free
    /// variables resolved as the module docs describe.
    ///
    /// # Errors
    ///
    /// Evaluation failure; a cell error when no value of the free variables is supported.
    pub fn evaluate(&self, provider: &EmpiricalTableProvider) -> Result<f64, EvalError> {
        let ctx = EvalContext::default();
        average_over_free_variables(provider, &self.free_variables, &Assignment::new(), |env| {
            self.program_evaluator.evaluate_with(provider, &ctx, env).map(|v| vec![v])
        })
        .map(|values| {
            antecedent_expr::execution_counts::note_static_work(
                antecedent_expr::execution_counts::StaticWork::Integration,
            );
            values[0]
        })
    }
}

/// Add the marginal law of the free variables to the factors built from the data, so its
/// empirical (or reweighted) masses are available as averaging weights.
pub(super) fn push_free_variable_law(factors: &mut Vec<FactorKey>, free: &[VariableId]) {
    if free.is_empty() {
        return;
    }
    let law: (Arc<[VariableId]>, Arc<[VariableId]>) = (Arc::from(free), Arc::from([]));
    if !factors.contains(&law) {
        factors.push(law);
    }
}

/// Discrete plug-in estimator for identified scalar functionals (contrasts).
#[derive(Clone, Debug)]
pub struct FunctionalEffect {
    /// Overlap policy.
    pub overlap: OverlapPolicy,
    /// Bootstrap replicates for the scalar SE (0 = skip).
    pub bootstrap_replicates: u32,
    /// How rows are resampled or reweighted.
    pub row_resampling: RowResampling,
    /// The sampling design the caller declared; a draw the design does not allow is refused.
    pub sampling_design: crate::util::SamplingDeclaration,
}

impl Default for FunctionalEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalEffect {
    /// Create with explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            overlap: OverlapPolicy::ExplicitOverride,
            bootstrap_replicates: 0,
            row_resampling: RowResampling::Rows,
            sampling_design: crate::util::SamplingDeclaration::Undeclared,
        }
    }

    /// Set the overlap policy recorded on the estimate artifact.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    crate::inference::with_bootstrap_replicates!();

    /// Resample (or reweight) rows as `resampling` under the declared `design`.
    #[must_use]
    pub const fn with_row_resampling(
        mut self,
        resampling: RowResampling,
        design: crate::util::SamplingDeclaration,
    ) -> Self {
        self.row_resampling = resampling;
        self.sampling_design = design;
        self
    }

    fn resample(
        &self,
        ctx: &ExecutionContext,
        stream_base: u64,
        n: usize,
        estimate: impl Fn(&[usize]) -> Result<Option<f64>, EstimationError> + Sync,
    ) -> Result<crate::util::BootstrapSeResult, EstimationError> {
        resample_rows(
            (self.bootstrap_replicates, self.row_resampling, self.sampling_design),
            ctx,
            stream_base,
            n,
            estimate,
        )
    }

    /// Prepare CPT plug-in for a path-specific / general-ID contrast functional.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, continuous columns, or compile failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        arena: &CausalExprArena,
        assumptions: AssumptionSet,
        extra_vars: &[VariableId],
    ) -> Result<PreparedFunctionalEffect, EstimationError> {
        require_method(
            estimand,
            &[EstimandMethod::PathSpecificNatural, EstimandMethod::GeneralId],
            "functional.effect requires path_specific.natural or general.id",
        )?;
        let mut factor_specs = collect_observational_factors(arena, estimand.functional);
        let signatures = collect_factor_signatures(arena, estimand.functional);
        // A scalar functional binds nothing from outside: every free variable is a
        // parameter of the identity.
        let free_variables = functional_free_variables(arena, estimand.functional, &[]);
        push_free_variable_law(&mut factor_specs, &free_variables);
        let mut vars_needed = HashSet::new();
        for (vars, cond) in &factor_specs {
            vars_needed.extend(vars.iter().copied());
            vars_needed.extend(cond.iter().copied());
        }
        vars_needed.extend(extra_vars.iter().copied());
        let (provider, columns) =
            build_empirical_provider(data, &vars_needed, &factor_specs, &signatures)?;
        let program = functional_program(data, arena, estimand.functional)?;
        let program_evaluator = program.compile().map_err(eval_err)?;
        let compiled = program.arena().compile(estimand.functional).map_err(eval_err)?;
        Ok(PreparedFunctionalEffect {
            semantic_schema: data.schema().clone(),
            vars_needed,
            estimand: estimand.clone(),
            arena: Arc::new(arena.clone()),
            compiled,
            program,
            program_evaluator,
            provider,
            assumptions,
            free_variables: Arc::from(free_variables),
            bootstrap_columns: columns,
            bootstrap_factors: factor_specs,
            bootstrap_signatures: signatures,
            source_rows: data.row_count(),
        })
    }

    /// Evaluate the scalar functional.
    ///
    /// # Errors
    ///
    /// Evaluation / missing CPT entries.
    pub fn estimate(
        &self,
        prepared: &PreparedFunctionalEffect,
        _workspace: &mut FunctionalDistributionWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<crate::adjustment::EffectEstimate, EstimationError> {
        let ate = prepared.evaluate(&prepared.provider).map_err(eval_err)?;
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            let n = prepared.bootstrap_columns.n();
            Some(self.resample(ctx, 0xF02D_u64, n, |idx| {
                let columns = prepared.bootstrap_columns.gather(idx);
                let provider = provider_from_columns(
                    &columns,
                    idx.len(),
                    &prepared.bootstrap_factors,
                    &prepared.bootstrap_signatures,
                    None,
                )?;
                match prepared.evaluate(&provider) {
                    Ok(v) if v.is_finite() => Ok(Some(v)),
                    _ => Ok(None),
                }
            })?)
        };
        let estimate = crate::adjustment::EffectEstimate::new(
            ate,
            // Multinomial delta-method analytic SE is out of scope for the discrete plug-in.
            f64::NAN,
            prepared.assumptions.clone(),
            self.overlap,
        )
        .with_bootstrap(boot);
        // Blocks of one series' consecutive rows: the SE is a row-block bootstrap SE.
        Ok(match self.row_resampling {
            RowResampling::SeriesBlocks | RowResampling::WindowBlocks { .. }
                if self.bootstrap_replicates > 0 =>
            {
                estimate.with_block_family(crate::temporal_block::CircularBlockFamily::RowBlock)
            }
            _ => estimate,
        })
    }

    /// Replicate values of several prepared functionals on the same resampled rows.
    ///
    /// Row `b` of the result holds, in input order, each functional's refit on replicate
    /// `b`, so the columns are the shared replicates a class-level identified-set interval
    /// reads. A replicate where any functional fails to evaluate is dropped for all of them.
    /// Empty when no replicates are requested.
    ///
    /// # Errors
    ///
    /// Prepared functionals over different row counts, or a resampling the design refuses.
    pub fn shared_replicate_draws(
        &self,
        prepared: &[&PreparedFunctionalEffect],
        ctx: &ExecutionContext,
    ) -> Result<Vec<Vec<f64>>, EstimationError> {
        let Some(first) = prepared.first() else {
            return Ok(Vec::new());
        };
        if self.bootstrap_replicates == 0 {
            return Ok(Vec::new());
        }
        let n = first.bootstrap_columns.n();
        if prepared.iter().any(|p| p.bootstrap_columns.n() != n) {
            return Err(EstimationError::data_msg(
                "shared replicates need functionals prepared on the same rows",
            ));
        }
        let draws = std::sync::Mutex::new(Vec::<Vec<f64>>::new());
        self.resample(ctx, 0xF02D_u64, n, |idx| {
            let mut row = Vec::with_capacity(prepared.len());
            for p in prepared {
                let columns = p.bootstrap_columns.gather(idx);
                let provider = provider_from_columns(
                    &columns,
                    idx.len(),
                    &p.bootstrap_factors,
                    &p.bootstrap_signatures,
                    None,
                )?;
                match p.evaluate(&provider) {
                    Ok(v) if v.is_finite() => row.push(v),
                    _ => return Ok(None),
                }
            }
            let head = row[0];
            draws
                .lock()
                .map_err(|_| EstimationError::data_msg("shared replicate store poisoned"))?
                .push(row);
            Ok(Some(head))
        })?;
        draws.into_inner().map_err(|_| EstimationError::data_msg("shared replicate store poisoned"))
    }

    /// Bayesian bootstrap of the identified functional. All CPT factors are
    /// derived from a shared Rubin (1981) Dirichlet(1, ..., 1) draw over
    /// empirical rows.
    /// Unobserved joint cells retain zero mass; this is an empirical-support
    /// posterior, not a positive pseudocount prior over unobserved categories.
    ///
    /// # Errors
    ///
    /// Evaluation / missing CPT entries, or a non-finite posterior.
    pub fn estimate_bayesian(
        &self,
        prepared: &PreparedFunctionalEffect,
        n_draws: usize,
        identification: antecedent_core::IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<crate::CausalPosterior, EstimationError> {
        let n = prepared.bootstrap_columns.n();
        if n == 0 {
            return Err(EstimationError::data_msg(
                "functional Bayesian requires complete discrete rows",
            ));
        }
        let draws = crate::require_bayesian_n_draws(n_draws)?;
        let _admitted = crate::util::admit_array::<f64>(ctx, draws, 1)?;
        let mut values = Vec::with_capacity(draws);
        let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, 0xF02E_u64);
        for _ in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("functional Bayesian cancelled"));
            }
            let provider = provider_from_bayesian_bootstrap(
                &prepared.bootstrap_columns,
                n,
                &prepared.bootstrap_factors,
                &prepared.bootstrap_signatures,
                (self.row_resampling.block_length(n), &mut rng),
            )?;
            let value = prepared.evaluate(&provider).map_err(eval_err)?;
            if !value.is_finite() {
                return Err(EstimationError::stats_msg("functional Bayesian draw was non-finite"));
            }
            values.push(value);
        }
        functional_posterior(values, prepared.assumptions.clone(), identification)
    }

    /// Check that several prepared functionals read the same rows, which is what makes their
    /// Bayesian-bootstrap posteriors and bootstrap replicates paired draw by draw: every
    /// functional consumes the same seeded stream of row weights (or row indices) over the
    /// same `n`, so draw `k` of each is evaluated at the same resampled table.
    ///
    /// # Errors
    ///
    /// Prepared functionals over different complete-row counts.
    pub fn assert_same_rows(
        &self,
        prepared: &[&PreparedFunctionalEffect],
    ) -> Result<(), EstimationError> {
        let Some(first) = prepared.first() else {
            return Ok(());
        };
        let n = first.bootstrap_columns.n();
        if prepared.iter().any(|p| p.bootstrap_columns.n() != n) {
            return Err(EstimationError::data_msg(
                "paired draws need functionals prepared on the same complete rows",
            ));
        }
        Ok(())
    }
}
