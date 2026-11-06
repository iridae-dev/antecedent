//! Deferred provider materialization for transport grids.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, Assignment, EmpiricalTableOptions, ExactLawWire, ExactTransportData, ExecutionContext,
    IoError, PreparedStudy, RetainedGridData, SampleSummary, SidLimits, StatisticalOptionsWire,
    TransportGridData, TransportGridQuery, TransportGridState, build_grid, err, grid_provider_data,
    grid_reasoning_view, rebind_snapshots, retain_grid_input, validate_grid,
};

/// Checked grid declarations whose statistical providers are fitted only on execution.
#[derive(Clone, Debug)]
pub struct DeferredTransportGrid {
    query: TransportGridQuery,
    input: Arc<RetainedGridData>,
    seed: u64,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    cache: Arc<super::super::statistical::MaterializationCache<GridProvider>>,
}
#[derive(Clone, Debug)]
struct GridProvider {
    data: ExactTransportData,
    samples: Vec<SampleSummary>,
    options: Option<StatisticalOptionsWire>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl DeferredTransportGrid {
    /// Retain checked proof, finite requests and raw providers without fitting.
    /// # Errors
    /// Invalid declarations, proof, cancellation or resource admission failure.
    pub fn compile(
        query: TransportGridQuery,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        Self::compile_retained(query, retain_grid_input(input)?, None, ctx)
    }
    pub(super) fn compile_retained(
        query: TransportGridQuery,
        input: Arc<RetainedGridData>,
        retained: Option<Arc<antecedent_core::execution::AdmissionLease>>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        super::super::statistical::check_deferred_cancel(ctx)?;
        antecedent_identify::verify_classical_transport(
            &query.diagram,
            query.functional.derivation().query(),
            query.functional.derivation(),
            SidLimits { steps: query.limits.operations, depth: query.limits.depth },
            ctx,
        )
        .map_err(err)?;
        validate_grid(&query.functional, &query.at, query.limits, ctx)?;
        let bytes = match input.as_ref() {
            RetainedGridData::Exact(data) => {
                if data
                    .laws()
                    .iter()
                    .any(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact)
                {
                    return Err(err("empirical laws require statistical transport"));
                }
                super::super::statistical::retained_law_bytes(data)?
            }
            RetainedGridData::Statistical(raw, options) => {
                antecedent_estimate::statistical_transport::check_statistical_resources(
                    raw,
                    &query.functional,
                    options,
                    ctx,
                )
                .map_err(err)?;
                // Validate recorded sample declarations without fitting their numerical laws.
                for sample in &raw.samples {
                    SampleSummary::from_sample(sample)?;
                }
                super::super::statistical::retained_input_bytes(raw)?
            }
        };
        let retention = match retained {
            Some(retention) => retention,
            None => Arc::new(
                ctx.acquire_admission(0, bytes)
                    .map_err(super::super::statistical::deferred_admission_error)?,
            ),
        };
        super::super::statistical::check_deferred_cancel(ctx)?;
        Ok(Self { query, input, seed: ctx.rng.master_seed(), retention, cache: Arc::default() })
    }
    /// Frozen structural request; it does not assert an executed grid identity.
    #[must_use]
    pub fn query(&self) -> &TransportGridQuery {
        &self.query
    }
    /// Structural reasoning before any provider is fitted or grid point executed.
    #[must_use]
    pub fn inspect(&self) -> antecedent_contract::ReasoningView {
        grid_reasoning_view(None, false)
    }
    /// Declared inference settings, if the providers are statistical.
    #[must_use]
    pub fn statistical_options(&self) -> Option<EmpiricalTableOptions> {
        match self.input.as_ref() {
            RetainedGridData::Statistical(_, options) => Some(*options),
            RetainedGridData::Exact(_) => None,
        }
    }
    /// Branch to a new ordered grid, sharing immutable raw evidence and compatible fitted laws.
    /// # Errors
    /// Invalid grid declarations, cancellation or computation budget failure.
    pub fn rebind(&self, at: Vec<Assignment>, ctx: &ExecutionContext) -> Result<Self, IoError> {
        super::super::statistical::check_deferred_cancel(ctx)?;
        validate_grid(&self.query.functional, &at, self.query.limits, ctx)?;
        let mut branch = self.clone();
        branch.query.at = at;
        Ok(branch)
    }
    /// Bind compatible provider snapshots without fitting or executing a target.
    /// # Errors
    /// Changed provider shape/options, invalid proof bindings or admission failure.
    pub fn bind_snapshot(
        &self,
        input: TransportGridData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let input = retain_grid_input(input)?;
        let shape = |input: &RetainedGridData| -> Result<String, IoError> {
            match input {
                RetainedGridData::Statistical(raw, options) => {
                    super::super::statistical::declared_statistical_shape(
                        raw,
                        &self.query.functional,
                        *options,
                    )
                }
                // Snapshot identities, tolerances and table masses are the observations a
                // rebinding replaces; only the declared axes and provider shape must agree.
                RetainedGridData::Exact(data) => {
                    super::super::statistical::statistical_shape_wires(
                        data.laws().iter().map(ExactLawWire::metadata).collect(),
                    )
                }
            }
        };
        if shape(&input)? != shape(&self.input)? {
            return Err(err("transport.reprepare_required"));
        }
        match (self.input.as_ref(), input.as_ref()) {
            (RetainedGridData::Exact(_), RetainedGridData::Exact(_)) => {}
            (RetainedGridData::Statistical(_, old), RetainedGridData::Statistical(_, new))
                if old == new => {}
            _ => return Err(err("transport.reprepare_required: inference settings changed")),
        }
        let mut query = self.query.clone();
        let catalog =
            rebind_snapshots(query.functional.catalog(), |regime| match input.as_ref() {
                RetainedGridData::Exact(data) => data
                    .laws()
                    .iter()
                    .filter(|law| law.regime() == regime)
                    .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity)
                    .collect(),
                RetainedGridData::Statistical(raw, _) => raw
                    .samples
                    .iter()
                    .filter(|sample| sample.regime == regime)
                    .map(|sample| sample.snapshot_identity.as_ref())
                    .chain(
                        raw.supplied
                            .iter()
                            .filter(|law| law.regime() == regime)
                            .map(antecedent_expr::ExactDiscreteLaw::snapshot_identity),
                    )
                    .collect(),
            })?;
        query.functional = query
            .functional
            .derivation()
            .bind_catalog_with_context(
                &catalog,
                SidLimits { steps: query.limits.operations, depth: query.limits.depth },
                ctx,
            )
            .map_err(err)?;
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        Self::compile_retained(query, input, None, &frozen)
    }
    /// Reconfigure an execution stream and physical policy without fitting or copying rows.
    /// # Errors
    /// Invalid declarations, cancelled execution or refused resource bounds.
    pub fn with_seed(&self, seed: u64, ctx: &ExecutionContext) -> Result<Self, IoError> {
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(seed);
        let mut branch = Self::compile_retained(
            self.query.clone(),
            Arc::clone(&self.input),
            Some(Arc::clone(&self.retention)),
            &frozen,
        )?;
        if seed == self.seed {
            branch.cache = Arc::clone(&self.cache);
        }
        Ok(branch)
    }
    /// Materialize finite provider laws under the actual caller's policy and admission pool.
    /// # Errors
    /// Provider fit, cancellation, admission or finite numerical plan failure.
    pub fn materialize(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PreparedStudy<TransportGridState>, IoError> {
        super::super::statistical::check_deferred_cancel(ctx)?;
        self.retention.observe_current();
        let mut frozen = ctx.clone();
        frozen.rng = antecedent_core::RngFactory::from_seed(self.seed);
        let provider = self.cache.materialize(ctx, || {
            let bytes = match self.input.as_ref() {
                RetainedGridData::Exact(data) => {
                    super::super::statistical::retained_law_bytes(data)?
                }
                RetainedGridData::Statistical(raw, _) => {
                    super::super::statistical::declared_provider_bytes(raw, &self.query.functional)?
                }
            };
            let retention = Arc::new(
                ctx.acquire_admission(0, bytes)
                    .map_err(super::super::statistical::deferred_admission_error)?,
            );
            let (data, samples, options) = grid_provider_data(&self.input, &self.query, &frozen)?;
            if super::super::statistical::retained_law_bytes(&data)? > bytes {
                return Err(err("transport retained provider memory budget quote exceeded"));
            }
            super::super::statistical::check_deferred_cancel(ctx)?;
            Ok(super::super::statistical::MaterializedProvider {
                value: GridProvider { data, samples, options, retention: Arc::clone(&retention) },
                bytes,
                retention,
                kernel: ctx.kernel_policy,
            })
        })?;
        let mut prepared = build_grid(
            self.query.clone(),
            Arc::clone(&self.input),
            provider.data,
            provider.samples,
            provider.options,
            &frozen,
        )?;
        prepared.state.retention = vec![Arc::clone(&self.retention), provider.retention];
        super::super::statistical::check_deferred_cancel(ctx)?;
        Ok(prepared)
    }
}
