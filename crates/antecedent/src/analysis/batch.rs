//! Batch multi-query: one table, N average-effect estimates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseQuery,
};
use antecedent_data::{TableView, TabularData, ValidityBitmap};
use antecedent_graph::Dag;

use crate::error::CausalError;
use crate::result::StudyResult;

use super::builder::RefuteSuite;
use super::execute::Study;
use super::latency::LatencyMode;
use super::prepared::PreparedStudy;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// How a batch family picked its surfaced candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateProcedure {
    /// Largest `|estimate|/SE` under the shared-row max-t family.
    MaxT,
    /// Smallest Benjamini–Hochberg q-value.
    BenjaminiHochberg,
    /// Smallest Benjamini–Yekutieli q-value.
    BenjaminiYekutieli,
    /// No screen was recorded; simultaneous inference covers only the fixed family.
    Unrecorded,
}

impl CandidateProcedure {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxT => "max_t",
            Self::BenjaminiHochberg => "bh",
            Self::BenjaminiYekutieli => "by",
            Self::Unrecorded => "unrecorded",
        }
    }
}

/// Declared screen / estimate split for a batch family.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScreen {
    /// Caller-stable id for the screen run.
    pub screen_id: std::sync::Arc<str>,
    /// Procedure used to pick the winner after the screen.
    pub procedure: CandidateProcedure,
    /// Row indexes used to screen candidates.
    pub screen_rows: std::sync::Arc<[u32]>,
    /// Row indexes used to estimate the surfaced family.
    pub estimate_rows: std::sync::Arc<[u32]>,
}

/// Provenance recorded on each batch result.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSelection {
    /// Screen run id, or `"unrecorded"`.
    pub screen_id: std::sync::Arc<str>,
    /// Selection rule.
    pub procedure: CandidateProcedure,
    /// Winning query index in the supplied family.
    pub winner_index: usize,
    /// Family size.
    pub family_size: usize,
    /// Screen half.
    pub screen_rows: std::sync::Arc<[u32]>,
    /// Estimate half.
    pub estimate_rows: std::sync::Arc<[u32]>,
    /// Whether the two halves are disjoint.
    pub disjoint: bool,
}

/// Shared-table batch of static average-effect queries.
///
/// Binds data once; each query runs identify → project → estimate independently
/// (shared ingest, not shared physical plan — plans stay per-query).
#[derive(Clone, Debug)]
pub struct BatchStudy {
    data: TabularData,
    graph: Dag,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
    latency_mode: Option<LatencyMode>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
    screen: Option<CandidateScreen>,
}

impl BatchStudy {
    /// Start a batch over `data` and a static DAG.
    #[must_use]
    pub fn new(data: TabularData, graph: Dag) -> Self {
        Self {
            data,
            graph,
            bootstrap_replicates: 50,
            refute: RefuteSuite::PlaceboAndRcc,
            latency_mode: None,
            identifier: None,
            estimator: None,
            screen: None,
        }
    }

    /// Record a screen/estimate split for candidate-selection provenance.
    #[must_use]
    pub fn candidate_screen(mut self, screen: CandidateScreen) -> Self {
        self.screen = Some(screen);
        self
    }

    /// Bootstrap replicates for every query.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self
    }

    /// Refute suite for every query.
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
        self
    }

    /// Optional latency tier applied to every query.
    #[must_use]
    pub fn latency_mode(mut self, mode: LatencyMode) -> Self {
        self.latency_mode = Some(mode);
        self
    }

    /// Optional identification strategy applied to every query.
    ///
    /// Parse a wire name with `"backdoor.adjustment".parse::<IdentifierId>()?`.
    #[must_use]
    pub const fn identifier(mut self, id: IdentifierId) -> Self {
        self.identifier = Some(id);
        self
    }

    /// Optional estimator applied to every query.
    ///
    /// Parse a wire name with `"propensity.weighting".parse::<EstimatorId>()?`.
    #[must_use]
    pub const fn estimator(mut self, id: EstimatorId) -> Self {
        self.estimator = Some(id);
        self
    }

    /// Estimate each query against the shared table.
    ///
    /// Shared fold assignment (`i % 5`) when the batch uses AIPW / score
    /// tables. Queries run in parallel under `ctx.parallelism.max_threads`.
    /// After the batch, joint IF covariance across claims that share rows
    /// supplies max-t and BH/BY diagnostics. An overlap validator failure
    /// does not abort a claim.
    ///
    /// # Errors
    ///
    /// Empty query list, or any per-query analysis failure.
    pub fn estimate_many(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyResult>, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch estimate_many requires at least one query".into(),
            });
        }
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let threads = ctx.parallelism.max_threads.get().max(1) as usize;
        let mut out = Vec::with_capacity(queries.len());
        for chunk in queries.chunks(threads) {
            std::thread::scope(|scope| {
                let handles: Vec<_> =
                    chunk.iter().map(|q| scope.spawn(|| self.run_one_on(&data, q, ctx))).collect();
                for handle in handles {
                    match handle.join() {
                        Ok(Ok(result)) => out.push(result),
                        Ok(Err(err)) => return Err(err),
                        Err(_) => {
                            return Err(CausalError::Compile {
                                message: "batch worker panicked".into(),
                            });
                        }
                    }
                }
                Ok(())
            })?;
        }
        attach_batch_joint_inference(&mut out, &data, queries, self.screen.as_ref());
        Ok(out)
    }

    /// Compile each query once into a reusable prepared batch.
    ///
    /// Identification and score tables are frozen per query. A later
    /// [`PreparedBatch::estimate`] reuses those plans on new rows under the
    /// same schema.
    ///
    /// # Errors
    ///
    /// Empty query list or any per-query prepare failure.
    pub fn prepare(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare requires at least one query".into(),
            });
        }
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let mut plans = Vec::with_capacity(queries.len());
        for query in queries {
            plans.push(self.study_for_data(&data, query)?.prepare(ctx)?);
        }
        Ok(PreparedBatch {
            plans,
            queries: queries.to_vec(),
            screen: self.screen.clone(),
        })
    }

    /// Compile discrete joint `InterventionResponse` cells into a reusable prepared batch.
    ///
    /// Defaults to [`EstimatorId::CellAipw`] when the batch has no estimator override.
    ///
    /// # Errors
    ///
    /// Empty query list, a non-Set intervention bundle, or any per-query prepare failure.
    pub fn prepare_cells(
        &self,
        queries: &[ResponseQuery],
        ctx: &ExecutionContext,
    ) -> Result<PreparedBatch, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch prepare_cells requires at least one query".into(),
            });
        }
        let data = subset_estimate_rows(&self.data, self.screen.as_ref())?;
        let mut plans = Vec::with_capacity(queries.len());
        let mut proxies = Vec::with_capacity(queries.len());
        for query in queries {
            plans.push(self.study_for_response_on(&data, query)?.prepare(ctx)?);
            proxies.push(cell_ate_proxy(query)?);
        }
        Ok(PreparedBatch { plans, queries: proxies, screen: self.screen.clone() })
    }

    fn study_for_response_on(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
    ) -> Result<Study, CausalError> {
        let mut builder = Study::tabular(data.clone())
            .graph(self.graph.clone())
            .query(query.clone())
            .refute(self.refute)
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(mode) = self.latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(id) = self.identifier {
            builder = builder.identifier(id);
        }
        builder = builder.estimator(self.estimator.unwrap_or(EstimatorId::CellAipw));
        builder.build()
    }

    fn run_one_on(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.study_for_data(data, query)?.run(ctx)
    }

    fn study_for_data(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
    ) -> Result<Study, CausalError> {
        let mut builder = Study::tabular(data.clone())
            .graph(self.graph.clone())
            .query(query.clone())
            .refute(self.refute)
            .bootstrap_replicates(self.bootstrap_replicates);
        if let Some(mode) = self.latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(id) = self.identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = self.estimator {
            builder = builder.estimator(est);
        }
        builder.build()
    }
}

/// Frozen batch of prepared average-effect (and joint-cell) plans.
#[derive(Clone, Debug)]
pub struct PreparedBatch {
    plans: Vec<PreparedStudy>,
    queries: Vec<AverageEffectQuery>,
    screen: Option<CandidateScreen>,
}

impl PreparedBatch {
    /// Queries frozen on this handle.
    #[must_use]
    pub fn queries(&self) -> &[AverageEffectQuery] {
        &self.queries
    }

    /// Prepared plans, one per query, sharing the batch schema.
    #[must_use]
    pub fn plans(&self) -> &[PreparedStudy] {
        &self.plans
    }

    /// Re-estimate every frozen plan on `data` and attach joint-family inference.
    ///
    /// # Errors
    ///
    /// Schema mismatch or any per-plan estimate failure.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyResult>, CausalError> {
        let data = subset_estimate_rows(data, self.screen.as_ref())?;
        let mut out = Vec::with_capacity(self.plans.len());
        for plan in &self.plans {
            out.push(plan.estimate(&data, ctx)?);
        }
        attach_batch_joint_inference(&mut out, &data, &self.queries, self.screen.as_ref());
        Ok(out)
    }
}

fn attach_batch_joint_inference(
    results: &mut [StudyResult],
    data: &TabularData,
    queries: &[AverageEffectQuery],
    screen: Option<&CandidateScreen>,
) {
    if results.len() < 2 {
        attach_candidate_selection(results, screen, &[], &[]);
        return;
    }
    // Row count alone is insufficient: different complete-case masks can have
    // equal sizes. Embed centered influence sequences in the original row universe.
    let mut aligned = Vec::new();
    let full_n = data.row_count();
    for (result, query) in results.iter().zip(queries) {
        if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved) {
            attach_batch_inference_unavailable(results, screen, "batch joint IF requires AllObserved");
            return;
        }
        let Some(inf) = result.estimate.influence.as_deref() else {
            attach_batch_inference_unavailable(results, screen, "batch joint IF requires per-claim influence");
            return;
        };
        let mut ids = vec![query.treatment, query.outcome];
        ids.extend(query.effect_modifiers.iter().copied());
        ids.extend(result.estimand.adjustment_set.iter().copied());
        ids.extend(result.estimand.instruments.iter().copied());
        ids.extend(result.estimand.mediators.iter().copied());
        if let Some(table) = result.estimate.score_table.as_ref() {
            ids.push(table.treatment);
            ids.extend(table.intervened.iter().copied());
        }
        let Ok(mask) = data.complete_case_mask(&ids) else {
            attach_batch_inference_unavailable(results, screen, "batch joint IF complete-case masks did not align");
            return;
        };
        let rows: Vec<_> =
            mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
        if rows.len() != inf.len() || rows.len() < 2 {
            attach_batch_inference_unavailable(
                results,
                screen,
                "batch joint IF influence length did not match the complete-case row universe",
            );
            return;
        }
        let mean = inf.iter().sum::<f64>() / inf.len() as f64;
        let mut col = vec![0.0; full_n];
        for (&r, &v) in rows.iter().zip(inf) {
            col[r] = (v - mean) * full_n as f64 / inf.len() as f64;
        }
        aligned.push(col);
    }
    let cols: Vec<&[f64]> = aligned.iter().map(Vec::as_slice).collect();
    let Ok(cov) = antecedent_estimate::joint_influence_covariance(&cols, None) else {
        attach_batch_inference_unavailable(results, screen, "batch joint IF covariance could not be formed");
        return;
    };
    let crit = antecedent_estimate::max_t_critical(&cov, 0.95, 4096, 1).ok();
    let mut p_values = Vec::with_capacity(results.len());
    for (i, result) in results.iter().enumerate() {
        let se = cov.se(i);
        let z = if se > 0.0 {
            result.estimate.ate.abs() / se
        } else if result.estimate.ate == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        p_values.push(2.0 * antecedent_stats::student_t_sf(z, 1.0e8));
    }
    let bh = antecedent_stats::benjamini_hochberg(&p_values);
    let by = antecedent_stats::benjamini_yekutieli(&p_values);
    for (i, result) in results.iter_mut().enumerate() {
        result.estimate.joint_covariance = Some(cov.clone());
        if let Some(c) = crit {
            result.estimate.simultaneous_interval = Some((
                result.estimate.ate - c * cov.se(i),
                result.estimate.ate + c * cov.se(i),
                0.95,
            ));
        }
        result.estimate.adjusted_p_values = Some((bh[i], by[i]));
        let mut d = antecedent_core::Diagnostic::new(
            "batch.joint_if",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            format!(
                "shared-row joint IF; BH={:.4} BY={:.4}{}",
                bh.get(i).copied().unwrap_or(f64::NAN),
                by.get(i).copied().unwrap_or(f64::NAN),
                crit.map(|c| format!(" max-t_0.95={c:.3}")).unwrap_or_default()
            ),
        );
        d.fields = std::sync::Arc::from([
            (
                std::sync::Arc::from("bh_q"),
                std::sync::Arc::from(format!("{}", bh.get(i).copied().unwrap_or(f64::NAN))),
            ),
            (
                std::sync::Arc::from("by_q"),
                std::sync::Arc::from(format!("{}", by.get(i).copied().unwrap_or(f64::NAN))),
            ),
        ]);
        result.diagnostics.push(d);
    }
    attach_candidate_selection(results, screen, &bh, &by);
}

fn subset_estimate_rows(
    data: &TabularData,
    screen: Option<&CandidateScreen>,
) -> Result<TabularData, CausalError> {
    let Some(screen) = screen else {
        return Ok(data.clone());
    };
    if screen.estimate_rows.is_empty() {
        return Ok(data.clone());
    }
    let n = data.row_count();
    let mut bytes = vec![0u8; n.div_ceil(8)];
    for &r in screen.estimate_rows.iter() {
        let i = r as usize;
        if i >= n {
            return Err(CausalError::Compile {
                message: "candidate-selection estimate_rows exceed the table".into(),
            });
        }
        bytes[i / 8] |= 1 << (i % 8);
    }
    let mask = ValidityBitmap::from_bytes(bytes, n).map_err(CausalError::from)?;
    data.with_analysis_mask(mask).map_err(CausalError::from)
}

fn attach_batch_inference_unavailable(
    results: &mut [StudyResult],
    screen: Option<&CandidateScreen>,
    reason: &str,
) {
    for result in results.iter_mut() {
        result.diagnostics.push(antecedent_core::Diagnostic::new(
            "batch.joint_if.unavailable",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Warning,
            reason,
        ));
    }
    attach_candidate_selection(results, screen, &[], &[]);
}

fn cell_ate_proxy(query: &ResponseQuery) -> Result<AverageEffectQuery, CausalError> {
    match &query.functional {
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            let Some(Intervention::Set { variable, .. }) = interventions.first() else {
                return Err(CausalError::Compile {
                    message: "batch prepare_cells requires a Set intervention".into(),
                });
            };
            Ok(AverageEffectQuery::binary_ate(*variable, *outcome))
        }
        _ => Err(CausalError::Compile {
            message: "batch prepare_cells is licensed for discrete joint InterventionResponse".into(),
        }),
    }
}

fn rows_disjoint(a: &[u32], b: &[u32]) -> bool {
    let mut left: Vec<u32> = a.to_vec();
    let mut right: Vec<u32> = b.to_vec();
    left.sort_unstable();
    right.sort_unstable();
    left.dedup();
    right.dedup();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => return false,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    true
}

fn attach_candidate_selection(
    results: &mut [StudyResult],
    screen: Option<&CandidateScreen>,
    bh: &[f64],
    by: &[f64],
) {
    let family_size = results.len();
    let (selection, recorded) = match screen {
        None => (
            CandidateSelection {
                screen_id: std::sync::Arc::from("unrecorded"),
                procedure: CandidateProcedure::Unrecorded,
                winner_index: 0,
                family_size,
                screen_rows: std::sync::Arc::from([]),
                estimate_rows: std::sync::Arc::from([]),
                disjoint: false,
            },
            false,
        ),
        Some(screen) => {
            let disjoint = rows_disjoint(&screen.screen_rows, &screen.estimate_rows);
            let winner_index = match screen.procedure {
                CandidateProcedure::BenjaminiHochberg => bh
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0),
                CandidateProcedure::BenjaminiYekutieli => by
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0),
                CandidateProcedure::MaxT | CandidateProcedure::Unrecorded => results
                    .iter()
                    .enumerate()
                    .max_by(|a, b| {
                        let sa = a.1.estimate.se_analytic.abs().max(1e-12);
                        let sb = b.1.estimate.se_analytic.abs().max(1e-12);
                        (a.1.estimate.ate.abs() / sa)
                            .partial_cmp(&(b.1.estimate.ate.abs() / sb))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0),
            };
            (
                CandidateSelection {
                    screen_id: std::sync::Arc::clone(&screen.screen_id),
                    procedure: screen.procedure,
                    winner_index,
                    family_size,
                    screen_rows: std::sync::Arc::clone(&screen.screen_rows),
                    estimate_rows: std::sync::Arc::clone(&screen.estimate_rows),
                    disjoint,
                },
                true,
            )
        }
    };
    for result in results.iter_mut() {
        result.candidate_selection = Some(selection.clone());
        result.estimate.candidate_selection = Some(antecedent_estimate::CandidateSelectionRecord {
            screen_id: std::sync::Arc::clone(&selection.screen_id),
            procedure: std::sync::Arc::from(selection.procedure.as_str()),
            winner_index: selection.winner_index,
            family_size: selection.family_size,
            screen_rows: std::sync::Arc::clone(&selection.screen_rows),
            estimate_rows: std::sync::Arc::clone(&selection.estimate_rows),
            disjoint: selection.disjoint,
        });
        if recorded {
            let mut d = antecedent_core::Diagnostic::new(
                "batch.candidate_selection",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                format!(
                    "screen={} procedure={} winner={} disjoint={}",
                    selection.screen_id,
                    selection.procedure.as_str(),
                    selection.winner_index,
                    selection.disjoint
                ),
            );
            d.fields = std::sync::Arc::from([
                (std::sync::Arc::from("screen_id"), std::sync::Arc::clone(&selection.screen_id)),
                (
                    std::sync::Arc::from("procedure"),
                    std::sync::Arc::from(selection.procedure.as_str()),
                ),
                (
                    std::sync::Arc::from("winner_index"),
                    std::sync::Arc::from(selection.winner_index.to_string()),
                ),
                (
                    std::sync::Arc::from("disjoint"),
                    std::sync::Arc::from(if selection.disjoint { "true" } else { "false" }),
                ),
            ]);
            result.diagnostics.push(d);
            if !selection.disjoint {
                result.diagnostics.push(antecedent_core::Diagnostic::new(
                    "batch.candidate_selection.overlap",
                    antecedent_core::DiagnosticKind::Scientific,
                    antecedent_core::DiagnosticSeverity::Warning,
                    "screen and estimate row sets are not disjoint",
                ));
            }
        } else {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.unrecorded",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "no candidate-selection split has been recorded; simultaneous inference covers only the fixed supplied family and excludes screening uncertainty",
            ));
        }
        if recorded {
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                "batch.candidate_selection.ranked_on_reported_family",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "winner is ranked on the reported estimate-sample family; screen_rows are recorded provenance and were not used to select",
            ));
        }
    }
}
