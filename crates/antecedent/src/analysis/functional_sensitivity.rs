// SPDX-License-Identifier: MIT OR Apache-2.0
//! The refuters of the functional validation state that need the data and a model beyond the
//! generic refits of [`target_validation`](super::target_validation):
//!
//! * a sensitivity range of a cross-world functional to the dependence it assumes away: the
//!   natural direct effect (and a path-specific effect through one mediator) against the
//!   correlation `rho` between the mediator and outcome errors, and a counterfactual against the
//!   correlation between the treatment and outcome errors. Each range is bound to its assumption
//!   and reports the break-even `rho`;
//! * the cross-world placebo, which permutes the mediator: the effect through it must vanish,
//!   and the natural direct effect must move to the total effect;
//! * the placebo of an interventional distribution, which permutes the intervened variable: the
//!   interventional law must equal the observed marginal law of the outcome;
//! * the noise-block permutation of an attribution: the abducted residuals of one block are
//!   permuted across units and the descendants regenerated, so the block's unit-level
//!   contribution (anomaly) or its share of the change (change attribution) must shrink.
//!
//! Every construction reads a supplied Dag (or, for the ranges, an Admg's directed part) and a
//! float64 table; a refuter that does not apply is recorded as a diagnostic naming its next
//! attempt.

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::{mean, shuffle};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::target_native::{
    RefutationTarget, TargetQuadrant, TargetRefuterKind, joint_mahalanobis, target_refuter_stream,
};
use antecedent_validate::{
    RefutationReport, cross_world_noise_range, mediation_range, mediation_sensitivity_fit,
    natural_direct_range, residual_correlation, total_effect_slope,
};

use super::execute::Study;
use super::prepared::PreparedStudy;
use super::target_validation::result_target_values;
use crate::accepted::GraphRef;
use crate::error::CausalError;
use crate::result::StudyResult;

/// Error correlation the sensitivity verdict is read at, and the smallest break-even that
/// passes.
const RHO_BAR: f64 = 0.1;
/// Fewest permutation replicates per attribution block.
const MIN_BLOCK_REPLICATES: u32 = 8;
/// Largest mean alignment of a permuted block's unit contributions with the original ones.
const ALIGNMENT_BAR: f64 = 0.5;

/// What the extra refuters read.
pub(crate) struct ExtraInput<'a> {
    pub(crate) study: &'a Study,
    pub(crate) prepared: &'a PreparedStudy,
    pub(crate) result: &'a StudyResult,
    pub(crate) data: &'a TabularData,
    pub(crate) replicates: u32,
    pub(crate) alpha: f64,
    pub(crate) ctx: &'a ExecutionContext,
    pub(crate) program: Option<[u8; 32]>,
}

/// Reports and diagnostics of the extra refuters.
pub(crate) type Extras = (Vec<RefutationReport>, Vec<Diagnostic>);

fn note(code: &'static str, message: String) -> Diagnostic {
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
}

fn not_applicable(refuter: &str, reason: &str, next: &str) -> Diagnostic {
    note(
        "refute.functional.refuter_refused",
        format!("refuter `{refuter}` did not run: {reason}; next attempt: {next}"),
    )
}

fn stamp(mut report: RefutationReport, program: Option<[u8; 32]>) -> RefutationReport {
    if let Some(program) = program {
        report.stamp_producer(program);
    }
    report
}

/// Run the extra refuters that apply to the query of `input.study`.
///
/// # Errors
///
/// Cancellation, admission or a failed refit.
pub(crate) fn extra_refuters(input: &ExtraInput<'_>) -> Result<Extras, CausalError> {
    let mut reports = Vec::new();
    let mut notes = Vec::new();
    match &input.study.query {
        CausalQuery::NestedCounterfactual(_)
        | CausalQuery::PathSpecific(_)
        | CausalQuery::Counterfactual(_) => {
            let (r, n) = cross_world_range(input);
            reports.extend(r);
            notes.extend(n);
            let (r, n) = mediator_placebo(input)?;
            reports.extend(r);
            notes.extend(n);
        }
        CausalQuery::Distribution(_) => {
            let (r, n) = distribution_placebo(input)?;
            reports.extend(r);
            notes.extend(n);
        }
        CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_) => {
            let (r, n) = noise_block_permutation(input)?;
            reports.extend(r);
            notes.extend(n);
        }
        _ => {}
    }
    Ok((reports, notes))
}

/// Directed parents of `variable` in a supplied Dag or in the directed part of an Admg.
fn graph_parents(study: &Study, variable: VariableId) -> Option<Vec<VariableId>> {
    let id = DenseNodeId::from_raw(variable.raw());
    let at = variable.raw() as usize;
    let lift = |parents: &[DenseNodeId]| -> Vec<VariableId> {
        parents.iter().map(|node| VariableId::from_raw(node.raw())).collect()
    };
    match study.graph.graph_ref() {
        GraphRef::Dag(graph) if at < graph.node_count() => Some(lift(graph.parents(id))),
        GraphRef::Admg(graph) if at < graph.node_count() => Some(lift(graph.parents(id))),
        _ => None,
    }
}

/// The columns of `ids` as finite floats.
fn columns(data: &TabularData, ids: &[VariableId]) -> Option<Vec<Vec<f64>>> {
    let cols: Vec<Vec<f64>> =
        ids.iter().map(|&id| data.float64_values(id).ok()).collect::<Option<_>>()?;
    cols.iter().all(|c| c.iter().all(|v| v.is_finite())).then_some(cols)
}

/// The scalar the query's functional reads from `result`. A natural direct effect that carries
/// the whole mediation decomposition (total, direct, indirect) is its `direct` coordinate, not
/// the first coordinate.
fn first_value(study: &Study, result: &StudyResult) -> Option<f64> {
    let (_, labels, values) = result_target_values(result).ok()?;
    if matches!(study.query, CausalQuery::NestedCounterfactual(_)) {
        if let Some(at) = labels.iter().position(|label| label.as_ref() == "direct") {
            return values.get(at).copied();
        }
    }
    values.first().copied()
}

fn target_of(input: &ExtraInput<'_>) -> Result<RefutationTarget, CausalError> {
    RefutationTarget::new(
        antecedent_validate::target_native::TargetFamily::ScalarAverageEffect,
        TargetQuadrant { temporal: false, bayesian: input.result.posterior.is_some() },
        ["effect"],
        input.program.unwrap_or([0; 32]),
    )
    .map_err(CausalError::from)
}

/// The mediator a cross-world query routes its effect through, and how.
enum Route {
    /// Natural direct effect holding `mediator` at its control-world value.
    Direct(VariableId),
    /// Path-specific effect through `mediator`.
    Through(VariableId),
    /// Unit-level counterfactual contrast of the treatment.
    Counterfactual,
}

fn route_of(study: &Study) -> Option<Route> {
    match &study.query {
        CausalQuery::NestedCounterfactual(q) => Some(Route::Direct(q.mediator)),
        CausalQuery::PathSpecific(q) if q.path_nodes.len() == 1 => {
            Some(Route::Through(q.path_nodes[0]))
        }
        CausalQuery::Counterfactual(_) => Some(Route::Counterfactual),
        _ => None,
    }
}

/// The sensitivity range of the cross-world functional to its dependence parameter.
fn cross_world_range(input: &ExtraInput<'_>) -> Extras {
    const NAME: &str = "target.cross_world_sensitivity";
    let mut notes = Vec::new();
    let Some(route) = route_of(input.study) else {
        return (Vec::new(), notes);
    };
    let (Some(treatment), outcome) = (input.result.treatment, input.result.outcome) else {
        return (Vec::new(), notes);
    };
    let Some(effect) = first_value(input.study, input.result) else {
        notes.push(not_applicable(
            NAME,
            "the result has no scalar effect",
            "a point-identified effect",
        ));
        return (Vec::new(), notes);
    };
    let Some(mut covariate_ids) = graph_parents(input.study, treatment) else {
        notes.push(not_applicable(
            NAME,
            "the range reads the parents of the treatment in a supplied Dag or Admg",
            "supply the structure as a Dag or an Admg",
        ));
        return (Vec::new(), notes);
    };
    let mediator = match route {
        Route::Direct(m) | Route::Through(m) => Some(m),
        Route::Counterfactual => None,
    };
    covariate_ids.retain(|&id| id != outcome && Some(id) != mediator && id != treatment);
    covariate_ids.sort_by_key(|id| id.raw());
    covariate_ids.dedup();
    let mut ids = vec![treatment, outcome];
    ids.extend(mediator);
    ids.extend(covariate_ids.iter().copied());
    let Some(cols) = columns(input.data, &ids) else {
        notes.push(not_applicable(
            NAME,
            "a column is not a complete float64 column",
            "encode the treatment, mediator, outcome and covariates as float64",
        ));
        return (Vec::new(), notes);
    };
    let skip = if mediator.is_some() { 3 } else { 2 };
    let covariates: Vec<Vec<f64>> = cols[skip..].to_vec();
    let target = match target_of(input) {
        Ok(target) => target,
        Err(error) => {
            notes.push(not_applicable(NAME, &error.to_string(), "a nonempty target"));
            return (Vec::new(), notes);
        }
    };
    let range = match (&route, mediator) {
        (Route::Direct(_) | Route::Through(_), Some(_)) => {
            let summary = mediation_sensitivity_fit(&cols[0], &cols[2], &cols[1], &covariates);
            let total = total_effect_slope(&cols[0], &cols[1], &covariates);
            match (summary, total, &route) {
                (Ok(summary), Ok(total), Route::Direct(_)) => {
                    natural_direct_range(target, &summary, total)
                }
                (Ok(summary), Ok(_), _) => mediation_range(target, &summary),
                (Err(error), _, _) | (_, Err(error), _) => Err(error),
            }
        }
        _ => residual_correlation(&cols[0], &cols[1], &covariates)
            .and_then(|r| cross_world_noise_range(target, effect, r)),
    };
    let range = match range {
        Ok(range) => range,
        Err(error) => {
            notes.push(not_applicable(
                NAME,
                &error.to_string(),
                "more rows, a full-rank design and a mediator or treatment with variance",
            ));
            return (Vec::new(), notes);
        }
    };
    publish_cross_world_range(input, effect, &range, notes)
}

/// The cross-world placebo: permute the mediator. The effect through it must vanish; the natural
/// direct effect must move to the total effect of the treatment on the outcome.
fn mediator_placebo(input: &ExtraInput<'_>) -> Result<Extras, CausalError> {
    const NAME: &str = "target.cross_world_placebo";
    let mut notes = Vec::new();
    let (mediator, expected_total) = match &input.study.query {
        CausalQuery::NestedCounterfactual(q) => {
            (q.mediator, Some(q.active_value() - q.control_value()))
        }
        CausalQuery::PathSpecific(q) if q.path_nodes.len() == 1 => (q.path_nodes[0], None),
        _ => return Ok((Vec::new(), notes)),
    };
    let (Some(treatment), outcome) = (input.result.treatment, input.result.outcome) else {
        return Ok((Vec::new(), notes));
    };
    let Some(original) = first_value(input.study, input.result) else {
        return Ok((Vec::new(), notes));
    };
    let Some(mut covariate_ids) = graph_parents(input.study, treatment) else {
        notes.push(not_applicable(
            NAME,
            "the reference total effect reads the parents of the treatment",
            "supply the structure as a Dag or an Admg",
        ));
        return Ok((Vec::new(), notes));
    };
    covariate_ids.retain(|&id| id != outcome && id != mediator && id != treatment);
    let mut ids = vec![treatment, outcome, mediator];
    ids.extend(covariate_ids);
    let Some(cols) = columns(input.data, &ids) else {
        notes.push(not_applicable(
            NAME,
            "a column is not a complete float64 column",
            "float64 columns",
        ));
        return Ok((Vec::new(), notes));
    };
    let expected = match expected_total {
        Some(delta) => match total_effect_slope(&cols[0], &cols[1], &cols[3..]) {
            Ok(slope) => slope * delta,
            Err(error) => {
                notes.push(not_applicable(NAME, &error.to_string(), "a full-rank design"));
                return Ok((Vec::new(), notes));
            }
        },
        None => 0.0,
    };
    let replicates = input.replicates;
    let mut values = Vec::with_capacity(replicates as usize);
    // The permuted column is rebuilt in one scratch buffer; each replicate starts from the
    // original order, exactly as a fresh copy would.
    let mut scratch = vec![0.0; cols[2].len()];
    for replicate in 0..replicates {
        if input.ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut rng =
            target_refuter_stream(TargetRefuterKind::RandomCommonCause, replicate, input.ctx);
        scratch.copy_from_slice(&cols[2]);
        shuffle(&mut rng, &mut scratch);
        let data = input.data.with_replaced_float(mediator, Arc::from(scratch.as_slice()))?;
        let refit = input.prepared.estimate(&data, input.ctx)?;
        if let Some(value) = first_value(input.study, &refit) {
            values.push(value);
        }
    }
    // A functional that sums the mediator out inside a district kernel never reads its
    // column: permuting it cannot move the effect, so the placebo says nothing about the
    // functional and is not run as a check.
    if !values.is_empty() && values.iter().all(|value| (value - original).abs() <= 1e-12) {
        notes.push(not_applicable(
            NAME,
            "the identified functional does not read the permuted mediator column, so \
             permuting it cannot move the effect",
            "a route whose functional reads the mediator (the edge g-formula)",
        ));
        return Ok((Vec::new(), notes));
    }
    let statistic = match joint_mahalanobis(&[expected], &values, 1) {
        Ok(statistic) => statistic,
        Err(error) => {
            notes.push(not_applicable(
                NAME,
                &error.to_string(),
                "more replicates, or a route whose refit varies with the permuted mediator",
            ));
            return Ok((Vec::new(), notes));
        }
    };
    let mean = mean(&values);
    let p = statistic.p_value.unwrap_or(0.0);
    // The linear total effect is a reference for the nonparametric route, not its twin.
    let near = expected_total.is_some() && (mean - expected).abs() <= 0.25 * expected.abs();
    let passed = p >= input.alpha || near;
    let report = RefutationReport::new(
        NAME,
        original,
        mean,
        p,
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "with the mediator permuted the effect is {mean:.4} where the cross-world \
                 functional implies {expected:.4} (joint chi-square {:.3}, p = {p:.4})",
                statistic.value
            ))
        }),
        u32::try_from(values.len()).unwrap_or(u32::MAX),
    );
    Ok((vec![stamp(report, input.program)], notes))
}

/// The placebo of an interventional distribution: with the intervened variable permuted the
/// interventional law is the observed marginal law of the outcome.
fn distribution_placebo(input: &ExtraInput<'_>) -> Result<Extras, CausalError> {
    const NAME: &str = "target.distribution_placebo";
    let mut notes = Vec::new();
    let CausalQuery::Distribution(query) = &input.study.query else {
        return Ok((Vec::new(), notes));
    };
    let Some(distribution) = input.result.distribution() else {
        return Ok((Vec::new(), notes));
    };
    if query.interventions.is_empty() {
        return Ok((Vec::new(), notes));
    }
    if !query.conditioning.is_empty() {
        notes.push(not_applicable(
            NAME,
            "the observed marginal is the placebo law of an unconditional intervention",
            "an unconditional intervention",
        ));
        return Ok((Vec::new(), notes));
    }
    // Every intervened variable is permuted on its own stream, so under the placebo the
    // intervened variables are independent of the outcome and of each other and the
    // interventional law of all of them together is the observed marginal law of the outcome.
    let mut treatments: Vec<VariableId> = Vec::new();
    for intervention in query.interventions.iter() {
        for variable in intervention.target_variables() {
            if !treatments.contains(&variable) {
                treatments.push(variable);
            }
        }
    }
    let atoms = &distribution.atoms;
    if atoms.len() < 2 {
        return Ok((Vec::new(), notes));
    }
    let kept = atoms.len() - 1;
    // Observed probability of each atom: the share of rows carrying its outcome values.
    let outcome_ids: Vec<VariableId> = query.outcomes.to_vec();
    let Some(cols) = columns(input.data, &outcome_ids) else {
        notes.push(not_applicable(
            NAME,
            "an outcome is not a complete float64 column",
            "float64 outcomes",
        ));
        return Ok((Vec::new(), notes));
    };
    let expected =
        observed_atom_probabilities(input.data.row_count(), &atoms[..kept], &outcome_ids, &cols);
    let Some(treatment_columns) = columns(input.data, &treatments) else {
        notes.push(not_applicable(
            NAME,
            "an intervened variable is not a complete float64 column",
            "float64",
        ));
        return Ok((Vec::new(), notes));
    };
    let required =
        u32::try_from(kept).ok().and_then(|count| count.checked_add(8)).ok_or_else(|| {
            CausalError::Compile {
                message: "the placebo outcome dimension exceeds the replicate count range".into(),
            }
        })?;
    let replicates = input.replicates.max(required);
    let rows =
        distribution_placebo_replicates(input, &treatments, &treatment_columns, kept, replicates)?;
    let statistic = match joint_mahalanobis(&expected, &rows, kept) {
        Ok(statistic) => statistic,
        Err(error) => {
            notes.push(not_applicable(NAME, &error.to_string(), "more replicates"));
            return Ok((Vec::new(), notes));
        }
    };
    let mut mean = vec![0.0; kept];
    for row in rows.chunks_exact(kept) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v / f64::from(replicates);
        }
    }
    let p = statistic.p_value.unwrap_or(0.0);
    let passed = p >= input.alpha;
    let report = RefutationReport::new(
        NAME,
        atoms[0].probability,
        mean[0],
        p,
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "with the intervened variable permuted the interventional law should be the \
                 observed marginal law of the outcome; it lies outside the refit replicate law \
                 (joint chi-square {:.3} on {kept} df, p = {p:.4})",
                statistic.value
            ))
        }),
        replicates,
    );
    Ok((vec![stamp(report, input.program)], notes))
}

/// Linear-additive mechanisms of a Dag, fitted by least squares on the table.
struct LinearScm {
    order: Vec<usize>,
    parents: Vec<Vec<usize>>,
    /// Intercept first, then one coefficient per parent.
    coefficients: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
    columns: Vec<Vec<f64>>,
    descendants: Vec<Vec<bool>>,
}

impl LinearScm {
    fn fit(dag: &Dag, data: &TabularData) -> Option<Self> {
        let k = dag.node_count();
        let node_ids: Vec<DenseNodeId> = (0..k)
            .map(|i| u32::try_from(i).ok().map(DenseNodeId::from_raw))
            .collect::<Option<_>>()?;
        let ids: Vec<VariableId> =
            node_ids.iter().map(|id| VariableId::from_raw(id.raw())).collect();
        let columns = columns(data, &ids)?;
        let n = columns.first()?.len();
        let order: Vec<usize> =
            dag.topological_order()?.into_iter().map(|id| id.raw() as usize).collect();
        let mut parents = vec![Vec::new(); k];
        let mut coefficients = vec![Vec::new(); k];
        let mut residuals = vec![Vec::new(); k];
        let mut ws = LeastSquaresWorkspace::default();
        for v in 0..k {
            let ps: Vec<usize> =
                dag.parents(node_ids[v]).iter().map(|p| p.raw() as usize).collect();
            let p = 1 + ps.len();
            if n <= p + 1 {
                return None;
            }
            let mut design = vec![1.0; n];
            for &parent in &ps {
                design.extend_from_slice(&columns[parent]);
            }
            let fit = FaerBackend.least_squares(&design, n, p, &columns[v], &mut ws).ok()?;
            if fit.rank < p {
                return None;
            }
            coefficients[v].clone_from(&fit.coefficients);
            residuals[v].clone_from(&fit.residuals);
            parents[v] = ps;
        }
        let descendants = (0..k)
            .map(|a| (0..k).map(|b| a != b && dag.reaches(node_ids[a], node_ids[b])).collect())
            .collect();
        Some(Self { order, parents, coefficients, residuals, columns, descendants })
    }

    /// The columns the permutation of `block`'s residuals changes: the block and its
    /// descendants, regenerated from the fitted mechanisms, in variable order. Untouched
    /// columns are read from the fitted table and never copied.
    fn permuted(&self, block: usize, permutation: &[usize]) -> Vec<(usize, Vec<f64>)> {
        let n = permutation.len();
        let mut regenerated: Vec<Option<Vec<f64>>> = vec![None; self.columns.len()];
        for &v in &self.order {
            if v != block && !self.descendants[block][v] {
                continue;
            }
            let mut fitted = vec![self.coefficients[v][0]; n];
            for (slot, &parent) in self.parents[v].iter().enumerate() {
                let beta = self.coefficients[v][1 + slot];
                let source = regenerated[parent].as_deref().unwrap_or(&self.columns[parent]);
                for (f, x) in fitted.iter_mut().zip(source) {
                    *f += beta * x;
                }
            }
            for (row, f) in fitted.iter_mut().enumerate() {
                *f += if v == block {
                    self.residuals[v][permutation[row]]
                } else {
                    self.residuals[v][row]
                };
            }
            regenerated[v] = Some(fitted);
        }
        regenerated.into_iter().enumerate().filter_map(|(v, column)| Some((v, column?))).collect()
    }
}

fn pearson(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n < 3 || b.len() != n {
        return None;
    }
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    (saa > 0.0 && sbb > 0.0).then(|| sab / (saa.sqrt() * sbb.sqrt()))
}

fn permutation(n: usize, rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(rng, &mut order);
    order
}

fn with_columns(
    data: &TabularData,
    new: Vec<(usize, Vec<f64>)>,
) -> Result<TabularData, CausalError> {
    let replacements: Vec<(VariableId, Arc<[f64]>)> = new
        .into_iter()
        .map(|(v, column)| {
            let id = u32::try_from(v).map_err(|_| CausalError::Compile {
                message: "a replacement column index exceeds the variable id range".into(),
            })?;
            Ok((VariableId::from_raw(id), Arc::from(column)))
        })
        .collect::<Result<_, CausalError>>()?;
    Ok(data.with_replaced_floats(&replacements)?)
}

/// The noise-block permutation of an attribution.
#[allow(
    clippy::too_many_lines,
    reason = "one refuter with an anomaly branch and a change branch that share the block loop"
)]
fn noise_block_permutation(input: &ExtraInput<'_>) -> Result<Extras, CausalError> {
    const NAME: &str = "target.noise_block_permutation";
    let mut notes = Vec::new();
    let Some(dag) = input.study.graph.as_dag() else {
        notes.push(not_applicable(
            NAME,
            "the permutation regenerates the descendants of a block from fitted mechanisms of a \
             supplied Dag",
            "supply the structure as a Dag",
        ));
        return Ok((Vec::new(), notes));
    };
    let Some(scm) = LinearScm::fit(dag, input.data) else {
        notes.push(not_applicable(
            NAME,
            "the table is not a complete float64 table with a full-rank mechanism per variable",
            "complete float64 columns and enough rows",
        ));
        return Ok((Vec::new(), notes));
    };
    let n = scm.columns[0].len();
    // A third of the requested replicates per block, never fewer than the minimum; the
    // request is honoured and the report carries the executed count.
    let replicates = (input.replicates / 3).max(MIN_BLOCK_REPLICATES);
    let anomaly = input.result.anomaly().and_then(|scores| scores.first());
    let change = input.result.change_attribution();
    if anomaly.is_none() && change.is_some() {
        // A change attribution allocates the shift of the mechanisms between two populations;
        // permuting the residuals of one block leaves its mechanism, and so its share of the
        // change, where it was. The refuter would report a surviving share for every correct
        // attribution, so it does not apply.
        notes.push(not_applicable(
            NAME,
            "a change attribution allocates a shift of mechanisms, which a permutation of the \
             residuals of a block does not remove",
            "use subset or bootstrap stability of the change vector",
        ));
        return Ok((Vec::new(), notes));
    }
    // Blocks: the noise components of the first scored target, or the change components.
    let blocks: Vec<usize> = if let Some(scores) = anomaly {
        scores.noise_components.iter().map(|c| c.variable().raw() as usize).collect()
    } else if let Some(change) = change {
        change.contributions.iter().map(|c| c.component.variable().raw() as usize).collect()
    } else {
        return Ok((Vec::new(), notes));
    };
    let mut worst: Option<(f64, usize)> = None;
    let mut refuted_value = 0.0;
    let mut tested = 0_u32;
    for (slot, &block) in blocks.iter().enumerate() {
        if block >= scm.columns.len() {
            continue;
        }
        let mut statistic = Vec::new();
        for replicate in 0..replicates {
            if input.ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_VALIDATE,
                });
            }
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PlaceboOutcome,
                replicate + 1000 * u32::try_from(block).unwrap_or(0),
                input.ctx,
            );
            let data = with_columns(input.data, scm.permuted(block, &permutation(n, &mut rng)))?;
            let refit = input.prepared.estimate(&data, input.ctx)?;
            if let Some(scores) = anomaly {
                let Some(refit_scores) = refit.anomaly().and_then(|s| s.first()) else {
                    continue;
                };
                let width = scores.noise_components.len();
                let refit_width = refit_scores.noise_components.len();
                let Some(other) = refit_scores
                    .noise_components
                    .iter()
                    .position(|c| c.variable().raw() as usize == block)
                else {
                    continue;
                };
                if scores.rows != refit_scores.rows {
                    continue;
                }
                let rows = scores.rows.len();
                let before: Vec<f64> =
                    (0..rows).map(|r| scores.noise_contributions[r * width + slot]).collect();
                let after: Vec<f64> = (0..rows)
                    .map(|r| refit_scores.noise_contributions[r * refit_width + other])
                    .collect();
                if let Some(rho) = pearson(&before, &after) {
                    statistic.push(rho);
                }
            } else if let (Some(change), Some(refit_change)) = (change, refit.change_attribution())
            {
                let share = |c: &antecedent_attribution::ChangeAttributionResult| {
                    c.contributions
                        .iter()
                        .find(|x| x.component.variable().raw() as usize == block)
                        .map(|x| x.contribution)
                };
                if let (Some(_), Some(after)) = (share(change), share(refit_change)) {
                    statistic.push(after);
                }
            }
        }
        if statistic.is_empty() {
            notes.push(not_applicable(
                NAME,
                &format!("block {block} produced no comparable refit"),
                "scored rows that the permutation leaves unchanged",
            ));
            continue;
        }
        tested += 1;
        let mean = mean(&statistic);
        // Anomaly: the alignment of unit contributions; change: the permuted block's share of
        // the change relative to its original share.
        let (score, shown) = if anomaly.is_some() {
            (mean.abs(), mean)
        } else {
            let original = change
                .and_then(|c| {
                    c.contributions
                        .iter()
                        .find(|x| x.component.variable().raw() as usize == block)
                        .map(|x| x.contribution)
                })
                .unwrap_or(0.0);
            if original.abs() <= 1e-12 {
                continue;
            }
            (mean.abs() / original.abs(), mean / original)
        };
        if worst.is_none_or(|(w, _)| score > w) {
            worst = Some((score, block));
            refuted_value = shown;
        }
    }
    let Some((score, block)) = worst else {
        return Ok((Vec::new(), notes));
    };
    let bar = if anomaly.is_some() { ALIGNMENT_BAR } else { 0.5 };
    let passed = score <= bar;
    let what = if anomaly.is_some() {
        "unit-level contributions stay aligned with the original ones"
    } else {
        "share of the change stays at a fraction of its original value"
    };
    let report = RefutationReport::new(
        NAME,
        1.0,
        refuted_value,
        score,
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "after permuting the abducted residuals of the block of variable {block} (and \
                 regenerating its descendants) the {what} ({score:.3} > {bar}): the \
                 attribution does not depend on the block's own noise"
            ))
        }),
        replicates * tested,
    );
    Ok((vec![stamp(report, input.program)], notes))
}

fn publish_cross_world_range(
    input: &ExtraInput<'_>,
    effect: f64,
    range: &antecedent_validate::AssumptionRange,
    mut notes: Vec<Diagnostic>,
) -> Extras {
    // The range is in the units of the linear model; rescale it to the route's own contrast.
    let at_zero = range.tested.iter().find(|p| p.parameter == 0.0).map(|p| p.target);
    let scale = match at_zero {
        Some(x0) if x0.abs() > 1e-12 => effect / x0,
        _ => {
            notes.push(not_applicable(
                "target.cross_world_sensitivity",
                "the linear model reads no effect at rho = 0 to rescale the range by",
                "a nonzero fitted effect",
            ));
            return (Vec::new(), notes);
        }
    };
    let bound = range
        .tested
        .iter()
        .filter(|p| (p.parameter.abs() - RHO_BAR).abs() < 1e-12)
        .map(|p| p.target * scale)
        .min_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(effect);
    let break_even = range.break_even;
    let passed = break_even.is_none_or(|b| b.abs() >= RHO_BAR);
    let curve = range
        .tested
        .iter()
        .map(|p| format!("{:+.2}: {:.5}", p.parameter, p.target * scale))
        .collect::<Vec<_>>()
        .join(", ");
    notes.push(note(
        "refute.functional.cross_world_range",
        format!(
            "assumption `{}`: {}. Effect by {} (rho: effect): {curve}. Break-even {}",
            range.assumption.as_str(),
            range.assumption.statement(),
            range.parameter,
            break_even
                .map_or_else(|| "not reached on the grid".to_owned(), |b| format!("rho = {b:.4}"))
        ),
    ));
    let report = RefutationReport::new(
        "target.cross_world_sensitivity",
        effect,
        bound,
        break_even.map_or(f64::INFINITY, f64::abs),
        true,
        passed,
        (!passed).then(|| {
            Arc::from(format!(
                "a correlation of {:.3} between the errors the cross-world functional assumes \
                 independent explains the effect away (bar {RHO_BAR})",
                break_even.unwrap_or(f64::NAN)
            ))
        }),
        0,
    );
    (vec![stamp(report, input.program)], notes)
}

fn observed_atom_probabilities(
    n: usize,
    atoms: &[antecedent_estimate::DistributionAtom],
    outcome_ids: &[VariableId],
    cols: &[Vec<f64>],
) -> Vec<f64> {
    let mut expected = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let mut hits = 0_usize;
        'rows: for row in 0..n {
            for (variable, value) in atom.outcomes.iter() {
                let Some(position) = outcome_ids.iter().position(|id| id == variable) else {
                    continue 'rows;
                };
                match value.as_f64() {
                    Some(v) if (cols[position][row] - v).abs() < 1e-9 => {}
                    _ => continue 'rows,
                }
            }
            hits += 1;
        }
        expected.push(hits as f64 / n as f64);
    }
    expected
}

/// Replicate-stream index of the `slot`-th intervened variable: the first keeps the
/// replicate's own stream, each further variable draws from a disjoint block of streams.
fn placebo_stream_index(replicate: u32, slot: usize) -> u32 {
    let offset = u32::try_from(slot).unwrap_or(u32::MAX).saturating_mul(1 << 20);
    replicate.saturating_add(offset)
}

fn distribution_placebo_replicates(
    input: &ExtraInput<'_>,
    treatments: &[VariableId],
    columns: &[Vec<f64>],
    kept: usize,
    replicates: u32,
) -> Result<Vec<f64>, CausalError> {
    let mut rows = Vec::with_capacity(replicates as usize * kept);
    let mut scratch: Vec<Vec<f64>> = columns.iter().map(|column| vec![0.0; column.len()]).collect();
    for replicate in 0..replicates {
        if input.ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut replacements: Vec<(VariableId, Arc<[f64]>)> = Vec::with_capacity(treatments.len());
        for (slot, ((&treatment, column), permuted)) in
            treatments.iter().zip(columns).zip(&mut scratch).enumerate()
        {
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PlaceboTreatment,
                placebo_stream_index(replicate, slot),
                input.ctx,
            );
            permuted.copy_from_slice(column);
            shuffle(&mut rng, permuted);
            replacements.push((treatment, Arc::from(permuted.as_slice())));
        }
        let data = input.data.with_replaced_floats(&replacements)?;
        let refit = input.prepared.estimate(&data, input.ctx)?;
        let (_, _, values) = result_target_values(&refit)?;
        if values.len() != kept {
            return Err(CausalError::Compile {
                message: "the placebo refit published a different number of atoms".into(),
            });
        }
        rows.extend_from_slice(&values);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::{pearson, placebo_stream_index};

    #[test]
    fn the_first_intervened_variable_keeps_the_replicate_stream() {
        assert_eq!(placebo_stream_index(7, 0), 7);
        assert_ne!(placebo_stream_index(7, 1), placebo_stream_index(7, 2));
        assert_ne!(placebo_stream_index(7, 1), placebo_stream_index(8, 0));
    }

    #[test]
    fn pearson_needs_three_rows_and_a_varying_column() {
        assert_eq!(pearson(&[1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
        let rho = pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
        assert!((rho - 1.0).abs() < 1e-12);
    }
}
