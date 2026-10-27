// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution of a counterfactual that contrasts several outcomes, several simultaneous
//! interventions, or nested interventions.
//!
//! One abduction of the exogenous state is replayed under the active and the control world, and
//! every outcome is contrasted from those two replays. The Dirichlet row-weight posterior and the
//! multinomial row bootstrap refit the mechanisms once per draw and contrast every outcome from
//! that draw's single abduction, so the outcomes' draws are jointly coherent. Support,
//! extrapolation and the homogeneity verdict are read per outcome over every intervened variable.
//! The one-outcome, one-`Set` counterfactual keeps its own execution in the parent module.

use super::{
    Arc, CausalError, CausalPosterior, CheckedCounterfactualPlan, CounterfactualProcedure,
    DenseNodeId, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId,
    ExecutionContext, GcmSlot, HomogeneityVerdict, IdentificationStatus, IdentifiedExecuteExtras,
    IdentifiedExecuteFinish, IdentifierId, Instant, OverlapPolicy, Strategy, StreamDomain,
    StudyResult, TableView, TabularData, UnitSupport, VariableId, counterfactual_posterior_columns,
    counterfactual_uncertainty_diagnostic, elapsed_ns, finish_identified_execute_with_context,
    homogeneity_diagnostics, homogeneity_verdict_over, map_mechanism_fit, mechanism_audit,
    noise_inference_label, observed_range, path_assignments_over, provenance_ids,
    unit_effect_posterior_summary,
};
use crate::gcm::{
    ContrastSetResult, ContrastTarget, ContrastedOutcome, CounterfactualContrasts, TreatmentLevels,
    UnitEffectIntervals, contrast_outcome_nodes, counterfactual_contrasts_with_noise,
};

/// The estimate each outcome's contrast carries, whichever procedure produced it.
struct ContrastEstimates {
    /// The first outcome's scalar estimate: the one the result publishes.
    primary: EffectEstimate,
    /// The joint posterior of every outcome's mean effect (Bayesian only).
    posterior: Option<CausalPosterior>,
    /// Disclosures the procedure adds (bootstrap accounting).
    notes: Vec<Diagnostic>,
    /// Mean effect of each outcome, in the query's order.
    means: Vec<f64>,
    /// Standard error of each mean, when the procedure forms one.
    standard_errors: Vec<Option<f64>>,
    /// Published per-unit effects of each outcome.
    unit_effects: Vec<Arc<[f64]>>,
    /// Per-unit intervals of each outcome (Bayesian only).
    intervals: Vec<Option<UnitEffectIntervals>>,
    /// Whether a bootstrap standard error was formed.
    bootstrapped: bool,
}

/// Inputs every refit of a contrast set reads.
struct ContrastRefit<'a> {
    data: &'a TabularData,
    fitted: &'a crate::gcm::FittedGcm,
    target: &'a ContrastTarget,
    nodes: Vec<DenseNodeId>,
    assumptions: &'a antecedent_core::AssumptionSet,
    status: IdentificationStatus,
}

fn variable_name(data: &TabularData, variable: VariableId) -> String {
    data.schema()
        .get(variable)
        .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
}

#[allow(
    clippy::too_many_lines,
    reason = "one contrast-set execution: the procedure's estimates, the per-outcome support and homogeneity verdicts and the identified finish share every fitted product"
)]
pub(super) fn execute_contrast_set_plan(
    plan: &CheckedCounterfactualPlan,
    target: &ContrastTarget,
    data: &TabularData,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let started = Instant::now();
    let identification = &plan.cache.identification;
    let estimand = plan.cache.estimand.clone();
    let graph = plan.operation.graph();
    let (fitted, set) = plan.operation.execute_contrasts(data, ctx)?;
    let assignments = mechanism_audit(&fitted.assignments);
    let refit = ContrastRefit {
        data,
        fitted: &fitted,
        target,
        nodes: contrast_outcome_nodes(&fitted.model, &target.outcomes)?,
        assumptions: &identification.required_assumptions,
        status: identification.status,
    };
    let estimates = match &plan.procedure {
        CounterfactualProcedure::DirichletRowWeights { draws } => {
            dirichlet_estimates(&refit, &set, *draws, ctx)?
        }
        CounterfactualProcedure::RowBootstrap { replicates } => {
            bootstrap_estimates(&refit, &set, *replicates, ctx)?
        }
        CounterfactualProcedure::Point => {
            point_estimates(&set, &identification.required_assumptions)
        }
    };
    let treatments = target.treatments();
    let verdicts: Vec<HomogeneityVerdict> = target
        .outcomes
        .iter()
        .map(|&outcome| homogeneity_verdict_over(graph, &fitted.assignments, &treatments, outcome))
        .collect();

    // The observed range of each intervened variable and whether either world leaves it.
    let mut ranges = Vec::with_capacity(target.levels.len());
    let mut pooled_extrapolative = false;
    for level in target.levels.iter() {
        let (min, max) = observed_range(&data.float64_values(level.variable)?);
        pooled_extrapolative |=
            level.control < min || level.control > max || level.active < min || level.active > max;
        ranges.push((variable_name(data, level.variable), min, max));
    }
    let mut unit_extrapolative = Vec::with_capacity(target.outcomes.len());
    let mut support_diagnostics = Vec::with_capacity(target.outcomes.len());
    for &outcome in target.outcomes.iter() {
        let support = unit_support_over(
            data,
            graph,
            &fitted.assignments,
            &set.exogenous,
            &target.levels,
            outcome,
        )?;
        support_diagnostics.push(joint_support_diagnostic(
            &support,
            &variable_name(data, outcome),
            &ranges,
            pooled_extrapolative,
        ));
        unit_extrapolative.push(Arc::<[bool]>::from(support.extrapolation_flags()));
    }

    let mut estimate = estimates.primary;
    estimate.heterogeneity.unit_effects_homogeneous =
        matches!(verdicts[0], HomogeneityVerdict::Structural(_));
    let mut diagnostics = vec![
        Diagnostic::new(
            "gcm.counterfactual.mechanisms",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            assignments,
        ),
        Diagnostic::new(
            "gcm.counterfactual",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "noise_inference={}; control=[{}]; active=[{}]; outcomes={}",
                noise_inference_label(set.noise_inference),
                level_list(data, &target.levels, |level| level.control),
                level_list(data, &target.levels, |level| level.active),
                target.outcomes.len(),
            ),
        ),
    ];
    diagnostics.extend(support_diagnostics);
    diagnostics.push(counterfactual_uncertainty_diagnostic(estimates.posterior.is_some()));
    if estimates.bootstrapped {
        diagnostics.retain(|diagnostic| {
            diagnostic.code.as_ref() != "gcm.counterfactual.uncertainty_unavailable"
        });
    }
    diagnostics.extend(estimates.notes);
    for ((&outcome, verdict), mean) in target.outcomes.iter().zip(&verdicts).zip(&estimates.means) {
        homogeneity_diagnostics(data, outcome, *mean, verdict, &mut diagnostics);
    }

    let contrasts: Vec<ContrastedOutcome> = target
        .outcomes
        .iter()
        .enumerate()
        .map(|(index, &outcome)| ContrastedOutcome {
            outcome,
            unit_effects: Arc::clone(&estimates.unit_effects[index]),
            mean_ite: estimates.means[index],
            standard_error: estimates.standard_errors[index],
            unit_effect_intervals: estimates.intervals[index].clone(),
            unit_extrapolative: Some(Arc::clone(&unit_extrapolative[index])),
            unit_effects_homogeneous: matches!(verdicts[index], HomogeneityVerdict::Structural(_)),
        })
        .collect();
    let payload = CounterfactualContrasts {
        treatments: Arc::from(treatments),
        contrasts,
        noise_inference: set.noise_inference,
        exogenous: set.exogenous,
    };
    let bootstrapped = estimates.bootstrapped;
    let posterior = estimates.posterior;
    Ok(finish_identified_execute_with_context(
        &plan.result_context,
        Some(data),
        IdentifiedExecuteFinish::new(
            &plan.physical,
            identification.clone(),
            estimand,
            estimate,
            Strategy::new(IdentifierId::GcmParametric, EstimatorId::GcmFit),
            target.levels[0].variable,
            target.outcomes[0],
        )
        .identify_cached(plan.identify_cached)
        .extra_diagnostics(diagnostics)
        .wall_time_ns(elapsed_ns(started))
        .extras(IdentifiedExecuteExtras {
            gcm: Some(GcmSlot::CounterfactualContrasts(Box::new(payload))),
            bootstrap_replicates_requested: Some(match plan.procedure {
                CounterfactualProcedure::RowBootstrap { replicates } if bootstrapped => {
                    Some(replicates)
                }
                _ => None,
            }),
            identify_provenance: Some(provenance_ids(
                "identify.gcm_parametric",
                "identify.gcm_parametric",
            )),
            estimate_provenance: Some(provenance_ids("counterfactual.aap", "counterfactual.aap")),
            n_draws: posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
            posterior,
            ..Default::default()
        }),
    ))
}

/// `name=level` for each intervened variable, comma separated.
fn level_list(
    data: &TabularData,
    levels: &[TreatmentLevels],
    pick: impl Fn(&TreatmentLevels) -> f64,
) -> String {
    levels
        .iter()
        .map(|level| format!("{}={}", variable_name(data, level.variable), pick(level)))
        .collect::<Vec<_>>()
        .join(",")
}

/// The point procedure: the abduct-act-predict means, with no sampling uncertainty.
fn point_estimates(
    set: &ContrastSetResult,
    assumptions: &antecedent_core::AssumptionSet,
) -> ContrastEstimates {
    ContrastEstimates {
        primary: EffectEstimate::new(
            set.contrasts[0].mean_ite,
            f64::NAN,
            assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        ),
        posterior: None,
        notes: Vec::new(),
        means: set.contrasts.iter().map(|c| c.mean_ite).collect(),
        standard_errors: vec![None; set.contrasts.len()],
        unit_effects: set.contrasts.iter().map(|c| Arc::clone(&c.unit_effects)).collect(),
        intervals: vec![None; set.contrasts.len()],
        bootstrapped: false,
    }
}

/// Bayesian-bootstrap refit of the mechanisms per draw. Every draw abducts once and contrasts
/// every outcome, so the posterior holds one mean-effect column per outcome from the same draws
/// and each outcome publishes its own per-unit intervals.
fn dirichlet_estimates(
    refit: &ContrastRefit<'_>,
    set: &ContrastSetResult,
    n_draws: usize,
    ctx: &ExecutionContext,
) -> Result<ContrastEstimates, CausalError> {
    let ContrastRefit { data, fitted, target, ref nodes, assumptions, status } = *refit;
    let base_model = &fitted.model;
    let k = target.outcomes.len();
    let n_units = set.contrasts[0].unit_effects.len();
    let request = target.request();
    let mut values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_draws); k];
    // Column-major `units × draws` per outcome.
    let mut unit_draws: Vec<Vec<f64>> = vec![vec![0.0; n_units * n_draws]; k];
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0x0CF0);
    // One stream through every draw: a categorical node's rank-preserving noise is drawn once
    // per posterior draw of the mechanisms, independently across draws.
    let mut noise_rng = ctx.rng.stream_for(StreamDomain::Counterfactual, 0x0CF1);
    let n = data.row_count();
    for draw in 0..n_draws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, false);
        let store = crate::gcm::MechanismRegistry::with_heterogeneity_families()
            .refit_weighted(base_model, data, &fitted.assignments, &weights)
            .map_err(|e| match e {
                crate::gcm::ModelError::NotConverged { .. } => map_mechanism_fit(e),
                other => CausalError::rejected(other),
            })?;
        let draw_set = counterfactual_contrasts_with_noise(
            base_model.clone().with_mechanisms(store),
            data,
            &request,
            nodes,
            ctx,
            &mut noise_rng,
        )?;
        if draw_set.contrasts.len() != k
            || draw_set.contrasts.iter().any(|c| c.unit_effects.len() != n_units)
        {
            return Err(CausalError::Compile {
                message: "counterfactual Bayesian draw changed the unit set".into(),
            });
        }
        for (index, contrast) in draw_set.contrasts.iter().enumerate() {
            values[index].push(contrast.mean_ite);
            for (unit, effect) in contrast.unit_effects.iter().enumerate() {
                unit_draws[index][unit * n_draws + draw] = *effect;
            }
        }
    }
    let columns = target
        .outcomes
        .iter()
        .zip(values)
        .map(|(outcome, column)| (format!("ite[{}]", outcome.raw()), column))
        .collect();
    let posterior = counterfactual_posterior_columns(columns, assumptions.clone(), status)?;
    let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "counterfactual posterior missing effect column".into(),
    })?;
    // Published unit effects stay the plain draw average (sequential sum, divided once).
    let scale = n_draws.max(1) as f64;
    let mut unit_effects = Vec::with_capacity(k);
    let mut intervals = Vec::with_capacity(k);
    for column in unit_draws {
        let means: Vec<f64> = (0..n_units)
            .map(|unit| {
                column[unit * n_draws..(unit + 1) * n_draws].iter().fold(0.0, |s, v| s + v) / scale
            })
            .collect();
        let summary = unit_effect_posterior_summary(column, n_units, n_draws)?;
        unit_effects.push(Arc::<[f64]>::from(means));
        intervals.push(Some(UnitEffectIntervals {
            lower: Arc::clone(&summary.q025),
            upper: Arc::clone(&summary.q975),
            level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
            method: antecedent_core::IntervalMethod::UnitPosteriorQuantile.as_str(),
        }));
    }
    let mut primary = EffectEstimate::new(
        posterior.summaries.mean[eq],
        posterior.summaries.sd[eq],
        posterior.assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    primary.uncertainty.analytic.se = posterior.summaries.sd[eq];
    Ok(ContrastEstimates {
        primary,
        means: (0..k).map(|index| posterior.summaries.mean[index]).collect(),
        standard_errors: (0..k).map(|index| Some(posterior.summaries.sd[index])).collect(),
        posterior: Some(posterior),
        notes: Vec::new(),
        unit_effects,
        intervals,
        bootstrapped: false,
    })
}

/// Frequentist whole-procedure row bootstrap of every outcome's mean effect: each replicate
/// refits every mechanism under multinomial row counts (conditional on the selected families),
/// abducts once and contrasts every outcome on the original units. The points stay the original
/// fit's means; the replicates give each its bootstrap standard error. Per-unit intervals are
/// not formed. A failed replicate is failed mass for every outcome.
fn bootstrap_estimates(
    refit: &ContrastRefit<'_>,
    set: &ContrastSetResult,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<ContrastEstimates, CausalError> {
    let ContrastRefit { data, fitted, target, ref nodes, assumptions, status: _ } = *refit;
    let k = target.outcomes.len();
    let n = data.row_count();
    let request = target.request();
    let _admitted = super::super::gcm_bootstrap::admit_replicates(ctx, replicates, k, n)?;
    let registry = crate::gcm::MechanismRegistry::with_heterogeneity_families();
    let cancelled = || ctx.cancellation.is_cancelled();
    let cancelled_error =
        || CausalError::Cancelled { stage: crate::analysis::stage::STAGE_UNCERTAINTY };
    let mut values: Vec<Vec<f64>> =
        vec![Vec::with_capacity(usize::try_from(replicates).unwrap_or(0)); k];
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if cancelled() {
            return Err(cancelled_error());
        }
        let mut rng = super::super::gcm_bootstrap::replicate_stream(ctx, 0x0CF0_B007, replicate);
        let counts = super::super::gcm_bootstrap::bootstrap_counts(&mut rng, n);
        let store = match registry.refit_frequency_weighted_polled(
            &fitted.model,
            data,
            &fitted.assignments,
            &counts,
            &cancelled,
        ) {
            Ok(store) => store,
            Err(_) if cancelled() => return Err(cancelled_error()),
            Err(_) => {
                failed += 1;
                continue;
            }
        };
        let mut noise_rng = ctx.rng.stream_for(
            StreamDomain::Counterfactual,
            0x0CF1_B007_u64.wrapping_add(u64::from(replicate)),
        );
        match counterfactual_contrasts_with_noise(
            fitted.model.clone().with_mechanisms(store),
            data,
            &request,
            nodes,
            ctx,
            &mut noise_rng,
        ) {
            Ok(draw)
                if draw.contrasts.len() == k
                    && draw
                        .contrasts
                        .iter()
                        .zip(&set.contrasts)
                        .all(|(d, c)| d.unit_effects.len() == c.unit_effects.len()) =>
            {
                for (column, contrast) in values.iter_mut().zip(&draw.contrasts) {
                    column.push(contrast.mean_ite);
                }
            }
            Err(_) if cancelled() => return Err(cancelled_error()),
            _ => failed += 1,
        }
    }
    let accountings: Vec<_> = values
        .iter()
        .map(|column| super::super::gcm_bootstrap::accounting(column, failed, replicates))
        .collect();
    let mut primary = EffectEstimate::new(
        set.contrasts[0].mean_ite,
        f64::NAN,
        assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    primary.uncertainty.bootstrap = accountings[0];
    let notes = target
        .outcomes
        .iter()
        .zip(&accountings)
        .map(|(outcome, accounting)| {
            super::super::gcm_bootstrap::bootstrap_diagnostic(
                &format!(
                    "mean ITE of {} over the observed units (per-unit intervals are not formed)",
                    variable_name(data, *outcome)
                ),
                accounting,
            )
        })
        .collect();
    Ok(ContrastEstimates {
        primary,
        posterior: None,
        notes,
        means: set.contrasts.iter().map(|c| c.mean_ite).collect(),
        standard_errors: accountings.iter().map(|accounting| accounting.se).collect(),
        unit_effects: set.contrasts.iter().map(|c| Arc::clone(&c.unit_effects)).collect(),
        intervals: vec![None; k],
        bootstrapped: true,
    })
}

/// [`super::unit_support`] for one outcome under several simultaneous interventions.
///
/// A unit belongs to the active arm when its observed treatment vector is nearer the active
/// levels than the control levels (summed absolute distance; for one treatment, the nearer
/// level). Covariates are the parents of the re-evaluated path mechanisms that no intervention
/// moves: not an intervened variable and not a descendant of one. With one treatment this is
/// exactly [`super::unit_support`].
fn unit_support_over(
    data: &TabularData,
    graph: &antecedent_graph::Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    exogenous: &crate::gcm::ExogenousPosterior,
    levels: &[TreatmentLevels],
    outcome: VariableId,
) -> Result<UnitSupport, CausalError> {
    let n = data.row_count();
    let treatments: Vec<VariableId> = levels.iter().map(|level| level.variable).collect();
    let observed =
        treatments.iter().map(|&t| data.float64_values(t)).collect::<Result<Vec<Vec<f64>>, _>>()?;
    let in_active: Vec<bool> = (0..n)
        .map(|row| {
            let (mut to_active, mut to_control) = (0.0, 0.0);
            for (level, column) in levels.iter().zip(&observed) {
                to_active += (column[row] - level.active).abs();
                to_control += (column[row] - level.control).abs();
            }
            to_active < to_control
        })
        .collect();
    let mut parent_cell = vec![false; n];
    let mut disturbance = vec![false; n];
    let mut covariates = Vec::new();
    let mut disturbance_nodes = Vec::new();
    let Some((on_path, treatment_nodes, _)) =
        path_assignments_over(graph, assignments, &treatments, outcome)
    else {
        return Ok(UnitSupport { parent_cell, disturbance, covariates, disturbance_nodes });
    };
    let variable_of =
        |node: DenseNodeId| assignments.iter().find(|a| a.node == node).map(|a| a.variable);
    // Range of `values` over the rows of each arm: `[control, active]`.
    let arm_ranges = |values: &[f64]| {
        let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 2];
        for (row, value) in values.iter().enumerate().take(n) {
            if !value.is_finite() {
                continue;
            }
            let arm = &mut ranges[usize::from(in_active[row])];
            arm.0 = arm.0.min(*value);
            arm.1 = arm.1.max(*value);
        }
        ranges
    };
    let flag = |values: &[f64], out: &mut [bool]| {
        let ranges = arm_ranges(values);
        for (row, value) in values.iter().enumerate().take(n) {
            // The opposite arm of a unit observed under `active` is `control`.
            let (lo, hi) = ranges[usize::from(!in_active[row])];
            // An arm with no observed rows has no support at all.
            if lo > hi || *value < lo || *value > hi {
                out[row] = true;
            }
        }
    };
    let mut seen = std::collections::BTreeSet::new();
    for assignment in &on_path {
        for &parent in graph.parents(assignment.node) {
            if treatment_nodes.contains(&parent)
                || treatment_nodes.iter().any(|&node| graph.reaches(node, parent))
            {
                continue;
            }
            if !seen.insert(parent.as_usize()) {
                continue;
            }
            let Some(variable) = variable_of(parent) else { continue };
            let values = data.float64_values(variable)?;
            flag(&values, &mut parent_cell);
            covariates.push(variable_name(data, variable));
        }
        if assignment.selected.has_additive_disturbance()
            && exogenous.n_units == n
            && assignment.node.as_usize() < exogenous.n_nodes
        {
            let start = assignment.node.as_usize() * n;
            flag(&exogenous.noise[start..start + n], &mut disturbance);
            disturbance_nodes.push(variable_name(data, assignment.variable));
        }
    }
    Ok(UnitSupport { parent_cell, disturbance, covariates, disturbance_nodes })
}

/// The support diagnostic of one outcome's contrast, naming each intervened variable's range.
fn joint_support_diagnostic(
    support: &UnitSupport,
    outcome: &str,
    ranges: &[(String, f64, f64)],
    pooled: bool,
) -> Diagnostic {
    let n = support.parent_cell.len();
    let cell = support.parent_cell.iter().filter(|f| **f).count();
    let disturbance = support.disturbance.iter().filter(|f| **f).count();
    let any = support.extrapolation_flags().iter().filter(|f| **f).count();
    let severity =
        if pooled || any > 0 { DiagnosticSeverity::Warning } else { DiagnosticSeverity::Info };
    let observed = ranges
        .iter()
        .map(|(name, min, max)| format!("{name}=[{min},{max}]"))
        .collect::<Vec<_>>()
        .join(",");
    Diagnostic::new(
        "gcm.counterfactual.support",
        DiagnosticKind::Support,
        severity,
        format!(
            "outcome={outcome}; observed treatment ranges=[{observed}]; extrapolative={pooled}; \
             per_unit_extrapolative={any}/{n} (covariate cell outside the opposite arm's \
             observed range: {cell}; abducted disturbance outside the opposite arm's residual \
             range: {disturbance}); covariates=[{}]; disturbance_nodes=[{}]",
            support.covariates.join(","),
            support.disturbance_nodes.join(",")
        ),
    )
    .with_fields([
        ("outcome", outcome.to_string()),
        ("extrapolative", pooled.to_string()),
        ("per_unit_extrapolative", any.to_string()),
        ("parent_cell_extrapolative", cell.to_string()),
        ("disturbance_extrapolative", disturbance.to_string()),
        ("n_units", n.to_string()),
    ])
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)] // SCM columns are named as in the graph (z, a, b, y).
mod tests {
    use super::super::{Dag, DenseNodeId, TabularData, UnitContrast, unit_support};
    use super::*;
    use crate::gcm::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// `z → a → y`, `z → y`, `b → y` with a binary `a` and an `a × b` interaction.
    fn table(n: usize) -> (TabularData, Dag) {
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.71).sin()).collect();
        let a: Vec<f64> = (0..n)
            .map(|i| f64::from(u8::from((i as f64 * 1.37).sin() + 0.8 * z[i] > 0.0)))
            .collect();
        let b: Vec<f64> =
            (0..n).map(|i| f64::from(u8::from((i as f64 * 2.11).cos() > 0.0))).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                0.8 * a[i] + 0.5 * b[i] + 0.6 * a[i] * b[i] + z[i] + 0.2 * (i as f64 * 0.29).sin()
            })
            .collect();
        let data = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("a", a.as_slice()),
            ("b", b.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(4);
        for (from, to) in [(0, 1), (0, 3), (1, 3), (2, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        (data, graph)
    }

    fn fitted(
        registry: &MechanismRegistry,
        data: &TabularData,
        graph: &Dag,
    ) -> Vec<crate::gcm::MechanismAssignment> {
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        registry.assign_and_fit(&compiled, data, SelectionPolicy::BestScore).unwrap().1
    }

    fn exogenous(data: &TabularData, graph: &Dag) -> crate::gcm::ExogenousPosterior {
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        let (store, _) = MechanismRegistry::with_heterogeneity_families()
            .assign_and_fit(&compiled, data, SelectionPolicy::BestScore)
            .unwrap();
        let engine = crate::gcm::CounterfactualEngine::new(compiled.with_mechanisms(store));
        engine
            .abduct(
                data,
                crate::gcm::AbductionMissingPolicy::Error,
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
    }

    #[test]
    fn support_over_one_treatment_is_the_single_treatment_support() {
        let (data, graph) = table(600);
        let assignments = fitted(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        let exo = exogenous(&data, &graph);
        let old = unit_support(
            &data,
            &graph,
            &assignments,
            &exo,
            UnitContrast { treatment: v(1), outcome: v(3), active: 1.0, control: 0.0 },
        )
        .unwrap();
        let new = unit_support_over(
            &data,
            &graph,
            &assignments,
            &exo,
            &[TreatmentLevels { variable: v(1), active: 1.0, control: 0.0 }],
            v(3),
        )
        .unwrap();
        assert_eq!(old.parent_cell, new.parent_cell);
        assert_eq!(old.disturbance, new.disturbance);
        assert_eq!(old.covariates, new.covariates);
        assert_eq!(old.disturbance_nodes, new.disturbance_nodes);
    }

    #[test]
    fn support_over_two_treatments_treats_each_as_moved_not_as_a_covariate() {
        let (data, graph) = table(600);
        let assignments = fitted(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        let exo = exogenous(&data, &graph);
        let both = unit_support_over(
            &data,
            &graph,
            &assignments,
            &exo,
            &[
                TreatmentLevels { variable: v(1), active: 1.0, control: 0.0 },
                TreatmentLevels { variable: v(2), active: 1.0, control: 0.0 },
            ],
            v(3),
        )
        .unwrap();
        assert_eq!(both.parent_cell.len(), 600);
        // `b` is intervened, so it is not a covariate held at the unit's own value.
        assert!(!both.covariates.iter().any(|name| name == "b"), "{:?}", both.covariates);
    }

    #[test]
    fn homogeneity_over_one_treatment_is_the_single_treatment_verdict() {
        let (data, graph) = table(600);
        for registry in
            [MechanismRegistry::standard(), MechanismRegistry::with_heterogeneity_families()]
        {
            let assignments = fitted(&registry, &data, &graph);
            let single = super::super::homogeneity_verdict(&graph, &assignments, v(1), v(3));
            let over = homogeneity_verdict_over(&graph, &assignments, &[v(1)], v(3));
            assert_eq!(format!("{single:?}"), format!("{over:?}"));
        }
    }

    #[test]
    fn homogeneity_over_two_treatments_is_heterogeneous_when_either_path_can_modify_the_effect() {
        let (data, graph) = table(1200);
        let assignments = fitted(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        // The outcome mechanism carries the `a × b` interaction, so the effect of `a` and of the
        // pair `(a, b)` can vary; a treatment missing from the model gives no structural claim.
        let joint = homogeneity_verdict_over(&graph, &assignments, &[v(1), v(2)], v(3));
        assert!(matches!(joint, HomogeneityVerdict::Heterogeneous), "{joint:?}");
        let absent = homogeneity_verdict_over(&graph, &assignments, &[v(9)], v(3));
        assert!(matches!(absent, HomogeneityVerdict::Heterogeneous));
    }
}
