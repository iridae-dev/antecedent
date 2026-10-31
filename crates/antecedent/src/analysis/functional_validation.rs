// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native validation of the functional queries: attribution, counterfactuals, nested and
//! path-specific effects, interventional distributions and conditional effects.
//!
//! The scalar refuters of the average-effect suites perturb the data and re-estimate one
//! treatment contrast; they do not denote for a vector of noise contributions, a unit-level
//! counterfactual, a cross-world functional or a law. A study of these queries that requests
//! `cheap` or `full` validation is therefore validated by the target-native suite of
//! [`target_validation`](super::target_validation): the study is prepared with no validation
//! (so the route runs exactly as at validation none), executed on the data, and the whole
//! target vector of that result is refuted by whole-procedure refits of the same compiled
//! program on perturbed data:
//!
//! * attribution (anomaly and change): the mean score and every noise component's mean
//!   Shapley contribution are refitted on random subsets of the rows (the held-out-population
//!   check of the fitted mechanisms) and, under `full`, on bootstrap resamples;
//! * an interventional distribution: the vector of atom probabilities is refitted the same
//!   way (the atoms are levels of a law, so the treatment-keyed null refuters do not apply);
//! * a counterfactual, a nested or path-specific effect and a conditional effect: the
//!   population value (the mean unit effect for a counterfactual) is refuted by the placebo
//!   treatment, which removes the effect the identified functional reads, and by subset
//!   stability; `full` adds the outcome placebo and bootstrap stability.
//!
//! On a table the refits are joined by `functional_sensitivity`: a sensitivity range of a
//! cross-world functional bound to its dependence parameter (with the break-even), the
//! cross-world placebo (the mediator permuted), the placebo of an interventional distribution
//! against the observed marginal of the outcome, and the noise-block permutation of an
//! attribution.
//!
//! Bayesian studies add the prior sensitivity of the posterior of the whole target where the
//! route retains its draws; a refuter the route cannot construct is recorded as a diagnostic
//! naming its next attempt, never as a pass. The state executes and stays unlicensed: its
//! verdicts are diagnostics, no coverage record names it, and the result reports the support
//! status `refused`.

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext,
    ResponseFunctional,
};
use antecedent_validate::RefutationReport;

use super::builder::{DataInput, RefuteSuite};
use super::execute::Study;
use super::prepared::PreparedStudy;
use super::target_validation::{
    PreparedTargetData, TargetRefitOptions, TargetRefutationConfig, TargetRefutationReport,
    TargetSuite, refute_prepared_target, result_target_values,
};
use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::result::StudyResult;

/// Replicates of a cheap refuter.
const CHEAP_REPLICATES: u32 = 30;
/// Replicates of a full refuter.
const FULL_REPLICATES: u32 = 60;

fn temporal_class_posterior_level(study: &Study) -> bool {
    use antecedent_discovery::GraphPosteriorAtomKind;
    let Some(posterior) = &study.graph_posterior else {
        return false;
    };
    posterior.max_lag().is_some()
        && matches!(
            posterior.atom_kind(),
            GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag
        )
        && matches!(&study.query, CausalQuery::Response(query)
            if matches!(&query.functional, ResponseFunctional::InterventionResponse { interventions, .. } if interventions.len() == 1))
}

/// Whether `study` requests a validation suite whose native suite is the functional one.
///
/// A path-specific effect and an interventional distribution on a supplied Dag keep their
/// own query-native suites; a conditional effect on a Dag, Cpdag or Pag keeps the scalar
/// suites. Every other cell of these queries is validated here.
pub(crate) fn requested(study: &Study) -> bool {
    if study.settings.refute == RefuteSuite::None {
        return false;
    }
    if temporal_class_posterior_level(study) {
        return true;
    }
    let class = study.graph.class();
    match &study.query {
        CausalQuery::AnomalyAttribution(_)
        | CausalQuery::ChangeAttribution(_)
        | CausalQuery::Counterfactual(_)
        | CausalQuery::NestedCounterfactual(_) => true,
        CausalQuery::Distribution(_) | CausalQuery::PathSpecific(_) => class != GraphClass::Dag,
        CausalQuery::ConditionalEffect(_) => {
            matches!(class, GraphClass::Admg | GraphClass::TemporalAdmg)
        }
        // A mean curve, or an intervention level over several coordinates, over a graph posterior
        // of mixed or class atoms has no atom-native scalar suite: its mixed value is refuted by
        // refitting the whole graph-posterior procedure. A curve or one-coordinate level over Dag or
        // TemporalDag atoms keeps its own suites.
        CausalQuery::Response(query) if study.graph_posterior.is_some() => {
            let mixed_or_class = matches!(
                class,
                GraphClass::Admg | GraphClass::TemporalAdmg | GraphClass::Cpdag | GraphClass::Pag
            );
            (mixed_or_class && matches!(query.functional, ResponseFunctional::MeanCurve { .. }))
                || ((mixed_or_class || class == GraphClass::Dag)
                    && matches!(
                        &query.functional,
                        ResponseFunctional::InterventionResponse { interventions, .. }
                            if interventions.len() > 1
                    ))
        }
        CausalQuery::TemporalEffect(query) => {
            !query.effect_modifiers.is_empty()
                && matches!(
                    class,
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                )
        }
        _ => false,
    }
}

/// The target-native suite a study's scalar suite requests.
const fn target_suite(suite: RefuteSuite) -> TargetSuite {
    match suite {
        RefuteSuite::None | RefuteSuite::Cheap => TargetSuite::Cheap,
        RefuteSuite::PlaceboAndRcc => TargetSuite::Placebo,
        RefuteSuite::Full => TargetSuite::Full,
    }
}

/// Replicate count of a refuter on a target of `dimension` coordinates.
///
/// The study's own bootstrap budget belongs to the interval of the estimate, which every
/// refit recomputes; the refuters' replicates are their own.
fn replicates(suite: TargetSuite, dimension: usize) -> u32 {
    let base = match suite {
        TargetSuite::Full => FULL_REPLICATES,
        TargetSuite::Cheap | TargetSuite::Placebo => CHEAP_REPLICATES,
    };
    // The covariance of a `k`-coordinate vector needs at least `k + 2` replicates, so the count
    // grows with the dimension of the target rather than being cut below that floor.
    let floor = u32::try_from(dimension.saturating_add(8)).unwrap_or(u32::MAX);
    base.max(floor)
}

/// One target-native report as a result refutation: the first coordinate of the original and
/// refuted vectors, and the tail probability (or the statistic) of the joint test.
fn as_refutation(report: &TargetRefutationReport, program: Option<[u8; 32]>) -> RefutationReport {
    let first = |values: &[f64]| values.first().copied().unwrap_or(f64::NAN);
    let comparison = report.statistic.p_value.unwrap_or(report.statistic.value);
    let mut out = RefutationReport::new(
        report.refuter(),
        first(&report.original),
        first(&report.refuted),
        comparison,
        report.informative,
        report.passed,
        report.failure.clone(),
        report.replicates,
    );
    if let Some(program) = program {
        out.stamp_producer(program);
    }
    out
}

/// [`run_validated`] over `data`, the rows a prepared handle was asked to estimate.
///
/// The study is cloned once, to prepare the route at validation none over `data`; the
/// refuters read the requested suite and the query from `study` itself.
///
/// # Errors
///
/// As [`run_validated`].
pub(crate) fn validate_data(
    study: &Study,
    data: DataInput,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let mut base = study.clone();
    base.settings.refute = RefuteSuite::None;
    base.retain_response_witnesses = temporal_class_posterior_level(study);
    if base.retain_response_witnesses {
        base.temporal_validation_budget =
            Some(antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits()));
    }
    if temporal_class_posterior_level(study) {
        base.declared.custom_validators.clear();
    }
    base.data = data;
    validate_over(study, &base, ctx)
}

/// Run `study` at validation none and refute the whole target vector of its result.
///
/// # Errors
///
/// The study cannot be prepared or executed at validation none, its result carries no target
/// vector, the data modality has no refit, cancellation, an admission refusal, a failed
/// refit, or no refuter of the requested suite could run.
pub(crate) fn run_validated(
    study: &Study,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let mut base = study.clone();
    base.settings.refute = RefuteSuite::None;
    base.retain_response_witnesses = temporal_class_posterior_level(study);
    if base.retain_response_witnesses {
        base.temporal_validation_budget =
            Some(antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits()));
    }
    if temporal_class_posterior_level(study) {
        base.declared.custom_validators.clear();
    }
    validate_over(study, &base, ctx)
}

/// Estimate `base` (the study at validation none) on its own data and refute the result
/// against the suite `study` requests.
fn validate_over(
    study: &Study,
    base: &Study,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let prepared = base.prepare(ctx)?;
    match &base.data {
        DataInput::Tabular(data) => {
            let result = prepared.estimate(data, ctx)?;
            validate(study, &prepared, result, PreparedTargetData::Tabular(data), ctx)
        }
        DataInput::Temporal(data) | DataInput::Event(data) => {
            let result = prepared.estimate_series(data, ctx)?;
            validate(study, &prepared, result, PreparedTargetData::Series(data), ctx)
        }
        DataInput::Panel(data) => {
            let result = prepared.estimate_panel(data, ctx)?;
            validate(study, &prepared, result, PreparedTargetData::Panel(data), ctx)
        }
        DataInput::MultiEnv(data) => {
            let result = prepared.estimate_multi_env(data, ctx)?;
            validate(study, &prepared, result, PreparedTargetData::MultiEnv(data), ctx)
        }
    }
}

// Keep set/graph-dependent point publication intact. Each report describes one
// point-producing completion's level under full-program refits, never the set.
fn validate_temporal_members(
    mut result: StudyResult,
    prepared: &PreparedStudy,
    data: PreparedTargetData<'_>,
    suite: TargetSuite,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let Some(witnesses) = result.temporal_posterior_witnesses.clone() else {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior", DiagnosticKind::Scientific,
            DiagnosticSeverity::Info, "completion-level refutations were unavailable because point-time witnesses were not retained",
        ));
        return Ok(result);
    };
    let expected = result
        .structural_response
        .as_ref()
        .map_or(0, |mixture| mixture.atoms.iter().filter(|atom| atom.value.is_some()).count());
    if witnesses.members.len() != expected {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior", DiagnosticKind::Scientific,
            DiagnosticSeverity::Info, "completion-level refutations cover only retained point-time members; they do not certify the outer set",
        ));
    }
    let program = result.lowered_identity.as_ref().map(|identity| *identity.as_bytes());
    // One member has at most the fixed target-native suite's report count.
    // Fund their retained vector, scopes and labels before running the suite.
    let mut member_count = 0u64;
    for member in &witnesses.members {
        member_count = member_count
            .checked_add(u64::try_from(member.witnesses.entries.len()).map_err(|_| {
                CausalError::Resource { message: "temporal report count overflow".into() }
            })?)
            .ok_or_else(|| CausalError::Resource {
                message: "temporal report count overflow".into(),
            })?;
    }
    let max_reports = member_count.checked_mul(16).ok_or_else(|| CausalError::Resource {
        message: "temporal report count overflow".into(),
    })?;
    let bytes =
        max_reports.checked_mul(4096).and_then(|n| n.checked_add(4096)).ok_or_else(|| {
            CausalError::Resource { message: "temporal report bytes overflow".into() }
        })?;
    if bytes > ctx.temporal_materialization_limits().max_bytes {
        return Err(CausalError::Resource { message: "temporal report byte limit".into() });
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal response refutation" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?;
    let mut budget = prepared.study().temporal_validation_budget.clone().unwrap_or_else(|| {
        antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
    });
    let suite_work = max_reports
        .checked_mul(u64::from(replicates(suite, 1)))
        .ok_or_else(|| CausalError::Resource { message: "temporal report work overflow".into() })?;
    budget
        .charge_work(suite_work)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let mut reports = Vec::new();
    let mut refutation_notes = Vec::new();
    reports
        .try_reserve_exact(usize::try_from(max_reports).map_err(|_| CausalError::Resource {
            message: "temporal report count exceeds address space".into(),
        })?)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    result
        .retained_resources
        .try_reserve_exact(1)
        .map_err(|error| CausalError::Resource { message: error.to_string() })?;
    let checked = super::target_validation::checked_refutation_input(&result, data, ctx)?;
    for member in &witnesses.members {
        for witness in &member.witnesses.entries {
            let config = TargetRefutationConfig {
                replicates: replicates(suite, 1),
                ..TargetRefutationConfig::default()
            };
            let (scope, run) = super::target_validation::refute_temporal_response_member(
                prepared,
                &result,
                data,
                member.outer_atom,
                witness,
                suite,
                &config,
                &checked,
                &mut budget,
                ctx,
            )?;
            for report in &run.reports {
                let mut report = as_refutation(report, program);
                report.scope = Some(scope.clone());
                reports.push(report);
            }
            for refusal in &run.refused {
                refutation_notes.push(Diagnostic::new(
                    "refute.temporal_completion.refused",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "refuter `{}` did not run for one completion: {}",
                        refusal.kind.as_str(),
                        refusal.reason
                    ),
                ));
            }
        }
    }
    drop(checked);
    result.view.diagnostics.extend(refutation_notes);
    if reports.is_empty() {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "no checked completion-level refuter produced a report",
        ));
        // Refusal notes remain in the result, so retain their admitted owner.
        // Do not mutably access the report view: the point result remains valid.
        result.retained_resources.push(std::sync::Arc::new(lease));
        return Ok(result);
    }
    result.refutations = reports;
    result.retained_resources.push(std::sync::Arc::new(lease));
    if !result.refutations.is_empty() {
        super::target_validation::seal_runtime_refutations(
            &mut result,
            crate::result::RuntimeRefutationOrigin::ScopedCompletion,
        )?;
    }
    Ok(result)
}

fn validate(
    study: &Study,
    prepared: &PreparedStudy,
    mut result: StudyResult,
    data: PreparedTargetData<'_>,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    if temporal_class_posterior_level(study) && !study.declared.custom_validators.is_empty() {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.temporal_response.custom_unavailable", DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "scalar custom validators do not check temporal response levels; no custom report was produced",
        ));
    }
    let suite = target_suite(study.settings.refute);
    if temporal_class_posterior_level(study) && result_target_values(&result).is_err() {
        return validate_temporal_members(result, prepared, data, suite, ctx);
    }
    let (_, _, values) = result_target_values(&result)?;
    let config = TargetRefutationConfig {
        replicates: replicates(suite, values.len()),
        ..TargetRefutationConfig::default()
    };
    let outcome = refute_prepared_target(
        prepared,
        &result,
        data,
        suite,
        &config,
        &TargetRefitOptions::default(),
        ctx,
    )?;
    if outcome.run.reports.is_empty() && temporal_class_posterior_level(study) {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.envelope.temporal_class_graph_posterior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "no whole-target check ran with all original temporal completion contributors",
        ));
        return Ok(result);
    }
    if outcome.run.reports.is_empty() {
        let first = outcome.run.refused.first().map_or_else(
            || "the suite names no refuter for this target".to_owned(),
            |refusal| {
                format!(
                    "{}: {}; next attempt: {}",
                    refusal.kind.as_str(),
                    refusal.reason,
                    refusal.next_attempt
                )
            },
        );
        return Err(CausalError::Compile {
            message: format!("no refuter of the requested suite could run: {first}"),
        });
    }
    let program = result.lowered_identity.as_ref().map(|identity| *identity.as_bytes());
    let (extra_reports, extra_notes) = match data {
        PreparedTargetData::Tabular(table) => super::functional_sensitivity::extra_refuters(
            &super::functional_sensitivity::ExtraInput {
                study,
                prepared,
                result: &result,
                data: table,
                replicates: config.replicates,
                alpha: config.alpha,
                ctx,
                program,
            },
        )?,
        _ => (Vec::new(), Vec::new()),
    };
    result.refutations = outcome.run.reports.iter().map(|r| as_refutation(r, program)).collect();
    if let Some(report) = curve_atom_support(&result, program) {
        result.refutations.push(report);
    }
    result.refutations.extend(extra_reports);
    result.view.diagnostics.extend(extra_notes);
    for refusal in &outcome.run.refused {
        result.view.diagnostics.push(Diagnostic::new(
            "refute.functional.refuter_refused",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "refuter `{}` did not run: {}; next attempt: {}",
                refusal.kind.as_str(),
                refusal.reason,
                refusal.next_attempt
            ),
        ));
    }
    let licensed = matches!(study.support_status(), Some(crate::support::CellStatus::Licensed));
    result.view.diagnostics.push(Diagnostic::new(
        "refute.functional.state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "functional refuter state: the {}-coordinate target vector of the result was \
             refitted by its own compiled program on perturbed data ({} replicates per \
             refuter) and each refuter reports one joint statistic over the whole vector. {} \
             The verdicts are diagnostics, not a calibration claim.",
            values.len(),
            config.replicates,
            if licensed {
                "The validated cell is licensed on known-truth refuter evidence."
            } else {
                "It executes unlicensed: no coverage or evidence record names it."
            }
        ),
    ));
    // The handle's contract names a run with no validation, which does not describe this
    // result. The result carries the status of the validated cell itself: licensed when a row
    // names it, refused otherwise.
    result.support_status = Some(if licensed {
        crate::support::CellStatus::Licensed
    } else {
        crate::support::CellStatus::Refused
    });
    result.executed_contract = None;
    super::target_validation::seal_runtime_refutations(
        &mut result,
        crate::result::RuntimeRefutationOrigin::WholeTarget,
    )?;
    Ok(result)
}

/// The support of a mixed curve over the atoms of its graph posterior: every atom that carries
/// posterior weight either published a finite value on the whole grid or is accounted for as
/// unidentified or unevaluable mass. `None` when the result is not a curve over a structural
/// mixture.
fn curve_atom_support(result: &StudyResult, program: Option<[u8; 32]>) -> Option<RefutationReport> {
    let response = result.response()?;
    if !matches!(response.estimand, ResponseFunctional::MeanCurve { .. }) {
        return None;
    }
    let mixture = result.structural_response.as_ref()?;
    let mut weight = 0.0;
    let mut finite = 0.0;
    for atom in &mixture.atoms {
        weight += atom.weight;
        let grid_finite = match &atom.value {
            Some(antecedent_core::ResponseValue::Scalar(value)) => value.is_finite(),
            Some(antecedent_core::ResponseValue::Vector(values)) => {
                values.iter().all(|value| value.is_finite())
            }
            Some(antecedent_core::ResponseValue::Surface { mean, .. }) => {
                mean.iter().all(|value| value.is_finite())
            }
            _ => false,
        };
        if grid_finite {
            finite += atom.weight;
        }
    }
    if weight <= 0.0 {
        return None;
    }
    let covered = finite / weight;
    let accounted =
        (mixture.identified_mass + mixture.unidentified_mass + mixture.unevaluable_mass)
            .max(f64::MIN_POSITIVE);
    let passed = covered > 0.0 && (accounted - 1.0).abs() < 1e-6;
    let mut report = RefutationReport::new(
        "curve.atom_support",
        mixture.identified_mass,
        covered,
        accounted,
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::<str>::from(
                "no atom published a finite curve on the grid, or the posterior mass is not \
                 accounted for",
            )
        }),
        u32::try_from(mixture.atoms.len()).unwrap_or(u32::MAX),
    );
    if let Some(program) = program {
        report.stamp_producer(program);
    }
    Some(report)
}
