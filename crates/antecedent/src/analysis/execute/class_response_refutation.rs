// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a class-aware (Cpdag / Pag, explicit or accepted) intervention level.
//!
//! Each evaluated completion is an atom of the identified set. The plugin-level suite
//! runs on every atom's own level against that atom's own adjustment estimand (overlap
//! under `cheap`; overlap plus per-completion sampling stability under `full`), exactly
//! as the graph-posterior cells of these classes do. Reports are mixed into one verdict
//! only when every atom answers the same level; otherwise each atom's reports are kept.
//! A disagreement diagnostic states how far the completions' levels spread. The state
//! executes and is unsupported.

use super::atom_mix::{ScalarRefuterMix, finish_atom_refutations};
use super::response_path::response_scalar_summary;
use super::{
    AverageEffectQuery, CausalError, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimationWorkspace, EstimatorId, ExecutionContext,
    OverlapPolicy, TabularData, VariableId,
};

/// Relative tolerance under which completion levels count as one level.
const SAME_LEVEL_TOLERANCE: f64 = 1e-9;

/// Run the per-completion refuter state of a class-aware intervention level.
///
/// # Errors
///
/// Cancellation or a validator failure.
#[allow(
    clippy::too_many_lines,
    reason = "native intervention refits share one completion traversal"
)]
pub(super) fn refute_class_intervention_atoms<G>(
    study: &super::Study,
    (data, query): (&TabularData, &antecedent_core::ResponseQuery),
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
    weighted: &[(u64, f64, CausalResponse)],
    (treatment, outcome): (VariableId, VariableId),
    estimator: EstimatorId,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let mut ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
    if let antecedent_core::ResponseFunctional::InterventionResponse { interventions, .. } =
        &query.functional
    {
        if let [action] = interventions.as_ref() {
            ate_query.active = action.clone();
        }
    }
    ate_query.target_population = query.target_population.clone();
    let mut workspace = EstimationWorkspace::default();
    let mut per_atom = Vec::new();
    let mut diagnostics = Vec::new();
    let mut levels = Vec::new();
    for (key, weight, response) in weighted {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let Some(case) = usize::try_from(*key).ok().and_then(|index| envelope.cases.get(index))
        else {
            continue;
        };
        let Some(estimand) = case.result.estimands.first() else { continue };
        let (level, se) = response_scalar_summary(response);
        if !level.is_finite() || *weight <= 0.0 {
            continue;
        }
        levels.push(level);
        let estimate = EffectEstimate::new(
            level,
            se,
            case.result.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let operational = std::cell::RefCell::new(None);
        let refitter = super::class_posterior_response::StaticResponseLevelRefit {
            study,
            query,
            estimand,
            status: case.result.status,
            assumptions: &case.result.required_assumptions,
            operational: Some(&operational),
        };
        let problem = antecedent_validate::RefutationProblem::new(
            data,
            estimand,
            &ate_query,
            &estimate,
            Some(estimator.as_str()),
            None,
        )
        .with_effect_refit(&refitter);
        let suite = match study.settings.refute {
            super::RefuteSuite::None => antecedent_validate::ValidationSuite::new(),
            super::RefuteSuite::Cheap | super::RefuteSuite::PlaceboAndRcc => {
                antecedent_validate::ValidationSuite::overlap_only()
            }
            super::RefuteSuite::Full => antecedent_validate::ValidationSuite::new()
                .with(antecedent_validate::ValidatorId::Overlap)
                .with(antecedent_validate::ValidatorId::OverlapRule)
                .with(antecedent_validate::ValidatorId::Bootstrap)
                .with(antecedent_validate::ValidatorId::DataSubset),
        };
        let outcomes = suite.run(&problem, &mut workspace, ctx);
        if let Some(error) = operational.borrow_mut().take() {
            return Err(CausalError::from(error));
        }
        let outcomes = outcomes.map_err(CausalError::from)?;
        diagnostics.extend(super::validator_not_applicable_diagnostics(&outcomes));
        let mut reports = antecedent_validate::ValidationSuite::reports_only(&outcomes);
        if matches!(study.settings.refute, super::RefuteSuite::Full) {
            match super::class_posterior_response::level_adjustment_sensitivity(
                &refitter, data, &estimate, treatment, outcome, ctx,
            ) {
                Ok(report) => reports.push(report),
                Err(antecedent_validate::ValidationError::NotApplicable { message }) => diagnostics
                    .push(
                        Diagnostic::new(
                            "refute.validator.not_applicable",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Info,
                            message,
                        )
                        .with_fields([("validator", "graph")]),
                    ),
                Err(error) => {
                    let reason = error.to_string();
                    let causal = CausalError::from(error);
                    if matches!(
                        causal,
                        CausalError::Cancelled { .. } | CausalError::Resource { .. }
                    ) {
                        return Err(causal);
                    }
                    diagnostics.push(
                        Diagnostic::new(
                            "refute.validator.failed",
                            DiagnosticKind::Scientific,
                            DiagnosticSeverity::Warning,
                            format!("validator graph could not compute a verdict: {reason}"),
                        )
                        .with_fields([("validator", "graph"), ("reason", reason.as_str())]),
                    );
                }
            }
        }
        per_atom.push((*weight, reports));
    }
    let lower = levels.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spread = if levels.is_empty() { f64::NAN } else { upper - lower };
    let same_level = spread.is_finite()
        && spread <= SAME_LEVEL_TOLERANCE * lower.abs().max(upper.abs()).max(1.0);
    diagnostics.push(
        Diagnostic::new(
            "refute.response.class_envelope_disagreement",
            DiagnosticKind::Scientific,
            if same_level { DiagnosticSeverity::Info } else { DiagnosticSeverity::Warning },
            format!(
                "{} evaluated completion(s); completion levels span [{lower}, {upper}] \
                 (spread {spread}). {}",
                levels.len(),
                if same_level {
                    "Every completion answers the same level, so the per-completion refuter \
                     reports are mixed by completion weight."
                } else {
                    "The completions disagree, so each completion's refuter reports are kept \
                     on their own; no mixed verdict is formed over an identified set."
                }
            ),
        )
        .with_fields([
            ("completions", levels.len().to_string()),
            ("lower", lower.to_string()),
            ("upper", upper.to_string()),
        ]),
    );
    diagnostics.push(Diagnostic::new(
        "refute.response.class_state",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "class-aware intervention-level refuter state: plugin-level overlap (cheap) and \
         overlap plus per-completion sampling stability (full) on each completion's own \
         level and estimand. It executes unsupported: no coverage or evidence record names it, \
         and the verdicts are checks, not a calibration claim.",
    ));
    Ok(finish_atom_refutations(
        per_atom,
        diagnostics,
        same_level,
        ScalarRefuterMix::Effect,
        None,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{CausalQuery, Intervention, ResponseFunctional, ResponseQuery, Value};
    use antecedent_graph::{Cpdag, DenseNodeId};

    fn fixture() -> (TabularData, Cpdag) {
        let z: Vec<f64> = (0..400).map(|i| f64::from((i * 31) % 103) / 20.0 - 2.5).collect();
        let t: Vec<f64> = z
            .iter()
            .enumerate()
            .map(|(i, z)| 0.5 * z + f64::from(u32::try_from((i * 17) % 101).unwrap()) / 20.0 - 2.5)
            .collect();
        let y: Vec<f64> = t
            .iter()
            .zip(&z)
            .enumerate()
            .map(|(i, (t, z))| {
                1.0 + 2.0 * t + z + f64::from(u32::try_from((i * 13) % 29).unwrap()) / 100.0
            })
            .collect();
        let data = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("t", t.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Cpdag::with_variables(3);
        graph.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        (data, graph)
    }

    #[test]
    fn class_intervention_full_refits_the_requested_level() {
        let (data, graph) = fixture();
        let treatment = VariableId::from_raw(1);
        let outcome = VariableId::from_raw(2);
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome,
            interventions: std::sync::Arc::from([Intervention::set(treatment, Value::f64(1.5))]),
        });
        let result = super::super::Study::tabular(data)
            .graph(graph)
            .query(CausalQuery::Response(query))
            .refute(super::super::RefuteSuite::Full)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(13))
            .unwrap();
        let reports = &result.refutations;
        let bootstrap = reports
            .iter()
            .filter(|report| report.refuter.as_ref() == "bootstrap.ci_coverage")
            .collect::<Vec<_>>();
        assert!(
            !bootstrap.is_empty(),
            "the full suite must run a bootstrap level refit: {reports:?}"
        );
        for report in bootstrap {
            assert!(
                (report.refuted_ate - report.original_ate).abs() < 0.8,
                "bootstrap must refit the intervention level, not a contrast: {report:?}"
            );
        }
    }
}
