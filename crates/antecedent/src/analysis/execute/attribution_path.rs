// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::StreamDomain;

impl super::Study {
    pub(super) fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (treatment, active, control) = binary_cf_interventions(query)?;
        let outcome = query.outcomes[0];
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    IdentifierId::GcmParametric,
                    graph,
                    &CausalQuery::Counterfactual(query.clone()),
                )?;
                let estimand = identification.estimands[0].clone();
                Ok((identification, estimand))
            })?;
        if let InferenceMode::Bayesian(cfg) = &self.inference {
            if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() || cfg.prior.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian counterfactuals require a declared mechanism mapping; \
                              a coefficient artifact cannot be applied as an isotropic GCM prior \
                              or hydrated onto fitted GCM mechanisms",
                });
            }
        }
        let fitted = fit_gcm_counterfactual(graph.clone(), data)?;
        let assignments = format!("{:?}", fitted.assignments);
        let mechanism_assignments = fitted.assignments.clone();
        let base_model = fitted.model.clone();
        let ite = counterfactual_ite(fitted.model, data, treatment, outcome, active, control, ctx)?;
        let (estimate, posterior, ite) = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            let n_draws = bayesian_draw_count(&self.inference)?;
            let n_units = ite.unit_effects.len();
            let mut values = Vec::with_capacity(n_draws);
            // Column-major `units × draws`: one posterior column per unit, read by
            // the same summary that publishes the mean-ITE interval.
            let mut unit_draws = vec![0.0; n_units * n_draws];
            let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0x0CF0);
            let n = data.row_count();
            for draw in 0..n_draws {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_ESTIMATE_POINT,
                    });
                }
                let weights: Vec<f64> = (0..n)
                    .map(|_| (-rng.next_f64().max(f64::MIN_POSITIVE).ln()).max(0.0))
                    .collect();
                // The same registry the point fit selected from, so every family a
                // node can have selected is one `refit_weighted` accepts.
                let store = crate::gcm::MechanismRegistry::with_heterogeneity_families()
                    .refit_weighted(&base_model, data, &mechanism_assignments, &weights)
                    .map_err(|e| match e {
                        crate::gcm::ModelError::NotConverged { .. } => map_mechanism_fit(e),
                        other => CausalError::Compile { message: other.to_string() },
                    })?;
                let draw_model = base_model.clone().with_mechanisms(store);
                let draw_ite =
                    counterfactual_ite(draw_model, data, treatment, outcome, active, control, ctx)?;
                if draw_ite.unit_effects.len() != n_units {
                    return Err(CausalError::Compile {
                        message: "counterfactual Bayesian draw changed the unit set".into(),
                    });
                }
                values.push(draw_ite.mean_ite);
                for (unit, effect) in draw_ite.unit_effects.iter().enumerate() {
                    unit_draws[unit * n_draws + draw] = *effect;
                }
            }
            let posterior = counterfactual_posterior(
                values,
                identification.required_assumptions.clone(),
                identification.status,
            )?;
            let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                message: "counterfactual posterior missing effect column".into(),
            })?;
            // Published unit effects stay the plain draw average (sequential sum,
            // divided once), exactly as before per-unit intervals existed.
            let scale = n_draws.max(1) as f64;
            let unit_means: Vec<f64> = (0..n_units)
                .map(|unit| {
                    unit_draws[unit * n_draws..(unit + 1) * n_draws].iter().fold(0.0, |s, v| s + v)
                        / scale
                })
                .collect();
            let unit_summary = unit_effect_posterior_summary(unit_draws, n_units, n_draws)?;
            let mut ite = ite;
            ite.unit_effects = std::sync::Arc::from(unit_means);
            ite.unit_effect_intervals = Some(crate::gcm::UnitEffectIntervals {
                lower: std::sync::Arc::clone(&unit_summary.q025),
                upper: std::sync::Arc::clone(&unit_summary.q975),
                level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                method: antecedent_core::IntervalMethod::UnitPosteriorQuantile.as_str(),
            });
            ite.mean_ite = posterior.summaries.mean[eq];
            let mut estimate = EffectEstimate::new(
                posterior.summaries.mean[eq],
                posterior.summaries.sd[eq],
                posterior.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
            );
            estimate.se_analytic = posterior.summaries.sd[eq];
            (estimate, Some(posterior), ite)
        } else {
            (
                EffectEstimate::new(
                    ite.mean_ite,
                    f64::NAN,
                    identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                None,
                ite,
            )
        };
        let verdict = homogeneity_verdict(graph, &mechanism_assignments, treatment, outcome);
        let homogeneous = match &verdict {
            HomogeneityVerdict::Structural(family) => Some(*family),
            _ => None,
        };
        let mut estimate = estimate;
        estimate.unit_effects_homogeneous = homogeneous.is_some();
        let observed = data.float64_values(treatment)?;
        let min = observed.iter().copied().fold(f64::INFINITY, f64::min);
        let max = observed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pooled_extrapolative = control < min || control > max || active < min || active > max;
        let support = unit_support(
            data,
            graph,
            &mechanism_assignments,
            &ite.exogenous,
            treatment,
            outcome,
            active,
            control,
        )?;
        let mut ite = ite;
        ite.unit_extrapolative = Some(std::sync::Arc::from(support.extrapolation_flags()));
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
                    "noise_inference={:?}; control={control}; active={active}",
                    ite.noise_inference
                ),
            ),
            support.support_diagnostic(min, max, pooled_extrapolative),
            if posterior.is_some() {
                Diagnostic::new(
                    "gcm.counterfactual.bayesian",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Dirichlet row-weight posterior of fitted GCM mechanisms, conditional on the selected families and empirical support; the family selection is made once on the unweighted data and is not revisited per draw. Each draw abducts–acts–predicts on the original units; published unit_effects are the posterior mean of those per-unit ITEs and unit_effect_intervals the equal-tailed posterior quantiles of each unit's ITE draws at the reported level. Both the mean_ite interval and the per-unit intervals carry mechanism-refit uncertainty only: abducted disturbances are recomputed from the observed rows, not drawn, so a per-unit interval is a credible interval for that observed unit's contrast under the fitted mechanism, not a predictive interval for a new unit, and the mean_ite interval is for the average over the observed units, not a population beyond them.",
                )
            } else {
                Diagnostic::new(
                    "gcm.counterfactual.uncertainty_unavailable",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Unit effects condition on fitted mechanisms and abducted disturbances; sampling uncertainty is unavailable.",
                )
            },
        ];
        if let Some(family) = homogeneous {
            let outcome_name = data
                .schema()
                .get(outcome)
                .map_or_else(|_| format!("v{}", outcome.raw()), |v| v.name.to_string());
            diagnostics.push(
                Diagnostic::new(
                    "gcm.counterfactual.unit_effects_homogeneous",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    format!(
                        "mechanism family {family:?} for {outcome_name} admits no effect \
                         modification; unit_effects equal the mechanism slope for every unit"
                    ),
                )
                .with_fields([
                    ("outcome", outcome_name),
                    ("family", family.id().to_string()),
                    ("unit_effect", ite.mean_ite.to_string()),
                ]),
            );
        }
        if let HomogeneityVerdict::Empirical { rejected } = &verdict {
            diagnostics.push(heterogeneity_rejected_diagnostic(data, rejected));
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GcmParametric,
            estimator_id: EstimatorId::GcmFit,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                gcm: Some(GcmSlot::Counterfactual(ite)),
                bootstrap_replicates_requested: Some(None),
                identify_provenance: Some(provenance_ids(
                    "identify.gcm_parametric",
                    "identify.gcm_parametric",
                )),
                estimate_provenance: Some(provenance_ids(
                    "counterfactual.aap",
                    "counterfactual.aap",
                )),
                n_draws: posterior
                    .as_ref()
                    .map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
                posterior,
                ..Default::default()
            },
        }))
    }

    pub(super) fn execute_anomaly(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::AnomalyAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                Ok(parametric_scm_identification(
                    CausalQuery::AnomalyAttribution(query.clone()),
                    outcome,
                    outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let scores = anomaly_attribution(
            &fitted.model,
            data,
            query.targets.iter().copied(),
            query.max_units,
        )?;
        Ok(self.finish_gcm(
            physical,
            CausalQuery::AnomalyAttribution(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Anomaly(scores),
            Vec::new(),
            identify_cached,
        ))
    }

    pub(super) fn execute_change_attribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::ChangeAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (_, _, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                Ok(parametric_scm_identification(
                    CausalQuery::ChangeAttribution(query.clone()),
                    query.outcome,
                    query.outcome,
                ))
            })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &antecedent_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let estimate = EffectEstimate::new(
            result.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        Ok(self.finish_gcm(
            physical,
            CausalQuery::ChangeAttribution(query.clone()),
            query.outcome,
            query.outcome,
            estimate,
            started,
            GcmSlot::Change(result),
            Vec::new(),
            identify_cached,
        ))
    }

    pub(super) fn execute_mechanism_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::MechanismChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let detections = mechanism_change_detection(
            &fitted.model,
            data,
            query,
            antecedent_attribution::MechanismChangeMethod::LikelihoodRatio,
            ctx,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        Ok(self.finish_gcm(
            physical,
            CausalQuery::MechanismChange(query.clone()),
            outcome,
            outcome,
            nan_effect(),
            started,
            GcmSlot::Mechanism(detections),
            Vec::new(),
            false,
        ))
    }

    pub(super) fn execute_unit_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &antecedent_core::UnitChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_unit_change(&fitted.model, data, query, ctx)?;
        Ok(self.finish_gcm(
            physical,
            CausalQuery::UnitChange(query.clone()),
            query.outcome,
            query.outcome,
            nan_effect(),
            started,
            GcmSlot::Unit(result),
            Vec::new(),
            false,
        ))
    }

    fn finish_gcm(
        &self,
        physical: &PhysicalExecutionPlan,
        query: CausalQuery,
        treatment: VariableId,
        outcome: VariableId,
        estimate: EffectEstimate,
        started: Instant,
        slot: GcmSlot,
        diagnostics: Vec<Diagnostic>,
        identify_cached: bool,
    ) -> StudyResult {
        let (identification, estimand) = parametric_scm_identification(query, treatment, outcome);
        self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id: IdentifierId::GcmParametric,
            estimator_id: EstimatorId::GcmFit,
            treatment,
            outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                diagnostics: Some(diagnostics),
                gcm: Some(slot),
                empty_provenance: true,
                ..Default::default()
            },
        })
    }
}

/// Why the published per-unit effects do, or do not, vary.
#[derive(Debug)]
enum HomogeneityVerdict {
    /// Some mechanism on a `treatment → outcome` path can modify the effect.
    Heterogeneous,
    /// Every path mechanism is additively separable, and no family that could
    /// have modified the effect was fit anywhere on those paths: the per-unit
    /// effects are equal by construction of the candidate set. Carries the
    /// outcome's selected family.
    Structural(crate::gcm::MechanismFamily),
    /// Every path mechanism is additively separable, but a heterogeneity-capable
    /// family *was* fit on some path node and lost on validation score: equal
    /// per-unit effects are a finding about the data, not a property fixed in
    /// advance.
    Empirical {
        /// `(node variable, rejected family, its score, selected family, selected score)`.
        rejected: Vec<RejectedHeterogeneity>,
    },
}

/// A heterogeneity-capable family scored on a path node and not selected.
#[derive(Debug)]
struct RejectedHeterogeneity {
    variable: VariableId,
    family: crate::gcm::MechanismFamily,
    score: f64,
    selected: crate::gcm::MechanismFamily,
    selected_score: f64,
}

/// Assignments whose mechanisms are re-evaluated in the acted world: strictly
/// downstream of the treatment and upstream of (or at) the outcome.
fn path_assignments<'a>(
    graph: &Dag,
    assignments: &'a [crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> Option<(Vec<&'a crate::gcm::MechanismAssignment>, DenseNodeId, DenseNodeId)> {
    let node_of =
        |variable: VariableId| assignments.iter().find(|a| a.variable == variable).map(|a| a.node);
    let treatment_node = node_of(treatment)?;
    let outcome_node = node_of(outcome)?;
    let on_path = assignments
        .iter()
        .filter(|a| {
            a.node != treatment_node
                && graph.reaches(treatment_node, a.node)
                && graph.reaches(a.node, outcome_node)
        })
        .collect();
    Some((on_path, treatment_node, outcome_node))
}

/// Classify the per-unit effects as structurally equal, empirically equal, or
/// free to vary.
///
/// A unit effect can only vary across units when some mechanism on a directed
/// `treatment → outcome` path bends with that unit's other parent values or its
/// abducted disturbance. When every such mechanism is additive and linear in its
/// parents, the two worlds differ by a fixed composition of slopes: the disturbances
/// cancel and `Y(a) − Y(a0)` is a constant.
///
/// That constant is disclosed as a property of the mechanism only when it *is*
/// one — when no family that could have represented effect modification was
/// successfully scored on any path node (none was in the registry for that node,
/// or every such family failed to fit, e.g. a single-parent node where no
/// cross-parent product exists). When such a family was scored and lost, the
/// data chose additivity, and the verdict records what was rejected instead.
fn homogeneity_verdict(
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    treatment: VariableId,
    outcome: VariableId,
) -> HomogeneityVerdict {
    let Some((on_path, _, outcome_node)) = path_assignments(graph, assignments, treatment, outcome)
    else {
        return HomogeneityVerdict::Heterogeneous;
    };
    let mut outcome_family = None;
    let mut rejected = Vec::new();
    for assignment in &on_path {
        if !assignment.fitted.admits_no_effect_modification() {
            return HomogeneityVerdict::Heterogeneous;
        }
        if assignment.node == outcome_node {
            outcome_family = Some(assignment.selected);
        }
        let selected_score = assignment
            .candidates
            .iter()
            .find(|c| c.family == assignment.selected)
            .map_or(f64::NAN, |c| c.score);
        for candidate in assignment.candidates.iter() {
            // `candidates` holds only families that scored; failures live in
            // `failed_families` and never gave the data a say.
            if candidate.family != assignment.selected
                && candidate.family.can_modify_effects()
                && candidate.score.is_finite()
            {
                rejected.push(RejectedHeterogeneity {
                    variable: assignment.variable,
                    family: candidate.family,
                    score: candidate.score,
                    selected: assignment.selected,
                    selected_score,
                });
            }
        }
    }
    match outcome_family {
        None => HomogeneityVerdict::Heterogeneous,
        Some(_) if !rejected.is_empty() => HomogeneityVerdict::Empirical { rejected },
        Some(family) => HomogeneityVerdict::Structural(family),
    }
}

/// Record that equal per-unit effects were chosen by validation score over a
/// family that could have made them differ.
///
/// Info, not Warning: nothing about how the number reads is hidden — the
/// selection is auditable in `gcm.counterfactual.mechanisms`, and this names the
/// rejected alternatives and the score margin so a reader does not have to parse
/// that dump to learn that heterogeneity was tested for.
fn heterogeneity_rejected_diagnostic(
    data: &TabularData,
    rejected: &[RejectedHeterogeneity],
) -> Diagnostic {
    let name = |variable: VariableId| {
        data.schema()
            .get(variable)
            .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
    };
    let parts: Vec<String> = rejected
        .iter()
        .map(|r| {
            format!(
                "{}: {} (score {:.6}) lost to {} (score {:.6})",
                name(r.variable),
                r.family.id(),
                r.score,
                r.selected.id(),
                r.selected_score
            )
        })
        .collect();
    let families: Vec<&str> = rejected.iter().map(|r| r.family.id()).collect();
    Diagnostic::new(
        "gcm.counterfactual.heterogeneity_rejected",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "heterogeneity-capable mechanism families were fit and lost on validation score, so \
             equal unit_effects are an empirical finding rather than a property of the mechanism: \
             {}",
            parts.join("; ")
        ),
    )
    .with_fields([("families", families.join(","))])
}

/// Per-unit support of the counterfactual prediction.
struct UnitSupport {
    /// Unit's covariate cell leaves the opposite arm's observed range.
    parent_cell: Vec<bool>,
    /// Unit's abducted disturbance leaves the opposite arm's residual range.
    disturbance: Vec<bool>,
    /// Covariates checked (names).
    covariates: Vec<String>,
    /// Additive-disturbance path nodes checked (names).
    disturbance_nodes: Vec<String>,
}

impl UnitSupport {
    fn extrapolation_flags(&self) -> Vec<bool> {
        self.parent_cell.iter().zip(&self.disturbance).map(|(a, b)| *a || *b).collect()
    }

    fn support_diagnostic(&self, min: f64, max: f64, pooled: bool) -> Diagnostic {
        let n = self.parent_cell.len();
        let cell = self.parent_cell.iter().filter(|f| **f).count();
        let disturbance = self.disturbance.iter().filter(|f| **f).count();
        let any = self.extrapolation_flags().iter().filter(|f| **f).count();
        let severity =
            if pooled || any > 0 { DiagnosticSeverity::Warning } else { DiagnosticSeverity::Info };
        Diagnostic::new(
            "gcm.counterfactual.support",
            DiagnosticKind::Support,
            severity,
            format!(
                "observed treatment range=[{min},{max}]; extrapolative={pooled}; \
                 per_unit_extrapolative={any}/{n} (covariate cell outside the opposite arm's \
                 observed range: {cell}; abducted disturbance outside the opposite arm's residual \
                 range: {disturbance}); covariates=[{}]; disturbance_nodes=[{}]",
                self.covariates.join(","),
                self.disturbance_nodes.join(",")
            ),
        )
        .with_fields([
            ("extrapolative", pooled.to_string()),
            ("per_unit_extrapolative", any.to_string()),
            ("parent_cell_extrapolative", cell.to_string()),
            ("disturbance_extrapolative", disturbance.to_string()),
            ("n_units", n.to_string()),
        ])
    }
}

/// Flag, per unit, a counterfactual prediction that leaves observed support.
///
/// Every unit is predicted into both arms; the arm it was observed in is
/// factual, the other is not. A unit is assigned to the arm whose level its
/// observed treatment is nearer (exact for a binary treatment; the midpoint
/// split for a continuous one). For the *opposite* arm two things must have
/// been seen:
///
/// * **Covariate cell.** Every parent of a re-evaluated path mechanism that the
///   intervention does not move (not the treatment, not a descendant of it) is
///   held at the unit's own value in both worlds. If that value lies outside the
///   range observed among units that actually received the opposite arm, an
///   interaction or spline term is being evaluated at a `(treatment, covariate)`
///   cell the fit never saw.
/// * **Disturbance.** For each additive-disturbance path mechanism, the unit's
///   abducted residual is inside the pooled residual support by construction,
///   but not necessarily inside the residuals of the opposite arm. A unit whose
///   disturbance is unlike any seen under the arm it is predicted into is
///   carried by the additivity assumption alone.
///
/// Categorical path mechanisms have uniform abducted noise with no residual
/// scale, so only their covariate cell is checked.
#[allow(clippy::too_many_arguments)]
fn unit_support(
    data: &TabularData,
    graph: &Dag,
    assignments: &[crate::gcm::MechanismAssignment],
    exogenous: &crate::gcm::ExogenousPosterior,
    treatment: VariableId,
    outcome: VariableId,
    active: f64,
    control: f64,
) -> Result<UnitSupport, CausalError> {
    let n = data.row_count();
    let observed = data.float64_values(treatment)?;
    let in_active: Vec<bool> =
        observed.iter().map(|t| (t - active).abs() < (t - control).abs()).collect();
    let name = |variable: VariableId| {
        data.schema()
            .get(variable)
            .map_or_else(|_| format!("v{}", variable.raw()), |v| v.name.to_string())
    };
    let mut parent_cell = vec![false; n];
    let mut disturbance = vec![false; n];
    let mut covariates = Vec::new();
    let mut disturbance_nodes = Vec::new();
    let Some((on_path, treatment_node, _)) =
        path_assignments(graph, assignments, treatment, outcome)
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
            if parent == treatment_node || graph.reaches(treatment_node, parent) {
                continue;
            }
            if !seen.insert(parent.as_usize()) {
                continue;
            }
            let Some(variable) = variable_of(parent) else { continue };
            let values = data.float64_values(variable)?;
            flag(&values, &mut parent_cell);
            covariates.push(name(variable));
        }
        if assignment.selected.has_additive_disturbance()
            && exogenous.n_units == n
            && assignment.node.as_usize() < exogenous.n_nodes
        {
            let start = assignment.node.as_usize() * n;
            flag(&exogenous.noise[start..start + n], &mut disturbance);
            disturbance_nodes.push(name(assignment.variable));
        }
    }
    Ok(UnitSupport { parent_cell, disturbance, covariates, disturbance_nodes })
}

/// Posterior summary of per-unit ITE draws, through the same summary that
/// publishes the mean-ITE interval, so a unit interval and the mean interval
/// share one quantile rule.
fn unit_effect_posterior_summary(
    unit_draws: Vec<f64>,
    n_units: usize,
    n_draws: usize,
) -> Result<antecedent_prob::PosteriorSummary, CausalError> {
    let quantities: Vec<antecedent_prob::PosteriorQuantityKind> = (0..n_units)
        .map(|unit| antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from(format!("ite[{unit}]")),
        })
        .collect();
    let schema = antecedent_prob::PosteriorSchema { quantities: std::sync::Arc::from(quantities) };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_draws,
        std::sync::Arc::<[f64]>::from(unit_draws),
    )
    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    Ok(draws.summarize())
}

fn counterfactual_posterior(
    values: Vec<f64>,
    mut assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("counterfactual.weighted_mechanisms"),
            description: Arc::from("Dirichlet row-weight posterior of standard mechanism fits, conditional on selected mechanism families and empirical support; abduction is repeated on the original units for every draw. This is not a parametric coefficient-prior posterior. The interval is for mean_ite over the observed units and reflects mechanism-refit uncertainty only; it is not a unit-level predictive interval."),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("gcm.fit.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let schema = antecedent_prob::PosteriorSchema {
        quantities: std::sync::Arc::from([antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from("ite"),
        }]),
    };
    let n = values.len();
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n,
        std::sync::Arc::<[f64]>::from(values),
    )
    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic("gcm.fit.bayesian"),
        assumptions,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
    })
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)] // SCM columns are named as in the graph (z, a, y).
mod tests {
    use super::*;
    use crate::gcm::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};
    use antecedent_core::CausalRng;

    fn var_id(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// `y = a·exp(z) + z + ε` with `a ~ Bern(σ(0.5 z))`: the unit effect is
    /// `exp(z)`, monotone in `z`. Columns `z, a, y`; graph `z → a`, `{z, a} → y`.
    fn exp_modifier_table(n: usize, seed: u64) -> (TabularData, Dag) {
        let mut rng = CausalRng::from_seed(seed);
        let (mut z, mut a, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            z[i] = antecedent_kernels::standard_normal(&mut rng);
            a[i] = f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-0.5 * z[i]).exp())));
            y[i] = a[i] * z[i].exp() + z[i] + antecedent_kernels::standard_normal(&mut rng);
        }
        let data = TabularData::from_f64_columns([
            ("z", z.as_slice()),
            ("a", a.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut g = Dag::with_variables(3);
        for (from, to) in [(0, 1), (0, 2), (1, 2)] {
            g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        (data, g)
    }

    fn assignments(
        registry: &MechanismRegistry,
        data: &TabularData,
        graph: &Dag,
    ) -> Vec<crate::gcm::MechanismAssignment> {
        let compiled = CompiledCausalModel::compile(graph.clone()).unwrap();
        registry.assign_and_fit(&compiled, data, SelectionPolicy::BestScore).unwrap().1
    }

    /// With only additive-linear families available the constant contrast is a
    /// property of the candidate set, and the verdict says so.
    #[test]
    fn linear_only_registry_makes_homogeneity_structural() {
        let (data, graph) = exp_modifier_table(1500, 5);
        let fitted = assignments(&MechanismRegistry::standard(), &data, &graph);
        match homogeneity_verdict(&graph, &fitted, var_id(1), var_id(2)) {
            HomogeneityVerdict::Structural(family) => {
                assert_eq!(family, crate::gcm::MechanismFamily::LinearGaussian);
            }
            other => panic!("linear-only registry must be structural, got {other:?}"),
        }
    }

    /// The same data under the counterfactual registry selects a family that
    /// modifies the effect, so there is nothing to disclose.
    #[test]
    fn heterogeneity_registry_lets_the_exp_modifier_through() {
        let (data, graph) = exp_modifier_table(1500, 5);
        let fitted = assignments(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        assert!(
            matches!(
                homogeneity_verdict(&graph, &fitted, var_id(1), var_id(2)),
                HomogeneityVerdict::Heterogeneous
            ),
            "{:?}",
            fitted.iter().map(|a| (a.variable, a.selected)).collect::<Vec<_>>()
        );
    }

    /// A single-parent outcome cannot fit any cross-parent product: the richer
    /// families fail, so the equal contrast is structural even under the
    /// counterfactual registry.
    #[test]
    fn failed_heterogeneity_families_leave_homogeneity_structural() {
        let n = 800;
        let mut rng = CausalRng::from_seed(9);
        let a: Vec<f64> = (0..n).map(|_| f64::from(u8::from(rng.next_f64() < 0.5))).collect();
        let y: Vec<f64> =
            a.iter().map(|ai| 0.8 * ai + antecedent_kernels::standard_normal(&mut rng)).collect();
        let data =
            TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let fitted = assignments(&MechanismRegistry::with_heterogeneity_families(), &data, &graph);
        assert!(matches!(
            homogeneity_verdict(&graph, &fitted, var_id(0), var_id(1)),
            HomogeneityVerdict::Structural(_)
        ));
    }
}
