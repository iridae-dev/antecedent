// SPDX-License-Identifier: MIT OR Apache-2.0


/// Build a GCM / parametric-SCM estimand and identification result for `treatment`/`outcome`.
///
/// The estimate itself is computed elsewhere (by the fitted parametric SCM, not by evaluating
/// a backdoor-adjustment formula); this function only produces the *inspectable* estimand
/// metadata. Like `stub_accepted_graph_for` (in `analysis::builder`), the adjustment set stays
/// deliberately empty — GCM does not identify via a backdoor covariate set — but unlike the
/// placeholder this replaced, `functional` is a real expression naming the actual
/// `treatment`/`outcome` pair rather than a nil `ExprId` into an empty arena, so the estimand
/// is honest about which variables it refers to. The node is a minimal
/// `Expectation`/`Distribution` leaf (not `CausalExprArena::backdoor_ate`'s `Product`/`SumOut`
/// shape) so it is inert if a future caller ever tries to mechanically re-evaluate
/// `functional` via the arena's generic evaluator — there is no adjustment-set
/// marginalization here to (mis)compute. Licensed transport and interference
/// routes do not call this; they stamp `transport.sid` / `interference.design`.
pub(crate) fn parametric_scm_identification(
    query: CausalQuery,
    treatment: VariableId,
    outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let y = arena.intern_var_set([outcome]);
    let do_t = arena.intern_intervention_set([treatment]);
    let empty = arena.empty_var_set();
    let distribution = arena.intern_distribution(y, empty, do_t, DomainRef::Interventional);
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            "gcm.parametric",
            Some(Arc::from(format!(
                "parametric SCM: treatment={treatment:?} outcome={outcome:?}; no adjustment \
                 set (GCM does not identify via backdoor covariates)"
            ))),
        ),
    );
    let estimand = IdentifiedEstimand::backdoor("gcm.parametric", Arc::from([]), functional);
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("gcm.supplied_structural_mechanisms"),
                description: Arc::from(
                    "Identification is conditional on the supplied acyclic structural mechanisms and their declared intervention semantics being an adequate model of the data-generating process.",
                ),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("gcm.parametric"),
        },
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let identification =
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
            query,
            estimands: vec![estimand.clone()],
            arena,
            derivation: DerivationTrace::default(),
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
    (identification, estimand)
}

pub(super) fn binary_cf_interventions(
    query: &antecedent_core::CounterfactualQuery,
) -> Result<(VariableId, f64, f64), CausalError> {
    if query.allow_nested || query.outcomes.len() != 1 || query.interventions.len() != 1 {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path currently supports a single hard \
                 intervention for ITE (use gcm helpers for multi-world predict)",
        });
    }
    let Intervention::Set { variable, value } = &query.interventions[0] else {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path requires a hard Set intervention",
        });
    };
    let active = value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual intervention value must be f64".into(),
    })?;
    let Intervention::Set { variable: control_var, value: control_value } = &query.control else {
        return Err(CausalError::Unsupported {
            message: "Study counterfactual path requires a hard Set control intervention",
        });
    };
    if control_var != variable {
        return Err(CausalError::Compile {
            message: format!(
                "counterfactual control targets {control_var:?}, expected {variable:?}"
            ),
        });
    }
    let control = control_value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual control value must be f64".into(),
    })?;
    if !control.is_finite() {
        return Err(CausalError::Unsupported { message: "counterfactual control must be finite" });
    }
    Ok((*variable, active, control))
}

/// The identified set of a scalar functional, in the coordinate-free shape the
/// portable format requires: no grid, one bound. The estimand already says what
/// the bound means, so no coordinate is invented for it.
///
/// Every route that publishes a scalar identified set builds it here: the
/// portable format accepts `dimension = 0` precisely so a scalar bound is not
/// dressed up with a `0.0` coordinate that nothing reads.
pub(super) fn scalar_identified_set(lower: f64, upper: f64) -> antecedent_core::ResponseEnvelope {
    antecedent_core::ResponseEnvelope {
        grid: Arc::from([]),
        dimension: 0,
        lower: Arc::from([lower]),
        upper: Arc::from([upper]),
    }
}

/// Completion-mass summary of a graph-class envelope: the one body behind every
/// `identify.*.envelope` diagnostic.
///
/// `prefix` names the class in the message (`generalized.adjustment envelope`,
/// `cpdag.mec envelope`), and each `extra` pair is appended as `, key=value`.
/// The structured half is [`mass_fields`], which the reasoning slot reads.
pub(super) fn class_envelope_diagnostic<G>(
    code: impl Into<Arc<str>>,
    prefix: &str,
    envelope: &IdentificationEnvelope<G>,
    extra: &[(&str, String)],
) -> Diagnostic {
    let mut message = format!(
        "{prefix}: identified_mass={}, unidentified_mass={}, cases={}",
        envelope.identified_weight.0,
        envelope.unidentified_weight.0,
        envelope.cases.len()
    );
    for (key, value) in extra {
        use std::fmt::Write as _;
        let _ = write!(message, ", {key}={value}");
    }
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
        .with_fields(mass_fields(
            Some(envelope.identified_weight.0),
            envelope.unidentified_weight.0,
        ))
}

/// Per-completion outcome of a graph-class (CPDAG/PAG, static or temporal)
/// envelope arm, in `envelope.cases` order.
#[derive(Clone, Copy, Debug, Default)]
pub(super) enum ClassAtomOutcome {
    /// Identified and evaluated: the completion's own point value.
    Evaluated(f64),
    /// Identified but left unevaluated by the Interactive latency tier's graph
    /// budget. Neither unidentified nor a failed estimate.
    SubsampledOut,
    /// Not evaluated for any other reason (unidentified completions land here
    /// too; their status decides which mass they join).
    #[default]
    NotEvaluated,
}

/// What a class mixture says about itself, beside the mixture: facts a caller
/// needs to decide whether publishing it is right.
pub(super) struct ClassMixtureFacts {
    /// Total weight the masses were divided by. Not positive means the
    /// envelope carried no mass at all and the masses are all zero.
    pub total_weight: f64,
    /// All mass identified, a degenerate identified set, and an identified
    /// envelope status: the mixture restates a point, and publishing it would
    /// dress a point up as bounds.
    pub point_identified: bool,
}

/// Structural uncertainty of a graph-class envelope: every completion with its
/// weight and status, the evaluated completions' point values, and the
/// identified set `[min, max]` over them.
///
/// The one class-mixture body. `outcomes` is indexed by `envelope.cases`;
/// `key_fn` supplies each atom's `graph_key` (the case index for a static
/// class, the completion fingerprint for a temporal one); `weights` overrides
/// the enumeration weights when the caller carries its own class mass.
///
/// Mass is a fraction of the total weight and is kept apart by kind:
/// identified-and-evaluated, unidentified, identified-but-unevaluable, and
/// identified-but-subsampled-out. Which cases are identified is
/// [`identification_status_carries_identified_mass`] — the envelope's own
/// split — so `identified_mass + unidentified_mass` here agrees with
/// `envelope.identified_weight` / `unidentified_weight` in the diagnostic
/// published beside it. `unevaluable_mass` is taken as the remainder, so the
/// four masses conserve exactly under division.
pub(super) fn class_structural_mixture<G>(
    envelope: &IdentificationEnvelope<G>,
    weight_basis: crate::result::StructuralWeightBasis,
    key_fn: impl Fn(usize, &antecedent_identify::GraphIdentificationCase<G>) -> u64,
    weights: Option<&[f64]>,
    outcomes: &[ClassAtomOutcome],
) -> (crate::result::StructuralResponseMixture, ClassMixtureFacts) {
    let mut atoms = Vec::with_capacity(envelope.cases.len());
    let (mut identified_weight, mut unidentified_weight) = (0.0, 0.0);
    let (mut unevaluable_weight, mut subsampled_weight) = (0.0, 0.0);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (index, case) in envelope.cases.iter().enumerate() {
        let weight = weights.and_then(|w| w.get(index).copied()).unwrap_or(case.weight.0);
        let outcome = outcomes.get(index).copied().unwrap_or_default();
        let value = match outcome {
            ClassAtomOutcome::Evaluated(value) if value.is_finite() => Some(value),
            _ => None,
        };
        if !identification_status_carries_identified_mass(case.result.status) {
            unidentified_weight += weight;
        } else if let Some(value) = value {
            identified_weight += weight;
            lo = lo.min(value);
            hi = hi.max(value);
        } else if matches!(outcome, ClassAtomOutcome::SubsampledOut) {
            subsampled_weight += weight;
        } else {
            unevaluable_weight += weight;
        }
        atoms.push(crate::result::StructuralResponseAtom {
            graph_key: key_fn(index, case),
            weight,
            status: case.result.status,
            value: value.map(antecedent_core::ResponseValue::Scalar),
            posterior: None,
            response: None,
        });
    }
    let total = identified_weight + unidentified_weight + unevaluable_weight + subsampled_weight;
    let positive = total > 0.0;
    let (identified_mass, unidentified_mass, subsampled_out_mass) = if positive {
        (identified_weight / total, unidentified_weight / total, subsampled_weight / total)
    } else {
        (0.0, 0.0, 0.0)
    };
    // The remainder, so the four masses conserve exactly under division.
    let unevaluable_mass = if positive {
        (1.0 - identified_mass - unidentified_mass - subsampled_out_mass).max(0.0)
    } else {
        0.0
    };
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    // Exact comparisons on purpose: "every completion is identified" and "they
    // all agree" are exact facts about the mass that was summed and the values
    // that were compared, not measurements with a tolerance.
    #[allow(
        clippy::float_cmp,
        reason = "\"every completion is identified\" and \"they all agree\" are exact facts about summed masses and compared values, not measurements with a tolerance"
    )]
    let facts = ClassMixtureFacts {
        total_weight: total,
        point_identified: identified_mass == 1.0
            && lo == hi
            && matches!(
                envelope.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
                    | IdentificationStatus::IdentifiedUnderPriorRestrictions
            ),
    };
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis,
        atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified: None,
        full_mass_scope: envelope.truncated_completions == 0,
        truncated_atoms: envelope.truncated_completions,
    };
    (mixture, facts)
}

/// Statuses the envelope itself counts as identified mass.
///
/// A re-export, not a second list: `antecedent-identify` owns the split —
/// [`antecedent_identify::carries_identified_mass`] is the same predicate
/// `IdentificationEnvelope::from_cases` uses to divide `identified_weight`
/// from `unidentified_weight`, so a mass this crate publishes cannot
/// contradict the envelope diagnostic beside it.
///
/// Wider than [`identification_status_ok_for_case`], which licenses
/// *estimating* a case: a completion identified only under prior restrictions
/// carries identified mass and its assumptions, but no frequentist arm
/// estimates it.
pub(crate) use antecedent_identify::carries_identified_mass as identification_status_carries_identified_mass;

/// Statuses a frequentist arm may estimate. Narrower than
/// [`identification_status_carries_identified_mass`]: a completion identified
/// only under prior restrictions is *not* estimable, but its mass is still
/// identified mass.
pub(crate) fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

pub(super) fn envelope_to_identification_result<G>(
    envelope: &IdentificationEnvelope<G>,
    query: &AverageEffectQuery,
) -> IdentificationResult {
    envelope_to_identification_result_for(envelope, CausalQuery::AverageEffect(query.clone()))
}

pub(super) fn envelope_to_identification_result_for<G>(
    envelope: &IdentificationEnvelope<G>,
    query: CausalQuery,
) -> IdentificationResult {
    // Every case's estimands carry a `functional: ExprId` into *that case's own*
    // `case.result.arena`, not a shared one. Each estimand copied into the combined
    // result must have its functional re-homed into the combined `arena` via
    // `CausalExprArena::import` (which re-interns the referenced subtree and returns
    // its id here) — otherwise the copied `ExprId` indexes nothing in the empty arena
    // this used to hand back, and every consumer that renders, evaluates, or encodes
    // the functional (including the wire encoder's arena-bounds check) sees a
    // dangling id.
    let mut arena = CausalExprArena::new();
    let mut estimands = Vec::new();
    let mut assumptions = antecedent_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    let mut first_identified_case_arena = None;
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            if first_identified_case_arena.is_none() {
                first_identified_case_arena = Some(&case.result.arena);
            }
            for est in &case.result.estimands {
                let mut est = est.clone();
                est.functional = arena.import(&case.result.arena, est.functional);
                estimands.push(est);
            }
        }
        // Every case the envelope counts as identified mass carries its
        // assumptions, including one identified only under prior restrictions:
        // its mass is reported, so the conditions behind it must be too.
        if identification_status_carries_identified_mass(case.result.status) {
            assumptions.extend_unique(&case.result.required_assumptions.entries);
        }
        // Refused and unverified cases explain the missing mass too.
        diagnostics.extend(case.result.diagnostics.iter().cloned());
    }
    for feature in &envelope.critical_graph_features {
        diagnostics.push(Diagnostic::new(
            Arc::from(format!("identify.envelope.{}", feature.kind)),
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            Arc::clone(&feature.detail),
        ));
    }
    if let Some(inv) = &envelope.invariant {
        if estimands.is_empty() {
            // `IdentificationEnvelope::from_cases` sets `invariant` to a clone of
            // the first identified case's first estimand, so its functional
            // belongs to that same case's arena.
            let mut inv = inv.clone();
            if let Some(source_arena) = first_identified_case_arena {
                inv.functional = arena.import(source_arena, inv.functional);
            }
            estimands.push(inv);
        }
    }
    IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
        status: envelope.status,
        query,
        estimands,
        arena,
        derivation: DerivationTrace::default(),
        required_assumptions: assumptions,
        diagnostics,
        performance: IdentificationPerformanceRecord::default(),
        hedge: None,
    })
}

pub(crate) fn admg_has_bidirected(admg: &Admg) -> bool {
    admg.has_bidirected()
}

/// Static ADMG whose general-ID functional matches this temporal ADMG.
///
/// The projection exists only when each variable appears once. Identification
/// and estimation then reuse the static ADMG path on that graph.
pub(crate) fn temporal_admg_projection(
    graph: &antecedent_graph::TemporalAdmg,
) -> Result<Admg, CausalError> {
    graph
        .project_preserving_functional()
        .map_err(CausalError::rejected)
}

/// The directed part of an ADMG as a DAG.
///
/// Only an ADMG without bidirected edges is a DAG: dropping a bidirected edge would present a
/// latent-confounded graph as causally sufficient, so it is refused here rather than left to
/// every caller's pre-check.
pub(super) fn admg_without_latents_to_dag(admg: &Admg) -> Result<Dag, CausalError> {
    if admg.has_bidirected() {
        return Err(CausalError::Unsupported {
            message: "an ADMG with bidirected edges (latent confounding) is not a DAG; \
                      dropping them would assume causal sufficiency",
        });
    }
    let n = u32::try_from(admg.node_count())
        .map_err(|_| CausalError::Compile { message: "ADMG too large".into() })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to)
                .map_err(CausalError::rejected)?;
        }
    }
    Ok(dag)
}

#[cfg(test)]
mod admg_dag_coercion_tests {
    use super::*;

    #[test]
    fn bidirected_edge_is_refused_and_directed_only_graph_converts() {
        let mut admg = Admg::with_variables(3);
        let (a, b, c) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        admg.insert_directed(a, b).unwrap();
        admg.insert_directed(b, c).unwrap();
        let dag = admg_without_latents_to_dag(&admg).unwrap();
        assert!(dag.reaches(a, c));
        admg.insert_bidirected(a, c).unwrap();
        assert!(matches!(admg_without_latents_to_dag(&admg), Err(CausalError::Unsupported { .. })));
    }
}

/// Copy inference notes from `sources` onto `target` (deduplicated), so a mixture or
/// composed posterior still records its atoms' dependence correction / draw floor.
pub(super) fn merge_posterior_notes<'a>(
    target: &mut CausalPosterior,
    sources: impl IntoIterator<Item = &'a CausalPosterior>,
) {
    for source in sources {
        for note in &source.diagnostics.notes {
            if !target.diagnostics.notes.contains(note) {
                target.diagnostics.notes.push(Arc::clone(note));
            }
        }
    }
}

/// Result diagnostics derived from posterior inference notes.
///
/// Notes are de-duplicated by text (an envelope mixture re-carries its atoms' notes), so the same
/// text raised by several fits is one note here: the messages state that a condition occurred and
/// never a count of fits, which the de-duplication cannot recover.
///
/// - `estimate.bayesian.temporal.dependence_correction`: the likelihood was tempered
///   for serial dependence; lists every fitted `κ̂`.
/// - `estimate.bayesian.hmc_draw_floor`: the HMC draw floor raised the requested
///   draw count (B-5).
/// - `estimate.bayesian.draw_controls`: the Monte Carlo standard errors of the interval
///   endpoints that [`crate::PosteriorDrawControls`] bounded.
pub(super) fn posterior_note_diagnostics<'a>(
    posteriors: impl IntoIterator<Item = &'a CausalPosterior>,
) -> Vec<Diagnostic> {
    let mut kappas = Vec::new();
    let mut floors = Vec::new();
    let mut endpoint_errors = Vec::new();
    let mut capped = false;
    let mut inestimable = false;
    let mut seen = std::collections::HashSet::new();
    for post in posteriors {
        for note in &post.diagnostics.notes {
            if !seen.insert(Arc::clone(note)) {
                continue;
            }
            let single = std::slice::from_ref(note);
            if let Some(kappa) = antecedent_estimate::tempering_kappa_from_notes(single) {
                kappas.push(kappa);
            }
            if antecedent_estimate::tempering_capped_from_notes(single) {
                capped = true;
            }
            if antecedent_estimate::tempering_inestimable_from_notes(single) {
                inestimable = true;
            }
            if let Some(floor) = antecedent_estimate::hmc_draw_floor_from_notes(single) {
                floors.push(floor);
            }
            if let Some(error) = antecedent_estimate::endpoint_mcse_from_notes(single) {
                endpoint_errors.push(error);
            }
        }
    }
    let mut out = Vec::new();
    if !kappas.is_empty() {
        let list = kappas.iter().map(|k| format!("{k:.3}")).collect::<Vec<_>>().join(", ");
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.dependence_correction",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "generalized (power) posterior with a serial-dependence correction: each \
                 Gaussian likelihood on time-ordered rows is tempered by 1/kappa, kappa = the \
                 larger of the autoregressive-prewhitened Newey-West long-run-variance ratio of \
                 the targeted slope score (scaled by its squared fixed-b factor) and the \
                 autoregressive-residual variance ratio given the design (AR(1), plus a \
                 BIC-selected AR(q) up to order 4), floored at 1 (kappa = [{list}], one entry \
                 per distinct fitted tempering note); this corrects short-memory serial \
                 dependence in the outcome residual, not long memory, heteroskedasticity or a \
                 misspecified mean",
            ),
        ));
    }
    if inestimable {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_inestimable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "long-run-variance tempering could not be estimated on at least one fit \
                 (n < max(8, p+2)); the published credible interval for it is the iid posterior \
                 and is likely too narrow"
                .to_string(),
        ));
    }
    if capped {
        out.push(Diagnostic::new(
            "estimate.bayesian.temporal.tempering_capped",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "long-run-variance tempering hit the n/(p+2) cap on at least one fit; kappa is \
                 known to be too small and the published credible interval is still too narrow"
                .to_string(),
        ));
    }
    if let Some(error) = endpoint_errors.first() {
        out.push(Diagnostic::new(
            "estimate.bayesian.draw_controls",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            format!(
                "interval endpoint Monte Carlo standard errors {:.3e} (lower) and {:.3e} (upper) \
                 are {:.4} of the interval width from {:.0} effective of {} draws, within the \
                 {} tolerance",
                error.lower_mcse,
                error.upper_mcse,
                error.width_fraction,
                error.effective_draws,
                error.draws,
                antecedent_estimate::count_fitted_posterior::MAX_ENDPOINT_MCSE_FRACTION
            ),
        ));
    }
    if let Some(&(requested, used)) = floors.first() {
        out.push(Diagnostic::new(
            "estimate.bayesian.hmc_draw_floor",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            format!(
                "HMC draw floor raised n_draws from {requested} to {used} so the MCMC \
                 publication gate (R-hat <= 1.01, bulk/tail ESS >= 100 per chain) is reachable"
            ),
        ));
    }
    out
}

pub(super) fn bayesian_gcomp(
    cfg: &BayesianConfig,
    ctx: &ExecutionContext,
) -> BayesianGComputationAte {
    BayesianGComputationAte {
        backend: cfg.backend,
        likelihood: cfg.likelihood,
        n_draws: cfg.n_draws,
        seed: ctx.rng.master_seed(),
        overlap: OverlapPolicy::ExplicitOverride,
        prior_scale: cfg.prior_scale,
        prior: None,
    }
}

pub(super) fn apply_temporal_prior_sensitivity(
    cfg: &BayesianConfig,
    bprep: &antecedent_estimate::PreparedBayesianProblem,
    status: IdentificationStatus,
    posterior: &antecedent_estimate::CausalPosterior,
    ate: f64,
    ctx: &ExecutionContext,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
) -> Result<antecedent_estimate::CausalPosterior, CausalError> {
    let mut est = bayesian_temporal_gcomp(cfg, ctx);
    let mut ws = BayesianGCompWorkspace::default();
    // Same resolution (and conflict shrink) as the fit, so the grid perturbs the
    // prior in force rather than a fresh isotropic one.
    est.inner.prior = resolve_bayesian_prior_with_conflict(cfg, bprep, Some(ctx))?.0;
    let (summary, sens) = evaluate_bayesian_prior_sensitivity(
        cfg, &est.inner, bprep, status, posterior, &mut ws, ctx,
    )?;
    refutations.push(sens.to_report(&summary, ate));
    Ok(with_prior_sensitivity(posterior.clone(), summary))
}

pub(super) fn bayesian_temporal_gcomp(
    cfg: &BayesianConfig,
    ctx: &ExecutionContext,
) -> BayesianTemporalGcomp {
    BayesianTemporalGcomp {
        inner: BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        },
    }
}

fn push_unique_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    diagnostic: Diagnostic,
) {
    if seen.insert(Arc::clone(&diagnostic.code)) {
        diagnostics.push(diagnostic);
    }
}

fn push_aipw_score_kind(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    estimator_id: EstimatorId,
    estimate: &EffectEstimate,
) {
    if seen.contains("estimate.aipw.crossfit_scores")
        || seen.contains("estimate.aipw.full_sample_residualized")
    {
        return;
    }
    match estimator_id {
        EstimatorId::CellAipw
        | EstimatorId::Aipw
        | EstimatorId::Dml
        | EstimatorId::DrLearner
        | EstimatorId::CausalForest
            if estimate.scores.table.is_some() =>
        {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.aipw.crossfit_scores",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "cross-fitted AIPW scores φᵢ^a; retarget averages this table. A residualized full-sample AIPW fit is a different object",
                ),
            );
        }
        EstimatorId::Aipw => {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.aipw.full_sample_residualized",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "this AIPW fit is full-sample residualized and has no score table; it is not the cross-fitted φ family that retarget averages. Prepare an AllObserved iid AIPW plan to retarget",
                ),
            );
        }
        _ => {}
    }
}

pub(crate) fn push_grid_scalar_cleared(
    diagnostics: &mut Vec<Diagnostic>,
    seen: &mut std::collections::HashSet<Arc<str>>,
    estimate: &EffectEstimate,
) {
    if let Some(inf) = estimate.scores.inference.as_ref() {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.cdf_inference",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "per-arm F_a(c) simultaneous bands describe raw CDF coordinates; rearranged exceedance_cdf values are not mixed with those intervals",
            ),
        );
        if inf.threshold_supported.iter().any(|ok| !ok) {
            push_unique_diagnostic(
                diagnostics,
                seen,
                Diagnostic::new(
                    "estimate.functional.threshold_tail.unsupported",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "at least one threshold tail lacks enough treated/control events for a tail probability; that coordinate's band is non-finite rather than an empty-cell or first-threshold SE",
                ),
            );
        }
    } else if estimate.scores.table.is_none() && estimate.joint.exceedance_cdf.is_some() {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.cdf_inference.unavailable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "conditional CDF values have no per-arm simultaneous bands or threshold tail-support evidence; joint covariance, when present, describes raw threshold contrasts, not the projected per-arm CDF",
            ),
        );
    }
    if estimate.joint.exceedance_cdf.as_ref().is_some_and(|cdf| cdf.len() > 2)
        && !estimate.ate.is_finite()
    {
        push_unique_diagnostic(
            diagnostics,
            seen,
            Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ),
        );
    }
}

impl super::Study {
    pub(super) fn require_execute_dag(&self, message: &'static str) -> Result<&Dag, CausalError> {
        self.graph.as_dag().ok_or(CausalError::Unsupported { message })
    }

    pub(super) fn attach_certificate(
        &self,
        mut result: StudyResult,
        identification: crate::Identification,
    ) -> StudyResult {
        result.certificate = Some(crate::AnalysisIdentification {
            identification,
            query: self.query.clone(),
            graph_class: self.graph.class(),
        });
        result
    }

    pub(super) fn finish_identified_execute(
        &self,
        args: IdentifiedExecuteFinish<'_>,
    ) -> StudyResult {
        let context = IdentifiedResultContext::from_study(self);
        let tabular_data = match &self.data {
            DataInput::Tabular(data) => Some(data),
            _ => None,
        };
        finish_identified_execute_with_context(&context, tabular_data, args)
    }
}

#[allow(clippy::too_many_lines, reason = "assembles one StudyResult from the finish record: diagnostics are pushed in a fixed order against one seen-code set, then provenance, estimate and certificate are read from the same record")]
fn finish_identified_execute_with_context(
    context: &IdentifiedResultContext,
    tabular_data: Option<&TabularData>,
    args: IdentifiedExecuteFinish<'_>,
) -> StudyResult {
    let extras = args.extras;
    let certificate = extras.certificate.or_else(|| {
        context.graph_has_certificate.then(|| crate::Identification::Point {
            result: args.identification.clone(),
            temporal_indexer: None,
            strategy: args.identifier_id,
            structure_version: context.graph_version,
        })
    });
    let mut diagnostics = if let Some(prebuilt) = extras.diagnostics {
        prebuilt
    } else {
        let mut diagnostics = args.identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(args.estimate.provenance.overlap));
        diagnostics.extend(args.extra_diagnostics);
        diagnostics
    };
    let mut seen: std::collections::HashSet<Arc<str>> =
        diagnostics.iter().map(|d| Arc::clone(&d.code)).collect();
    // Envelope routes supply their own diagnostic seed, but a prepared
    // envelope still must expose cache reuse just like a single-graph path.
    if args.identify_cached {
        push_unique_diagnostic(&mut diagnostics, &mut seen, identify_cached_diagnostic());
    }
    push_aipw_score_kind(&mut diagnostics, &mut seen, args.estimator_id, &args.estimate);
    push_grid_scalar_cleared(&mut diagnostics, &mut seen, &args.estimate);
    if matches!(
        args.estimator_id,
        EstimatorId::FunctionalEffect | EstimatorId::FunctionalDistribution
    ) {
        if let Some(diagnostic) = free_variables_averaged(&args.identification, &args.estimand) {
            push_unique_diagnostic(&mut diagnostics, &mut seen, diagnostic);
        }
    }
    let structural_posteriors = extras
        .structural_response
        .iter()
        .flat_map(|mixture| mixture.atoms.iter().filter_map(|atom| atom.posterior.as_ref()));
    for diagnostic in
        posterior_note_diagnostics(extras.posterior.iter().chain(structural_posteriors))
    {
        push_unique_diagnostic(&mut diagnostics, &mut seen, diagnostic);
    }
    if let (Some(mode), Some(n)) = (context.latency_mode, extras.n_draws) {
        let tier = match mode {
            crate::analysis::latency::LatencyMode::Interactive => {
                crate::analysis::latency::INTERACTIVE_N_DRAWS
            }
            crate::analysis::latency::LatencyMode::Standard => {
                crate::analysis::latency::STANDARD_N_DRAWS
            }
            crate::analysis::latency::LatencyMode::Report => {
                crate::analysis::latency::REPORT_N_DRAWS
            }
        };
        if usize::try_from(n).ok() != Some(tier) {
            diagnostics.push(Diagnostic::new(
                "latency.explicit_budget_kept",
                DiagnosticKind::Execution,
                DiagnosticSeverity::Info,
                format!("explicit n_draws={n} kept over {} tier default {tier}", mode.as_str()),
            ));
        }
    }
    let (id_artifact, id_op) = extras.identify_provenance.unwrap_or_else(|| {
        let (a, b) = identify_provenance_step(args.identifier_id);
        provenance_ids(a, b)
    });
    let (est_artifact, est_op) = extras.estimate_provenance.unwrap_or_else(|| {
        let (a, b) = estimate_provenance_step(args.estimator_id);
        provenance_ids(a, b)
    });
    let provenance = if extras.empty_provenance {
        ProvenanceGraph::new()
    } else {
        provenance_pair(
            (id_artifact.as_ref(), id_op.as_ref(), &[], &args.identification.required_assumptions),
            (
                est_artifact.as_ref(),
                est_op.as_ref(),
                &[id_artifact.as_ref()],
                &args.estimate.provenance.assumptions,
            ),
        )
    };
    let mut physical_record = args.physical.record.clone();
    if context.custom_validators {
        let (record, diagnostic) = mark_python_callback_plan(physical_record, "validator");
        physical_record = record;
        diagnostics.push(diagnostic);
    }
    let payload = crate::result::ResultPayload::one_of([
        args.distribution.map(crate::result::ResultPayload::Distribution),
        (args.mediation.is_some() || extras.mediation_grid.is_some()).then(|| {
            crate::result::ResultPayload::from_mediation(args.mediation, extras.mediation_grid)
        }),
        extras.gcm.map(GcmSlot::into_payload),
        extras.response.map(crate::result::ResultPayload::Response),
    ]);
    let mut result = assemble_result(AssembleArgs {
        logical: &args.physical.logical.record,
        physical: &physical_record,
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        posterior: extras.posterior,
        payload,
        refutations: args.refutations,
        diagnostics,
        provenance,
        treatment: args.treatment,
        outcome: args.outcome,
        wall_time_ns: args.wall_time_ns,
        latency_mode: context.latency_mode.map(|m| Arc::from(m.as_str())),
        stage_timings_ns: extras.stage_timings_ns,
        bootstrap_replicates_requested: extras
            .bootstrap_replicates_requested
            .unwrap_or(Some(context.bootstrap_replicates)),
        bootstrap_replicates_ok: args.bootstrap_replicates_ok,
        n_draws: extras.n_draws,
        cancelled: args.cancelled,
        early_stopped: args.early_stopped,
        bayesian: context.inference.is_bayesian(),
    });
    result.certificate = certificate.map(|identification| crate::AnalysisIdentification {
        identification,
        query: context.query.clone(),
        graph_class: context.graph_class,
    });
    let is_quantile = match &context.query {
        CausalQuery::AverageEffect(q) => q.outcome_functional.quantile_level().is_some(),
        CausalQuery::ConditionalEffect(q) => q.inner.outcome_functional.quantile_level().is_some(),
        CausalQuery::Response(q) => q.outcome_functional.quantile_level().is_some(),
        _ => false,
    };
    if is_quantile {
        if let Some(effect) = result.estimate.as_effect_mut() {
            effect.diagnostics.evalue = None;
        }
        result.diagnostics.push(super::helpers::quantile_scope_diagnostic());
        if matches!(context.query, CausalQuery::ConditionalEffect(_)) {
            result.diagnostics.push(Diagnostic::new("estimate.functional.conditional_quantile",
                    DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    "quantile contrast inverts arm CDFs standardized over the retained modifier distribution; this is not a pointwise conditional-quantile surface or an average of individual quantile effects"));
        }
    }
    result.predictive_checks = extras.predictive_checks;
    result.structural_response = extras.structural_response;
    result.rebind_interval(context.inference.is_bayesian());
    result.support_status = context.support_status;
    result.structure_source = context.structure_source;
    // A named predicate or custom distribution is a handle; encoding the
    // executed query (certificates, artifacts) needs its bindings.
    result.population_registry.clone_from(&context.population_registry);
    if !is_quantile {
        super::helpers::mirror_refuted_evalue(&mut result.estimate, &result.refutations);
    }
    if let Some(data) = tabular_data {
        super::helpers::attach_tiered_evalue_for_cell(
            &mut result.estimate,
            data,
            args.outcome,
            is_quantile,
            context.tiered.as_ref(),
            &mut result.diagnostics,
        );
    }
    if let Some(crate::support::CellStatus::Allowlisted { reason, parent }) = context.support_status
    {
        result.diagnostics.push(Diagnostic {
            code: Arc::from("support.allowed_unlicensed"),
            kind: DiagnosticKind::Support,
            severity: DiagnosticSeverity::Warning,
            message: Arc::from(
                "this estimate executed an allowlisted cell; it is not a licensed claim",
            ),
            artifact_id: None,
            fields: Arc::from([
                (Arc::from("reason"), Arc::from(reason)),
                (Arc::from("parent"), Arc::from(parent)),
            ]),
        });
    }
    if let Some((tier, configured)) = context.latency_bootstrap_not_applied {
        result.diagnostics.push(
            crate::analysis::helpers::latency_bootstrap_not_applied_diagnostic(tier, configured),
        );
    }
    if let Some(requested) = context.refute_default_downgrade {
        let requested_id = requested.validation_suite_id().unwrap_or("none");
        result.diagnostics.push(Diagnostic {
            code: Arc::from("exec.refute.default_suite_unsupported"),
            kind: DiagnosticKind::Scientific,
            severity: DiagnosticSeverity::Info,
            message: Arc::from(format!(
                "no .refute(..) was set; the default validation suite \
                     ({requested_id}) is not supported for this cell, so validation was \
                     silently downgraded to none (no refuters ran)"
            )),
            artifact_id: None,
            fields: Arc::from([
                (Arc::from("requested_suite"), Arc::from(requested_id)),
                (Arc::from("applied_suite"), Arc::from("none")),
            ]),
        });
    }
    result
}

/// Match the estimator failure policy without counting unattempted, cancelled
/// replicates as fitting failures, and require enough successes to earn a
/// nominal 0.95 band under the usual (B+1) order-statistic floor.
///
/// Cancellation that stops after two successes must not publish a 0.95 pointwise
/// or simultaneous band: `bootstrap_has_enough_successes(2, attempted)` is false
/// whenever `attempted > 2`, and also when `completed` is below
/// [`PERCENTILE_95_BAND_MIN_SUCCESSES`]. Adaptive early-stop that actually reaches
/// that floor with a majority of successes still passes.
pub(super) fn bootstrap_has_enough_successes(completed: usize, attempted: usize) -> bool {
    completed >= PERCENTILE_95_BAND_MIN_SUCCESSES
        && completed >= attempted.saturating_sub(completed)
}

/// Fewest successful replicates that may license a facade-published nominal 0.95
/// pointwise / simultaneous band. Alias of [`crate::result::PERCENTILE_95_MIN_REPLICATES`].
pub(super) const PERCENTILE_95_BAND_MIN_SUCCESSES: usize =
    crate::result::PERCENTILE_95_MIN_REPLICATES as usize;

#[cfg(test)]
mod bootstrap_success_floor_tests {
    use super::{PERCENTILE_95_BAND_MIN_SUCCESSES, bootstrap_has_enough_successes};

    #[test]
    fn cancelled_two_success_bootstrap_cannot_license_nominal_band() {
        assert!(!bootstrap_has_enough_successes(2, 2));
        assert!(!bootstrap_has_enough_successes(2, 3));
        assert!(!bootstrap_has_enough_successes(2, 199));
        assert!(!bootstrap_has_enough_successes(2, PERCENTILE_95_BAND_MIN_SUCCESSES));
    }

    #[test]
    fn completed_budget_at_earned_minimum_licenses_band() {
        let min = PERCENTILE_95_BAND_MIN_SUCCESSES;
        assert!(bootstrap_has_enough_successes(min, min));
        assert!(!bootstrap_has_enough_successes(min - 1, min - 1));
        // Majority-failure policy still binds above the floor.
        assert!(!bootstrap_has_enough_successes(min, min * 2 + 1));
        assert!(bootstrap_has_enough_successes(min, min * 2));
    }
}

#[cfg(test)]
mod refutation_mixture_coverage_tests {
    use super::mix_atom_refutation_reports;
    use antecedent_validate::RefutationReport;

    fn report(id: &str, passed: bool) -> RefutationReport {
        RefutationReport::new(id, 1.0, 1.0, 0.5, true, passed, None, 10)
    }

    #[test]
    fn refuter_run_on_part_of_the_mass_reports_its_coverage() {
        // Atom A (0.6) ran the placebo; atom B (0.4) had it not applicable. The mixed placebo
        // "passed" covers 0.6 of the mass and the result must say so.
        let per_atom = vec![(0.6, vec![report("placebo", true)]), (0.4, Vec::new())];
        let (reports, diagnostics) = mix_atom_refutation_reports(&per_atom);
        assert_eq!(reports.len(), 1);
        assert!(reports[0].passed);
        let partial: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.code.as_ref() == "refute.envelope.partial_coverage")
            .collect();
        assert_eq!(partial.len(), 1, "{diagnostics:?}");
        assert!(partial[0].message.contains("`placebo`"), "{}", partial[0].message);
        assert!(partial[0].message.contains("0.600000"), "{}", partial[0].message);
    }

    #[test]
    fn fully_covering_refuter_and_zero_weight_atoms_raise_no_coverage_warning() {
        let per_atom = vec![
            (0.5, vec![report("placebo", true)]),
            (0.5, vec![report("placebo", false)]),
            (0.0, Vec::new()),
        ];
        let (reports, diagnostics) = mix_atom_refutation_reports(&per_atom);
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].passed);
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
    }
}

// Aggregation consumes only effect draws. Keep each contributing model's
// prior/estimation restrictions on the resulting mixture as well.
pub(super) fn retain_envelope_assumptions(
    posterior: &mut CausalPosterior,
    atoms: &[EnvelopeAtomFit],
) {
    for atom in atoms {
        posterior.assumptions.entries.extend(
            atom.posterior.assumptions.entries.iter().cloned().map(|mut record| {
                match &mut record.assumption {
                    antecedent_core::Assumption::PriorRestriction(prior) => {
                        prior.description =
                            Arc::from(format!("graph atom {}: {}", atom.key, prior.description));
                    }
                    antecedent_core::Assumption::ParametricRestriction(model) => {
                        model.description =
                            Arc::from(format!("graph atom {}: {}", atom.key, model.description));
                    }
                    _ => {}
                }
                record
            }),
        );
    }
}

#[cfg(test)]
mod posterior_status_ladder_tests {
    use super::*;

    #[test]
    fn resolve_posterior_status_uses_one_tolerance() {
        let base = IdentificationStatus::IdentifiedUnderParametricRestrictions;
        let policy = crate::result::StructuralAggregationPolicy::SameEstimandWeightedMean;
        let dust = resolve_posterior_status(
            base,
            policy,
            PosteriorMassSummary { unidentified: 1e-13, ..PosteriorMassSummary::default() },
        );
        let real = resolve_posterior_status(
            base,
            policy,
            PosteriorMassSummary { unidentified: 1e-11, ..PosteriorMassSummary::default() },
        );
        assert_eq!(dust, base);
        assert_eq!(real, IdentificationStatus::GraphDependent);
        let kept = resolve_posterior_status(base, policy, PosteriorMassSummary::default());
        assert_eq!(kept, base);
        let partial = resolve_posterior_status(
            base,
            crate::result::StructuralAggregationPolicy::IdentifiedSetEnvelope,
            PosteriorMassSummary::default(),
        );
        assert_eq!(partial, IdentificationStatus::PartiallyIdentified);
    }

    #[test]
    fn temporal_refuter_keeps_the_query_contrast() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let mut query = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
        query.active = Intervention::set(treatment, antecedent_core::Value::f64(2.0));
        query.control = Intervention::set(treatment, antecedent_core::Value::f64(-1.0));
        let average = temporal_refuter_average(&query);
        let bare = AverageEffectQuery::binary_ate(treatment, outcome);
        assert_eq!(average.active, query.active);
        assert_eq!(average.control, query.control);
        assert_eq!(average.target_population, query.target_population);
        assert_ne!(average.active, bare.active);
        assert_ne!(SUSTAINED_LINEAR_SEM, UNFOLDED_LINEAR_SEM);
        assert_ne!(SUSTAINED_LINEAR_SEM, SEQUENCE_LINEAR_SEM);
        assert_ne!(UNFOLDED_LINEAR_SEM, DBN_LINEAR_SEM);
    }
}

#[cfg(test)]
mod graph_mixture_timing {
    use super::*;

    /// Timing harness for appending unidentified samples to a mixture
    /// (`cargo test -p antecedent --lib graph_mixture_timing -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing harness"]
    fn unidentified_samples_scale_linearly() {
        for samples in [2_000_usize, 8_000, 32_000] {
            let keys: Vec<u64> = (0..samples as u64).collect();
            let graphs = WeightedGraphSamples::new(
                vec![1.0 / samples as f64; samples],
                vec![GraphIdentFlag::Unidentified; samples],
                keys,
            )
            .unwrap();
            let started = std::time::Instant::now();
            let mixed = mix_graph_posterior_identified_atoms(&graphs, &[], 1.0, 0.0, 0.0, 0);
            eprintln!(
                "samples={samples:>6} atoms={} append={:.1} ms",
                mixed.mixture.atoms.len(),
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}
