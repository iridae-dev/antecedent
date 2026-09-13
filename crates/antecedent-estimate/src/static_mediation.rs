//! Static additive-linear natural mediation on an identified DAG.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
use crate::{
    EffectEstimate, EstimationError, HydrateMapping, OverlapPolicy, TemporalMediationEstimate,
    hydrate_prior,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ExecutionContext, MediationContrast, MediationQuery, ParametricAssumption,
    TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_prob::{PosteriorQuantityKind, PriorSet};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use std::sync::Arc;

/// Fit parent regressions and propagate the active–control contrast along all
/// paths (total) or paths avoiding the mediators (natural direct). Their
/// difference is the natural indirect effect under additive linear mechanisms.
/// `extra` are exogenous nuisance covariates used by the native RCC refuter.
///
/// # Errors
/// Invalid query/data, unsupported population, singular regression, cancellation.
#[allow(clippy::too_many_lines)]
pub fn estimate_static_mediation(
    data: &TabularData,
    graph: &Dag,
    query: &MediationQuery,
    mut assumptions: AssumptionSet,
    replicates: u32,
    extra: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<TemporalMediationEstimate, EstimationError> {
    query.validate()?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::unsupported("static mediation requires AllObserved"));
    }
    let delta = crate::adjustment::intervention_f64(&query.active)?
        - crate::adjustment::intervention_f64(&query.control)?;
    let order = graph
        .topological_order()
        .ok_or_else(|| EstimationError::unsupported("mediation requires a DAG"))?;
    let columns: Vec<_> = (0..graph.node_count())
        .map(|i| data.float64_values(VariableId::from_raw(i as u32)))
        .collect::<Result<_, _>>()?;
    let extras: Vec<_> =
        extra.iter().map(|&id| data.float64_values(id)).collect::<Result<_, _>>()?;
    let validities: Vec<_> = (0..graph.node_count())
        .map(|i| {
            data.column(VariableId::from_raw(i as u32)).map(antecedent_data::ColumnView::validity)
        })
        .collect::<Result<_, _>>()?;
    let rows: Vec<_> = (0..data.row_count())
        .filter(|&r| {
            data.storage().analysis_mask().is_none_or(|mask| mask.is_valid(r))
                && validities.iter().all(|mask| mask.is_valid(r))
                && columns.iter().chain(&extras).all(|c| c[r].is_finite())
        })
        .collect();
    let mut ls_ws = LeastSquaresWorkspace::default();
    let mut fit = |rows: &[usize]| -> Result<(f64, f64), EstimationError> {
        let mut total = vec![0.0; graph.node_count()];
        let mut direct = total.clone();
        for &node in &order {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("static mediation cancelled"));
            }
            let i = node.as_usize();
            if i == query.treatment.as_usize() {
                total[i] = delta;
                direct[i] = delta;
                continue;
            }
            let parents = graph.parents(DenseNodeId::from_raw(i as u32));
            if parents.is_empty() {
                continue;
            }
            let p = 1 + parents.len() + extras.len();
            if rows.len() <= p {
                return Err(EstimationError::unsupported("insufficient complete mediation rows"));
            }
            let mut matrix = vec![1.0; rows.len()];
            for parent in parents {
                matrix.extend(rows.iter().map(|&r| columns[parent.as_usize()][r]));
            }
            for column in &extras {
                matrix.extend(rows.iter().map(|&r| column[r]));
            }
            let y: Vec<_> = rows.iter().map(|&r| columns[i][r]).collect();
            let fitted = FaerBackend.least_squares(&matrix, rows.len(), p, &y, &mut ls_ws)?;
            if fitted.rank < p {
                return Err(EstimationError::unsupported("singular static mediation regression"));
            }
            let coefficients = fitted.coefficients;
            total[i] = parents
                .iter()
                .enumerate()
                .map(|(j, p)| coefficients[j + 1] * total[p.as_usize()])
                .sum();
            if !query.mediators.contains(&VariableId::from_raw(i as u32)) {
                direct[i] = parents
                    .iter()
                    .enumerate()
                    .map(|(j, p)| coefficients[j + 1] * direct[p.as_usize()])
                    .sum();
            }
        }
        Ok((total[query.outcome.as_usize()], direct[query.outcome.as_usize()]))
    };
    let contrast = |(total, direct): (f64, f64)| match query.contrast {
        MediationContrast::Total => total,
        MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => total - direct,
    };
    let (total, direct) = fit(&rows)?;
    let mut draws = Vec::new();
    for rep in 0..replicates {
        let mut rng = ctx.rng.stream(0x1300_1000 + u64::from(rep));
        let sample: Vec<_> = (0..rows.len())
            .map(|_| rows[(rng.next_f64() * rows.len() as f64) as usize % rows.len()])
            .collect();
        draws.push(contrast(fit(&sample)?));
    }
    let se = if draws.len() > 1 {
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    assumptions.push(AssumptionRecord {
        assumption:Assumption::ParametricRestriction(ParametricAssumption {
            id:Arc::from("mediation.additive_linear"),
            description:Arc::from("Natural effects use linear additive DAG mechanisms with independent disturbances, no treatment-mediator interactions, and the identified path restriction. Direct/mediated aliases denote natural direct/indirect effects in this model."),
        }), source:AssumptionSource::AlgorithmDefault{algorithm:Arc::from("estimate.mediation.linear")},
        scope:AssumptionScope::Estimation,status:AssumptionStatus::Declared,
    });
    let mut effect = EffectEstimate::new(
        contrast((total, direct)),
        f64::NAN,
        assumptions,
        OverlapPolicy::ExplicitOverride,
    );
    effect.se_bootstrap = se.is_finite().then_some(se);
    effect.bootstrap_replicates_ok = (replicates > 0).then_some(replicates);
    effect.bootstrap_replicates_failed = (replicates > 0).then_some(0);
    Ok(TemporalMediationEstimate {
        effect,
        total: Some(total),
        direct: Some(direct),
        mediated: Some(total - direct),
    })
}

/// Mapped prior summaries hydrated independently onto each linear mechanism.
#[derive(Clone, Copy, Debug)]
pub struct MediationPriorBridge<'a> {
    /// Declared hydrate mapping.
    pub mapping: &'a HydrateMapping,
    /// Source posterior quantity kinds (artifact column schema).
    pub quantities: &'a [PosteriorQuantityKind],
    /// Source posterior means, aligned to [`Self::quantities`].
    pub mean: &'a [f64],
    /// Source posterior SDs, aligned to [`Self::quantities`].
    pub sd: &'a [f64],
}

/// Bayesian linear natural effects: independent Gaussian mechanism posteriors,
/// then the same additive-linear natural-effect product as [`estimate_static_mediation`].
///
/// A shared coefficient [`PriorSet`] cannot be assigned to both mechanisms.
/// A [`MediationPriorBridge`] hydrates the mapped artifact onto each node that
/// can bind; unbound nodes keep that node's isotropic
/// [`crate::BayesianGComputationAte::prior_scale`].
///
/// # Errors
///
/// Same as [`estimate_static_mediation`], HMC composition, a shared prior, a
/// mapping that binds no mechanism, or a non-finite posterior.
pub fn estimate_static_mediation_bayesian(
    data: &TabularData,
    graph: &Dag,
    query: &MediationQuery,
    assumptions: AssumptionSet,
    extra: &[VariableId],
    estimator: &crate::BayesianGComputationAte,
    identification: antecedent_core::IdentificationStatus,
    bridge: Option<MediationPriorBridge<'_>>,
    ctx: &ExecutionContext,
) -> Result<(TemporalMediationEstimate, crate::CausalPosterior), EstimationError> {
    crate::bayesian_mediation::require_gaussian_mediation(estimator)?;
    if estimator.prior.is_some() {
        return Err(EstimationError::unsupported(
            "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
        ));
    }
    let point = estimate_static_mediation(data, graph, query, assumptions.clone(), 0, extra, ctx)?;
    query.validate()?;
    let delta = crate::adjustment::intervention_f64(&query.active)?
        - crate::adjustment::intervention_f64(&query.control)?;
    let order = graph
        .topological_order()
        .ok_or_else(|| EstimationError::unsupported("mediation requires a DAG"))?;
    let columns: Vec<_> = (0..graph.node_count())
        .map(|i| data.float64_values(VariableId::from_raw(i as u32)))
        .collect::<Result<_, _>>()?;
    let extras: Vec<_> =
        extra.iter().map(|&id| data.float64_values(id)).collect::<Result<_, _>>()?;
    let validities: Vec<_> = (0..graph.node_count())
        .map(|i| {
            data.column(VariableId::from_raw(i as u32)).map(antecedent_data::ColumnView::validity)
        })
        .collect::<Result<_, _>>()?;
    let rows: Vec<_> = (0..data.row_count())
        .filter(|&r| {
            data.storage().analysis_mask().is_none_or(|mask| mask.is_valid(r))
                && validities.iter().all(|mask| mask.is_valid(r))
                && columns.iter().chain(&extras).all(|c| c[r].is_finite())
        })
        .collect();
    let n = rows.len();
    let mut mechanisms = vec![None; graph.node_count()];
    let mut ws = crate::BayesianGCompWorkspace::default();
    let mechanism_ctx = ExecutionContext {
        parallelism: ctx.parallelism,
        determinism: ctx.determinism,
        rng: ctx.rng.clone(),
        memory: ctx.memory,
        cancellation: ctx.cancellation.clone(),
        progress: ctx.progress.clone(),
        kernel_policy: ctx.kernel_policy,
        cache_policy: ctx.cache_policy,
        adaptive_bootstrap: ctx.adaptive_bootstrap,
        adaptive_draws: antecedent_core::AdaptiveDrawBudget::disabled(),
    };
    let mut hydrated = Vec::new();
    for &node in &order {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("static mediation cancelled"));
        }
        let i = node.as_usize();
        if i == query.treatment.as_usize() {
            continue;
        }
        let parents = graph.parents(DenseNodeId::from_raw(i as u32));
        if parents.is_empty() {
            continue;
        }
        if n <= 1 + parents.len() + extras.len() {
            return Err(EstimationError::unsupported("insufficient complete mediation rows"));
        }
        let first = parents[0].as_usize();
        let treatment: Vec<f64> = rows.iter().map(|&r| columns[first][r]).collect();
        let mut covs: Vec<(VariableId, Vec<f64>)> = parents
            .iter()
            .skip(1)
            .map(|parent| {
                let id = VariableId::from_raw(parent.raw());
                let col: Vec<f64> = rows.iter().map(|&r| columns[parent.as_usize()][r]).collect();
                (id, col)
            })
            .collect();
        for (k, column) in extras.iter().enumerate() {
            covs.push((extra[k], rows.iter().map(|&r| column[r]).collect()));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, col)| (*id, col.as_slice())).collect();
        let y: Vec<f64> = rows.iter().map(|&r| columns[i][r]).collect();
        let design =
            antecedent_stats::CompiledDesign::linear_adjustment(&treatment, &cov_refs, &y, &[])?;
        let schema = data.schema();
        let name_of = |id: VariableId| -> Arc<str> {
            schema
                .get(id)
                .ok()
                .map_or_else(|| Arc::from(format!("var_{}", id.raw())), |v| Arc::clone(&v.name))
        };
        let mut coef_names = vec![Arc::from("intercept")];
        for parent in parents {
            coef_names
                .push(Arc::from(format!("coef_{}", name_of(VariableId::from_raw(parent.raw())))));
        }
        for &id in extra {
            coef_names.push(Arc::from(format!("coef_{}", name_of(id))));
        }
        let treatment_col =
            parents.iter().position(|p| p.raw() == query.treatment.raw()).map(|j| j + 1);
        let prep = crate::PreparedBayesianProblem {
            design,
            method: Arc::from("mediation.linear"),
            adjustment_set: Arc::from([]),
            active: 1.0,
            control: 0.0,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: Some(Arc::from(coef_names.clone())),
            unit_ids: None,
        };
        let mut node_est = estimator.clone();
        node_est.seed = estimator.seed.wrapping_add(i as u64 + 1);
        node_est.n_draws = estimator.n_draws.max(2);
        // Independent mechanisms must all supply every composed draw.

        if let Some(bridge) = bridge {
            if let Some(prior) =
                hydrate_mechanism_prior(bridge, &coef_names, treatment_col, estimator.prior_scale)?
            {
                node_est.prior = Some(prior);
                hydrated.push(name_of(VariableId::from_raw(i as u32)));
            }
        }
        let posterior = node_est.fit(
            &prep,
            antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions,
            &mut ws,
            &mechanism_ctx,
        )?;
        let mut parent_draws = Vec::with_capacity(parents.len());
        for j in 0..parents.len() {
            parent_draws.push(coefficient_draws(&posterior, j + 1)?);
        }
        mechanisms[i] = Some((parents.to_vec(), parent_draws));
    }
    if bridge.is_some() && hydrated.is_empty() {
        return Err(EstimationError::unsupported(
            "mapped prior did not bind any mediation mechanism",
        ));
    }
    let draws_n = estimator.n_draws.max(2);
    let mut effect = Vec::with_capacity(draws_n);
    let mut totals = Vec::with_capacity(draws_n);
    let mut directs = Vec::with_capacity(draws_n);
    let mut mediated = Vec::with_capacity(draws_n);
    for draw in 0..draws_n {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("static mediation cancelled"));
        }
        let (total, direct) =
            compose_linear_natural(graph.node_count(), query, delta, &order, &mechanisms, draw)?;
        totals.push(total);
        directs.push(direct);
        mediated.push(total - direct);
        effect.push(match query.contrast {
            MediationContrast::Total => total,
            MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => total - direct,
        });
    }
    if effect.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg("mediation mechanism product was non-finite"));
    }
    let n_keep = effect.len();
    let values: Vec<f64> =
        effect.iter().chain(&totals).chain(&directs).chain(&mediated).copied().collect();
    let schema = antecedent_prob::PosteriorSchema {
        quantities: Arc::from([
            antecedent_prob::PosteriorQuantityKind::Effect { name: Arc::from("mediation") },
            antecedent_prob::PosteriorQuantityKind::Scalar { name: Arc::from("total") },
            antecedent_prob::PosteriorQuantityKind::Scalar { name: Arc::from("direct") },
            antecedent_prob::PosteriorQuantityKind::Scalar { name: Arc::from("mediated") },
        ]),
    };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_keep,
        Arc::<[f64]>::from(values),
    )
    .map_err(|e| EstimationError::stats_msg(e.to_string()))?;
    let summaries = draws.summarize();
    let mut posterior_assumptions = point.effect.assumptions.clone();
    posterior_assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("mediation.gaussian_product"),
            description: Arc::from(if hydrated.is_empty() {
                "Independent Gaussian linear mechanism posteriors composed into natural effects; no treatment-mediator interaction; empirical complete rows fixed".to_string()
            } else {
                format!(
                    "Independent Gaussian linear mechanism posteriors composed into natural effects; mapped prior hydrated onto [{}]; unbound mechanisms keep isotropic prior_scale",
                    hydrated.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>().join(", ")
                )
            }),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.mediation.linear.bayesian"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    let posterior = crate::CausalPosterior {
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic("mediation.linear.bayesian"),
        assumptions: posterior_assumptions,
        unidentified_mass: match identification {
            antecedent_core::IdentificationStatus::NotIdentified => 1.0,
            _ => 0.0,
        },
        early_stopped: false,
    };
    let eq = posterior
        .effect_column()
        .ok_or_else(|| EstimationError::stats_msg("mediation posterior missing effect column"))?;
    let mut effect = point.effect.clone();
    effect.ate = posterior.summaries.mean[eq];
    effect.se_analytic = posterior.summaries.sd[eq];
    effect.assumptions = posterior.assumptions.clone();
    Ok((
        TemporalMediationEstimate {
            effect,
            total: Some(posterior.summaries.mean[1]),
            direct: Some(posterior.summaries.mean[2]),
            mediated: Some(posterior.summaries.mean[3]),
        },
        posterior,
    ))
}

fn hydrate_mechanism_prior(
    bridge: MediationPriorBridge<'_>,
    coef_names: &[Arc<str>],
    treatment_col: Option<usize>,
    prior_scale: f64,
) -> Result<Option<PriorSet>, EstimationError> {
    let mut baseline = PriorSet::new();
    baseline.push(antecedent_prob::PriorSpec::GaussianCoefficients(
        antecedent_prob::GaussianCoefficientPrior::isotropic(coef_names.len(), prior_scale),
    ));
    match bridge.mapping {
        HydrateMapping::EffectFunctional { .. } if treatment_col.is_none() => return Ok(None),
        HydrateMapping::NamedParameters { pairs } => {
            let names: std::collections::HashSet<&str> =
                coef_names.iter().map(std::convert::AsRef::as_ref).collect();
            if !pairs.iter().any(|(_, target)| names.contains(target.as_str())) {
                return Ok(None);
            }
        }
        HydrateMapping::IdenticalCoefficientSubspace | HydrateMapping::EffectFunctional { .. } => {}
    }
    match hydrate_prior(
        bridge.mapping,
        bridge.quantities,
        bridge.mean,
        bridge.sd,
        &baseline,
        coef_names,
        treatment_col,
    ) {
        Ok(prior) => Ok(Some(prior)),
        Err(err)
            if matches!(bridge.mapping, HydrateMapping::IdenticalCoefficientSubspace)
                && (err.to_string().contains("n_coef")
                    || err.to_string().contains("dimension")
                    || err.to_string().contains("expected n_coef")) =>
        {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

fn coefficient_draws(
    posterior: &crate::CausalPosterior,
    index: usize,
) -> Result<Vec<f64>, EstimationError> {
    let col = posterior
        .draws
        .schema
        .quantities
        .iter()
        .position(|q| {
            matches!(
                q,
                antecedent_prob::PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index
            )
        })
        .ok_or_else(|| EstimationError::stats_msg("mediation posterior missing coefficient"))?;
    Ok(posterior.draws.column(col)?.to_vec())
}

fn compose_linear_natural(
    nodes: usize,
    query: &MediationQuery,
    delta: f64,
    order: &[DenseNodeId],
    mechanisms: &[Option<(Vec<DenseNodeId>, Vec<Vec<f64>>)>],
    draw: usize,
) -> Result<(f64, f64), EstimationError> {
    let mut total = vec![0.0; nodes];
    let mut direct = total.clone();
    total[query.treatment.as_usize()] = delta;
    direct[query.treatment.as_usize()] = delta;
    for &node in order {
        let i = node.as_usize();
        if i == query.treatment.as_usize() {
            continue;
        }
        let Some((parents, coefs)) = mechanisms[i].as_ref() else {
            continue;
        };
        let parent_total = parents.iter().enumerate().try_fold(0.0, |acc, (j, parent)| {
            let beta = *coefs
                .get(j)
                .and_then(|c| c.get(draw))
                .ok_or_else(|| EstimationError::stats_msg("mediation mechanism draw is short"))?;
            Ok::<f64, EstimationError>(acc + beta * total[parent.as_usize()])
        })?;
        total[i] = parent_total;
        if !query.mediators.contains(&VariableId::from_raw(i as u32)) {
            direct[i] = parents.iter().enumerate().try_fold(0.0, |acc, (j, parent)| {
                let beta = *coefs.get(j).and_then(|c| c.get(draw)).ok_or_else(|| {
                    EstimationError::stats_msg("mediation mechanism draw is short")
                })?;
                Ok::<f64, EstimationError>(acc + beta * direct[parent.as_usize()])
            })?;
        }
    }
    Ok((total[query.outcome.as_usize()], direct[query.outcome.as_usize()]))
}
