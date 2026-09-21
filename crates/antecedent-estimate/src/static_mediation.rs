//! Static additive-linear natural mediation on an identified DAG.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
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
/// The path-specific identifier certifies the *pure* natural indirect effect
/// `E[Y(a₀, M(a₁))] − E[Y(a₀)]` (treatment at the active level only on edges
/// that start a mediated path). `total − direct` is by definition the *total*
/// natural indirect effect `E[Y(a₁)] − E[Y(a₁, M(a₀))]`. The two agree only
/// without treatment–mediator interaction, which the additive linear
/// mechanisms impose; indirect contrasts record that reliance as the
/// `mediation.no_interaction` assumption.
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
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("population_not_estimable"),
            "static mediation requires AllObserved",
        ));
    }
    let MediationRows { delta, order, columns, extras, rows } =
        MediationRows::new(data, graph, query, extra)?;
    let mut ls_ws = LeastSquaresWorkspace::default();
    let fit = |rows: &[usize],
               ls_ws: &mut LeastSquaresWorkspace|
     -> Result<(f64, f64), EstimationError> {
        propagate(graph, query, delta, &order, |i| {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::unsupported("static mediation cancelled"));
            }
            let parents = graph.parents(DenseNodeId::from_raw(i as u32));
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
            let fitted = FaerBackend.least_squares(&matrix, rows.len(), p, &y, ls_ws)?;
            if fitted.rank < p {
                return Err(EstimationError::unsupported("singular static mediation regression"));
            }
            Ok(fitted.coefficients[1..=parents.len()].to_vec())
        })
    };
    let contrast = |(total, direct): (f64, f64)| match query.contrast {
        MediationContrast::Total => total,
        MediationContrast::Direct | MediationContrast::NaturalDirect => direct,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => total - direct,
    };
    let (total, direct) = fit(&rows, &mut ls_ws)?;
    // Shared tolerant bootstrap: a singular or under-determined replicate is a
    // soft failure (counted, the loop continues), cancellation aborts, and more
    // than half failed withholds the SE. Replicate accounting is the real count.
    let boot = crate::util::bootstrap_se_with_scratch(
        replicates,
        ctx,
        0x1300_1000,
        rows.len(),
        || (Vec::with_capacity(rows.len()), LeastSquaresWorkspace::default()),
        |(sample, ws), idx| {
            sample.clear();
            sample.extend(idx.iter().map(|&i| rows[i]));
            match fit(sample, ws) {
                Ok(fitted) => Ok(Some(contrast(fitted))),
                Err(_) if ctx.cancellation.is_cancelled() => {
                    Err(EstimationError::unsupported("static mediation cancelled"))
                }
                Err(_) => Ok(None),
            }
        },
    )?;
    assumptions.push(AssumptionRecord {
        assumption:Assumption::ParametricRestriction(ParametricAssumption {
            id:Arc::from("mediation.additive_linear"),
            description:Arc::from("Natural effects use linear additive DAG mechanisms with independent disturbances, no treatment-mediator interactions, and the identified path restriction. Direct/mediated aliases denote natural direct/indirect effects in this model."),
        }), source:AssumptionSource::AlgorithmDefault{algorithm:Arc::from("estimate.mediation.linear")},
        scope:AssumptionScope::Estimation,status:AssumptionStatus::Declared,
    });
    if matches!(query.contrast, MediationContrast::Mediated | MediationContrast::NaturalIndirect) {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("mediation.no_interaction"),
                description: Arc::from(
                    "The identified functional is the pure natural indirect effect E[Y(a0, M(a1))] - E[Y(a0)]; the estimate is total - direct, the total natural indirect effect E[Y(a1)] - E[Y(a1, M(a0))]. They coincide only without treatment-mediator interaction, which the additive linear mechanisms impose; under interaction the estimate is not the certified quantity.",
                ),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("estimate.mediation.linear"),
            },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
    }
    let effect = EffectEstimate::new(
        contrast((total, direct)),
        f64::NAN,
        assumptions,
        OverlapPolicy::ExplicitOverride,
    )
    .with_bootstrap((replicates > 0).then_some(boot));
    Ok(TemporalMediationEstimate {
        effect,
        total: Some(total),
        direct: Some(direct),
        mediated: Some(total - direct),
    })
}

/// The restriction under which `total − direct` is the pure natural indirect effect.
///
/// Identification scope: the path-specific identifier certifies `E[Y(a0, M(a1))] − E[Y(a0)]`,
/// and this estimator computes a different functional of the observed law, `total − direct`
/// (the total natural indirect effect `E[Y(a1)] − E[Y(a1, M(a0))]`), that coincides with it
/// only under this restriction.
#[must_use]
pub fn linear_no_interaction_restriction() -> antecedent_core::AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("mediation.linear_no_interaction"),
            description: Arc::from(
                "The natural direct and total effects are computed as products of additive linear parent-regression coefficients along direct and full paths, and the indirect estimate is their difference total - direct, the total natural indirect effect E[Y(a1)] - E[Y(a1, M(a0))], not by evaluating the pure natural indirect effect functional E[Y(a0, M(a1))] - E[Y(a0)] the identifier certifies. They coincide only if the structural mechanisms are additive and linear with no treatment-mediator interaction.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.mediation.linear"),
        },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    }
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
    /// Source treatment contrast `active − control` recorded on the artifact.
    pub source_contrast: Option<f64>,
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
    let draws_n = crate::require_bayesian_n_draws(estimator.n_draws)?;
    if estimator.prior.is_some() {
        return Err(EstimationError::unsupported(
            "Bayesian mediation currently supports isotropic mechanism priors; a shared coefficient prior cannot be assigned to both mechanisms",
        ));
    }
    let point = estimate_static_mediation(data, graph, query, assumptions.clone(), 0, extra, ctx)?;
    query.validate()?;
    let MediationRows { delta, order, columns, extras, rows } =
        MediationRows::new(data, graph, query, extra)?;
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
    let mut mechanism_assumptions = AssumptionSet::new();
    let mut bound_targets = std::collections::HashSet::new();
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
            serial_dependence: crate::SerialDependence::Iid,
        };
        let mut node_est = estimator.clone();
        node_est.seed = estimator.seed.wrapping_add(i as u64 + 1);
        node_est.n_draws = draws_n;
        // Independent mechanisms must all supply every composed draw.

        if let Some(bridge) = bridge {
            let is_outcome = i == query.outcome.as_usize();
            if let Some(prior) = hydrate_mechanism_prior(
                bridge,
                &coef_names,
                treatment_col,
                estimator.prior_scale,
                is_outcome,
            )? {
                node_est.prior = Some(prior);
                hydrated.push(name_of(VariableId::from_raw(i as u32)));
                bound_targets.extend(coef_names.iter().map(std::string::ToString::to_string));
            }
        }
        let posterior = node_est.fit(
            &prep,
            antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions,
            &mut ws,
            &mechanism_ctx,
        )?;
        // Retain the actual coefficient and residual priors, scoped to the
        // mechanism whose likelihood consumed them (as in temporal composition).
        mechanism_assumptions.entries.extend(posterior.assumptions.entries.iter().cloned().map(
            |mut record| {
                record.scope = AssumptionScope::Variables {
                    variables: Arc::from([VariableId::from_raw(i as u32)]),
                };
                record
            },
        ));
        let mut parent_draws = Vec::with_capacity(parents.len());
        for j in 0..parents.len() {
            parent_draws.push(coefficient_draws(&posterior, j + 1)?);
        }
        mechanisms[i] = Some(parent_draws);
    }
    if let Some(MediationPriorBridge {
        mapping: HydrateMapping::NamedParameters { pairs }, ..
    }) = bridge
    {
        if let Some((_, target)) = pairs.iter().find(|(_, target)| !bound_targets.contains(target))
        {
            return Err(EstimationError::stats_msg(format!(
                "mapped prior target `{target}` did not bind any mediation mechanism"
            )));
        }
    }
    if bridge.is_some() && hydrated.is_empty() {
        return Err(EstimationError::unsupported(
            "mapped prior did not bind any mediation mechanism",
        ));
    }
    let mut effect = Vec::with_capacity(draws_n);
    let mut totals = Vec::with_capacity(draws_n);
    let mut directs = Vec::with_capacity(draws_n);
    let mut mediated = Vec::with_capacity(draws_n);
    for draw in 0..draws_n {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("static mediation cancelled"));
        }
        let (total, direct) = propagate(graph, query, delta, &order, |i| {
            mechanisms[i]
                .iter()
                .flatten()
                .map(|coefs| {
                    coefs.get(draw).copied().ok_or_else(|| {
                        EstimationError::stats_msg("mediation mechanism draw is short")
                    })
                })
                .collect()
        })?;
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
    posterior_assumptions.entries.extend(mechanism_assumptions.entries);
    posterior_assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("mediation.gaussian_product"),
            description: Arc::from(if hydrated.is_empty() {
                "Independent Gaussian linear mechanism posteriors composed into natural effects; no treatment-mediator interaction; empirical complete rows fixed".to_string()
            } else if !matches!(bridge, Some(MediationPriorBridge { mapping: HydrateMapping::EffectFunctional { .. }, .. })) {
                format!(
                    "Independent Gaussian linear mechanism posteriors composed into natural effects; declared coefficient mapping hydrated onto mechanisms [{}]; unbound coefficients and mechanisms keep isotropic prior_scale",
                    hydrated.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>().join(", "),
                )
            } else {
                let implied = match bridge {
                    Some(MediationPriorBridge {
                        mapping: HydrateMapping::EffectFunctional { source_quantity },
                        quantities,
                        mean,
                        ..
                    }) => quantities.iter().zip(mean.iter()).find_map(|(q, m)| {
                        match q {
                            PosteriorQuantityKind::Effect { name } if name.as_ref() == source_quantity.as_str() => {
                                Some(*m)
                            }
                            _ => None,
                        }
                    }),
                    _ => None,
                };
                format!(
                    "Independent Gaussian linear mechanism posteriors composed into natural effects; mapped ATE/Δ prior hydrated onto outcome-mechanism [{}]{}; unbound mechanisms keep isotropic prior_scale",
                    hydrated.iter().map(std::convert::AsRef::as_ref).collect::<Vec<_>>().join(", "),
                    implied.map(|m| format!("; implied NDE/ATE mean {m}")).unwrap_or_default(),
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
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
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
        treatment_contrast: Some(delta),
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
    outcome_mechanism: bool,
) -> Result<Option<PriorSet>, EstimationError> {
    let mut baseline = PriorSet::new();
    baseline.push(antecedent_prob::PriorSpec::GaussianCoefficients(
        antecedent_prob::GaussianCoefficientPrior::isotropic(coef_names.len(), prior_scale),
    ));
    match bridge.mapping {
        HydrateMapping::EffectFunctional { .. }
            if !outcome_mechanism || treatment_col.is_none() =>
        {
            return Ok(None);
        }
        HydrateMapping::NamedParameters { pairs } => {
            let names: std::collections::HashSet<&str> =
                coef_names.iter().map(std::convert::AsRef::as_ref).collect();
            let local_pairs: Vec<_> = pairs
                .iter()
                .filter(|(_, target)| names.contains(target.as_str()))
                .cloned()
                .collect();
            if local_pairs.is_empty() {
                return Ok(None);
            }
            return hydrate_prior(
                &HydrateMapping::NamedParameters { pairs: local_pairs },
                bridge.quantities,
                bridge.mean,
                bridge.sd,
                &baseline,
                coef_names,
                treatment_col,
                bridge.source_contrast,
            )
            .map(Some);
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
        bridge.source_contrast,
    ) {
        Ok(prior) => Ok(Some(prior)),
        // A banked posterior of another shape does not bind this mechanism; every
        // other failure is a real error.
        Err(EstimationError::PriorDimensionMismatch { .. })
            if matches!(bridge.mapping, HydrateMapping::IdenticalCoefficientSubspace) =>
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

/// Complete-case rows and column values both mediation estimators regress on.
struct MediationRows {
    /// Treatment contrast `active − control`.
    delta: f64,
    /// Topological order of the mediation DAG.
    order: Vec<DenseNodeId>,
    /// One value column per graph node.
    columns: Vec<Vec<f64>>,
    /// Extra adjustment columns entering every mechanism.
    extras: Vec<Vec<f64>>,
    /// Rows valid, unmasked, and finite in every node and extra column.
    rows: Vec<usize>,
}

impl MediationRows {
    fn new(
        data: &TabularData,
        graph: &Dag,
        query: &MediationQuery,
        extra: &[VariableId],
    ) -> Result<Self, EstimationError> {
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
                data.column(VariableId::from_raw(i as u32))
                    .map(antecedent_data::ColumnView::validity)
            })
            .collect::<Result<_, _>>()?;
        let rows: Vec<_> = (0..data.row_count())
            .filter(|&r| {
                data.storage().analysis_mask().is_none_or(|mask| mask.is_valid(r))
                    && validities.iter().all(|mask| mask.is_valid(r))
                    && columns.iter().chain(&extras).all(|c| c[r].is_finite())
            })
            .collect();
        Ok(Self { delta, order, columns, extras, rows })
    }
}

/// Total and mediator-blocked direct effects of the treatment contrast on the outcome.
///
/// Walks `order`; a node's total effect is `Σ_p β_p · total[p]` over its parents, and its
/// direct effect the same sum over `direct` unless the node is a mediator (whose
/// direct-path effect is blocked at zero). `betas(i)` returns node `i`'s parent
/// coefficients, in parent order, for a node with parents. Frequentist and Bayesian
/// mediation share this recursion so both describe the same estimand.
fn propagate(
    graph: &Dag,
    query: &MediationQuery,
    delta: f64,
    order: &[DenseNodeId],
    mut betas: impl FnMut(usize) -> Result<Vec<f64>, EstimationError>,
) -> Result<(f64, f64), EstimationError> {
    let mut total = vec![0.0; graph.node_count()];
    let mut direct = total.clone();
    total[query.treatment.as_usize()] = delta;
    direct[query.treatment.as_usize()] = delta;
    for &node in order {
        let i = node.as_usize();
        if i == query.treatment.as_usize() {
            continue;
        }
        let parents = graph.parents(node);
        if parents.is_empty() {
            continue;
        }
        let beta = betas(i)?;
        total[i] = parents.iter().zip(&beta).map(|(p, b)| b * total[p.as_usize()]).sum();
        if !query.mediators.contains(&VariableId::from_raw(i as u32)) {
            direct[i] = parents.iter().zip(&beta).map(|(p, b)| b * direct[p.as_usize()]).sum();
        }
    }
    Ok((total[query.outcome.as_usize()], direct[query.outcome.as_usize()]))
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use super::*;
    use antecedent_core::{Intervention, Value};

    fn coefficient_summary(n: usize) -> (Vec<PosteriorQuantityKind>, Vec<f64>, Vec<f64>) {
        let mut kinds: Vec<_> =
            (0..n).map(|index| PosteriorQuantityKind::Coefficient { index, name: None }).collect();
        kinds.push(PosteriorQuantityKind::ResidualVariance);
        (kinds, vec![0.5; n + 1], vec![0.1; n + 1])
    }

    #[test]
    fn prior_of_another_shape_leaves_the_mechanism_unbound_but_other_errors_surface() {
        let names: Vec<Arc<str>> = vec![Arc::from("intercept"), Arc::from("coef_t")];
        let mapping = HydrateMapping::IdenticalCoefficientSubspace;
        // Three banked coefficients cannot describe a two-coefficient mechanism.
        let (quantities, mean, sd) = coefficient_summary(3);
        let bridge = MediationPriorBridge {
            mapping: &mapping,
            quantities: &quantities,
            mean: &mean,
            sd: &sd,
            source_contrast: None,
        };
        assert!(hydrate_mechanism_prior(bridge, &names, Some(1), 1.0, true).unwrap().is_none());
        // The matching shape binds.
        let (quantities, mean, sd) = coefficient_summary(2);
        let bridge = MediationPriorBridge {
            mapping: &mapping,
            quantities: &quantities,
            mean: &mean,
            sd: &sd,
            source_contrast: None,
        };
        assert!(hydrate_mechanism_prior(bridge, &names, Some(1), 1.0, true).unwrap().is_some());
        // A malformed artifact is an error, not an unbound mechanism.
        let short_sd = [0.1];
        let bridge = MediationPriorBridge {
            mapping: &mapping,
            quantities: &quantities,
            mean: &mean,
            sd: &short_sd,
            source_contrast: None,
        };
        assert!(hydrate_mechanism_prior(bridge, &names, Some(1), 1.0, true).is_err());
        // The mismatch is a typed variant carrying both counts.
        let (quantities, ..) = coefficient_summary(3);
        let err = crate::bayesian::hydrate_prior_from_quantity_summaries(
            &quantities,
            &[0.5; 4],
            &[0.1; 4],
            Some(2),
        )
        .unwrap_err();
        assert_eq!(err, EstimationError::PriorDimensionMismatch { posterior: 3, design: 2 });
    }

    /// `w`, `t` independent; `m = 0.6 t + 0.5 w + e`, `y = 0.4 t + 0.5 m + 0.7 w + e`.
    /// `w` confounds the mediator-outcome relation but not `t -> y`.
    #[allow(clippy::many_single_char_names)]
    fn confounded_mediator(n: usize) -> TabularData {
        let mut rng =
            ExecutionContext::for_tests(5).rng.stream_for(StreamDomain::Estimate, 0x0057_A71C);
        let mut draw = || rng.next_f64() - 0.5;
        let (mut t, mut m, mut y, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            w[i] = draw();
            t[i] = draw();
            m[i] = 0.6 * t[i] + 0.5 * w[i] + 0.3 * draw();
            y[i] = 0.4 * t[i] + 0.5 * m[i] + 0.7 * w[i] + 0.3 * draw();
        }
        TabularData::from_f64_columns([("t", &t[..]), ("m", &m[..]), ("y", &y[..]), ("w", &w[..])])
            .unwrap()
    }

    fn graph(with_confounder: bool) -> Dag {
        let mut dag = Dag::with_variables(4);
        let [t, m, y, w] = [0, 1, 2, 3].map(DenseNodeId::from_raw);
        dag.insert_directed(t, m).unwrap();
        dag.insert_directed(t, y).unwrap();
        dag.insert_directed(m, y).unwrap();
        if with_confounder {
            dag.insert_directed(w, m).unwrap();
            dag.insert_directed(w, y).unwrap();
        }
        dag
    }

    /// The static path regresses every node on its full graph parent set, so a
    /// mediator-outcome confounder that is a graph parent of `m` and `y` is
    /// adjusted in both inference modes (the temporal-path omission does not
    /// exist here). Dropping the `w` edges from the graph
    /// reproduces the omitted-confounder bias, so the check is sensitive.
    #[test]
    fn mediator_outcome_confounder_is_adjusted_in_both_modes() {
        let data = confounded_mediator(20_000);
        let ctx = ExecutionContext::for_tests(3);
        let truths = [0.4 + 0.6 * 0.5, 0.4, 0.6 * 0.5];
        let mut q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::NaturalIndirect,
        );
        q.control = Intervention::set(q.treatment, Value::f64(0.0));
        q.active = Intervention::set(q.treatment, Value::f64(1.0));
        let freq =
            estimate_static_mediation(&data, &graph(true), &q, AssumptionSet::new(), 0, &[], &ctx)
                .unwrap();
        let bayes = estimate_static_mediation_bayesian(
            &data,
            &graph(true),
            &q,
            AssumptionSet::new(),
            &[],
            &crate::BayesianGComputationAte::conjugate(),
            antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions,
            None,
            &ctx,
        )
        .unwrap()
        .0;
        for (label, est) in [("frequentist", &freq), ("bayesian", &bayes)] {
            for (name, got, want) in [
                ("total", est.total.unwrap(), truths[0]),
                ("direct", est.direct.unwrap(), truths[1]),
                ("mediated", est.mediated.unwrap(), truths[2]),
            ] {
                assert!((got - want).abs() < 0.02, "{label} {name}: {got} vs {want}");
            }
        }
        let omitted =
            estimate_static_mediation(&data, &graph(false), &q, AssumptionSet::new(), 0, &[], &ctx)
                .unwrap();
        assert!(
            (omitted.mediated.unwrap() - truths[2]).abs() > 0.1,
            "omitting w must bias the mediated effect, got {}",
            omitted.mediated.unwrap()
        );
    }

    /// The indirect estimate is `total − direct` (the total natural indirect
    /// effect) while the certificate names the pure one; the indirect contrast
    /// records the no-interaction assumption that equates them, in both modes.
    #[test]
    fn indirect_contrast_records_no_interaction_assumption() {
        let data = confounded_mediator(2_000);
        let ctx = ExecutionContext::for_tests(3);
        let declares = |set: &AssumptionSet| {
            set.entries.iter().any(|r| {
                matches!(&r.assumption, Assumption::ParametricRestriction(p)
                    if p.id.as_ref() == "mediation.no_interaction")
            })
        };
        for (contrast, expected) in [
            (MediationContrast::NaturalIndirect, true),
            (MediationContrast::Mediated, true),
            (MediationContrast::NaturalDirect, false),
            (MediationContrast::Total, false),
        ] {
            let q = MediationQuery::binary(
                VariableId::from_raw(0),
                VariableId::from_raw(2),
                [VariableId::from_raw(1)],
                contrast,
            );
            let freq = estimate_static_mediation(
                &data,
                &graph(true),
                &q,
                AssumptionSet::new(),
                0,
                &[],
                &ctx,
            )
            .unwrap();
            assert_eq!(declares(&freq.effect.assumptions), expected, "{contrast:?}");
        }
        let q = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::NaturalIndirect,
        );
        let (bayes, posterior) = estimate_static_mediation_bayesian(
            &data,
            &graph(true),
            &q,
            AssumptionSet::new(),
            &[],
            &crate::BayesianGComputationAte::conjugate(),
            antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions,
            None,
            &ctx,
        )
        .unwrap();
        assert!(declares(&bayes.effect.assumptions) && declares(&posterior.assumptions));
    }

    /// A rare binary mediator makes some pairs resamples singular (the mediator
    /// is constant); those replicates are counted as failures and the rest still
    /// publish an SE, instead of the first singular replicate aborting the run.
    #[test]
    fn singular_bootstrap_replicates_are_counted_not_fatal() {
        let n = 14u32;
        let t: Vec<f64> = (0..n).map(|i| f64::from(i) / 7.0 - 1.0).collect();
        let m: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i == 3 || i == 10))).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let r = i as usize;
                0.5 * t[r] + 2.0 * m[r] + 0.1 * (f64::from(i) * 1.7).sin()
            })
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            Arc::from([VariableId::from_raw(1)]),
            MediationContrast::NaturalDirect,
        );
        let ctx = ExecutionContext::for_tests(5);
        let out =
            estimate_static_mediation(&data, &graph, &query, AssumptionSet::new(), 60, &[], &ctx)
                .unwrap();
        let ok = out.effect.bootstrap_replicates_ok.unwrap();
        let failed = out.effect.bootstrap_replicates_failed.unwrap();
        assert_eq!(ok + failed, 60);
        assert!(failed > 0, "a constant-mediator resample must be counted as failed");
        assert!(out.effect.se_bootstrap.is_some_and(f64::is_finite));
    }
}
