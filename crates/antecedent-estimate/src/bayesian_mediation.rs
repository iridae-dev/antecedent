//! Gaussian posterior mechanism products for linear temporal mediation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, IdentificationStatus, Lag, MediationContrast, MediationQuery,
};
use antecedent_data::{LaggedColumn, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_model::history_operators::HistoryScan;
use antecedent_prob::{BayesLikelihood, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use antecedent_stats::CompiledDesign;

use crate::joint_coefficients::{CoefficientDraws, joint_draws, with_coefficient_columns};
use crate::serial_dependence::{DEPENDENCE_ASSUMPTION_ID, DependenceScope, SerialDependence};
use crate::{
    BayesianGComputationAte, CausalPosterior, EstimationError, OverlapPolicy,
    PreparedBayesianProblem,
};

/// Prepare the mediator and outcome mechanisms on the same lag-aligned rows.
/// Innovations independent across the two mechanisms and independent coefficient
/// priors give a posterior product; no prior changes the supplied identification
/// status.
///
/// Rows are time-ordered, so each mechanism's Gaussian likelihood is tempered by
/// the long-run-variance ratio of its path combinations
/// ([`crate::serial_dependence`], [`DependenceScope::MediationPaths`]): the
/// treatment slope `a` of the mediator mechanism, and the direct `c'`, mediated
/// `b` and total `c' + â b` gradients of the outcome mechanism (`â` the OLS
/// mediator slope).
pub fn prepare_temporal_mediation(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    ctx: &ExecutionContext,
) -> Result<[PreparedBayesianProblem; 2], EstimationError> {
    prepare_temporal_mediation_adjusted(data, estimand, query, &[], ctx)
}

/// Prepare both mechanisms with the same graph-derived baseline covariates.
pub fn prepare_temporal_mediation_adjusted(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    adjustment: &[LaggedColumn],
    ctx: &ExecutionContext,
) -> Result<[PreparedBayesianProblem; 2], EstimationError> {
    prepare_temporal_mediation_with_gaps(data, estimand, query, adjustment, ctx, false)
}

/// [`prepare_temporal_mediation_adjusted`], optionally on a series with missing time points.
///
/// With `gaps` both mechanisms are fitted on the intact lag windows (each at its original
/// time with its own lags) and their likelihoods are tempered by the segment-wise long-run
/// variance ([`SerialDependence::SegmentedTempering`]): no autocovariance crosses a gap.
///
/// # Errors
///
/// As [`prepare_temporal_mediation_adjusted`].
pub fn prepare_temporal_mediation_with_gaps(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    adjustment: &[LaggedColumn],
    ctx: &ExecutionContext,
    gaps: bool,
) -> Result<[PreparedBayesianProblem; 2], EstimationError> {
    query.validate()?;
    if !estimand.method_kind().is_ok_and(antecedent_expr::EstimandMethod::is_temporal_mediation)
        || estimand.mediators.len() != 1
    {
        return Err(EstimationError::unsupported(
            "Bayesian temporal mediation requires one identified mediator",
        ));
    }
    let mediator = estimand.mediators[0];
    let treatment_lag = query.horizons.first().copied().filter(|&h| h >= 1).unwrap_or(1);
    let mut columns = vec![
        LaggedColumn { variable: query.treatment, lag: Lag::from_raw(treatment_lag) },
        LaggedColumn { variable: mediator, lag: Lag::CONTEMPORANEOUS },
        LaggedColumn { variable: query.outcome, lag: Lag::CONTEMPORANEOUS },
    ];
    columns.extend_from_slice(adjustment);
    let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(1);
    let mut scan = HistoryScan::plan(data, max_lag, Arc::from(columns))?;
    let (sample, complete) = scan.windows(data, &ctx.kernel_policy, gaps)?;
    let segments = complete.map(|complete| crate::serial_dependence::window_segments(&complete));
    let active = crate::adjustment::intervention_f64(&query.active)?;
    let control = crate::adjustment::intervention_f64(&query.control)?;
    if active.to_bits() == control.to_bits() || (active == 0.0 && control == 0.0) {
        return Err(EstimationError::unsupported("mediation contrast must have distinct levels"));
    }
    let build = |outcome: &[f64],
                 covs: &[(antecedent_core::VariableId, &[f64])]|
     -> Result<_, EstimationError> {
        Ok(PreparedBayesianProblem {
            design: CompiledDesign::linear_adjustment(sample.column(0), covs, outcome, &[])?,
            method: estimand.method.clone(),
            adjustment_set: Arc::from([]),
            active,
            control,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
            preparation: None,
        })
    };
    let covs: Vec<_> = adjustment
        .iter()
        .enumerate()
        .map(|(i, column)| (column.variable, sample.column(3 + i)))
        .collect();
    let mut outcome_covs = vec![(mediator, sample.column(1))];
    outcome_covs.extend_from_slice(&covs);
    let mediator_mechanism = build(sample.column(1), &covs)?;
    let outcome_mechanism = build(sample.column(2), &outcome_covs)?;
    // Mediator design [1 | T | Z…]: the path slope `a` is column 1. Outcome design
    // [1 | T | M | Z…]: `c'` is column 1 and `b` column 2.
    let unit = |p: usize, entries: &[(usize, f64)]| -> Arc<[f64]> {
        let mut c = vec![0.0; p];
        for &(i, v) in entries {
            c[i] = v;
        }
        Arc::from(c)
    };
    let a_hat = ols_coefficient(&mediator_mechanism.design, 1)?;
    let pm = mediator_mechanism.design.ncols;
    let po = outcome_mechanism.design.ncols;
    let mediator_paths = DependenceScope::MediationPaths(Arc::from([unit(pm, &[(1, 1.0)])]));
    let outcome_paths = DependenceScope::MediationPaths(Arc::from([
        unit(po, &[(1, 1.0)]),
        unit(po, &[(2, 1.0)]),
        unit(po, &[(1, 1.0), (2, a_hat)]),
    ]));
    let dependence = |scope: DependenceScope| match &segments {
        Some(segments) => SerialDependence::SegmentedTempering {
            scope,
            segments: std::sync::Arc::clone(segments),
        },
        None => SerialDependence::LongRunTempering(scope),
    };
    Ok([
        mediator_mechanism.with_serial_dependence(dependence(mediator_paths)),
        outcome_mechanism.with_serial_dependence(dependence(outcome_paths)),
    ])
}

/// OLS coefficient `index` of `design` (the plug-in path slope for a tempering gradient).
fn ols_coefficient(design: &CompiledDesign, index: usize) -> Result<f64, EstimationError> {
    use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
    let fit = FaerBackend.least_squares(
        &design.matrix[..design.nrows * design.ncols],
        design.nrows,
        design.ncols,
        &design.outcome,
        &mut LeastSquaresWorkspace::default(),
    )?;
    fit.coefficients
        .get(index)
        .copied()
        .filter(|v| v.is_finite())
        .ok_or_else(|| EstimationError::stats_msg("mediation path slope is not estimable"))
}

/// Diagnostics-note prefix of mechanisms whose draws are one joint law.
pub const MEDIATION_JOINT_NOTE: &str = "mediation.joint_mechanisms";

/// Diagnostics-note prefix of mechanisms that keep independent draws because the joint
/// dependence could not be formed.
pub const MEDIATION_JOINT_UNAVAILABLE_NOTE: &str = "mediation.joint_unavailable";

/// Tie the fitted mediator and outcome mechanisms into one joint law.
///
/// Both mechanisms are regressions on the same lag-aligned rows, so their posteriors share the
/// calendar, the treatment regressor and, through the series, their innovations. Each keeps the
/// mean and covariance of its own posterior draws and the two are tied by the long-run
/// covariance of their stacked normal-equation scores
/// ([`crate::joint_coefficients`]); draw `r` of `a`, `c'` and `b` is then one draw of the whole
/// system, so `a·b`, `c' + a·b` and every functional of them inherit the dependence. Rows are
/// taken as adjacent in time; on a series with gaps only the cross products across a gap
/// boundary are affected. The non-coefficient quantities and the unevaluable, unidentified and
/// subsampled masses of each posterior are kept as fitted.
///
/// When the joint dependence cannot be formed the mechanisms are returned unchanged with a
/// [`MEDIATION_JOINT_UNAVAILABLE_NOTE`] note, and their composition is the independent product.
#[must_use]
pub fn couple_mediation_mechanisms(
    preparations: &[PreparedBayesianProblem],
    mechanisms: Vec<CausalPosterior>,
    structural_span: usize,
    seed: u64,
    ctx: &ExecutionContext,
) -> Vec<CausalPosterior> {
    let note_all = |mut posts: Vec<CausalPosterior>, note: String| {
        for post in &mut posts {
            post.diagnostics.notes.push(Arc::from(note.as_str()));
        }
        posts
    };
    let coupled = (|| -> Result<(Vec<CausalPosterior>, usize), EstimationError> {
        if preparations.len() != 2 || mechanisms.len() != 2 {
            return Err(EstimationError::unsupported("mediation couples exactly two mechanisms"));
        }
        let rows = preparations[0].design.nrows;
        if preparations[1].design.nrows != rows {
            return Err(EstimationError::unsupported("the mechanisms do not share rows"));
        }
        let fits = [0, 1]
            .into_iter()
            .map(|k| {
                CoefficientDraws::of(&mechanisms[k], &preparations[k].design, (0..rows).collect())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (theta, block) = joint_draws(&[&fits[0], &fits[1]], structural_span, seed, &ctx.rng)?;
        let posts = [0, 1]
            .into_iter()
            .map(|k| with_coefficient_columns(&mechanisms[k], &theta[k]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((posts, block))
    })();
    match coupled {
        Ok((posts, block)) => note_all(
            posts,
            format!(
                "{MEDIATION_JOINT_NOTE}: mediator and outcome coefficients are one joint draw tied \
                 by the long-run covariance of their stacked scores (block length {block})"
            ),
        ),
        Err(error) => note_all(
            mechanisms,
            format!("{MEDIATION_JOINT_UNAVAILABLE_NOTE}: {error}; independent draws"),
        ),
    }
}

/// Compose mediator/outcome mechanism draws into a posterior over the requested contrast,
/// retaining all three decomposition quantities. Mechanisms tied by
/// [`couple_mediation_mechanisms`] compose as a joint law; others compose as an independent
/// product.
pub fn compose_temporal_mediation(
    mediator: &CausalPosterior,
    outcome: &CausalPosterior,
    query: &MediationQuery,
    status: IdentificationStatus,
) -> Result<CausalPosterior, EstimationError> {
    let coefficient = |post: &CausalPosterior, index| -> Result<Vec<f64>, EstimationError> {
        let col = post
            .draws
            .schema
            .quantities
            .iter()
            .position(
                |q| matches!(q, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index),
            )
            .ok_or_else(|| EstimationError::stats_msg("mediation posterior missing coefficient"))?;
        Ok(post.draws.column(col)?.to_vec())
    };
    let a = coefficient(mediator, 1)?;
    let cp = coefficient(outcome, 1)?;
    let b = coefficient(outcome, 2)?;
    let delta = crate::adjustment::intervention_f64(&query.active)?
        - crate::adjustment::intervention_f64(&query.control)?;
    let n = a.len().min(b.len());
    let direct: Vec<_> = cp.iter().take(n).map(|v| v * delta).collect();
    let indirect: Vec<_> = a.iter().zip(&b).map(|(a, b)| a * b * delta).collect();
    let total: Vec<_> = direct.iter().zip(&indirect).map(|(d, i)| d + i).collect();
    let effect = match query.contrast {
        MediationContrast::Total => &total,
        MediationContrast::Direct | MediationContrast::NaturalDirect => &direct,
        MediationContrast::Mediated | MediationContrast::NaturalIndirect => &indirect,
    };
    let values: Vec<_> =
        effect.iter().chain(&total).chain(&direct).chain(&indirect).copied().collect();
    let schema = PosteriorSchema {
        quantities: Arc::from([
            PosteriorQuantityKind::Effect { name: Arc::from("mediation") },
            PosteriorQuantityKind::Scalar { name: Arc::from("total") },
            PosteriorQuantityKind::Scalar { name: Arc::from("direct") },
            PosteriorQuantityKind::Scalar { name: Arc::from("mediated") },
        ]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n, values)?;
    let tempered = [mediator, outcome].iter().any(|post| {
        post.assumptions.entries.iter().any(|record| {
            matches!(&record.assumption, antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == DEPENDENCE_ASSUMPTION_ID)
        })
    });
    let joint = [mediator, outcome].iter().any(|post| {
        post.diagnostics.notes.iter().any(|note| note.starts_with(MEDIATION_JOINT_NOTE))
    });
    let mut post = outcome.clone();
    post.identification = status;
    post.summaries = draws.summarize();
    post.draws = draws;
    // Both mechanisms' tempering notes stay on the composed diagnostics.
    post.diagnostics.notes.extend(mediator.diagnostics.notes.iter().cloned());
    post.assumptions.entries.extend(mediator.assumptions.entries.iter().cloned());
    post.assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("temporal.mediation.gaussian_product"),
            description: Arc::from(if joint {
                "linear additive mediator and outcome mechanisms; the coefficients of both are one joint Gaussian draw that keeps each mechanism's posterior mean and covariance and takes the dependence between the mechanisms from the long-run covariance of their stacked normal-equation scores on the shared rows, so the path products inherit the dependence of the shared series and innovations; no treatment-mediator interaction; empirical complete rows fixed"
            } else if tempered {
                "linear additive mediator and outcome mechanisms with Gaussian innovations independent across the two mechanisms and independent coefficient priors; each mechanism's likelihood is tempered by the long-run-variance ratio of its path coefficients for serial dependence (a generalized posterior, not a correlated-innovation likelihood); no treatment-mediator interaction; empirical complete rows fixed"
            } else {
                "linear additive mediator and outcome mechanisms with independent Gaussian innovations and independent coefficient priors; no treatment-mediator interaction; empirical complete rows fixed"
            }),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("temporal.mediation.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    Ok(post)
}

/// Refuse likelihoods that do not implement the linear mechanism product.
pub fn require_gaussian_mediation(
    estimator: &BayesianGComputationAte,
) -> Result<(), EstimationError> {
    if estimator.likelihood != BayesLikelihood::GaussianIdentity {
        return Err(EstimationError::unsupported("Bayesian mediation requires GaussianIdentity"));
    }
    if estimator.backend == crate::BayesianBackendKind::Hmc {
        return Err(EstimationError::unsupported(
            "composed mediation HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::VariableId;

    use super::*;
    use crate::BayesianGCompWorkspace;

    fn normal(state: &mut u64) -> f64 {
        let mut uniform = || {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        (0..12).map(|_| uniform()).sum::<f64>() - 6.0
    }

    /// `T` persistent, `M = 0.8 T + e1`, `Y = 0.5 T + 0.7 M + 0.6 e1 + w`: the two mechanisms'
    /// innovations are correlated, so the mediator slope and the direct effect are too.
    fn series(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(5);
        let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            t[i] = if i == 0 { 0.0 } else { 0.5 * t[i - 1] } + normal(&mut state);
            let e1 = normal(&mut state);
            m[i] = 0.8 * t[i] + e1;
            y[i] = 0.5 * t[i] + 0.7 * m[i] + 0.6 * e1 + 0.5 * normal(&mut state);
        }
        (t, m, y)
    }

    /// OLS slope of `m` on `t` and of `y` on `(t, m)`, by centred normal equations.
    fn ols(t: &[f64], m: &[f64], y: &[f64]) -> (f64, f64) {
        let n = t.len() as f64;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
        let (mt, mm, my) = (mean(t), mean(m), mean(y));
        let s = |a: &[f64], ma: f64, b: &[f64], mb: f64| {
            a.iter().zip(b).map(|(x, z)| (x - ma) * (z - mb)).sum::<f64>()
        };
        let (stt, stm, smm) = (s(t, mt, t, mt), s(t, mt, m, mm), s(m, mm, m, mm));
        let (sty, smy) = (s(t, mt, y, my), s(m, mm, y, my));
        let a = stm / stt;
        let det = stt * smm - stm * stm;
        let direct = (smm * sty - stm * smy) / det;
        (a, direct)
    }

    fn problem(
        treatment: &[f64],
        covariate: Option<(&[f64], VariableId)>,
        outcome: &[f64],
    ) -> PreparedBayesianProblem {
        let covs: Vec<(VariableId, &[f64])> =
            covariate.into_iter().map(|(values, id)| (id, values)).collect();
        PreparedBayesianProblem {
            design: CompiledDesign::linear_adjustment(treatment, &covs, outcome, &[]).unwrap(),
            method: Arc::from("temporal.mediation.bayesian"),
            adjustment_set: Arc::from([]),
            active: 1.0,
            control: 0.0,
            overlap: OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: SerialDependence::Iid,
            preparation: None,
        }
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|x| (x - ma) * (x - ma)).sum();
        let vb: f64 = b.iter().map(|y| (y - mb) * (y - mb)).sum();
        cov / (va * vb).sqrt()
    }

    #[test]
    fn coupled_mechanisms_carry_the_dependence_the_repeated_simulation_shows() {
        // Independent reference: the correlation of the mediator slope and the direct effect
        // over fresh simulated series.
        let (mut slopes, mut directs) = (Vec::new(), Vec::new());
        for seed in 1..=400 {
            let (t, m, y) = series(250, 500 + seed);
            let (a, direct) = ols(&t, &m, &y);
            slopes.push(a);
            directs.push(direct);
        }
        let reference = corr(&slopes, &directs);
        assert!(reference.abs() > 0.3, "the design must carry dependence: {reference}");

        let ctx = ExecutionContext::for_tests(9);
        let (t, m, y) = series(250, 3);
        let mediator_id = VariableId::from_raw(1);
        let preparations = [problem(&t, None, &m), problem(&t, Some((&m, mediator_id)), &y)];
        let estimator = BayesianGComputationAte::conjugate();
        let fitted: Vec<CausalPosterior> = preparations
            .iter()
            .enumerate()
            .map(|(i, prep)| {
                let mut est = estimator.clone();
                est.seed = est.seed.wrapping_add(i as u64);
                est.fit(
                    prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut BayesianGCompWorkspace::default(),
                    &ctx,
                )
                .unwrap()
            })
            .collect();
        let column = |post: &CausalPosterior, index: usize| -> Vec<f64> {
            let q = post
                .draws
                .schema
                .quantities
                .iter()
                .position(|k| matches!(k, PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index))
                .unwrap();
            post.draws.column(q).unwrap().to_vec()
        };
        let (a, c) = (column(&fitted[0], 1), column(&fitted[1], 1));
        let n = a.len().min(c.len());
        let independent = corr(&a[..n], &c[..n]);
        assert!(
            independent.abs() < 0.1,
            "separately fitted mechanisms are independent: {independent}"
        );

        let coupled = couple_mediation_mechanisms(&preparations, fitted.clone(), 1, 17, &ctx);
        assert!(coupled[0].diagnostics.notes.iter().any(|n| n.starts_with(MEDIATION_JOINT_NOTE)));
        let (a, c) = (column(&coupled[0], 1), column(&coupled[1], 1));
        let joint = corr(&a, &c);
        assert!(
            (joint - reference).abs() < 0.2,
            "joint correlation {joint} vs simulated {reference}"
        );
        // Each mechanism keeps its own marginal spread.
        for (before, after) in [(&fitted[0], &coupled[0]), (&fitted[1], &coupled[1])] {
            let sd = |v: &[f64]| {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
            };
            let ratio = sd(&column(after, 1)) / sd(&column(before, 1));
            assert!((ratio - 1.0).abs() < 0.1, "marginal spread ratio {ratio}");
        }
    }
}
