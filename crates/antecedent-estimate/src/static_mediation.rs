//! Static additive-linear natural mediation on an identified DAG.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_arguments
)]
use crate::{EffectEstimate, EstimationError, OverlapPolicy, TemporalMediationEstimate};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, ExecutionContext, MediationContrast, MediationQuery, ParametricAssumption,
    TargetPopulation, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
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
