//! Bayesian temporal response estimation.
//!
//! Horizons are separate fits. Independent draws may be paired by index for a
//! simultaneous band; that product of marginals is not a joint trajectory posterior,
//! and shared time labels do not make it one.

use std::sync::Arc;

use crate::error::EstimationError;
use crate::serial_dependence::{DependenceScope, SerialDependence};
use crate::temporal_adjustment::TemporalLinearAdjustment;
use antecedent_core::{
    CausalResponse, CredibleDraws, Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity,
    ExecutionContext, IdentificationStatus, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, SupportDiagnostic,
    TargetPopulation, TemporalEffectQuery, Value,
};
use antecedent_data::TimeSeriesData;

use super::{
    TemporalIdentification, TemporalResponseEstimator, assumptions::HorizonTempering,
    assumptions::INDEPENDENT_DRAW_STATEMENT, assumptions::TEMPORAL_BAYESIAN_TEMPERING_CAPPED,
    assumptions::TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC,
    assumptions::TEMPORAL_BAYESIAN_TEMPERING_INESTIMABLE, assumptions::is_tempering_record,
    assumptions::temporal_bayesian_product_posterior_assumption,
    assumptions::temporal_bayesian_tempering_assumption, band::max_deviation_band_columns,
    band::publish_simultaneous_band, plan::TemporalInterventionPlan,
    plan::plan_temporal_intervention, support::intervention_support, support::mean_curve_support,
    surface::TREATMENT_COL, surface::design_column_means, surface::flatten_dose_horizon_grid,
    surface::horizon_identification_of,
};

impl TemporalResponseEstimator {
    /// Bayesian Gaussian linear-additive response on the identified unfolded
    /// design at each horizon. Intervals are pointwise posterior quantiles.
    #[allow(
        clippy::too_many_lines,
        reason = "validate the identifications, fit each horizon, then pair the draws across horizons to summarize every dose cell"
    )]
    pub fn estimate_bayesian(
        &self,
        data: &TimeSeriesData,
        identified: TemporalIdentification<'_>,
        query: &ResponseQuery,
        estimator: &crate::BayesianGComputationAte,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        let TemporalIdentification {
            horizons: identifications,
            status: identification_status,
            mut assumptions,
        } = identified;
        query.validate()?;
        if query.observation != antecedent_core::ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || estimator.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
        {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires complete observations, AllObserved, and GaussianIdentity",
            ));
        }
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::unsupported(
                "Bayesian temporal response requires point identification at every horizon",
            ));
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported("missing temporal response specification")
        })?;
        if identifications.len() != temporal.horizons.len() {
            return Err(EstimationError::unsupported("one identification required per horizon"));
        }
        let (outcome, treatment, doses, intervention) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                (*outcome, treatment.variable, treatment.grid.values()?, None)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                match plan_temporal_intervention(interventions, temporal)? {
                    TemporalInterventionPlan::Single { treatment, level, shift } => {
                        (*outcome, treatment, Vec::new(), Some((level, shift)))
                    }
                    TemporalInterventionPlan::Sequential { .. }
                    | TemporalInterventionPlan::Mechanisms { .. } => {
                        return Err(EstimationError::unsupported(
                            "Bayesian multi-step Sequence uses sequential mechanism overlays, \
                             not response.temporal.bayesian / bayesian.gcomp",
                        ));
                    }
                }
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "Bayesian temporal response supports only curves and intervention responses",
                ));
            }
        };
        let mut rows = Vec::new();
        let mut ranges = Vec::new();
        let mut horizons = Vec::new();
        let mut levels = Vec::new();
        let mut tempering: Vec<HorizonTempering> = Vec::new();
        for (h, (&horizon_steps, &(estimand, indexer))) in
            temporal.horizons.iter().zip(identifications).enumerate()
        {
            let pulse = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(1.0)),
                horizon_steps,
                max_history_lag: temporal.max_history_lag,
                target_population: TargetPopulation::AllObserved,
                effect_modifiers: std::sync::Arc::from([]),
            };
            let prep = TemporalLinearAdjustment::new().prepare(
                data,
                estimand,
                &pulse,
                indexer,
                None,
                &ctx.kernel_policy,
            )?;
            let n = prep.design.nrows;
            let t = &prep.design.matrix[n..2 * n];
            ranges.push((
                t.iter().copied().fold(f64::INFINITY, f64::min),
                t.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ));
            horizons.push(horizon_identification_of(
                horizon_steps,
                estimand,
                indexer,
                identification_status,
            )?);
            let mut weights = design_column_means(&prep.design);
            let grid = if let Some((level, shift)) = intervention {
                vec![level.unwrap_or(weights[1] + shift)]
            } else {
                doses.clone()
            };
            let mut est = estimator.clone();
            est.seed = est.seed.wrapping_add(h as u64);
            let mut bprep = crate::BayesianGComputationAte::from_prepared_estimation(&prep);
            // Lag-aligned rows are time-ordered: at h ≥ 2 the unfolded regression omits the
            // intermediate treatments and outcome innovations, so its residuals are MA(h−1)
            // whenever the outcome or treatment is persistent, and serially dependent at every
            // horizon under autocorrelated innovations. Each horizon's likelihood is tempered
            // by the long-run-variance ratio of its grid-cell levels (floored at 1), the same
            // generalized posterior as the Bayesian Pulse / Sustained cells.
            bprep.serial_dependence = SerialDependence::LongRunTempering(DependenceScope::Levels(
                grid.iter()
                    .map(|&dose| {
                        let mut direction = weights.clone();
                        direction[TREATMENT_COL] = dose;
                        Arc::from(direction)
                    })
                    .collect(),
            ));
            let posterior = est.fit(
                &bprep,
                identification_status,
                &mut crate::BayesianGCompWorkspace::default(),
                ctx,
            )?;
            if h == 0 {
                assumptions.entries.extend(
                    posterior
                        .assumptions
                        .entries
                        .iter()
                        .filter(|record| !is_tempering_record(record))
                        .cloned(),
                );
            }
            tempering.push(HorizonTempering::from_notes(&posterior.diagnostics.notes));
            if intervention.is_some() {
                levels.push(grid[0]);
            }
            let mut row = Vec::new();
            for dose in grid {
                weights[1] = dose;
                let values = crate::bayesian::linear_response_draws(&posterior, &weights)?;
                let summary =
                    crate::bayesian::summarize_linear_response_draws(values.clone(), 0.95)?;
                row.push((summary, values));
            }
            rows.push(row);
        }
        let mut mean = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut cell_draws: Vec<&[f64]> = Vec::new();
        // Full-length per-cell draw columns in the published order, retained for a
        // single-coordinate `InterventionResponse` so the facade can surface its
        // scalar credible interval as a `Scalar` uncertainty whose draws
        // reconstruct the published `[lower, upper]`. Curves keep `draws: None`.
        let mut band_columns: Vec<Vec<f64>> = Vec::new();
        for d in 0..if intervention.is_some() { 1 } else { doses.len() } {
            for row in &rows {
                let ((m, lo, hi, _), values) = &row[d];
                mean.push(*m);
                lower.push(*lo);
                upper.push(*hi);
                cell_draws.push(values);
                band_columns.push(values.clone());
            }
        }
        // Horizons are separate conjugate fits with independent draws. Pairing draw r
        // across horizons samples a product of marginals, used only for the simultaneous
        // band. That alignment is not a joint trajectory posterior, and shared time
        // labels do not make it one. Within a horizon every dose shares one coefficient
        // draw.
        let n_joint = cell_draws.iter().map(|values| values.len()).min().unwrap_or(0);
        let cell_draws: Vec<&[f64]> = cell_draws.iter().map(|values| &values[..n_joint]).collect();
        let simultaneous = max_deviation_band_columns(&mean, &cell_draws, 0.95);
        let (grid, dimension, mut support) = if let Some((level, shift)) = intervention {
            (
                temporal.horizons.iter().map(|&h| f64::from(h)).collect(),
                1,
                intervention_support(&levels, level, shift, temporal, &ranges),
            )
        } else {
            (
                flatten_dose_horizon_grid(&doses, &temporal.horizons)?,
                2,
                mean_curve_support(&doses, temporal, &ranges),
            )
        };
        let simultaneous_detail = format!(
            "simultaneous CREDIBLE band: max studentized deviation of the posterior draws of the \
             whole grid around the posterior mean; doses at one horizon share each coefficient \
             draw, while horizons are separate fits whose independent draws are paired by index \
             (a product of per-horizon posteriors, not a joint horizon posterior). \
             {INDEPENDENT_DRAW_STATEMENT} Conditional on the observed adjustment distribution; \
             each horizon's posterior is the long-run-tempered generalized posterior (support \
             diagnostic response.temporal_bayesian.tempering)"
        );
        publish_simultaneous_band(&mut support, simultaneous, &simultaneous_detail);
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC),
            values: Arc::from(tempering.iter().map(|t| t.kappa).collect::<Vec<_>>()),
            detail: Arc::from(
                "per-horizon likelihood tempering factor kappa (rows weighted 1/kappa): the \
                 variance ratio of that horizon's driving grid-cell-level combination under a \
                 REML-fitted autoregressive residual, scaled by the residual-scale and \
                 delta-method spread factors, bounded by three times the prewhitened Newey-West \
                 long-run-variance ratio of the driving score, floored at 1 and capped at \
                 n/(p+2) (support diagnostics response.temporal_bayesian.tempering_capped / \
                 _inestimable disclose which horizons hit either bound)",
            ),
            scope: DiagnosticScope::Global,
        });
        let capped_horizons: Vec<u32> = temporal
            .horizons
            .iter()
            .zip(&tempering)
            .filter(|(_, t)| t.capped)
            .map(|(&h, _)| h)
            .collect();
        let inestimable_horizons: Vec<u32> = temporal
            .horizons
            .iter()
            .zip(&tempering)
            .filter(|(_, t)| t.inestimable)
            .map(|(&h, _)| h)
            .collect();
        if !inestimable_horizons.is_empty() {
            support.warnings.push(Diagnostic::new(
                TEMPORAL_BAYESIAN_TEMPERING_INESTIMABLE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the long-run-variance tempering factor could not be estimated at horizon(s) \
                     {inestimable_horizons:?} (n below max(8, p+2)); those cells keep the \
                     untempered iid posterior and their credible interval is likely too narrow"
                ),
            ));
        }
        if !capped_horizons.is_empty() {
            support.warnings.push(Diagnostic::new(
                TEMPORAL_BAYESIAN_TEMPERING_CAPPED,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the long-run-variance tempering factor hit the n/(p+2) cap at horizon(s) \
                     {capped_horizons:?}; kappa is known to be too small there and the published \
                     credible interval is still too narrow"
                ),
            ));
        }
        assumptions.push(temporal_bayesian_product_posterior_assumption());
        assumptions.push(temporal_bayesian_tempering_assumption(&temporal.horizons, &tempering));
        let band_draws = if intervention.is_some() {
            retained_response_draws(&band_columns, temporal.horizons.len())
        } else {
            None
        };
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from(grid),
                dimension,
                mean: Arc::from(mean),
            }),
            uncertainty: ResponseUncertainty::PointwiseBand {
                level: 0.95,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                draws: band_draws,
            },
            support,
            assumptions,
            provenance_id: Arc::from("estimate.response.temporal.bayesian"),
            horizon_identification: Some(Arc::from(horizons)),
            interaction_structurally_zero: false,
        })
    }
}

/// Draws retained for a single-coordinate intervention response.
///
/// Horizons are separate conjugate fits on separately prepared designs, so
/// their draws are independent marginals: index r of one horizon has no
/// relation to index r of another. Only a response with exactly one horizon
/// has a single coordinate whose retained draws are a posterior in their own
/// right; several horizons retain nothing, so no consumer can read the
/// index-paired product as a joint trajectory posterior. Shared time labels
/// do not change that (a nonlinear functional across horizons has no draws to
/// evaluate and falls to the refusal path).
pub(super) fn retained_response_draws(
    columns: &[Vec<f64>],
    horizon_count: usize,
) -> Option<CredibleDraws> {
    if horizon_count != 1 || columns.len() != 1 {
        return None;
    }
    let n_draws = columns[0].len();
    (n_draws >= 2).then(|| CredibleDraws::columns(n_draws, columns))
}

#[cfg(test)]
mod tests {
    use antecedent_core::Assumption;

    use crate::temporal_response::assumptions::{
        INDEPENDENT_DRAW_STATEMENT, TEMPORAL_BAYESIAN_PRODUCT_POSTERIOR_ID, TrajectoryDrawReading,
        temporal_bayesian_product_posterior_assumption,
    };

    use super::retained_response_draws;

    #[test]
    fn independent_horizon_draws_are_a_product_posterior_not_a_trajectory() {
        let record = temporal_bayesian_product_posterior_assumption();
        let Assumption::ParametricRestriction(restriction) = &record.assumption else {
            panic!("expected a parametric restriction");
        };
        assert_eq!(restriction.id.as_ref(), TEMPORAL_BAYESIAN_PRODUCT_POSTERIOR_ID);
        let text = restriction.description.as_ref();
        assert!(text.contains("product-posterior band, not a joint horizon posterior"), "{text}");
        assert!(text.contains(INDEPENDENT_DRAW_STATEMENT), "{text}");
        assert!(text.contains("The per-horizon draws are independent."));
        assert!(text.contains("not a joint trajectory posterior"));
        assert!(text.contains("shared time labels do not make it one"));
        assert_eq!(
            TrajectoryDrawReading::IndependentProduct.statement(),
            INDEPENDENT_DRAW_STATEMENT
        );
        let column = vec![0.5, 1.5, 2.5];
        assert!(retained_response_draws(std::slice::from_ref(&column), 1).is_some());
        let columns = vec![column, vec![9.0, 8.0, 7.0]];
        assert!(retained_response_draws(&columns, 2).is_none());
    }
}
