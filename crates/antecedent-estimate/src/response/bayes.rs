//! Bayesian posterior-draw response estimation.

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, CredibleDraws, DerivativeWeighting, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, IdentificationStatus, Intervention, ParametricAssumption,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    StreamDomain, SupportRegion, SupportStatus, VariableId,
};
use antecedent_data::TabularData;
use antecedent_stats::{
    AdditiveDesign, GlmFamily, QuantileRule, equal_tail_interval_sorted, silverman_bandwidth,
};

use crate::EstimationError;
use crate::tiled_prediction::{
    AxisOrder, Encoding, GridPoint, OutcomeDraws, Reduction, StreamSummary, StreamedValue,
    TileRequest, stream_tile,
};
use crate::util::range;

use super::{
    ContinuousResponseEstimator, bias_corrected_interval_note, bootstrap_weights,
    local_design_refusal, sample::CompleteSample, sample::read_shared_complete_samples,
    simultaneous_posterior_band, static_bayesian_policy, support_report, transform_derivative,
    transform_point_derivative, with_estimation_assumptions,
};

fn non_derivative_functional() -> EstimationError {
    EstimationError::unsupported("Bayesian derivative estimation requires a derivative functional")
}

/// One streamed response grid.
///
/// Rows are visited and reduced. The row × draw × grid block is not stored.
struct ResponseGrid<'a> {
    /// Draw-major coefficients: draw `k` occupies `[k * width, (k + 1) * width)`.
    coefficients: &'a [f64],
    /// Joint failure mask. `true` keeps draw `k` and reports it as non-finite.
    failed_draws: &'a [bool],
    /// Row-major features after the encoding columns.
    row_features: &'a [f64],
    /// Rows in `row_features`.
    row_count: u64,
    /// Feature columns after the encoding.
    row_width: u64,
    /// Dose coordinate of each grid point, in published order.
    coordinates: &'a [f64],
    /// Polynomial degree of the invariant dose encoding. Width is `degree + 1`.
    encoding_degree: u64,
    /// Inverse link applied to each row's linear predictor before the row reduction.
    family: GlmFamily,
    /// Average the row means. A single mean-row leaves the streamed value unchanged.
    average_rows: bool,
}

/// Pack coefficient columns into draw-major order.
///
/// A draw is failed when any of its coefficients is non-finite.
fn aligned_coefficient_draws(
    columns: &[&[f64]],
) -> Result<(Vec<f64>, Vec<bool>), EstimationError> {
    if columns.is_empty() {
        return Err(EstimationError::data_msg("Bayesian response has no coefficient columns"));
    }
    let n_draws = columns[0].len();
    if n_draws == 0 || columns.iter().any(|column| column.len() != n_draws) {
        return Err(EstimationError::data_msg(
            "Bayesian response coefficient draws are not aligned",
        ));
    }
    let width = columns.len();
    let cells = n_draws.checked_mul(width).ok_or_else(|| {
        EstimationError::data_msg("Bayesian response coefficient grid overflows")
    })?;
    let mut coefficients = vec![0.0_f64; cells];
    let mut failed = vec![false; n_draws];
    for (column_index, column) in columns.iter().enumerate() {
        for (draw, &coefficient) in column.iter().enumerate() {
            coefficients[draw * width + column_index] = coefficient;
            if !coefficient.is_finite() {
                failed[draw] = true;
            }
        }
    }
    Ok((coefficients, failed))
}

/// Stream the response grid through [`stream_tile`].
///
/// Each returned column is one grid point and contains every draw. A failed draw stays at its
/// index as NaN. Encodings are formed once per grid point.
fn stream_response_grid(
    request: &ResponseGrid<'_>,
) -> Result<(Vec<Vec<f64>>, StreamSummary), EstimationError> {
    let n_draws = request.failed_draws.len();
    let n_grid = request.coordinates.len();
    let cells = n_grid.checked_mul(n_draws).ok_or_else(|| {
        EstimationError::data_msg("Bayesian response draw grid overflows")
    })?;
    let mut sums = vec![0.0_f64; cells];
    let mut sites = Vec::with_capacity(n_grid);
    for (index, &coordinate) in request.coordinates.iter().enumerate() {
        let id = u64::try_from(index).map_err(|_| {
            EstimationError::data_msg("Bayesian response grid index does not fit u64")
        })?;
        sites.push(GridPoint { id, coordinate, factor: 1.0_f64 });
    }
    let grid_count = u64::try_from(sites.len()).map_err(|_| {
        EstimationError::data_msg("Bayesian response grid count does not fit u64")
    })?;
    let draw_count = u64::try_from(n_draws).map_err(|_| {
        EstimationError::data_msg("Bayesian response draw count does not fit u64")
    })?;
    let outcomes = [OutcomeDraws { id: 0, coefficients: request.coefficients }];
    let tile = TileRequest {
        row_features: request.row_features,
        row_start: 0,
        row_count: request.row_count,
        row_width: request.row_width,
        outcomes: &outcomes,
        failed_draws: request.failed_draws,
        draw_start: 0,
        draw_count,
        grid: &sites,
        grid_start: 0,
        grid_count,
        encoding: Encoding::Polynomial { degree: request.encoding_degree },
        order: AxisOrder::row_major(),
        reduction: Reduction::Full,
    };
    let family = request.family;
    let mut refused = false;
    let summary = stream_tile(&tile, |emitted| {
        let StreamedValue::Full { draw, grid_id, value, .. } = emitted else {
            refused = true;
            return;
        };
        let (Ok(draw_index), Ok(grid_index)) = (usize::try_from(draw), usize::try_from(grid_id))
        else {
            refused = true;
            return;
        };
        if grid_index >= n_grid || draw_index >= n_draws {
            refused = true;
            return;
        }
        sums[grid_index * n_draws + draw_index] += family.mean_from_eta(value);
    })
    .map_err(|error| EstimationError::data_msg(error.to_string()))?;
    if refused {
        return Err(EstimationError::data_msg(
            "Bayesian response tile emitted an index outside the grid",
        ));
    }
    if request.average_rows {
        let divisor = request.row_count as f64;
        for sum in &mut sums {
            *sum /= divisor;
        }
    }
    let columns = sums.chunks(n_draws).map(<[f64]>::to_vec).collect();
    Ok((columns, summary))
}

impl ContinuousResponseEstimator {
    /// Bayesian generalized-linear response levels, with posterior
    /// coefficient uncertainty propagated through every intervention coordinate.
    /// This estimator is parametric; it makes no doubly robust claim.
    #[allow(
        clippy::too_many_lines,
        reason = "one estimator over every response functional: validate, read the sample, fit, then summarize each grid point from the same draws"
    )]
    pub fn estimate_bayesian(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        mut assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if matches!(
            &query.functional,
            ResponseFunctional::AverageDerivative { .. }
                | ResponseFunctional::PointDerivative { .. }
                | ResponseFunctional::DirectionalDerivative { .. }
                | ResponseFunctional::Jacobian { .. }
        ) {
            return self.estimate_bayesian_derivative(
                data,
                query,
                identification_status,
                assumptions,
                estimator,
                ctx,
            );
        }
        self.validate(query, identification_status)?;
        if self.options.export_row_diagnostics {
            return Err(EstimationError::unsupported(
                "Bayesian response does not export frequentist row influence diagnostics",
            ));
        }
        let mut support_grid = None;
        let mut joint_levels = Vec::new();
        let mut primary_shift = None;
        let mut stochastic = false;
        let (outcome, treatment, mut grid, scalar) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => {
                (*outcome, treatment.variable, treatment.grid.values()?, false)
            }
            ResponseFunctional::InterventionResponse { outcome, interventions } => {
                if interventions.iter().any(|iv| matches!(iv, Intervention::Sequence(_))) {
                    return Err(EstimationError::unsupported(
                        "static Bayesian response does not accept temporal Sequence",
                    ));
                }
                stochastic =
                    interventions.iter().any(|iv| matches!(iv, Intervention::Stochastic { .. }));
                let (t, level, shift) = static_bayesian_policy(&interventions[0])?;
                for intervention in &interventions[1..] {
                    let (target, level, shift) = static_bayesian_policy(intervention)?;
                    joint_levels.push((target, level, shift));
                }
                let sample = CompleteSample::read(data, *outcome, &[t], &self.adjustment_set)?;
                if level.is_none() {
                    primary_shift = Some(shift);
                    let (lo, hi) = range(&sample.treatments);
                    support_grid = Some(vec![lo + shift, hi + shift]);
                }
                let level = level.unwrap_or_else(|| {
                    sample.treatments.iter().sum::<f64>() / sample.len() as f64 + shift
                });
                (*outcome, t, vec![level], true)
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "Bayesian response supports MeanCurve, InterventionResponse, and derivative functionals",
                ));
            }
        };
        let treatments: Vec<_> = std::iter::once(treatment)
            .chain(joint_levels.iter().map(|(target, _, _)| *target))
            .collect();
        let sample = CompleteSample::read(data, outcome, &treatments, &self.adjustment_set)?;
        let family = estimator.glm_family();
        if stochastic && !matches!(family, antecedent_stats::GlmFamily::GaussianIdentity) {
            return Err(EstimationError::unsupported(
                "non-Gaussian Bayesian response does not integrate stochastic policies by their mean",
            ));
        }
        let n = sample.len();
        if let Some(shift) = primary_shift {
            grid[0] = sample.treatments.iter().sum::<f64>() / n as f64 + shift;
            let (lo, hi) = range(&sample.treatments);
            support_grid = Some(vec![lo + shift, hi + shift]);
        }

        let mut covs: Vec<_> = treatments
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &id)| (id, &sample.treatment_matrix[i * n..(i + 1) * n]))
            .collect();
        covs.extend(
            self.adjustment_set
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, &sample.adjustment[i * n..(i + 1) * n])),
        );
        let design = antecedent_stats::CompiledDesign::linear_adjustment(
            &sample.treatments,
            &covs,
            &sample.outcome,
            &sample.keep,
        )?;
        let prep = crate::PreparedBayesianProblem {
            design,
            method: Arc::from("response.backdoor"),
            adjustment_set: self.adjustment_set.clone(),
            active: 1.0,
            control: 0.0,
            overlap: crate::OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: crate::SerialDependence::Iid,
            preparation: None,
        };
        let posterior = estimator.fit(
            &prep,
            identification_status,
            &mut crate::BayesianGCompWorkspace::default(),
            ctx,
        )?;
        let mut weights: Vec<_> =
            prep.design.matrix.chunks(n).map(|c| c.iter().sum::<f64>() / n as f64).collect();
        for (i, (_, level, shift)) in joint_levels.iter().enumerate() {
            weights[i + 2] = level.unwrap_or(weights[i + 2] + shift);
        }
        let mut means = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut sds = Vec::new();
        let bandwidth = self.options.bandwidth.unwrap_or(silverman_bandwidth(&sample.treatments)?);
        let mut ess = Vec::new();
        let mut density = Vec::new();
        // The draw vector each grid point's interval is the quantiles of is
        // retained on the published uncertainty (`CredibleDraws`).
        let mut retained: Vec<Vec<f64>> = Vec::with_capacity(grid.len());
        let coefficient_columns: Vec<&[f64]> = (0..prep.design.ncols)
            .map(|index| {
                let column = posterior
                    .draws
                    .schema
                    .quantities
                    .iter()
                    .position(|quantity| matches!(quantity,
                        antecedent_prob::PosteriorQuantityKind::Coefficient { index: i, .. } if *i == index))
                    .ok_or_else(|| EstimationError::stats_msg("Bayesian response missing coefficient draw"))?;
                posterior.draws.column(column).map_err(EstimationError::from)
            })
            .collect::<Result<_, _>>()?;
        let (coefficients, failed_draws) = aligned_coefficient_draws(&coefficient_columns)?;
        let gaussian = matches!(family, GlmFamily::GaussianIdentity);
        // One tile over the whole draw vector. A failed draw stays in its column as NaN.
        let (columns, _summary) = if gaussian {
            if weights.len() < 2 {
                return Err(EstimationError::data_msg(
                    "Bayesian response design is missing the intercept and dose columns",
                ));
            }
            let row_width = u64::try_from(weights.len() - 2).map_err(|_| {
                EstimationError::data_msg("Bayesian response row width does not fit u64")
            })?;
            stream_response_grid(&ResponseGrid {
                coefficients: &coefficients,
                failed_draws: &failed_draws,
                row_features: &weights[2..],
                row_count: 1,
                row_width,
                coordinates: &grid,
                encoding_degree: 1,
                family,
                average_rows: false,
            })?
        } else {
            // A primary shift puts the dose on each row. Otherwise the dose is the grid coordinate.
            let encoding_degree = u64::from(primary_shift.is_none());
            let encoding_columns = usize::from(primary_shift.is_none()) + 1;
            if coefficient_columns.len() < encoding_columns {
                return Err(EstimationError::data_msg(
                    "Bayesian response has fewer coefficient columns than the dose encoding",
                ));
            }
            let row_width_usize = coefficient_columns.len() - encoding_columns;
            let feature_cells = n.checked_mul(row_width_usize).ok_or_else(|| {
                EstimationError::data_msg("Bayesian response row features overflow")
            })?;
            let mut row_features = vec![0.0_f64; feature_cells];
            for row in 0..n {
                for local in 0..row_width_usize {
                    let column = encoding_columns + local;
                    let value = if column == 1 {
                        let shift = primary_shift.ok_or_else(|| {
                            EstimationError::data_msg(
                                "Bayesian response put the dose column in the row features",
                            )
                        })?;
                        sample.treatments[row] + shift
                    } else if column >= 2 && column < 1 + treatments.len() {
                        let (level, shift) =
                            (joint_levels[column - 2].1, joint_levels[column - 2].2);
                        level.unwrap_or(sample.treatment_matrix[(column - 1) * n + row] + shift)
                    } else {
                        prep.design.matrix[column * n + row]
                    };
                    row_features[row * row_width_usize + local] = value;
                }
            }
            let row_count = u64::try_from(n).map_err(|_| {
                EstimationError::data_msg("Bayesian response row count does not fit u64")
            })?;
            let row_width = u64::try_from(row_width_usize).map_err(|_| {
                EstimationError::data_msg("Bayesian response row width does not fit u64")
            })?;
            stream_response_grid(&ResponseGrid {
                coefficients: &coefficients,
                failed_draws: &failed_draws,
                row_features: &row_features,
                row_count,
                row_width,
                coordinates: &grid,
                encoding_degree,
                family,
                average_rows: true,
            })?
        };
        for values in columns {
            let (mean, lo, hi, sd) = crate::bayesian::summarize_linear_response_draws(
                values.clone(),
                self.options.confidence_level,
            )?;
            retained.push(values);
            means.push(mean);
            lower.push(lo);
            upper.push(hi);
            sds.push(sd);
        }
        let draws = Some(CredibleDraws::columns(posterior.draws.n_draws, &retained));
        if self
            .options
            .simultaneous_replicates
            .is_some_and(|minimum| posterior.draws.n_draws < minimum as usize)
        {
            return Err(EstimationError::unsupported(
                "Bayesian simultaneous band requires at least the requested number of posterior draws",
            ));
        }
        let support_points = support_grid.as_deref().unwrap_or(&grid);
        for &dose in support_points {
            let kernels: Vec<_> = sample
                .treatments
                .iter()
                .map(|t| (-0.5 * ((t - dose) / bandwidth).powi(2)).exp())
                .collect();
            let sum = kernels.iter().sum::<f64>();
            ess.push(sum * sum / kernels.iter().map(|k| k * k).sum::<f64>().max(f64::MIN_POSITIVE));
            density.push(sum / (n as f64 * bandwidth * (2.0 * std::f64::consts::PI).sqrt()));
        }
        let mut support = support_report(
            support_points,
            &sample.treatments,
            &ess,
            density,
            self.options.minimum_local_ess,
            0,
        );
        // A shifted-policy query is assessed over its shifted range, not at the
        // response grid, so it has no one-label-per-grid-coordinate evidence.
        let per_grid = support_grid.is_none();
        if !per_grid {
            support.point_status = None;
        }
        if treatments.len() > 1 {
            support.status = SupportStatus::Extrapolative;
            support.point_status =
                per_grid.then(|| vec![SupportStatus::Extrapolative; support_points.len()].into());
            support.query_region = SupportRegion {
                minima: (0..treatments.len()).map(|i| sample.treatment_column_range(i).0).collect(),
                maxima: (0..treatments.len()).map(|i| sample.treatment_column_range(i).1).collect(),
            };
            support.warnings.push(Diagnostic::new(
                "response.joint_support_unverified", DiagnosticKind::Support, DiagnosticSeverity::Warning,
                "per-treatment bounds do not certify joint policy support; posterior uncertainty conditions on the declared additive outcome model and empirical covariate distribution",
            ));
        }
        if stochastic {
            support.status = SupportStatus::Extrapolative;
            support.point_status =
                per_grid.then(|| vec![SupportStatus::Extrapolative; support_points.len()].into());
            support.warnings.push(Diagnostic::new(
                "response.stochastic_policy_support_unverified", DiagnosticKind::Support, DiagnosticSeverity::Warning,
                "the Gaussian additive model integrates stochastic policies by their exact means; local support at the mean does not certify support over the policy distribution; intervals describe the policy mean, not a predictive draw",
            ));
        }
        if !per_grid && scalar {
            // A shifted-policy query answers one scalar at the shifted mean and is
            // assessed over the shifted range; its single label is that range's
            // (worst-endpoint) summary, never a per-grid-point claim.
            support.point_status = Some(Arc::from([support.status]));
        }
        assumptions.entries.extend(posterior.assumptions.entries);
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("bayesian.response.linear_additive"), description: Arc::from(format!("{family:?} additive outcome mechanism on its link scale; each posterior draw is evaluated on the outcome scale over the empirical covariate distribution; credible intervals use coherent joint draws over the requested grid, without nuisance-distribution uncertainty")),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("response.bayesian") },
            scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
        });
        let value = if scalar {
            ResponseValue::Scalar(means[0])
        } else {
            ResponseValue::Surface {
                dimension: 1,
                grid: Arc::from(grid),
                mean: Arc::from(means.clone()),
            }
        };
        let uncertainty = if scalar {
            ResponseUncertainty::Scalar {
                standard_error: sds[0],
                level: self.options.confidence_level,
                lower: lower[0],
                upper: upper[0],
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                draws,
            }
        } else if self.options.simultaneous_replicates.is_some() {
            simultaneous_posterior_band(
                &means,
                &retained,
                &lower,
                &upper,
                &sds,
                self.options.confidence_level,
            )?
        } else {
            ResponseUncertainty::PointwiseBand {
                level: self.options.confidence_level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Credible,
                draws,
            }
        };
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(value),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from("estimate.response.bayesian"),
            horizon_identification: None,
            interaction_structurally_zero: matches!(
                &query.functional,
                ResponseFunctional::InterventionResponse { interventions, .. } if interventions.len() > 1
            ),
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one arm per derivative functional over the same fitted draws, each assembling its own estimand and provenance"
    )]
    fn estimate_bayesian_derivative(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
        estimator: &crate::BayesianGComputationAte,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        if estimator.prior.is_some() {
            return Err(EstimationError::unsupported(
                "coefficient priors require a mapping to the Riesz/local-polynomial/GAM parameterization; a linear response prior cannot be silently ignored",
            ));
        }
        query.validate()?;
        if !matches!(
            identification_status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "Bayesian derivatives require point identification",
            });
        }
        self.validate(query, identification_status)?;
        let mut assumptions = with_estimation_assumptions(assumptions, &query.functional);
        let level = self.options.confidence_level;
        let draws_n = crate::require_bayesian_n_draws(estimator.n_draws)?;
        let (value, uncertainty, support, assumption_id, assumption_text, provenance) = match &query
            .functional
        {
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                if !matches!(weighting, DerivativeWeighting::Observed) {
                    return Err(EstimationError::unsupported(
                        "Bayesian ADE currently supports observed-law weighting only",
                    ));
                }
                let sample =
                    CompleteSample::read(data, *outcome, &[*treatment], &self.adjustment_set)?;
                let (_, _, support) =
                    self.average_derivative(data, *outcome, *treatment, weighting)?;
                // Folds, their training designs and knots are fixed across draws:
                // expanded once here, refit under each draw's weights.
                let folds = self.cross_fit_folds(&sample)?;
                // One RNG stream per draw: draws are independent, so they run across
                // the context's thread budget and return in draw order.
                let values = ctx.map_indexed::<_, EstimationError, _>(draws_n, |draw, ctx| {
                    if ctx.cancellation.is_cancelled() {
                        return Err(EstimationError::unsupported("Bayesian ADE cancelled"));
                    }
                    let mut rng = ctx
                        .rng
                        .stream_for(StreamDomain::Bayesian, (0xADEB_0001_u64 << 32) | draw as u64);
                    let weights = bootstrap_weights(sample.len(), &mut rng);
                    self.weighted_cross_fitted_ade(&sample, &folds, &weights)
                })?;
                let (mean, lo, hi, sd) = summarize_scalar_draws(&values, level)?;
                (
                    ResponseValue::Scalar(mean),
                    credible_scalar_uncertainty(sd, level, lo, hi, values),
                    support,
                    "bayesian.derivative.riesz_weighted_cross_fit",
                    "Each Rubin Bayesian-bootstrap draw cross-fits the additive-GAM outcome μ and Gaussian treatment law α under Dirichlet(1,...,1)/Exp(1) row weights: every fold refits both nuisances on its weighted training rows and scores its held-out rows with the Riesz ADE φ = ∂_a μ̂_w + α_w (Y − μ̂_w), and the draw is the weighted mean of the held-out scores. This is not a frozen-score reweight of the first-fit φ_i. Held fixed: fold assignment, spline knots, and penalty. Estimator identity stays response.riesz_ade",
                    "estimate.response.riesz_ade",
                )
            }
            ResponseFunctional::PointDerivative { .. }
            | ResponseFunctional::DirectionalDerivative { .. }
            | ResponseFunctional::Jacobian { .. } => {
                self.validate(query, identification_status)?;
                let (point, _, mut support, provenance) = match &query.functional {
                    ResponseFunctional::PointDerivative {
                        outcome,
                        treatment,
                        at,
                        order,
                        scale,
                    } => {
                        let (value, uncertainty, support) =
                            self.point_derivative(data, *outcome, *treatment, *at, *order, *scale)?;
                        (value, uncertainty, support, "estimate.response.point_derivative")
                    }
                    ResponseFunctional::DirectionalDerivative {
                        outcomes,
                        treatments,
                        at,
                        direction,
                    } => {
                        let (value, uncertainty, support) =
                            self.directional_derivative(data, outcomes, treatments, at, direction)?;
                        (value, uncertainty, support, "estimate.response.gam_derivative")
                    }
                    ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                        let (value, uncertainty, support) =
                            self.jacobian(data, outcomes, treatments, at, *scale)?;
                        (value, uncertainty, support, "estimate.response.gam_derivative")
                    }
                    ResponseFunctional::MeanCurve { .. }
                    | ResponseFunctional::AverageDerivative { .. }
                    | ResponseFunctional::InterventionResponse { .. } => {
                        return Err(non_derivative_functional());
                    }
                };
                // The Frequentist withheld / bias-corrected notes describe the
                // analytic interval; the draws below supply their own interval and
                // their own disclosure.
                support.warnings.retain(|warning| {
                    !matches!(
                        warning.code.as_ref(),
                        "response.derivative_interval_withheld"
                            | "response.derivative_interval_bias_corrected"
                            | "response.derivative_interval_delta_method"
                            | "response.derivative_interval_fieller"
                            | "response.derivative_interval_unbounded"
                    )
                });
                let point_derivative =
                    matches!(query.functional, ResponseFunctional::PointDerivative { .. });
                // Every draw refits its nuisances: point draws rebuild the
                // cross-fitted Kennedy pseudo-outcome under the draw's row weights
                // (the bandwidth is the caller's fixed value, never data-selected);
                // GAM draws refit coefficients with the same continuous weights.
                let (samples, bandwidth) = match &query.functional {
                    ResponseFunctional::PointDerivative { outcome, treatment, .. } => {
                        let sample = CompleteSample::read(
                            data,
                            *outcome,
                            &[*treatment],
                            &self.adjustment_set,
                        )?;
                        let bandwidth = self
                            .options
                            .bandwidth
                            .unwrap_or(silverman_bandwidth(&sample.treatments)?);
                        (vec![sample], bandwidth)
                    }
                    ResponseFunctional::DirectionalDerivative { outcomes, treatments, .. }
                    | ResponseFunctional::Jacobian { outcomes, treatments, .. } => (
                        read_shared_complete_samples(
                            data,
                            outcomes,
                            treatments,
                            &self.adjustment_set,
                        )?,
                        0.0,
                    ),
                    ResponseFunctional::MeanCurve { .. }
                    | ResponseFunctional::AverageDerivative { .. }
                    | ResponseFunctional::InterventionResponse { .. } => {
                        return Err(non_derivative_functional());
                    }
                };
                let n = samples[0].len();
                // Fold assignment, knots and bases are fixed across draws: the point
                // route's cross-fitting folds and the GAM routes' plug-in target
                // designs are expanded once here and refit under each draw's weights.
                let folds =
                    if point_derivative { self.cross_fit_folds(&samples[0])? } else { vec![] };
                let target_designs: Vec<AdditiveDesign> = if point_derivative {
                    vec![]
                } else {
                    samples
                        .iter()
                        .map(|sample| self.outcome_target_design(sample))
                        .collect::<Result<_, _>>()?
                };
                // One RNG stream per draw: draws are independent, so they run across the
                // context's thread budget and return in draw order. Each draw yields its
                // point coordinates (local-quadratic, bias-corrected) or its GAM vector,
                // plus the GAM fits' effective degrees of freedom for the HC1-type spread
                // correction (`inflate_draws_by_edf`).
                let draws = ctx.map_indexed::<_, EstimationError, _>(draws_n, |draw, ctx| {
                    if ctx.cancellation.is_cancelled() {
                        return Err(EstimationError::unsupported("Bayesian derivative cancelled"));
                    }
                    let mut rng = ctx
                        .rng
                        .stream_for(StreamDomain::Bayesian, (0xADEB_0002_u64 << 32) | draw as u64);
                    let weights = bootstrap_weights(n, &mut rng);
                    let mut out = DerivativeDraw {
                        scalar: 0.0,
                        corrected: 0.0,
                        vector: Vec::new(),
                        edf: vec![0.0; samples.len()],
                    };
                    match &query.functional {
                        ResponseFunctional::PointDerivative { at, order, scale, .. } => {
                            let pseudo = self
                                .cross_fitted_pseudo_outcome_weighted(
                                    &samples[0],
                                    &folds,
                                    Some(&weights),
                                )?
                                .values;
                            let p = antecedent_stats::gaussian_local_quadratic_weighted(
                                &samples[0].treatments,
                                &pseudo,
                                *at,
                                bandwidth,
                                &weights,
                            )
                            .map_err(|err| {
                                local_design_refusal(err, &samples[0].treatments, *at, bandwidth)
                            })?;
                            out.scalar = transform_point_derivative(
                                p.value,
                                p.first_derivative,
                                p.second_derivative,
                                *at,
                                *order,
                                *scale,
                            )?;
                            let c = antecedent_stats::gaussian_local_quadratic_bias_corrected(
                                &samples[0].treatments,
                                &pseudo,
                                *at,
                                bandwidth,
                                Some(&weights),
                            )
                            .map_err(|err| {
                                local_design_refusal(err, &samples[0].treatments, *at, bandwidth)
                            })?;
                            out.corrected = transform_point_derivative(
                                c.value,
                                c.first_derivative,
                                c.second_derivative,
                                *at,
                                *order,
                                *scale,
                            )?;
                        }
                        ResponseFunctional::DirectionalDerivative { at, direction, .. } => {
                            for (s, sample) in samples.iter().enumerate() {
                                let fit = Self::fit_outcome_target_weighted(
                                    sample,
                                    &target_designs[s],
                                    Some(&weights),
                                )?;
                                out.edf[s] += fit.edf_approx;
                                let (_, gradient) =
                                    Self::plugin_gradient_at_fit(&fit, sample, at, Some(&weights))?;
                                out.vector.push(
                                    gradient.iter().zip(direction.iter()).map(|(a, b)| a * b).sum(),
                                );
                            }
                        }
                        ResponseFunctional::Jacobian { at, scale, .. } => {
                            for (s, sample) in samples.iter().enumerate() {
                                let fit = Self::fit_outcome_target_weighted(
                                    sample,
                                    &target_designs[s],
                                    Some(&weights),
                                )?;
                                out.edf[s] += fit.edf_approx;
                                let (level, gradient) =
                                    Self::plugin_gradient_at_fit(&fit, sample, at, Some(&weights))?;
                                for (j, raw) in gradient.into_iter().enumerate() {
                                    out.vector
                                        .push(transform_derivative(raw, at[j], level, *scale)?);
                                }
                            }
                        }
                        ResponseFunctional::MeanCurve { .. }
                        | ResponseFunctional::AverageDerivative { .. }
                        | ResponseFunctional::InterventionResponse { .. } => {
                            return Err(non_derivative_functional());
                        }
                    }
                    Ok(out)
                })?;
                let scalars: Vec<f64> = draws.iter().map(|d| d.scalar).collect();
                let corrected_scalars: Vec<f64> = draws.iter().map(|d| d.corrected).collect();
                let mut edf_sum = vec![0.0; samples.len()];
                for draw in &draws {
                    for (total, edf) in edf_sum.iter_mut().zip(&draw.edf) {
                        *total += edf;
                    }
                }
                let vectors: Vec<Vec<f64>> = draws.into_iter().map(|d| d.vector).collect();
                let (value, uncertainty) = if point_derivative {
                    // Reported value: posterior mean of the local-quadratic estimator.
                    // Interval and SD: the bias-corrected draws on the same weights.
                    let (mean, _, _, _) = summarize_scalar_draws(&scalars, level)?;
                    let (corrected_mean, lo, hi, sd) =
                        summarize_scalar_draws(&corrected_scalars, level)?;
                    support.warnings.push(bias_corrected_interval_note(true, mean, corrected_mean));
                    // The retained draws are the bias-corrected ones: the interval's.
                    (
                        ResponseValue::Scalar(mean),
                        credible_scalar_uncertainty(sd, level, lo, hi, corrected_scalars),
                    )
                } else {
                    let dim = vectors.first().map_or(0, Vec::len);
                    let mut means = vec![0.0; dim];
                    let mut lower = vec![0.0; dim];
                    let mut upper = vec![0.0; dim];
                    // Retained per coordinate *after* the edf inflation: the
                    // vector the band's quantiles were taken from.
                    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(dim);
                    for j in 0..dim {
                        let mut col: Vec<f64> = vectors.iter().map(|row| row[j]).collect();
                        // Columns are outcome-major (Jacobian) or one per outcome
                        // (directional); either way the outcome index is j·S/dim.
                        let edf = edf_sum[j * samples.len() / dim] / draws_n as f64;
                        inflate_draws_by_edf(&mut col, n, edf);
                        let (mean, lo, hi, _) = summarize_scalar_draws(&col, level)?;
                        means[j] = mean;
                        lower[j] = lo;
                        upper[j] = hi;
                        columns.push(col);
                    }
                    let draws = Some(CredibleDraws::columns(draws_n, &columns));
                    let value = match &point {
                        ResponseValue::Jacobian { outcomes, treatments, .. } => {
                            ResponseValue::Jacobian {
                                outcomes: *outcomes,
                                treatments: *treatments,
                                values: Arc::from(means),
                            }
                        }
                        _ => ResponseValue::Vector(Arc::from(means)),
                    };
                    (
                        value,
                        ResponseUncertainty::PointwiseBand {
                            level,
                            lower: Arc::from(lower),
                            upper: Arc::from(upper),
                            interpretation: antecedent_core::IntervalInterpretation::Credible,
                            draws,
                        },
                    )
                };
                (
                    value,
                    uncertainty,
                    support,
                    "bayesian.derivative.estimator_bootstrap",
                    if point_derivative {
                        "Dirichlet(1,...,1)/Exp(1) row-weight posterior of the Kennedy-DR point derivative: every draw refits the cross-fitted additive-GAM outcome and Gaussian treatment nuisances on its weighted training folds, rebuilds the pseudo-outcome, and evaluates the weighted local quadratic (posterior mean = reported value) and its robust bias-corrected coordinate at the same caller-fixed bandwidth (local-cubic slope for a first derivative, local-quartic level and curvature otherwise) (quantiles = credible interval, SD = standard_error). Held fixed: the caller bandwidth, fold assignment, spline knots, and penalty. Not a frozen-pseudo-outcome reweight. Estimator identity stays estimate.response.point_derivative"
                    } else {
                        "Dirichlet(1,...,1)/Exp(1) row-weight posterior of the additive-GAM plug-in gradient: every draw refits the GAM coefficients under the draw's row weights with fixed knots and penalty; the draw spread is inflated around the posterior mean by sqrt(n/(n − edf)) (the row-weight spread is an HC0 sandwich, which understates a fitted-coefficient functional's variance by the fit's leverage; edf is the penalized fit's effective degrees of freedom), and the pointwise band takes exchangeable-rank (type-6) quantiles of the draws. The band inherits the additive-surface restriction and any penalized-spline smoothing bias at the evaluation point; no nuisance-selection uncertainty"
                    },
                    provenance,
                )
            }
            _ => {
                return Err(EstimationError::unsupported("not a derivative functional"));
            }
        };
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from(assumption_id),
                description: Arc::from(assumption_text),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("response.bayesian.derivative"),
            },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        Ok(CausalResponse {
            estimand: query.functional.clone(),
            identification_status,
            estimate: ResponseIdentification::PointIdentified(value),
            uncertainty,
            support,
            assumptions,
            provenance_id: Arc::from(provenance),
            horizon_identification: None,
            interaction_structurally_zero: false,
        })
    }

    /// Linear Gaussian design used by Bayesian response and derivative transfer.
    ///
    /// # Errors
    ///
    /// Incomplete rows or a singular design.
    pub fn prepare_linear_response_problem(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatments: &[VariableId],
    ) -> Result<crate::PreparedBayesianProblem, EstimationError> {
        let sample = CompleteSample::read(data, outcome, treatments, &self.adjustment_set)?;
        let n = sample.len();
        let mut covs: Vec<_> = treatments
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &id)| (id, &sample.treatment_matrix[i * n..(i + 1) * n]))
            .collect();
        covs.extend(
            self.adjustment_set
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, &sample.adjustment[i * n..(i + 1) * n])),
        );
        let design = antecedent_stats::CompiledDesign::linear_adjustment(
            &sample.treatments,
            &covs,
            &sample.outcome,
            &sample.keep,
        )?;
        Ok(crate::PreparedBayesianProblem {
            design,
            method: Arc::from("response.linear"),
            adjustment_set: self.adjustment_set.clone(),
            active: 1.0,
            control: 0.0,
            overlap: crate::OverlapPolicy::ExplicitOverride,
            coef_names: None,
            unit_ids: None,
            serial_dependence: crate::SerialDependence::Iid,
            preparation: None,
        })
    }
}

/// Posterior mean, equal-tailed credible interval, and SD of a scalar's draws.
///
/// The interval uses *exchangeable-rank* quantiles (Hyndman–Fan type 6: the
/// `p`-quantile sits at rank `p·(D + 1)` of the `D` sorted draws), not the
/// `1 + p·(D − 1)` interpolation of a sample quantile. The reason is what a
/// credible interval built from finitely many draws claims: when the posterior
/// is calibrated the truth is exchangeable with the draws, and the interval
/// between order statistics `r` and `s` then covers it with probability exactly
/// `(s − r)/(D + 1)`, whatever the posterior's shape. Rank `p·(D + 1)` makes
/// that probability the level. The sample-quantile ranks sit about one order
/// statistic inside on each side, which at `D = 200` and a 90% level is about
/// one coverage point (0.890 expected) and at the interactive tier's `D = 64`
/// almost three (0.873); the exchangeable ranks give 0.900 at either count,
/// at the price of an interval that is wider on average by 1% (`D = 200`) or
/// 4% (`D = 64`) than the large-`D` limit. Ranks below 1 or above `D` clamp to
/// the extreme draws, which is the only case where the rule under-covers, and
/// a caller can widen it with more draws.
pub(super) fn summarize_scalar_draws(
    values: &[f64],
    level: f64,
) -> Result<(f64, f64, f64, f64), EstimationError> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg(
            "derivative posterior needs at least two finite draws",
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = antecedent_stats::sample_std(values);
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lower, upper) = equal_tail_interval_sorted(&sorted, level, QuantileRule::ExchangeableRank);
    Ok((mean, lower, upper, sd))
}

/// Inflate `draws` around their mean by `sqrt(n / (n − edf))`, the
/// degrees-of-freedom correction of a row-weight posterior spread.
///
/// A Dirichlet(1, …, 1) row-weight (Rubin Bayesian-bootstrap) posterior of a
/// fitted-coefficient functional `c'β̂` has, to first order in the weights,
/// the spread of `Σ_i w_i a_i e_i` with `a = B(B'B + λP)⁻¹c` the functional's
/// influence direction and `e` the full-sample residuals: its variance is the
/// HC0 sandwich `Σ_i a_i² e_i²`. Under homoskedastic errors `E[e_i²] =
/// σ²(1 − h_ii)`, so the sandwich understates the sampling variance
/// `σ² Σ a_i²` by the leverage `h_ii` of the rows that carry the functional;
/// replacing each row's leverage by the average `tr(H)/n = edf/n` gives the
/// HC1 factor `n/(n − edf)`, with `edf` the effective degrees of freedom of
/// the penalized fit (the penalty lowers the hat-matrix trace, so the
/// correction is smaller for a smoother fit). The correction is exact when
/// leverage is flat across the rows the functional loads on and conservative
/// when those rows sit at lower-than-average leverage (interior evaluation
/// points of a spline); it does not repair a mis-centred posterior. Its size
/// is bounded by the coefficient-to-row ratio: with the default six-function
/// bases on three columns (`edf ≈ 14`) and 1000 rows the factor is 1.007, so
/// the band widens by under one percent; the correction matters at small
/// samples or wide bases, not here.
pub(super) fn inflate_draws_by_edf(draws: &mut [f64], n: usize, edf: f64) {
    let n = n as f64;
    if draws.len() < 2 || !(edf.is_finite() && edf >= 0.0 && edf < n) {
        return;
    }
    let factor = (n / (n - edf)).sqrt();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    for v in draws {
        *v = mean + (*v - mean) * factor;
    }
}

/// A posterior summary published as scalar uncertainty: `sd` is the posterior standard
/// deviation and `[lo, hi]` the `level` credible interval, tagged so a consumer cannot
/// read either as a frequentist standard error or confidence interval. `draws` is the
/// vector `[lo, hi]` are the quantiles of, retained for re-summarization.
fn credible_scalar_uncertainty(
    sd: f64,
    level: f64,
    lo: f64,
    hi: f64,
    draws: Vec<f64>,
) -> ResponseUncertainty {
    ResponseUncertainty::Scalar {
        standard_error: sd,
        level,
        lower: lo,
        upper: hi,
        interpretation: antecedent_core::IntervalInterpretation::Credible,
        draws: Some(CredibleDraws::scalar(draws)),
    }
}

/// One Bayesian-bootstrap draw of a derivative functional.
struct DerivativeDraw {
    /// Local-quadratic point coordinate (point derivatives).
    scalar: f64,
    /// Robust bias-corrected point coordinate (point derivatives).
    corrected: f64,
    /// Plug-in gradient coordinates (directional derivative / Jacobian).
    vector: Vec<f64>,
    /// Effective degrees of freedom of each sample's GAM fit.
    edf: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::{GlmFamily, ResponseGrid, stream_response_grid};

    fn bits(value: f64) -> u64 {
        value.to_bits()
    }

    #[test]
    fn response_grid_streams_through_the_tile_and_keeps_a_failed_draw() {
        // Two rows, covariate 0.5 and 1.5. eta = 1 + 2 * dose + 4 * z on the complete draw.
        let coefficients = [1.0_f64, 2.0, 4.0, 5.0, 6.0, 7.0];
        let failed = [false, true];
        let features = [0.5_f64, 1.5];
        let doses = [0.0_f64, 1.0];
        let (columns, summary) = stream_response_grid(&ResponseGrid {
            coefficients: &coefficients,
            failed_draws: &failed,
            row_features: &features,
            row_count: 2,
            row_width: 1,
            coordinates: &doses,
            encoding_degree: 1,
            family: GlmFamily::GaussianIdentity,
            average_rows: true,
        })
        .unwrap();
        let stored: usize = columns.iter().map(Vec::len).sum();
        assert_eq!(summary.invariant_evaluations, 2);
        assert_eq!(summary.emitted, 2 * 2 * 2);
        assert_eq!(stored, 2 * 2);
        assert!(summary.emitted > u64::try_from(stored).unwrap_or(summary.emitted));
        assert_eq!(columns.len(), 2);
        assert_eq!(bits(columns[0][0]), bits(5.0));
        assert_eq!(bits(columns[1][0]), bits(7.0));
        assert_eq!(columns[0].len(), 2);
        assert!(columns[0][1].is_nan());
        assert!(columns[1][1].is_nan());
    }

    #[test]
    fn complete_response_grid_matches_the_row_draw_evaluation() {
        let coefficients = [1.0_f64, 2.0, 4.0, 5.0, 6.0, 7.0];
        let failed = [false, false];
        let features = [0.25_f64];
        let doses = [0.0_f64, 3.0];
        let (columns, summary) = stream_response_grid(&ResponseGrid {
            coefficients: &coefficients,
            failed_draws: &failed,
            row_features: &features,
            row_count: 1,
            row_width: 1,
            coordinates: &doses,
            encoding_degree: 1,
            family: GlmFamily::GaussianIdentity,
            average_rows: false,
        })
        .unwrap();
        assert_eq!(summary.invariant_evaluations, 2);
        assert_eq!(summary.emitted, 4);
        assert_eq!(bits(columns[0][0]), bits(2.0));
        assert_eq!(bits(columns[0][1]), bits(6.75));
        assert_eq!(bits(columns[1][0]), bits(8.0));
        assert_eq!(bits(columns[1][1]), bits(24.75));

        let link = GlmFamily::BinomialLogit;
        let (linked, _) = stream_response_grid(&ResponseGrid {
            coefficients: &[1.0_f64, 2.0, 4.0],
            failed_draws: &[false],
            row_features: &[0.5_f64, 1.5],
            row_count: 2,
            row_width: 1,
            coordinates: &[0.0_f64],
            encoding_degree: 1,
            family: link,
            average_rows: true,
        })
        .unwrap();
        let per_row = (link.mean_from_eta(3.0_f64) + link.mean_from_eta(7.0_f64)) / 2.0_f64;
        assert_eq!(bits(linked[0][0]), bits(per_row));
        assert_ne!(bits(per_row), bits(link.mean_from_eta(5.0_f64)));

        // Degree 0: the dose column is a row feature (shifted treatment), and a failed draw stays.
        let (shifted, shift_summary) = stream_response_grid(&ResponseGrid {
            coefficients: &[1.0_f64, 2.0, 9.0, 0.0],
            failed_draws: &[false, true],
            row_features: &[10.0_f64, 30.0],
            row_count: 2,
            row_width: 1,
            coordinates: &[0.0_f64],
            encoding_degree: 0,
            family: GlmFamily::GaussianIdentity,
            average_rows: true,
        })
        .unwrap();
        assert_eq!(shift_summary.invariant_evaluations, 1);
        assert_eq!(shift_summary.emitted, 4);
        assert_eq!(shifted[0].len(), 2);
        assert_eq!(bits(shifted[0][0]), bits(41.0));
        assert!(shifted[0][1].is_nan());
    }
}
