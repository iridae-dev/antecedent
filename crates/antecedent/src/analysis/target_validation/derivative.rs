//! Finite-difference checks for derivative targets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalQuery, ContinuousDomain, DERIVATIVE_POINTS_PER_COORDINATE, DerivativeCheckPoint,
    DerivativeCoordinate, DerivativeScale, DerivativeWeighting, ExecutionContext, GridSpec,
    Intervention, PreparedRefit, PreparedTargetData, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, TargetRefitError, TargetRefutationReport, TargetRefuterKind,
    ValidationError, Value, VariableId, admit_target_cells, finite_difference_consistency,
    float_column, local_linearity, merge_derivative_checks, sample_sd, unsupported,
};

impl PreparedRefit<'_> {
    fn derivative_treatment_column(&self, variable: VariableId) -> Vec<f64> {
        match self.data {
            PreparedTargetData::Panel(panel) => panel
                .units()
                .iter()
                .filter_map(|unit| float_column(unit.series.storage(), variable))
                .flatten()
                .copied()
                .collect(),
            PreparedTargetData::MultiEnv(environments) => environments
                .environments()
                .iter()
                .filter_map(|series| float_column(series.storage(), variable))
                .flatten()
                .copied()
                .collect(),
            _ => float_column(self.storage(), variable).map(<[f64]>::to_vec).unwrap_or_default(),
        }
    }

    /// The fitted response curve of a derivative at the points the derivative diagnostics read,
    /// through the compiler: the study's mean-curve query, under its own temporal
    /// specification, on the same data.
    ///
    /// A point derivative is read at its evaluation point. An average derivative is read at
    /// seven nodes of its weighting law (quantiles of the observed treatment, or midpoints of
    /// the uniform interval), so its finite difference is the mean of the nodes' point
    /// differences. Each node contributes the five grid points `x`, `x +- step / 2` and
    /// `x +- step`. A multi-horizon specification reads one curve column per horizon.
    ///
    /// Returns the nodes, the reported value of each horizon and the step.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "one check that classifies the derivative, picks its nodes and step and fits its curve once"
    )]
    fn response_curve(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), TargetRefitError> {
        const NODES: usize = 7;
        let Some(response) = self.result.response() else {
            return Err(unsupported(
                "the derivative diagnostics read a response result",
                "a derivative response query",
            ));
        };
        let treatment_column = || {
            let mut column = self.derivative_treatment_column(self.treatment);
            column.retain(|value| value.is_finite());
            column
        };
        let (outcome, treatment, nodes) = match &response.estimand {
            ResponseFunctional::PointDerivative {
                outcome,
                treatment,
                at,
                order: 1,
                scale: DerivativeScale::Identity,
            } => (*outcome, *treatment, vec![*at]),
            ResponseFunctional::PointDerivative { .. } => {
                return Err(unsupported(
                    "the consistency check compares a first derivative on the identity scale",
                    "query the first derivative on the identity scale",
                ));
            }
            ResponseFunctional::AverageDerivative { outcome, treatment, weighting } => {
                let nodes: Vec<f64> = match weighting {
                    DerivativeWeighting::Observed => {
                        let mut column = treatment_column();
                        column.sort_by(f64::total_cmp);
                        if column.len() < NODES {
                            return Err(unsupported(
                                "the observed treatment has too few values to place the nodes \
                                 of the weighting law",
                                "more data",
                            ));
                        }
                        (0..NODES)
                            .map(|k| {
                                let rank = ((2 * k + 1) * column.len()) / (2 * NODES);
                                column[rank.min(column.len() - 1)]
                            })
                            .collect()
                    }
                    DerivativeWeighting::Uniform { lower, upper } => (0..NODES)
                        .map(|k| lower + (upper - lower) * (k as f64 + 0.5) / NODES as f64)
                        .collect(),
                    DerivativeWeighting::Custom(_) => {
                        return Err(unsupported(
                            "caller-supplied row weights have no support points to place nodes at",
                            "check point derivatives at chosen points",
                        ));
                    }
                };
                (*outcome, *treatment, nodes)
            }
            _ => {
                return Err(unsupported(
                    "a directional derivative or Jacobian has several treatments; its finite \
                     difference needs the joint response surface",
                    "check each treatment's point derivative",
                ));
            }
        };
        let horizons = match &self.study().query {
            CausalQuery::Response(original) => {
                original.temporal.as_ref().map_or(1, |spec| spec.horizons.len())
            }
            _ => 1,
        };
        if self.original.len() != horizons {
            return Err(unsupported(
                "the reported derivative does not have one value per horizon of the specification",
                "a derivative query with one value per horizon",
            ));
        }
        let reported = self.original.clone();
        let step = if let Some(step) = self.options.derivative_step {
            step
        } else {
            0.25 * sample_sd(&treatment_column())
        };
        if !(step.is_finite() && step > 0.0) {
            return Err(unsupported(
                "the treatment has no spread to take a derivative step from",
                "declare derivative_step",
            ));
        }
        if self.curve.get().is_none() {
            let half = 0.5 * step;
            // The grid of a mean curve is strictly increasing, and the neighbourhoods of the
            // nodes overlap.
            let mut points: Vec<f64> = nodes
                .iter()
                .flat_map(|&node| [-step, -half, 0.0, half, step].map(|offset| node + offset))
                .collect();
            points.sort_by(f64::total_cmp);
            points.dedup_by(|a, b| a.to_bits() == b.to_bits());
            let mut curve_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome,
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from(points.clone())),
                ),
            });
            // A temporal derivative's curve is fit under the same temporal specification.
            if let CausalQuery::Response(original) = &self.study().query {
                curve_query.temporal.clone_from(&original.temporal);
            }
            let query = CausalQuery::Response(curve_query);
            let mut study = self.study().clone();
            study.query = query;
            let refitted =
                if let Ok(result) = Self::compile_and_run(study.clone(), &self.modal_data(), ctx) {
                    result
                } else {
                    // The derivative's own estimator may not serve a mean curve; let the
                    // compiler choose the curve estimator of the same data and graph.
                    study.declared.estimator = None;
                    study.declared.estimator_spec = None;
                    Self::compile_and_run(study, &self.modal_data(), ctx).map_err(|error| {
                        unsupported(
                            format!("the study's mean curve did not run: {error}"),
                            "a graph and data that identify the dose-response curve",
                        )
                    })?
                };
            let curve = refitted
                .response()
                .and_then(|r| match &r.estimate {
                    ResponseIdentification::PointIdentified(ResponseValue::Surface {
                        mean,
                        ..
                    }) if mean.len() == points.len() * horizons => Some(
                        points
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(dose, point)| {
                                (point, mean[dose * horizons..(dose + 1) * horizons].to_vec())
                            })
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .ok_or_else(|| {
                    unsupported(
                        "the mean curve is not point identified on the check grid",
                        "a point-identified dose-response curve",
                    )
                })?;
            let _ = self.curve.set(curve);
        }
        Ok((nodes, reported, step))
    }

    fn curve_value(&self, x: f64, horizon: usize) -> Result<f64, ValidationError> {
        self.curve
            .get()
            .and_then(|curve| curve.iter().find(|(p, _)| p.to_bits() == x.to_bits()))
            .and_then(|(_, values)| values.get(horizon).copied())
            .ok_or_else(|| ValidationError::estimation_msg("response curve read off its grid"))
    }

    /// The mean over the nodes of the curve shifted by `offset`: its derivative at zero is the
    /// mean of the nodes' derivatives, which for an average derivative is the finite
    /// difference of the weighted average.
    fn node_mean(
        &self,
        nodes: &[f64],
        offset: f64,
        horizon: usize,
    ) -> Result<f64, ValidationError> {
        let mut total = 0.0;
        for &node in nodes {
            total += self.curve_value(node + offset, horizon)?;
        }
        Ok(total / nodes.len() as f64)
    }

    pub(super) fn derivative_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        if let Some(response) = self.result.response()
            && matches!(
                response.estimand,
                ResponseFunctional::DirectionalDerivative { .. }
                    | ResponseFunctional::Jacobian { .. }
            )
        {
            return self.vector_derivative_check(kind, ctx);
        }
        let (nodes, reported, step) = self.response_curve(ctx)?;
        let mut coordinates = Vec::with_capacity(reported.len());
        for (horizon, &value) in reported.iter().enumerate() {
            coordinates.push(self.check_coordinate(
                kind,
                |offset| self.node_mean(&nodes, offset, horizon),
                value,
                step,
            )?);
        }
        self.merge_coordinates(kind, coordinates)
    }

    /// One coordinate's check along a response function of the offset from the evaluation
    /// point.
    fn check_coordinate(
        &self,
        kind: TargetRefuterKind,
        response: impl FnMut(f64) -> Result<f64, ValidationError>,
        reported: f64,
        step: f64,
    ) -> Result<DerivativeCoordinate, TargetRefitError> {
        Ok(if kind == TargetRefuterKind::FiniteDifferenceConsistency {
            DerivativeCoordinate::FiniteDifference(finite_difference_consistency(
                response,
                &[DerivativeCheckPoint { x: 0.0, reported }],
                step,
                self.options.derivative_tolerance,
            )?)
        } else {
            DerivativeCoordinate::Linearity(local_linearity(
                response,
                &[0.0],
                step,
                self.options.linearity_tolerance,
                1e-9,
            )?)
        })
    }

    /// One report on the whole target vector from the per-coordinate checks.
    fn merge_coordinates(
        &self,
        kind: TargetRefuterKind,
        coordinates: Vec<DerivativeCoordinate>,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        merge_derivative_checks(kind, coordinates, &self.target)
    }

    /// The first-order check of a directional derivative or a Jacobian.
    ///
    /// Every coordinate (an outcome, and for a Jacobian a treatment) is the derivative of the
    /// outcome's interventional mean along a direction at the evaluation point, so its
    /// finite difference is read off the study's own `InterventionResponse` query at the
    /// shifted points `at + t * direction`: a Jacobian entry along the treatment's axis, a
    /// directional derivative along the declared direction.
    #[allow(
        clippy::too_many_lines,
        reason = "one check that classifies the derivative, places its points and fits each mean"
    )]
    fn vector_derivative_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let Some(response) = self.result.response() else {
            return Err(unsupported(
                "the derivative diagnostics read a response result",
                "a derivative response query",
            ));
        };
        let (outcomes, treatments, at, directions): (
            Vec<VariableId>,
            Vec<VariableId>,
            Vec<f64>,
            Vec<Vec<f64>>,
        ) = match &response.estimand {
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                (outcomes.to_vec(), treatments.to_vec(), at.to_vec(), vec![direction.to_vec()])
            }
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if *scale != DerivativeScale::Identity {
                    return Err(unsupported(
                        "the consistency check compares derivatives on the identity scale",
                        "query the Jacobian on the identity scale",
                    ));
                }
                let axes = (0..treatments.len())
                    .map(|k| (0..treatments.len()).map(|j| f64::from(u8::from(j == k))).collect())
                    .collect();
                (outcomes.to_vec(), treatments.to_vec(), at.to_vec(), axes)
            }
            _ => {
                return Err(unsupported("not a vector derivative", "a derivative query"));
            }
        };
        let coordinates: Vec<(usize, usize)> =
            (0..outcomes.len()).flat_map(|o| (0..directions.len()).map(move |d| (o, d))).collect();
        if coordinates.len() != self.original.len() {
            return Err(unsupported(
                "the reported derivative vector does not match its outcomes and directions",
                "check the derivative query's outcomes and directions",
            ));
        }
        // Each coordinate reads a few interventional means; the cache and check buffers are
        // admitted against the context's budget instead of a coordinate count.
        let _lease = admit_target_cells(ctx, coordinates.len(), DERIVATIVE_POINTS_PER_COORDINATE)?;
        // The step along a direction: a share of the smallest treatment spread it crosses.
        let spread = |variable: VariableId| -> f64 {
            sample_sd(&self.derivative_treatment_column(variable))
        };
        let step = if let Some(step) = self.options.derivative_step {
            step
        } else {
            directions
                .iter()
                .flat_map(|direction| {
                    treatments
                        .iter()
                        .zip(direction)
                        .filter(|(_, d)| **d != 0.0)
                        .map(|(&variable, d)| 0.25 * spread(variable) / d.abs())
                        .collect::<Vec<_>>()
                })
                .fold(f64::INFINITY, f64::min)
        };
        if !(step.is_finite() && step > 0.0) {
            return Err(unsupported(
                "the treatments have no spread to take a derivative step from",
                "declare derivative_step",
            ));
        }
        let cache: std::cell::RefCell<std::collections::HashMap<(usize, Vec<u64>), f64>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
        let mean_at = |outcome_index: usize, point: &[f64]| -> Result<f64, ValidationError> {
            let key = (outcome_index, point.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            if let Some(found) = cache.borrow().get(&key) {
                return Ok(*found);
            }
            let interventions: Vec<Intervention> = treatments
                .iter()
                .zip(point)
                .map(|(&variable, &level)| Intervention::set(variable, Value::f64(level)))
                .collect();
            let mut study = self.study().clone();
            study.query = CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::InterventionResponse {
                    outcome: outcomes[outcome_index],
                    interventions: Arc::from(interventions),
                },
            ));
            let run = |study: crate::analysis::execute::Study| {
                Self::compile_and_run(study, &self.modal_data(), ctx)
            };
            let result = if let Ok(result) = run(study.clone()) {
                result
            } else {
                study.declared.estimator = None;
                study.declared.estimator_spec = None;
                run(study).map_err(|error| {
                    ValidationError::estimation_msg(format!(
                        "the interventional mean did not run: {error}"
                    ))
                })?
            };
            let value = result
                .response()
                .and_then(|r| match &r.estimate {
                    ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => Some(*v),
                    _ => None,
                })
                .ok_or_else(|| {
                    ValidationError::estimation_msg(
                        "the interventional mean is not point identified at a check point",
                    )
                })?;
            cache.borrow_mut().insert(key, value);
            Ok(value)
        };
        let mut checks = Vec::with_capacity(coordinates.len());
        for (index, &(outcome_index, direction_index)) in coordinates.iter().enumerate() {
            let direction = &directions[direction_index];
            let reported = self.original[index];
            checks.push(self.check_coordinate(
                kind,
                |offset| {
                    let point: Vec<f64> =
                        at.iter().zip(direction).map(|(a, d)| a + offset * d).collect();
                    mean_at(outcome_index, &point)
                },
                reported,
                step,
            )?);
        }
        self.merge_coordinates(kind, checks)
    }
}
