//! Prepared ADMG response-curve execution.

use super::{
    Arc, AverageEffectQuery, CausalError, CausalQuery, CheckedAdmgResponseCurveOperation,
    CheckedFunctionalEffectResponseMember, EffectEstimate, EstimationWorkspace, ExecutionContext,
    GraphClass, InferenceMode, Instant, OverlapPolicy, RefutationRun, RefuteSuite,
    STAGE_ESTIMATE_POINT, StudyResult, TableView, TabularData,
};

/// Per-member results of evaluating an ADMG response grid.
struct MemberEvaluations {
    means: Vec<f64>,
    point_status: Vec<antecedent_core::SupportStatus>,
    member_posteriors: Vec<antecedent_estimate::CausalPosterior>,
    support_status: antecedent_core::SupportStatus,
    support_warnings: Vec<antecedent_core::Diagnostic>,
    freq_bounds: Vec<(f64, f64)>,
    freq_scalar_se: f64,
    freq_replicates_ok: u32,
}

impl CheckedAdmgResponseCurveOperation {
    pub(in crate::analysis::prepared) fn sealed_for_direct_execution(&self) -> bool {
        matches!(
            self.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        ) && self.identifier == crate::strategy_table::IdentifierId::GeneralId
            && self.estimator == crate::strategy_table::EstimatorId::FunctionalEffect
            && matches!(self.inference, InferenceMode::Frequentist | InferenceMode::Bayesian(_))
            && matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            && self.custom_validator_names.is_empty()
            && self.members.len() == self.grid.len()
    }

    pub(in crate::analysis::prepared) fn rebind(
        &self,
        data: &TabularData,
    ) -> Result<Self, CausalError> {
        let members = self
            .members
            .iter()
            .map(|member| {
                let prepared = member.prepared.rebind_checked(data).map_err(CausalError::from)?;
                Ok(CheckedFunctionalEffectResponseMember {
                    grid_value: member.grid_value,
                    query: member.query.clone(),
                    identification: member.identification.clone(),
                    estimand: member.estimand.clone(),
                    prepared,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?;
        if matches!(self.inference, InferenceMode::Bayesian(_)) {
            // Member calls use the same seeded Bayesian-bootstrap stream. Require
            // every member to use every source row so its stream positions name
            // identical rows and therefore one shared row-weight draw.
            for member in &members {
                let snapshot = member.prepared.factor_snapshot().map_err(CausalError::from)?;
                if snapshot.provenance.source_rows != data.row_count()
                    || snapshot.provenance.complete_case_rows != data.row_count()
                {
                    return Err(CausalError::Unsupported {
                        message: "Bayesian ADMG response curves require a common complete-case row set across grid members",
                    });
                }
            }
        }
        Ok(Self {
            factor_snapshot_digest: data.storage().content_digest(),
            query: self.query.clone(),
            grid: Arc::clone(&self.grid),
            members: Arc::from(members),
            fitter: self.fitter.clone(),
            common: Arc::clone(&self.common),
        })
    }

    pub(in crate::analysis::prepared) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if self.members.is_empty() || self.members.len() != self.grid.len() {
            return Err(CausalError::Compile {
                message: "retained ADMG response grid/member mapping is incomplete".into(),
            });
        }
        let MemberEvaluations {
            means,
            point_status,
            member_posteriors,
            support_status,
            support_warnings,
            freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
        } = self.evaluate_members(ctx)?;
        let n_draws = member_posteriors
            .first()
            .map(|posterior| u32::try_from(posterior.draws.n_draws).unwrap_or(u32::MAX));
        let posterior = self.joint_posterior(member_posteriors)?;
        let first = self.members.first().ok_or(CausalError::Unsupported {
            message: "a general-identification response needs at least one member",
        })?;
        let mut support = self.response_support(support_status, point_status, &support_warnings)?;
        let scalar_intervention = matches!(
            self.query.functional,
            antecedent_core::ResponseFunctional::InterventionResponse { .. }
        );
        let scalar = if scalar_intervention { means[0] } else { f64::NAN };
        let freq_uncertainty = self.frequentist_uncertainty(
            &mut support,
            &freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
            means.len(),
            scalar_intervention,
        );
        let frequentist_bootstrap = matches!(self.inference, InferenceMode::Frequentist)
            && self.fitter.bootstrap_replicates > 0;
        let bootstrap_requested =
            if frequentist_bootstrap { self.fitter.bootstrap_replicates } else { 0 };
        let bootstrap_ok = frequentist_bootstrap.then_some(freq_replicates_ok);
        let response = antecedent_core::CausalResponse {
            estimand: self.query.functional.clone(),
            identification_status: first.identification.status,
            estimate: antecedent_core::ResponseIdentification::PointIdentified(
                self.response_value(scalar_intervention, scalar, means),
            ),
            uncertainty: posterior
                .as_ref()
                .and_then(|posterior| credible_response_uncertainty(posterior, scalar_intervention))
                .or(freq_uncertainty)
                .unwrap_or(antecedent_core::ResponseUncertainty::None),
            support,
            assumptions: first.identification.required_assumptions.clone(),
            provenance_id: Arc::from("estimate.response.general_id"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        };
        let mut diagnostics = general_id_diagnostics(first, support_warnings);
        let (treatment, outcome) =
            first.query.functional.primary_pair().ok_or(CausalError::Unsupported {
                message: "the response functional names no treatment and outcome pair",
            })?;
        let refutations =
            if scalar_intervention && scalar.is_finite() && self.refute != RefuteSuite::None {
                let (reports, refute_diags) =
                    self.plugin_level_refutations(data, first, scalar, treatment, outcome, ctx)?;
                diagnostics.extend(refute_diags);
                reports
            } else {
                Vec::new()
            };
        let estimate = EffectEstimate::new(
            f64::NAN,
            f64::NAN,
            first.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let result = crate::analysis::execute::CheckedResultFinish::new(
            &self.common,
            &first.identification,
            &first.estimand,
            estimate,
            started,
            (treatment, outcome),
            (CausalQuery::Response(self.query.clone()), GraphClass::Admg),
        )
        .with_estimate_assumptions(first.prepared.assumptions.clone())
        .with_diagnostics(diagnostics)
        .with_refutations(refutations)
        .with_posterior(posterior)
        .with_n_draws(n_draws)
        .with_bootstrap(Some(bootstrap_requested), bootstrap_ok)
        .with_response(response)
        .without_evalue_mirror()
        .with_rebound_interval()
        .finish();
        Ok(result)
    }

    /// Point value of the response: the single plug-in level, or the mean surface over the grid.
    fn response_value(
        &self,
        scalar_intervention: bool,
        scalar: f64,
        means: Vec<f64>,
    ) -> antecedent_core::ResponseValue {
        if scalar_intervention {
            antecedent_core::ResponseValue::Scalar(scalar)
        } else {
            antecedent_core::ResponseValue::Surface {
                grid: Arc::clone(&self.grid),
                dimension: 1,
                mean: Arc::from(means),
            }
        }
    }

    /// Evaluates every grid member under the configured inference mode, in grid order,
    /// retaining per-member posteriors and the frequentist bootstrap bounds.
    fn evaluate_members(&self, ctx: &ExecutionContext) -> Result<MemberEvaluations, CausalError> {
        let mut means = Vec::with_capacity(self.members.len());
        let mut point_status = Vec::with_capacity(self.members.len());
        let mut member_posteriors = Vec::new();
        let mut support_status = antecedent_core::SupportStatus::Supported;
        let mut support_warnings = Vec::new();
        // Frequentist general-ID responses publish a normal interval from the
        // per-member front-door plug-in bootstrap SE (the sibling AverageEffect /
        // InterventionalDistribution ADMG cells use the same resample). One
        // `(lower, upper)` per member; the scalar case keeps its single SE.
        let mut freq_bounds: Vec<(f64, f64)> = Vec::new();
        let mut freq_scalar_se = f64::NAN;
        let mut freq_replicates_ok: u32 = 0;
        for member in self.members.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled { stage: STAGE_ESTIMATE_POINT });
            }
            let value = match &self.inference {
                InferenceMode::Frequentist => {
                    match member.prepared.evaluate(&member.prepared.provider) {
                        Ok(value) => value,
                        Err(error) if antecedent_estimate::functional_cell_unevaluable(&error) => {
                            support_status =
                                antecedent_core::SupportStatus::OutsideEmpiricalSupport;
                            support_warnings.push(antecedent_core::Diagnostic::new(
                                "functional.required_cell",
                                antecedent_core::DiagnosticKind::Scientific,
                                antecedent_core::DiagnosticSeverity::Warning,
                                error.to_string(),
                            ));
                            f64::NAN
                        }
                        Err(error) => {
                            return Err(CausalError::from(
                                antecedent_estimate::EstimationError::data_msg(error.to_string()),
                            ));
                        }
                    }
                }
                InferenceMode::Bayesian(_) => {
                    let n_draws =
                        crate::analysis::execute::functional_bayesian_draws(&self.inference)?
                            .ok_or(CausalError::Unsupported {
                                message: "Bayesian inference selects no draw count",
                            })?;
                    let fit = self
                        .fitter
                        .estimate_bayesian(
                            &member.prepared,
                            n_draws,
                            member.identification.status,
                            ctx,
                        )
                        .map_err(CausalError::from)?;
                    let mean = crate::analysis::execute::effect_from_posterior(&fit)?.ate;
                    // `estimate_bayesian` opens the same named RNG stream for each
                    // member. Its Bayesian-bootstrap row weights therefore align
                    // draw-for-draw across the fixed grid; retain every member
                    // column below so the resulting posterior keeps that joint law.
                    member_posteriors.push(fit);
                    mean
                }
            };
            point_status.push(if value.is_finite() {
                antecedent_core::SupportStatus::Supported
            } else {
                antecedent_core::SupportStatus::OutsideEmpiricalSupport
            });
            means.push(value);
            if matches!(self.inference, InferenceMode::Frequentist)
                && value.is_finite()
                && self.fitter.bootstrap_replicates > 0
            {
                let mut ws = antecedent_estimate::FunctionalDistributionWorkspace::default();
                if let Ok(effect) = self.fitter.estimate(&member.prepared, &mut ws, ctx) {
                    if let Some(se) = effect.se_bootstrap() {
                        if se.is_finite() && se > 0.0 {
                            let z = crate::result::reported_se_interval_z();
                            freq_bounds.push((value - z * se, value + z * se));
                            freq_scalar_se = se;
                            freq_replicates_ok =
                                effect.uncertainty.bootstrap.replicates_ok.unwrap_or(0);
                        }
                    }
                }
            }
        }
        Ok(MemberEvaluations {
            means,
            point_status,
            member_posteriors,
            support_status,
            support_warnings,
            freq_bounds,
            freq_scalar_se,
            freq_replicates_ok,
        })
    }

    /// Stacks the per-member posterior effect columns into one joint posterior over the
    /// grid, reusing the first member's posterior as the carrier.
    fn joint_posterior(
        &self,
        mut member_posteriors: Vec<antecedent_estimate::CausalPosterior>,
    ) -> Result<Option<antecedent_estimate::CausalPosterior>, CausalError> {
        if member_posteriors.is_empty() {
            Ok(None)
        } else {
            let n = member_posteriors[0].draws.n_draws;
            let mut columns = Vec::with_capacity(member_posteriors.len());
            for member_posterior in &member_posteriors {
                if member_posterior.draws.n_draws != n {
                    return Err(CausalError::Compile {
                        message:
                            "response curve members did not retain a common posterior draw count"
                                .into(),
                    });
                }
                let quantity =
                    member_posterior.effect_column().ok_or_else(|| CausalError::Compile {
                        message: "response curve member posterior omitted its effect quantity"
                            .into(),
                    })?;
                columns.push(member_posterior.draws.column(quantity)?.to_vec());
            }
            let schema = antecedent_prob::PosteriorSchema {
                quantities: Arc::from(
                    self.grid
                        .iter()
                        .map(|value| antecedent_prob::PosteriorQuantityKind::Scalar {
                            name: Arc::from(format!("response_at_{value}")),
                        })
                        .collect::<Vec<_>>(),
                ),
            };
            let draws = antecedent_prob::PosteriorDraws::from_column_major(
                schema,
                n,
                columns.into_iter().flatten().collect::<Vec<_>>(),
            )?;
            let summaries = draws.summarize();
            let mut joint = member_posteriors.remove(0);
            joint.draws = draws;
            joint.summaries = summaries;
            Ok(Some(joint))
        }
    }

    /// Support report for the grid: the worst member status, per-point status, the grid
    /// range as the query region, and the warnings raised while evaluating members.
    fn response_support(
        &self,
        support_status: antecedent_core::SupportStatus,
        point_status: Vec<antecedent_core::SupportStatus>,
        support_warnings: &[antecedent_core::Diagnostic],
    ) -> Result<antecedent_core::SupportReport, CausalError> {
        let mut support =
            antecedent_estimate::support_from_functional_eval(None).map_err(|error| {
                CausalError::from(antecedent_estimate::EstimationError::data_msg(error.to_string()))
            })?;
        support.status = support_status;
        support.point_status = Some(Arc::from(point_status));
        support.query_region = antecedent_core::SupportRegion {
            minima: Arc::from([self.grid.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([self.grid.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        };
        support.warnings.extend(support_warnings.iter().cloned());
        Ok(support)
    }

    /// Normal interval from the per-member bootstrap SE, published only when every
    /// member returned one; records the bootstrap-SE support marker when it does.
    fn frequentist_uncertainty(
        &self,
        support: &mut antecedent_core::SupportReport,
        freq_bounds: &[(f64, f64)],
        freq_scalar_se: f64,
        freq_replicates_ok: u32,
        member_count: usize,
        scalar_intervention: bool,
    ) -> Option<antecedent_core::ResponseUncertainty> {
        // Publish the frequentist bootstrap interval only when every member
        // returned a positive-finite SE; otherwise fail open to `None`. The
        // support marker makes the facade report `bootstrap_se` for this band.
        if matches!(self.inference, InferenceMode::Frequentist)
            && !freq_bounds.is_empty()
            && freq_bounds.len() == member_count
        {
            support.diagnostics.push(antecedent_core::SupportDiagnostic {
                id: Arc::from(crate::result::RESPONSE_BOOTSTRAP_SE),
                values: Arc::from([
                    f64::from(self.fitter.bootstrap_replicates),
                    f64::from(freq_replicates_ok),
                ]),
                detail: Arc::from(
                    "general-ID functional.effect response interval is a normal interval from the \
                     front-door plug-in bootstrap SE [requested replicates, successful replicates]",
                ),
                scope: antecedent_core::DiagnosticScope::Global,
            });
            if scalar_intervention {
                let (lower, upper) = freq_bounds[0];
                Some(antecedent_core::ResponseUncertainty::Scalar {
                    standard_error: freq_scalar_se,
                    lower,
                    upper,
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                })
            } else {
                Some(antecedent_core::ResponseUncertainty::PointwiseBand {
                    level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
                    lower: Arc::from(freq_bounds.iter().map(|(l, _)| *l).collect::<Vec<_>>()),
                    upper: Arc::from(freq_bounds.iter().map(|(_, u)| *u).collect::<Vec<_>>()),
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                })
            }
        } else {
            None
        }
    }

    /// Cheap/full refutations of the plugin intervention level for the scalar response,
    /// with the diagnostics they raise.
    fn plugin_level_refutations(
        &self,
        data: &TabularData,
        first: &CheckedFunctionalEffectResponseMember,
        scalar: f64,
        treatment: antecedent_core::VariableId,
        outcome: antecedent_core::VariableId,
        ctx: &ExecutionContext,
    ) -> Result<
        (Vec<antecedent_validate::RefutationReport>, Vec<antecedent_core::Diagnostic>),
        CausalError,
    > {
        let mut diagnostics = Vec::new();
        if matches!(self.refute, RefuteSuite::Cheap | RefuteSuite::Full) {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "refute.evalue.not_a_contrast",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "contrast-shaped refuters are not supported for a plugin intervention level; \
                 cheap runs overlap only and full runs overlap plus sampling-stability of the \
                 identified intervention mean",
            ));
        }
        let plugin_estimate = EffectEstimate::new(
            scalar,
            f64::NAN,
            first.identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut refute_ws = EstimationWorkspace::default();
        let (reports, refute_diags) = RefutationRun {
            data,
            estimand: &first.estimand,
            query: &AverageEffectQuery::binary_ate(treatment, outcome),
            estimate: &plugin_estimate,
            suite: self.refute,
            estimator: self.estimator.as_str(),
            custom: &[],
            temporal: None,
        }
        .plugin_level(&mut refute_ws, ctx)?;
        diagnostics.extend(refute_diags);
        Ok((reports, diagnostics))
    }
}

/// Identification diagnostics of the first member followed by the grid's support warnings.
fn general_id_diagnostics(
    first: &CheckedFunctionalEffectResponseMember,
    support_warnings: Vec<antecedent_core::Diagnostic>,
) -> Vec<antecedent_core::Diagnostic> {
    let mut diagnostics = first.identification.diagnostics.clone();
    diagnostics.push(antecedent_core::Diagnostic::new(
        "identify.response.general_id",
        antecedent_core::DiagnosticKind::Scientific,
        antecedent_core::DiagnosticSeverity::Info,
        "intervention mean identified by general ID; grid members are evaluated from their retained checked programs",
    ));
    diagnostics.extend(support_warnings);
    diagnostics
}

/// Credible-interval uncertainty of a response posterior, scalar or pointwise by response shape.
fn credible_response_uncertainty(
    posterior: &antecedent_estimate::CausalPosterior,
    scalar_intervention: bool,
) -> Option<antecedent_core::ResponseUncertainty> {
    if scalar_intervention {
        crate::analysis::helpers::credible_scalar_uncertainty(posterior)
    } else {
        crate::analysis::helpers::credible_pointwise_uncertainty(posterior)
    }
}
