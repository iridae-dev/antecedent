//! Validation of composed temporal effects using the same sequential mechanisms.
use super::*;
use antecedent_estimate::temporal_sequential::{
    SequentialBayesianMechanism, estimate_sustained_window,
};
use antecedent_validate::RefutationProblem;
use antecedent_validate::common::EffectRefit;

#[derive(Clone, Debug)]
pub(super) struct SequentialValidationAtom {
    pub weight: f64,
    pub graph: TemporalDag,
    pub indexer: antecedent_core::TemporalIndexer,
    pub estimand: IdentifiedEstimand,
    pub status: IdentificationStatus,
    pub estimate: EffectEstimate,
    pub mechanisms: Vec<SequentialBayesianMechanism>,
}

#[derive(Debug)]
struct SequentialRefitter<'a> {
    atom: &'a SequentialValidationAtom,
    query: &'a TemporalEffectQuery,
    time: &'a TimeIndex,
    bayes: Option<&'a BayesianGComputationAte>,
}
impl EffectRefit for SequentialRefitter<'_> {
    fn refit(
        &self,
        data: &TabularData,
        extras: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, antecedent_validate::ValidationError> {
        let mut time = self.time.clone();
        time.length = data.row_count();
        let series = TimeSeriesData::try_new(data.storage().clone(), time)?;
        let mut graph = self.atom.graph.clone();
        let indexer =
            if extras.is_empty() {
                self.atom.indexer.clone()
            } else {
                // Independent nuisance variables become parents of each contemporaneous
                // mechanism. Refit the whole factorization, including all mediator paths.
                let children: Vec<_> = graph.nodes().iter().enumerate().filter_map(|(i, node)| {
                matches!(node, antecedent_core::NodeRef::Lagged { lag, .. } if lag.raw() == 0)
                    .then_some(DenseNodeId::from_raw(u32::try_from(i).expect("dense node")))
            }).collect();
                for &extra in extras {
                    let parent =
                        graph.add_lagged(extra, antecedent_core::Lag::CONTEMPORANEOUS).map_err(
                            |e| antecedent_validate::ValidationError::data_msg(e.to_string()),
                        )?;
                    for &child in &children {
                        graph.insert_directed(parent, child).map_err(|e| {
                            antecedent_validate::ValidationError::data_msg(e.to_string())
                        })?;
                    }
                }
                antecedent_core::TemporalIndexer::new(
                    u32::try_from(data.schema().variables().len()).map_err(|_| {
                        antecedent_validate::ValidationError::data_msg("too many variables")
                    })?,
                    self.atom.indexer.history(),
                    self.atom.indexer.horizon(),
                )
                .map_err(|e| antecedent_validate::ValidationError::data_msg(e.to_string()))?
            };
        Ok(estimate_sustained_window(
            &series,
            &graph,
            &indexer,
            &self.atom.estimand,
            self.query,
            self.atom.status,
            self.atom.estimate.assumptions.clone(),
            0,
            self.bayes,
            ctx,
        )?
        .0)
    }

    /// Refits evaluate the OLS contrast on lag-aligned rows of the original series,
    /// in circular blocks of the Frequentist contrast interval's length
    /// ([`antecedent_estimate::SequentialContrastDesign::block_length`]).
    ///
    /// For Bayesian refits the replicate OLS contrast stands in for the posterior
    /// mean ([`least_squares_stand_in`]); an informative coefficient prior has no
    /// aligned-row posterior-mean evaluation, and the check is not applicable.
    fn prepare_aligned(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Option<
        Result<antecedent_validate::common::AlignedRefit, antecedent_validate::ValidationError>,
    > {
        if self.bayes.is_some_and(|bayes| bayes.prior.is_some()) {
            return Some(Err(antecedent_validate::ValidationError::NotApplicable {
                message: "a composed Bayesian contrast under an informative coefficient prior \
                          has no lag-aligned posterior-mean refit",
            }));
        }
        let mut time = self.time.clone();
        time.length = data.row_count();
        let prepared = TimeSeriesData::try_new(data.storage().clone(), time)
            .map_err(antecedent_validate::ValidationError::from)
            .and_then(|series| {
                antecedent_estimate::SequentialContrastDesign::prepare(
                    &series,
                    &self.atom.graph,
                    &self.atom.indexer,
                    &self.atom.estimand,
                    self.query,
                    self.atom.status,
                    ctx,
                )
                .map_err(antecedent_validate::ValidationError::from)
            });
        let stand_in = self.bayes.map(least_squares_stand_in);
        let design = match prepared {
            Ok(design) => design,
            Err(error) => return Some(Err(error)),
        };
        if self.bayes.is_some() && !stand_in_is_faithful(&design, &self.atom.estimate) {
            return Some(Err(antecedent_validate::ValidationError::NotApplicable {
                message: "prior shrinkage separates the posterior mean from the least-squares \
                          contrast by more than 0.25 posterior SD, so the least-squares \
                          stand-in would resample a different estimator than the posterior mean",
            }));
        }
        Some(Ok(antecedent_validate::common::AlignedRefit {
            rows: design.aligned_rows().rows,
            block_length: design.block_length(),
            stand_in,
            estimate: Box::new(move |rows| design.estimate_on_rows(rows).ok()),
        }))
    }
}

/// Largest gap, in posterior SDs, between the Bayesian posterior mean and the
/// least-squares contrast on the same rows for the stand-in to be used.
///
/// The posterior mean is a Monte Carlo average, so even without shrinkage it
/// sits about `1/sqrt(draws)` SDs from least squares (0.1 at 100 draws); 0.25
/// leaves that noise room while a tight scale moves the mean by many SDs.
const STAND_IN_MAX_GAP_SD: f64 = 0.25;

/// Whether the least-squares contrast on all aligned rows is within
/// [`STAND_IN_MAX_GAP_SD`] posterior SDs of the published posterior mean.
///
/// Under a tight isotropic scale the ridge shrinkage `κ̂/s²` moves the posterior
/// mean away from least squares; the bootstrap check would then compare the
/// published mean against the wrong estimator and report a spurious refutation.
/// A missing or degenerate posterior SD leaves the stand-in in place.
fn stand_in_is_faithful(
    design: &antecedent_estimate::SequentialContrastDesign,
    posterior: &EffectEstimate,
) -> bool {
    let sd = posterior.se_analytic;
    if !(sd.is_finite() && sd > 0.0) {
        return true;
    }
    let all_rows: Vec<usize> = (0..design.aligned_rows().rows).collect();
    design
        .estimate_on_rows(&all_rows)
        .is_ok_and(|ls| (ls - posterior.ate).abs() <= STAND_IN_MAX_GAP_SD * sd)
}

/// Why the least-squares contrast stands in for a Bayesian sequential posterior mean
/// in `bootstrap.ci_coverage`.
///
/// Each mechanism's likelihood is tempered by `1/κ̂` under the isotropic prior
/// `β | σ² ~ N(0, σ² s² I)`, so its posterior mean is the ridge fit
/// `(X'X + (κ̂/s²) I)⁻¹ X'y`: tempering multiplies the prior's relative weight by
/// `κ̂`, and the stand-in is exact only up to `O(κ̂/(n s²))` shrinkage.
fn least_squares_stand_in(bayes: &BayesianGComputationAte) -> Arc<str> {
    Arc::from(format!(
        "bootstrap.ci_coverage refit the least-squares contrast on block-resampled \
         lag-aligned rows as a stand-in for the posterior mean: with each mechanism's \
         likelihood tempered by 1/kappa under the isotropic prior (scale {}), the posterior \
         mean is the ridge fit (X'X + kappa/scale^2 I)^-1 X'y, the least-squares fit up to \
         O(kappa/(n scale^2)) shrinkage; the check is not applicable when that shrinkage \
         moves the posterior mean more than 0.25 posterior SD from least squares, or under an \
         informative coefficient prior. It asks whether the posterior mean lies inside that \
         least-squares bootstrap interval; it does not test the published credible interval, \
         so it cannot detect a miscalibrated (too narrow or too wide) posterior",
        bayes.prior_scale
    ))
}

type SequentialValidationResults =
    (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>, Vec<PredictiveCheckReport>);

pub(super) fn validate_sequential(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atoms: &[SequentialValidationAtom],
    suite: RefuteSuite,
    custom: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    bayes: Option<&BayesianGComputationAte>,
    posterior: Option<&mut CausalPosterior>,
    original: f64,
    ctx: &ExecutionContext,
    predictive_sims: u32,
) -> Result<SequentialValidationResults, CausalError> {
    let table = TabularData::new(data.storage().clone());
    let mut average = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
    average.active = query.active.clone();
    average.control = query.control.clone();
    average.target_population = query.target_population.clone();
    let mut reports: std::collections::BTreeMap<
        Arc<str>,
        Vec<(f64, antecedent_validate::RefutationReport)>,
    > = std::collections::BTreeMap::new();
    let mut diagnostics = Vec::new();
    let mut priors =
        std::collections::BTreeMap::<VariableId, Vec<(f64, PredictiveCheckReport)>>::new();
    let mut posts =
        std::collections::BTreeMap::<VariableId, Vec<(f64, PredictiveCheckReport)>>::new();
    let mut sensitivity = Vec::new();
    let grid = antecedent_validate::PriorSensitivity::standard_grid();
    for atom in atoms {
        let refitter = SequentialRefitter { atom, query, time: data.time_index(), bayes };
        let temporal = TemporalRefitContext {
            indexer: &atom.indexer,
            temporal_query: query,
            split: None,
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
        };
        let problem = RefutationProblem::new(
            &table,
            &atom.estimand,
            &average,
            &atom.estimate,
            Some("temporal.sequential.gcomp"),
            Some(temporal),
        )
        .with_effect_refit(&refitter);
        let mut validation = match suite {
            RefuteSuite::None => ValidationSuite::new(),
            RefuteSuite::Cheap => ValidationSuite::overlap_and_evalue(),
            RefuteSuite::PlaceboAndRcc => ValidationSuite::placebo_and_rcc(),
            RefuteSuite::Full => ValidationSuite::full_effect(),
        };
        for validator in custom {
            validation = validation.with_custom(Arc::clone(validator));
        }
        let outcomes = validation.run(&problem, &mut EstimationWorkspace::default(), ctx)?;
        diagnostics
            .extend(crate::analysis::helpers::validator_not_applicable_diagnostics(&outcomes));
        if let Some(estimator) = bayes {
            let checked = ValidationSuite::reports_only(&outcomes)
                .iter()
                .any(|report| report.refuter.as_ref() == "bootstrap.ci_coverage");
            let code = "refute.bootstrap.ci_coverage.least_squares_stand_in";
            if checked && !diagnostics.iter().any(|d: &Diagnostic| d.code.as_ref() == code) {
                diagnostics.push(Diagnostic::new(
                    code,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    least_squares_stand_in(estimator).to_string(),
                ));
            }
        }
        for report in ValidationSuite::reports_only(&outcomes) {
            reports.entry(Arc::clone(&report.refuter)).or_default().push((atom.weight, report));
        }
        if let Some(estimator) = bayes.filter(|_| suite != RefuteSuite::None) {
            // Multi-step prior transfer is refused before fitting, so the isotropic
            // per-mechanism prior (`prior_scale`) is the prior in force. Refuse rather
            // than validate against a prior the mechanisms were not fitted under.
            if estimator.prior.is_some() {
                return Err(CausalError::Unsupported {
                    message: "multi-step sustained validation requires isotropic per-mechanism priors",
                });
            }
            for mechanism in &atom.mechanisms {
                let prior = estimator.prior_in_force(mechanism.prepared.design.ncols);
                let prior = PriorPredictiveCheck::for_estimator(estimator, ctx)
                    .with_n_sims(predictive_sims)
                    .check_with_prior(&mechanism.prepared, &prior, ctx)?;
                // Mechanism rows are time-ordered: `full` adds the lag-1 residual
                // autocorrelation discrepancy (C-4).
                let post_check = PosteriorPredictiveCheck::for_estimator(estimator, ctx)
                    .with_n_sims(predictive_sims);
                let post = if suite == RefuteSuite::Full {
                    post_check.check_temporal(
                        &mechanism.prepared,
                        &mechanism.posterior,
                        ctx.rng.master_seed(),
                    )?
                } else {
                    post_check.check(&mechanism.prepared, &mechanism.posterior)?
                };
                priors.entry(mechanism.variable).or_default().push((atom.weight, prior));
                posts.entry(mechanism.variable).or_default().push((atom.weight, post));
            }
            if suite == RefuteSuite::Full {
                let mut means = Vec::new();
                let mut sds = Vec::new();
                for &scale in grid.scales.iter() {
                    let mut est = estimator.clone();
                    est.prior_scale = scale;
                    let (effect, _) = estimate_sustained_window(
                        data,
                        &atom.graph,
                        &atom.indexer,
                        &atom.estimand,
                        query,
                        atom.status,
                        atom.estimate.assumptions.clone(),
                        0,
                        Some(&est),
                        ctx,
                    )?;
                    means.push(effect.ate);
                    sds.push(effect.se_analytic);
                }
                sensitivity.push((
                    atom.weight,
                    antecedent_prob::PriorSensitivitySummary {
                        family: antecedent_prob::PriorSensitivityFamily::IsotropicScale,
                        prior_scales: Arc::clone(&grid.scales),
                        effect_means: means.into(),
                        effect_sds: sds.into(),
                        ..Default::default()
                    },
                ));
            }
        }
    }
    let mut mixed_reports: Vec<_> = reports
        .values()
        .filter_map(|items| {
            antecedent_validate::RefutationReport::mixture_weighted(
                &items.iter().map(|(w, r)| (*w, r)).collect::<Vec<_>>(),
            )
        })
        .collect();
    // Graph atoms describe alternatives for the same child. Distinct mechanism
    // outcomes have different units; never average their PPC statistics together.
    let mut predictive = Vec::new();
    for checks in [&priors, &posts] {
        for (variable, items) in checks {
            if let Some(mixed) = PredictiveCheckReport::mixture_weighted(
                &items.iter().map(|(w, r)| (*w, r)).collect::<Vec<_>>(),
            ) {
                let atom_reports: Vec<_> = items
                    .iter()
                    .map(|(weight, check)| (*weight, check.to_refutation_report(original, 0.05)))
                    .collect();
                let mut report = antecedent_validate::RefutationReport::mixture_weighted(
                    &atom_reports
                        .iter()
                        .map(|(weight, check)| (*weight, check))
                        .collect::<Vec<_>>(),
                )
                .expect("positive mechanism weight");
                report.refuter =
                    Arc::from(format!("{}.mechanism.{}", report.refuter, variable.raw()));
                mixed_reports.push(report);
                predictive.push(mixed);
            }
        }
    }
    if let (Some(post), Some(summary)) = (
        posterior,
        mix_prior_sensitivity_summaries(
            &sensitivity.iter().map(|(w, s)| (*w, s)).collect::<Vec<_>>(),
        ),
    ) {
        mixed_reports.push(grid.to_report(&summary, original));
        *post = with_prior_sensitivity(post.clone(), summary);
    }
    Ok((mixed_reports, diagnostics, predictive))
}
