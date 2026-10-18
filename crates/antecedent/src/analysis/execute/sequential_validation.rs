//! Validation of composed temporal effects using the same sequential mechanisms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BayesianGComputationAte, CausalError, CausalPosterior, DenseNodeId, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimationWorkspace, ExecutionContext,
    IdentificationStatus, IdentifiedEstimand, PosteriorPredictiveCheck, PredictiveCheckReport,
    PriorPredictiveCheck, RefuteSuite, TableView, TabularData, TemporalDag, TemporalEffectQuery,
    TemporalRefitContext, TimeIndex, TimeSeriesData, ValidationSuite, VariableId,
    mix_prior_sensitivity_summaries, temporal_refuter_average, with_prior_sensitivity,
};
use antecedent_estimate::temporal_sequential::{
    SequentialBayesianMechanism, estimate_sustained_window, estimate_sustained_window_gaps_if,
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
    /// The series has missing time points: refits fit the intact lag windows.
    gaps: bool,
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
        let indexer = if extras.is_empty() {
            self.atom.indexer.clone()
        } else {
            // Independent nuisance variables become parents of each contemporaneous
            // mechanism. Refit the whole factorization, including all mediator paths.
            let children: Vec<_> = graph.nodes().iter().enumerate().filter_map(|(i, node)| {
                let contemporaneous =
                    matches!(node, antecedent_core::NodeRef::Lagged { lag, .. } if lag.raw() == 0);
                u32::try_from(i).ok().filter(|_| contemporaneous).map(DenseNodeId::from_raw)
            }).collect();
            for &extra in extras {
                let parent = graph
                    .add_lagged(extra, antecedent_core::Lag::CONTEMPORANEOUS)
                    .map_err(|e| antecedent_validate::ValidationError::data_msg(e.to_string()))?;
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
        Ok(estimate_sustained_window_gaps_if(
            &antecedent_estimate::SequentialProblem {
                data: &series,
                graph: &graph,
                indexer: &indexer,
                estimand: &self.atom.estimand,
            },
            self.query,
            antecedent_estimate::SequentialInference {
                status: self.atom.status,
                assumptions: self.atom.estimate.provenance.assumptions.clone(),
                bootstrap_replicates: 0,
                bayesian: self.bayes,
            },
            ctx,
            None,
            self.gaps,
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
        self.prepare_aligned_windows(data, ctx).map(|prepared| {
            prepared.and_then(|(refit, windows)| match windows {
                None => Ok(refit),
                Some(_) => Err(antecedent_validate::ValidationError::NotApplicable {
                    message: "the aligned rows of a series with missing time points are not \
                              consecutive; resample them through the intact-window map",
                }),
            })
        })
    }

    /// [`Self::prepare_aligned`], with the intact-window map when the series has gaps.
    fn prepare_aligned_windows(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Option<
        Result<
            (
                antecedent_validate::common::AlignedRefit,
                Option<antecedent_validate::common::WindowCompleteness>,
            ),
            antecedent_validate::ValidationError,
        >,
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
                antecedent_estimate::SequentialContrastDesign::prepare_with_gaps(
                    &antecedent_estimate::SequentialProblem {
                        data: &series,
                        graph: &self.atom.graph,
                        indexer: &self.atom.indexer,
                        estimand: &self.atom.estimand,
                    },
                    self.query,
                    self.atom.status,
                    ctx,
                    self.gaps,
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
        let rows = design.aligned_rows().rows;
        let window_complete = design.has_gaps().then(|| design.gapped_rows().complete);
        // A bootstrapped Frequentist contrast carries the length its interval
        // resampled; otherwise derive it once from the prepared design.
        let block_length = match self.atom.estimate.uncertainty.block.resampling {
            // On a series with gaps the compacted rows are not consecutive: the structural
            // rule on the original window axis stands.
            _ if window_complete.is_some() => antecedent_data::circular_block_length(
                design.structural_span(),
                design.gapped_rows().complete.len(),
            ),
            Some(geometry)
                if self.bayes.is_none()
                    && self.atom.estimate.uncertainty.bootstrap.replicates_ok.is_some()
                    && geometry.rows == rows =>
            {
                geometry.block_length
            }
            _ => design.block_length(),
        };
        Some(Ok((
            antecedent_validate::common::AlignedRefit {
                rows,
                block_length,
                stand_in,
                estimate: Box::new(move |rows| design.estimate_on_rows(rows).ok()),
            },
            window_complete,
        )))
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
    let sd = posterior.se_analytic();
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

/// What a sequential validation pass runs against: the series and query, the suite and
/// custom validators, and the predictive-check budget.
#[derive(Clone, Copy)]
pub(super) struct SequentialValidation<'a> {
    pub(super) data: &'a TimeSeriesData,
    pub(super) query: &'a TemporalEffectQuery,
    pub(super) suite: RefuteSuite,
    pub(super) custom: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    pub(super) predictive_sims: u32,
    /// The series has missing time points: refuters refit the intact lag windows and resample
    /// blocks of the original window axis.
    pub(super) gaps: bool,
}

type SequentialValidationResults =
    (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>, Vec<PredictiveCheckReport>);

pub(super) fn validate_sequential(
    setup: SequentialValidation<'_>,
    atoms: &[SequentialValidationAtom],
    bayes: Option<&BayesianGComputationAte>,
    posterior: Option<&mut CausalPosterior>,
    original: f64,
    ctx: &ExecutionContext,
) -> Result<SequentialValidationResults, CausalError> {
    let pass = SequentialPass {
        table: TabularData::new(setup.data.storage().clone()),
        average: temporal_refuter_average(setup.query),
        grid: antecedent_validate::PriorSensitivity::standard_grid(),
        setup,
        bayes,
    };
    let mut accumulated = SequentialAccumulators::default();
    for atom in atoms {
        pass.validate_atom(atom, &mut accumulated, ctx)?;
    }
    Ok(mix_sequential_results(accumulated, &pass.grid, original, posterior))
}

/// The shared inputs every atom of one sequential validation pass reads.
struct SequentialPass<'a> {
    setup: SequentialValidation<'a>,
    bayes: Option<&'a BayesianGComputationAte>,
    table: TabularData,
    average: antecedent_core::AverageEffectQuery,
    grid: antecedent_validate::PriorSensitivity,
}

/// Per-atom validation products awaiting the cross-atom mixture.
#[derive(Default)]
struct SequentialAccumulators {
    reports:
        std::collections::BTreeMap<Arc<str>, Vec<(f64, antecedent_validate::RefutationReport)>>,
    diagnostics: Vec<Diagnostic>,
    priors: std::collections::BTreeMap<VariableId, Vec<(f64, PredictiveCheckReport)>>,
    posts: std::collections::BTreeMap<VariableId, Vec<(f64, PredictiveCheckReport)>>,
    sensitivity: Vec<(f64, antecedent_prob::PriorSensitivitySummary)>,
}

impl SequentialPass<'_> {
    /// Runs the validation suite, predictive checks, and prior-sensitivity sweep for one graph atom.
    fn validate_atom(
        &self,
        atom: &SequentialValidationAtom,
        acc: &mut SequentialAccumulators,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let SequentialValidation { data, query, suite, custom, .. } = self.setup;
        let bayes = self.bayes;
        let table = &self.table;
        let average = &self.average;
        let refitter = SequentialRefitter {
            atom,
            query,
            time: data.time_index(),
            bayes,
            gaps: self.setup.gaps,
        };
        let temporal = TemporalRefitContext {
            indexer: &atom.indexer,
            temporal_query: query,
            split: None,
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
            gaps: self.setup.gaps,
        };
        let problem = RefutationProblem::new(
            table,
            &atom.estimand,
            average,
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
        acc.diagnostics
            .extend(crate::analysis::helpers::validator_not_applicable_diagnostics(&outcomes));
        if let Some(estimator) = bayes {
            let checked = ValidationSuite::reports_only(&outcomes)
                .iter()
                .any(|report| report.refuter.as_ref() == "bootstrap.ci_coverage");
            let code = "refute.bootstrap.ci_coverage.least_squares_stand_in";
            if checked && !acc.diagnostics.iter().any(|d: &Diagnostic| d.code.as_ref() == code) {
                acc.diagnostics.push(Diagnostic::new(
                    code,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    least_squares_stand_in(estimator).to_string(),
                ));
            }
        }
        for report in ValidationSuite::reports_only(&outcomes) {
            acc.reports.entry(Arc::clone(&report.refuter)).or_default().push((atom.weight, report));
        }
        if let Some(estimator) = bayes.filter(|_| suite != RefuteSuite::None) {
            self.check_mechanisms(atom, estimator, acc, ctx)?;
        }
        Ok(())
    }

    /// Prior and posterior predictive checks per mechanism, and under the full suite the prior-scale sensitivity sweep.
    fn check_mechanisms(
        &self,
        atom: &SequentialValidationAtom,
        estimator: &BayesianGComputationAte,
        acc: &mut SequentialAccumulators,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let SequentialValidation { data, query, suite, predictive_sims, .. } = self.setup;
        let grid = &self.grid;
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
            // autocorrelation discrepancy.
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
            acc.priors.entry(mechanism.variable).or_default().push((atom.weight, prior));
            acc.posts.entry(mechanism.variable).or_default().push((atom.weight, post));
        }
        if suite == RefuteSuite::Full {
            let mut means = Vec::new();
            let mut sds = Vec::new();
            for &scale in grid.scales.iter() {
                let mut est = estimator.clone();
                est.prior_scale = scale;
                let (effect, _) = estimate_sustained_window(
                    &antecedent_estimate::SequentialProblem {
                        data,
                        graph: &atom.graph,
                        indexer: &atom.indexer,
                        estimand: &atom.estimand,
                    },
                    query,
                    antecedent_estimate::SequentialInference {
                        status: atom.status,
                        assumptions: atom.estimate.provenance.assumptions.clone(),
                        bootstrap_replicates: 0,
                        bayesian: Some(&est),
                    },
                    ctx,
                )?;
                means.push(effect.ate);
                sds.push(effect.se_analytic());
            }
            acc.sensitivity.push((
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
        Ok(())
    }
}

/// Mixes the per-atom reports, predictive checks, and prior sensitivity into the published results.
fn mix_sequential_results(
    acc: SequentialAccumulators,
    grid: &antecedent_validate::PriorSensitivity,
    original: f64,
    posterior: Option<&mut CausalPosterior>,
) -> SequentialValidationResults {
    let SequentialAccumulators { reports, diagnostics, priors, posts, sensitivity } = acc;
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
                let Some(mut report) = antecedent_validate::RefutationReport::mixture_weighted(
                    &atom_reports
                        .iter()
                        .map(|(weight, check)| (*weight, check))
                        .collect::<Vec<_>>(),
                ) else {
                    continue;
                };
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
    (mixed_reports, diagnostics, predictive)
}
