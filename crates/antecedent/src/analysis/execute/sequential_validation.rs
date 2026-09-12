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
        for report in ValidationSuite::reports_only(&outcomes) {
            reports.entry(Arc::clone(&report.refuter)).or_default().push((atom.weight, report));
        }
        if let Some(estimator) = bayes.filter(|_| suite != RefuteSuite::None) {
            for mechanism in &atom.mechanisms {
                let prior = PriorSet {
                    specs: vec![antecedent_prob::PriorSpec::GaussianCoefficients(
                        antecedent_prob::GaussianCoefficientPrior::isotropic(
                            mechanism.prepared.design.ncols,
                            estimator.prior_scale,
                        ),
                    )],
                    contrast: None,
                    categorical: Vec::new(),
                    restrictions: Vec::new(),
                };
                let prior = PriorPredictiveCheck {
                    n_sims: 200,
                    seed: ctx.rng.master_seed(),
                    ..PriorPredictiveCheck::new()
                }
                .check_with_prior(&mechanism.prepared, &prior, ctx)?;
                let post = PosteriorPredictiveCheck::new()
                    .check(&mechanism.prepared, &mechanism.posterior)?;
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
                        prior_scales: Arc::clone(&grid.scales),
                        alphas: Arc::from([]),
                        effect_means: means.into(),
                        effect_sds: sds.into(),
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
