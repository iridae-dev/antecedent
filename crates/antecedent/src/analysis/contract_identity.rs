//! Structured identity projections of study configuration.
//!
//! Every field that changes the science is mapped onto a scalar wire or a
//! [`PayloadDigestWire`] here, never through `Debug` or `Display`. Data-sized
//! vectors are digested once, when the study is built.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, ObservationAssumption, ObservationSpec, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{
    AnalyticSeKind, CaliperScale, ContinuousResponseOptions, LinearFitKind,
    ObservationEstimatorOptions, OverlapPolicy, SelectedOutcomeCorrection,
};
use antecedent_io::{
    BayesianBindingWire, EstimatorConfigWire, EstimatorSpecWire, GlmOptionsWire,
    InterferenceSnapshotWire, ObservationOptionsWire, OverlapPolicyWire, PayloadDigestWire,
    PriorMappingIdentityWire, ResponseOptionsWire, SplitIdentityWire, external_compose_identity,
    prior_set_identity,
};
use antecedent_stats::{GlmFamily, GlmOptions, NbAlphaPolicy};

use crate::estimator_spec::EstimatorSpec;
use crate::inference::BayesianConfig;

/// Structured identity of a configured estimator. Called once per build.
pub(crate) fn estimator_spec_identity(spec: &EstimatorSpec) -> EstimatorSpecWire {
    match spec {
        EstimatorSpec::Default(id) => EstimatorSpecWire::Default(id.as_str().into()),
        EstimatorSpec::LinearAdjustmentAte(cfg) => {
            EstimatorSpecWire::LinearAdjustmentAte(EstimatorConfigWire {
                se_kind: Some(se_kind(cfg.se_kind)),
                fit_kind: Some(fit_kind(cfg.fit_kind)),
                cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.panel_times.as_deref().map(panel_times),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::PropensityWeighting(cfg) => {
            EstimatorSpecWire::PropensityWeighting(EstimatorConfigWire {
                glm: Some(glm_options(&cfg.glm_options)),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::PropensityMatching(cfg) => {
            EstimatorSpecWire::PropensityMatching(EstimatorConfigWire {
                glm: Some(glm_options(&cfg.glm_options)),
                se_kind: Some(se_kind(cfg.se_kind)),
                caliper_bits: cfg.caliper.map(f64::to_bits),
                caliper_scale: Some(
                    match cfg.caliper_scale {
                        CaliperScale::Logit => "logit",
                        CaliperScale::Raw => "raw",
                    }
                    .into(),
                ),
                cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.panel_times.as_deref().map(panel_times),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::PropensityStratification(cfg) => {
            EstimatorSpecWire::PropensityStratification(EstimatorConfigWire {
                glm: Some(glm_options(&cfg.glm_options)),
                n_strata: Some(cfg.n_strata),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::DistanceMatching(cfg) => {
            EstimatorSpecWire::DistanceMatching(EstimatorConfigWire {
                glm: Some(glm_options(&cfg.glm_options)),
                se_kind: Some(se_kind(cfg.se_kind)),
                caliper_bits: cfg.caliper.map(f64::to_bits),
                cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.panel_times.as_deref().map(panel_times),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::Aipw(cfg) => EstimatorSpecWire::Aipw(EstimatorConfigWire {
            glm: Some(glm_options(&cfg.glm_options)),
            se_kind: Some(se_kind(cfg.se_kind)),
            cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
            multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.panel_times.as_deref().map(panel_times),
            population_registry: cfg.population_registry.as_ref().map(population_registry),
            ..base(cfg.bootstrap_replicates, cfg.overlap)
        }),
        EstimatorSpec::GlmAdjustment(cfg) => {
            EstimatorSpecWire::GlmAdjustment(EstimatorConfigWire {
                glm: Some(glm_options(&cfg.glm_options)),
                se_kind: Some(se_kind(cfg.se_kind)),
                family: Some(glm_family(cfg.family).into()),
                cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.panel_times.as_deref().map(panel_times),
                population_registry: cfg.population_registry.as_ref().map(population_registry),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::FrontDoorTwoStage(cfg) => {
            EstimatorSpecWire::FrontDoorTwoStage(EstimatorConfigWire {
                se_kind: Some(se_kind(cfg.se_kind)),
                cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
                ..base(cfg.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::IvWald(cfg) => EstimatorSpecWire::IvWald(EstimatorConfigWire {
            backend: "none".into(),
            se_kind: Some(se_kind(cfg.se_kind)),
            cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
            multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.panel_times.as_deref().map(panel_times),
            ..base(cfg.bootstrap_replicates, cfg.overlap)
        }),
        EstimatorSpec::Iv2Sls(cfg) => EstimatorSpecWire::Iv2Sls(EstimatorConfigWire {
            se_kind: Some(se_kind(cfg.se_kind)),
            cluster_ids: cfg.cluster_ids.as_deref().map(cluster_ids),
            multiway_ids: cfg.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.panel_times.as_deref().map(panel_times),
            ..base(cfg.bootstrap_replicates, cfg.overlap)
        }),
        EstimatorSpec::Dml(cfg) => EstimatorSpecWire::Dml(EstimatorConfigWire {
            folds: Some(u32::try_from(cfg.folds).unwrap_or(u32::MAX)),
            outcome: Some(cfg.outcome.name().into()),
            outcome_config: Some(learner_identity(cfg.outcome)),
            treatment: Some(cfg.treatment.name().into()),
            treatment_config: Some(learner_identity(cfg.treatment)),
            score: Some(
                match cfg.score {
                    antecedent_estimate::DmlScore::Aipw => "aipw",
                    antecedent_estimate::DmlScore::PartiallyLinear => "partially_linear",
                }
                .into(),
            ),
            ..base(0, cfg.overlap)
        }),
        EstimatorSpec::DrLearner(cfg) => EstimatorSpecWire::DrLearner(EstimatorConfigWire {
            folds: Some(u32::try_from(cfg.folds).unwrap_or(u32::MAX)),
            outcome: Some(cfg.outcome.name().into()),
            outcome_config: Some(learner_identity(cfg.outcome)),
            treatment: Some(cfg.treatment.name().into()),
            treatment_config: Some(learner_identity(cfg.treatment)),
            final_learner: Some(cfg.final_learner.name().into()),
            final_learner_config: Some(learner_identity(cfg.final_learner)),
            ..base(0, cfg.overlap)
        }),
        EstimatorSpec::CausalForest(cfg) => EstimatorSpecWire::CausalForest(EstimatorConfigWire {
            n_trees: Some(u32::try_from(cfg.n_trees).unwrap_or(u32::MAX)),
            min_leaf: Some(u32::try_from(cfg.min_leaf).unwrap_or(u32::MAX)),
            max_depth: Some(u32::try_from(cfg.max_depth).unwrap_or(u32::MAX)),
            honesty: Some(cfg.honesty),
            ..base(0, antecedent_estimate::DmlAte::new().overlap)
        }),
    }
}

/// Fields every configured estimator carries; the dense-algebra backend is
/// the only one the facade builds (`FaerBackend`).
fn base(bootstrap_replicates: u32, overlap: OverlapPolicy) -> EstimatorConfigWire {
    EstimatorConfigWire {
        backend: "faer".into(),
        bootstrap_replicates,
        overlap: overlap_policy(overlap),
        glm: None,
        se_kind: None,
        caliper_bits: None,
        caliper_scale: None,
        n_strata: None,
        family: None,
        fit_kind: None,
        cluster_ids: None,
        multiway_ids: None,
        panel_times: None,
        population_registry: None,
        folds: None,
        outcome: None,
        treatment: None,
        score: None,
        final_learner: None,
        outcome_config: None,
        treatment_config: None,
        final_learner_config: None,
        n_trees: None,
        min_leaf: None,
        max_depth: None,
        honesty: None,
    }
}

fn overlap_policy(policy: OverlapPolicy) -> OverlapPolicyWire {
    match policy {
        OverlapPolicy::ExplicitOverride => OverlapPolicyWire::ExplicitOverride,
        OverlapPolicy::RequireDiagnostics { clip, trim } => OverlapPolicyWire::RequireDiagnostics {
            clip_bits: clip.map(f64::to_bits),
            trim_bits: trim.map(f64::to_bits),
        },
    }
}

fn se_kind(kind: AnalyticSeKind) -> String {
    kind.as_str().into()
}

fn fit_kind(kind: LinearFitKind) -> (String, Option<u64>) {
    match kind {
        LinearFitKind::Ols => ("ols".into(), None),
        LinearFitKind::Ridge { lambda } => ("ridge".into(), Some(lambda.to_bits())),
        LinearFitKind::Lasso { lambda } => ("lasso".into(), Some(lambda.to_bits())),
        LinearFitKind::Huber { c } => ("huber".into(), Some(c.to_bits())),
    }
}

const fn glm_family(family: GlmFamily) -> &'static str {
    match family {
        GlmFamily::BinomialLogit => "binomial_logit",
        GlmFamily::BinomialProbit => "binomial_probit",
        GlmFamily::GaussianIdentity => "gaussian_identity",
        GlmFamily::PoissonLog => "poisson_log",
        GlmFamily::NegativeBinomial => "negative_binomial",
    }
}

fn glm_options(options: &GlmOptions) -> GlmOptionsWire {
    GlmOptionsWire {
        max_iter: options.max_iter,
        tol_bits: options.tol.to_bits(),
        nb_alpha: match options.nb_alpha {
            NbAlphaPolicy::Fixed(alpha) => format!("fixed:{:016x}", alpha.to_bits()),
            NbAlphaPolicy::MethodOfMoments => "method_of_moments".into(),
            NbAlphaPolicy::NestedMle { max_outer, tol_alpha } => {
                format!("nested_mle:{max_outer}:{:016x}", tol_alpha.to_bits())
            }
        },
        ridge_on_separation_bits: options.ridge_on_separation.map(f64::to_bits),
    }
}

fn cluster_ids(ids: &[u32]) -> PayloadDigestWire {
    PayloadDigestWire::u32s("estimator.cluster_ids", ids)
}

fn multiway_ids(groups: &[Vec<u32>]) -> PayloadDigestWire {
    PayloadDigestWire::u32_groups("estimator.multiway_ids", groups)
}

fn panel_times(times: &[i64]) -> PayloadDigestWire {
    PayloadDigestWire::i64s("estimator.panel_times", times)
}

/// Registry predicates and distributions (with declared parents) as one digest.
pub(crate) fn population_registry(
    registry: &antecedent_core::PopulationRegistry,
) -> PayloadDigestWire {
    let mut bytes = Vec::new();
    let mut entries = 0u64;
    for (name, rows) in registry.predicates() {
        entries += 1;
        bytes.push(0);
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
        for &row in rows {
            bytes.extend_from_slice(&(row as u64).to_le_bytes());
        }
    }
    for (handle, weights, parents) in registry.distributions() {
        entries += 1;
        bytes.push(1);
        bytes.extend_from_slice(&handle.raw().to_le_bytes());
        bytes.extend_from_slice(&(weights.len() as u64).to_le_bytes());
        for weight in weights {
            bytes.extend_from_slice(&weight.to_bits().to_le_bytes());
        }
        bytes.extend_from_slice(&(parents.len() as u64).to_le_bytes());
        for parent in parents {
            bytes.extend_from_slice(&parent.raw().to_le_bytes());
        }
    }
    PayloadDigestWire {
        digest: antecedent_io::payload_digest("population.registry", &bytes),
        len: entries,
    }
}

/// Backend, likelihood, draws, and the resolved prior contents.
pub(crate) fn bayesian_binding(cfg: &BayesianConfig) -> BayesianBindingWire {
    BayesianBindingWire {
        backend: match cfg.backend {
            antecedent_estimate::BayesianBackendKind::ConjugateGaussian => "conjugate_gaussian",
            antecedent_estimate::BayesianBackendKind::Laplace => "laplace",
            antecedent_estimate::BayesianBackendKind::Hmc => "hmc",
        }
        .into(),
        likelihood: match cfg.likelihood {
            antecedent_prob::BayesLikelihood::GaussianIdentity => "gaussian_identity",
            antecedent_prob::BayesLikelihood::BernoulliLogit => "bernoulli_logit",
            antecedent_prob::BayesLikelihood::BernoulliProbit => "bernoulli_probit",
            antecedent_prob::BayesLikelihood::PoissonLog => "poisson_log",
        }
        .into(),
        n_draws: u64::try_from(cfg.n_draws).unwrap_or(u64::MAX),
        prior_scale_bits: (cfg.prior.is_none() && cfg.prior_artifact.is_none())
            .then(|| cfg.prior_scale.to_bits()),
        prior: cfg.prior.as_ref().map(prior_set_identity),
        prior_artifact: cfg
            .prior_artifact
            .as_deref()
            .map(|bytes| PayloadDigestWire::bytes("prior.artifact", bytes)),
        prior_mapping: cfg.prior_mapping.as_ref().map(PriorMappingIdentityWire::from_mapping),
        external_compose: cfg.external_compose.as_deref().map(|compose| {
            external_compose_identity(
                &compose.sources,
                compose.conflict_policy.map(|policy| (policy.p_min, policy.kl_scale)),
            )
        }),
    }
}

/// Every response-surface option that changes the estimate.
///
/// `export_row_diagnostics` only appends diagnostic channels (the estimate is
/// identical either way), so it is not an identity input.
pub(crate) fn response_options(options: &ContinuousResponseOptions) -> ResponseOptionsWire {
    ResponseOptionsWire {
        folds: options.folds as u64,
        nuisance_basis: options.nuisance_basis as u64,
        nuisance_lambda_bits: options.nuisance_lambda.to_bits(),
        bandwidth_bits: options.bandwidth.map(f64::to_bits),
        minimum_local_ess_bits: options.minimum_local_ess.to_bits(),
        confidence_level_bits: options.confidence_level.to_bits(),
        simultaneous_replicates: options.simultaneous_replicates,
        multiplier_seed: options.multiplier_seed,
    }
}

pub(crate) fn observation_options(options: &ObservationEstimatorOptions) -> ObservationOptionsWire {
    ObservationOptionsWire {
        selected_correction: match options.selected_correction {
            SelectedOutcomeCorrection::Ipw => "ipw",
            SelectedOutcomeCorrection::Aipw => "aipw",
        }
        .into(),
        observation_probability_floor_bits: options.observation_probability_floor.to_bits(),
        censoring_survival_floor_bits: options.censoring_survival_floor.to_bits(),
        crossfit_folds: options.crossfit_folds as u64,
    }
}

pub(crate) fn split(split: &antecedent_data::DiscoveryEstimationSplit) -> SplitIdentityWire {
    SplitIdentityWire {
        discovery: (split.discovery.start as u64, split.discovery.end as u64),
        estimation: (split.estimation.start as u64, split.estimation.end as u64),
        gap: split.gap as u64,
        series_len: split.series_len as u64,
    }
}

/// Observation-contract tags with their variable arguments.
///
/// Distinct from the human obligation labels: identity tags keep every
/// variable id, so `IndependentGiven(z)` and `IndependentGiven(w)` differ.
pub(crate) fn observation_identity_tags(
    query: &CausalQuery,
    delayed_entry: Option<VariableId>,
) -> Vec<String> {
    let mut tags = Vec::new();
    if let CausalQuery::Response(response) = query {
        for assumption in response.observation_assumptions.iter() {
            tags.push(match assumption {
                ObservationAssumption::IndependentGiven(vars) => {
                    format!("independent_given:{}", sorted_ids(vars))
                }
                ObservationAssumption::OutcomeIndependentGiven(vars) => {
                    format!("outcome_independent_given:{}", sorted_ids(vars))
                }
                ObservationAssumption::Structural(name) => format!("structural:{name}"),
            });
        }
        if let Some(spec) = observation_spec_identity_tag(&response.observation) {
            tags.push(spec);
        }
    }
    if let Some(entry) = delayed_entry {
        tags.push(format!("delayed_entry:{}", entry.raw()));
    }
    tags
}

fn sorted_ids(vars: &[VariableId]) -> String {
    let mut ids: Vec<u32> = vars.iter().map(|id| id.raw()).collect();
    ids.sort_unstable();
    ids.dedup();
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn optional_id(id: Option<VariableId>) -> String {
    id.map_or_else(|| "none".into(), |id| id.raw().to_string())
}

fn observation_spec_identity_tag(spec: &ObservationSpec) -> Option<String> {
    Some(match spec {
        ObservationSpec::Complete => return None,
        ObservationSpec::RightCensored { latent, observed, censoring, event } => format!(
            "right_censored:latent={},observed={},censoring={},event={}",
            latent.raw(),
            observed.raw(),
            censoring.raw(),
            event.raw()
        ),
        ObservationSpec::LeftCensored { latent, observed, censoring, event } => format!(
            "left_censored:latent={},observed={},censoring={},event={}",
            latent.raw(),
            observed.raw(),
            censoring.raw(),
            event.raw()
        ),
        ObservationSpec::IntervalCensored { latent, lower, upper } => format!(
            "interval_censored:latent={},lower={},upper={}",
            latent.raw(),
            lower.raw(),
            upper.raw()
        ),
        ObservationSpec::Truncated { latent, observed, lower, upper } => format!(
            "truncated:latent={},observed={},lower={},upper={}",
            latent.raw(),
            observed.raw(),
            optional_id(*lower),
            optional_id(*upper)
        ),
        ObservationSpec::Selected { latent, observed, indicator } => format!(
            "selected:latent={},observed={},indicator={}",
            latent.raw(),
            observed.raw(),
            indicator.raw()
        ),
    })
}

/// Fixed network and realized assignment of an interference study.
pub(crate) fn interference_snapshot(
    spec: &super::builder::InterferenceSpec,
) -> InterferenceSnapshotWire {
    let units = spec.network.units();
    let mut edges: Vec<(u32, u32, u64)> = spec
        .network
        .edges()
        .iter()
        .map(|edge| (edge.from, edge.to, edge.weight.to_bits()))
        .collect();
    edges.sort_unstable();
    let mut bytes = Vec::with_capacity(edges.len() * 16);
    for (from, to, weight) in &edges {
        bytes.extend_from_slice(&from.to_le_bytes());
        bytes.extend_from_slice(&to.to_le_bytes());
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    let assignment: Vec<u8> = spec.assignment.iter().map(|&treated| u8::from(treated)).collect();
    InterferenceSnapshotWire {
        units: antecedent_io::DataPartitionIdentityWire {
            content: units.storage().content_digest(),
            row_count: units.row_count() as u64,
            unit_id: None,
            regularity: None,
        },
        edges: PayloadDigestWire {
            digest: antecedent_io::payload_digest("interference.edges", &bytes),
            len: edges.len() as u64,
        },
        assignment: PayloadDigestWire::bytes("interference.assignment", &assignment),
    }
}

fn learner_identity(spec: antecedent_estimate::LearnerSpec) -> String {
    spec.identity()
}

#[cfg(test)]
mod learner_identity_tests {
    use super::*;
    #[test]
    fn learner_hyperparameters_change_contract_identity() {
        let a = EstimatorSpec::from(antecedent_estimate::DmlAte::new());
        let b = EstimatorSpec::from(antecedent_estimate::DmlAte::new().with_outcome(
            antecedent_estimate::LearnerSpec::Ridge(antecedent_estimate::RidgeSpec { lambda: 2.0 }),
        ));
        assert_ne!(estimator_spec_identity(&a), estimator_spec_identity(&b));
    }
}
