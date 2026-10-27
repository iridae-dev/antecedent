//! Structured identity projections of study configuration.
//!
//! Every field that changes the science is mapped onto a scalar wire or a
//! [`PayloadDigestWire`] here, never through `Debug` or `Display`. Data-sized
//! vectors are digested once, when the study is built.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CanonicalWriter, CausalQuery, ObservationAssumption, ObservationSpec, VariableId, to_hex,
};
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

/// Exact query bytes for retained-target guards, using the artifact query codec.
/// Floating-point fields keep the wire's declared bit representation.
pub(crate) fn canonical_query_bytes(
    query: &CausalQuery,
) -> Result<std::sync::Arc<[u8]>, crate::CausalError> {
    let wire = antecedent_io::causal_query_to_wire(query).map_err(crate::CausalError::rejected)?;
    antecedent_io::to_cbor(&wire).map(std::sync::Arc::from).map_err(crate::CausalError::rejected)
}

/// Overlay the declared grouping inputs `$field` of `$cfg` (cluster labels, multiway
/// groups, panel times, population registry) onto `$wire` as digests; absent ones stay
/// `None`.
macro_rules! grouped {
    (@digest cluster_ids, $v:expr) => { $v.as_deref().map(cluster_ids) };
    (@digest multiway_ids, $v:expr) => { $v.as_deref().map(multiway_ids) };
    (@digest panel_times, $v:expr) => { $v.as_deref().map(panel_times) };
    (@digest population_registry, $v:expr) => { $v.as_ref().map(population_registry) };
    (@digest sampling_weights, $v:expr) => { $v.as_ref().map(sampling_weights) };
    ($cfg:expr, $wire:expr; $($field:ident),+ $(,)?) => {{
        let mut wire = $wire;
        $( wire.$field = grouped!(@digest $field, $cfg.$field); )+
        wire
    }};
}

/// Structured identity of a configured estimator. Called once per build.
#[allow(
    clippy::too_many_lines,
    reason = "projects each EstimatorSpec variant onto the EstimatorSpecWire variant that digests exactly that estimator's settings, so the arm list is the identity schema and each arm reads only its own variant's fields over the shared `base`"
)]
pub(crate) fn estimator_spec_identity(spec: &EstimatorSpec) -> EstimatorSpecWire {
    match spec {
        EstimatorSpec::Default(id) => EstimatorSpecWire::Default(id.as_str().into()),
        EstimatorSpec::LinearAdjustmentAte(cfg) => {
            EstimatorSpecWire::LinearAdjustmentAte(EstimatorConfigWire {
                bootstrap_interval: interval_key(cfg.inference.bootstrap_interval),
                se_kind: Some(se_kind(cfg.inference.se_kind)),
                fit_kind: Some(fit_kind(cfg.fit_kind)),
                cluster_ids: declared_clusters(
                    cfg.inference.cluster_ids.as_deref(),
                    cfg.inference.source_cluster_ids.as_deref(),
                ),
                multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::PropensityWeighting(cfg) => {
            EstimatorSpecWire::PropensityWeighting(EstimatorConfigWire {
                bootstrap_interval: interval_key(cfg.inference.bootstrap_interval),
                outcome: cfg.covariate_balance.then(|| "covariate_balanced_linear_hc3".into()),
                glm: Some(glm_options(&cfg.glm_options)),
                other_levels: other_levels_key(cfg.other_levels),
                cluster_ids: declared_clusters(
                    cfg.inference.cluster_ids.as_deref(),
                    cfg.inference.source_cluster_ids.as_deref(),
                ),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::PropensityMatching(cfg) => {
            EstimatorSpecWire::PropensityMatching(EstimatorConfigWire {
                outcome: cfg.population_variance.then(|| "population_linear_variance".into()),
                glm: Some(glm_options(&cfg.glm_options)),
                other_levels: other_levels_key(cfg.other_levels),
                se_kind: Some(se_kind(cfg.inference.se_kind)),
                cluster_ids: cfg.inference.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
                caliper_bits: cfg.caliper.map(f64::to_bits),
                caliper_scale: Some(
                    match cfg.caliper_scale {
                        CaliperScale::Logit => "logit",
                        CaliperScale::Raw => "raw",
                    }
                    .into(),
                ),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::PropensityStratification(cfg) => {
            EstimatorSpecWire::PropensityStratification(EstimatorConfigWire {
                bootstrap_interval: interval_key(cfg.inference.bootstrap_interval),
                glm: Some(glm_options(&cfg.glm_options)),
                other_levels: other_levels_key(cfg.other_levels),
                n_strata: Some(cfg.n_strata),
                outcome: cfg.regression_adjustment.then(|| "stratum_intercepts_linear_hc3".into()),
                cluster_ids: declared_clusters(
                    cfg.inference.cluster_ids.as_deref(),
                    cfg.inference.source_cluster_ids.as_deref(),
                ),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::DistanceMatching(cfg) => {
            EstimatorSpecWire::DistanceMatching(EstimatorConfigWire {
                outcome: cfg.population_variance.then(|| "population_linear_variance".into()),
                glm: Some(glm_options(&cfg.glm_options)),
                se_kind: Some(se_kind(cfg.inference.se_kind)),
                cluster_ids: cfg.inference.cluster_ids.as_deref().map(cluster_ids),
                multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
                caliper_bits: cfg.caliper.map(f64::to_bits),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::Aipw(cfg) => EstimatorSpecWire::Aipw(EstimatorConfigWire {
            bootstrap_interval: interval_key(cfg.inference.bootstrap_interval),
            glm: Some(glm_options(&cfg.glm_options)),
            other_levels: other_levels_key(cfg.other_levels),
            // The declared propensity nuisance (penalty grid, inner folds, fallback) changes
            // the science; the default is omitted so existing digests are unchanged.
            treatment_config: (!cfg.propensity.is_default())
                .then(|| cfg.propensity.canonical_key()),
            // The declared independence unit and minimum cluster count (the labels are
            // digested as `cluster_ids`, a dyadic unit's second-endpoint labels inside this
            // key); omitted by default.
            score: cfg.cluster_dml.as_ref().map(|spec| {
                cluster_dml_key(
                    spec,
                    cfg.cluster_ids_second.as_deref(),
                    cfg.source_cluster_ids_second.as_deref(),
                )
            }),
            se_kind: Some(se_kind(cfg.inference.se_kind)),
            cluster_ids: declared_clusters(
                cfg.inference.cluster_ids.as_deref(),
                cfg.inference.source_cluster_ids.as_deref(),
            ),
            multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
            ..grouped!(
                cfg,
                base(cfg.inference.bootstrap_replicates, cfg.overlap);
                population_registry, sampling_weights
            )
        }),
        EstimatorSpec::GlmAdjustment(cfg) => {
            EstimatorSpecWire::GlmAdjustment(EstimatorConfigWire {
                bootstrap_interval: interval_key(cfg.inference.bootstrap_interval),
                glm: Some(glm_options(&cfg.glm_options)),
                se_kind: Some(se_kind(cfg.inference.se_kind)),
                family: Some(glm_family(cfg.family).into()),
                cluster_ids: declared_clusters(
                    cfg.inference.cluster_ids.as_deref(),
                    cfg.inference.source_cluster_ids.as_deref(),
                ),
                multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
                panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
                ..grouped!(
                    cfg,
                    base(cfg.inference.bootstrap_replicates, cfg.overlap);
                    population_registry, sampling_weights
                )
            })
        }
        EstimatorSpec::FrontDoorTwoStage(cfg) => {
            EstimatorSpecWire::FrontDoorTwoStage(EstimatorConfigWire {
                se_kind: Some(se_kind(cfg.inference.se_kind)),
                cluster_ids: declared_clusters(
                    cfg.inference.cluster_ids.as_deref(),
                    cfg.inference.source_cluster_ids.as_deref(),
                ),
                ..base(cfg.inference.bootstrap_replicates, cfg.overlap)
            })
        }
        EstimatorSpec::IvWald(cfg) => EstimatorSpecWire::IvWald(EstimatorConfigWire {
            backend: "none".into(),
            se_kind: Some(se_kind(cfg.inference.se_kind)),
            cluster_ids: cfg.inference.cluster_ids.as_deref().map(cluster_ids),
            multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
            ..base(cfg.inference.bootstrap_replicates, cfg.overlap)
        }),
        EstimatorSpec::Iv2Sls(cfg) => EstimatorSpecWire::Iv2Sls(EstimatorConfigWire {
            se_kind: Some(se_kind(cfg.inference.se_kind)),
            cluster_ids: cfg.inference.cluster_ids.as_deref().map(cluster_ids),
            multiway_ids: cfg.inference.multiway_ids.as_deref().map(multiway_ids),
            panel_times: cfg.inference.panel_times.as_deref().map(panel_times),
            ..base(cfg.inference.bootstrap_replicates, cfg.overlap)
        }),
        EstimatorSpec::Dml(cfg) => EstimatorSpecWire::Dml(EstimatorConfigWire {
            folds: Some(u32::try_from(cfg.folds).unwrap_or(u32::MAX)),
            outcome: Some(cfg.outcome.name().into()),
            outcome_config: Some(learner_identity(cfg.outcome)),
            treatment: Some(cfg.treatment.name().into()),
            treatment_config: Some(learner_identity(cfg.treatment)),
            score: Some(match (cfg.score, cfg.units.spec.as_ref()) {
                (antecedent_estimate::DmlScore::Aipw, None) => "aipw".into(),
                (antecedent_estimate::DmlScore::PartiallyLinear, None) => "partially_linear".into(),
                // The unit declaration (and a dyad's second labels) change the science, so
                // they extend the score name only when declared.
                (score, Some(spec)) => format!(
                    "{};{}",
                    match score {
                        antecedent_estimate::DmlScore::Aipw => "aipw",
                        antecedent_estimate::DmlScore::PartiallyLinear => "partially_linear",
                    },
                    cluster_dml_key(
                        spec,
                        cfg.units.second.as_deref(),
                        cfg.units.second_source.as_deref(),
                    )
                ),
            }),
            other_levels: other_levels_key(cfg.other_levels),
            cluster_ids: declared_clusters(
                cfg.units.first.as_deref(),
                cfg.units.first_source.as_deref(),
            ),
            sampling_weights: cfg.sampling_weights.as_ref().map(sampling_weights),
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
            score: cfg.units.spec.as_ref().map(|spec| {
                cluster_dml_key(
                    spec,
                    cfg.units.second.as_deref(),
                    cfg.units.second_source.as_deref(),
                )
            }),
            cluster_ids: declared_clusters(
                cfg.units.first.as_deref(),
                cfg.units.first_source.as_deref(),
            ),
            sampling_weights: cfg.sampling_weights.as_ref().map(sampling_weights),
            ..base(0, cfg.overlap)
        }),
        EstimatorSpec::CausalForest(cfg) => EstimatorSpecWire::CausalForest(EstimatorConfigWire {
            n_trees: Some(u32::try_from(cfg.n_trees).unwrap_or(u32::MAX)),
            min_leaf: Some(u32::try_from(cfg.min_leaf).unwrap_or(u32::MAX)),
            max_depth: Some(u32::try_from(cfg.max_depth).unwrap_or(u32::MAX)),
            honesty: Some(cfg.honesty),
            // The unit declaration changes the subsample, the honest split and the standard
            // error, so it is part of the identity when declared.
            score: cfg.units.spec.as_ref().map(|spec| cluster_dml_key(spec, None, None)),
            cluster_ids: declared_clusters(
                cfg.units.first.as_deref(),
                cfg.units.first_source.as_deref(),
            ),
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
        sampling_weights: None,
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
        other_levels: None,
        bootstrap_interval: None,
    }
}

/// Identity key of a requested bootstrap interval; no request is omitted.
fn interval_key(request: Option<antecedent_estimate::IntervalRequest>) -> Option<String> {
    request.map(|r| {
        let bits = |value: Option<f64>| {
            value.map_or_else(|| "-".to_owned(), |v| format!("{:016x}", v.to_bits()))
        };
        format!(
            "level={:016x};wave={};min={};se={};half_width={}",
            r.level.to_bits(),
            r.wave_size,
            r.min_replicates,
            bits(r.se_rel_tol),
            bits(r.half_width_rel_tol)
        )
    })
}

/// Identity key of a declared exclusion; the default refusal is omitted.
fn other_levels_key(policy: antecedent_estimate::OtherLevels) -> Option<String> {
    match policy {
        antecedent_estimate::OtherLevels::Refuse => None,
        antecedent_estimate::OtherLevels::ExcludeUndeclared => Some("exclude_undeclared".into()),
    }
}

pub(crate) fn overlap_policy(policy: OverlapPolicy) -> OverlapPolicyWire {
    match policy.bounds() {
        None => OverlapPolicyWire::ExplicitOverride,
        Some((clip, trim)) => OverlapPolicyWire::RequireDiagnostics {
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

pub(crate) const fn glm_family(family: GlmFamily) -> &'static str {
    match family {
        GlmFamily::BinomialLogit => "binomial_logit",
        GlmFamily::BinomialProbit => "binomial_probit",
        GlmFamily::GaussianIdentity => "gaussian_identity",
        GlmFamily::PoissonLog => "poisson_log",
        GlmFamily::NegativeBinomial => "negative_binomial",
    }
}

pub(crate) fn glm_options(options: &GlmOptions) -> GlmOptionsWire {
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

/// Canonical key of a cluster-DML declaration, with the digest of a dyadic unit's
/// second-endpoint labels so that other second labels give another identity.
fn cluster_dml_key(
    spec: &antecedent_estimate::ClusterDml,
    second: Option<&[u32]>,
    second_source: Option<&[u32]>,
) -> String {
    let mut key = spec.canonical_key();
    if let Some(second) = second {
        key.push_str(";second_ids=");
        key.push_str(&to_hex(
            &PayloadDigestWire::u32s("estimator.cluster_ids_second", second).digest,
        ));
    } else if let Some(second) = second_source {
        key.push_str(";second_source_ids=");
        key.push_str(&to_hex(
            &PayloadDigestWire::u32s("estimator.source_cluster_ids_second", second).digest,
        ));
    }
    key
}

fn cluster_ids(ids: &[u32]) -> PayloadDigestWire {
    PayloadDigestWire::u32s("estimator.cluster_ids", ids)
}

/// The declared cluster labels as one digest. Labels per source row and labels per retained
/// row digest under different names, so the same numbers declared the other way are another
/// identity.
fn declared_clusters(
    retained: Option<&[u32]>,
    source: Option<&[u32]>,
) -> Option<PayloadDigestWire> {
    retained
        .map(cluster_ids)
        .or_else(|| source.map(|ids| PayloadDigestWire::u32s("estimator.source_cluster_ids", ids)))
}

/// The declared sampling weights of every source row, as the exact bits of each weight.
fn sampling_weights(weights: &antecedent_estimate::SamplingWeights) -> PayloadDigestWire {
    PayloadDigestWire::f64s("estimator.sampling_weights", weights.source())
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
    let mut stream = CanonicalWriter::new("antecedent.population.registry.v1");
    let mut entries = 0u64;
    for (name, rows) in registry.predicates() {
        entries += 1;
        stream.word(0).text(name).word(rows.len() as u64);
        for &row in rows {
            stream.word(row as u64);
        }
    }
    for (handle, weights, parents) in registry.distributions() {
        entries += 1;
        stream.word(1).word(u64::from(handle.raw())).reals(weights).word(parents.len() as u64);
        for parent in parents {
            stream.word(u64::from(parent.raw()));
        }
    }
    PayloadDigestWire { digest: stream.finish(), len: entries }
}

/// Backend, likelihood, draws, and the resolved prior contents.
pub(crate) fn bayesian_binding(
    cfg: &BayesianConfig,
    draw_controls: Option<crate::PosteriorDrawControls>,
) -> BayesianBindingWire {
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
        prior_source_snapshot: cfg.prior_artifact.as_deref().and_then(|bytes| {
            let artifact = antecedent_io::read_and_migrate(bytes).ok()?;
            let meta = antecedent_io::extract_prior_source_meta(&artifact).ok()??;
            let snapshot = meta.provenance.get("bayesian.verified_source_snapshot.v1")?;
            (snapshot.len() == 64 && snapshot.bytes().all(|byte| byte.is_ascii_hexdigit()))
                .then(|| snapshot.clone())
        }),
        prior_mapping: cfg.prior_mapping.as_ref().map(PriorMappingIdentityWire::from_mapping),
        external_compose: cfg.external_compose.as_deref().map(|compose| {
            external_compose_identity(
                &compose.sources,
                compose.conflict_policy.map(|policy| (policy.p_min, policy.kl_scale)),
            )
        }),
        draw_controls: draw_controls.map(|controls| controls.as_str().into()),
        design_model: cfg.design_model.as_ref().map(crate::DesignPosteriorModel::identity),
        white_innovation: cfg.white_innovation.map(|gate| {
            format!("ljung_box:max_lag={}:level_bits={:016x}", gate.max_lag, gate.level.to_bits())
        }),
        predictive_replicates: cfg
            .predictive_replicates
            .map(|bounds| format!("n_sims={}:max_rows={}", bounds.n_sims, bounds.max_rows)),
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
        observation_probability_floor_bits: options.observation_floor().to_bits(),
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
    let mut edges: Vec<(u64, u64, u64)> = spec
        .network
        .edges()
        .iter()
        .map(|edge| (edge.from.get(), edge.to.get(), edge.weight.to_bits()))
        .collect();
    edges.sort_unstable();
    let assignment: Vec<u8> = spec.assignment.iter().map(|&treated| u8::from(treated)).collect();
    InterferenceSnapshotWire {
        units: antecedent_io::DataPartitionIdentityWire {
            content: units.storage().content_digest(),
            row_count: units.row_count() as u64,
            unit_id: None,
            regularity: None,
        },
        edges: edges_digest(&edges),
        assignment: PayloadDigestWire::bytes("interference.assignment", &assignment),
    }
}

/// Digest of the sorted `(from, to, weight bits)` edge list of an interference network.
fn edges_digest(edges: &[(u64, u64, u64)]) -> PayloadDigestWire {
    let mut stream = CanonicalWriter::new("antecedent.interference.edges.v1");
    stream.word(edges.len() as u64);
    for &(from, to, weight_bits) in edges {
        stream.word(from).word(to).word(weight_bits);
    }
    PayloadDigestWire { digest: stream.finish(), len: edges.len() as u64 }
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

#[cfg(test)]
mod tests {
    use antecedent_estimate::{AipwAte, ClusterDml};

    use super::*;

    fn dyadic(first: &[u32], second: &[u32]) -> EstimatorSpecWire {
        let cfg = AipwAte::new()
            .with_cluster_dml(ClusterDml::dyadic(20, 3).unwrap())
            .with_cluster_ids(first.to_vec())
            .with_second_cluster_ids(second.to_vec());
        estimator_spec_identity(&EstimatorSpec::Aipw(Box::new(cfg)))
    }

    #[test]
    fn the_population_registry_digest_is_a_pinned_golden_vector() {
        let mut registry = antecedent_core::PopulationRegistry::new();
        registry.insert_predicate("treated_rows", vec![0_usize, 2, 3]);
        registry
            .insert_distribution(antecedent_core::DistributionRef::from_raw(7), vec![0.25, 0.75]);
        let wire = population_registry(&registry);
        assert_eq!(wire.len, 2);
        assert_eq!(
            to_hex(&wire.digest),
            "93e97abf1d9db165c9a30fead5afd73a01b090b76483c6aa9ade151e4d5aa0de"
        );
        let mut other = antecedent_core::PopulationRegistry::new();
        other.insert_predicate("treated_rows", vec![0_usize, 2, 4]);
        assert_ne!(population_registry(&other).digest, wire.digest);
    }

    #[test]
    fn the_interference_edge_digest_is_a_pinned_golden_vector() {
        let edges = [(0, 1, 1.0_f64.to_bits()), (1, 2, 0.5_f64.to_bits())];
        let wire = edges_digest(&edges);
        assert_eq!(wire.len, 2);
        assert_eq!(
            to_hex(&wire.digest),
            "658b99cf80f2cd0c065aa7fd9f8369c0a468f7ababc66d29bfdfd16d74d5578b"
        );
        assert_ne!(edges_digest(&edges[..1]).digest, wire.digest);
    }

    #[test]
    fn the_second_endpoint_labels_of_a_dyadic_unit_are_part_of_the_identity() {
        let first = [0, 0, 1, 1, 2, 2];
        let same = dyadic(&first, &[3, 4, 3, 5, 4, 5]);
        assert_eq!(same, dyadic(&first, &[3, 4, 3, 5, 4, 5]));
        assert_ne!(same, dyadic(&first, &[3, 4, 3, 5, 5, 4]));
    }

    #[test]
    fn a_declared_dyadic_unit_differs_from_the_same_labels_without_a_second_dimension() {
        let first = [0, 0, 1, 1, 2, 2];
        let without = AipwAte::new()
            .with_cluster_dml(ClusterDml::new(20).unwrap())
            .with_cluster_ids(first.to_vec());
        let without = estimator_spec_identity(&EstimatorSpec::Aipw(Box::new(without)));
        assert_ne!(without, dyadic(&first, &[3, 4, 3, 5, 4, 5]));
    }
}
