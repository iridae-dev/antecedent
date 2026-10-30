//! Calibration key assembly and the claim envelope calibration view.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(
    clippy::wildcard_imports,
    reason = "the contract module is split by concern and these files share its imports and private items"
)]
use super::*;

/// Match the historical cluster constructions by their actual published reference.
/// The unweighted normal key's `iid` spelling is a legacy key convention; its `cluster`
/// SE kind still denotes clustered uncertainty. It does not assert independent errors.
pub(super) fn cluster_reference_key(
    result: &StudyResult,
    binding: &IntervalBinding,
    contract: &CausalContract,
) -> (&'static str, String, &'static str) {
    let ordinary = (binding.method.as_str(), contract.functional.to_string(), binding.dependence);
    if binding.method != IntervalMethod::AnalyticSe
        || binding.se_kind != Some(antecedent_estimate::AnalyticSeKind::Cluster)
        || result.logical_plan.data_classification
            != antecedent_contract::DataClassification::Tabular
    {
        return ordinary;
    }
    let student = result.estimate.as_effect().is_some_and(|estimate| {
        estimate.uncertainty.analytic.reference_df.is_some_and(|df| df.is_finite() && df > 0.0)
    });
    match (contract.functional.strip_suffix("+weights=sampling"), student) {
        (Some(functional), true) => (
            "student_analytic",
            format!("{functional}+reference=cluster_df+weights=sampling"),
            "cluster",
        ),
        (Some(_), false) | (None, true) => (ordinary.0, ordinary.1, "cluster"),
        (None, false) => ordinary,
    }
}

impl StudyResult {
    /// Calibration match bases of every interval this execution reported
    /// ([`StudyResult::reported_interval_bindings`]), primary first: the keys
    /// [`StudyResult::claim`] matches against the coverage records, and the
    /// keys a coverage test records.
    ///
    /// # Errors
    ///
    /// Structural mass totals that do not conserve.
    pub fn calibration_bases(
        &self,
        contract: &CausalContract,
    ) -> Result<Vec<antecedent_io::calibration::CalibrationBasisWire>, CausalError> {
        let identification = identification_slot_from_result(self)?;
        Ok(self.calibration_bases_with(contract, Some(&identification)))
    }

    pub(super) fn calibration_bases_with(
        &self,
        contract: &CausalContract,
        identification: Option<&IdentificationSlot>,
    ) -> Vec<antecedent_io::calibration::CalibrationBasisWire> {
        use antecedent_io::calibration::{
            CalibrationBasisWire, CalibrationInference, CalibrationKeyWire, CalibrationScopeWire,
            CalibrationStructure,
        };
        let coordinate = contract
            .reasoning
            .support
            .as_ref()
            .and_then(|slot| slot.matrix_coordinate.as_deref())
            .and_then(crate::support::support_cell_from_coordinate);
        let (query, graph_class, structure, inference) = match coordinate {
            Some(cell) => (
                cell.query.to_string(),
                cell.graph_class.to_string(),
                CalibrationStructure::from_axis(cell.structure.as_str()),
                cell.inference.to_string(),
            ),
            None => (
                contract.query_kind.to_string(),
                contract.graph_class.as_str().to_string(),
                if contract.structure_source == StructureSource::GraphPosterior {
                    CalibrationStructure::GraphPosterior
                } else {
                    CalibrationStructure::Fixed
                },
                CalibrationInference::from_label(&contract.inference).as_str().to_string(),
            ),
        };
        // The snapshot's modality, not one re-derived from the compiled plan's
        // data classification: a temporal plan is compiled as `Temporal` for
        // event data too, so re-deriving stamped `series` on an event
        // execution whose snapshot said `event`, and the claim then failed to
        // verify against its own body.
        let modality = contract.modality.as_ref();
        let (label, unidentified_mass) = identification_label(identification);
        let label = antecedent_io::calibration::identification_key(
            label,
            self.identification.derivation.steps.iter().map(|step| step.rule.as_ref()),
        );
        let bayesian = contract.inference.eq_ignore_ascii_case("bayesian");
        // The fields shared by every reported interval are built once; the
        // per-binding fields start empty (no allocation) and are filled below.
        // The contract support cell is the coordinate admitted for this program.
        // Prepared contracts require the complete execution identity. Cheap inspection
        // intentionally has no program identity, so require a stamped program plus every
        // identity it does carry, including the inference binding (which commits validation)
        // and the engine-stamped base data snapshot. The full execution snapshot continues
        // to bind any replay evidence. A caller-edited suite cannot select a different mode.
        let validation = match (self.executed_contract, coordinate) {
            (Some(executed), Some(cell))
                if (executed.identities == contract.identities
                    || (contract.identities.program.is_none()
                        && executed.identities.program.is_some()
                        && executed.identities.target == contract.identities.target
                        && executed.identities.identification
                            == contract.identities.identification
                        && executed.identities.inference_binding
                            == contract.identities.inference_binding
                        && executed.identities.observation == contract.identities.observation
                        && executed.inspection_data_snapshot
                            == contract.identities.data_snapshot))
                    && crate::support::validation_axis(executed.refute) == cell.validation =>
            {
                cell.validation.as_str()
            }
            // These two checked graphless routes refuse a validation suite at
            // preparation. Only an exact execution stamp and the licensed,
            // result-bound route coordinate establish that fact; a detached or
            // merely inspected result remains unknown.
            (Some(executed), None)
                if executed.identities == contract.identities
                    && executed.refute == crate::RefuteSuite::None
                    && contract.graph_class == GraphClass::RandomizedTrial
                    && contract.structure_source == StructureSource::RandomizedTrial
                    && contract.reasoning.support.as_ref().is_some_and(|slot| {
                        slot.matrix_status.as_ref() == "licensed"
                            && slot.matrix_coordinate.as_deref().is_some_and(|coordinate| {
                                match (&self.payload, &contract.body.query) {
                                    (ResultPayload::PanelDid(_), antecedent_io::CausalQueryWire::PanelDid(_)) =>
                                        coordinate.starts_with("graphless:difference_in_differences/"),
                                    (ResultPayload::PosteriorRow(row), antecedent_io::CausalQueryWire::RandomizedEffect(_)) =>
                                        row.randomized_ipow.is_some()
                                            && coordinate.starts_with("graphless:randomized_effect/")
                                            && coordinate.ends_with("/gaussian_ipow_modular_posterior/conditional_pseudo_posterior_pointwise_95_quantiles"),
                                    _ => false,
                                }
                            })
                    }) => "not_applicable",
            _ => "",
        };
        let (query, structure) = if validation == "not_applicable" {
            let query = match &self.payload {
                ResultPayload::PanelDid(_) => "PanelDid",
                ResultPayload::PosteriorRow(_) => "RandomizedEffect",
                _ => unreachable!("graphless validation was established only for these routes"),
            };
            (query.to_owned(), "design".to_owned())
        } else {
            (query, structure.as_str().to_string())
        };
        let shared = CalibrationKeyWire {
            validation: validation.to_owned(),
            query,
            graph_class,
            structure,
            modality: modality.to_string(),
            inference,
            estimator: self.logical_plan.estimator.as_deref().unwrap_or("").to_string(),
            interval_method: String::new(),
            se_kind: String::new(),
            dependence: String::new(),
            posterior: contract.posterior.to_string(),
            functional: String::new(),
            level: 0.0,
            identification: label.to_string(),
        };
        let row_count = analysis_row_count(self, contract);
        self.reported_interval_bindings(bayesian)
            .into_iter()
            .map(|binding| {
                let (interval_method, functional, dependence) =
                    cluster_reference_key(self, &binding, contract);
                CalibrationBasisWire {
                    key: CalibrationKeyWire {
                        interval_method: interval_method.to_string(),
                        se_kind: binding.se_kind.map_or("", |kind| kind.as_str()).to_string(),
                        dependence: dependence.to_string(),
                        functional,
                        level: if binding.level.is_finite() { binding.level } else { 0.0 },
                        ..shared.clone()
                    },
                    scope: CalibrationScopeWire {
                        row_count,
                        replicates_ok: binding.replicates_ok,
                        posterior_draws: binding.posterior_draws.or_else(|| {
                            (binding.method == IntervalMethod::PosteriorQuantile)
                                .then_some(contract.posterior_draws)
                                .flatten()
                        }),
                        unidentified_mass,
                    },
                }
            })
            .collect()
    }
}

/// `point` when every structural mass is identified under an identified
/// status; `partial` otherwise. Returns the non-identified mass beside it.
pub(super) fn identification_label(slot: Option<&IdentificationSlot>) -> (&'static str, f64) {
    let Some(slot) = slot else {
        return ("partial", 1.0);
    };
    let unidentified = slot.unidentified_mass + slot.unevaluable_mass + slot.incomplete_search_mass;
    let identified_status = matches!(
        slot.status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    );
    if identified_status && unidentified <= 1e-12 {
        ("point", unidentified)
    } else {
        ("partial", unidentified)
    }
}

/// Functional label of `query` for the calibration key: what the support
/// axis query name leaves open and a coverage measurement depends on.
pub(super) fn functional_label(query: &CausalQuery) -> String {
    fn population(population: &TargetPopulation) -> &'static str {
        match population {
            TargetPopulation::AllObserved => "all_observed",
            TargetPopulation::Treated => "treated",
            TargetPopulation::Untreated => "untreated",
            TargetPopulation::Environment(_) => "environment",
            TargetPopulation::Predicate(_) => "predicate",
            TargetPopulation::CustomDistribution(_) => "custom_distribution",
            TargetPopulation::RowWeights { .. } => "row_weights",
            TargetPopulation::LocalAtCutoff { .. } => "local_at_cutoff",
            _ => "other_population",
        }
    }
    fn outcome(functional: &antecedent_core::OutcomeFunctional) -> String {
        match functional {
            antecedent_core::OutcomeFunctional::Mean => "mean".into(),
            antecedent_core::OutcomeFunctional::Exceedance(_) => "exceedance".into(),
            antecedent_core::OutcomeFunctional::ExceedanceGrid(_) => "exceedance_grid".into(),
            antecedent_core::OutcomeFunctional::Quantile(tau) => {
                format!("quantile:{}", tau.to_f64())
            }
            _ => "other_functional".into(),
        }
    }
    fn policy(policy: &antecedent_core::TemporalPolicy) -> String {
        match policy {
            antecedent_core::TemporalPolicy::Pulse { .. } => "pulse".into(),
            antecedent_core::TemporalPolicy::Sustained { from, until } => {
                format!("sustained:{}", i64::from(*until) - i64::from(*from) + 1)
            }
            antecedent_core::TemporalPolicy::Dynamic { active_at, .. } => {
                format!("dynamic:{}", active_at.len())
            }
            _ => "other_policy".into(),
        }
    }
    match query {
        CausalQuery::AverageEffect(q) => {
            format!("{}.{}", population(&q.target_population), outcome(&q.outcome_functional))
        }
        CausalQuery::ConditionalEffect(q) => format!(
            "{}.{}",
            population(&q.inner.target_population),
            outcome(&q.inner.outcome_functional)
        ),
        CausalQuery::Mediation(q) => {
            let contrast = match q.contrast {
                antecedent_core::MediationContrast::Total => "total",
                antecedent_core::MediationContrast::Direct => "direct",
                antecedent_core::MediationContrast::Mediated => "mediated",
                antecedent_core::MediationContrast::NaturalDirect => "natural_direct",
                antecedent_core::MediationContrast::NaturalIndirect => "natural_indirect",
            };
            format!("{contrast}.{}", population(&q.target_population))
        }
        CausalQuery::NestedCounterfactual(_) => "natural_direct_shared_exogenous".into(),
        CausalQuery::TemporalEffect(q) => format!(
            "{}.h{}.{}",
            policy(&q.policy),
            q.horizon_steps,
            population(&q.target_population)
        ),
        CausalQuery::Response(q) => {
            let observation = match q.observation {
                antecedent_core::ObservationSpec::Complete => "complete",
                _ => "observation_adjusted",
            };
            let temporal = q.temporal.as_ref().map_or_else(String::new, |spec| {
                let horizons: Vec<String> = spec.horizons.iter().map(u32::to_string).collect();
                format!(".{}.h{}", policy(&spec.policy), horizons.join(","))
            });
            format!(
                "{}.{}.{observation}{temporal}",
                population(&q.target_population),
                outcome(&q.outcome_functional)
            )
        }
        CausalQuery::Distribution(q) => population(&q.target_population).into(),
        CausalQuery::PathSpecific(q) => population(&q.target_population).into(),
        CausalQuery::Discover(q) => {
            format!("discover.{}.{}.{}", q.method.id(), q.output().as_str(), q.inference().as_str())
        }
        CausalQuery::Attribute(q) => format!("attribute.{}", q.target.as_str()),
        CausalQuery::Simulate(q) => format!("simulate.{}", q.target.as_str()),
        CausalQuery::TemporalDesign(q) => format!("temporal_design.{}", q.design_name()),
        CausalQuery::IvBounds(_) => "iv_bounds.binary_instrument".into(),
        CausalQuery::Association(q) => format!("association.{}", q.functional.as_str()),
        _ => String::new(),
    }
}

/// Calibration label of a non-default propensity overlap policy.
///
/// A propensity-score estimator (weighting, matching, stratification,
/// distance matching, AIPW) clips propensities into `[0.01, 0.99]` and trims
/// nothing by default, and every coverage record for those estimators was
/// measured there. Any other clip or trim is a different construction (a trim
/// also changes the population the interval covers), so it is appended to the
/// functional label and binds only to records measured under it. `None` at the
/// default policy and for estimators that fit no propensity model.
pub(super) fn overlap_label(study: &Study, resolved_estimator: Option<&str>) -> Option<String> {
    let configured = study
        .declared
        .estimator_spec
        .as_deref()
        .and_then(crate::estimator_spec::EstimatorSpec::propensity_overlap);
    let policy = if let Some(policy) = configured {
        policy
    } else {
        let estimator = study.declared.estimator.map(|id| id.as_str()).or(resolved_estimator)?;
        if !matches!(
            estimator,
            "propensity.weighting"
                | "propensity.matching"
                | "propensity.stratification"
                | "distance.matching"
                | "aipw"
        ) {
            return None;
        }
        study.declared.overlap_policy?
    };
    let policy = policy.canonical();
    if policy == antecedent_estimate::default_propensity_overlap().canonical() {
        return None;
    }
    Some(match policy.bounds() {
        None => "overlap=explicit_override".into(),
        Some((clip, trim)) => {
            let bound = |value: Option<f64>| value.map_or_else(|| "none".into(), |v| v.to_string());
            format!("overlap=clip:{},trim:{}", bound(clip), bound(trim))
        }
    })
}

/// Posterior construction label: backend, likelihood and prior.
pub(super) fn posterior_label(inference: &InferenceMode, rank_interval: bool) -> String {
    let InferenceMode::Bayesian(cfg) = inference else {
        return String::new();
    };
    let backend = match cfg.backend {
        antecedent_estimate::BayesianBackendKind::ConjugateGaussian => "conjugate_gaussian",
        antecedent_estimate::BayesianBackendKind::Laplace => "laplace",
        antecedent_estimate::BayesianBackendKind::Hmc => "hmc",
    };
    let likelihood = match cfg.likelihood {
        antecedent_prob::BayesLikelihood::GaussianIdentity => "gaussian_identity",
        antecedent_prob::BayesLikelihood::BernoulliLogit => "bernoulli_logit",
        antecedent_prob::BayesLikelihood::BernoulliProbit => "bernoulli_probit",
        antecedent_prob::BayesLikelihood::PoissonLog => "poisson_log",
    };
    let prior =
        if cfg.prior.is_some() || cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
            "prior=supplied".to_string()
        } else {
            format!("prior_scale={}", cfg.prior_scale)
        };
    // Below the measured draw count the g-computation's effect interval is the exchangeable-rank
    // interval of its draws, not the type-7 interval its 1,000-draw records measured: another
    // construction, keyed apart. The default count and above keep the original label.
    let interval = if rank_interval
        && matches!(cfg.backend, antecedent_estimate::BayesianBackendKind::Laplace)
        && cfg.n_draws < antecedent_estimate::LAPLACE_RANK_INTERVAL_BELOW_DRAWS
    {
        ".rank_quantile_interval"
    } else {
        ""
    };
    format!("{backend}.{likelihood}.{prior}{interval}")
}

/// Project a computed slot onto the claim envelope's view.
pub(super) fn calibration_view(slot: &CalibrationSlotWire) -> CalibrationView {
    CalibrationView {
        status: Arc::from(slot.status.as_str()),
        record_id: slot.record_id.as_deref().map(Arc::from),
        reason: slot.reason.as_deref().map(Arc::from),
        scope_n: slot.scope_n,
        scope_dependence: slot.scope_dependence.as_deref().map(Arc::from),
        calibration_sha: slot.calibration_sha.as_deref().map(Arc::from),
        scope_n_max: slot.scope_n_max,
        nominal: slot.nominal,
        observed: slot.observed,
        basis: slot.basis.as_ref().map(|basis| {
            let key = &basis.key;
            antecedent_core::CalibrationBasis::new(
                [
                    Arc::from(key.query.as_str()),
                    Arc::from(key.graph_class.as_str()),
                    Arc::from(key.structure.as_str()),
                    Arc::from(key.modality.as_str()),
                    Arc::from(key.inference.as_str()),
                    Arc::from(key.estimator.as_str()),
                    Arc::from(key.interval_method.as_str()),
                    Arc::from(key.se_kind.as_str()),
                    Arc::from(key.dependence.as_str()),
                    Arc::from(key.posterior.as_str()),
                    Arc::from(key.functional.as_str()),
                ],
                key.level,
                Arc::from(key.identification.as_str()),
                basis.scope.row_count,
                basis.scope.replicates_ok,
                basis.scope.posterior_draws,
                basis.scope.unidentified_mass,
            )
            .with_validation(Arc::<str>::from(key.validation.as_str()))
        }),
        secondary: slot.secondary.iter().map(calibration_view).collect(),
    }
}

/// The inverse of [`calibration_view`]: the claim section stores the slot the
/// claim computed, basis included, so the consumer re-derives it.
pub(super) fn calibration_wire(view: &CalibrationView) -> CalibrationSlotWire {
    CalibrationSlotWire {
        status: view.status.as_ref().into(),
        record_id: view.record_id.as_ref().map(ToString::to_string),
        reason: view.reason.as_ref().map(ToString::to_string),
        scope_n: view.scope_n,
        scope_dependence: view.scope_dependence.as_ref().map(ToString::to_string),
        calibration_sha: view.calibration_sha.as_ref().map(ToString::to_string),
        scope_n_max: view.scope_n_max,
        nominal: view.nominal,
        observed: view.observed,
        basis: view.basis.as_ref().map(antecedent_io::calibration::CalibrationBasisWire::from),
        secondary: view.secondary.iter().map(calibration_wire).collect(),
    }
}

pub(super) fn overlap_policy_tag(policy: antecedent_estimate::OverlapPolicy) -> String {
    match policy.bounds() {
        None => "explicit_override".into(),
        Some((clip, trim)) => {
            format!(
                "require_diagnostics:clip={}:trim={}",
                clip.map_or_else(|| "none".into(), ordered_float_bits),
                trim.map_or_else(|| "none".into(), ordered_float_bits),
            )
        }
    }
}

pub(super) fn ordered_float_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}
