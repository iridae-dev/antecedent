//! Options every prepared entry point shares.
//!
//! One parser for the omitted-aware budget (`refute` / `bootstrap` /
//! `latency`), caller custom validators, population bindings, and the
//! execution controls used while preparing. An omitted budget field is never
//! written to the builder, so the builder's own omitted table, latency-tier
//! mapping, and refute downgrade apply. Unknown keys are refused rather than
//! ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{LatencyMode, RefuteSuite, StudyBuilder};
use antecedent_core::{PopulationRegistry, TargetPopulation};
use antecedent_validate::CustomEffectValidator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

const KEYS: &[&str] = &[
    "refute",
    "bootstrap",
    "latency",
    "validators",
    "target_population",
    "population_predicates",
    "population_distributions",
    "cancel",
    "on_progress",
    "inference",
    "discovery_algorithm",
];

/// Declared inference: mode plus the Bayesian prior and draw budget.
///
/// An absent mapping keeps the builder's Frequentist default. `n_draws` is
/// written only when the caller set it, so a latency tier can map the draw
/// budget; the prior scale is part of the declared prior and always sent.
#[derive(Default)]
pub(crate) struct InferenceSpec {
    pub mode: Option<String>,
    pub n_draws: Option<usize>,
    pub prior_scale: f64,
    pub prior_artifact: Option<Vec<u8>>,
    pub prior_mapping: Option<antecedent_io::PriorMapping>,
    pub composed_prior: Option<crate::prior_bank::OwnedComposedPrior>,
}

impl InferenceSpec {
    fn parse(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
        };
        Ok(Self {
            mode: item("mode")?.map(|value| value.extract()).transpose()?,
            n_draws: item("n_draws")?.map(|value| value.extract()).transpose()?,
            prior_scale: item("prior_scale")?
                .map(|value| value.extract())
                .transpose()?
                .ok_or_else(|| PyValueError::new_err("inference requires prior_scale"))?,
            prior_artifact: item("prior_artifact")?.map(|value| value.extract()).transpose()?,
            prior_mapping: item("prior_mapping")?
                .map(|value| crate::prior_bank::mapping_from_dict(value.cast::<PyDict>()?))
                .transpose()?,
            composed_prior: item("composed_prior")?
                .map(|value| {
                    crate::prior_bank::owned_composed_prior_from_dict(value.cast::<PyDict>()?)
                })
                .transpose()?,
        })
    }

    /// Whether this is a Bayesian declaration.
    pub(crate) fn is_bayesian(&self) -> bool {
        self.mode.as_deref().is_some_and(|mode| !mode.eq_ignore_ascii_case("frequentist"))
    }
}

/// Parsed shared prepare options.
#[derive(Default)]
pub(crate) struct PrepareOptions {
    pub refute: Option<RefuteSuite>,
    pub bootstrap: Option<u32>,
    pub latency: Option<LatencyMode>,
    pub validators: Vec<Arc<dyn CustomEffectValidator>>,
    pub target_population: Option<TargetPopulation>,
    pub registry: Option<PopulationRegistry>,
    pub cancel: Option<antecedent_core::CancellationToken>,
    pub progress: Option<Arc<dyn antecedent_core::ProgressSink>>,
    pub inference: InferenceSpec,
    /// Discovery algorithm of an accepted structure (provenance only).
    pub discovery_algorithm: Option<String>,
}

impl PrepareOptions {
    /// Parse the `options` mapping passed by `PreparedAnalysis.prepare`.
    pub(crate) fn parse(options: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let Some(dict) = options else {
            return Ok(Self::default());
        };
        for (key, _) in dict.iter() {
            let key: String = key.extract()?;
            if !KEYS.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown prepare option {key:?}; expected one of {KEYS:?}"
                )));
            }
        }
        let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
        };
        let refute =
            item("refute")?.map(|value| crate::suite_from_refute(Some(&value))).transpose()?;
        let bootstrap = item("bootstrap")?.map(|value| value.extract::<u32>()).transpose()?;
        let latency = item("latency")?
            .map(|value| {
                let text: String = value.extract()?;
                LatencyMode::parse(&text).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "unknown latency={text:?}; use interactive|standard|report"
                    ))
                })
            })
            .transpose()?;
        let validators = crate::callbacks::parse_validators(item("validators")?.as_ref())?;
        let target_population = match item("target_population")? {
            Some(value) => crate::ate_api::parse_target_population(Some(value.cast::<PyDict>()?))?,
            None => None,
        };
        let predicates = item("population_predicates")?;
        let distributions = item("population_distributions")?;
        let registry = crate::ate_api::parse_population_registry(
            predicates.as_ref().map(|value| value.cast::<PyDict>()).transpose()?,
            distributions.as_ref().map(|value| value.cast::<PyDict>()).transpose()?,
        )?;
        let cancel = item("cancel")?
            .map(|value| {
                value
                    .extract::<crate::PyCancellationToken>()
                    .map(|token| token.inner)
                    .map_err(PyErr::from)
            })
            .transpose()?;
        let progress = crate::callbacks::progress_sink_from_py(item("on_progress")?.as_ref())?;
        let inference = match item("inference")? {
            Some(value) => InferenceSpec::parse(value.cast::<PyDict>()?)?,
            None => InferenceSpec::default(),
        };
        let discovery_algorithm =
            item("discovery_algorithm")?.map(|value| value.extract()).transpose()?;
        Ok(Self {
            refute,
            bootstrap,
            latency,
            validators,
            target_population,
            registry,
            cancel,
            progress,
            inference,
            discovery_algorithm,
        })
    }

    /// Discovery algorithm of the accepted structure, when it came from discovery.
    pub(crate) fn discovery_algorithm(&self) -> Option<&str> {
        self.discovery_algorithm.as_deref()
    }

    /// Apply the declared inference, moving any prior transfer onto the builder.
    pub(crate) fn apply_inference(&mut self, builder: StudyBuilder) -> PyResult<StudyBuilder> {
        let spec = &mut self.inference;
        let artifact = spec.prior_artifact.take();
        crate::temporal_api::apply_temporal_inference_transfer(
            builder,
            spec.mode.as_deref(),
            spec.n_draws,
            spec.prior_scale,
            artifact.as_deref(),
            spec.prior_mapping.take(),
            spec.composed_prior.take(),
        )
    }

    /// Refuse prior transfer on a route whose executor cannot hydrate a prior.
    pub(crate) fn refuse_prior_transfer(&self, route: &str) -> PyResult<()> {
        let spec = &self.inference;
        let named: Vec<&str> = [
            ("prior_artifact", spec.prior_artifact.is_some()),
            ("prior_mapping", spec.prior_mapping.is_some()),
            ("composed_prior", spec.composed_prior.is_some()),
        ]
        .into_iter()
        .filter_map(|(name, set)| set.then_some(name))
        .collect();
        if named.is_empty() {
            return Ok(());
        }
        Err(crate::refusal(
            antecedent_core::reason_code!("prior_transfer_not_hydrated"),
            format!(
                "{route} fits its Bayesian model under the isotropic prior_scale and does not \
                 hydrate {}",
                named.join(", ")
            ),
        ))
    }

    /// Write only what the caller supplied onto the builder.
    pub(crate) fn apply(&self, mut builder: StudyBuilder) -> StudyBuilder {
        if let Some(suite) = self.refute {
            builder = builder.refute(suite);
        }
        if let Some(replicates) = self.bootstrap {
            builder = builder.bootstrap_replicates(replicates);
        }
        if let Some(mode) = self.latency {
            builder = builder.latency_mode(mode);
        }
        if !self.validators.is_empty() {
            builder = builder.custom_validators(self.validators.clone());
        }
        if let Some(registry) = &self.registry {
            builder = builder.population_registry(registry.clone());
        }
        builder
    }

    /// [`Self::apply`], except that a configured estimator owns the replicate count
    /// (it inherited [`Self::ambient_bootstrap`] when it was parsed).
    pub(crate) fn apply_budget_for(&self, builder: StudyBuilder, configured: bool) -> StudyBuilder {
        if !configured {
            return self.apply(builder);
        }
        let without_bootstrap = Self {
            refute: self.refute,
            bootstrap: None,
            latency: self.latency,
            validators: self.validators.clone(),
            target_population: None,
            registry: self.registry.clone(),
            cancel: None,
            progress: None,
            inference: InferenceSpec::default(),
            discovery_algorithm: None,
        };
        without_bootstrap.apply(builder)
    }

    /// Replicate count a configured estimator inherits when it names none.
    pub(crate) fn ambient_bootstrap(&self) -> u32 {
        self.bootstrap.unwrap_or_else(|| {
            self.latency.map_or(StudyBuilder::OMITTED_BOOTSTRAP, |mode| {
                antecedent::analysis::ResolvedLatencyBudget::from_mode(mode).bootstrap
            })
        })
    }

    /// Refuse a non-default target population on a route that cannot carry one.
    pub(crate) fn refuse_population(&self, route: &str) -> PyResult<()> {
        match &self.target_population {
            Some(population) if *population != TargetPopulation::AllObserved => {
                Err(crate::refusal(
                    antecedent_core::reason_code!("population_not_estimable"),
                    format!(
                        "{route} estimates the all-observed population; this route has no \
                         weighting or subpopulation estimator for another target"
                    ),
                ))
            }
            _ => Ok(()),
        }
    }

    /// Execution context for prepare-time identification.
    pub(crate) fn ctx(&self, seed: u64, threads: u32) -> antecedent_core::ExecutionContext {
        crate::py_execution_context_ext(
            seed,
            threads,
            self.cancel.clone(),
            self.progress.clone(),
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        )
    }
}
