//! Python surface for four committed batch-6 contracts.
//!
//! Retention `require` refusals, rewrite-legality checks, summary-observation
//! adaptation, and feature-version reads. A refusal is a Python exception.
//! Checking a rewrite does not execute it. Adapting a summary does not form a
//! posterior. Classifying a feature does not decode a container.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use antecedent_contract::rewrite_legality::{
    ColumnFilterPushdown, NuisanceEffect, NuisanceReuse, PreparationEffect, PreservationAxis,
    PreservedIdentities, PushdownEffect, RewriteKind, RewriteRefusal, SharedRowDesignPreparation,
    SufficientStatistics, SummaryEffect, TileEffect, TiledEvaluation,
};
use antecedent_contract::summary_observation::{
    Dependence, FrequentistSummary, LikelihoodUpdateAdmission, Origin, SamplingApproximation,
    Selection, SummaryObservationError, SummaryObservationModel, SummaryUse,
};
use antecedent_core::reason_code;
use antecedent_io_core::feature_version::{
    FeatureRead, FeatureRefusal, FeatureVersion, RequestedVersion, classify_feature_read,
};
use antecedent_io_core::{FormatVersion, SemanticVersion};
use antecedent_state::retention::{
    RetainedOperation, RetentionMode, RetentionPolicy, RetentionRefusal,
};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::{CausalStateError, CausalUnsupportedError};

fn retention_err(refusal: RetentionRefusal) -> PyErr {
    let mode = refusal.mode().as_str();
    let operation = refusal.operation().as_str();
    let err = CausalStateError::new_err(refusal.to_string());
    Python::attach(|py| {
        let value = err.value(py);
        let _ = value.setattr("mode", mode);
        let _ = value.setattr("operation", operation);
    });
    err
}

fn rewrite_err(refusal: RewriteRefusal) -> PyErr {
    let detail = refusal.detail();
    let axis = refusal.axis().as_str();
    let next = refusal.next_attempt();
    let kind = refusal.kind();
    let err = crate::value_err_code(refusal.code(), refusal.to_string());
    Python::attach(|py| {
        let value = err.value(py);
        let _ = value.setattr("detail", detail);
        let _ = value.setattr("axis", axis);
        let _ = value.setattr("next_attempt", next.as_str());
        let _ = value.setattr("remedy", next.sentence());
        if let Some(kind) = kind {
            let _ = value.setattr("kind", kind);
        }
    });
    err
}

fn summary_err(error: SummaryObservationError) -> PyErr {
    let code = error.code();
    let detail = error.detail();
    let next = error.next_attempt();
    let kind = error.summary_kind();
    let err = if code == reason_code!("invalid_argument") {
        crate::value_err_code(code, error.to_string())
    } else {
        crate::refusal(code, error.to_string())
    };
    Python::attach(|py| {
        let value = err.value(py);
        let _ = value.setattr("detail", detail);
        if let Some(next) = next {
            let _ = value.setattr("next_attempt", next.as_str());
            let _ = value.setattr("remedy", next.phrase());
        }
        if let Some(kind) = kind {
            let _ = value.setattr("summary_kind", kind.as_str());
        }
    });
    err
}

fn feature_err(refusal: FeatureRefusal) -> PyErr {
    crate::unsupported_message(refusal.to_string())
}

fn unknown_effect(family: &str, name: &str) -> PyErr {
    crate::value_err_code(
        reason_code!("invalid_argument"),
        format!("unknown {family} effect `{name}`"),
    )
}

fn unknown_name(what: &str, name: &str) -> PyErr {
    crate::value_err_code(reason_code!("invalid_argument"), format!("unknown {what} `{name}`"))
}

/// `FeatureVersion::name` is `'static`. Each distinct Python name is owned once.
fn intern_feature_name(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let table = NAMES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = table.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = guard.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    guard.insert(leaked.to_owned(), leaked);
    leaked
}

fn pushdown_effect(name: &str) -> Option<PushdownEffect> {
    Some(match name {
        "declared" => PushdownEffect::Declared,
        "omits_required_column" => PushdownEffect::OmitsRequiredColumn,
        "reassigns_folds" => PushdownEffect::ReassignsFolds,
        "undeclared_population_cut" => PushdownEffect::UndeclaredPopulationCut,
        "cuts_history" => PushdownEffect::CutsHistory,
        "drops_dependence_factor" => PushdownEffect::DropsDependenceFactor,
        "drops_uncertainty" => PushdownEffect::DropsUncertainty,
        _ => return None,
    })
}

fn preparation_effect(name: &str) -> Option<PreparationEffect> {
    Some(match name {
        "shared" => PreparationEffect::Shared,
        "mixes_samples" => PreparationEffect::MixesSamples,
        "mixes_folds" => PreparationEffect::MixesFolds,
        "mixes_populations" => PreparationEffect::MixesPopulations,
        "mixes_histories" => PreparationEffect::MixesHistories,
        "mixes_dependence" => PreparationEffect::MixesDependence,
        "mixes_uncertainty" => PreparationEffect::MixesUncertainty,
        _ => return None,
    })
}

fn nuisance_effect(name: &str) -> Option<NuisanceEffect> {
    Some(match name {
        "certified" => NuisanceEffect::Certified,
        "changes_estimation_sample" => NuisanceEffect::ChangesEstimationSample,
        "trains_on_evaluation_fold" => NuisanceEffect::TrainsOnEvaluationFold,
        "fit_on_other_population" => NuisanceEffect::FitOnOtherPopulation,
        "fit_on_other_history" => NuisanceEffect::FitOnOtherHistory,
        "drops_shared_factor" => NuisanceEffect::DropsSharedFactor,
        "replaces_uncertainty" => NuisanceEffect::ReplacesUncertainty,
        _ => return None,
    })
}

fn statistics_effect(name: &str) -> Option<SummaryEffect> {
    Some(match name {
        "sufficient" => SummaryEffect::Sufficient,
        "evicts_row" => SummaryEffect::EvictsRow,
        "pools_folds" => SummaryEffect::PoolsFolds,
        "marginalizes_population" => SummaryEffect::MarginalizesPopulation,
        "collapses_history" => SummaryEffect::CollapsesHistory,
        "drops_cluster" => SummaryEffect::DropsCluster,
        "drops_sampling_law" => SummaryEffect::DropsSamplingLaw,
        _ => return None,
    })
}

fn tile_effect(name: &str) -> Option<TileEffect> {
    Some(match name {
        "faithful" => TileEffect::Faithful,
        "drops_rows" => TileEffect::DropsRows,
        "splits_fold" => TileEffect::SplitsFold,
        "retargets_population" => TileEffect::RetargetsPopulation,
        "cuts_horizon" => TileEffect::CutsHorizon,
        "drops_cross_terms" => TileEffect::DropsCrossTerms,
        "drops_failed_draws" => TileEffect::DropsFailedDraws,
        _ => return None,
    })
}

fn adapt_summary(
    summary: FrequentistSummary,
    sampling_approximation: &str,
    selection: &str,
    dependence: &str,
    origin: &str,
) -> Result<SummaryObservationModel, SummaryObservationError> {
    SummaryObservationModel::try_adapt(
        summary,
        SamplingApproximation::new(sampling_approximation)?,
        Selection::new(selection)?,
        Dependence::new(dependence)?,
        Origin::new(origin)?,
    )
}

fn read_feature(
    container: FormatVersion,
    name: &str,
    version: u32,
    accepted: &[FeatureVersion],
) -> Result<FeatureRead, FeatureRefusal> {
    classify_feature_read(container, RequestedVersion::Feature { name, version }, accepted)
}

fn read_package(
    container: FormatVersion,
    package: &SemanticVersion,
    accepted: &[FeatureVersion],
) -> Result<FeatureRead, FeatureRefusal> {
    classify_feature_read(container, RequestedVersion::Package(package), accepted)
}

fn accepted_features(
    py: Python<'_>,
    accepted: &[Py<PyFeatureVersion>],
) -> Vec<FeatureVersion> {
    accepted
        .iter()
        .map(|item| {
            let view = item.borrow(py);
            FeatureVersion { name: intern_feature_name(&view.name), version: view.version }
        })
        .collect()
}

#[pyclass(name = "RetentionPolicy", frozen)]
struct PyRetentionPolicy {
    inner: RetentionPolicy,
}

#[pymethods]
impl PyRetentionPolicy {
    #[classmethod]
    fn raw_history(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionPolicy::RawHistory }
    }

    #[classmethod]
    fn bounded_window(_cls: &Bound<'_, PyType>, max_rows: u64) -> Self {
        Self { inner: RetentionPolicy::BoundedWindow { max_rows } }
    }

    #[classmethod]
    fn sufficient_statistics(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionPolicy::SufficientStatisticsOnly }
    }

    fn kind(&self) -> &'static str {
        match self.inner {
            RetentionPolicy::RawHistory => "raw_history",
            RetentionPolicy::BoundedWindow { .. } => "bounded_window",
            RetentionPolicy::SufficientStatisticsOnly => "sufficient_statistics",
        }
    }

    fn result_mode(&self) -> PyRetentionMode {
        PyRetentionMode { inner: self.inner.result_mode() }
    }

    fn max_retained_rows(&self) -> Option<u64> {
        self.inner.max_retained_rows()
    }

    fn no_longer_serves(&self) -> Vec<PyRetainedOperation> {
        self.inner.no_longer_serves().iter().copied().map(PyRetainedOperation::new).collect()
    }

    /// Accept `operation` under this policy's result mode, or raise.
    fn require(&self, py: Python<'_>, operation: Py<PyRetainedOperation>) -> PyResult<()> {
        self.inner.require(operation.borrow(py).inner).map_err(retention_err)
    }
}

#[pyclass(name = "RetentionMode", frozen)]
struct PyRetentionMode {
    inner: RetentionMode,
}

#[pymethods]
impl PyRetentionMode {
    #[classmethod]
    fn full_reusable(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionMode::FullReusable }
    }

    #[classmethod]
    fn summary_only(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionMode::SummaryOnly }
    }

    #[classmethod]
    fn disk_backed(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionMode::DiskBacked }
    }

    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        Self { inner: RetentionMode::default() }
    }

    fn as_str(&self) -> &'static str {
        self.inner.as_str()
    }

    fn serves(&self, py: Python<'_>, operation: Py<PyRetainedOperation>) -> bool {
        self.inner.serves(operation.borrow(py).inner)
    }

    fn no_longer_serves(&self) -> Vec<PyRetainedOperation> {
        self.inner.no_longer_serves().iter().copied().map(PyRetainedOperation::new).collect()
    }

    /// Accept `operation`, or raise. Nothing is written and nothing is spilled.
    fn require(&self, py: Python<'_>, operation: Py<PyRetainedOperation>) -> PyResult<()> {
        self.inner.require(operation.borrow(py).inner).map_err(retention_err)
    }
}

#[pyclass(name = "RetainedOperation", frozen)]
struct PyRetainedOperation {
    inner: RetainedOperation,
}

impl PyRetainedOperation {
    fn new(inner: RetainedOperation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRetainedOperation {
    #[classmethod]
    fn row_reconstruction(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::RowReconstruction)
    }

    #[classmethod]
    fn refit_from_retained_rows(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::RefitFromRetainedRows)
    }

    #[classmethod]
    fn exact_row_eviction(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::ExactRowEviction)
    }

    #[classmethod]
    fn in_place_column_insert(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::InPlaceColumnInsert)
    }

    #[classmethod]
    fn in_place_weight_or_mask_update(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::InPlaceWeightOrMaskUpdate)
    }

    #[classmethod]
    fn reuse_full_state(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::ReuseFullState)
    }

    #[classmethod]
    fn reuse_summary(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::ReuseSummary)
    }

    #[classmethod]
    fn read_disk_artifact(_cls: &Bound<'_, PyType>) -> Self {
        Self::new(RetainedOperation::ReadDiskArtifact)
    }

    fn as_str(&self) -> &'static str {
        self.inner.as_str()
    }
}

#[pyclass(name = "PreservedIdentities", frozen)]
struct PyPreservedIdentities {
    inner: PreservedIdentities,
}

#[pymethods]
impl PyPreservedIdentities {
    #[new]
    fn new(
        sample_selection: &str,
        folds: &str,
        population: &str,
        history: &str,
        dependence: &str,
        uncertainty: &str,
    ) -> PyResult<Self> {
        PreservedIdentities::try_new(
            sample_selection,
            folds,
            population,
            history,
            dependence,
            uncertainty,
        )
        .map(|inner| Self { inner })
        .map_err(rewrite_err)
    }

    fn identity(&self, axis: &str) -> PyResult<String> {
        let axis = PreservationAxis::from_name(axis)
            .ok_or_else(|| unknown_name("preservation axis", axis))?;
        Ok(self.inner.identity(axis).to_owned())
    }
}

enum HeldRewrite {
    Pushdown(Box<ColumnFilterPushdown>),
    Preparation(Box<SharedRowDesignPreparation>),
    Nuisance(Box<NuisanceReuse>),
    Statistics(Box<SufficientStatistics>),
    Tile(Box<TiledEvaluation>),
}

impl HeldRewrite {
    fn check(&self) -> Result<(), RewriteRefusal> {
        match self {
            Self::Pushdown(rewrite) => rewrite.check(),
            Self::Preparation(rewrite) => rewrite.check(),
            Self::Nuisance(rewrite) => rewrite.check(),
            Self::Statistics(rewrite) => rewrite.check(),
            Self::Tile(rewrite) => rewrite.check(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Pushdown(rewrite) => rewrite.name(),
            Self::Preparation(rewrite) => rewrite.name(),
            Self::Nuisance(rewrite) => rewrite.name(),
            Self::Statistics(rewrite) => rewrite.name(),
            Self::Tile(rewrite) => rewrite.name(),
        }
    }
}

#[pyclass(name = "FrameRewrite", frozen)]
struct PyFrameRewrite {
    inner: HeldRewrite,
}

#[pymethods]
impl PyFrameRewrite {
    #[classmethod]
    fn column_filter_pushdown(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        cohort: Py<PyPreservedIdentities>,
        pushed: Py<PyPreservedIdentities>,
        effect: &str,
    ) -> PyResult<Self> {
        let effect = pushdown_effect(effect).ok_or_else(|| unknown_effect("pushdown", effect))?;
        let cohort = cohort.borrow(py).inner.clone();
        let pushed = pushed.borrow(py).inner.clone();
        Ok(Self {
            inner: HeldRewrite::Pushdown(Box::new(ColumnFilterPushdown::new(
                cohort, pushed, effect,
            ))),
        })
    }

    #[classmethod]
    fn shared_row_design_preparation(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        preparation: Py<PyPreservedIdentities>,
        left: Py<PyPreservedIdentities>,
        right: Py<PyPreservedIdentities>,
        effect: &str,
    ) -> PyResult<Self> {
        let effect =
            preparation_effect(effect).ok_or_else(|| unknown_effect("preparation", effect))?;
        let preparation = preparation.borrow(py).inner.clone();
        let left = left.borrow(py).inner.clone();
        let right = right.borrow(py).inner.clone();
        Ok(Self {
            inner: HeldRewrite::Preparation(Box::new(SharedRowDesignPreparation::new(
                preparation,
                left,
                right,
                effect,
            ))),
        })
    }

    #[classmethod]
    fn nuisance_reuse(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        fit: Py<PyPreservedIdentities>,
        consumer: Py<PyPreservedIdentities>,
        effect: &str,
    ) -> PyResult<Self> {
        let effect = nuisance_effect(effect).ok_or_else(|| unknown_effect("nuisance", effect))?;
        let fit = fit.borrow(py).inner.clone();
        let consumer = consumer.borrow(py).inner.clone();
        Ok(Self {
            inner: HeldRewrite::Nuisance(Box::new(NuisanceReuse::new(fit, consumer, effect))),
        })
    }

    #[classmethod]
    fn sufficient_statistics(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        rows: Py<PyPreservedIdentities>,
        summary: Py<PyPreservedIdentities>,
        effect: &str,
    ) -> PyResult<Self> {
        let effect =
            statistics_effect(effect).ok_or_else(|| unknown_effect("statistics", effect))?;
        let rows = rows.borrow(py).inner.clone();
        let summary = summary.borrow(py).inner.clone();
        Ok(Self {
            inner: HeldRewrite::Statistics(Box::new(SufficientStatistics::new(
                rows, summary, effect,
            ))),
        })
    }

    #[classmethod]
    fn tiled_evaluation(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        full: Py<PyPreservedIdentities>,
        tile: Py<PyPreservedIdentities>,
        effect: &str,
    ) -> PyResult<Self> {
        let effect = tile_effect(effect).ok_or_else(|| unknown_effect("tile", effect))?;
        let full = full.borrow(py).inner.clone();
        let tile = tile.borrow(py).inner.clone();
        Ok(Self {
            inner: HeldRewrite::Tile(Box::new(TiledEvaluation::new(full, tile, effect))),
        })
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Admit this rewrite, or raise. Does not execute the rewrite.
    fn check(&self) -> PyResult<()> {
        self.inner.check().map_err(rewrite_err)
    }
}

#[pyclass(name = "FrequentistSummary", frozen)]
struct PyFrequentistSummary {
    inner: FrequentistSummary,
}

#[pymethods]
impl PyFrequentistSummary {
    #[classmethod]
    fn estimate_covariance(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        FrequentistSummary::estimate_covariance(name)
            .map(|inner| Self { inner })
            .map_err(summary_err)
    }

    #[classmethod]
    fn interval(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<Self> {
        FrequentistSummary::interval(name).map(|inner| Self { inner }).map_err(summary_err)
    }

    fn kind(&self) -> &'static str {
        self.inner.kind().as_str()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Refuse this summary as a Bayesian likelihood.
    fn as_bayesian_likelihood(&self) -> PyResult<()> {
        Err(summary_err(self.inner.as_bayesian_likelihood()))
    }
}

#[pyclass(name = "SummaryObservationModel", frozen)]
struct PySummaryObservationModel {
    inner: SummaryObservationModel,
}

#[pymethods]
impl PySummaryObservationModel {
    /// Adapt an estimate and covariance when the four records are present.
    ///
    /// An interval is refused. The caller's summary object is left unchanged.
    #[classmethod]
    fn try_adapt(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        summary: Py<PyFrequentistSummary>,
        sampling_approximation: &str,
        selection: &str,
        dependence: &str,
        origin: &str,
    ) -> PyResult<Self> {
        adapt_summary(
            summary.borrow(py).inner.clone(),
            sampling_approximation,
            selection,
            dependence,
            origin,
        )
        .map(|inner| Self { inner })
        .map_err(summary_err)
    }

    fn summary_kind(&self) -> &'static str {
        self.inner.summary().kind().as_str()
    }

    fn summary_name(&self) -> &str {
        self.inner.summary().name()
    }

    fn sampling_approximation(&self) -> &str {
        self.inner.sampling_approximation().name()
    }

    fn selection(&self) -> &str {
        self.inner.selection().name()
    }

    fn dependence(&self) -> &str {
        self.inner.dependence().name()
    }

    fn origin(&self) -> &str {
        self.inner.origin().name()
    }

    /// Admit a likelihood update. A Bayesian law is refused.
    fn validate_use(&self, requested: &str) -> PyResult<PyLikelihoodUpdateAdmission> {
        let requested = match requested {
            "likelihood_update" => SummaryUse::LikelihoodUpdate,
            "bayesian_law" => SummaryUse::BayesianLaw,
            _ => return Err(unknown_name("summary use", requested)),
        };
        self.inner
            .validate_use(requested)
            .map(|inner| PyLikelihoodUpdateAdmission { inner })
            .map_err(summary_err)
    }
}

#[pyclass(name = "LikelihoodUpdateAdmission", frozen)]
struct PyLikelihoodUpdateAdmission {
    inner: LikelihoodUpdateAdmission,
}

#[pymethods]
impl PyLikelihoodUpdateAdmission {
    fn summary_kind(&self) -> &'static str {
        self.inner.summary_kind().as_str()
    }

    fn summary_name(&self) -> &str {
        self.inner.summary_name()
    }
}

#[pyclass(name = "FeatureVersion", frozen)]
struct PyFeatureVersion {
    name: String,
    version: u32,
}

#[pymethods]
impl PyFeatureVersion {
    #[new]
    fn new(name: String, version: u32) -> Self {
        Self { name, version }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> u32 {
        self.version
    }
}

#[pyclass(name = "FeatureRead", frozen)]
struct PyFeatureRead {
    kind: &'static str,
    name: String,
    version: u32,
    source_major: Option<u16>,
    source_minor: Option<u16>,
}

impl PyFeatureRead {
    fn from_read(read: FeatureRead) -> PyResult<Self> {
        match read {
            FeatureRead::Compatible { feature } => Ok(Self {
                kind: "compatible",
                name: feature.name.to_owned(),
                version: feature.version,
                source_major: None,
                source_minor: None,
            }),
            FeatureRead::RequiresMigration { from, feature } => Ok(Self {
                kind: "requires_migration",
                name: feature.name.to_owned(),
                version: feature.version,
                source_major: Some(from.major),
                source_minor: Some(from.minor),
            }),
            _ => Err(CausalUnsupportedError::new_err("feature read is not a bound variant")),
        }
    }
}

#[pymethods]
impl PyFeatureRead {
    fn kind(&self) -> &'static str {
        self.kind
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn source_major(&self) -> Option<u16> {
        self.source_major
    }

    fn source_minor(&self) -> Option<u16> {
        self.source_minor
    }

    /// Container `0.5` with an accepted feature is compatible. An unknown feature is refused.
    #[classmethod]
    fn classify(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        container_major: u16,
        container_minor: u16,
        name: &str,
        version: u32,
        accepted: Vec<Py<PyFeatureVersion>>,
    ) -> PyResult<Self> {
        let accepted = accepted_features(py, &accepted);
        let container = FormatVersion { major: container_major, minor: container_minor };
        read_feature(container, name, version, &accepted)
            .map_err(feature_err)
            .and_then(Self::from_read)
    }

    /// A package SemVer is refused, including when a component equals an accepted feature version.
    #[classmethod]
    fn classify_package(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        container_major: u16,
        container_minor: u16,
        major: u32,
        minor: u32,
        patch: u32,
        accepted: Vec<Py<PyFeatureVersion>>,
    ) -> PyResult<Self> {
        let accepted = accepted_features(py, &accepted);
        let container = FormatVersion { major: container_major, minor: container_minor };
        let package = SemanticVersion { major, minor, patch };
        read_package(container, &package, &accepted)
            .map_err(feature_err)
            .and_then(Self::from_read)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRetentionPolicy>()?;
    module.add_class::<PyRetentionMode>()?;
    module.add_class::<PyRetainedOperation>()?;
    module.add_class::<PyPreservedIdentities>()?;
    module.add_class::<PyFrameRewrite>()?;
    module.add_class::<PyFrequentistSummary>()?;
    module.add_class::<PySummaryObservationModel>()?;
    module.add_class::<PyLikelihoodUpdateAdmission>()?;
    module.add_class::<PyFeatureVersion>()?;
    module.add_class::<PyFeatureRead>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_contract::rewrite_legality::PreservationAxis;
    use antecedent_contract::summary_observation::{
        FrequentistSummary, FrequentistSummaryKind, SummaryNextAttempt, SummaryUse,
    };
    use antecedent_io_core::feature_version::{FeatureRead, FeatureRefusal, FeatureVersion};
    use antecedent_io_core::{FormatVersion, STABLE_FORMAT, SemanticVersion};
    use antecedent_state::retention::{RetainedOperation, RetentionMode, RetentionPolicy};

    use super::{
        ColumnFilterPushdown, NuisanceReuse, RewriteKind, SharedRowDesignPreparation,
        SufficientStatistics, TiledEvaluation, adapt_summary, intern_feature_name,
        nuisance_effect, preparation_effect, pushdown_effect, read_feature, read_package,
        statistics_effect, tile_effect,
    };

    fn frame() -> super::PreservedIdentities {
        super::PreservedIdentities::try_new(
            "sample",
            "folds",
            "population",
            "history",
            "dependence",
            "uncertainty",
        )
        .expect("frame")
    }

    #[test]
    fn require_refuses_what_the_mode_cannot_serve() {
        let refusal = RetentionMode::SummaryOnly
            .require(RetainedOperation::RowReconstruction)
            .expect_err("rows");
        assert_eq!(refusal.mode(), RetentionMode::SummaryOnly);
        assert_eq!(refusal.operation(), RetainedOperation::RowReconstruction);
        assert_eq!(refusal.to_string(), "summary-only cannot serve row reconstruction");

        let policy = RetentionPolicy::SufficientStatisticsOnly;
        assert_eq!(policy.result_mode(), RetentionMode::SummaryOnly);
        assert!(policy.require(RetainedOperation::ReuseSummary).is_ok());
        assert!(policy.require(RetainedOperation::RefitFromRetainedRows).is_err());
        assert_eq!(
            RetentionPolicy::BoundedWindow { max_rows: 0 }.result_mode(),
            RetentionMode::SummaryOnly,
        );
        assert_eq!(RetentionPolicy::RawHistory.result_mode(), RetentionMode::FullReusable);
        assert_eq!(RetentionPolicy::RawHistory.max_retained_rows(), None);

        assert!(RetentionMode::DiskBacked.require(RetainedOperation::ReadDiskArtifact).is_ok());
        assert!(RetentionMode::default().require(RetainedOperation::ReuseFullState).is_ok());
        assert!(RetentionMode::default().require(RetainedOperation::ReadDiskArtifact).is_err());
    }

    #[test]
    fn rewrite_that_changes_the_frame_is_refused() {
        let frame = frame();
        let altered = pushdown_effect("undeclared_population_cut").expect("effect");
        let refusal = ColumnFilterPushdown::new(frame.clone(), frame.clone(), altered)
            .check()
            .expect_err("effect");
        assert_eq!(refusal.axis(), PreservationAxis::Population);
        assert_eq!(refusal.next_attempt(), PreservationAxis::Population.next_attempt());
        assert_eq!(refusal.code(), "invalid_argument");
        assert_eq!(refusal.kind(), Some(ColumnFilterPushdown::NAME));

        let moved = super::PreservedIdentities::try_new(
            "sample",
            "folds",
            "other",
            "history",
            "dependence",
            "uncertainty",
        )
        .expect("moved");
        let declared = pushdown_effect("declared").expect("declared");
        let refusal = ColumnFilterPushdown::new(frame.clone(), moved, declared)
            .check()
            .expect_err("identity");
        assert_eq!(refusal.axis(), PreservationAxis::Population);
        assert!(ColumnFilterPushdown::new(frame.clone(), frame.clone(), declared).check().is_ok());
        assert!(pushdown_effect("not_an_effect").is_none());

        let population = preparation_effect("mixes_populations").expect("preparation");
        assert_eq!(
            SharedRowDesignPreparation::new(frame.clone(), frame.clone(), frame.clone(), population)
                .check()
                .expect_err("preparation")
                .axis(),
            PreservationAxis::Population,
        );
        let shared = preparation_effect("shared").expect("shared");
        assert!(
            SharedRowDesignPreparation::new(frame.clone(), frame.clone(), frame.clone(), shared)
                .check()
                .is_ok()
        );

        let history = nuisance_effect("fit_on_other_history").expect("nuisance");
        assert_eq!(
            NuisanceReuse::new(frame.clone(), frame.clone(), history)
                .check()
                .expect_err("nuisance")
                .axis(),
            PreservationAxis::History,
        );
        let certified = nuisance_effect("certified").expect("certified");
        assert!(NuisanceReuse::new(frame.clone(), frame.clone(), certified).check().is_ok());

        let uncertainty = statistics_effect("drops_sampling_law").expect("statistics");
        assert_eq!(
            SufficientStatistics::new(frame.clone(), frame.clone(), uncertainty)
                .check()
                .expect_err("statistics")
                .axis(),
            PreservationAxis::Uncertainty,
        );
        let sufficient = statistics_effect("sufficient").expect("sufficient");
        assert!(
            SufficientStatistics::new(frame.clone(), frame.clone(), sufficient).check().is_ok()
        );

        let draws = tile_effect("drops_failed_draws").expect("tile");
        assert_eq!(
            TiledEvaluation::new(frame.clone(), frame.clone(), draws)
                .check()
                .expect_err("tile")
                .axis(),
            PreservationAxis::Uncertainty,
        );
        let faithful = tile_effect("faithful").expect("faithful");
        assert!(TiledEvaluation::new(frame.clone(), frame, faithful).check().is_ok());
    }

    #[test]
    fn interval_is_not_a_bayesian_law_and_covariance_needs_the_four_fields() {
        let interval = FrequentistSummary::interval("risk_difference").expect("interval");
        let refusal = adapt_summary(
            interval.clone(),
            "normal_approximation",
            "complete_cases",
            "cluster_dependence",
            "trial_estimator",
        )
        .expect_err("interval");
        assert_eq!(interval.kind(), FrequentistSummaryKind::Interval);
        assert_eq!(refusal.detail(), "summary_observation.interval_stays_frequentist");
        assert_eq!(refusal.code(), "distribution_meaning_mismatch");
        assert_eq!(
            refusal.next_attempt(),
            Some(SummaryNextAttempt::ReportAsFrequentistInterval),
        );
        assert!(interval.as_bayesian_likelihood().message().contains("not a Bayesian law"));

        assert!(FrequentistSummary::estimate_covariance(" ").is_err());
        let summary = FrequentistSummary::estimate_covariance("ate").expect("estimate");
        assert!(
            adapt_summary(
                summary.clone(),
                " ",
                "complete_cases",
                "cluster_dependence",
                "trial_estimator",
            )
            .is_err()
        );
        let model = adapt_summary(
            summary,
            "normal_approximation",
            "complete_cases",
            "cluster_dependence",
            "trial_estimator",
        )
        .expect("adapt");
        assert_eq!(model.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(model.sampling_approximation().name(), "normal_approximation");
        assert_eq!(model.selection().name(), "complete_cases");
        assert_eq!(model.dependence().name(), "cluster_dependence");
        assert_eq!(model.origin().name(), "trial_estimator");
        assert!(model.validate_use(SummaryUse::LikelihoodUpdate).is_ok());
        let law = model.validate_use(SummaryUse::BayesianLaw).expect_err("law");
        assert_eq!(law.detail(), "summary_observation.not_bayesian_law");
        assert_eq!(law.next_attempt(), Some(SummaryNextAttempt::UseAsLikelihoodUpdate));
        assert_eq!(model.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
    }

    #[test]
    fn container_0_5_is_compatible_and_unknown_or_semver_is_refused() {
        let name = intern_feature_name("family_payload");
        assert_eq!(intern_feature_name("family_payload"), name);
        let feature = FeatureVersion { name, version: 1 };
        assert_eq!(STABLE_FORMAT, FormatVersion { major: 0, minor: 5 });
        assert_eq!(
            read_feature(STABLE_FORMAT, feature.name, feature.version, &[feature]),
            Ok(FeatureRead::Compatible { feature }),
        );
        assert_eq!(
            read_feature(STABLE_FORMAT, "family_payload", 2, &[feature]),
            Err(FeatureRefusal::UnknownFeature {
                name: "family_payload".to_owned(),
                version: 2,
            }),
        );
        let package = SemanticVersion { major: feature.version, minor: 1, patch: 4 };
        assert_eq!(
            read_package(STABLE_FORMAT, &package, &[feature]),
            Err(FeatureRefusal::PackageSemVer {
                major: package.major,
                minor: package.minor,
                patch: package.patch,
            }),
        );
    }
}
