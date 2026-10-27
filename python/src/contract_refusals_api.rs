//! Python surface for retention, rewrite-legality, summary-observation, and
//! feature-version contract refusals.
//!
//! Retention `require` refusals, rewrite-legality checks, summary-observation
//! adaptation, and feature-version reads. A refusal is a Python exception.
//! Checking a rewrite does not execute it. Adapting a summary does not form a
//! posterior. Classifying a feature does not decode a container.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::rewrite_legality::{
    FrameContents, NuisanceEffect, PreparationEffect, PreservationAxis, PreservedIdentities,
    PushdownEffect, Rewrite, RewriteRefusal, SummaryEffect, TileEffect,
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

use crate::name_errors::unknown_name;
use crate::{CausalStateError, CausalUnsupportedError};

/// Set attributes on the exception value. A failed `setattr` replaces the
/// refusal with that failure so it is never dropped silently.
fn attach_attributes(err: PyErr, set: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<()>) -> PyErr {
    match Python::attach(|py| set(err.value(py).as_any())) {
        Ok(()) => err,
        Err(failure) => failure,
    }
}

fn retention_err(refusal: RetentionRefusal) -> PyErr {
    let mode = refusal.mode().as_str();
    let operation = refusal.operation().as_str();
    let lost: Vec<&'static str> =
        refusal.no_longer_serves().iter().copied().map(RetainedOperation::as_str).collect();
    let err = CausalStateError::new_err(refusal.to_string());
    attach_attributes(err, |value| {
        value.setattr("mode", mode)?;
        value.setattr("operation", operation)?;
        value.setattr("no_longer_serves", lost)
    })
}

fn rewrite_err(refusal: RewriteRefusal) -> PyErr {
    let detail = refusal.detail();
    let axis = refusal.axis().as_str();
    let next = refusal.next_attempt();
    let kind = refusal.kind();
    let err = crate::value_err_code(refusal.code(), refusal.to_string());
    attach_attributes(err, |value| {
        value.setattr("detail", detail)?;
        value.setattr("axis", axis)?;
        value.setattr("next_attempt", next.as_str())?;
        value.setattr("remedy", next.sentence())?;
        if let Some(kind) = kind {
            value.setattr("kind", kind)?;
        }
        Ok(())
    })
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
    attach_attributes(err, |value| {
        value.setattr("detail", detail)?;
        if let Some(next) = next {
            value.setattr("next_attempt", next.as_str())?;
            value.setattr("remedy", next.sentence())?;
        }
        if let Some(kind) = kind {
            value.setattr("summary_kind", kind.as_str())?;
        }
        Ok(())
    })
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

/// Parse an effect name through the Rust enum's own name table, so a variant
/// added there is reachable here without a second list.
macro_rules! effect_names {
    ($name:ident, $ty:ident) => {
        fn $name(name: &str) -> Option<$ty> {
            $ty::from_name(name)
        }
    };
}

effect_names!(pushdown_effect, PushdownEffect);
effect_names!(preparation_effect, PreparationEffect);
effect_names!(nuisance_effect, NuisanceEffect);
effect_names!(statistics_effect, SummaryEffect);
effect_names!(tile_effect, TileEffect);

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
) -> PyResult<Vec<FeatureVersion>> {
    accepted
        .iter()
        .map(|item| {
            let view = item.borrow(py);
            Ok(FeatureVersion { name: Arc::<str>::from(view.name.as_str()), version: view.version })
        })
        .collect()
}

#[pyclass(name = "RetentionPolicy", frozen)]
pub(crate) struct PyRetentionPolicy {
    pub(crate) inner: RetentionPolicy,
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

    /// The refusal of `operation` under this policy's result mode, or `None` when it is served.
    fn refusal(
        &self,
        py: Python<'_>,
        operation: Py<PyRetainedOperation>,
    ) -> Option<PyRetentionRefusal> {
        self.inner.require(operation.borrow(py).inner).err().map(PyRetentionRefusal::new)
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

    /// The refusal of `operation`, or `None` when this mode serves it.
    fn refusal(
        &self,
        py: Python<'_>,
        operation: Py<PyRetainedOperation>,
    ) -> Option<PyRetentionRefusal> {
        self.inner.require(operation.borrow(py).inner).err().map(PyRetentionRefusal::new)
    }
}

/// A refusal of a later operation by a reduced retention mode, as a value rather than a raised
/// error: the mode that was asked, the operation it cannot serve, and everything that mode
/// can no longer serve.
#[pyclass(name = "RetentionRefusal", frozen)]
struct PyRetentionRefusal {
    inner: RetentionRefusal,
}

impl PyRetentionRefusal {
    fn new(inner: RetentionRefusal) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRetentionRefusal {
    fn mode(&self) -> PyRetentionMode {
        PyRetentionMode { inner: self.inner.mode() }
    }

    fn operation(&self) -> PyRetainedOperation {
        PyRetainedOperation::new(self.inner.operation())
    }

    fn no_longer_serves(&self) -> Vec<PyRetainedOperation> {
        self.inner.no_longer_serves().iter().copied().map(PyRetainedOperation::new).collect()
    }

    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Raise this refusal as the `CausalStateError` a failed `require` raises.
    fn raise_error(&self) -> PyResult<()> {
        Err(retention_err(self.inner))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
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

#[pyclass(name = "FrameRewrite", frozen)]
struct PyFrameRewrite {
    inner: Rewrite,
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
        Ok(Self { inner: Rewrite::column_filter_pushdown(cohort, pushed, effect) })
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
        Ok(Self { inner: Rewrite::shared_row_design_preparation(preparation, left, right, effect) })
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
        Ok(Self { inner: Rewrite::nuisance_reuse(fit, consumer, effect) })
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
        Ok(Self { inner: Rewrite::sufficient_statistics(rows, summary, effect) })
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
        Ok(Self { inner: Rewrite::tiled_evaluation(full, tile, effect) })
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Bind one frame's selected rows, folds, population, and history to every side.
    ///
    /// `check` compares those contents. Matching identity strings are not enough.
    fn with_matching_frame(
        &self,
        row_ids: Vec<String>,
        fold_ids: Vec<String>,
        population: &str,
        history: &str,
    ) -> PyResult<Self> {
        let rows: Vec<&str> = row_ids.iter().map(String::as_str).collect();
        let folds: Vec<&str> = fold_ids.iter().map(String::as_str).collect();
        let frame =
            FrameContents::try_new(&rows, &folds, population, history).map_err(rewrite_err)?;
        let claims = if self.inner.name() == Rewrite::SHARED_ROW_DESIGN_PREPARATION {
            vec![&frame, &frame]
        } else {
            vec![&frame]
        };
        let inner = self.inner.clone().with_frames(&frame, &claims).map_err(rewrite_err)?;
        Ok(Self { inner })
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
    fn estimate_covariance(
        _cls: &Bound<'_, PyType>,
        name: &str,
        estimate: f64,
        covariance: f64,
    ) -> PyResult<Self> {
        FrequentistSummary::estimate_covariance(name, estimate, covariance)
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
                name: feature.name.to_string(),
                version: feature.version,
                source_major: None,
                source_minor: None,
            }),
            FeatureRead::RequiresMigration { from, feature } => Ok(Self {
                kind: "requires_migration",
                name: feature.name.to_string(),
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
        let accepted = accepted_features(py, &accepted)?;
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
        let accepted = accepted_features(py, &accepted)?;
        let container = FormatVersion { major: container_major, minor: container_minor };
        let package = SemanticVersion { major, minor, patch };
        read_package(container, &package, &accepted).map_err(feature_err).and_then(Self::from_read)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRetentionPolicy>()?;
    module.add_class::<PyRetentionMode>()?;
    module.add_class::<PyRetainedOperation>()?;
    module.add_class::<PyRetentionRefusal>()?;
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
    use std::sync::Arc;

    use antecedent_contract::rewrite_legality::{FrameContents, PreservationAxis};
    use antecedent_contract::summary_observation::{
        FrequentistSummary, FrequentistSummaryKind, REPORT_AS_FREQUENTIST_INTERVAL, SummaryUse,
        USE_AS_LIKELIHOOD_UPDATE,
    };
    use antecedent_io_core::feature_version::{FeatureRead, FeatureRefusal, FeatureVersion};
    use antecedent_io_core::{FormatVersion, STABLE_FORMAT, SemanticVersion};
    use antecedent_state::retention::{RetainedOperation, RetentionMode, RetentionPolicy};

    use super::{
        Rewrite, adapt_summary, nuisance_effect, preparation_effect, pushdown_effect, read_feature,
        read_package, statistics_effect, tile_effect,
    };

    fn frame_contents() -> FrameContents {
        FrameContents::try_new(&["row-1"], &["fold-1"], "population", "history").expect("contents")
    }

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
        let refusal = Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), altered)
            .check()
            .expect_err("effect");
        assert_eq!(refusal.axis(), PreservationAxis::Population);
        assert_eq!(refusal.next_attempt(), PreservationAxis::Population.next_attempt());
        assert_eq!(refusal.code(), "invalid_argument");
        assert_eq!(refusal.kind(), Some(Rewrite::COLUMN_FILTER_PUSHDOWN));

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
        let refusal = Rewrite::column_filter_pushdown(frame.clone(), moved, declared)
            .check()
            .expect_err("identity");
        assert_eq!(refusal.axis(), PreservationAxis::Population);
        let same = frame_contents();
        assert!(
            Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), declared)
                .with_frames(&same, &[&same])
                .expect("frames")
                .check()
                .is_ok()
        );
        assert!(pushdown_effect("not_an_effect").is_none());

        let population = preparation_effect("mixes_populations").expect("preparation");
        assert_eq!(
            Rewrite::shared_row_design_preparation(
                frame.clone(),
                frame.clone(),
                frame.clone(),
                population
            )
            .check()
            .expect_err("preparation")
            .axis(),
            PreservationAxis::Population,
        );
        let shared = preparation_effect("shared").expect("shared");
        let same = frame_contents();
        assert!(
            Rewrite::shared_row_design_preparation(
                frame.clone(),
                frame.clone(),
                frame.clone(),
                shared
            )
            .with_frames(&same, &[&same, &same])
            .expect("frames")
            .check()
            .is_ok()
        );

        let history = nuisance_effect("fit_on_other_history").expect("nuisance");
        assert_eq!(
            Rewrite::nuisance_reuse(frame.clone(), frame.clone(), history)
                .check()
                .expect_err("nuisance")
                .axis(),
            PreservationAxis::History,
        );
        let certified = nuisance_effect("certified").expect("certified");
        let same = frame_contents();
        assert!(
            Rewrite::nuisance_reuse(frame.clone(), frame.clone(), certified)
                .with_frames(&same, &[&same])
                .expect("frames")
                .check()
                .is_ok()
        );

        let uncertainty = statistics_effect("drops_sampling_law").expect("statistics");
        assert_eq!(
            Rewrite::sufficient_statistics(frame.clone(), frame.clone(), uncertainty)
                .check()
                .expect_err("statistics")
                .axis(),
            PreservationAxis::Uncertainty,
        );
        let sufficient = statistics_effect("sufficient").expect("sufficient");
        let same = frame_contents();
        assert!(
            Rewrite::sufficient_statistics(frame.clone(), frame.clone(), sufficient)
                .with_frames(&same, &[&same])
                .expect("frames")
                .check()
                .is_ok()
        );

        let draws = tile_effect("drops_failed_draws").expect("tile");
        assert_eq!(
            Rewrite::tiled_evaluation(frame.clone(), frame.clone(), draws)
                .check()
                .expect_err("tile")
                .axis(),
            PreservationAxis::Uncertainty,
        );
        let faithful = tile_effect("faithful").expect("faithful");
        let same = frame_contents();
        assert!(
            Rewrite::tiled_evaluation(frame.clone(), frame, faithful)
                .with_frames(&same, &[&same])
                .expect("frames")
                .check()
                .is_ok()
        );
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
        assert_eq!(refusal.next_attempt(), Some(REPORT_AS_FREQUENTIST_INTERVAL),);
        assert!(interval.as_bayesian_likelihood().message().contains("not a Bayesian law"));

        assert!(FrequentistSummary::estimate_covariance(" ", 1.0_f64, 1.0_f64).is_err());
        let summary =
            FrequentistSummary::estimate_covariance("ate", 1.5_f64, 0.04_f64).expect("estimate");
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
        assert_eq!(law.next_attempt(), Some(USE_AS_LIKELIHOOD_UPDATE));
        assert_eq!(model.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
    }

    #[test]
    fn container_0_5_is_compatible_and_unknown_or_semver_is_refused() {
        let feature = FeatureVersion { name: Arc::from("family_payload"), version: 1 };
        assert_eq!(STABLE_FORMAT, FormatVersion { major: 0, minor: 5 });
        let got =
            read_feature(STABLE_FORMAT, feature.name.as_ref(), feature.version, &[feature.clone()]);
        assert_eq!(got, Ok(FeatureRead::Compatible { feature: feature.clone() }));
        assert_eq!(
            read_feature(STABLE_FORMAT, "family_payload", 2, &[feature.clone()]),
            Err(FeatureRefusal::UnknownFeature { name: "family_payload".to_owned(), version: 2 }),
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
