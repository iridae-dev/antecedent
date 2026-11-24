//! Python bridge for learning preferences from recorded choices: the recorded-choice wire,
//! the compatible-set learner, the noisy-choice estimate and the accepted-preferences
//! lifecycle.
//!
//! As for the decisions, the learning, the review lifecycle, the artifacts and every refusal
//! are Rust's: a refusal comes back as the structured JSON of the underlying error.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::native_claims::NativeDecisionSource;
use antecedent::analysis::preference_decision::{
    AcceptedPreferences, AcceptedPreferencesArtifact, BeliefSource, ChoiceModelEstimate,
    ChoiceModelOptions, ChoiceModelResult, ChoiceOption, ChoiceRecord, ChoiceRecordSet,
    ConflictKind, CovarianceKind, EstimateOutcome, LearnedPreferenceSet, LearningOptions,
    OptionBelief, PendingReview, Pooling, PreferenceBasis, SetLearningOptions, SetLearningResult,
    SetOutcome, analysis_options, estimate_accepted_with_context,
    estimate_choice_model_with_context, learn_compatible_set_accepted_with_context,
    learn_compatible_set_with_context,
};
use antecedent::contract::ExternalRefusal;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    AttributeWire, Reply, attribute_from, contract_from, debug_kind, mean_source_from,
    measure_value, model_from, model_value, parameter_to, parse, refusal, refusal_value,
    relation_to, reply,
};
use crate::distribution_api::PyJointDistributionArtifact;

/// An accepted-preferences handle or the refusal JSON.
type HandleReply = (Option<PyAcceptedPreferences>, Option<String>);

// ---------------------------------------------------------------------------
// Recorded choices.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum BeliefWire {
    Means { values: Vec<f64> },
    Draws { rows: Vec<f64>, weights: Option<Vec<f64>> },
    Unavailable { constraint: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SourceWire {
    Forecast { source_id: String },
    AnalysisAsOf { result_identity: String, as_of: String },
    Realized { assumption: String },
    Stated { elicitation: String, assumption: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionWire {
    id: String,
    belief: BeliefWire,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordWire {
    id: String,
    decider: String,
    options: Vec<OptionWire>,
    chosen: String,
    source: SourceWire,
    observation_rows: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PoolingWire {
    PerDecider,
    Pooled { assumption: String },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordsWire {
    attributes: Vec<AttributeWire>,
    population: String,
    records: Vec<RecordWire>,
    pooling: PoolingWire,
}

fn belief_from(w: BeliefWire) -> OptionBelief {
    match w {
        BeliefWire::Means { values } => OptionBelief::Means(values),
        BeliefWire::Draws { rows, weights } => OptionBelief::Draws { rows, weights },
        BeliefWire::Unavailable { constraint } => {
            OptionBelief::Unavailable { constraint_id: constraint }
        }
    }
}

pub(super) fn belief_to(b: &OptionBelief) -> BeliefWire {
    match b {
        OptionBelief::Means(values) => BeliefWire::Means { values: values.clone() },
        OptionBelief::Draws { rows, weights } => {
            BeliefWire::Draws { rows: rows.clone(), weights: weights.clone() }
        }
        OptionBelief::Unavailable { constraint_id } => {
            BeliefWire::Unavailable { constraint: constraint_id.clone() }
        }
    }
}

fn records_from(json: &str) -> PyResult<ChoiceRecordSet> {
    let wire: RecordsWire = parse(json, "choice records")?;
    Ok(ChoiceRecordSet {
        attributes: wire.attributes.into_iter().map(attribute_from).collect(),
        population: wire.population,
        records: wire
            .records
            .into_iter()
            .map(|r| ChoiceRecord {
                id: r.id,
                decider: r.decider,
                options: r
                    .options
                    .into_iter()
                    .map(|o| ChoiceOption { id: o.id, belief: belief_from(o.belief) })
                    .collect(),
                chosen: r.chosen,
                belief_source: match r.source {
                    SourceWire::Forecast { source_id } => {
                        BeliefSource::RecordedForecast { source_id }
                    }
                    SourceWire::AnalysisAsOf { result_identity, as_of } => {
                        BeliefSource::AnalysisAsOf { result_identity, as_of }
                    }
                    SourceWire::Realized { assumption } => {
                        BeliefSource::RealizedOutcome { assumption_id: assumption }
                    }
                    SourceWire::Stated { elicitation, assumption } => BeliefSource::Stated {
                        elicitation_id: elicitation,
                        assumption_id: assumption,
                    },
                },
                observation_rows: r.observation_rows,
            })
            .collect(),
        pooling: match wire.pooling {
            PoolingWire::PerDecider => Pooling::PerDecider,
            PoolingWire::Pooled { assumption } => Pooling::Pooled { assumption_id: assumption },
        },
    })
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SetOptionsWire {
    tolerance: Option<f64>,
    conflict_search_size: Option<usize>,
    max_boxes: Option<usize>,
    min_width_fraction: Option<f64>,
}

fn set_options_from(json: &str) -> PyResult<SetLearningOptions> {
    let wire: SetOptionsWire = parse(json, "compatible-set options")?;
    let mut options = SetLearningOptions::default();
    if let Some(tolerance) = wire.tolerance {
        options.tolerance = tolerance;
    }
    if let Some(size) = wire.conflict_search_size {
        options.conflict_search_size = size;
    }
    if let Some(max_boxes) = wire.max_boxes {
        options.max_boxes = max_boxes;
    }
    if let Some(width) = wire.min_width_fraction {
        options.min_width_fraction = width;
    }
    Ok(options)
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ChoiceOptionsWire {
    level: Option<f64>,
    n_draws: Option<usize>,
    seed: Option<u64>,
    shape_grid_points: Option<usize>,
    shape_max_dimensions: Option<usize>,
    shape_profile_points: Option<usize>,
    shape_max_refinements: Option<usize>,
}

fn choice_options_from(json: &str) -> PyResult<ChoiceModelOptions> {
    let wire: ChoiceOptionsWire = parse(json, "choice-model options")?;
    let mut options = ChoiceModelOptions::default();
    if let Some(level) = wire.level {
        options.level = level;
    }
    if let Some(n_draws) = wire.n_draws {
        options.n_draws = n_draws;
    }
    if let Some(seed) = wire.seed {
        options.seed = seed;
    }
    if let Some(points) = wire.shape_grid_points {
        options.shape_search.grid_points = points;
    }
    if let Some(dimensions) = wire.shape_max_dimensions {
        options.shape_search.max_dimensions = dimensions;
    }
    if let Some(points) = wire.shape_profile_points {
        options.shape_search.profile_points = points;
    }
    if let Some(refinements) = wire.shape_max_refinements {
        options.shape_search.max_refinements = refinements;
    }
    Ok(options)
}

// ---------------------------------------------------------------------------
// Result JSON.
// ---------------------------------------------------------------------------

/// The kept shape boxes of a compatible set under a value with shape parameters.
fn region_json(
    shape_parameters: &[usize],
    linear_parameters: &[usize],
    cells: Vec<Value>,
    boxes_used: usize,
    min_width_fraction: f64,
    budget_exhausted: bool,
) -> Value {
    json!({
        "shape_parameters": shape_parameters,
        "linear_parameters": linear_parameters,
        "cells": cells,
        "boxes_used": boxes_used,
        "min_width_fraction": min_width_fraction,
        "budget_exhausted": budget_exhausted,
    })
}

macro_rules! region_value {
    ($region:expr) => {{
        let region = $region;
        let cells: Vec<Value> = region
            .cells
            .iter()
            .map(|c| {
                json!({
                    "shape_lower": c.shape_lower,
                    "shape_upper": c.shape_upper,
                    "status": debug_kind(&format!("{:?}", c.status)),
                    "linear_outer": c.linear_outer,
                })
            })
            .collect();
        region_json(
            &region.shape_parameters,
            &region.linear_parameters,
            cells,
            region.boxes_used,
            region.min_width_fraction,
            region.budget_exhausted,
        )
    }};
}

fn learned_json(l: &LearnedPreferenceSet) -> Value {
    let ranges: Vec<Value> = l
        .ranges
        .iter()
        .map(|r| {
            json!({
                "parameter_id": r.parameter_id,
                "declared": r.declared,
                "learned": r.learned,
                "varied_in_records": r.varied_in_records,
                "narrowed": r.narrowed,
                "basis": debug_kind(&format!("{:?}", r.basis)),
            })
        })
        .collect();
    let value_relations: Vec<Value> = l
        .set
        .value_relations
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "chosen": {"rows": r.chosen.rows, "weights": r.chosen.weights},
                "rejected": {"rows": r.rejected.rows, "weights": r.rejected.weights},
                "tolerance": r.tolerance,
            })
        })
        .collect();
    let binding: Vec<Value> = l
        .binding
        .iter()
        .map(|b| json!({"record_id": b.record_id, "rejected_option": b.rejected_option}))
        .collect();
    json!({
        "model_identity": l.model_identity,
        "records_digest": l.records_digest,
        "decider": l.decider,
        "parameters": l.set.parameters.iter().map(parameter_to).collect::<Vec<_>>(),
        "relations": l.set.relations.iter().map(relation_to).collect::<Vec<_>>(),
        "value_relations": value_relations,
        "region": l.region.as_ref().map(|region| region_value!(region)),
        "ranges": ranges,
        "binding": binding,
        "tolerance": l.tolerance,
        "n_records": l.n_records,
        "assumptions": l.assumptions,
    })
}

fn outcome_json(outcome: &SetOutcome) -> Value {
    match outcome {
        SetOutcome::Learned(learned) => {
            json!({"kind": "learned", "learned": learned_json(learned)})
        }
        SetOutcome::Contradictory(conflict) => json!({
            "kind": "contradictory",
            "conflict": {
                "records": conflict.records,
                "kind": match conflict.kind {
                    ConflictKind::Smallest => "smallest",
                    ConflictKind::Irreducible => "irreducible",
                    ConflictKind::NotMinimised => "not_minimised",
                },
                "tolerance_needed": conflict.tolerance_needed,
            },
        }),
        SetOutcome::Undetermined(undetermined) => json!({
            "kind": "undetermined",
            "undetermined": {
                "model_identity": undetermined.model_identity,
                "records_digest": undetermined.records_digest,
                "decider": undetermined.decider,
                "n_records": undetermined.n_records,
                "tolerance": undetermined.tolerance,
                "region": region_value!(&undetermined.region),
            },
        }),
    }
}

fn set_result_json(result: &SetLearningResult) -> Value {
    json!({
        "pooled": result.pooled.as_ref().map(outcome_json),
        "per_decider": result
            .per_decider
            .iter()
            .map(|d| json!({
                "decider": d.decider,
                "n_records": d.n_records,
                "outcome": outcome_json(&d.outcome),
            }))
            .collect::<Vec<_>>(),
        "overlap": result.overlap.as_ref().map(|o| json!({
            "common": outcome_json(&o.common),
            "disagreeing": o.disagreeing,
        })),
    })
}

fn estimate_json(e: &ChoiceModelEstimate) -> Value {
    let parameters: Vec<Value> = e
        .parameters
        .iter()
        .map(|p| {
            json!({
                "parameter_id": p.parameter_id,
                "estimate": p.estimate,
                "standard_error": p.standard_error,
                "interval": p.interval,
                "declared": p.declared,
                "inside_declared": p.inside_declared,
                "interval_method": debug_kind(&format!("{:?}", p.interval_method)),
            })
        })
        .collect();
    json!({
        "identity": e.identity,
        "model_identity": e.model_identity,
        "records_digest": e.records_digest,
        "decider": e.decider,
        "consistency": e.consistency,
        "consistency_standard_error": e.consistency_standard_error,
        "parameters": parameters,
        "not_learnable": e.not_learnable,
        "log_likelihood": e.log_likelihood,
        "null_log_likelihood": e.null_log_likelihood,
        "n_records": e.n_records,
        "n_deciders": e.n_deciders,
        "covariance_kind": match e.covariance_kind {
            CovarianceKind::Model => "model",
            CovarianceKind::ClusteredByDecider => "clustered_by_decider",
        },
        "coefficient_covariance": e.coefficient_covariance,
        "level": e.level,
        "interval_calibration": debug_kind(&format!("{:?}", e.interval_calibration)),
        "measure": measure_value(&e.measure),
        "share_outside_declared": e.share_outside_declared,
        "assumptions": e.assumptions,
        "shape_parameters": e.shape_parameters,
    })
}

fn choice_result_json(result: &ChoiceModelResult) -> Value {
    json!({
        "pooled": result.pooled.as_deref().map(estimate_json),
        "per_decider": result
            .per_decider
            .iter()
            .map(|(decider, outcome)| match outcome {
                EstimateOutcome::Estimated(estimate) => json!({
                    "decider": decider,
                    "estimate": estimate_json(estimate),
                    "refusal": Value::Null,
                }),
                EstimateOutcome::Refused(error) => json!({
                    "decider": decider,
                    "estimate": Value::Null,
                    "refusal": refusal_value(&error.to_refusal()),
                }),
            })
            .collect::<Vec<_>>(),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum PendingWire {
    ParameterNotVaried { parameter_id: String },
    ParameterNotNarrowed { parameter_id: String },
    ToleranceApplied { tolerance: f64 },
    ParameterNotLearnable { parameter_id: String },
    EstimateOutsideDeclared { parameter_id: String },
    DrawsOutsideDeclared { share: f64 },
    StatedComparisonsUsed { count: usize },
}

fn pending_to(item: &PendingReview) -> Value {
    let wire = match item {
        PendingReview::ParameterNotVaried { parameter_id } => {
            PendingWire::ParameterNotVaried { parameter_id: parameter_id.clone() }
        }
        PendingReview::ParameterNotNarrowed { parameter_id } => {
            PendingWire::ParameterNotNarrowed { parameter_id: parameter_id.clone() }
        }
        PendingReview::ToleranceApplied { tolerance } => {
            PendingWire::ToleranceApplied { tolerance: *tolerance }
        }
        PendingReview::ParameterNotLearnable { parameter_id } => {
            PendingWire::ParameterNotLearnable { parameter_id: parameter_id.clone() }
        }
        PendingReview::EstimateOutsideDeclared { parameter_id } => {
            PendingWire::EstimateOutsideDeclared { parameter_id: parameter_id.clone() }
        }
        PendingReview::DrawsOutsideDeclared { share } => {
            PendingWire::DrawsOutsideDeclared { share: *share }
        }
        PendingReview::StatedComparisonsUsed { count } => {
            PendingWire::StatedComparisonsUsed { count: *count }
        }
    };
    serde_json::to_value(wire).unwrap_or(Value::Null)
}

fn pending_from(wire: PendingWire) -> PendingReview {
    match wire {
        PendingWire::ParameterNotVaried { parameter_id } => {
            PendingReview::ParameterNotVaried { parameter_id }
        }
        PendingWire::ParameterNotNarrowed { parameter_id } => {
            PendingReview::ParameterNotNarrowed { parameter_id }
        }
        PendingWire::ToleranceApplied { tolerance } => {
            PendingReview::ToleranceApplied { tolerance }
        }
        PendingWire::ParameterNotLearnable { parameter_id } => {
            PendingReview::ParameterNotLearnable { parameter_id }
        }
        PendingWire::EstimateOutsideDeclared { parameter_id } => {
            PendingReview::EstimateOutsideDeclared { parameter_id }
        }
        PendingWire::DrawsOutsideDeclared { share } => {
            PendingReview::DrawsOutsideDeclared { share }
        }
        PendingWire::StatedComparisonsUsed { count } => {
            PendingReview::StatedComparisonsUsed { count }
        }
    }
}

fn learning_json(options: Option<LearningOptions>) -> Value {
    match options {
        None => Value::Null,
        Some(LearningOptions::CompatibleSet(o)) => json!({
            "kind": "compatible_set",
            "tolerance": o.tolerance,
            "conflict_search_size": o.conflict_search_size,
            "max_boxes": o.max_boxes,
            "min_width_fraction": o.min_width_fraction,
        }),
        Some(LearningOptions::ChoiceEstimate(o)) => json!({
            "kind": "choice_model",
            "level": o.level,
            "n_draws": o.n_draws,
            "seed": o.seed,
            "shape_grid_points": o.shape_search.grid_points,
            "shape_max_dimensions": o.shape_search.max_dimensions,
            "shape_profile_points": o.shape_search.profile_points,
            "shape_max_refinements": o.shape_search.max_refinements,
        }),
    }
}

fn accepted_json(a: &AcceptedPreferences) -> Value {
    let basis = match a.basis() {
        PreferenceBasis::Declared => json!({"kind": "declared"}),
        PreferenceBasis::CompatibleSet(learned) => {
            json!({"kind": "compatible_set", "learned": learned_json(learned)})
        }
        PreferenceBasis::ChoiceEstimate(estimate) => {
            json!({"kind": "choice_estimate", "estimate": estimate_json(estimate)})
        }
    };
    json!({
        "identity": a.identity(),
        "version": a.version(),
        "declared_identity": a.declared_identity(),
        "decider": a.decider(),
        "basis": basis,
        "model": model_value(a.model()),
        "declared_model": model_value(a.declared_model()),
        "measure": a.measure().map(measure_value),
        "pending": a.pending().iter().map(pending_to).collect::<Vec<_>>(),
        "acknowledged": a.acknowledged().iter().map(pending_to).collect::<Vec<_>>(),
        "assumptions": a.assumptions(),
        "stated_comparisons": a.stated_comparisons(),
        "learning": learning_json(a.learning_options()),
    })
}

// ---------------------------------------------------------------------------
// Learning.
// ---------------------------------------------------------------------------

/// Learn the set of preference parameters compatible with recorded choices; returns
/// `{pooled, per_decider, overlap}` or a refusal. The learner reads `cancel` before every
/// linear program.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, options_json, cancel=None))]
fn learn_preference_set(
    model_json: &str,
    records_json: &str,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = set_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            learn_compatible_set_with_context(&model, &records, options, &ctx)
                .map(|result| set_result_json(&result))
                .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Estimate the preference parameters from choices read as noisy; returns
/// `{pooled, per_decider}` or a refusal. The estimator reads `cancel` before every fit,
/// Newton step and block of measure draws.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, options_json, cancel=None))]
fn estimate_preference_choice_model(
    model_json: &str,
    records_json: &str,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = choice_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(reply(
            estimate_choice_model_with_context(&model, &records, options, &ctx)
                .map(|result| choice_result_json(&result))
                .map_err(|e| refusal(&e.to_refusal())),
        ))
    })
}

/// Reviewed, versioned preferences: an immutable native handle.
#[pyclass(name = "_AcceptedPreferences", frozen, skip_from_py_object)]
pub(crate) struct PyAcceptedPreferences {
    inner: AcceptedPreferences,
}

fn handle(outcome: Result<AcceptedPreferences, ExternalRefusal>) -> HandleReply {
    match outcome {
        Ok(inner) => (Some(PyAcceptedPreferences { inner }), None),
        Err(error) => (None, Some(refusal(&error))),
    }
}

#[pymethods]
impl PyAcceptedPreferences {
    #[getter]
    fn identity(&self) -> String {
        self.inner.identity().to_owned()
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version()
    }

    /// Everything the handle holds, as JSON: identity, version, basis, accepted model,
    /// measure, pending and acknowledged review items, assumptions and learning options.
    fn summary_json(&self) -> String {
        accepted_json(&self.inner).to_string()
    }

    /// A new handle with the review items acknowledged, or a refusal.
    fn acknowledge(&self, items_json: &str) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let items: Vec<PendingWire> = parse(items_json, "review items")?;
            let items: Vec<PendingReview> = items.into_iter().map(pending_from).collect();
            Ok(handle(self.inner.acknowledge(&items).map_err(|e| e.to_refusal())))
        })
    }

    /// Relearn the compatible set from `records`: version + 1, acknowledgements cleared. The
    /// learner reads `cancel` before every linear program.
    #[pyo3(signature = (records_json, decider, options_json, cancel=None))]
    fn relearn_compatible_set(
        &self,
        records_json: &str,
        decider: Option<&str>,
        options_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let records = records_from(records_json)?;
            let options = set_options_from(options_json)?;
            let ctx = crate::transport_common::execution_context(0, None, cancel);
            Ok(handle(
                self.inner
                    .relearn_compatible_set_with_context(&records, decider, options, &ctx)
                    .map_err(|e| e.to_refusal()),
            ))
        })
    }

    /// Re-estimate from `records`: version + 1, acknowledgements cleared. The estimator reads
    /// `cancel` before every fit, Newton step and block of measure draws.
    #[pyo3(signature = (records_json, decider, options_json, cancel=None))]
    fn relearn_choice_model(
        &self,
        records_json: &str,
        decider: Option<&str>,
        options_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<HandleReply> {
        crate::catch_ffi(|| {
            let records = records_from(records_json)?;
            let options = choice_options_from(options_json)?;
            let ctx = crate::transport_common::execution_context(0, None, cancel);
            Ok(handle(
                self.inner
                    .relearn_estimate_with_context(&records, decider, options, &ctx)
                    .map_err(|e| e.to_refusal()),
            ))
        })
    }

    /// The checksummed record of the handle, or a refusal.
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
    ) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<String>)> {
        crate::catch_ffi(|| {
            Ok(match AcceptedPreferencesArtifact::new(self.inner.clone()).to_bytes(artifact_id) {
                Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
                Err(error) => (None, Some(refusal(&error.to_refusal()))),
            })
        })
    }
}

/// Accept the declared preference set as it stands (version 1, nothing pending).
#[pyfunction]
fn accept_preferences_declared(model_json: &str) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        Ok(handle(AcceptedPreferences::declared(model).map_err(|e| e.to_refusal())))
    })
}

/// Learn the compatible set and accept it; `decider` selects one decider's outcome, `None`
/// the pooled one. The learner reads `cancel` before every linear program.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, decider, options_json, cancel=None))]
fn accept_preferences_compatible_set(
    model_json: &str,
    records_json: &str,
    decider: Option<&str>,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = set_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(handle(
            learn_compatible_set_accepted_with_context(&model, &records, decider, options, &ctx)
                .map_err(|e| e.to_refusal()),
        ))
    })
}

/// Estimate from choices read as noisy and accept the estimate. The estimator reads `cancel`
/// before every fit, Newton step and block of measure draws.
#[pyfunction]
#[pyo3(signature = (model_json, records_json, decider, options_json, cancel=None))]
fn accept_preferences_choice_model(
    model_json: &str,
    records_json: &str,
    decider: Option<&str>,
    options_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let model = model_from(parse(model_json, "value model")?);
        let records = records_from(records_json)?;
        let options = choice_options_from(options_json)?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        Ok(handle(
            estimate_accepted_with_context(&model, &records, decider, options, &ctx)
                .map_err(|e| e.to_refusal()),
        ))
    })
}

/// What was expected of each action of a contract, read from a source, as the options of a
/// recorded choice: `{options: [{id, belief}]}`. Exactly one of `source` and
/// `mean_source_json` is given.
#[pyfunction]
#[pyo3(signature = (contract_json, source, mean_source_json))]
fn preference_analysis_options(
    contract_json: &str,
    source: Option<&PyJointDistributionArtifact>,
    mean_source_json: Option<&str>,
) -> PyResult<Reply> {
    crate::catch_ffi(|| {
        let contract = contract_from(contract_json)?;
        let native = match (source, mean_source_json) {
            (Some(joint), None) => NativeDecisionSource::JointLaw(Box::new(joint.inner().clone())),
            (None, Some(json)) => NativeDecisionSource::Mean(mean_source_from(json)?),
            _ => return Err(crate::value_err("pass joint draws or a mean source, not both")),
        };
        Ok(reply(
            analysis_options(&contract, &native)
                .map(|options| {
                    json!({
                        "options": options
                            .iter()
                            .map(|o| json!({"id": o.id, "belief": belief_to(&o.belief)}))
                            .collect::<Vec<_>>(),
                    })
                })
                .map_err(|e| refusal(&e)),
        ))
    })
}

/// Load accepted preferences under an identity the consumer retained; with records, a
/// learned basis is relearned from them and must be reproduced, reading `cancel` as the
/// learner or the estimator does.
#[pyfunction]
#[pyo3(signature = (data, expected_identity, records_json, cancel=None))]
fn consume_accepted_preferences(
    data: &[u8],
    expected_identity: &str,
    records_json: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<HandleReply> {
    crate::catch_ffi(|| {
        let records = records_json.map(records_from).transpose()?;
        let ctx = crate::transport_common::execution_context(0, None, cancel);
        let outcome = AcceptedPreferencesArtifact::from_bytes(data, expected_identity)
            .and_then(|artifact| match &records {
                Some(records) => artifact.replay_with_context(records, &ctx).map(|()| artifact),
                None => Ok(artifact),
            })
            .map(|artifact| artifact.accepted().clone())
            .map_err(|e| e.to_refusal());
        Ok(handle(outcome))
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAcceptedPreferences>()?;
    m.add_function(wrap_pyfunction!(learn_preference_set, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_preference_choice_model, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_declared, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_compatible_set, m)?)?;
    m.add_function(wrap_pyfunction!(accept_preferences_choice_model, m)?)?;
    m.add_function(wrap_pyfunction!(preference_analysis_options, m)?)?;
    m.add_function(wrap_pyfunction!(consume_accepted_preferences, m)?)
}
