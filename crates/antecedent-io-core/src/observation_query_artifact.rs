//! Replayable artifact for an observation-conditioned scientific query.
//!
//! Format version 1, kind `observation_query_v1`. The artifact keeps every
//! premise the answer depends on: for a query on a composed causal model, the
//! canonical composition premises (the same wire the mechanism-composition
//! artifact uses); for a query on a declared finite or conjugate Bayes model,
//! the declared prior and likelihood inside the query itself. It also keeps
//! the query, the operation the query performs, how the value is to be read
//! (`exact_given_declared_model` or `point_only_sample_fitted`) and the stored
//! result as IEEE bit patterns.
//!
//! # Conditioning and intervention are different claims
//!
//! Every query variant names its operation, and the operation is part of the
//! identity: conditioning on `S = s`, intervening with `do(S = s)` and the
//! paired comparison of the two over one model have three different
//! identities and three differently-shaped results. A measured value is
//! carried with the measurement model that relates it to the construct; the
//! artifact has no way to say "this recorded value is the construct".
//!
//! A consumer trusts none of it. It recomputes the identity over the canonical
//! premises, query and reading, recompiles the composition under the coherence
//! checker, replays the query and accepts only the identical result bits.
//! What replay protects against: corruption, edits that are not resealed, and
//! resealed edits that change the answer. It does not protect against a
//! producer that seals wrong but coherent premises, or a bug shared by
//! producer and consumer; the independent oracles are in the test suites.
//!
//! No interval and no calibration is carried.
//!
//! # General queries
//!
//! Queries on continuous variables (conditioning a linear-Gaussian composition
//! on observed real values, with or without an intervention), observation after
//! an intervention on a finite model, and parameter uncertainty by retained-data
//! redraw are carried in format version 2 under the feature marker
//! `observation_query_general_v1`. A version-1 artifact keeps its bytes and its
//! identity; a version-2 artifact is refused by a version-1 reader before its
//! payload is interpreted. Real values are IEEE bit patterns, never strings.
//! Redrawn parameters are returned as aligned draws (one row is one joint
//! realization of every queried quantity) with the alignment identity of the
//! repo's distribution artifact, and carry no coverage statement.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod error;
mod wire;

pub use error::{ObservationQueryArtifactError, query_error_refusals};
use error::{composition_error, query_error, refused, registry_refusal};

pub use wire::{
    DependentMeasurementWire, MeasuredSlotWire, ObservationQueryWire, ObservationResultWire,
    RedrawPriorWire, RedrawTargetWire, RetainedMechanismWire,
};
use wire::{level_bits, real_refs};

use antecedent_contract::{
    InferenceBasis, MeasuredIndicator, MechanismOriginKind, ObservationContext, ObservedSlot,
    QuantityRole, ScientificQuantity, SlotProvenance, SlotValue, measured_likelihood,
    mixed_conditioning_refusal,
};
use antecedent_core::measurement_model::{MeasurementKind, RepresentationVersion};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, FiniteLaw, MAX_COMPOSED_MECHANISMS,
    MechanismOrigin,
};
use antecedent_model::observation_query::{
    BetaBernoulli, Direction, FiniteBayesModel, MAX_BAYES_HYPOTHESES, MAX_BAYES_OBSERVATION_LEVELS,
    MAX_BAYES_UPDATE_WORK, condition, condition_gaussian, condition_on_measured,
    conditioning_and_intervention, intervene, intervened_law, observations_required_binary_signal,
    observations_required_conjugate, required_bayes_factor, threshold_crossing_observations,
};
use antecedent_model::observation_uncertainty::{
    GaussianSufficient, MAX_REDRAW_QUANTITIES, MAX_REDRAWS, MAX_RETAINED_REGRESSORS,
    ParameterDraws, RetainedMechanism, TableCountRow, TableCounts, redraw_conditional,
};
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use crate::measurement_model_wire::model_from_wire;
use crate::mechanism_composition_artifact::MechanismCompositionPremisesWire;
use crate::quantity_wire::DistributionMeaningWire;

/// The artifact format this reader writes and accepts.
pub const OBSERVATION_QUERY_ARTIFACT_VERSION: u32 = 1;
/// The feature marker (and artifact kind) of the accepted format.
pub const OBSERVATION_QUERY_ARTIFACT_FEATURE: &str = "observation_query_v1";
/// The format version that carries the general queries (continuous evidence,
/// observation after an intervention, retained-data redraw). A reader of
/// [`OBSERVATION_QUERY_ARTIFACT_VERSION`] alone refuses it before the payload
/// is interpreted.
pub const OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION: u32 = 2;
/// The feature marker of the general queries.
pub const OBSERVATION_QUERY_GENERAL_FEATURE: &str = "observation_query_general_v1";
/// Additive format for a declared joint measurement likelihood.
pub const OBSERVATION_QUERY_JOINT_MEASURED_VERSION: u32 = 3;
/// Required feature of the joint measured-signal query.
pub const OBSERVATION_QUERY_JOINT_MEASURED_FEATURE: &str = "observation_query_joint_measured_v1";
/// Additive format for a complete dependent-error measurement law.
pub const OBSERVATION_QUERY_DEPENDENT_MEASURED_VERSION: u32 = 4;
/// Required feature of dependent-error conditioning.
pub const OBSERVATION_QUERY_DEPENDENT_MEASURED_FEATURE: &str =
    "observation_query_dependent_measured_v1";
/// Most retained mechanisms, and most parent-configuration rows of one retained
/// table, a redraw query may carry.
pub const OBSERVATION_QUERY_MAX_RETAINED_ROWS: usize = 4096;
/// The reading of the spread of redrawn parameters: a spread of exact posterior
/// draws with no coverage statement.
pub const PARAMETER_DRAWS_READING: &str = antecedent_contract::UNCALIBRATED_SPREAD_LABEL;

/// The versioned artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationQueryArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The composed model premises; present exactly when the query runs on a
    /// composed model.
    pub premises: Option<MechanismCompositionPremisesWire>,
    /// The question.
    pub query: ObservationQueryWire,
    /// How the value is to be read.
    pub inference_basis: String,
    /// The stored answer.
    pub result: ObservationResultWire,
    /// Digest of the premises, query and reading.
    pub identity: String,
}

/// A consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedObservationQuery {
    /// The identity that was recomputed and verified.
    pub identity: String,
    /// The operation the query performed.
    pub operation: String,
    /// How the value is to be read.
    pub inference_basis: String,
    /// The replayed result, bit-identical to the stored one.
    pub result: ObservationResultWire,
}

// ------------------------------------------------------------------ identity

/// Identity of the premises, query and reading.
///
/// The digest is over the canonical wire encoding under a versioned tag. The
/// operation tag is bound in explicitly as well as through the query variant,
/// and the stored result is excluded: the result is an output replayed from
/// the premises, not a premise.
///
/// # Errors
/// Encoding failure.
pub fn observation_query_identity(
    premises: Option<&MechanismCompositionPremisesWire>,
    query: &ObservationQueryWire,
    inference_basis: &str,
) -> Result<String, IoError> {
    if query.is_dependent_measured() {
        return crate::digest_hex(
            IdentityDomain::Program,
            &(
                OBSERVATION_QUERY_DEPENDENT_MEASURED_FEATURE,
                OBSERVATION_QUERY_DEPENDENT_MEASURED_VERSION,
                query.operation(),
                inference_basis,
                premises,
                query,
            ),
        );
    }
    if query.is_joint_measured() {
        return crate::digest_hex(
            IdentityDomain::Program,
            &(
                OBSERVATION_QUERY_JOINT_MEASURED_FEATURE,
                OBSERVATION_QUERY_JOINT_MEASURED_VERSION,
                query.operation(),
                inference_basis,
                premises,
                query,
            ),
        );
    }
    if query.is_general() {
        return crate::digest_hex(
            IdentityDomain::Program,
            &(
                OBSERVATION_QUERY_GENERAL_FEATURE,
                OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION,
                query.operation(),
                inference_basis,
                premises,
                query,
            ),
        );
    }
    crate::digest_hex(
        IdentityDomain::Program,
        &(
            "observation_query_v1",
            OBSERVATION_QUERY_ARTIFACT_VERSION,
            query.operation(),
            inference_basis,
            premises,
            query,
        ),
    )
}

// ----------------------------------------------------------------- execution

/// Check the structural bounds of a declared observation query without
/// evaluating its model or conditioning a law. Assignment lists carry no
/// count cap beyond the artifact byte budget; model evaluation checks names.
///
/// # Errors
/// A malformed or oversized declaration.
#[allow(clippy::match_same_arms, reason = "the cases retain distinct wire or replay semantics")]
#[allow(clippy::too_many_lines, reason = "one arm per query declaration form")]
pub fn check_query_bounds(
    query: &ObservationQueryWire,
) -> Result<(), ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::ConditionalOutcome { .. }
        | ObservationQueryWire::InterventionalOutcome { .. }
        | ObservationQueryWire::ConditioningVersusIntervention { .. }
        | ObservationQueryWire::GaussianOutcome { .. }
        | ObservationQueryWire::ConditionalInterventionalOutcome { .. }
        | ObservationQueryWire::ThresholdCrossing { .. } => Ok(()),
        ObservationQueryWire::JointMeasuredOutcome { measured, dependence, .. } => {
            if measured.len() < 2
                || measured.len() > 16
                || dependence != "conditionally_independent_given_construct"
            {
                return Err(ObservationQueryArtifactError::Malformed(
                    "joint measurement needs 2-16 indicators and an explicit conditional-independence declaration".into(),
                ));
            }
            let construct = &measured[0].construct;
            let mut ids = std::collections::BTreeSet::new();
            if measured
                .iter()
                .any(|slot| slot.construct != *construct || !ids.insert(slot.indicator_id.as_str()))
            {
                return Err(ObservationQueryArtifactError::Malformed(
                    "joint indicators must be distinct and name one construct".into(),
                ));
            }
            Ok(())
        }
        ObservationQueryWire::DependentMeasuredOutcome { measured, error_law, .. } => {
            let n = error_law.indicator_ids.len();
            if !(2..=16).contains(&n)
                || measured.len() != n
                || error_law.indicator_levels.len() != n
                || error_law.construct_levels.is_empty()
                || error_law.construct_levels.len() > 16
                || error_law.construct.trim().is_empty()
                || error_law.population_id.trim().is_empty()
                || error_law.regime_id.trim().is_empty()
                || error_law.evidence_id.trim().is_empty()
                || error_law.indicator_ids.windows(2).any(|pair| pair[0] >= pair[1])
                || error_law.indicator_ids.iter().any(|id| id.trim().is_empty())
                || error_law.construct_levels.iter().any(|level| level.trim().is_empty())
                || error_law
                    .construct_levels
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != error_law.construct_levels.len()
            {
                return Err(ObservationQueryArtifactError::Malformed(
                    "dependent measurement needs distinct indicators and a complete scoped construct law".into(),
                ));
            }
            let mut cells = 1usize;
            for (index, levels) in error_law.indicator_levels.iter().enumerate() {
                if levels.is_empty()
                    || levels.len() > 16
                    || levels.iter().any(|level| level.trim().is_empty())
                    || levels.iter().collect::<std::collections::BTreeSet<_>>().len()
                        != levels.len()
                    || measured[index].0.as_str() != error_law.indicator_ids[index].as_str()
                    || !levels.contains(&measured[index].1)
                {
                    return Err(ObservationQueryArtifactError::Malformed(
                        "indicator identities, levels or recorded values differ from the dependent-error table".into(),
                    ));
                }
                cells = cells.checked_mul(levels.len()).ok_or(
                    ObservationQueryArtifactError::LimitsExceeded("dependent measurement cells"),
                )?;
                if cells > 65_536 {
                    return Err(ObservationQueryArtifactError::LimitsExceeded(
                        "dependent measurement cells",
                    ));
                }
            }
            if error_law.probability_bits.len() != cells * error_law.construct_levels.len() {
                return Err(ObservationQueryArtifactError::Malformed(
                    "dependent-error table has the wrong number of cells".into(),
                ));
            }
            for row in error_law.probability_bits.chunks_exact(cells) {
                let probabilities: Vec<f64> =
                    row.iter().map(|bits| f64::from_bits(*bits)).collect();
                if probabilities
                    .iter()
                    .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                    || (probabilities.iter().sum::<f64>() - 1.0).abs() > 1e-9
                {
                    return Err(ObservationQueryArtifactError::Malformed(
                        "each construct row must be a normalized joint indicator law".into(),
                    ));
                }
            }
            Ok(())
        }
        ObservationQueryWire::ParameterRedraw { retained, draws, .. } => {
            if retained.len() > MAX_COMPOSED_MECHANISMS || *draws > MAX_REDRAWS {
                return Err(ObservationQueryArtifactError::LimitsExceeded("redraw"));
            }
            for item in retained {
                match item {
                    RetainedMechanismWire::Gaussian { parents, gram_bits, xty_bits, .. } => {
                        if parents.len() >= MAX_RETAINED_REGRESSORS
                            || gram_bits.len() > MAX_RETAINED_REGRESSORS * MAX_RETAINED_REGRESSORS
                            || xty_bits.len() > MAX_RETAINED_REGRESSORS
                        {
                            return Err(ObservationQueryArtifactError::LimitsExceeded(
                                "retained regression",
                            ));
                        }
                    }
                    RetainedMechanismWire::Table { rows, .. } => {
                        if rows.len() > OBSERVATION_QUERY_MAX_RETAINED_ROWS
                            || rows.iter().any(|(config, counts)| {
                                config.len() > MAX_COMPOSED_MECHANISMS
                                    || counts.len()
                                        > antecedent_model::compose::MAX_LEVELS_PER_VARIABLE
                            })
                        {
                            return Err(ObservationQueryArtifactError::LimitsExceeded(
                                "retained table",
                            ));
                        }
                    }
                }
            }
            Ok(())
        }
        ObservationQueryWire::FinitePosterior {
            hypotheses,
            prior_bits,
            observation_levels,
            likelihood_bits,
            observations,
        } => {
            if hypotheses.len() > MAX_BAYES_HYPOTHESES
                || prior_bits.len() > MAX_BAYES_HYPOTHESES
                || observation_levels.len() > MAX_BAYES_OBSERVATION_LEVELS
                || likelihood_bits.len() > MAX_BAYES_HYPOTHESES * MAX_BAYES_OBSERVATION_LEVELS
            {
                return Err(ObservationQueryArtifactError::LimitsExceeded("bayes model"));
            }
            if observations.len() as u64 > MAX_BAYES_UPDATE_WORK {
                return Err(ObservationQueryArtifactError::LimitsExceeded("observations"));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_premises_bounds(
    premises: &MechanismCompositionPremisesWire,
) -> Result<(), ObservationQueryArtifactError> {
    if premises.mechanisms.is_empty() || premises.mechanisms.len() > MAX_COMPOSED_MECHANISMS {
        return Err(ObservationQueryArtifactError::LimitsExceeded("mechanisms"));
    }
    if premises.edges.len() > MAX_COMPOSED_MECHANISMS * MAX_COMPOSED_MECHANISMS {
        return Err(ObservationQueryArtifactError::LimitsExceeded("edges"));
    }
    Ok(())
}

fn refs(pairs: &[(String, String)]) -> Vec<(&str, &str)> {
    pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect()
}

fn direction(name: &str) -> Result<Direction, ObservationQueryArtifactError> {
    Direction::from_name(name)
        .ok_or_else(|| ObservationQueryArtifactError::Malformed(format!("direction `{name}`")))
}

fn bits(values: &[u64]) -> Vec<f64> {
    values.iter().map(|b| f64::from_bits(*b)).collect()
}

fn measured_slot_likelihood(
    measured: &MeasuredSlotWire,
    law: &FiniteLaw,
    spec: &ComposedModelSpec,
) -> Result<Vec<f64>, ObservationQueryArtifactError> {
    let Some(index) = law.variables().iter().position(|v| v == &measured.construct) else {
        return Err(refused(registry_refusal(
            antecedent_core::reason_code!("unknown_variable"),
            antecedent_core::reason_details::observation_query::UNKNOWN_VARIABLE,
            measured.construct.clone(),
            "name a variable of the composed model",
        )));
    };
    let model = model_from_wire(&measured.model)
        .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
    let kind = MeasurementKind::from_name(&measured.indicator_kind).ok_or_else(|| {
        ObservationQueryArtifactError::Malformed(format!(
            "indicator kind `{}`",
            measured.indicator_kind
        ))
    })?;
    let slot = ObservedSlot {
        variable_id: measured.construct.clone(),
        level: measured.recorded_level.clone(),
        provenance: SlotProvenance::Recorded,
        value: SlotValue::Measured(MeasuredIndicator {
            indicator_id: measured.indicator_id.clone(),
            kind,
            model: Some(model),
            positive_level: measured.positive_level.clone(),
            representation: RepresentationVersion {
                coordinate_system_id: measured.representation_system.clone(),
                version: measured.representation_version,
            },
        }),
    };
    let context = ObservationContext {
        population_id: &spec.target.population,
        regime_id: &spec.target.regime,
        arm_id: None,
    };
    measured_likelihood(&slot, &law.levels()[index], &context).map_err(refused)
}

fn evaluate_composed(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::GaussianOutcome { .. } => evaluate_gaussian(spec, query, ctx),
        ObservationQueryWire::ParameterRedraw { .. } => evaluate_redraw(spec, query, ctx),
        _ => evaluate_finite(spec, query, ctx),
    }
}

fn evaluate_gaussian(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let ObservationQueryWire::GaussianOutcome { outcome, interventions, observed } = query else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a Gaussian evaluation was given another query".to_owned(),
        ));
    };
    let model = ComposedLinearGaussian::compile(spec).map_err(composition_error)?;
    let answer =
        condition_gaussian(&model, outcome, &real_refs(interventions), &real_refs(observed), ctx)
            .map_err(query_error)?;
    Ok(ObservationResultWire::GaussianConditional {
        operation: query.operation().to_owned(),
        mean_bits: answer.mean.to_bits(),
        variance_bits: answer.variance.to_bits(),
        evidence_log_density_bits: answer.evidence_log_density.map(f64::to_bits),
    })
}

fn retained_from_wire(item: &RetainedMechanismWire) -> RetainedMechanism {
    match item {
        RetainedMechanismWire::Gaussian {
            variable,
            parents,
            gram_bits,
            xty_bits,
            yty_bits,
            rows,
        } => RetainedMechanism::Gaussian(GaussianSufficient {
            variable: variable.clone(),
            parents: parents.clone(),
            gram: bits(gram_bits),
            xty: bits(xty_bits),
            yty: f64::from_bits(*yty_bits),
            rows: *rows,
        }),
        RetainedMechanismWire::Table { variable, rows } => RetainedMechanism::Table(TableCounts {
            variable: variable.clone(),
            rows: rows
                .iter()
                .map(|(parent_levels, counts)| TableCountRow {
                    parent_levels: parent_levels.clone(),
                    counts: counts.clone(),
                })
                .collect(),
        }),
    }
}

/// How the intervened regime of redrawn draws is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegimeNaming {
    /// v2: `do(v=level,...)` when no variable or level contains `=`, `,`, `(` or `)` (the
    /// names every regime in the library uses, unchanged from v1), else `do[...]` of
    /// JSON-quoted `[variable, level]` pairs, so no text can imitate a separator.
    Framed,
    /// v1 of earlier releases: `do(v=level,...)` joined without framing; kept to verify
    /// artifacts sealed under it.
    Joined,
}

/// A recorded or assigned Gaussian value as coordinate text: its shortest round-trip
/// decimal (distinct for distinct finite values, `-0` for negative zero).
fn real_text(bits: u64) -> String {
    format!("{}", f64::from_bits(bits))
}

/// The regime id of an intervention assignment `(variable, level text)` under `naming`.
fn regime_id_of(mut pairs: Vec<(String, String)>, naming: RegimeNaming) -> String {
    match naming {
        RegimeNaming::Framed => {
            pairs.sort();
            let separator = |text: &str| text.contains(['=', ',', '(', ')']);
            if pairs.iter().any(|(v, level)| separator(v) || separator(level)) {
                let framed = serde_json::to_string(&pairs).unwrap_or_default();
                format!("do{framed}")
            } else {
                // Sorted as text, exactly as v1, so these names keep their v1 regime id.
                let mut joined: Vec<String> =
                    pairs.iter().map(|(v, level)| format!("{v}={level}")).collect();
                joined.sort();
                format!("do({})", joined.join(","))
            }
        }
        RegimeNaming::Joined => {
            let mut set: Vec<String> =
                pairs.iter().map(|(v, level)| format!("{v}={level}")).collect();
            set.sort();
            format!("do({})", set.join(","))
        }
    }
}

/// The quantity coordinates of redrawn draws: one per queried functional of the
/// outcome, at the composition's population, under the intervened regime, with
/// the recorded values as conditioning.
fn draw_quantities(
    spec: &ComposedModelSpec,
    target: &RedrawTargetWire,
    functionals: &[String],
) -> Vec<ScientificQuantity> {
    draw_quantities_named(spec, target, functionals, RegimeNaming::Framed)
}

fn draw_quantities_named(
    spec: &ComposedModelSpec,
    target: &RedrawTargetWire,
    functionals: &[String],
    naming: RegimeNaming,
) -> Vec<ScientificQuantity> {
    let outcome = target.outcome();
    let units = spec
        .mechanisms
        .iter()
        .find(|m| m.variable == outcome)
        .map(|m| m.output_units.clone())
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| "dimensionless".to_owned());
    let mut conditions: Vec<(String, String)> = match target {
        RedrawTargetWire::Gaussian { observed, .. } => {
            observed.iter().map(|(v, b)| (v.clone(), real_text(*b))).collect()
        }
        RedrawTargetWire::Finite { observed, .. } => observed.clone(),
    };
    conditions.sort();
    let regime = if target.has_interventions() {
        let pairs: Vec<(String, String)> = match target {
            RedrawTargetWire::Gaussian { interventions, .. } => {
                interventions.iter().map(|(v, b)| (v.clone(), real_text(*b))).collect()
            }
            RedrawTargetWire::Finite { interventions, .. } => interventions.clone(),
        };
        regime_id_of(pairs, naming)
    } else {
        spec.target.regime.clone()
    };
    functionals
        .iter()
        .map(|functional| ScientificQuantity {
            variable_id: format!("{}:{outcome}", spec.target.population),
            variable_name: outcome.to_owned(),
            role: QuantityRole::Outcome,
            units: units.clone(),
            population_id: spec.target.population.clone(),
            regime_id: regime.clone(),
            horizon: 0,
            functional_id: functional.clone(),
            conditioning: conditions
                .iter()
                .map(|(variable, value)| antecedent_contract::QuantityCondition {
                    variable_id: variable.clone(),
                    value_id: value.clone(),
                })
                .collect(),
            transform_id: "identity".to_owned(),
        })
        .collect()
}

fn digest_error(error: &IoError) -> ObservationQueryArtifactError {
    ObservationQueryArtifactError::Encode(error.to_string())
}

/// The replayed result with the alignment identity an artifact sealed before the framed
/// regime naming carries ([`RegimeNaming::Joined`]); any other result is returned unchanged.
fn legacy_alignment(
    spec: Option<&ComposedModelSpec>,
    query: &ObservationQueryWire,
    result: &ObservationResultWire,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let (
        Some(spec),
        ObservationResultWire::ParameterDraws {
            quantities,
            draws,
            point_bits,
            values_bits,
            reading,
            ..
        },
    ) = (spec, result)
    else {
        return Ok(result.clone());
    };
    let alignment_identity = parameter_draws_distribution_named(
        spec,
        query,
        &ParameterDraws {
            quantities: quantities.clone(),
            draws: *draws as usize,
            values: bits(values_bits),
            point: bits(point_bits),
        },
        RegimeNaming::Joined,
    )?
    .alignment_identity()
    .map_err(|error| digest_error(&error))?;
    Ok(ObservationResultWire::ParameterDraws {
        quantities: quantities.clone(),
        draws: *draws,
        point_bits: point_bits.clone(),
        values_bits: values_bits.clone(),
        alignment_identity,
        reading: reading.clone(),
    })
}

/// The aligned draws of a redraw query as the repo's distribution artifact.
///
/// Every row is one joint realization of the queried quantities (one parameter
/// draw produced them all), so the alignment is [`DrawAlignment::Joint`]. The
/// source is the identity of the query on these premises, the snapshot is the
/// digest of the retained data and priors, and the stream is the seed.
fn parameter_draws_distribution(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    draws: &ParameterDraws,
) -> Result<DistributionArtifact, ObservationQueryArtifactError> {
    parameter_draws_distribution_named(spec, query, draws, RegimeNaming::Framed)
}

fn parameter_draws_distribution_named(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    draws: &ParameterDraws,
    naming: RegimeNaming,
) -> Result<DistributionArtifact, ObservationQueryArtifactError> {
    let ObservationQueryWire::ParameterRedraw { target, retained, prior, draws: n, seed } = query
    else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a distribution was requested for a query that redraws nothing".to_owned(),
        ));
    };
    let premises = MechanismCompositionPremisesWire::from_spec(spec);
    let basis = inference_basis_of(spec);
    let source = observation_query_identity(Some(&premises), query, basis.as_str())
        .map_err(|error| digest_error(&error))?;
    let snapshot = crate::digest_hex(
        IdentityDomain::Program,
        &("observation_query_retained_data_v1", retained, prior),
    )
    .map_err(|error| digest_error(&error))?;
    let contract =
        crate::digest_hex(IdentityDomain::Program, &("observation_query_premises_v1", &premises))
            .map_err(|error| digest_error(&error))?;
    let quantities = draw_quantities_named(spec, target, &draws.quantities, naming);
    let semantic = if target.has_interventions() {
        DistributionMeaningWire::InterventionalPredictive
    } else {
        DistributionMeaningWire::PosteriorPredictive
    };
    let identity = DistributionIdentity::new(
        semantic,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: source,
            provider_id: "observation_query.parameter_redraw".to_owned(),
            rng_id: format!("splitmix64.derived_stream:seed={seed}:draws={n}"),
            snapshot_id: snapshot,
            causal_contract_id: contract,
        },
    )
    .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".to_owned(), "quantity".to_owned()],
            shape: [draws.draws, draws.width()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws.values.clone(),
    )
    .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))
}

fn evaluate_redraw(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let ObservationQueryWire::ParameterRedraw { target, retained, prior, draws, seed } = query
    else {
        return Err(ObservationQueryArtifactError::Malformed(
            "a redraw evaluation was given another query".to_owned(),
        ));
    };
    let retained: Vec<RetainedMechanism> = retained.iter().map(retained_from_wire).collect();
    let drawn = redraw_conditional(
        spec,
        &target.to_target(),
        &retained,
        &prior.to_prior(),
        *draws,
        *seed,
        ctx,
    )
    .map_err(query_error)?;
    if drawn.width() > MAX_REDRAW_QUANTITIES {
        return Err(ObservationQueryArtifactError::LimitsExceeded("quantities"));
    }
    let alignment_identity = parameter_draws_distribution(spec, query, &drawn)?
        .alignment_identity()
        .map_err(|error| digest_error(&error))?;
    Ok(ObservationResultWire::ParameterDraws {
        quantities: drawn.quantities.clone(),
        draws: *draws,
        point_bits: drawn.point.iter().map(|v| v.to_bits()).collect(),
        values_bits: drawn.values.iter().map(|v| v.to_bits()).collect(),
        alignment_identity,
        reading: PARAMETER_DRAWS_READING.to_owned(),
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per composed-model query, each a short call into the numerical layer"
)]
fn evaluate_finite(
    spec: &ComposedModelSpec,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    let model = ComposedFinite::compile(spec).map_err(composition_error)?;
    match query {
        ObservationQueryWire::DependentMeasuredOutcome {
            outcome,
            interventions,
            observed,
            measured,
            error_law,
        } => {
            let observed = refs(observed);
            let interventions_ref = refs(interventions);
            let law =
                intervened_law(&model, &interventions_ref, &observed, ctx).map_err(query_error)?;
            if interventions_ref
                .iter()
                .any(|(variable, _)| *variable == error_law.construct.as_str())
            {
                return Err(refused(mixed_conditioning_refusal(&error_law.construct)));
            }
            let mut regime = interventions.clone();
            regime.sort();
            let expected_regime = if regime.is_empty() {
                spec.target.regime.clone()
            } else {
                format!(
                    "do({})",
                    regime
                        .iter()
                        .map(|(v, level)| format!("{v}={level}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            if error_law.population_id != spec.target.population
                || error_law.regime_id != expected_regime
            {
                return Err(ObservationQueryArtifactError::Malformed(
                    "dependent measurement law is outside the query population or regime".into(),
                ));
            }
            let index = law
                .variables()
                .iter()
                .position(|variable| variable == &error_law.construct)
                .ok_or_else(|| {
                    ObservationQueryArtifactError::Malformed(
                        "dependent measurement construct is absent from the finite model".into(),
                    )
                })?;
            if law.levels()[index].as_slice() != error_law.construct_levels.as_slice() {
                return Err(ObservationQueryArtifactError::Malformed(
                    "dependent measurement construct levels differ from the finite model".into(),
                ));
            }
            let mut cell = 0usize;
            for ((id, recorded), (table_id, levels)) in
                measured.iter().zip(error_law.indicator_ids.iter().zip(&error_law.indicator_levels))
            {
                if id != table_id {
                    return Err(ObservationQueryArtifactError::Malformed(
                        "dependent measurement indicator order differs".into(),
                    ));
                }
                let position =
                    levels.iter().position(|level| level == recorded).ok_or_else(|| {
                        ObservationQueryArtifactError::Malformed(
                            "dependent measurement recorded level is absent".into(),
                        )
                    })?;
                cell = cell * levels.len() + position;
            }
            let cells = error_law.probability_bits.len() / error_law.construct_levels.len();
            let likelihood: Vec<f64> = (0..error_law.construct_levels.len())
                .map(|row| f64::from_bits(error_law.probability_bits[row * cells + cell]))
                .collect();
            let answer = condition_on_measured(
                &law,
                outcome,
                &error_law.construct,
                &observed,
                &likelihood,
                ctx,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::Conditional {
                operation: query.operation().to_owned(),
                evidence_probability_bits: answer.evidence_probability.to_bits(),
                levels: level_bits(&answer.outcome),
                construct_posterior: Some(level_bits(&answer.construct_posterior)),
            })
        }
        ObservationQueryWire::JointMeasuredOutcome {
            outcome,
            interventions,
            observed,
            measured,
            ..
        } => {
            let observed = refs(observed);
            let interventions = refs(interventions);
            let law =
                intervened_law(&model, &interventions, &observed, ctx).map_err(query_error)?;
            let construct = &measured[0].construct;
            if interventions.iter().any(|(variable, _)| *variable == construct) {
                return Err(refused(mixed_conditioning_refusal(construct)));
            }
            let mut likelihood = measured_slot_likelihood(&measured[0], &law, spec)?;
            for indicator in &measured[1..] {
                let next = measured_slot_likelihood(indicator, &law, spec)?;
                if next.len() != likelihood.len() {
                    return Err(ObservationQueryArtifactError::Malformed(
                        "joint indicator likelihoods have different construct levels".into(),
                    ));
                }
                for (joint, factor) in likelihood.iter_mut().zip(next) {
                    *joint *= factor;
                }
            }
            let answer =
                condition_on_measured(&law, outcome, construct, &observed, &likelihood, ctx)
                    .map_err(query_error)?;
            Ok(ObservationResultWire::Conditional {
                operation: query.operation().to_owned(),
                evidence_probability_bits: answer.evidence_probability.to_bits(),
                levels: level_bits(&answer.outcome),
                construct_posterior: Some(level_bits(&answer.construct_posterior)),
            })
        }
        ObservationQueryWire::ConditionalInterventionalOutcome {
            outcome,
            interventions,
            observed,
            measured,
        } => {
            let observed = refs(observed);
            let interventions = refs(interventions);
            let law =
                intervened_law(&model, &interventions, &observed, ctx).map_err(query_error)?;
            if let Some(m) = measured {
                if interventions.iter().any(|(v, _)| *v == m.construct) {
                    return Err(refused(mixed_conditioning_refusal(&m.construct)));
                }
                let likelihood = measured_slot_likelihood(m, &law, spec)?;
                let answer =
                    condition_on_measured(&law, outcome, &m.construct, &observed, &likelihood, ctx)
                        .map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: query.operation().to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.outcome),
                    construct_posterior: Some(level_bits(&answer.construct_posterior)),
                })
            } else {
                let answer = condition(&law, outcome, &observed, ctx).map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: query.operation().to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.levels),
                    construct_posterior: None,
                })
            }
        }
        ObservationQueryWire::ConditionalOutcome { outcome, observed, measured } => {
            let law = model.law(&[], ctx).map_err(composition_error)?;
            let observed = refs(observed);
            if let Some(m) = measured {
                let likelihood = measured_slot_likelihood(m, &law, spec)?;
                let answer =
                    condition_on_measured(&law, outcome, &m.construct, &observed, &likelihood, ctx)
                        .map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: "conditioning".to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.outcome),
                    construct_posterior: Some(level_bits(&answer.construct_posterior)),
                })
            } else {
                let answer = condition(&law, outcome, &observed, ctx).map_err(query_error)?;
                Ok(ObservationResultWire::Conditional {
                    operation: "conditioning".to_owned(),
                    evidence_probability_bits: answer.evidence_probability.to_bits(),
                    levels: level_bits(&answer.levels),
                    construct_posterior: None,
                })
            }
        }
        ObservationQueryWire::InterventionalOutcome { outcome, interventions } => {
            let levels =
                intervene(&model, outcome, &refs(interventions), ctx).map_err(query_error)?;
            Ok(ObservationResultWire::Interventional {
                operation: "intervention".to_owned(),
                levels: level_bits(&levels),
            })
        }
        ObservationQueryWire::ConditioningVersusIntervention { outcome, assignment } => {
            let pair = conditioning_and_intervention(&model, outcome, &refs(assignment), ctx)
                .map_err(query_error)?;
            Ok(ObservationResultWire::Paired {
                conditioning_evidence_bits: pair.conditioning.evidence_probability.to_bits(),
                conditioning: level_bits(&pair.conditioning.levels),
                intervention: level_bits(&pair.intervention),
                max_abs_difference_bits: pair.max_abs_difference.to_bits(),
            })
        }
        ObservationQueryWire::ThresholdCrossing {
            outcome,
            outcome_level,
            signal,
            background,
            threshold_bits,
            direction: direction_name,
        } => {
            let law = model.law(&[], ctx).map_err(composition_error)?;
            let crossing = threshold_crossing_observations(
                &law,
                outcome,
                outcome_level,
                signal,
                &refs(background),
                f64::from_bits(*threshold_bits),
                direction(direction_name)?,
                ctx,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::ThresholdCrossing {
                crossing_probability_bits: crossing.crossing_probability.to_bits(),
                rows: crossing
                    .levels
                    .iter()
                    .map(|r| {
                        (
                            r.level.clone(),
                            r.probability.to_bits(),
                            r.conditional.to_bits(),
                            r.crosses,
                        )
                    })
                    .collect(),
            })
        }
        _ => Err(ObservationQueryArtifactError::Malformed(
            "a declared-model query was given composed premises".to_owned(),
        )),
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "one arm per declared-model query, each a short call into the numerical layer"
)]
fn evaluate_declared(
    query: &ObservationQueryWire,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    match query {
        ObservationQueryWire::FinitePosterior {
            hypotheses,
            prior_bits,
            observation_levels,
            likelihood_bits,
            observations,
        } => {
            let model = FiniteBayesModel::new(
                hypotheses.clone(),
                bits(prior_bits),
                observation_levels.clone(),
                bits(likelihood_bits),
            )
            .map_err(query_error)?;
            let observations: Vec<&str> = observations.iter().map(String::as_str).collect();
            let posterior = model.posterior(&observations).map_err(query_error)?;
            Ok(ObservationResultWire::FinitePosterior {
                posterior: level_bits(&posterior.hypotheses),
                log_evidence_bits: posterior.log_evidence.to_bits(),
                evidence_probability_bits: posterior.evidence_probability.to_bits(),
                predictive: level_bits(&posterior.predictive),
            })
        }
        ObservationQueryWire::ConjugatePosterior { alpha_bits, beta_bits, successes, failures } => {
            let prior = BetaBernoulli::new(f64::from_bits(*alpha_bits), f64::from_bits(*beta_bits))
                .map_err(query_error)?;
            let posterior = prior.update(*successes, *failures).map_err(query_error)?;
            Ok(ObservationResultWire::ConjugatePosterior {
                alpha_bits: posterior.alpha().to_bits(),
                beta_bits: posterior.beta().to_bits(),
                predictive_success_bits: posterior.mean().to_bits(),
            })
        }
        ObservationQueryWire::RequiredBayesFactor {
            prior_bits,
            level_bits: target_bits,
            direction: direction_name,
        } => {
            let required = required_bayes_factor(
                f64::from_bits(*prior_bits),
                f64::from_bits(*target_bits),
                direction(direction_name)?,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredBayesFactor {
                factor_bits: required.factor.to_bits(),
                already_satisfied: required.already_satisfied,
            })
        }
        ObservationQueryWire::RequiredObservationsBinarySignal {
            prior_bits,
            level_bits: target_bits,
            direction: direction_name,
            rate_given_hypothesis_bits,
            rate_given_complement_bits,
            max_observations,
        } => {
            let required = observations_required_binary_signal(
                f64::from_bits(*prior_bits),
                f64::from_bits(*target_bits),
                direction(direction_name)?,
                f64::from_bits(*rate_given_hypothesis_bits),
                f64::from_bits(*rate_given_complement_bits),
                *max_observations,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredObservations {
                observations: required.observations,
                value_bits: required.value_at_requirement.to_bits(),
                already_satisfied: required.already_satisfied,
                bayes_factor_bits: required.bayes_factor_per_observation.map(f64::to_bits),
            })
        }
        ObservationQueryWire::RequiredObservationsConjugate {
            alpha_bits,
            beta_bits,
            level_bits: target_bits,
            direction: direction_name,
            max_observations,
        } => {
            let prior = BetaBernoulli::new(f64::from_bits(*alpha_bits), f64::from_bits(*beta_bits))
                .map_err(query_error)?;
            let required = observations_required_conjugate(
                &prior,
                f64::from_bits(*target_bits),
                direction(direction_name)?,
                *max_observations,
            )
            .map_err(query_error)?;
            Ok(ObservationResultWire::RequiredObservations {
                observations: required.observations,
                value_bits: required.value_at_requirement.to_bits(),
                already_satisfied: required.already_satisfied,
                bayes_factor_bits: None,
            })
        }
        _ => Err(ObservationQueryArtifactError::Malformed(
            "a composed-model query was given no composed premises".to_owned(),
        )),
    }
}

/// How the value of a query on `spec` is to be read: declared only when every
/// mechanism was written down by the analyst.
#[must_use]
pub fn inference_basis_of(spec: &ComposedModelSpec) -> InferenceBasis {
    let origins: Vec<MechanismOriginKind> = spec
        .mechanisms
        .iter()
        .map(|m| match m.provenance.origin {
            MechanismOrigin::Declared => MechanismOriginKind::Declared,
            MechanismOrigin::Fitted => MechanismOriginKind::Fitted,
            MechanismOrigin::ExternalBinding => MechanismOriginKind::ExternalBinding,
        })
        .collect();
    InferenceBasis::from_origins(&origins)
}

fn basis_from_name(name: &str) -> Result<InferenceBasis, ObservationQueryArtifactError> {
    [InferenceBasis::DeclaredModel, InferenceBasis::SampleFittedPointOnly]
        .into_iter()
        .find(|b| b.as_str() == name)
        .ok_or_else(|| {
            ObservationQueryArtifactError::Malformed(format!("inference basis `{name}`"))
        })
}

/// Run `query`: against the composed model when it needs one, against its own
/// declared model otherwise.
///
/// # Errors
/// Incoherent mechanisms (structured refusals), a refused request (each with a
/// registered code and remedy, including an unreachable level and a measured
/// value without an applicable model), a shape that disagrees with the
/// premises, or cancellation.
pub fn evaluate_observation_query(
    spec: Option<&ComposedModelSpec>,
    query: &ObservationQueryWire,
    ctx: &ExecutionContext,
) -> Result<ObservationResultWire, ObservationQueryArtifactError> {
    check_query_bounds(query)?;
    if ctx.cancellation.is_cancelled() {
        return Err(ObservationQueryArtifactError::Cancelled);
    }
    match (query.needs_composed_model(), spec) {
        (true, Some(spec)) => evaluate_composed(spec, query, ctx),
        (false, None) => evaluate_declared(query),
        (true, None) => Err(ObservationQueryArtifactError::Malformed(
            "the query needs composed premises and none were supplied".to_owned(),
        )),
        (false, Some(_)) => Err(ObservationQueryArtifactError::Malformed(
            "a declared-model query was given composed premises".to_owned(),
        )),
    }
}

/// The format version and feature marker a query is carried under.
const fn format_of(query: &ObservationQueryWire) -> (u32, &'static str) {
    if query.is_dependent_measured() {
        (OBSERVATION_QUERY_DEPENDENT_MEASURED_VERSION, OBSERVATION_QUERY_DEPENDENT_MEASURED_FEATURE)
    } else if query.is_joint_measured() {
        (OBSERVATION_QUERY_JOINT_MEASURED_VERSION, OBSERVATION_QUERY_JOINT_MEASURED_FEATURE)
    } else if query.is_general() {
        (OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION, OBSERVATION_QUERY_GENERAL_FEATURE)
    } else {
        (OBSERVATION_QUERY_ARTIFACT_VERSION, OBSERVATION_QUERY_ARTIFACT_FEATURE)
    }
}

impl ObservationQueryArtifactWire {
    /// The aligned draws of a parameter-redraw claim as the repo's distribution
    /// artifact, rebuilt from the stored result.
    ///
    /// # Errors
    /// A claim that is not a parameter redraw, premises that no longer compose,
    /// or draws that do not form a valid distribution.
    pub fn distribution(&self) -> Result<DistributionArtifact, ObservationQueryArtifactError> {
        let (
            Some(premises),
            ObservationResultWire::ParameterDraws {
                quantities,
                draws,
                values_bits,
                point_bits,
                ..
            },
        ) = (&self.premises, &self.result)
        else {
            return Err(ObservationQueryArtifactError::Malformed(
                "the claim carries no parameter draws".to_owned(),
            ));
        };
        let spec = premises
            .to_spec()
            .map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?;
        parameter_draws_distribution(
            &spec,
            &self.query,
            &ParameterDraws {
                quantities: quantities.clone(),
                draws: *draws as usize,
                values: bits(values_bits),
                point: bits(point_bits),
            },
        )
    }

    /// Execute and seal an artifact for `query`.
    ///
    /// `spec` is the composed model for a query that runs on one and `None`
    /// otherwise. For a composed model the reading is derived from the
    /// mechanisms' origins and `declared_basis` is ignored; a declared-model
    /// query takes `declared_basis` from the caller, who states whether the
    /// prior and likelihood were written down or fitted to a sample.
    ///
    /// # Errors
    /// Incoherent mechanisms, a refused request, a failed execution or an
    /// encoding failure.
    pub fn seal(
        spec: Option<&ComposedModelSpec>,
        query: ObservationQueryWire,
        declared_basis: InferenceBasis,
        ctx: &ExecutionContext,
    ) -> Result<Self, ObservationQueryArtifactError> {
        let query = query.canonical();
        let premises = spec.map(MechanismCompositionPremisesWire::from_spec);
        if let Some(p) = &premises {
            check_premises_bounds(p)?;
        }
        let canonical = match &premises {
            Some(p) => Some(
                p.to_spec().map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?,
            ),
            None => None,
        };
        let result = evaluate_observation_query(canonical.as_ref(), &query, ctx)?;
        let basis = match &canonical {
            Some(spec) => inference_basis_of(spec),
            None => declared_basis,
        };
        let inference_basis = basis.as_str().to_owned();
        let identity = observation_query_identity(premises.as_ref(), &query, &inference_basis)
            .map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))?;
        let (version, feature) = format_of(&query);
        Ok(Self {
            version,
            required_features: vec![feature.to_owned()],
            premises,
            query,
            inference_basis,
            result,
            identity,
        })
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ObservationQueryArtifactError> {
        if let Some(p) = &self.premises {
            check_premises_bounds(p)?;
        }
        check_query_bounds(&self.query)?;
        crate::to_cbor(self).map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, ObservationQueryArtifactError> {
        let wire: Self = crate::decode_with_version(bytes, |version: u32| {
            if version == OBSERVATION_QUERY_ARTIFACT_VERSION
                || version == OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION
                || version == OBSERVATION_QUERY_JOINT_MEASURED_VERSION
                || version == OBSERVATION_QUERY_DEPENDENT_MEASURED_VERSION
            {
                Ok(())
            } else {
                Err(ObservationQueryArtifactError::UnsupportedVersion { version })
            }
        })?;
        wire.check_format()?;
        Ok(wire)
    }

    /// As [`Self::decode`], admitting only the artifact the consumer retained.
    ///
    /// The bytes bind to `retained`
    /// ([`crate::retained_artifact::bind_retained_artifact`]) before they are read.
    ///
    /// # Errors
    ///
    /// As [`Self::decode`], and `external_binding_mismatch` when the bytes are not the
    /// retained artifact.
    pub fn decode_expecting(
        bytes: &[u8],
        retained: &crate::retained_artifact::RetainedArtifactIdentity,
    ) -> Result<Self, ObservationQueryArtifactError> {
        crate::retained_artifact::load_expecting(bytes, retained, Self::decode)
    }

    /// The version, the feature marker and the query variant must agree: a
    /// version-1 artifact holds only the version-1 queries and a version-2
    /// artifact only the general ones.
    fn check_format(&self) -> Result<(), ObservationQueryArtifactError> {
        let (version, feature) = format_of(&self.query);
        if self.version != version {
            return Err(ObservationQueryArtifactError::UnsupportedVersion {
                version: self.version,
            });
        }
        if self.required_features != [feature] {
            return Err(ObservationQueryArtifactError::UnsupportedSemantics("required features"));
        }
        Ok(())
    }

    /// Verify and replay.
    ///
    /// `expected_identity`, when given, is the identity the caller holds
    /// independently of the artifact.
    ///
    /// # Errors
    /// Any of the refusals of [`ObservationQueryArtifactError`].
    pub fn consume(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedObservationQuery, ObservationQueryArtifactError> {
        self.check_format()?;
        if let Some(p) = &self.premises {
            check_premises_bounds(p)?;
        }
        check_query_bounds(&self.query)?;
        let recomputed =
            observation_query_identity(self.premises.as_ref(), &self.query, &self.inference_basis)
                .map_err(|e| ObservationQueryArtifactError::Encode(e.to_string()))?;
        if recomputed != self.identity {
            return Err(ObservationQueryArtifactError::IdentityMismatch);
        }
        if let Some(expected) = expected_identity
            && expected != recomputed
        {
            return Err(ObservationQueryArtifactError::IdentityMismatch);
        }
        let stored_basis = basis_from_name(&self.inference_basis)?;
        let spec = match &self.premises {
            Some(p) => Some(
                p.to_spec().map_err(|e| ObservationQueryArtifactError::Malformed(e.to_string()))?,
            ),
            None => None,
        };
        if let Some(spec) = &spec {
            // The reading of a composed model is a function of its premises;
            // a stored reading that differs is an edit, not a declaration.
            if inference_basis_of(spec) != stored_basis {
                return Err(ObservationQueryArtifactError::IdentityMismatch);
            }
        }
        let result = evaluate_observation_query(spec.as_ref(), &self.query, ctx)?;
        if !self.result.replays_as(&result)
            && !self.result.replays_as(&legacy_alignment(spec.as_ref(), &self.query, &result)?)
        {
            return Err(ObservationQueryArtifactError::ResultMismatch);
        }
        Ok(ConsumedObservationQuery {
            identity: recomputed,
            operation: self.query.operation().to_owned(),
            inference_basis: self.inference_basis.clone(),
            result,
        })
    }
}

#[cfg(test)]
mod identity_site_tests {
    use super::*;
    use crate::identity_site_laws::{multiset_laws, set_laws};

    fn pairs(items: &[(&str, &str)]) -> Vec<(String, String)> {
        items.iter().map(|(v, l)| ((*v).to_owned(), (*l).to_owned())).collect()
    }

    /// Site `io_core.observation_query_artifact.draw_quantities`: the intervened regime is a
    /// set of assignments, and its id is injective (v2): a variable or level text containing
    /// a separator can no longer imitate two assignments, while names without separators keep
    /// their v1 id.
    #[test]
    fn site_draw_quantities_regime_id_is_a_framed_set() {
        let assignments = pairs(&[("a", "1"), ("b", "2"), ("c", "x")]);
        set_laws(&assignments, |items| {
            let mut distinct = items.to_vec();
            distinct.sort();
            distinct.dedup();
            Some(regime_id_of(distinct, RegimeNaming::Framed))
        });
        // v1 collision: one variable named `a=1,b` set to 2 reads as two assignments.
        let crafted = pairs(&[("a=1,b", "2")]);
        let honest = pairs(&[("a", "1"), ("b", "2")]);
        assert_eq!(
            regime_id_of(crafted.clone(), RegimeNaming::Joined),
            regime_id_of(honest.clone(), RegimeNaming::Joined)
        );
        assert_ne!(
            regime_id_of(crafted, RegimeNaming::Framed),
            regime_id_of(honest.clone(), RegimeNaming::Framed)
        );
        // Plain names keep the v1 id (sorted as text: `a1` before `a`).
        let plain = pairs(&[("a", "1"), ("a1", "2")]);
        assert_eq!(regime_id_of(plain.clone(), RegimeNaming::Framed), "do(a1=2,a=1)");
        assert_eq!(
            regime_id_of(plain.clone(), RegimeNaming::Framed),
            regime_id_of(plain, RegimeNaming::Joined)
        );
        // Every separator-bearing spelling of these parts is told apart.
        let texts = ["a", "a=", "=a", "a,", ",a", "(", ")", "a=1", "1,b"];
        let mut seen = std::collections::BTreeMap::new();
        for variable in texts {
            for level in texts {
                let id = regime_id_of(pairs(&[(variable, level)]), RegimeNaming::Framed);
                if let Some(previous) = seen.insert(id.clone(), (variable, level)) {
                    panic!("{previous:?} and {:?} share {id}", (variable, level));
                }
            }
        }
        // Gaussian levels render by shortest round trip; `-0.0` and `0.0` differ.
        assert_ne!(real_text((-0.0f64).to_bits()), real_text(0.0f64.to_bits()));
    }

    fn gaussian(observed: &[(String, u64)]) -> ObservationQueryWire {
        ObservationQueryWire::GaussianOutcome {
            outcome: "y".into(),
            interventions: vec![("a".into(), 1.0f64.to_bits())],
            observed: observed.to_vec(),
        }
    }

    /// Site `io_core.observation_query_artifact.wire.ObservationQueryWire.canonical`: every
    /// assignment list is a set (canonical form v2: an exact repeat is written once; one
    /// variable with two levels is refused by the evaluator).
    #[test]
    fn site_observation_query_canonical_assignment_lists_are_sets() {
        let observed: Vec<(String, u64)> = vec![
            ("m".into(), 0.5f64.to_bits()),
            ("w".into(), 1.5f64.to_bits()),
            ("z".into(), (-2.0f64).to_bits()),
        ];
        set_laws(&observed, |items| {
            observation_query_identity(None, &gaussian(items).canonical(), "declared").ok()
        });
        let background = pairs(&[("b", "0"), ("c", "1"), ("d", "1")]);
        set_laws(&background, |items| {
            let query = ObservationQueryWire::ThresholdCrossing {
                outcome: "y".into(),
                outcome_level: "1".into(),
                signal: "s".into(),
                background: items.to_vec(),
                threshold_bits: 0.5f64.to_bits(),
                direction: "above".into(),
            };
            observation_query_identity(None, &query.canonical(), "declared").ok()
        });
        // Retained table rows are a weighted dataset: a repeated row is more data.
        let parent = |level: &str| vec![("x".to_owned(), level.to_owned())];
        let rows: Vec<(Vec<(String, String)>, Vec<(String, u64)>)> = vec![
            (parent("0"), vec![("0".into(), 3), ("1".into(), 1)]),
            (parent("1"), vec![("0".into(), 1), ("1".into(), 2)]),
            (parent("2"), vec![("1".into(), 4)]),
        ];
        multiset_laws(&rows, |items| {
            let mut query = ObservationQueryWire::ParameterRedraw {
                target: RedrawTargetWire::Finite {
                    outcome: "y".into(),
                    interventions: Vec::new(),
                    observed: Vec::new(),
                },
                retained: vec![RetainedMechanismWire::Table {
                    variable: "y".into(),
                    rows: items.to_vec(),
                }],
                prior: RedrawPriorWire {
                    dirichlet_alpha_bits: 1.0f64.to_bits(),
                    coefficient_precision_bits: 1.0f64.to_bits(),
                    variance_shape_bits: 1.0f64.to_bits(),
                    variance_scale_bits: 1.0f64.to_bits(),
                },
                draws: 10,
                seed: 1,
            };
            query = query.canonical();
            observation_query_identity(None, &query, "declared").ok()
        });
    }
}

#[cfg(test)]
mod retained_tests {
    use super::*;
    use crate::retained_artifact::{RETAINED_ARTIFACT_REFUSAL, RetainedArtifactIdentity};

    #[test]
    fn observation_query_artifact_expecting_binds_the_retained_bytes() {
        let bytes: &[u8] = b"not an artifact";
        let retained = RetainedArtifactIdentity::of_bytes(bytes);
        let other = RetainedArtifactIdentity::of_bytes(b"another artifact");
        // ObservationQueryArtifactWire::decode_expecting
        assert_eq!(
            format!("{:?}", ObservationQueryArtifactWire::decode_expecting(bytes, &retained).err()),
            format!("{:?}", ObservationQueryArtifactWire::decode(bytes).err()),
        );
        let refused =
            format!("{:?}", ObservationQueryArtifactWire::decode_expecting(bytes, &other).err());
        assert!(refused.contains(RETAINED_ARTIFACT_REFUSAL), "{refused}");
    }
}
