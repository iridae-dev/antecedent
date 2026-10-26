// SPDX-License-Identifier: MIT OR Apache-2.0
//! Identity-bound evidence and decision composition of native cyclic products.
//! Numerical roots, posteriors, bootstrap laws and selected algebraic sets keep
//! their distinct meanings. No adapter performs implicit root selection, drops
//! failed draw slots or upgrades a specified model to graphical identification.
use super::{PreparedStudy, cyclic::CyclicFamilyPreparedState};
use antecedent_contract::{
    EvidenceBinding, EvidenceCalibration, EvidenceEnvelope, EvidenceOrigins, EvidencePort,
    EvidenceSupport, QuantityRole, ScientificQuantity, UncertaintyAxes, WorldTime,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, SupportStatus};
use antecedent_io::{
    cyclic_family_artifact::{
        CyclicFamilyAnswer as A, CyclicFamilyReport, CyclicFamilyRequest as R,
        consume_cyclic_family_with_request,
    },
    cyclic_system_artifact::{
        CyclicArtifactError, MAX_CYCLIC_ARTIFACT_BYTES, cyclic_decode_admission,
        cyclic_serialized_size,
    },
    distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    },
    quantity_wire::{DistributionMeaningWire, ScientificQuantityWire},
};
use antecedent_model::{equation_system::Interval, nonlinear_equilibrium::EquilibriumOutcome};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Declared physical variable identities and units; never inferred from a graph index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceSchema {
    /// Schema/model-system identity shared by the variables.
    pub id: String,
    /// Target population, explicit even for a specified physical system.
    pub population: String,
    /// Stable variable names and units in endogenous coordinate order.
    pub variables: Vec<(String, String)>,
}
/// Typed native output coordinate; array positions are never accepted as scientific labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicEvidenceSelector {
    /// Prespecified structural action and noise row, or its abducted factual response.
    Action {
        /// World index or stable declared world identity.
        world: usize,
        /// Aligned explicit disturbance row; zero for an abduction posterior.
        noise_row: usize,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Read the abducted factual response instead of a supplied-noise row.
        counterfactual: bool,
    },
    /// Aligned posterior/bootstrap response column. Noise row applies only to parameter/ bootstrap products.
    Response {
        /// World index or stable declared world identity.
        world: usize,
        /// Aligned explicit disturbance row; zero for an abduction posterior.
        noise_row: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// State of a deterministic ODE path.
    Ode {
        /// Observation-time index; None denotes stationarity.
        time: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// A stochastic-process moment; absence of time denotes stationarity.
    OuMean {
        /// Observation-time index; None denotes stationarity.
        time: Option<usize>,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// Row-major covariance of the same declared stochastic law.
    OuCovariance {
        /// Observation-time index; None denotes stationarity.
        time: Option<usize>,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Second dense endogenous coordinate of the covariance.
        other: usize,
    },
    /// A stationary stochastic moment after a named equation-intervention world.
    OuRegime {
        /// Declared equation-replacement world index.
        world: usize,
        /// Endogenous coordinate.
        variable: usize,
        /// Second coordinate for a covariance; None selects the mean.
        other: Option<usize>,
    },
    /// Certified enclosure of one root; this selector supplies no equilibrium probabilities.
    Root {
        /// Certified root enclosure index, with no probability implied.
        root: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// Outer solution set, including all unresolved boxes without a selection law.
    SolutionSet {
        /// Endogenous coordinate of the bounded equation system.
        variable: usize,
    },
    /// Theorem-derived finite-law functional evaluation.
    Functional {
        /// Retained identified-functional assignment index.
        assignment: usize,
    },
    /// An actual likelihood of a fitted model, or one finite polynomial candidate.
    Likelihood {
        /// Finite likelihood candidate index; None denotes the fitted continuous model.
        candidate: Option<usize>,
    },
    /// A graphical derivation/PAG. Qualitative evidence supplies no numeric outcome law.
    Graph,
    /// Coupled finite-law polynomial outcomes with mass and numerical enclosures retained.
    PolynomialWorld {
        /// World index or stable declared world identity.
        world: String,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// The interventional law within one recovered draw: its equilibrium mean,
    /// with its variance retained as a diagnostic. Not parameter uncertainty.
    RecoveredDraw {
        /// World position in the request.
        world: usize,
        /// Draw position in the law.
        draw: usize,
        /// Dense endogenous coordinate.
        variable: usize,
    },
    /// One typed world of a specified model, by scientific world id.
    World {
        /// Scientific world id.
        world: u32,
        /// Dense endogenous coordinate.
        variable: usize,
        /// Observation-time index of a trajectory; absent for an equilibrium.
        time: Option<usize>,
    },
    /// One declared quantity of a fitted continuous-time family, across the
    /// replicates, draws, or delta-method law of its declared uncertainty method.
    DynamicQuantity {
        /// Position in the request's quantity list.
        quantity: usize,
    },
}
/// One coordinate's native scientific meaning and original aligned payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceColumn {
    /// Full semantic scientific coordinate, derived from the selector and schema.
    pub quantity: ScientificQuantityWire,
    /// Existing contract evidence-envelope identity.
    pub envelope_identity: String,
    /// Existing typed evidence-port spelling.
    pub port: String,
    /// Independent uncertainty axes; an empty list explicitly means none supplied.
    pub uncertainty_axes: Vec<String>,
    /// Original aligned samples; None retains a failed/undefined slot.
    pub samples: Vec<Option<f64>>,
    /// Numerical enclosures, never relabeled as confidence/credible intervals.
    pub numerical_enclosures: Vec<Option<Interval>>,
    /// Probability bounds when supplied by the finite joint selection/parameter law.
    pub mass_bounds: Option<Vec<antecedent_estimate::polynomial_scm::MassBounds>>,
    /// Actual semantic time and world identifiers, not merely a plotting index.
    pub world: String,
    /// Explicit initial-value time or stationary marker.
    pub time: String,
    /// Numerical/selection failures and qualitative/likelihood details.
    pub diagnostics: serde_json::Value,
}
/// Checked, replayable projection of a native cyclic source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicEvidenceReport {
    /// Complete source, target schema and projection identity.
    pub identity: String,
    /// Independently retained original scientific-source identity.
    pub source_identity: String,
    /// The specified-model/identification distinction remains explicit.
    pub scope: String,
    /// Every original requested coordinate and unresolved sample slot.
    pub columns: Vec<CyclicEvidenceColumn>,
    /// Actual retained source diagnostics, including convergence and failed mass.
    pub source: CyclicFamilyReport,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u16,
    #[serde(with = "source_bytes")]
    source: Vec<u8>,
    source_identity: String,
    schema: CyclicEvidenceSchema,
    selectors: Vec<CyclicEvidenceSelector>,
    report: CyclicEvidenceReport,
}
mod source_bytes {
    pub fn serialize<S: serde::Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        struct Bytes;
        impl serde::de::Visitor<'_> for Bytes {
            type Value = Vec<u8>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("bounded original cyclic source bytes")
            }
            fn visit_bytes<E: serde::de::Error>(self, value: &[u8]) -> Result<Vec<u8>, E> {
                if value.len() > super::MAX_CYCLIC_ARTIFACT_BYTES {
                    return Err(E::custom("source byte limit"));
                }
                Ok(value.to_vec())
            }
            fn visit_byte_buf<E: serde::de::Error>(self, value: Vec<u8>) -> Result<Vec<u8>, E> {
                if value.len() > super::MAX_CYCLIC_ARTIFACT_BYTES {
                    return Err(E::custom("source byte limit"));
                }
                Ok(value)
            }
        }
        deserializer.deserialize_byte_buf(Bytes)
    }
}
/// Native evidence whose constructor can only read an executing prepared handle
/// or an independently replayed original artifact.
#[derive(Clone, Debug)]
pub struct CyclicEvidence {
    report: CyclicEvidenceReport,
    source: Arc<[u8]>,
    schema: CyclicEvidenceSchema,
    selectors: Vec<CyclicEvidenceSelector>,
    envelopes: Vec<EvidenceEnvelope>,
    _retention: Arc<Vec<antecedent_core::execution::AdmissionLease>>,
}
fn invalid(s: impl std::fmt::Display) -> CyclicArtifactError {
    CyclicArtifactError::Invalid(s.to_string())
}
fn digest<T: Serialize>(tag: &str, value: &T) -> Result<String, CyclicArtifactError> {
    Ok(CanonicalWriter::digest_bytes_hex(tag, &serde_json::to_vec(value).map_err(invalid)?))
}
fn state(outcome: &EquilibriumOutcome, variable: usize) -> Option<f64> {
    match outcome {
        EquilibriumOutcome::Solved(value) => value.values.get(variable).copied(),
        EquilibriumOutcome::Failed { .. } => None,
    }
}
fn licensed_state(outcome: &EquilibriumOutcome, variable: usize) -> Option<f64> {
    match outcome {
        EquilibriumOutcome::Solved(value) if value.globally_unique && value.existence_certified => {
            value.values.get(variable).copied()
        }
        _ => None,
    }
}

struct Projection {
    samples: Vec<Option<f64>>,
    bounds: Vec<Option<Interval>>,
    masses: Option<Vec<antecedent_estimate::polynomial_scm::MassBounds>>,
    port: EvidencePort,
    axes: Vec<String>,
    world: String,
    time: String,
    functional: &'static str,
    variable: usize,
    /// Second coordinate of a covariance quantity.
    other: Option<usize>,
    diagnostics: serde_json::Value,
}
fn point(
    value: Option<f64>,
    variable: usize,
    world: String,
    time: String,
    functional: &'static str,
    port: EvidencePort,
) -> Projection {
    Projection {
        samples: vec![value],
        bounds: vec![],
        masses: None,
        port,
        axes: vec![],
        world,
        time,
        functional,
        variable,
        other: None,
        diagnostics: serde_json::Value::Null,
    }
}

fn selector_mismatch() -> CyclicArtifactError {
    invalid("cyclic evidence selector differs from source family or coordinate bounds")
}

/// Draw-aligned projection of one recovered world.
///
/// Supplied draws declare no meaning and stay on the interventional port. Draws
/// from a declared sampling law or posterior project the per-draw interventional
/// mean onto that port: parameter uncertainty is not relabelled as outcome
/// variation, and a sampling law is not read as a posterior.
fn recovered_response(
    found: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: &antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire,
    world: usize,
    variable: usize,
) -> Result<Projection, CyclicArtifactError> {
    let mut samples = Vec::with_capacity(found.draws.len());
    let mut masses = Vec::with_capacity(found.draws.len());
    for draw in &found.draws {
        masses.push(antecedent_estimate::polynomial_scm::MassBounds {
            lower: draw.weight,
            upper: draw.weight,
        });
        samples.push(match &draw.outcome {
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
                mean,
                ..
            } => Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Inadmissible { .. }
            | antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::NotCertified { .. } => {
                None
            }
        });
    }
    let (port, axis, functional) = match origin.law_meaning() {
        None => (EvidencePort::InterventionalLaw, "predictive", "recovered_intervention"),
        Some("sampling_law") => {
            (EvidencePort::SamplingLaw, "sampling", "recovered_interventional_mean")
        }
        Some(_) => (EvidencePort::ParameterPosterior, "parameter", "recovered_interventional_mean"),
    };
    let mut p =
        point(None, variable, format!("world:{world}"), "equilibrium".into(), functional, port);
    p.samples = samples;
    p.masses = Some(masses);
    p.axes = vec![axis.into()];
    p.diagnostics = if origin.is_supplied() {
        serde_json::to_value(found).map_err(invalid)?
    } else {
        serde_json::json!({ "world": found, "origin": origin })
    };
    Ok(p)
}

/// The interventional law within one recovered draw.
fn recovered_within(
    found: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    world: usize,
    draw: usize,
    variable: usize,
) -> Result<Projection, CyclicArtifactError> {
    let item = found.draws.get(draw).ok_or_else(selector_mismatch)?;
    let (value, variance) = match &item.outcome {
        antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
            mean,
            covariance,
            ..
        } => {
            let n = mean.len();
            let value = *mean.get(variable).ok_or_else(selector_mismatch)?;
            (Some(value), covariance.get(variable * n + variable).copied())
        }
        _ => (None, None),
    };
    let mut p = point(
        value,
        variable,
        format!("world:{world}:draw:{draw}"),
        "equilibrium".into(),
        "interventional_outcome",
        EvidencePort::InterventionalLaw,
    );
    p.axes = vec!["predictive".into()];
    p.masses = Some(vec![antecedent_estimate::polynomial_scm::MassBounds {
        lower: item.weight,
        upper: item.weight,
    }]);
    p.diagnostics = serde_json::json!({ "variance": variance, "draw": item });
    Ok(p)
}

/// Projection of one typed world of a specified model.
fn typed_world(
    found: &antecedent_io::cyclic_family_artifact::CyclicWorldAnswer,
    variable: usize,
    time: Option<usize>,
) -> Result<Projection, CyclicArtifactError> {
    use antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as O;
    if variable >= found.variables.len() {
        return Err(selector_mismatch());
    }
    let label = format!("world:{}", found.world);
    let context = serde_json::json!({
        "certificate": found.certificate,
        "solved_by": found.solved_by,
        "failed_mass": found.failed_mass,
        "withheld_mass": found.withheld_mass,
        "actions": found.actions,
    });
    let mut p = match (&found.outcome, time) {
        (O::GaussianLaw { mean, covariance, .. }, None) => {
            let n = mean.len();
            let mut p = point(
                Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
                variable,
                label,
                "equilibrium".into(),
                "interventional_mean",
                EvidencePort::InterventionalLaw,
            );
            p.axes = vec!["predictive".into()];
            p.diagnostics = serde_json::json!({
                "variance": covariance.get(variable * n + variable),
                "world": context,
            });
            p
        }
        (O::EquilibriumDraws { outcomes }, None) => {
            let mut p = point(
                None,
                variable,
                label,
                "equilibrium".into(),
                "specified_noise_response",
                EvidencePort::Scenario,
            );
            p.samples = outcomes.iter().map(|outcome| state(outcome, variable)).collect();
            p.diagnostics = context;
            p
        }
        (O::Trajectory { trajectory }, Some(index)) => {
            let value = trajectory
                .states
                .get(index)
                .and_then(|row| row.get(variable))
                .copied()
                .ok_or_else(selector_mismatch)?;
            let mut p = point(
                Some(value),
                variable,
                label,
                format!("time:{index}"),
                "trajectory_state",
                EvidencePort::Scenario,
            );
            p.diagnostics = context;
            p
        }
        (O::NoSolution { .. } | O::Withheld, _) => {
            let mut p = point(
                None,
                variable,
                label,
                "equilibrium".into(),
                "no_solution",
                EvidencePort::Scenario,
            );
            p.diagnostics = serde_json::json!({ "outcome": found.outcome, "world": context });
            p
        }
        _ => return Err(selector_mismatch()),
    };
    p.bounds.clear();
    Ok(p)
}

/// Endogenous coordinates of one dynamic quantity.
const fn quantity_coordinates(
    quantity: &antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity,
) -> (usize, Option<usize>) {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    match quantity {
        Q::OdeState { variable, .. }
        | Q::OuStationaryMean { variable, .. }
        | Q::OuMechanismMean { variable, .. } => (*variable, None),
        Q::OuStationaryCovariance { variable, other, .. }
        | Q::OuMechanismCovariance { variable, other, .. } => (*variable, Some(*other)),
    }
}

/// Projection of one declared quantity of a fitted continuous-time family.
fn dynamic_quantity(
    quantities: &[antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity],
    inference: &antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer,
    quantity: usize,
) -> Result<Projection, CyclicArtifactError> {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    use antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer as D;
    let spec = quantities.get(quantity).ok_or_else(selector_mismatch)?;
    let (variable, other) = quantity_coordinates(spec);
    let time = match spec {
        Q::OdeState { time, .. } => format!("time:{time}"),
        _ => "stationary".into(),
    };
    let label = format!("quantity:{quantity}");
    let functional = if other.is_some() { "dynamic_covariance" } else { "dynamic_quantity" };
    let mut p = match inference {
        D::SamplingLaw(law) => {
            let mut p = point(None, variable, label, time, functional, EvidencePort::SamplingLaw);
            p.samples = law
                .replicates
                .iter()
                .map(|replicate| replicate.quantities.get(quantity).copied().flatten())
                .collect();
            p.masses = Some(
                law.weights
                    .iter()
                    .map(|weight| antecedent_estimate::polynomial_scm::MassBounds {
                        lower: *weight,
                        upper: *weight,
                    })
                    .collect(),
            );
            p.axes = vec!["sampling".into()];
            p.diagnostics = serde_json::json!({
                "estimate": law.quantity_estimates.get(quantity),
                "standard_error": law.quantity_standard_errors.get(quantity),
                "failed_mass": law.quantity_failed_mass.get(quantity),
                "replicate_failed_mass": law.failed_mass,
                "method": law.method,
            });
            p
        }
        D::Information(information) => {
            let mut p = point(
                information.quantity_estimates.get(quantity).copied().flatten(),
                variable,
                label,
                time,
                functional,
                EvidencePort::CovarianceInfluence,
            );
            p.axes = vec!["sampling".into()];
            p.diagnostics = serde_json::json!({
                "standard_error": information.quantity_standard_errors.get(quantity),
                "regularity": information.regularity,
            });
            p
        }
        D::Posterior(posterior) => {
            let mut p =
                point(None, variable, label, time, functional, EvidencePort::ParameterPosterior);
            p.samples = posterior
                .quantity_draws
                .iter()
                .map(|draw| draw.get(quantity).copied().flatten())
                .collect();
            p.axes = vec!["parameter".into()];
            p.diagnostics = serde_json::json!({
                "failed_mass": posterior.quantity_failed_mass.get(quantity),
                "max_rhat": posterior.max_rhat,
                "min_bulk_ess": posterior.min_bulk_ess,
                "min_tail_ess": posterior.min_tail_ess,
                "diagnostics_pass": posterior.diagnostics_pass,
            });
            p
        }
    };
    p.other = other;
    Ok(p)
}
#[allow(
    clippy::too_many_lines,
    reason = "typed family dispatch preserves distinct uncertainty ports and explicit failure slots"
)]
fn project(
    request: &R,
    answer: &A,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection, CyclicArtifactError> {
    let bad =
        || invalid("cyclic evidence selector differs from source family or coordinate bounds");
    Ok(match (request, answer, selector) {
        (
            _,
            A::EquilibriumActions(worlds),
            CyclicEvidenceSelector::Action { world, noise_row, variable, counterfactual },
        ) => {
            let w = worlds.get(*world).ok_or_else(bad)?;
            let value =
                if *counterfactual { w.counterfactual.as_ref() } else { w.rows.get(*noise_row) }
                    .ok_or_else(bad)?;
            let mut p = point(
                state(value, *variable),
                *variable,
                w.id.clone(),
                "equilibrium".into(),
                if *counterfactual { "factual_counterfactual" } else { "specified_noise_response" },
                EvidencePort::Scenario,
            );
            // A residual-checked numeric state is not an outer enclosure of all solutions.
            p.bounds.clear();
            p.diagnostics = serde_json::to_value(value).map_err(invalid)?;
            p
        }
        (
            _,
            A::ParameterPosterior { worlds, .. },
            CyclicEvidenceSelector::Response { world, noise_row, variable },
        ) => {
            let draws = worlds.get(*world).ok_or_else(bad)?;
            let values = draws
                .iter()
                .map(|draw| {
                    draw.get(*noise_row)
                        .map(|value| licensed_state(value, *variable))
                        .ok_or_else(bad)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut p = point(
                None,
                *variable,
                format!("world:{world}:noise:{noise_row}"),
                "equilibrium".into(),
                "conditional_structural_response",
                EvidencePort::ParameterPosterior,
            );
            p.samples = values;
            p.axes = vec!["parameter".into()];
            p
        }
        (
            _,
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. },
            CyclicEvidenceSelector::Response { world, noise_row, variable },
        ) => {
            if *noise_row != 0 {
                return Err(bad());
            }
            let draws = worlds.get(*world).ok_or_else(bad)?;
            let mut p = point(
                None,
                *variable,
                format!("world:{world}"),
                "equilibrium".into(),
                "outcome",
                EvidencePort::InterventionalLaw,
            );
            p.samples = draws.iter().map(|v| licensed_state(v, *variable)).collect();
            p.axes = vec!["predictive".into()];
            if matches!(answer, A::JointCounterfactual { .. }) {
                p.axes.push("parameter".into());
            }
            p
        }
        (
            _,
            A::FrequentistBootstrap { worlds, .. },
            CyclicEvidenceSelector::Response { world, noise_row, variable },
        ) => {
            let draws = worlds.get(*world).ok_or_else(bad)?;
            let mut p = point(
                None,
                *variable,
                format!("world:{world}:noise:{noise_row}"),
                "equilibrium".into(),
                "conditional_structural_response",
                EvidencePort::SamplingLaw,
            );
            p.samples = draws
                .iter()
                .map(|draw| match draw {
                    Some(rows) => {
                        rows.get(*noise_row).map(|v| licensed_state(v, *variable)).ok_or_else(bad)
                    }
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?;
            p.axes = vec!["sampling".into()];
            p
        }
        (
            _,
            A::OuMechanisms { laws, .. },
            CyclicEvidenceSelector::OuRegime { world, variable, other },
        ) => {
            let outcome = laws.get(*world).ok_or_else(bad)?;
            let antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) = outcome
            else {
                let mut p = point(
                    None,
                    *variable,
                    format!("equation-world:{world}"),
                    "stationary".into(),
                    if other.is_some() { "predictive_covariance" } else { "mean" },
                    if other.is_some() { EvidencePort::Likelihood } else { EvidencePort::Mean },
                );
                p.diagnostics = serde_json::to_value(outcome).map_err(invalid)?;
                return Ok(p);
            };
            let n = law.mean.len();
            if *variable >= n || other.is_some_and(|j| j >= n) {
                return Err(bad());
            }
            point(
                Some(other.map_or(law.mean[*variable], |j| law.covariance[*variable * n + j])),
                *variable,
                format!("equation-world:{world}"),
                "stationary".into(),
                if other.is_some() { "predictive_covariance" } else { "mean" },
                if other.is_some() { EvidencePort::Likelihood } else { EvidencePort::Mean },
            )
        }
        (
            _,
            A::OuMechanisms { fit: Some(fit), .. },
            CyclicEvidenceSelector::Likelihood { candidate: None },
        ) => point(
            Some(fit.log_likelihood),
            0,
            "training".into(),
            "stationary".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (_, A::Ode(path), CyclicEvidenceSelector::Ode { time, variable }) => point(
            Some(*path.states.get(*time).and_then(|s| s.get(*variable)).ok_or_else(bad)?),
            *variable,
            "initial_value_world".into(),
            format!("t:{}", path.times[*time]),
            "conditional_trajectory",
            EvidencePort::Mean,
        ),
        (
            R::OuStationary { model },
            A::OuStationary(_),
            CyclicEvidenceSelector::OuMean { time: None, variable },
        ) => point(
            Some(*model.center.get(*variable).ok_or_else(bad)?),
            *variable,
            "stationary".into(),
            "stationary".into(),
            "mean",
            EvidencePort::Mean,
        ),
        (
            R::OuStationary { model },
            A::OuStationary(cov),
            CyclicEvidenceSelector::OuCovariance { time: None, variable, other },
        ) => {
            let n = model.center.len();
            if *variable >= n || *other >= n {
                return Err(bad());
            }
            point(
                Some(cov[*variable * n + other]),
                *variable,
                "stationary".into(),
                "stationary".into(),
                "predictive_covariance",
                EvidencePort::Likelihood,
            )
        }
        (
            _,
            A::OuTransient(path),
            CyclicEvidenceSelector::OuMean { time: Some(time), variable },
        ) => point(
            Some(*path.means.get(*time).and_then(|s| s.get(*variable)).ok_or_else(bad)?),
            *variable,
            "initial_law_world".into(),
            format!("t:{}", path.times[*time]),
            "mean",
            EvidencePort::Mean,
        ),
        (
            _,
            A::OuTransient(path),
            CyclicEvidenceSelector::OuCovariance { time: Some(time), variable, other },
        ) => {
            let row = path.means.get(*time).ok_or_else(bad)?;
            let n = row.len();
            if *variable >= n || *other >= n {
                return Err(bad());
            }
            point(
                Some(path.covariances[*time][*variable * n + other]),
                *variable,
                "initial_law_world".into(),
                format!("t:{}", path.times[*time]),
                "predictive_covariance",
                EvidencePort::Likelihood,
            )
        }
        (_, A::Polynomial(solutions), CyclicEvidenceSelector::Root { root, variable }) => {
            let enclosure =
                *solutions.roots.get(*root).and_then(|s| s.get(*variable)).ok_or_else(bad)?;
            let mut p = point(
                None,
                *variable,
                format!("root_enclosure:{root}"),
                "algebraic".into(),
                "solution_coordinate",
                EvidencePort::Scenario,
            );
            p.bounds = vec![Some(enclosure)];
            p.diagnostics = serde_json::json!({"unresolved_regions":solutions.unresolved.len(),"selection_law":false});
            p
        }
        (_, A::Polynomial(solutions), CyclicEvidenceSelector::SolutionSet { variable }) => {
            let mut p = point(
                None,
                *variable,
                "all_model_solutions".into(),
                "algebraic".into(),
                "model_solution_set",
                EvidencePort::Scenario,
            );
            p.samples.clear();
            p.bounds = solutions
                .roots
                .iter()
                .chain(&solutions.unresolved)
                .map(|region| region.get(*variable).copied().map(Some).ok_or_else(bad))
                .collect::<Result<Vec<_>, _>>()?;
            p.diagnostics = serde_json::json!({"certified_roots":solutions.roots.len(),"unresolved_regions":solutions.unresolved.len(),"selection_law":false,"outer_enclosure":true});
            p
        }
        (
            R::Functional { source, .. },
            A::Functional(report),
            CyclicEvidenceSelector::Functional { assignment },
        ) => {
            use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
            let outcomes = match source {
                G::Identify { outcomes, .. } | G::Adjust { outcomes, .. } => outcomes,
                G::Calculus { y, .. } => y,
                _ => return Err(bad()),
            };
            if outcomes.len() != 1 {
                return Err(invalid(
                    "scalar evidence projection requires a single outcome; joint events retain the full functional report",
                ));
            }

            let value = report.values.get(*assignment).ok_or_else(bad)?;
            let mut p = point(
                value.value,
                outcomes[0],
                format!("assignment:{assignment}"),
                "static".into(),
                "identified_probability",
                EvidencePort::Mean,
            );
            p.diagnostics = serde_json::to_value(value).map_err(invalid)?;
            p
        }
        (
            R::GaussianFunctional { source, .. },
            A::GaussianFunctional(report),
            CyclicEvidenceSelector::Functional { assignment },
        ) => {
            use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
            let outcomes = match source {
                G::Identify { outcomes, .. } | G::Adjust { outcomes, .. } => outcomes,
                G::Calculus { y, .. } => y,
                _ => return Err(bad()),
            };
            if outcomes.len() != 1 {
                return Err(invalid("scalar Gaussian density evidence requires one outcome"));
            }
            let value = report.values.get(*assignment).ok_or_else(bad)?;
            let mut p = point(
                value.value,
                outcomes[0],
                format!("assignment:{assignment}"),
                "static".into(),
                "identified_density",
                EvidencePort::Likelihood,
            );
            p.diagnostics = serde_json::to_value(value).map_err(invalid)?;
            p
        }
        (_, A::AdamFit(fit), CyclicEvidenceSelector::Likelihood { candidate: None }) => point(
            Some(fit.training_log_likelihood),
            0,
            "training".into(),
            "equilibrium".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (_, A::GradientFit(fit), CyclicEvidenceSelector::Likelihood { candidate: None }) => point(
            Some(fit.log_likelihood),
            0,
            "training".into(),
            "static".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (_, A::OdeFit(fit), CyclicEvidenceSelector::Likelihood { candidate: None }) => point(
            Some(fit.log_likelihood),
            0,
            "training".into(),
            "trajectory".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (_, A::OuFit(fit), CyclicEvidenceSelector::Likelihood { candidate: None }) => point(
            Some(fit.log_likelihood),
            0,
            "training".into(),
            "stationary".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (
            _,
            A::StructureFit { fit, .. },
            CyclicEvidenceSelector::Likelihood { candidate: None },
        ) => point(
            Some(fit.fit.log_likelihood),
            0,
            "training".into(),
            "equilibrium".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (
            _,
            A::PolynomialFit(fit),
            CyclicEvidenceSelector::Likelihood { candidate: Some(candidate) },
        ) => {
            let c = fit.candidates.get(*candidate).ok_or_else(bad)?;
            let mut p = point(
                None,
                0,
                c.candidate.clone(),
                "algebraic".into(),
                "log_likelihood",
                EvidencePort::Likelihood,
            );
            p.bounds = vec![c.lower.zip(c.upper).map(|(lower, upper)| Interval { lower, upper })];
            p.diagnostics = serde_json::to_value(c).map_err(invalid)?;
            p
        }
        (
            R::RecoveredLinear { uncertainty, .. },
            A::RecoveredLinear(report),
            CyclicEvidenceSelector::Response { world, noise_row, variable },
        ) => {
            if *noise_row != 0 {
                return Err(bad());
            }
            let found = report.worlds.get(*world).ok_or_else(bad)?;
            recovered_response(found, uncertainty, *world, *variable)?
        }
        (
            _,
            A::ExperimentalRecovery(report),
            CyclicEvidenceSelector::Response { world, noise_row, variable },
        ) => {
            if *noise_row != 0 {
                return Err(bad());
            }
            let found = &report.worlds.get(*world).ok_or_else(bad)?.answer;
            recovered_response(found, &report.law.origin, *world, *variable)?
        }
        (
            _,
            A::RecoveredLinear(report),
            CyclicEvidenceSelector::RecoveredDraw { world, draw, variable },
        ) => {
            recovered_within(report.worlds.get(*world).ok_or_else(bad)?, *world, *draw, *variable)?
        }
        (
            _,
            A::ExperimentalRecovery(report),
            CyclicEvidenceSelector::RecoveredDraw { world, draw, variable },
        ) => recovered_within(
            &report.worlds.get(*world).ok_or_else(bad)?.answer,
            *world,
            *draw,
            *variable,
        )?,
        (_, A::Worlds(worlds), CyclicEvidenceSelector::World { world, variable, time }) => {
            let found = worlds.iter().find(|item| item.world == *world).ok_or_else(bad)?;
            typed_world(found, *variable, *time)?
        }
        (
            R::DynamicInference { quantities, .. },
            A::DynamicInference { inference, .. },
            CyclicEvidenceSelector::DynamicQuantity { quantity },
        ) => dynamic_quantity(quantities, inference, *quantity)?,
        (
            _,
            A::DynamicInference { fit, .. },
            CyclicEvidenceSelector::Likelihood { candidate: None },
        ) => point(
            Some(fit.log_likelihood()),
            0,
            "training".into(),
            "dynamic".into(),
            "log_likelihood",
            EvidencePort::Likelihood,
        ),
        (
            _,
            A::System(report),
            CyclicEvidenceSelector::Action { world, noise_row, variable, counterfactual },
        ) => {
            let value = if *counterfactual {
                report.counterfactuals.as_ref().and_then(|rows| rows.get(*world))
            } else {
                report.worlds.get(*world).and_then(|rows| rows.get(*noise_row))
            }
            .ok_or_else(bad)?;
            let mut p = point(
                Some(*value.values.get(*variable).ok_or_else(bad)?),
                *variable,
                format!("world:{world}"),
                "equilibrium".into(),
                if *counterfactual { "factual_counterfactual" } else { "specified_noise_response" },
                EvidencePort::Scenario,
            );
            p.diagnostics = serde_json::to_value(value).map_err(invalid)?;
            p
        }
        (_, A::Graph(graph), CyclicEvidenceSelector::Graph) => {
            let mut p = point(
                None,
                0,
                "graphical".into(),
                "static".into(),
                "graphical_evidence",
                EvidencePort::Scenario,
            );
            p.diagnostics = serde_json::to_value(graph).map_err(invalid)?;
            p
        }
        (
            _,
            A::PolynomialCausal(report),
            CyclicEvidenceSelector::PolynomialWorld { world, variable },
        ) => {
            let mut p = point(
                None,
                *variable,
                world.clone(),
                "algebraic".into(),
                "counterfactual_outcome_set",
                EvidencePort::Scenario,
            );
            p.samples.clear();
            p.bounds.clear();
            let mut masses = Vec::new();
            for atom in &report.atoms {
                let bound = match atom.states.get(world).ok_or_else(bad)? {
                    antecedent_estimate::polynomial_scm::PolynomialSelectedState::Selected {
                        enclosure,
                    } => Some(*enclosure.get(*variable).ok_or_else(bad)?),
                    _ => None,
                };
                p.samples.push(None);
                p.bounds.push(bound);
                if let Some(mass) = atom.posterior_mass {
                    masses.push(mass);
                }
            }
            p.diagnostics = serde_json::json!({"world_mass":report.world_mass,"positive_evidence_certified":masses.len()==report.atoms.len() && masses.iter().any(|m|m.lower>0.0)});
            p.masses = Some(masses);
            p.axes = vec!["predictive".into(), "parameter".into()];
            p
        }
        _ => return Err(bad()),
    })
}
fn source_order(request: &R) -> usize {
    use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
    let graph_order = |graph: &G| match graph {
        G::Identify { graph, .. }
        | G::Backdoor { graph, .. }
        | G::Adjust { graph, .. }
        | G::Calculus { graph, .. } => graph.nodes as usize,
        G::Bipartite { variables, .. } | G::DomainInvariance { variables, .. } => *variables,
        G::Discover { columns, oracle, .. } => {
            oracle.as_ref().map_or(columns.len(), |g| g.nodes as usize)
        }
    };
    match request {
        R::AdamFit { model, .. }
        | R::EquilibriumRoots { model, .. }
        | R::EquilibriumActions { model, .. }
        | R::GradientFit { model, .. }
        | R::FrequentistBootstrap { model, .. }
        | R::ParameterPosterior { model, .. }
        | R::JointCounterfactual { model, .. }
        | R::PartialCounterfactual { model, .. }
        | R::StructureFit { model, .. } => model.order(),
        R::Ode { model, .. } | R::OdeFit { model, .. } => model.equations.len(),
        R::OuMechanisms { model, .. } => model.equations.len(),
        R::OuStationary { model, .. } | R::OuTransient { model, .. } | R::OuFit { model, .. } => {
            model.center.len()
        }
        R::Polynomial { model, .. } => model.variables.len(),
        R::PolynomialCausal { program } => {
            program.candidates.first().map_or(0, |c| c.model.variables.len())
        }
        R::PolynomialFit { candidates, .. } => {
            candidates.first().map_or(0, |c| c.model.variables.len())
        }
        R::Graph { request } => graph_order(request),
        R::Functional { source, .. } | R::GaussianFunctional { source, .. } => graph_order(source),
        R::RecoveredLinear { nodes, .. } => nodes.len(),
        R::ExperimentalRecovery { regimes, dropped, .. } => {
            let dropped: std::collections::BTreeSet<u32> = dropped.iter().copied().collect();
            regimes
                .iter()
                .flat_map(|regime| regime.units.iter().copied())
                .filter(|unit| !dropped.contains(unit))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        }
        R::LinearWorlds { variables, .. }
        | R::EquilibriumWorlds { variables, .. }
        | R::OdeWorlds { variables, .. } => variables.len(),
        R::DynamicInference { problem, .. } => problem.order(),
        R::System { request } => request.model.order(),
    }
}

impl PreparedStudy<CyclicFamilyPreparedState> {
    /// Bind retained native outputs to typed evidence without fitting or sampling again.
    /// # Errors
    /// Invalid schema, incompatible selectors, cancellation or resource refusal.
    pub fn cyclic_evidence(
        &self,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicEvidence, CyclicArtifactError> {
        let (report, bytes) = self.export(ctx)?;
        CyclicEvidence::build(
            self.state.program.request(),
            &report,
            bytes.into(),
            schema,
            selectors,
            ctx,
        )
    }
}
impl CyclicEvidence {
    #[allow(
        clippy::too_many_lines,
        reason = "source-bound evidence construction validates coordinates and retains native envelopes atomically"
    )]
    fn build(
        request: &R,
        source: &CyclicFamilyReport,
        artifact: Arc<[u8]>,
        schema: CyclicEvidenceSchema,
        selectors: Vec<CyclicEvidenceSelector>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        if schema.id.trim().is_empty()
            || schema.id.len() > 256
            || schema.population.len() > 256
            || schema.population.trim().is_empty()
            || schema.variables.is_empty()
            || schema.variables.len() > 128
            || schema.variables.len() != source_order(request)
            || schema.variables.iter().enumerate().any(|(i, (id, units))| {
                id.trim().is_empty()
                    || id.len() > 256
                    || units.len() > 128
                    || units.trim().is_empty()
                    || schema.variables[..i].iter().any(|(old, _)| old == id)
            })
            || selectors.is_empty()
            || selectors.len() > 1024
        {
            return Err(invalid("cyclic evidence schema/selector limits"));
        }
        let draws = match &source.answer {
            A::ParameterPosterior { worlds, .. } => worlds.first().map_or(0, Vec::len),
            A::FrequentistBootstrap { worlds, .. } => worlds.first().map_or(0, Vec::len),
            A::JointCounterfactual { worlds, .. } | A::PartialCounterfactual { worlds, .. } => {
                worlds.first().map_or(0, Vec::len)
            }
            A::PolynomialCausal(report) => report.atoms.len(),
            A::RecoveredLinear(report) => {
                report.worlds.first().map_or(0, |world| world.draws.len())
            }
            A::ExperimentalRecovery(report) => report.law.draws.len(),
            A::Worlds(worlds) => worlds
                .iter()
                .map(|world| match &world.outcome {
                    antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer::EquilibriumDraws {
                        outcomes,
                    } => outcomes.len(),
                    _ => 1,
                })
                .max()
                .unwrap_or(1),
            A::DynamicInference { inference, .. } => match inference.as_ref() {
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::SamplingLaw(law) => {
                    law.replicates.len()
                }
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::Posterior(
                    posterior,
                ) => posterior.quantity_draws.len(),
                antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer::Information(_) => 1,
            },
            A::Polynomial(report) => report.roots.len().saturating_add(report.unresolved.len()),
            _ => 1,
        };
        if selectors.len().saturating_mul(draws) > 1_000_000 {
            return Err(invalid("cyclic evidence projection cell limit"));
        }
        let reserve = (artifact.len() as u64) * 4
            + (cyclic_serialized_size(&(source, &schema, &selectors), ctx)? as u64) * 8
            + 4096;
        let mut retention =
            vec![ctx.acquire_admission(0, reserve).map_err(CyclicArtifactError::Admission)?];
        let identity =
            digest("antecedent.cyclic_evidence.v1", &(&source.identity, &schema, &selectors))?;
        let mut columns = Vec::new();
        let mut envelopes = Vec::new();
        for selector in &selectors {
            let p = project(request, &source.answer, selector)?;
            let projection_bytes = cyclic_serialized_size(
                &(&p.diagnostics, &p.samples, &p.bounds, &p.masses, &p.world, &p.time),
                ctx,
            )? as u64;
            retention.push(
                ctx.acquire_admission(0, projection_bytes * 8 + 4096)
                    .map_err(CyclicArtifactError::Admission)?,
            );
            let (name, units) = schema
                .variables
                .get(p.variable)
                .ok_or_else(|| invalid("evidence variable outside declared schema"))?;
            let regime = digest(
                "antecedent.cyclic_evidence.regime.v1",
                &(&source.identity, &p.world, &p.time),
            )?;
            let quantity = ScientificQuantity {
                variable_id: format!("{}:{name}", schema.id),
                variable_name: name.clone(),
                role: QuantityRole::Outcome,
                units: if p.functional == "identified_density" {
                    format!("1/({units})")
                } else if matches!(p.functional, "log_likelihood" | "identified_probability") {
                    "dimensionless".into()
                } else if let Some(other) = p.other {
                    let second = schema
                        .variables
                        .get(other)
                        .ok_or_else(|| invalid("covariance variable outside schema"))?;
                    format!("({units})*({})", second.1)
                } else if let CyclicEvidenceSelector::OuCovariance { other, .. }
                | CyclicEvidenceSelector::OuRegime { other: Some(other), .. } = selector
                {
                    let second = schema
                        .variables
                        .get(*other)
                        .ok_or_else(|| invalid("covariance variable outside schema"))?;
                    format!("({units})*({})", second.1)
                } else {
                    units.clone()
                },
                population_id: schema.population.clone(),
                regime_id: regime,
                horizon: 0,
                functional_id: p.functional.into(),
                conditioning: vec![],
                transform_id: "identity".into(),
            };
            let premises: Arc<[Arc<str>]> = vec![
                Arc::from("conditional_on_declared_source_semantics_and_assumptions"),
                Arc::from("numerical_diagnostics_are_not_identification_certificates"),
            ]
            .into();
            let envelope = EvidenceEnvelope::try_new(
                EvidenceBinding::try_new(
                    quantity.clone(),
                    WorldTime { world_id: p.world.clone().into(), time_id: p.time.clone().into() },
                    premises,
                    EvidenceOrigins {
                        source: Some(source.identity.clone().into()),
                        selection: if matches!(
                            request,
                            R::PolynomialCausal { .. } | R::PolynomialFit { .. }
                        ) {
                            Some(source.identity.clone().into())
                        } else {
                            None
                        },
                        training: if matches!(
                            request,
                            R::AdamFit { .. }
                                | R::GaussianFunctional {law:antecedent_io::cyclic_gaussian_functional::GaussianCyclicLaw::Fit{..},..}
                                | R::OuMechanisms { .. }
                                | R::GradientFit { .. }
                                | R::OdeFit { .. }
                                | R::OuFit { .. }
                                | R::PolynomialFit { .. }
                                | R::StructureFit { .. }
                                | R::ParameterPosterior { .. }
                                | R::JointCounterfactual { .. }
                                | R::FrequentistBootstrap { .. }
                                | R::ExperimentalRecovery { .. }
                                | R::DynamicInference { .. }
                                | R::System {
                                    request: antecedent_io::cyclic_system_artifact::CyclicExecutionRequest {
                                        fit: Some(_),
                                        ..
                                    }
                                }
                        ) {
                            Some(source.identity.clone().into())
                        } else {
                            None
                        },
                        prior: if p.axes.iter().any(|a| a == "parameter") {
                            Some(source.identity.clone().into())
                        } else {
                            None
                        },
                    },
                    EvidenceSupport {
                        status: SupportStatus::MissingEvidence,
                        domain_id: "empirical_support_not_assessed".into(),
                    },
                    EvidenceCalibration::ScopeNotAssessed,
                )
                .map_err(invalid)?,
                UncertaintyAxes::absent(),
            )
            .map_err(invalid)?;
            columns.push(CyclicEvidenceColumn {
                quantity: ScientificQuantityWire::from(&quantity),
                envelope_identity: envelope.identity().to_hex(),
                port: p.port.as_str().into(),
                uncertainty_axes: p.axes,
                samples: p.samples,
                numerical_enclosures: p.bounds,
                mass_bounds: p.masses,
                world: p.world,
                time: p.time,
                diagnostics: p.diagnostics,
            });
            envelopes.push(envelope);
        }
        let scope = if matches!(source.answer, A::GaussianFunctional(_)) {
            "theorem_identified_conditional_on_declared_Gaussian_observational_density"
        } else if matches!(source.answer, A::Functional(_)) {
            "theorem_identified_conditional_on_declared_observational_law"
        } else if matches!(source.answer, A::Graph(_)) {
            "theorem_scoped_graphical_evidence"
        } else if matches!(source.answer, A::ExperimentalRecovery(_)) {
            "recovered_from_declared_clamp_experiments_conditional_on_the_recipe"
        } else {
            "specified_or_fitted_model_conditional"
        }
        .into();
        Ok(Self {
            report: CyclicEvidenceReport {
                identity,
                source_identity: source.identity.clone(),
                scope,
                columns,
                source: source.clone(),
            },
            source: artifact,
            schema,
            selectors,
            envelopes,
            _retention: Arc::new(retention),
        })
    }
    /// Checked projection, with failures and numerical enclosures in original slots.
    #[must_use]
    pub fn report(&self) -> &CyclicEvidenceReport {
        &self.report
    }
    /// Existing contract envelopes, suitable for evidence binding/port checks.
    #[must_use]
    pub fn envelopes(&self) -> &[EvidenceEnvelope] {
        &self.envelopes
    }
    /// Export the original source plus projection; independent loading re-executes the source.
    /// # Errors
    /// Resource or container/size refusal.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CyclicArtifactError::Cancelled);
        }
        let _admission = ctx
            .acquire_admission(
                0,
                (self.source.len() as u64) * 4
                    + (cyclic_serialized_size(&(&self.report, &self.schema, &self.selectors), ctx)?
                        as u64)
                        * 8
                    + 65536,
            )
            .map_err(CyclicArtifactError::Admission)?;
        let bytes = antecedent_io::to_cbor(&Record {
            version: 2,
            source: self.source.to_vec(),
            source_identity: self.report.source_identity.clone(),
            schema: self.schema.clone(),
            selectors: self.selectors.clone(),
            report: self.report.clone(),
        })
        .map_err(invalid)?;
        if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("cyclic evidence byte limit"));
        }
        let _lease = ctx
            .acquire_admission(0, (bytes.len() as u64).saturating_mul(4))
            .map_err(CyclicArtifactError::Admission)?;
        antecedent_io::encode_container(
            &antecedent_io::ArtifactContainer {
                kind: "cyclic_evidence",
                note: "typed_native_cyclic_projection",
                max_bytes: MAX_CYCLIC_ARTIFACT_BYTES,
                artifact_id: &self.report.identity,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[antecedent_io::OutSection {
                id: "cyclic_evidence_v1",
                content_type: antecedent_io::CONTENT_BYTES,
                bytes: &bytes,
            }],
        )
        .map_err(|e| invalid(e.into_io("cyclic evidence layout")))
    }
    /// Independently verify source, target, projection, uncertainty meaning and numerical output.
    /// # Errors
    /// Identity mismatch, malformed envelope, replay failure, cancellation or resources.
    pub fn consume(
        bytes: &[u8],
        expected_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<Self, CyclicArtifactError> {
        if expected_identity.len() != 64
            || !expected_identity.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(invalid("cyclic evidence identity"));
        }
        if bytes.len() > MAX_CYCLIC_ARTIFACT_BYTES {
            return Err(invalid("cyclic evidence byte limit"));
        }
        let _lease = cyclic_decode_admission(bytes, ctx)?;
        let parts = antecedent_io::decode_container(
            "cyclic_evidence",
            MAX_CYCLIC_ARTIFACT_BYTES,
            &["cyclic_evidence_v1"],
            bytes,
        )
        .map_err(|e| invalid(e.into_io("cyclic evidence layout")))?;
        let record: Record = antecedent_io::from_cbor(
            parts.first().ok_or_else(|| invalid("missing evidence section"))?,
        )
        .map_err(invalid)?;
        if record.version != 2
            || record.report.identity != expected_identity
            || digest(
                "antecedent.cyclic_evidence.v1",
                &(&record.source_identity, &record.schema, &record.selectors),
            )? != expected_identity
        {
            return Err(invalid("cyclic evidence identity mismatch"));
        }
        let (request, source) =
            consume_cyclic_family_with_request(&record.source, &record.source_identity, ctx)?;
        let evidence = Self::build(
            &request,
            &source,
            record.source.into(),
            record.schema,
            record.selectors,
            ctx,
        )?;
        if evidence.report != record.report {
            return Err(invalid("cyclic evidence differs from original source replay"));
        }
        Ok(evidence)
    }
    /// Existing decision engine's affine-mean route, conditional on the reported source.
    /// A posterior/ bootstrap column is not silently averaged into an outcome law.
    /// # Errors
    /// Wrong evidence port, unresolved slots, incompatible decision contract or numeric failure.
    pub fn decide_means(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        let mut coordinates = Vec::new();
        let mut means = Vec::new();
        for column in &self.report.columns {
            if column.port != "mean" || column.samples.len() != 1 || column.samples[0].is_none() {
                return Err(invalid(
                    "decision needs point/mean evidence; unresolved states and other uncertainty ports are retained, not coerced",
                ));
            }
            coordinates
                .push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
            means.push(column.samples[0].ok_or_else(|| invalid("missing mean"))?);
        }
        let source = antecedent_design::decision_eval::MeanSource {
            coordinates,
            means,
            estimate: antecedent_design::decision_eval::MeanEstimate::Frequentist,
            provider_id: "native:cyclic_model_conditional".into(),
            snapshot_id: self.report.source_identity.clone(),
            causal_contract_id: self.report.identity.clone(),
            rng_id: "deterministic_native_projection".into(),
        };
        let mut result =
            antecedent_design::decision_eval::evaluate_contract_on_means(contract, &source)
                .map_err(|e| invalid(format!("{e:?}")))?;
        result.assumptions.push(self.report.scope.clone());
        let result = antecedent_design::decision_artifact::mean_result_to_json(&result, &source)
            .map_err(invalid)?;
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":self.report.scope,"result":serde_json::from_str::<serde_json::Value>(&result).map_err(invalid)?}),
        )
    }
    /// Evaluate conservative solution/scenario utility enclosures with the existing
    /// robust set decision engine. No equilibrium probabilities or nonparametric
    /// identification claim are inferred. Unresolved boxes remain inside the set.
    /// # Errors
    /// Missing/undefined states, incompatible quantities, probability criteria,
    /// hard chance constraints or an invalid existing decision contract.
    pub fn decide_enclosures(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
    ) -> Result<serde_json::Value, CyclicArtifactError> {
        use antecedent_design::{
            decision_adapters::{IdentifiedUtility, evaluate_identified_sets},
            decision_contract::{AdmissibilityRules, AdmissibleDecisionContract},
            decision_robustness::AtomSupport,
        };
        if !contract.constraints.is_empty() {
            return Err(invalid(
                "a solution enclosure does not supply a chance-constraint probability law",
            ));
        }
        let coordinates = self
            .report
            .columns
            .iter()
            .map(|c| ScientificQuantity::try_from(c.quantity.clone()).map_err(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let mut utilities = Vec::new();
        for action in &contract.actions {
            let mut inputs = Vec::new();
            for quantity in &action.inputs {
                let i = coordinates
                    .iter()
                    .position(|c| c.require_same_coordinate(quantity).is_ok())
                    .ok_or_else(|| {
                        invalid("solution decision quantity does not match original evidence")
                    })?;
                let column = &self.report.columns[i];
                if column.port != "scenario" || column.numerical_enclosures.is_empty() {
                    return Err(invalid(
                        "solution decision requires a declared numerical solution/scenario enclosure",
                    ));
                }
                if column.mass_bounds.is_some()
                    && (column
                        .diagnostics
                        .get("positive_evidence_certified")
                        .and_then(serde_json::Value::as_bool)
                        != Some(true)
                        || column
                            .mass_bounds
                            .as_ref()
                            .is_some_and(|m| m.len() != column.numerical_enclosures.len()))
                {
                    return Err(invalid("posterior selection evidence is not certified positive"));
                }
                let mut lower = f64::INFINITY;
                let mut upper = f64::NEG_INFINITY;
                for (k, bound) in column.numerical_enclosures.iter().enumerate() {
                    if column
                        .mass_bounds
                        .as_ref()
                        .is_some_and(|m| m.get(k).is_some_and(|p| p.upper == 0.0))
                    {
                        continue;
                    }
                    let value=bound.ok_or_else(||invalid("undefined or unresolved selected-state mass has no decision utility enclosure"))?;
                    lower = lower.min(value.lower);
                    upper = upper.max(value.upper);
                }
                if !lower.is_finite() || !upper.is_finite() {
                    return Err(invalid("empty/undefined solution enclosure"));
                }
                inputs.push(Interval { lower, upper });
            }
            let enclosure = enclose_utility(&action.utility, &inputs)?;
            utilities.push(IdentifiedUtility {
                action_id: action.id.clone(),
                utility: antecedent_core::IdentifiedSet::try_new(enclosure.lower, enclosure.upper)
                    .map_err(invalid)?,
                hard_exclusions: vec![],
                support: AtomSupport::unassessed(),
            });
        }
        let result = evaluate_identified_sets(
            &AdmissibleDecisionContract {
                contract: contract.clone(),
                rules: AdmissibilityRules::default(),
            },
            &utilities,
        )
        .map_err(|e| invalid(format!("{e:?}")))?;
        let actions=result.actions.iter().map(|a|serde_json::json!({"id":a.id,"utility":a.utility,"eligible":a.eligible,"dominated_by":a.dominated_by,"possibly_optimal":a.possibly_optimal,"necessarily_optimal":a.necessarily_optimal,"max_regret":a.max_regret})).collect::<Vec<_>>();
        Ok(
            serde_json::json!({"evidence_identity":self.report.identity,"source_identity":self.report.source_identity,"scope":"conditional_model_solution_outer_enclosure","actions":actions,"verdict":format!("{:?}",result.verdict),"lower_leader":result.lower_leader,"upper_leader":result.upper_leader,"conflicting_leaders":result.conflicting_leaders,"selection_probabilities_inferred":false}),
        )
    }
    /// Build the existing aligned joint-law artifact for a complete posterior column set.
    /// Bootstrap, root sets, finite mass bounds and failed slots cannot be coerced to a belief law.
    /// # Errors
    /// Different axes/alignments, missing mass, wrong port or invalid artifact coordinates.
    pub fn joint_law(&self) -> Result<DistributionArtifact, CyclicArtifactError> {
        let first = self.report.columns.first().ok_or_else(|| invalid("empty evidence"))?;
        if !matches!(first.port.as_str(), "parameter_posterior" | "interventional_law") {
            return Err(invalid("evidence port is not a posterior or outcome law"));
        }
        let count = first.samples.len();
        if count == 0 {
            return Err(invalid("empty posterior"));
        }
        let mut quantities = Vec::new();
        for column in &self.report.columns {
            if column.port != first.port
                || column.uncertainty_axes != first.uncertainty_axes
                || column.samples.len() != count
                || column.samples.iter().any(Option::is_none)
                || column.mass_bounds.is_some()
            {
                return Err(invalid(
                    "joint law needs complete aligned draws with identical uncertainty meaning; failed or unresolved mass is retained",
                ));
            }
            quantities
                .push(ScientificQuantity::try_from(column.quantity.clone()).map_err(invalid)?);
        }
        let mut rows = Vec::new();
        for i in 0..count {
            for column in &self.report.columns {
                rows.push(column.samples[i].ok_or_else(|| invalid("missing draw"))?);
            }
        }
        let meaning = if first.port == "parameter_posterior" {
            DistributionMeaningWire::CausalFunctionalPosterior
        } else {
            DistributionMeaningWire::InterventionalPredictive
        };
        let identity = DistributionIdentity::new(
            meaning,
            &quantities,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: self.report.source_identity.clone(),
                provider_id: "native:cyclic_model_conditional".into(),
                rng_id: format!("retained:{}", self.report.source_identity),
                snapshot_id: self.report.source_identity.clone(),
                causal_contract_id: self.report.identity.clone(),
            },
        )
        .map_err(invalid)?;
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [count, quantities.len()],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Unmeasured,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            rows,
        )
        .map_err(invalid)
    }
}

fn enclose_utility(
    expression: &antecedent_design::decision_contract::UtilityExpr,
    inputs: &[Interval],
) -> Result<Interval, CyclicArtifactError> {
    use antecedent_design::decision_contract::UtilityExpr as U;
    let output = match expression {
        U::Const(value) => Interval { lower: *value, upper: *value },
        U::Input(i) => *inputs.get(*i).ok_or_else(|| invalid("utility input index"))?,
        U::Neg(a) => enclose_utility(a, inputs)?.scale_outward(-1.0),
        U::Add(a, b) => enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?),
        U::Sub(a, b) => {
            enclose_utility(a, inputs)?.add_outward(enclose_utility(b, inputs)?.scale_outward(-1.0))
        }
        U::Mul(a, b) => enclose_utility(a, inputs)?.multiply_outward(enclose_utility(b, inputs)?),
        U::Min(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.min(b.lower), upper: a.upper.min(b.upper) }
        }
        U::Max(a, b) => {
            let a = enclose_utility(a, inputs)?;
            let b = enclose_utility(b, inputs)?;
            Interval { lower: a.lower.max(b.lower), upper: a.upper.max(b.upper) }
        }
    };
    if !output.lower.is_finite() || !output.upper.is_finite() {
        return Err(invalid("nonfinite utility enclosure"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Study;
    #[test]
    fn retained_evidence_preserves_ports_identity_and_replay_without_calibration_gate() {
        let ctx = ExecutionContext::for_tests(8);
        let request = R::OuStationary {
            model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck {
                drift: vec![-2.0],
                center: vec![3.0],
                diffusion: vec![4.0],
            },
        };
        let prepared = Study::prepare_cyclic_family(&request, &ctx).unwrap();
        let schema = CyclicEvidenceSchema {
            id: "physical-system".into(),
            population: "system".into(),
            variables: vec![("position".into(), "meter".into())],
        };
        let selectors = vec![CyclicEvidenceSelector::OuMean { time: None, variable: 0 }];
        let evidence = prepared.cyclic_evidence(schema.clone(), selectors.clone(), &ctx).unwrap();
        assert_eq!(prepared.inspect().preparation_evaluations, 0);
        assert_eq!(evidence.report.columns[0].samples, vec![Some(3.0)]);
        assert_eq!(evidence.report.columns[0].port, "mean");
        assert!(matches!(
            evidence.envelopes[0].binding().calibration,
            EvidenceCalibration::ScopeNotAssessed
        ));
        assert!(evidence.envelopes[0].binding().origins.training.is_none());
        let bytes = evidence.export(&ctx).unwrap();
        let consumed = CyclicEvidence::consume(&bytes, &evidence.report.identity, &ctx).unwrap();
        assert_eq!(consumed.report, evidence.report);
        assert!(CyclicEvidence::consume(&bytes, &"0".repeat(64), &ctx).is_err());
        let mut different = schema;
        different.variables[0].1 = "centimeter".into();
        let refreshed = prepared.cyclic_evidence(different, selectors, &ctx).unwrap();
        assert_ne!(refreshed.report.identity, evidence.report.identity);
        assert!(evidence.joint_law().is_err());
        ctx.cancellation.cancel();
        assert!(
            prepared
                .cyclic_evidence(refreshed.schema.clone(), refreshed.selectors.clone(), &ctx)
                .is_err()
        );
    }
}
