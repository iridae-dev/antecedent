//! A native response as a typed claim a decision or inverse flow can consume.
//!
//! A [`CausalResponse`] says what the estimator computed, but a decision needs to
//! know which scientific quantity each number is, how well supported it is, who
//! produced it and what law it supplies. [`NativeResponseClaim`] carries exactly
//! that: one [`ScientificQuantity`] per grid point (variable, units, population,
//! regime, horizon, functional, conditioning, transform), a support label per
//! coordinate, native provider trust, the calibration status and the identity of
//! the identified program the response was checked against.
//!
//! The coordinates are derived from the response's own estimand and grid and
//! checked against the [`ProgramBinding`], so a response computed for another
//! outcome, treatment, grid or population is refused instead of silently relabeled.
//! [`NativeResponseClaim::decision_source`] then hands the decision engine a mean
//! source, or an aligned joint law when the response retained credible draws of
//! every coordinate. A point or mean response never becomes an outcome law: a
//! requirement that needs a distribution refuses with
//! `native_claims.source_not_supplied`.
//!
//! The joint law is a posterior over the *causal mean curve*
//! (`CausalFunctionalPosterior`), not an interventional outcome law, so the
//! decision engine still refuses outcome-threshold inputs on it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalResponse, CausalSchema, CredibleDraws, ExternalProgramClaim, ExternalRefusal,
    IntervalInterpretation, ProgramBinding, ResponseFunctional, ResponseIdentification,
    ResponseUncertainty, ResponseValue, ScientificQuantity, SupportStatus, VariableId,
    check_external_against_program, check_static_point_labels,
};
use antecedent_design::decision_contract::{
    DecisionContractError, SourceRepresentation, SourceRequirement,
};
use antecedent_design::decision_eval::MeanSource;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

/// Everything about a native response that its result type does not carry.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeResponseContext {
    /// The identified program and request the response must answer.
    pub program: ProgramBinding,
    /// Data snapshot identity the response was estimated from.
    pub snapshot_id: String,
    /// RNG algorithm, seed and stream identity, or a deterministic marker.
    pub rng_id: String,
    /// Calibration status of the response's uncertainty. A response with no
    /// measured calibration record is `Unmeasured`; a point value is `PointOnly`.
    pub calibration: DistributionCalibration,
}

/// A coordinate withheld from a decision source for lack of support.
#[derive(Clone, Debug, PartialEq)]
pub struct WithheldCoordinate {
    /// Position in the claim's grid.
    pub index: usize,
    /// The withheld coordinate.
    pub coordinate: ScientificQuantity,
    /// Why: its support label.
    pub status: SupportStatus,
}

/// The representation handed to the decision engine.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeDecisionSource {
    /// Means only; answers an expectation of an affine utility and nothing else.
    Mean(MeanSource),
    /// Aligned joint draws of the mean curve's posterior.
    JointLaw(Box<DistributionArtifact>),
}

/// A decision source with the support, trust and calibration that travel with it.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeDecisionInput {
    /// The source for the decision engine.
    pub source: NativeDecisionSource,
    /// The representation that satisfied the requirement.
    pub representation: SourceRepresentation,
    /// Coordinates of the source's columns, in order.
    pub coordinates: Vec<ScientificQuantity>,
    /// Support label of each of those coordinates.
    pub point_status: Vec<SupportStatus>,
    /// Coordinates left out of a mean source, or masked in a joint law, because
    /// they are outside empirical support or lack evidence.
    pub withheld: Vec<WithheldCoordinate>,
    /// Provider trust; always native for a claim built here.
    pub trust: DistributionTrust,
    /// Calibration status of the response's uncertainty.
    pub calibration: DistributionCalibration,
    /// Durable identity of the identified program and request.
    pub program_identity: String,
}

/// A native response with its scientific coordinates, support and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeResponseClaim {
    coordinates: Vec<ScientificQuantity>,
    means: Vec<f64>,
    point_status: Vec<SupportStatus>,
    summary_status: SupportStatus,
    draws: Option<CredibleDraws>,
    calibration: DistributionCalibration,
    program_identity: String,
    contract_id: String,
    provenance_id: String,
    snapshot_id: String,
    rng_id: String,
}

fn refusal(
    code: &'static str,
    stage: &'static str,
    detail: &str,
    expected: Option<String>,
    supplied: Option<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage,
        detail: detail.to_owned(),
        offending: None,
        expected,
        supplied,
        capability: None,
        remedy: Some(remedy),
    }
}

fn bind_refusal(detail: &str, expected: String, supplied: String) -> ExternalRefusal {
    refusal(
        antecedent_core::reason_code!("quantity_semantics_mismatch"),
        "bind",
        detail,
        Some(expected),
        Some(supplied),
        "evaluate the response on the identified program's grid and variables",
    )
}

fn representation_name(representation: SourceRepresentation) -> &'static str {
    match representation {
        SourceRepresentation::Mean => "mean",
        SourceRepresentation::MeanAndCovariance => "mean_and_covariance",
        SourceRepresentation::Cdf => "cdf",
        SourceRepresentation::QuantileFunction => "quantile_function",
        SourceRepresentation::MarginalDraws => "marginal_draws",
        SourceRepresentation::JointDraws => "joint_draws",
    }
}

fn names(representations: &[SourceRepresentation]) -> String {
    representations.iter().map(|r| representation_name(*r)).collect::<Vec<_>>().join(",")
}

impl NativeResponseContext {
    // The refusal is the cold path of a once-per-claim check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    fn check(&self) -> Result<(), ExternalRefusal> {
        self.program.validate()?;
        if self.snapshot_id.trim().is_empty() || self.rng_id.trim().is_empty() {
            return Err(refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "bind",
                "native_claims.invalid_context",
                None,
                None,
                "name the data snapshot and the RNG identity of the native response",
            ));
        }
        Ok(())
    }
}

/// The mean curve's outcome, treatment and dose grid, read from the response itself.
#[allow(clippy::result_large_err)]
fn mean_curve(
    response: &CausalResponse,
) -> Result<(VariableId, VariableId, Vec<f64>), ExternalRefusal> {
    let ResponseFunctional::MeanCurve { outcome, treatment } = &response.estimand else {
        return Err(refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "bind",
            "native_claims.unsupported_estimand",
            Some("mean_curve".to_owned()),
            None,
            "only a mean response curve carries one coordinate per dose",
        ));
    };
    let doses = treatment.grid.values().map_err(|_| {
        bind_refusal(
            "native_claims.grid_mismatch",
            "a valid dose grid".to_owned(),
            "invalid".into(),
        )
    })?;
    Ok((*outcome, treatment.variable, doses))
}

#[allow(clippy::result_large_err)]
fn point_surface<'a>(
    response: &'a CausalResponse,
    doses: &[f64],
) -> Result<&'a [f64], ExternalRefusal> {
    let ResponseIdentification::PointIdentified(value) = &response.estimate else {
        return Err(refusal(
            antecedent_core::reason_code!("effect_not_identified"),
            "bind",
            "native_claims.response_not_point_identified",
            Some("point_identified".to_owned()),
            Some(response.identification_status.as_str().to_owned()),
            "an envelope, graph-dependent or unidentified response supplies no point mean",
        ));
    };
    let ResponseValue::Surface { grid, dimension, mean } = value else {
        return Err(refusal(
            antecedent_core::reason_code!("route_not_supported"),
            "bind",
            "native_claims.unsupported_estimand",
            Some("surface".to_owned()),
            None,
            "only a mean response curve carries one coordinate per dose",
        ));
    };
    let same_grid = grid.len() == doses.len()
        && grid.iter().zip(doses).all(|(a, b)| a.to_bits() == b.to_bits());
    if *dimension != 1
        || !same_grid
        || mean.len() != doses.len()
        || mean.iter().any(|m| !m.is_finite())
    {
        return Err(bind_refusal(
            "native_claims.grid_mismatch",
            format!("{} finite means on the estimand grid", doses.len()),
            format!("{} means on a {}-point grid", mean.len(), grid.len()),
        ));
    }
    Ok(&mean[..])
}

#[allow(clippy::result_large_err)]
fn variable_name(
    schema: &CausalSchema,
    id: VariableId,
) -> Result<&antecedent_core::VariableSchema, ExternalRefusal> {
    schema.get(id).map_err(|_| {
        refusal(
            antecedent_core::reason_code!("unknown_variable"),
            "bind",
            "native_claims.unknown_variable",
            None,
            Some(id.raw().to_string()),
            "derive coordinates with the schema the response was estimated on",
        )
    })
}

/// The scientific coordinates of a native mean-curve response, one per grid point.
///
/// This is the native getter for a response's coordinates: it reads the outcome,
/// treatment and grid from the response's own estimand, names them with `schema`,
/// and checks the result against the identified program. No query is reconstructed
/// from anywhere else.
///
/// # Errors
/// Refuses a response that is not a mean curve, a variable unknown to `schema`, a
/// schema unit that contradicts the program's outcome units, and any substitution
/// of treatment, outcome, grid or population against the program
/// (`program_binding.*` details).
// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
pub fn native_response_coordinates(
    response: &CausalResponse,
    schema: &CausalSchema,
    context: &NativeResponseContext,
) -> Result<Vec<ScientificQuantity>, ExternalRefusal> {
    context.check()?;
    let program = &context.program;
    let (outcome, treatment, doses) = mean_curve(response)?;
    let outcome = variable_name(schema, outcome)?;
    let treatment = variable_name(schema, treatment)?;
    if let Some(unit) = &outcome.unit {
        if &**unit != program.outcome_units.as_str() {
            return Err(bind_refusal(
                "native_claims.units_conflict",
                program.outcome_units.clone(),
                unit.to_string(),
            ));
        }
    }
    let derived = ProgramBinding {
        treatment_id: treatment.name.to_string(),
        outcome_id: outcome.name.to_string(),
        dose_grid: doses.clone(),
        ..program.clone()
    };
    let claim = ExternalProgramClaim {
        contract_id: program.contract_id.clone(),
        graph_id: program.graph_id.clone(),
        declared_identity: program.identity(),
        treatment_id: derived.treatment_id.clone(),
        outcome_id: derived.outcome_id.clone(),
        population_id: program.population_id.clone(),
        doses,
        dose_units: program.dose_units.clone(),
        quantities: derived.expected_quantities(),
    };
    check_external_against_program(program, &claim)?;
    Ok(claim.quantities)
}

/// Credible draws of every coordinate that the response retained, if any.
fn aligned_draws(response: &CausalResponse, width: usize) -> Option<CredibleDraws> {
    match &response.uncertainty {
        ResponseUncertainty::PointwiseBand {
            interpretation: IntervalInterpretation::Credible,
            draws: Some(draws),
            ..
        } if draws.n_draws > 0 && draws.n_coordinates() == width => Some(draws.clone()),
        _ => None,
    }
}

impl NativeResponseClaim {
    /// Build the claim from a native point-identified mean-curve response.
    ///
    /// # Errors
    /// Everything [`native_response_coordinates`] refuses, a response that is not
    /// point identified (`native_claims.response_not_point_identified`), a surface
    /// that is not the estimand's grid (`native_claims.grid_mismatch`), missing or
    /// inconsistent per-coordinate support (`coordinate_support.*`,
    /// `native_claims.point_support_missing`), and an invalid context.
    // The refusal is the cold path of a once-per-claim check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    pub fn from_response(
        response: &CausalResponse,
        schema: &CausalSchema,
        context: &NativeResponseContext,
    ) -> Result<Self, ExternalRefusal> {
        let coordinates = native_response_coordinates(response, schema, context)?;
        let (_, _, doses) = mean_curve(response)?;
        let means = point_surface(response, &doses)?.to_vec();
        let Some(labels) = response.support.point_status.as_deref() else {
            return Err(refusal(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                "support",
                "native_claims.point_support_missing",
                Some(format!("{} coordinate labels", coordinates.len())),
                None,
                "a decision input carries a support label for every coordinate",
            ));
        };
        check_static_point_labels(labels, response.support.status, coordinates.len())?;
        Ok(Self {
            draws: aligned_draws(response, coordinates.len()),
            point_status: labels.to_vec(),
            summary_status: response.support.status,
            coordinates,
            means,
            calibration: context.calibration,
            program_identity: context.program.identity(),
            contract_id: context.program.contract_id.clone(),
            provenance_id: response.provenance_id.to_string(),
            snapshot_id: context.snapshot_id.clone(),
            rng_id: context.rng_id.clone(),
        })
    }

    /// One scientific coordinate per grid point, in grid order.
    #[must_use]
    pub fn coordinates(&self) -> &[ScientificQuantity] {
        &self.coordinates
    }

    /// The mean response at each coordinate.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Support label of each coordinate.
    #[must_use]
    pub fn point_status(&self) -> &[SupportStatus] {
        &self.point_status
    }

    /// The response's summary support label (the worst coordinate).
    #[must_use]
    pub const fn support_status(&self) -> SupportStatus {
        self.summary_status
    }

    /// Provider trust: always native, because the claim is built in process from a
    /// typed response and never from artifact metadata.
    #[must_use]
    pub const fn trust(&self) -> DistributionTrust {
        DistributionTrust::NativeLicensed
    }

    /// Calibration status of the response's uncertainty.
    #[must_use]
    pub const fn calibration(&self) -> DistributionCalibration {
        self.calibration
    }

    /// Durable identity of the identified program and request.
    #[must_use]
    pub fn program_identity(&self) -> &str {
        &self.program_identity
    }

    /// Provenance operation id of the native response.
    #[must_use]
    pub fn provenance_id(&self) -> &str {
        &self.provenance_id
    }

    /// Whether the response retained aligned credible draws of every coordinate.
    #[must_use]
    pub const fn has_joint_law(&self) -> bool {
        self.draws.is_some()
    }

    /// Produce what the decision engine needs, consuming the claim.
    ///
    /// # Errors
    /// See [`Self::decision_source`].
    // The refusal is the cold path of a once-per-claim check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    pub fn into_decision_source(
        self,
        requirement: &SourceRequirement,
    ) -> Result<NativeDecisionInput, ExternalRefusal> {
        self.decision_source(requirement)
    }

    /// Produce what the decision engine needs for `requirement`.
    ///
    /// A mean is available always; aligned joint draws only when the response
    /// retained them. A requirement that needs a distribution (a probability, a
    /// quantile, a nonlinear utility) is refused when only means exist.
    ///
    /// # Errors
    /// `native_claims.source_not_supplied` when no available representation
    /// satisfies the requirement, `native_claims.no_supported_coordinate` when every
    /// coordinate lacks support, and `native_claims.joint_law_invalid` when the
    /// retained draws cannot form a valid artifact.
    // The refusal is the cold path of a once-per-claim check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    pub fn decision_source(
        &self,
        requirement: &SourceRequirement,
    ) -> Result<NativeDecisionInput, ExternalRefusal> {
        let mut available = vec![SourceRepresentation::Mean];
        if self.draws.is_some() {
            available.push(SourceRepresentation::JointDraws);
        }
        let mode = requirement.check(&available).map_err(|error| match error {
            DecisionContractError::MissingSource { needed } => refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                "evaluate",
                "native_claims.source_not_supplied",
                Some(names(&needed)),
                Some(names(&available)),
                "a point or mean response is not an outcome law; supply aligned joint draws",
            ),
            other => other.to_refusal(),
        })?;
        let withheld: Vec<WithheldCoordinate> = self
            .point_status
            .iter()
            .enumerate()
            .filter(|(_, status)| status.severity() > SupportStatus::Extrapolative.severity())
            .map(|(index, status)| WithheldCoordinate {
                index,
                coordinate: self.coordinates[index].clone(),
                status: *status,
            })
            .collect();
        if withheld.len() == self.coordinates.len() {
            return Err(refusal(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                "evaluate",
                "native_claims.no_supported_coordinate",
                Some("at least one supported coordinate".to_owned()),
                Some(self.summary_status.as_str().to_owned()),
                "no coordinate of this response may feed a decision",
            ));
        }
        let (source, coordinates, point_status) = match mode.representation {
            SourceRepresentation::JointDraws => {
                let artifact = self.joint_artifact()?;
                (
                    NativeDecisionSource::JointLaw(Box::new(artifact)),
                    self.coordinates.clone(),
                    self.point_status.clone(),
                )
            }
            _ => self.mean_source(&withheld),
        };
        Ok(NativeDecisionInput {
            source,
            representation: mode.representation,
            coordinates,
            point_status,
            withheld,
            trust: self.trust(),
            calibration: self.calibration,
            program_identity: self.program_identity.clone(),
        })
    }

    fn mean_source(
        &self,
        withheld: &[WithheldCoordinate],
    ) -> (NativeDecisionSource, Vec<ScientificQuantity>, Vec<SupportStatus>) {
        let kept: Vec<usize> = (0..self.coordinates.len())
            .filter(|i| withheld.iter().all(|w| w.index != *i))
            .collect();
        let coordinates: Vec<ScientificQuantity> =
            kept.iter().map(|i| self.coordinates[*i].clone()).collect();
        let source = MeanSource {
            coordinates: coordinates.clone(),
            means: kept.iter().map(|i| self.means[*i]).collect(),
            provider_id: format!("native:{}", self.provenance_id),
            snapshot_id: self.snapshot_id.clone(),
            causal_contract_id: self.contract_id.clone(),
            rng_id: "none:mean_grid".to_owned(),
        };
        let status = kept.iter().map(|i| self.point_status[*i]).collect();
        (NativeDecisionSource::Mean(source), coordinates, status)
    }

    // The refusal is the cold path of a once-per-claim check; boxing would not pay.
    #[allow(clippy::result_large_err)]
    fn joint_artifact(&self) -> Result<DistributionArtifact, ExternalRefusal> {
        let invalid = |why: String| {
            refusal(
                antecedent_core::reason_code!("invalid_argument"),
                "evaluate",
                "native_claims.joint_law_invalid",
                None,
                Some(why),
                "retain finite, aligned credible draws of every coordinate",
            )
        };
        let draws = self.draws.as_ref().ok_or_else(|| invalid("no retained draws".to_owned()))?;
        let width = self.coordinates.len();
        let mut rows = Vec::with_capacity(draws.n_draws * width);
        for draw in 0..draws.n_draws {
            for column in 0..width {
                let value = draws
                    .column(column)
                    .and_then(|values| values.get(draw))
                    .copied()
                    .ok_or_else(|| invalid("ragged draws".to_owned()))?;
                rows.push(value);
            }
        }
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::CausalFunctionalPosterior,
            &self.coordinates,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: self.provenance_id.clone(),
                provider_id: "native:antecedent".to_owned(),
                rng_id: self.rng_id.clone(),
                snapshot_id: self.snapshot_id.clone(),
                causal_contract_id: self.contract_id.clone(),
            },
        )
        .map_err(|error| invalid(error.to_string()))?;
        let metadata = DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".to_owned(), "quantity".to_owned()],
            shape: [draws.n_draws, width],
            weights: None,
            supported: Some(
                self.point_status
                    .iter()
                    .map(|status| status.severity() <= SupportStatus::Extrapolative.severity())
                    .collect(),
            ),
            calibration: self.calibration,
            trust: DistributionTrust::NativeLicensed,
            legacy_posterior: None,
            legacy_bindings: None,
        };
        DistributionArtifact::new(metadata, rows).map_err(|error| invalid(error.to_string()))
    }
}
