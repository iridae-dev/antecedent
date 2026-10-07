//! Independent artifact of the bounded binary nested-Markov likelihood pilot (2.3A A3,
//! record `2.3A.X4.binary_nested_markov_pilot`). Internal and non-routed: the public
//! route `antecedent.transport.binary_nested_markov` stays closed
//! (`cell_not_licensed`, `nested_markov.route_frozen`) until calibration is measured, so
//! this artifact publishes a fit and a point contrast, never an interval or posterior.
//!
//! Format version 1. The artifact carries the declared ADMG (variables, directed and
//! bidirected edges), the graph id and parameterization id, the regime-specific cell
//! counts, the fit options, and the fit receipt: the nested-Markov parameters, the 16
//! fitted cell probabilities, the exact log-likelihood, the fit diagnostics (method,
//! sweeps, convergence, saturated and model log-likelihood, deviance, the four
//! equality-constraint residuals of the empirical law, normalization, boundary margin),
//! the model-versus-plug-in target contrast, and **three separate status fields**:
//! nonparametric identification, likelihood fit and inferential standing. Calibration is
//! `unmeasured`.
//!
//! A consumer trusts none of the receipt. Under its own bounds it checks the two digests,
//! rebuilds the declared graph and counts, re-runs the pilot (scope check, fit, residuals
//! and the target contrast against the empirical plug-in of the same identifying formula)
//! and accepts only a receipt identical bit for bit. A changed graph, regime or count
//! therefore refuses: outside the pilot class with the engine's own
//! `nested_markov.outside_binary_pilot` refusal, or as a receipt that does not replay.
//! The digests are separate: the premises digest (graph, parameterization, options, regime
//! structure) and the data-identity digest (every count bit for bit). A consumer holding
//! an expected identity passes a [`NestedMarkovExpectation`].
//!
//! What replay does not protect against: a producer that supplies fabricated counts
//! consistently, and that the declared graph describes the world; the graph and its
//! identification standing are declared premises bound into the premises digest.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::EstimationError;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, ConstraintResiduals, ConstraintStatus, FitMethod, FitOptions,
    IdentificationStanding, InferenceStanding, LikelihoodCheck, LikelihoodFitStanding,
    MAX_ITERATIONS_CAP, MAX_OBSERVED, NestedMarkovInput, NestedMarkovRefusal, PilotReport, Regime,
    RegimeCounts, evaluate_nested_markov_pilot,
};
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const NESTED_MARKOV_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const NESTED_MARKOV_ARTIFACT_FEATURE: &str = "binary_nested_markov_pilot_v1";
/// Id of the one selected graph (Verma graph, four binary variables).
pub const NESTED_MARKOV_GRAPH_ID: &str = "binary_admg.verma.x1_x2_x3_x4.v1";
/// Id of the district (c-factor) Mobius parameterization.
pub const NESTED_MARKOV_PARAMETERIZATION: &str = "district_mobius_c_factor.v1";
/// The only calibration status of this cell until coverage is measured.
pub const NESTED_MARKOV_CALIBRATION: &str = "unmeasured";

/// Why a nested-Markov artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum NestedMarkovArtifactError {
    /// The feature marker, a tag or a claim is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored count or size exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored counts.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The stored identity is not the one the consumer expects.
    #[error("identity differs from the consumer's expectation: {0}")]
    ExpectationMismatch(&'static str),
    /// The stored graph or parameterization id is not the selected pilot's.
    #[error("graph or parameterization id is not the selected pilot's")]
    GraphIdMismatch,
    /// The re-fitted receipt differs from the stored receipt.
    #[error("fit receipt does not replay")]
    ReceiptMismatch,
}

impl From<NestedMarkovArtifactError> for IoError {
    fn from(error: NestedMarkovArtifactError) -> Self {
        Self::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("nested markov artifact: {error}"),
        }
    }
}

fn estimation_refusal(error: &EstimationError) -> IoError {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => {
            IoError::Refused { code, message: message.clone() }
        }
        other => IoError::Convert(other.to_string()),
    }
}

fn pilot_refusal(refusal: &NestedMarkovRefusal) -> IoError {
    estimation_refusal(&refusal.error)
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct NestedMarkovConsumeLimits {
    /// Largest stored sweep bound the consumer will replay.
    pub max_iterations: usize,
    /// Most regimes.
    pub max_regimes: usize,
    /// Most observed variables.
    pub max_variables: usize,
    /// Most cells of one regime table.
    pub max_cells: usize,
}

impl Default for NestedMarkovConsumeLimits {
    fn default() -> Self {
        Self {
            max_iterations: 200_000,
            max_regimes: 8,
            max_variables: MAX_OBSERVED,
            max_cells: 4096,
        }
    }
}

/// An identity a consumer expects the artifact to carry; `None` skips that check.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NestedMarkovExpectation {
    /// Expected premises digest (graph, parameterization, options, regime structure).
    pub premises_digest: Option<String>,
    /// Expected data-identity digest (the counts).
    pub data_digest: Option<String>,
}

/// The stored declared ADMG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NestedGraphWire {
    /// Variable names in coordinate order.
    pub variables: Vec<String>,
    /// Directed edges `(from, to)`, sorted.
    pub directed: Vec<(usize, usize)>,
    /// Bidirected edges `(low, high)`, sorted.
    pub bidirected: Vec<(usize, usize)>,
}

impl NestedGraphWire {
    fn from_graph(graph: &AdmgDeclaration) -> Self {
        let mut directed = graph.directed.clone();
        directed.sort_unstable();
        directed.dedup();
        let mut bidirected: Vec<(usize, usize)> =
            graph.bidirected.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        bidirected.sort_unstable();
        bidirected.dedup();
        Self { variables: graph.variables.clone(), directed, bidirected }
    }

    fn to_graph(&self) -> AdmgDeclaration {
        AdmgDeclaration {
            variables: self.variables.clone(),
            directed: self.directed.clone(),
            bidirected: self.bidirected.clone(),
        }
    }
}

/// One stored regime table: its kind, domain sizes (premises) and cells (data).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedRegimeWire {
    /// `observational` or `interventional`.
    pub regime: String,
    /// Variables an interventional regime fixes; empty for the observational one.
    pub fixed: Vec<usize>,
    /// Declared levels per variable.
    pub levels: Vec<usize>,
    /// Cell counts, `x1` most significant.
    pub cells: Vec<f64>,
}

impl NestedRegimeWire {
    fn from_counts(counts: &RegimeCounts) -> Self {
        let (regime, fixed) = match &counts.regime {
            Regime::Observational => ("observational", Vec::new()),
            Regime::Interventional(fixed) => ("interventional", fixed.clone()),
        };
        Self {
            regime: regime.into(),
            fixed,
            levels: counts.levels.clone(),
            cells: counts.cells.clone(),
        }
    }

    fn to_counts(&self) -> Result<RegimeCounts, NestedMarkovArtifactError> {
        let regime = match self.regime.as_str() {
            "observational" => Regime::Observational,
            "interventional" => Regime::Interventional(self.fixed.clone()),
            _ => return Err(NestedMarkovArtifactError::UnsupportedSemantics("regime kind")),
        };
        Ok(RegimeCounts { regime, levels: self.levels.clone(), cells: self.cells.clone() })
    }
}

/// The stored fit options.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedOptionsWire {
    /// Sweep bound.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Refuse data whose empirical constraint residual exceeds this, when set.
    pub refuse_constraint_residual_above: Option<f64>,
}

impl NestedOptionsWire {
    const fn from_options(options: &FitOptions) -> Self {
        Self {
            max_iterations: options.max_iterations,
            tolerance: options.tolerance,
            refuse_constraint_residual_above: options.refuse_constraint_residual_above,
        }
    }

    const fn to_options(self) -> FitOptions {
        FitOptions {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            refuse_constraint_residual_above: self.refuse_constraint_residual_above,
        }
    }
}

/// The stored equality-constraint residuals of the empirical law.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedResidualsWire {
    /// (E1) `X3` independent of `X1` given `X2`, for `x2 = 0, 1`.
    pub x3_independent_of_x1: [f64; 2],
    /// (E2) Verma constraint, for `x3 = 0, 1`.
    pub verma: [f64; 2],
    /// Largest absolute residual.
    pub max_abs: f64,
}

impl NestedResidualsWire {
    const fn from_residuals(r: &ConstraintResiduals) -> Self {
        Self { x3_independent_of_x1: r.x3_independent_of_x1, verma: r.verma, max_abs: r.max_abs }
    }
}

/// The stored normalization and positivity check of the fitted cells.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedNormalizationWire {
    /// Sum of the fitted cells.
    pub total: f64,
    /// `|total - 1|`.
    pub normalization_error: f64,
    /// Smallest fitted cell.
    pub min_cell: f64,
    /// Whether the cells sum to one.
    pub normalized: bool,
    /// Whether every cell is positive.
    pub positive: bool,
}

impl NestedNormalizationWire {
    const fn from_check(c: &LikelihoodCheck) -> Self {
        Self {
            total: c.total,
            normalization_error: c.normalization_error,
            min_cell: c.min_cell,
            normalized: c.normalized,
            positive: c.positive,
        }
    }
}

/// The stored fit diagnostics.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedDiagnosticsWire {
    /// `saturated_feasible` or `coordinate_ascent`.
    pub method: String,
    /// Sweeps run.
    pub iterations: usize,
    /// Whether the fit converged.
    pub converged: bool,
    /// Largest parameter change of the last sweep.
    pub final_change: Option<f64>,
    /// Log-likelihood of the saturated model.
    pub saturated_log_likelihood: f64,
    /// Log-likelihood of the nested model.
    pub model_log_likelihood: Option<f64>,
    /// `2 (saturated - model)`.
    pub deviance: Option<f64>,
    /// Constraint residuals of the empirical law.
    pub empirical_residuals: NestedResidualsWire,
    /// `satisfied` or `violated_by_data`.
    pub constraint_status: String,
    /// Normalization and positivity of the fitted cells.
    pub normalization: Option<NestedNormalizationWire>,
    /// Smallest `min(p, 1 - p)` over fitted probabilities.
    pub boundary_margin: Option<f64>,
    /// Fitted probabilities within the boundary tolerance.
    pub boundary_count: Option<usize>,
}

/// The stored nested-Markov parameters (each is the probability of level `0`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedParametersWire {
    /// `P(X1 = 0)`.
    pub a: f64,
    /// `P(X3 = 0 | X2 = x2)`.
    pub c: [f64; 2],
    /// `Q(X2 = 0 | x1)`.
    pub q2: [f64; 2],
    /// `Q(X4 = 0 | x3)`.
    pub q4: [f64; 2],
    /// `Q(X2 = 0, X4 = 0 | x1, x3)`, indexed `[x1][x3]`.
    pub g: [[f64; 2]; 2],
}

/// The stored model-versus-plug-in target contrast.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedContrastWire {
    /// Model means of `X4` under `do(X2 = 0)` and `do(X2 = 1)`.
    pub model_means: [f64; 2],
    /// Plug-in means.
    pub plugin_means: [f64; 2],
    /// Model contrast.
    pub model_contrast: f64,
    /// Plug-in contrast.
    pub plugin_contrast: f64,
    /// `model_contrast - plugin_contrast`.
    pub difference: f64,
}

/// The three statuses, kept as separate fields.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NestedStatusWire {
    /// Nonparametric identification standing.
    pub identification: String,
    /// Likelihood-fit standing.
    pub likelihood_fit: String,
    /// Inferential standing.
    pub inference: String,
}

/// The stored fit receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NestedReceiptWire {
    /// Fitted parameters.
    pub parameters: NestedParametersWire,
    /// The 16 fitted cell probabilities.
    pub cells: Vec<f64>,
    /// Exact log-likelihood at the fit.
    pub log_likelihood: f64,
    /// Diagnostics.
    pub diagnostics: NestedDiagnosticsWire,
    /// Target contrast.
    pub contrast: NestedContrastWire,
    /// Separate statuses.
    pub status: NestedStatusWire,
}

impl NestedReceiptWire {
    fn from_report(report: &PilotReport) -> Self {
        let fit = &report.fit;
        let d = &fit.diagnostics;
        let p = &fit.parameters;
        let c = &report.comparison;
        let method = match d.method {
            FitMethod::SaturatedFeasible => "saturated_feasible",
            FitMethod::CoordinateAscent => "coordinate_ascent",
        };
        let constraint_status = match d.constraint_status {
            ConstraintStatus::Satisfied => "satisfied",
            ConstraintStatus::ViolatedByData => "violated_by_data",
        };
        let identification = match report.status.identification {
            IdentificationStanding::NonparametricallyIdentified => "nonparametrically_identified",
        };
        let likelihood_fit = match report.status.likelihood_fit {
            LikelihoodFitStanding::ConvergedConstraintsSatisfied => {
                "converged_constraints_satisfied"
            }
            LikelihoodFitStanding::ConvergedConstraintsViolatedByData => {
                "converged_constraints_violated_by_data"
            }
        };
        let inference = match report.status.inference {
            InferenceStanding::IntervalWithheldCalibrationUnmeasured => {
                "interval_withheld_calibration_unmeasured"
            }
        };
        Self {
            parameters: NestedParametersWire { a: p.a, c: p.c, q2: p.q2, q4: p.q4, g: p.g },
            cells: fit.cells.to_vec(),
            log_likelihood: fit.log_likelihood,
            diagnostics: NestedDiagnosticsWire {
                method: method.into(),
                iterations: d.iterations,
                converged: d.converged,
                final_change: d.final_change,
                saturated_log_likelihood: d.saturated_log_likelihood,
                model_log_likelihood: d.model_log_likelihood,
                deviance: d.deviance,
                empirical_residuals: NestedResidualsWire::from_residuals(&d.empirical_residuals),
                constraint_status: constraint_status.into(),
                normalization: d.normalization.as_ref().map(NestedNormalizationWire::from_check),
                boundary_margin: d.boundary_margin,
                boundary_count: d.boundary_count,
            },
            contrast: NestedContrastWire {
                model_means: c.model_means,
                plugin_means: c.plugin_means,
                model_contrast: c.model_contrast,
                plugin_contrast: c.plugin_contrast,
                difference: c.difference,
            },
            status: NestedStatusWire {
                identification: identification.into(),
                likelihood_fit: likelihood_fit.into(),
                inference: inference.into(),
            },
        }
    }
}

/// Versioned nested-Markov pilot receipt with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NestedMarkovArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Id of the selected graph.
    pub graph_id: String,
    /// Id of the parameterization.
    pub parameterization: String,
    /// The declared ADMG.
    pub graph: NestedGraphWire,
    /// Regime-specific counts.
    pub regimes: Vec<NestedRegimeWire>,
    /// Fit options.
    pub options: NestedOptionsWire,
    /// The fit receipt.
    pub receipt: NestedReceiptWire,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Digest of the scientific premises.
    pub premises_digest: String,
    /// Digest of the counts.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph_id: &'a str,
    parameterization: &'a str,
    graph: &'a NestedGraphWire,
    options: &'a NestedOptionsWire,
    regimes: Vec<(&'a str, &'a [usize], &'a [usize])>,
}

#[derive(Serialize)]
struct DataView {
    tag: &'static str,
    cells: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl NestedMarkovArtifactWire {
    /// Run the pilot through the engine and build the artifact from its inputs and
    /// receipt, with the report returned alongside.
    ///
    /// # Errors
    /// The engine's refusal (outside the pilot class, positivity, numerical failure) with
    /// its reason code and `nested_markov.*` detail, or an encoding failure.
    pub fn build(
        input: &NestedMarkovInput,
        options: &FitOptions,
        ctx: &ExecutionContext,
    ) -> Result<(Self, PilotReport), IoError> {
        let report = evaluate_nested_markov_pilot(input, options, ctx)
            .map_err(|refusal| pilot_refusal(&refusal))?;
        let mut wire = Self {
            version: NESTED_MARKOV_ARTIFACT_VERSION,
            required_features: vec![NESTED_MARKOV_ARTIFACT_FEATURE.into()],
            graph_id: NESTED_MARKOV_GRAPH_ID.into(),
            parameterization: NESTED_MARKOV_PARAMETERIZATION.into(),
            graph: NestedGraphWire::from_graph(&input.graph),
            regimes: input.regimes.iter().map(NestedRegimeWire::from_counts).collect(),
            options: NestedOptionsWire::from_options(options),
            receipt: NestedReceiptWire::from_report(&report),
            calibration: NESTED_MARKOV_CALIBRATION.into(),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.check_limits(&NestedMarkovConsumeLimits::default())?;
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        Ok((wire, report))
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-fits and compares the whole receipt.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let view = PremisesView {
            tag: "binary_nested_markov_premises_v1",
            graph_id: &self.graph_id,
            parameterization: &self.parameterization,
            graph: &self.graph,
            options: &self.options,
            regimes: self
                .regimes
                .iter()
                .map(|r| (r.regime.as_str(), r.fixed.as_slice(), r.levels.as_slice()))
                .collect(),
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    /// The data-identity digest the stored counts should carry.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let view = DataView {
            tag: "binary_nested_markov_data_v1",
            cells: self
                .regimes
                .iter()
                .map(|r| r.cells.iter().map(|v| v.to_bits()).collect())
                .collect(),
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &view)?.to_hex())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure or a foreign feature or claim.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != NESTED_MARKOV_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        let unsupported = NestedMarkovArtifactError::UnsupportedSemantics;
        if wire.required_features != [NESTED_MARKOV_ARTIFACT_FEATURE] {
            return Err(unsupported("required features").into());
        }
        if wire.calibration != NESTED_MARKOV_CALIBRATION {
            return Err(unsupported("calibration is unmeasured").into());
        }
        if wire.receipt.status.inference != "interval_withheld_calibration_unmeasured" {
            return Err(unsupported("this pilot publishes no interval").into());
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &NestedMarkovConsumeLimits,
    ) -> Result<(), NestedMarkovArtifactError> {
        let exceeded = NestedMarkovArtifactError::LimitsExceeded;
        if self.options.max_iterations > limits.max_iterations
            || self.options.max_iterations > MAX_ITERATIONS_CAP
        {
            return Err(exceeded("sweep bound"));
        }
        if self.regimes.len() > limits.max_regimes {
            return Err(exceeded("regime count"));
        }
        if self.graph.variables.len() > limits.max_variables {
            return Err(exceeded("variable count"));
        }
        if self
            .regimes
            .iter()
            .any(|r| r.cells.len() > limits.max_cells || r.levels.len() > limits.max_variables)
        {
            return Err(exceeded("regime table size"));
        }
        if self.graph.directed.len() > 64 || self.graph.bidirected.len() > 64 {
            return Err(exceeded("edge count"));
        }
        Ok(())
    }

    /// Decode and recheck everything under the consumer's limits, then re-run the pilot
    /// from the stored graph and counts and accept only an identical receipt.
    ///
    /// # Errors
    /// A limit, digest, id or receipt mismatch, or the engine's own refusal with its
    /// reason code and detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: NestedMarkovConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, PilotReport), IoError> {
        Self::consume_expecting(bytes, &NestedMarkovExpectation::default(), limits, ctx)
    }

    /// [`Self::consume_with_limits`] that additionally requires the stored premises and
    /// data digests to equal the consumer's expected identity.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`], and an expectation mismatch.
    pub fn consume_expecting(
        bytes: &[u8],
        expected: &NestedMarkovExpectation,
        limits: NestedMarkovConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, PilotReport), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if wire.expected_premises_digest()? != wire.premises_digest {
            return Err(NestedMarkovArtifactError::PremisesMismatch.into());
        }
        if wire.expected_data_digest()? != wire.data_digest {
            return Err(NestedMarkovArtifactError::DataIdentityMismatch.into());
        }
        if expected.premises_digest.as_ref().is_some_and(|d| *d != wire.premises_digest) {
            return Err(NestedMarkovArtifactError::ExpectationMismatch("premises").into());
        }
        if expected.data_digest.as_ref().is_some_and(|d| *d != wire.data_digest) {
            return Err(NestedMarkovArtifactError::ExpectationMismatch("data identity").into());
        }
        if wire.graph_id != NESTED_MARKOV_GRAPH_ID
            || wire.parameterization != NESTED_MARKOV_PARAMETERIZATION
        {
            return Err(NestedMarkovArtifactError::GraphIdMismatch.into());
        }
        let regimes =
            wire.regimes.iter().map(NestedRegimeWire::to_counts).collect::<Result<Vec<_>, _>>()?;
        let input = NestedMarkovInput { graph: wire.graph.to_graph(), regimes };
        let report = evaluate_nested_markov_pilot(&input, &wire.options.to_options(), ctx)
            .map_err(|refusal| pilot_refusal(&refusal))?;
        if crate::to_cbor(&NestedReceiptWire::from_report(&report))?
            != crate::to_cbor(&wire.receipt)?
        {
            return Err(NestedMarkovArtifactError::ReceiptMismatch.into());
        }
        Ok((wire, report))
    }
}
