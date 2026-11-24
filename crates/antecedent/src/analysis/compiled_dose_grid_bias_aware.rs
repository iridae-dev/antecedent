//! Compiled bias-aware dose-grid row.
//!
//! [`Study::compile_dose_grid_bias_aware`] binds a randomized-dose response-grid request, its
//! declared bound on the absolute third derivative of the dose response, the band's
//! multiplier replicate count and the declared row weights or dependence into a lowered
//! program: the premises are the program's question, the `(dose, outcome)` table (and its
//! weights) its source. [`CompiledDoseGridBiasAware::execute`] evaluates the bound request
//! through `antecedent_estimate::dose_grid_functional::estimate_dose_grid_bias_aware_declared`,
//! the kernel `estimate_dose_grid_bias_aware` runs with plain rows, under the caller's
//! execution context (whose seed draws the band's Gaussian multipliers). The result is the
//! kernel's [`BiasAwareDoseGrid`]: the unrestricted row, the worst-case smoothing bias of each
//! coordinate, the bias-aware (folded-normal) pointwise intervals and, for a grid of two or
//! more doses, the simultaneous band.
//!
//! [`Study::seal_dose_grid_bias_aware`] is the sealed row of a bias-aware request: it checks the
//! request as `DoseGridFunctional::evaluate_with_context` checks it, evaluates the compiled
//! program once under the band's declared seed and seals the program's fit, so the artifact is
//! the one that entry seals and the kernel runs once.
//!
//! The coverage of these intervals and of the band is measured for the estimator
//! `dose_grid_bias_aware` (the coverage records cited by the `estimate.dose_grid_bias_aware`
//! capability record); [`DoseGridBiasAwareRun::calibration_estimator`] names that key.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    ScoreFrequentist, ScoreModel, SemanticDigest,
};
use antecedent_estimate::EstimationError;
use antecedent_estimate::dose_grid_functional::{
    BiasAwareDoseGrid, ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, RowDeclaration,
    RowDependence, estimate_dose_grid_bias_aware_declared,
};
use antecedent_io::dose_grid_functional_artifact::{
    DoseGridArtifactError, DoseGridArtifactWire, DoseGridRequestWire,
};

use super::dose_grid_functional::DoseGridFunctional;
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "dose_grid_bias_aware.v1";
/// The estimator key the measured coverage of this kernel is recorded under.
const CALIBRATION_ESTIMATOR: &str = "dose_grid_bias_aware";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// How the rows of a bound request depend on one another.
#[derive(Clone, Debug, PartialEq)]
enum BoundDependence {
    Independent,
    Clustered(Vec<u32>),
    Serial(usize),
}

/// A randomized-dose response-grid request with its bias-aware premises, owned.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseGridBiasAwareRequest {
    dose: Vec<f64>,
    outcome: Vec<f64>,
    grid: Vec<f64>,
    bandwidth: f64,
    bandwidth_range: (f64, f64),
    minimum_local_ess: f64,
    functional: DoseFunctional,
    claims: ClaimSet,
    design: DoseDesign,
    curvature_bound: f64,
    band_replicates: usize,
    sampling_weights: Option<Vec<f64>>,
    dependence: BoundDependence,
}

impl DoseGridBiasAwareRequest {
    /// Copy a row request and add its declared bound on `|m'''|` and the band's multiplier
    /// replicate count (the arguments `estimate_dose_grid_bias_aware` takes). The rows are
    /// independent and unweighted until declared otherwise.
    #[must_use]
    pub fn new(
        request: &DoseGridRequest<'_>,
        curvature_bound: f64,
        band_replicates: usize,
    ) -> Self {
        Self {
            dose: request.dose.to_vec(),
            outcome: request.outcome.to_vec(),
            grid: request.grid.to_vec(),
            bandwidth: request.bandwidth,
            bandwidth_range: request.bandwidth_range,
            minimum_local_ess: request.minimum_local_ess,
            functional: request.functional,
            claims: request.claims,
            design: request.design,
            curvature_bound,
            band_replicates,
            sampling_weights: None,
            dependence: BoundDependence::Independent,
        }
    }

    /// Declare sampling weights, one per row.
    #[must_use]
    pub fn sampling_weights(mut self, weights: Vec<f64>) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    /// Declare one cluster id per row; replaces a declared serial lag.
    #[must_use]
    pub fn clusters(mut self, clusters: Vec<u32>) -> Self {
        self.dependence = BoundDependence::Clustered(clusters);
        self
    }

    /// Declare rows in time order with serial dependence up to `lag`; replaces declared
    /// clusters.
    #[must_use]
    pub fn serial_lag(mut self, lag: usize) -> Self {
        self.dependence = BoundDependence::Serial(lag);
        self
    }

    /// The bound row request, borrowed.
    #[must_use]
    pub fn row_request(&self) -> DoseGridRequest<'_> {
        DoseGridRequest {
            dose: &self.dose,
            outcome: &self.outcome,
            grid: &self.grid,
            bandwidth: self.bandwidth,
            bandwidth_range: self.bandwidth_range,
            minimum_local_ess: self.minimum_local_ess,
            functional: self.functional,
            claims: self.claims,
            design: self.design,
        }
    }

    /// The declared row weights and dependence, borrowed.
    #[must_use]
    pub fn row_declaration(&self) -> RowDeclaration<'_> {
        RowDeclaration {
            sampling_weights: self.sampling_weights.as_deref(),
            dependence: match &self.dependence {
                BoundDependence::Independent => RowDependence::Independent,
                BoundDependence::Clustered(groups) => RowDependence::Clustered(groups),
                BoundDependence::Serial(lag) => RowDependence::Serial { lag: *lag },
            },
        }
    }

    /// Declared bound on the absolute third derivative of the dose response.
    #[must_use]
    pub const fn curvature_bound(&self) -> f64 {
        self.curvature_bound
    }

    /// Gaussian multiplier replicates of the band.
    #[must_use]
    pub const fn band_replicates(&self) -> usize {
        self.band_replicates
    }
}

/// A bias-aware dose-grid request bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledDoseGridBiasAware {
    request: DoseGridBiasAwareRequest,
    lowered: Arc<LoweredProgram>,
}

/// The bias-aware row the bound request evaluates to, bound to the program that evaluated it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DoseGridBiasAwareRun {
    /// The row, worst-case biases, bias-aware intervals and band, as the kernel returned them.
    pub fit: BiasAwareDoseGrid,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// The estimator key the measured coverage of these intervals is recorded under.
    pub calibration_estimator: &'static str,
}

/// A bias-aware row sealed from the one evaluation of its compiled program.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SealedDoseGridBiasAware {
    /// The sealed row: the artifact `DoseGridFunctional::evaluate_with_context` seals for the
    /// same request and context.
    pub sealed: DoseGridFunctional,
    /// The program's run the row was sealed from.
    pub run: DoseGridBiasAwareRun,
    /// The program that evaluated the row.
    pub compiled: CompiledDoseGridBiasAware,
}

/// The compiled request of a checked bias-aware wire request: its premises and rows.
fn bound_request(
    wire: &DoseGridRequestWire,
    replicates: usize,
) -> Result<DoseGridBiasAwareRequest, DoseGridArtifactError> {
    let unsupported = DoseGridArtifactError::UnsupportedSemantics;
    let declared =
        wire.bias_aware.ok_or(unsupported("a bias-aware row declares a curvature bound"))?;
    let design = match wire.design.as_str() {
        "randomized_dose" => DoseDesign::RandomizedDose,
        "observational" => DoseDesign::Observational,
        _ => return Err(unsupported("design spelling")),
    };
    let functional = match (wire.functional.kind.as_str(), wire.functional.from, wire.functional.to)
    {
        ("level", None, None) => DoseFunctional::Level,
        ("derivative", None, None) => DoseFunctional::Derivative,
        ("contrast", Some(from), Some(to)) => DoseFunctional::Contrast { from, to },
        _ => return Err(unsupported("functional kind")),
    };
    let row = DoseGridRequest {
        dose: &wire.dose,
        outcome: &wire.outcome,
        grid: &wire.grid,
        bandwidth: wire.bandwidth,
        bandwidth_range: wire.bandwidth_range,
        minimum_local_ess: wire.minimum_local_ess,
        functional,
        claims: ClaimSet {
            pointwise_level: wire.claims.pointwise_level,
            derivative: wire.claims.derivative,
            simultaneous_band: wire.claims.simultaneous_band,
        },
        design,
    };
    let mut request = DoseGridBiasAwareRequest::new(&row, declared.curvature_bound, replicates);
    if let Some(weights) = &wire.sampling_weights {
        request = request.sampling_weights(weights.clone());
    }
    if let Some(clusters) = &wire.cluster {
        request = request.clusters(clusters.clone());
    }
    if let Some(lag) = wire.serial_lag {
        let lag = usize::try_from(lag)
            .map_err(|_| DoseGridArtifactError::LimitsExceeded("serial lag"))?;
        request = request.serial_lag(lag);
    }
    Ok(request)
}

fn functional_word(functional: DoseFunctional) -> (u64, f64, f64) {
    match functional {
        DoseFunctional::Level => (0, 0.0, 0.0),
        DoseFunctional::Derivative => (1, 0.0, 0.0),
        DoseFunctional::Contrast { from, to } => (2, from, to),
    }
}

fn question_digest(request: &DoseGridBiasAwareRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.dose_grid_bias_aware.question.v1");
    let (functional, from, to) = functional_word(request.functional);
    writer.word(functional).real(from).real(to);
    writer.word(match request.design {
        DoseDesign::RandomizedDose => 0,
        DoseDesign::Observational => 1,
    });
    writer
        .word(u64::from(request.claims.pointwise_level))
        .word(u64::from(request.claims.derivative))
        .word(u64::from(request.claims.simultaneous_band));
    writer.reals(&request.grid);
    writer.real(request.bandwidth).real(request.bandwidth_range.0).real(request.bandwidth_range.1);
    writer.real(request.minimum_local_ess);
    writer.real(request.curvature_bound).word(request.band_replicates as u64);
    match &request.dependence {
        BoundDependence::Independent => writer.word(0),
        BoundDependence::Clustered(groups) => writer.word(1).u32s(groups),
        BoundDependence::Serial(lag) => writer.word(2).word(*lag as u64),
    };
    writer.finish()
}

fn source_digest(request: &DoseGridBiasAwareRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.dose_grid_bias_aware.rows.v1");
    writer.reals(&request.dose).reals(&request.outcome);
    match &request.sampling_weights {
        Some(weights) => writer.word(1).reals(weights),
        None => writer.word(0),
    };
    writer.finish()
}

fn lower(request: &DoseGridBiasAwareRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let question = question_digest(request);
    let source = source_digest(request);
    let rows = request.dose.len() as u64;
    let columns = 2 + u64::from(request.sampling_weights.is_some());
    let design = DesignShape::quoted(rows, columns, true);
    let kind = match request.functional {
        DoseFunctional::Derivative => FunctionalKind::ResponseDerivative,
        DoseFunctional::Level | DoseFunctional::Contrast { .. } => {
            FunctionalKind::InterventionalDistribution
        }
    };
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional { kind, outcome: None, query: question },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::Tabular,
        graph_class: "caller_attested_randomized_dose",
        query_variables: Arc::from([]),
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let score = ScoreModel::new("gaussian_local_quadratic_bias_aware").map_err(|_| {
        CausalError::Compile { message: format!("{PLAN_ID}: the score cannot be declared") }
    })?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef { role: EvidenceRole::Trial, digest: Some(source) }]),
        composition: Composition::Chained { stages: 2 },
        decision: None,
        planning: None,
        chain: None,
    };
    let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
        let declared = PhysicalOperator::declare(effect);
        let declared = match bytes {
            Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
            None => declared,
        };
        declared.build().map_err(|_| CausalError::Compile {
            message: format!("{PLAN_ID}: a physical operator cannot be declared"),
        })
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read, Some(rows.saturating_mul(columns).saturating_mul(8)))?,
            operator(PhysicalEffect::Fit, None)?,
        ]),
        design,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::DoseGridBiasAware,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Sampling,
                kind: Some(ClaimKind::Response),
            },
        }
        .finish(),
    ))
}

impl Study {
    /// Bind a bias-aware dose-grid request into a compiled program.
    ///
    /// # Errors
    ///
    /// A score or physical operator that cannot be declared.
    pub fn compile_dose_grid_bias_aware(
        request: DoseGridBiasAwareRequest,
    ) -> Result<CompiledDoseGridBiasAware, CausalError> {
        let lowered = lower(&request)?;
        Ok(CompiledDoseGridBiasAware { request, lowered })
    }

    /// Check, evaluate and seal a bias-aware dose-grid request through its compiled program.
    ///
    /// The request is checked exactly as `DoseGridFunctional::evaluate_with_context` checks it
    /// before fitting, bound into the `DoseGridBiasAware` program, evaluated once under the
    /// band's declared seed (as the sealed artifact draws its band, whatever the caller's seed)
    /// and sealed from that fit. The sealed artifact equals the one
    /// `DoseGridFunctional::evaluate_with_context` seals for the same request and context.
    ///
    /// # Errors
    ///
    /// The refusals `DoseGridFunctional::evaluate_with_context` returns for the same request
    /// (spellings, limits, declared rows, the separate exact-quadratic premise, every refusal of
    /// the row and the bound, a non-finite result, cancellation), and the support gate once it
    /// is on.
    pub fn seal_dose_grid_bias_aware(
        request: DoseGridRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<SealedDoseGridBiasAware, DoseGridArtifactError> {
        let replicates = request.check_bias_aware()?;
        let band_seed = request.bias_aware.map_or(0, |declared| declared.band_seed);
        let compiled = Self::compile_dose_grid_bias_aware(bound_request(&request, replicates)?)
            .map_err(|error| DoseGridArtifactError::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: antecedent_core::reason_details::dose_grid::INVALID_REQUEST.to_owned(),
                message: error.to_string(),
            })?;
        // The band is drawn from its declared seed, as the sealed artifact draws it.
        let mut band_ctx = ctx.clone();
        band_ctx.rng = antecedent_core::RngFactory::from_seed(band_seed);
        let run = compiled.evaluate(&band_ctx)?;
        let artifact = DoseGridArtifactWire::seal_bias_aware_fit(request, &run.fit)?;
        Ok(SealedDoseGridBiasAware {
            sealed: DoseGridFunctional::from_artifact(artifact),
            run,
            compiled,
        })
    }
}

impl CompiledProgramHandle for CompiledDoseGridBiasAware {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledDoseGridBiasAware {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &DoseGridBiasAwareRequest {
        &self.request
    }

    /// Evaluate the bound request: the row, the bias-aware intervals and, for a grid, the band.
    ///
    /// # Errors
    ///
    /// Every refusal of the row (design, claims, bandwidth, support, local weight, a negative or
    /// non-finite curvature bound, too few band replicates, declared rows), admission,
    /// cancellation, or the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<DoseGridBiasAwareRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let fit = estimate_dose_grid_bias_aware_declared(
            &self.request.row_request(),
            &self.request.row_declaration(),
            self.request.curvature_bound,
            self.request.band_replicates,
            &scoped,
        )?;
        Ok(DoseGridBiasAwareRun {
            fit,
            lowered_program_identity: self.lowered.identity(),
            calibration_estimator: CALIBRATION_ESTIMATOR,
        })
    }

    /// [`Self::execute`] with the kernel's own refusal, for the sealed row.
    fn evaluate(&self, ctx: &ExecutionContext) -> Result<DoseGridBiasAwareRun, EstimationError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let fit = estimate_dose_grid_bias_aware_declared(
            &self.request.row_request(),
            &self.request.row_declaration(),
            self.request.curvature_bound,
            self.request.band_replicates,
            &scoped,
        )?;
        Ok(DoseGridBiasAwareRun {
            fit,
            lowered_program_identity: self.lowered.identity(),
            calibration_estimator: CALIBRATION_ESTIMATOR,
        })
    }
}
