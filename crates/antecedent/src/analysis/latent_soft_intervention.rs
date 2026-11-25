//! Compiled discovery of latent additive soft interventions on a declared DAG.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_graph::Dag;
use serde::{Deserialize, Serialize};

use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::Study;
use crate::error::CausalError;
use crate::planner::{PlanStage, PlanView, lowered::LoweredProgram};

mod artifact;
mod fit;

pub use artifact::{LatentSoftInterventionExport, ReplayedLatentSoftIntervention, replay_forecast};

pub(crate) const METHOD: &str = "estimate.latent_soft_intervention";

const MAX_NODES: usize = 16;
const MAX_ROWS: usize = 100_000;
const MAX_CANDIDATES: usize = 6;
const MAX_CONTEXT: usize = 6;
const MAX_SEARCH_OPERATIONS: u32 = 41_000;

const fn default_search_operations() -> u32 {
    MAX_SEARCH_OPERATIONS
}

/// Observations and the declared scope of the latent intervention search.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LatentSoftInterventionRequest {
    /// Unique names of the observed DAG variables, in data-column order.
    pub variables: Vec<String>,
    /// Directed edges as variable indices, parent first.
    pub edges: Vec<(usize, usize)>,
    /// Pre-intervention DAG variables available to the membership model and forecast.
    pub context: Vec<usize>,
    /// Equations that may receive additive shifts.
    pub candidates: Vec<usize>,
    /// Independent control-arm rows.
    pub baseline: Vec<Vec<f64>>,
    /// Independent recorded-action rows. Execution-version labels are latent.
    pub action: Vec<Vec<f64>>,
    /// Stable identity of the sampled population.
    pub population: String,
    /// The recorded action was randomized against the baseline.
    pub randomized: bool,
    /// Given the declared context, latent version assignment is independent of structural errors.
    pub version_exogenous_given_context: bool,
    /// Stable seed of the outcome-independent split and numerical restarts.
    pub seed: u64,
    /// Optional split-preserving whole-procedure bootstrap refits, at most 32.
    #[serde(default)]
    pub refits: u8,
    /// Maximum EM iterations across every mask and restart of one fit.
    #[serde(default = "default_search_operations")]
    pub search_operations: u32,
}

pub(super) fn invalid(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!(
        "invalid_argument",
        "{}: {message}",
        antecedent_core::reason_details::latent_soft_intervention::INVALID_REQUEST
    )
}

pub(super) fn cancelled(ctx: &ExecutionContext, stage: &'static str) -> Result<(), CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage });
    }
    Ok(())
}

impl LatentSoftInterventionRequest {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the variable count is validated to at most MAX_NODES, far below u32::MAX"
    )]
    fn validate(&mut self, ctx: &ExecutionContext) -> Result<(), CausalError> {
        fn indices(values: &[usize], n: usize) -> bool {
            values.iter().all(|&value| value < n)
                && values.iter().collect::<BTreeSet<_>>().len() == values.len()
        }
        let n = self.variables.len();
        if !(3..=MAX_NODES).contains(&n)
            || self.population.trim().is_empty()
            || self.population.len() > 256
            || self.variables.iter().any(|name| name.trim().is_empty() || name.len() > 256)
            || self.variables.iter().collect::<BTreeSet<_>>().len() != n
        {
            return Err(invalid("invalid variable or population declaration"));
        }
        if !self.randomized || !self.version_exogenous_given_context {
            return Err(invalid(
                "recorded-action randomization and conditional version exogeneity must be declared",
            ));
        }
        if self.refits > 32
            || !(1..=MAX_SEARCH_OPERATIONS).contains(&self.search_operations)
            || self.context.len() > MAX_CONTEXT
            || !(2..=MAX_CANDIDATES).contains(&self.candidates.len())
            || self.edges.len() > n * (n - 1) / 2
            || self.baseline.len().saturating_add(self.action.len()) > MAX_ROWS
            || self.baseline.len() < 6 * n + 16
            || self.action.len() < 6 * n + 16
        {
            return Err(invalid("declaration exceeds the supported shape or has too few rows"));
        }
        if !indices(&self.context, n)
            || !indices(&self.candidates, n)
            || self.context.iter().any(|index| self.candidates.contains(index))
        {
            return Err(invalid(
                "context and candidate targets must be disjoint unique variable indices",
            ));
        }
        self.edges.sort_unstable();
        let mut dag = Dag::with_variables(n as u32);
        for &(parent, child) in &self.edges {
            if parent >= n || child >= n {
                return Err(invalid("edge references an unknown variable"));
            }
            dag.insert_directed(
                antecedent_graph::DenseNodeId::from_raw(parent as u32),
                antecedent_graph::DenseNodeId::from_raw(child as u32),
            )
            .map_err(invalid)?;
        }
        if self.edges.iter().any(|&(_, child)| self.context.contains(&child)) {
            return Err(invalid("forecast context must consist of pre-intervention DAG roots"));
        }
        self.candidates.sort_unstable();
        for row in self.baseline.iter().chain(&self.action) {
            cancelled(ctx, "latent soft intervention source validation")?;
            if row.len() != n || row.iter().any(|value| !value.is_finite()) {
                return Err(invalid("observations must be finite and match the declared schema"));
            }
        }
        Ok(())
    }

    pub(crate) fn source_digest(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SemanticDigest, CausalError> {
        let mut writer = CanonicalWriter::new("antecedent.latent_soft_intervention.source.v1");
        writer.texts(&self.variables).text(&self.population);
        writer
            .word(u64::from(self.randomized))
            .word(u64::from(self.version_exogenous_given_context))
            .word(self.seed)
            .word(u64::from(self.refits))
            .word(u64::from(self.search_operations));
        writer.word(self.edges.len() as u64);
        for &(parent, child) in &self.edges {
            writer.word(parent as u64).word(child as u64);
        }
        for indices in [&self.context, &self.candidates] {
            writer.word(indices.len() as u64);
            for &index in indices {
                writer.word(index as u64);
            }
        }
        for arm in [&self.baseline, &self.action] {
            writer.word(arm.len() as u64);
            for row in arm {
                cancelled(ctx, "latent soft intervention source identity")?;
                writer.reals(row);
            }
        }
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }
}

/// The immutable source and native lowered discovery program.
#[derive(Clone)]
pub struct PreparedLatentSoftIntervention {
    request: Arc<LatentSoftInterventionRequest>,
    program: Arc<LoweredProgram>,
    source: SemanticDigest,
    _source_lease: Arc<antecedent_core::execution::AdmissionLease>,
}

impl Study {
    /// Compile a bounded, model-conditional latent soft-intervention study without fitting.
    ///
    /// # Errors
    /// Invalid declarations, unsupported inputs, resource refusal or cancellation.
    pub fn prepare_latent_soft_intervention(
        mut request: LatentSoftInterventionRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedLatentSoftIntervention, CausalError> {
        super::lowering::refuse_reentry(ctx)?;
        request.validate(ctx)?;
        let bytes = (request.baseline.len() + request.action.len()) as u64
            * (24 + 8 * request.variables.len() as u64);
        let source_lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "latent soft intervention preparation" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let source = request.source_digest(ctx)?;
        let program = super::lowering::latent_soft_intervention::lower(&request, source, bytes)?;
        Ok(PreparedLatentSoftIntervention {
            request: Arc::new(request),
            program: Arc::new(program),
            source,
            _source_lease: Arc::new(source_lease),
        })
    }
}

impl CompiledProgramHandle for PreparedLatentSoftIntervention {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.program
    }
}

impl PreparedLatentSoftIntervention {
    /// Inspect the native compiled plan without executing it.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.program), PlanStage::Prepared)
    }

    /// Identity of the native program that owns the fit.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
    }
}

/// Scientific status of the latent-version comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionEvidence {
    /// The observed data did not resolve a split from the declared comparator.
    SplitUnresolved,
    /// The split passed, while competing target hypotheses remain plausible.
    TargetsUnresolved,
    /// The split passed and the target search had a separated preferred fit.
    TargetsPreferred,
}

/// A fitted pair of target masks, treated as an unordered hypothesis.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentTargetHypothesis {
    /// The two candidate equation sets, as indices into the discovery variable order.
    pub target_sets: [Vec<usize>; 2],
    /// Penalized training log-likelihood gap from the best retained fit, without a confidence claim.
    pub training_gap: f64,
}

/// Model-conditional result of the retained search.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentSoftInterventionDiscovery {
    /// Hex identity of the compiler program that executed.
    pub program: String,
    /// Variable names in the order used by shifts and target indices.
    pub variables: Vec<String>,
    /// Scope of the evidence for the reported versions.
    pub evidence: VersionEvidence,
    /// Whether the release support registry attests the route.
    pub release_supported: bool,
    /// This route's evidence refusal message when previewed in a development build.
    pub would_refuse: Option<String>,
    /// Held-out log likelihood ratio against the fitted-on-evaluation Gaussian null.
    /// Absent when the bounded two-version search returned no eligible fit.
    pub log_evidence: Option<f64>,
    /// Whether the recorded-action forecast passed the fit-admissibility guard.
    pub forecast_admissible: bool,
    /// Preferred and runner-up target hypotheses when the split passed evaluation.
    pub hypotheses: Vec<LatentTargetHypothesis>,
    /// Additive equation shifts in declared variable order for each fitted component.
    pub shifts: Vec<Vec<f64>>,
    /// Action-arm observations used to fit candidate models.
    pub training_rows: usize,
    /// Action-arm observations held out for the split comparison.
    pub evaluation_rows: usize,
}

/// Conditional Gaussian component in a joint predictive law.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentSoftInterventionComponent {
    /// Model-conditional version probability at the supplied context.
    pub weight: f64,
    /// Mean vector in the forecast's variable order.
    pub mean: Vec<f64>,
    /// Row-major covariance in the forecast's variable order.
    pub covariance: Vec<f64>,
}

/// The model-conditional outcome law and baseline at one pre-intervention context.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentSoftInterventionForecast {
    /// Hex identity binding the program, fitted result and exact context.
    pub query: String,
    /// Non-context DAG variables shared by all reported laws.
    pub variables: Vec<String>,
    /// Joint Gaussian baseline.
    pub baseline: LatentSoftInterventionComponent,
    /// Model-conditional joint outcome mixture under the recorded action.
    pub action: Vec<LatentSoftInterventionComponent>,
    /// Action mean minus baseline mean at the same context.
    pub mean_difference: Vec<f64>,
    /// Whole-fit resampling sensitivity, absent when refits were not requested.
    pub refits: Option<LatentSoftInterventionRefits>,
}

impl LatentSoftInterventionForecast {
    /// Pass one action marginal to the existing known-weight propagation route.
    ///
    /// This reading conditions on the fitted version weights and law at this query.
    /// It carries the forecast identity as shared evidence for every mixture component.
    ///
    /// # Errors
    /// Unknown coordinate, malformed components or a propagation refusal.
    pub fn action_marginal(
        &self,
        variable: &str,
        ctx: &ExecutionContext,
    ) -> Result<super::joint_law_propagation::PropagationClaim, CausalError> {
        use super::joint_law_propagation::{MixtureComponent, propagate_mixture};

        let width = self.variables.len();
        let index = self
            .variables
            .iter()
            .position(|name| name == variable)
            .ok_or_else(|| invalid("unknown forecast coordinate"))?;
        let diagonal = index
            .checked_mul(width)
            .and_then(|offset| offset.checked_add(index))
            .ok_or_else(|| invalid("forecast covariance size overflow"))?;
        let components = self
            .action
            .iter()
            .enumerate()
            .map(|(version, law)| {
                let mean =
                    *law.mean.get(index).ok_or_else(|| invalid("forecast mean is missing"))?;
                let variance = *law
                    .covariance
                    .get(diagonal)
                    .ok_or_else(|| invalid("forecast covariance is missing"))?;
                Ok(MixtureComponent {
                    id: format!("{}:{index}:{version}", self.query),
                    evidence_id: self.query.clone(),
                    weight: law.weight,
                    mean,
                    variance,
                })
            })
            .collect::<Result<Vec<_>, CausalError>>()?;
        propagate_mixture(&components, ctx)
            .map_err(|error| CausalError::Compile { message: error.to_string() })
    }
}

/// Joint predictive laws from one complete resampling refit.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentSoftInterventionRefit {
    /// Baseline and action components share a refit, context, and coordinate order.
    pub baseline: LatentSoftInterventionComponent,
    /// Action components, with their own membership weights from this refit.
    pub action: Vec<LatentSoftInterventionComponent>,
    /// Aggregate action mean minus baseline mean for this refit.
    pub mean_difference: Vec<f64>,
}

/// Diagnostic sampling spread across aligned baseline, shifts, and gating refits.
/// The draws do not supply confidence or credible intervals.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct LatentSoftInterventionRefits {
    /// Refit count requested at compilation.
    pub attempted: u8,
    /// Replicates without a valid model or with insufficient support at this context.
    pub unavailable: u8,
    /// Successful replicates in deterministic attempt order; version labels are local to a draw.
    pub draws: Vec<LatentSoftInterventionRefit>,
}

struct RetainedLaw {
    fit: fit::Fit,
    baseline: antecedent_model::compose::JointGaussian,
    versions: Vec<antecedent_model::compose::JointGaussian>,
}

/// Retained fit produced by a native lowered program.
pub struct FittedLatentSoftIntervention {
    prepared: PreparedLatentSoftIntervention,
    law: RetainedLaw,
    refits: Vec<RetainedLaw>,
    _refit_lease: antecedent_core::execution::AdmissionLease,
    discovery: LatentSoftInterventionDiscovery,
    fit_identity: SemanticDigest,
}

impl PreparedLatentSoftIntervention {
    /// Execute the retained program once and keep its fit for repeated forecasts.
    ///
    /// # Errors
    /// Fitting or composition refusal, cancellation or resource limits.
    #[allow(clippy::too_many_lines, reason = "one pass from the base fit through the refit loop")]
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<FittedLatentSoftIntervention, CausalError> {
        super::lowering::refuse_reentry(ctx)?;
        let scoped = ctx.enter_lowered_dispatch(self.program.identity());
        cancelled(&scoped, "latent soft intervention execution")?;
        let release_supported = super::lowering::latent_soft_intervention::release_supported();
        gate_program(&scoped, self)?;
        let rows = (self.request.baseline.len() + self.request.action.len()) as u64;
        let width = self.request.variables.len() as u64;
        let _workspace = scoped
            .acquire_serial_numerical_admission(rows * (512 + 64 * width))
            .map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "latent soft intervention execution" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
        let law = self.fit_law(None, &scoped)?;
        let fit = &law.fit;
        let refit_lease = scoped
            .acquire_admission(
                0,
                (1 + u64::from(self.request.refits)) * (4096 + width * width * 64),
            )
            .map_err(|error| {
                crate::error::resource_admission_error(error, "latent soft intervention refits")
            })?;
        let mut refits = Vec::with_capacity(usize::from(self.request.refits));
        if fit.forecast_admissible {
            for replicate in 0..self.request.refits {
                cancelled(&scoped, "latent soft intervention refits")?;
                let seed = antecedent_core::mix64(
                    self.request.seed ^ 0x5245_4649_545f_4c53 ^ u64::from(replicate),
                );
                match self.fit_law(Some(seed), &scoped) {
                    Ok(refit) if refit.fit.forecast_admissible => refits.push(refit),
                    Ok(_) | Err(CausalError::Compile { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        let evidence = if fit.mixture.shifts.len() == 1 {
            VersionEvidence::SplitUnresolved
        } else if fit.targets_resolved {
            VersionEvidence::TargetsPreferred
        } else {
            VersionEvidence::TargetsUnresolved
        };
        let discovery = LatentSoftInterventionDiscovery {
            program: self.program_identity().to_hex(),
            variables: self.request.variables.clone(),
            evidence,
            release_supported,
            would_refuse: scoped
                .would_refuse()
                .into_iter()
                .find(|entry| entry.detail == self.program.route())
                .map(|entry| entry.to_string()),
            log_evidence: fit.log_evidence.is_finite().then_some(fit.log_evidence),
            forecast_admissible: fit.forecast_admissible,
            hypotheses: fit.hypotheses.clone(),
            shifts: fit.mixture.shifts.clone(),
            training_rows: fit.training_rows,
            evaluation_rows: fit.evaluation_rows,
        };
        let mut identity = CanonicalWriter::new("antecedent.latent_soft_intervention.fit.v1");
        identity.bytes(self.program_identity().as_bytes());
        // The standing of release evidence is an annotation and never changes this fit identity.
        identity.bytes(
            &serde_json::to_vec(&(
                discovery.evidence,
                discovery.log_evidence,
                discovery.forecast_admissible,
                &discovery.hypotheses,
                &discovery.shifts,
                discovery.training_rows,
                discovery.evaluation_rows,
            ))
            .map_err(invalid)?,
        );
        for equation in &fit.equations {
            identity.reals(&equation.beta).real(equation.sigma);
        }
        fit.write_forecast_identity(&mut identity);
        identity.word(refits.len() as u64);
        for refit in &refits {
            refit.fit.write_forecast_identity(&mut identity);
            for equation in &refit.fit.equations {
                identity.reals(&equation.beta).real(equation.sigma);
            }
            identity.word(refit.fit.mixture.shifts.len() as u64);
            for shift in &refit.fit.mixture.shifts {
                identity.reals(shift);
            }
        }
        Ok(FittedLatentSoftIntervention {
            prepared: self.clone(),
            law,
            refits,
            _refit_lease: refit_lease,
            discovery,
            fit_identity: SemanticDigest::from_bytes(identity.finish()),
        })
    }

    fn fit_law(
        &self,
        refit_seed: Option<u64>,
        ctx: &ExecutionContext,
    ) -> Result<RetainedLaw, CausalError> {
        let fit = fit::run(&self.request, self.program_identity(), refit_seed, ctx)?;
        let baseline = self.gaussian_law(&fit.equations, &vec![0.0; fit.equations.len()])?;
        let mut versions = Vec::with_capacity(fit.mixture.shifts.len());
        for shift in &fit.mixture.shifts {
            cancelled(ctx, "latent soft intervention composition")?;
            versions.push(self.gaussian_law(&fit.equations, shift)?);
        }
        Ok(RetainedLaw { fit, baseline, versions })
    }

    fn gaussian_law(
        &self,
        equations: &[fit::Equation],
        shift: &[f64],
    ) -> Result<antecedent_model::compose::JointGaussian, CausalError> {
        use antecedent_model::compose::{
            ComposedLinearGaussian, ComposedModelSpec, Domain, MechanismContext,
            MechanismDescriptor, MechanismForm, MechanismOrigin, MechanismProvenance,
            ParentBinding, PosteriorDependence,
        };

        let names = &self.request.variables;
        let context = MechanismContext {
            population: self.request.population.clone(),
            regime: "latent_soft_intervention_model".into(),
        };
        let evidence = self.source.to_hex();
        let artifact = self.program_identity().to_hex();
        let mechanisms = equations
            .iter()
            .enumerate()
            .map(|(i, equation)| MechanismDescriptor {
                variable: names[i].clone(),
                form: MechanismForm::LinearGaussian {
                    intercept: equation.beta[0] + shift[i],
                    sigma: equation.sigma,
                    coefficients: equation
                        .parents
                        .iter()
                        .zip(&equation.beta[1..])
                        .map(|(&p, &beta)| (names[p].clone(), beta))
                        .collect(),
                },
                parents: equation
                    .parents
                    .iter()
                    .map(|&p| ParentBinding {
                        variable: names[p].clone(),
                        units: "native".into(),
                        transform: "identity".into(),
                        domain: Domain::Real,
                    })
                    .collect(),
                conditioning_set: equation.parents.iter().map(|&p| names[p].clone()).collect(),
                output_units: "native".into(),
                output_transform: "identity".into(),
                output_support: Domain::Real,
                context: context.clone(),
                provenance: MechanismProvenance {
                    origin: MechanismOrigin::Fitted,
                    artifact_id: artifact.clone(),
                    evidence_id: evidence.clone(),
                },
                latents: Vec::new(),
                uncertainty: None,
            })
            .collect();
        let model = ComposedModelSpec {
            target: context,
            mechanisms,
            edges: self
                .request
                .edges
                .iter()
                .map(|&(p, c)| (names[p].clone(), names[c].clone()))
                .collect(),
            invariances: Vec::new(),
            selection_diagrams: Vec::new(),
            dependence: PosteriorDependence::Unknown,
            admit_declared_mechanisms: false,
        };
        ComposedLinearGaussian::compile(&model).and_then(|model| model.law(&[])).map_err(invalid)
    }
}

impl FittedLatentSoftIntervention {
    /// Read the discovery result from the retained fit.
    #[must_use]
    pub const fn discovery(&self) -> &LatentSoftInterventionDiscovery {
        &self.discovery
    }

    /// Bind and evaluate a pre-intervention context against the retained model.
    ///
    /// # Errors
    /// Malformed or unsupported context, nonfinite law, or cancellation.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the draw count never exceeds the refit count, which validation bounds by 32"
    )]
    pub fn forecast(
        &self,
        values: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<LatentSoftInterventionForecast, CausalError> {
        super::lowering::refuse_reentry(ctx)?;
        let scoped = ctx.enter_lowered_dispatch(self.prepared.program_identity());
        cancelled(&scoped, "latent soft intervention forecast")?;
        if !self.law.fit.forecast_admissible {
            return Err(crate::compile_reason!(
                "not_supported",
                "{}: the recorded-action law failed its evaluation guard",
                antecedent_core::reason_details::latent_soft_intervention::FORECAST_INADEQUATE
            ));
        }
        self.law.fit.context.forecast_design(values)?;
        let observed: Vec<_> = self
            .prepared
            .request
            .context
            .iter()
            .zip(values)
            .map(|(&node, &value)| (self.prepared.request.variables[node].as_str(), value))
            .collect();
        let (variables, point) = self.law.condition(values, &observed, &scoped)?;
        let refits = if self.prepared.request.refits > 0 {
            let mut draws = Vec::new();
            for refit in &self.refits {
                cancelled(&scoped, "latent soft intervention refit forecast")?;
                if refit.fit.context.forecast_design(values).is_ok() {
                    let (coordinates, draw) = refit.condition(values, &observed, &scoped)?;
                    if coordinates != variables {
                        return Err(invalid("refit response coordinates disagree"));
                    }
                    draws.push(draw);
                }
            }
            let attempted = self.prepared.request.refits;
            Some(LatentSoftInterventionRefits {
                attempted,
                unavailable: attempted - draws.len() as u8,
                draws,
            })
        } else {
            None
        };
        let mut query = CanonicalWriter::new("antecedent.latent_soft_intervention.forecast.v1");
        query.bytes(self.fit_identity.as_bytes()).reals(values);
        Ok(LatentSoftInterventionForecast {
            query: query.finish_hex(),
            variables,
            baseline: point.baseline,
            action: point.action,
            mean_difference: point.mean_difference,
            refits,
        })
    }
}

impl RetainedLaw {
    fn condition(
        &self,
        values: &[f64],
        observed: &[(&str, f64)],
        ctx: &ExecutionContext,
    ) -> Result<(Vec<String>, LatentSoftInterventionRefit), CausalError> {
        let design = self.fit.context.forecast_design(values)?;
        let base = self.baseline.condition(observed).map_err(invalid)?;
        let variables = base.variables().to_vec();
        let baseline = LatentSoftInterventionComponent {
            weight: 1.0,
            mean: base.mean().to_vec(),
            covariance: base.covariance_matrix().to_vec(),
        };
        let mut action = Vec::with_capacity(self.versions.len());
        let mut mean_difference: Vec<_> = baseline.mean.iter().map(|mean| -mean).collect();
        for (law, weight) in self.versions.iter().zip(self.fit.mixture.weights(&design)) {
            cancelled(ctx, "latent soft intervention conditional law")?;
            let conditional = law.condition(observed).map_err(invalid)?;
            if conditional.variables() != variables.as_slice() {
                return Err(invalid("conditioned response coordinates disagree"));
            }
            for (difference, mean) in mean_difference.iter_mut().zip(conditional.mean()) {
                *difference += weight * mean;
            }
            action.push(LatentSoftInterventionComponent {
                weight,
                mean: conditional.mean().to_vec(),
                covariance: conditional.covariance_matrix().to_vec(),
            });
        }
        if mean_difference.iter().any(|value| !value.is_finite()) {
            return Err(invalid("response comparison is not finite"));
        }
        Ok((variables, LatentSoftInterventionRefit { baseline, action, mean_difference }))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::ExecutionContext;

    use super::LatentSoftInterventionRequest;
    use crate::identity_laws::{bit_exact, ordered_laws, set_laws};

    fn rows(shift: f64) -> Vec<Vec<f64>> {
        (0..48_u32)
            .map(|row| {
                let x = f64::from(row);
                vec![x, 0.5 * x, x + shift, 0.25 * x, shift]
            })
            .collect()
    }

    /// Five variables whose roots are `a` and `b`.
    fn request() -> LatentSoftInterventionRequest {
        LatentSoftInterventionRequest {
            variables: ["a", "b", "c", "d", "e"].iter().map(|name| (*name).to_owned()).collect(),
            edges: vec![(0, 2), (1, 2), (2, 3), (3, 4)],
            context: vec![0, 1],
            candidates: vec![2, 3, 4],
            baseline: rows(0.0),
            action: rows(1.0),
            population: "units".to_owned(),
            randomized: true,
            version_exogenous_given_context: true,
            seed: 7,
            refits: 0,
            search_operations: 100,
        }
    }

    /// The source digest of a request preparation accepts, `None` when validation refuses it.
    fn source(mut request: LatentSoftInterventionRequest) -> Option<[u8; 32]> {
        let ctx = ExecutionContext::for_tests(7);
        request.validate(&ctx).ok()?;
        request.source_digest(&ctx).ok().map(|digest| *digest.as_bytes())
    }

    #[test]
    fn site_latent_soft_intervention_source_sets_and_orders() {
        // The DAG's edges and the shiftable equations are sets: validation sorts them and
        // refuses a repeat.
        set_laws(&[(0, 2), (1, 2), (2, 3), (3, 4)], |edges| {
            source(LatentSoftInterventionRequest { edges: edges.to_vec(), ..request() })
        });
        set_laws(&[2, 3, 4], |candidates| {
            source(LatentSoftInterventionRequest { candidates: candidates.to_vec(), ..request() })
        });
        // The context is the forecast's argument order; the variables are the column order.
        ordered_laws(&[0, 1], |context| {
            source(LatentSoftInterventionRequest { context: context.to_vec(), ..request() })
        });
        ordered_laws(&["a", "b", "c", "d", "e"], |names| {
            let variables = names.iter().map(|name| (*name).to_owned()).collect();
            source(LatentSoftInterventionRequest { variables, ..request() })
        });
        bit_exact(|value| {
            let mut declared = request();
            declared.baseline[0][4] = value;
            source(declared)
        });
    }
}
