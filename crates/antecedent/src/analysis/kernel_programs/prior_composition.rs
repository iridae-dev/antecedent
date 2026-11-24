//! External prior composition and the transport of external priors, as compiled program
//! families.
//!
//! A declared set of external prior sources, each with its power-prior discount (and optional
//! mixture weight), composes with a baseline prior into one prior: Gaussian coefficient sources
//! by the precision sum (or a moment-matched mixture), Beta-Binomial count sources by adding the
//! discounted successes and failures, Gamma-Poisson count sources by adding the discounted
//! counts and exposure. These compile to the `ExternalPriorComposition` route. Moving the
//! sources toward a target population before composing — a tag-defined target under a declared
//! invariance policy and an optional unit reweighting, a target declared as a weighting over the
//! source rows, and the sensitivity of a kernel-weighted target to its bandwidth — compiles to
//! the `ExternalPriorTransport` route.
//!
//! Compiling binds the digest of the declared input into a six-layer program. These inputs are
//! not wire types, so the identity bytes are their `Debug` rendering: every field in order,
//! with every real in Rust's shortest round-trip form, so equal declarations give equal bytes.
//! Every run passes the support gate of the route's unit and then calls the very kernel
//! function the direct path calls (`antecedent_prob::compose_external_priors`,
//! `compose_beta_power_priors`, `compose_gamma_power_priors`, `apply_transport`,
//! `compose_with_transport`, `apply_weighted_target_transport`,
//! `compose_with_weighted_target_transport`, `kernel_bandwidth_sensitivity`) with the same
//! arguments, so every composed mean, variance, discount, effective sample size, transported
//! source, overlap diagnostic and sensitivity row is the direct path's bit for bit. Nothing
//! here reads the context's seed.
//!
//! The routes' unit (`prior.external_composition`) has not yet earned its support, so the runs
//! pass its support gate only once `GATE_ON_UNIT_SUPPORT` is turned on; until then no valid use
//! refuses.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_prob::{
    BandwidthSensitivityRow, BetaHyperparameters, ComposedPrior, ExternalPriorSource,
    GammaHyperparameters, KernelFamily, PriorSet, ProbError, TargetOverlap, TransportAdjustment,
    TransportContext, TransportError, TransportOutcome, TransportPolicy, WeightedTarget,
    apply_transport, apply_weighted_target_transport, compose_beta_power_priors,
    compose_external_priors, compose_gamma_power_priors, compose_with_transport,
    compose_with_weighted_target_transport, kernel_bandwidth_sensitivity,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

/// Whether the runs pass the support gate of the unit the routes are bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the routes
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

const COMPOSITION_DESIGN: &str = "declared external prior sources composed with a baseline at \
                                  their declared power-prior discounts";

const TRANSPORT_DESIGN: &str = "declared external prior sources moved toward a declared target \
                                population before composition";

/// Gaussian coefficient sources: the power-prior precision sum or the moment-matched mixture.
const GAUSSIAN_COMPOSITION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorComposition,
    plan_id: "prior.external_composition.gaussian",
    functional: FunctionalKind::Association,
    evaluator: "gaussian_external_prior_power_or_mixture_composition",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::PriorArtifact,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: COMPOSITION_DESIGN,
};

/// Beta-Binomial count sources: `Beta(a + sum alpha k, b + sum alpha (n - k))`.
const BETA_COMPOSITION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorComposition,
    plan_id: "prior.external_composition.beta_counts",
    functional: FunctionalKind::Association,
    evaluator: "beta_binomial_count_power_prior_composition",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::PriorArtifact,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: COMPOSITION_DESIGN,
};

/// Gamma-Poisson count sources: `Gamma(s + sum alpha S, r + sum alpha E)`.
const GAMMA_COMPOSITION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorComposition,
    plan_id: "prior.external_composition.gamma_counts",
    functional: FunctionalKind::Association,
    evaluator: "gamma_poisson_count_power_prior_composition",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::PriorArtifact,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: COMPOSITION_DESIGN,
};

/// A tag-defined target under a declared invariance policy and an optional unit reweighting.
const TAG_TRANSPORT: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorTransport,
    plan_id: "prior.external_transport.population_tag",
    functional: FunctionalKind::Association,
    evaluator: "external_prior_population_tag_transport",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "none",
    data: DataClassification::MultiEnvironment,
    design: TRANSPORT_DESIGN,
};

/// A target declared as a weighting over the source rows, checked for overlap.
const WEIGHTED_TARGET_TRANSPORT: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorTransport,
    plan_id: "prior.external_transport.weighted_target",
    functional: FunctionalKind::Association,
    evaluator: "external_prior_weighted_target_transport",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "none",
    data: DataClassification::MultiEnvironment,
    design: TRANSPORT_DESIGN,
};

/// The kernel-weighted transported mean and its overlap across declared bandwidths.
const BANDWIDTH_SENSITIVITY: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ExternalPriorTransport,
    plan_id: "prior.external_transport.kernel_bandwidth_sensitivity",
    functional: FunctionalKind::Association,
    evaluator: "kernel_weighted_target_bandwidth_sensitivity",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "none",
    data: DataClassification::MultiEnvironment,
    design: TRANSPORT_DESIGN,
};

/// The declared Gaussian composition: sources and baseline.
#[derive(Clone, Debug)]
struct GaussianCompositionInput {
    sources: Vec<ExternalPriorSource>,
    baseline: PriorSet,
}

/// The declared Beta count composition: base and `(successes, trials, alpha)` sources.
#[derive(Clone, Debug)]
struct BetaCompositionInput {
    base: BetaHyperparameters,
    sources: Vec<(u64, u64, f64)>,
}

/// The declared Gamma count composition: base and `(total count, exposure, alpha)` sources.
#[derive(Clone, Debug)]
struct GammaCompositionInput {
    base: GammaHyperparameters,
    sources: Vec<(u64, f64, f64)>,
}

/// The owned parts of a [`TransportContext`].
#[derive(Clone, Debug)]
struct OwnedTransportContext {
    source_populations: Vec<Option<String>>,
    target_population: Option<String>,
    policy: Option<TransportPolicy>,
    adjustment: Option<TransportAdjustment>,
    coef_index: Option<usize>,
}

impl OwnedTransportContext {
    fn of(context: &TransportContext<'_>) -> Self {
        Self {
            source_populations: context
                .source_populations
                .iter()
                .copied()
                .map(|population| population.map(str::to_owned))
                .collect(),
            target_population: context.target_population.map(str::to_owned),
            policy: context.policy,
            adjustment: context.adjustment.cloned(),
            coef_index: context.coef_index,
        }
    }

    /// The source population tags as the borrowed slice a [`TransportContext`] reads.
    fn populations(&self) -> Vec<Option<&str>> {
        self.source_populations.iter().map(Option::as_deref).collect()
    }

    /// The context the kernel reads, over `populations` (from [`Self::populations`]).
    fn view<'a>(&'a self, populations: &'a [Option<&'a str>]) -> TransportContext<'a> {
        TransportContext {
            source_populations: populations,
            target_population: self.target_population.as_deref(),
            policy: self.policy,
            adjustment: self.adjustment.as_ref(),
            coef_index: self.coef_index,
        }
    }
}

/// The declared tag-defined transport: sources, baseline and transport context.
#[derive(Clone, Debug)]
struct TagTransportInput {
    sources: Vec<ExternalPriorSource>,
    baseline: PriorSet,
    context: OwnedTransportContext,
}

/// The declared weighted-target transport: sources, baseline, policy, target and coefficient.
#[derive(Clone, Debug)]
struct WeightedTargetTransportInput {
    sources: Vec<ExternalPriorSource>,
    baseline: PriorSet,
    policy: TransportPolicy,
    target: WeightedTarget,
    coef_index: Option<usize>,
}

/// The declared bandwidth sensitivity: kernel family, bandwidths, rows and the overlap floor.
#[derive(Clone, Debug)]
struct BandwidthSensitivityInput {
    family: KernelFamily,
    bandwidths: Vec<f64>,
    distances: Vec<f64>,
    unit_effects: Vec<f64>,
    min_ess_fraction: f64,
}

/// A compiled composition of external Gaussian coefficient priors with a baseline.
#[derive(Clone, Debug)]
pub struct CompiledExternalPriorComposition {
    program: KernelProgram<GaussianCompositionInput>,
}

/// A compiled composition of Beta-Binomial count sources into one Beta power prior.
#[derive(Clone, Debug)]
pub struct CompiledBetaPowerPriorComposition {
    program: KernelProgram<BetaCompositionInput>,
}

/// A compiled composition of Gamma-Poisson count sources into one Gamma power prior.
#[derive(Clone, Debug)]
pub struct CompiledGammaPowerPriorComposition {
    program: KernelProgram<GammaCompositionInput>,
}

/// A compiled transport of external priors toward a tag-defined target population.
#[derive(Clone, Debug)]
pub struct CompiledPriorTagTransport {
    program: KernelProgram<TagTransportInput>,
}

/// A compiled transport of external priors toward a target declared as a row weighting.
#[derive(Clone, Debug)]
pub struct CompiledPriorWeightedTargetTransport {
    program: KernelProgram<WeightedTargetTransportInput>,
}

/// A compiled sensitivity of a kernel-weighted target transport to its bandwidth.
#[derive(Clone, Debug)]
pub struct CompiledKernelBandwidthSensitivity {
    program: KernelProgram<BandwidthSensitivityInput>,
}

/// The identity digest of a declared input: its `Debug` rendering under the family's tag.
fn debug_canonical<T: std::fmt::Debug>(tag: &str, input: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(tag);
    writer.text(&format!("{input:?}"));
    writer.finish()
}

/// The support refusal as a probability refusal (`not_supported`, detail
/// `prior_composition.unit_not_supported`).
fn prob_refusal(refusal: &NotSupported) -> ProbError {
    ProbError::Refused {
        code: antecedent_core::reason_code!("not_supported"),
        message: format!("prior_composition.unit_not_supported: {}", refusal.message()),
    }
}

/// The support refusal as a transport refusal. `TransportError` carries no owned message, so
/// the refusal is reported under its fixed text; the unit's decision is on the plan view.
fn transport_refusal(_refusal: &NotSupported) -> TransportError {
    TransportError::InvalidWeights {
        message: "not_supported: the support unit of the compiled external prior transport \
                  route is not supported in this build; its decision is on the plan view",
    }
}

fn gate<I>(program: &KernelProgram<I>, ctx: &ExecutionContext) -> Result<(), NotSupported> {
    if GATE_ON_UNIT_SUPPORT {
        program.gate(ctx)?;
    }
    Ok(())
}

impl Study {
    /// Compile the composition of external Gaussian coefficient priors with a baseline (the
    /// inputs of `compose_external_priors`) under the `ExternalPriorComposition` route.
    /// Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_external_prior_composition(
        sources: Vec<ExternalPriorSource>,
        baseline: PriorSet,
        ctx: &ExecutionContext,
    ) -> Result<CompiledExternalPriorComposition, CausalError> {
        let input = GaussianCompositionInput { sources, baseline };
        let canonical = debug_canonical("antecedent.prior_composition.gaussian.v1", &input);
        let program = KernelProgram::compile(&GAUSSIAN_COMPOSITION, input, &canonical, ctx)?;
        Ok(CompiledExternalPriorComposition { program })
    }

    /// Compile the composition of Beta-Binomial count sources `(successes, trials, alpha)` with
    /// a base Beta prior (the inputs of `compose_beta_power_priors`) under the
    /// `ExternalPriorComposition` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_beta_power_prior_composition(
        base: BetaHyperparameters,
        sources: Vec<(u64, u64, f64)>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledBetaPowerPriorComposition, CausalError> {
        let input = BetaCompositionInput { base, sources };
        let canonical = debug_canonical("antecedent.prior_composition.beta_counts.v1", &input);
        let program = KernelProgram::compile(&BETA_COMPOSITION, input, &canonical, ctx)?;
        Ok(CompiledBetaPowerPriorComposition { program })
    }

    /// Compile the composition of Gamma-Poisson count sources `(total count, exposure, alpha)`
    /// with a base Gamma prior (the inputs of `compose_gamma_power_priors`) under the
    /// `ExternalPriorComposition` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_gamma_power_prior_composition(
        base: GammaHyperparameters,
        sources: Vec<(u64, f64, f64)>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledGammaPowerPriorComposition, CausalError> {
        let input = GammaCompositionInput { base, sources };
        let canonical = debug_canonical("antecedent.prior_composition.gamma_counts.v1", &input);
        let program = KernelProgram::compile(&GAMMA_COMPOSITION, input, &canonical, ctx)?;
        Ok(CompiledGammaPowerPriorComposition { program })
    }

    /// Compile the transport of external priors toward a tag-defined target population (the
    /// inputs of `apply_transport` and, with the baseline, of `compose_with_transport`) under
    /// the `ExternalPriorTransport` route. The context's population tags and reweighting are
    /// copied into the program. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_prior_tag_transport(
        sources: Vec<ExternalPriorSource>,
        baseline: PriorSet,
        context: &TransportContext<'_>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPriorTagTransport, CausalError> {
        let input =
            TagTransportInput { sources, baseline, context: OwnedTransportContext::of(context) };
        let canonical = debug_canonical("antecedent.prior_transport.population_tag.v1", &input);
        let program = KernelProgram::compile(&TAG_TRANSPORT, input, &canonical, ctx)?;
        Ok(CompiledPriorTagTransport { program })
    }

    /// Compile the transport of external priors toward a target declared as a weighting over
    /// the source rows (the inputs of `apply_weighted_target_transport` and, with the baseline,
    /// of `compose_with_weighted_target_transport`) under the `ExternalPriorTransport` route.
    /// Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_prior_weighted_target_transport(
        sources: Vec<ExternalPriorSource>,
        baseline: PriorSet,
        policy: TransportPolicy,
        target: WeightedTarget,
        coef_index: Option<usize>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPriorWeightedTargetTransport, CausalError> {
        let input = WeightedTargetTransportInput { sources, baseline, policy, target, coef_index };
        let canonical = debug_canonical("antecedent.prior_transport.weighted_target.v1", &input);
        let program = KernelProgram::compile(&WEIGHTED_TARGET_TRANSPORT, input, &canonical, ctx)?;
        Ok(CompiledPriorWeightedTargetTransport { program })
    }

    /// Compile the sensitivity of a kernel-weighted target transport to its bandwidth (the
    /// inputs of `kernel_bandwidth_sensitivity`) under the `ExternalPriorTransport` route.
    /// Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_kernel_bandwidth_sensitivity(
        family: KernelFamily,
        bandwidths: Vec<f64>,
        distances: Vec<f64>,
        unit_effects: Vec<f64>,
        min_ess_fraction: f64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledKernelBandwidthSensitivity, CausalError> {
        let input = BandwidthSensitivityInput {
            family,
            bandwidths,
            distances,
            unit_effects,
            min_ess_fraction,
        };
        let canonical =
            debug_canonical("antecedent.prior_transport.bandwidth_sensitivity.v1", &input);
        let program = KernelProgram::compile(&BANDWIDTH_SENSITIVITY, input, &canonical, ctx)?;
        Ok(CompiledKernelBandwidthSensitivity { program })
    }
}

impl CompiledExternalPriorComposition {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and compose, as `compose_external_priors` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) once the gate is on, in a release build, while the
    /// unit is not supported, and the refusals of the composition.
    pub fn compose(&self, ctx: &ExecutionContext) -> Result<ComposedPrior, ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        compose_external_priors(&input.sources, &input.baseline)
    }
}

impl CompiledBetaPowerPriorComposition {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and compose, as `compose_beta_power_priors` does.
    ///
    /// # Errors
    ///
    /// As [`CompiledExternalPriorComposition::compose`].
    pub fn compose(&self, ctx: &ExecutionContext) -> Result<BetaHyperparameters, ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        compose_beta_power_priors(input.base, &input.sources)
    }
}

impl CompiledGammaPowerPriorComposition {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and compose, as `compose_gamma_power_priors` does.
    ///
    /// # Errors
    ///
    /// As [`CompiledExternalPriorComposition::compose`].
    pub fn compose(&self, ctx: &ExecutionContext) -> Result<GammaHyperparameters, ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        compose_gamma_power_priors(input.base, &input.sources)
    }
}

impl CompiledPriorTagTransport {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and transport the sources, as `apply_transport` does: each
    /// source whose population differs from the target takes the declared policy, the
    /// reweighted coefficient moments, or a discount forced to zero with its reason.
    ///
    /// # Errors
    ///
    /// The support refusal once the gate is on, in a release build, while the unit is not
    /// supported (reported as `transport_invalid_weights` with a `not_supported` message), and
    /// the refusals of the transport.
    pub fn apply_transport(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<ExternalPriorSource>, Vec<TransportOutcome>), TransportError> {
        gate(&self.program, ctx).map_err(|refusal| transport_refusal(&refusal))?;
        let input = self.program.input();
        let populations = input.context.populations();
        apply_transport(&input.sources, &input.context.view(&populations))
    }

    /// Pass the support gate, transport the sources and compose them with the baseline, as
    /// `compose_with_transport` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) once the gate is on, in a release build, while the
    /// unit is not supported, and the refusals of the transport and the composition.
    pub fn compose_with_transport(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(ComposedPrior, Vec<TransportOutcome>), ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        let populations = input.context.populations();
        compose_with_transport(&input.sources, &input.baseline, &input.context.view(&populations))
    }
}

impl CompiledPriorWeightedTargetTransport {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The declared target weighting.
    #[must_use]
    pub fn target(&self) -> &WeightedTarget {
        &self.program.input().target
    }

    /// Pass the support gate, check the target's overlap and transport the sources, as
    /// `apply_weighted_target_transport` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) once the gate is on, in a release build, while the
    /// unit is not supported, the overlap refusal and the refusals of the transport.
    pub fn apply_weighted_target_transport(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<ExternalPriorSource>, TargetOverlap), ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        apply_weighted_target_transport(
            &input.sources,
            input.policy,
            &input.target,
            input.coef_index,
        )
    }

    /// Pass the support gate, transport the sources toward the target and compose them with
    /// the baseline, as `compose_with_weighted_target_transport` does.
    ///
    /// # Errors
    ///
    /// As [`Self::apply_weighted_target_transport`], and the refusals of the composition.
    pub fn compose_with_weighted_target_transport(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(ComposedPrior, TargetOverlap), ProbError> {
        gate(&self.program, ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        compose_with_weighted_target_transport(
            &input.sources,
            &input.baseline,
            input.policy,
            &input.target,
            input.coef_index,
        )
    }
}

impl CompiledKernelBandwidthSensitivity {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and report the kernel-weighted mean, its sampling variance and
    /// the overlap at every declared bandwidth, as `kernel_bandwidth_sensitivity` does.
    ///
    /// # Errors
    ///
    /// The support refusal once the gate is on, in a release build, while the unit is not
    /// supported (reported as `transport_invalid_weights` with a `not_supported` message), and
    /// the refusals of the sensitivity.
    pub fn sensitivity(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<BandwidthSensitivityRow>, TransportError> {
        gate(&self.program, ctx).map_err(|refusal| transport_refusal(&refusal))?;
        let input = self.program.input();
        kernel_bandwidth_sensitivity(
            input.family,
            &input.bandwidths,
            &input.distances,
            &input.unit_effects,
            input.min_ess_fraction,
        )
    }
}
