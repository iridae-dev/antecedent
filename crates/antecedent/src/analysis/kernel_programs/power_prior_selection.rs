//! The power-prior discount chosen from the target data, as compiled program families.
//!
//! A power prior discounts a source by an exponent `alpha` in `[0, 1]`. The selection chooses
//! `alpha` on a grid from the target data (the marginal likelihood, or the `K`-fold log
//! predictive density of held-out target rows) and reports the posterior mean of the target
//! functional with its conditional, uniform and (under a declared discrepancy bound) bias-aware
//! intervals, with the score, estimate and conditional interval at every grid point.
//!
//! Three conjugate targets compile here, each to its own route: a Beta-Binomial proportion
//! (`PowerPriorBetaSelection`), a Gamma-Poisson rate (`PowerPriorGammaSelection`) and a Gaussian
//! linear functional under external Gaussian sources (`PowerPriorLinearSelection`). Compiling
//! binds the digest of the declared source, target data and selection options into a six-layer
//! program. Every run passes the support gate of the route's unit and then calls the kernel
//! selection the direct path calls (`antecedent_prob::power_prior_alpha`), with the same
//! arguments, so the selected discount, the estimate, every interval and the sensitivity table
//! are the direct path's bit for bit. The selection reads only the options' seed for its fold
//! assignment and critical-value draws, never the context's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_prob::{
    AlphaCriterion, AlphaSelectionOptions, BetaHyperparameters, BetaSource, ComposedPrior,
    ExternalPriorSource, GammaHyperparameters, GammaSource, LinearTarget, PriorSet, ProbError,
    SelectedAlphaReport, select_beta_power_prior, select_beta_prior, select_gamma_power_prior,
    select_gamma_prior, select_linear_power_prior, select_linear_prior,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const SELECTION_DESIGN: &str = "power-prior discount chosen on a grid from the target data by \
                                marginal likelihood or K-fold held-out predictive density";

const BETA_SELECTION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::PowerPriorBetaSelection,
    plan_id: "prior.power_prior_beta_selection",
    functional: FunctionalKind::Association,
    evaluator: "beta_binomial_power_prior_alpha_selection",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: SELECTION_DESIGN,
};

const GAMMA_SELECTION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::PowerPriorGammaSelection,
    plan_id: "prior.power_prior_gamma_selection",
    functional: FunctionalKind::Association,
    evaluator: "gamma_poisson_power_prior_alpha_selection",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: SELECTION_DESIGN,
};

const LINEAR_SELECTION: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::PowerPriorLinearSelection,
    plan_id: "prior.power_prior_linear_selection",
    functional: FunctionalKind::Association,
    evaluator: "gaussian_linear_power_prior_alpha_selection",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::Observations,
    graph_class: "none",
    data: DataClassification::Tabular,
    design: SELECTION_DESIGN,
};

/// The declared Beta-Binomial selection: source counts, target trials and options.
#[derive(Clone, Debug)]
struct BetaSelectionInput {
    source: BetaSource,
    outcomes: Vec<bool>,
    options: AlphaSelectionOptions,
}

/// The declared Gamma-Poisson selection: source counts, target counts and options.
#[derive(Clone, Debug)]
struct GammaSelectionInput {
    source: GammaSource,
    counts: Vec<u64>,
    options: AlphaSelectionOptions,
}

/// The declared Gaussian linear selection: external sources, baseline, target and options.
#[derive(Clone, Debug)]
struct LinearSelectionInput {
    sources: Vec<ExternalPriorSource>,
    baseline: PriorSet,
    x: Vec<f64>,
    y: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    functional: Vec<f64>,
    options: AlphaSelectionOptions,
}

impl LinearSelectionInput {
    /// The target view the kernel reads, over the owned rows.
    fn target(&self) -> LinearTarget<'_> {
        LinearTarget {
            x: &self.x,
            y: &self.y,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            functional: &self.functional,
        }
    }
}

/// A compiled Beta-Binomial power-prior discount selection.
#[derive(Clone, Debug)]
pub struct CompiledPowerPriorBetaSelection {
    program: KernelProgram<BetaSelectionInput>,
}

/// A compiled Gamma-Poisson power-prior discount selection.
#[derive(Clone, Debug)]
pub struct CompiledPowerPriorGammaSelection {
    program: KernelProgram<GammaSelectionInput>,
}

/// A compiled Gaussian linear power-prior discount selection over external sources.
#[derive(Clone, Debug)]
pub struct CompiledPowerPriorLinearSelection {
    program: KernelProgram<LinearSelectionInput>,
}

fn write_options(writer: &mut CanonicalWriter, options: &AlphaSelectionOptions) {
    let folds = match options.criterion {
        AlphaCriterion::EmpiricalBayes => 0,
        AlphaCriterion::CrossValidated { folds } => folds as u64,
    };
    writer
        .text(options.criterion.name())
        .word(folds)
        .reals(&options.grid)
        .real(options.level)
        .opt_real(options.discrepancy_bound)
        .word(options.seed);
}

fn beta_canonical(input: &BetaSelectionInput) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.power_prior_selection.beta.v1");
    writer
        .real(input.source.base.alpha)
        .real(input.source.base.beta)
        .word(input.source.successes)
        .word(input.source.trials)
        .bytes(&input.outcomes.iter().map(|&outcome| u8::from(outcome)).collect::<Vec<_>>());
    write_options(&mut writer, &input.options);
    writer.finish()
}

fn gamma_canonical(input: &GammaSelectionInput) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.power_prior_selection.gamma.v1");
    writer
        .real(input.source.base.shape)
        .real(input.source.base.rate)
        .word(input.source.total_count)
        .real(input.source.exposure)
        .words(&input.counts);
    write_options(&mut writer, &input.options);
    writer.finish()
}

/// The sources and baseline enter through their `Debug` rendering: ordered vectors and options
/// whose reals print in their shortest round-trip form, so equal declarations give equal bytes.
fn linear_canonical(input: &LinearSelectionInput) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.power_prior_selection.linear.v1");
    writer.word(input.sources.len() as u64);
    for source in &input.sources {
        writer
            .text(&source.id)
            .text(&format!("{:?}", source.prior))
            .real(source.weight.alpha)
            .opt_real(source.weight.mixture_weight)
            .opt_real(source.ess);
    }
    writer
        .text(&format!("{:?}", input.baseline))
        .word(input.n_rows as u64)
        .word(input.n_cols as u64)
        .reals(&input.x)
        .reals(&input.y)
        .reals(&input.functional);
    write_options(&mut writer, &input.options);
    writer.finish()
}

/// The support refusal as a probability refusal, for the selection's own error type.
fn prob_refusal(refusal: &NotSupported) -> ProbError {
    ProbError::Refused {
        code: antecedent_core::reason_code!("not_supported"),
        message: refusal.message().to_owned(),
    }
}

impl Study {
    /// Compile a Beta-Binomial power-prior discount selection (the inputs of
    /// `select_beta_power_prior` and `select_beta_prior`) under the `PowerPriorBetaSelection`
    /// route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_beta_power_prior_selection(
        source: BetaSource,
        outcomes: Vec<bool>,
        options: AlphaSelectionOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPowerPriorBetaSelection, CausalError> {
        let input = BetaSelectionInput { source, outcomes, options };
        let canonical = beta_canonical(&input);
        let program = KernelProgram::compile(&BETA_SELECTION, input, &canonical, ctx)?;
        Ok(CompiledPowerPriorBetaSelection { program })
    }

    /// Compile a Gamma-Poisson power-prior discount selection (the inputs of
    /// `select_gamma_power_prior` and `select_gamma_prior`) under the `PowerPriorGammaSelection`
    /// route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_gamma_power_prior_selection(
        source: GammaSource,
        counts: Vec<u64>,
        options: AlphaSelectionOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPowerPriorGammaSelection, CausalError> {
        let input = GammaSelectionInput { source, counts, options };
        let canonical = gamma_canonical(&input);
        let program = KernelProgram::compile(&GAMMA_SELECTION, input, &canonical, ctx)?;
        Ok(CompiledPowerPriorGammaSelection { program })
    }

    /// Compile a Gaussian linear power-prior discount selection over external sources (the
    /// inputs of `select_linear_power_prior` and `select_linear_prior`) under the
    /// `PowerPriorLinearSelection` route. The target's rows are copied into the program.
    /// Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_linear_power_prior_selection(
        sources: Vec<ExternalPriorSource>,
        baseline: PriorSet,
        target: LinearTarget<'_>,
        options: AlphaSelectionOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPowerPriorLinearSelection, CausalError> {
        let input = LinearSelectionInput {
            sources,
            baseline,
            x: target.x.to_vec(),
            y: target.y.to_vec(),
            n_rows: target.n_rows,
            n_cols: target.n_cols,
            functional: target.functional.to_vec(),
            options,
        };
        let canonical = linear_canonical(&input);
        let program = KernelProgram::compile(&LINEAR_SELECTION, input, &canonical, ctx)?;
        Ok(CompiledPowerPriorLinearSelection { program })
    }
}

impl CompiledPowerPriorBetaSelection {
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

    /// Pass the support gate and choose the discount, as `select_beta_power_prior` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the selection.
    pub fn select_power_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SelectedAlphaReport, ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_beta_power_prior(input.source, &input.outcomes, &input.options)
    }

    /// Pass the support gate and return the Beta power prior at the chosen discount with the
    /// selection report, as `select_beta_prior` does.
    ///
    /// # Errors
    ///
    /// As [`Self::select_power_prior`].
    pub fn select_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(BetaHyperparameters, SelectedAlphaReport), ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_beta_prior(input.source, &input.outcomes, &input.options)
    }
}

impl CompiledPowerPriorGammaSelection {
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

    /// Pass the support gate and choose the discount, as `select_gamma_power_prior` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the selection.
    pub fn select_power_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SelectedAlphaReport, ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_gamma_power_prior(input.source, &input.counts, &input.options)
    }

    /// Pass the support gate and return the Gamma power prior at the chosen discount with the
    /// selection report, as `select_gamma_prior` does.
    ///
    /// # Errors
    ///
    /// As [`Self::select_power_prior`].
    pub fn select_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(GammaHyperparameters, SelectedAlphaReport), ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_gamma_prior(input.source, &input.counts, &input.options)
    }
}

impl CompiledPowerPriorLinearSelection {
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

    /// Pass the support gate and choose the discount, as `select_linear_power_prior` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the selection.
    pub fn select_power_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SelectedAlphaReport, ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_linear_power_prior(&input.sources, &input.baseline, input.target(), &input.options)
    }

    /// Pass the support gate and return the prior composed at the chosen discount with the
    /// selection report, as `select_linear_prior` does.
    ///
    /// # Errors
    ///
    /// As [`Self::select_power_prior`], or a composition failure.
    pub fn select_prior(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(ComposedPrior, SelectedAlphaReport), ProbError> {
        self.program.gate(ctx).map_err(|refusal| prob_refusal(&refusal))?;
        let input = self.program.input();
        select_linear_prior(&input.sources, &input.baseline, input.target(), &input.options)
    }
}
