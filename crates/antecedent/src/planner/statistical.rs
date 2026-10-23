//! Layer three of the lowered program: the statistical program.
//!
//! Models, nuisance fits, scores or likelihood, priors, folds, resampling and
//! posterior computation, held as one object the physical plan is sized from.
//! The realization reuses the checked realization objects of
//! [`antecedent_core::inference_realization`]; nothing here converts one
//! realization into another, and a prior never identifies an effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CanonicalWriter, DeclaredSamplingDesign, GeneralizedBayes, LikelihoodModel, Operator,
    OrdinaryBayesianLikelihood, PartialIdentificationEnvelope, Prior, ScoreFrequentist, ScoreModel,
    SemanticDigest, VariableId,
};

use crate::error::CausalError;
use crate::planner::lowered::DesignShape;
use crate::{EstimatorId, IdentifierId, InferenceMode};

const STATISTICAL_PROGRAM_TAG: &str = "antecedent.lowered_program.statistical";
/// Declaration name used when the route record names no estimator.
const UNNAMED_ESTIMATOR: &str = "unnamed";

/// How the program turns data into an answer.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Realization {
    /// Score-based frequentist estimate with a sampling law.
    Frequentist(ScoreFrequentist),
    /// Ordinary posterior under a declared likelihood and prior.
    Bayesian(OrdinaryBayesianLikelihood),
    /// Generalized posterior: a loss, a prior and a learning rate.
    GeneralizedBayes(GeneralizedBayes),
    /// Identified set or envelope with its supported estimation uncertainty.
    Envelope(PartialIdentificationEnvelope),
    /// Exact evaluation of an identified functional, with no sampling law.
    ExactPoint {
        /// Name of the evaluator.
        evaluator: Arc<str>,
    },
}

impl Realization {
    fn write(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Frequentist(realization) => {
                writer.text("frequentist").bytes(realization.identity().as_bytes());
            }
            Self::Bayesian(realization) => {
                writer.text("bayesian").bytes(realization.identity().as_bytes());
            }
            Self::GeneralizedBayes(realization) => {
                writer.text("generalized_bayes").bytes(realization.identity().as_bytes());
            }
            Self::Envelope(realization) => {
                writer.text("envelope").bytes(realization.identity().as_bytes());
            }
            Self::ExactPoint { evaluator } => {
                writer.text("exact_point").text(evaluator);
            }
        }
    }
}

/// What a fitted model is for.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ModelRole {
    /// Outcome regression.
    Outcome,
    /// Treatment assignment (propensity) model.
    Propensity,
    /// Mediator model.
    Mediator,
    /// Instrument stage.
    Instrument,
    /// Structural mechanism of one node.
    Mechanism,
    /// Any other nuisance the estimator fits.
    Nuisance,
}

impl ModelRole {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Outcome => "outcome",
            Self::Propensity => "propensity",
            Self::Mediator => "mediator",
            Self::Instrument => "instrument",
            Self::Mechanism => "mechanism",
            Self::Nuisance => "nuisance",
        }
    }
}

/// One fitted model of the program.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModelSlot {
    pub(crate) role: ModelRole,
    pub(crate) family: Arc<str>,
    pub(crate) inputs: Arc<[VariableId]>,
    pub(crate) output: Option<VariableId>,
}

impl ModelSlot {
    /// What the model is for.
    #[must_use]
    pub const fn role(&self) -> ModelRole {
        self.role
    }

    /// Model family or kernel label.
    #[must_use]
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Variables the model reads.
    #[must_use]
    pub fn inputs(&self) -> &[VariableId] {
        &self.inputs
    }

    /// Variable the model predicts, when it predicts one.
    #[must_use]
    pub const fn output(&self) -> Option<VariableId> {
        self.output
    }
}

/// Unit a resample or a fold draws.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResampleUnit {
    /// One row.
    Row,
    /// One subject's whole trajectory.
    Subject,
    /// One cluster.
    Cluster,
    /// One contiguous block of calendar time.
    CalendarBlock,
}

impl ResampleUnit {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Subject => "subject",
            Self::Cluster => "cluster",
            Self::CalendarBlock => "calendar_block",
        }
    }

    /// The unit a declared sampling design resamples. No declaration resamples rows.
    #[must_use]
    pub const fn of_design(design: Option<DeclaredSamplingDesign>) -> Self {
        match design {
            None => Self::Row,
            Some(DeclaredSamplingDesign::IndependentSubjectTrajectories) => Self::Subject,
            Some(DeclaredSamplingDesign::ClusteredPanel) => Self::Cluster,
            Some(DeclaredSamplingDesign::DependentTimeSeries) => Self::CalendarBlock,
        }
    }
}

/// How the program resamples for uncertainty.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResamplingMethod {
    /// Nonparametric bootstrap over independent units.
    Bootstrap,
    /// Block bootstrap over contiguous time.
    BlockBootstrap,
    /// Permutation or placebo reassignment.
    Permutation,
    /// Leave-one-unit-out.
    Jackknife,
}

impl ResamplingMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::BlockBootstrap => "block_bootstrap",
            Self::Permutation => "permutation",
            Self::Jackknife => "jackknife",
        }
    }
}

/// Resampling the program performs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ResamplingPlan {
    pub(crate) method: ResamplingMethod,
    pub(crate) replicates: u32,
    pub(crate) unit: ResampleUnit,
}

impl ResamplingPlan {
    /// Resampling method.
    #[must_use]
    pub const fn method(&self) -> ResamplingMethod {
        self.method
    }

    /// Requested replicates.
    #[must_use]
    pub const fn replicates(&self) -> u32 {
        self.replicates
    }

    /// Unit each replicate draws.
    #[must_use]
    pub const fn unit(&self) -> ResampleUnit {
        self.unit
    }
}

/// Sample splitting for cross-fitting.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FoldPlan {
    pub(crate) folds: usize,
    pub(crate) unit: ResampleUnit,
    pub(crate) embargo: Option<usize>,
}

impl FoldPlan {
    /// Number of folds.
    #[must_use]
    pub const fn folds(&self) -> usize {
        self.folds
    }

    /// Unit each fold holds whole.
    #[must_use]
    pub const fn unit(&self) -> ResampleUnit {
        self.unit
    }

    /// Gap kept between training and evaluation time, when the split needs one.
    #[must_use]
    pub const fn embargo(&self) -> Option<usize> {
        self.embargo
    }
}

/// Posterior computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PosteriorAlgorithm {
    /// Analytic conjugate update.
    Conjugate,
    /// Laplace approximation.
    Laplace,
    /// Hamiltonian Monte Carlo.
    Hmc,
    /// Mixture of per-atom posteriors over a graph posterior.
    AtomMixture,
}

impl PosteriorAlgorithm {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Conjugate => "conjugate",
            Self::Laplace => "laplace",
            Self::Hmc => "hmc",
            Self::AtomMixture => "atom_mixture",
        }
    }
}

/// How the posterior is computed.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PosteriorPlan {
    pub(crate) algorithm: PosteriorAlgorithm,
    pub(crate) draws: usize,
    pub(crate) prior: Prior,
}

impl PosteriorPlan {
    /// Posterior algorithm.
    #[must_use]
    pub const fn algorithm(&self) -> PosteriorAlgorithm {
        self.algorithm
    }

    /// Requested draws.
    #[must_use]
    pub const fn draws(&self) -> usize {
        self.draws
    }

    /// Declared prior.
    #[must_use]
    pub const fn prior(&self) -> &Prior {
        &self.prior
    }
}

/// Layer three: the statistical program of one route.
#[derive(Clone, Debug)]
pub struct StatisticalProgram {
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) realization: Realization,
    pub(crate) models: Arc<[ModelSlot]>,
    pub(crate) folds: Option<FoldPlan>,
    pub(crate) resampling: Option<ResamplingPlan>,
    pub(crate) posterior: Option<PosteriorPlan>,
    pub(crate) design: DesignShape,
    pub(crate) operators: Arc<[Operator]>,
}

/// What [`StatisticalProgram::declared`] reads from a study.
pub(crate) struct DeclaredProgram<'a> {
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) inference: &'a InferenceMode,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) sampling_design: Option<DeclaredSamplingDesign>,
    pub(crate) design: DesignShape,
    pub(crate) operators: Vec<Operator>,
}

impl StatisticalProgram {
    /// The program a study declares: its identifier and estimator, its inference
    /// mode, its bootstrap request under its sampling design. Models and folds are
    /// left to the route's own lowering.
    ///
    /// # Errors
    ///
    /// A declaration name is empty.
    pub(crate) fn declared(declared: DeclaredProgram<'_>) -> Result<Self, CausalError> {
        let DeclaredProgram {
            identifier,
            estimator,
            inference,
            bootstrap_replicates,
            sampling_design,
            design,
            operators,
        } = declared;
        let name = estimator.map_or(UNNAMED_ESTIMATOR, |estimator| estimator.as_str());
        let (realization, posterior) = match inference {
            InferenceMode::Frequentist => (
                Realization::Frequentist(ScoreFrequentist::new(
                    ScoreModel::new(name).map_err(declaration_error)?,
                )),
                None,
            ),
            InferenceMode::Bayesian(config) => {
                let prior = Prior::new(prior_name(config)).map_err(declaration_error)?;
                let likelihood = LikelihoodModel::new(name).map_err(declaration_error)?;
                let algorithm = match config.backend {
                    antecedent_estimate::BayesianBackendKind::ConjugateGaussian => {
                        PosteriorAlgorithm::Conjugate
                    }
                    antecedent_estimate::BayesianBackendKind::Laplace => {
                        PosteriorAlgorithm::Laplace
                    }
                    antecedent_estimate::BayesianBackendKind::Hmc => PosteriorAlgorithm::Hmc,
                };
                (
                    Realization::Bayesian(OrdinaryBayesianLikelihood::new(
                        likelihood,
                        prior.clone(),
                    )),
                    Some(PosteriorPlan { algorithm, draws: config.n_draws, prior }),
                )
            }
        };
        let unit = ResampleUnit::of_design(sampling_design);
        let resampling = (bootstrap_replicates > 0).then_some(ResamplingPlan {
            method: if unit == ResampleUnit::CalendarBlock {
                ResamplingMethod::BlockBootstrap
            } else {
                ResamplingMethod::Bootstrap
            },
            replicates: bootstrap_replicates,
            unit,
        });
        Ok(Self {
            identifier,
            estimator,
            realization,
            models: Arc::from(Vec::new()),
            folds: None,
            resampling,
            posterior,
            design,
            operators: Arc::from(operators),
        })
    }

    /// Identification strategy the program evaluates, when the route names one.
    #[must_use]
    pub const fn identifier(&self) -> Option<IdentifierId> {
        self.identifier
    }

    /// Estimator, when the route names one.
    #[must_use]
    pub const fn estimator(&self) -> Option<EstimatorId> {
        self.estimator
    }

    /// Inference realization.
    #[must_use]
    pub const fn realization(&self) -> &Realization {
        &self.realization
    }

    /// Fitted models.
    #[must_use]
    pub fn models(&self) -> &[ModelSlot] {
        &self.models
    }

    /// Cross-fitting split, when the estimator splits.
    #[must_use]
    pub const fn folds(&self) -> Option<FoldPlan> {
        self.folds
    }

    /// Resampling for uncertainty, when requested.
    #[must_use]
    pub const fn resampling(&self) -> Option<ResamplingPlan> {
        self.resampling
    }

    /// Posterior computation, when the realization is a posterior.
    #[must_use]
    pub const fn posterior(&self) -> Option<&PosteriorPlan> {
        self.posterior.as_ref()
    }

    /// Design the program fits.
    #[must_use]
    pub const fn design(&self) -> DesignShape {
        self.design
    }

    /// Scientific and evidence-use operator contracts.
    #[must_use]
    pub fn operators(&self) -> &[Operator] {
        &self.operators
    }

    /// Identity of the program. Row counts are excluded: a data refresh with the
    /// same encoded design does not change the program.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(STATISTICAL_PROGRAM_TAG);
        writer
            .opt_text(self.identifier.as_ref().map(IdentifierId::as_str))
            .opt_text(self.estimator.as_ref().map(EstimatorId::as_str));
        self.realization.write(&mut writer);
        writer.word(self.models.len() as u64);
        for model in &*self.models {
            writer.text(model.role.as_str()).text(&model.family).word(model.inputs.len() as u64);
            for input in &*model.inputs {
                writer.word(u64::from(input.raw()));
            }
            writer.opt_word(model.output.map(|output| u64::from(output.raw())));
        }
        match self.folds {
            Some(folds) => writer
                .word(1)
                .word(folds.folds as u64)
                .text(folds.unit.as_str())
                .opt_word(folds.embargo.map(|gap| gap as u64)),
            None => writer.word(0),
        };
        match self.resampling {
            Some(plan) => writer
                .word(1)
                .text(plan.method.as_str())
                .word(u64::from(plan.replicates))
                .text(plan.unit.as_str()),
            None => writer.word(0),
        };
        match &self.posterior {
            Some(plan) => writer
                .word(1)
                .text(plan.algorithm.as_str())
                .word(plan.draws as u64)
                .text(plan.prior.name()),
            None => writer.word(0),
        };
        writer
            .opt_word(self.design.columns())
            .text(self.design.source().as_str())
            .word(self.operators.len() as u64);
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// The declared name of a study's prior. The name is the prior's identity, so the
/// isotropic scale is part of it.
fn prior_name(config: &crate::BayesianConfig) -> String {
    if config.prior.is_some() {
        "explicit_prior_set".to_owned()
    } else if config.prior_artifact.is_some() {
        "prior_artifact".to_owned()
    } else {
        format!("isotropic_normal(scale={})", config.prior_scale)
    }
}

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}
