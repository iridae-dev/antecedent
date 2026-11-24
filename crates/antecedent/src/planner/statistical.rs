//! Layer three of the lowered program: the statistical program.
//!
//! Models, nuisance fits, scores or likelihood, priors, folds, resampling and
//! posterior computation, held as one object the physical plan is sized from.
//! The realization reuses the checked realization objects of
//! [`antecedent_core::inference_realization`]; nothing here converts one
//! realization into another, and a prior never identifies an effect.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CanonicalWriter, DeclaredSamplingDesign, GeneralizedBayes, LikelihoodModel, Operator,
    OrdinaryBayesianLikelihood, PartialIdentificationEnvelope, Prior, ScoreFrequentist, ScoreModel,
    SemanticDigest, VariableId,
};

use crate::error::CausalError;
use crate::planner::lowered::DesignShape;
use crate::{EstimatorId, IdentifierId, InferenceMode};

/// Versioned with the identity layout: v2 writes the typed quasi-experimental identifiers of
/// difference-in-differences, synthetic control and the local ratios, which v1 recorded as
/// absent, and the declared posterior algorithm.
const STATISTICAL_PROGRAM_TAG: &str = "antecedent.lowered_program.statistical.v2";
/// v3 binds each operator's content (effect, graph, input and output roles in declaration
/// order, causal feedback and frozen-fit reuse) where v2 wrote only how many there are.
const STATISTICAL_PROGRAM_TAG_V3: &str = "antecedent.lowered_program.statistical.v3";
/// Declaration name used when the route record names no estimator.
const UNNAMED_ESTIMATOR: &str = "unnamed";

/// Law retained by a nested native fitted producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NestedFitLaw {
    /// Parameter or causal-functional posterior, not outcome risk.
    Posterior,
    /// Predictive outcome law.
    Predictive,
}

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
    /// The fixed, bounded graph-selection and row-imputation realization.
    GraphImputationV1,
    /// Fixed, scientifically closed categorical posterior candidate.
    CategoricalPosteriorCandidateV1,
    /// Monte Carlo utilities from a declared finite sequential model.
    SequentialKnownLawMonteCarlo,
    /// Whole-trajectory bootstrap utilities from logged sequential evidence.
    SequentialTrajectoryBootstrap,
    /// Exact evaluation of an identified functional, with no sampling law.
    ExactPoint {
        /// Name of the evaluator.
        evaluator: Arc<str>,
    },
    /// A declared, aligned parameter posterior evaluated by the program.
    DeclaredPosterior {
        /// Name of the posterior evaluator.
        evaluator: Arc<str>,
    },
    /// A declared, aligned predictive outcome law transformed by the program.
    DeclaredPredictive {
        /// Name of the predictive evaluator.
        evaluator: Arc<str>,
    },
    /// A compiler-owned producer refit from raw evidence before evaluation.
    NestedFit {
        /// Child program retaining the actual fit likelihood and prior.
        producer_program: antecedent_core::SemanticDigest,
        /// Stable native method name.
        method: Arc<str>,
        /// Meaning of the child output law.
        law: NestedFitLaw,
    },
    /// Aligned draws evaluated under each admissible interval assumption.
    DeclaredAssumptionEnvelope {
        /// Name of the interval evaluator.
        evaluator: Arc<str>,
        /// Whether the aligned row axis is a sampling law or a posterior.
        draw_meaning: antecedent_contract::propagation_chain::LawMeaning,
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
            Self::GraphImputationV1 => {
                writer.text("graph_imputation_v1");
            }
            Self::CategoricalPosteriorCandidateV1 => {
                writer.text("categorical_posterior_candidate_v1");
            }
            Self::SequentialKnownLawMonteCarlo => {
                writer.text("sequential_known_law_monte_carlo_v1");
            }
            Self::SequentialTrajectoryBootstrap => {
                writer.text("sequential_trajectory_bootstrap_v1");
            }
            Self::ExactPoint { evaluator } => {
                writer.text("exact_point").text(evaluator);
            }
            Self::DeclaredPosterior { evaluator } => {
                writer.text("declared_posterior").text(evaluator);
            }
            Self::DeclaredPredictive { evaluator } => {
                writer.text("declared_predictive").text(evaluator);
            }
            Self::NestedFit { producer_program, method, law } => {
                writer.text("nested_fit").bytes(producer_program.as_bytes()).text(method).text(
                    match law {
                        NestedFitLaw::Posterior => "posterior",
                        NestedFitLaw::Predictive => "predictive",
                    },
                );
            }
            Self::DeclaredAssumptionEnvelope { evaluator, draw_meaning } => {
                writer
                    .text("declared_assumption_envelope")
                    .text(evaluator)
                    .word(draw_meaning.tag());
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
    /// Exact enumeration of a discrete posterior (structures or their edge marginals).
    Enumeration,
    /// Markov chain Monte Carlo over a discrete space (orders or structures).
    DiscreteMcmc,
    /// Bayesian bootstrap: Dirichlet weights on rows, or on contiguous blocks of a series.
    BayesianBootstrap,
    /// Random-walk Metropolis over jointly fitted cyclic mechanisms.
    RandomWalkMetropolis,
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
            Self::Enumeration => "enumeration",
            Self::DiscreteMcmc => "discrete_mcmc",
            Self::BayesianBootstrap => "bayesian_bootstrap",
            Self::RandomWalkMetropolis => "random_walk_metropolis",
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
                let algorithm = if estimator
                    .is_some_and(EstimatorId::posterior_is_bayesian_bootstrap)
                {
                    PosteriorAlgorithm::BayesianBootstrap
                } else {
                    match config.backend {
                        antecedent_estimate::BayesianBackendKind::ConjugateGaussian => {
                            PosteriorAlgorithm::Conjugate
                        }
                        antecedent_estimate::BayesianBackendKind::Laplace => {
                            PosteriorAlgorithm::Laplace
                        }
                        antecedent_estimate::BayesianBackendKind::Hmc => PosteriorAlgorithm::Hmc,
                    }
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
        self.digest(STATISTICAL_PROGRAM_TAG_V3, true)
    }

    /// The layer digest a lowered program's identity binds: the v2 layout, which writes the
    /// operators' count and not their content. Every route builds its statistical operators
    /// as a fixed list of its program cell (none, a literal list, or the linear-Gaussian,
    /// temporal or response list of that cell), and the lowered identity binds the cell, so
    /// the content adds nothing there; keeping v2 keeps every released program identity, which
    /// bundle replay and the portable consume paths re-check against stored identities.
    pub(crate) fn lowered_layer_digest(&self) -> SemanticDigest {
        self.digest(STATISTICAL_PROGRAM_TAG, false)
    }

    fn digest(&self, tag: &str, operator_content: bool) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(tag);
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
        if operator_content {
            // Ordered: an operator list is the program's pipeline, in execution order.
            for operator in &*self.operators {
                write_operator(&mut writer, operator);
            }
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// One operator edge: its effect and graph, its roles in declaration order (count-prefixed),
/// whether it carries causal feedback, and whether it allows frozen-fit reuse.
fn write_operator(writer: &mut CanonicalWriter, operator: &Operator) {
    writer
        .text(operator.effect().as_str())
        .text(operator.graph().as_str())
        .word(operator.inputs().len() as u64);
    for role in operator.inputs() {
        writer.text(role.as_str());
    }
    writer.word(operator.outputs().len() as u64);
    for role in operator.outputs() {
        writer.text(role.as_str());
    }
    writer.word(u64::from(operator.causal_feedback())).text(operator.frozen_fit_reuse().as_str());
}

/// The declared name of a study's prior. The name is the prior's identity, so the
/// isotropic scale is part of it, exactly by its bits.
fn prior_name(config: &crate::BayesianConfig) -> String {
    if config.prior.is_some() {
        "explicit_prior_set".to_owned()
    } else if config.prior_artifact.is_some() {
        "prior_artifact".to_owned()
    } else {
        isotropic_prior_name(config.prior_scale)
    }
}

/// The released name renders the scale through `Display`, which is injective on every bit
/// pattern but NaN: the shortest round-trip decimal names a finite value exactly, `-0` and `0`
/// differ, and the infinities are `inf` and `-inf`. Every NaN rendered as `NaN`, so a NaN other
/// than the canonical `f64::NAN` is named by its bits; the canonical NaN, and every other
/// scale, keep their released name.
fn isotropic_prior_name(scale: f64) -> String {
    if scale.is_nan() && scale.to_bits() != f64::NAN.to_bits() {
        format!("isotropic_normal(scale_bits=0x{:016x})", scale.to_bits())
    } else {
        format!("isotropic_normal(scale={scale})")
    }
}

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};
    use antecedent_core::{EvidenceClaims, OperatorDeclaration, OperatorEffect, OperatorGraph};
    use antecedent_core::{OperatorPorts, OperatorRole};

    fn declared_algorithm(estimator: EstimatorId) -> PosteriorAlgorithm {
        let inference = InferenceMode::Bayesian(crate::BayesianConfig::laplace());
        StatisticalProgram::declared(DeclaredProgram {
            identifier: None,
            estimator: Some(estimator),
            inference: &inference,
            bootstrap_replicates: 0,
            sampling_design: None,
            design: DesignShape::not_design_based("test"),
            operators: Vec::new(),
        })
        .expect("declared program")
        .posterior()
        .expect("a Bayesian program plans a posterior")
        .algorithm()
    }

    #[test]
    fn bootstrap_estimators_declare_the_bayesian_bootstrap_whatever_the_backend() {
        for estimator in EstimatorId::ALL {
            let expected = if estimator.posterior_is_bayesian_bootstrap() {
                PosteriorAlgorithm::BayesianBootstrap
            } else {
                PosteriorAlgorithm::Laplace
            };
            assert_eq!(declared_algorithm(*estimator), expected, "{}", estimator.as_str());
        }
        assert!(EstimatorId::BayesianRobustAte.posterior_is_bayesian_bootstrap());
        assert!(!EstimatorId::BayesianGcomp.posterior_is_bayesian_bootstrap());
    }

    fn declared() -> StatisticalProgram {
        StatisticalProgram::declared(DeclaredProgram {
            identifier: None,
            estimator: None,
            inference: &InferenceMode::Frequentist,
            bootstrap_replicates: 0,
            sampling_design: None,
            design: DesignShape::not_design_based("test"),
            operators: Vec::new(),
        })
        .expect("declared program")
    }

    fn slot(role: ModelRole, inputs: &[u32]) -> ModelSlot {
        ModelSlot {
            role,
            family: Arc::from("ols"),
            inputs: inputs.iter().copied().map(VariableId::from_raw).collect(),
            output: Some(VariableId::from_raw(9)),
        }
    }

    fn with_models(models: Vec<ModelSlot>) -> SemanticDigest {
        let mut program = declared();
        program.models = Arc::from(models);
        program.identity()
    }

    #[test]
    fn site_statistical_models_and_their_inputs_are_ordered() {
        // Slots are in the estimator's fixed role order (the plan view lists them and the
        // associational route reads the first); inputs are design columns in column order.
        ordered_laws(
            &[ModelRole::Propensity, ModelRole::Outcome, ModelRole::Nuisance],
            |roles: &[ModelRole]| {
                Some(with_models(roles.iter().map(|&role| slot(role, &[0, 1])).collect()))
            },
        );
        ordered_laws(&[0_u32, 1, 2], |inputs: &[u32]| {
            Some(with_models(vec![slot(ModelRole::Outcome, inputs)]))
        });
        all_distinct(&[
            with_models(Vec::new()),
            with_models(vec![slot(ModelRole::Outcome, &[0, 1])]),
            with_models(vec![slot(ModelRole::Outcome, &[0]), slot(ModelRole::Outcome, &[1])]),
            with_models(vec![ModelSlot { output: None, ..slot(ModelRole::Outcome, &[0, 1]) }]),
            with_models(vec![ModelSlot {
                family: Arc::from("glm"),
                ..slot(ModelRole::Outcome, &[0, 1])
            }]),
        ]);
    }

    fn with(edit: fn(&mut StatisticalProgram)) -> SemanticDigest {
        let mut program = declared();
        edit(&mut program);
        program.identity()
    }

    #[test]
    fn site_statistical_realization_and_plans_are_tagged() {
        all_distinct(&[
            declared().identity(),
            with(|p| p.realization = Realization::ExactPoint { evaluator: Arc::from("e") }),
            with(|p| p.realization = Realization::DeclaredPosterior { evaluator: Arc::from("e") }),
            with(|p| p.realization = Realization::DeclaredPredictive { evaluator: Arc::from("e") }),
            with(|p| p.realization = Realization::GraphImputationV1),
            with(|p| p.realization = Realization::CategoricalPosteriorCandidateV1),
            with(|p| p.realization = Realization::SequentialKnownLawMonteCarlo),
            with(|p| p.realization = Realization::SequentialTrajectoryBootstrap),
            with(|p| {
                p.realization = Realization::NestedFit {
                    producer_program: SemanticDigest::from_bytes([0; 32]),
                    method: Arc::from("e"),
                    law: NestedFitLaw::Posterior,
                };
            }),
            with(|p| {
                p.realization = Realization::NestedFit {
                    producer_program: SemanticDigest::from_bytes([0; 32]),
                    method: Arc::from("e"),
                    law: NestedFitLaw::Predictive,
                };
            }),
            with(|p| p.folds = Some(FoldPlan { folds: 2, unit: ResampleUnit::Row, embargo: None })),
            with(|p| {
                p.folds = Some(FoldPlan { folds: 2, unit: ResampleUnit::Row, embargo: Some(0) });
            }),
            with(|p| {
                p.resampling = Some(ResamplingPlan {
                    method: ResamplingMethod::Bootstrap,
                    replicates: 2,
                    unit: ResampleUnit::Row,
                });
            }),
            with(|p| p.design = DesignShape::quoted(10, 2, true)),
        ]);
    }

    /// (layer identity, the released layer digest the lowered identity binds)
    fn with_operators(operators: Vec<Operator>) -> (SemanticDigest, SemanticDigest) {
        let mut program = declared();
        program.operators = Arc::from(operators);
        (program.identity(), program.lowered_layer_digest())
    }

    fn fit(inputs: &[OperatorRole]) -> Operator {
        Operator::new(
            OperatorDeclaration::new(
                OperatorEffect::Fit,
                OperatorGraph::Execution,
                OperatorPorts::new(inputs.iter().copied(), [OperatorRole::FitHandle]),
            )
            .with_evidence_claims(EvidenceClaims::nuisance_fit()),
        )
        .expect("a fit edge")
    }

    #[test]
    fn site_statistical_operators_are_bound_by_content_in_order() {
        let operators = crate::planner::linear_gaussian_program_operators().expect("operators");
        ordered_laws(&operators, |ops: &[Operator]| Some(with_operators(ops.to_vec()).0));
        // A fit reads its input roles in declaration order.
        ordered_laws(&[OperatorRole::Outcome, OperatorRole::WorldOrSchedule], |roles| {
            Some(with_operators(vec![fit(roles)]).0)
        });
        // One operator each, different content: the layer identity tells them apart.
        let read = with_operators(vec![operators[0].clone()]);
        let intervene = with_operators(vec![operators[2].clone()]);
        assert_ne!(read.0, intervene.0, "the layer identity binds each operator's content");
        // The lowered identity binds the released digest, which counts operators: every route's
        // operator list is fixed by its cell, which the lowered identity binds.
        assert_eq!(read.1, intervene.1, "the released digest is unchanged");
        assert_ne!(read.1, with_operators(Vec::new()).1, "the released digest counts operators");
        assert_ne!(read.0, read.1, "the layer identity is versioned apart from the released one");
    }

    #[test]
    fn site_statistical_lowered_layer_digest_keeps_the_released_layout() {
        let program = declared();
        let mut writer = CanonicalWriter::new("antecedent.lowered_program.statistical.v2");
        writer.opt_text(None).opt_text(None);
        program.realization.write(&mut writer);
        writer.word(0).word(0).word(0).word(0).opt_word(None).text("not_design_based").word(0);
        assert_eq!(program.lowered_layer_digest(), SemanticDigest::from_bytes(writer.finish()));
    }

    #[test]
    fn site_statistical_prior_scale_is_named_by_its_bits() {
        // Released names are unchanged, the canonical NaN included.
        assert_eq!(isotropic_prior_name(10.0), "isotropic_normal(scale=10)");
        assert_eq!(isotropic_prior_name(f64::NAN), "isotropic_normal(scale=NaN)");
        bit_exact(|scale| Some(isotropic_prior_name(scale)));
        all_distinct(
            &[0.1, 0.2, 1e-300, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, f64::NAN]
                .map(isotropic_prior_name),
        );
        // Through the program: the prior name is the posterior plan's prior and the realization's.
        bit_exact(|scale| {
            let inference =
                InferenceMode::Bayesian(crate::BayesianConfig::laplace().prior_scale(scale));
            StatisticalProgram::declared(DeclaredProgram {
                identifier: None,
                estimator: None,
                inference: &inference,
                bootstrap_replicates: 0,
                sampling_design: None,
                design: DesignShape::not_design_based("test"),
                operators: Vec::new(),
            })
            .ok()
            .map(|program| (program.identity(), program.lowered_layer_digest()))
        });
    }
}
