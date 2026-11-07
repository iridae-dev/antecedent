// SPDX-License-Identifier: MIT OR Apache-2.0
//! Experimental recovery declarations, conversion and execution.

use super::{
    A, BTreeSet, BootstrapSpec, CoefficientCovarianceEstimator, CoefficientPrior,
    CompiledSystemPlan, ConstraintWeighting, CyclicAction, CyclicArtifactError, CyclicFamily,
    CyclicVariables, CyclicWorldRequest, CyclicWorldWire, DEFAULT_FIT_BYTES, DependencePremise,
    Deserialize, DirectCoefficient, DisturbanceDivisor, DisturbancePrior, DrawStatus,
    ExecutionContext, ExperimentRegime, ExperimentRows, ExperimentalFitError,
    ExperimentalFitOptions, JointRecoveryDraw, MAX_DRAWS, MAX_EXPERIMENT_VALUES,
    MAX_RECOVERED_NODES, PopulationId, PosteriorSpec, RecoveredCoefficientUncertainty,
    RecoveredDirectionWire, RecoveredDrawOrigin, RecoveredEstimate, RecoveredSupportWire,
    RecoveredUnresolvedWire, RecoveredWorldAnswer, RecoveryDrawInput, RecoveryPlanParts,
    RecoveryPrior, RegimeContract, RegimeIntervention, RegimeSample, RetainedRecoveryPlan,
    Serialize, SourceIdentity, SystemStability, UnitColumn, UnitMask, UnresolvedDirection,
    UnresolvedReason, Value, VariableId, asymptotic_normal_recovery,
    bootstrap_experimental_recovery, consume_retained_recovery_plan, fit_experimental_recovery,
    invalid, posterior_recovery, refused, resolve_cyclic_world, variables_of, world_requests,
};

/// Declared sampling design of the experimental units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingDesignWire {
    /// Independent subjects; a subject's trajectory is one unit.
    IndependentSubjectTrajectories,
    /// Rows nested in clusters; the cluster is the unit.
    ClusteredPanel,
    /// One dependent series.
    DependentTimeSeries,
}

impl SamplingDesignWire {
    pub(super) const fn engine(self) -> antecedent_core::DeclaredSamplingDesign {
        use antecedent_core::DeclaredSamplingDesign as D;
        match self {
            Self::IndependentSubjectTrajectories => D::IndependentSubjectTrajectories,
            Self::ClusteredPanel => D::ClusteredPanel,
            Self::DependentTimeSeries => D::DependentTimeSeries,
        }
    }
    const fn of(design: antecedent_core::DeclaredSamplingDesign) -> Self {
        use antecedent_core::DeclaredSamplingDesign as D;
        match design {
            D::IndependentSubjectTrajectories => Self::IndependentSubjectTrajectories,
            D::ClusteredPanel => Self::ClusteredPanel,
            D::DependentTimeSeries => Self::DependentTimeSeries,
        }
    }
    /// Wire spelling.
    const fn label(self) -> &'static str {
        match self {
            Self::IndependentSubjectTrajectories => "independent_subject_trajectories",
            Self::ClusteredPanel => "clustered_panel",
            Self::DependentTimeSeries => "dependent_time_series",
        }
    }
}

/// Resampling unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResamplingRuleWire {
    /// Whole subjects.
    RowSubjectIid,
    /// Whole clusters.
    WholeCluster,
    /// Contiguous blocks.
    ContiguousBlock,
    /// Individual rows. No declared design allows it; kept so the refusal is typed.
    RowIid,
}

impl ResamplingRuleWire {
    pub(super) const fn engine(self) -> antecedent_core::ResamplingRule {
        use antecedent_core::ResamplingRule as R;
        match self {
            Self::RowSubjectIid => R::RowSubjectIid,
            Self::WholeCluster => R::WholeCluster,
            Self::ContiguousBlock => R::ContiguousBlock,
            Self::RowIid => R::RowIid,
        }
    }
    const fn of(rule: antecedent_core::ResamplingRule) -> Self {
        use antecedent_core::ResamplingRule as R;
        match rule {
            R::RowSubjectIid => Self::RowSubjectIid,
            R::WholeCluster => Self::WholeCluster,
            R::ContiguousBlock => Self::ContiguousBlock,
            R::RowIid => Self::RowIid,
        }
    }
}

/// Coefficient prior of a recovered-law posterior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CoefficientPriorWire {
    /// Improper flat prior on the identified directions.
    Flat,
    /// Independent normal prior, projected onto the identified subspace only.
    Normal {
        /// Prior mean of every coefficient.
        mean: f64,
        /// Prior variance of every coefficient.
        variance: f64,
    },
}

/// Disturbance prior of a recovered-law posterior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DisturbancePriorWire {
    /// Known disturbance variances, in node order.
    Known {
        /// Variance of each node's disturbance.
        variances: Vec<f64>,
    },
    /// Scaled inverse chi-square on each diagonal disturbance variance.
    InverseWishart {
        /// Prior degrees of freedom.
        degrees_of_freedom: f64,
        /// Prior scale per node.
        scale: Vec<f64>,
    },
}

/// Prior of a recovered-law posterior. The intercept prior is flat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPriorWire {
    /// Coefficient prior.
    pub coefficients: CoefficientPriorWire,
    /// Disturbance prior.
    pub disturbance: DisturbancePriorWire,
}

impl RecoveryPriorWire {
    fn engine(&self) -> RecoveryPrior {
        RecoveryPrior {
            coefficients: match self.coefficients {
                CoefficientPriorWire::Flat => CoefficientPrior::Flat,
                CoefficientPriorWire::Normal { mean, variance } => {
                    CoefficientPrior::Normal { mean, variance }
                }
            },
            disturbance: match &self.disturbance {
                DisturbancePriorWire::Known { variances } => {
                    DisturbancePrior::Known { variances: variances.clone() }
                }
                DisturbancePriorWire::InverseWishart { degrees_of_freedom, scale } => {
                    DisturbancePrior::InverseWishart {
                        degrees_of_freedom: *degrees_of_freedom,
                        scale: scale.clone(),
                    }
                }
            },
        }
    }
    fn of(prior: &RecoveryPrior) -> Self {
        Self {
            coefficients: match prior.coefficients {
                CoefficientPrior::Flat => CoefficientPriorWire::Flat,
                CoefficientPrior::Normal { mean, variance } => {
                    CoefficientPriorWire::Normal { mean, variance }
                }
            },
            disturbance: match &prior.disturbance {
                DisturbancePrior::Known { variances } => {
                    DisturbancePriorWire::Known { variances: variances.clone() }
                }
                DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
                    DisturbancePriorWire::InverseWishart {
                        degrees_of_freedom: *degrees_of_freedom,
                        scale: scale.clone(),
                    }
                }
            },
        }
    }
}

/// How the draws of a recovered law were produced.
///
/// Supplied draws declare no meaning: an evidence projection keeps them on the
/// interventional port only. A sampling law is never read as a posterior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredDrawOriginWire {
    /// The caller supplied the draws.
    #[default]
    Supplied,
    /// Joint asymptotic normal law of the experimental fit. A sampling law.
    AsymptoticNormal {
        /// Number of draws.
        draws: usize,
        /// Seed.
        seed: u64,
    },
    /// Whole-recovery bootstrap of the experimental units. A sampling law.
    Bootstrap {
        /// Replicates, failed ones included.
        replicates: usize,
        /// Seed.
        seed: u64,
        /// Declared sampling design.
        design: SamplingDesignWire,
        /// Resampling rule the design allowed.
        rule: ResamplingRuleWire,
    },
    /// Gaussian posterior over the identified coefficient subspace.
    Posterior {
        /// Prior.
        prior: RecoveryPriorWire,
        /// Retained draws.
        draws: usize,
        /// Discarded sweeps of each chain.
        burn_in: usize,
        /// Seed.
        seed: u64,
    },
}

impl RecoveredDrawOriginWire {
    /// Whether the draws were supplied without a declared meaning.
    #[must_use]
    pub const fn is_supplied(&self) -> bool {
        matches!(self, Self::Supplied)
    }

    /// Evidence meaning of the draws: `sampling_law`, `parameter_posterior`, or
    /// `None` for supplied draws.
    #[must_use]
    pub const fn law_meaning(&self) -> Option<&'static str> {
        match self {
            Self::Supplied => None,
            Self::AsymptoticNormal { .. } | Self::Bootstrap { .. } => Some("sampling_law"),
            Self::Posterior { .. } => Some("parameter_posterior"),
        }
    }

    fn of(origin: &RecoveredDrawOrigin) -> Self {
        match origin {
            RecoveredDrawOrigin::Supplied => Self::Supplied,
            RecoveredDrawOrigin::AsymptoticNormal { draws, seed } => {
                Self::AsymptoticNormal { draws: *draws, seed: *seed }
            }
            RecoveredDrawOrigin::Bootstrap { replicates, seed, design, rule } => Self::Bootstrap {
                replicates: *replicates,
                seed: *seed,
                design: SamplingDesignWire::of(*design),
                rule: ResamplingRuleWire::of(*rule),
            },
            RecoveredDrawOrigin::Posterior { prior, draws, burn_in, seed } => Self::Posterior {
                prior: RecoveryPriorWire::of(prior),
                draws: *draws,
                burn_in: *burn_in,
                seed: *seed,
            },
        }
    }
}

/// One direct coefficient of a generated draw. An absent value is not finite.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawCoefficient {
    /// Target equation id.
    pub into: u32,
    /// Source variable id.
    pub from: u32,
    /// Value; `None` when the generator left it non-finite.
    pub value: Option<f64>,
}

/// One generated draw of a recovered law. `None` entries mark unresolved rows,
/// whose values the generator did not complete.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawDraw {
    /// Direct effects named by this draw.
    pub coefficients: Vec<RecoveredLawCoefficient>,
    /// Intercept in node order.
    pub intercept: Vec<Option<f64>>,
    /// Disturbance mean in node order.
    pub noise_mean: Vec<Option<f64>>,
    /// Row-major disturbance covariance.
    pub noise_covariance: Vec<Option<f64>>,
    /// Unresolved directions carried by the draw.
    pub unresolved: Vec<RecoveredDirectionWire>,
    /// Weight; the weights of the law sum to one.
    pub weight: f64,
    /// Inadmissible regardless of its numbers (a failed replicate, for one).
    pub marked_inadmissible: bool,
}

/// Status of one generated draw.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveredDrawStatusWire {
    /// Every row resolved.
    Complete {
        /// Invertibility of `I − U K` on every declared experiment, when computed.
        /// Not physical stability.
        weakly_stable: Option<bool>,
        /// Whether `ρ(|K|) < 1` was established, when computed.
        family_admitted: Option<bool>,
    },
    /// The draw carries unresolved directions; none was completed.
    Unresolved {
        /// Number of unresolved directions.
        directions: usize,
    },
    /// The replicate's recovery failed; its weight stays as failed mass.
    Failed {
        /// Why.
        error: String,
    },
    /// Marked inadmissible by the caller.
    Marked,
}

/// The recovered law an experimental fit produced, without its worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredLawProduct {
    /// Retained unit ids, strictly increasing; positions index the draws.
    pub nodes: Vec<u32>,
    /// How the draws were produced.
    pub origin: RecoveredDrawOriginWire,
    /// Draws in law order, failed and inadmissible ones included.
    pub draws: Vec<RecoveredLawDraw>,
    /// One status per draw.
    pub draw_status: Vec<RecoveredDrawStatusWire>,
    /// Mass of replicates whose recovery failed.
    pub failed_mass: f64,
}

/// One hard-clamp world of a recovered law, keyed by its scientific world id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredKeyedWorld {
    /// Scientific world id.
    pub world: u32,
    /// Draw-aligned evaluation; failed and inadmissible draws stay at their weights.
    pub answer: RecoveredWorldAnswer,
}

/// A law recovered from clamp experiments and its intervention worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalRecoveryReport {
    /// The recovered law.
    #[serde(with = "shared_recovered_law")]
    pub law: std::sync::Arc<RecoveredLawProduct>,
    /// Evidence meaning of the draws.
    pub law_meaning: Option<String>,
    /// Worlds in request order.
    pub worlds: Vec<RecoveredKeyedWorld>,
}

mod shared_recovered_law {
    use super::RecoveredLawProduct;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    pub fn serialize<S: serde::Serializer>(
        law: &Arc<RecoveredLawProduct>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        law.as_ref().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Arc<RecoveredLawProduct>, D::Error> {
        RecoveredLawProduct::deserialize(deserializer).map(Arc::new)
    }
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn direction_wire(direction: &UnresolvedDirection) -> RecoveredDirectionWire {
    RecoveredDirectionWire {
        support: direction
            .support
            .iter()
            .map(|item| RecoveredSupportWire { into: item.into.raw(), from: item.from.raw() })
            .collect(),
        reason: match &direction.reason {
            UnresolvedReason::IncompletePairCoverage { target, missing_clamps } => {
                RecoveredUnresolvedWire::IncompletePairCoverage {
                    target: target.raw(),
                    missing_clamps: missing_clamps.iter().map(|id| id.raw()).collect(),
                }
            }
            UnresolvedReason::DeficientConstraintRank { target, column_rank, width } => {
                RecoveredUnresolvedWire::DeficientConstraintRank {
                    target: target.raw(),
                    column_rank: *column_rank,
                    width: *width,
                }
            }
        },
    }
}

fn law_draw(draw: &JointRecoveryDraw) -> RecoveredLawDraw {
    RecoveredLawDraw {
        coefficients: draw
            .coefficients()
            .iter()
            .map(|(coefficient, value)| RecoveredLawCoefficient {
                into: coefficient.into.raw(),
                from: coefficient.from.raw(),
                value: finite(*value),
            })
            .collect(),
        intercept: draw.intercept().iter().copied().map(finite).collect(),
        noise_mean: draw.noise_mean().iter().copied().map(finite).collect(),
        noise_covariance: draw.noise_covariance().iter().copied().map(finite).collect(),
        unresolved: draw.unresolved().iter().map(direction_wire).collect(),
        weight: draw.weight(),
        marked_inadmissible: draw.marked_inadmissible(),
    }
}

fn draw_input(draw: &RecoveredLawDraw) -> RecoveryDrawInput {
    let restore = |values: &[Option<f64>]| -> Vec<f64> {
        values.iter().map(|value| value.unwrap_or(f64::NAN)).collect()
    };
    RecoveryDrawInput {
        coefficients: draw
            .coefficients
            .iter()
            .map(|item| {
                (
                    DirectCoefficient {
                        into: VariableId::from_raw(item.into),
                        from: VariableId::from_raw(item.from),
                    },
                    item.value.unwrap_or(f64::NAN),
                )
            })
            .collect(),
        intercept: restore(&draw.intercept),
        noise_mean: restore(&draw.noise_mean),
        noise_covariance: restore(&draw.noise_covariance),
        unresolved: draw.unresolved.iter().map(super::super::compiled::direction_input).collect(),
        weight: draw.weight,
        marked_inadmissible: draw.marked_inadmissible,
    }
}

fn status_wire(status: &DrawStatus) -> RecoveredDrawStatusWire {
    match status {
        DrawStatus::Complete { stability } => RecoveredDrawStatusWire::Complete {
            weakly_stable: stability.as_ref().map(SystemStability::weakly_stable),
            family_admitted: stability.as_ref().map(|item| item.family_admitted),
        },
        DrawStatus::Unresolved { directions } => {
            RecoveredDrawStatusWire::Unresolved { directions: *directions }
        }
        DrawStatus::Failed { error } => {
            RecoveredDrawStatusWire::Failed { error: error.to_string() }
        }
        DrawStatus::Marked => RecoveredDrawStatusWire::Marked,
    }
}

impl RecoveredLawProduct {
    fn of(estimate: &RecoveredEstimate) -> Self {
        let law = estimate.law();
        Self {
            nodes: law.nodes().iter().map(|node| node.raw()).collect(),
            origin: RecoveredDrawOriginWire::of(estimate.origin()),
            draws: law.draws().iter().map(law_draw).collect(),
            draw_status: estimate.draw_status().iter().map(status_wire).collect(),
            failed_mass: estimate.failed_mass(),
        }
    }

    fn law(&self) -> Result<RecoveredCoefficientUncertainty, CyclicArtifactError> {
        if self.draws.is_empty() || self.nodes.len() > MAX_RECOVERED_NODES {
            return Err(invalid("a recovered law needs a draw and at most 64 nodes"));
        }
        let ids: Vec<VariableId> = self.nodes.iter().copied().map(VariableId::from_raw).collect();
        RecoveredCoefficientUncertainty::retain(&ids, self.draws.iter().map(draw_input).collect())
            .map_err(invalid)
    }
}

// ---------------------------------------------------------------------------
// Experimental recovery request
// ---------------------------------------------------------------------------

/// One clamp of an experiment regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeClampWire {
    /// Clamped unit id.
    pub unit: u32,
    /// Declared level, or the mean of the randomized clamp draws.
    pub value: f64,
}

/// One declared experiment regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentRegimeWire {
    /// Randomized surgical clamps of this regime; empty is the passive regime.
    pub clamps: Vec<RegimeClampWire>,
    /// Sampling units of the regime.
    pub units: Vec<u32>,
    /// Population the units are drawn from.
    #[serde(default)]
    pub population: u32,
    /// The regime claims to share its disturbance with another regime.
    #[serde(default)]
    pub dependent: bool,
    /// Source identity of the regime's sample.
    pub source: u64,
}

/// One observed column of a regime sample.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnitColumnWire {
    /// Unit id.
    pub unit: u32,
    /// Observed values, one per row, randomized clamp draws included.
    pub values: Vec<f64>,
}

/// Sample of one regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimeSampleWire {
    /// Source the columns were drawn from; must equal the regime's source.
    pub source: u64,
    /// Columns of the retained units.
    pub columns: Vec<UnitColumnWire>,
    /// Sampling-unit label of every row, for whole-subject or whole-cluster resampling.
    #[serde(default)]
    pub sampling_units: Option<Vec<u64>>,
}

/// Row weighting of the constraint least-squares solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintWeightingWire {
    /// Inverse model-based covariance of the rows.
    #[default]
    Efficient,
    /// Ordinary least squares.
    Unweighted,
}

/// Joint coefficient covariance estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoefficientCovarianceWire {
    /// Heteroskedasticity-robust sandwich.
    #[default]
    Sandwich,
    /// Model-based covariance under the declared Gaussian law.
    ModelBased,
    /// Not computed; asymptotic draws are then refused.
    NotComputed,
}

/// Divisor of the pooled residual variance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisturbanceDivisorWire {
    /// Row count (maximum likelihood).
    #[default]
    RowCount,
    /// Row count minus one.
    RowCountMinusOne,
}

/// Options of the experimental fit. The byte budget comes from the execution context.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalFitOptionsWire {
    /// Row weighting.
    #[serde(default)]
    pub weighting: ConstraintWeightingWire,
    /// Coefficient covariance estimator.
    #[serde(default)]
    pub covariance: CoefficientCovarianceWire,
    /// Disturbance divisor.
    #[serde(default)]
    pub divisor: DisturbanceDivisorWire,
    /// Declared sampling design of the experimental rows, when the study declares one.
    ///
    /// An asymptotic-normal or posterior origin treats rows as independent within each
    /// regime, so it is refused under a clustered or time-series design; a bootstrap
    /// must resample under this same design. Absent, the encoding and the request
    /// identity are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_design: Option<SamplingDesignWire>,
}

/// Refuse an uncertainty origin that the declared row design does not allow.
///
/// The refusal carries the registered `estimator_inference_mismatch` reason and names
/// the next attempt: a bootstrap under the declared design.
fn check_recovery_design(
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
) -> Result<(), CyclicArtifactError> {
    let Some(declared) = options.sampling_design else {
        return Ok(());
    };
    let code = antecedent_core::reason_code!("estimator_inference_mismatch");
    let origin = match inference {
        RecoveryInferenceWire::Bootstrap { design, .. } if *design == declared => return Ok(()),
        RecoveryInferenceWire::Bootstrap { design, .. } => {
            return Err(CyclicArtifactError::ScientificRefusal {
                code,
                detail: "cyclic_recovery.bootstrap_sampling_design_mismatch",
                message: format!(
                    "the bootstrap resamples under the `{}` design but the rows declare `{}`",
                    design.label(),
                    declared.label()
                ),
                next_attempt: format!(
                    "use a bootstrap under the declared `{}` design",
                    declared.label()
                ),
            });
        }
        RecoveryInferenceWire::AsymptoticNormal { .. } => "asymptotic_normal",
        RecoveryInferenceWire::Posterior { .. } => "posterior",
    };
    match declared {
        SamplingDesignWire::IndependentSubjectTrajectories => Ok(()),
        SamplingDesignWire::ClusteredPanel | SamplingDesignWire::DependentTimeSeries => {
            Err(CyclicArtifactError::ScientificRefusal {
                code,
                detail: "cyclic_recovery.row_independence_unsupported",
                message: format!(
                    "the `{origin}` origin treats rows as independent within each regime, which the declared `{}` design does not allow",
                    declared.label()
                ),
                next_attempt: format!(
                    "use a bootstrap under the declared `{}` design, which resamples the unit the design names",
                    declared.label()
                ),
            })
        }
    }
}

impl ExperimentalFitOptionsWire {
    fn engine(self, ctx: &ExecutionContext) -> ExperimentalFitOptions {
        ExperimentalFitOptions {
            weighting: match self.weighting {
                ConstraintWeightingWire::Efficient => ConstraintWeighting::Efficient,
                ConstraintWeightingWire::Unweighted => ConstraintWeighting::Unweighted,
            },
            covariance: match self.covariance {
                CoefficientCovarianceWire::Sandwich => CoefficientCovarianceEstimator::Sandwich,
                CoefficientCovarianceWire::ModelBased => CoefficientCovarianceEstimator::ModelBased,
                CoefficientCovarianceWire::NotComputed => {
                    CoefficientCovarianceEstimator::NotComputed
                }
            },
            divisor: match self.divisor {
                DisturbanceDivisorWire::RowCount => DisturbanceDivisor::RowCount,
                DisturbanceDivisorWire::RowCountMinusOne => DisturbanceDivisor::RowCountMinusOne,
            },
            max_bytes: ctx
                .memory_limit_bytes()
                .map_or(DEFAULT_FIT_BYTES, |limit| limit.min(DEFAULT_FIT_BYTES)),
        }
    }
}

/// How the uncertainty of the recovered law is produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum RecoveryInferenceWire {
    /// Joint asymptotic normal law of the fit. A sampling law.
    AsymptoticNormal {
        /// Number of draws.
        draws: usize,
        /// Seed.
        seed: u64,
    },
    /// Whole-recovery bootstrap of the experimental units. A sampling law.
    Bootstrap {
        /// Replicates, failed ones included.
        replicates: usize,
        /// Seed.
        seed: u64,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Resampling rule; the design must allow it.
        rule: ResamplingRuleWire,
        /// Block length of contiguous-block resampling.
        #[serde(default)]
        block_length: Option<usize>,
    },
    /// Gaussian posterior over the identified coefficient subspace.
    Posterior {
        /// Prior.
        prior: RecoveryPriorWire,
        /// Retained draws.
        draws: usize,
        /// Discarded sweeps of each Gibbs chain.
        #[serde(default)]
        burn_in: usize,
        /// Seed.
        seed: u64,
    },
}

/// Retained unit ids of a declared contract: every declared unit minus the mask.
pub(in crate::cyclic_family_artifact) fn retained_units(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
) -> Vec<u32> {
    let dropped: BTreeSet<u32> = dropped.iter().copied().collect();
    regimes
        .iter()
        .flat_map(|regime| regime.units.iter().copied())
        .filter(|unit| !dropped.contains(unit))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn regime_contract(regimes: &[ExperimentRegimeWire], dropped: &[u32]) -> RegimeContract {
    RegimeContract {
        regimes: regimes
            .iter()
            .map(|regime| ExperimentRegime {
                targets: regime
                    .clamps
                    .iter()
                    .map(|clamp| RegimeIntervention {
                        unit: VariableId::from_raw(clamp.unit),
                        value: Value::f64(clamp.value),
                    })
                    .collect(),
                units: regime.units.iter().copied().map(VariableId::from_raw).collect(),
                population: PopulationId::from_raw(regime.population),
                premise: if regime.dependent {
                    DependencePremise::Dependence
                } else {
                    DependencePremise::Independence
                },
                source: SourceIdentity::from_raw(regime.source),
            })
            .collect(),
        mask: UnitMask { dropped: dropped.iter().copied().map(VariableId::from_raw).collect() },
    }
}

fn check_experiment(
    regimes: &[ExperimentRegimeWire],
    samples: &[RegimeSampleWire],
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<(), CyclicArtifactError> {
    if regimes.is_empty() || samples.len() != regimes.len() {
        return Err(invalid("experimental recovery needs one sample per declared regime"));
    }
    let values = samples
        .iter()
        .flat_map(|sample| sample.columns.iter())
        .try_fold(0usize, |sum, column| sum.checked_add(column.values.len()))
        .ok_or_else(|| invalid("experiment value count overflow"))?;
    if values > MAX_EXPERIMENT_VALUES {
        return Err(invalid("experimental recovery value limit"));
    }
    if source_id.is_empty()
        || certifying_evidence.is_empty()
        || certifying_evidence.iter().any(|item| item.is_empty() || item == source_id)
    {
        return Err(invalid(
            "experimental recovery needs a source id and certifying evidence other than the \
             source; next attempt: cite a held-out regime or other evidence that may certify \
             the component",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(in crate::cyclic_family_artifact) fn experimental_recovery_plans(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    // Compilation admits the worlds on the retained units and the uncertainty
    // origin under the declared row design. It does not fit.
    check_experiment(regimes, samples, source_id, certifying_evidence)?;
    check_recovery_design(options, inference)?;
    let units = retained_units(regimes, dropped);
    if units.len() > MAX_RECOVERED_NODES {
        // The design's unresolved directions are quadratic in the retained units; the
        // recovered law admits no more, so refuse before any pass is sized by them.
        return Err(invalid(
            "experimental recovery retains more units than a recovered law admits",
        ));
    }
    let variables = variables_of(&units)?;
    if variables.position(VariableId::from_raw(outcome)).is_none() {
        return Err(invalid(
            "experimental recovery outcome is not a retained unit; next attempt: name an \
             unmasked unit",
        ));
    }
    let requests = world_requests(worlds)?;
    for request in &requests {
        recovered_clamp(&variables, request)?;
    }
    Ok(Vec::new())
}

fn recovered_clamp(
    variables: &CyclicVariables,
    request: &CyclicWorldRequest,
) -> Result<(Vec<usize>, Vec<f64>), CyclicArtifactError> {
    let resolved = resolve_cyclic_world(CyclicFamily::LinearEquilibrium, variables, &[], request)
        .map_err(refused)?;
    let mut nodes = Vec::new();
    let mut values = Vec::new();
    for action in resolved.actions() {
        match action {
            CyclicAction::HardClamp { node, level, .. } => {
                nodes.push(*node);
                values.push(*level);
            }
            other => {
                return Err(CyclicArtifactError::ScientificRefusal {
                    code: antecedent_core::reason_code!("route_not_supported"),
                    detail: "cyclic_recovery.hard_clamp_required",
                    message: format!("world {}: `{}` on variable {}: a recovered law supports hard clamps only",
                        request.world().raw(),other.kind(),other.variable().raw()),
                    next_attempt: "evaluate this action on a separately declared eligible specified model; keep the recovered law and its original uncertainty origin explicit".into(),
                });
            }
        }
    }
    Ok((nodes, values))
}

fn fit_error(error: &ExperimentalFitError) -> CyclicArtifactError {
    CyclicArtifactError::RecoveryFit(error.clone())
}

#[cfg(test)]
std::thread_local! { static RECOVERY_FITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
pub(in crate::cyclic_family_artifact) fn recovery_fit_count() -> usize {
    RECOVERY_FITS.with(std::cell::Cell::get)
}

fn estimate_law(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    ctx: &ExecutionContext,
) -> Result<RecoveredLawProduct, CyclicArtifactError> {
    #[cfg(test)]
    RECOVERY_FITS.with(|count| count.set(count.get() + 1));
    check_recovery_design(options, inference)?;
    let contract = regime_contract(regimes, dropped);
    let columns: Vec<Vec<UnitColumn<'_>>> = samples
        .iter()
        .map(|sample| {
            sample
                .columns
                .iter()
                .map(|column| UnitColumn {
                    unit: VariableId::from_raw(column.unit),
                    values: column.values.as_slice(),
                })
                .collect()
        })
        .collect();
    let regime_samples: Vec<RegimeSample<'_>> = columns
        .iter()
        .zip(samples)
        .map(|(columns, sample)| RegimeSample {
            columns: columns.as_slice(),
            source: SourceIdentity::from_raw(sample.source),
        })
        .collect();
    let rows = ExperimentRows { contract: &contract, samples: &regime_samples };
    let options = options.engine(ctx);
    let estimate = match inference {
        RecoveryInferenceWire::AsymptoticNormal { draws, seed } => {
            if *draws > MAX_DRAWS {
                return Err(invalid("asymptotic draw limit"));
            }
            let fit = fit_experimental_recovery(&rows, &options).map_err(|e| fit_error(&e))?;
            asymptotic_normal_recovery(&fit, *draws, *seed)
        }
        RecoveryInferenceWire::Bootstrap { replicates, seed, design, rule, block_length } => {
            if *replicates > MAX_DRAWS {
                return Err(invalid("bootstrap replicate limit"));
            }
            let labelled = samples.iter().filter(|sample| sample.sampling_units.is_some()).count();
            if labelled != 0 && labelled != samples.len() {
                return Err(invalid(
                    "sampling-unit labels must be given for every regime or for none",
                ));
            }
            let labels: Vec<&[u64]> =
                samples.iter().filter_map(|sample| sample.sampling_units.as_deref()).collect();
            let spec = BootstrapSpec {
                design: design.engine(),
                rule: rule.engine(),
                replicates: *replicates,
                seed: *seed,
                sampling_units: (labelled == samples.len()).then_some(labels.as_slice()),
                block_length: *block_length,
                batch: (*replicates).clamp(1, 64),
            };
            bootstrap_experimental_recovery(&rows, &options, &spec, ctx)
        }
        RecoveryInferenceWire::Posterior { prior, draws, burn_in, seed } => {
            if *draws > MAX_DRAWS || *burn_in > MAX_DRAWS {
                return Err(invalid("posterior draw limit"));
            }
            let fit = fit_experimental_recovery(&rows, &options).map_err(|e| fit_error(&e))?;
            posterior_recovery(
                &fit,
                &PosteriorSpec {
                    prior: prior.engine(),
                    draws: *draws,
                    burn_in: *burn_in,
                    seed: *seed,
                },
            )
        }
    }
    .map_err(|e| fit_error(&e))?;
    Ok(RecoveredLawProduct::of(&estimate))
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(in crate::cyclic_family_artifact) fn execute_experimental_recovery(
    regimes: &[ExperimentRegimeWire],
    dropped: &[u32],
    samples: &[RegimeSampleWire],
    options: ExperimentalFitOptionsWire,
    inference: &RecoveryInferenceWire,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
    frozen: Option<&std::sync::Arc<RecoveredLawProduct>>,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    check_experiment(regimes, samples, source_id, certifying_evidence)?;
    let law = match frozen {
        Some(retained) => retained.clone(),
        None => {
            std::sync::Arc::new(estimate_law(regimes, dropped, samples, options, inference, ctx)?)
        }
    };
    if ctx.cancellation.is_cancelled() {
        return Err(invalid("experimental recovery cancelled"));
    }
    let worlds = evaluate_recovered_worlds(&law, worlds, outcome, source_id, certifying_evidence)?;
    Ok(A::ExperimentalRecovery(ExperimentalRecoveryReport {
        law_meaning: law.origin.law_meaning().map(str::to_owned),
        law,
        worlds,
    }))
}

/// Evaluate new worlds from the exact original law; no recovery or sampling.
pub(in crate::cyclic_family_artifact) fn execute_retained_recovery(
    law: &std::sync::Arc<RecoveredLawProduct>,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    evidence: &[String],
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CyclicArtifactError::Cancelled);
    }
    let answers = evaluate_recovered_worlds(law, worlds, outcome, source_id, evidence)?;
    Ok(A::ExperimentalRecovery(ExperimentalRecoveryReport {
        law: std::sync::Arc::clone(law),
        law_meaning: law.origin.law_meaning().map(str::to_owned),
        worlds: answers,
    }))
}

fn evaluate_recovered_worlds(
    product: &RecoveredLawProduct,
    worlds: &[CyclicWorldWire],
    outcome: u32,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<Vec<RecoveredKeyedWorld>, CyclicArtifactError> {
    // Every world is evaluated from the portable law, so a replay reads the same numbers.
    let law = product.law()?;
    let variables = CyclicVariables::try_new(law.nodes().to_vec()).map_err(refused)?;
    let outcome = variables.position(VariableId::from_raw(outcome)).ok_or_else(|| {
        invalid("experimental recovery outcome is not a retained unit; next attempt: name an unmasked unit")
    })?;
    let requests = world_requests(worlds)?;
    requests
        .iter()
        .map(|request| {
            let (clamp_nodes, clamp_values) = recovered_clamp(&variables, request)?;
            let plan = RetainedRecoveryPlan::try_new(RecoveryPlanParts {
                uncertainty: law.clone(),
                clamp_nodes,
                clamp_values,
                outcome,
                source_id: source_id.to_owned(),
                certifying_evidence: certifying_evidence.to_vec(),
            })
            .map_err(invalid)?;
            let evaluation = consume_retained_recovery_plan(&plan);
            Ok(RecoveredKeyedWorld {
                world: request.world().raw(),
                answer: super::super::compiled::world_answer(&evaluation),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Dynamic-fit inference
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
