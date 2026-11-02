//! Carrier for a selected — and optionally fully configured — estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::{
    AipwAte, CausalForest, DistanceMatching, DmlAte, DrLearner, FrontDoorTwoStage,
    GlmAdjustmentAte, LinearAdjustmentAte, PropensityMatching, PropensityStratification,
    PropensityWeighting, TwoStageLeastSquares, WaldIv,
};

use crate::strategy_table::EstimatorId;

/// A selected estimator, optionally fully configured.
///
/// [`Self::Default`] selects by id and lets the study fill bootstrap-replicate /
/// overlap defaults exactly as it does today. Every other variant carries a
/// caller-configured estimator (built via its `with_*` setters) that the study
/// uses verbatim — none of its fields are overwritten by builder-level
/// bootstrap / overlap settings.
///
/// Only estimators actually constructed by
/// the static estimate dispatcher have a configured variant
/// here; estimators driven through other paths (`rd.sharp`, `bayesian.gcomp`,
/// the temporal / conditional / mediation / functional-plug-in families) are
/// selectable only via [`Self::Default`].
///
/// Configured payloads are boxed: several of the estimator config structs
/// carry enough optional cluster / multiway / panel scratch fields that the
/// unboxed enum would trip clippy's `large_enum_variant` lint, and boxing
/// keeps [`Self`] itself cheap to move regardless of which concrete estimator
/// is selected.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum EstimatorSpec {
    /// Id only — the study fills bootstrap-replicate / overlap defaults.
    Default(EstimatorId),
    /// Caller-configured OLS g-computation / linear adjustment ATE.
    LinearAdjustmentAte(Box<LinearAdjustmentAte>),
    /// Caller-configured inverse-probability weighting.
    PropensityWeighting(Box<PropensityWeighting>),
    /// Caller-configured propensity-score matching.
    PropensityMatching(Box<PropensityMatching>),
    /// Caller-configured propensity stratification (raw means or declared regression adjustment).
    PropensityStratification(Box<PropensityStratification>),
    /// Caller-configured covariate distance matching.
    DistanceMatching(Box<DistanceMatching>),
    /// Caller-configured augmented IPW.
    Aipw(Box<AipwAte>),
    /// Caller-configured GLM (logit) adjustment.
    GlmAdjustment(Box<GlmAdjustmentAte>),
    /// Caller-configured front-door two-stage.
    FrontDoorTwoStage(Box<FrontDoorTwoStage>),
    /// Caller-configured Wald IV.
    IvWald(Box<WaldIv>),
    /// Caller-configured two-stage least squares.
    Iv2Sls(Box<TwoStageLeastSquares>),
    /// Caller-configured cross-fitted DML / AIPW.
    Dml(Box<DmlAte>),
    /// Caller-configured DR-Learner.
    DrLearner(Box<DrLearner>),
    /// Caller-configured honest causal forest.
    CausalForest(Box<CausalForest>),
}

impl EstimatorSpec {
    /// The estimator id this spec selects.
    #[must_use]
    pub fn id(&self) -> EstimatorId {
        match self {
            Self::Default(id) => *id,
            Self::LinearAdjustmentAte(_) => EstimatorId::LinearAdjustmentAte,
            Self::PropensityWeighting(_) => EstimatorId::PropensityWeighting,
            Self::PropensityMatching(_) => EstimatorId::PropensityMatching,
            Self::PropensityStratification(_) => EstimatorId::PropensityStratification,
            Self::DistanceMatching(_) => EstimatorId::DistanceMatching,
            Self::Aipw(_) => EstimatorId::Aipw,
            Self::GlmAdjustment(_) => EstimatorId::GlmAdjustment,
            Self::FrontDoorTwoStage(_) => EstimatorId::FrontDoorTwoStage,
            Self::IvWald(_) => EstimatorId::IvWald,
            Self::Iv2Sls(_) => EstimatorId::Iv2Sls,
            Self::Dml(_) => EstimatorId::Dml,
            Self::DrLearner(_) => EstimatorId::DrLearner,
            Self::CausalForest(_) => EstimatorId::CausalForest,
        }
    }

    /// The inference declarations (standard-error kind, cluster, multiway and panel labels) of a
    /// configured estimator that reads them; `None` for an id-only spec and for the estimators
    /// that publish their own uncertainty without them.
    #[must_use]
    pub fn inference_spec(&self) -> Option<&antecedent_estimate::InferenceSpec> {
        match self {
            Self::LinearAdjustmentAte(cfg) => Some(&cfg.inference),
            Self::PropensityWeighting(cfg) => Some(&cfg.inference),
            Self::PropensityMatching(cfg) => Some(&cfg.inference),
            Self::PropensityStratification(cfg) => Some(&cfg.inference),
            Self::DistanceMatching(cfg) => Some(&cfg.inference),
            Self::Aipw(cfg) => Some(&cfg.inference),
            Self::GlmAdjustment(cfg) => Some(&cfg.inference),
            Self::FrontDoorTwoStage(cfg) => Some(&cfg.inference),
            Self::IvWald(cfg) => Some(&cfg.inference),
            Self::Iv2Sls(cfg) => Some(&cfg.inference),
            Self::Default(_) | Self::Dml(_) | Self::DrLearner(_) | Self::CausalForest(_) => None,
        }
    }

    /// Bind the study-level sampling design to a configured estimator that resamples or splits.
    ///
    /// First the shared admission [`ensure_dependence_honoured`] holds the estimator (an id-only
    /// spec resolved to its default configuration) to the declared dependence. An estimator that
    /// declared no design then takes the study's. An estimator that declared a different design
    /// than the study is a conflict, never a silent override. An id-only spec and an estimator
    /// that neither bootstraps by design nor splits are left as they are.
    ///
    /// # Errors
    ///
    /// `option_not_applicable` when the estimator reads none of the declared dependence (see
    /// [`ensure_dependence_honoured`]); [`crate::CausalError::Conflict`] when the estimator
    /// declared another design.
    pub fn bind_sampling_design(
        &mut self,
        design: antecedent_core::DeclaredSamplingDesign,
    ) -> Result<(), crate::CausalError> {
        use antecedent_estimate::util::SamplingDeclaration;
        ensure_dependence_honoured(self, design)?;
        let slot: &mut SamplingDeclaration = match self {
            Self::LinearAdjustmentAte(cfg) => &mut cfg.inference.sampling_design,
            Self::PropensityWeighting(cfg) => &mut cfg.inference.sampling_design,
            Self::PropensityStratification(cfg) => &mut cfg.inference.sampling_design,
            Self::Aipw(cfg) => &mut cfg.inference.sampling_design,
            Self::GlmAdjustment(cfg) => &mut cfg.inference.sampling_design,
            Self::FrontDoorTwoStage(cfg) => &mut cfg.inference.sampling_design,
            _ => return Ok(()),
        };
        match *slot {
            SamplingDeclaration::Undeclared => {
                *slot = SamplingDeclaration::Declared(design);
                Ok(())
            }
            SamplingDeclaration::Declared(own) if own == design => Ok(()),
            SamplingDeclaration::Declared(_) => Err(crate::CausalError::Conflict {
                what: "sampling_design",
                detail: "the estimator declares a sampling design other than the study's; \
                         declare it once on the study",
            }),
        }
    }

    /// Request a bootstrap interval produced by the complete resampling procedure (waves that
    /// stop on the request's Monte Carlo tolerances, capped by the estimator's replicate count).
    ///
    /// A configured linear adjustment, weighting, stratification, AIPW or GLM estimator takes the
    /// request as its own. An id-only spec of those estimators is built with the study's
    /// `defaults` (replicates, overlap, population registry, sampling design), exactly as the
    /// dispatcher builds it, then takes the request.
    ///
    /// # Errors
    ///
    /// [`crate::CausalError::Conflict`] for an estimator whose bootstrap is not the wave
    /// procedure (matching, IV, front-door, DML, forests), or one that already carries a
    /// different request.
    pub fn with_bootstrap_interval(
        self,
        request: antecedent_estimate::IntervalRequest,
        defaults: BuildDefaults<'_>,
    ) -> Result<Self, crate::CausalError> {
        let refuse = || crate::CausalError::Conflict {
            what: "bootstrap_interval",
            detail: "a requested bootstrap interval is produced by linear adjustment, propensity \
                     weighting, propensity stratification, AIPW or GLM adjustment; the selected \
                     estimator's bootstrap is not the wave procedure",
        };
        let taken = || crate::CausalError::Conflict {
            what: "bootstrap_interval",
            detail: "the estimator already carries a different bootstrap interval request",
        };
        let claim = |slot: &mut Option<antecedent_estimate::IntervalRequest>| match *slot {
            None => {
                *slot = Some(request);
                Ok(())
            }
            Some(own) if own == request => Ok(()),
            Some(_) => Err(taken()),
        };
        let spec = match self {
            Self::Default(id) => defaults.build(id).ok_or_else(refuse)?,
            other => other,
        };
        let mut spec = spec;
        match &mut spec {
            Self::LinearAdjustmentAte(cfg) => claim(&mut cfg.inference.bootstrap_interval)?,
            Self::PropensityWeighting(cfg) => claim(&mut cfg.inference.bootstrap_interval)?,
            Self::PropensityStratification(cfg) => claim(&mut cfg.inference.bootstrap_interval)?,
            Self::Aipw(cfg) => claim(&mut cfg.inference.bootstrap_interval)?,
            Self::GlmAdjustment(cfg) => claim(&mut cfg.inference.bootstrap_interval)?,
            _ => return Err(refuse()),
        }
        Ok(spec)
    }

    /// The requested bootstrap interval a configured estimator carries.
    #[must_use]
    pub fn bootstrap_interval(&self) -> Option<antecedent_estimate::IntervalRequest> {
        match self {
            Self::LinearAdjustmentAte(cfg) => cfg.inference.bootstrap_interval,
            Self::PropensityWeighting(cfg) => cfg.inference.bootstrap_interval,
            Self::PropensityStratification(cfg) => cfg.inference.bootstrap_interval,
            Self::Aipw(cfg) => cfg.inference.bootstrap_interval,
            Self::GlmAdjustment(cfg) => cfg.inference.bootstrap_interval,
            _ => None,
        }
    }

    /// Separate an interval stopped by Monte Carlo tolerances from the fixed-count bootstrap the
    /// calibration gates certify: the coverage of the stop-and-interval procedure is its own
    /// measured property, so the key must not bind the two.
    #[must_use]
    pub fn bootstrap_interval_calibration_label(&self) -> Option<&'static str> {
        self.bootstrap_interval().map(|_| "bootstrap=adaptive_waves")
    }

    /// Whether the caller supplied an explicit configuration (versus [`Self::Default`]).
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !matches!(self, Self::Default(_))
    }

    /// Replicate count a configured estimator carries; `None` for an id-only spec.
    ///
    /// The study reports and executes one replicate budget, so it reads this
    /// rather than keeping a second count beside a configured estimator.
    #[must_use]
    pub fn bootstrap_replicates(&self) -> Option<u32> {
        match self {
            Self::Default(_) | Self::Dml(_) | Self::DrLearner(_) | Self::CausalForest(_) => None,
            Self::LinearAdjustmentAte(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::PropensityWeighting(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::PropensityMatching(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::PropensityStratification(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::DistanceMatching(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::Aipw(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::GlmAdjustment(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::FrontDoorTwoStage(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::IvWald(cfg) => Some(cfg.inference.bootstrap_replicates),
            Self::Iv2Sls(cfg) => Some(cfg.inference.bootstrap_replicates),
        }
    }

    /// Overlap policy a configured propensity-score estimator (weighting,
    /// matching, stratification, distance matching, AIPW) clips and trims
    /// with; `None` for an id-only spec and for estimators without a
    /// propensity model.
    #[must_use]
    pub fn propensity_overlap(&self) -> Option<antecedent_estimate::OverlapPolicy> {
        match self {
            Self::PropensityWeighting(cfg) => Some(cfg.overlap),
            Self::PropensityMatching(cfg) => Some(cfg.overlap),
            Self::PropensityStratification(cfg) => Some(cfg.overlap),
            Self::DistanceMatching(cfg) => Some(cfg.overlap),
            Self::Aipw(cfg) => Some(cfg.overlap),
            Self::Dml(cfg) => Some(cfg.overlap),
            Self::DrLearner(cfg) => Some(cfg.overlap),
            _ => None,
        }
    }

    /// Whether the configured estimator declares the exclusion of treatment rows at neither
    /// declared level. That restricts the target population to the rows at the two declared
    /// levels, which the checked all-observed routes do not cover, so such a spec executes
    /// through the ordinary dispatcher.
    #[must_use]
    pub fn excludes_undeclared_levels(&self) -> bool {
        let policy = match self {
            Self::PropensityWeighting(cfg) => cfg.other_levels,
            Self::PropensityMatching(cfg) => cfg.other_levels,
            Self::PropensityStratification(cfg) => cfg.other_levels,
            Self::Aipw(cfg) => cfg.other_levels,
            Self::Dml(cfg) => cfg.other_levels,
            _ => return false,
        };
        policy != antecedent_estimate::OtherLevels::Refuse
    }

    /// The suffix declared sampling weights add to the calibration key's functional label, so
    /// an interval measured on unweighted rows never binds to a weighted-population estimate
    /// (the standard error, the estimand and the effective sample size all differ); `None`
    /// when the configured estimator reads no weights.
    #[must_use]
    pub fn sampling_weights_calibration_label(&self) -> Option<&'static str> {
        let weighted = match self {
            Self::LinearAdjustmentAte(cfg) => cfg.sampling_weights.is_some(),
            Self::PropensityWeighting(cfg) => cfg.sampling_weights.is_some(),
            Self::PropensityMatching(cfg) => cfg.sampling_weights.is_some(),
            Self::PropensityStratification(cfg) => cfg.sampling_weights.is_some(),
            Self::Aipw(cfg) => cfg.sampling_weights.is_some(),
            Self::DistanceMatching(cfg) => cfg.sampling_weights.is_some(),
            Self::GlmAdjustment(cfg) => cfg.sampling_weights.is_some(),
            Self::Dml(cfg) => cfg.sampling_weights.is_some(),
            Self::DrLearner(cfg) => cfg.sampling_weights.is_some(),
            _ => false,
        };
        weighted.then_some(SAMPLING_WEIGHTS_LABEL)
    }

    /// The suffix a declared non-default AIPW propensity nuisance (a ridge or lasso penalty,
    /// or a GLM-to-penalized fallback) adds to the calibration key's functional label, so an
    /// interval measured under one nuisance never binds to another; `None` otherwise.
    #[must_use]
    pub fn propensity_calibration_label(&self) -> Option<String> {
        match self {
            Self::Aipw(cfg) => cfg.propensity.calibration_label(),
            Self::PropensityWeighting(cfg) if cfg.covariate_balance => {
                Some("weighting=balanced_linear_hc3".into())
            }
            _ => None,
        }
    }

    /// Distinguish declared regression-standardized strata from raw stratum means.
    #[must_use]
    pub fn stratification_calibration_label(&self) -> Option<&'static str> {
        match self {
            Self::PropensityStratification(cfg) if cfg.regression_adjustment => {
                Some("stratification=regression_linear_hc3")
            }
            _ => None,
        }
    }

    /// Separate population matching variance from the raw constant-effect approximation.
    #[must_use]
    pub fn matching_calibration_label(&self) -> Option<&'static str> {
        let declared = match self {
            Self::PropensityMatching(cfg) => cfg.population_variance,
            Self::DistanceMatching(cfg) => cfg.population_variance,
            _ => false,
        };
        declared.then_some("matching=population_linear")
    }
}

/// How a resolved estimator reads a declared clustered-panel or dependent-time-series design.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DependenceRead {
    /// Through its inference declarations (a cluster or Newey-West error, or a resample under
    /// the declared design); the study-level check that the configuration really does so is
    /// `require_declared_dependence_read` at build.
    Inference,
    /// Through a whole-cluster cross-fit over declared independence units (cross-fitted AIPW
    /// scores summed by unit, with the cluster sandwich). A clustered panel only; the units
    /// are `declared` or the rows stand alone.
    Units { declared: bool },
    /// Not at all: matching and instrumental fits.
    MatchingOrInstrument,
    /// Not at all: the front-door functional plug-in, which carries no unit labels.
    FrontDoorFunctional,
}

impl EstimatorSpec {
    /// The estimator an id-only spec stands for, with its default configuration; `None` for an
    /// id with no configured variant.
    fn default_configuration(id: EstimatorId) -> Option<Self> {
        Some(match id {
            EstimatorId::PropensityMatching => PropensityMatching::new().into(),
            EstimatorId::DistanceMatching => DistanceMatching::new().into(),
            EstimatorId::IvWald => WaldIv::new().into(),
            EstimatorId::Iv2Sls => TwoStageLeastSquares::new().into(),
            EstimatorId::Dml => DmlAte::new().into(),
            EstimatorId::DrLearner => DrLearner::new().into(),
            EstimatorId::CausalForest => CausalForest::new().into(),
            _ => return None,
        })
    }

    /// How this estimator, resolved from its id when id-only, reads a declared dependent design.
    fn dependence_read(&self) -> DependenceRead {
        match self {
            Self::Default(EstimatorId::FrontDoorFunctional) => DependenceRead::FrontDoorFunctional,
            Self::Default(id) => Self::default_configuration(*id)
                .map_or(DependenceRead::Inference, |resolved| resolved.dependence_read()),
            Self::PropensityMatching(_)
            | Self::DistanceMatching(_)
            | Self::IvWald(_)
            | Self::Iv2Sls(_) => DependenceRead::MatchingOrInstrument,
            Self::Dml(cfg) => DependenceRead::Units { declared: cfg.units.spec.is_some() },
            Self::DrLearner(cfg) => DependenceRead::Units { declared: cfg.units.spec.is_some() },
            Self::CausalForest(cfg) => DependenceRead::Units { declared: cfg.units.spec.is_some() },
            _ => DependenceRead::Inference,
        }
    }

    /// Whether the estimator reads a declared clustered panel through its own independence
    /// units, so no standard-error kind or bootstrap setting has to carry the declaration.
    #[must_use]
    pub(crate) fn reads_dependence_through_units(&self) -> bool {
        self.dependence_read() == DependenceRead::Units { declared: true }
    }
}

/// The one admission of an estimator under a declared sampling design: the estimator must use
/// the declared dependence or the study is refused.
///
/// An id-only spec is resolved to the estimator it stands for first, so the id and the same
/// estimator configured by hand are held to the same rule. Independent subject trajectories
/// declare no dependence to read. A clustered panel or dependent time series is read by an
/// estimator that carries its inference declarations (checked against the configuration by the
/// study), and by a cross-fitted estimator (double ML, DR-learner, causal forest) only through a
/// declared whole-cluster independence unit and only for a panel. Matching, instrumental fits and
/// the front-door functional read neither, and the cross-fit of a series reads no serial lag, so
/// each is refused rather than answered with an error that treats the rows as independent.
///
/// Every dispatch that builds an estimator under a declared design calls this.
///
/// # Errors
///
/// `option_not_applicable` naming why the estimator reads none of the declared dependence and a
/// next attempt.
pub(crate) fn ensure_dependence_honoured(
    spec: &EstimatorSpec,
    design: antecedent_core::DeclaredSamplingDesign,
) -> Result<(), crate::CausalError> {
    use antecedent_core::DeclaredSamplingDesign as Design;
    if design == Design::IndependentSubjectTrajectories {
        return Ok(());
    }
    let series = design == Design::DependentTimeSeries;
    match (spec.dependence_read(), series) {
        (DependenceRead::Inference, _) | (DependenceRead::Units { declared: true }, false) => {
            Ok(())
        }
        (DependenceRead::MatchingOrInstrument, _) => Err(crate::unsupported_reason!(
            "option_not_applicable",
            "this estimator reads no cluster or serial dependence: matching has no valid \
             cluster or HAC error (donors belong to other clusters and times than the \
             matched unit) and instrumental fits publish heteroskedasticity-robust errors \
             only, so a clustered-panel or dependent-time-series design would be silently \
             ignored. Next attempt: declare independent subject trajectories, or use \
             LinearAdjustmentAte, AIPW or propensity weighting with a cluster or Newey-West \
             standard error"
        )),
        (DependenceRead::FrontDoorFunctional, _) => Err(crate::unsupported_reason!(
            "option_not_applicable",
            "this estimator reads no cluster or serial dependence: the front-door functional \
             publishes an influence-function error and resamples single rows, and carries no \
             cluster label or lag, so a clustered-panel or dependent-time-series design would \
             be silently ignored and its error too small. Next attempt: declare independent \
             subject trajectories if the rows are independent, or use FrontDoorTwoStage with \
             cluster labels for a clustered panel"
        )),
        (DependenceRead::Units { .. }, true) => Err(crate::unsupported_reason!(
            "option_not_applicable",
            "this estimator reads no serial dependence: a cross-fitted estimator folds rows \
             (or whole clusters) at random and sums its scores as independent terms, so a \
             declared dependent-time-series design would be silently ignored and its error too \
             small. Next attempt: declare independent subject trajectories if the rows are \
             independent, or use LinearAdjustmentAte, AIPW or propensity weighting with a \
             Newey-West standard error"
        )),
        (DependenceRead::Units { declared: false }, false) => Err(crate::unsupported_reason!(
            "option_not_applicable",
            "this estimator reads no cluster dependence without a declared independence unit: \
             a cross-fitted estimator folds and sandwiches single rows as independent, so a \
             clustered-panel design would be silently ignored and its error too small. Next \
             attempt: declare the units with with_cluster_dml and the cluster labels \
             (with_cluster_ids), declare independent subject trajectories if the rows are \
             independent, or use LinearAdjustmentAte, AIPW or propensity weighting with a \
             cluster standard error"
        )),
    }
}

/// What a study fills into an estimator built from an id alone.
#[derive(Clone, Copy)]
pub struct BuildDefaults<'a> {
    /// Bootstrap replicates (the replicate cap of a requested interval).
    pub bootstrap_replicates: u32,
    /// Overlap policy, when the study set one.
    pub overlap_policy: Option<antecedent_estimate::OverlapPolicy>,
    /// Population registry, when the study set one.
    pub population_registry: Option<&'a antecedent_core::PopulationRegistry>,
    /// Sampling design the study declared.
    pub sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
}

impl BuildDefaults<'_> {
    /// The configured estimator the dispatcher would build for `id`, for the estimators whose
    /// bootstrap is the wave procedure; `None` for every other id.
    fn build(self, id: EstimatorId) -> Option<EstimatorSpec> {
        let sampling = declared_sampling(self.sampling_design);
        Some(match id {
            EstimatorId::LinearAdjustmentAte => {
                let mut est = LinearAdjustmentAte::new();
                est.inference.sampling_design = sampling;
                est.inference.bootstrap_replicates = self.bootstrap_replicates;
                est.overlap = antecedent_estimate::OverlapPolicy::ExplicitOverride;
                EstimatorSpec::LinearAdjustmentAte(Box::new(est))
            }
            EstimatorId::PropensityWeighting => {
                let mut est = PropensityWeighting::new();
                est.inference.sampling_design = sampling;
                est.inference.bootstrap_replicates = self.bootstrap_replicates;
                if let Some(policy) = self.overlap_policy {
                    est.overlap = policy;
                }
                est.population_registry = self.population_registry.cloned();
                EstimatorSpec::PropensityWeighting(Box::new(est))
            }
            EstimatorId::PropensityStratification => {
                let mut est = PropensityStratification::new();
                est.inference.sampling_design = sampling;
                est.inference.bootstrap_replicates = self.bootstrap_replicates;
                if let Some(policy) = self.overlap_policy {
                    est.overlap = policy;
                }
                est.population_registry = self.population_registry.cloned();
                EstimatorSpec::PropensityStratification(Box::new(est))
            }
            EstimatorId::Aipw => {
                let mut est = AipwAte::new();
                est.inference.sampling_design = sampling;
                est.inference.bootstrap_replicates = self.bootstrap_replicates;
                if let Some(policy) = self.overlap_policy {
                    est.overlap = policy;
                }
                est.population_registry = self.population_registry.cloned();
                EstimatorSpec::Aipw(Box::new(est))
            }
            EstimatorId::GlmAdjustment => {
                let mut est = GlmAdjustmentAte::new();
                est.inference.sampling_design = sampling;
                est.inference.bootstrap_replicates = self.bootstrap_replicates;
                EstimatorSpec::GlmAdjustment(Box::new(est))
            }
            _ => return None,
        })
    }
}

/// The sampling declaration an estimator built by default takes from the study's design.
pub(crate) fn declared_sampling(
    design: Option<antecedent_core::DeclaredSamplingDesign>,
) -> antecedent_estimate::util::SamplingDeclaration {
    design.map_or(
        antecedent_estimate::util::SamplingDeclaration::Undeclared,
        antecedent_estimate::util::SamplingDeclaration::Declared,
    )
}

/// The functional-label suffix of a weighted-population estimate.
pub const SAMPLING_WEIGHTS_LABEL: &str = "weights=sampling";

impl From<EstimatorId> for EstimatorSpec {
    fn from(id: EstimatorId) -> Self {
        Self::Default(id)
    }
}

/// `From<Cfg> for EstimatorSpec` for each configured estimator, boxing it into its variant.
macro_rules! spec_from_config {
    ($($cfg:ident => $variant:ident),+ $(,)?) => {
        $(
            impl From<$cfg> for EstimatorSpec {
                fn from(cfg: $cfg) -> Self {
                    Self::$variant(Box::new(cfg))
                }
            }
        )+
    };
}

spec_from_config! {
    LinearAdjustmentAte => LinearAdjustmentAte,
    PropensityWeighting => PropensityWeighting,
    PropensityMatching => PropensityMatching,
    PropensityStratification => PropensityStratification,
    DistanceMatching => DistanceMatching,
    AipwAte => Aipw,
    GlmAdjustmentAte => GlmAdjustment,
    FrontDoorTwoStage => FrontDoorTwoStage,
    WaldIv => IvWald,
    TwoStageLeastSquares => Iv2Sls,
    DmlAte => Dml,
    DrLearner => DrLearner,
    CausalForest => CausalForest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_spec_carries_id_only_and_is_unconfigured() {
        let spec: EstimatorSpec = EstimatorId::Aipw.into();
        assert_eq!(spec.id(), EstimatorId::Aipw);
        assert!(!spec.is_configured());
    }

    #[test]
    fn configured_spec_carries_its_id_and_is_configured() {
        let spec: EstimatorSpec = LinearAdjustmentAte::new().with_bootstrap_replicates(500).into();
        assert_eq!(spec.id(), EstimatorId::LinearAdjustmentAte);
        assert!(spec.is_configured());
    }

    #[test]
    fn configured_spec_round_trips_its_fields() {
        let spec: EstimatorSpec = LinearAdjustmentAte::new().with_bootstrap_replicates(500).into();
        let EstimatorSpec::LinearAdjustmentAte(cfg) = &spec else {
            panic!("expected EstimatorSpec::LinearAdjustmentAte");
        };
        assert_eq!(cfg.inference.bootstrap_replicates, 500);
    }

    #[test]
    fn matching_and_iv_default_ids_refuse_the_same_dependent_designs_as_configurations() {
        use antecedent_core::DeclaredSamplingDesign as Design;

        for (id, configured) in [
            (EstimatorId::PropensityMatching, PropensityMatching::new().into()),
            (EstimatorId::DistanceMatching, DistanceMatching::new().into()),
            (EstimatorId::IvWald, WaldIv::new().into()),
            (EstimatorId::Iv2Sls, TwoStageLeastSquares::new().into()),
        ] {
            let configured: EstimatorSpec = configured;
            for design in [Design::ClusteredPanel, Design::DependentTimeSeries] {
                for mut spec in [EstimatorSpec::Default(id), configured.clone()] {
                    let error = spec.bind_sampling_design(design).unwrap_err();
                    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{id:?}");
                    assert!(error.to_string().contains("this estimator reads no cluster"));
                }
            }
            EstimatorSpec::Default(id)
                .bind_sampling_design(Design::IndependentSubjectTrajectories)
                .unwrap();
        }
    }

    #[test]
    fn the_shared_admission_resolves_ids_and_reads_only_declared_units() {
        use antecedent_core::DeclaredSamplingDesign as Design;
        use antecedent_estimate::ClusterDml;

        let units = ClusterDml::new(20).unwrap();
        // An id and the estimator it resolves to are held to the same rule: with no declared
        // unit a cross-fit reads neither a clustered panel nor a series.
        for (id, configured) in [
            (EstimatorId::Dml, DmlAte::new().into()),
            (EstimatorId::DrLearner, DrLearner::new().into()),
            (EstimatorId::CausalForest, CausalForest::new().into()),
        ] {
            let configured: EstimatorSpec = configured;
            for design in [Design::ClusteredPanel, Design::DependentTimeSeries] {
                for spec in [EstimatorSpec::Default(id), configured.clone()] {
                    let error = ensure_dependence_honoured(&spec, design).unwrap_err();
                    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{id:?}");
                    assert!(error.to_string().contains("Next attempt"), "{error}");
                }
            }
            ensure_dependence_honoured(
                &EstimatorSpec::Default(id),
                Design::IndependentSubjectTrajectories,
            )
            .unwrap();
        }
        // A declared independence unit is read for a clustered panel and never for a series.
        let declared: [EstimatorSpec; 3] = [
            DmlAte::new().with_cluster_dml(units).into(),
            DrLearner::new().with_cluster_dml(units).into(),
            CausalForest::new().with_cluster_dml(units).into(),
        ];
        for spec in &declared {
            assert!(spec.reads_dependence_through_units());
            ensure_dependence_honoured(spec, Design::ClusteredPanel).unwrap();
            let error = ensure_dependence_honoured(spec, Design::DependentTimeSeries).unwrap_err();
            assert_eq!(error.reason_code(), Some("option_not_applicable"));
        }
        // The front-door functional carries no labels at all; it is selectable by id only.
        let functional = EstimatorSpec::Default(EstimatorId::FrontDoorFunctional);
        for design in [Design::ClusteredPanel, Design::DependentTimeSeries] {
            let error = ensure_dependence_honoured(&functional, design).unwrap_err();
            assert_eq!(error.reason_code(), Some("option_not_applicable"));
            assert!(error.to_string().contains("front-door functional"), "{error}");
        }
        ensure_dependence_honoured(&functional, Design::IndependentSubjectTrajectories).unwrap();
        // An estimator that carries its inference declarations is checked against them by the
        // study, not refused here.
        for id in [EstimatorId::Aipw, EstimatorId::LinearAdjustmentAte, EstimatorId::GlmAdjustment]
        {
            let spec = EstimatorSpec::Default(id);
            assert!(!spec.reads_dependence_through_units());
            ensure_dependence_honoured(&spec, Design::ClusteredPanel).unwrap();
        }
    }
}
