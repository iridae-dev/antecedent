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
/// [`crate::strategy_table::estimate_static_effect`] have a configured variant
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
    /// Caller-configured propensity stratification.
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
            Self::LinearAdjustmentAte(cfg) => Some(cfg.bootstrap_replicates),
            Self::PropensityWeighting(cfg) => Some(cfg.bootstrap_replicates),
            Self::PropensityMatching(cfg) => Some(cfg.bootstrap_replicates),
            Self::PropensityStratification(cfg) => Some(cfg.bootstrap_replicates),
            Self::DistanceMatching(cfg) => Some(cfg.bootstrap_replicates),
            Self::Aipw(cfg) => Some(cfg.bootstrap_replicates),
            Self::GlmAdjustment(cfg) => Some(cfg.bootstrap_replicates),
            Self::FrontDoorTwoStage(cfg) => Some(cfg.bootstrap_replicates),
            Self::IvWald(cfg) => Some(cfg.bootstrap_replicates),
            Self::Iv2Sls(cfg) => Some(cfg.bootstrap_replicates),
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
}

impl From<EstimatorId> for EstimatorSpec {
    fn from(id: EstimatorId) -> Self {
        Self::Default(id)
    }
}

impl From<LinearAdjustmentAte> for EstimatorSpec {
    fn from(cfg: LinearAdjustmentAte) -> Self {
        Self::LinearAdjustmentAte(Box::new(cfg))
    }
}

impl From<PropensityWeighting> for EstimatorSpec {
    fn from(cfg: PropensityWeighting) -> Self {
        Self::PropensityWeighting(Box::new(cfg))
    }
}

impl From<PropensityMatching> for EstimatorSpec {
    fn from(cfg: PropensityMatching) -> Self {
        Self::PropensityMatching(Box::new(cfg))
    }
}

impl From<PropensityStratification> for EstimatorSpec {
    fn from(cfg: PropensityStratification) -> Self {
        Self::PropensityStratification(Box::new(cfg))
    }
}

impl From<DistanceMatching> for EstimatorSpec {
    fn from(cfg: DistanceMatching) -> Self {
        Self::DistanceMatching(Box::new(cfg))
    }
}

impl From<AipwAte> for EstimatorSpec {
    fn from(cfg: AipwAte) -> Self {
        Self::Aipw(Box::new(cfg))
    }
}

impl From<GlmAdjustmentAte> for EstimatorSpec {
    fn from(cfg: GlmAdjustmentAte) -> Self {
        Self::GlmAdjustment(Box::new(cfg))
    }
}

impl From<FrontDoorTwoStage> for EstimatorSpec {
    fn from(cfg: FrontDoorTwoStage) -> Self {
        Self::FrontDoorTwoStage(Box::new(cfg))
    }
}

impl From<WaldIv> for EstimatorSpec {
    fn from(cfg: WaldIv) -> Self {
        Self::IvWald(Box::new(cfg))
    }
}

impl From<TwoStageLeastSquares> for EstimatorSpec {
    fn from(cfg: TwoStageLeastSquares) -> Self {
        Self::Iv2Sls(Box::new(cfg))
    }
}

impl From<DmlAte> for EstimatorSpec {
    fn from(cfg: DmlAte) -> Self {
        Self::Dml(Box::new(cfg))
    }
}

impl From<DrLearner> for EstimatorSpec {
    fn from(cfg: DrLearner) -> Self {
        Self::DrLearner(Box::new(cfg))
    }
}

impl From<CausalForest> for EstimatorSpec {
    fn from(cfg: CausalForest) -> Self {
        Self::CausalForest(Box::new(cfg))
    }
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
        assert_eq!(cfg.bootstrap_replicates, 500);
    }
}
