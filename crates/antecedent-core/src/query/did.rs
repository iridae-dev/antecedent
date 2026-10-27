//! Two-period and selected staggered group-time difference-in-differences queries.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QueryError;
use super::fault::PanelDesignFault;
use crate::canonical_encode::{
    CanonicalEncode, encode_flags, encode_i64s, encode_option, encode_variables, flag, i64_bits,
};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use std::sync::Arc;

/// Sampling design for a difference-in-differences comparison.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DidSamplingDesign {
    /// The same subjects contribute one observation in each period.
    #[default]
    BalancedPanel,
    /// Each subject contributes one observation in exactly one period.
    RepeatedCrossSection,
    /// A balanced staggered panel, compared with never-treated subjects at a selected time.
    StaggeredGroupTime,
    /// All cohort-specific event-time contrasts relative to adoption period minus one.
    StaggeredEventStudy,
    /// One row per subject with supplied propensity and untreated-change predictions.
    AugmentedPanel,
    /// A staggered rollout estimated cell by cell without already-treated controls
    /// ([`HeterogeneityRobustDid`]); balanced or unbalanced.
    StaggeredHeterogeneityRobust,
    /// A two-period panel whose parallel untreated trends hold conditional on baseline
    /// covariates, estimated by outcome regression, inverse probability weighting or the doubly
    /// robust 2x2.
    CovariateAdjustedTwoPeriod,
}

impl CanonicalEncode for DidSamplingDesign {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let tag = match self {
            Self::BalancedPanel => 1,
            Self::RepeatedCrossSection => 2,
            Self::StaggeredGroupTime => 3,
            Self::StaggeredEventStudy => 4,
            Self::AugmentedPanel => 5,
            Self::StaggeredHeterogeneityRobust => 6,
            Self::CovariateAdjustedTwoPeriod => 7,
        };
        writer.word(tag);
    }
}

/// Nuisance columns for the augmented two-period panel ATT.
#[derive(Clone, Debug, PartialEq)]
pub struct AugmentedPanelNuisance {
    pub outcome_pre: VariableId,
    pub propensity: VariableId,
    pub untreated_change_prediction: VariableId,
    /// Caller declaration only; the engine does not certify model fitting.
    pub predictions_cross_fitted: bool,
}

/// Every field is written, the cross-fitting declaration included.
impl CanonicalEncode for AugmentedPanelNuisance {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome_pre.encode(writer);
        self.propensity.encode(writer);
        self.untreated_change_prediction.encode(writer);
        writer.word(flag(self.predictions_cross_fitted));
    }
}

/// Estimator family of a heterogeneity-robust rollout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum RolloutDidEstimator {
    /// Group-time 2x2 comparisons with never-treated or not-yet-treated units.
    #[default]
    CallawaySantAnna,
    /// Interaction-weighted cohort-by-event-time contrasts against one control cohort.
    SunAbraham,
    /// Untreated outcomes imputed from unit and period effects fitted on untreated rows.
    Imputation,
}

impl RolloutDidEstimator {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CallawaySantAnna => "callaway_santanna",
            Self::SunAbraham => "sun_abraham",
            Self::Imputation => "imputation",
        }
    }
}

/// Units a treated cohort is compared with.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DidComparisonGroup {
    /// Units never treated in the panel.
    #[default]
    NeverTreated,
    /// Never-treated units and units adopting after both compared periods.
    NotYetTreated,
}

impl DidComparisonGroup {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NeverTreated => "never_treated",
            Self::NotYetTreated => "not_yet_treated",
        }
    }
}

/// How baseline covariates enter a 2x2 comparison.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DidNuisanceMethod {
    /// Regression of the untreated outcome change on the covariates among comparison units.
    OutcomeRegression,
    /// Normalized odds weights of a logistic treated-cohort propensity.
    InverseProbabilityWeighting,
    /// The outcome regression corrected by the odds-weighted comparison residuals.
    #[default]
    DoublyRobust,
}

impl DidNuisanceMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OutcomeRegression => "outcome_regression",
            Self::InverseProbabilityWeighting => "inverse_probability_weighting",
            Self::DoublyRobust => "doubly_robust",
        }
    }
}

/// Base period of a pre-adoption contrast.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DidBasePeriod {
    /// Each pre-adoption period against the period before it.
    #[default]
    Varying,
    /// Every period against the last period before adoption (minus anticipation).
    Universal,
}

impl DidBasePeriod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Varying => "varying",
            Self::Universal => "universal",
        }
    }
}

/// Aggregation of group-time effects into the reported summary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DidAggregation {
    /// Cohort-size-weighted average over post-adoption cells.
    #[default]
    Simple,
    /// Event-time profile; the summary averages the post-adoption event times.
    Dynamic,
    /// Per-cohort averages; the summary weights them by cohort size.
    Group,
    /// Per-calendar-period averages; the summary averages the periods.
    Calendar,
}

impl DidAggregation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Dynamic => "dynamic",
            Self::Group => "group",
            Self::Calendar => "calendar",
        }
    }
}

/// Heterogeneity-robust specification of a staggered or covariate-adjusted rollout.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HeterogeneityRobustDid {
    /// Estimator family.
    pub estimator: RolloutDidEstimator,
    /// Comparison group.
    pub comparison: DidComparisonGroup,
    /// Covariate adjustment of a 2x2 comparison.
    pub nuisance: DidNuisanceMethod,
    /// Base period of pre-adoption contrasts.
    pub base_period: DidBasePeriod,
    /// Aggregation that defines the reported summary.
    pub aggregation: DidAggregation,
    /// Periods before adoption in which units may already respond.
    pub anticipation: u32,
    /// Baseline covariate columns, read at each unit's first observed row.
    pub covariates: Arc<[VariableId]>,
}

/// Every field is written; covariates keep their declared order.
impl CanonicalEncode for HeterogeneityRobustDid {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let estimator = match self.estimator {
            RolloutDidEstimator::CallawaySantAnna => 1,
            RolloutDidEstimator::SunAbraham => 2,
            RolloutDidEstimator::Imputation => 3,
        };
        let comparison = match self.comparison {
            DidComparisonGroup::NeverTreated => 1,
            DidComparisonGroup::NotYetTreated => 2,
        };
        let nuisance = match self.nuisance {
            DidNuisanceMethod::OutcomeRegression => 1,
            DidNuisanceMethod::InverseProbabilityWeighting => 2,
            DidNuisanceMethod::DoublyRobust => 3,
        };
        let base = match self.base_period {
            DidBasePeriod::Varying => 1,
            DidBasePeriod::Universal => 2,
        };
        let aggregation = match self.aggregation {
            DidAggregation::Simple => 1,
            DidAggregation::Dynamic => 2,
            DidAggregation::Group => 3,
            DidAggregation::Calendar => 4,
        };
        writer
            .word(estimator)
            .word(comparison)
            .word(nuisance)
            .word(base)
            .word(aggregation)
            .word(u64::from(self.anticipation));
        encode_variables(writer, &self.covariates);
    }
}

/// How a whole-procedure resample reweights the inference clusters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum DidResamplingScheme {
    /// Nonparametric bootstrap of whole clusters.
    #[default]
    ClusterBootstrap,
    /// Exponential multiplier weights on whole clusters.
    ClusterMultiplier,
}

/// Whole-procedure resample of a rollout or augmented panel: every replicate refits the
/// entire estimator on reweighted clusters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DidResampling {
    /// Cluster reweighting.
    pub scheme: DidResamplingScheme,
    /// Replicates.
    pub replicates: u32,
}

/// Every field is written.
impl CanonicalEncode for DidResampling {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let scheme = match self.scheme {
            DidResamplingScheme::ClusterBootstrap => 1,
            DidResamplingScheme::ClusterMultiplier => 2,
        };
        writer.word(scheme).word(u64::from(self.replicates));
    }
}

/// Marker word ahead of the rollout fields, written only when one of them is present so a
/// query without them keeps its canonical bytes.
const ROLLOUT_FIELDS: u64 = 0x524f_4c4c_4f55_5400;

/// Difference in outcome changes for a two-period or selected group-time design.
///
/// This estimates a point and cluster-robust standard error. Its causal
/// interpretation requires consistency, no anticipation, and parallel trends.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelDidQuery {
    /// Whether subjects recur across periods or are sampled once.
    pub design: DidSamplingDesign,
    /// Outcome column.
    pub outcome: VariableId,
    /// Stable treatment group, aligned to outcome rows.
    pub treated: Arc<[bool]>,
    /// Pre/post period indicator, aligned to outcome rows.
    pub post: Arc<[bool]>,
    /// Subject identifier for each observation.
    pub subjects: Arc<[Arc<str>]>,
    /// Inference cluster for each observation, constant within subject.
    pub clusters: Arc<[Arc<str>]>,
    /// Calendar period of each row for a staggered comparison.
    pub periods: Arc<[i64]>,
    /// First treated period; zero denotes a never-treated subject.
    pub cohorts: Arc<[i64]>,
    /// Selected adoption cohort and post-adoption comparison period.
    pub target: Option<(i64, i64)>,
    /// Supplied nuisance columns for the one-row-per-subject augmented design.
    pub augmented: Option<AugmentedPanelNuisance>,
    /// Heterogeneity-robust specification of a staggered or covariate-adjusted rollout.
    pub robust: Option<HeterogeneityRobustDid>,
    /// Whole-procedure resample of a rollout or augmented panel.
    pub resampling: Option<DidResampling>,
}

/// Every field is written; every array is row-aligned and keeps its order.
impl CanonicalEncode for PanelDidQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.design.encode(writer);
        self.outcome.encode(writer);
        encode_flags(writer, &self.treated);
        encode_flags(writer, &self.post);
        writer.texts(&self.subjects).texts(&self.clusters);
        encode_i64s(writer, &self.periods);
        encode_i64s(writer, &self.cohorts);
        match self.target {
            Some((cohort, period)) => {
                writer.word(1).word(i64_bits(cohort)).word(i64_bits(period));
            }
            None => {
                writer.word(0);
            }
        }
        encode_option(writer, self.augmented.as_ref());
        if self.robust.is_some() || self.resampling.is_some() {
            writer.word(ROLLOUT_FIELDS);
            encode_option(writer, self.robust.as_ref());
            encode_option(writer, self.resampling.as_ref());
        }
    }
}

impl PanelDidQuery {
    /// Construct the row-aligned design contract.
    #[must_use]
    pub fn new(
        outcome: VariableId,
        treated: impl Into<Arc<[bool]>>,
        post: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        Self {
            design: DidSamplingDesign::BalancedPanel,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects: subjects.into(),
            clusters: clusters.into(),
            periods: Arc::from([]),
            cohorts: Arc::from([]),
            target: None,
            augmented: None,
            robust: None,
            resampling: None,
        }
    }

    /// Select one group-time ATT from a staggered adoption panel.
    #[must_use]
    pub fn staggered_group_time(
        outcome: VariableId,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        cohorts: impl Into<Arc<[i64]>>,
        target_cohort: i64,
        target_period: i64,
    ) -> Self {
        let subjects = subjects.into();
        let clusters = clusters.into();
        let periods = periods.into();
        let cohorts = cohorts.into();
        let treated = cohorts.iter().map(|value| *value == target_cohort).collect::<Vec<_>>();
        let post = periods.iter().map(|value| *value == target_period).collect::<Vec<_>>();
        Self {
            design: DidSamplingDesign::StaggeredGroupTime,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects,
            clusters,
            periods,
            cohorts,
            target: Some((target_cohort, target_period)),
            augmented: None,
            robust: None,
            resampling: None,
        }
    }

    /// Compare every adoption cohort with never-treated controls over the panel.
    #[must_use]
    pub fn staggered_event_study(
        outcome: VariableId,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        cohorts: impl Into<Arc<[i64]>>,
    ) -> Self {
        let subjects = subjects.into();
        let cohorts = cohorts.into();
        let periods = periods.into();
        let treated = cohorts.iter().map(|value| *value > 0).collect::<Vec<_>>();
        let post = periods
            .iter()
            .zip(cohorts.iter())
            .map(|(period, cohort)| *cohort > 0 && *period >= *cohort)
            .collect::<Vec<_>>();
        Self {
            design: DidSamplingDesign::StaggeredEventStudy,
            outcome,
            treated: treated.into(),
            post: post.into(),
            subjects,
            clusters: clusters.into(),
            periods,
            cohorts,
            target: None,
            augmented: None,
            robust: None,
            resampling: None,
        }
    }

    /// Construct an augmented panel ATT with supplied, row-aligned nuisance predictions.
    // The augmented panel binds four frozen columns plus assignment, subject,
    // cluster, and cross-fit metadata; each is a distinct caller input.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "binds four frozen columns plus assignment, subject, cluster and cross-fit metadata; each is a distinct caller input"
    )]
    pub fn augmented_panel(
        outcome_post: VariableId,
        outcome_pre: VariableId,
        propensity: VariableId,
        untreated_change_prediction: VariableId,
        treated: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        predictions_cross_fitted: bool,
    ) -> Self {
        let treated = treated.into();
        let post = vec![true; treated.len()];
        let mut query = Self::new(outcome_post, treated, post, subjects, clusters);
        query.design = DidSamplingDesign::AugmentedPanel;
        query.augmented = Some(AugmentedPanelNuisance {
            outcome_pre,
            propensity,
            untreated_change_prediction,
            predictions_cross_fitted,
        });
        query
    }

    /// Construct a repeated-cross-section design with one row per subject.
    #[must_use]
    pub fn repeated_cross_section(
        outcome: VariableId,
        treated: impl Into<Arc<[bool]>>,
        post: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        let mut query = Self::new(outcome, treated, post, subjects, clusters);
        query.design = DidSamplingDesign::RepeatedCrossSection;
        query
    }

    /// A staggered rollout estimated without already-treated controls, on a balanced or
    /// unbalanced panel. `cohorts` holds each row's first treated period (zero for never
    /// treated).
    #[must_use]
    pub fn heterogeneity_robust(
        outcome: VariableId,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        periods: impl Into<Arc<[i64]>>,
        cohorts: impl Into<Arc<[i64]>>,
        robust: HeterogeneityRobustDid,
    ) -> Self {
        let mut query = Self::staggered_event_study(outcome, subjects, clusters, periods, cohorts);
        query.design = DidSamplingDesign::StaggeredHeterogeneityRobust;
        query.robust = Some(robust);
        query
    }

    /// A two-period panel with parallel untreated trends conditional on baseline covariates.
    /// `treated` marks the treated group and `post` the second period; the design is recorded
    /// as cohort two (treated) or zero (comparison) over periods one and two.
    #[must_use]
    pub fn covariate_adjusted_two_period(
        outcome: VariableId,
        treated: impl Into<Arc<[bool]>>,
        post: impl Into<Arc<[bool]>>,
        subjects: impl Into<Arc<[Arc<str>]>>,
        clusters: impl Into<Arc<[Arc<str>]>>,
        covariates: impl Into<Arc<[VariableId]>>,
        nuisance: DidNuisanceMethod,
    ) -> Self {
        let mut query = Self::new(outcome, treated, post, subjects, clusters);
        query.periods = query.post.iter().map(|post| if *post { 2 } else { 1 }).collect();
        query.cohorts = query.treated.iter().map(|treated| if *treated { 2 } else { 0 }).collect();
        query.design = DidSamplingDesign::CovariateAdjustedTwoPeriod;
        query.robust = Some(HeterogeneityRobustDid {
            nuisance,
            covariates: covariates.into(),
            ..HeterogeneityRobustDid::default()
        });
        query
    }

    /// Attach a whole-procedure resample to a rollout or augmented panel.
    #[must_use]
    pub fn with_resampling(mut self, scheme: DidResamplingScheme, replicates: u32) -> Self {
        self.resampling = Some(DidResampling { scheme, replicates });
        self
    }

    /// The query restricted to the rows where `keep` is true, preserving row order, subject,
    /// cluster, period and cohort labels. Nothing is relabelled.
    ///
    /// Returns `None` for an augmented panel, whose nuisance columns are row-aligned in the
    /// data and would need the same restriction, and when `keep` does not match the rows.
    #[must_use]
    pub fn restrict_rows(&self, keep: &[bool]) -> Option<Self> {
        // Optional label columns are empty when undeclared and stay empty.
        fn pick<T: Clone>(values: &Arc<[T]>, keep: &[bool]) -> Arc<[T]> {
            if values.len() != keep.len() {
                return Arc::from([]);
            }
            values.iter().zip(keep).filter(|(_, kept)| **kept).map(|(v, _)| v.clone()).collect()
        }
        let n = self.treated.len();
        if keep.len() != n || self.augmented.is_some() {
            return None;
        }
        Some(Self {
            design: self.design,
            outcome: self.outcome,
            treated: pick(&self.treated, keep),
            post: pick(&self.post, keep),
            subjects: pick(&self.subjects, keep),
            clusters: pick(&self.clusters, keep),
            periods: pick(&self.periods, keep),
            cohorts: pick(&self.cohorts, keep),
            target: self.target,
            augmented: None,
            robust: self.robust.clone(),
            resampling: self.resampling,
        })
    }

    /// Validate metadata dimensions and labels.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError::InvalidRandomizedEffect`] when fewer than four rows
    /// are supplied or the metadata vectors are misaligned, subject or cluster
    /// labels are blank, an augmented panel is missing or has non-distinct
    /// columns or duplicate subjects, nuisance columns are attached without an
    /// augmented panel, or a staggered design has an invalid cohort/period
    /// selection or misaligned cohort/period vectors.
    pub fn validate(&self) -> Result<(), QueryError> {
        let n = self.treated.len();
        if n < 4 || self.post.len() != n || self.subjects.len() != n || self.clusters.len() != n {
            return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::DidRows));
        }
        if self.subjects.iter().any(|x| x.trim().is_empty())
            || self.clusters.iter().any(|x| x.trim().is_empty())
        {
            return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::BlankLabels));
        }
        if self.design == DidSamplingDesign::AugmentedPanel {
            let Some(nuisance) = &self.augmented else {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::AugmentedNeedsNuisance,
                ));
            };
            let variables = [
                self.outcome,
                nuisance.outcome_pre,
                nuisance.propensity,
                nuisance.untreated_change_prediction,
            ];
            if variables.iter().collect::<std::collections::BTreeSet<_>>().len() != variables.len()
                || self.subjects.iter().collect::<std::collections::BTreeSet<_>>().len() != n
                || self.post.iter().any(|post| !post)
                || !self.periods.is_empty()
                || !self.cohorts.is_empty()
                || self.target.is_some()
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::AugmentedColumns,
                ));
            }
        } else if self.augmented.is_some() {
            return Err(QueryError::InvalidRandomizedEffect(
                PanelDesignFault::NuisanceNeedsAugmented,
            ));
        }
        if matches!(
            self.design,
            DidSamplingDesign::StaggeredGroupTime | DidSamplingDesign::StaggeredEventStudy
        ) {
            let (cohort, period) = if self.design == DidSamplingDesign::StaggeredGroupTime {
                let Some((cohort, period)) = self.target else {
                    return Err(QueryError::InvalidRandomizedEffect(
                        PanelDesignFault::StaggeredTarget,
                    ));
                };
                (cohort, period)
            } else {
                (2, 2)
            };
            if (self.design == DidSamplingDesign::StaggeredGroupTime
                && (cohort <= 1 || period < cohort))
                || self.periods.len() != n
                || self.cohorts.len() != n
                || self.periods.iter().any(|p| *p <= 0)
                || self.cohorts.iter().any(|g| *g < 0)
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::StaggeredPeriods,
                ));
            }
        }
        self.validate_rollout()
    }

    /// The rollout fields agree with the design.
    fn validate_rollout(&self) -> Result<(), QueryError> {
        let n = self.treated.len();
        let rollout = matches!(
            self.design,
            DidSamplingDesign::StaggeredHeterogeneityRobust
                | DidSamplingDesign::CovariateAdjustedTwoPeriod
        );
        match (&self.robust, rollout) {
            (Some(_), false) => {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::RobustSpecificationDesign,
                ));
            }
            (None, true) => {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::RobustDesignSpecification,
                ));
            }
            _ => {}
        }
        if let Some(robust) = &self.robust {
            if self.periods.len() != n
                || self.cohorts.len() != n
                || self.periods.iter().any(|period| *period <= 0)
                || self.cohorts.iter().any(|cohort| *cohort < 0)
                || !self.cohorts.iter().any(|cohort| *cohort > 0)
                || self.target.is_some()
                || self.augmented.is_some()
            {
                return Err(QueryError::InvalidRandomizedEffect(PanelDesignFault::RolloutPeriods));
            }
            if robust.covariates.contains(&self.outcome)
                || robust.covariates.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != robust.covariates.len()
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::RolloutCovariates,
                ));
            }
            if self.design == DidSamplingDesign::CovariateAdjustedTwoPeriod
                && (robust.covariates.is_empty()
                    || robust.estimator != RolloutDidEstimator::CallawaySantAnna
                    || robust.anticipation != 0)
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::CovariateAdjustedTwoPeriod,
                ));
            }
        }
        if let Some(resampling) = &self.resampling {
            if resampling.replicates < 2
                || !(rollout || self.design == DidSamplingDesign::AugmentedPanel)
            {
                return Err(QueryError::InvalidRandomizedEffect(
                    PanelDesignFault::WholeProcedureResample,
                ));
            }
        }
        Ok(())
    }
}
