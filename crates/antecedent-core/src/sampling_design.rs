//! Sampling-design contract for resampling, splitting, variance, and effective sample size.
//!
//! The design is an input. A time column on a table does not select it.
//! [`SamplingDesign::IndependentSubjectTrajectories`] may use row/subject
//! independent rules. A [`SamplingDesign::ClusteredPanel`] is resampled and
//! split by cluster, and its effective sample size counts clusters or a declared
//! cluster-aware size. A [`SamplingDesign::DependentTimeSeries`] keeps serial
//! dependence and cannot use an independent variance formula.
//!
//! Applying a rule the design does not allow is a failed attempt. The error
//! names that design and the rule it allows.
//!
//! This is not the transport-regime label [`crate::SamplingDesign`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// How the rows were sampled. Callers pass this; nothing here reads a table.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SamplingDesign {
    /// Independent subjects. A subject's trajectory is one sampling unit, so
    /// row/subject independent rules are the rules this design allows.
    IndependentSubjectTrajectories,
    /// Rows nested in clusters. The cluster is the sampling unit.
    ClusteredPanel,
    /// One dependent series. A time point is not an independent subject.
    DependentTimeSeries,
}

impl SamplingDesign {
    /// The three designs, in report order.
    ///
    /// A time column is not a fourth design and does not choose among these.
    pub const ALL: [Self; 3] =
        [Self::IndependentSubjectTrajectories, Self::ClusteredPanel, Self::DependentTimeSeries];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentSubjectTrajectories => "independent_subject_trajectories",
            Self::ClusteredPanel => "clustered_panel",
            Self::DependentTimeSeries => "dependent_time_series",
        }
    }

    /// Resampling, splitting, variance, and effective-sample-size rules this design allows.
    #[must_use]
    pub const fn rules(self) -> DesignRules {
        match self {
            Self::IndependentSubjectTrajectories => DesignRules {
                resampling: ResamplingRule::RowSubjectIid,
                splitting: SplittingRule::RowSubjectIid,
                variance: VarianceRule::RowSubjectIid,
                effective_sample_size: Self::INDEPENDENT_EFFECTIVE_SAMPLE_SIZE,
            },
            Self::ClusteredPanel => DesignRules {
                resampling: ResamplingRule::WholeCluster,
                splitting: SplittingRule::WholeCluster,
                variance: VarianceRule::ClusterRobust,
                effective_sample_size: Self::PANEL_EFFECTIVE_SAMPLE_SIZE,
            },
            Self::DependentTimeSeries => DesignRules {
                resampling: ResamplingRule::ContiguousBlock,
                splitting: SplittingRule::ContiguousBlock,
                variance: VarianceRule::DependencePreserving,
                effective_sample_size: Self::SERIES_EFFECTIVE_SAMPLE_SIZE,
            },
        }
    }

    /// Apply a resampling rule.
    ///
    /// # Errors
    ///
    /// [`SamplingDesignError::FailedAttempt`] when `rule` is not the resampling
    /// rule [`Self::rules`] binds. The error names the design and that allowed rule.
    pub const fn apply_resampling(
        self,
        rule: ResamplingRule,
    ) -> Result<BoundResampling, SamplingDesignError> {
        let allowed = self.rules().resampling();
        if same_resampling(rule, allowed) {
            Ok(BoundResampling { design: self, rule })
        } else {
            Err(failed(self, AttemptedRule::Resampling(rule), AllowedRule::Resampling(allowed)))
        }
    }

    /// Apply a splitting rule.
    ///
    /// # Errors
    ///
    /// [`SamplingDesignError::FailedAttempt`] when `rule` is not the splitting
    /// rule [`Self::rules`] binds. The error names the design and that allowed rule.
    pub const fn apply_splitting(
        self,
        rule: SplittingRule,
    ) -> Result<BoundSplitting, SamplingDesignError> {
        let allowed = self.rules().splitting();
        if same_splitting(rule, allowed) {
            Ok(BoundSplitting { design: self, rule })
        } else {
            Err(failed(self, AttemptedRule::Splitting(rule), AllowedRule::Splitting(allowed)))
        }
    }

    /// Apply a variance rule.
    ///
    /// # Errors
    ///
    /// [`SamplingDesignError::FailedAttempt`] when `rule` is not the variance
    /// rule [`Self::rules`] binds. The error names the design and that allowed rule.
    pub const fn apply_variance(
        self,
        rule: VarianceRule,
    ) -> Result<BoundVariance, SamplingDesignError> {
        let allowed = self.rules().variance();
        if same_variance(rule, allowed) {
            Ok(BoundVariance { design: self, rule })
        } else {
            Err(failed(self, AttemptedRule::Variance(rule), AllowedRule::Variance(allowed)))
        }
    }

    /// Apply an effective-sample-size basis.
    ///
    /// `size` is already expressed in the unit of `rule`. It is recorded only
    /// when `rule` is allowed. A rejected basis does not inspect `size`.
    ///
    /// # Errors
    ///
    /// [`SamplingDesignError::FailedAttempt`] when `rule` is not one of the
    /// bases [`Self::rules`] binds. The error names the design and each basis
    /// that design allows.
    pub const fn apply_effective_sample_size(
        self,
        rule: EffectiveSampleSizeRule,
        size: u64,
    ) -> Result<BoundEffectiveSampleSize, SamplingDesignError> {
        let allowed = self.rules().effective_sample_size();
        if ess_allowed(rule, allowed) {
            Ok(BoundEffectiveSampleSize { design: self, rule, size })
        } else {
            Err(failed(
                self,
                AttemptedRule::EffectiveSampleSize(rule),
                AllowedRule::EffectiveSampleSize(allowed),
            ))
        }
    }

    const INDEPENDENT_EFFECTIVE_SAMPLE_SIZE: &'static [EffectiveSampleSizeRule] =
        &[EffectiveSampleSizeRule::RowOrSubjectCount];
    const PANEL_EFFECTIVE_SAMPLE_SIZE: &'static [EffectiveSampleSizeRule] =
        &[EffectiveSampleSizeRule::ClusterCount, EffectiveSampleSizeRule::DeclaredClusterAware];
    const SERIES_EFFECTIVE_SAMPLE_SIZE: &'static [EffectiveSampleSizeRule] =
        &[EffectiveSampleSizeRule::DeclaredSeriesAware];
}

impl fmt::Display for SamplingDesign {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A caller-supplied design, plus whether the table has a time column.
///
/// The column is recorded and is not an input to [`SamplingDesign::rules`] or
/// to any `apply_*` method. Two declarations that differ only in the column
/// have the same design and the same rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SamplingDesignDeclaration {
    design: SamplingDesign,
    time_column_present: bool,
}

impl SamplingDesignDeclaration {
    /// Record `design`. `time_column_present` does not select or change it.
    #[must_use]
    pub const fn new(design: SamplingDesign, time_column_present: bool) -> Self {
        Self { design, time_column_present }
    }

    /// The design passed to [`Self::new`].
    #[must_use]
    pub const fn design(self) -> SamplingDesign {
        self.design
    }

    /// Whether the table has a time column. This is not a design.
    #[must_use]
    pub const fn time_column_present(self) -> bool {
        self.time_column_present
    }
}

/// Resampling unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResamplingRule {
    /// Resample each row or subject independently.
    RowSubjectIid,
    /// Resample whole clusters. Rows inside a drawn cluster stay together.
    WholeCluster,
    /// Resample contiguous blocks so dependence inside a block stays intact.
    ContiguousBlock,
}

impl ResamplingRule {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowSubjectIid => "row_subject_iid",
            Self::WholeCluster => "whole_cluster",
            Self::ContiguousBlock => "contiguous_block",
        }
    }
}

impl fmt::Display for ResamplingRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Split unit. Held-out rows follow the same unit as resampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SplittingRule {
    /// Split rows as if each row were an independent subject.
    RowSubjectIid,
    /// Assign whole clusters to folds.
    WholeCluster,
    /// Assign contiguous blocks to folds.
    ContiguousBlock,
}

impl SplittingRule {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowSubjectIid => "row_subject_iid",
            Self::WholeCluster => "whole_cluster",
            Self::ContiguousBlock => "contiguous_block",
        }
    }
}

impl fmt::Display for SplittingRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Variance formula.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VarianceRule {
    /// Independent row or subject variance formula.
    RowSubjectIid,
    /// Cluster-robust variance. Rows in a cluster are not independent draws.
    ClusterRobust,
    /// Variance that retains serial dependence. Not an independent-row formula.
    DependencePreserving,
}

impl VarianceRule {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowSubjectIid => "row_subject_iid",
            Self::ClusterRobust => "cluster_robust",
            Self::DependencePreserving => "dependence_preserving",
        }
    }
}

impl fmt::Display for VarianceRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Unit of an effective sample size.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EffectiveSampleSizeRule {
    /// Count of rows or independent subjects.
    RowOrSubjectCount,
    /// Count of clusters, not rows.
    ClusterCount,
    /// Caller-declared size that already accounts for within-cluster dependence.
    DeclaredClusterAware,
    /// Caller-declared size that already accounts for serial dependence.
    /// The length of the series is not this size.
    DeclaredSeriesAware,
}

impl EffectiveSampleSizeRule {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowOrSubjectCount => "row_or_subject_count",
            Self::ClusterCount => "cluster_count",
            Self::DeclaredClusterAware => "declared_cluster_aware",
            Self::DeclaredSeriesAware => "declared_series_aware",
        }
    }
}

impl fmt::Display for EffectiveSampleSizeRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Rules [`SamplingDesign::rules`] binds to one design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DesignRules {
    resampling: ResamplingRule,
    splitting: SplittingRule,
    variance: VarianceRule,
    effective_sample_size: &'static [EffectiveSampleSizeRule],
}

impl DesignRules {
    /// Resampling rule this design allows.
    #[must_use]
    pub const fn resampling(self) -> ResamplingRule {
        self.resampling
    }

    /// Splitting rule this design allows.
    #[must_use]
    pub const fn splitting(self) -> SplittingRule {
        self.splitting
    }

    /// Variance rule this design allows.
    #[must_use]
    pub const fn variance(self) -> VarianceRule {
        self.variance
    }

    /// Effective-sample-size bases this design allows, in report order.
    #[must_use]
    pub const fn effective_sample_size(self) -> &'static [EffectiveSampleSizeRule] {
        self.effective_sample_size
    }
}

/// Rule an attempt applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AttemptedRule {
    /// A resampling rule.
    Resampling(ResamplingRule),
    /// A splitting rule.
    Splitting(SplittingRule),
    /// A variance rule.
    Variance(VarianceRule),
    /// An effective-sample-size basis.
    EffectiveSampleSize(EffectiveSampleSizeRule),
}

impl fmt::Display for AttemptedRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resampling(rule) => write!(formatter, "{rule} resampling"),
            Self::Splitting(rule) => write!(formatter, "{rule} splitting"),
            Self::Variance(rule) => write!(formatter, "{rule} variance"),
            Self::EffectiveSampleSize(rule) => write!(formatter, "{rule} effective sample size"),
        }
    }
}

/// Rule, or effective-sample-size bases, that a design allows for the attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AllowedRule {
    /// The resampling rule the design allows.
    Resampling(ResamplingRule),
    /// The splitting rule the design allows.
    Splitting(SplittingRule),
    /// The variance rule the design allows.
    Variance(VarianceRule),
    /// Effective-sample-size bases the design allows, in report order.
    EffectiveSampleSize(&'static [EffectiveSampleSizeRule]),
}

impl fmt::Display for AllowedRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resampling(rule) => write!(formatter, "{rule} resampling"),
            Self::Splitting(rule) => write!(formatter, "{rule} splitting"),
            Self::Variance(rule) => write!(formatter, "{rule} variance"),
            Self::EffectiveSampleSize(rules) => {
                write_ess_list(rules, formatter)?;
                formatter.write_str(" effective sample size")
            }
        }
    }
}

/// A resampling, splitting, variance, or effective-sample-size rule was applied
/// to a design that does not allow it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SamplingDesignError {
    /// The attempt used a rule other than the rule this design allows.
    #[error("failed attempt: {design} does not allow {attempted}; {design} allows {allowed}")]
    FailedAttempt {
        /// Design the attempt was applied to.
        design: SamplingDesign,
        /// Rule the attempt applied.
        attempted: AttemptedRule,
        /// Rule this design allows for that kind of attempt.
        allowed: AllowedRule,
    },
}

/// Checked resampling for one design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundResampling {
    design: SamplingDesign,
    rule: ResamplingRule,
}

impl BoundResampling {
    /// Design the rule was applied to.
    #[must_use]
    pub const fn design(self) -> SamplingDesign {
        self.design
    }

    /// Resampling rule that design allows.
    #[must_use]
    pub const fn rule(self) -> ResamplingRule {
        self.rule
    }
}

/// Checked splitting for one design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundSplitting {
    design: SamplingDesign,
    rule: SplittingRule,
}

impl BoundSplitting {
    /// Design the rule was applied to.
    #[must_use]
    pub const fn design(self) -> SamplingDesign {
        self.design
    }

    /// Splitting rule that design allows.
    #[must_use]
    pub const fn rule(self) -> SplittingRule {
        self.rule
    }
}

/// Checked variance for one design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundVariance {
    design: SamplingDesign,
    rule: VarianceRule,
}

impl BoundVariance {
    /// Design the rule was applied to.
    #[must_use]
    pub const fn design(self) -> SamplingDesign {
        self.design
    }

    /// Variance rule that design allows.
    #[must_use]
    pub const fn rule(self) -> VarianceRule {
        self.rule
    }
}

/// Checked effective sample size for one design.
///
/// [`Self::size`] is in the unit of [`Self::rule`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BoundEffectiveSampleSize {
    design: SamplingDesign,
    rule: EffectiveSampleSizeRule,
    size: u64,
}

impl BoundEffectiveSampleSize {
    /// Design the basis was applied to.
    #[must_use]
    pub const fn design(self) -> SamplingDesign {
        self.design
    }

    /// Basis the size is expressed in.
    #[must_use]
    pub const fn rule(self) -> EffectiveSampleSizeRule {
        self.rule
    }

    /// Size in the unit of [`Self::rule`].
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
}

const fn failed(
    design: SamplingDesign,
    attempted: AttemptedRule,
    allowed: AllowedRule,
) -> SamplingDesignError {
    SamplingDesignError::FailedAttempt { design, attempted, allowed }
}

const fn same_resampling(left: ResamplingRule, right: ResamplingRule) -> bool {
    match left {
        ResamplingRule::RowSubjectIid => matches!(right, ResamplingRule::RowSubjectIid),
        ResamplingRule::WholeCluster => matches!(right, ResamplingRule::WholeCluster),
        ResamplingRule::ContiguousBlock => matches!(right, ResamplingRule::ContiguousBlock),
    }
}

const fn same_splitting(left: SplittingRule, right: SplittingRule) -> bool {
    match left {
        SplittingRule::RowSubjectIid => matches!(right, SplittingRule::RowSubjectIid),
        SplittingRule::WholeCluster => matches!(right, SplittingRule::WholeCluster),
        SplittingRule::ContiguousBlock => matches!(right, SplittingRule::ContiguousBlock),
    }
}

const fn same_variance(left: VarianceRule, right: VarianceRule) -> bool {
    match left {
        VarianceRule::RowSubjectIid => matches!(right, VarianceRule::RowSubjectIid),
        VarianceRule::ClusterRobust => matches!(right, VarianceRule::ClusterRobust),
        VarianceRule::DependencePreserving => matches!(right, VarianceRule::DependencePreserving),
    }
}

const fn same_ess(left: EffectiveSampleSizeRule, right: EffectiveSampleSizeRule) -> bool {
    match left {
        EffectiveSampleSizeRule::RowOrSubjectCount => {
            matches!(right, EffectiveSampleSizeRule::RowOrSubjectCount)
        }
        EffectiveSampleSizeRule::ClusterCount => {
            matches!(right, EffectiveSampleSizeRule::ClusterCount)
        }
        EffectiveSampleSizeRule::DeclaredClusterAware => {
            matches!(right, EffectiveSampleSizeRule::DeclaredClusterAware)
        }
        EffectiveSampleSizeRule::DeclaredSeriesAware => {
            matches!(right, EffectiveSampleSizeRule::DeclaredSeriesAware)
        }
    }
}

const fn ess_allowed(rule: EffectiveSampleSizeRule, allowed: &[EffectiveSampleSizeRule]) -> bool {
    let mut index = 0;
    while index < allowed.len() {
        if same_ess(rule, allowed[index]) {
            return true;
        }
        index += 1;
    }
    false
}

fn write_ess_list(
    rules: &[EffectiveSampleSizeRule],
    formatter: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let mut index = 0;
    while index < rules.len() {
        if index > 0 {
            formatter.write_str(" or ")?;
        }
        formatter.write_str(rules[index].as_str())?;
        index += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AllowedRule, AttemptedRule, EffectiveSampleSizeRule, ResamplingRule, SamplingDesign,
        SamplingDesignDeclaration, SamplingDesignError, SplittingRule, VarianceRule,
    };

    fn assert_failed(
        error: &SamplingDesignError,
        design: SamplingDesign,
        attempted: AttemptedRule,
        attempted_text: &str,
        allowed: AllowedRule,
        allowed_text: &str,
    ) {
        let SamplingDesignError::FailedAttempt {
            design: got_design,
            attempted: got_attempted,
            allowed: got_allowed,
        } = error;
        assert_eq!(*got_design, design);
        assert_eq!(*got_attempted, attempted);
        assert_eq!(got_attempted.to_string(), attempted_text);
        assert_eq!(*got_allowed, allowed);
        assert_eq!(got_allowed.to_string(), allowed_text);
        let message = error.to_string();
        let head = format!("failed attempt: {design} does not allow {attempted_text}; ");
        let tail = format!("{design} allows {allowed_text}");
        assert_eq!(message, format!("{head}{tail}"));
    }

    #[test]
    fn independent_subject_trajectories_bind_row_subject_iid_rules() {
        let design = SamplingDesign::IndependentSubjectTrajectories;
        assert_eq!(design.as_str(), "independent_subject_trajectories");
        let rules = design.rules();
        assert_eq!(rules.resampling(), ResamplingRule::RowSubjectIid);
        assert_eq!(rules.splitting(), SplittingRule::RowSubjectIid);
        assert_eq!(rules.variance(), VarianceRule::RowSubjectIid);
        assert_eq!(rules.effective_sample_size(), &[EffectiveSampleSizeRule::RowOrSubjectCount]);

        let resampled = design
            .apply_resampling(ResamplingRule::RowSubjectIid)
            .expect("independent trajectories resample by subject");
        assert_eq!(resampled.design(), design);
        assert_eq!(resampled.rule(), ResamplingRule::RowSubjectIid);
        let split = design
            .apply_splitting(SplittingRule::RowSubjectIid)
            .expect("independent trajectories split by subject");
        assert_eq!(split.design(), design);
        assert_eq!(split.rule(), SplittingRule::RowSubjectIid);
        let variance = design
            .apply_variance(VarianceRule::RowSubjectIid)
            .expect("independent trajectories use the row or subject variance");
        assert_eq!(variance.design(), design);
        assert_eq!(variance.rule(), VarianceRule::RowSubjectIid);
        let size = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::RowOrSubjectCount, 40)
            .expect("independent trajectories count subjects");
        assert_eq!(size.design(), design);
        assert_eq!(size.rule(), EffectiveSampleSizeRule::RowOrSubjectCount);
        assert_eq!(size.size(), 40);

        let error = design
            .apply_resampling(ResamplingRule::WholeCluster)
            .expect_err("cluster resampling is not the independent-trajectory rule");
        assert_failed(
            &error,
            design,
            AttemptedRule::Resampling(ResamplingRule::WholeCluster),
            "whole_cluster resampling",
            AllowedRule::Resampling(ResamplingRule::RowSubjectIid),
            "row_subject_iid resampling",
        );
    }

    #[test]
    fn clustered_panel_binds_cluster_rules() {
        let design = SamplingDesign::ClusteredPanel;
        assert_eq!(design.as_str(), "clustered_panel");
        let rules = design.rules();
        assert_eq!(rules.resampling(), ResamplingRule::WholeCluster);
        assert_eq!(rules.splitting(), SplittingRule::WholeCluster);
        assert_eq!(rules.variance(), VarianceRule::ClusterRobust);
        assert_eq!(
            rules.effective_sample_size(),
            &[EffectiveSampleSizeRule::ClusterCount, EffectiveSampleSizeRule::DeclaredClusterAware,]
        );

        let resampled = design
            .apply_resampling(ResamplingRule::WholeCluster)
            .expect("a panel resamples clusters");
        assert_eq!(resampled.design(), design);
        assert_eq!(resampled.rule(), ResamplingRule::WholeCluster);
        let split =
            design.apply_splitting(SplittingRule::WholeCluster).expect("a panel splits clusters");
        assert_eq!(split.design(), design);
        assert_eq!(split.rule(), SplittingRule::WholeCluster);
        let variance = design
            .apply_variance(VarianceRule::ClusterRobust)
            .expect("a panel uses cluster-robust variance");
        assert_eq!(variance.design(), design);
        assert_eq!(variance.rule(), VarianceRule::ClusterRobust);

        let clusters = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::ClusterCount, 4)
            .expect("panel effective sample size counts clusters");
        assert_eq!(clusters.design(), design);
        assert_eq!(clusters.rule(), EffectiveSampleSizeRule::ClusterCount);
        assert_eq!(clusters.size(), 4);
        let declared = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::DeclaredClusterAware, 9)
            .expect("a declared cluster-aware size is a panel size");
        assert_eq!(declared.design(), design);
        assert_eq!(declared.rule(), EffectiveSampleSizeRule::DeclaredClusterAware);
        assert_eq!(declared.size(), 9);
    }

    #[test]
    fn dependent_time_series_binds_dependence_preserving_rules() {
        let design = SamplingDesign::DependentTimeSeries;
        assert_eq!(design.as_str(), "dependent_time_series");
        let rules = design.rules();
        assert_eq!(rules.resampling(), ResamplingRule::ContiguousBlock);
        assert_eq!(rules.splitting(), SplittingRule::ContiguousBlock);
        assert_eq!(rules.variance(), VarianceRule::DependencePreserving);
        assert_eq!(rules.effective_sample_size(), &[EffectiveSampleSizeRule::DeclaredSeriesAware]);

        let resampled = design
            .apply_resampling(ResamplingRule::ContiguousBlock)
            .expect("a series resamples contiguous blocks");
        assert_eq!(resampled.design(), design);
        assert_eq!(resampled.rule(), ResamplingRule::ContiguousBlock);
        let split = design
            .apply_splitting(SplittingRule::ContiguousBlock)
            .expect("a series splits contiguous blocks");
        assert_eq!(split.design(), design);
        assert_eq!(split.rule(), SplittingRule::ContiguousBlock);
        let variance = design
            .apply_variance(VarianceRule::DependencePreserving)
            .expect("a series keeps dependence in the variance");
        assert_eq!(variance.design(), design);
        assert_eq!(variance.rule(), VarianceRule::DependencePreserving);
        let size = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::DeclaredSeriesAware, 6)
            .expect("a series size is a declared dependence-aware size");
        assert_eq!(size.design(), design);
        assert_eq!(size.rule(), EffectiveSampleSizeRule::DeclaredSeriesAware);
        assert_eq!(size.size(), 6);
    }

    #[test]
    fn clustered_panel_rejects_row_iid_treatment() {
        let design = SamplingDesign::ClusteredPanel;
        let resampling = design
            .apply_resampling(ResamplingRule::RowSubjectIid)
            .expect_err("a panel cannot resample rows as independent subjects");
        assert_failed(
            &resampling,
            design,
            AttemptedRule::Resampling(ResamplingRule::RowSubjectIid),
            "row_subject_iid resampling",
            AllowedRule::Resampling(ResamplingRule::WholeCluster),
            "whole_cluster resampling",
        );
        let splitting = design
            .apply_splitting(SplittingRule::RowSubjectIid)
            .expect_err("a panel cannot split rows as independent subjects");
        assert_failed(
            &splitting,
            design,
            AttemptedRule::Splitting(SplittingRule::RowSubjectIid),
            "row_subject_iid splitting",
            AllowedRule::Splitting(SplittingRule::WholeCluster),
            "whole_cluster splitting",
        );
        let variance = design
            .apply_variance(VarianceRule::RowSubjectIid)
            .expect_err("a panel cannot use an independent-row variance");
        assert_failed(
            &variance,
            design,
            AttemptedRule::Variance(VarianceRule::RowSubjectIid),
            "row_subject_iid variance",
            AllowedRule::Variance(VarianceRule::ClusterRobust),
            "cluster_robust variance",
        );
        let size = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::RowOrSubjectCount, 100)
            .expect_err("a panel effective sample size is not the row count");
        assert_failed(
            &size,
            design,
            AttemptedRule::EffectiveSampleSize(EffectiveSampleSizeRule::RowOrSubjectCount),
            "row_or_subject_count effective sample size",
            AllowedRule::EffectiveSampleSize(design.rules().effective_sample_size()),
            "cluster_count or declared_cluster_aware effective sample size",
        );
    }

    #[test]
    fn dependent_time_series_rejects_row_iid_treatment() {
        let design = SamplingDesign::DependentTimeSeries;
        let variance = design
            .apply_variance(VarianceRule::RowSubjectIid)
            .expect_err("a dependent series cannot use an independent variance formula");
        assert_failed(
            &variance,
            design,
            AttemptedRule::Variance(VarianceRule::RowSubjectIid),
            "row_subject_iid variance",
            AllowedRule::Variance(VarianceRule::DependencePreserving),
            "dependence_preserving variance",
        );
        let resampling = design
            .apply_resampling(ResamplingRule::RowSubjectIid)
            .expect_err("a series cannot resample rows as independent subjects");
        assert_failed(
            &resampling,
            design,
            AttemptedRule::Resampling(ResamplingRule::RowSubjectIid),
            "row_subject_iid resampling",
            AllowedRule::Resampling(ResamplingRule::ContiguousBlock),
            "contiguous_block resampling",
        );
        let splitting = design
            .apply_splitting(SplittingRule::RowSubjectIid)
            .expect_err("a series cannot split rows as independent subjects");
        assert_failed(
            &splitting,
            design,
            AttemptedRule::Splitting(SplittingRule::RowSubjectIid),
            "row_subject_iid splitting",
            AllowedRule::Splitting(SplittingRule::ContiguousBlock),
            "contiguous_block splitting",
        );
        let size = design
            .apply_effective_sample_size(EffectiveSampleSizeRule::RowOrSubjectCount, 100)
            .expect_err("series length is not an effective sample size");
        assert_failed(
            &size,
            design,
            AttemptedRule::EffectiveSampleSize(EffectiveSampleSizeRule::RowOrSubjectCount),
            "row_or_subject_count effective sample size",
            AllowedRule::EffectiveSampleSize(design.rules().effective_sample_size()),
            "declared_series_aware effective sample size",
        );
    }

    #[test]
    fn time_column_does_not_select_the_design() {
        for design in SamplingDesign::ALL {
            let with_time = SamplingDesignDeclaration::new(design, true);
            let without_time = SamplingDesignDeclaration::new(design, false);
            assert!(with_time.time_column_present());
            assert!(!without_time.time_column_present());
            assert_eq!(with_time.design(), design);
            assert_eq!(without_time.design(), design);
            assert_eq!(with_time.design().rules(), without_time.design().rules());
        }

        let panel = SamplingDesignDeclaration::new(SamplingDesign::ClusteredPanel, true);
        let panel_rows = panel
            .design()
            .apply_resampling(ResamplingRule::RowSubjectIid)
            .expect_err("a time column does not make panel rows independent subjects");
        assert_failed(
            &panel_rows,
            SamplingDesign::ClusteredPanel,
            AttemptedRule::Resampling(ResamplingRule::RowSubjectIid),
            "row_subject_iid resampling",
            AllowedRule::Resampling(ResamplingRule::WholeCluster),
            "whole_cluster resampling",
        );

        let series = SamplingDesignDeclaration::new(SamplingDesign::DependentTimeSeries, true);
        let series_variance = series
            .design()
            .apply_variance(VarianceRule::RowSubjectIid)
            .expect_err("a time column does not license an independent variance formula");
        assert_failed(
            &series_variance,
            SamplingDesign::DependentTimeSeries,
            AttemptedRule::Variance(VarianceRule::RowSubjectIid),
            "row_subject_iid variance",
            AllowedRule::Variance(VarianceRule::DependencePreserving),
            "dependence_preserving variance",
        );
    }
}
