//! Typed findings, the recommendation, and the report identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::CanonicalWriter;
use antecedent_core::recalc::{
    RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_io_core::dependency_key::{DependencyClass, ReusableProduct};

const REPORT_TAG: &str = "antecedent.validate.staleness.report.v1";

/// What a host should do with a stored plan or claim, given new data.
///
/// The order is the order of severity: a report takes the most severe recommendation any of
/// its findings forces.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Recommendation {
    /// No finding exceeds its threshold: the stored claim stands for the new data.
    Reuse,
    /// The data changed but the identification-level premises hold: keep the plan and its
    /// identification, refit (score or fit artifact, law, decision).
    Refresh,
    /// A premise of the identification product or of the adjustment set no longer holds
    /// (overlap, support, a recorded identification check, a period or population change):
    /// identify again before estimating.
    Reidentify,
    /// The new data cannot be analysed under the plan at all (schema or target incompatible,
    /// or too little data to assess uncertainty).
    Refuse,
}

impl Recommendation {
    /// Stable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Reuse => "reuse",
            Self::Refresh => "refresh",
            Self::Reidentify => "reidentify",
            Self::Refuse => "refuse",
        }
    }
}

/// Kind of a staleness finding.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum FindingKind {
    /// The marginal distribution of one column moved (mean or binned density ratio).
    CovariateShift,
    /// The joint distribution of the covariates moved (Mahalanobis or energy statistic).
    JointShift,
    /// More propensities fall outside the positivity margin than the policy allows.
    PositivityChange,
    /// The effective sample size of the inverse-propensity weights fell.
    OverlapChange,
    /// A check the plan recorded as passed fails on the new data.
    AssumptionRegression,
    /// New values fall outside the recorded range.
    SupportLoss,
    /// A level appeared that the baseline never saw, or a recorded level disappeared.
    LevelChange,
    /// The stored posterior and the likelihood of the new data disagree.
    PriorLikelihoodConflict,
    /// The per-period summaries of the new data drift.
    PeriodDrift,
    /// Columns were added, removed, retyped or reordered.
    SchemaChange,
    /// The missingness rate of a column, or the missingness patterns, changed.
    MissingnessChange,
    /// Too few rows (in total or in an arm) to assess uncertainty.
    InsufficientData,
    /// The target cannot be formed on the new data (a role column or an arm is missing).
    TargetIncompatible,
}

impl FindingKind {
    /// Stable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CovariateShift => "covariate_shift",
            Self::JointShift => "joint_shift",
            Self::PositivityChange => "positivity_change",
            Self::OverlapChange => "overlap_change",
            Self::AssumptionRegression => "assumption_regression",
            Self::SupportLoss => "support_loss",
            Self::LevelChange => "level_change",
            Self::PriorLikelihoodConflict => "prior_likelihood_conflict",
            Self::PeriodDrift => "period_drift",
            Self::SchemaChange => "schema_change",
            Self::MissingnessChange => "missingness_change",
            Self::InsufficientData => "insufficient_data",
            Self::TargetIncompatible => "target_incompatible",
        }
    }
}

/// One evaluated diagnostic: a statistic, the policy threshold it was judged by, and the
/// dependency keys it invalidates when it exceeds that threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct StalenessFinding {
    /// Kind.
    pub kind: FindingKind,
    /// What was assessed: a column, `joint`, a check name, or `column:level`.
    pub subject: String,
    /// Name of the statistic.
    pub statistic_name: &'static str,
    /// Value of the statistic on the new data (or the new-to-recorded comparison).
    pub statistic: f64,
    /// Name of the policy field the statistic is judged by.
    pub threshold_name: &'static str,
    /// Threshold value from the policy (or from the recorded check).
    pub threshold: f64,
    /// Unadjusted p-value of a test-based diagnostic.
    pub p_value: Option<f64>,
    /// Multiplicity-adjusted p-value of a test-based diagnostic.
    pub adjusted_p_value: Option<f64>,
    /// Whether the diagnostic exceeded its threshold (and, when test-based, was significant
    /// at the policy level after adjustment).
    pub exceeded: bool,
    /// The recommendation this diagnostic forces when it exceeds.
    pub forces: Recommendation,
    /// Dependency keys (declared recalculation inputs) it changes when it exceeds.
    pub keys: Vec<Stage>,
    /// Auxiliary statistics, by name.
    pub auxiliary: Vec<(String, f64)>,
    /// Plain-language detail.
    pub detail: String,
}

impl StalenessFinding {
    /// A diagnostic judged by an effect-size or count threshold only.
    pub(crate) fn threshold(
        kind: FindingKind,
        subject: impl Into<String>,
        statistic: (&'static str, f64),
        threshold: (&'static str, f64),
        exceeded: bool,
        forces: Recommendation,
        keys: Vec<Stage>,
    ) -> Self {
        Self {
            kind,
            subject: subject.into(),
            statistic_name: statistic.0,
            statistic: statistic.1,
            threshold_name: threshold.0,
            threshold: threshold.1,
            p_value: None,
            adjusted_p_value: None,
            exceeded,
            forces,
            keys,
            auxiliary: Vec::new(),
            detail: String::new(),
        }
    }

    /// Attach an auxiliary statistic.
    #[must_use]
    pub(crate) fn with(mut self, name: &str, value: f64) -> Self {
        self.auxiliary.push((name.into(), value));
        self
    }

    /// Attach the detail text.
    #[must_use]
    pub(crate) fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = detail.into();
        self
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        writer
            .text(self.kind.label())
            .text(&self.subject)
            .text(self.statistic_name)
            .real(self.statistic)
            .text(self.threshold_name)
            .real(self.threshold)
            .opt_real(self.p_value)
            .opt_real(self.adjusted_p_value)
            .word(u64::from(self.exceeded))
            .text(self.forces.label())
            .word(self.keys.len() as u64);
        for key in &self.keys {
            writer.text(&key.label());
        }
        writer.word(self.auxiliary.len() as u64);
        for (name, value) in &self.auxiliary {
            writer.text(name).real(*value);
        }
    }
}

/// The result of comparing new data with a stored plan or artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct StalenessReport {
    /// What the host should do.
    pub recommendation: Recommendation,
    /// The attempt to make next, in words.
    pub next_attempt: String,
    /// Every evaluated diagnostic, exceeded or not, in evaluation order.
    pub evaluations: Vec<StalenessFinding>,
    /// Diagnostics that could not be evaluated, with the reason.
    pub not_evaluated: Vec<(String, String)>,
    /// Declared recalculation inputs the findings change, in stage order.
    pub changed_keys: Vec<Stage>,
    /// Stages the recalculation planner recomputes when those inputs change.
    pub invalidated: Vec<Stage>,
    /// Semantic dependency classes of the changed inputs.
    pub changed_classes: Vec<DependencyClass>,
    /// Reusable products whose dependency key reads a changed class.
    pub invalidated_products: Vec<ReusableProduct>,
    /// Identity of the plan the baseline describes.
    pub plan_identity: String,
    /// Content identity of the baseline's data snapshot.
    pub old_snapshot: String,
    /// Content identity of the new data snapshot.
    pub new_snapshot: String,
    /// Seal of the baseline.
    pub baseline_identity: String,
    /// Identity of the policy the report was judged under.
    pub policy_identity: String,
    /// Rows of the new data.
    pub new_rows: u64,
    /// Identity of the report: old snapshot, new snapshot, policy and every evaluation.
    pub identity: String,
}

impl StalenessReport {
    /// The diagnostics that exceeded their thresholds.
    #[must_use = "the findings are the diagnostics that exceeded"]
    pub fn findings(&self) -> impl Iterator<Item = &StalenessFinding> {
        self.evaluations.iter().filter(|f| f.exceeded)
    }

    /// Stable labels of [`Self::changed_keys`].
    #[must_use]
    pub fn changed_key_labels(&self) -> Vec<String> {
        self.changed_keys.iter().map(|s| s.label()).collect()
    }

    /// Stable labels of [`Self::invalidated`].
    #[must_use]
    pub fn invalidated_labels(&self) -> Vec<String> {
        self.invalidated.iter().map(|s| s.label()).collect()
    }

    /// Canonical identity of the report fields (old snapshot, new snapshot, policy, baseline,
    /// recommendation, every evaluation, and the keys).
    #[must_use]
    pub(crate) fn compute_identity(&self) -> String {
        let mut writer = CanonicalWriter::new(REPORT_TAG);
        writer
            .text(&self.plan_identity)
            .text(&self.old_snapshot)
            .text(&self.new_snapshot)
            .text(&self.baseline_identity)
            .text(&self.policy_identity)
            .word(self.new_rows)
            .text(self.recommendation.label())
            .text(&self.next_attempt)
            .word(self.evaluations.len() as u64);
        for finding in &self.evaluations {
            finding.write(&mut writer);
        }
        writer.word(self.not_evaluated.len() as u64);
        for (what, why) in &self.not_evaluated {
            writer.text(what).text(why);
        }
        writer.word(self.changed_keys.len() as u64);
        for key in &self.changed_keys {
            writer.text(&key.label());
        }
        writer.finish_hex()
    }
}

/// The semantic dependency class of a declared recalculation input.
#[must_use]
pub fn dependency_class(stage: Stage) -> Option<DependencyClass> {
    match stage {
        Stage::DataSnapshot => Some(DependencyClass::SourceSnapshot),
        Stage::RowDesign | Stage::LearnerFoldsRng => Some(DependencyClass::RowsHistoryFolds),
        Stage::Graph
        | Stage::Query
        | Stage::Regime
        | Stage::Evidence
        | Stage::SourcePopulation
        | Stage::TargetPopulation
        | Stage::Identification => Some(DependencyClass::TargetIdentification),
        Stage::Prior(_) | Stage::ProviderRequest(_) | Stage::Representation => {
            Some(DependencyClass::ModelPriorProvider)
        }
        Stage::TreatmentGrid => Some(DependencyClass::InterventionTargets),
        Stage::Calibration => Some(DependencyClass::Inference),
        _ => None,
    }
}

/// The canonical workflow the staleness planner reasons over: every declared input of an
/// estimated claim and its derived stages, with the prior stage of a Bayesian plan. Each key in
/// `keys` gets a requested identity that differs from its previous one; every other stage keeps
/// its identity, so the planner recomputes exactly what the changed keys reach.
fn workflow(
    keys: &BTreeSet<Stage>,
    changed_digest: &str,
    bayesian: bool,
) -> (StageIdentities, StageIdentities) {
    let mut stages = vec![
        Stage::Graph,
        Stage::Query,
        Stage::Regime,
        Stage::Evidence,
        Stage::SourcePopulation,
        Stage::TargetPopulation,
        Stage::DataSnapshot,
        Stage::RowDesign,
        Stage::TreatmentGrid,
        Stage::LearnerFoldsRng,
        Stage::Utility,
        Stage::Identification,
        Stage::ScoreArtifact,
        Stage::Law,
        Stage::Decision,
    ];
    if bayesian {
        stages.extend(antecedent_core::recalc::Branch::new(0).map(Stage::Prior));
    }
    let mut previous = StageIdentities::new();
    let mut requested = StageIdentities::new();
    for stage in stages {
        let label = stage.label();
        let old = StageIdentity::of("staleness.workflow", &[label.as_bytes()]);
        let new = if keys.contains(&stage) {
            StageIdentity::of("staleness.changed", &[label.as_bytes(), changed_digest.as_bytes()])
        } else {
            old
        };
        previous.set(stage, old);
        requested.set(stage, new);
    }
    (previous, requested)
}

/// Stages the recalculation planner recomputes when `keys` change, in stage order.
pub(crate) fn invalidated_stages(
    keys: &BTreeSet<Stage>,
    changed_digest: &str,
    bayesian: bool,
) -> Vec<Stage> {
    if keys.is_empty() {
        return Vec::new();
    }
    let (previous, requested) = workflow(keys, changed_digest, bayesian);
    RecalcPlan::plan(
        &previous,
        &requested,
        &RecalcCapabilities::in_process(RetargetSupport::Licensed),
    )
    .recomputed()
}

/// Dependency classes of `keys`, and the reusable products that read any of them.
pub(crate) fn classes_and_products(keys: &[Stage]) -> (Vec<DependencyClass>, Vec<ReusableProduct>) {
    let classes: BTreeSet<DependencyClass> =
        keys.iter().filter_map(|k| dependency_class(*k)).collect();
    let products = ReusableProduct::ALL
        .into_iter()
        .filter(|product| classes.iter().any(|class| product.declares(*class)))
        .collect();
    (classes.into_iter().collect(), products)
}
