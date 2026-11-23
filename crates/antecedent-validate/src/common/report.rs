//! Refutation report scope and named construction.

use std::sync::Arc;

use crate::target_native::RefutationTarget;

/// Descriptive target scope of a refutation report; not execution authority.
///
/// A completion-level report is evidence only about that completion, even when
/// the outer program produced it. An absent scope denotes a legacy unscoped report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefutationScope {
    /// The complete target vector of the producing program.
    WholeTarget {
        /// Canonical target identity and coordinates.
        target: RefutationTarget,
    },
    /// One temporal posterior/class completion and horizon.
    TemporalClassCompletion {
        /// Posterior atom key.
        outer_atom: u64,
        /// Zero-based horizon coordinate.
        horizon_index: usize,
        /// Temporal horizon in steps.
        horizon: u32,
        /// Full canonical completion graph digest.
        completion_digest: [u8; 32],
        /// Canonical scoped level target.
        target: RefutationTarget,
    },
    /// One static posterior/class completion.
    StaticClassCompletion {
        /// Posterior atom key.
        outer_atom: u64,
        /// Full canonical completion graph digest.
        completion_digest: [u8; 32],
        /// Canonical scoped level target.
        target: RefutationTarget,
    },
}

/// Comparison of original vs refuted estimates.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RefutationReport {
    /// Refuter id.
    pub refuter: Arc<str>,
    /// Original ATE.
    pub original_ate: f64,
    /// Refuted / transformed ATE (mean across replicates when applicable).
    pub refuted_ate: f64,
    /// Scale-free comparison statistic. Replicate-based refuters store the two-sided
    /// p-value of the null value under the replicate distribution; sensitivity grids
    /// store the simulated tipping partial R² (the "robustness value"; not the closed-form
    /// Cinelli–Hazlett RV, see [`crate::sensitivity`]); overlap/e-value checks store their
    /// own statistic.
    pub comparison: f64,
    /// Whether the check is informative for the estimator used.
    pub informative: bool,
    /// Whether the check passed the configured threshold.
    pub passed: bool,
    /// Failure condition description when `passed` is false.
    pub failure_condition: Option<Arc<str>>,
    /// Number of replicate estimates.
    pub replicates: u32,
    /// Identity of the compiled program whose own execution produced this report, stamped
    /// when that execution returns its result (see [`Self::stamped_by`]). `None` for a report
    /// built outside a compiled execution: such a report cannot certify a claim on any target
    /// other than the scalar average effect.
    pub produced_by: Option<[u8; 32]>,
    /// Descriptive target scope, or legacy/unscoped when absent. Scope alone is not authority.
    pub scope: Option<RefutationScope>,
}

/// Which [`RefutationReport`] fields a [`RefutationReportBuilder`] has already set.
const REPORT_ORIGINAL: u8 = 1 << 0;
const REPORT_REFUTED: u8 = 1 << 1;
const REPORT_COMPARISON: u8 = 1 << 2;
const REPORT_INFORMATIVE: u8 = 1 << 3;
const REPORT_PASSED: u8 = 1 << 4;
const REPORT_FAILURE: u8 = 1 << 5;
const REPORT_REPLICATES: u8 = 1 << 6;
const REPORT_ALL: u8 = (1 << 7) - 1;

/// Named construction of a [`RefutationReport`].
///
/// `build` records a field only after its setter has run, so a call site cannot omit one
/// and have a zero or `false` stand in for it. Field order in the finished report is the
/// struct's own order.
#[must_use]
#[derive(Clone, Debug)]
pub(crate) struct RefutationReportBuilder {
    refuter: Arc<str>,
    original_ate: f64,
    refuted_ate: f64,
    comparison: f64,
    informative: bool,
    passed: bool,
    failure_condition: Option<Arc<str>>,
    replicates: u32,
    set: u8,
}

impl RefutationReport {
    /// Record that the compiled program `program` produced this report for its own result.
    /// A report already stamped keeps its producer.
    pub fn stamp_producer(&mut self, program: [u8; 32]) {
        if self.produced_by.is_none() {
            self.produced_by = Some(program);
        }
    }

    /// Whether the compiled program `program` produced this report.
    #[must_use]
    pub fn stamped_by(&self, program: &[u8; 32]) -> bool {
        self.produced_by.as_ref() == Some(program)
    }

    /// Start a report named `refuter`. Every other field is set by the builder.
    pub(crate) fn builder(refuter: impl Into<Arc<str>>) -> RefutationReportBuilder {
        RefutationReportBuilder {
            refuter: refuter.into(),
            original_ate: 0.0,
            refuted_ate: 0.0,
            comparison: 0.0,
            informative: false,
            passed: false,
            failure_condition: None,
            replicates: 0,
            set: 0,
        }
    }

    /// Construct a refutation report.
    ///
    /// The positional form of [`Self::builder`]. Callers outside this crate keep this
    /// signature; construction inside the crate names each field.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "positional form of the named builder, kept for callers outside this crate"
    )]
    pub fn new(
        refuter: impl Into<Arc<str>>,
        original_ate: f64,
        refuted_ate: f64,
        comparison: f64,
        informative: bool,
        passed: bool,
        failure_condition: Option<Arc<str>>,
        replicates: u32,
    ) -> Self {
        Self::builder(refuter)
            .original_ate(original_ate)
            .refuted_ate(refuted_ate)
            .comparison(comparison)
            .informative(informative)
            .passed(passed)
            .failure_condition(failure_condition)
            .replicates(replicates)
            .build()
    }

    /// Mixture-weighted aggregation of per-atom reports of the same refuter id.
    ///
    /// Weights are graph-posterior mass. Location statistics are weighted means;
    /// `replicates` is the maximum across atoms. `informative` is true when any
    /// contributing atom is informative. `passed` is fail-closed: the mixed check
    /// passes only when every positive-weight contributing atom passed.
    ///
    /// Atoms with non-positive weight, and atoms whose `refuter` id differs from
    /// the first contributing report, are dropped. Returns `None` when nothing
    /// contributes.
    #[must_use]
    pub fn mixture_weighted(items: &[(f64, &Self)]) -> Option<Self> {
        // A generic numeric mixture cannot synthesize authority over distinct
        // target scopes. Scoped reports require a separately checked target mix.
        if items.iter().any(|(weight, report)| *weight > 0.0 && report.scope.is_some()) {
            return None;
        }
        let first = items.iter().find(|(w, _)| *w > 0.0)?.1;
        let id = Arc::clone(&first.refuter);
        // Weights are rescaled by the largest contributing one, so masses of any magnitude
        // (1e300 or 1e-300) neither overflow the weighted sums nor underflow the total.
        let max_weight = items
            .iter()
            .filter(|(w, report)| *w > 0.0 && report.refuter == id)
            .map(|(w, _)| *w)
            .fold(0.0_f64, f64::max);
        if !max_weight.is_finite() {
            return None;
        }
        let mut w_sum = 0.0;
        let mut original = 0.0;
        let mut mixed_ate = 0.0;
        let mut comparison = 0.0;
        let mut any_failed = false;
        let mut informative = false;
        let mut replicates = 0_u32;
        let mut failure = None;
        for (w, report) in items {
            if *w <= 0.0 || report.refuter != id {
                continue;
            }
            let w = *w / max_weight;
            w_sum += w;
            original += w * report.original_ate;
            mixed_ate += w * report.refuted_ate;
            comparison += w * report.comparison;
            if !report.passed {
                // One failing contributing atom fails the mixture, whatever its mass.
                any_failed = true;
                if failure.is_none() {
                    failure.clone_from(&report.failure_condition);
                }
            }
            informative |= report.informative;
            replicates = replicates.max(report.replicates);
        }
        if w_sum <= 0.0 {
            return None;
        }
        let passed = !any_failed;
        Some(
            Self::builder(id)
                .original_ate(original / w_sum)
                .refuted_ate(mixed_ate / w_sum)
                .comparison(comparison / w_sum)
                .informative(informative)
                .passed(passed)
                .failure_condition(if passed { None } else { failure })
                .replicates(replicates)
                .build(),
        )
    }
}

impl RefutationReportBuilder {
    /// Published estimate the refuter is checking.
    pub fn original_ate(mut self, original_ate: f64) -> Self {
        self.original_ate = original_ate;
        self.set |= REPORT_ORIGINAL;
        self
    }

    /// Refuted or transformed estimate.
    pub fn refuted_ate(mut self, refuted_ate: f64) -> Self {
        self.refuted_ate = refuted_ate;
        self.set |= REPORT_REFUTED;
        self
    }

    /// Scale-free comparison statistic.
    pub fn comparison(mut self, comparison: f64) -> Self {
        self.comparison = comparison;
        self.set |= REPORT_COMPARISON;
        self
    }

    /// Whether the check can falsify the estimator that produced the estimate.
    pub fn informative(mut self, informative: bool) -> Self {
        self.informative = informative;
        self.set |= REPORT_INFORMATIVE;
        self
    }

    /// Whether the check passed its configured threshold.
    pub fn passed(mut self, passed: bool) -> Self {
        self.passed = passed;
        self.set |= REPORT_PASSED;
        self
    }

    /// Failure text when `passed` is false, and `None` when it is true.
    pub fn failure_condition(mut self, failure_condition: Option<Arc<str>>) -> Self {
        self.failure_condition = failure_condition;
        self.set |= REPORT_FAILURE;
        self
    }

    /// Number of replicate estimates that entered the report.
    pub fn replicates(mut self, replicates: u32) -> Self {
        self.replicates = replicates;
        self.set |= REPORT_REPLICATES;
        self
    }

    /// Finish the report.
    ///
    /// # Panics
    ///
    /// Panics if any field setter was skipped. That is a programming error at the call
    /// site; the stored values are exactly the ones the setters received.
    #[must_use]
    pub fn build(self) -> RefutationReport {
        assert_eq!(
            self.set, REPORT_ALL,
            "refutation report builder is missing a field (set bits {:#04x})",
            self.set
        );
        RefutationReport {
            refuter: self.refuter,
            original_ate: self.original_ate,
            refuted_ate: self.refuted_ate,
            comparison: self.comparison,
            informative: self.informative,
            passed: self.passed,
            failure_condition: self.failure_condition,
            replicates: self.replicates,
            produced_by: None,
            scope: None,
        }
    }
}
