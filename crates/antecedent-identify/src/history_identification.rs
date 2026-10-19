//! History and initial-state identification for temporal routes.
//!
//! [`HistoryIdentification::Certified`] is a finite feedback ancestry together
//! with a declared [`InitialLaw`]. Ancestry past the declared search bound is
//! [`HistoryIdentification::Uncertified`]: the product keeps that bound and
//! returns a bounded-search diagnostic naming the missing ancestor. Nothing
//! here increases the cap or treats that history as identified.
//!
//! [`HistoryIdentification::PulseFallback`] is only the existing narrow pulse
//! route `identify_pulse_by_parent_adjustment`
//! ([`PARENT_ADJUSTMENT_RULE`](crate::temporal_backdoor::PARENT_ADJUSTMENT_RULE)).
//! It does not certify feedback ancestry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, InitialLaw, TemporalNodeKey,
};
use thiserror::Error;

/// Diagnostic code when feedback ancestry stops at the declared search bound.
pub const HISTORY_SEARCH_BOUNDED_CODE: &str = "identify.history.search_bounded";

/// Diagnostic code for the narrow pulse fallback.
///
/// The fallback is not a feedback-ancestry certificate.
pub const HISTORY_PULSE_FALLBACK_CODE: &str = "identify.history.pulse_fallback";

/// Route the caller asked history identification to answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HistoryIdentificationTarget {
    /// Finite feedback ancestry closed under a declared initial law.
    FeedbackAncestry,
    /// Existing narrow pulse route `identify_pulse_by_parent_adjustment`
    /// ([`PARENT_ADJUSTMENT_RULE`](crate::temporal_backdoor::PARENT_ADJUSTMENT_RULE)).
    PulseParentAdjustment,
}

/// Declared bound on feedback-ancestry search.
///
/// The stored step count is the caller's bound. No method returns a larger one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HistorySearchBound {
    steps: u32,
}

impl HistorySearchBound {
    /// Bound of `steps` lags, as already declared by the caller.
    #[must_use]
    pub const fn declared(steps: u32) -> Self {
        Self { steps }
    }

    /// Declared step count. This is the bound that was given, not a wider one.
    #[must_use]
    pub const fn steps(self) -> u32 {
        self.steps
    }
}

/// One node of a feedback ancestry, with the depth the search bound compares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FeedbackAncestor {
    node: TemporalNodeKey,
    lag: u32,
}

impl FeedbackAncestor {
    /// Ancestor `node` at `lag` steps.
    ///
    /// `lag` is the depth compared with [`HistorySearchBound`]. A lag greater
    /// than that bound is outside the certified history.
    #[must_use]
    pub const fn new(node: TemporalNodeKey, lag: u32) -> Self {
        Self { node, lag }
    }

    /// Ancestor node.
    #[must_use]
    pub const fn node(self) -> TemporalNodeKey {
        self.node
    }

    /// Ancestry depth in steps.
    #[must_use]
    pub const fn lag(self) -> u32 {
        self.lag
    }
}

/// Finite ancestry and the initial law that certify it.
///
/// Both are present: the ancestry list is non-empty, and the law was declared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedHistory {
    ancestry: Vec<FeedbackAncestor>,
    initial_law: InitialLaw,
    bound: HistorySearchBound,
}

impl CertifiedHistory {
    /// Finite ancestry. Non-empty, and every lag is within [`Self::search_bound`].
    #[must_use]
    pub fn ancestry(&self) -> &[FeedbackAncestor] {
        &self.ancestry
    }

    /// Declared initial law.
    #[must_use]
    pub const fn initial_law(&self) -> &InitialLaw {
        &self.initial_law
    }

    /// Search bound the ancestry fit inside. Equal to the bound that was given.
    #[must_use]
    pub const fn search_bound(&self) -> HistorySearchBound {
        self.bound
    }
}

/// Narrow pulse fallback. Feedback ancestry is not certified.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PulseFallbackHistory {
    diagnostic: Diagnostic,
}

impl PulseFallbackHistory {
    /// Diagnostic stating that this is the pulse fallback.
    #[must_use]
    pub const fn diagnostic(&self) -> &Diagnostic {
        &self.diagnostic
    }
}

/// Feedback ancestry that the declared bound does not finitely certify.
///
/// `bound` is the bound the caller declared. The missing ancestor lies past it.
/// This is not [`HistoryIdentification::Certified`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UncertifiedHistory {
    bound: HistorySearchBound,
    missing_ancestor: FeedbackAncestor,
    diagnostic: Diagnostic,
}

impl UncertifiedHistory {
    /// Declared search bound, unchanged.
    #[must_use]
    pub const fn search_bound(&self) -> HistorySearchBound {
        self.bound
    }

    /// Ancestor whose lag is past [`Self::search_bound`].
    #[must_use]
    pub const fn missing_ancestor(&self) -> FeedbackAncestor {
        self.missing_ancestor
    }

    /// Bounded-search diagnostic for this uncertified history.
    #[must_use]
    pub const fn diagnostic(&self) -> &Diagnostic {
        &self.diagnostic
    }
}

/// History-identification product for a temporal route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HistoryIdentification {
    /// Finite ancestry and a declared initial law.
    Certified(CertifiedHistory),
    /// Existing narrow pulse route. Not a feedback-ancestry certificate.
    PulseFallback(PulseFallbackHistory),
    /// Declared bound reached before feedback ancestry was finite.
    Uncertified(UncertifiedHistory),
}

/// Why history identification was refused before a product existed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum HistoryIdentificationRefusal {
    /// Feedback ancestry was requested without an initial law.
    #[error("initial law is missing; next attempt: declare the initial law")]
    MissingInitialLaw,
    /// Feedback ancestry was requested with an empty ancestor list.
    #[error("feedback ancestry is empty; next attempt: supply the finite ancestor list")]
    EmptyAncestry,
}

/// Identify history for `target` under the already declared `bound`.
///
/// [`HistoryIdentificationTarget::PulseParentAdjustment`] returns
/// [`HistoryIdentification::PulseFallback`] and does not certify `ancestry`.
/// [`HistoryIdentificationTarget::FeedbackAncestry`] returns
/// [`HistoryIdentification::Certified`] only when `ancestry` is non-empty,
/// every lag is within `bound`, and `initial_law` is present. A lag past
/// `bound` returns [`HistoryIdentification::Uncertified`] carrying that same
/// bound and the missing ancestor.
///
/// # Errors
///
/// [`HistoryIdentificationRefusal::MissingInitialLaw`] when feedback ancestry
/// is requested and `initial_law` is absent.
/// [`HistoryIdentificationRefusal::EmptyAncestry`] when feedback ancestry is
/// requested and `ancestry` is empty.
#[must_use = "a refusal is not a history identification"]
pub fn identify_history(
    target: HistoryIdentificationTarget,
    ancestry: &[FeedbackAncestor],
    initial_law: Option<&InitialLaw>,
    bound: HistorySearchBound,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    match target {
        HistoryIdentificationTarget::PulseParentAdjustment => Ok(pulse_fallback()),
        HistoryIdentificationTarget::FeedbackAncestry => {
            identify_feedback(ancestry, initial_law, bound)
        }
    }
}

/// Whether `identification` certifies finite feedback ancestry.
///
/// [`HistoryIdentification::PulseFallback`] and
/// [`HistoryIdentification::Uncertified`] do not.
#[must_use]
pub const fn is_feedback_certified(identification: &HistoryIdentification) -> bool {
    matches!(identification, HistoryIdentification::Certified(_))
}

/// Declared search bound carried by `identification`, when it has one.
///
/// The step count is the bound given to [`identify_history`], never a larger
/// one. The pulse fallback has no feedback-ancestry bound.
#[must_use]
pub const fn search_bound(identification: &HistoryIdentification) -> Option<HistorySearchBound> {
    match identification {
        HistoryIdentification::Certified(certified) => Some(certified.search_bound()),
        HistoryIdentification::Uncertified(uncertified) => Some(uncertified.search_bound()),
        HistoryIdentification::PulseFallback(_) => None,
    }
}

fn identify_feedback(
    ancestry: &[FeedbackAncestor],
    initial_law: Option<&InitialLaw>,
    bound: HistorySearchBound,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    let Some(initial_law) = initial_law else {
        return Err(HistoryIdentificationRefusal::MissingInitialLaw);
    };
    if ancestry.is_empty() {
        return Err(HistoryIdentificationRefusal::EmptyAncestry);
    }
    let missing = ancestry.iter().copied().find(|ancestor| ancestor.lag() > bound.steps());
    if let Some(missing_ancestor) = missing {
        return Ok(HistoryIdentification::Uncertified(uncertified_history(
            bound,
            missing_ancestor,
        )));
    }
    Ok(HistoryIdentification::Certified(CertifiedHistory {
        ancestry: ancestry.to_vec(),
        initial_law: initial_law.clone(),
        bound,
    }))
}

fn pulse_fallback() -> HistoryIdentification {
    let rule = crate::temporal_backdoor::PARENT_ADJUSTMENT_RULE;
    let diagnostic = Diagnostic::new(
        HISTORY_PULSE_FALLBACK_CODE,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        format!(
            "pulse fallback: identify_pulse_by_parent_adjustment ({rule}) does not certify \
             feedback ancestry"
        ),
    );
    HistoryIdentification::PulseFallback(PulseFallbackHistory { diagnostic })
}

fn uncertified_history(bound: HistorySearchBound, missing: FeedbackAncestor) -> UncertifiedHistory {
    let steps = bound.steps();
    let lag = missing.lag();
    let variable = missing.node().variable.raw();
    let diagnostic = crate::result::search_bounded_diagnostic(
        HISTORY_SEARCH_BOUNDED_CODE,
        format!(
            "uncertified history: bounded search stopped at the declared bound {steps}; \
             missing ancestor variable {variable} at lag {lag}"
        ),
    )
    .with_fields([
        ("bound", steps.to_string()),
        ("missing_ancestor_variable", variable.to_string()),
        ("missing_ancestor_lag", lag.to_string()),
    ]);
    UncertifiedHistory { bound, missing_ancestor: missing, diagnostic }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{InitialLaw, TemporalNodeKey, VariableId};

    use super::{
        FeedbackAncestor, HistoryIdentification, HistoryIdentificationRefusal,
        HistoryIdentificationTarget, HistorySearchBound, identify_history, is_feedback_certified,
        search_bound,
    };

    fn ancestor(variable: u32, lag: u32) -> FeedbackAncestor {
        let offset = -i32::try_from(lag).unwrap();
        FeedbackAncestor::new(
            TemporalNodeKey { variable: VariableId::from_raw(variable), offset },
            lag,
        )
    }

    fn bound_field(product: &HistoryIdentification) -> Option<u32> {
        let HistoryIdentification::Uncertified(uncertified) = product else {
            return None;
        };
        uncertified.diagnostic().fields.iter().find_map(|(key, value)| {
            (key.as_ref() == "bound").then(|| value.as_ref().parse().unwrap())
        })
    }

    #[test]
    fn finite_ancestry_and_initial_law_is_certified() {
        let bound = HistorySearchBound::declared(3);
        let law = InitialLaw::named("origin").unwrap();
        let ancestry = [ancestor(0, 0), ancestor(1, 3)];
        let product = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &ancestry,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(is_feedback_certified(&product));
        let HistoryIdentification::Certified(certified) = product else {
            panic!("finite ancestry and an initial law are certified");
        };
        assert_eq!(certified.ancestry(), &ancestry);
        assert_eq!(certified.initial_law(), &law);
        assert_eq!(certified.search_bound(), bound);
    }

    #[test]
    fn missing_initial_law_is_refused() {
        let error = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 1)],
            None,
            HistorySearchBound::declared(4),
        )
        .unwrap_err();
        assert_eq!(error, HistoryIdentificationRefusal::MissingInitialLaw);
    }

    #[test]
    fn empty_ancestry_is_refused() {
        let law = InitialLaw::contemporaneous_observed();
        let error = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[],
            Some(&law),
            HistorySearchBound::declared(4),
        )
        .unwrap_err();
        assert_eq!(error, HistoryIdentificationRefusal::EmptyAncestry);
    }

    #[test]
    fn ancestry_past_the_declared_bound_is_uncertified_and_the_bound_is_unchanged() {
        let bound = HistorySearchBound::declared(2);
        let missing = ancestor(7, 5);
        let product = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(1, 2), missing],
            Some(&InitialLaw::contemporaneous_observed()),
            bound,
        )
        .unwrap();
        assert!(!is_feedback_certified(&product));
        let HistoryIdentification::Uncertified(uncertified) = &product else {
            panic!("ancestry past the bound is uncertified");
        };
        assert_eq!(uncertified.search_bound(), bound);
        assert_eq!(uncertified.search_bound().steps(), bound.steps());
        assert_eq!(uncertified.missing_ancestor(), missing);
        let diagnostic = uncertified.diagnostic();
        assert!(diagnostic.code.ends_with(".search_bounded"));
        assert!(diagnostic.message.contains("uncertified history"));
        assert_eq!(bound_field(&product), Some(bound.steps()));
    }

    #[test]
    fn pulse_fallback_is_not_feedback_certified() {
        let product = identify_history(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &[ancestor(0, 9)],
            None,
            HistorySearchBound::declared(1),
        )
        .unwrap();
        assert!(!is_feedback_certified(&product));
        let HistoryIdentification::PulseFallback(fallback) = &product else {
            panic!("the narrow pulse route yields the pulse fallback");
        };
        assert!(fallback.diagnostic().message.contains("pulse fallback"));
        assert!(fallback.diagnostic().message.contains("identify_pulse_by_parent_adjustment"));
    }

    #[test]
    fn pulse_fallback_is_only_the_narrow_pulse_route() {
        let law = InitialLaw::contemporaneous_observed();
        let feedback = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 1)],
            Some(&law),
            HistorySearchBound::declared(1),
        )
        .unwrap();
        assert!(!matches!(feedback, HistoryIdentification::PulseFallback(_)));
        let pulse = identify_history(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &[],
            None,
            HistorySearchBound::declared(1),
        )
        .unwrap();
        assert!(matches!(pulse, HistoryIdentification::PulseFallback(_)));
    }

    #[test]
    fn no_api_returns_a_bound_greater_than_the_one_given() {
        let declared = HistorySearchBound::declared(4);
        let law = InitialLaw::contemporaneous_observed();
        let within = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 4)],
            Some(&law),
            declared,
        )
        .unwrap();
        let past = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 1), ancestor(1, 9)],
            Some(&law),
            declared,
        )
        .unwrap();
        let pulse = identify_history(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &[ancestor(1, 9)],
            None,
            declared,
        )
        .unwrap();

        for product in [&within, &past, &pulse] {
            assert!(search_bound(product).is_none_or(|bound| bound.steps() <= declared.steps()));
            assert!(bound_field(product).is_none_or(|steps| steps <= declared.steps()));
        }
        assert_eq!(search_bound(&within).map(HistorySearchBound::steps), Some(declared.steps()));
        assert_eq!(search_bound(&past).map(HistorySearchBound::steps), Some(declared.steps()));
        assert!(search_bound(&pulse).is_none());
        let HistoryIdentification::Certified(certified) = &within else {
            panic!("ancestry inside the bound is certified");
        };
        assert_eq!(certified.search_bound().steps(), declared.steps());
        let HistoryIdentification::Uncertified(uncertified) = &past else {
            panic!("ancestry past the bound is uncertified");
        };
        assert_eq!(uncertified.search_bound().steps(), declared.steps());
        assert!(!is_feedback_certified(&past));
    }
}
