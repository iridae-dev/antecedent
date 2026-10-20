//! History and initial-state identification for temporal routes.
//!
//! [`HistoryIdentification::Certified`] is a finite feedback ancestry together
//! with a declared [`InitialLaw`] and a declaration that the search frontier is
//! closed ([`AncestryFrontier::Closed`]). This module does not search a graph.
//! It checks that the declared ancestry is consistent with the declared bound
//! and that the caller declared the search finished; the ancestry list and the
//! frontier are the caller's. Each ancestor's lag is derived from the temporal
//! offset of its node, not taken from the caller.
//!
//! Ancestry past the declared search bound, or a frontier left open, is
//! [`HistoryIdentification::Uncertified`]: the product keeps that bound and
//! returns a bounded-search diagnostic. Nothing here increases the cap or
//! treats that history as identified.
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

/// Whether the bounded ancestry search ran to exhaustion.
///
/// This is the caller's declaration about its own search. It is not checked
/// here, and it is the one fact that separates a finished ancestry from a list
/// that merely stopped at the bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AncestryFrontier {
    /// No unexpanded ancestor remains: the search finished inside the bound.
    Closed,
    /// The search stopped with ancestors still to expand. Ancestry may continue
    /// past the bound.
    Open,
}

/// One node of a feedback ancestry, with the depth the search bound compares.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FeedbackAncestor {
    node: TemporalNodeKey,
}

impl FeedbackAncestor {
    /// Ancestor `node`, at the lag its temporal offset gives.
    ///
    /// The lag is `-node.offset`, compared with [`HistorySearchBound`]. A lag
    /// greater than that bound is outside the certified history.
    ///
    /// # Errors
    ///
    /// [`HistoryIdentificationRefusal::FutureAncestor`] when the offset is
    /// positive, which is not an ancestor.
    #[must_use = "a refused ancestor is not part of an ancestry"]
    pub const fn new(node: TemporalNodeKey) -> Result<Self, HistoryIdentificationRefusal> {
        if node.offset > 0 {
            return Err(HistoryIdentificationRefusal::FutureAncestor { offset: node.offset });
        }
        Ok(Self { node })
    }

    /// Ancestor node.
    #[must_use]
    pub const fn node(self) -> TemporalNodeKey {
        self.node
    }

    /// Ancestry depth in steps, `-offset` of the node.
    #[must_use]
    pub const fn lag(self) -> u32 {
        self.node.offset.unsigned_abs()
    }
}

/// Finite ancestry, the declared initial law, and the declared closed frontier.
///
/// All three are present: the ancestry list is non-empty with every lag inside
/// the bound, the law was declared, and the caller declared the frontier closed.
/// This is a consistency check on those declarations, not a graph search.
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

/// Why a feedback ancestry is not finitely certified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UncertifiedReason {
    /// A declared ancestor lies past the bound.
    AncestorPastBound(FeedbackAncestor),
    /// Every declared ancestor is inside the bound, but the search frontier was
    /// left open, so ancestry may continue past the bound.
    FrontierOpen,
}

/// Feedback ancestry that the declared bound does not finitely certify.
///
/// `bound` is the bound the caller declared. This is not
/// [`HistoryIdentification::Certified`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UncertifiedHistory {
    bound: HistorySearchBound,
    reason: UncertifiedReason,
    diagnostic: Diagnostic,
}

impl UncertifiedHistory {
    /// Declared search bound, unchanged.
    #[must_use]
    pub const fn search_bound(&self) -> HistorySearchBound {
        self.bound
    }

    /// Why the ancestry is not certified.
    #[must_use]
    pub const fn reason(&self) -> UncertifiedReason {
        self.reason
    }

    /// Declared ancestor whose lag is past [`Self::search_bound`], when that is
    /// the reason. [`None`] when the frontier was left open.
    #[must_use]
    pub const fn missing_ancestor(&self) -> Option<FeedbackAncestor> {
        match self.reason {
            UncertifiedReason::AncestorPastBound(ancestor) => Some(ancestor),
            UncertifiedReason::FrontierOpen => None,
        }
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
    /// An ancestor had a positive temporal offset, which is a later time.
    #[error("ancestor offset {offset} is in the future; next attempt: supply a lagged node")]
    FutureAncestor {
        /// The offending offset.
        offset: i32,
    },
}

/// Identify history for `target` under the already declared `bound`.
///
/// [`HistoryIdentificationTarget::PulseParentAdjustment`] returns
/// [`HistoryIdentification::PulseFallback`] and does not certify `ancestry`.
/// [`HistoryIdentificationTarget::FeedbackAncestry`] returns
/// [`HistoryIdentification::Certified`] only when `ancestry` is non-empty,
/// every lag is within `bound`, `initial_law` is present, and `frontier` is
/// [`AncestryFrontier::Closed`]. A lag past `bound`, or an
/// [`AncestryFrontier::Open`] frontier, returns
/// [`HistoryIdentification::Uncertified`] carrying that same bound. The
/// ancestry and the frontier are the caller's declarations; no graph is
/// searched here.
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
    frontier: AncestryFrontier,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    match target {
        HistoryIdentificationTarget::PulseParentAdjustment => Ok(pulse_fallback()),
        HistoryIdentificationTarget::FeedbackAncestry => {
            identify_feedback(ancestry, initial_law, bound, frontier)
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
    frontier: AncestryFrontier,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    let Some(initial_law) = initial_law else {
        return Err(HistoryIdentificationRefusal::MissingInitialLaw);
    };
    if ancestry.is_empty() {
        return Err(HistoryIdentificationRefusal::EmptyAncestry);
    }
    let missing = ancestry.iter().copied().find(|ancestor| ancestor.lag() > bound.steps());
    if let Some(missing_ancestor) = missing {
        return Ok(HistoryIdentification::Uncertified(past_bound_history(bound, missing_ancestor)));
    }
    if frontier == AncestryFrontier::Open {
        return Ok(HistoryIdentification::Uncertified(open_frontier_history(bound)));
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

fn open_frontier_history(bound: HistorySearchBound) -> UncertifiedHistory {
    let steps = bound.steps();
    let diagnostic = crate::result::search_bounded_diagnostic(
        HISTORY_SEARCH_BOUNDED_CODE,
        format!(
            "uncertified history: the search frontier is open at the declared bound {steps}; \
             ancestry may continue past it"
        ),
    )
    .with_fields([("bound", steps.to_string()), ("frontier", "open".to_owned())]);
    UncertifiedHistory { bound, reason: UncertifiedReason::FrontierOpen, diagnostic }
}

fn past_bound_history(bound: HistorySearchBound, missing: FeedbackAncestor) -> UncertifiedHistory {
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
    UncertifiedHistory { bound, reason: UncertifiedReason::AncestorPastBound(missing), diagnostic }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{InitialLaw, TemporalNodeKey, VariableId};

    use super::{
        AncestryFrontier, FeedbackAncestor, HistoryIdentification, HistoryIdentificationRefusal,
        HistoryIdentificationTarget, HistorySearchBound, UncertifiedReason, identify_history,
        is_feedback_certified, search_bound,
    };

    fn ancestor(variable: u32, lag: u32) -> FeedbackAncestor {
        let offset = -i32::try_from(lag).unwrap();
        FeedbackAncestor::new(TemporalNodeKey { variable: VariableId::from_raw(variable), offset })
            .unwrap()
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
            AncestryFrontier::Closed,
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
            AncestryFrontier::Closed,
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
            AncestryFrontier::Closed,
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
            AncestryFrontier::Closed,
        )
        .unwrap();
        assert!(!is_feedback_certified(&product));
        let HistoryIdentification::Uncertified(uncertified) = &product else {
            panic!("ancestry past the bound is uncertified");
        };
        assert_eq!(uncertified.search_bound(), bound);
        assert_eq!(uncertified.search_bound().steps(), bound.steps());
        assert_eq!(uncertified.missing_ancestor(), Some(missing));
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
            AncestryFrontier::Closed,
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
            AncestryFrontier::Closed,
        )
        .unwrap();
        assert!(!matches!(feedback, HistoryIdentification::PulseFallback(_)));
        let pulse = identify_history(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &[],
            None,
            HistorySearchBound::declared(1),
            AncestryFrontier::Closed,
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
            AncestryFrontier::Closed,
        )
        .unwrap();
        let past = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 1), ancestor(1, 9)],
            Some(&law),
            declared,
            AncestryFrontier::Closed,
        )
        .unwrap();
        let pulse = identify_history(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &[ancestor(1, 9)],
            None,
            declared,
            AncestryFrontier::Closed,
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

    #[test]
    fn lag_is_derived_from_the_offset_and_a_future_offset_is_refused() {
        let node = TemporalNodeKey { variable: VariableId::from_raw(3), offset: -5 };
        assert_eq!(FeedbackAncestor::new(node).unwrap().lag(), 5);
        let future = TemporalNodeKey { variable: VariableId::from_raw(3), offset: 1 };
        assert_eq!(
            FeedbackAncestor::new(future).unwrap_err(),
            HistoryIdentificationRefusal::FutureAncestor { offset: 1 }
        );
        // A declared lag of 0 cannot hide an offset of -7: the only lag is the offset's.
        let deep = TemporalNodeKey { variable: VariableId::from_raw(4), offset: -7 };
        let bound = HistorySearchBound::declared(3);
        let product = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &[ancestor(0, 0), FeedbackAncestor::new(deep).unwrap()],
            Some(&InitialLaw::contemporaneous_observed()),
            bound,
            AncestryFrontier::Closed,
        )
        .unwrap();
        let HistoryIdentification::Uncertified(uncertified) = product else {
            panic!("a lag of 7 is past a bound of 3");
        };
        assert_eq!(uncertified.missing_ancestor().map(FeedbackAncestor::lag), Some(7));
    }

    #[test]
    fn an_open_frontier_is_uncertified_even_when_every_listed_lag_fits() {
        // The caller listed ancestors at lags 0 and 3 under a bound of 3, but the search
        // stopped there: true ancestry continues to lag 7.
        let bound = HistorySearchBound::declared(3);
        let law = InitialLaw::named("origin").unwrap();
        let listed = [ancestor(0, 0), ancestor(1, 3)];
        let open = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &listed,
            Some(&law),
            bound,
            AncestryFrontier::Open,
        )
        .unwrap();
        assert!(!is_feedback_certified(&open));
        let HistoryIdentification::Uncertified(uncertified) = &open else {
            panic!("an open frontier is uncertified");
        };
        assert_eq!(uncertified.reason(), UncertifiedReason::FrontierOpen);
        assert!(uncertified.missing_ancestor().is_none());
        assert_eq!(uncertified.search_bound(), bound);
        assert!(uncertified.diagnostic().code.ends_with(".search_bounded"));
        assert_eq!(bound_field(&open), Some(3));

        let closed = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &listed,
            Some(&law),
            bound,
            AncestryFrontier::Closed,
        )
        .unwrap();
        assert!(is_feedback_certified(&closed));
    }
}
