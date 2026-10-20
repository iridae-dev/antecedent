//! History and initial-state identification for temporal routes.
//!
//! [`HistoryIdentification::Certified`] is the feedback ancestry a search walked
//! on a [`TemporalDag`], together with a declared [`InitialLaw`], and only when
//! that search closed: every reached node's parents lie inside
//! [`HistorySearchBound`]. A caller-supplied ancestor list is not an input and
//! does not certify. Each ancestor's lag is the temporal offset of its node on
//! the graph.
//!
//! An ancestor that is not a node of the graph, a node whose lag is past the
//! bound, or a frontier left open because a parent lies past the bound, is
//! [`HistoryIdentification::Uncertified`]. The product keeps the bound that was
//! given. Nothing here increases that bound or treats the partial history as
//! identified.
//!
//! [`HistoryIdentification::PulseFallback`] is only the existing narrow pulse
//! route `identify_pulse_by_parent_adjustment`
//! ([`PARENT_ADJUSTMENT_RULE`](crate::temporal_backdoor::PARENT_ADJUSTMENT_RULE)).
//! It does not read the graph and does not certify feedback ancestry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::VecDeque;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, InitialLaw, TemporalNodeKey,
};
use antecedent_graph::{DenseNodeId, TemporalDag};
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

/// Bound on feedback-ancestry search, in lags.
///
/// The stored step count is the caller's bound. The search does not return a
/// larger one, and a larger ancestry is not certified by widening it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HistorySearchBound {
    steps: u32,
}

impl HistorySearchBound {
    /// Bound of `steps` lags.
    #[must_use]
    pub const fn declared(steps: u32) -> Self {
        Self { steps }
    }

    /// Step count. This is the bound that was given, not a wider one.
    #[must_use]
    pub const fn steps(self) -> u32 {
        self.steps
    }
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

/// Finite ancestry the graph search closed, and the declared initial law.
///
/// The ancestry is non-empty, every lag is inside the bound, the law was
/// declared, and the parent walk found no further parent past the bound.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedHistory {
    ancestry: Vec<FeedbackAncestor>,
    initial_law: InitialLaw,
    bound: HistorySearchBound,
}

impl CertifiedHistory {
    /// Ancestors the search reached, breadth-first from the focus, parents in
    /// graph order. The focus is first.
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
    /// The node under search lies past the bound.
    AncestorPastBound(FeedbackAncestor),
    /// The walk reached a node inside the bound that still has a parent past
    /// the bound, so the frontier did not close.
    FrontierOpen,
    /// The node under search is not a node of the graph.
    AncestorAbsent(FeedbackAncestor),
}

/// Feedback ancestry that the search did not finitely certify.
///
/// `bound` is the bound that was given. This is not
/// [`HistoryIdentification::Certified`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UncertifiedHistory {
    bound: HistorySearchBound,
    reason: UncertifiedReason,
    diagnostic: Diagnostic,
}

impl UncertifiedHistory {
    /// Search bound, unchanged.
    #[must_use]
    pub const fn search_bound(&self) -> HistorySearchBound {
        self.bound
    }

    /// Why the ancestry is not certified.
    #[must_use]
    pub const fn reason(&self) -> UncertifiedReason {
        self.reason
    }

    /// Ancestor the search could not certify, when one is known.
    ///
    /// [`Some`] for an ancestor past the bound or absent from the graph.
    /// [`None`] when the frontier stayed open.
    #[must_use]
    pub const fn missing_ancestor(&self) -> Option<FeedbackAncestor> {
        match self.reason {
            UncertifiedReason::AncestorPastBound(ancestor)
            | UncertifiedReason::AncestorAbsent(ancestor) => Some(ancestor),
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
    /// Finite ancestry the graph search closed, and a declared initial law.
    Certified(CertifiedHistory),
    /// Existing narrow pulse route. Not a feedback-ancestry certificate.
    PulseFallback(PulseFallbackHistory),
    /// The graph search did not close inside the given bound.
    Uncertified(UncertifiedHistory),
}

/// Why history identification was refused before a product existed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum HistoryIdentificationRefusal {
    /// Feedback ancestry was requested without an initial law.
    #[error("initial law is missing; next attempt: declare the initial law")]
    MissingInitialLaw,
    /// An ancestor had a positive temporal offset, which is a later time.
    #[error("ancestor offset {offset} is in the future; next attempt: supply a lagged node")]
    FutureAncestor {
        /// The offending offset.
        offset: i32,
    },
}

/// Identify history for `target` by searching `graph` from `focus`.
///
/// [`HistoryIdentificationTarget::PulseParentAdjustment`] returns
/// [`HistoryIdentification::PulseFallback`] and does not read `graph` or
/// certify ancestry. [`HistoryIdentificationTarget::FeedbackAncestry`] walks
/// parents on `graph` from `focus`. It returns
/// [`HistoryIdentification::Certified`] only when `focus` is a node of
/// `graph`, every reached lag is within `bound`, `initial_law` is present, and
/// no reached node still has a parent past `bound`. A `focus` absent from
/// `graph`, a `focus` whose lag is past `bound`, or a parent past `bound`
/// returns [`HistoryIdentification::Uncertified`] carrying that same bound.
///
/// # Errors
///
/// [`HistoryIdentificationRefusal::MissingInitialLaw`] when feedback ancestry
/// is requested and `initial_law` is absent.
/// [`HistoryIdentificationRefusal::FutureAncestor`] when `focus` has a positive
/// offset.
#[must_use = "a refusal is not a history identification"]
pub fn identify_history(
    target: HistoryIdentificationTarget,
    graph: &TemporalDag,
    focus: TemporalNodeKey,
    initial_law: Option<&InitialLaw>,
    bound: HistorySearchBound,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    match target {
        HistoryIdentificationTarget::PulseParentAdjustment => Ok(pulse_fallback()),
        HistoryIdentificationTarget::FeedbackAncestry => {
            identify_feedback(graph, focus, initial_law, bound)
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

/// Search bound carried by `identification`, when it has one.
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
    graph: &TemporalDag,
    focus: TemporalNodeKey,
    initial_law: Option<&InitialLaw>,
    bound: HistorySearchBound,
) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
    let Some(initial_law) = initial_law else {
        return Err(HistoryIdentificationRefusal::MissingInitialLaw);
    };
    let focus_ancestor = FeedbackAncestor::new(focus)?;
    let Some(start) = find_node(graph, focus) else {
        return Ok(HistoryIdentification::Uncertified(absent_history(bound, focus_ancestor)));
    };
    match walk_ancestry(graph, start, bound) {
        Ok(ancestry) => Ok(HistoryIdentification::Certified(CertifiedHistory {
            ancestry,
            initial_law: initial_law.clone(),
            bound,
        })),
        Err(WalkFailure::PastBound(missing)) => {
            Ok(HistoryIdentification::Uncertified(past_bound_history(bound, missing)))
        }
        Err(WalkFailure::Open) => {
            Ok(HistoryIdentification::Uncertified(open_frontier_history(bound)))
        }
    }
}

/// Why the parent walk did not close.
enum WalkFailure {
    PastBound(FeedbackAncestor),
    Open,
}

fn find_node(graph: &TemporalDag, key: TemporalNodeKey) -> Option<DenseNodeId> {
    (0..graph.node_count()).find_map(|index| {
        let id = DenseNodeId::try_from_usize(index).ok()?;
        graph.temporal_key(id).is_some_and(|found| found == key).then_some(id)
    })
}

fn named_ancestor(graph: &TemporalDag, id: DenseNodeId) -> Option<FeedbackAncestor> {
    FeedbackAncestor::new(graph.temporal_key(id)?).ok()
}

/// Walk parents from `start`. A parent past `bound` leaves the frontier open
/// and drops the nodes already reached: that partial list is not a certificate.
fn walk_ancestry(
    graph: &TemporalDag,
    start: DenseNodeId,
    bound: HistorySearchBound,
) -> Result<Vec<FeedbackAncestor>, WalkFailure> {
    let mut ancestry = Vec::new();
    let mut seen = vec![false; graph.node_count()];
    let mut queue = VecDeque::new();
    seen[start.as_usize()] = true;
    queue.push_back(start);
    let limit = bound.steps();
    while let Some(id) = queue.pop_front() {
        let Some(ancestor) = named_ancestor(graph, id) else {
            return Err(WalkFailure::Open);
        };
        if ancestor.lag() > limit {
            return Err(WalkFailure::PastBound(ancestor));
        }
        let mut frontier_open = false;
        for &parent in graph.parents(id) {
            let within_bound = named_ancestor(graph, parent)
                .is_some_and(|parent_ancestor| parent_ancestor.lag() <= limit);
            if within_bound {
                let index = parent.as_usize();
                if !seen[index] {
                    seen[index] = true;
                    queue.push_back(parent);
                }
            } else {
                frontier_open = true;
            }
        }
        if frontier_open {
            return Err(WalkFailure::Open);
        }
        ancestry.push(ancestor);
    }
    Ok(ancestry)
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

fn absent_history(bound: HistorySearchBound, missing: FeedbackAncestor) -> UncertifiedHistory {
    let steps = bound.steps();
    let lag = missing.lag();
    let variable = missing.node().variable.raw();
    let diagnostic = crate::result::search_bounded_diagnostic(
        HISTORY_SEARCH_BOUNDED_CODE,
        format!(
            "uncertified history: ancestor variable {variable} at lag {lag} is absent from the \
             graph; the search did not close"
        ),
    )
    .with_fields([
        ("bound", steps.to_string()),
        ("missing_ancestor_variable", variable.to_string()),
        ("missing_ancestor_lag", lag.to_string()),
        ("frontier", "absent".to_owned()),
    ]);
    UncertifiedHistory { bound, reason: UncertifiedReason::AncestorAbsent(missing), diagnostic }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{InitialLaw, Lag, TemporalNodeKey, VariableId};
    use antecedent_graph::{DenseNodeId, TemporalDag};

    use super::{
        FeedbackAncestor, HistoryIdentification, HistoryIdentificationRefusal,
        HistoryIdentificationTarget, HistorySearchBound, UncertifiedReason, identify_history,
        is_feedback_certified, search_bound,
    };

    fn add(graph: &mut TemporalDag, variable: u32, lag: u32) -> DenseNodeId {
        graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    }

    fn key(variable: u32, lag: u32) -> TemporalNodeKey {
        let offset = -i32::try_from(lag).unwrap();
        TemporalNodeKey { variable: VariableId::from_raw(variable), offset }
    }

    fn ancestor(variable: u32, lag: u32) -> FeedbackAncestor {
        FeedbackAncestor::new(key(variable, lag)).unwrap()
    }

    /// Focus at lag 0, parents at lags 1 and 2, and a non-ancestor child.
    /// `deep_lag` adds a parent of the lag-2 node.
    fn ancestry_graph(deep_lag: Option<u32>) -> TemporalDag {
        let mut graph = TemporalDag::empty();
        let past = add(&mut graph, 1, 2);
        let mid = add(&mut graph, 2, 1);
        let focus = add(&mut graph, 0, 0);
        let child = add(&mut graph, 3, 0);
        graph.insert_directed(past, focus).unwrap();
        graph.insert_directed(mid, focus).unwrap();
        graph.insert_directed(past, mid).unwrap();
        graph.insert_directed(focus, child).unwrap();
        if let Some(lag) = deep_lag {
            let deep = add(&mut graph, 7, lag);
            graph.insert_directed(deep, past).unwrap();
        }
        graph
    }

    fn identify(
        target: HistoryIdentificationTarget,
        graph: &TemporalDag,
        variable: u32,
        lag: u32,
        law: Option<&InitialLaw>,
        bound: HistorySearchBound,
    ) -> Result<HistoryIdentification, HistoryIdentificationRefusal> {
        identify_history(target, graph, key(variable, lag), law, bound)
    }

    fn bound_field(product: &HistoryIdentification) -> Option<u32> {
        let HistoryIdentification::Uncertified(uncertified) = product else {
            return None;
        };
        uncertified.diagnostic().fields.iter().find_map(|(field, value)| {
            (field.as_ref() == "bound").then(|| value.as_ref().parse().unwrap())
        })
    }

    #[test]
    fn a_search_that_closes_on_the_graph_is_certified() {
        let bound = HistorySearchBound::declared(2);
        let law = InitialLaw::named("origin").unwrap();
        let graph = ancestry_graph(None);
        let product = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            0,
            0,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(is_feedback_certified(&product));
        let HistoryIdentification::Certified(certified) = product else {
            panic!("a search that closes on the graph is certified");
        };
        let found: Vec<(u32, u32)> = certified
            .ancestry()
            .iter()
            .map(|node| (node.node().variable.raw(), node.lag()))
            .collect();
        assert_eq!(found, vec![(0, 0), (1, 2), (2, 1)]);
        assert!(certified.ancestry().iter().all(|node| node.node().variable.raw() != 3));
        assert_eq!(certified.initial_law(), &law);
        assert_eq!(certified.search_bound(), bound);
    }

    #[test]
    fn a_caller_named_ancestor_list_does_not_certify() {
        let bound = HistorySearchBound::declared(2);
        let law = InitialLaw::named("origin").unwrap();
        let named = [ancestor(0, 0), ancestor(1, 2), ancestor(2, 1)];
        let mut graph = TemporalDag::empty();
        let _focus = add(&mut graph, 0, 0);
        let _past = add(&mut graph, 1, 2);
        let _mid = add(&mut graph, 2, 1);
        let product = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            0,
            0,
            Some(&law),
            bound,
        )
        .unwrap();
        let HistoryIdentification::Certified(certified) = product else {
            panic!("a node with no parents closes inside the bound");
        };
        let found: Vec<(u32, u32)> = certified
            .ancestry()
            .iter()
            .map(|node| (node.node().variable.raw(), node.lag()))
            .collect();
        let caller_list: Vec<(u32, u32)> =
            named.iter().map(|node| (node.node().variable.raw(), node.lag())).collect();
        assert_ne!(found, caller_list);
        assert_eq!(found, vec![(0, 0)]);
    }

    #[test]
    fn an_ancestor_absent_from_the_graph_is_uncertified() {
        let bound = HistorySearchBound::declared(2);
        let law = InitialLaw::named("origin").unwrap();
        let graph = ancestry_graph(None);
        let present = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            0,
            0,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(is_feedback_certified(&present));

        let absent = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            9,
            1,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(!is_feedback_certified(&absent));
        let HistoryIdentification::Uncertified(uncertified) = &absent else {
            panic!("an ancestor absent from the graph is uncertified");
        };
        assert_eq!(uncertified.reason(), UncertifiedReason::AncestorAbsent(ancestor(9, 1)));
        assert_eq!(uncertified.missing_ancestor(), Some(ancestor(9, 1)));
        assert_eq!(uncertified.search_bound(), bound);
        assert!(uncertified.diagnostic().message.contains("absent"));
        assert!(uncertified.diagnostic().code.ends_with(".search_bounded"));
        assert_eq!(bound_field(&absent), Some(bound.steps()));

        // Variable 1 is on the graph at lag 2. Naming it at lag 1 is a different node.
        let wrong_lag = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            1,
            1,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(!is_feedback_certified(&wrong_lag));
        let HistoryIdentification::Uncertified(uncertified) = wrong_lag else {
            panic!("a lag the graph does not contain is uncertified");
        };
        assert_eq!(uncertified.reason(), UncertifiedReason::AncestorAbsent(ancestor(1, 1)));
    }

    #[test]
    fn an_open_frontier_is_uncertified() {
        let bound = HistorySearchBound::declared(2);
        let law = InitialLaw::contemporaneous_observed();
        let open_graph = ancestry_graph(Some(5));
        let open = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &open_graph,
            0,
            0,
            Some(&law),
            bound,
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
        assert_eq!(bound_field(&open), Some(bound.steps()));

        let closed = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &ancestry_graph(None),
            0,
            0,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(is_feedback_certified(&closed));
    }

    #[test]
    fn missing_initial_law_is_refused() {
        let error = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &ancestry_graph(None),
            0,
            0,
            None,
            HistorySearchBound::declared(4),
        )
        .unwrap_err();
        assert_eq!(error, HistoryIdentificationRefusal::MissingInitialLaw);
    }

    #[test]
    fn an_ancestor_past_the_bound_is_uncertified_and_the_bound_is_unchanged() {
        let bound = HistorySearchBound::declared(2);
        let mut graph = TemporalDag::empty();
        let deep = add(&mut graph, 7, 5);
        let nearer = add(&mut graph, 1, 2);
        graph.insert_directed(deep, nearer).unwrap();
        let product = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            7,
            5,
            Some(&InitialLaw::contemporaneous_observed()),
            bound,
        )
        .unwrap();
        assert!(!is_feedback_certified(&product));
        let HistoryIdentification::Uncertified(uncertified) = &product else {
            panic!("an ancestor past the bound is uncertified");
        };
        assert_eq!(uncertified.search_bound(), bound);
        assert_eq!(uncertified.search_bound().steps(), bound.steps());
        assert_eq!(uncertified.missing_ancestor(), Some(ancestor(7, 5)));
        assert_eq!(uncertified.reason(), UncertifiedReason::AncestorPastBound(ancestor(7, 5)));
        let diagnostic = uncertified.diagnostic();
        assert!(diagnostic.code.ends_with(".search_bounded"));
        assert!(diagnostic.message.contains("uncertified history"));
        assert_eq!(bound_field(&product), Some(bound.steps()));
    }

    #[test]
    fn pulse_fallback_is_not_feedback_certified() {
        let graph = ancestry_graph(None);
        let law = InitialLaw::contemporaneous_observed();
        let bound = HistorySearchBound::declared(2);
        let feedback = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            0,
            0,
            Some(&law),
            bound,
        )
        .unwrap();
        assert!(is_feedback_certified(&feedback));
        let product = identify(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &graph,
            0,
            9,
            None,
            bound,
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
        let graph = ancestry_graph(None);
        let feedback = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            0,
            0,
            Some(&law),
            HistorySearchBound::declared(2),
        )
        .unwrap();
        assert!(!matches!(feedback, HistoryIdentification::PulseFallback(_)));
        let pulse = identify(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &TemporalDag::empty(),
            0,
            0,
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
        let mut within_graph = TemporalDag::empty();
        let parent = add(&mut within_graph, 1, 4);
        let focus = add(&mut within_graph, 0, 0);
        within_graph.insert_directed(parent, focus).unwrap();
        let within = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &within_graph,
            0,
            0,
            Some(&law),
            declared,
        )
        .unwrap();
        let mut past_graph = TemporalDag::empty();
        let deep = add(&mut past_graph, 1, 9);
        let later = add(&mut past_graph, 0, 1);
        past_graph.insert_directed(deep, later).unwrap();
        let past = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &past_graph,
            1,
            9,
            Some(&law),
            declared,
        )
        .unwrap();
        let pulse = identify(
            HistoryIdentificationTarget::PulseParentAdjustment,
            &past_graph,
            1,
            9,
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

    #[test]
    fn lag_is_derived_from_the_offset_and_a_future_offset_is_refused() {
        let node = TemporalNodeKey { variable: VariableId::from_raw(3), offset: -5 };
        assert_eq!(FeedbackAncestor::new(node).unwrap().lag(), 5);
        let future = TemporalNodeKey { variable: VariableId::from_raw(3), offset: 1 };
        assert_eq!(
            FeedbackAncestor::new(future).unwrap_err(),
            HistoryIdentificationRefusal::FutureAncestor { offset: 1 }
        );
        let refused = identify_history(
            HistoryIdentificationTarget::FeedbackAncestry,
            &TemporalDag::empty(),
            future,
            Some(&InitialLaw::contemporaneous_observed()),
            HistorySearchBound::declared(3),
        )
        .unwrap_err();
        assert_eq!(refused, HistoryIdentificationRefusal::FutureAncestor { offset: 1 });

        let mut graph = TemporalDag::empty();
        add(&mut graph, 4, 7);
        let bound = HistorySearchBound::declared(3);
        let product = identify(
            HistoryIdentificationTarget::FeedbackAncestry,
            &graph,
            4,
            7,
            Some(&InitialLaw::contemporaneous_observed()),
            bound,
        )
        .unwrap();
        let HistoryIdentification::Uncertified(uncertified) = product else {
            panic!("a lag of 7 is past a bound of 3");
        };
        assert_eq!(uncertified.missing_ancestor().map(FeedbackAncestor::lag), Some(7));
    }
}
