//! `antecedent_core::recalc::Revalidation::worst` combines classes as a join, and
//! `ChangeKind::revalidation` never leaves a dependent still valid.
//!
//! Both domains are finite, so these harnesses cover every input. The Lean statements
//! are `Antecedent.Revalidation.worst_*` and `ChangeKind.revalidation_lowers`; these two
//! items are checked here rather than translated, because a second Aeneas translation of
//! `Revalidation` beside the one in `formal/rust` would clash.

use antecedent_core::recalc::{ChangeKind, Revalidation};

fn any_class() -> Revalidation {
    match kani::any::<u8>() % 3 {
        0 => Revalidation::StillValid,
        1 => Revalidation::Recompute,
        _ => Revalidation::Withdrawn,
    }
}

fn any_change() -> ChangeKind {
    match kani::any::<u8>() % 3 {
        0 => ChangeKind::Added,
        1 => ChangeKind::Removed,
        _ => ChangeKind::Modified,
    }
}

/// Severity: larger is worse, stated independently of the derived order.
fn rank(class: Revalidation) -> u8 {
    match class {
        Revalidation::StillValid => 0,
        Revalidation::Recompute => 1,
        Revalidation::Withdrawn => 2,
    }
}

/// `worst` is the more severe class: commutative, associative and idempotent, with
/// still valid as identity and withdrawn absorbing.
#[kani::proof]
fn worst_is_a_join() {
    let (a, b, c) = (any_class(), any_class(), any_class());
    let w = a.worst(b);
    assert!(rank(w) == rank(a).max(rank(b)));
    assert!(w == b.worst(a));
    assert!(a.worst(b).worst(c) == a.worst(b.worst(c)));
    assert!(a.worst(a) == a);
    assert!(a.worst(Revalidation::StillValid) == a);
    assert!(a.worst(Revalidation::Withdrawn) == Revalidation::Withdrawn);
}

/// A change never leaves its dependents still valid, and withdraws them exactly when
/// the input is gone.
#[kani::proof]
fn a_change_never_keeps_a_dependent_valid() {
    let change = any_change();
    let class = change.revalidation();
    assert!(class != Revalidation::StillValid);
    assert!((class == Revalidation::Withdrawn) == (change == ChangeKind::Removed));
}
