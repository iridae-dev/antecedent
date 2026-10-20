//! Delta contract of every incrementally kept route.
//!
//! A kept summary or cache answers each change in exactly one of four ways: it is
//! `Unaffected`, it is updated `Exactly` (equal to a fresh reconstruction from the retained
//! observations within [`crate::merge_within_tolerance`]), it is dropped and recomputed from
//! the retained observations, or the change is refused because it breaks a premise of the
//! route. A stale result is never a fifth way. The table below is the contract; the routes
//! that carry a summary apply it in [`crate::OlsRefresh`] and [`crate::ConjugateOlsPosterior`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// One incrementally kept route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum RefreshRoute {
    /// Gram, cross-product and centered scatter of a linear least-squares design.
    LinearOlsGram,
    /// Streaming mean and centered scatter of a vector series.
    StreamingCovariance,
    /// Normal-inverse-gamma posterior of a Gaussian linear model, kept as additive natural
    /// parameters.
    ConjugateGaussianPosterior,
    /// Lagged sample-index metadata.
    LagIndexCache,
    /// Local graph-score slots.
    GraphScoreCache,
    /// Particle-filter state slots.
    ParticleFilter,
}

impl RefreshRoute {
    /// Every route, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::LinearOlsGram,
        Self::StreamingCovariance,
        Self::ConjugateGaussianPosterior,
        Self::LagIndexCache,
        Self::GraphScoreCache,
        Self::ParticleFilter,
    ];
}

/// One change to the inputs a route was built on.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum RefreshChange {
    /// New rows after the last retained row, from the same source.
    AppendRows,
    /// A summary of a contiguous window of the same source and columns.
    AppendWindow,
    /// A previously summarized row is removed.
    EvictRow,
    /// The weight of a summarized row, or every weight, changed.
    WeightChange,
    /// The row or column mask changed.
    MaskChange,
    /// A design column was inserted.
    ColumnInserted,
    /// The data source was replaced.
    SourceChanged,
    /// The observation period moved.
    PeriodChanged,
    /// The prior changed.
    PriorChanged,
    /// Observations a route already used are offered to it a second time.
    ObservationsReused,
}

impl RefreshChange {
    /// Every change, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::AppendRows,
        Self::AppendWindow,
        Self::EvictRow,
        Self::WeightChange,
        Self::MaskChange,
        Self::ColumnInserted,
        Self::SourceChanged,
        Self::PeriodChanged,
        Self::PriorChanged,
        Self::ObservationsReused,
    ];
}

/// What a route does with one change.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshDisposition {
    /// The route does not read the changed input.
    Unaffected,
    /// The route updates in place and equals a fresh reconstruction.
    ExactUpdate,
    /// The route is dropped and rebuilt from the retained observations.
    InvalidateAndRecompute {
        /// Why no in-place update is claimed.
        reason: &'static str,
    },
    /// The change breaks a premise of the route. Nothing is updated.
    Refuse {
        /// The premise the change breaks.
        premise: &'static str,
        /// The attempt that reaches the same end state.
        next_attempt: &'static str,
    },
}

const NO_DOWNDATE: &str = "no exact downdate of the summary is claimed";
const WEIGHT_MASK: &str = "a weight or mask enters every summarized term";
const WIDER_DESIGN: &str = "a new column changes every summarized cross-product";
const NEW_SOURCE: &str = "the summary describes the old source only";
const NEW_PERIOD: &str = "a summary carried across periods asserts one mechanism for both";
const DATA_BOUND: &str = "the cache is bound to the data it was filled from";
const REUSE_PREMISE: &str = "each observation enters a route once";
const REUSE_NEXT: &str = "drop the duplicate block, or rebuild the route from the distinct blocks";
const PRIOR_PREMISE: &str = "a posterior is the prior times the likelihood of the observations \
                             used; a different prior is a different posterior";
const PRIOR_NEXT: &str = "rebuild from every retained block under the new prior";

/// The contract of `route` for `change`.
#[must_use]
pub const fn route_contract(route: RefreshRoute, change: RefreshChange) -> RefreshDisposition {
    use RefreshChange as C;
    use RefreshDisposition as D;
    use RefreshRoute as R;
    match route {
        R::LinearOlsGram | R::StreamingCovariance => match change {
            C::AppendRows | C::AppendWindow => D::ExactUpdate,
            C::EvictRow => D::InvalidateAndRecompute { reason: NO_DOWNDATE },
            C::WeightChange | C::MaskChange => D::InvalidateAndRecompute { reason: WEIGHT_MASK },
            C::ColumnInserted => D::InvalidateAndRecompute { reason: WIDER_DESIGN },
            C::SourceChanged => D::InvalidateAndRecompute { reason: NEW_SOURCE },
            C::PeriodChanged => D::InvalidateAndRecompute { reason: NEW_PERIOD },
            C::PriorChanged => D::Unaffected,
            C::ObservationsReused => D::Refuse { premise: REUSE_PREMISE, next_attempt: REUSE_NEXT },
        },
        R::ConjugateGaussianPosterior => match change {
            C::AppendRows | C::AppendWindow => D::ExactUpdate,
            C::EvictRow => D::InvalidateAndRecompute { reason: NO_DOWNDATE },
            C::WeightChange | C::MaskChange => D::InvalidateAndRecompute { reason: WEIGHT_MASK },
            C::ColumnInserted => D::InvalidateAndRecompute { reason: WIDER_DESIGN },
            C::SourceChanged => D::InvalidateAndRecompute { reason: NEW_SOURCE },
            C::PeriodChanged => D::InvalidateAndRecompute { reason: NEW_PERIOD },
            C::PriorChanged => D::Refuse { premise: PRIOR_PREMISE, next_attempt: PRIOR_NEXT },
            C::ObservationsReused => D::Refuse { premise: REUSE_PREMISE, next_attempt: REUSE_NEXT },
        },
        R::LagIndexCache | R::GraphScoreCache | R::ParticleFilter => match change {
            C::PriorChanged => D::Unaffected,
            C::ObservationsReused => D::Refuse { premise: REUSE_PREMISE, next_attempt: REUSE_NEXT },
            _ => D::InvalidateAndRecompute { reason: DATA_BOUND },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_appends_update_in_place_and_every_route_answers_every_change() {
        for route in RefreshRoute::ALL {
            for change in RefreshChange::ALL {
                let exact = route_contract(route, change) == RefreshDisposition::ExactUpdate;
                let append =
                    matches!(change, RefreshChange::AppendRows | RefreshChange::AppendWindow);
                let summary = matches!(
                    route,
                    RefreshRoute::LinearOlsGram
                        | RefreshRoute::StreamingCovariance
                        | RefreshRoute::ConjugateGaussianPosterior
                );
                assert_eq!(exact, append && summary, "{route:?} {change:?}");
            }
        }
    }

    #[test]
    fn a_posterior_refuses_a_prior_change_that_a_gram_ignores() {
        assert!(matches!(
            route_contract(RefreshRoute::ConjugateGaussianPosterior, RefreshChange::PriorChanged),
            RefreshDisposition::Refuse { .. }
        ));
        assert_eq!(
            route_contract(RefreshRoute::LinearOlsGram, RefreshChange::PriorChanged),
            RefreshDisposition::Unaffected
        );
    }
}
