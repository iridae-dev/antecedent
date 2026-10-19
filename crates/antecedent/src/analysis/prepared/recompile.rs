//! Dependency-directed recompilation for a prepared lifecycle.
//!
//! Each product names the inputs it reads and the products it is built from.
//! A change invalidates the products that read that input and their
//! descendants. A product that does not read the changed input stays, including
//! structural identification that does not use a prior and any product outside
//! the source, period, and history chain.
//!
//! The static ordinary-least-squares route consults this state before fitting.
//! Utility and action are decision inputs, so a change to either of them leaves
//! the causal fit in place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

/// An input the prepared lifecycle watches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LifecycleInput {
    /// Decision utility. It does not enter the causal fit.
    Utility,
    /// Intervention action. It does not enter the causal fit.
    Action,
    /// Model prior. It enters prior-bound structure and the causal fit.
    Prior,
    /// Observation source.
    Source,
    /// Time period the source covers.
    Period,
    /// History or initial state the fit reads.
    History,
}

/// A product retained by the prepared lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LifecycleProduct {
    /// Identification that does not read the prior.
    Structural,
    /// Identification that is valid only for the current prior.
    PriorStructural,
    /// Causal fit. On the linear/Gaussian route this is the ordinary-least-squares fit.
    CausalFit,
    /// Decision scored from the causal fit, the action, and the utility.
    Decision,
    /// A product that reads none of the watched inputs.
    Unrelated,
}

impl LifecycleProduct {
    const ALL: [Self; 5] =
        [Self::Structural, Self::PriorStructural, Self::CausalFit, Self::Decision, Self::Unrelated];

    const fn bit(self) -> u8 {
        match self {
            Self::Structural => 1_u8 << 0,
            Self::PriorStructural => 1_u8 << 1,
            Self::CausalFit => 1_u8 << 2,
            Self::Decision => 1_u8 << 3,
            Self::Unrelated => 1_u8 << 4,
        }
    }

    /// Inputs this product reads directly.
    const fn reads(self, input: LifecycleInput) -> bool {
        matches!(
            (self, input),
            (Self::PriorStructural, LifecycleInput::Prior)
                | (
                    Self::CausalFit,
                    LifecycleInput::Prior
                        | LifecycleInput::Source
                        | LifecycleInput::Period
                        | LifecycleInput::History
                )
                | (Self::Decision, LifecycleInput::Utility | LifecycleInput::Action)
        )
    }

    /// Whether `self` is built from `parent` and must drop when `parent` drops.
    const fn depends_on(self, parent: Self) -> bool {
        matches!(
            (self, parent),
            (Self::CausalFit, Self::Structural | Self::PriorStructural)
                | (Self::Decision, Self::CausalFit)
        )
    }
}

/// Validity of the prepared products, and how many causal fits have run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedRecompile {
    valid: u8,
    causal_fits: u64,
}

impl PreparedRecompile {
    /// State after the linear/Gaussian lower, before the causal fit.
    ///
    /// Structural identification, the prior-bound structure that matches this
    /// lower, and the unrelated product are valid. The fit and the decision
    /// have not run.
    #[must_use = "install this state on the lowered route"]
    pub(crate) const fn after_lower() -> Self {
        Self {
            valid: LifecycleProduct::Structural.bit()
                | LifecycleProduct::PriorStructural.bit()
                | LifecycleProduct::Unrelated.bit(),
            causal_fits: 0,
        }
    }

    /// Whether the causal fit may be reused.
    #[must_use = "reuse the retained fit only when this is true"]
    pub(crate) const fn reuses_causal_fit(self) -> bool {
        self.is_valid(LifecycleProduct::CausalFit)
    }

    /// Whether `product` survived the changes applied so far.
    #[must_use = "an invalid product has to be rebuilt"]
    pub(crate) const fn is_valid(self, product: LifecycleProduct) -> bool {
        self.valid & product.bit() != 0
    }

    /// Causal fits recorded by [`Self::retain_causal_fit`].
    #[must_use = "compare this with the count from before the change"]
    pub(crate) const fn causal_fits(self) -> u64 {
        self.causal_fits
    }

    /// Invalidate every product that reads `input`, then their descendants.
    pub(crate) fn apply(&mut self, input: LifecycleInput) {
        let mut marked = 0_u8;
        for product in LifecycleProduct::ALL {
            if product.reads(input) {
                marked |= product.bit();
            }
        }
        loop {
            let mut grew = false;
            for product in LifecycleProduct::ALL {
                if marked & product.bit() != 0 {
                    continue;
                }
                for parent in LifecycleProduct::ALL {
                    if product.depends_on(parent) && marked & parent.bit() != 0 {
                        marked |= product.bit();
                        grew = true;
                        break;
                    }
                }
            }
            if !grew {
                break;
            }
        }
        self.valid &= !marked;
    }

    /// Record one causal fit and the decision built from it.
    pub(crate) const fn retain_causal_fit(&mut self) {
        self.valid |= LifecycleProduct::CausalFit.bit() | LifecycleProduct::Decision.bit();
        self.causal_fits = self.causal_fits.saturating_add(1);
    }

    /// Record a decision rebuilt from a causal fit that is still valid.
    pub(crate) const fn retain_decision(&mut self) {
        if self.reuses_causal_fit() {
            self.valid |= LifecycleProduct::Decision.bit();
        }
    }
}
