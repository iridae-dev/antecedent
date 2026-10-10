//! Exact evaluation of the ordered-response recovery row (2.3 B2 / X10 second
//! row). The graph decision, the formula and its derivation live in
//! [`antecedent_identify::recovery_chain`]; this module evaluates the formula on
//! an exact nine-cell observed pattern law.
//!
//! The pattern law is `P(R1, R2, X*1, X*2)` with the deterministic proxy model
//! built in: the nine cells are `(R1=1, R2=1, x1, x2)` (four), `(R1=1, R2=0, x1)`
//! (two), `(R1=0, R2=1, x2)` (two) and `(R1=0, R2=0)`; a cell the proxy model
//! forbids cannot be expressed. Positivity (the head response and every
//! complete-case cell have mass, and the head-variable margin used by a
//! dependent tail has mass) is checked before any division. Exact laws only: no
//! sampled provider is composed here, and calibration of one is separate,
//! measured only at the release cut, and unmeasured and closed in this slice.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_identify::{ChainRecoveryDetail, ChainRecoveryError, ChainRecoveryPlan};

/// Tolerance on the total mass of an observed pattern law.
const MASS_TOLERANCE: f64 = 1e-9;

fn refuse(detail: ChainRecoveryDetail, message: impl Into<String>) -> ChainRecoveryError {
    ChainRecoveryError::new(detail, message)
}

/// An exact observed pattern law over the nine proxy cells. Axis order is the
/// declaration order of the plan's query (first, second).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainPatternLaw {
    both: [[f64; 2]; 2],
    only_first: [f64; 2],
    only_second: [f64; 2],
    neither: f64,
}

impl ChainPatternLaw {
    /// The law from its cells: `both[x1][x2] = P(R1=1, R2=1, X*1=x1, X*2=x2)`,
    /// `only_first[x1] = P(R1=1, R2=0, X*1=x1)`,
    /// `only_second[x2] = P(R1=0, R2=1, X*2=x2)`, `neither = P(R1=0, R2=0)`.
    ///
    /// # Errors
    ///
    /// `invalid_observed_law` when a cell is negative or non-finite or the
    /// cells do not sum to one within `1e-9`.
    pub fn new(
        both: [[f64; 2]; 2],
        only_first: [f64; 2],
        only_second: [f64; 2],
        neither: f64,
    ) -> Result<Self, ChainRecoveryError> {
        let cells: Vec<f64> = both
            .iter()
            .flatten()
            .chain(only_first.iter())
            .chain(only_second.iter())
            .copied()
            .chain(std::iter::once(neither))
            .collect();
        if cells.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err(refuse(
                ChainRecoveryDetail::InvalidObservedLaw,
                "an observed pattern cell is negative or not finite",
            ));
        }
        let total: f64 = cells.iter().sum();
        if (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(refuse(
                ChainRecoveryDetail::InvalidObservedLaw,
                format!("the observed pattern cells sum to {total}, not 1"),
            ));
        }
        Ok(Self { both, only_first, only_second, neither })
    }

    /// `P(R1=1, R2=1, X*1=x1, X*2=x2)`.
    #[must_use]
    pub const fn both(&self) -> &[[f64; 2]; 2] {
        &self.both
    }

    /// `P(R1=1, R2=0, X*1=x1)`.
    #[must_use]
    pub const fn only_first(&self) -> &[f64; 2] {
        &self.only_first
    }

    /// `P(R1=0, R2=1, X*2=x2)`.
    #[must_use]
    pub const fn only_second(&self) -> &[f64; 2] {
        &self.only_second
    }

    /// `P(R1=0, R2=0)`.
    #[must_use]
    pub const fn neither(&self) -> f64 {
        self.neither
    }

    /// `P(R1=1, R2=1, X*_h = vh, X*_t = vt)` with axes in head/tail order.
    fn both_ht(&self, head: usize, vh: usize, vt: usize) -> f64 {
        if head == 0 { self.both[vh][vt] } else { self.both[vt][vh] }
    }

    /// `P(R_h=1, R_t=0, X*_h = vh)`.
    fn head_only(&self, head: usize, vh: usize) -> f64 {
        if head == 0 { self.only_first[vh] } else { self.only_second[vh] }
    }
}

/// The recovered full law `P(X1, X2)` with the identity of the plan it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainRecoveredLaw {
    cells: [[f64; 2]; 2],
    rule_version: &'static str,
}

impl ChainRecoveredLaw {
    /// `P(X1 = x1, X2 = x2)` (declaration order of the plan's query).
    #[must_use]
    pub const fn cells(&self) -> &[[f64; 2]; 2] {
        &self.cells
    }

    /// The rule version that produced it.
    #[must_use]
    pub const fn rule_version(&self) -> &'static str {
        self.rule_version
    }
}

/// Evaluate the ordered-response recovery formula of `plan` on `observed`.
///
/// With head axis `h` and tail axis `t`:
/// `p(x_h, x_t) = both(x_h, x_t) / (P(R_h=1) q(x_h))`, where `q` is
/// `P(R_h=1, R_t=1, X*_h = x_h) / P(R_h=1, X*_h = x_h)` when the tail response
/// depends on the head variable and `P(R_h=1, R_t=1) / P(R_h=1)` otherwise.
///
/// # Errors
///
/// `invalid_derivation` for a malformed plan; `transport_support_failure`
/// (`recovery_chain.positivity`) when the head response, a complete-case cell or
/// a required denominator has no mass.
pub fn evaluate_chain_recovery(
    plan: &ChainRecoveryPlan,
    observed: &ChainPatternLaw,
) -> Result<ChainRecoveredLaw, ChainRecoveryError> {
    if !plan.is_intact() || plan.head > 1 {
        return Err(refuse(
            ChainRecoveryDetail::InvalidDerivation,
            "the checked recovery plan was altered or has an invalid head axis",
        ));
    }
    let head = plan.head;
    let zero = || {
        refuse(
            ChainRecoveryDetail::Positivity,
            "a complete-case cell or a denominator of the recovery formula has no observed mass",
        )
    };
    let mut p_head = 0.0;
    let mut complete_total = 0.0;
    for vh in 0..2 {
        p_head += observed.head_only(head, vh);
        for vt in 0..2 {
            let cell = observed.both_ht(head, vh, vt);
            if cell <= 0.0 {
                return Err(zero());
            }
            p_head += cell;
            complete_total += cell;
        }
    }
    if p_head <= 0.0 {
        return Err(zero());
    }
    let mut cells = [[0.0; 2]; 2];
    for vh in 0..2 {
        let numerator_margin: f64 = (0..2).map(|vt| observed.both_ht(head, vh, vt)).sum();
        let denominator_margin = numerator_margin + observed.head_only(head, vh);
        for vt in 0..2 {
            let both = observed.both_ht(head, vh, vt);
            let value = if plan.tail_depends_on_head_variable {
                if denominator_margin <= 0.0 || numerator_margin <= 0.0 {
                    return Err(zero());
                }
                both * denominator_margin / (p_head * numerator_margin)
            } else {
                both / complete_total
            };
            if head == 0 {
                cells[vh][vt] = value;
            } else {
                cells[vt][vh] = value;
            }
        }
    }
    Ok(ChainRecoveredLaw { cells, rule_version: plan.rule_version })
}
