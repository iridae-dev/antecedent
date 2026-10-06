//! Deterministic screen/estimate splits from caller-declared data ownership (2.2 E4).
//!
//! A screen/estimate split is only honest when no unit of dependence contributes rows to
//! both halves: a repeated entity, a cluster, or a dyad whose endpoint also appears in
//! another dyad would leak what the screen saw into the estimate half. The generator takes
//! the caller's ownership declaration (a label per row, or two endpoint labels per row) and
//! assigns whole units to a half, with a seeded, order-independent shuffle:
//!
//! - **Entity / cluster labels**: each distinct label is a unit.
//! - **Dyads**: the endpoint labels live in one namespace and every row links its two
//!   endpoints, so a unit is a *connected component of the endpoint graph* (union-find). Two
//!   rows that share an endpoint, directly or through a chain of rows, always land in the
//!   same half; this is the no-shared-endpoint rule. The components are computed by
//!   `antecedent_estimate::endpoint_components`, the same union-find that owns the folds of
//!   a dyadic cluster-DML declaration, so the screen split and the cross-fit agree on units.
//!
//! The split is a function of the set of units and the seed alone, so it does not depend on
//! row order. A component that covers the whole graph leaves a single unit and the split is
//! refused: no honest split exists.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Write;
use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_io::PayloadDigestWire;

use super::batch::{CandidateProcedure, CandidateScreen};
use crate::error::CausalError;

/// Caller-declared ownership of the rows of a data set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScreenUnits<'a> {
    /// One owner label per data row (an entity or a cluster); each distinct label is a unit.
    Entity(&'a [u32]),
    /// Two endpoint labels per data row (a dyad), in one shared label namespace. A unit is a
    /// connected component of the graph that joins `first[i]` and `second[i]` for every row.
    Dyad {
        /// First endpoint of each row.
        first: &'a [u32],
        /// Second endpoint of each row.
        second: &'a [u32],
    },
}

/// What a generated split was built from, recorded beside the [`CandidateScreen`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScreenSplitReceipt {
    /// Seed of the unit shuffle.
    pub seed: u64,
    /// `"entity"` for owner labels, `"dyad_component"` for connected endpoint components.
    pub unit_kind: &'static str,
    /// Units that were dealt to a half.
    pub n_units: usize,
    /// Units in the screen half.
    pub screen_units: usize,
    /// Units in the estimate half.
    pub estimate_units: usize,
    /// Hex digest of the sorted distinct unit ids (each unit is named by its smallest label),
    /// so it does not depend on row order.
    pub unit_digest: String,
}

impl CandidateScreen {
    /// Split the rows of a data set into disjoint screen and estimate halves that never
    /// share a unit of dependence, from the caller's ownership declaration.
    ///
    /// `screen_fraction` in `(0, 1)` is the share of units given to the screen half (rounded,
    /// and kept so both halves hold at least one unit). The returned screen's `screen_id` is
    /// `"{screen_id};seed={seed:016x};units={digest}"`, so the seed and the unit digest ride
    /// along in every recorded selection; the receipt carries them as fields.
    ///
    /// # Errors
    ///
    /// `invalid_argument` for a fraction outside `(0, 1)`, mismatched or empty labels, a row
    /// index beyond `u32`, or fewer than two units (for dyads: a single connected component).
    pub fn from_units(
        screen_id: &str,
        procedure: CandidateProcedure,
        units: ScreenUnits<'_>,
        screen_fraction: f64,
        seed: u64,
    ) -> Result<(Self, ScreenSplitReceipt), CausalError> {
        if !(screen_fraction > 0.0 && screen_fraction < 1.0) {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "screen_fraction must lie strictly between 0 and 1, got {screen_fraction}"
            ));
        }
        let (row_unit, unit_kind) = row_units(units)?;
        let mut distinct: Vec<u32> = row_unit.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 2 {
            return Err(crate::compile_reason!(
                "invalid_argument",
                "a screen/estimate split needs at least two independent units; the declared \
                 ownership leaves {}",
                distinct.len()
            ));
        }
        let mut rng = CausalRng::from_seed(seed ^ 0x5C7E_E2A1_D00D_F00D);
        let mut order = distinct.clone();
        for i in (1..order.len()).rev() {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a uniform draw in [0, 1) scaled by i + 1 is a small non-negative index"
            )]
            let j = (rng.next_f64() * (i as f64 + 1.0)) as usize;
            order.swap(i, j.min(i));
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the product of a fraction in (0, 1) and a unit count is a small count"
        )]
        let n_screen =
            ((screen_fraction * order.len() as f64).round() as usize).clamp(1, order.len() - 1);
        let screen_units: std::collections::BTreeSet<u32> =
            order[..n_screen].iter().copied().collect();
        let mut screen_rows = Vec::new();
        let mut estimate_rows = Vec::new();
        for (row, unit) in row_unit.iter().enumerate() {
            let index = u32::try_from(row).map_err(|_| {
                crate::compile_reason!("invalid_argument", "row index exceeds the u32 row-id range")
            })?;
            if screen_units.contains(unit) {
                screen_rows.push(index);
            } else {
                estimate_rows.push(index);
            }
        }
        let digest = PayloadDigestWire::u32s("candidate_screen.units.v1", &distinct).digest;
        let mut unit_digest = String::with_capacity(2 * digest.len());
        for byte in digest {
            let _ = write!(unit_digest, "{byte:02x}");
        }
        let screen = Self {
            screen_id: Arc::from(format!("{screen_id};seed={seed:016x};units={unit_digest}")),
            procedure,
            screen_rows: screen_rows.into(),
            estimate_rows: estimate_rows.into(),
        };
        let receipt = ScreenSplitReceipt {
            seed,
            unit_kind,
            n_units: distinct.len(),
            screen_units: n_screen,
            estimate_units: distinct.len() - n_screen,
            unit_digest,
        };
        Ok((screen, receipt))
    }
}

/// The unit of each row, named by the smallest label of its connected component.
fn row_units(units: ScreenUnits<'_>) -> Result<(Vec<u32>, &'static str), CausalError> {
    match units {
        ScreenUnits::Entity(labels) => {
            if labels.is_empty() {
                return Err(crate::compile_reason!(
                    "invalid_argument",
                    "ownership labels must not be empty"
                ));
            }
            Ok((labels.to_vec(), "entity"))
        }
        ScreenUnits::Dyad { first, second } => {
            if first.is_empty() || first.len() != second.len() {
                return Err(crate::compile_reason!(
                    "invalid_argument",
                    "dyad endpoints must be non-empty and one pair per row (got {} and {})",
                    first.len(),
                    second.len()
                ));
            }
            let rows = antecedent_estimate::endpoint_components(first, second);
            Ok((rows, "dyad_component"))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    fn split(units: ScreenUnits<'_>, seed: u64) -> (CandidateScreen, ScreenSplitReceipt) {
        CandidateScreen::from_units("screen", CandidateProcedure::MaxT, units, 0.5, seed).unwrap()
    }

    /// Entities of each half, by label.
    fn halves(labels: &[u32], screen: &CandidateScreen) -> (BTreeSet<u32>, BTreeSet<u32>) {
        let pick = |rows: &[u32]| rows.iter().map(|&r| labels[r as usize]).collect();
        (pick(&screen.screen_rows), pick(&screen.estimate_rows))
    }

    #[test]
    fn an_entity_never_contributes_rows_to_both_halves() {
        // 40 entities with 1..=4 rows each.
        let labels: Vec<u32> = (0..40u32).flat_map(|e| (0..=e % 4).map(move |_| e)).collect();
        let (screen, receipt) = split(ScreenUnits::Entity(&labels), 7);
        let (screen_entities, estimate_entities) = halves(&labels, &screen);
        assert!(screen_entities.is_disjoint(&estimate_entities));
        assert_eq!(screen_entities.len() + estimate_entities.len(), 40);
        assert_eq!(screen.screen_rows.len() + screen.estimate_rows.len(), labels.len());
        assert_eq!((receipt.screen_units, receipt.estimate_units, receipt.n_units), (20, 20, 40));
        assert_eq!(receipt.unit_kind, "entity");
    }

    #[test]
    fn the_split_is_replayable_row_order_free_and_records_seed_and_digest() {
        let labels: Vec<u32> = (0..30u32).flat_map(|e| [e, e]).collect();
        let (a, ra) = split(ScreenUnits::Entity(&labels), 11);
        let (b, rb) = split(ScreenUnits::Entity(&labels), 11);
        assert_eq!((a.clone(), ra.clone()), (b, rb));
        assert!(a.screen_id.contains(&format!("seed={:016x}", 11)));
        assert!(a.screen_id.contains(&ra.unit_digest));
        let (other, _) = split(ScreenUnits::Entity(&labels), 12);
        assert_ne!(a.screen_rows, other.screen_rows);

        // Reverse the rows: the same entities land in the same halves, and the digest is the same.
        let reversed: Vec<u32> = labels.iter().rev().copied().collect();
        let (c, rc) = split(ScreenUnits::Entity(&reversed), 11);
        assert_eq!(ra.unit_digest, rc.unit_digest);
        let (a_screen, _) = halves(&labels, &a);
        let (c_screen, _) = halves(&reversed, &c);
        assert_eq!(a_screen, c_screen);
    }

    #[test]
    fn dyads_sharing_an_endpoint_never_cross_the_split() {
        // Components: a path 0-1-2-3, a star 10-{11,12,13}, and 12 isolated pairs.
        let mut first = vec![0, 1, 2, 10, 10, 10];
        let mut second = vec![1, 2, 3, 11, 12, 13];
        for pair in 0..12u32 {
            first.push(100 + 2 * pair);
            second.push(101 + 2 * pair);
        }
        let (screen, receipt) = split(ScreenUnits::Dyad { first: &first, second: &second }, 3);
        assert_eq!(receipt.unit_kind, "dyad_component");
        assert_eq!(receipt.n_units, 14);
        let endpoints = |rows: &[u32]| -> BTreeSet<u32> {
            rows.iter().flat_map(|&r| [first[r as usize], second[r as usize]]).collect()
        };
        assert!(
            endpoints(&screen.screen_rows).is_disjoint(&endpoints(&screen.estimate_rows)),
            "an endpoint appears on both sides of the split"
        );
        // The chain 0-1-2-3 stays whole: all four dyads of it share one half.
        let side: BTreeMap<u32, bool> = screen
            .screen_rows
            .iter()
            .map(|&r| (r, true))
            .chain(screen.estimate_rows.iter().map(|&r| (r, false)))
            .collect();
        assert!(side[&0] == side[&1] && side[&1] == side[&2]);
        assert!(side[&3] == side[&4] && side[&4] == side[&5]);
    }

    #[test]
    fn dyad_components_do_not_depend_on_row_order() {
        let first = vec![5, 1, 9, 2, 7, 3, 20, 21];
        let second = vec![1, 2, 7, 3, 8, 4, 22, 23];
        let (a, ra) = split(ScreenUnits::Dyad { first: &first, second: &second }, 5);
        let order: Vec<usize> = (0..first.len()).rev().collect();
        let first_r: Vec<u32> = order.iter().map(|&i| first[i]).collect();
        let second_r: Vec<u32> = order.iter().map(|&i| second[i]).collect();
        let (b, rb) = split(ScreenUnits::Dyad { first: &first_r, second: &second_r }, 5);
        assert_eq!(ra.unit_digest, rb.unit_digest);
        let rows_a: BTreeSet<(u32, u32)> =
            a.screen_rows.iter().map(|&r| (first[r as usize], second[r as usize])).collect();
        let rows_b: BTreeSet<(u32, u32)> =
            b.screen_rows.iter().map(|&r| (first_r[r as usize], second_r[r as usize])).collect();
        assert_eq!(rows_a, rows_b);
    }

    #[test]
    fn a_single_component_or_bad_input_is_refused() {
        // One connected component: no honest split exists.
        let (first, second) = (vec![0, 1, 2, 3], vec![1, 2, 3, 0]);
        for units in [
            ScreenUnits::Dyad { first: &first, second: &second },
            ScreenUnits::Entity(&[4, 4, 4]),
            ScreenUnits::Entity(&[]),
            ScreenUnits::Dyad { first: &[0, 1], second: &[1] },
        ] {
            let error = CandidateScreen::from_units("s", CandidateProcedure::MaxT, units, 0.5, 1)
                .unwrap_err();
            assert!(error.to_string().contains("invalid_argument"), "{error}");
        }
        for fraction in [0.0, 1.0, -0.5, f64::NAN] {
            let error = CandidateScreen::from_units(
                "s",
                CandidateProcedure::MaxT,
                ScreenUnits::Entity(&[0, 1, 2]),
                fraction,
                1,
            )
            .unwrap_err();
            assert!(error.to_string().contains("invalid_argument"), "{error}");
        }
    }
}
