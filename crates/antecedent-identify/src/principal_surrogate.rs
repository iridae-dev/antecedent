//! Sharp binary principal-stratum bounds from a randomized joint endpoint law.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Sharp identified set for P(Y(1) > Y(0) | S(1) > S(0)), as exact ratios.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrincipalSurrogateBounds {
    /// Least compatible numerator, in units of the common arm-law denominator.
    pub lower_numerator: u64,
    /// Greatest compatible numerator in the same units.
    pub upper_numerator: u64,
    /// Positive mass of surrogate improvers, in the same units.
    pub improver_numerator: u64,
    /// Denominator of the normalized two-arm joint endpoint laws.
    pub law_denominator: u64,
}

impl PrincipalSurrogateBounds {
    /// Display endpoints of the exact conditional probability interval.
    #[must_use]
    pub fn endpoints(self) -> [f64; 2] {
        [self.lower_numerator, self.upper_numerator]
            .map(|value| value as f64 / self.improver_numerator as f64)
    }

    /// Probability of the conditioning principal stratum.
    #[must_use]
    pub fn improver_probability(self) -> f64 {
        self.improver_numerator as f64 / self.law_denominator as f64
    }

    /// Whether the declared evidence fixes a single compatible value.
    #[must_use]
    pub const fn point_identified(self) -> bool {
        self.lower_numerator == self.upper_numerator
    }
}

/// A refusal of the monotone binary principal-stratum target.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PrincipalSurrogateError {
    /// Each arm needs positive joint law mass with total representable as a u32.
    #[error("each arm needs a nonempty joint endpoint law with total mass within u32")]
    JointLaw,
    /// The observed treatment margin makes the asserted surrogate monotonicity impossible.
    #[error(
        "the treated surrogate prevalence is below the control prevalence under declared monotonicity"
    )]
    Monotonicity,
    /// A conditional probability for the surrogate-improver stratum needs positive mass.
    #[error("the surrogate-improver principal stratum has zero mass")]
    EmptyImproverStratum,
}

fn mass(cell: [u32; 4]) -> Result<u32, PrincipalSurrogateError> {
    cell.iter()
        .try_fold(0_u32, |total, &value| total.checked_add(value))
        .filter(|&total| total > 0)
        .ok_or(PrincipalSurrogateError::JointLaw)
}

/// Identify the sharp interval under individual monotonicity S(1) >= S(0).
///
/// The four masses in each arm are for (S,Y) = (0,0), (0,1), (1,0), (1,1).
/// Under random assignment these are the row and column margins of a 4x4
/// potential endpoint table. Monotonicity forbids rows with S(0)=1 from
/// using columns with S(1)=0, fixing total improver mass at d. The target
/// numerator is the (0,0) -> (1,1) cell. Its upper capacity is the smaller
/// of its row, column and d. For the lower bound that row can use every
/// other column, except that total mass sent to S(1)=1 is constrained by
/// the lower block's fixed mass. The remaining capacities give the
/// Fréchet lower bound. After fixing this cell anywhere between those
/// limits, the remaining upper-left and lower-right blocks can be filled
/// by arbitrary nonnegative couplings of their residual margins.
///
/// # Errors
/// The arm laws are invalid, violate a necessary monotonicity inequality,
/// or identify zero probability of belonging to the conditioning stratum.
pub fn monotone_principal_surrogate_bounds(
    control: [u32; 4],
    treatment: [u32; 4],
) -> Result<PrincipalSurrogateBounds, PrincipalSurrogateError> {
    let n0 = u64::from(mass(control)?);
    let n1 = u64::from(mass(treatment)?);
    let law_denominator = n0 * n1;
    let control = control.map(|cell| u64::from(cell) * n1);
    let treatment = treatment.map(|cell| u64::from(cell) * n0);
    let improver_numerator = (treatment[2] + treatment[3])
        .checked_sub(control[2] + control[3])
        .ok_or(PrincipalSurrogateError::Monotonicity)?;
    if improver_numerator == 0 {
        return Err(PrincipalSurrogateError::EmptyImproverStratum);
    }
    Ok(PrincipalSurrogateBounds {
        lower_numerator: control[0].saturating_sub(law_denominator - treatment[3]),
        upper_numerator: control[0].min(treatment[3]).min(improver_numerator),
        improver_numerator,
        law_denominator,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Enumerate every 4-subject latent table directly by unordered assignments
    /// to the twelve allowed cells. This oracle uses no margin-bound formulas.
    #[test]
    fn independent_exhaustive_monotone_surrogate_couplings() {
        type Margins = ([u32; 4], [u32; 4]);
        let cells: Vec<_> = (0..4)
            .flat_map(|row| (0..4).map(move |column| (row, column)))
            .filter(|&(row, column)| row / 2 <= column / 2)
            .collect();
        let mut fibers: BTreeMap<Margins, (u32, u32, u32)> = BTreeMap::new();
        for a in 0..cells.len() {
            for b in a..cells.len() {
                for c in b..cells.len() {
                    for d in c..cells.len() {
                        let (mut control, mut treatment) = ([0; 4], [0; 4]);
                        let (mut improved, mut endpoint_benefit) = (0, 0);
                        for i in [a, b, c, d] {
                            let (row, column) = cells[i];
                            control[row] += 1;
                            treatment[column] += 1;
                            improved += u32::from(row < 2 && column >= 2);
                            endpoint_benefit += u32::from(row == 0 && column == 3);
                        }
                        let fiber = fibers.entry((control, treatment)).or_insert((
                            improved,
                            endpoint_benefit,
                            endpoint_benefit,
                        ));
                        assert_eq!(fiber.0, improved);
                        fiber.1 = fiber.1.min(endpoint_benefit);
                        fiber.2 = fiber.2.max(endpoint_benefit);
                    }
                }
            }
        }
        let (mut ambiguous, mut tied, mut empty) = (0, 0, 0);
        for ((control, treatment), (improvers, lower, upper)) in fibers {
            let found = monotone_principal_surrogate_bounds(control, treatment);
            if improvers == 0 {
                assert_eq!(found, Err(PrincipalSurrogateError::EmptyImproverStratum));
                empty += 1;
            } else {
                let found = found.unwrap();
                assert_eq!(found.improver_numerator, 4 * u64::from(improvers));
                assert_eq!(found.lower_numerator, 4 * u64::from(lower));
                assert_eq!(found.upper_numerator, 4 * u64::from(upper));
                assert_eq!(found.point_identified(), lower == upper);
                tied += usize::from(lower == upper);
                ambiguous += usize::from(lower < upper);
            }
        }
        assert!(ambiguous > 0 && tied > 0 && empty > 0);
        assert_eq!(
            monotone_principal_surrogate_bounds([0, 0, 4, 0], [4, 0, 0, 0]),
            Err(PrincipalSurrogateError::Monotonicity),
        );
    }
}
