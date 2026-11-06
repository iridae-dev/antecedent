//! Exact binary trial compatibility test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::IoError;

/// Relative tolerance with which a table counts as exactly as probable as the observed one.
///
/// Masses are generated from the mode by the hypergeometric ratio recurrence, each step of
/// which carries a few ulps of relative error: at the one-million-row capacity the
/// accumulated relative error stays below `5e-10`. Two masses within `1e-7` of each other are
/// therefore indistinguishable at this precision, and counting them as ties errs on the side
/// of the conservative (larger) p-value, as exact-arithmetic ties such as a symmetric table
/// require.
const TIE_RELATIVE_SLACK: f64 = 1e-7;

/// A contingency table the exact test cannot evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct InvalidContingencyTable;

impl std::fmt::Display for InvalidContingencyTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("invalid binary trial contingency table")
    }
}

impl std::error::Error for InvalidContingencyTable {}

impl From<InvalidContingencyTable> for IoError {
    fn from(error: InvalidContingencyTable) -> Self {
        Self::Convert(error.to_string())
    }
}

/// Two-sided Fisher exact p-value of the 2x2 table `[[a, b], [c, d]]` flattened row-major.
///
/// The probability of each table with the observed margins is proportional to
/// `C(K, k) C(N - K, draws - k)`. Rather than evaluating each through log-gamma differences,
/// the masses are generated relative to the mode (mass one there) by the exact ratio
/// recurrence `m(k + 1) / m(k) = (K - k)(draws - k) / ((k + 1)(N - K - draws + k + 1))`, whose
/// factors are integers below `2^53` here. The p-value is the compensated sum of the masses
/// not exceeding the observed one (within [`TIE_RELATIVE_SLACK`]) over the compensated sum of
/// all masses, so no normalising constant is evaluated. Masses far from the mode underflow to
/// zero, which only drops terms below the working precision.
#[allow(
    clippy::cast_precision_loss,
    reason = "all counts are bounded by one million and exactly represented as f64"
)]
pub(super) fn fisher_exact(counts: [u64; 4]) -> Result<f64, InvalidContingencyTable> {
    use antecedent_kernels::numeric::CompensatedSum;
    let n = counts
        .iter()
        .try_fold(0u64, |sum, c| sum.checked_add(*c))
        .ok_or(InvalidContingencyTable)?;
    if n == 0 || n > 1_000_000 {
        return Err(InvalidContingencyTable);
    }
    let draws = counts[2] + counts[3];
    let successes = counts[1] + counts[3];
    let lo = draws.saturating_sub(n - successes);
    let hi = draws.min(successes);
    let mode = (((draws + 1) * (successes + 1)) / (n + 2)).clamp(lo, hi);
    // Mass of the table with `k + 1` successes relative to the table with `k`.
    let up = |k: u64| {
        ((successes - k) as f64 * (draws - k) as f64)
            / ((k + 1) as f64 * (n - successes + k + 1 - draws) as f64)
    };
    let observed_k = counts[3];
    let mut observed = 1.0_f64;
    if observed_k >= mode {
        for k in mode..observed_k {
            observed *= up(k);
        }
    } else {
        for k in observed_k..mode {
            observed /= up(k);
        }
    }
    let threshold = observed * (1.0 + TIE_RELATIVE_SLACK);
    let mut total = CompensatedSum::default();
    let mut tail = CompensatedSum::default();
    let mut take = |mass: f64| {
        total.add(mass);
        if mass <= threshold {
            tail.add(mass);
        }
    };
    let mut mass = 1.0_f64;
    take(mass);
    for k in mode..hi {
        mass *= up(k);
        take(mass);
    }
    mass = 1.0;
    for k in (lo..mode).rev() {
        mass /= up(k);
        take(mass);
    }
    Ok((tail.total() / total.total()).min(1.0))
}
