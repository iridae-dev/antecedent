//! Finite marginal laws and coupling envelopes.

use antecedent_core::CancellationToken;

use super::{
    CANCEL_STRIDE, JointLawError, MASS_TOLERANCE, MAX_LATTICE_CHECKS, MAX_MARGINAL_ATOMS,
    PairStructure, STRUCTURE_RELATIVE_SLACK, ValueBounds, ValueFunctional, check_cancel,
    require_finite,
};

/// A finite marginal law: distinct ascending atoms with positive masses that
/// sum to one. Draws are equal-mass atoms.
#[derive(Clone, Debug, PartialEq)]
pub struct Marginal {
    atoms: Vec<(f64, f64)>,
    cumulative: Vec<f64>,
}

impl Marginal {
    /// A marginal from draws, each with mass `1 / n`.
    ///
    /// # Errors
    /// Empty input, more than [`MAX_MARGINAL_ATOMS`] draws, or non-finite values.
    pub fn from_draws(draws: &[f64]) -> Result<Self, JointLawError> {
        let masses = vec![1.0 / draws.len().max(1) as f64; draws.len()];
        Self::from_weighted(draws, &masses)
    }

    /// A marginal from values and masses; repeated values are merged.
    ///
    /// # Errors
    /// Empty or unequal inputs, more than [`MAX_MARGINAL_ATOMS`] atoms,
    /// non-finite values, negative masses or masses not summing to one.
    pub fn from_weighted(values: &[f64], masses: &[f64]) -> Result<Self, JointLawError> {
        if values.is_empty() {
            return Err(JointLawError::Shape("a marginal needs at least one atom"));
        }
        if values.len() != masses.len() {
            return Err(JointLawError::Shape("values and masses differ in length"));
        }
        if values.len() > MAX_MARGINAL_ATOMS {
            return Err(JointLawError::BudgetExceeded("more atoms than the marginal bound"));
        }
        require_finite(values, "marginal value")?;
        require_finite(masses, "marginal mass")?;
        if masses.iter().any(|&m| m < 0.0) {
            return Err(JointLawError::InvalidWeights("masses must be non-negative"));
        }
        // Canonical order (value, then mass) before any summation, so the
        // atoms, their masses and every bound derived from them are the same
        // bits for any listing order of the input.
        let mut pairs: Vec<(f64, f64)> = values
            .iter()
            .zip(masses)
            .filter(|(_, m)| **m > 0.0)
            // `+ 0.0` turns negative zero into positive zero.
            .map(|(v, m)| (v + 0.0, *m))
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let total: f64 = pairs.iter().map(|(_, m)| m).sum();
        if (total - 1.0).abs() > MASS_TOLERANCE {
            return Err(JointLawError::InvalidWeights("masses must sum to one"));
        }
        for pair in &mut pairs {
            pair.1 /= total;
        }
        let mut atoms: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
        for (value, mass) in pairs {
            match atoms.last_mut() {
                Some(last) if last.0.total_cmp(&value).is_eq() => last.1 += mass,
                _ => atoms.push((value, mass)),
            }
        }
        let mut cumulative = Vec::with_capacity(atoms.len());
        let mut running = 0.0;
        for (_, mass) in &atoms {
            running += mass;
            cumulative.push(running);
        }
        if let Some(last) = cumulative.last_mut() {
            *last = 1.0;
        }
        Ok(Self { atoms, cumulative })
    }

    /// Ascending `(value, mass)` atoms.
    #[must_use]
    pub fn atoms(&self) -> &[(f64, f64)] {
        &self.atoms
    }

    /// Mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.atoms.iter().map(|(v, m)| v * m).sum()
    }

    /// Population variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.atoms.iter().map(|(v, m)| m * (v - mean) * (v - mean)).sum()
    }

    /// `P(X <= x)`.
    #[must_use]
    pub fn mass_at_most(&self, x: f64) -> f64 {
        match self.atoms.partition_point(|a| a.0 <= x) {
            0 => 0.0,
            idx => self.cumulative[idx - 1],
        }
    }

    /// `P(X < x)`.
    #[must_use]
    pub fn mass_below(&self, x: f64) -> f64 {
        match self.atoms.partition_point(|a| a.0 < x) {
            0 => 0.0,
            idx => self.cumulative[idx - 1],
        }
    }

    /// The generalized inverse `inf { x : P(X <= x) >= p }` (Hyndman-Fan type
    /// 1), for `p` in `(0, 1]`.
    ///
    /// # Errors
    /// A `p` outside `(0, 1]`.
    pub fn quantile(&self, p: f64) -> Result<f64, JointLawError> {
        if !(p.is_finite() && p > 0.0 && p <= 1.0) {
            return Err(JointLawError::InvalidLevel("a quantile level is in (0, 1]"));
        }
        let idx = self.cumulative.partition_point(|c| *c < p).min(self.atoms.len() - 1);
        Ok(self.atoms[idx].0)
    }

    /// The marginal of `-X`.
    #[must_use]
    pub fn negated(&self) -> Self {
        let values: Vec<f64> = self.atoms.iter().rev().map(|(v, _)| -v).collect();
        let masses: Vec<f64> = self.atoms.iter().rev().map(|(_, m)| *m).collect();
        // The mass vector sums to one by construction, so this cannot refuse.
        Self::from_weighted(&values, &masses).unwrap_or_else(|_| self.clone())
    }
}

// ------------------------------------------------------- coupling extremes

fn coupled_expectation(
    x: &[(f64, f64)],
    y: &[(f64, f64)],
    value: &dyn Fn(f64, f64) -> f64,
    cancel: &CancellationToken,
) -> Result<f64, JointLawError> {
    let (mut i, mut j) = (0_usize, 0_usize);
    let (mut rx, mut ry) = (x[0].1, y[0].1);
    let mut terms: Vec<f64> = Vec::new();
    let mut steps = 0_usize;
    loop {
        steps += 1;
        if steps % CANCEL_STRIDE == 0 {
            check_cancel(cancel)?;
        }
        let m = rx.min(ry);
        let v = value(x[i].0, y[j].0);
        if !v.is_finite() {
            return Err(JointLawError::NonFiniteValue);
        }
        terms.push(m * v);
        rx -= m;
        ry -= m;
        // One of `rx` and `ry` is exactly the selected minimum, so its
        // subtraction reaches zero. Do not use a probability tolerance here:
        // even a tiny positive atom is part of the declared marginal and can
        // change an allegedly exact coupling envelope.
        if rx <= 0.0 {
            i += 1;
            if i == x.len() {
                break;
            }
            rx = x[i].1;
        }
        if ry <= 0.0 {
            j += 1;
            if j == y.len() {
                break;
            }
            ry = y[j].1;
        }
    }
    // Sum the cell contributions in a total order so the value does not depend
    // on the direction the coupling was walked.
    terms.sort_by(f64::total_cmp);
    Ok(terms.iter().sum())
}

/// `E[g(X, Y)]` under the comonotone and countermonotone couplings.
fn monotone_couplings(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    cancel: &CancellationToken,
) -> Result<(f64, f64), JointLawError> {
    let comonotone = coupled_expectation(&x.atoms, &y.atoms, value, cancel)?;
    // The countermonotone coupling is one coupling however it is walked; walk it
    // from the canonically smaller marginal so that swapping the two arguments
    // visits the same cells in the same order with the same masses.
    let swap = x
        .atoms
        .iter()
        .zip(&y.atoms)
        .map(|(a, b)| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
        .find(|o| o.is_ne())
        .unwrap_or_else(|| x.atoms.len().cmp(&y.atoms.len()))
        == std::cmp::Ordering::Greater;
    let countermonotone = if swap {
        let reversed: Vec<(f64, f64)> = x.atoms.iter().rev().copied().collect();
        coupled_expectation(&y.atoms, &reversed, &|a, b| value(b, a), cancel)?
    } else {
        let reversed: Vec<(f64, f64)> = y.atoms.iter().rev().copied().collect();
        coupled_expectation(&x.atoms, &reversed, value, cancel)?
    };
    Ok((comonotone, countermonotone))
}

/// Verify the mixed second difference of `value` on every adjacent cell of the
/// product of the atom lattices.
///
/// Adjacent-cell sign is equivalent to the global lattice property on a product
/// of chains, so this is a complete check, not a sample.
fn verify_structure(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    structure: PairStructure,
    cancel: &CancellationToken,
) -> Result<(), JointLawError> {
    let (nx, ny) = (x.atoms.len(), y.atoms.len());
    if (nx.saturating_sub(1)).saturating_mul(ny.saturating_sub(1)) > MAX_LATTICE_CHECKS {
        return Err(JointLawError::BudgetExceeded(
            "the atom lattice is larger than the structure-verification bound; coarsen the marginals",
        ));
    }
    let row = |xv: f64| -> Result<Vec<f64>, JointLawError> {
        y.atoms
            .iter()
            .map(|(yv, _)| {
                let v = value(xv, *yv);
                if v.is_finite() { Ok(v) } else { Err(JointLawError::NonFiniteValue) }
            })
            .collect()
    };
    let mut lower_row = row(x.atoms[0].0)?;
    for &(x_value, _) in x.atoms.iter().skip(1) {
        check_cancel(cancel)?;
        let upper_row = row(x_value)?;
        for j in 1..ny {
            let (a, b, c, d) = (upper_row[j], lower_row[j - 1], lower_row[j], upper_row[j - 1]);
            let mixed = (a + b) - (c + d);
            let slack = STRUCTURE_RELATIVE_SLACK * (a.abs() + b.abs() + c.abs() + d.abs());
            let violated = match structure {
                PairStructure::Supermodular => mixed < -slack,
                PairStructure::Submodular => mixed > slack,
            };
            if violated {
                return Err(JointLawError::StructureNotVerified { structure: structure.as_str() });
            }
        }
        lower_row = upper_row;
    }
    Ok(())
}

/// Exact extremes of `E[g(X, Y)]` over every coupling of the two marginals,
/// for a value function with a verified mixed-difference sign.
///
/// For a supermodular `g` the largest value is attained at the comonotone
/// coupling and the smallest at the countermonotone coupling; for a
/// submodular `g` the roles swap (Lorentz 1953; Tchen 1980). Both couplings are
/// built from the marginals' quantile functions and evaluated exactly on the
/// finite atoms. The sign of the mixed difference is checked on the whole atom
/// lattice first, and the call refuses when it does not hold.
///
/// # Errors
/// A structure that is not verified, a non-finite value, a lattice above
/// [`MAX_LATTICE_CHECKS`], or cancellation.
pub fn frechet_expectation_envelope(
    x: &Marginal,
    y: &Marginal,
    value: &dyn Fn(f64, f64) -> f64,
    structure: PairStructure,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    verify_structure(x, y, value, structure, cancel)?;
    let (co, counter) = monotone_couplings(x, y, value, cancel)?;
    Ok(ValueBounds::ordered(co, counter))
}

/// [`frechet_expectation_envelope`] for one of the closed value functionals.
///
/// # Errors
/// As [`frechet_expectation_envelope`].
pub fn functional_envelope(
    x: &Marginal,
    y: &Marginal,
    functional: ValueFunctional,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    frechet_expectation_envelope(
        x,
        y,
        &|a, b| functional.value(a, b),
        functional.structure(),
        cancel,
    )
}

/// Extremes of `E[X Y]` over every coupling. `x * y` is supermodular on any
/// lattice, so no lattice verification is needed and the cost is linear in the
/// number of atoms.
///
/// # Errors
/// A non-finite product or cancellation.
pub fn product_moment_envelope(
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    let (co, counter) = monotone_couplings(x, y, &|a, b| a * b, cancel)?;
    Ok(ValueBounds::ordered(co, counter))
}

/// Extremes of `Cov(X, Y)` over every coupling.
///
/// # Errors
/// As [`product_moment_envelope`].
pub fn covariance_envelope(
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    let product = product_moment_envelope(x, y, cancel)?;
    let centre = x.mean() * y.mean();
    let lower = product.lower - centre;
    let upper = product.upper - centre;
    if !(centre.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(JointLawError::NonFiniteValue);
    }
    Ok(ValueBounds { lower, upper })
}

/// Extremes of `Var(a X + b Y)` over every coupling.
///
/// `Var = a^2 Var X + b^2 Var Y + 2 a b Cov`, linear in the covariance, whose
/// extremes are attained at the comonotone and countermonotone couplings, so
/// the extremes are exact for every sum, difference and weighted sum of two
/// marginals.
///
/// # Errors
/// Non-finite coefficients, or as [`product_moment_envelope`].
pub fn linear_variance_envelope(
    a: f64,
    b: f64,
    x: &Marginal,
    y: &Marginal,
    cancel: &CancellationToken,
) -> Result<ValueBounds, JointLawError> {
    require_finite(&[a, b], "coefficient")?;
    let cov = covariance_envelope(x, y, cancel)?;
    let base = a * a * x.variance() + b * b * y.variance();
    let cross = 2.0 * a * b;
    let (lo, hi) = (base + cross * cov.lower, base + cross * cov.upper);
    if !(base.is_finite() && cross.is_finite() && lo.is_finite() && hi.is_finite()) {
        return Err(JointLawError::NonFiniteValue);
    }
    let bounds = ValueBounds::ordered(lo, hi);
    Ok(ValueBounds { lower: bounds.lower.max(0.0), upper: bounds.upper.max(0.0) })
}

/// Exact extremes of `P(X + Y <= t)` over every coupling of the marginals
/// (Makarov 1982; Rüschendorf 1982):
/// `sup_x max(F(x) + G(t - x) - 1, 0)` and `inf_x min(F(x) + G(t - x), 1)` with
/// the right-continuous distribution functions `F` and `G`.
///
/// Both functions of `x` are step functions; the extreme is evaluated exactly
/// at each atom of either marginal (its value, the value just before and the
/// value just after), which covers every constancy interval.
///
/// # Errors
/// A non-finite threshold.
pub fn sum_cdf_envelope(
    x: &Marginal,
    y: &Marginal,
    threshold: f64,
) -> Result<ValueBounds, JointLawError> {
    if !threshold.is_finite() {
        return Err(JointLawError::NonFinite("threshold"));
    }
    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    let mut take = |value: f64| {
        lower = lower.max(value - 1.0);
        upper = upper.min(value);
    };
    for &(a, _) in &x.atoms {
        let (f_at, f_left) = (x.mass_at_most(a), x.mass_below(a));
        let u = threshold - a;
        let (g_at, g_after) = (y.mass_at_most(u), y.mass_below(u));
        take(f_at + g_at);
        take(f_at + g_after);
        take(f_left + g_at);
    }
    for &(b, _) in &y.atoms {
        let p = threshold - b;
        let (f_at, f_left) = (x.mass_at_most(p), x.mass_below(p));
        let (g_at, g_after) = (y.mass_at_most(b), y.mass_below(b));
        take(f_at + g_at);
        take(f_at + g_after);
        take(f_left + g_at);
    }
    Ok(ValueBounds { lower: lower.clamp(0.0, 1.0), upper: upper.clamp(0.0, 1.0) })
}

/// Exact extremes of `P(X - Y <= t)` over every coupling.
///
/// # Errors
/// As [`sum_cdf_envelope`].
pub fn difference_cdf_envelope(
    x: &Marginal,
    y: &Marginal,
    threshold: f64,
) -> Result<ValueBounds, JointLawError> {
    sum_cdf_envelope(x, &y.negated(), threshold)
}
