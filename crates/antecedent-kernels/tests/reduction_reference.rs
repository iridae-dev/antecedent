//! Reductions checked against an independent double-double reference.
//!
//! The scalar and portable reductions are both left-to-right `f64` sums. They
//! can agree with each other to the last bit and still share one rounding error,
//! so a differential between them only guards index order. Here each result is
//! compared with a double-double evaluation (error-free `TwoSum` and `TwoProd`
//! transformations, about 106 bits). That is a different algorithm from the
//! kernels' accumulation, so a shared drift is visible.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "counts are converted to f64 for reference arithmetic"
)]
#![allow(clippy::float_cmp, reason = "hand-computed references are exactly representable")]

use antecedent_kernels::{BitMaskView, F64VectorView, portable, scalar};

const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd { hi: s, lo: b - (s - a) }
}

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd { hi: s, lo: (a - (s - bb)) + (b - bb) }
}

fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd { hi: p, lo: a.mul_add(b, -p) }
}

impl Dd {
    const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    const fn of(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    fn plus(self, other: Self) -> Self {
        let s = two_sum(self.hi, other.hi);
        let t = two_sum(self.lo, other.lo);
        let u = quick_two_sum(s.hi, s.lo + t.hi);
        quick_two_sum(u.hi, u.lo + t.lo)
    }

    fn negated(self) -> Self {
        Self { hi: -self.hi, lo: -self.lo }
    }

    fn minus(self, other: Self) -> Self {
        self.plus(other.negated())
    }

    fn times(self, other: Self) -> Self {
        let p = two_prod(self.hi, other.hi);
        quick_two_sum(p.hi, p.lo + (self.hi * other.lo + self.lo * other.hi))
    }

    fn div_f64(self, divisor: f64) -> Self {
        let q1 = self.hi / divisor;
        let r = self.minus(two_prod(q1, divisor));
        let q2 = r.hi / divisor;
        let r = r.minus(two_prod(q2, divisor));
        let q3 = r.hi / divisor;
        quick_two_sum(q1, q2).plus(Self::of(q3))
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// Deterministic uniform draw in `[0, 1)`.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

/// Values spanning many binades, with both signs.
fn wide_range(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let magnitude = 10f64.powf(uniform(&mut state) * 12.0 - 4.0);
            if uniform(&mut state) < 0.5 { -magnitude } else { magnitude }
        })
        .collect()
}

/// A large offset with unit-scale noise: the variance is tiny next to the mean.
fn offset_noise(n: usize, offset: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n).map(|_| offset + (uniform(&mut state) - 0.5) * 4.0).collect()
}

fn mask_bytes(keep: impl Fn(usize) -> bool, len: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; len.div_ceil(8)];
    for i in 0..len {
        if keep(i) {
            bytes[i / 8] |= 1 << (i % 8);
        }
    }
    bytes
}

fn selected(x: &[f64], bytes: Option<&[u8]>) -> Vec<f64> {
    match bytes {
        None => x.to_vec(),
        Some(bytes) => x
            .iter()
            .enumerate()
            .filter(|(i, _)| (bytes[i / 8] >> (i % 8)) & 1 == 1)
            .map(|(_, v)| *v)
            .collect(),
    }
}

fn reference_sum(values: &[f64]) -> Dd {
    values.iter().fold(Dd::ZERO, |acc, &v| acc.plus(Dd::of(v)))
}

fn reference_mean(values: &[f64]) -> Dd {
    reference_sum(values).div_f64(values.len() as f64)
}

fn reference_variance(values: &[f64]) -> Dd {
    let mean = reference_mean(values);
    let ss = values.iter().fold(Dd::ZERO, |acc, &v| {
        let d = Dd::of(v).minus(mean);
        acc.plus(d.times(d))
    });
    ss.div_f64(values.len() as f64)
}

fn reference_covariance(x: &[f64], y: &[f64]) -> Dd {
    let mx = reference_mean(x);
    let my = reference_mean(y);
    let acc = x
        .iter()
        .zip(y)
        .fold(Dd::ZERO, |acc, (&a, &b)| acc.plus(Dd::of(a).minus(mx).times(Dd::of(b).minus(my))));
    acc.div_f64(x.len() as f64)
}

fn abs_sum(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).sum()
}

struct Case {
    name: &'static str,
    data: Vec<f64>,
    mask: Option<Vec<u8>>,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let ramp: Vec<f64> = (0..128).map(|i| f64::from(i) * 0.5 - 10.0).collect();
    out.push(Case { name: "ramp", data: ramp.clone(), mask: None });
    out.push(Case {
        name: "ramp_masked",
        mask: Some(mask_bytes(|i| i % 3 != 0, ramp.len())),
        data: ramp,
    });
    // Cancellation: the plain left-to-right sum loses every unit term.
    let mut cancelling = Vec::new();
    for _ in 0..16 {
        cancelling.extend_from_slice(&[1e16, 1.0, -1e16, 1.0]);
    }
    out.push(Case { name: "cancelling", data: cancelling, mask: None });
    let wide = wide_range(500, 0x5eed);
    out.push(Case {
        name: "wide_range_masked",
        mask: Some(mask_bytes(|i| i % 5 != 2, wide.len())),
        data: wide.clone(),
    });
    out.push(Case { name: "wide_range", data: wide, mask: None });
    out.push(Case { name: "offset_noise", data: offset_noise(400, 1.0e8, 7), mask: None });
    let shifted = offset_noise(256, 3.0e7, 11);
    out.push(Case {
        name: "offset_noise_masked",
        mask: Some(mask_bytes(|i| i % 2 == 0, shifted.len())),
        data: shifted,
    });
    out
}

#[test]
fn reference_distinguishes_a_plain_sum_from_the_exact_one() {
    // 1e16 + 1 rounds back to 1e16, so the plain sum of this sequence is 1.
    let values = [1e16, 1.0, -1e16, 1.0];
    let plain: f64 = values.iter().sum();
    assert_eq!(plain, 1.0);
    assert_eq!(reference_sum(&values).value(), 2.0);
    // The reference is not the kernels' accumulation: it keeps what they drop.
    let x = F64VectorView::contiguous(&values);
    let bound = values.len() as f64 * EPS * abs_sum(&values);
    for got in [scalar::masked_sum(x, None).unwrap(), portable::masked_sum(x, None).unwrap()] {
        assert!((got - 2.0).abs() <= bound, "{got} is outside the rounding bound of 2");
    }
}

#[test]
fn sum_and_mean_stay_within_the_rounding_bound_of_the_reference() {
    for case in cases() {
        let mask =
            case.mask.as_deref().map(|bytes| BitMaskView::new(bytes, case.data.len()).unwrap());
        let x = F64VectorView::contiguous(&case.data);
        let kept = selected(&case.data, case.mask.as_deref());
        let n = kept.len() as f64;
        let bound = n * EPS * abs_sum(&kept);
        let exact = reference_sum(&kept).value();
        for (label, got) in [
            ("scalar", scalar::masked_sum(x, mask).unwrap()),
            ("portable", portable::masked_sum(x, mask).unwrap()),
        ] {
            assert!((got - exact).abs() <= bound, "{} {label} sum {got} vs {exact}", case.name);
        }
        let exact_mean = reference_mean(&kept).value();
        for (label, got) in [
            ("scalar", scalar::masked_mean(x, mask).unwrap().unwrap()),
            ("portable", portable::masked_mean(x, mask).unwrap().unwrap()),
        ] {
            let mean_bound = bound / n + EPS * exact_mean.abs();
            assert!(
                (got - exact_mean).abs() <= mean_bound,
                "{} {label} mean {got} vs {exact_mean}",
                case.name
            );
        }
    }
}

#[test]
fn variance_matches_the_reference_for_ill_conditioned_data() {
    for case in cases() {
        // The cancelling and wide-range sets are conditioned for sums, not for a
        // variance that must stay close in relative terms.
        if matches!(case.name, "cancelling" | "wide_range" | "wide_range_masked") {
            continue;
        }
        let mask =
            case.mask.as_deref().map(|bytes| BitMaskView::new(bytes, case.data.len()).unwrap());
        let x = F64VectorView::contiguous(&case.data);
        let kept = selected(&case.data, case.mask.as_deref());
        let exact = reference_variance(&kept).value();
        let tolerance = 8.0 * kept.len() as f64 * EPS;
        for (label, got) in [
            ("scalar", scalar::masked_variance(x, mask).unwrap().unwrap()),
            ("portable", portable::masked_variance(x, mask).unwrap().unwrap()),
        ] {
            let relative = (got - exact).abs() / exact;
            assert!(relative <= tolerance, "{} {label} variance {got} vs {exact}", case.name);
        }
    }
}

#[test]
fn covariance_matches_the_reference_for_ill_conditioned_data() {
    let x_data = offset_noise(300, 5.0e7, 21);
    let mut state = 99_u64;
    let y_data: Vec<f64> =
        x_data.iter().map(|v| 0.7 * (v - 5.0e7) + 2.0e6 + (uniform(&mut state) - 0.5)).collect();
    for keep_every in [None, Some(3_usize)] {
        let bytes = keep_every.map(|k| mask_bytes(|i| i % k != 1, x_data.len()));
        let mask = bytes.as_deref().map(|b| BitMaskView::new(b, x_data.len()).unwrap());
        let x = F64VectorView::contiguous(&x_data);
        let y = F64VectorView::contiguous(&y_data);
        let kx = selected(&x_data, bytes.as_deref());
        let ky = selected(&y_data, bytes.as_deref());
        let exact = reference_covariance(&kx, &ky).value();
        let tolerance = 8.0 * kx.len() as f64 * EPS;
        for (label, got) in [
            ("scalar", scalar::masked_covariance(x, y, mask).unwrap().unwrap()),
            ("portable", portable::masked_covariance(x, y, mask).unwrap().unwrap()),
        ] {
            let relative = (got - exact).abs() / exact.abs();
            assert!(relative <= tolerance, "{label} covariance {got} vs {exact}");
        }
    }
}

#[test]
fn weighted_sum_stays_within_the_rounding_bound_of_the_reference() {
    let x = wide_range(300, 3);
    let mut state = 17_u64;
    let weights: Vec<f64> = (0..x.len())
        .map(|i| match i % 11 {
            // Non-finite and negative weights are treated as zero by the kernels.
            0 => f64::NAN,
            1 => -uniform(&mut state),
            _ => uniform(&mut state) * 3.0,
        })
        .collect();
    let sanitized = |w: f64| if w.is_finite() { w.max(0.0) } else { 0.0 };
    let exact = x
        .iter()
        .zip(&weights)
        .fold(Dd::ZERO, |acc, (&v, &w)| acc.plus(two_prod(sanitized(w), v)))
        .value();
    let scale: f64 = x.iter().zip(&weights).map(|(v, &w)| (sanitized(w) * v).abs()).sum();
    let bound = (x.len() as f64 + 1.0) * EPS * scale;
    for (label, got) in [
        ("scalar", scalar::weighted_sum(&x, &weights).unwrap()),
        ("portable", portable::weighted_sum(&x, &weights).unwrap()),
    ] {
        assert!((got - exact).abs() <= bound, "{label} weighted sum {got} vs {exact}");
    }
}

#[test]
fn a_dropped_term_or_wrong_divisor_is_outside_the_bound() {
    // The reference must be tight enough to reject plausible mistakes.
    let values = offset_noise(64, 1.0e3, 5);
    let mean = reference_mean(&values).value();
    let wrong_divisor = reference_sum(&values).value() / (values.len() as f64 - 1.0);
    assert!((wrong_divisor - mean).abs() > 100.0 * EPS * mean.abs());
    let dropped = reference_sum(&values[1..]).value();
    let full = reference_sum(&values).value();
    assert!((dropped - full).abs() > values.len() as f64 * EPS * abs_sum(&values));
}
