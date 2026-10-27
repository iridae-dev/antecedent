//! Peak memory and copy counts of the dense least-squares paths, observed in a counting
//! allocator and in the backend's own gather counters.
//!
//! One design copy is the equilibrated factor the pivoted QR needs. Everything else (the
//! scratch buffer, the fallback SVD's view of the caller's design, the solve vectors) must fit
//! beside that one copy. A path that held a second design-sized buffer would exceed the bound
//! asserted here by the size of the design.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, reason = "integration test crate root")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use antecedent_stats::faer_backend::{SolverTraffic, observe_solver_traffic};
use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, IvBlocks, LeastSquaresWorkspace,
    RidgeIntercept, StatsError, fit_2sls, fit_glm, fit_glm_weighted, fit_ridge_with, fit_wls,
};

/// System allocator that tracks live and peak bytes.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded unchanged to `System`; the counters are bookkeeping only.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The counters are process-wide, so the measuring tests run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Peak live bytes above the level at the start, and the traffic the backend reported.
fn measure<T>(step: impl FnOnce() -> T) -> (T, usize, SolverTraffic) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let (value, traffic) = observe_solver_traffic(step);
    (value, PEAK.load(Ordering::Relaxed).saturating_sub(base), traffic)
}

fn design(n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n * p];
    let mut y = vec![0.5; n];
    for row in 0..n {
        x[row] = 1.0;
        for column in 1..p {
            let value = ((row as f64 + 1.0) * (column as f64 + 0.7) * 0.0031).sin();
            x[column * n + row] = value;
            y[row] += value * column as f64 * 0.01;
        }
    }
    (x, y)
}

/// Bytes outside the design copy and scratch that a solve may hold: a handful of
/// `n`-length vectors (right-hand side, residuals, their copies) and small factors.
fn vector_allowance(n: usize) -> usize {
    12 * n * size_of::<f64>() + 64 * 1024
}

#[test]
fn solve_paths_hold_one_design_copy_beside_their_scratch() {
    let _serial = serial();
    let (n, p) = (16_384usize, 64usize);
    let design_bytes = n * p * size_of::<f64>();
    let (x, y) = design(n, p);

    // Full-rank coefficients.
    let (coefficients, peak, traffic) =
        measure(|| FaerBackend.least_squares_coefficients(&x, n, p, &y).unwrap());
    assert_eq!(coefficients.len(), p);
    assert_eq!((traffic.design_gathers, traffic.gathered_cells), (1, (n * p) as u64));
    assert_eq!(traffic.scratch_allocations, 1);
    assert!(peak >= design_bytes, "the gather is the design copy: {peak}");
    let scratch = usize::try_from(traffic.largest_scratch_bytes).unwrap();
    assert!(
        peak <= design_bytes + scratch + vector_allowance(n),
        "peak {peak} exceeds one design copy ({design_bytes}) + scratch ({scratch}) + vectors"
    );

    // Full fit through the shared-scratch entry point.
    let mut workspace = LeastSquaresWorkspace::default();
    let (fit, peak, traffic) =
        measure(|| FaerBackend.least_squares_or_min_norm(&x, n, p, &y, &mut workspace).unwrap());
    assert_eq!(fit.coefficients, coefficients);
    assert_eq!(traffic.design_gathers, 1);
    let scratch = usize::try_from(traffic.largest_scratch_bytes).unwrap();
    assert!(
        peak <= design_bytes + scratch + vectors_with_residuals(n),
        "peak {peak} exceeds one design copy ({design_bytes}) + scratch ({scratch})"
    );
    // The workspace holds residuals and the right-hand side, not a second design.
    assert!(workspace.scratch.is_empty());

    // Ill-conditioned fallback: a duplicated column sends the QR result to the SVD.
    let mut deficient = x.clone();
    for row in 0..n {
        deficient[(p - 1) * n + row] = 2.0 * deficient[2 * n + row];
    }
    let (fallback, peak, traffic) = measure(|| {
        FaerBackend
            .least_squares_or_min_norm(&deficient, n, p, &y, &mut LeastSquaresWorkspace::default())
            .unwrap()
    });
    assert_eq!(fallback.diagnostics.backend, "faer-svd");
    assert_eq!(fallback.rank, p - 1);
    assert_eq!(traffic.design_gathers, 1, "the SVD reads the caller's design in place");
    assert_eq!(traffic.in_place_views, 1);
    assert!(traffic.scratch_allocations <= 2);
    let scratch = usize::try_from(traffic.largest_scratch_bytes).unwrap();
    // The SVD's U factor is design-sized, so the fallback may hold one design-sized buffer
    // at a time (the QR factor, then U) but never both with the scratch of the other phase.
    assert!(
        peak <= design_bytes + scratch + vectors_with_residuals(n) + p * p * size_of::<f64>() * 4,
        "fallback peak {peak} exceeds one design-sized buffer ({design_bytes}) + scratch ({scratch})"
    );

    // A rank refusal made its one gather and kept nothing.
    let (refused, peak, traffic) =
        measure(|| FaerBackend.least_squares_coefficients(&deficient, n, p, &y));
    assert!(matches!(refused, Err(StatsError::RankDeficient { .. })));
    assert_eq!(traffic.design_gathers, 1);
    assert!(peak >= design_bytes);
}

/// A weighted solve that folds the row scale into the factorization gather holds one design
/// copy; building the scaled design first holds that copy and the gather.
#[test]
fn row_scaled_solve_does_not_hold_a_second_design() {
    use antecedent_stats::DenseLinearAlgebra;

    let _serial = serial();

    let (n, p) = (16_384usize, 64usize);
    let design_bytes = n * p * size_of::<f64>();
    let (x, y) = design(n, p);
    let root: Vec<f64> = (0..n).map(|row| 0.5 + (row % 5) as f64 * 0.1).collect();
    let y_w: Vec<f64> = y.iter().zip(&root).map(|(v, s)| v * s).collect();

    let (folded, folded_peak, folded_traffic) = measure(|| {
        FaerBackend
            .row_scaled_least_squares(&x, n, p, &root, &y_w, &mut LeastSquaresWorkspace::default())
            .unwrap()
    });
    let (materialized, materialized_peak, _) = measure(|| {
        let mut scaled = x.clone();
        for column in 0..p {
            for row in 0..n {
                scaled[column * n + row] *= root[row];
            }
        }
        FaerBackend
            .least_squares(&scaled, n, p, &y_w, &mut LeastSquaresWorkspace::default())
            .unwrap()
    });
    assert_eq!(folded.coefficients, materialized.coefficients);
    assert_eq!(folded_traffic.design_gathers, 1);
    assert!(
        materialized_peak >= folded_peak + design_bytes / 2,
        "folded peak {folded_peak}, materialized peak {materialized_peak}, design {design_bytes}"
    );
    let scratch = usize::try_from(folded_traffic.largest_scratch_bytes).unwrap();
    assert!(folded_peak <= design_bytes + scratch + vectors_with_residuals(n));
}

fn vectors_with_residuals(n: usize) -> usize {
    vector_allowance(n) + 4 * n * size_of::<f64>()
}

/// Every estimator family that reaches the backend gathers its design once per solve and holds
/// no second design-sized buffer beside the factor, scratch and `n`-length vectors.
#[test]
fn estimator_families_gather_once_and_hold_one_design() {
    let _serial = serial();
    let (n, p) = (16_384usize, 64usize);
    let design_bytes = n * p * size_of::<f64>();
    let (x, y) = design(n, p);
    let weights: Vec<f64> = (0..n).map(|row| 0.5 + (row % 5) as f64 * 0.1).collect();
    let bound = |traffic: &SolverTraffic, extra_designs: usize| {
        let scratch = usize::try_from(traffic.largest_scratch_bytes).unwrap();
        (1 + extra_designs) * design_bytes + scratch + vectors_with_residuals(n)
    };

    // Weighted least squares: the row weights ride the one gather.
    let (fit, peak, traffic) = measure(|| {
        fit_wls(&x, n, p, &y, &weights, &FaerBackend, &mut LeastSquaresWorkspace::default())
            .unwrap()
    });
    assert_eq!(fit.coefficients.len(), p);
    assert_eq!(traffic.design_gathers, 1);
    assert!(peak <= bound(&traffic, 0), "WLS peak {peak} over one design copy");

    // Gaussian GLM with prior weights takes the same path.
    let glm_design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &y };
    let options = GlmOptions::default();
    let (fit, peak, traffic) = measure(|| {
        fit_glm_weighted(
            GlmFamily::GaussianIdentity,
            glm_design,
            &weights,
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &options,
        )
        .unwrap()
    });
    assert_eq!(fit.coefficients.len(), p);
    assert_eq!(traffic.design_gathers, 1);
    assert!(peak <= bound(&traffic, 0), "weighted Gaussian peak {peak}");

    // IRLS: one gather per iteration, never a scaled design beside it.
    let binary: Vec<f64> = (0..n)
        .map(|row| {
            let signal = 0.4 * x[n + row] + ((row * 7919) % 100) as f64 / 100.0 - 0.5;
            f64::from(signal > 0.0)
        })
        .collect();
    let logit_design = GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &binary };
    let (fit, peak, traffic) = measure(|| {
        fit_glm(
            GlmFamily::BinomialLogit,
            logit_design,
            &FaerBackend,
            &mut LeastSquaresWorkspace::default(),
            &options,
        )
        .unwrap()
    });
    assert!(fit.iterations >= 2);
    assert!(traffic.design_gathers >= u64::from(fit.iterations));
    assert!(peak <= bound(&traffic, 0), "IRLS peak {peak}");

    // Ridge works from the Gram matrix and never copies the design.
    let (_, peak, traffic) =
        measure(|| fit_ridge_with(&x, n, p, &y, 1.0, RidgeIntercept::sniff(&x, n, p)).unwrap());
    assert_eq!(traffic.design_gathers, 0);
    assert!(peak < design_bytes / 4, "ridge peak {peak}");

    // 2SLS: the instrument-plus-exogenous block and the fitted-plus-exogenous block are each
    // assembled once, and each regression gathers once (first stage, the restricted model of
    // the weak-instrument test, second stage).
    let (z_ncols, x_ncols) = (8usize, 8usize);
    let instruments = &x[..n * z_ncols];
    let exogenous = &x[n * z_ncols..n * (z_ncols + x_ncols)];
    let endogenous: Vec<f64> = (0..n)
        .map(|row| instruments[row] + 0.5 * instruments[n + row] + 0.1 * exogenous[n + row])
        .collect();
    let blocks = IvBlocks {
        nrows: n,
        instruments_colmajor: instruments,
        z_ncols,
        exogenous_colmajor: exogenous,
        x_ncols,
    };
    let (fit, peak, traffic) = measure(|| {
        fit_2sls(&blocks, &endogenous, &y, &FaerBackend, &mut LeastSquaresWorkspace::default())
            .unwrap()
    });
    assert_eq!(fit.second_stage.coefficients.len(), 1 + x_ncols);
    assert!(traffic.design_gathers >= 3);
    let widest = (z_ncols + x_ncols) * n * size_of::<f64>();
    let scratch = usize::try_from(traffic.largest_scratch_bytes).unwrap();
    // The stage-one block stays live for the fitted values while stage two is assembled.
    assert!(
        peak <= 2 * widest
            + (1 + x_ncols) * n * size_of::<f64>()
            + scratch
            + 3 * vectors_with_residuals(n),
        "2SLS peak {peak}"
    );
}
