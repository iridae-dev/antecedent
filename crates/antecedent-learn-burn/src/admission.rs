//! Shared CPU, RAM, and device-memory admission for a neural fit.
//!
//! A fit asks the shared [`ExecutionContext`] pool for CPU slots, host RAM, and host scratch
//! through [`antecedent_core::device_reservation::admit_fold`]. A device route additionally
//! asks the [`DeviceReservationPool`] for its device bytes. The order is CPU first, device
//! second: a refused device request returns the CPU hold before the error leaves, so a
//! refusal leaves both accounts as they were. Dropping a [`FitAdmission`], observing a
//! cancelled token, or unwinding out of a failed device fit returns every hold.
//!
//! [`footprint`] sizes the three accounts from the fit shape with checked arithmetic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::device_reservation::{
    CpuAdmissionError, DeviceReservationError, DeviceReservationLease, DeviceReservationPool,
    FoldAdmission, admit_fold,
};
use antecedent_core::execution::CancellationToken;

use super::policy::Route;

/// Bytes a fit holds in each account.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Host RAM: the `f64` design and its standardized copy, the `f32` row-major copy, the
    /// targets, and the host weights.
    pub host_bytes: u64,
    /// Host scratch: the `f32` staging buffers handed to the device or the CPU backend.
    pub scratch_bytes: u64,
    /// Device memory: inputs, activations and their gradients, parameters, and the two Adam
    /// moments. Zero is never returned for a non-empty fit.
    pub device_bytes: u64,
}

/// Why a fit was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdmissionRefusal {
    /// The shared CPU and RAM pool refused or was cancelled.
    Cpu(CpuAdmissionError),
    /// The device pool refused or was cancelled. The CPU hold was returned.
    Device(DeviceReservationError),
    /// A byte count did not fit in `u64`.
    Overflow,
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(error) => write!(f, "cpu admission: {error}"),
            Self::Device(error) => write!(f, "device admission: {error}"),
            Self::Overflow => f.write_str("fit footprint overflowed and was refused"),
        }
    }
}

impl std::error::Error for AdmissionRefusal {}

/// Size the host, scratch, and device accounts of a two-hidden-layer fit.
///
/// # Errors
///
/// [`AdmissionRefusal::Overflow`] when a product or sum does not fit in `u64`.
pub fn footprint(nrows: usize, ncols: usize, hidden: usize) -> Result<Footprint, AdmissionRefusal> {
    let n = nrows as u64;
    let c = ncols as u64;
    let h = hidden as u64;
    let checked = |value: Option<u64>| value.ok_or(AdmissionRefusal::Overflow);
    let design = checked(n.checked_mul(c))?;
    let params = checked(
        c.checked_mul(h)
            .and_then(|a| h.checked_mul(h).and_then(|b| a.checked_add(b)))
            .and_then(|ab| ab.checked_add(h))
            .and_then(|p| p.checked_add(1)),
    )?;
    // f64 design + f64 standardized copy + f32 row-major copy, f64 y + f32 y, f32 weights.
    let host_elements_f64 = checked(design.checked_mul(2).and_then(|d| d.checked_add(n)))?;
    let host_f32 = checked(design.checked_add(n).and_then(|d| d.checked_add(params)))?;
    let host_bytes = checked(
        host_elements_f64
            .checked_mul(8)
            .and_then(|a| host_f32.checked_mul(4).and_then(|b| a.checked_add(b))),
    )?;
    let scratch_bytes = checked(design.checked_add(n).and_then(|d| d.checked_mul(4)))?;
    // Inputs, forward activations with their gradients (two hidden layers plus the output),
    // parameters with their gradients and two Adam moments.
    let activations = checked(n.checked_mul(checked(h.checked_mul(2).map(|v| v + 1))?))?;
    let device_elements = checked(
        design
            .checked_add(n)
            .and_then(|d| activations.checked_mul(2).and_then(|a| d.checked_add(a)))
            .and_then(|d| params.checked_mul(4).and_then(|p| d.checked_add(p))),
    )?;
    let device_bytes = checked(device_elements.checked_mul(4))?;
    Ok(Footprint { host_bytes, scratch_bytes, device_bytes })
}

/// A fit's holds on the shared CPU/RAM pool and, on a device route, the device pool.
#[derive(Debug)]
#[must_use = "dropping the admission returns the cpu, ram, scratch, and device holds"]
pub struct FitAdmission {
    fold: FoldAdmission,
    device: Option<DeviceReservationLease>,
}

impl FitAdmission {
    /// CPU slots held.
    #[must_use]
    pub fn held_threads(&self) -> u32 {
        self.fold.held_threads()
    }

    /// Device bytes held. Zero on a CPU route.
    #[must_use]
    pub fn held_device_bytes(&self) -> u64 {
        self.device.as_ref().map_or(0, |lease| lease.remainder().device_bytes())
    }

    /// Return every hold when `token` is cancelled. `false` leaves the holds in place.
    #[must_use = "a cancelled fit has no result"]
    pub fn observe_cancel(&mut self, token: &CancellationToken) -> bool {
        let cancelled = self.fold.observe_cancel(token);
        let freed = self.device.as_ref().is_some_and(|lease| {
            let returned = lease.release_on_token(token);
            returned.device_bytes() > 0 || returned.spill_bytes() > 0
        });
        if cancelled || freed {
            self.device = None;
        }
        cancelled || freed
    }
}

/// Admit a fit on `route`.
///
/// One CPU slot per fold is the unit of fold concurrency: a pool with `k` slots admits at
/// most `k` folds at once and refuses the rest without taking anything. Device bytes are
/// taken only for [`Route::Wgpu`].
///
/// # Errors
///
/// [`AdmissionRefusal::Cpu`] when the shared pool cannot hold the fold or is cancelled,
/// [`AdmissionRefusal::Device`] when the device pool cannot hold `footprint.device_bytes`
/// (the CPU hold is returned first).
#[must_use = "a refused fit is not permission to run it"]
pub fn admit_fit(
    ctx: &ExecutionContext,
    device_pool: &DeviceReservationPool,
    route: Route,
    footprint: &Footprint,
) -> Result<FitAdmission, AdmissionRefusal> {
    let fold = admit_fold(ctx, 1, footprint.host_bytes, footprint.scratch_bytes)
        .map_err(AdmissionRefusal::Cpu)?;
    let device = match route {
        Route::Cpu => None,
        Route::Wgpu => {
            match device_pool.try_grant_with_token(footprint.device_bytes, 0, &ctx.cancellation) {
                Ok(lease) => Some(lease),
                Err(error) => {
                    drop(fold);
                    return Err(AdmissionRefusal::Device(error));
                }
            }
        }
    };
    Ok(FitAdmission { fold, device })
}

#[cfg(test)]
mod tests {
    use antecedent_core::execution::AdmissionPool;

    use super::*;

    fn context(threads: u32, bytes: u64) -> (ExecutionContext, AdmissionPool) {
        let pool = AdmissionPool::fixed(threads, bytes);
        (ExecutionContext::for_tests(1).with_admission(pool.clone()), pool)
    }

    #[test]
    fn footprint_grows_with_every_dimension_and_refuses_overflow() {
        let base = footprint(100, 4, 8).unwrap();
        for bigger in [footprint(200, 4, 8), footprint(100, 8, 8), footprint(100, 4, 16)] {
            let bigger = bigger.unwrap();
            assert!(bigger.host_bytes > base.host_bytes);
            assert!(bigger.device_bytes > base.device_bytes);
            assert!(bigger.scratch_bytes >= base.scratch_bytes);
        }
        assert!(base.device_bytes > 0 && base.host_bytes > 0 && base.scratch_bytes > 0);
        assert_eq!(footprint(usize::MAX, usize::MAX, 1), Err(AdmissionRefusal::Overflow));
    }

    #[test]
    fn cpu_route_takes_cpu_and_ram_and_no_device_bytes() {
        let fp = footprint(64, 4, 8).unwrap();
        let (ctx, pool) = context(2, fp.host_bytes + fp.scratch_bytes + 10);
        let device = DeviceReservationPool::fixed(1 << 20, 0);
        let admission = admit_fit(&ctx, &device, Route::Cpu, &fp).unwrap();
        assert_eq!(admission.held_threads(), 1);
        assert_eq!(admission.held_device_bytes(), 0);
        assert_eq!(pool.available().threads(), 1);
        assert_eq!(pool.available().bytes(), 10);
        assert_eq!(device.available().device_bytes(), 1 << 20);
        drop(admission);
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(pool.available().bytes(), fp.host_bytes + fp.scratch_bytes + 10);
    }

    #[test]
    fn device_refusal_returns_the_cpu_hold() {
        let fp = footprint(64, 4, 8).unwrap();
        let (ctx, pool) = context(2, 10 * (fp.host_bytes + fp.scratch_bytes));
        let device = DeviceReservationPool::fixed(fp.device_bytes - 1, 0);
        let before = pool.available();
        let error = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap_err();
        assert!(matches!(
            error,
            AdmissionRefusal::Device(DeviceReservationError::ExceedsRemainder { .. })
        ));
        assert_eq!(pool.available(), before);
        assert_eq!(device.available().device_bytes(), fp.device_bytes - 1);
    }

    #[test]
    fn device_route_holds_device_bytes_until_dropped() {
        let fp = footprint(64, 4, 8).unwrap();
        let (ctx, pool) = context(3, 10 * (fp.host_bytes + fp.scratch_bytes));
        let device = DeviceReservationPool::fixed(2 * fp.device_bytes, 0);
        let first = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap();
        assert_eq!(first.held_device_bytes(), fp.device_bytes);
        assert_eq!(device.available().device_bytes(), fp.device_bytes);
        let second = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap();
        assert_eq!(device.available().device_bytes(), 0);
        let third = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap_err();
        assert!(matches!(third, AdmissionRefusal::Device(_)));
        drop(first);
        drop(second);
        assert_eq!(device.available().device_bytes(), 2 * fp.device_bytes);
        assert_eq!(pool.available().threads(), 3);
    }

    #[test]
    fn cancellation_returns_cpu_ram_scratch_and_device_holds() {
        let fp = footprint(64, 4, 8).unwrap();
        let (ctx, pool) = context(2, 10 * (fp.host_bytes + fp.scratch_bytes));
        let device = DeviceReservationPool::fixed(fp.device_bytes, 0);
        let mut admission = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap();
        assert!(!admission.observe_cancel(&ctx.cancellation));
        assert_eq!(device.available().device_bytes(), 0);
        ctx.cancellation.cancel();
        assert!(admission.observe_cancel(&ctx.cancellation));
        assert_eq!(device.available().device_bytes(), fp.device_bytes);
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(pool.available().bytes(), 10 * (fp.host_bytes + fp.scratch_bytes));
        let refused = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap_err();
        assert!(matches!(refused, AdmissionRefusal::Cpu(CpuAdmissionError::Cancelled)));
        assert_eq!(device.available().device_bytes(), fp.device_bytes);
    }

    #[test]
    fn a_failing_worker_returns_every_hold_on_unwind() {
        let fp = footprint(64, 4, 8).unwrap();
        let (ctx, pool) = context(1, 10 * (fp.host_bytes + fp.scratch_bytes));
        let device = DeviceReservationPool::fixed(fp.device_bytes, 0);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = admit_fit(&ctx, &device, Route::Wgpu, &fp).unwrap();
            panic!("simulated device allocation failure");
        }));
        assert!(outcome.is_err());
        assert_eq!(device.available().device_bytes(), fp.device_bytes);
        assert_eq!(pool.available().threads(), 1);
    }
}
