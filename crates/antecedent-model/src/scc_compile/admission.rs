//! Work and peak-memory estimates for SCC compilation.

use super::SccCompileError;

pub(super) fn work_overflow() -> SccCompileError {
    SccCompileError::Shape { message: "admission work overflows u64" }
}

pub(super) fn plan_work(n: u64, clamps: u64, blocks: &[u64]) -> Result<u64, SccCompileError> {
    let square = n.checked_mul(n).ok_or_else(work_overflow)?;
    let mut work = square.checked_mul(n).ok_or_else(work_overflow)?;
    work = work.checked_add(square).ok_or_else(work_overflow)?;
    let mut earlier = 0u64;
    for &m in blocks {
        let step = block_work(m, earlier, n, clamps)?;
        let condition = condition_work(m)?;
        work = work
            .checked_add(step)
            .and_then(|sum| sum.checked_add(condition))
            .ok_or_else(work_overflow)?;
        earlier = earlier.checked_add(m).ok_or_else(work_overflow)?;
    }
    Ok(work)
}

fn block_work(m: u64, earlier: u64, n: u64, clamps: u64) -> Result<u64, SccCompileError> {
    let m2 = m.checked_mul(m).ok_or_else(work_overflow)?;
    let factor = m2.checked_mul(m).ok_or_else(work_overflow)?;
    let rhs_cols =
        m.checked_add(earlier).and_then(|sum| sum.checked_add(n)).ok_or_else(work_overflow)?;
    let solves = m2.checked_mul(rhs_cols).ok_or_else(work_overflow)?;
    let span =
        m.checked_add(earlier).and_then(|sum| sum.checked_add(n)).ok_or_else(work_overflow)?;
    let cross =
        m.checked_mul(earlier).and_then(|prod| prod.checked_mul(span)).ok_or_else(work_overflow)?;
    let mean = clamps
        .checked_add(earlier)
        .and_then(|sum| sum.checked_add(1))
        .and_then(|sum| m.checked_mul(sum))
        .ok_or_else(work_overflow)?;
    factor
        .checked_add(solves)
        .and_then(|sum| sum.checked_add(cross))
        .and_then(|sum| sum.checked_add(mean))
        .ok_or_else(work_overflow)
}

fn condition_work(m: u64) -> Result<u64, SccCompileError> {
    m.checked_mul(m).and_then(|square| square.checked_add(16)).ok_or_else(work_overflow)
}

fn byte_overflow() -> SccCompileError {
    SccCompileError::Shape { message: "admission bytes overflow u64" }
}

pub(super) fn plan_bytes(n: u64) -> Result<u64, SccCompileError> {
    let square = n.checked_mul(n).ok_or_else(byte_overflow)?;
    // Peak dense storage of one n-by-n block: `14 n²` floats across the covariance states,
    // the factor, the solve right-hand sides and the cross terms live together in
    // `solve_block`, plus `n²` for the covariance the compiled result keeps, plus `16 n`
    // for the mean, the diagonal scaling and the pivots; times 8 bytes.
    let linear = n.checked_mul(16).ok_or_else(byte_overflow)?;
    let floats =
        square.checked_mul(15).and_then(|big| big.checked_add(linear)).ok_or_else(byte_overflow)?;
    floats.checked_mul(8).ok_or_else(byte_overflow)
}
