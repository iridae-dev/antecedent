//! Bounded-tile streaming of complete fitted-law responses and their linear reductions.
//!
//! The retained full-output route keeps every draw by every world by every period. The
//! routes here evaluate the same shared compiled plan on the same draw order, hold at most
//! one tile of draws live, and hand each tile to a reduction. Slots keep their original
//! positions, numerical failures stay failed slots, and any other failure refuses the
//! whole stream, exactly as in the full-output route.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::time_varying::{TemporalPanelDraw, evaluate_temporal_bound};
use super::{
    Arc, ExecutionContext, PanelData, PanelDraw, PanelError, PanelProgram, PanelResponse,
    check_cancelled, check_memory, evaluate_bound, invalid, likelihood, map_execution_error,
    project_panel_responses, response_storage_bytes, response_workspace_bytes,
    validate_panel_programs,
};
use std::sync::Mutex;

/// What a completed stream evaluated, without any per-draw responses.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelStreamReceipt {
    /// Point responses in the declared world order.
    pub point: Vec<PanelResponse>,
    /// Tiles handed to the reduction.
    pub tiles: usize,
    /// Most draws ever live at once; never above the requested tile size.
    pub peak_tile_draws: usize,
    /// Draw slots that produced a complete response row.
    pub complete: usize,
    /// Draw slots that stayed failed, original position preserved.
    pub failed: usize,
}

type Evaluator<T> = fn(
    &likelihood::CompiledPanel<'_>,
    &T,
    &[PanelProgram],
    &ExecutionContext,
) -> Result<Vec<PanelResponse>, PanelError>;

/// A consumer receives an aligned tile without retaining the complete fitted law.
type TileSink<'a> = dyn FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), PanelError> + 'a;

fn tile_bytes(programs: &[PanelProgram], tile_draws: usize) -> Result<u64, PanelError> {
    let slot = response_storage_bytes(programs)?
        .checked_add(std::mem::size_of::<Option<Vec<PanelResponse>>>() as u64)
        .ok_or_else(|| invalid("streamed tile size overflow"))?;
    slot.checked_mul(u64::try_from(tile_draws).map_err(|_| invalid("tile size overflow"))?)
        .ok_or_else(|| invalid("streamed tile size overflow"))
}

#[allow(clippy::too_many_arguments)]
fn stream_law<T>(
    data: &PanelData,
    point: &T,
    draws: &[Option<T>],
    programs: &[PanelProgram],
    tile_draws: usize,
    ctx: &ExecutionContext,
    evaluate: Evaluator<T>,
    sink: &mut TileSink<'_>,
) -> Result<PanelStreamReceipt, PanelError> {
    if tile_draws == 0 {
        return Err(invalid("a streamed tile must hold at least one draw"));
    }
    validate_panel_programs(data, programs)?;
    check_cancelled(ctx)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let live = tile_draws.min(draws.len().max(1));
    let tile = tile_bytes(programs, live)?
        .checked_add(response_storage_bytes(programs)?)
        .ok_or_else(|| invalid("streamed tile size overflow"))?;
    let workspace = response_workspace_bytes(&compiled)?;
    check_memory(
        ctx,
        tile.checked_add(workspace).ok_or_else(|| invalid("streamed peak size overflow"))?,
    )?;
    let _tile =
        ctx.acquire_admission(0, tile).map_err(|error| PanelError::Resource(error.to_string()))?;
    let _working = ctx
        .acquire_admission(0, workspace)
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let point = evaluate(&compiled, point, programs, ctx)?;
    let mut receipt =
        PanelStreamReceipt { point, tiles: 0, peak_tile_draws: 0, complete: 0, failed: 0 };
    let mut rows: Vec<Option<Vec<PanelResponse>>> = Vec::with_capacity(live);
    for (index, chunk) in draws.chunks(tile_draws).enumerate() {
        check_cancelled(ctx)?;
        rows.clear();
        for draw in chunk {
            check_cancelled(ctx)?;
            rows.push(match draw.as_ref().map(|draw| evaluate(&compiled, draw, programs, ctx)) {
                None | Some(Err(PanelError::Numerical(_))) => None,
                Some(Ok(row)) => Some(row),
                Some(Err(error)) => return Err(error),
            });
        }
        let complete = rows.iter().filter(|row| row.is_some()).count();
        receipt.complete += complete;
        receipt.failed += rows.len() - complete;
        receipt.peak_tile_draws = receipt.peak_tile_draws.max(rows.len());
        receipt.tiles += 1;
        sink(index * tile_draws, &rows)?;
    }
    Ok(receipt)
}

#[allow(clippy::too_many_arguments)]
fn run<T, S>(
    data: &PanelData,
    point: &T,
    draws: &[Option<T>],
    programs: &[PanelProgram],
    tile_draws: usize,
    ctx: &ExecutionContext,
    evaluate: Evaluator<T>,
    sink: S,
) -> Result<PanelStreamReceipt, PanelError>
where
    T: Sync,
    S: FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), PanelError> + Send,
{
    let sink = Mutex::new(sink);
    ctx.try_map_indexed(1, |_, worker| {
        let mut guard = sink.lock().map_err(|_| invalid("streamed reduction was poisoned"))?;
        stream_law(data, point, draws, programs, tile_draws, worker, evaluate, &mut *guard)
    })
    .map_err(map_execution_error)?
    .pop()
    .ok_or_else(|| invalid("missing admitted streamed law worker"))
}

/// Stream a stationary fitted law through `sink` in bounded tiles of `tile_draws` draws.
///
/// `sink` receives the index of the tile's first draw and that tile's aligned slots; a
/// failed slot is `None` at its original position. The tile buffer is reused, so a sink that
/// keeps a row must clone it.
///
/// # Errors
/// An empty tile, invalid programs or draws, resource refusal, cancellation, or any
/// error the sink returns.
pub fn stream_panel_law<S>(
    data: &PanelData,
    point: &PanelDraw,
    draws: &[Option<PanelDraw>],
    programs: &[PanelProgram],
    tile_draws: usize,
    ctx: &ExecutionContext,
    sink: S,
) -> Result<PanelStreamReceipt, PanelError>
where
    S: FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), PanelError> + Send,
{
    run(data, point, draws, programs, tile_draws, ctx, evaluate_bound, sink)
}

/// [`stream_panel_law`] for a temporal mechanism path, keeping selection and path coupling.
///
/// # Errors
/// The refusals of [`stream_panel_law`] plus unrequested extrapolation beyond the path.
pub fn stream_temporal_law<S>(
    data: &PanelData,
    point: &TemporalPanelDraw,
    draws: &[Option<TemporalPanelDraw>],
    programs: &[PanelProgram],
    tile_draws: usize,
    ctx: &ExecutionContext,
    sink: S,
) -> Result<PanelStreamReceipt, PanelError>
where
    S: FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), PanelError> + Send,
{
    run(data, point, draws, programs, tile_draws, ctx, evaluate_temporal_bound, sink)
}

/// One linear functional per world for the point and for every aligned draw slot.
#[derive(Debug)]
pub struct PanelProjectedLaw {
    point: Vec<f64>,
    draws: Vec<Option<Vec<f64>>>,
    receipt: PanelStreamReceipt,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl PanelProjectedLaw {
    /// Projected point value per world.
    #[must_use]
    pub fn point(&self) -> &[f64] {
        &self.point
    }
    /// Projected value per world for each original draw slot; failed slots stay `None`.
    #[must_use]
    pub fn draws(&self) -> &[Option<Vec<f64>>] {
        self.retention.observe_current();
        &self.draws
    }
    /// Tile accounting for the stream that produced these values.
    #[must_use]
    pub fn receipt(&self) -> &PanelStreamReceipt {
        &self.receipt
    }
}

fn project<S>(
    weights: &[Vec<f64>],
    programs: &[PanelProgram],
    slots: usize,
    ctx: &ExecutionContext,
    stream: S,
) -> Result<PanelProjectedLaw, PanelError>
where
    S: FnOnce(
        &mut (dyn FnMut(usize, &[Option<Vec<PanelResponse>>]) -> Result<(), PanelError> + Send),
    ) -> Result<PanelStreamReceipt, PanelError>,
{
    if weights.len() != programs.len()
        || weights.iter().zip(programs).any(|(weights, program)| weights.len() != program.horizon())
    {
        return Err(invalid("projection weights must cover each program horizon exactly"));
    }
    let bytes = slots
        .checked_mul(
            programs
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(std::mem::size_of::<Option<Vec<f64>>>()))
                .ok_or_else(|| invalid("projected law size overflow"))?,
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| invalid("projected law size overflow"))?;
    check_memory(ctx, bytes)?;
    let retention = Arc::new(
        ctx.acquire_admission(0, bytes).map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let mut draws: Vec<Option<Vec<f64>>> = Vec::with_capacity(slots);
    let receipt = stream(&mut |_, tile| {
        for row in tile {
            draws
                .push(row.as_deref().map(|row| project_panel_responses(row, weights)).transpose()?);
        }
        Ok(())
    })?;
    let point = project_panel_responses(&receipt.point, weights)?;
    Ok(PanelProjectedLaw { point, draws, receipt, retention })
}

/// Reduce a stationary fitted law to one linear functional per world without ever holding
/// more than one tile of full response rows. Values equal projecting the full-output rows.
///
/// # Errors
/// The refusals of [`stream_panel_law`], plus weights that do not cover each horizon.
pub fn project_panel_law(
    data: &PanelData,
    point: &PanelDraw,
    draws: &[Option<PanelDraw>],
    programs: &[PanelProgram],
    weights: &[Vec<f64>],
    tile_draws: usize,
    ctx: &ExecutionContext,
) -> Result<PanelProjectedLaw, PanelError> {
    project(weights, programs, draws.len(), ctx, |sink| {
        stream_panel_law(data, point, draws, programs, tile_draws, ctx, |first, tile| {
            sink(first, tile)
        })
    })
}

/// [`project_panel_law`] for a temporal mechanism path.
///
/// # Errors
/// The refusals of [`stream_temporal_law`], plus weights that do not cover each horizon.
pub fn project_temporal_law(
    data: &PanelData,
    point: &TemporalPanelDraw,
    draws: &[Option<TemporalPanelDraw>],
    programs: &[PanelProgram],
    weights: &[Vec<f64>],
    tile_draws: usize,
    ctx: &ExecutionContext,
) -> Result<PanelProjectedLaw, PanelError> {
    project(weights, programs, draws.len(), ctx, |sink| {
        stream_temporal_law(data, point, draws, programs, tile_draws, ctx, |first, tile| {
            sink(first, tile)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_cover_every_slot_in_order_and_match_the_full_output_route() {
        let (data, baseline, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let ctx = ExecutionContext::for_tests(8);
        let mut shifted = baseline.clone();
        shifted.group_intercepts.iter_mut().for_each(|v| *v += 0.25);
        let draws = vec![
            Some(baseline.clone()),
            None,
            Some(shifted.clone()),
            Some(baseline.clone()),
            Some(shifted.clone()),
        ];
        let full =
            super::super::evaluate_panel_law(&data, &baseline, &draws, &programs, &ctx).unwrap();
        let mut seen = Vec::new();
        let mut starts = Vec::new();
        let receipt =
            stream_panel_law(&data, &baseline, &draws, &programs, 2, &ctx, |first, tile| {
                starts.push(first);
                seen.extend(tile.iter().cloned());
                Ok(())
            })
            .unwrap();
        assert_eq!(starts, vec![0, 2, 4]);
        assert_eq!(receipt.tiles, 3);
        assert_eq!(receipt.peak_tile_draws, 2);
        assert_eq!((receipt.complete, receipt.failed), (4, 1));
        assert_eq!(receipt.point, full.point());
        assert_eq!(seen.as_slice(), full.draws());
        // The reduction equals projecting the retained rows, bit for bit, for any tile size.
        let weights = programs
            .iter()
            .map(|program| {
                let mut w = vec![0.0; program.horizon()];
                *w.last_mut().unwrap() = 1.0;
                w[0] = -0.5;
                w
            })
            .collect::<Vec<_>>();
        for tile in [1, 2, 3, 5, 64] {
            let law = project_panel_law(&data, &baseline, &draws, &programs, &weights, tile, &ctx)
                .unwrap();
            assert_eq!(law.point(), project_panel_responses(full.point(), &weights).unwrap());
            for (streamed, retained) in law.draws().iter().zip(full.draws()) {
                let expected = retained
                    .as_deref()
                    .map(|rows| project_panel_responses(rows, &weights).unwrap());
                assert_eq!(streamed, &expected);
            }
            assert!(law.draws()[1].is_none());
            assert!(law.receipt().peak_tile_draws <= tile);
        }
    }

    #[test]
    fn empty_tiles_and_misaligned_weights_refuse_and_cancellation_stops_streaming() {
        let (data, baseline, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let ctx = ExecutionContext::for_tests(8);
        let draws = vec![Some(baseline.clone())];
        assert!(
            stream_panel_law(&data, &baseline, &draws, &programs, 0, &ctx, |_, _| Ok(())).is_err()
        );
        assert!(
            project_panel_law(&data, &baseline, &draws, &programs, &[vec![1.0]], 1, &ctx).is_err()
        );
        let failing = stream_panel_law(&data, &baseline, &draws, &programs, 1, &ctx, |_, _| {
            Err(invalid("sink refused"))
        });
        assert_eq!(failing.unwrap_err(), invalid("sink refused"));
        let cancelled = ExecutionContext::for_tests(8);
        cancelled.cancellation.cancel();
        assert!(
            stream_panel_law(&data, &baseline, &draws, &programs, 1, &cancelled, |_, _| Ok(()))
                .is_err()
        );
    }
}
