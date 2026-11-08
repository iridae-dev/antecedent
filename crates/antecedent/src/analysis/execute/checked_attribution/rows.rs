//! Retained row selection and temporal attribution row mapping.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AttributionOutcome, CausalError, CausalQuery, CheckedAttributionOperation,
    ExecutionContext, GcmSlot, TableView, TabularData, VariableId,
};

impl CheckedAttributionOperation {
    /// The table without its masked-out rows and the submitted positions of the rows it keeps,
    /// or `None` when no row is hidden.
    pub(super) fn visible_rows(
        data: &TabularData,
    ) -> Result<Option<(TabularData, Vec<usize>)>, CausalError> {
        let Some(mask) = data.storage().analysis_mask().filter(|mask| !mask.is_all_valid()) else {
            return Ok(None);
        };
        let kept: Vec<usize> = (0..data.row_count()).filter(|&row| mask.is_valid(row)).collect();
        if kept.len() < 2 {
            return Err(CausalError::Compile {
                message: "an analysis mask leaves fewer than two rows to fit the mechanisms".into(),
            });
        }
        let storage = data.storage();
        let columns = (0..storage.schema().len())
            .map(|index| {
                storage
                    .column(VariableId::from_raw(u32::try_from(index).unwrap_or(u32::MAX)))?
                    .gather_rows(&kept)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = storage
            .weights()
            .map(|weights| kept.iter().map(|&row| weights[row]).collect::<Arc<[f64]>>());
        let table = TabularData::new(antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            None,
            weights,
        )?);
        Ok(Some((table, kept)))
    }

    /// Submitted row positions as positions in the retained rows; a hidden row drops out.
    pub(super) fn retained_positions(rows: &[usize], kept: &[usize]) -> Arc<[usize]> {
        rows.iter().filter_map(|row| kept.binary_search(row).ok()).collect()
    }

    /// A population selector over the submitted rows as one over the retained rows.
    pub(super) fn retained_selector(
        selector: &antecedent_core::PopulationSelector,
        kept: &[usize],
    ) -> antecedent_core::PopulationSelector {
        use antecedent_core::PopulationSelector;
        match selector {
            PopulationSelector::Rows(rows) => {
                PopulationSelector::Rows(Self::retained_positions(rows, kept))
            }
            PopulationSelector::TimeRange { start, end } => PopulationSelector::TimeRange {
                start: kept.partition_point(|&row| row < *start),
                end: kept.partition_point(|&row| row < *end),
            },
            other => other.clone(),
        }
    }

    pub(super) fn series_model_query(
        query: &CausalQuery,
        ticks: &[usize],
        history: u32,
        source_bound: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            CausalQuery,
            Option<antecedent_core::execution::AdmissionLease>,
            Option<antecedent_core::execution::AdmissionLease>,
        ),
        CausalError,
    > {
        use antecedent_core::{AttributeTarget, PopulationSelector};
        let mut query = query.clone();
        let mut first = None;
        let mut second = None;
        let mut change = |baseline: &mut PopulationSelector,
                          comparison: &mut PopulationSelector|
         -> Result<(), CausalError> {
            let (mapped, lease) =
                Self::series_retained_selector(baseline, ticks, source_bound, budget, ctx)?;
            *baseline = mapped;
            first = lease;
            let (mapped, lease) =
                Self::series_retained_selector(comparison, ticks, source_bound, budget, ctx)?;
            *comparison = mapped;
            second = lease;
            Ok(())
        };
        match &mut query {
            CausalQuery::MechanismChange(q) => change(&mut q.baseline, &mut q.comparison)?,
            CausalQuery::Attribute(q) => match &mut q.target {
                AttributeTarget::DistributionChange { change: c, .. }
                | AttributeTarget::RootCauses { change: c, .. }
                | AttributeTarget::StructureChange { change: c, .. } => {
                    change(&mut c.baseline, &mut c.comparison)?
                }
                AttributeTarget::MechanismChange { query: c, .. } => {
                    change(&mut c.baseline, &mut c.comparison)?
                }
                _ => {}
            },
            _ => {}
        }
        let unit_rows = match &mut query {
            CausalQuery::UnitChange(q) => Some(&mut q.unit_rows),
            CausalQuery::Attribute(q) => match &mut q.target {
                AttributeTarget::Anomaly(a) => Some(&mut a.unit_rows),
                AttributeTarget::UnitChange(a) => Some(&mut a.unit_rows),
                _ => None,
            },
            _ => None,
        };
        if let Some(Some(rows)) = unit_rows {
            let count = u64::try_from(rows.len()).map_err(|_| CausalError::Resource {
                message: "unit row request size overflow".into(),
            })?;
            budget
                .charge_work(count)
                .map_err(|e| CausalError::Resource { message: e.to_string() })?;
            if source_bound.is_some() {
                for &row in rows.iter() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if row < history as usize {
                        return Err(CausalError::Compile {
                            message: format!(
                                "requested original row {row} without complete observed history"
                            ),
                        });
                    }
                }
            }
            let (mapped, lease) = Self::series_retained_selector(
                &PopulationSelector::Rows(rows.clone()),
                ticks,
                source_bound,
                budget,
                ctx,
            )?;
            let PopulationSelector::Rows(mapped) = mapped else { unreachable!() };
            *rows = mapped;
            first = lease;
        }
        Ok((query, first, second))
    }

    pub(super) fn retain_mapped_rows(
        &self,
        count: usize,
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<antecedent_core::execution::AdmissionLease>>, CausalError> {
        if count == 0 {
            return Ok(None);
        }
        let Some(shared) = self.temporal_budget.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal row mapping lost its budget".into(),
            });
        };
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let mut budget = shared.clone();
        let n = u64::try_from(count).map_err(|_| CausalError::Resource {
            message: "temporal result row count overflow".into(),
        })?;
        budget.charge_work(n).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = n.checked_mul(16).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
            CausalError::Resource { message: "temporal result row bytes overflow".into() }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal result row byte limit".into() });
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| {
            crate::error::resource_admission_error(error, "temporal attribution")
        })?;
        Ok(Some(Arc::new(lease)))
    }

    pub(super) fn model_row_count(slot: &GcmSlot) -> Result<usize, CausalError> {
        match slot {
            GcmSlot::Unit(result) => Ok(result.unit_rows.len()),
            GcmSlot::Attribution(payload) => match &payload.outcome {
                AttributionOutcome::Anomaly(scores) => {
                    scores.iter().try_fold(0_usize, |n, score| {
                        n.checked_add(score.rows.len()).ok_or_else(|| CausalError::Resource {
                            message: "temporal result row count overflow".into(),
                        })
                    })
                }
                AttributionOutcome::UnitChange(result) => Ok(result.unit_rows.len()),
                _ => Ok(0),
            },
            _ => Ok(0),
        }
    }

    pub(super) fn remap_model_rows(
        slot: &mut GcmSlot,
        ticks: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        fn remap(
            rows: &mut Arc<[usize]>,
            ticks: &[usize],
            ctx: &ExecutionContext,
        ) -> Result<(), CausalError> {
            for row in Arc::make_mut(rows).iter_mut() {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                *row = *ticks.get(*row).ok_or_else(|| CausalError::Compile {
                    message: "attribution returned an invalid compact row".into(),
                })?;
            }
            Ok(())
        }
        match slot {
            GcmSlot::Unit(result) => remap(&mut result.unit_rows, ticks, ctx)?,
            GcmSlot::Attribution(payload) => match &mut payload.outcome {
                AttributionOutcome::Anomaly(scores) => {
                    for score in scores {
                        remap(&mut score.rows, ticks, ctx)?;
                    }
                }
                AttributionOutcome::UnitChange(result) => remap(&mut result.unit_rows, ticks, ctx)?,
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }

    /// Translate original coordinates to compact positions, including every
    /// occurrence in a gathered refit. The lease remains live with the query.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn series_retained_selector(
        selector: &antecedent_core::PopulationSelector,
        kept: &[usize],
        source_bound: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<
        (antecedent_core::PopulationSelector, Option<antecedent_core::execution::AdmissionLease>),
        CausalError,
    > {
        use antecedent_core::PopulationSelector;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        if let (Some(n), PopulationSelector::TimeRange { end, .. }) = (source_bound, selector)
            && *end > n
        {
            return Err(CausalError::Compile {
                message: format!("temporal population range ends at {end}, beyond {n} source rows"),
            });
        }
        let scans = match selector {
            PopulationSelector::Rows(rows) => rows.len().checked_mul(kept.len()),
            PopulationSelector::TimeRange { .. } => Some(kept.len()),
            _ => return Ok((selector.clone(), None)),
        }
        .ok_or_else(|| CausalError::Resource {
            message: "temporal selector work overflow".into(),
        })?;
        let work = u64::try_from(scans).map_err(|_| CausalError::Resource {
            message: "temporal selector work overflow".into(),
        })?;
        budget
            .charge_work(work)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let mut count = 0_usize;
        match selector {
            PopulationSelector::Rows(rows) => {
                for row in rows.iter() {
                    if source_bound.is_some_and(|n| *row >= n) {
                        return Err(CausalError::Compile {
                            message: format!("temporal population row {row} is out of range"),
                        });
                    }
                    for tick in kept {
                        if ctx.cancellation.is_cancelled() {
                            return Err(CausalError::Cancelled { stage: "temporal attribution" });
                        }
                        if row == tick {
                            count += 1;
                        }
                    }
                }
            }
            PopulationSelector::TimeRange { start, end } => {
                for tick in kept {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if *tick >= *start && *tick < *end {
                        count += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| CausalError::Resource {
                message: "temporal selector size overflow".into(),
            })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal selector byte limit".into() });
        }
        budget
            .charge_work(work)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| {
            crate::error::resource_admission_error(error, "temporal attribution")
        })?;
        let mut positions = Vec::new();
        positions.try_reserve_exact(count).map_err(|_| CausalError::Resource {
            message: "temporal selector allocation refused".into(),
        })?;
        match selector {
            PopulationSelector::Rows(rows) => {
                for row in rows.iter() {
                    for (i, tick) in kept.iter().enumerate() {
                        if ctx.cancellation.is_cancelled() {
                            return Err(CausalError::Cancelled { stage: "temporal attribution" });
                        }
                        if row == tick {
                            positions.push(i);
                        }
                    }
                }
            }
            PopulationSelector::TimeRange { start, end } => {
                for (i, tick) in kept.iter().enumerate() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if *tick >= *start && *tick < *end {
                        positions.push(i);
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok((PopulationSelector::Rows(positions.into()), Some(lease)))
    }
}
