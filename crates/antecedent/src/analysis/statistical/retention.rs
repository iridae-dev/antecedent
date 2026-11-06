//! Admission byte quotes for retained statistical inputs and laws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BoundTransportFunctional, ExactTransportData, IoError, StatisticalTransportInput, err,
    estimate_err,
};

fn retained_memory_overflow() -> IoError {
    err("transport retained provider memory budget overflow")
}

fn retained_add(total: &mut u64, count: usize, width: usize) -> Result<(), IoError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(retained_memory_overflow)?;
    *total = total.checked_add(bytes).ok_or_else(retained_memory_overflow)?;
    Ok(())
}

/// Conservative bound on the known finite law buffers, including axis metadata and Arc headers.
/// This does not claim to account for arbitrary application-owned provider objects.
pub(in crate::analysis) fn retained_law_bytes(data: &ExactTransportData) -> Result<u64, IoError> {
    let mut total = 0;
    for law in data.laws() {
        retained_add(
            &mut total,
            1,
            std::mem::size_of::<antecedent_expr::ExactDiscreteLaw>() + 256,
        )?;
        retained_add(&mut total, law.probabilities().len(), std::mem::size_of::<f64>())?;
        retained_add(&mut total, law.empirical_counts().map_or(0, <[u64]>::len), 8)?;
        retained_add(&mut total, law.population().len() + law.snapshot_identity().len(), 1)?;
        for axis in law.axes() {
            retained_add(&mut total, 1, std::mem::size_of::<antecedent_expr::DiscreteAxis>() + 64)?;
            retained_add(
                &mut total,
                axis.values.len(),
                std::mem::size_of::<antecedent_core::Value>(),
            )?;
            for value in axis.values.iter() {
                if let antecedent_core::Value::Label(label) = value {
                    retained_add(&mut total, label.len(), 1)?;
                }
            }
        }
        retained_add(
            &mut total,
            law.interventions().len(),
            std::mem::size_of::<antecedent_expr::InterventionAssignment>() + 64,
        )?;
    }
    Ok(total)
}

/// Quote the finite output joints from declared domains before fitting them. This covers
/// probabilities and empirical counts even when a learned provider retains only one of them.
pub(in crate::analysis) fn declared_provider_bytes(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
) -> Result<u64, IoError> {
    let supplied = ExactTransportData::try_new(input.supplied.clone(), usize::MAX).map_err(err)?;
    let mut total = retained_law_bytes(&supplied)?;
    for sample in &input.samples {
        let axes = antecedent_estimate::catalog_axes(functional.catalog(), sample)
            .map_err(estimate_err)?;
        let cells = axes.iter().try_fold(1usize, |n, axis| {
            n.checked_mul(axis.values.len()).ok_or_else(retained_memory_overflow)
        })?;
        retained_add(
            &mut total,
            1,
            std::mem::size_of::<antecedent_expr::ExactDiscreteLaw>() + 256,
        )?;
        retained_add(&mut total, cells, 16)?;
        retained_add(&mut total, sample.population.len() + sample.snapshot_identity.len(), 1)?;
        for axis in axes {
            retained_add(&mut total, 1, std::mem::size_of::<antecedent_expr::DiscreteAxis>() + 64)?;
            retained_add(
                &mut total,
                axis.values.len(),
                std::mem::size_of::<antecedent_core::Value>(),
            )?;
        }
        retained_add(
            &mut total,
            sample.interventions.len(),
            std::mem::size_of::<antecedent_expr::InterventionAssignment>() + 64,
        )?;
        for assignment in sample.interventions.iter() {
            if let antecedent_core::Value::Label(label) = &assignment.value {
                retained_add(&mut total, label.len(), 1)?;
            }
        }
    }
    Ok(total)
}

pub(in crate::analysis) fn retained_input_bytes(
    input: &StatisticalTransportInput,
) -> Result<u64, IoError> {
    let data = ExactTransportData::try_new(input.supplied.clone(), usize::MAX).map_err(err)?;
    let mut total = retained_law_bytes(&data)?;
    retained_add(
        &mut total,
        input.supplied.capacity(),
        std::mem::size_of::<antecedent_expr::ExactDiscreteLaw>(),
    )?;
    retained_add(
        &mut total,
        input.samples.capacity(),
        std::mem::size_of::<antecedent_estimate::RegimeSample>(),
    )?;
    for sample in &input.samples {
        retained_add(
            &mut total,
            1,
            std::mem::size_of::<antecedent_estimate::RegimeSample>() + 128,
        )?;
        retained_add(&mut total, sample.population.len() + sample.snapshot_identity.len(), 1)?;
        retained_add(
            &mut total,
            sample.interventions.len(),
            std::mem::size_of::<antecedent_expr::InterventionAssignment>() + 64,
        )?;
        for column in sample.columns.values() {
            retained_add(&mut total, 1, 128)?;
            retained_add(&mut total, column.capacity(), std::mem::size_of::<Option<f64>>())?;
        }
    }
    Ok(total)
}
