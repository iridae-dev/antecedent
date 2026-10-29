//! Categorical assignment channels as dummy-coded levels with a declared level set.
//!
//! The panel mechanism reads each channel as a nonnegative dose that adstocks and saturates.
//! A categorical assignment has no dose: its levels are alternatives. [`dummy_code_panel`]
//! rewrites each declared categorical channel as one 0/1 indicator channel per non-reference
//! level, so that the fitted response differs between levels by exactly the indicator
//! coefficients (for an indicator, `x / (x + k)` is `1 / (1 + k)` when the level is assigned
//! and zero otherwise). The level set is declared by the caller and must equal the column's
//! domain; a level the data never assign has no identified effect and is refused by name, and
//! an invalid cell stays missing in every indicator of its channel so the missing-data policy
//! applies to it as to any other assignment cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelColumns, JointPanelError};
use antecedent_core::VariableId;
use antecedent_data::{
    ColumnView, PanelData, PanelUnit, SamplingRegularity, TableView, TimeSeriesData,
};
use std::sync::Arc;

/// One categorical assignment channel with its complete declared level set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelCategoricalChannel {
    /// The categorical source column.
    pub variable: VariableId,
    /// Every level the channel may take, as the labels of the column's domain.
    pub levels: Vec<String>,
    /// The reference level, coded as all indicators zero. It must be one of `levels`.
    pub reference: String,
}

/// An indicator channel produced for one non-reference level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelDummyChannel {
    /// The indicator column in the dummy-coded panel.
    pub variable: VariableId,
    /// The categorical source column it was derived from.
    pub source: VariableId,
    /// The level it indicates.
    pub level: String,
}

/// A panel whose declared categorical channels are replaced by their indicator channels.
#[derive(Clone, Debug)]
pub struct DummyCodedPanel {
    /// The rewritten panel: expanded channels, then outcome, confounder and season.
    pub data: PanelData,
    /// Column roles in the rewritten panel; programs and worlds name these channels.
    pub columns: JointPanelColumns,
    /// Each indicator channel with its source and level, in channel order.
    pub indicators: Vec<PanelDummyChannel>,
}

/// Replace each declared categorical channel of `columns` by indicator channels.
///
/// Weights and analysis masks of every unit are carried over unchanged; a category cell that
/// is invalid becomes missing in all indicators of its channel.
///
/// # Errors
/// An undeclared or non-categorical column, a declared level set that differs from the
/// column's domain, a reference outside the level set, fewer than two levels, a declared level
/// that no unit assigns, or an irregular cadence.
pub fn dummy_code_panel(
    data: &PanelData,
    columns: &JointPanelColumns,
    categorical: &[PanelCategoricalChannel],
) -> Result<DummyCodedPanel, JointPanelError> {
    for declared in categorical {
        if !columns.channels.contains(&declared.variable) {
            return Err(JointPanelError::compile(format!(
                "categorical channel {:?} is not one of the declared channels",
                declared.variable
            )));
        }
        if categorical.iter().filter(|other| other.variable == declared.variable).count() != 1 {
            return Err(JointPanelError::compile("a categorical channel is declared twice"));
        }
        let mut unique = declared.levels.clone();
        unique.sort();
        unique.dedup();
        if declared.levels.len() < 2
            || unique.len() != declared.levels.len()
            || !declared.levels.contains(&declared.reference)
        {
            return Err(JointPanelError::compile(
                "a categorical channel needs at least two distinct declared levels, one of them the reference",
            ));
        }
    }
    // Per channel and level: the number of cells assigned, over every unit.
    let mut counts: Vec<Vec<usize>> =
        categorical.iter().map(|declared| vec![0; declared.levels.len()]).collect();
    for unit in data.units() {
        for (index, declared) in categorical.iter().enumerate() {
            let ColumnView::Categorical(view) =
                unit.series.column(declared.variable).map_err(JointPanelError::compile)?
            else {
                return Err(JointPanelError::compile(format!(
                    "channel {:?} is declared categorical but its column is not",
                    declared.variable
                )));
            };
            let mut domain: Vec<&str> =
                view.domain.levels.iter().map(|level| level.label.as_ref()).collect();
            domain.sort_unstable();
            let mut declared_sorted: Vec<&str> =
                declared.levels.iter().map(String::as_str).collect();
            declared_sorted.sort_unstable();
            if domain != declared_sorted {
                return Err(JointPanelError::compile(format!(
                    "the declared level set {declared_sorted:?} differs from the domain {domain:?} of channel {:?}",
                    declared.variable
                )));
            }
            for (row, code) in view.codes.iter().enumerate() {
                if !view.validity.is_valid(row) {
                    continue;
                }
                let label = &view.domain.levels[code.raw() as usize].label;
                let slot = declared
                    .levels
                    .iter()
                    .position(|level| level.as_str() == label.as_ref())
                    .ok_or_else(|| JointPanelError::compile("category code outside the domain"))?;
                counts[index][slot] += 1;
            }
        }
    }
    for (declared, counts) in categorical.iter().zip(&counts) {
        for (level, &count) in declared.levels.iter().zip(counts) {
            if count == 0 {
                return Err(JointPanelError::compile(format!(
                    "declared level {level:?} of channel {:?} is assigned in no unit; its effect is not identified",
                    declared.variable
                )));
            }
        }
    }

    // New column order: expanded channels, outcome, confounder, season.
    struct Slot {
        name: String,
        source: VariableId,
        /// `Some((categorical index, level index))` for an indicator.
        level: Option<(usize, usize)>,
    }
    let mut slots: Vec<Slot> = Vec::new();
    for &channel in &columns.channels {
        let name = data.schema().get(channel).map_err(JointPanelError::compile)?.name.to_string();
        match categorical.iter().position(|declared| declared.variable == channel) {
            None => slots.push(Slot { name, source: channel, level: None }),
            Some(index) => {
                for (level_index, level) in categorical[index].levels.iter().enumerate() {
                    if *level != categorical[index].reference {
                        slots.push(Slot {
                            name: format!("{name}={level}"),
                            source: channel,
                            level: Some((index, level_index)),
                        });
                    }
                }
            }
        }
    }
    let channel_count = slots.len();
    let mut units = Vec::with_capacity(data.units().len());
    for unit in data.units() {
        let series = &unit.series;
        let interval = match series.time_index().regularity {
            SamplingRegularity::Regular { interval_ns } if interval_ns > 0 => interval_ns,
            _ => {
                return Err(JointPanelError::compile(
                    "panel requires explicit positive regular cadence",
                ));
            }
        };
        let mut values: Vec<Vec<f64>> = Vec::new();
        for slot in &slots {
            values.push(match slot.level {
                None => {
                    series.float64_cow(slot.source).map_err(JointPanelError::compile)?.into_owned()
                }
                Some((index, level_index)) => {
                    let ColumnView::Categorical(view) =
                        series.column(slot.source).map_err(JointPanelError::compile)?
                    else {
                        return Err(JointPanelError::compile("categorical column expected"));
                    };
                    let wanted = categorical[index].levels[level_index].as_str();
                    view.codes
                        .iter()
                        .enumerate()
                        .map(|(row, code)| {
                            if !view.validity.is_valid(row) {
                                f64::NAN
                            } else if view.domain.levels[code.raw() as usize].label.as_ref()
                                == wanted
                            {
                                1.0
                            } else {
                                0.0
                            }
                        })
                        .collect()
                }
            });
        }
        for role in [columns.outcome, columns.confounder, columns.season] {
            values.push(series.float64_cow(role).map_err(JointPanelError::compile)?.into_owned());
        }
        let names: Vec<String> = slots
            .iter()
            .map(|slot| slot.name.clone())
            .chain(
                [columns.outcome, columns.confounder, columns.season]
                    .into_iter()
                    .map(|role| data.schema().get(role).map(|variable| variable.name.to_string()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(JointPanelError::compile)?,
            )
            .collect();
        let mut rebuilt = TimeSeriesData::from_f64_columns(
            names.iter().map(String::as_str).zip(values.iter().map(Vec::as_slice)),
            interval,
        )
        .map_err(JointPanelError::compile)?;
        if let Some(weights) = series.storage().weights() {
            rebuilt = rebuilt.with_weights(Arc::from(weights)).map_err(JointPanelError::compile)?;
        }
        if let Some(mask) = series.storage().analysis_mask() {
            rebuilt = rebuilt.with_analysis_mask(mask.clone()).map_err(JointPanelError::compile)?;
        }
        units.push(PanelUnit { unit_id: unit.unit_id, series: rebuilt });
    }
    let id = |position: usize| VariableId::from_raw(u32::try_from(position).unwrap_or(u32::MAX));
    let indicators = slots
        .iter()
        .enumerate()
        .filter_map(|(position, slot)| {
            slot.level.map(|(index, level_index)| PanelDummyChannel {
                variable: id(position),
                source: slot.source,
                level: categorical[index].levels[level_index].clone(),
            })
        })
        .collect();
    Ok(DummyCodedPanel {
        data: PanelData::try_new(Arc::from(units)).map_err(JointPanelError::compile)?,
        columns: JointPanelColumns {
            channels: (0..channel_count).map(id).collect(),
            outcome: id(channel_count),
            confounder: id(channel_count + 1),
            season: id(channel_count + 2),
        },
        indicators,
    })
}
