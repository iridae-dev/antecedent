//! Encoded width of a linear adjustment design, quoted before the design is built.
//!
//! The design is `[1 | T | Z…]`: an intercept, the treatment, and one column per
//! numeric adjustment variable or one column per generated contrast column of a
//! categorical adjustment variable. Physical planning sizes the design from this
//! quote; preparation then builds a design with exactly this many columns.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;
use antecedent_data::{ColumnView, Contrast, TableView, TabularData};

use crate::error::EstimationError;

/// Rows, encoded columns and categorical expansion of one adjustment design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DesignWidthQuote {
    /// Source rows the design is sized for, before any complete-case selection.
    pub rows: usize,
    /// Encoded columns, including the intercept and the treatment.
    pub columns: usize,
    /// Bytes of expanded contrast columns, source-row aligned, before selection.
    pub expansion_bytes: u64,
    /// Whether every adjustment column is numeric, so the design can be gathered
    /// directly from the source columns without materializing a contrast expansion.
    pub direct: bool,
}

/// Quote the design `[1 | T | Z…]` over `adjustment` under `contrasts`.
///
/// A categorical adjustment variable contributes the columns its contrast
/// generates: one per level for a full-rank indicator, the custom matrix's column
/// count, and one per level but the reference otherwise.
///
/// # Errors
///
/// An adjustment column is missing, a categorical adjustment variable has no
/// contrast coding, or the size arithmetic overflows.
pub fn quote_adjustment_design(
    data: &TabularData,
    adjustment: &[VariableId],
    contrasts: &[(VariableId, Contrast)],
) -> Result<DesignWidthQuote, EstimationError> {
    let overflow = || EstimationError::data_msg("design width arithmetic overflow");
    let rows = data.row_count();
    let mut columns = 2_usize;
    let mut expansion_bytes = 0_u64;
    let mut direct = true;
    for &id in adjustment {
        let width = match data.column(id)? {
            ColumnView::Categorical(column) => {
                direct = false;
                let (_, contrast) =
                    contrasts.iter().find(|(variable, _)| *variable == id).ok_or_else(|| {
                        EstimationError::unsupported(
                            "categorical adjustment requires explicit contrast coding",
                        )
                    })?;
                let levels = column.domain.len();
                let width = match contrast {
                    Contrast::Custom(matrix) => matrix.n_columns,
                    Contrast::FullRankIndicator => levels,
                    _ => levels.saturating_sub(1),
                };
                let cells = (rows as u64).checked_add(levels as u64).ok_or_else(overflow)?;
                let bytes = cells
                    .checked_mul(width as u64)
                    .and_then(|cells| cells.checked_mul(8))
                    .ok_or_else(overflow)?;
                expansion_bytes = expansion_bytes.checked_add(bytes).ok_or_else(overflow)?;
                width
            }
            _ => 1,
        };
        columns = columns.checked_add(width).ok_or_else(overflow)?;
    }
    Ok(DesignWidthQuote { rows, columns, expansion_bytes, direct })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_adjustment_quotes_one_column_each_and_matches_the_built_design() {
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[2.0, 4.0, 8.0, 16.0][..]),
            ("z", &[3.0, 5.0, 7.0, 11.0][..]),
            ("w", &[1.0, 0.0, 2.0, 1.0][..]),
        ])
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let adjustment = [VariableId::from_raw(2), VariableId::from_raw(3)];
        let quote = quote_adjustment_design(&data, &adjustment, &[]).unwrap();
        let expected = DesignWidthQuote { rows: 4, columns: 4, expansion_bytes: 0, direct: true };
        assert_eq!(quote, expected);
        let mask = [true; 4];
        let built =
            crate::prepare::direct_adjustment_design(&data, t, y, &adjustment, &[], &mask).unwrap();
        assert_eq!(built.ncols, quote.columns);
    }
}
