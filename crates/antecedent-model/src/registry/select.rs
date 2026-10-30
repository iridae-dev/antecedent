//! Node typing for mechanism assignment.

use antecedent_core::{ValueType, VariableId};
use antecedent_data::{TableView, TabularData};

use super::{LOW_CARDINALITY_MAX_LEVELS, LOW_CARDINALITY_MIN_ROWS_PER_LEVEL, MechanismTyping};

/// Number of distinct finite values (signed zeroes compare equal) when it is
/// at most `max_levels`, else `None`; `None` also for an all-non-finite column.
///
/// Early-exiting scan: at most `max_levels` keys are ever retained, and the
/// (`max_levels` + 1)-th distinct key bails out immediately, so no full-column
/// sort or copy.
pub(super) fn distinct_level_count(y: &[f64], max_levels: usize) -> Option<usize> {
    let mut seen: Vec<f64> = Vec::with_capacity(max_levels.min(64));
    for &value in y.iter().filter(|v| v.is_finite()) {
        if !seen.contains(&value) {
            if seen.len() == max_levels {
                return None;
            }
            seen.push(value);
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}

/// Whether `y` has at most `max_levels` distinct finite values.
#[cfg(test)]
pub(super) fn is_low_cardinality(y: &[f64], max_levels: usize) -> bool {
    distinct_level_count(y, max_levels).is_some()
}

/// Decide which family list a node draws from.
///
/// A declared discrete value type (binary, categorical, ordinal) is
/// authoritative; an unspecified type (schema migrated from a names-only artifact) is not a
/// declaration. Otherwise the level count decides, but only when the levels
/// are observed repeatedly ([`LOW_CARDINALITY_MIN_ROWS_PER_LEVEL`]): a short
/// column of a continuous variable must not become a multinomial whose
/// counterfactual support is the few values that happened to be observed.
pub(super) fn classify_node(data: &TabularData, var: VariableId, y: &[f64]) -> MechanismTyping {
    let declared_discrete = data.schema().get(var).is_ok_and(|v| {
        matches!(v.value_type, ValueType::Binary | ValueType::Categorical | ValueType::Ordinal)
    });
    if declared_discrete {
        return MechanismTyping::DeclaredDiscrete;
    }
    match distinct_level_count(y, LOW_CARDINALITY_MAX_LEVELS) {
        Some(levels) if y.len() >= levels * LOW_CARDINALITY_MIN_ROWS_PER_LEVEL => {
            MechanismTyping::LowCardinality
        }
        _ => MechanismTyping::Continuous,
    }
}
