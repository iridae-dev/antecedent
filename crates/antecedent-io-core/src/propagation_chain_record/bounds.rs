//! Resource limits for retained propagation-chain records.

use super::{
    ChainChangeWire, ChainOpWire, Error, PropagationChainLimits, PropagationChainRecordWire,
};
use antecedent_stats::propagation_chain::{
    MAX_CHAIN_CELLS, MAX_CHAIN_NODES, MAX_CHAIN_ROWS, MAX_EXPRESSION_TERMS,
};

fn text_ok(text: &str, limits: &PropagationChainLimits) -> Result<(), Error> {
    if text.len() > limits.max_text_bytes {
        Err(Error::LimitsExceeded("text bytes"))
    } else {
        Ok(())
    }
}

pub(super) fn check_bounds(
    record: &PropagationChainRecordWire,
    limits: &PropagationChainLimits,
) -> Result<(), Error> {
    if record.sources.len() + record.stages.len() > MAX_CHAIN_NODES {
        return Err(Error::LimitsExceeded("nodes"));
    }
    if record.changes.len() > limits.max_changes {
        return Err(Error::LimitsExceeded("changes"));
    }
    if record.results.len() > MAX_CHAIN_NODES {
        return Err(Error::LimitsExceeded("results"));
    }
    if record.output_draws.as_ref().is_some_and(|values| values.len() > MAX_CHAIN_ROWS)
        || record.output_weights.as_ref().is_some_and(|values| values.len() > MAX_CHAIN_ROWS)
        || record.output_lower_bounds.as_ref().is_some_and(|values| values.len() > MAX_CHAIN_ROWS)
        || record.output_upper_bounds.as_ref().is_some_and(|values| values.len() > MAX_CHAIN_ROWS)
    {
        return Err(Error::LimitsExceeded("output law rows"));
    }
    if record.joint_outputs.len() > MAX_CHAIN_NODES
        || record.joint_output_draws.as_ref().is_some_and(|columns| {
            columns.len() > MAX_CHAIN_NODES
                || columns.iter().any(|column| column.len() > MAX_CHAIN_ROWS)
                || columns
                    .iter()
                    .try_fold(0_usize, |cells, column| cells.checked_add(column.len()))
                    .is_none_or(|cells| cells > MAX_CHAIN_CELLS)
        })
        || record.joint_output_weights.as_ref().is_some_and(|values| values.len() > MAX_CHAIN_ROWS)
    {
        return Err(Error::LimitsExceeded("joint output law"));
    }
    for text in [&record.construction, &record.reading, &record.output, &record.label] {
        text_ok(text, limits)?;
    }
    if let Some(quantity) = &record.output_quantity {
        if quantity.conditioning.len() > MAX_CHAIN_NODES {
            return Err(Error::LimitsExceeded("output conditions"));
        }
        for text in [
            &quantity.variable_id,
            &quantity.variable_name,
            &quantity.role,
            &quantity.units,
            &quantity.population_id,
            &quantity.regime_id,
            &quantity.functional_id,
            &quantity.transform_id,
        ] {
            text_ok(text, limits)?;
        }
        for condition in &quantity.conditioning {
            text_ok(&condition.variable_id, limits)?;
            text_ok(&condition.value_id, limits)?;
        }
    }
    for output in &record.joint_outputs {
        text_ok(&output.stage, limits)?;
        let quantity = &output.quantity;
        if quantity.conditioning.len() > MAX_CHAIN_NODES {
            return Err(Error::LimitsExceeded("joint output conditions"));
        }
        for text in [
            &quantity.variable_id,
            &quantity.variable_name,
            &quantity.role,
            &quantity.units,
            &quantity.population_id,
            &quantity.regime_id,
            &quantity.functional_id,
            &quantity.transform_id,
        ] {
            text_ok(text, limits)?;
        }
        for condition in &quantity.conditioning {
            text_ok(&condition.variable_id, limits)?;
            text_ok(&condition.value_id, limits)?;
        }
    }
    for source in &record.sources {
        text_ok(&source.id, limits)?;
        if source.values.len() > MAX_CHAIN_ROWS {
            return Err(Error::LimitsExceeded("rows"));
        }
        for factor in &source.factors {
            text_ok(&factor.id, limits)?;
        }
    }
    for stage in &record.stages {
        text_ok(&stage.id, limits)?;
        for factor in &stage.own_factors {
            text_ok(&factor.id, limits)?;
        }
        if let ChainOpWire::Preposterior { noise, .. } = &stage.op
            && noise.len() > MAX_CHAIN_ROWS
        {
            return Err(Error::LimitsExceeded("rows"));
        }
        if let ChainOpWire::Reweight { weights } = &stage.op
            && weights.len() > MAX_CHAIN_NODES
        {
            return Err(Error::LimitsExceeded("weights"));
        }
        if let ChainOpWire::Expression { terms } = &stage.op
            && terms.len() > MAX_EXPRESSION_TERMS
        {
            return Err(Error::LimitsExceeded("expression terms"));
        }
    }
    for change in &record.changes {
        match change {
            ChainChangeWire::Withdraw { source } => text_ok(source, limits)?,
            ChainChangeWire::Reweight { source, log_weights } => {
                text_ok(source, limits)?;
                if log_weights.len() > MAX_CHAIN_ROWS {
                    return Err(Error::LimitsExceeded("rows"));
                }
            }
        }
    }
    for result in &record.results {
        if result.values.len() > limits.max_stage_values {
            return Err(Error::LimitsExceeded("result values"));
        }
        text_ok(&result.stage, limits)?;
    }
    Ok(())
}
