//! Resource limits for retained propagation-chain records.

use super::{
    ChainChangeWire, ChainOpWire, Error, PropagationChainLimits, PropagationChainRecordWire,
};
use antecedent_stats::propagation_chain::{MAX_CHAIN_NODES, MAX_CHAIN_ROWS, MAX_EXPRESSION_TERMS};

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
    for text in [&record.construction, &record.reading, &record.output, &record.label] {
        text_ok(text, limits)?;
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
        if let ChainOpWire::Preposterior { noise, .. } = &stage.op {
            if noise.len() > MAX_CHAIN_ROWS {
                return Err(Error::LimitsExceeded("rows"));
            }
        }
        if let ChainOpWire::Reweight { weights } = &stage.op {
            if weights.len() > MAX_CHAIN_NODES {
                return Err(Error::LimitsExceeded("weights"));
            }
        }
        if let ChainOpWire::Expression { terms } = &stage.op {
            if terms.len() > MAX_EXPRESSION_TERMS {
                return Err(Error::LimitsExceeded("expression terms"));
            }
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
