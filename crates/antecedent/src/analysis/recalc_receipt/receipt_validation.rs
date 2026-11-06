//! Validation of work recorded at each recalculation stage.

use super::{Counter, ReceiptEntry, ReceiptError, Stage, StageStatus, required_work};

fn entry_recomputed(entry: &ReceiptEntry) -> bool {
    matches!(entry.status, StageStatus::Recomputed { .. })
}

/// Work counted at a stage that does not own it: only a recomputed stage may carry work, and
/// only of its own counters (a provider request also carries its branch's invocations).
fn reject_foreign_work(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    let stage = entry.stage;
    for counter in Counter::ALL.into_iter().chain([Counter::ModelFit]).chain(Counter::EXTENDED) {
        let allowed = entry_recomputed(entry)
            && (counter.stage() == stage
                || matches!(
                    (counter, stage),
                    (Counter::ExternalInvocation(_), Stage::ProviderRequest(_))
                ));
        if entry.counts.get(counter) > 0 && !allowed {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
    }
    Ok(())
}

/// The work a recomputed score-artifact stage must show, which depends on which route ran.
fn score_artifact_required(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    let stage = entry.stage;
    let c = &entry.counts;
    let unexpected = Err(ReceiptError::UnexpectedWork { stage });
    if c.posterior_draws > 0 {
        if c.fold_fits > 0
            || c.score_computations > 0
            || c.provider_bindings > 0
            || c.factor_builds > 0
            || c.program_compilations > 0
        {
            return unexpected;
        }
        Ok(vec![Counter::ModelFit, Counter::PosteriorDraw])
    } else if c.provider_bindings > 0 {
        if c.fold_fits > 0 || c.model_fits > 0 || c.score_computations > 0 {
            return unexpected;
        }
        Ok(vec![Counter::ProviderBinding])
    } else if c.factor_builds > 0 || c.program_compilations > 0 {
        Err(ReceiptError::MissingWork { stage, counter: Counter::ProviderBinding })
    } else if c.score_computations == 0 && c.model_fits > 0 {
        if c.fold_fits > 0 {
            return unexpected;
        }
        Ok(vec![Counter::ModelFit])
    } else {
        Ok(vec![Counter::FoldFit, Counter::ScoreComputation])
    }
}

/// The work a recomputed law stage must show: a summary, an integration, or a reweight.
fn law_required(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    let stage = entry.stage;
    let c = &entry.counts;
    if c.law_summaries > 0 {
        if c.reweights > 0 || c.integrations > 0 || c.factor_evaluations > 0 || c.provider_calls > 0
        {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        Ok(vec![Counter::LawSummary])
    } else if c.integrations > 0 || c.factor_evaluations > 0 || c.provider_calls > 0 {
        if c.reweights > 0 {
            return Err(ReceiptError::UnexpectedWork { stage });
        }
        Ok(vec![Counter::Integration])
    } else {
        Ok(required_work(stage).to_vec())
    }
}

/// The counters a stage's entry must show, given its status and the route its counts reveal.
fn required_counters(entry: &ReceiptEntry) -> Result<Vec<Counter>, ReceiptError> {
    if !entry_recomputed(entry) {
        return Ok(Vec::new());
    }
    match entry.stage {
        Stage::ProviderRequest(branch) => Ok(vec![Counter::ExternalInvocation(branch)]),
        Stage::ScoreArtifact => score_artifact_required(entry),
        Stage::Law => law_required(entry),
        stage => Ok(required_work(stage).to_vec()),
    }
}

pub(super) fn verify_entry(entry: &ReceiptEntry) -> Result<(), ReceiptError> {
    reject_foreign_work(entry)?;
    for counter in required_counters(entry)? {
        if entry.counts.get(counter) == 0 {
            return Err(ReceiptError::MissingWork { stage: entry.stage, counter });
        }
    }
    Ok(())
}
