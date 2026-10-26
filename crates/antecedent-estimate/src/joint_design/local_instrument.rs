//! Encouragement-specific local effects under exclusion and monotonicity.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

/// A local effect belongs to this encouragement and its compliers, not the full population.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalInstrumentDesign {
    /// Stable complete-program-versus-reference encouragement contrast.
    pub key: String,
    /// Source population containing this encouragement's compliers.
    pub population: String,
    /// Exclusion of direct encouragement effects on the outcome.
    pub exclusion: String,
    /// No units whose receipt decreases under encouragement.
    pub monotonicity: String,
    /// Consistency and no interference between independently assigned units.
    pub consistency_and_isolation: String,
}
/// Original independently encouraged unit, including known assignment probability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalInstrumentUnit {
    /// Stable original sampling identity.
    pub id: String,
    /// Encouragement assignment.
    pub assigned: bool,
    /// Receipt of the complete program.
    pub received: bool,
    /// Known Bernoulli encouragement probability.
    pub probability: f64,
    /// Observed response.
    pub outcome: f64,
}
/// Actual ratio output. Posterior draws are retained without asserting existence of ratio moments.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalInstrumentResult {
    /// Local target key.
    pub key: String,
    /// Wald ratio of the two identified score expectations.
    pub effect: f64,
    /// Encouragement effect on response.
    pub intention_to_treat: f64,
    /// Encouragement effect on receipt.
    pub first_stage: f64,
    /// Asymptotic independent-unit ratio influence variance.
    pub variance: f64,
    /// Frequentist normal interval; absent for weak or insufficient support.
    pub interval_95: Option<[f64; 2]>,
    /// Aligned posterior draws (ratio, numerator, denominator), if requested.
    pub posterior: Option<Vec<[f64; 3]>>,
    /// Explicit statistical model; no conversion to a population ATE.
    pub statistical_meaning: String,
}
/// Execute the existing independent-encouragement Wald and paired Bayesian-bootstrap owners.
/// # Errors
/// Missing causal restrictions, invalid original units, unsupported first stage or procedure.
pub fn local_instrument_effect(
    design: &LocalInstrumentDesign,
    units: &[LocalInstrumentUnit],
    posterior_draws: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<LocalInstrumentResult, String> {
    if [
        design.key.as_str(),
        design.population.as_str(),
        design.exclusion.as_str(),
        design.monotonicity.as_str(),
        design.consistency_and_isolation.as_str(),
    ]
    .iter()
    .any(|s| s.trim().is_empty())
        || units.len() < 2
        || units.iter().any(|u| u.id.trim().is_empty())
        || units.iter().map(|u| &u.id).collect::<std::collections::BTreeSet<_>>().len()
            != units.len()
    {
        return Err("local IV requires explicit exclusion, monotonicity, consistency, source population and independent original identities".into());
    }
    let outcomes = units.iter().map(|u| u.outcome).collect::<Vec<_>>();
    let assigned = units.iter().map(|u| u.assigned).collect::<Vec<_>>();
    let received = units.iter().map(|u| u.received).collect::<Vec<_>>();
    let probabilities = units.iter().map(|u| u.probability).collect::<Vec<_>>();
    let point = crate::randomized_scores::complier_wald_effect(
        &outcomes,
        &assigned,
        &received,
        &probabilities,
    )
    .ok_or("local encouragement effect lacks a positive identified receipt first stage")?;
    let posterior = posterior_draws
        .map(|draws| {
            if draws < 2 {
                return Err("local posterior draw count must be at least 2".into());
            }
            let _lease = super::budget::admit_cells(ctx, "local posterior draws", draws, 3)?;
            crate::complier_posterior::bayesian_complier_effect(
                &outcomes,
                &assigned,
                &received,
                &probabilities,
                crate::score_posterior::ScorePosteriorOptions { draws, coverage: 0.95 },
                ctx,
            )
            .map(|p| {
                p.complier
                    .draws
                    .iter()
                    .zip(&p.intention_to_treat.draws)
                    .zip(&p.first_stage.draws)
                    .map(|((ratio, itt), stage)| [*ratio, *itt, *stage])
                    .collect()
            })
            .map_err(|e| e.to_string())
        })
        .transpose()?;
    Ok(LocalInstrumentResult {
        key: design.key.clone(),
        effect: point.effect,
        intention_to_treat: point.intention_to_treat_effect,
        first_stage: point.first_stage,
        variance: point.variance,
        interval_95: point.interval_95,
        posterior,
        statistical_meaning: if posterior_draws.is_some() {
            "paired Dirichlet(1,...,1) Bayesian-bootstrap score-law draws; encouragement-specific complier ratio; retained empirical support and weak-stage refusal; finite-draw first-stage guard is not a posterior support certificate; empirical score law is not a principal-strata likelihood enforcing monotonicity; no assertion of finite ratio posterior moments".into()
        } else {
            "independent-unit Wald ratio influence-function variance; normal interval withheld at weak stage or insufficient support; no weak-instrument robust coverage claim".into()
        },
    })
}
