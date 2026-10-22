//! Independent replay of general finite counterfactual transport artifacts
//! (kind `counterfactual_transport_v1`, format version 1).
//!
//! The wire, the digests and the typed errors are estimator-free and live in
//! `antecedent-io-core` ([`antecedent_io_core::counterfactual_transport_wire`]);
//! this module adds what needs the estimator: the producer's point, and the
//! consumer, which trusts nothing in the bytes.
//!
//! A consumer holding only the bytes and an expected identity
//!
//! 1. refuses another version or required feature before reading the payload;
//! 2. refuses stored search limits, memory cap, evaluation budget, law count or
//!    law size above its own bounds before any work (a limits refusal the caller
//!    can retry with larger bounds, not a claim the artifact is invalid);
//! 3. recomputes the premises and data digests and compares them with the stored
//!    ones and with the identity the caller holds independently;
//! 4. for an identified query, re-decides the stored model class, query and
//!    catalog under the stored limits and accepts only the identical
//!    derivation, search receipt included, then rebuilds the laws, re-evaluates
//!    the point under the stored evaluation limits and requires every stored
//!    number bit for bit together with the identity of every law read;
//! 5. for a proven obstruction, re-verifies the two-model witness by exact
//!    rational enumeration against the stored query and catalog and requires the
//!    stored values to be what the verification establishes.
//!
//! Every failure is a [`CounterfactualTransportArtifactError`] naming its slot.
//! The artifact is point-only; this module never produces an interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_estimate::{
    CounterfactualTransportPoint, EstimationError, evaluate_counterfactual_transport,
};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_MAX_LIMITS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
    CounterfactualTransportDecision, CounterfactualTransportDerivation,
    CounterfactualTransportObstruction, CounterfactualTransportProblem,
    CounterfactualTransportQuery, CounterfactualTransportWitness,
    CounterfactualTransportWitnessCheck, IdentificationError, counterfactual_transport_refusal,
    replay_counterfactual_transport, verify_counterfactual_transport_witness,
};

use crate::exact_law_wire::ExactLawWire;
use crate::{ConsumeLimits, IoError};

use antecedent_io_core::counterfactual_transport_wire::{
    COUNTERFACTUAL_TRANSPORT_IDENTIFIED, CounterfactualTransportArtifactError,
    CounterfactualTransportArtifactWire, CounterfactualTransportCitedLawWire,
    CounterfactualTransportDerivationWire, CounterfactualTransportExpectedIdentity,
    CounterfactualTransportObstructionWire, CounterfactualTransportResultWire,
    CounterfactualTransportSearchLimitsWire,
};

/// The artifact error type, as the producer and consumer return it.
type Error = CounterfactualTransportArtifactError;

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct CounterfactualTransportConsumeLimits {
    /// Largest search limits a stored decision may have run under (the
    /// contract's maxima by default).
    pub search: SearchLimits,
    /// Largest search memory cap (bytes) a stored decision may have run under.
    pub search_memory_bytes: u64,
    /// Exact-evaluation operation and depth limits of the point replay.
    pub evaluation: ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
}

impl Default for CounterfactualTransportConsumeLimits {
    fn default() -> Self {
        Self {
            search: COUNTERFACTUAL_TRANSPORT_MAX_LIMITS,
            search_memory_bytes: COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
            evaluation: ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
        }
    }
}

impl CounterfactualTransportConsumeLimits {
    /// The exact-law replay bounds, as the shared checks read them.
    #[must_use]
    pub const fn replay(&self) -> ConsumeLimits {
        ConsumeLimits {
            evaluation: self.evaluation,
            max_support_rows: self.max_support_rows,
            max_laws: self.max_laws,
            max_law_cells: self.max_law_cells,
        }
    }
}

/// The wire form of an evaluated point: IEEE bits and the identity of every law
/// read.
#[must_use]
// doc(hidden): pure projection shared by the producer and the consumer.
#[doc(hidden)]
pub fn counterfactual_transport_result_wire(
    point: &CounterfactualTransportPoint,
) -> CounterfactualTransportResultWire {
    CounterfactualTransportResultWire {
        numerator: point.numerator.to_bits(),
        denominator: point.denominator.to_bits(),
        probability: point.probability.to_bits(),
        conditioning_mass: point.conditioning_mass.map(f64::to_bits),
        conditioning_mass_checked: point.conditioning_mass_checked,
        cells_read: point.cells_read,
        cited_laws: point
            .evidence
            .iter()
            .map(|law| CounterfactualTransportCitedLawWire {
                population: law.population.to_string(),
                regime: law.regime.raw(),
                interventions: law
                    .interventions
                    .iter()
                    .map(|(variable, level)| (variable.raw(), level.to_bits()))
                    .collect(),
                snapshot_identity: law.snapshot_identity.clone(),
                content_identity: law.content_identity.clone(),
            })
            .collect(),
    }
}

/// Seal an identified query: the derivation, the exact laws its point was
/// computed on, the evaluation limits and the point.
///
/// # Errors
/// A bound exceeded, laws that do not encode, or names that do not cover the
/// model class.
pub fn identified_counterfactual_transport_artifact(
    derivation: &CounterfactualTransportDerivation,
    data: &ExactTransportData,
    limits: ExactEvaluationLimits,
    variable_names: &[String],
    point: &CounterfactualTransportPoint,
) -> Result<CounterfactualTransportArtifactWire, Error> {
    CounterfactualTransportArtifactWire::identified(
        derivation,
        data,
        limits,
        variable_names,
        counterfactual_transport_result_wire(point),
    )
}

/// Seal a proven obstruction. `search` and `memory_limit_bytes` are the limits
/// the decision ran under (`memory_limit_bytes` the effective cap: the smaller
/// of the context's hard limit and the contract's).
///
/// # Errors
/// A bound exceeded or names that do not cover the model class.
pub fn obstructed_counterfactual_transport_artifact(
    obstruction: &CounterfactualTransportObstruction,
    search: SearchLimits,
    memory_limit_bytes: u64,
    variable_names: &[String],
) -> Result<CounterfactualTransportArtifactWire, Error> {
    CounterfactualTransportArtifactWire::obstructed(
        &obstruction.problem,
        &obstruction.query,
        &obstruction.catalog,
        CounterfactualTransportSearchLimitsWire {
            operations: search.operations,
            depth: search.depth,
            memory_limit_bytes,
        },
        variable_names,
        CounterfactualTransportObstructionWire {
            witness: obstruction.witness.clone(),
            cells_compared: u64::try_from(obstruction.check.cells_compared).unwrap_or(u64::MAX),
            first_value: obstruction.check.first_value.clone(),
            second_value: obstruction.check.second_value.clone(),
            reason: obstruction.reason.clone(),
        },
    )
}

/// What a replay of an identified artifact re-decided and recomputed.
#[derive(Debug)]
pub struct ConsumedIdentifiedCounterfactualTransport {
    /// The re-decided derivation.
    pub derivation: CounterfactualTransportDerivation,
    /// The re-validated exact laws.
    pub data: ExactTransportData,
    /// The recomputed point.
    pub point: CounterfactualTransportPoint,
    /// The model class.
    pub problem: CounterfactualTransportProblem,
    /// The canonical query.
    pub query: CounterfactualTransportQuery,
}

/// What a replay of an obstruction artifact re-verified.
#[derive(Debug)]
pub struct ConsumedObstructedCounterfactualTransport {
    /// The model class the witness lives in.
    pub problem: CounterfactualTransportProblem,
    /// The canonical query.
    pub query: CounterfactualTransportQuery,
    /// The evidence catalog the declared family was built from.
    pub catalog: antecedent_core::EvidenceCatalog,
    /// The re-verified witness.
    pub witness: CounterfactualTransportWitness,
    /// What the verification established.
    pub check: CounterfactualTransportWitnessCheck,
    /// Why no derivation exists (descriptive, as stored).
    pub reason: String,
}

/// The conclusion a replay established.
#[derive(Debug)]
pub enum ConsumedCounterfactualTransportOutcome {
    /// An identified query and its recomputed point.
    Identified(Box<ConsumedIdentifiedCounterfactualTransport>),
    /// A proven obstruction and its re-verified witness.
    Obstructed(Box<ConsumedObstructedCounterfactualTransport>),
}

/// Everything a replay reconstructed.
#[derive(Debug)]
pub struct ConsumedCounterfactualTransport {
    /// The conclusion.
    pub outcome: ConsumedCounterfactualTransportOutcome,
    /// The decoded artifact.
    pub wire: CounterfactualTransportArtifactWire,
}

fn replay_refusal(error: &IdentificationError) -> Error {
    match counterfactual_transport_refusal(error) {
        Some((code, detail)) => Error::Replay { code, message: format!("{detail}: {error}") },
        None => Error::DerivationMismatch(error.to_string()),
    }
}

fn evaluation_refusal(error: &EstimationError) -> Error {
    match error {
        EstimationError::Refused { code, message }
        | EstimationError::RefusedWithFields { code, message, .. } => {
            Error::Replay { code, message: message.clone() }
        }
        other => Error::LawInvalid(other.to_string()),
    }
}

/// The first stored result field that differs from the recomputed one.
fn first_difference(
    stored: &CounterfactualTransportResultWire,
    replayed: &CounterfactualTransportResultWire,
) -> Option<&'static str> {
    if stored.numerator != replayed.numerator {
        Some("numerator")
    } else if stored.denominator != replayed.denominator {
        Some("denominator")
    } else if stored.probability != replayed.probability {
        Some("probability")
    } else if stored.conditioning_mass != replayed.conditioning_mass
        || stored.conditioning_mass_checked != replayed.conditioning_mass_checked
    {
        Some("conditioning_mass")
    } else if stored.cells_read != replayed.cells_read {
        Some("cells_read")
    } else if stored.cited_laws != replayed.cited_laws {
        Some("cited_laws")
    } else {
        None
    }
}

fn check_limits(
    wire: &CounterfactualTransportArtifactWire,
    limits: &CounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<(), Error> {
    let exceeded = Error::LimitsExceeded;
    if let Some(what) =
        crate::search_bounds(wire.search.operations, wire.search.depth, limits.search)
    {
        return Err(exceeded(what));
    }
    // The replay runs under the producer's stored memory cap, so the consumer's
    // own limit (its consume limits and the context's hard limit) must cover it.
    let memory_cap = ctx
        .memory_limit_bytes()
        .map_or(limits.search_memory_bytes, |hard| hard.min(limits.search_memory_bytes));
    if wire.search.memory_limit_bytes > memory_cap {
        return Err(exceeded("search memory limit"));
    }
    let replay = limits.replay();
    if wire.status == COUNTERFACTUAL_TRANSPORT_IDENTIFIED {
        if let Some(what) = replay.budget(
            wire.operation_limit,
            wire.depth_limit,
            wire.max_support_rows,
            wire.laws.len(),
        ) {
            return Err(exceeded(what));
        }
        if let Some(what) = replay.law_cells(&wire.laws, false) {
            return Err(exceeded(what));
        }
    }
    Ok(())
}

/// Decode and recheck everything under the consumer's limits, then re-decide and
/// recompute (identified) or re-verify the witness (obstruction). No external
/// provider is accessed.
///
/// `expected` is the identity the caller holds independently of the bytes: an
/// artifact whose own digests are consistent but which is not the one the caller
/// expects is refused (`ExpectedIdentityMismatch`).
///
/// # Errors
/// Any [`CounterfactualTransportArtifactError`]; its [`slot`](CounterfactualTransportArtifactError::slot)
/// names the failed check and [`refusal`](CounterfactualTransportArtifactError::refusal) its
/// registered reason code.
pub fn consume_counterfactual_transport_with_limits(
    bytes: &[u8],
    expected: &CounterfactualTransportExpectedIdentity,
    limits: CounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ConsumedCounterfactualTransport, Error> {
    let wire = CounterfactualTransportArtifactWire::decode(bytes)?;
    check_limits(&wire, &limits, ctx)?;
    wire.check_digests(expected)?;
    let problem = wire.problem.to_problem()?;
    let query = wire.query.to_query()?;
    let catalog = wire
        .catalog
        .to_catalog()
        .map_err(|error| Error::Malformed { slot: "catalog", message: error.to_string() })?;
    let outcome = if wire.status == COUNTERFACTUAL_TRANSPORT_IDENTIFIED {
        ConsumedCounterfactualTransportOutcome::Identified(Box::new(consume_identified(
            &wire, problem, query, &catalog, ctx,
        )?))
    } else {
        ConsumedCounterfactualTransportOutcome::Obstructed(Box::new(consume_obstructed(
            &wire, problem, query, catalog, ctx,
        )?))
    };
    Ok(ConsumedCounterfactualTransport { outcome, wire })
}

fn consume_identified(
    wire: &CounterfactualTransportArtifactWire,
    problem: CounterfactualTransportProblem,
    query: CounterfactualTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    ctx: &ExecutionContext,
) -> Result<ConsumedIdentifiedCounterfactualTransport, Error> {
    let (Some(stored), Some(result)) = (&wire.derivation, &wire.result) else {
        return Err(Error::Malformed {
            slot: "status",
            message: "an identified artifact carries a derivation and a result".into(),
        });
    };
    let decision = replay_counterfactual_transport(&problem, &query, catalog, &stored.search, ctx)
        .map_err(|error| replay_refusal(&error))?;
    let derivation = match decision {
        CounterfactualTransportDecision::Identified(derivation) => *derivation,
        CounterfactualTransportDecision::Exhausted(receipt) => {
            return Err(Error::Replay {
                code: antecedent_core::reason_code!("transport_budget_cancel"),
                message: format!(
                    "counterfactual_transport.budget: the replayed decision stopped under the \
                     stored limits: {}",
                    receipt.summary()
                ),
            });
        }
        other => {
            return Err(Error::DerivationMismatch(format!(
                "the stored query no longer decides as identified under the stored limits: {}",
                other.detail_code().unwrap_or("not_identified")
            )));
        }
    };
    if CounterfactualTransportDerivationWire::from_derivation(&derivation) != *stored {
        return Err(Error::DerivationMismatch(
            "the re-decided derivation, search receipt or cited regimes differ from the stored \
             ones"
                .into(),
        ));
    }
    let laws = wire
        .laws
        .iter()
        .map(ExactLawWire::to_law)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Error::LawInvalid(error.to_string()))?;
    let data = ExactTransportData::try_new(laws, wire.max_support_rows)
        .map_err(|error| Error::LawInvalid(error.to_string()))?;
    let evaluation =
        ExactEvaluationLimits { operations: wire.operation_limit, depth: wire.depth_limit };
    let point = evaluate_counterfactual_transport(&derivation, &data, evaluation, ctx)
        .map_err(|error| evaluation_refusal(&error))?;
    if let Some(field) = first_difference(result, &counterfactual_transport_result_wire(&point)) {
        return Err(Error::ResultMismatch { field });
    }
    Ok(ConsumedIdentifiedCounterfactualTransport { derivation, data, point, problem, query })
}

fn consume_obstructed(
    wire: &CounterfactualTransportArtifactWire,
    problem: CounterfactualTransportProblem,
    query: CounterfactualTransportQuery,
    catalog: antecedent_core::EvidenceCatalog,
    ctx: &ExecutionContext,
) -> Result<ConsumedObstructedCounterfactualTransport, Error> {
    let Some(stored) = &wire.obstruction else {
        return Err(Error::Malformed {
            slot: "status",
            message: "an obstruction artifact carries an obstruction".into(),
        });
    };
    if ctx.cancellation.is_cancelled() {
        return Err(Error::Replay {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            message: "counterfactual_transport.budget: the witness verification was cancelled"
                .into(),
        });
    }
    let check =
        verify_counterfactual_transport_witness(&problem, &query, &catalog, &stored.witness)
            .map_err(|error| Error::ObstructionInvalid(error.to_string()))?;
    if check.first_value == check.second_value {
        return Err(Error::ObstructionInvalid("the two models give the same query value".into()));
    }
    if check.first_value != stored.first_value
        || check.second_value != stored.second_value
        || u64::try_from(check.cells_compared).ok() != Some(stored.cells_compared)
    {
        return Err(Error::ObstructionInvalid(
            "the stored values differ from what the witness establishes".into(),
        ));
    }
    Ok(ConsumedObstructedCounterfactualTransport {
        problem,
        query,
        catalog,
        witness: stored.witness.clone(),
        check,
        reason: stored.reason.clone(),
    })
}

/// [`consume_counterfactual_transport_with_limits`] as the io error a facade
/// returns: the typed failure's registered reason code and message.
///
/// # Errors
/// [`IoError::Refused`] carrying the failure's registered code, with the
/// `counterfactual_transport.*` detail and the failed slot in the message.
// doc(hidden): error-shape adapter of the routed consumer; the facade calls it.
#[doc(hidden)]
pub fn consume_counterfactual_transport(
    bytes: &[u8],
    expected: &CounterfactualTransportExpectedIdentity,
    limits: CounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<ConsumedCounterfactualTransport, IoError> {
    consume_counterfactual_transport_with_limits(bytes, expected, limits, ctx).map_err(|error| {
        let (code, message) = error.refusal();
        IoError::Refused { code, message }
    })
}
