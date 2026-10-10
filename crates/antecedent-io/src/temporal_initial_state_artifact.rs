//! Independent artifact for the target-marginal initial-state estimand of the
//! finite two-step sequence (2.3A, X5 `uncertain_initial_state`).
//!
//! Format version 1. The artifact records the initial-state variable and the time
//! order of the sequence's coordinates, the source and target regimes, the graph
//! and proof identities, the ordered action sequence, the **target** initial-state
//! law with its own snapshot id, a summary of the panel snapshot the histories came
//! from, and the two labelled results: the marginalized value
//! `sum_s0 P_target(s0) R(s0)` and, optionally, the fixed-state value `R(s0)` of a
//! single state. A consumer trusts nothing: it rebuilds the law, refuses a
//! source-labelled law or a point state as a target law, re-evaluates both sums
//! from the embedded law and the panel summary with the producer's arithmetic, and
//! accepts only an artifact whose stored values, labels and identities are
//! bit-for-bit what the replay produced.
//!
//! This artifact is not the 2.2 specified-initial-state sequence artifact: it
//! carries a different feature marker, a different `kind`, no exact laws and no
//! catalog, and neither decoder accepts the other's bytes. The two result labels
//! are fixed strings and a consumer refuses a swapped or renamed label, so a
//! fixed-state value is never read as the marginalized one. The claim is point only.

use crate::IoError;
use antecedent_core::{IdentityDomain, reason_code};
use antecedent_estimate::temporal_dependent_interval::TemporalUnitPanel;
use antecedent_estimate::temporal_initial_state::{
    FixedStateEffect, FixedStateQuery, INITIAL_STATE_INFERENCE_CLAIM, InitialStateLaw,
    InitialStatePopulation, InitialStateSpec, MarginalizedEffect, MarginalizedQuery,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The artifact format this reader writes and accepts.
pub const TEMPORAL_INITIAL_STATE_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TEMPORAL_INITIAL_STATE_ARTIFACT_FEATURE: &str = "temporal_initial_state_target_law_v1";
/// The artifact kind; a different kind is a different artifact.
pub const TEMPORAL_INITIAL_STATE_ARTIFACT_KIND: &str = "temporal_initial_state_marginalization";

const PREFIX: &str = "temporal_initial_state_artifact";

/// A consumer-detected change, as a conversion failure with a stable prefix.
pub(crate) fn mismatch(prefix: &str, detail: &str) -> IoError {
    IoError::Convert(format!("{prefix}.{detail}"))
}

/// Lower-case hex of a 64-bit identity.
pub(crate) fn hex64(value: u64) -> String {
    format!("{value:016x}")
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct TemporalInitialStateConsumeLimits {
    /// Most summary rows (initial state, step-2 covariate level) a panel summary may carry.
    pub max_summary_rows: usize,
    /// Most unit ids a panel summary may list.
    pub max_units: usize,
}

impl Default for TemporalInitialStateConsumeLimits {
    fn default() -> Self {
        Self { max_summary_rows: 1_000_000, max_units: 100_000 }
    }
}

/// What the initial state is and where it sits: the premises both temporal
/// extension artifacts bind.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TemporalPremisesWire {
    /// Name of the pre-action state variable.
    pub initial_state_variable: String,
    /// Names of the sequence's coordinates in time order; the state comes first.
    pub time_order: Vec<String>,
    /// Source regime.
    pub source_regime: String,
    /// Target regime.
    pub target_regime: String,
    /// Identity of the graph the sequence was identified on.
    pub graph_id: String,
    /// Identity of the proof the sequence was identified under.
    pub proof_id: String,
}

impl TemporalPremisesWire {
    /// Refuse premises that do not name a state, a time order, two regimes and ids.
    ///
    /// # Errors
    /// `invalid_argument` / `initial_state.invalid_premises`.
    pub fn validate(&self) -> Result<(), IoError> {
        let invalid = |message: &str| IoError::Refused {
            code: reason_code!("invalid_argument"),
            message: format!("initial_state.invalid_premises: {message}"),
        };
        if self.initial_state_variable.is_empty() || self.time_order.is_empty() {
            return Err(invalid("name the initial-state variable and the time order"));
        }
        if self.time_order[0] != self.initial_state_variable {
            return Err(invalid("the initial-state variable must come first in the time order"));
        }
        let mut seen = std::collections::BTreeSet::new();
        if self.time_order.iter().any(|name| name.is_empty() || !seen.insert(name.as_str())) {
            return Err(invalid("the time order needs distinct, non-empty coordinate names"));
        }
        if self.source_regime.is_empty()
            || self.target_regime.is_empty()
            || self.source_regime == self.target_regime
        {
            return Err(invalid("declare distinct, non-empty source and target regimes"));
        }
        if self.graph_id.is_empty() || self.proof_id.is_empty() {
            return Err(invalid("declare the graph id and the proof id"));
        }
        Ok(())
    }
}

/// One support point of a law.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StateMassWire {
    /// State level.
    pub state: u32,
    /// Mass.
    pub mass: f64,
}

/// A finite initial-state law with its population label and own snapshot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InitialStateLawWire {
    /// `target` or `source`; only a target law answers a target-marginal query.
    pub population: String,
    /// The law's own snapshot id.
    pub snapshot_id: String,
    /// Support points and masses, in declared order.
    pub states: Vec<StateMassWire>,
    /// Identity of the population, snapshot, states and masses (hex).
    pub digest: String,
}

impl InitialStateLawWire {
    /// Encode a law.
    #[must_use]
    pub fn from_law(law: &InitialStateLaw) -> Self {
        Self {
            population: match law.population() {
                InitialStatePopulation::Target => "target",
                InitialStatePopulation::Source => "source",
            }
            .into(),
            snapshot_id: law.snapshot_id().into(),
            states: law
                .states()
                .iter()
                .map(|&(state, mass)| StateMassWire { state, mass })
                .collect(),
            digest: hex64(law.digest()),
        }
    }

    /// Rebuild the law (validating it) without trusting the stored digest.
    ///
    /// # Errors
    /// A foreign population label, or a law the core refuses.
    pub fn to_law_unchecked(&self) -> Result<InitialStateLaw, IoError> {
        let population = match self.population.as_str() {
            "target" => InitialStatePopulation::Target,
            "source" => InitialStatePopulation::Source,
            _ => return Err(mismatch(PREFIX, "law_population")),
        };
        let states = self.states.iter().map(|s| (s.state, s.mass)).collect();
        Ok(InitialStateLaw::new(population, self.snapshot_id.clone(), states)?)
    }

    /// Whether the stored digest is the rebuilt law's.
    ///
    /// # Errors
    /// A digest that is not the law's.
    pub fn check_digest(&self, law: &InitialStateLaw) -> Result<(), IoError> {
        if self.digest == hex64(law.digest()) {
            Ok(())
        } else {
            Err(mismatch(PREFIX, "law_digest"))
        }
    }
}

/// Tallies of one `(initial state, step-2 covariate)` cell under the first action.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SummaryRowWire {
    /// Initial state.
    pub s0: u32,
    /// Step-2 covariate level.
    pub l2: u32,
    /// Histories with this state and covariate under the first action.
    pub n_covariate: u64,
    /// Of those, histories that took the second action.
    pub n_cell: u64,
    /// Sum of their outcomes, in panel order.
    pub y_sum: f64,
}

/// One state's contribution to the marginalized response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ContributionWire {
    /// State level.
    pub state: u32,
    /// Target mass.
    pub mass: f64,
    /// Response of the whole sequence given the state.
    pub response: f64,
}

/// Everything about a panel snapshot an initial-state evaluation and a window
/// identity read: the snapshot and its units and time range, and the tallies.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PanelSummaryWire {
    /// Snapshot id.
    pub snapshot_id: String,
    /// Identity of the panel's snapshot, units, times and values (hex).
    pub panel_digest: String,
    /// Unit ids, strictly increasing.
    pub unit_ids: Vec<u64>,
    /// Complete histories over all units.
    pub histories: u64,
    /// Earliest time id of any history.
    pub time_min: Option<u64>,
    /// Latest time id of any history.
    pub time_max: Option<u64>,
    /// Tallies, strictly increasing in `(s0, l2)`.
    pub rows: Vec<SummaryRowWire>,
}

impl PanelSummaryWire {
    /// Summarize a panel for a fixed action sequence.
    #[must_use]
    pub fn from_panel(panel: &TemporalUnitPanel, sequence: [u32; 2]) -> Self {
        let mut cells: BTreeMap<(u32, u32), (u64, u64, f64)> = BTreeMap::new();
        let mut time_min: Option<u64> = None;
        let mut time_max: Option<u64> = None;
        for history in panel.units().iter().flat_map(|u| u.histories.iter()) {
            time_min = Some(time_min.map_or(history.time_id, |t| t.min(history.time_id)));
            time_max = Some(time_max.map_or(history.time_id, |t| t.max(history.time_id)));
            if history.a1 != sequence[0] {
                continue;
            }
            let cell = cells.entry((history.s0, history.l2)).or_insert((0, 0, 0.0));
            cell.0 += 1;
            if history.a2 == sequence[1] {
                cell.1 += 1;
                cell.2 += history.y;
            }
        }
        let mut unit_ids = panel.units().iter().map(|u| u.unit_id).collect::<Vec<_>>();
        unit_ids.sort_unstable();
        Self {
            snapshot_id: panel.snapshot_id().into(),
            panel_digest: hex64(panel.digest()),
            unit_ids,
            histories: panel.history_count() as u64,
            time_min,
            time_max,
            rows: cells
                .into_iter()
                .map(|((s0, l2), (n_covariate, n_cell, y_sum))| SummaryRowWire {
                    s0,
                    l2,
                    n_covariate,
                    n_cell,
                    y_sum,
                })
                .collect(),
        }
    }

    /// Canonical-form and bound checks on a decoded summary.
    ///
    /// # Errors
    /// An over-limit, unsorted or inconsistent summary.
    pub fn check(
        &self,
        prefix: &str,
        limits: &TemporalInitialStateConsumeLimits,
    ) -> Result<(), IoError> {
        if self.rows.len() > limits.max_summary_rows || self.unit_ids.len() > limits.max_units {
            return Err(mismatch(prefix, "limits_exceeded"));
        }
        if self.snapshot_id.is_empty() || self.unit_ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(mismatch(prefix, "panel_summary_not_canonical"));
        }
        let ordered = self.rows.windows(2).all(|w| (w[0].s0, w[0].l2) < (w[1].s0, w[1].l2));
        let counted = self.rows.iter().try_fold(0_u64, |acc, r| acc.checked_add(r.n_covariate));
        let consistent = self.rows.iter().all(|r| r.n_cell <= r.n_covariate && r.y_sum.is_finite());
        if !ordered || !consistent || counted.is_none_or(|n| n > self.histories) {
            return Err(mismatch(prefix, "panel_summary_not_canonical"));
        }
        if self.time_min.is_none() != self.time_max.is_none()
            || self.time_min.zip(self.time_max).is_some_and(|(lo, hi)| lo > hi)
        {
            return Err(mismatch(prefix, "panel_summary_not_canonical"));
        }
        Ok(())
    }

    fn support_gap(s0: u32, what: &str) -> IoError {
        IoError::Refused {
            code: reason_code!("transport_support_failure"),
            message: format!(
                "initial_state.support_gap: initial state {s0} has target mass but {what}"
            ),
        }
    }

    /// Response of the whole sequence given the initial state `s0`.
    ///
    /// # Errors
    /// `initial_state.support_gap` when the state has no support.
    pub fn response(&self, s0: u32) -> Result<f64, IoError> {
        let rows = self.rows.iter().filter(|r| r.s0 == s0).collect::<Vec<_>>();
        let n_state: u64 = rows.iter().map(|r| r.n_covariate).sum();
        if n_state == 0 {
            return Err(Self::support_gap(s0, "no history starts in it under the first action"));
        }
        let mut total = 0.0;
        for row in rows {
            if row.n_cell == 0 {
                return Err(Self::support_gap(
                    s0,
                    &format!("no history with step-2 covariate {} takes the second action", row.l2),
                ));
            }
            total += (row.n_covariate as f64 / n_state as f64) * (row.y_sum / row.n_cell as f64);
        }
        Ok(total)
    }

    /// The target-marginal sum over `law`, with every positive-mass contribution.
    ///
    /// # Errors
    /// `initial_state.support_gap` when a state with positive mass has no support.
    pub fn marginalized(
        &self,
        law: &InitialStateLaw,
    ) -> Result<(f64, Vec<ContributionWire>), IoError> {
        let mut contributions = Vec::with_capacity(law.states().len());
        for &(state, mass) in law.states() {
            if mass <= 0.0 {
                continue;
            }
            contributions.push(ContributionWire { state, mass, response: self.response(state)? });
        }
        let value = contributions.iter().map(|c| c.mass * c.response).sum::<f64>();
        Ok((value, contributions))
    }
}

/// The stored marginalized result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MarginalizedResultWire {
    /// Always `marginalized_initial_state`.
    pub label: String,
    /// `sum_s0 P_target(s0) R(s0)`.
    pub value: f64,
    /// Every positive-mass state with its mass and conditional response.
    pub contributions: Vec<ContributionWire>,
    /// Snapshot id of the target law.
    pub state_snapshot_id: String,
    /// Identity of the target law (hex).
    pub state_law_digest: String,
    /// Snapshot id of the panel.
    pub panel_snapshot_id: String,
    /// Always `point_only`.
    pub inference_claim: String,
}

impl MarginalizedResultWire {
    /// Encode a core result.
    #[must_use]
    pub fn from_effect(effect: &MarginalizedEffect) -> Self {
        Self {
            label: effect.label().into(),
            value: effect.value,
            contributions: effect
                .contributions
                .iter()
                .map(|c| ContributionWire { state: c.s0, mass: c.mass, response: c.response })
                .collect(),
            state_snapshot_id: effect.state_snapshot_id.clone(),
            state_law_digest: hex64(effect.state_law_digest),
            panel_snapshot_id: effect.panel_snapshot_id.clone(),
            inference_claim: effect.inference_claim.into(),
        }
    }
}

/// The stored fixed-state companion result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FixedResultWire {
    /// Always `fixed_initial_state`.
    pub label: String,
    /// The state held fixed.
    pub state: u32,
    /// `R(state)`.
    pub value: f64,
    /// Snapshot id of the panel.
    pub panel_snapshot_id: String,
}

/// The replayed values of a consumed artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialStateReplay {
    /// The re-evaluated marginalized value.
    pub value: f64,
    /// The re-evaluated contributions.
    pub contributions: Vec<ContributionWire>,
    /// The re-evaluated fixed-state `(state, value)`, when the artifact carries one.
    pub fixed: Option<(u32, f64)>,
}

/// The versioned target-marginal initial-state artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalInitialStateArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// Initial-state variable, time order, regimes, graph and proof.
    pub premises: TemporalPremisesWire,
    /// The ordered action sequence.
    pub sequence: [u32; 2],
    /// The target initial-state law and its snapshot.
    pub law: InitialStateLawWire,
    /// The panel snapshot the histories came from.
    pub panel: PanelSummaryWire,
    /// The marginalized result.
    pub marginalized: MarginalizedResultWire,
    /// The fixed-state companion, when one was evaluated.
    pub fixed: Option<FixedResultWire>,
    /// Seal over every field above.
    pub seal: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl TemporalInitialStateArtifactWire {
    /// Build an artifact from the queries and the panel, and replay it once so a
    /// producer never writes an artifact its own consumer would refuse.
    ///
    /// # Errors
    /// Invalid premises, a state without support, or an encoding failure.
    pub fn checked(
        premises: TemporalPremisesWire,
        sequence: [u32; 2],
        query: &MarginalizedQuery,
        fixed: Option<&FixedStateQuery>,
        panel: &TemporalUnitPanel,
    ) -> Result<Self, IoError> {
        premises.validate()?;
        let effect = query.effect(panel)?;
        if effect.sequence != sequence || fixed.is_some_and(|q| q.sequence != sequence) {
            return Err(mismatch(PREFIX, "sequence"));
        }
        let fixed = match fixed {
            Some(q) => {
                let fixed_effect = q.effect(panel)?;
                Some(FixedResultWire {
                    label: fixed_effect.label().into(),
                    state: fixed_effect.s0,
                    value: fixed_effect.value,
                    panel_snapshot_id: fixed_effect.panel_snapshot_id,
                })
            }
            None => None,
        };
        let mut wire = Self {
            version: TEMPORAL_INITIAL_STATE_ARTIFACT_VERSION,
            required_features: vec![TEMPORAL_INITIAL_STATE_ARTIFACT_FEATURE.into()],
            kind: TEMPORAL_INITIAL_STATE_ARTIFACT_KIND.into(),
            premises,
            sequence,
            law: InitialStateLawWire::from_law(query.law()),
            panel: PanelSummaryWire::from_panel(panel, sequence),
            marginalized: MarginalizedResultWire::from_effect(&effect),
            fixed,
            seal: String::new(),
        };
        wire.seal = wire.compute_seal()?;
        wire.verify(&TemporalInitialStateConsumeLimits::default())?;
        Ok(wire)
    }

    /// The seal these fields imply. A consumer never trusts it: a resealed
    /// mutation still fails the replay.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn compute_seal(&self) -> Result<String, IoError> {
        let mut body = self.clone();
        body.seal = String::new();
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &("temporal_initial_state_seal_v1", body),
        )?
        .to_hex())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version, feature or kind first.
    ///
    /// # Errors
    /// An unsupported version, a decoding failure, or a foreign feature or kind.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != TEMPORAL_INITIAL_STATE_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [TEMPORAL_INITIAL_STATE_ARTIFACT_FEATURE]
            || wire.kind != TEMPORAL_INITIAL_STATE_ARTIFACT_KIND
        {
            return Err(mismatch(PREFIX, "unsupported_semantics"));
        }
        Ok(wire)
    }

    /// Replay every stored value from the embedded law and panel summary.
    ///
    /// # Errors
    /// A changed seal, a source-labelled law (`transport_missing_evidence`), a
    /// support gap, or any stored label, identity or value that the replay does
    /// not reproduce bit for bit.
    pub fn verify(
        &self,
        limits: &TemporalInitialStateConsumeLimits,
    ) -> Result<InitialStateReplay, IoError> {
        if self.compute_seal()? != self.seal {
            return Err(mismatch(PREFIX, "seal"));
        }
        self.premises.validate()?;
        let query = MarginalizedQuery::new(
            self.sequence,
            InitialStateSpec::Law(self.law.to_law_unchecked()?),
        )?;
        self.law.check_digest(query.law())?;
        self.panel.check(PREFIX, limits)?;
        let (value, contributions) = self.panel.marginalized(query.law())?;
        self.check_marginalized(query.law(), value, &contributions)?;
        let fixed = self.check_fixed()?;
        Ok(InitialStateReplay { value, contributions, fixed })
    }

    fn check_marginalized(
        &self,
        law: &InitialStateLaw,
        value: f64,
        contributions: &[ContributionWire],
    ) -> Result<(), IoError> {
        let stored = &self.marginalized;
        if stored.label != MarginalizedEffect::LABEL {
            return Err(mismatch(PREFIX, "label"));
        }
        if stored.inference_claim != INITIAL_STATE_INFERENCE_CLAIM {
            return Err(mismatch(PREFIX, "inference_claim"));
        }
        if stored.state_snapshot_id != law.snapshot_id()
            || stored.state_law_digest != hex64(law.digest())
            || stored.panel_snapshot_id != self.panel.snapshot_id
        {
            return Err(mismatch(PREFIX, "identity"));
        }
        let rows_match = stored.contributions.len() == contributions.len()
            && stored.contributions.iter().zip(contributions).all(|(a, b)| {
                a.state == b.state && same(a.mass, b.mass) && same(a.response, b.response)
            });
        if !same(stored.value, value) || !rows_match {
            return Err(mismatch(PREFIX, "value"));
        }
        Ok(())
    }

    fn check_fixed(&self) -> Result<Option<(u32, f64)>, IoError> {
        let Some(stored) = &self.fixed else {
            return Ok(None);
        };
        if stored.label != FixedStateEffect::LABEL {
            return Err(mismatch(PREFIX, "label"));
        }
        if stored.panel_snapshot_id != self.panel.snapshot_id {
            return Err(mismatch(PREFIX, "identity"));
        }
        let value = self.panel.response(stored.state)?;
        if !same(stored.value, value) {
            return Err(mismatch(PREFIX, "value"));
        }
        Ok(Some((stored.state, value)))
    }

    /// Decode, verify and replay.
    ///
    /// # Errors
    /// As [`Self::decode`] and [`Self::verify`].
    pub fn consume(
        bytes: &[u8],
        limits: &TemporalInitialStateConsumeLimits,
    ) -> Result<(Self, InitialStateReplay), IoError> {
        let wire = Self::decode(bytes)?;
        let replay = wire.verify(limits)?;
        Ok((wire, replay))
    }
}
