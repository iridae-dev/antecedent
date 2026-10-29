//! General ID of a whole-program joint intervention on an unrolled panel graph.
//!
//! A panel program sets every channel at every period. Its observational
//! identification is a statement about a point graph over the unrolled
//! per-period variables: pretreatment context, one node per channel and period,
//! and one outcome node per period. [`UnrolledPanelSpec`] builds that ADMG from
//! the declared dependence structure (channel-to-outcome carryover reach, outcome
//! and channel persistence, history feedback of past outcomes onto later channels,
//! and declared latent confounding between named nodes), and
//! [`identify_joint_panel_intervention`] runs the complete ID algorithm for the
//! joint `do(all channels, all periods)` on the requested outcome periods.
//!
//! The result is the nonparametric functional of the observed law, or a verified
//! hedge. It is a statement about the graph only: a fitted parametric mechanism
//! (for example a Gaussian/AR panel) realizes this functional only under its own
//! restrictions, so the functional here is never an identification of that fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};

use crate::{
    HedgeCertificate, IdIdentifier, IdentificationError, IdentificationResult,
    IdentificationStatus, IdentificationWorkspace,
};

/// A node of the unrolled panel graph.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PanelNode {
    /// A pretreatment context variable, a parent of every channel and outcome node.
    Context(usize),
    /// Channel `channel` at `period`.
    Channel {
        /// Channel index.
        channel: usize,
        /// Period index.
        period: usize,
    },
    /// The outcome at `period`.
    Outcome(usize),
}

/// The declared dependence structure of a panel program, per period.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnrolledPanelSpec {
    /// Periods, at least one.
    pub periods: usize,
    /// Channels set by the program, at least one.
    pub channels: usize,
    /// Pretreatment context variables.
    pub contexts: usize,
    /// Channel `(c, t)` reaches outcomes `t ..= t + carryover_reach` (within the periods).
    pub carryover_reach: usize,
    /// The outcome at `t - 1` is a parent of the outcome at `t`.
    pub outcome_persistence: bool,
    /// Channel `(c, t - 1)` is a parent of channel `(c, t)`.
    pub channel_persistence: bool,
    /// The outcome at `t - 1` is a parent of every channel at `t`: the assignment
    /// responds to the history (time-varying confounding by an observed outcome).
    pub history_feedback: bool,
    /// Declared unmeasured common causes, as bidirected edges between nodes.
    pub latent_confounding: Vec<(PanelNode, PanelNode)>,
}

/// The unrolled ADMG with its node naming.
#[derive(Clone, Debug)]
pub struct UnrolledPanelGraph {
    spec: UnrolledPanelSpec,
    admg: Admg,
}

impl UnrolledPanelGraph {
    /// Build the unrolled graph.
    ///
    /// # Errors
    /// No periods or channels, a latent edge naming a node outside the program or joining a
    /// node to itself, or a graph construction failure.
    pub fn build(spec: UnrolledPanelSpec) -> Result<Self, IdentificationError> {
        if spec.periods == 0 || spec.channels == 0 {
            return Err(IdentificationError::unsupported(
                "an unrolled panel graph needs at least one period and one channel",
            ));
        }
        let mut graph = Self { admg: Admg::with_variables(0), spec };
        let n = graph.node_count();
        let mut admg = Admg::with_variables(u32::try_from(n).map_err(|_| {
            IdentificationError::unsupported("unrolled panel graph exceeds the node bound")
        })?);
        let spec = &graph.spec;
        let d = |node: PanelNode| -> Result<DenseNodeId, IdentificationError> {
            graph.node_index(node).map(|i| DenseNodeId::from_raw(i as u32)).ok_or_else(|| {
                IdentificationError::unsupported("a named node lies outside the panel program")
            })
        };
        for z in 0..spec.contexts {
            for t in 0..spec.periods {
                for c in 0..spec.channels {
                    admg.insert_directed(
                        d(PanelNode::Context(z))?,
                        d(PanelNode::Channel { channel: c, period: t })?,
                    )?;
                }
                admg.insert_directed(d(PanelNode::Context(z))?, d(PanelNode::Outcome(t))?)?;
            }
        }
        for t in 0..spec.periods {
            for c in 0..spec.channels {
                let channel = PanelNode::Channel { channel: c, period: t };
                for s in t..spec.periods.min(t + spec.carryover_reach + 1) {
                    admg.insert_directed(d(channel)?, d(PanelNode::Outcome(s))?)?;
                }
                if t > 0 {
                    if spec.channel_persistence {
                        admg.insert_directed(
                            d(PanelNode::Channel { channel: c, period: t - 1 })?,
                            d(channel)?,
                        )?;
                    }
                    if spec.history_feedback {
                        admg.insert_directed(d(PanelNode::Outcome(t - 1))?, d(channel)?)?;
                    }
                }
            }
            if t > 0 && spec.outcome_persistence {
                admg.insert_directed(d(PanelNode::Outcome(t - 1))?, d(PanelNode::Outcome(t))?)?;
            }
        }
        for &(a, b) in &spec.latent_confounding {
            if a == b {
                return Err(IdentificationError::unsupported(
                    "a latent confounder joins two distinct nodes",
                ));
            }
            admg.insert_bidirected(d(a)?, d(b)?)?;
        }
        graph.admg = admg;
        Ok(graph)
    }

    fn node_count(&self) -> usize {
        self.spec.contexts + self.spec.periods * (self.spec.channels + 1)
    }

    /// Dense index of a node (contexts first, then each period's channels followed by its
    /// outcome), or `None` outside the program.
    #[must_use]
    pub fn node_index(&self, node: PanelNode) -> Option<usize> {
        let s = &self.spec;
        match node {
            PanelNode::Context(z) if z < s.contexts => Some(z),
            PanelNode::Channel { channel, period }
                if channel < s.channels && period < s.periods =>
            {
                Some(s.contexts + period * (s.channels + 1) + channel)
            }
            PanelNode::Outcome(period) if period < s.periods => {
                Some(s.contexts + period * (s.channels + 1) + s.channels)
            }
            _ => None,
        }
    }

    /// The variable id of a node (its dense index).
    #[must_use]
    pub fn variable(&self, node: PanelNode) -> Option<VariableId> {
        self.node_index(node).map(|i| VariableId::from_raw(i as u32))
    }

    /// The declared structure.
    #[must_use]
    pub fn spec(&self) -> &UnrolledPanelSpec {
        &self.spec
    }

    /// The unrolled graph.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.admg
    }

    /// Every channel node at every period, period-major.
    #[must_use]
    pub fn program_nodes(&self) -> Vec<VariableId> {
        let mut out = Vec::new();
        for period in 0..self.spec.periods {
            for channel in 0..self.spec.channels {
                out.extend(self.variable(PanelNode::Channel { channel, period }));
            }
        }
        out
    }
}

/// The general-ID verdict on a whole-program joint intervention.
#[derive(Clone, Debug)]
pub enum UnrolledPanelIdentification {
    /// A nonparametric functional of the observed law.
    Identified(Box<IdentificationResult>),
    /// A hedge: no functional of the observed law exists for this graph.
    NotIdentified {
        /// The hedge, verified against the unrolled graph.
        hedge: Box<HedgeCertificate>,
        /// The full result.
        result: Box<IdentificationResult>,
    },
}

/// Identify `P(outcomes at the given periods | do(every channel at every period))`.
///
/// # Errors
/// No outcome period, an outcome period outside the program, a failure of the ID run, or a
/// hedge that does not verify against the unrolled graph.
pub fn identify_joint_panel_intervention(
    graph: &UnrolledPanelGraph,
    outcome_periods: &[usize],
) -> Result<UnrolledPanelIdentification, IdentificationError> {
    if outcome_periods.is_empty() {
        return Err(IdentificationError::unsupported("name at least one outcome period"));
    }
    let outcomes = outcome_periods
        .iter()
        .map(|&p| {
            graph.variable(PanelNode::Outcome(p)).ok_or_else(|| {
                IdentificationError::unsupported("an outcome period lies outside the program")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let treatments = graph.program_nodes();
    let interventions: Vec<_> =
        treatments.iter().map(|&t| Intervention::set(t, Value::f64(1.0))).collect();
    let query = CausalQuery::Distribution(
        InterventionalDistributionQuery::new(outcomes[0], interventions)
            .with_outcomes(outcomes.clone()),
    );
    let id = IdIdentifier::new();
    let prepared = id.prepare(graph.admg())?;
    let mut workspace = IdentificationWorkspace::default();
    let result = id.identify(&prepared, &query, &mut workspace)?;
    match result.status {
        IdentificationStatus::NonparametricallyIdentified => {
            Ok(UnrolledPanelIdentification::Identified(Box::new(result)))
        }
        IdentificationStatus::NotIdentified => {
            let hedge = result.hedge.clone().ok_or_else(|| {
                IdentificationError::unsupported("a non-identified verdict carried no hedge")
            })?;
            hedge.verify(&prepared, &treatments, &outcomes)?;
            Ok(UnrolledPanelIdentification::NotIdentified {
                hedge: Box::new(hedge),
                result: Box::new(result),
            })
        }
        _ => Err(IdentificationError::unsupported(
            "general ID returned a status other than identified or not identified",
        )),
    }
}
