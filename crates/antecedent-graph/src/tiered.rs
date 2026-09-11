//! Tier-rule background over existing ADMG / PAG semantics.
//!
//! Not a new graph class. [`TieredBackground`] constructs an [`Admg`] or
//! [`Pag`] from a declared tier order and certifies the tier-closure
//! adjustment set in `O(p)` when same-tier nodes are [`WithinTier::CoDetermined`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalSchema, VariableId};

use crate::admg::Admg;
use crate::error::GraphError;
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint, MarkedEdge};

/// How edges among nodes that share a tier are interpreted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WithinTier {
    /// Same-tier nodes are non-causal to each other (bidirected within the
    /// tier). The tier-closure set is a generalized adjustment set in `O(p)`.
    CoDetermined,
    /// Within-tier orientation is unknown. Canonical sets stay an envelope.
    Unknown,
}

/// Declared tier order over existing ADMG / PAG semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TieredBackground {
    /// Tiers in causal order (earlier tiers may cause later ones).
    pub tiers: Arc<[Arc<[VariableId]>]>,
    /// Within-tier interpretation.
    pub within_tier: WithinTier,
}

impl TieredBackground {
    /// Construct from an ordered list of tiers.
    ///
    /// # Errors
    ///
    /// Empty tiers, empty names, or a variable appearing in two tiers.
    pub fn new(
        tiers: impl IntoIterator<Item = impl Into<Arc<[VariableId]>>>,
        within_tier: WithinTier,
    ) -> Result<Self, GraphError> {
        let tiers: Arc<[Arc<[VariableId]>]> = tiers.into_iter().map(Into::into).collect();
        if tiers.is_empty() || tiers.iter().any(|t| t.is_empty()) {
            return Err(GraphError::InvalidEndpoints {
                message: "TieredBackground requires non-empty tiers",
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for tier in tiers.iter() {
            for &v in tier.iter() {
                if !seen.insert(v.raw()) {
                    return Err(GraphError::InvalidEndpoints {
                        message: "TieredBackground variables must be unique across tiers",
                    });
                }
            }
        }
        Ok(Self { tiers, within_tier })
    }

    /// Named-tier constructor (`schema` resolves names to ids).
    ///
    /// # Errors
    ///
    /// Unknown names or invalid tier geometry.
    pub fn from_named(
        schema: &CausalSchema,
        tiers: &[Vec<&str>],
        within_tier: WithinTier,
    ) -> Result<Self, GraphError> {
        let mut resolved = Vec::with_capacity(tiers.len());
        for tier in tiers {
            let mut ids = Vec::with_capacity(tier.len());
            for name in tier {
                let id = schema.id_of(name).map_err(|_| GraphError::InvalidEndpoints {
                    message: "unknown tier variable",
                })?;
                ids.push(id);
            }
            resolved.push(Arc::from(ids));
        }
        Self::new(resolved, within_tier)
    }

    /// Tier index of `id`, if declared.
    #[must_use]
    pub fn tier_of(&self, id: VariableId) -> Option<usize> {
        self.tiers.iter().position(|t| t.iter().any(|&v| v == id))
    }

    /// Tier-closure adjustment set: all nodes in tiers `≤ k` except `treatment`,
    /// where `k` is the treatment's tier. `O(p)`.
    ///
    /// # Errors
    ///
    /// Treatment or outcome missing, or outcome not strictly after treatment.
    pub fn tier_closure(
        &self,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Result<Arc<[VariableId]>, GraphError> {
        self.tier_closure_set(&[treatment], outcome)
    }

    /// Closure of a treatment *set*: all nodes in tiers `≤ k` except the
    /// treatments, where `k` is the latest treatment tier. `O(p)`.
    ///
    /// # Errors
    ///
    /// Empty set, a treatment or outcome missing from the tiers, a duplicate
    /// treatment, or outcome not strictly after every treatment.
    pub fn tier_closure_set(
        &self,
        treatments: &[VariableId],
        outcome: VariableId,
    ) -> Result<Arc<[VariableId]>, GraphError> {
        if treatments.is_empty() {
            return Err(GraphError::InvalidEndpoints { message: "treatment set is empty" });
        }
        let y = self
            .tier_of(outcome)
            .ok_or(GraphError::InvalidEndpoints { message: "outcome is not in a declared tier" })?;
        let mut k_max = 0usize;
        let mut seen = std::collections::BTreeSet::new();
        for &treatment in treatments {
            let k = self.tier_of(treatment).ok_or(GraphError::InvalidEndpoints {
                message: "treatment is not in a declared tier",
            })?;
            if !seen.insert(treatment) {
                return Err(GraphError::InvalidEndpoints {
                    message: "treatment set must be distinct",
                });
            }
            if y <= k {
                return Err(GraphError::InvalidEndpoints {
                    message: "outcome must sit in a later tier than every treatment",
                });
            }
            k_max = k_max.max(k);
        }
        let mut set = Vec::new();
        for tier in self.tiers.iter().take(k_max + 1) {
            for &v in tier.iter() {
                if !seen.contains(&v) {
                    set.push(v);
                }
            }
        }
        Ok(Arc::from(set))
    }

    /// Canonical envelope sets for [`WithinTier::Unknown`].
    ///
    /// Never collapses to one set: pretreatment-only and tier-closure.
    ///
    /// # Errors
    ///
    /// Same as [`Self::tier_closure`].
    pub fn unknown_canonical_sets(
        &self,
        treatment: VariableId,
        outcome: VariableId,
    ) -> Result<[Arc<[VariableId]>; 2], GraphError> {
        let k = self.tier_of(treatment).ok_or(GraphError::InvalidEndpoints {
            message: "treatment is not in a declared tier",
        })?;
        let _ = self.tier_closure(treatment, outcome)?;
        let mut pre = Vec::new();
        for tier in self.tiers.iter().take(k) {
            pre.extend(tier.iter().copied());
        }
        let closure = self.tier_closure(treatment, outcome)?;
        Ok([Arc::from(pre), closure])
    }

    /// Materialize the `CoDetermined` background as an ADMG.
    ///
    /// Directed edges run from every earlier-tier node to every later-tier
    /// node. `CoDetermined` tiers get a bidirected clique. Unknown tiers
    /// are refused because an ADMG cannot preserve their circles.
    ///
    /// # Errors
    ///
    /// Schema mismatch or insert failure.
    pub fn to_admg(&self, schema: &CausalSchema) -> Result<Admg, GraphError> {
        if self.within_tier == WithinTier::Unknown {
            return Err(GraphError::InvalidEndpoints {
                message: "Unknown tiers cannot be collapsed to an ADMG; use to_pag or explicit orientation scenarios",
            });
        }
        let n = u32::try_from(schema.len()).map_err(|_| GraphError::TooManyNodes)?;
        let mut g = Admg::with_variables(n);
        for (i, earlier) in self.tiers.iter().enumerate() {
            for later in self.tiers.iter().skip(i + 1) {
                for &a in earlier.iter() {
                    for &b in later.iter() {
                        g.insert_directed(
                            DenseNodeId::from_raw(a.raw()),
                            DenseNodeId::from_raw(b.raw()),
                        )?;
                    }
                }
            }
        }
        if self.within_tier == WithinTier::CoDetermined {
            for tier in self.tiers.iter() {
                for (ia, &a) in tier.iter().enumerate() {
                    for &b in tier.iter().skip(ia + 1) {
                        g.insert_bidirected(
                            DenseNodeId::from_raw(a.raw()),
                            DenseNodeId::from_raw(b.raw()),
                        )?;
                    }
                }
            }
        }
        Ok(g)
    }

    /// Materialize as a PAG: directed between tiers; within-tier circles
    /// when [`WithinTier::Unknown`], bidirected when `CoDetermined`.
    ///
    /// # Errors
    ///
    /// Schema mismatch or insert failure.
    pub fn to_pag(&self, schema: &CausalSchema) -> Result<Pag, GraphError> {
        let n = u32::try_from(schema.len()).map_err(|_| GraphError::TooManyNodes)?;
        let mut g = Pag::with_variables(n);
        for (i, earlier) in self.tiers.iter().enumerate() {
            for later in self.tiers.iter().skip(i + 1) {
                for &a in earlier.iter() {
                    for &b in later.iter() {
                        g.insert_marked(MarkedEdge::directed(
                            DenseNodeId::from_raw(a.raw()),
                            DenseNodeId::from_raw(b.raw()),
                        ))?;
                    }
                }
            }
        }
        for tier in self.tiers.iter() {
            for (ia, &a) in tier.iter().enumerate() {
                for &b in tier.iter().skip(ia + 1) {
                    let edge = match self.within_tier {
                        WithinTier::CoDetermined => MarkedEdge::bidirected(
                            DenseNodeId::from_raw(a.raw()),
                            DenseNodeId::from_raw(b.raw()),
                        ),
                        WithinTier::Unknown => MarkedEdge {
                            a: DenseNodeId::from_raw(a.raw()),
                            b: DenseNodeId::from_raw(b.raw()),
                            at_a: Endpoint::Circle,
                            at_b: Endpoint::Circle,
                            middle: crate::types::MiddleMark::Empty,
                        },
                    };
                    g.insert_marked(edge)?;
                }
            }
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::CausalSchemaBuilder;

    fn schema() -> CausalSchema {
        CausalSchemaBuilder::new()
            .continuous("era")
            .finish()
            .continuous("scale")
            .finish()
            .continuous("design")
            .finish()
            .continuous("t")
            .finish()
            .continuous("m")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap()
    }

    #[test]
    fn closure_is_o_p_and_excludes_treatment() {
        let s = schema();
        let bg = TieredBackground::from_named(
            &s,
            &[vec!["era"], vec!["scale"], vec!["design", "t"], vec!["m"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let t = s.id_of("t").unwrap();
        let y = s.id_of("y").unwrap();
        let set = bg.tier_closure(t, y).unwrap();
        assert!(!set.iter().any(|&v| v == t));
        assert!(set.iter().any(|&v| v == s.id_of("era").unwrap()));
        assert!(set.iter().any(|&v| v == s.id_of("design").unwrap()));
        assert!(!set.iter().any(|&v| v == s.id_of("m").unwrap()));
    }

    #[test]
    fn joint_closure_excludes_every_treatment() {
        let s = schema();
        let bg = TieredBackground::from_named(
            &s,
            &[vec!["era"], vec!["scale"], vec!["design", "t"], vec!["m"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let set = bg
            .tier_closure_set(
                &[s.id_of("design").unwrap(), s.id_of("t").unwrap()],
                s.id_of("y").unwrap(),
            )
            .unwrap();
        assert!(!set.iter().any(|&v| v == s.id_of("design").unwrap()));
        assert!(!set.iter().any(|&v| v == s.id_of("t").unwrap()));
        assert!(set.iter().any(|&v| v == s.id_of("era").unwrap()));
        assert!(set.iter().any(|&v| v == s.id_of("scale").unwrap()));
        assert!(!set.iter().any(|&v| v == s.id_of("m").unwrap()));
    }

    #[test]
    fn unknown_envelope_keeps_two_sets() {
        let s = schema();
        let bg = TieredBackground::from_named(
            &s,
            &[vec!["era", "scale"], vec!["design", "t"], vec!["y"]],
            WithinTier::Unknown,
        )
        .unwrap();
        let [pre, closure] =
            bg.unknown_canonical_sets(s.id_of("t").unwrap(), s.id_of("y").unwrap()).unwrap();
        assert!(pre.len() < closure.len() || pre.len() == 2);
        assert_ne!(pre.as_ref(), closure.as_ref());
    }
}
