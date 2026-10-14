//! End-to-end composition verification: one claim-level invalidation graph.
//!
//! The planner in `antecedent_core::recalc` decides, per fixed pipeline stage,
//! whether an artifact is reused, recomputed or refused. This module adds the
//! layer above it: a graph whose nodes are claims and objects (a measurement
//! model, a mechanism composition, a bridge, a ledger result, a transported
//! result, a decision, a cohort source, a selection diagram, a target
//! population) with content identities, and whose edges are typed
//! dependencies. The fixed stage enum is not touched; a stage status, a
//! ledger standing or a claim freshness enters as an [`ObservedEffect`] on the
//! node it concerns.
//!
//! A change upstream classifies every downstream claim as
//! [`Revalidation::StillValid`], [`Revalidation::Recompute`] or
//! [`Revalidation::Withdrawn`], with the reason per dependency kind.
//! Numerical invalidation (data, evidence, model version, structural
//! derivation) is kept separate from withdrawal of a license (support,
//! provider trust, calibration, a retired input). Stage reuse alone never
//! proves a claim still licensed: support, trust and calibration are
//! rechecked at the current coordinates.
//!
//! This module reads graphs and ledgers; it computes no statistic and
//! declares no calibration record. It never turns a point value into an
//! interval: a claim whose calibration no longer holds is withdrawn for
//! calibration, and its point value is left as it was.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod graph;
mod revalidate;
mod view;

pub use graph::{
    COMPOSITION_GRAPH_VERSION, CalibrationDeclaration, ClaimStanding, CompositionEdge,
    CompositionGraph, CompositionNode, DependencyKind, GraphError, MAX_EDGES, MAX_ITEMS_PER_NODE,
    MAX_NODES, MAX_TEXT_BYTES, NodeKind, SupportStanding, TrustStanding,
};
pub use revalidate::{
    CalibrationLineage, CalibrationState, ClaimRevalidation, GraphRevalidation, NumericalCause,
    NumericalReason, ObservedEffect, PriorAuthority, Withdrawal, WithdrawalKind, ledger_effects,
    revalidate,
};
pub use view::{
    BoundaryAssumption, BoundaryTransformation, CompositionView, LineageEdge, LineageEntry,
    MAX_OVERLAP_CHECKS, SharedDataRelation, UnsupportedCoordinate, ViewError,
    shared_data_relations,
};

pub use crate::evidence_ledger::Revalidation;
