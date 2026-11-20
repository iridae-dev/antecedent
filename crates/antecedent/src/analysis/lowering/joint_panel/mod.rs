//! Joint-panel family lowering: every prepared joint-panel handle carries a lowered program.
//!
//! The joint-panel families prepare through their own entry points, not through
//! `Study::compile`, and lower to the same six layers through `JointPanelLoweredParts`, which
//! names the program cell by its `JointPanelRoute`:
//!
//! - `response`: a fitted joint panel response (`JointPanelResponse`) and a response under a
//!   time-varying mechanism (`JointPanelTimeVarying`);
//! - `imputation`: a response pooled over multiply imputed completed panels
//!   (`JointPanelImputation`);
//! - `search`: program search over a fitted response (`JointPanelProgramSearch`) and held-out
//!   program selection (`JointPanelProgramSelection`);
//! - `experiment`: an experiment conditioning a panel or design law, and the decision and
//!   next-study value read from the conditioned law (`JointPanelExperiment`);
//! - `design`: a design-native joint functional (`JointDesign`).
//!
//! A lowering reads declarations and quotes only: it does not fit, sample, search, impute or
//! encode an artifact, and takes no admission lease. It grants no scientific support: a fitted
//! response stays conditional on the specified panel model unless a retained design proof
//! identifies the target. The support status it records comes from the route's capability unit
//! (`analysis::joint_panel::compiled::support_status`) and does not enter the program identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod design;
pub(crate) mod experiment;
pub(crate) mod imputation;
pub(crate) mod response;
pub(crate) mod search;
