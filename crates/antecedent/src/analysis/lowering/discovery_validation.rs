//! Lowering of a validation check of a compiled temporal structure search.
//!
//! The check is composed over the search's own lowered program: its scientific problem is the
//! structure-search functional keyed by the search program's identity and the check's canonical
//! encoding, so a check of another search, on other data or with other options is another
//! program. Identification, evidence and the physical layer are the search's: the check reads
//! the same observational law through the same engine. The statistical program declares what the
//! check does to that law: a resampling check (block bootstrap, orientation, regime stability)
//! declares its block resampling at its replicate count, a surrogate check its permutations, a
//! parameter grid or a holdout one evaluation of the decision rule per grid point or split.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ScoreFrequentist, ScoreModel};

use crate::analysis::discovery_validation::DiscoveryCheck;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, DiscoveryValidationLoweredParts, FunctionalKind, LoweredProgram, ProblemTarget,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};

/// Domain tag of the check target a validation program is keyed by.
const CHECK_TARGET_TAG: &str = "antecedent.discovery_validation.target.v1";

fn error(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!(
        "invalid_argument",
        "{}: {message}",
        antecedent_core::reason_details::discovery_validation::LOWERING
    )
}

/// The resampling `check` draws from the search's data, if any.
fn resampling(check: &DiscoveryCheck) -> Option<ResamplingPlan> {
    match check {
        DiscoveryCheck::BlockBootstrap { replicates, .. }
        | DiscoveryCheck::Orientation { replicates, .. }
        | DiscoveryCheck::RegimeStability { replicates, .. } => Some(ResamplingPlan {
            method: ResamplingMethod::BlockBootstrap,
            replicates: *replicates,
            unit: ResampleUnit::CalendarBlock,
        }),
        DiscoveryCheck::FalsePositive { replicates, .. } => Some(ResamplingPlan {
            method: ResamplingMethod::Permutation,
            replicates: *replicates,
            unit: ResampleUnit::Row,
        }),
        DiscoveryCheck::Alpha { .. }
        | DiscoveryCheck::LagWindow { .. }
        | DiscoveryCheck::CiTest { .. }
        | DiscoveryCheck::SyntheticNull { .. }
        | DiscoveryCheck::EnvironmentHoldout { .. } => None,
    }
}

/// Lower `check` of the compiled temporal search `search` into its own six-layer program.
///
/// # Errors
///
/// A search program that is not a temporal structure search, or a check whose score name is
/// blank.
pub(crate) fn lower(
    search: &LoweredProgram,
    check: &DiscoveryCheck,
) -> Result<LoweredProgram, CausalError> {
    if search.route() != "DiscoverTemporal" {
        return Err(error(format!(
            "a discovery validation check runs on a compiled temporal structure search, not on \
             the {} route",
            search.route()
        )));
    }
    let mut writer = CanonicalWriter::new(CHECK_TARGET_TAG);
    writer.bytes(search.identity().as_bytes());
    check.write(&mut writer);
    let query = writer.finish();
    let child = search.scientific();
    let scientific = ScientificProblem {
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::StructureSearch,
            outcome: None,
            query,
        },
        identity_scope: search.identity_scope(),
        runtime_query: None,
        plan_id: Arc::from(format!("discovery_validation_{}.v1", check.id())),
        data: child.data,
        graph_class: child.graph_class,
        query_variables: Arc::clone(&child.query_variables),
        source_digest: child.source_digest,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let score = ScoreModel::new(format!("{}_discovery_check", check.id()))
        .map_err(|declaration| error(declaration.to_string()))?;
    let resampling = resampling(check);
    let statistical = StatisticalProgram {
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        resampling,
        posterior: None,
        ..search.statistical().clone()
    };
    let claim = ClaimDemand {
        uncertainty: if resampling.is_some() {
            UncertaintyReading::Sampling
        } else {
            UncertaintyReading::None
        },
        kind: None,
    };
    Ok(DiscoveryValidationLoweredParts {
        scientific,
        identification: search.identification_layer().clone(),
        statistical,
        evidence: search.evidence().clone(),
        physical: search.physical().clone(),
        claim,
    }
    .finish())
}
