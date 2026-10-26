//! Bind existing joint-cell reductions to their checked scientific target.
//!
//! The joint-adjustment proof is invariant to hard intervention levels. The source result
//! retains that proof; each required world additionally needs observed cell support from
//! the same frozen score table. No new numerical fit or identification is performed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, GraphFamily, InferenceSelection, Intervention, InterventionContrast,
    InterventionWorld, ModelAssumptions, ObservationSpec, OutcomeFunctional, ResponseFunctional,
    ResponseQuery, ScientificTarget, ScientificWorldId, TargetForm, TargetPopulation, Value,
    VariableId, WorldBinding, WorldCollection,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{ScoreTable, effect_estimate::CompiledFamilyContrast};
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::{
    CompiledIdentificationProduct, CompiledProductParts, IdentificationClaimClass,
    IdentificationOutcome, RequiredFactor,
};

use super::batch::CellFamilyContrast;
use super::lowering::{response_world, response_world_binding};
use crate::{GraphClass, StudyResult};

/// Compile the scientific contract of an already executed, joint-identified cell family.
/// Numerical reduction is permitted only after this returns successfully.
pub(super) fn bind(
    data: &TabularData,
    result: &StudyResult,
    query: &ResponseQuery,
    table: &ScoreTable,
    kind: CellFamilyContrast,
) -> Result<Arc<CompiledFamilyContrast>, String> {
    if result.identification.query != CausalQuery::Response(query.clone())
        || query.temporal.is_some()
        || query.observation != ObservationSpec::Complete
        || query.target_population != TargetPopulation::AllObserved
        || !matches!(
            query.outcome_functional,
            OutcomeFunctional::Mean | OutcomeFunctional::Exceedance(_)
        )
    {
        return Err("family contrast does not match a complete static identified response".into());
    }
    let cells = contrast_cells(query, table, kind)?;
    require_observed_support(query, table, &cells.world_arms)?;
    let target = scientific_target(data, result, query, kind, &cells)?;
    let identified_outcome = IdentificationOutcome::Identified;
    let claim = IdentificationClaimClass::IdentifiedClaim;
    let mut parts = CompiledProductParts::new(identified_outcome, claim);
    parts.world_obligations = cells
        .world_arms
        .iter()
        .map(|&world| {
            IdentificationObligation::for_world(
                ScientificWorldId::from_raw(world),
                identified_outcome,
                claim,
                vec![],
                None,
            )
        })
        .collect();
    parts.assembled =
        Some(IdentificationObligation::assembled(identified_outcome, claim, vec![], None));
    let mut variables = cells.treatments;
    variables.push(cells.outcome);
    variables.extend(table.adjustment_set.iter().copied());
    parts.required_factors.push(RequiredFactor::new("joint_cell_adjustment_scores", variables));
    let product =
        CompiledIdentificationProduct::retain_for_target(&result.identification, parts, &target)
            .map_err(|error| error.to_string())?;
    CompiledFamilyContrast::new(target, product).map(Arc::new).map_err(|error| error.to_string())
}

struct ContrastCells {
    outcome: VariableId,
    treatments: Vec<VariableId>,
    arm: u32,
    world_arms: Vec<u32>,
}

fn contrast_cells(
    query: &ResponseQuery,
    table: &ScoreTable,
    kind: CellFamilyContrast,
) -> Result<ContrastCells, String> {
    let ResponseFunctional::InterventionResponse { outcome, interventions } = &query.functional
    else {
        return Err("family contrast requires a joint intervention response".into());
    };
    // Joint score engines retain the complete bit-coordinate order, including
    // the primary treatment. Prepending it would duplicate coordinate zero.
    let treatments =
        if table.intervened.is_empty() { vec![table.treatment] } else { table.intervened.to_vec() };
    if treatments.first() != Some(&table.treatment)
        || treatments
            .iter()
            .enumerate()
            .any(|(index, variable)| treatments[..index].contains(variable))
    {
        return Err("family contrast has invalid frozen treatment coordinates".into());
    }
    let mut requested = Vec::new();
    for intervention in interventions.iter() {
        let Intervention::Set { variable, value } = intervention else {
            return Err("family contrast requires hard binary intervention levels".into());
        };
        let Some(level) =
            value.as_f64().filter(|level| *level == 0.0 || level.to_bits() == 1.0_f64.to_bits())
        else {
            return Err("family contrast requires hard binary intervention levels".into());
        };
        requested.push((*variable, level));
    }
    if requested.len() != treatments.len()
        || treatments
            .iter()
            .any(|variable| requested.iter().filter(|(id, _)| id == variable).count() != 1)
        || !(2..=antecedent_estimate::cell_aipw::MAX_JOINT_BINARY).contains(&treatments.len())
    {
        return Err("family contrast treatment coordinates differ from the frozen scores".into());
    }
    let mut arm = 0_u32;
    for (index, variable) in treatments.iter().enumerate() {
        if requested
            .iter()
            .any(|(id, level)| id == variable && level.to_bits() == 1.0_f64.to_bits())
        {
            arm |= 1_u32 << index;
        }
    }
    let world_arms = match kind {
        CellFamilyContrast::CellMinusControl if arm != 0 => vec![0, arm],
        CellFamilyContrast::CellMinusControl => {
            return Err("control minus itself is not a distinct-world contrast".into());
        }
        CellFamilyContrast::Interaction => (0..1_u32 << treatments.len()).collect(),
    };
    Ok(ContrastCells { outcome: *outcome, treatments, arm, world_arms })
}

fn require_observed_support(
    query: &ResponseQuery,
    table: &ScoreTable,
    world_arms: &[u32],
) -> Result<(), String> {
    let threshold = match &query.outcome_functional {
        OutcomeFunctional::Mean => None,
        OutcomeFunctional::Exceedance(threshold) => Some(threshold.to_f64()),
        _ => return Err("family contrast requires one scalar outcome functional".into()),
    };
    let support = table.support(None);
    if !support.overlap.overlap_ok {
        return Err("family contrast withheld: frozen score table lacks joint-cell overlap".into());
    }
    for &world in world_arms {
        let positions: Vec<_> = table
            .columns
            .iter()
            .enumerate()
            .filter_map(|(index, column)| {
                (column.arm == world && column.threshold == threshold).then_some(index)
            })
            .collect();
        if positions.len() != 1
            || !support.threshold_supported.get(positions[0]).copied().unwrap_or(false)
        {
            return Err(format!(
                "family contrast withheld: world {world} lacks observed outcome/cell support"
            ));
        }
    }
    Ok(())
}

fn scientific_target(
    data: &TabularData,
    result: &StudyResult,
    query: &ResponseQuery,
    kind: CellFamilyContrast,
    cells: &ContrastCells,
) -> Result<ScientificTarget, String> {
    let graph = match result.certificate.as_ref().map(|certificate| certificate.graph_class) {
        Some(GraphClass::Dag) => GraphFamily::Dag,
        Some(GraphClass::Admg) => GraphFamily::Admg,
        _ => {
            return Err(
                "family contrast requires a retained point-graph identification certificate".into(),
            );
        }
    };
    let binding = response_world_binding(data.schema(), query, cells.outcome)
        .map_err(|error| error.to_string())?;
    let worlds = cells
        .world_arms
        .iter()
        .map(|&world| cell_world(world, &cells.treatments, &binding))
        .collect::<Result<Vec<_>, _>>()?;
    let worlds = WorldCollection::new(worlds).map_err(|error| error.to_string())?;
    let id = ScientificWorldId::from_raw;
    let contrast = match kind {
        CellFamilyContrast::CellMinusControl => {
            InterventionContrast::difference(worlds, id(cells.arm), id(0), id(0))
        }
        CellFamilyContrast::Interaction if cells.treatments.len() == 2 => {
            InterventionContrast::interaction(worlds, id(3), id(1), id(2), id(0))
        }
        // The highest-order interaction of `k` factors: cell `a` carries `(-1)^(k - |a|)`,
        // the same coefficients the estimator reduces the score table with.
        CellFamilyContrast::Interaction => {
            let factors = u32::try_from(cells.treatments.len()).map_err(|error| error.to_string())?;
            InterventionContrast::linear(
                worlds,
                cells.world_arms.iter().map(|&arm| {
                    let sign = if (factors - arm.count_ones()) % 2 == 0 { 1.0 } else { -1.0 };
                    (id(arm), sign)
                }),
                id(0),
            )
        }
    }
    .map_err(|error| error.to_string())?;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(graph, result.identification.required_assumptions.clone()),
        InferenceSelection::Frequentist,
    ))
}

fn cell_world(
    arm: u32,
    treatments: &[VariableId],
    binding: &WorldBinding,
) -> Result<InterventionWorld, String> {
    let interventions = treatments.iter().enumerate().map(|(index, &variable)| {
        Intervention::set(variable, Value::f64(f64::from((arm >> index) & 1)))
    });
    response_world(arm, interventions, arm == 0, binding).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchStudy, RefuteSuite};
    use antecedent_core::IdentificationStatus;
    use antecedent_graph::Dag;

    #[test]
    fn production_binding_refuses_unidentified_source_and_unsupported_worlds() {
        let mut columns = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0_u32..256 {
            let a = f64::from(i & 1);
            let b = f64::from((i >> 1) & 1);
            for (column, value) in
                columns.iter_mut().zip([a, b, a + b + (f64::from(i) * 0.23).sin()])
            {
                column.push(value);
            }
        }
        let data = TabularData::from_f64_columns(
            ["a", "b", "y"].into_iter().zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap();
        let graph = Dag::from_named_edges(data.schema(), &[("a", "y"), ("b", "y")]).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(2),
            interventions: Arc::from([
                Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
                Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            ]),
        });
        let ctx = antecedent_core::ExecutionContext::for_tests(19);
        let prepared = BatchStudy::new(data.clone(), graph)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .family_contrast(Some(CellFamilyContrast::Interaction))
            .prepare_cells(&[query.clone(), query.clone()], &ctx)
            .unwrap();
        let mut result = prepared.estimate(&data, &ctx).unwrap().remove(0);
        let mut table = result.estimate.as_effect().unwrap().scores.table.clone().unwrap();
        bind(&data, &result, &query, &table, CellFamilyContrast::Interaction).unwrap();
        result.identification.status = IdentificationStatus::NotIdentified;
        assert!(bind(&data, &result, &query, &table, CellFamilyContrast::Interaction).is_err());
        result.identification.status = IdentificationStatus::NonparametricallyIdentified;
        // A numerical column is not evidence that its world was observed.
        table.observed_arm = table
            .observed_arm
            .iter()
            .map(|&arm| if arm == 3 { 0 } else { arm })
            .collect::<Vec<_>>()
            .into();
        assert!(bind(&data, &result, &query, &table, CellFamilyContrast::Interaction).is_err());
    }
}
