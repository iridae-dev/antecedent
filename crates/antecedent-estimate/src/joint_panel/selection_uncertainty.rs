//! Change-point, smoothing and selection uncertainty carried into a reference decomposition.
//!
//! A temporal law draws a specification (no change, a change point, or a smooth path with a
//! penalty) together with its parameter path. A decomposition computed per draw therefore
//! already mixes over specifications. This module splits that mixture by the specification
//! each draw selected, so the part of the spread in every component that comes from
//! choosing the change point or the smoothing is reported next to the part that remains
//! inside a fixed specification. It is descriptive: neither share is a coverage claim.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::decomposition::PanelDecomposition;
use super::time_varying::{TemporalModelChoice, TemporalPanelDraw};
use super::{PanelError, invalid};
use serde::{Deserialize, Serialize};

/// Decomposition moments conditional on one selected specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpecificationShare {
    /// The selected specification, including its change period or smoothing penalty.
    pub choice: TemporalModelChoice,
    /// Complete law slots that selected it.
    pub slots: usize,
    /// Normalized law mass of those slots.
    pub mass: f64,
    /// Mean Shapley contribution given this specification.
    pub shapley_mean: [f64; 4],
    /// Plug-in variance of each Shapley contribution given this specification.
    pub shapley_variance: [f64; 4],
}

/// Law-of-total-variance split of each contribution by selected specification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionDecomposition {
    /// Requested slots, including failed ones that stay failed.
    pub requested: usize,
    /// Complete slots with positive mass.
    pub complete: usize,
    /// Specification of the point fit; it need not be the most probable one.
    pub point_choice: TemporalModelChoice,
    /// One entry per distinct specification, in order of first appearance.
    pub specifications: Vec<SpecificationShare>,
    /// Mixture mean of each Shapley contribution.
    pub shapley_mean: [f64; 4],
    /// Mass-weighted variance that remains inside a fixed specification.
    pub within_variance: [f64; 4],
    /// Variance of the specification means: spread owed to the selection itself.
    pub between_variance: [f64; 4],
    /// Share of each contribution's variance owed to selection; zero without any variance.
    pub selection_share: [f64; 4],
    /// Mass on each distinct change period, ascending.
    pub change_point_mass: Vec<(usize, f64)>,
    /// Mass on smooth paths.
    pub smooth_mass: f64,
    /// Mass on the specification with no change.
    pub no_change_mass: f64,
}

/// One complete positive-mass source slot, retaining the law's selected specification.
struct SelectionRow<'a> {
    choice: &'a TemporalModelChoice,
    mass: f64,
    shapley: [f64; 4],
}

/// Members of one specification, in source-slot order and normalized to total law mass.
struct SpecificationGroup<'a> {
    choice: &'a TemporalModelChoice,
    members: Vec<(f64, [f64; 4])>,
}

/// Validate source alignment before grouping or computing any conditional moments.
fn aligned_selection_rows<'a>(
    decomposition: &PanelDecomposition,
    draws: &'a [Option<TemporalPanelDraw>],
) -> Result<(f64, Vec<SelectionRow<'a>>), PanelError> {
    if draws.len() != decomposition.draws.len()
        || decomposition.weights.as_ref().is_some_and(|w| w.len() != draws.len())
    {
        return Err(invalid("selection split requires the exact aligned decomposed law"));
    }
    let mut total_mass = 0.0;
    let mut rows = Vec::new();
    for (index, (decomposed, law)) in decomposition.draws.iter().zip(draws).enumerate() {
        let mass = decomposition.weights.as_ref().map_or(1.0, |w| w[index]);
        if !mass.is_finite() || mass < 0.0 {
            return Err(invalid("selection split requires finite nonnegative law mass"));
        }
        match (decomposed, law) {
            (Some(row), Some(law)) => {
                if mass > 0.0 {
                    total_mass += mass;
                    rows.push(SelectionRow { choice: &law.choice, mass, shapley: row.shapley });
                }
            }
            (None, _) => {}
            (Some(_), None) => {
                return Err(invalid("a decomposed slot has no aligned temporal law"));
            }
        }
    }
    if rows.is_empty() || total_mass <= 0.0 || !total_mass.is_finite() {
        return Err(invalid("selection split requires at least one complete positive-mass slot"));
    }
    Ok((total_mass, rows))
}

/// Split a temporal decomposition by the specification each aligned draw selected.
///
/// `draws` must be the exact aligned law the decomposition was computed from.
///
/// # Errors
/// Misaligned axes, a decomposed slot whose law slot failed, or no complete slot.
pub fn decomposition_selection(
    decomposition: &PanelDecomposition,
    point: &TemporalPanelDraw,
    draws: &[Option<TemporalPanelDraw>],
) -> Result<SelectionDecomposition, PanelError> {
    let (total_mass, rows) = aligned_selection_rows(decomposition, draws)?;
    let mut groups: Vec<SpecificationGroup<'_>> = Vec::new();
    for row in &rows {
        match groups.iter_mut().find(|group| group.choice == row.choice) {
            Some(group) => group.members.push((row.mass / total_mass, row.shapley)),
            None => groups.push(SpecificationGroup {
                choice: row.choice,
                members: vec![(row.mass / total_mass, row.shapley)],
            }),
        }
    }
    let mut mean = [0.0; 4];
    let mut specifications = Vec::with_capacity(groups.len());
    for group in &groups {
        let members = &group.members;
        let group_mass: f64 = members.iter().map(|(m, _)| m).sum();
        let mut group_mean = [0.0; 4];
        for (m, shapley) in members {
            for j in 0..4 {
                group_mean[j] += m / group_mass * shapley[j];
            }
        }
        let mut group_variance = [0.0; 4];
        for (m, shapley) in members {
            for j in 0..4 {
                group_variance[j] += m / group_mass * (shapley[j] - group_mean[j]).powi(2);
            }
        }
        for j in 0..4 {
            mean[j] += group_mass * group_mean[j];
        }
        specifications.push(SpecificationShare {
            choice: group.choice.clone(),
            slots: members.len(),
            mass: group_mass,
            shapley_mean: group_mean,
            shapley_variance: group_variance,
        });
    }
    let mut within = [0.0; 4];
    let mut between = [0.0; 4];
    for share in &specifications {
        for j in 0..4 {
            within[j] += share.mass * share.shapley_variance[j];
            between[j] += share.mass * (share.shapley_mean[j] - mean[j]).powi(2);
        }
    }
    let selection_share = std::array::from_fn(|j| {
        let total = within[j] + between[j];
        if total > 0.0 { between[j] / total } else { 0.0 }
    });
    let mut change_point_mass: Vec<(usize, f64)> = Vec::new();
    let (mut smooth_mass, mut no_change_mass) = (0.0, 0.0);
    for share in &specifications {
        match share.choice {
            TemporalModelChoice::NoChange => no_change_mass += share.mass,
            TemporalModelChoice::Smooth { .. } => smooth_mass += share.mass,
            TemporalModelChoice::Piecewise { at } => {
                match change_point_mass.iter_mut().find(|(period, _)| *period == at) {
                    Some((_, mass)) => *mass += share.mass,
                    None => change_point_mass.push((at, share.mass)),
                }
            }
        }
    }
    change_point_mass.sort_by_key(|(period, _)| *period);
    Ok(SelectionDecomposition {
        requested: draws.len(),
        complete: rows.len(),
        point_choice: point.choice.clone(),
        specifications,
        shapley_mean: mean,
        within_variance: within,
        between_variance: between,
        selection_share,
        change_point_mass,
        smooth_mass,
        no_change_mass,
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::{
        ChangeComponent, DecompositionEndpoint, MechanismPeriodMap, PanelDecompositionRequest,
        decompose_temporal_law,
    };
    use super::*;
    use antecedent_core::ExecutionContext;

    #[test]
    fn selection_share_splits_the_decomposed_law_by_selected_change_point_and_smoothing() {
        let (data, draw, _) = super::super::tests::fixture();
        let programs = super::super::tests::programs(&data);
        let request = PanelDecompositionRequest {
            before: DecompositionEndpoint {
                program: programs[0].clone(),
                population_weights: vec![0.25; 4],
            },
            after: DecompositionEndpoint {
                program: programs[3].clone(),
                population_weights: vec![0.25; 4],
            },
            time_weights: vec![1.0 / 24.0; 24],
            order: [
                ChangeComponent::Exposure,
                ChangeComponent::Saturation,
                ChangeComponent::Population,
                ChangeComponent::Mechanism,
            ],
        };
        let periods = MechanismPeriodMap { before: vec![0; 24], after: (0..24).collect() };
        let temporal = |choice, shift| {
            let parameters = (0..24)
                .map(|time| {
                    let mut value = draw.parameters.clone();
                    value.intercept += shift * time as f64;
                    value
                })
                .collect();
            TemporalPanelDraw { baseline: draw.clone(), parameters, choice }
        };
        let point = temporal(TemporalModelChoice::NoChange, 0.0);
        let laws = [
            Some(temporal(TemporalModelChoice::Piecewise { at: 12 }, 0.1)),
            Some(temporal(TemporalModelChoice::Piecewise { at: 12 }, 0.3)),
            None,
            Some(temporal(TemporalModelChoice::Smooth { penalty: 1.0 }, 0.5)),
            Some(temporal(TemporalModelChoice::Piecewise { at: 8 }, 0.1)),
        ];
        let result = decompose_temporal_law(
            &data,
            &request,
            &periods,
            &point,
            &laws,
            &ExecutionContext::for_tests(85),
        )
        .unwrap();
        let split = decomposition_selection(&result, &point, &laws).unwrap();
        assert_eq!((split.requested, split.complete), (5, 4));
        assert_eq!(split.point_choice, TemporalModelChoice::NoChange);
        assert_eq!(split.specifications.len(), 3);
        assert_eq!(split.change_point_mass, vec![(8, 0.25), (12, 0.5)]);
        assert!((split.smooth_mass - 0.25).abs() < 1e-12);
        assert_eq!(split.no_change_mass, 0.0);
        // Within plus between reproduces the plug-in variance of the whole decomposed law.
        let complete = result.draws.iter().flatten().collect::<Vec<_>>();
        for j in 0..4 {
            let mean = complete.iter().map(|d| d.shapley[j]).sum::<f64>() / 4.0;
            let variance =
                complete.iter().map(|d| (d.shapley[j] - mean).powi(2)).sum::<f64>() / 4.0;
            assert!((split.shapley_mean[j] - mean).abs() < 1e-10);
            assert!(
                (split.within_variance[j] + split.between_variance[j] - variance).abs() < 1e-10
            );
        }
        // Choosing different mechanism paths moves the mechanism contribution between
        // specifications; the unchanged exposure contribution is not owed to selection.
        assert!(split.between_variance[3] > 0.0);
        assert!(split.selection_share[3] > 0.0 && split.selection_share[3] <= 1.0);
        assert!(split.selection_share[2].abs() < 1e-12);
        // A decomposed slot without its aligned law, or a shifted axis, refuses.
        assert!(decomposition_selection(&result, &point, &laws[..4]).is_err());
        let mut broken = laws.clone();
        broken[0] = None;
        assert!(decomposition_selection(&result, &point, &broken).is_err());
    }
}
