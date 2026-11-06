//! Classification of decisions over an assumption grid.
// SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// A change of leader between grid points.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSwitch {
    /// Leading actions before the switch.
    pub from: Vec<String>,
    /// Leading actions after the switch.
    pub to: Vec<String>,
    /// Lower end of the coordinate bracket containing the switch.
    pub lower: f64,
    /// Upper end of the coordinate bracket containing the switch.
    pub upper: f64,
    /// Whether the switch is an exact tie at a grid coordinate (`lower` equals `upper`).
    pub exact: bool,
    /// Linear interpolation of the crossing between the two bracketing grid points. It
    /// is exact only if the utility difference is linear in the coordinate between them;
    /// present only for a point surface.
    pub interpolated: Option<f64>,
}

/// Why no action is robust.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoRobustReason {
    /// At these coordinates the best action depends on where in the assumption range
    /// the surface lies.
    MixedWithinRange {
        /// The coordinates.
        coordinates: Vec<f64>,
    },
    /// The leader changes more than once along the coordinate.
    MultipleSwitches,
    /// Actions tie at these coordinates without a change of leader.
    TieWithoutSwitch {
        /// The coordinates.
        coordinates: Vec<f64>,
    },
}

/// What the decision over the assumption range can claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityOutcome {
    /// One action is uniquely best at every grid point under every range vertex.
    InvariantAction {
        /// The action.
        action: String,
    },
    /// The unique leader changes exactly once along the coordinate.
    AssumptionDependent {
        /// The switch and its tipping coordinate.
        switch: DecisionSwitch,
    },
    /// No action is robust over the range.
    NoRobustAction {
        /// Why.
        reason: NoRobustReason,
        /// Switches found, when the leader moves along the coordinate.
        switches: Vec<DecisionSwitch>,
    },
    /// A grid point was not evaluated; invariance is not checked there.
    Unresolved {
        /// The first unevaluated coordinate.
        coordinate: f64,
    },
}

/// What is known at one grid point for classification.
#[derive(Clone, Debug, PartialEq)]
pub enum PointInput {
    /// The point was not evaluated.
    Unevaluated,
    /// The leading actions under each range-vertex scenario at this point.
    Scenarios {
        /// Leading actions (ids) of each scenario; one id is a unique leader.
        leaders: Vec<Vec<String>>,
        /// Utility of each action; present when the point has a single scenario.
        values: Option<Vec<(String, f64)>>,
    },
}

/// One grid point handed to [`classify_decision_points`].
#[derive(Clone, Debug, PartialEq)]
pub struct PointLeaders {
    /// Coordinate value.
    pub coordinate: f64,
    /// Evidence at the point.
    pub input: PointInput,
}

#[derive(Clone, Debug, PartialEq)]
enum PointState {
    Unique(String),
    Tie(Vec<String>),
    Mixed,
}

struct Run {
    state: PointState,
    first: usize,
    last: usize,
}

fn state_of(leaders: &[Vec<String>]) -> PointState {
    let normalized: Vec<Vec<String>> = leaders
        .iter()
        .map(|ids| {
            let mut sorted = ids.clone();
            sorted.sort();
            sorted
        })
        .collect();
    let Some(first) = normalized.first() else {
        return PointState::Mixed;
    };
    if normalized.iter().any(|ids| ids != first) {
        return PointState::Mixed;
    }
    match first.len() {
        0 => PointState::Mixed,
        1 => PointState::Unique(first[0].clone()),
        _ => PointState::Tie(first.clone()),
    }
}

fn runs_of(states: Vec<PointState>) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (index, state) in states.into_iter().enumerate() {
        if runs.last().is_some_and(|run| run.state == state) {
            if let Some(run) = runs.last_mut() {
                run.last = index;
            }
        } else {
            runs.push(Run { state, first: index, last: index });
        }
    }
    runs
}

fn leaders_of_state(state: &PointState) -> Vec<String> {
    match state {
        PointState::Unique(id) => vec![id.clone()],
        PointState::Tie(ids) => ids.clone(),
        PointState::Mixed => Vec::new(),
    }
}

fn interpolate(points: &[PointLeaders], i: usize, j: usize, a: &str, b: &str) -> Option<f64> {
    let value = |point: &PointLeaders, id: &str| -> Option<f64> {
        match &point.input {
            PointInput::Scenarios { values: Some(values), .. } => {
                values.iter().find(|(name, _)| name == id).map(|(_, x)| *x)
            }
            _ => None,
        }
    };
    let (left, right) = (&points[i], &points[j]);
    let before = value(left, a)? - value(left, b)?;
    let after = value(right, a)? - value(right, b)?;
    if before <= 0.0 || after >= 0.0 {
        return None;
    }
    Some(left.coordinate + before / (before - after) * (right.coordinate - left.coordinate))
}

fn switch_between(current: &Run, next: &Run, points: &[PointLeaders]) -> DecisionSwitch {
    let at = |index: usize| points[index].coordinate;
    let (lower, upper, exact) = match (&current.state, &next.state) {
        (_, PointState::Tie(_)) => (at(next.first), at(next.last), next.first == next.last),
        (PointState::Tie(_), _) => {
            (at(current.first), at(current.last), current.first == current.last)
        }
        _ => (at(current.last), at(next.first), false),
    };
    let interpolated = match (&current.state, &next.state) {
        (PointState::Unique(a), PointState::Unique(b)) => {
            interpolate(points, current.last, next.first, a, b)
        }
        _ => None,
    };
    DecisionSwitch {
        from: leaders_of_state(&current.state),
        to: leaders_of_state(&next.state),
        lower,
        upper,
        exact,
        interpolated,
    }
}

/// Switches along the coordinate and the coordinates where a tie is touched and left
/// again without a change of leader.
fn switches_of(runs: &[Run], points: &[PointLeaders]) -> (Vec<DecisionSwitch>, Vec<f64>) {
    let mut switches = Vec::new();
    let mut touched = Vec::new();
    let mut index = 0;
    while index + 1 < runs.len() {
        if let (PointState::Unique(before), PointState::Tie(_), Some(after)) =
            (&runs[index].state, &runs[index + 1].state, runs.get(index + 2))
        {
            if let PointState::Unique(beyond) = &after.state {
                let tie = &runs[index + 1];
                if before == beyond {
                    touched.extend((tie.first..=tie.last).map(|k| points[k].coordinate));
                } else {
                    switches.push(DecisionSwitch {
                        from: vec![before.clone()],
                        to: vec![beyond.clone()],
                        lower: points[tie.first].coordinate,
                        upper: points[tie.last].coordinate,
                        exact: tie.first == tie.last,
                        interpolated: None,
                    });
                }
                index += 2;
                continue;
            }
        }
        switches.push(switch_between(&runs[index], &runs[index + 1], points));
        index += 1;
    }
    (switches, touched)
}

/// Classify the decision over an ordered grid of coordinates.
///
/// `points` must be in ascending coordinate order. A point is `Unique` when every range
/// scenario has the same single leader, `Tie` when every scenario has the same set of
/// tied leaders and `Mixed` otherwise. All points `Unique` with one action is an
/// invariant action; no `Mixed` point and exactly one change of leader (an exact tie
/// bridging two leaders counts as the change) is an assumption-dependent switch with its
/// tipping coordinate; everything else has no robust action.
#[must_use]
pub fn classify_decision_points(points: &[PointLeaders]) -> SensitivityOutcome {
    if let Some(point) = points.iter().find(|p| matches!(p.input, PointInput::Unevaluated)) {
        return SensitivityOutcome::Unresolved { coordinate: point.coordinate };
    }
    let states: Vec<PointState> = points
        .iter()
        .map(|point| match &point.input {
            PointInput::Scenarios { leaders, .. } => state_of(leaders),
            PointInput::Unevaluated => PointState::Mixed,
        })
        .collect();
    if let Some(PointState::Unique(first)) = states.first() {
        if states.iter().all(|state| matches!(state, PointState::Unique(id) if id == first)) {
            return SensitivityOutcome::InvariantAction { action: first.clone() };
        }
    }
    let coordinates_where = |wanted: fn(&PointState) -> bool| -> Vec<f64> {
        points.iter().zip(&states).filter(|(_, s)| wanted(s)).map(|(p, _)| p.coordinate).collect()
    };
    let mixed = coordinates_where(|s| matches!(s, PointState::Mixed));
    if !mixed.is_empty() {
        return SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MixedWithinRange { coordinates: mixed },
            switches: Vec::new(),
        };
    }
    let tied = coordinates_where(|s| matches!(s, PointState::Tie(_)));
    let runs = runs_of(states);
    let (mut switches, touched) = switches_of(&runs, points);
    if !touched.is_empty() {
        let reason = NoRobustReason::TieWithoutSwitch { coordinates: touched };
        return SensitivityOutcome::NoRobustAction { reason, switches };
    }
    match switches.len() {
        0 => SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::TieWithoutSwitch { coordinates: tied },
            switches,
        },
        1 => SensitivityOutcome::AssumptionDependent { switch: switches.remove(0) },
        _ => SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MultipleSwitches,
            switches,
        },
    }
}
