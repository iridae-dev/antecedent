//! The declared validation design and the dependence of its cases.
//!
//! A score is a mean over cases; how uncertain a mean or a difference of means
//! is depends on how the cases depend on each other, which the data cannot
//! reveal. The design therefore declares both:
//!
//! * the [`ValidationDesign`]: held-out split, rolling origin, blocked or
//!   clustered, or simulation stream;
//! * the [`DependenceDeclaration`]: iid, serial (with the truncation lag of the
//!   long-run variance) or clustered.
//!
//! An iid declaration is refused for a rolling-origin or blocked design. With no
//! dependence declaration, point scores are still reported but every standard
//! error is withheld: independence is never assumed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;

use super::law::ObservationKey;
use super::refusal::{PredictiveAssessmentError, detail, invalid, refuse};

/// How the held-out cases were formed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationDesign {
    /// One held-out split of units not used to fit the law.
    HeldOutSplit,
    /// Forecast origins moved forward through time; each case is a forecast made
    /// at its origin for `horizon_steps` ahead.
    RollingOrigin {
        /// Forecast origin of each case, in the order of the observations.
        origins: Vec<u64>,
        /// Steps ahead of the origin each forecast targets (at least one).
        horizon_steps: u32,
    },
    /// Blocked or clustered folds: cases within a block are not independent.
    Blocked {
        /// Block label of each case, in the order of the observations.
        blocks: Vec<u32>,
    },
    /// Cases drawn from a simulation stream with a known generator.
    SimulationStream {
        /// Identity of the stream (generator and seed), for the record.
        stream: String,
    },
}

impl ValidationDesign {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::HeldOutSplit => "held_out_split",
            Self::RollingOrigin { .. } => "rolling_origin",
            Self::Blocked { .. } => "blocked",
            Self::SimulationStream { .. } => "simulation_stream",
        }
    }

    /// Whether cases of this design are temporal or clustered, so that iid
    /// cases are not a default.
    #[must_use]
    pub const fn is_dependent_by_construction(&self) -> bool {
        matches!(self, Self::RollingOrigin { .. } | Self::Blocked { .. })
    }
}

/// How the cases of an assessment depend on each other.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DependenceDeclaration {
    /// The case scores are independent. Never a default; refused for rolling
    /// origin and blocked designs.
    Iid,
    /// Serial dependence: scores are ordered by `time_index`, and the long-run
    /// variance of a mean (or a mean difference) uses a Bartlett kernel with the
    /// declared truncation lag (Newey and West 1987; Diebold and Mariano 1995).
    Serial {
        /// Time index of each case, in the order of the observations.
        time_index: Vec<u64>,
        /// Truncation lag of the kernel. For an `h`-step forecast declare at
        /// least `h - 1`.
        truncation_lag: u32,
    },
    /// Cluster dependence: scores within a cluster may be arbitrarily dependent,
    /// clusters are independent.
    Cluster {
        /// Cluster label of each case, in the order of the observations.
        labels: Vec<u32>,
    },
}

impl DependenceDeclaration {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Iid => "iid",
            Self::Serial { .. } => "serial",
            Self::Cluster { .. } => "cluster",
        }
    }
}

/// Stable name of an optional dependence declaration.
#[must_use]
pub const fn dependence_label(dependence: Option<&DependenceDeclaration>) -> &'static str {
    match dependence {
        Some(declaration) => declaration.name(),
        None => "undeclared",
    }
}

/// The design resolved into canonical case order and groupings.
#[derive(Clone, Debug)]
pub(crate) struct ResolvedDesign {
    /// Case indices (into the supplied order) in canonical order: sorted by the
    /// design's time or origin, then by observation key, so that every sum is
    /// formed in an order independent of how the caller listed the cases.
    pub order: Vec<usize>,
    /// Fold label of each case in canonical order (`None` for a design without folds).
    pub fold_of: Vec<Option<u64>>,
    /// Serial time index per case in canonical order, when declared.
    pub time: Option<Vec<u64>>,
    /// Cluster label per case in canonical order, when declared.
    pub cluster: Option<Vec<u32>>,
}

fn length_check(
    what: &str,
    supplied: usize,
    cases: usize,
) -> Result<(), PredictiveAssessmentError> {
    if supplied == cases {
        Ok(())
    } else {
        Err(invalid(
            detail::DESIGN_INVALID,
            format!("{what} has {supplied} entries for {cases} cases; one per case is required"),
        ))
    }
}

/// Validate the design and dependence against the observation keys and resolve
/// the canonical order.
pub(crate) fn resolve_design(
    design: &ValidationDesign,
    dependence: Option<&DependenceDeclaration>,
    keys: &[ObservationKey],
) -> Result<ResolvedDesign, PredictiveAssessmentError> {
    let n = keys.len();
    let mut primary: Vec<u64> = vec![0; n];
    let mut fold: Option<Vec<u64>> = None;
    match design {
        ValidationDesign::HeldOutSplit => {}
        ValidationDesign::SimulationStream { stream } => {
            if stream.trim().is_empty() {
                return Err(invalid(
                    detail::DESIGN_INVALID,
                    "a simulation stream design names its generator and seed",
                ));
            }
        }
        ValidationDesign::RollingOrigin { origins, horizon_steps } => {
            length_check("the rolling-origin design", origins.len(), n)?;
            if *horizon_steps == 0 {
                return Err(invalid(
                    detail::DESIGN_INVALID,
                    "a rolling-origin design forecasts at least one step ahead",
                ));
            }
            primary.clone_from(origins);
            fold = Some(origins.clone());
        }
        ValidationDesign::Blocked { blocks } => {
            length_check("the blocked design", blocks.len(), n)?;
            fold = Some(blocks.iter().map(|b| u64::from(*b)).collect());
        }
    }
    let mut time: Option<Vec<u64>> = None;
    let mut cluster: Option<Vec<u32>> = None;
    match dependence {
        None | Some(DependenceDeclaration::Iid) => {}
        Some(DependenceDeclaration::Serial { time_index, .. }) => {
            length_check("the serial dependence declaration", time_index.len(), n)?;
            primary.clone_from(time_index);
            time = Some(time_index.clone());
        }
        Some(DependenceDeclaration::Cluster { labels }) => {
            length_check("the cluster dependence declaration", labels.len(), n)?;
            cluster = Some(labels.clone());
        }
    }
    if design.is_dependent_by_construction()
        && matches!(dependence, Some(DependenceDeclaration::Iid))
    {
        return Err(refuse(
            reason_code!("sampling_dependence_unknown"),
            detail::IID_NOT_LICENSED,
            format!(
                "an iid dependence was declared for a {} design, whose cases are temporal or \
                 clustered; independence is never assumed. Declare serial dependence with its \
                 truncation lag, or cluster dependence with the cluster labels",
                design.name()
            ),
        ));
    }
    if let (
        ValidationDesign::RollingOrigin { horizon_steps, .. },
        Some(DependenceDeclaration::Serial { truncation_lag, .. }),
    ) = (design, dependence)
    {
        if u64::from(*truncation_lag) + 1 < u64::from(*horizon_steps) {
            return Err(refuse(
                reason_code!("sampling_dependence_unknown"),
                detail::SERIAL_LAG_BELOW_HORIZON,
                format!(
                    "forecasts {horizon_steps} steps ahead have errors that overlap across up to \
                     {} origins, but the declared truncation lag is {truncation_lag}; declare a \
                     truncation lag of at least {}",
                    horizon_steps,
                    horizon_steps - 1
                ),
            ));
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| primary[a].cmp(&primary[b]).then(keys[a].cmp(&keys[b])));
    let permute_u64 = |values: &[u64]| order.iter().map(|&i| values[i]).collect::<Vec<u64>>();
    Ok(ResolvedDesign {
        fold_of: match &fold {
            Some(values) => permute_u64(values).into_iter().map(Some).collect(),
            None => vec![None; n],
        },
        time: time.as_deref().map(permute_u64),
        cluster: cluster.map(|labels| order.iter().map(|&i| labels[i]).collect()),
        order,
    })
}
