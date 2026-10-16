//! Measured posterior interval of a conditional probability computed from tables fitted
//! to counts.
//!
//! A declared model's posterior probability is exact and has no sampling distribution.
//! A model whose conditionals were estimated from a sample does. This route retains the
//! per-parent-configuration counts and the row count `n`, places an independent
//! Dirichlet posterior over every fitted conditional, and reports the equal-tailed
//! posterior quantiles of `P(target = level | evidence)` by Monte Carlo over joint table
//! draws and exact enumeration. A consumer replays the draws from the retained counts,
//! prior, seed and level and recomputes the interval bit for bit; the reported scalar
//! must resolve to a currently attesting coverage record. Declared nodes (a measured
//! signal with known error rates) are fixed, not redrawn.
//!
//! Any draw count from the supported minimum ([`MEASURED_DRAWS`]) up to the bound is
//! accepted, and the actual draws and seed are part of the replay identity. The
//! endpoints are quantiles of a finite Monte Carlo sample, so the route reports their
//! Monte Carlo standard errors and refuses an interval whose larger error exceeds
//! [`MAX_ENDPOINT_MCSE_FRACTION`] of its width: the draws have not converged enough
//! for the endpoints to mean what they say.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use antecedent_core::IdentityDomain;
use antecedent_estimate::count_fitted_posterior::{
    CountFittedModel, CountNode, FUNCTIONAL_ID, INTERVAL_METHOD, MAX_DRAWS, NodeTable,
    PosteriorInterval, PosteriorQuery, calibration_basis,
};
use serde::{Deserialize, Serialize};

/// Route name the measured envelope is consumed under.
pub const ROUTE: &str = "count_fitted_interval";
/// The only prior concentration that was measured.
pub const MEASURED_PRIOR_ALPHA: f64 = 1.0;
/// The supported minimum (and default) Monte Carlo draw count. Larger counts are
/// accepted; each measured draw class
/// ([`MEASURED_DRAW_CLASSES`](antecedent_estimate::count_fitted_posterior::MEASURED_DRAW_CLASSES))
/// has its own coverage record.
pub const MEASURED_DRAWS: u32 = 2000;
/// Largest accepted ratio of an endpoint's Monte Carlo standard error to the interval
/// width. The measured design sits near 0.015 at the minimum draw count; five times
/// that or more is a sign the draws, not the posterior, set the endpoints.
pub const MAX_ENDPOINT_MCSE_FRACTION: f64 = 0.05;
const MAX_NODES: usize = 16;
const MAX_LEVELS: usize = 16;

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("{detail}: {message}"),
    }
}

fn protocol(message: &str) -> IoError {
    refused("measured_interval.protocol_not_measured", message)
}

/// One node of the retained model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountFittedNodeWire {
    /// Variable name.
    pub name: String,
    /// Level names, in table column order.
    pub levels: Vec<String>,
    /// Parent node indices (each earlier than this node).
    pub parents: Vec<u32>,
    /// Counts per parent configuration (first parent slowest), for a fitted node.
    pub counts: Option<Vec<u64>>,
    /// Declared conditional probabilities as IEEE bit patterns, for a declared node.
    pub declared_bits: Option<Vec<u64>>,
}

/// The retained source of a count-fitted interval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountFittedIntervalArtifactWire {
    /// Format version.
    pub version: u32,
    /// Nodes in topological order.
    pub nodes: Vec<CountFittedNodeWire>,
    /// Rows the counts were fitted from.
    pub rows: u64,
    /// Target variable name.
    pub target: String,
    /// Target level name.
    pub target_level: String,
    /// Observed `(variable, level)` evidence.
    pub evidence: Vec<(String, String)>,
    /// Nominal credible mass bits.
    pub level_bits: u64,
    /// Dirichlet prior concentration bits.
    pub prior_alpha_bits: u64,
    /// Monte Carlo draws.
    pub draws: u32,
    /// Monte Carlo seed.
    pub seed: u64,
}

struct Resolved {
    model: CountFittedModel,
    query: PosteriorQuery,
    level: f64,
    prior_alpha: f64,
}

impl CountFittedIntervalArtifactWire {
    fn resolve(&self) -> Result<Resolved, IoError> {
        if self.version != 1 || self.nodes.is_empty() || self.nodes.len() > MAX_NODES {
            return Err(refused(
                "measured_inference.artifact_mismatch",
                "count-fitted source is not a version 1 model within bounds",
            ));
        }
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if node.levels.len() > MAX_LEVELS {
                return Err(protocol("a node exceeds the measured level bound"));
            }
            let table = match (&node.counts, &node.declared_bits) {
                (Some(counts), None) => NodeTable::Counts(counts.clone()),
                (None, Some(bits)) => {
                    NodeTable::Declared(bits.iter().map(|b| f64::from_bits(*b)).collect())
                }
                _ => {
                    return Err(refused(
                        "measured_inference.artifact_mismatch",
                        "a node carries both or neither of counts and a declared table",
                    ));
                }
            };
            nodes.push(CountNode {
                cardinality: node.levels.len(),
                parents: node.parents.iter().map(|p| *p as usize).collect(),
                table,
            });
        }
        // Complete rows: every fitted node was counted from exactly `rows` observations.
        for node in &nodes {
            if let NodeTable::Counts(counts) = &node.table {
                if counts.iter().sum::<u64>() != self.rows {
                    return Err(protocol(
                        "every fitted node must be counted from the same complete rows",
                    ));
                }
            }
        }
        let model = CountFittedModel::new(nodes, self.rows)
            .map_err(|e| refused("measured_inference.invalid_report", &e.to_string()))?;
        let find = |name: &str, level: &str| -> Result<(usize, usize), IoError> {
            let node = self
                .nodes
                .iter()
                .position(|n| n.name == name)
                .ok_or_else(|| refused("observation_query.unknown_variable", name))?;
            let l = self.nodes[node]
                .levels
                .iter()
                .position(|l| l == level)
                .ok_or_else(|| refused("observation_query.unknown_level", level))?;
            Ok((node, l))
        };
        let (target, level_index) = find(&self.target, &self.target_level)?;
        let evidence =
            self.evidence.iter().map(|(n, l)| find(n, l)).collect::<Result<Vec<_>, IoError>>()?;
        Ok(Resolved {
            model,
            query: PosteriorQuery { target, level: level_index, evidence },
            level: f64::from_bits(self.level_bits),
            prior_alpha: f64::from_bits(self.prior_alpha_bits),
        })
    }

    /// The conditional probability under the posterior-mean tables (counts plus the prior).
    ///
    /// # Errors
    /// A malformed source, an unknown variable or level, or evidence of probability zero.
    pub fn point(&self) -> Result<f64, IoError> {
        let resolved = self.resolve()?;
        resolved
            .model
            .point(&resolved.query, resolved.prior_alpha)
            .map_err(|e| refused("measured_inference.invalid_report", &e.to_string()))
    }

    /// Serialize a retained count-fitted source.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn encode(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// The draw count lies in the supported range, from the minimum to the bound.
    ///
    /// # Errors
    /// A count below [`MEASURED_DRAWS`] or above the estimator's draw bound.
    pub fn check_draws(&self) -> Result<(), IoError> {
        if self.draws < MEASURED_DRAWS || self.draws as usize > MAX_DRAWS {
            return Err(protocol(&format!(
                "posterior draws must lie in {MEASURED_DRAWS}..={MAX_DRAWS}, got {}",
                self.draws
            )));
        }
        Ok(())
    }

    /// Replay the posterior draws from the retained counts, check the measured protocol
    /// and the Monte Carlo error of the endpoints against `mcse_tolerance`.
    fn replay(&self, mcse_tolerance: f64) -> Result<PosteriorInterval, IoError> {
        let resolved = self.resolve()?;
        if resolved.prior_alpha.to_bits() != MEASURED_PRIOR_ALPHA.to_bits() {
            return Err(protocol("only the uniform Dirichlet prior was measured"));
        }
        self.check_draws()?;
        let interval = resolved
            .model
            .posterior_interval(
                &resolved.query,
                resolved.level,
                resolved.prior_alpha,
                self.draws,
                self.seed,
            )
            .map_err(|e| refused("measured_inference.invalid_report", &e.to_string()))?;
        let fraction = interval.mcse_width_fraction();
        if fraction.is_nan() || fraction > mcse_tolerance {
            return Err(refused(
                "measured_interval.draws_not_converged",
                &format!(
                    "the Monte Carlo standard error of an interval endpoint ({:.3e} lower, \
                     {:.3e} upper) is {fraction:.3} of the interval width, above the \
                     {mcse_tolerance} tolerance; use more posterior draws",
                    interval.lower_mcse, interval.upper_mcse
                ),
            ));
        }
        Ok(interval)
    }

    /// Replay the posterior draws from the retained counts and bind the interval to the
    /// current calibration evidence.
    ///
    /// # Errors
    /// A non-canonical or malformed source, a construction outside the measured
    /// protocol (prior, a draw count below the supported minimum or above the bound,
    /// complete rows), endpoints whose Monte Carlo error exceeds
    /// [`MAX_ENDPOINT_MCSE_FRACTION`] of the width, evidence of probability zero, or a
    /// construction with no currently attesting record.
    pub fn measure(bytes: &[u8]) -> Result<MeasuredInference, IoError> {
        let wire: Self = crate::from_cbor(bytes)?;
        if crate::to_cbor(&wire)? != bytes {
            return Err(refused(
                "measured_inference.artifact_mismatch",
                "count-fitted source is not canonical",
            ));
        }
        let interval = wire.replay(MAX_ENDPOINT_MCSE_FRACTION)?;
        let premises = crate::digest_hex(
            IdentityDomain::Claim,
            &(
                "count_fitted_premises_v1",
                wire.nodes
                    .iter()
                    .map(|n| (&n.name, &n.levels, &n.parents, &n.declared_bits))
                    .collect::<Vec<_>>(),
                &wire.target,
                &wire.target_level,
                &wire.evidence,
                wire.level_bits,
                wire.prior_alpha_bits,
                wire.draws,
                wire.seed,
            ),
        )?;
        let data = crate::digest_hex(
            IdentityDomain::Claim,
            &(wire.rows, wire.nodes.iter().map(|n| &n.counts).collect::<Vec<_>>()),
        )?;
        let scope = serde_json::json!({
            "validation_design": "confounded_common_cause_finite_tables_counts_fitted_with_and_without_declared_signal",
            "evidence_interpretation": "finite validation grid for count-fitted binary tables of a confounded three-variable model with an optional declared measured signal; sparse parent configurations and rare conditioning events are named boundaries",
            "declared_model_assumptions": [
                "independent complete rows",
                "model structure (parents) correct",
                "declared nodes (known signal error rates) exact"
            ],
            "interval_method": INTERVAL_METHOD,
            "posterior": "dirichlet_uniform_prior_per_parent_configuration",
            "rows": wire.rows,
            "draws": wire.draws,
            "seed": wire.seed,
            "endpoint_mcse": {
                "lower": interval.lower_mcse,
                "upper": interval.upper_mcse,
                "width_fraction": interval.mcse_width_fraction(),
                "tolerance": MAX_ENDPOINT_MCSE_FRACTION,
            },
            "level": interval.level,
            "target": wire.target,
            "target_level": wire.target_level,
            "evidence": wire.evidence,
            "reported_scalar": FUNCTIONAL_ID,
        });
        authorize(
            "count_fitted_interval",
            bytes.to_vec(),
            &premises,
            &data,
            scope,
            vec![ScalarExecution {
                name: FUNCTIONAL_ID.into(),
                point: interval.point,
                interval: (interval.lower, interval.upper),
                basis: calibration_basis(interval.rows, interval.level, interval.draws),
            }],
        )
    }

    /// Consume a measured envelope by replaying its source and comparing the report.
    ///
    /// # Errors
    /// Any source, expected identity, protocol, evidence or scalar mismatch.
    pub fn consume_measured(
        bytes: &[u8],
        expected: &MeasuredExpectation,
    ) -> Result<MeasuredInference, IoError> {
        let wire = decode(bytes, expected)?;
        let measured = Self::measure(&wire.source)?;
        compare_replay(&wire, &measured)?;
        Ok(measured)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels() -> Vec<String> {
        vec!["0".into(), "1".into()]
    }

    /// `Z -> Y`, 100 rows: Z = 1 in 40 rows, and Y = 1 in 30 of those; Y = 1 in 12 of the
    /// 60 rows with Z = 0.
    fn wire(draws: u32) -> CountFittedIntervalArtifactWire {
        CountFittedIntervalArtifactWire {
            version: 1,
            nodes: vec![
                CountFittedNodeWire {
                    name: "Z".into(),
                    levels: levels(),
                    parents: vec![],
                    counts: Some(vec![60, 40]),
                    declared_bits: None,
                },
                CountFittedNodeWire {
                    name: "Y".into(),
                    levels: levels(),
                    parents: vec![0],
                    counts: Some(vec![48, 12, 10, 30]),
                    declared_bits: None,
                },
            ],
            rows: 100,
            target: "Y".into(),
            target_level: "1".into(),
            evidence: vec![("Z".into(), "1".into())],
            level_bits: 0.95_f64.to_bits(),
            prior_alpha_bits: MEASURED_PRIOR_ALPHA.to_bits(),
            draws,
            seed: 7,
        }
    }

    fn detail(error: &IoError) -> String {
        let IoError::Refused { message, .. } = error else { panic!("not a refusal: {error:?}") };
        message.clone()
    }

    #[test]
    fn every_draw_count_from_the_supported_minimum_replays() {
        let mut fractions = Vec::new();
        for draws in [2000, 3000, 4000, 8000] {
            let interval = wire(draws).replay(MAX_ENDPOINT_MCSE_FRACTION).expect("replays");
            assert_eq!(interval.draws, draws);
            assert!(interval.lower < interval.point && interval.point < interval.upper);
            assert!(interval.lower_mcse > 0.0 && interval.upper_mcse > 0.0);
            fractions.push(interval.mcse_width_fraction());
        }
        // Four times the draws halves the error; the order-statistic estimate is noisy
        // enough that only the ends of the range are compared.
        assert!(fractions[3] < fractions[0], "{fractions:?}");
        // The retained draws and seed are the replay: the same source reproduces the
        // interval bit for bit, and a different draw count or seed is another source.
        let (a, b) = (wire(3000), wire(3000));
        let (first, second) = (a.replay(1.0).unwrap(), b.replay(1.0).unwrap());
        assert_eq!(first.lower.to_bits(), second.lower.to_bits());
        assert_eq!(first.upper.to_bits(), second.upper.to_bits());
        assert_ne!(a.encode().unwrap(), wire(4000).encode().unwrap());
        let reseeded = CountFittedIntervalArtifactWire { seed: 8, ..wire(3000) };
        assert_ne!(a.encode().unwrap(), reseeded.encode().unwrap());
    }

    #[test]
    fn draws_below_the_supported_minimum_or_above_the_bound_are_refused() {
        for draws in [0, 2, 1000, MEASURED_DRAWS - 1, u32::try_from(MAX_DRAWS).unwrap() + 1] {
            let error = wire(draws).replay(MAX_ENDPOINT_MCSE_FRACTION).unwrap_err();
            let message = detail(&error);
            assert!(message.contains("protocol_not_measured"), "{draws}: {message}");
            assert!(message.contains(&format!("got {draws}")), "{message}");
        }
        assert!(wire(u32::try_from(MAX_DRAWS).unwrap()).replay(1.0).is_ok());
    }

    #[test]
    fn a_non_measured_prior_is_still_refused() {
        let other =
            CountFittedIntervalArtifactWire { prior_alpha_bits: 0.5_f64.to_bits(), ..wire(2000) };
        let message = detail(&other.replay(1.0).unwrap_err());
        assert!(message.contains("protocol_not_measured"), "{message}");
    }

    #[test]
    fn endpoints_whose_monte_carlo_error_exceeds_the_tolerance_are_refused() {
        let source = wire(2000);
        let fraction = source.replay(1.0).unwrap().mcse_width_fraction();
        assert!(fraction > 0.0 && fraction < MAX_ENDPOINT_MCSE_FRACTION, "{fraction}");
        // The same draws pass at a tolerance just above their error and are refused
        // just below it, naming the error, the fraction and the tolerance.
        assert!(source.replay(fraction * 1.001).is_ok());
        let message = detail(&source.replay(fraction * 0.999).unwrap_err());
        assert!(message.contains("measured_interval.draws_not_converged"), "{message}");
        assert!(message.contains("more posterior draws"), "{message}");
    }
}
