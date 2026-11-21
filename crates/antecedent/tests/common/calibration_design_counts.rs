//! Extra counts for studies that distinguish returned intervals from refused panels.

use super::{CoverageTally, grid_point, last_map_end, sharding, smoke};

impl CoverageTally {
    /// Publish the actual attempted, returned and covered counts for a scoped panel study.
    pub fn assert_panel_interval(self) {
        self.check_panel_interval(false);
    }

    /// The four short-history average-effect cells must also establish nominal
    /// coverage within the reporting precision at a single 2,000-attempt look.
    pub fn assert_panel_nominal_at_final_look(self) {
        self.check_panel_interval(true);
    }

    fn check_panel_interval(mut self, final_look: bool) {
        let scoped = self.record.as_ref().and_then(|r| r.construction.as_ref()).is_some_and(|c| {
            c.query == "SyntheticControl" && c.functional.starts_with("synthetic_control/")
        });
        assert!(!final_look || scoped || sharding(), "panel has no matching construction");
        if sharding() || !scoped {
            self.assert();
            return;
        }
        // The fixed map owns the attempts. Its end includes all earlier shards and
        // prefix runs, including refusals whose event digests remain in the tally.
        let attempted = last_map_end().expect("panel coverage requires a fixed attempt budget");
        assert_eq!(self.scored, self.with_interval, "invalid returned panel interval");
        self.skipped = attempted.checked_sub(self.scored).expect("more returns than attempts");
        if !smoke() {
            self.check_skips();
        }
        self.skipped = 0;
        if final_look {
            let prediction = self.level;
            if !smoke() && attempted < super::RECHECK_N_SIM {
                eprintln!(
                    "calibration-recheck {}: nominal panel validation requires 2000 attempts",
                    self.name
                );
                if let Some(test) = std::thread::current().name() {
                    eprintln!("calibration-recheck-test {test}");
                }
            } else if !smoke() {
                assert_eq!(attempted, super::RECHECK_N_SIM, "unregistered panel validation look");
                let rate = self.rate();
                // Half of a five-percentage-point reporting step. Test both
                // boundaries of this prediction with the exact binomial tails.
                // A pass needs positive evidence that the held-out rate is inside.
                let margin = 0.025;
                let p_lower = super::decision::binomial_p_value(
                    self.scored,
                    self.scored - self.covered,
                    1.0 - (prediction - margin),
                    true,
                )
                .expect("valid nominal panel level");
                let p_upper = super::decision::binomial_p_value(
                    self.scored,
                    self.covered,
                    prediction + margin,
                    true,
                )
                .expect("valid nominal panel level");
                // Four fixed predictions, one final look, simultaneous 95% assurance.
                let alpha = 0.05 / 4.0;
                let mcse = (rate * (1.0 - rate) / f64::from(self.scored)).sqrt();
                assert!(
                    mcse <= 0.01 && p_lower <= alpha && p_upper <= alpha,
                    "nominal panel coverage {prediction}, validation {}/{}, MCSE {mcse}, equivalence tail probabilities {p_lower}, {p_upper}",
                    self.covered,
                    self.scored
                );
            }
        }
        self.assert();
        self.emit_design_counts(serde_json::json!({
            "interval_counts": {
                "attempted": attempted, "returned": self.with_interval, "covered": self.covered
            }
        }));
    }

    /// Sum the per-replicate refusal counts with the same ordered extension and shard
    /// machinery as interval lengths. Counts up to 1023 per replicate are exactly
    /// representable at every supported simulation budget.
    pub fn assert_with_overlap_refusals(&self, refusals: &Self) {
        refusals.finish();
        self.assert();
        if !sharding() {
            assert_eq!(refusals.attempts(), self.attempts());
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the tally sums small nonnegative integer refusal counts"
            )]
            let total = refusals.length_sum as u64;
            self.emit_design_counts(serde_json::json!({"overlap_refusals": total}));
        }
    }

    fn emit_design_counts(&self, counts: serde_json::Value) {
        let line = serde_json::json!({
            "id": self.record_id().expect("study counts need a bound record"),
            "grid_point": grid_point(),
            "counts": counts
        });
        eprintln!("calibration-design-counts {line}");
    }
}
