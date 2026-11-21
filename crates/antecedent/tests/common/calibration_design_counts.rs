//! Extra counts for studies that distinguish returned intervals from refused panels.

use super::{CoverageTally, grid_point, last_map_end, sharding, smoke};

impl CoverageTally {
    /// Publish the actual attempted, returned and covered counts for a scoped panel study.
    pub fn assert_panel_interval(mut self) {
        let scoped = self.record.as_ref().and_then(|r| r.construction.as_ref()).is_some_and(|c| {
            c.query == "SyntheticControl" && c.functional.starts_with("synthetic_control/")
        });
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
