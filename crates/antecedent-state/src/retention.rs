//! Retention policies for state components, and explicit result-retention modes.
//!
//! [`RetentionPolicy`] is what raw history a component requires. Its variants are
//! unchanged: full raw history, a bounded window, or sufficient statistics.
//! [`RetentionMode`] is what a result keeps for later operations. The default
//! mode is full reusable state, which is the existing result requirement.
//! Summary-only and disk-backed output are reduced modes. Each reports the
//! later operations it can no longer serve. Disk-backed output is a declared
//! mode: the variant carries no path and no byte count, and nothing here spills
//! to disk. [`RetentionMode::require`] refuses an operation the mode cannot serve.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// What raw history a component requires.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RetentionPolicy {
    /// Full raw history must be retained.
    RawHistory,
    /// Only a bounded trailing window is required.
    BoundedWindow {
        /// Maximum rows / events retained.
        max_rows: u64,
    },
    /// Sufficient statistics alone reconstruct the needed view.
    SufficientStatisticsOnly,
}

impl RetentionPolicy {
    /// Result mode this component requirement already is.
    ///
    /// [`Self::RawHistory`] and a [`Self::BoundedWindow`] that retains at least
    /// one row are [`RetentionMode::FullReusable`]. A window of zero rows
    /// retains nothing, so it is [`RetentionMode::SummaryOnly`], the same mode
    /// as [`Self::SufficientStatisticsOnly`]. A positive window still reports
    /// its cap through [`Self::max_retained_rows`]. [`RetentionMode::DiskBacked`]
    /// is never implied. Disk-backed output has to be declared on its own.
    #[must_use]
    pub const fn result_mode(self) -> RetentionMode {
        match self {
            Self::BoundedWindow { max_rows: 0 } | Self::SufficientStatisticsOnly => {
                RetentionMode::SummaryOnly
            }
            Self::RawHistory | Self::BoundedWindow { .. } => RetentionMode::FullReusable,
        }
    }

    /// Row cap this policy already states.
    ///
    /// `None` is full raw history, which has no cap. `Some(0)` retains no rows:
    /// sufficient statistics, or a window whose cap is zero. `Some(max_rows)`
    /// with `max_rows > 0` is a bounded window.
    #[must_use]
    pub const fn max_retained_rows(self) -> Option<u64> {
        match self {
            Self::RawHistory => None,
            Self::BoundedWindow { max_rows } => Some(max_rows),
            Self::SufficientStatisticsOnly => Some(0),
        }
    }

    /// Later operations this policy's result mode can no longer serve.
    ///
    /// Empty for raw history and for a window that retains rows. Sufficient
    /// statistics, and a window of zero rows, report the summary-only losses.
    #[must_use]
    pub const fn no_longer_serves(self) -> &'static [RetainedOperation] {
        self.result_mode().no_longer_serves()
    }

    /// Accept `operation` under this policy's result mode, or refuse it.
    ///
    /// # Errors
    ///
    /// The result mode cannot serve `operation`.
    #[must_use = "an operation this retention cannot serve is refused"]
    pub const fn require(self, operation: RetainedOperation) -> Result<(), RetentionRefusal> {
        self.result_mode().require(operation)
    }
}

/// How a result is retained for later operations.
///
/// [`Self::FullReusable`] is the default. It keeps the existing result
/// requirements. The other two modes are reduced: [`Self::no_longer_serves`]
/// names what each one drops, and [`Self::require`] refuses a request for any
/// operation that mode cannot serve.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum RetentionMode {
    /// Full reusable in-memory state. This is the existing result requirement.
    #[default]
    FullReusable,
    /// Summary only. Rows, and anything that needs rows, are not kept.
    SummaryOnly,
    /// Declared disk-backed output.
    ///
    /// This variant is the declaration. It stores no path and no byte count.
    /// Selecting it does not write, and no other mode spills to disk when memory
    /// is short.
    DiskBacked,
}

impl RetentionMode {
    /// Every mode, default first.
    pub const ALL: [Self; 3] = [Self::FullReusable, Self::SummaryOnly, Self::DiskBacked];

    /// Stable name used in refusals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullReusable => "full reusable state",
            Self::SummaryOnly => "summary-only",
            Self::DiskBacked => "disk-backed output",
        }
    }

    /// Whether this mode can serve `operation`.
    ///
    /// Full reusable state serves every in-memory operation and refuses a disk
    /// read, because that read was not declared. Summary-only serves reuse of
    /// the summary and nothing that needs rows or a disk artifact. Disk-backed
    /// output serves only a read of the declared artifact.
    #[must_use]
    pub const fn serves(self, operation: RetainedOperation) -> bool {
        match self {
            Self::FullReusable => !matches!(operation, RetainedOperation::ReadDiskArtifact),
            Self::SummaryOnly => matches!(operation, RetainedOperation::ReuseSummary),
            Self::DiskBacked => matches!(operation, RetainedOperation::ReadDiskArtifact),
        }
    }

    /// Later operations this mode can no longer serve, relative to full reusable state.
    ///
    /// The default mode returns an empty slice. A disk read is absent from every
    /// slice: full reusable state never served one, so a reduced mode did not lose it.
    #[must_use]
    pub const fn no_longer_serves(self) -> &'static [RetainedOperation] {
        match self {
            Self::FullReusable => FULL_REUSABLE_NO_LONGER_SERVES,
            Self::SummaryOnly => SUMMARY_ONLY_NO_LONGER_SERVES,
            Self::DiskBacked => DISK_BACKED_NO_LONGER_SERVES,
        }
    }

    /// Accept `operation`, or refuse it.
    ///
    /// # Errors
    ///
    /// This mode cannot serve `operation`. The refusal names the mode and the
    /// operation. No state is written and nothing is spilled.
    #[must_use = "an operation this mode cannot serve is refused"]
    pub const fn require(self, operation: RetainedOperation) -> Result<(), RetentionRefusal> {
        if self.serves(operation) {
            Ok(())
        } else {
            Err(RetentionRefusal { mode: self, operation })
        }
    }
}

impl fmt::Display for RetentionMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A later operation that may need retained state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RetainedOperation {
    /// Reconstruct individual rows from what was kept.
    RowReconstruction,
    /// Fit again from the rows that were retained.
    RefitFromRetainedRows,
    /// Drop one retained row and keep an exact result for the rows that remain.
    ///
    /// Full reusable state does this from the retained rows. A summary has no
    /// exact downdate, and summary-only keeps no rows to refit.
    ExactRowEviction,
    /// Insert a column and keep an exact result.
    ///
    /// The rows have to be retained. A summary does not accept an inserted column.
    InPlaceColumnInsert,
    /// Change a weight or a mask and keep an exact result.
    ///
    /// A summary invalidates on that change. The next attempt is a refit from
    /// the retained rows, which summary-only does not have.
    InPlaceWeightOrMaskUpdate,
    /// Reuse the full in-memory state without another scan and without a disk load.
    ReuseFullState,
    /// Reuse the retained summary without the rows it was built from.
    ReuseSummary,
    /// Read the artifact declared by [`RetentionMode::DiskBacked`].
    ///
    /// This is not a spill, and no other mode implies it.
    ReadDiskArtifact,
}

impl RetainedOperation {
    /// Every later operation, in the order reduced modes report losses.
    pub const ALL: [Self; 8] = [
        Self::RowReconstruction,
        Self::RefitFromRetainedRows,
        Self::ExactRowEviction,
        Self::InPlaceColumnInsert,
        Self::InPlaceWeightOrMaskUpdate,
        Self::ReuseFullState,
        Self::ReuseSummary,
        Self::ReadDiskArtifact,
    ];

    /// Stable name used in refusals.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowReconstruction => "row reconstruction",
            Self::RefitFromRetainedRows => "refit from retained rows",
            Self::ExactRowEviction => "exact row eviction",
            Self::InPlaceColumnInsert => "in-place column insert",
            Self::InPlaceWeightOrMaskUpdate => "in-place weight or mask update",
            Self::ReuseFullState => "reuse of full state",
            Self::ReuseSummary => "reuse of the summary",
            Self::ReadDiskArtifact => "read of the declared disk artifact",
        }
    }
}

impl fmt::Display for RetainedOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Full reusable state drops nothing relative to the existing result requirements.
const FULL_REUSABLE_NO_LONGER_SERVES: &[RetainedOperation] = &[];

/// What summary-only can no longer serve.
///
/// Row reconstruction and refit need retained rows. Exact row eviction, an
/// inserted column, and a weight or mask change are the updates the summary
/// type cannot apply; each one asks for a refit from retained rows, which this
/// mode does not have. Full in-memory state is gone with the rows. Reuse of the
/// summary itself remains.
const SUMMARY_ONLY_NO_LONGER_SERVES: &[RetainedOperation] = &[
    RetainedOperation::RowReconstruction,
    RetainedOperation::RefitFromRetainedRows,
    RetainedOperation::ExactRowEviction,
    RetainedOperation::InPlaceColumnInsert,
    RetainedOperation::InPlaceWeightOrMaskUpdate,
    RetainedOperation::ReuseFullState,
];

/// What declared disk-backed output can no longer serve.
///
/// The in-memory result is not kept. Resident reconstruction, refit, in-place
/// updates, and reuse of either the full state or the summary are gone. The
/// mode still serves a read of the declared artifact, and that read is not a loss
/// relative to full reusable state because the default never served a disk read.
const DISK_BACKED_NO_LONGER_SERVES: &[RetainedOperation] = &[
    RetainedOperation::RowReconstruction,
    RetainedOperation::RefitFromRetainedRows,
    RetainedOperation::ExactRowEviction,
    RetainedOperation::InPlaceColumnInsert,
    RetainedOperation::InPlaceWeightOrMaskUpdate,
    RetainedOperation::ReuseFullState,
    RetainedOperation::ReuseSummary,
];

/// Refusal of one later operation.
///
/// The attempt is this value. No rows were reconstructed, no refit ran, and
/// nothing was spilled to disk.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[error("{mode} cannot serve {operation}")]
pub struct RetentionRefusal {
    mode: RetentionMode,
    operation: RetainedOperation,
}

impl RetentionRefusal {
    /// Mode that was asked.
    #[must_use]
    pub const fn mode(self) -> RetentionMode {
        self.mode
    }

    /// Operation that mode cannot serve.
    #[must_use]
    pub const fn operation(self) -> RetainedOperation {
        self.operation
    }

    /// Later operations `mode` can no longer serve.
    #[must_use]
    pub const fn no_longer_serves(self) -> &'static [RetainedOperation] {
        self.mode.no_longer_serves()
    }
}

#[cfg(test)]
mod tests {
    use super::{RetainedOperation, RetentionMode, RetentionPolicy};

    #[test]
    fn default_mode_keeps_full_reusable_state() {
        let mode = RetentionMode::default();
        assert_eq!(mode, RetentionMode::FullReusable);
        assert!(mode.no_longer_serves().is_empty());
        for operation in RetainedOperation::ALL {
            let served = !matches!(operation, RetainedOperation::ReadDiskArtifact);
            assert_eq!(mode.serves(operation), served, "{operation}");
            assert_eq!(mode.require(operation).is_ok(), served, "{operation}");
        }
        let refusal = mode.require(RetainedOperation::ReadDiskArtifact).unwrap_err();
        assert_eq!(refusal.mode(), RetentionMode::FullReusable);
        assert_eq!(refusal.operation(), RetainedOperation::ReadDiskArtifact);
        assert!(refusal.no_longer_serves().is_empty());
        assert_eq!(
            refusal.to_string(),
            "full reusable state cannot serve read of the declared disk artifact"
        );
    }

    #[test]
    fn summary_only_names_the_operations_it_can_no_longer_serve() {
        let mode = RetentionMode::SummaryOnly;
        assert_eq!(
            mode.no_longer_serves(),
            &[
                RetainedOperation::RowReconstruction,
                RetainedOperation::RefitFromRetainedRows,
                RetainedOperation::ExactRowEviction,
                RetainedOperation::InPlaceColumnInsert,
                RetainedOperation::InPlaceWeightOrMaskUpdate,
                RetainedOperation::ReuseFullState,
            ]
        );
        assert!(mode.serves(RetainedOperation::ReuseSummary));
        assert!(mode.require(RetainedOperation::ReuseSummary).is_ok());
        for operation in mode.no_longer_serves() {
            let refusal = mode.require(*operation).unwrap_err();
            assert_eq!(refusal.mode(), mode);
            assert_eq!(refusal.operation(), *operation);
            assert_eq!(refusal.to_string(), format!("summary-only cannot serve {operation}"));
        }
        let disk = mode.require(RetainedOperation::ReadDiskArtifact).unwrap_err();
        assert_eq!(disk.operation(), RetainedOperation::ReadDiskArtifact);
        assert!(!mode.no_longer_serves().contains(&RetainedOperation::ReadDiskArtifact));
    }

    #[test]
    fn disk_backed_output_is_declared_and_refuses_resident_operations() {
        let mode = RetentionMode::DiskBacked;
        assert_eq!(
            mode.no_longer_serves(),
            &[
                RetainedOperation::RowReconstruction,
                RetainedOperation::RefitFromRetainedRows,
                RetainedOperation::ExactRowEviction,
                RetainedOperation::InPlaceColumnInsert,
                RetainedOperation::InPlaceWeightOrMaskUpdate,
                RetainedOperation::ReuseFullState,
                RetainedOperation::ReuseSummary,
            ]
        );
        assert!(mode.require(RetainedOperation::ReadDiskArtifact).is_ok());
        for operation in mode.no_longer_serves() {
            let refusal = mode.require(*operation).unwrap_err();
            assert_eq!(refusal.operation(), *operation);
            assert_eq!(refusal.to_string(), format!("disk-backed output cannot serve {operation}"));
        }
        assert_ne!(mode, RetentionMode::default());
    }

    #[test]
    fn reduced_modes_match_what_full_reusable_state_still_serves() {
        let full = RetentionMode::FullReusable;
        for mode in RetentionMode::ALL {
            for operation in RetainedOperation::ALL {
                let dropped = mode.no_longer_serves().contains(&operation);
                assert_eq!(mode.require(operation).is_ok(), mode.serves(operation));
                if mode == full {
                    assert!(!dropped);
                } else if full.serves(operation) {
                    assert_eq!(dropped, !mode.serves(operation));
                } else {
                    assert!(!dropped);
                }
            }
        }
    }

    #[test]
    fn component_policies_keep_their_requirements() {
        let raw = RetentionPolicy::RawHistory;
        let window = RetentionPolicy::BoundedWindow { max_rows: 7 };
        let summary = RetentionPolicy::SufficientStatisticsOnly;
        assert_eq!(raw.result_mode(), RetentionMode::FullReusable);
        assert_eq!(window.result_mode(), RetentionMode::FullReusable);
        assert_eq!(summary.result_mode(), RetentionMode::SummaryOnly);
        assert_ne!(raw.result_mode(), RetentionMode::DiskBacked);
        assert_ne!(window.result_mode(), RetentionMode::DiskBacked);
        assert_ne!(summary.result_mode(), RetentionMode::DiskBacked);
        assert_eq!(raw.max_retained_rows(), None);
        assert_eq!(window.max_retained_rows(), Some(7));
        assert_eq!(summary.max_retained_rows(), Some(0));
        assert!(raw.no_longer_serves().is_empty());
        assert!(window.no_longer_serves().is_empty());
        assert_eq!(summary.no_longer_serves(), RetentionMode::SummaryOnly.no_longer_serves());
        assert!(raw.require(RetainedOperation::RefitFromRetainedRows).is_ok());
        assert!(window.require(RetainedOperation::RowReconstruction).is_ok());
        assert!(summary.require(RetainedOperation::RowReconstruction).is_err());
        assert!(summary.require(RetainedOperation::RefitFromRetainedRows).is_err());
        assert!(raw.require(RetainedOperation::ReadDiskArtifact).is_err());
        let empty_window = RetentionPolicy::BoundedWindow { max_rows: 0 };
        assert_eq!(empty_window.result_mode(), RetentionMode::SummaryOnly);
        assert_eq!(empty_window.max_retained_rows(), Some(0));
        assert_eq!(empty_window.no_longer_serves(), summary.no_longer_serves());
        assert!(empty_window.require(RetainedOperation::RowReconstruction).is_err());
        assert!(empty_window.require(RetainedOperation::ExactRowEviction).is_err());
        assert!(empty_window.require(RetainedOperation::ReuseSummary).is_ok());
    }
}
