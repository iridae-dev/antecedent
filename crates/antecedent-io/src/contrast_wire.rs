//! Contrast / categorical coding wire (ADR 0003).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// One recorded contrast on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RecordedContrastWire {
    /// Variable dense id.
    pub variable: u32,
    /// Coding kind tag.
    pub coding: String,
    /// Reference category when treatment coding.
    pub reference: Option<u32>,
    /// Custom matrix row-major when present (`levels * columns`).
    pub custom_matrix: Option<Vec<f64>>,
    /// Matrix rows.
    pub custom_rows: Option<u32>,
    /// Matrix cols.
    pub custom_cols: Option<u32>,
}

/// Bundle of contrasts for categorical variables.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ContrastBundleWire {
    /// Contrasts.
    pub contrasts: Vec<RecordedContrastWire>,
}
