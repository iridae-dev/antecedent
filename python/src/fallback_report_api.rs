//! The descriptive-fallback report, rendered by Rust from the parts Python composed.
//!
//! The wording, escaping and layout of the report of a refusal beside its noncausal description
//! and evidence plan are Rust's (`report::render_fallback_input`, the same renderer
//! `render_fallback_report` uses for an `AnalysisWithFallback`). Python hands it the plain values
//! of the parts it holds, as JSON, and receives Markdown or one self-contained HTML document. No
//! figure is computed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::report::{
    FallbackAssociationInput, FallbackEvidenceInput, FallbackFigures, FallbackObligationInput,
    FallbackOptionInput, FallbackRankingInput, FallbackReportInput, ReportFormat,
    render_fallback_input,
};
use pyo3::prelude::*;
use serde::Deserialize;

/// A figure that is a finite number or the text of a non-finite one.
#[derive(Deserialize)]
#[serde(untagged)]
enum Figure {
    Number(f64),
    Text(String),
}

impl Figure {
    fn value(&self) -> PyResult<f64> {
        match self {
            Self::Number(value) => Ok(*value),
            Self::Text(text) => match text.as_str() {
                "nan" => Ok(f64::NAN),
                "inf" => Ok(f64::INFINITY),
                "-inf" => Ok(f64::NEG_INFINITY),
                other => Err(crate::value_err(format!("a figure must be a number, got {other:?}"))),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FiguresWire {
    analysis: String,
    rows: usize,
    uncertainty: String,
    calibration: String,
    refused_after: Option<String>,
    terms: Vec<(String, Figure)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DescribedWire {
    claim_kind: String,
    identification: String,
    rule: String,
    figures: Option<FiguresWire>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum AssociationWire {
    Unavailable(String),
    Described(DescribedWire),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObligationWire {
    id: String,
    kind: String,
    route: String,
    proof_step: Option<String>,
    satisfiable_by_study: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OptionWire {
    rank: String,
    route: String,
    studies: Vec<String>,
    gate: String,
    cost: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum RankingWire {
    NotRequested,
    Unavailable(String),
    Ranked(Vec<OptionWire>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NeedsWire {
    obligations: Vec<ObligationWire>,
    next_actions: Vec<(String, String)>,
    ranking: RankingWire,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceWire {
    Unavailable(String),
    Needs(NeedsWire),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReportWire {
    reason_code: Option<String>,
    message: String,
    remedy: Option<String>,
    associational: AssociationWire,
    evidence: EvidenceWire,
}

impl ReportWire {
    fn into_input(self) -> PyResult<FallbackReportInput> {
        let associational = match self.associational {
            AssociationWire::Unavailable(reason) => Err(reason),
            AssociationWire::Described(part) => Ok(FallbackAssociationInput {
                claim_kind: part.claim_kind,
                identification: part.identification,
                rule: part.rule,
                figures: part
                    .figures
                    .map(|figures| -> PyResult<FallbackFigures> {
                        Ok(FallbackFigures {
                            analysis: figures.analysis,
                            rows: figures.rows,
                            uncertainty: figures.uncertainty,
                            calibration: figures.calibration,
                            refused_after: figures.refused_after,
                            terms: figures
                                .terms
                                .into_iter()
                                .map(|(label, figure)| Ok((label, figure.value()?)))
                                .collect::<PyResult<Vec<_>>>()?,
                        })
                    })
                    .transpose()?,
            }),
        };
        let evidence = match self.evidence {
            EvidenceWire::Unavailable(reason) => Err(reason),
            EvidenceWire::Needs(needs) => Ok(FallbackEvidenceInput {
                obligations: needs
                    .obligations
                    .into_iter()
                    .map(|o| FallbackObligationInput {
                        id: o.id,
                        kind: o.kind,
                        route: o.route,
                        proof_step: o.proof_step,
                        satisfiable_by_study: o.satisfiable_by_study,
                    })
                    .collect(),
                next_actions: needs.next_actions,
                ranking: match needs.ranking {
                    RankingWire::NotRequested => FallbackRankingInput::NotRequested,
                    RankingWire::Unavailable(reason) => FallbackRankingInput::Unavailable(reason),
                    RankingWire::Ranked(options) => FallbackRankingInput::Ranked(
                        options
                            .into_iter()
                            .map(|o| FallbackOptionInput {
                                rank: o.rank,
                                route: o.route,
                                studies: o.studies,
                                gate: o.gate,
                                cost: o.cost,
                            })
                            .collect(),
                    ),
                },
            }),
        };
        Ok(FallbackReportInput {
            reason_code: self.reason_code,
            message: self.message,
            remedy: self.remedy,
            associational,
            evidence,
        })
    }
}

/// Render the report of a refusal with its descriptive fallback as `markdown` or `html`, from
/// the JSON values of its parts.
#[pyfunction]
fn render_fallback_report(parts_json: &str, format: &str) -> PyResult<String> {
    let format = ReportFormat::from_name(format).ok_or_else(|| {
        crate::value_err(format!("unknown report format {format:?}; use markdown or html"))
    })?;
    let wire: ReportWire = serde_json::from_str(parts_json)
        .map_err(|e| crate::value_err(format!("invalid fallback report parts: {e}")))?;
    Ok(render_fallback_input(&wire.into_input()?, format))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(render_fallback_report, module)?)?;
    Ok(())
}
