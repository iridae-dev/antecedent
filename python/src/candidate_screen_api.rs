//! Python binding for the deterministic screen/estimate split from declared ownership (2.2 E4).
use antecedent::{CandidateProcedure, CandidateScreen, ScreenUnits};
use pyo3::prelude::*;

/// Receipt fields of a generated split, beside the screen itself.
type SplitFields = (String, Vec<u32>, Vec<u32>, u64, String, usize, usize, usize, String);

/// Split rows into disjoint screen and estimate halves that never share an entity, cluster or
/// connected dyad component: `(screen_id, screen_rows, estimate_rows, seed, unit_kind,
/// n_units, screen_units, estimate_units, unit_digest)`.
#[pyfunction]
#[pyo3(signature=(screen_id, procedure, *, screen_fraction, seed, entity_ids=None, first=None, second=None))]
fn candidate_screen_from_units(
    screen_id: &str,
    procedure: &str,
    screen_fraction: f64,
    seed: u64,
    entity_ids: Option<Vec<u32>>,
    first: Option<Vec<u32>>,
    second: Option<Vec<u32>>,
) -> PyResult<SplitFields> {
    let procedure = match procedure {
        "max_t" => CandidateProcedure::MaxT,
        "bh" => CandidateProcedure::BenjaminiHochberg,
        "by" => CandidateProcedure::BenjaminiYekutieli,
        "unrecorded" => CandidateProcedure::Unrecorded,
        other => {
            return Err(crate::value_err(format!(
                "unknown screen procedure {other:?}; use max_t|bh|by|unrecorded"
            )));
        }
    };
    let units = match (&entity_ids, &first, &second) {
        (Some(ids), None, None) => ScreenUnits::Entity(ids),
        (None, Some(first), Some(second)) => ScreenUnits::Dyad { first, second },
        _ => {
            return Err(crate::value_err(
                "declare either entity_ids, or both first and second dyad endpoints",
            ));
        }
    };
    let (screen, receipt) =
        CandidateScreen::from_units(screen_id, procedure, units, screen_fraction, seed)
            .map_err(crate::py_err)?;
    Ok((
        screen.screen_id.to_string(),
        screen.screen_rows.to_vec(),
        screen.estimate_rows.to_vec(),
        receipt.seed,
        receipt.unit_kind.to_string(),
        receipt.n_units,
        receipt.screen_units,
        receipt.estimate_units,
        receipt.unit_digest,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(candidate_screen_from_units, module)?)?;
    Ok(())
}
