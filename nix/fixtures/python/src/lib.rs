use pyo3::prelude::*;
#[pyfunction]
fn add(left: i64, right: i64) -> i64 { left + right }
#[pymodule]
fn nix_python_boundary(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(add, module)?)?;
    Ok(())
}
