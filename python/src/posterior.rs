//! The decoded posterior artifact handed to Python consumers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::*;

/// Decoded posterior artifact for Python consumers.
#[pyclass(frozen)]
pub(crate) struct PosteriorArtifact {
    #[pyo3(get)]
    pub(crate) n_draws: usize,
    #[pyo3(get)]
    pub(crate) mean: Vec<f64>,
    #[pyo3(get)]
    pub(crate) sd: Vec<f64>,
    #[pyo3(get)]
    pub(crate) q025: Vec<f64>,
    #[pyo3(get)]
    pub(crate) q975: Vec<f64>,
    pub(crate) draws: Vec<f64>,
    #[pyo3(get)]
    pub(crate) backend_id: String,
    #[pyo3(get)]
    pub(crate) identification: String,
    #[pyo3(get)]
    pub(crate) unidentified_mass: f64,
    /// Identified graph mass a latency tier left out of the envelope subsample.
    #[pyo3(get)]
    pub(crate) subsampled_out_mass: f64,
    #[pyo3(get)]
    pub(crate) converged: bool,
    #[pyo3(get)]
    pub(crate) hessian_condition: f64,
    #[pyo3(get)]
    pub(crate) quantity_names: Vec<String>,
    /// Source active minus control contrast for effect-to-slope hydration.
    #[pyo3(get)]
    pub(crate) treatment_contrast: Option<f64>,
    // Preserve roles, indices, and names across decode/encode. Python getters
    // return copies, so the decoded schema cannot drift from quantity_names.
    pub(crate) quantity_schema: Option<Vec<PosteriorQuantityWire>>,
}

#[pymethods]
impl PosteriorArtifact {
    /// Posterior draws as a float64 array (one memcpy, no boxed floats). `numpy.asarray(artifact)`
    /// views the same storage through the buffer protocol without copying at all.
    #[getter]
    fn draws<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.draws)
    }

    /// Read-only float64 array over this artifact's own draw storage: no copy, and the array keeps
    /// the artifact alive. `draws` returns an independent writable copy instead.
    fn draws_view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py().import("numpy")?.call_method1("asarray", (slf,))
    }

    #[new]
    #[pyo3(signature = (
        n_draws,
        mean,
        sd,
        q025,
        q975,
        draws,
        backend_id,
        identification,
        quantity_names,
        unidentified_mass=0.0,
        converged=true,
        hessian_condition=f64::NAN,
        treatment_contrast=None,
        subsampled_out_mass=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_draws: usize,
        mean: Vec<f64>,
        sd: Vec<f64>,
        q025: Vec<f64>,
        q975: Vec<f64>,
        draws: Vec<f64>,
        backend_id: String,
        identification: String,
        quantity_names: Vec<String>,
        unidentified_mass: f64,
        converged: bool,
        hessian_condition: f64,
        treatment_contrast: Option<f64>,
        subsampled_out_mass: f64,
    ) -> Self {
        Self {
            n_draws,
            mean,
            sd,
            q025,
            q975,
            draws,
            backend_id,
            identification,
            unidentified_mass,
            subsampled_out_mass,
            converged,
            hessian_condition,
            quantity_names,
            treatment_contrast,
            quantity_schema: None,
        }
    }

    /// Build a summary-only artifact (mean / SD / quantiles, no draws).
    ///
    /// For callers who only hold posterior moments (e.g. from a conjugate update
    /// computed elsewhere) and would otherwise have to fabricate a fake draws array
    /// just to construct an artifact. `n_draws` records the draw count the moments
    /// were computed from; no samples are stored. `encode_posterior_artifact` emits
    /// `draws_encoding = "none"` for artifacts built this way, and
    /// `decode_posterior_artifact` round-trips them back with an empty `draws`.
    /// Supply `treatment_contrast` (source active minus control) for an ATE
    /// summary that will be mapped onto a target regression slope.
    #[staticmethod]
    #[pyo3(signature = (
        n_draws,
        mean,
        sd,
        q025,
        q975,
        backend_id,
        identification,
        quantity_names,
        unidentified_mass=0.0,
        converged=true,
        hessian_condition=f64::NAN,
        treatment_contrast=None,
        subsampled_out_mass=0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_moments(
        n_draws: usize,
        mean: Vec<f64>,
        sd: Vec<f64>,
        q025: Vec<f64>,
        q975: Vec<f64>,
        backend_id: String,
        identification: String,
        quantity_names: Vec<String>,
        unidentified_mass: f64,
        converged: bool,
        hessian_condition: f64,
        treatment_contrast: Option<f64>,
        subsampled_out_mass: f64,
    ) -> Self {
        Self {
            n_draws,
            mean,
            sd,
            q025,
            q975,
            draws: Vec::new(),
            backend_id,
            identification,
            unidentified_mass,
            subsampled_out_mass,
            converged,
            hessian_condition,
            quantity_names,
            treatment_contrast,
            quantity_schema: None,
        }
    }

    /// Number of posterior draws (``n_draws``, independent of whether `draws` is populated).
    fn __len__(&self) -> usize {
        self.n_draws
    }

    /// NumPy array protocol — ``np.asarray(artifact)`` returns the flat posterior draws
    /// directly, without needing ``np.asarray(artifact.draws)``.
    ///
    /// `dtype` and `copy` match the NumPy 2 calling convention.
    /// Default / ``copy=False`` wraps the live `draws` buffer (no extra `Vec` clone).
    /// ``copy=True`` copies into a fresh NumPy-owned array.
    /// ``copy=None`` (default) copies only when a dtype cast requires it.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Tri-state: None = copy-if-needed, Some(false) = never, Some(true) = always.
        let force_copy = match &copy {
            Some(c) => Some(c.is_truthy()?),
            None => None,
        };
        let np = py.import("numpy")?;
        let float64 = np.getattr("float64")?;
        let arr = if force_copy == Some(true) {
            let draws = &slf.borrow().draws;
            PyArray1::from_slice(py, draws).into_any()
        } else {
            let mv = py.import("builtins")?.call_method1("memoryview", (slf,))?;
            np.getattr("asarray")?.call((mv, &float64), None)?
        };
        match dtype {
            None => Ok(arr),
            Some(dt) => match force_copy {
                // copy=None → NumPy default (copy only if the cast requires it).
                None => Ok(np.getattr("asarray")?.call((arr, dt), None)?),
                Some(force) => {
                    // NumPy 2: forward explicit copy= so False cannot silently allocate.
                    // NumPy 1.x lacks copy=; fall back to an explicit same-dtype check.
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", force)?;
                    match np.getattr("asarray")?.call((&arr, &dt), Some(&kwargs)) {
                        Ok(out) => Ok(out),
                        Err(err) if !force => {
                            let msg = err.to_string();
                            if msg.contains("unexpected keyword argument") || msg.contains("copy") {
                                let cur = arr.getattr("dtype")?;
                                if cur.eq(dt)? {
                                    Ok(arr)
                                } else {
                                    Err(crate::value_err(
                                        "Unable to avoid a copy while casting PosteriorArtifact \
                                         dtype; pass copy=True to allow a conversion",
                                    ))
                                }
                            } else {
                                Err(err)
                            }
                        }
                        Err(err) => Err(err),
                    }
                }
            },
        }
    }
}
