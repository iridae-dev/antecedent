//! Python bindings for the preflight, rank-drop and cost surface.
//!
//! Reports cross the boundary as JSON text (the Rust report types are `Serialize`) and the
//! Python wrapper parses them into frozen dataclasses, so the two languages share one shape.
//! The fit-free report ([`antecedent::PreflightReport`]) and the fitted diagnostics
//! ([`antecedent::NuisanceFitDiagnostics`]) are separate calls, as they are in Rust.

use std::sync::Arc;

use antecedent::{
    CausalError, ColumnPriority, CostModel, EstimatorId, PreflightInput, RankDropPolicy,
    estimate_with_rank_drop, fit_diagnostics_design, plan_rank_drop, preflight_design,
};
use antecedent_core::{CausalSchema, VariableId};
use antecedent_data::TableView;
use pyo3::prelude::*;
use serde::Serialize;

use crate::ate_api::PyPreparedBatch;
use crate::prepared_api::PyPreparedAnalysis;
use crate::{
    detach_catch, py_err, py_execution_context, resolve_user_threads, tabular_from_py_columns,
};

use serde::ser::{self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple, SerializeTupleStruct, SerializeTupleVariant};
use std::io::{self, Write};

const META: &str = "nonfinite_numeric_fields";

// A path-only serialization pass. It retains only the current path and the paths found.
#[derive(Default)]
struct Scan {
    path: Vec<String>,
    found: Vec<Vec<String>>,
    root: RootShape,
    object_entries: std::collections::BTreeSet<String>,
}
#[derive(Clone, Copy, Default, PartialEq)]
enum RootShape { #[default] Unknown, Object, Array, Other }

impl Scan {
    fn shape(&mut self, shape: RootShape) {
        if self.path.is_empty() { self.root = shape; }
        else if self.root == RootShape::Array && self.path.len() == 1 && shape == RootShape::Object {
            self.object_entries.insert(self.path[0].clone());
        }
    }
    fn child<T: ?Sized + Serialize>(&mut self, key: String, value: &T) -> Result<(), serde_json::Error> {
        self.path.push(key);
        let result = value.serialize(&mut *self);
        self.path.pop();
        result
    }
    fn key(&self, key: &str) -> Result<(), serde_json::Error> {
        if key == META && ((self.path.is_empty() && self.root == RootShape::Object) || (self.root == RootShape::Array && self.path.len() == 1 && self.object_entries.contains(&self.path[0]))) {
            return Err(ser::Error::custom("reserved nonfinite metadata key"));
        }
        Ok(())
    }
}
struct Compound<'a> { scan: &'a mut Scan, index: usize, key: Option<String>, variant: bool }
impl<'a> Compound<'a> {
    fn new(scan: &'a mut Scan, variant: bool) -> Self { Self { scan, index: 0, key: None, variant } }
    fn item<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), serde_json::Error> {
        let i = self.index; self.index += 1; self.scan.child(i.to_string(), v)
    }
    fn field<T: ?Sized + Serialize>(&mut self, key: &str, v: &T) -> Result<(), serde_json::Error> {
        self.scan.key(key)?; self.scan.child(key.to_owned(), v)
    }
    fn finish(self) { if self.variant { self.scan.path.pop(); } }
}
macro_rules! seq_impl { ($trait:ident, $method:ident) => { impl $trait for Compound<'_> { type Ok = (); type Error = serde_json::Error; fn $method<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), Self::Error> { self.item(v) } fn end(self) -> Result<(), Self::Error> { self.finish(); Ok(()) } } }; }
seq_impl!(SerializeSeq, serialize_element);
seq_impl!(SerializeTuple, serialize_element);
seq_impl!(SerializeTupleStruct, serialize_field);
seq_impl!(SerializeTupleVariant, serialize_field);
impl SerializeStruct for Compound<'_> { type Ok=(); type Error=serde_json::Error; fn serialize_field<T: ?Sized + Serialize>(&mut self, k: &'static str, v: &T) -> Result<(),Self::Error> { self.field(k,v) } fn end(self)->Result<(),Self::Error>{self.finish();Ok(())} }
impl SerializeStructVariant for Compound<'_> { type Ok=(); type Error=serde_json::Error; fn serialize_field<T: ?Sized + Serialize>(&mut self, k: &'static str, v: &T) -> Result<(),Self::Error> { self.field(k,v) } fn end(self)->Result<(),Self::Error>{self.finish();Ok(())} }
impl SerializeMap for Compound<'_> {
    type Ok=(); type Error=serde_json::Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(),Self::Error> {
        // Ask serde_json's own map-key serializer to validate and encode exactly one key.
        // Only the current key is buffered; no report tree is materialized.
        struct OneKey<'a, T: ?Sized>(&'a T);
        impl<T: ?Sized + Serialize> Serialize for OneKey<'_, T> {
            fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(self.0, &())?;
                map.end()
            }
        }
        let encoded = serde_json::to_string(&OneKey(key))?;
        let keys: std::collections::BTreeMap<String, ()> = serde_json::from_str(&encoded)?;
        let key = keys.into_keys().next().ok_or_else(|| <serde_json::Error as ser::Error>::custom("missing JSON map key"))?;
        self.scan.key(&key)?; self.key=Some(key); Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self,v:&T)->Result<(),Self::Error>{let k=self.key.take().ok_or_else(|| <serde_json::Error as ser::Error>::custom("map value without key"))?;self.scan.child(k,v)}
    fn end(self)->Result<(),Self::Error>{self.finish();Ok(())}
}
macro_rules! scalar { ($($name:ident($ty:ty)),*) => { $(fn $name(self,_: $ty)->Result<(),Self::Error>{self.shape(RootShape::Other);Ok(())})* }; }
impl<'a> ser::Serializer for &'a mut Scan {
    type Ok=(); type Error=serde_json::Error;
    type SerializeSeq=Compound<'a>; type SerializeTuple=Compound<'a>; type SerializeTupleStruct=Compound<'a>; type SerializeTupleVariant=Compound<'a>; type SerializeMap=Compound<'a>; type SerializeStruct=Compound<'a>; type SerializeStructVariant=Compound<'a>;
    scalar!(serialize_bool(bool),serialize_i8(i8),serialize_i16(i16),serialize_i32(i32),serialize_i64(i64),serialize_i128(i128),serialize_u8(u8),serialize_u16(u16),serialize_u32(u32),serialize_u64(u64),serialize_u128(u128),serialize_char(char),serialize_str(&str),serialize_bytes(&[u8]));
    fn serialize_f32(self,v:f32)->Result<(),Self::Error>{self.shape(RootShape::Other);if !v.is_finite(){self.found.push(self.path.clone())} Ok(())}
    fn serialize_f64(self,v:f64)->Result<(),Self::Error>{self.shape(RootShape::Other);if !v.is_finite(){self.found.push(self.path.clone())} Ok(())}
    fn serialize_none(self)->Result<(),Self::Error>{self.shape(RootShape::Other);Ok(())} fn serialize_some<T:?Sized+Serialize>(self,v:&T)->Result<(),Self::Error>{v.serialize(self)}
    fn serialize_unit(self)->Result<(),Self::Error>{self.shape(RootShape::Other);Ok(())} fn serialize_unit_struct(self,_:&'static str)->Result<(),Self::Error>{self.shape(RootShape::Other);Ok(())} fn serialize_unit_variant(self,_:&'static str,_:u32,_:&'static str)->Result<(),Self::Error>{self.shape(RootShape::Other);Ok(())}
    fn serialize_newtype_struct<T:?Sized+Serialize>(self,_:&'static str,v:&T)->Result<(),Self::Error>{v.serialize(self)}
    fn serialize_newtype_variant<T:?Sized+Serialize>(self,_:&'static str,_:u32,k:&'static str,v:&T)->Result<(),Self::Error>{self.shape(RootShape::Object);self.key(k)?;self.child(k.into(),v)}
    fn serialize_seq(self,_:Option<usize>)->Result<Self::SerializeSeq,Self::Error>{self.shape(RootShape::Array);Ok(Compound::new(self,false))}
    fn serialize_tuple(self,_:usize)->Result<Self::SerializeTuple,Self::Error>{self.shape(RootShape::Array);Ok(Compound::new(self,false))}
    fn serialize_tuple_struct(self,_:&'static str,_:usize)->Result<Self::SerializeTupleStruct,Self::Error>{self.shape(RootShape::Array);Ok(Compound::new(self,false))}
    fn serialize_tuple_variant(self,_:&'static str,_:u32,k:&'static str,_:usize)->Result<Self::SerializeTupleVariant,Self::Error>{self.shape(RootShape::Object);self.key(k)?;self.path.push(k.into());Ok(Compound::new(self,true))}
    fn serialize_map(self,_:Option<usize>)->Result<Self::SerializeMap,Self::Error>{self.shape(RootShape::Object);Ok(Compound::new(self,false))}
    fn serialize_struct(self,_:&'static str,_:usize)->Result<Self::SerializeStruct,Self::Error>{self.shape(RootShape::Object);Ok(Compound::new(self,false))}
    fn serialize_struct_variant(self,_:&'static str,_:u32,k:&'static str,_:usize)->Result<Self::SerializeStructVariant,Self::Error>{self.shape(RootShape::Object);self.key(k)?;self.path.push(k.into());Ok(Compound::new(self,true))}
}
fn pointer(path: &[String]) -> String { path.iter().map(|s| format!("/{}", s.replace('~', "~0").replace('/', "~1"))).collect() }

// Insert metadata only into the root object, or each object in a root array.
struct MetadataFormatter { paths: Vec<Vec<String>>, stack: Vec<(bool, usize)>, batch: bool }
impl serde_json::ser::Formatter for MetadataFormatter {
    fn begin_array<W: ?Sized + Write>(&mut self,w:&mut W)->io::Result<()> { if self.stack.is_empty(){self.batch=true;} self.stack.push((false,0)); w.write_all(b"[") }
    fn end_array<W: ?Sized + Write>(&mut self,w:&mut W)->io::Result<()> { self.stack.pop(); w.write_all(b"]") }
    fn begin_array_value<W: ?Sized + Write>(&mut self,w:&mut W,first:bool)->io::Result<()> { if !first {w.write_all(b",")?;} if let Some(top)=self.stack.last_mut(){top.1+=1;} Ok(()) }
    fn begin_object<W: ?Sized + Write>(&mut self,w:&mut W)->io::Result<()> { self.stack.push((true,0)); w.write_all(b"{") }
    fn begin_object_key<W: ?Sized + Write>(&mut self,w:&mut W,first:bool)->io::Result<()> { if !first {w.write_all(b",")?;} if let Some(top)=self.stack.last_mut(){top.1+=1;} Ok(()) }
    fn end_object<W: ?Sized + Write>(&mut self,w:&mut W)->io::Result<()> {
        let root = self.stack.len()==1 && !self.batch;
        let entry = self.batch && self.stack.len()==2 && !self.stack[0].0;
        if root || entry {
            let index = if entry {Some((self.stack[0].1-1).to_string())} else {None};
            let paths: Vec<String> = self.paths.iter().filter_map(|p| match &index { Some(i) if p.first()==Some(i) => Some(pointer(&p[1..])), Some(_) => None, None => Some(pointer(p)) }).collect();
            if !paths.is_empty() { if self.stack.last().unwrap().1>0 {w.write_all(b",")?;} w.write_all(b"\"nonfinite_numeric_fields\":")?; serde_json::to_writer(&mut *w,&paths).map_err(io::Error::other)?; }
        }
        self.stack.pop(); w.write_all(b"}")
    }
}
fn checked_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut scan=Scan::default(); value.serialize(&mut scan)?;
    for path in &scan.found {
        let supported = match scan.root {
            RootShape::Object => true,
            RootShape::Array => path.first().is_some_and(|index| scan.object_entries.contains(index)),
            _ => false,
        };
        if !supported { return Err(ser::Error::custom("nonfinite value in unsupported JSON report shape")); }
    }
    let formatter=MetadataFormatter { paths: scan.found, stack: Vec::new(), batch: false };
    let mut out=Vec::new(); value.serialize(&mut serde_json::Serializer::with_formatter(&mut out,formatter))?;
    String::from_utf8(out).map_err(ser::Error::custom)
}

fn to_json<T: Serialize>(value: &T) -> PyResult<String> {
    checked_json(value).map_err(|e| crate::value_err(e.to_string()))
}

/// Resolve a declared priority (column names, highest first) against the frozen schema.
fn policy_for(schema: &CausalSchema, priority: Option<Vec<String>>) -> PyResult<RankDropPolicy> {
    let priority = match priority {
        None => ColumnPriority::AdjustmentOrder,
        Some(names) => ColumnPriority::Declared(
            names
                .iter()
                .map(|name| schema.id_of(name))
                .collect::<Result<Vec<VariableId>, _>>()
                .map_err(|e| py_err(CausalError::from(e)))?,
        ),
    };
    Ok(RankDropPolicy { priority })
}

/// Structured refusal fields as JSON (`null` and empty lists for absent entries).
pub(crate) fn refusal_fields_value(
    fields: &antecedent_estimate::RefusalFields,
) -> serde_json::Value {
    serde_json::json!({
        "stage": fields.stage,
        "subject": fields.subject,
        "reason": fields.reason,
        "arm_ess": fields.arm_ess.iter().map(|(l, e)| (l, e.0)).collect::<Vec<_>>(),
        "propensity_min": fields.propensity_min.map(|v| v.0),
        "propensity_max": fields.propensity_max.map(|v| v.0),
        "propensity_quantiles": fields
            .propensity_quantiles
            .iter()
            .map(|(p, v)| (p.0, v.0))
            .collect::<Vec<_>>(),
        "cluster_count": fields.cluster_count,
        "cluster_minimum": fields.cluster_minimum,
        "numerical_rank": fields.numerical_rank,
        "design_columns": fields.design_columns,
        "implicated_columns": fields.implicated_columns,
        "remedy": fields.remedy,
        "glm_iterations": fields.glm_iterations,
        "boundary_margin": fields.boundary_margin.map(|v| v.0),
        "boundary_count": fields.boundary_count,
    })
}

/// Structured refusal fields as a Python dict (`None` and empty lists for absent entries).
pub(crate) fn refusal_fields_dict<'py>(
    py: Python<'py>,
    fields: &antecedent_estimate::RefusalFields,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (refusal_fields_value(fields).to_string(),))
}

/// Attach a refusal's structured fields (`refusal_fields`) to the raised exception.
///
/// Additive: set only when the refusal carries them, so every other exception reads the
/// `refusal_fields = None` class default registered with the errors.
pub(crate) fn with_refusal_fields(
    err: PyErr,
    fields: Option<&antecedent_estimate::RefusalFields>,
) -> PyErr {
    if let Some(fields) = fields {
        Python::attach(|py| {
            if let Ok(dict) = refusal_fields_dict(py, fields) {
                let _ = err.value(py).setattr("refusal_fields", dict);
            }
        });
    }
    err
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Fit-free preflight report of this plan, as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_json(&self, py: Python<'_>, seed: u64, threads: Option<u32>) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose(&ctx).map_err(py_err)?)
        })
    }

    /// Propensity-fit diagnostics (fit-requiring), as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_fit_json(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose_fit(&ctx).map_err(py_err)?)
        })
    }

    /// Opt-in rank-drop plan under a declared priority (`None` = adjustment-set order).
    #[pyo3(signature = (priority=None, *, seed=1, threads=None))]
    fn plan_rank_drop_json(
        &self,
        py: Python<'_>,
        priority: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let policy = policy_for(inner.schema(), priority)?;
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.plan_rank_drop(&policy, &ctx).map_err(py_err)?)
        })
    }

    /// Planning-time cost counts, as JSON. `cost_model` replaces the process-wide model
    /// (see `build_cost_model`), so the seconds do not depend on the working directory.
    #[pyo3(signature = (cost_model=None))]
    fn estimate_cost_json(&self, cost_model: Option<CostModelParts>) -> PyResult<String> {
        match cost_model {
            None => to_json(&self.inner.estimate_cost().map_err(py_err)?),
            Some(parts) => {
                let model = model_from_parts(parts)?;
                to_json(&self.inner.estimate_cost_with(Some(&model)).map_err(py_err)?)
            }
        }
    }
}

#[pymethods]
impl PyPreparedBatch {
    /// Fit-free preflight report of every plan, as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_json(&self, py: Python<'_>, seed: u64, threads: Option<u32>) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose(&ctx).map_err(py_err)?)
        })
    }

    /// Propensity-fit diagnostics of every plan (fit-requiring), as JSON.
    #[pyo3(signature = (*, seed=1, threads=None))]
    fn diagnose_fit_json(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.diagnose_fit(&ctx).map_err(py_err)?)
        })
    }

    /// Opt-in rank-drop plans under a declared priority (`None` = adjustment-set order).
    #[pyo3(signature = (priority=None, *, seed=1, threads=None))]
    fn plan_rank_drop_json(
        &self,
        py: Python<'_>,
        priority: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let inner = Arc::clone(&self.inner);
        let policy = {
            let first = inner
                .plans()
                .first()
                .ok_or_else(|| crate::value_err("a prepared batch holds no plans"))?;
            policy_for(first.schema(), priority)?
        };
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            to_json(&inner.plan_rank_drop(&policy, &ctx).map_err(py_err)?)
        })
    }

    /// Planning-time cost counts of every plan and their totals, as JSON. `cost_model`
    /// replaces the process-wide model for every plan.
    #[pyo3(signature = (cost_model=None))]
    fn estimate_cost_json(&self, cost_model: Option<CostModelParts>) -> PyResult<String> {
        match cost_model {
            None => to_json(&self.inner.estimate_cost().map_err(py_err)?),
            Some(parts) => {
                let model = model_from_parts(parts)?;
                to_json(&self.inner.estimate_cost_with(Some(&model)).map_err(py_err)?)
            }
        }
    }
}

/// `(benchmark, platform, cpu_count, [(estimator id, a, b)])` of a cost model.
type CostModelParts = (String, String, u64, Vec<(String, f64, f64)>);

fn cost_model_error(error: &antecedent::CostModelError) -> PyErr {
    crate::value_err_code(antecedent_core::reason_code!("invalid_argument"), error.to_string())
}

fn model_from_parts(
    (benchmark, platform, cpu_count, coefficients): CostModelParts,
) -> PyResult<CostModel> {
    CostModel::new(&benchmark, &platform, cpu_count, coefficients).map_err(|e| cost_model_error(&e))
}

fn parts_of(model: &CostModel) -> CostModelParts {
    (
        model.benchmark().to_owned(),
        model.platform().to_owned(),
        model.cpu_count(),
        model.coefficients().map(|(id, a, b)| (id.to_owned(), a, b)).collect(),
    )
}

/// A cost model from its declared parts; refuses an empty benchmark or platform, a zero CPU
/// count, no coefficients and a negative or non-finite coefficient.
#[pyfunction]
fn build_cost_model(
    benchmark: &str,
    platform: &str,
    cpu_count: u64,
    coefficients: Vec<(String, f64, f64)>,
) -> PyResult<CostModelParts> {
    let model =
        model_from_parts((benchmark.to_owned(), platform.to_owned(), cpu_count, coefficients))?;
    Ok(parts_of(&model))
}

/// A cost model from the TOML `scripts/bench_cost_model.py` writes; refuses malformed text and
/// a model without a benchmark, a machine or valid coefficients.
#[pyfunction]
fn parse_cost_model(text: &str) -> PyResult<CostModelParts> {
    let model = CostModel::from_toml(text).map_err(|e| cost_model_error(&e))?;
    Ok(parts_of(&model))
}

/// A table and the resolved columns of a binary-effect design.
struct BoundDesign {
    data: antecedent_data::TabularData,
    treatment: VariableId,
    outcome: VariableId,
    adjustment: Vec<VariableId>,
}

impl BoundDesign {
    fn bind(
        _py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: &str,
        outcome: &str,
        adjustment: &[String],
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(names, columns)?;
        let lookup =
            |name: &str| data.schema().id_of(name).map_err(|e| py_err(CausalError::from(e)));
        let treatment = lookup(treatment)?;
        let outcome = lookup(outcome)?;
        let adjustment = adjustment.iter().map(|n| lookup(n)).collect::<PyResult<Vec<_>>>()?;
        Ok(Self { data, treatment, outcome, adjustment })
    }

    fn input(&self, control: f64, active: f64) -> PreflightInput<'_> {
        PreflightInput::binary_effect(
            &self.data,
            self.treatment,
            self.outcome,
            &self.adjustment,
            control,
            active,
        )
    }
}

/// Fit-free preflight of a declared binary-effect design, before any study is prepared (a
/// design a fit would refuse cannot be prepared, so this is the entry point for it).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn preflight_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&preflight_design(&design.input(control, active), &ctx).map_err(py_err)?)
    })
}

/// Propensity-fit diagnostics of a declared binary-effect design (fit-requiring).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn fit_diagnostics_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&fit_diagnostics_design(&design.input(control, active), &ctx).map_err(py_err)?)
    })
}

/// Rank-drop plan of a declared binary-effect design under a declared priority.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, priority=None, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn rank_drop_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    priority: Option<Vec<String>>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    let policy = policy_for(design.data.schema(), priority)?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(&plan_rank_drop(&design.input(control, active), &policy, &ctx).map_err(py_err)?)
    })
}

/// Estimate a declared binary-effect design on the span-preserving reduced adjustment set of
/// a rank-drop plan (`estimator` is a licensed estimator id, e.g. `aipw`).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, adjustment, estimator, priority=None, control=0.0, active=1.0, *, seed=1, threads=None))]
#[allow(clippy::too_many_arguments, reason = "mirrors the Python keyword surface")]
fn estimate_with_rank_drop_json(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    treatment: &str,
    outcome: &str,
    adjustment: Vec<String>,
    estimator: &str,
    priority: Option<Vec<String>>,
    control: f64,
    active: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    let design = BoundDesign::bind(py, names, columns, treatment, outcome, &adjustment)?;
    let policy = policy_for(design.data.schema(), priority)?;
    let estimator =
        estimator.parse::<EstimatorId>().map_err(|e| crate::value_err(e.to_string()))?;
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        to_json(
            &estimate_with_rank_drop(&design.input(control, active), &policy, estimator, &ctx)
                .map_err(py_err)?,
        )
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(preflight_json, module)?)?;
    module.add_function(wrap_pyfunction!(fit_diagnostics_json, module)?)?;
    module.add_function(wrap_pyfunction!(rank_drop_json, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_with_rank_drop_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_cost_model, module)?)?;
    module.add_function(wrap_pyfunction!(parse_cost_model, module)?)?;
    Ok(())
}

#[cfg(test)]
mod nonfinite_json_tests {
    use super::checked_json;
    use serde::Serialize;
    use std::collections::BTreeMap;

    #[derive(Serialize)]
    struct Example {
        nested: Vec<BTreeMap<String, Option<f64>>>,
        absent: Option<f64>,
        finite: f32,
        single: f32,
    }
    fn example() -> Example {
        Example {
            nested: vec![BTreeMap::from([
                ("a~/b".into(), Some(f64::NAN)),
                ("absent".into(), None),
                ("infinite".into(), Some(f64::INFINITY)),
            ])],
            absent: None, finite: 1.0, single: f32::NEG_INFINITY,
        }
    }
    #[test]
    fn paths_distinguish_nonfinite_from_none_and_escape_keys() {
        let result: serde_json::Value = serde_json::from_str(&checked_json(&example()).unwrap()).unwrap();
        assert_eq!(result["nonfinite_numeric_fields"], serde_json::json!(["/nested/0/a~0~1b", "/nested/0/infinite", "/single"]));
        assert!(result["absent"].is_null());
        assert!(result["nested"][0]["absent"].is_null());
    }
    #[test]
    fn batch_paths_are_relative_to_each_object_and_finite_bytes_are_unchanged() {
        let clean = Example { nested: vec![], absent: None, finite: 1.0, single: 2.0 };
        let batch = [clean, example()];
        let json = checked_json(&batch).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed[0].get("nonfinite_numeric_fields").is_none());
        assert_eq!(parsed[1]["nonfinite_numeric_fields"][0], "/nested/0/a~0~1b");
        assert_eq!(checked_json(&batch[0]).unwrap(), serde_json::to_string(&batch[0]).unwrap());
    }
    #[test]
    fn finite_float_map_keys_match_json_and_nested_numeric_keys_do_not_collide() {
        struct FloatKey;
        impl Serialize for FloatKey {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(&1.25_f64, &f64::NAN)?;
                map.end()
            }
        }
        let json = checked_json(&FloatKey).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["nonfinite_numeric_fields"], serde_json::json!(["/1.25"]));
        let nested = BTreeMap::from([("0", BTreeMap::from([("nonfinite_numeric_fields", 7)]))]);
        assert_eq!(checked_json(&nested).unwrap(), serde_json::to_string(&nested).unwrap());
    }
    #[test]
    fn rejects_nonfinite_unsupported_root_shapes() {
        assert!(checked_json(&f64::NAN).is_err());
        assert!(checked_json(&vec![vec![f64::INFINITY]]).is_err());
        assert_eq!(checked_json(&vec![1.0]).unwrap(), "[1.0]");
    }
    #[test]
    fn reserved_enum_variant_at_root_or_batch_entry_is_rejected() {
        #[derive(Serialize)]
        enum Collision { #[serde(rename = "nonfinite_numeric_fields")] Value(f64) }
        assert!(checked_json(&Collision::Value(f64::NAN)).is_err());
        assert!(checked_json(&vec![Collision::Value(f64::NAN)]).is_err());
    }
    #[test]
    fn reserved_field_is_rejected() {
        let object = BTreeMap::from([("nonfinite_numeric_fields", 1)]);
        assert!(checked_json(&object).is_err());
    }
}
