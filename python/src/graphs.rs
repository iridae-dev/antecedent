//! PyO3 typed graph wrappers (`Dag`, `Cpdag`, `Pag`, `Admg`, `TemporalDag`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::io::{
    admg_from_dot as facade_admg_from_dot, admg_from_gml as facade_admg_from_gml,
    admg_from_json as facade_admg_from_json,
    admg_from_networkx_node_link as facade_admg_from_networkx_node_link,
    admg_to_dot as facade_admg_to_dot, admg_to_gml as facade_admg_to_gml,
    admg_to_json as facade_admg_to_json,
    admg_to_networkx_node_link as facade_admg_to_networkx_node_link,
    cpdag_from_dot as facade_cpdag_from_dot, cpdag_from_gml as facade_cpdag_from_gml,
    cpdag_from_json as facade_cpdag_from_json,
    cpdag_from_networkx_node_link as facade_cpdag_from_networkx_node_link,
    cpdag_to_dot as facade_cpdag_to_dot, cpdag_to_gml as facade_cpdag_to_gml,
    cpdag_to_json as facade_cpdag_to_json,
    cpdag_to_networkx_node_link as facade_cpdag_to_networkx_node_link,
    dag_from_dot as facade_dag_from_dot, dag_from_gml as facade_dag_from_gml,
    dag_from_json as facade_dag_from_json,
    dag_from_networkx_adjacency as facade_dag_from_networkx_adjacency,
    dag_from_networkx_node_link as facade_dag_from_networkx_node_link,
    dag_to_dot as facade_dag_to_dot, dag_to_gml as facade_dag_to_gml,
    dag_to_json as facade_dag_to_json,
    dag_to_networkx_adjacency as facade_dag_to_networkx_adjacency,
    dag_to_networkx_node_link as facade_dag_to_networkx_node_link,
    pag_from_dot as facade_pag_from_dot, pag_from_gml as facade_pag_from_gml,
    pag_from_json as facade_pag_from_json,
    pag_from_networkx_node_link as facade_pag_from_networkx_node_link,
    pag_to_dot as facade_pag_to_dot, pag_to_gml as facade_pag_to_gml,
    pag_to_json as facade_pag_to_json,
    pag_to_networkx_node_link as facade_pag_to_networkx_node_link,
};
use antecedent_core::{Lag, VariableId};
use antecedent_graph::{
    Dag as RustDag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, NodeRef,
    TemporalDag as RustTemporalDag, ensure_lagged,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::{CausalGraphError, py_err};

fn unknown_node(name: &str) -> PyErr {
    CausalGraphError::new_err(format!("unknown node '{name}'"))
}

fn parse_endpoint(s: &str) -> PyResult<Endpoint> {
    match s.to_ascii_lowercase().as_str() {
        "tail" => Ok(Endpoint::Tail),
        "arrow" => Ok(Endpoint::Arrow),
        "circle" => Ok(Endpoint::Circle),
        "conflict" => Ok(Endpoint::Conflict),
        other => Err(PyValueError::new_err(format!(
            "unknown endpoint mark '{other}' (expected tail|arrow|circle|conflict)"
        ))),
    }
}

fn endpoint_str(e: Endpoint) -> &'static str {
    match e {
        Endpoint::Tail => "tail",
        Endpoint::Arrow => "arrow",
        Endpoint::Circle => "circle",
        Endpoint::Conflict => "conflict",
    }
}

fn cpdag_mark_str(edge: MarkedEdge) -> &'static str {
    if edge.is_undirected() {
        "undirected"
    } else if edge.is_conflict() {
        "conflict"
    } else if edge.parent_child().is_some() {
        "directed"
    } else {
        "other"
    }
}

fn default_names(n: usize) -> Vec<String> {
    (0..n).map(|i| i.to_string()).collect()
}

fn resolve_name_index(names: &[String], name: &str) -> PyResult<DenseNodeId> {
    names
        .iter()
        .position(|n| n == name)
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node index fits u32")))
        .ok_or_else(|| unknown_node(name))
}

fn id_name(names: &[String], id: DenseNodeId) -> PyResult<String> {
    names
        .get(id.as_usize())
        .cloned()
        .ok_or_else(|| CausalGraphError::new_err(format!("dense id {} out of range", id.raw())))
}

/// Named static DAG.
#[pyclass(name = "Dag", from_py_object)]
#[derive(Clone)]
pub struct Dag {
    pub(crate) dag: RustDag,
    pub(crate) names: Vec<String>,
}

impl Dag {
    /// Resolve a variable name to a dense node id.
    pub(crate) fn name_index(&self, name: &str) -> PyResult<DenseNodeId> {
        resolve_name_index(&self.names, name)
    }

    pub(crate) fn from_rust(dag: RustDag, names: Vec<String>) -> PyResult<Self> {
        if names.len() != dag.node_count() {
            return Err(PyValueError::new_err(format!(
                "names length {} must equal node_count {}",
                names.len(),
                dag.node_count()
            )));
        }
        Ok(Self { dag, names })
    }
}

#[pymethods]
impl Dag {
    #[classmethod]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        edges: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let n = u32::try_from(names.len()).map_err(|_| PyValueError::new_err("too many nodes"))?;
        let mut dag = RustDag::with_variables(n);
        for (from, to) in &edges {
            let f = resolve_name_index(&names, from)?;
            let t = resolve_name_index(&names, to)?;
            dag.insert_directed(f, t).map_err(py_err)?;
        }
        Ok(Self { dag, names })
    }

    #[classmethod]
    fn from_dot(_cls: &Bound<'_, PyType>, dot: &str) -> PyResult<Self> {
        let dag = facade_dag_from_dot(dot).map_err(py_err)?;
        let names = default_names(dag.node_count());
        Ok(Self { dag, names })
    }

    fn nodes(&self) -> Vec<String> {
        self.names.clone()
    }

    fn edges(&self) -> PyResult<Vec<(String, String)>> {
        let mut out = Vec::new();
        for e in self.dag.edges() {
            let Some((from, to)) = e.parent_child() else {
                continue;
            };
            out.push((id_name(&self.names, from)?, id_name(&self.names, to)?));
        }
        Ok(out)
    }

    fn parents(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.dag.parents(id).iter().map(|&p| id_name(&self.names, p)).collect()
    }

    fn children(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.dag.children(id).iter().map(|&c| id_name(&self.names, c)).collect()
    }

    fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    fn to_dot(&self) -> PyResult<String> {
        facade_dag_to_dot(&self.dag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let dag = facade_dag_from_json(json).map_err(py_err)?;
        let names = default_names(dag.node_count());
        Ok(Self { dag, names })
    }

    fn to_json(&self) -> PyResult<String> {
        facade_dag_to_json(&self.dag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_gml(_cls: &Bound<'_, PyType>, gml: &str) -> PyResult<Self> {
        let dag = facade_dag_from_gml(gml).map_err(py_err)?;
        let names = default_names(dag.node_count());
        Ok(Self { dag, names })
    }

    fn to_gml(&self) -> PyResult<String> {
        facade_dag_to_gml(&self.dag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_networkx_node_link(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let dag = facade_dag_from_networkx_node_link(json).map_err(py_err)?;
        let names = default_names(dag.node_count());
        Ok(Self { dag, names })
    }

    fn to_networkx_node_link(&self) -> PyResult<String> {
        facade_dag_to_networkx_node_link(&self.dag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_networkx_adjacency(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let dag = facade_dag_from_networkx_adjacency(json).map_err(py_err)?;
        let names = default_names(dag.node_count());
        Ok(Self { dag, names })
    }

    fn to_networkx_adjacency(&self) -> PyResult<String> {
        facade_dag_to_networkx_adjacency(&self.dag, Some(self.names.as_slice())).map_err(py_err)
    }

    /// Whether ``x`` is d-separated from ``y`` given ``z``.
    #[pyo3(signature = (x, y, z=None))]
    fn d_separated(&self, x: &str, y: &str, z: Option<Vec<String>>) -> PyResult<bool> {
        use antecedent_graph::DSeparationWorkspace;
        let xid = self.name_index(x)?;
        let yid = self.name_index(y)?;
        let mut zids = Vec::new();
        if let Some(names) = z {
            for n in names {
                zids.push(self.name_index(&n)?);
            }
        }
        let mut ws = DSeparationWorkspace::default();
        self.dag.is_d_separated(xid, yid, &zids, &mut ws).map_err(py_err)
    }

    /// Latent-project onto ``observed`` variable names; returns an ``Admg``.
    fn latent_project(&self, observed: Vec<String>) -> PyResult<Admg> {
        use antecedent::graph::latent_project;
        let mut obs = Vec::with_capacity(observed.len());
        let mut obs_names = Vec::with_capacity(observed.len());
        for n in &observed {
            obs.push(self.name_index(n)?);
            obs_names.push(n.clone());
        }
        let admg = latent_project(&self.dag, &obs).map_err(py_err)?;
        Ok(Admg { admg, names: obs_names })
    }

    fn __repr__(&self) -> String {
        format!("Dag(nodes={}, edges={})", self.dag.node_count(), self.dag.edges().count())
    }
}

/// Named static CPDAG.
#[pyclass(name = "Cpdag", from_py_object)]
#[derive(Clone)]
pub struct Cpdag {
    pub(crate) cpdag: antecedent_graph::Cpdag,
    pub(crate) names: Vec<String>,
}

impl Cpdag {
    pub(crate) fn name_index(&self, name: &str) -> PyResult<DenseNodeId> {
        resolve_name_index(&self.names, name)
    }
}

#[pymethods]
impl Cpdag {
    #[classmethod]
    #[pyo3(signature = (names, directed, undirected=None))]
    fn from_directed_undirected(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        directed: Vec<(String, String)>,
        undirected: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let n = u32::try_from(names.len()).map_err(|_| PyValueError::new_err("too many nodes"))?;
        let mut g = antecedent_graph::Cpdag::with_variables(n);
        for (from, to) in &directed {
            let f = resolve_name_index(&names, from)?;
            let t = resolve_name_index(&names, to)?;
            g.insert_directed(f, t).map_err(py_err)?;
        }
        if let Some(undirected) = undirected {
            for (a, b) in &undirected {
                let ia = resolve_name_index(&names, a)?;
                let ib = resolve_name_index(&names, b)?;
                g.insert_undirected(ia, ib).map_err(py_err)?;
            }
        }
        Ok(Self { cpdag: g, names })
    }

    #[classmethod]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        edges: Vec<(String, String, String)>,
    ) -> PyResult<Self> {
        let n = u32::try_from(names.len()).map_err(|_| PyValueError::new_err("too many nodes"))?;
        let mut g = antecedent_graph::Cpdag::with_variables(n);
        for (from, to, kind) in &edges {
            let f = resolve_name_index(&names, from)?;
            let t = resolve_name_index(&names, to)?;
            match kind.to_ascii_lowercase().as_str() {
                "directed" => g.insert_directed(f, t).map_err(py_err)?,
                "undirected" => g.insert_undirected(f, t).map_err(py_err)?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown edge kind '{other}' (expected directed|undirected)"
                    )));
                }
            }
        }
        Ok(Self { cpdag: g, names })
    }

    fn nodes(&self) -> Vec<String> {
        self.names.clone()
    }

    fn edges(&self) -> PyResult<Vec<(String, String, String)>> {
        let mut out = Vec::new();
        for e in self.cpdag.edges() {
            let mark = cpdag_mark_str(e);
            if let Some((from, to)) = e.parent_child() {
                out.push((
                    id_name(&self.names, from)?,
                    id_name(&self.names, to)?,
                    mark.to_string(),
                ));
            } else {
                out.push((
                    id_name(&self.names, e.a)?,
                    id_name(&self.names, e.b)?,
                    mark.to_string(),
                ));
            }
        }
        Ok(out)
    }

    fn parents(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.cpdag.parents(id).into_iter().map(|p| id_name(&self.names, p)).collect()
    }

    fn children(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.cpdag.children(id).into_iter().map(|c| id_name(&self.names, c)).collect()
    }

    fn undirected_neighbors(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.cpdag.undirected_neighbors(id).into_iter().map(|n| id_name(&self.names, n)).collect()
    }

    fn try_into_dag(&self) -> PyResult<Dag> {
        let dag = self.cpdag.try_into_dag().map_err(py_err)?;
        Dag::from_rust(dag, self.names.clone())
    }

    fn node_count(&self) -> usize {
        self.cpdag.node_count()
    }

    #[classmethod]
    fn from_dot(_cls: &Bound<'_, PyType>, dot: &str) -> PyResult<Self> {
        let g = facade_cpdag_from_dot(dot).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { cpdag: g, names })
    }

    fn to_dot(&self) -> PyResult<String> {
        facade_cpdag_to_dot(&self.cpdag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_cpdag_from_json(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { cpdag: g, names })
    }

    fn to_json(&self) -> PyResult<String> {
        facade_cpdag_to_json(&self.cpdag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_gml(_cls: &Bound<'_, PyType>, gml: &str) -> PyResult<Self> {
        let g = facade_cpdag_from_gml(gml).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { cpdag: g, names })
    }

    fn to_gml(&self) -> PyResult<String> {
        facade_cpdag_to_gml(&self.cpdag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_networkx_node_link(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_cpdag_from_networkx_node_link(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { cpdag: g, names })
    }

    fn to_networkx_node_link(&self) -> PyResult<String> {
        facade_cpdag_to_networkx_node_link(&self.cpdag, Some(self.names.as_slice())).map_err(py_err)
    }

    fn __repr__(&self) -> String {
        format!("Cpdag(nodes={}, edges={})", self.cpdag.node_count(), self.cpdag.edges().len())
    }
}

/// Named static PAG.
#[pyclass(name = "Pag", from_py_object)]
#[derive(Clone)]
pub struct Pag {
    pub(crate) pag: antecedent_graph::Pag,
    pub(crate) names: Vec<String>,
}

impl Pag {
    pub(crate) fn name_index(&self, name: &str) -> PyResult<DenseNodeId> {
        resolve_name_index(&self.names, name)
    }
}

#[pymethods]
impl Pag {
    #[classmethod]
    fn from_marked_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        edges: Vec<(String, String, String, String)>,
    ) -> PyResult<Self> {
        let n = u32::try_from(names.len()).map_err(|_| PyValueError::new_err("too many nodes"))?;
        let mut g = antecedent_graph::Pag::with_variables(n);
        for (a, b, at_a, at_b) in &edges {
            let ia = resolve_name_index(&names, a)?;
            let ib = resolve_name_index(&names, b)?;
            let edge = MarkedEdge {
                a: ia,
                b: ib,
                at_a: parse_endpoint(at_a)?,
                at_b: parse_endpoint(at_b)?,
                middle: MiddleMark::Empty,
            };
            g.insert_marked(edge).map_err(py_err)?;
        }
        Ok(Self { pag: g, names })
    }

    fn nodes(&self) -> Vec<String> {
        self.names.clone()
    }

    fn neighbors(&self, name: &str) -> PyResult<Vec<(String, String, String)>> {
        let id = self.name_index(name)?;
        let mut out = Vec::new();
        for (nbr, at_self, at_nbr) in self.pag.neighbors(id) {
            out.push((
                id_name(&self.names, nbr)?,
                endpoint_str(at_self).to_string(),
                endpoint_str(at_nbr).to_string(),
            ));
        }
        Ok(out)
    }

    fn directed_children(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.pag.directed_children(id).into_iter().map(|c| id_name(&self.names, c)).collect()
    }

    fn node_count(&self) -> usize {
        self.pag.node_count()
    }

    #[classmethod]
    fn from_dot(_cls: &Bound<'_, PyType>, dot: &str) -> PyResult<Self> {
        let g = facade_pag_from_dot(dot).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { pag: g, names })
    }

    fn to_dot(&self) -> PyResult<String> {
        facade_pag_to_dot(&self.pag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_pag_from_json(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { pag: g, names })
    }

    fn to_json(&self) -> PyResult<String> {
        facade_pag_to_json(&self.pag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_gml(_cls: &Bound<'_, PyType>, gml: &str) -> PyResult<Self> {
        let g = facade_pag_from_gml(gml).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { pag: g, names })
    }

    fn to_gml(&self) -> PyResult<String> {
        facade_pag_to_gml(&self.pag, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_networkx_node_link(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_pag_from_networkx_node_link(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { pag: g, names })
    }

    fn to_networkx_node_link(&self) -> PyResult<String> {
        facade_pag_to_networkx_node_link(&self.pag, Some(self.names.as_slice())).map_err(py_err)
    }

    /// Definite-status m-separation of ``x`` and ``y`` given ``z``.
    #[pyo3(signature = (x, y, z=None, *, max_paths=32, max_len=6))]
    fn m_separated(
        &self,
        x: &str,
        y: &str,
        z: Option<Vec<String>>,
        max_paths: usize,
        max_len: usize,
    ) -> PyResult<bool> {
        let xid = self.name_index(x)?;
        let yid = self.name_index(y)?;
        let mut zids = Vec::new();
        if let Some(names) = z {
            for n in names {
                zids.push(self.name_index(&n)?);
            }
        }
        self.pag.is_m_separated(xid, yid, &zids, max_paths, max_len).map_err(py_err)
    }

    fn __repr__(&self) -> String {
        format!("Pag(nodes={})", self.pag.node_count())
    }
}

/// Named ADMG (directed + bidirected).
#[pyclass(name = "Admg", from_py_object)]
#[derive(Clone)]
pub struct Admg {
    pub(crate) admg: antecedent_graph::Admg,
    pub(crate) names: Vec<String>,
}

impl Admg {
    pub(crate) fn name_index(&self, name: &str) -> PyResult<DenseNodeId> {
        resolve_name_index(&self.names, name)
    }
}

#[pymethods]
impl Admg {
    #[classmethod]
    #[pyo3(signature = (names, directed, bidirected=None))]
    fn from_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        directed: Vec<(String, String)>,
        bidirected: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let n = u32::try_from(names.len()).map_err(|_| PyValueError::new_err("too many nodes"))?;
        let mut g = antecedent_graph::Admg::with_variables(n);
        for (from, to) in &directed {
            let f = resolve_name_index(&names, from)?;
            let t = resolve_name_index(&names, to)?;
            g.insert_directed(f, t).map_err(py_err)?;
        }
        if let Some(bidirected) = bidirected {
            for (a, b) in &bidirected {
                let ia = resolve_name_index(&names, a)?;
                let ib = resolve_name_index(&names, b)?;
                g.insert_bidirected(ia, ib).map_err(py_err)?;
            }
        }
        Ok(Self { admg: g, names })
    }

    fn nodes(&self) -> Vec<String> {
        self.names.clone()
    }

    fn parents(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.admg.parents(id).iter().map(|&p| id_name(&self.names, p)).collect()
    }

    fn children(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.admg.children(id).iter().map(|&c| id_name(&self.names, c)).collect()
    }

    fn bidirected_neighbors(&self, name: &str) -> PyResult<Vec<String>> {
        let id = self.name_index(name)?;
        self.admg.bidirected_neighbors(id).iter().map(|&n| id_name(&self.names, n)).collect()
    }

    fn node_count(&self) -> usize {
        self.admg.node_count()
    }

    #[classmethod]
    fn from_dot(_cls: &Bound<'_, PyType>, dot: &str) -> PyResult<Self> {
        let g = facade_admg_from_dot(dot).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { admg: g, names })
    }

    fn to_dot(&self) -> PyResult<String> {
        facade_admg_to_dot(&self.admg, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_admg_from_json(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { admg: g, names })
    }

    fn to_json(&self) -> PyResult<String> {
        facade_admg_to_json(&self.admg, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_gml(_cls: &Bound<'_, PyType>, gml: &str) -> PyResult<Self> {
        let g = facade_admg_from_gml(gml).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { admg: g, names })
    }

    fn to_gml(&self) -> PyResult<String> {
        facade_admg_to_gml(&self.admg, Some(self.names.as_slice())).map_err(py_err)
    }

    #[classmethod]
    fn from_networkx_node_link(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        let g = facade_admg_from_networkx_node_link(json).map_err(py_err)?;
        let names = default_names(g.node_count());
        Ok(Self { admg: g, names })
    }

    fn to_networkx_node_link(&self) -> PyResult<String> {
        facade_admg_to_networkx_node_link(&self.admg, Some(self.names.as_slice())).map_err(py_err)
    }

    /// Whether ``x`` is m-separated from ``y`` given ``z``.
    #[pyo3(signature = (x, y, z=None))]
    fn m_separated(&self, x: &str, y: &str, z: Option<Vec<String>>) -> PyResult<bool> {
        use antecedent_graph::DSeparationWorkspace;
        let xid = self.name_index(x)?;
        let yid = self.name_index(y)?;
        let mut zids = Vec::new();
        if let Some(names) = z {
            for n in names {
                zids.push(self.name_index(&n)?);
            }
        }
        let mut ws = DSeparationWorkspace::default();
        self.admg.is_m_separated(xid, yid, &zids, &mut ws).map_err(py_err)
    }

    fn __repr__(&self) -> String {
        format!("Admg(nodes={})", self.admg.node_count())
    }
}

/// Named temporal DAG over lagged variables.
#[pyclass(name = "TemporalDag", from_py_object)]
#[derive(Clone)]
pub struct TemporalDag {
    pub(crate) dag: RustTemporalDag,
    /// Base variable names (index = [`VariableId`] raw).
    pub(crate) names: Vec<String>,
}

impl TemporalDag {
    #[allow(dead_code)] // reserved for later OO analyze / discovery wiring
    pub(crate) fn var_index(&self, name: &str) -> PyResult<VariableId> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| VariableId::from_raw(u32::try_from(i).expect("var index fits u32")))
            .ok_or_else(|| unknown_node(name))
    }

    fn node_label(&self, id: DenseNodeId) -> PyResult<(String, u32)> {
        match self.dag.nodes().get(id.as_usize()) {
            Some(NodeRef::Lagged { variable, lag }) => {
                let name = self.names.get(variable.as_usize()).cloned().ok_or_else(|| {
                    CausalGraphError::new_err(format!(
                        "variable id {} out of range",
                        variable.raw()
                    ))
                })?;
                Ok((name, lag.raw()))
            }
            _ => {
                Err(CausalGraphError::new_err(format!("temporal node {} is not lagged", id.raw())))
            }
        }
    }
}

#[pymethods]
impl TemporalDag {
    #[classmethod]
    fn from_lagged_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        edges: Vec<(String, u32, String, u32)>,
    ) -> PyResult<Self> {
        let mut g = RustTemporalDag::empty();
        let var_of = |nm: &str| -> PyResult<VariableId> {
            names
                .iter()
                .position(|n| n == nm)
                .map(|i| VariableId::from_raw(u32::try_from(i).expect("var index fits u32")))
                .ok_or_else(|| unknown_node(nm))
        };
        for (src, slag, tgt, tlag) in &edges {
            let s = ensure_lagged(&mut g, var_of(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
            let t = ensure_lagged(&mut g, var_of(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
            g.insert_directed(s, t).map_err(py_err)?;
        }
        Ok(Self { dag: g, names })
    }

    /// Lagged nodes as `(name, lag)` pairs.
    fn nodes(&self) -> PyResult<Vec<(String, u32)>> {
        let mut out = Vec::with_capacity(self.dag.node_count());
        for i in 0..self.dag.node_count() {
            out.push(self.node_label(DenseNodeId::from_raw(u32::try_from(i).expect("fit")))?);
        }
        Ok(out)
    }

    /// Directed edges as `(src, src_lag, tgt, tgt_lag)`.
    fn edges(&self) -> PyResult<Vec<(String, u32, String, u32)>> {
        let mut out = Vec::new();
        for e in self.dag.edges() {
            let Some((from, to)) = e.parent_child() else {
                continue;
            };
            let (sn, sl) = self.node_label(from)?;
            let (tn, tl) = self.node_label(to)?;
            out.push((sn, sl, tn, tl));
        }
        Ok(out)
    }

    fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    fn __repr__(&self) -> String {
        format!("TemporalDag(variables={}, nodes={})", self.names.len(), self.dag.node_count())
    }
}

/// Named temporal CPDAG over lagged variables.
#[pyclass(name = "TemporalCpdag", from_py_object)]
#[derive(Clone)]
pub struct TemporalCpdag {
    pub(crate) cpdag: antecedent_graph::TemporalCpdag,
    pub(crate) names: Vec<String>,
}

#[pymethods]
impl TemporalCpdag {
    #[classmethod]
    fn from_lagged_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        directed: Vec<(String, u32, String, u32)>,
        undirected: Option<Vec<(String, u32, String, u32)>>,
    ) -> PyResult<Self> {
        let mut g = antecedent_graph::TemporalCpdag::empty();
        let var_of = |nm: &str| -> PyResult<VariableId> {
            names
                .iter()
                .position(|n| n == nm)
                .map(|i| VariableId::from_raw(u32::try_from(i).expect("var index fits u32")))
                .ok_or_else(|| unknown_node(nm))
        };
        for (src, slag, tgt, tlag) in &directed {
            let s = ensure_lagged_cpdag(&mut g, var_of(src)?, Lag::from_raw(*slag))?;
            let t = ensure_lagged_cpdag(&mut g, var_of(tgt)?, Lag::from_raw(*tlag))?;
            g.insert_directed(s, t).map_err(py_err)?;
        }
        if let Some(undirected) = undirected {
            for (a, al, b, bl) in &undirected {
                let ia = ensure_lagged_cpdag(&mut g, var_of(a)?, Lag::from_raw(*al))?;
                let ib = ensure_lagged_cpdag(&mut g, var_of(b)?, Lag::from_raw(*bl))?;
                g.insert_undirected(ia, ib).map_err(py_err)?;
            }
        }
        Ok(Self { cpdag: g, names })
    }

    fn try_into_temporal_dag(&self) -> PyResult<TemporalDag> {
        let dag = self.cpdag.try_into_temporal_dag().map_err(py_err)?;
        Ok(TemporalDag { dag, names: self.names.clone() })
    }

    fn node_count(&self) -> usize {
        self.cpdag.node_count()
    }

    fn __repr__(&self) -> String {
        format!("TemporalCpdag(variables={}, nodes={})", self.names.len(), self.cpdag.node_count())
    }
}

/// Named temporal PAG over lagged variables.
#[pyclass(name = "TemporalPag", from_py_object)]
#[derive(Clone)]
pub struct TemporalPag {
    pub(crate) pag: antecedent_graph::TemporalPag,
    pub(crate) names: Vec<String>,
}

#[pymethods]
impl TemporalPag {
    #[classmethod]
    fn from_marked_lagged_edges(
        _cls: &Bound<'_, PyType>,
        names: Vec<String>,
        edges: Vec<(String, u32, String, u32, String, String)>,
    ) -> PyResult<Self> {
        let mut g = antecedent_graph::TemporalPag::empty();
        let var_of = |nm: &str| -> PyResult<VariableId> {
            names
                .iter()
                .position(|n| n == nm)
                .map(|i| VariableId::from_raw(u32::try_from(i).expect("var index fits u32")))
                .ok_or_else(|| unknown_node(nm))
        };
        for (a, al, b, bl, at_a, at_b) in &edges {
            let ia = ensure_lagged_pag(&mut g, var_of(a)?, Lag::from_raw(*al))?;
            let ib = ensure_lagged_pag(&mut g, var_of(b)?, Lag::from_raw(*bl))?;
            let edge = MarkedEdge {
                a: ia,
                b: ib,
                at_a: parse_endpoint(at_a)?,
                at_b: parse_endpoint(at_b)?,
                middle: MiddleMark::Empty,
            };
            g.insert_marked(edge).map_err(py_err)?;
        }
        Ok(Self { pag: g, names })
    }

    fn node_count(&self) -> usize {
        self.pag.node_count()
    }

    fn __repr__(&self) -> String {
        format!("TemporalPag(variables={}, nodes={})", self.names.len(), self.pag.node_count())
    }
}

fn ensure_lagged_cpdag(
    g: &mut antecedent_graph::TemporalCpdag,
    variable: VariableId,
    lag: Lag,
) -> PyResult<DenseNodeId> {
    // Reuse existing node if present.
    for (i, node) in g.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable: v, lag: l } = node {
            if *v == variable && *l == lag {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    g.add_lagged(variable, lag).map_err(py_err)
}

fn ensure_lagged_pag(
    g: &mut antecedent_graph::TemporalPag,
    variable: VariableId,
    lag: Lag,
) -> PyResult<DenseNodeId> {
    for (i, node) in g.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable: v, lag: l } = node {
            if *v == variable && *l == lag {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    g.add_lagged(variable, lag).map_err(py_err)
}
