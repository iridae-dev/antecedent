//! Markov-equivalence-class membership of a DAG in a CPDAG.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{is_mec_member, Dag as RustDag};
use pyo3::prelude::*;

use crate::graphs::{Cpdag, Dag};
use crate::{py_err, value_err};

#[pymethods]
impl Cpdag {
    /// Whether `dag` is a member of this CPDAG's Markov equivalence class: same skeleton,
    /// same unshielded colliders and every compelled directed edge present.
    ///
    /// `dag` must be over exactly the same variable names; nodes are matched by name, so the
    /// order the two graphs list them in does not matter.
    fn is_mec_member(&self, dag: &Dag) -> PyResult<bool> {
        let mut own: Vec<&String> = self.names.iter().collect();
        let mut theirs: Vec<&String> = dag.names.iter().collect();
        own.sort();
        theirs.sort();
        if own != theirs {
            return Err(value_err(
                "the DAG must be over exactly the CPDAG's variables to be tested for membership",
            ));
        }
        let mut aligned = RustDag::with_variables(
            u32::try_from(self.names.len()).map_err(|_| value_err("too many nodes"))?,
        );
        for edge in dag.dag.edges() {
            let Some((from, to)) = edge.parent_child() else { continue };
            let from = self.name_index(&dag.names[from.as_usize()])?;
            let to = self.name_index(&dag.names[to.as_usize()])?;
            aligned.insert_directed(from, to).map_err(py_err)?;
        }
        Ok(is_mec_member(&self.cpdag, &aligned))
    }
}
