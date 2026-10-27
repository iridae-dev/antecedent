//! Borrowed encoding of the canonical expression wire, without owned DTO tables.
#![allow(
    clippy::cast_possible_truncation,
    reason = "constructor validates every table index before iteration"
)]
use super::*;
use serde::ser::{SerializeSeq, SerializeStruct};

/// A borrowed canonical expression-arena wire view.
///
/// This uses the same field order, tags and omission rules as [`ExprArenaWire`].
/// Construction validates the existing wire index limits without allocating.
/// Serialization traverses borrowed tables and strings; no copied arena or
/// intermediate JSON value tree is constructed.
pub struct BorrowedExprArenaWire<'a>(&'a CausalExprArena);

impl<'a> BorrowedExprArenaWire<'a> {
    /// Borrow an arena under the existing `u32` wire-coordinate bounds.
    /// # Errors
    /// An arena table length cannot be represented on the canonical wire.
    pub fn new(arena: &'a CausalExprArena) -> Result<Self, IoError> {
        for count in
            [arena.len(), arena.var_set_count(), arena.intervention_set_count(), arena.list_count()]
        {
            if count > 0 {
                u32::try_from(count - 1).map_err(|_| IoError::TooLarge)?;
            }
        }
        Ok(Self(arena))
    }
}

pub(crate) struct Sequence<F>(pub(crate) F);
impl<F, I> Serialize for Sequence<F>
where
    F: Fn() -> I,
    I: IntoIterator,
    I::Item: Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((self.0)())
    }
}

impl Serialize for BorrowedExprArenaWire<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let arena = self.0;
        let derivation_count = (0..arena.len())
            .filter(|&index| arena.derivation(ExprId::from_raw(index as u32)).is_some())
            .count();
        let mut wire =
            serializer.serialize_struct("ExprArenaWire", 4 + usize::from(derivation_count > 0))?;
        if derivation_count > 0 {
            wire.serialize_field("derivations", &Derivations(arena, derivation_count))?;
        }
        wire.serialize_field(
            "var_sets",
            &Sequence(|| {
                (0..arena.var_set_count()).map(|index| {
                    Sequence(move || {
                        arena.var_set(VarSetId::from_raw(index as u32)).iter().map(|v| v.raw())
                    })
                })
            }),
        )?;
        wire.serialize_field(
            "interventions",
            &Sequence(|| {
                (0..arena.intervention_set_count()).map(|index| {
                    Sequence(move || {
                        arena
                            .intervention_assignments(InterventionSetId::from_raw(index as u32))
                            .iter()
                            .map(BorrowedAssignment)
                    })
                })
            }),
        )?;
        wire.serialize_field(
            "lists",
            &Sequence(|| {
                (0..arena.list_count()).map(|index| {
                    Sequence(move || {
                        arena.list(ExprListId::from_raw(index as u32)).iter().map(|v| v.raw())
                    })
                })
            }),
        )?;
        wire.serialize_field(
            "nodes",
            &Sequence(|| {
                (0..arena.len()).map(|index| {
                    BorrowedNode::new(arena.node(ExprId::from_raw(index as u32)), arena)
                })
            }),
        )?;
        wire.end()
    }
}

struct Derivations<'a>(&'a CausalExprArena, usize);
impl Serialize for Derivations<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.1))?;
        for index in 0..self.0.len() {
            let id = ExprId::from_raw(index as u32);
            if let Some(d) = self.0.derivation(id) {
                sequence.serialize_element(&(id.raw(), BorrowedDerivation(d)))?;
            }
        }
        sequence.end()
    }
}

struct BorrowedDerivation<'a>(&'a antecedent_expr::DerivationMeta);
impl Serialize for BorrowedDerivation<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let d = self.0;
        let mut wire = serializer.serialize_struct("DerivationWire", 8)?;
        wire.serialize_field("rule", d.rule.as_ref())?;
        wire.serialize_field("note", &d.note.as_deref())?;
        wire.serialize_field("input", &d.input.map(ExprId::raw))?;
        wire.serialize_field("output", &d.output.map(ExprId::raw))?;
        wire.serialize_field("graph_operation", &d.graph_operation.as_deref())?;
        wire.serialize_field(
            "premises",
            &Sequence(|| d.premises.iter().map(AsRef::<str>::as_ref)),
        )?;
        wire.serialize_field("evidence", &Sequence(|| d.evidence.iter().map(|v| v.raw())))?;
        wire.serialize_field("parents", &Sequence(|| d.parents.iter().map(|v| v.raw())))?;
        wire.end()
    }
}

// Mirror the canonical scalar tags, borrowing only the one variable-length case.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum BorrowedValue<'a> {
    Float64(f64),
    Int64(i64),
    Bool(bool),
    Category(u32),
    Label(&'a str),
}
impl<'a> From<&'a Value> for BorrowedValue<'a> {
    fn from(value: &'a Value) -> Self {
        match value {
            Value::Float64(x) => Self::Float64(*x),
            Value::Int64(x) => Self::Int64(*x),
            Value::Bool(x) => Self::Bool(*x),
            Value::Category(x) => Self::Category(*x),
            Value::Label(x) => Self::Label(x.as_ref()),
        }
    }
}
struct BorrowedAssignment<'a>(&'a InterventionAssignment);
impl Serialize for BorrowedAssignment<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let symbolic = self.0.is_symbolic();
        let mut wire =
            serializer.serialize_struct("InterventionAssignmentWire", 2 + usize::from(symbolic))?;
        if symbolic {
            wire.serialize_field("symbolic", &true)?;
        }
        wire.serialize_field("variable", &self.0.variable.raw())?;
        wire.serialize_field(
            "value",
            &if symbolic {
                BorrowedValue::Float64(0.0)
            } else {
                BorrowedValue::from(&self.0.value)
            },
        )?;
        wire.end()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum BorrowedNode<'a> {
    Distribution {
        variables: u32,
        conditioned_on: u32,
        intervention: u32,
        domain: &'static str,
        #[serde(skip_serializing_if = "str::is_empty")]
        population: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        regime: Option<u32>,
    },
    Kernel {
        body: u32,
        bound: u32,
        #[serde(skip_serializing_if = "str::is_empty")]
        population: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        regime: Option<u32>,
    },
    Product(u32),
    SumOut {
        variables: u32,
        expr: u32,
    },
    IntegralOut {
        variables: u32,
        expr: u32,
    },
    Ratio {
        numerator: u32,
        denominator: u32,
    },
    Expectation {
        function: u32,
        distribution: u32,
    },
    Contrast {
        left: u32,
        right: u32,
        op: &'static str,
    },
}
impl<'a> BorrowedNode<'a> {
    fn new(node: &'a ExprNode, arena: &'a CausalExprArena) -> Self {
        match node {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => Self::Distribution {
                variables: variables.raw(),
                conditioned_on: conditioned_on.raw(),
                intervention: intervention.raw(),
                domain: match domain {
                    DomainRef::Observational => "observational",
                    DomainRef::Interventional => "interventional",
                },
                population: arena.population(*population),
                regime: regime.map(RegimeId::raw),
            },
            ExprNode::Kernel { body, bound, population, regime } => Self::Kernel {
                body: body.raw(),
                bound: bound.raw(),
                population: arena.population(*population),
                regime: regime.map(RegimeId::raw),
            },
            ExprNode::Product(list) => Self::Product(list.raw()),
            ExprNode::SumOut { variables, expr } => {
                Self::SumOut { variables: variables.raw(), expr: expr.raw() }
            }
            ExprNode::IntegralOut { variables, expr } => {
                Self::IntegralOut { variables: variables.raw(), expr: expr.raw() }
            }
            ExprNode::Ratio { numerator, denominator } => {
                Self::Ratio { numerator: numerator.raw(), denominator: denominator.raw() }
            }
            ExprNode::Expectation { function, distribution } => Self::Expectation {
                function: function.variable().raw(),
                distribution: distribution.raw(),
            },
            ExprNode::Contrast { left, right, op } => Self::Contrast {
                left: left.raw(),
                right: right.raw(),
                op: match op {
                    ContrastOp::Difference => "difference",
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_bytes(arena: &CausalExprArena) {
        let original = expr_arena_to_wire(arena).unwrap();
        let borrowed = BorrowedExprArenaWire::new(arena).unwrap();
        assert_eq!(serde_json::to_vec(&borrowed).unwrap(), serde_json::to_vec(&original).unwrap());
        let mut old_cbor = Vec::new();
        let mut new_cbor = Vec::new();
        ciborium::into_writer(&original, &mut old_cbor).unwrap();
        ciborium::into_writer(&borrowed, &mut new_cbor).unwrap();
        assert_eq!(new_cbor, old_cbor, "canonical field order and definite collection lengths");
    }

    #[test]
    fn borrowed_arena_preserves_existing_json_and_cbor_tags_omissions_and_order() {
        let mut arena = CausalExprArena::new();
        assert_same_bytes(&arena);
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let variables = arena.intern_var_set([x, y]);
        let conditioned_on = arena.empty_var_set();
        let intervention = arena.intern_intervention_assignments([
            InterventionAssignment { variable: x, value: Value::symbolic_intervention() },
            InterventionAssignment { variable: y, value: Value::Label(Arc::from("escaped\\\"λ")) },
        ]);
        let leaf = arena
            .intern_distribution_tagged(
                DistributionLeaf {
                    variables,
                    conditioned_on,
                    intervention,
                    domain: DomainRef::Interventional,
                },
                "source\\\"λ",
                Some(RegimeId::from_raw(3)),
                None,
            )
            .unwrap();
        let kernel = arena.intern_kernel(leaf, variables, "target", None);
        let list = arena.intern_list([leaf, kernel]);
        let product = arena.intern(ExprNode::Product(list));
        let sum = arena.intern(ExprNode::SumOut { variables, expr: product });
        let integral = arena.intern(ExprNode::IntegralOut { variables, expr: sum });
        let ratio = arena.intern(ExprNode::Ratio { numerator: integral, denominator: leaf });
        let expectation = arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(y),
            distribution: ratio,
        });
        let contrast = arena.intern(ExprNode::Contrast {
            left: expectation,
            right: leaf,
            op: ContrastOp::Difference,
        });
        assert_same_bytes(&arena);
        arena.set_derivation(
            contrast,
            antecedent_expr::DerivationMeta {
                rule: Arc::from("rule\\\"λ"),
                note: Some(Arc::from("original note")),
                input: Some(leaf),
                output: Some(contrast),
                graph_operation: Some(Arc::from("project")),
                premises: Arc::from([Arc::from("first"), Arc::from("second")]),
                evidence: Arc::from([RegimeId::from_raw(3)]),
                parents: Arc::from([leaf, expectation]),
            },
        );
        assert_same_bytes(&arena);
    }
}
