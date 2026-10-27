//! The semantic coordinate of one joint event over several schema variables.
//!
//! A [`ScientificQuantity`] names exactly one variable, but the probability or density of
//! an event over several outcomes is not a property of any one of them. A
//! [`JointEventQuantity`] names an ordered set of distinct schema variables and the kind of
//! event functional, and lowers to a `ScientificQuantity` whose identity is distinct from
//! every single-variable coordinate: its functional identity is `joint_probability` or
//! `joint_density`, and its variable identity frames the ordered variable names. The
//! lowered coordinate therefore flows unchanged through every encoding that carries a
//! `ScientificQuantity`, and single-variable encodings are byte-identical to before.

use crate::quantity::{QuantityCondition, QuantityMismatch, QuantityRole, ScientificQuantity};

/// The functional a joint event reports.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum JointEventKind {
    /// Probability of a discrete joint event; dimensionless.
    Probability,
    /// Joint density at a point; carries the inverse product of the variables' units.
    Density,
}

impl JointEventKind {
    /// Stable functional identity of the lowered coordinate.
    #[must_use]
    pub const fn functional_id(self) -> &'static str {
        match self {
            Self::Probability => "joint_probability",
            Self::Density => "joint_density",
        }
    }

    fn from_functional_id(id: &str) -> Option<Self> {
        match id {
            "joint_probability" => Some(Self::Probability),
            "joint_density" => Some(Self::Density),
            _ => None,
        }
    }
}

/// One schema variable of a joint event, with its declared units.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JointEventVariable {
    /// Schema-local variable name.
    pub name: String,
    /// Declared units of the variable.
    pub units: String,
}

/// The full semantic coordinate of the probability or density of a joint event.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JointEventQuantity {
    /// Stable schema identity qualifying every variable.
    pub schema_id: String,
    /// The ordered, distinct variables of the event.
    pub variables: Vec<JointEventVariable>,
    /// Probability or density.
    pub kind: JointEventKind,
    /// Stable population identity.
    pub population_id: String,
    /// Exact intervention or regime identity.
    pub regime_id: String,
    /// Number of ordered causal steps; zero is static.
    pub horizon: u32,
    /// Explicit conditioning coordinates in canonical order.
    pub conditioning: Vec<QuantityCondition>,
    /// Declared scale and transform.
    pub transform_id: String,
}

impl JointEventQuantity {
    /// Unit string of the lowered coordinate.
    fn units(&self) -> String {
        match self.kind {
            JointEventKind::Probability => "dimensionless".into(),
            JointEventKind::Density => {
                let product: Vec<String> =
                    self.variables.iter().map(|v| format!("({})", v.units)).collect();
                format!("1/({})", product.join("*"))
            }
        }
    }

    /// Length-framed, order-sensitive variable identity.
    fn variable_id(&self) -> String {
        let mut id = format!("{}:joint({}", self.schema_id, self.variables.len());
        for variable in &self.variables {
            id.push_str(&format!(";{}:{}", variable.name.len(), variable.name));
        }
        id.push(')');
        id
    }

    /// Validate nonempty and distinct variables with declared names and units.
    ///
    /// # Errors
    /// An empty variable list, a blank or duplicate variable, or a blank identity.
    pub fn validate(&self) -> Result<(), QuantityMismatch> {
        if self.schema_id.trim().is_empty()
            || self.variables.is_empty()
            || self.variables.iter().any(|v| v.name.trim().is_empty() || v.units.trim().is_empty())
            || self
                .variables
                .iter()
                .enumerate()
                .any(|(i, v)| self.variables[..i].iter().any(|old| old.name == v.name))
        {
            return Err(QuantityMismatch::InvalidCoordinate);
        }
        self.to_coordinate().map(|_| ())
    }

    /// Validate, and require every variable to be declared in `schema` with the same units.
    ///
    /// # Errors
    /// Any [`Self::validate`] failure, or a variable absent from the schema.
    pub fn validate_in_schema(&self, schema: &[(String, String)]) -> Result<(), QuantityMismatch> {
        self.validate()?;
        let declared = |v: &JointEventVariable| {
            schema.iter().any(|(name, units)| *name == v.name && *units == v.units)
        };
        if self.variables.iter().all(declared) { Ok(()) } else { Err(QuantityMismatch::Variable) }
    }

    /// Lower to the single-coordinate form every contract encoding carries.
    ///
    /// # Errors
    /// An invalid joint event, or an invalid lowered coordinate.
    pub fn to_coordinate(&self) -> Result<ScientificQuantity, QuantityMismatch> {
        if self.schema_id.trim().is_empty() || self.variables.is_empty() {
            return Err(QuantityMismatch::InvalidCoordinate);
        }
        let names: Vec<&str> = self.variables.iter().map(|v| v.name.as_str()).collect();
        let quantity = ScientificQuantity {
            variable_id: self.variable_id(),
            variable_name: format!("({})", names.join(", ")),
            role: QuantityRole::Outcome,
            units: self.units(),
            population_id: self.population_id.clone(),
            regime_id: self.regime_id.clone(),
            horizon: self.horizon,
            functional_id: self.kind.functional_id().into(),
            conditioning: self.conditioning.clone(),
            transform_id: self.transform_id.clone(),
        };
        quantity.validate()?;
        Ok(quantity)
    }

    /// Recover the joint event a lowered coordinate encodes, resolving units in `schema`.
    ///
    /// Returns `None` for any coordinate that is not a joint event of `schema_id`, so a
    /// single-variable coordinate is never read as one.
    #[must_use]
    pub fn from_coordinate(
        quantity: &ScientificQuantity,
        schema_id: &str,
        schema: &[(String, String)],
    ) -> Option<Self> {
        let kind = JointEventKind::from_functional_id(&quantity.functional_id)?;
        let body = quantity.variable_id.strip_prefix(schema_id)?.strip_prefix(":joint(")?;
        let body = body.strip_suffix(')')?;
        let (count, mut rest) = body.split_once(';').map_or((body, ""), |(c, r)| (c, r));
        let count: usize = count.parse().ok()?;
        let mut variables = Vec::with_capacity(count);
        while !rest.is_empty() {
            let (len, tail) = rest.split_once(':')?;
            let len: usize = len.parse().ok()?;
            let name = tail.get(..len)?;
            let units = &schema.iter().find(|(declared, _)| declared == name)?.1;
            variables.push(JointEventVariable { name: name.into(), units: units.clone() });
            rest = tail.get(len..)?;
            rest = rest.strip_prefix(';').unwrap_or(rest);
        }
        if variables.len() != count {
            return None;
        }
        let event = Self {
            schema_id: schema_id.into(),
            variables,
            kind,
            population_id: quantity.population_id.clone(),
            regime_id: quantity.regime_id.clone(),
            horizon: quantity.horizon,
            conditioning: quantity.conditioning.clone(),
            transform_id: quantity.transform_id.clone(),
        };
        let lowered = event.to_coordinate().ok()?;
        (lowered.variable_id == quantity.variable_id && lowered.units == quantity.units)
            .then_some(event)
    }

    /// Write every identity field into `stream`, tagged so it never collides with a
    /// single-variable coordinate.
    pub fn write_canonical(&self, stream: &mut antecedent_core::identity::CanonicalWriter) {
        stream.text("joint_event").text(&self.schema_id).word(self.variables.len() as u64);
        for variable in &self.variables {
            stream.text(&variable.name).text(&variable.units);
        }
        stream
            .text(self.kind.functional_id())
            .text(&self.population_id)
            .text(&self.regime_id)
            .text(&self.transform_id)
            .word(u64::from(self.horizon))
            .word(self.conditioning.len() as u64);
        for condition in &self.conditioning {
            stream.text(&condition.variable_id).text(&condition.value_id);
        }
    }
}

impl std::fmt::Display for JointEventQuantity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.variables.iter().map(|v| v.name.as_str()).collect();
        write!(f, "{}({}:{})", self.kind.functional_id(), self.schema_id, names.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Vec<(String, String)> {
        vec![("a".into(), "kg".into()), ("b".into(), "m".into()), ("c".into(), "s".into())]
    }

    fn event(kind: JointEventKind, names: &[&str]) -> JointEventQuantity {
        let schema = schema();
        JointEventQuantity {
            schema_id: "sch".into(),
            variables: names
                .iter()
                .map(|n| {
                    let units = schema.iter().find(|(name, _)| name == n).unwrap().1.clone();
                    JointEventVariable { name: (*n).into(), units }
                })
                .collect(),
            kind,
            population_id: "pop".into(),
            regime_id: "obs".into(),
            horizon: 0,
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    #[test]
    fn lowering_round_trips_and_is_distinct_from_single_variables() {
        let joint = event(JointEventKind::Density, &["b", "a"]);
        joint.validate_in_schema(&schema()).unwrap();
        let coordinate = joint.to_coordinate().unwrap();
        assert_eq!(coordinate.functional_id, "joint_density");
        assert_eq!(coordinate.units, "1/((m)*(kg))");
        assert_eq!(JointEventQuantity::from_coordinate(&coordinate, "sch", &schema()), Some(joint));
        let single = event(JointEventKind::Probability, &["a"]).to_coordinate().unwrap();
        assert_ne!(single.variable_id, "sch:a");
        assert_eq!(single.units, "dimensionless");
    }

    #[test]
    fn order_changes_identity() {
        let ab = event(JointEventKind::Probability, &["a", "b"]).to_coordinate().unwrap();
        let ba = event(JointEventKind::Probability, &["b", "a"]).to_coordinate().unwrap();
        assert_ne!(ab.canonical_identity(), ba.canonical_identity());
        assert!(ab.require_same_coordinate(&ba).is_err());
    }

    #[test]
    fn single_variable_coordinates_are_not_joint_events() {
        let plain = ScientificQuantity {
            variable_id: "sch:a".into(),
            variable_name: "a".into(),
            role: QuantityRole::Outcome,
            units: "kg".into(),
            population_id: "pop".into(),
            regime_id: "obs".into(),
            horizon: 0,
            functional_id: "identified_probability".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        };
        assert_eq!(JointEventQuantity::from_coordinate(&plain, "sch", &schema()), None);
    }

    #[test]
    fn empty_duplicate_and_unknown_variables_are_refused() {
        let mut joint = event(JointEventKind::Probability, &["a", "b"]);
        joint.variables.clear();
        assert_eq!(joint.validate(), Err(QuantityMismatch::InvalidCoordinate));
        let mut joint = event(JointEventKind::Probability, &["a", "b"]);
        joint.variables[1] = joint.variables[0].clone();
        assert_eq!(joint.validate(), Err(QuantityMismatch::InvalidCoordinate));
        let mut joint = event(JointEventKind::Probability, &["a", "b"]);
        joint.variables[1].name = "zzz".into();
        assert_eq!(joint.validate_in_schema(&schema()), Err(QuantityMismatch::Variable));
        let mut joint = event(JointEventKind::Probability, &["a", "b"]);
        joint.variables[1].units = "wrong".into();
        assert_eq!(joint.validate_in_schema(&schema()), Err(QuantityMismatch::Variable));
    }

    #[test]
    fn display_and_canonical_digest_are_stable_and_ordered() {
        let joint = event(JointEventKind::Probability, &["a", "c"]);
        assert_eq!(joint.to_string(), "joint_probability(sch:a, c)");
        let digest = |q: &JointEventQuantity| {
            let mut stream = antecedent_core::identity::CanonicalWriter::new("test.joint");
            q.write_canonical(&mut stream);
            stream.finish_hex()
        };
        assert_eq!(digest(&joint), digest(&joint.clone()));
        assert_ne!(digest(&joint), digest(&event(JointEventKind::Probability, &["c", "a"])));
        assert_ne!(digest(&joint), digest(&event(JointEventKind::Density, &["a", "c"])));
    }
}
