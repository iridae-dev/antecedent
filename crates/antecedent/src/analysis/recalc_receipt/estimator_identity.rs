//! Canonical estimator identity encoding for recalculation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::AipwAte;

/// Append-only builder of an estimator's identity bytes.
///
/// Integers are little endian at their declared width, floats are bit patterns, every sum
/// type starts with a one-byte tag and every sequence with its length, so unequal
/// configurations cannot share bytes. The layout is part of the stage identities that resume
/// and reuse key on: changing it changes every retained identity.
struct IdentityBytes(Vec<u8>);

impl IdentityBytes {
    fn new(domain: &[u8]) -> Self {
        Self(domain.to_vec())
    }

    fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.0.extend_from_slice(bytes);
        self
    }

    fn tag(&mut self, tag: u8) -> &mut Self {
        self.0.push(tag);
        self
    }

    fn u32(&mut self, value: u32) -> &mut Self {
        self.raw(&value.to_le_bytes())
    }

    fn usize(&mut self, value: usize) -> &mut Self {
        self.raw(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes())
    }

    fn f64(&mut self, value: f64) -> &mut Self {
        self.raw(&value.to_bits().to_le_bytes())
    }

    fn opt_f64(&mut self, value: Option<f64>) -> &mut Self {
        match value {
            None => self.tag(0),
            Some(value) => self.tag(1).f64(value),
        }
    }

    fn f64s(&mut self, values: &[f64]) -> &mut Self {
        self.usize(values.len());
        for value in values {
            self.f64(*value);
        }
        self
    }

    fn u32s(&mut self, values: &[u32]) -> &mut Self {
        self.usize(values.len());
        for value in values {
            self.u32(*value);
        }
        self
    }

    fn opt_u32s(&mut self, values: Option<&[u32]>) -> &mut Self {
        match values {
            None => self.tag(0),
            Some(values) => self.tag(1).u32s(values),
        }
    }

    fn opt_panel(&mut self, values: Option<&[i64]>) -> &mut Self {
        match values {
            None => self.tag(0),
            Some(values) => {
                self.tag(1).usize(values.len());
                for value in values {
                    self.raw(&value.to_le_bytes());
                }
                self
            }
        }
    }

    fn write(&mut self, part: &impl CanonicalIdentity) -> &mut Self {
        part.write_identity(self);
        self
    }
}

/// A part of an estimator specification with a canonical identity encoding.
trait CanonicalIdentity {
    fn write_identity(&self, out: &mut IdentityBytes);
}

impl CanonicalIdentity for antecedent_estimate::OverlapPolicy {
    fn write_identity(&self, out: &mut IdentityBytes) {
        match self.bounds() {
            None => {
                out.tag(0);
            }
            Some((clip, trim)) => {
                out.tag(1).opt_f64(clip).opt_f64(trim);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_stats::NbAlphaPolicy {
    fn write_identity(&self, out: &mut IdentityBytes) {
        match *self {
            Self::Fixed(alpha) => {
                out.tag(0).f64(alpha);
            }
            Self::MethodOfMoments => {
                out.tag(1);
            }
            Self::NestedMle { max_outer, tol_alpha } => {
                out.tag(2).u32(max_outer).f64(tol_alpha);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_stats::GlmOptions {
    fn write_identity(&self, out: &mut IdentityBytes) {
        out.u32(self.max_iter)
            .f64(self.tol)
            .write(&self.nb_alpha)
            .opt_f64(self.ridge_on_separation);
    }
}

impl CanonicalIdentity for antecedent_estimate::AnalyticSeKind {
    fn write_identity(&self, out: &mut IdentityBytes) {
        use antecedent_estimate::AnalyticSeKind as Kind;
        match *self {
            Kind::Homoskedastic => {
                out.tag(0);
            }
            Kind::Hc0 => {
                out.tag(1);
            }
            Kind::Hc1 => {
                out.tag(2);
            }
            Kind::Hc2 => {
                out.tag(3);
            }
            Kind::Hc3 => {
                out.tag(4);
            }
            Kind::Cluster => {
                out.tag(5);
            }
            Kind::Multiway => {
                out.tag(6);
            }
            Kind::NeweyWest { lag } => {
                out.tag(7).usize(lag);
            }
            Kind::PanelClusterHac { lag } => {
                out.tag(8).usize(lag);
            }
        }
    }
}

impl CanonicalIdentity for antecedent_core::PopulationRegistry {
    fn write_identity(&self, out: &mut IdentityBytes) {
        for (name, rows) in self.predicates() {
            out.raw(name.as_bytes()).tag(0).usize(rows.len());
            for row in rows {
                out.usize(*row);
            }
        }
        out.tag(0xff);
        for (id, weights, parents) in self.distributions() {
            out.u32(id.raw()).f64s(weights).usize(parents.len());
            for parent in parents {
                out.u32(parent.raw());
            }
        }
    }
}

pub(super) fn estimator_identity_bytes(estimator: &AipwAte) -> Vec<u8> {
    let mut out = IdentityBytes::new(b"aipw.v1");
    out.raw(&estimator.inference.bootstrap_replicates.to_le_bytes());
    out.write(&estimator.overlap).write(&estimator.glm_options).write(&estimator.inference.se_kind);
    out.opt_u32s(estimator.inference.cluster_ids.as_deref())
        .opt_u32s(estimator.cluster_ids_second.as_deref())
        .opt_panel(estimator.inference.panel_times.as_deref());
    match estimator.inference.multiway_ids.as_deref() {
        None => {
            out.tag(0);
        }
        Some(groups) => {
            out.tag(1).usize(groups.len());
            for group in groups {
                out.u32s(group);
            }
        }
    }
    out.raw(&estimator.propensity.identity_digest());
    match estimator.cluster_dml.as_ref() {
        None => {
            out.tag(0);
        }
        Some(cluster) => {
            out.tag(1).raw(cluster.canonical_key().as_bytes());
        }
    }
    match estimator.population_registry.as_ref() {
        None => {
            out.tag(0);
        }
        Some(registry) => {
            out.tag(1).write(registry);
        }
    }
    // Labels per source row are appended only when declared, so a configuration without
    // them keeps its earlier encoding.
    if let Some(labels) = estimator.inference.source_cluster_ids.as_deref() {
        out.tag(2).u32s(labels);
    }
    if let Some(labels) = estimator.source_cluster_ids_second.as_deref() {
        out.tag(3).u32s(labels);
    }
    if let Some(weights) = estimator.sampling_weights.as_ref() {
        out.tag(4).f64s(weights.source());
    }
    out.0
}
