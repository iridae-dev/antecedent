// SPDX-License-Identifier: MIT OR Apache-2.0

//! Joint-law alignment by axis identity.
//!
//! A joint law is one measure indexed by the identities of its axes: graph
//! atom, bootstrap replicate, parameter draw, outcome, trajectory,
//! counterfactual world, and history. The identity of a replicate stream, a
//! posterior, or a world is the digest of that producing law. A caller-chosen
//! display name is not an axis. Two arrays align when those identities are
//! the same. Equal length is not an identity. A contrast of intervention
//! worlds and a response surface that share world, parameter, replicate, and history
//! axes are one measured joint, and that joint records covariance cross terms.
//! Equal length is not an identity, but arrays on the same parameter-draw or
//! bootstrap-replicate axis index the same draws, so a measured joint requires
//! their lengths (the number of draws or replicates) to be equal.
//!
//! Failed mass and unidentified mass stay in the law. Dropping either is
//! refused. A product law is legal only when it is declared, and a declared
//! product is recorded as a product, not as a measured joint. A marginal array
//! is not a joint uncertainty procedure.
//!
//! Every refusal is a [`JointAlignmentError`] and names the next attempt.
//! Library paths do not panic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

use antecedent_core::text::is_blank;
use antecedent_core::{CanonicalWriter, SemanticDigest};

/// Domain tag for [`ProducingLaw::identity`].
const PRODUCING_LAW_TAG: &str = "antecedent.contract.producing_law";

const JOINT_CODE: &str = antecedent_core::reason_code!("joint_law_required");
const INVALID_CODE: &str = antecedent_core::reason_code!("invalid_argument");

const UNRELATED_DETAIL: &str = "joint_alignment.unrelated_axes";
const DROPPED_MASS_DETAIL: &str = "joint_alignment.dropped_mass";
const UNDECLARED_PRODUCT_DETAIL: &str = "joint_alignment.undeclared_product";
const MARGINAL_DETAIL: &str = "joint_alignment.marginal_array";
const INVALID_DETAIL: &str = "joint_alignment.invalid_declaration";
const DRAW_COUNT_DETAIL: &str = "joint_alignment.draw_count_mismatch";

const DECLARE_SHARED_AXES: &str =
    "declare the shared axes, including covariance cross terms for a joint contrast";
const RETAIN_DROPPED_MASS: &str = "retain failed and unidentified mass in the joint law";
const DECLARE_PRODUCT_LAW: &str = "declare the product law";
const REPAIR_DECLARATION: &str = "repair the axis declaration";

/// An axis a joint law is indexed by.
///
/// Declaration order is the canonical order of [`JointAlignment::shared_axes`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum JointAxisKind {
    /// Graph atom of a posterior over graphs.
    GraphAtom,
    /// Bootstrap replicate.
    BootstrapReplicate,
    /// Parameter draw.
    ParameterDraw,
    /// Outcome coordinate.
    Outcome,
    /// Trajectory or path.
    Trajectory,
    /// Counterfactual world, including an intervention world of a contrast.
    CounterfactualWorld,
    /// History, initial state, or time path of a response surface.
    History,
}

impl JointAxisKind {
    /// Every axis, in canonical order.
    pub const ALL: [Self; 7] = [
        Self::GraphAtom,
        Self::BootstrapReplicate,
        Self::ParameterDraw,
        Self::Outcome,
        Self::Trajectory,
        Self::CounterfactualWorld,
        Self::History,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GraphAtom => "graph_atom",
            Self::BootstrapReplicate => "bootstrap_replicate",
            Self::ParameterDraw => "parameter_draw",
            Self::Outcome => "outcome",
            Self::Trajectory => "trajectory",
            Self::CounterfactualWorld => "counterfactual_world",
            Self::History => "history",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not an axis"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// A replicate stream, posterior, or world produces this kind.
    const fn is_law_axis(self) -> bool {
        matches!(
            self,
            Self::GraphAtom
                | Self::BootstrapReplicate
                | Self::ParameterDraw
                | Self::CounterfactualWorld
        )
    }
}

impl fmt::Display for JointAxisKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Law that produced the positions on one axis.
///
/// [`Self::identity`] is the canonical digest of the stream key, posterior id,
/// or world id. [`Self::display_name`] is a caller-chosen label and is not
/// that identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ProducingLaw {
    /// Bootstrap or Monte Carlo replicate stream.
    ReplicateStream {
        /// Key of the replicate stream. This is the producing-law identity.
        stream_key: Arc<str>,
        /// Caller-chosen label. It is not the axis identity.
        display_name: Arc<str>,
    },
    /// Posterior whose draws or graph atoms index the array.
    Posterior {
        /// Identity of the posterior. This is the producing-law identity.
        posterior_id: Arc<str>,
        /// Caller-chosen label. It is not the axis identity.
        display_name: Arc<str>,
    },
    /// Counterfactual world, including an intervention world.
    World {
        /// Identity of the world. This is the producing-law identity.
        world_id: Arc<str>,
        /// Caller-chosen label. It is not the axis identity.
        display_name: Arc<str>,
    },
}

impl ProducingLaw {
    /// Replicate stream keyed by `stream_key`, labeled `display_name`.
    ///
    /// # Errors
    ///
    /// Returns an error when `stream_key` or `display_name` is empty or only
    /// whitespace.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a blank stream key or display name is refused"]
    pub fn replicate_stream(
        stream_key: &str,
        display_name: &str,
    ) -> Result<Self, JointAlignmentError> {
        Ok(Self::ReplicateStream {
            stream_key: law_text(stream_key, "replicate stream key")?,
            display_name: law_text(display_name, "producing-law display name")?,
        })
    }

    /// Posterior identified by `posterior_id`, labeled `display_name`.
    ///
    /// # Errors
    ///
    /// Returns an error when `posterior_id` or `display_name` is empty or only
    /// whitespace.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a blank posterior id or display name is refused"]
    pub fn posterior(posterior_id: &str, display_name: &str) -> Result<Self, JointAlignmentError> {
        Ok(Self::Posterior {
            posterior_id: law_text(posterior_id, "posterior id")?,
            display_name: law_text(display_name, "producing-law display name")?,
        })
    }

    /// World identified by `world_id`, labeled `display_name`.
    ///
    /// # Errors
    ///
    /// Returns an error when `world_id` or `display_name` is empty or only
    /// whitespace.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a blank world id or display name is refused"]
    pub fn world(world_id: &str, display_name: &str) -> Result<Self, JointAlignmentError> {
        Ok(Self::World {
            world_id: law_text(world_id, "world id")?,
            display_name: law_text(display_name, "producing-law display name")?,
        })
    }

    /// Stable `snake_case` name of the producing law.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ReplicateStream { .. } => "replicate_stream",
            Self::Posterior { .. } => "posterior",
            Self::World { .. } => "world",
        }
    }

    /// Caller-chosen label. This is not [`Self::identity`].
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::ReplicateStream { display_name, .. }
            | Self::Posterior { display_name, .. }
            | Self::World { display_name, .. } => display_name,
        }
    }

    /// Canonical digest of the stream key, posterior id, or world id.
    ///
    /// The inputs are the law kind and that producing field. [`Self::display_name`]
    /// is not an input. A shared display name does not make two producing fields
    /// one axis, and distinct display names do not split one producing field
    /// into two digests.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(PRODUCING_LAW_TAG);
        writer.text(self.as_str());
        writer.text(self.key());
        SemanticDigest::from_bytes(writer.finish())
    }

    /// Whether this law produces positions of `kind`.
    ///
    /// A replicate stream produces bootstrap replicates. A posterior produces
    /// parameter draws and graph atoms. A world produces counterfactual worlds.
    #[must_use]
    pub const fn produces(&self, kind: JointAxisKind) -> bool {
        match self {
            Self::ReplicateStream { .. } => matches!(kind, JointAxisKind::BootstrapReplicate),
            Self::Posterior { .. } => {
                matches!(kind, JointAxisKind::ParameterDraw | JointAxisKind::GraphAtom)
            }
            Self::World { .. } => matches!(kind, JointAxisKind::CounterfactualWorld),
        }
    }

    fn key(&self) -> &str {
        match self {
            Self::ReplicateStream { stream_key, .. } => stream_key,
            Self::Posterior { posterior_id, .. } => posterior_id,
            Self::World { world_id, .. } => world_id,
        }
    }
}

impl fmt::Display for ProducingLaw {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One position on a [`JointAxisKind`].
///
/// For a replicate stream, a posterior, or a world, [`Self::id`] is the hex
/// digest from [`ProducingLaw::identity`]. A caller-chosen display name is not
/// an axis. Outcome, trajectory, and history are coordinates; their id is the
/// coordinate token. Two positions are the same only when the kind and that
/// identity are the same.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct JointAxis {
    kind: JointAxisKind,
    id: Arc<str>,
}

impl JointAxis {
    /// Coordinate `kind` at token `id`.
    ///
    /// Outcome, trajectory, and history are coordinates. A replicate stream,
    /// posterior, or world is refused: that identity is [`Self::try_from_law`].
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is empty or only whitespace, or when `kind`
    /// is produced by a law.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a blank coordinate token, or a producing-law kind, is refused"]
    pub fn try_new(kind: JointAxisKind, id: &str) -> Result<Self, JointAlignmentError> {
        if kind.is_law_axis() {
            return Err(invalid(format!(
                "{kind} identity is the digest of its producing law, not a caller-chosen name",
            )));
        }
        if is_blank(id) {
            return Err(invalid("an axis id is blank"));
        }
        Ok(Self { kind, id: Arc::from(id) })
    }

    /// Axis `kind` produced by `law`.
    ///
    /// [`Self::id`] is the lowercase hex of [`ProducingLaw::identity`]. The
    /// law's display name is not stored and is not compared.
    ///
    /// # Errors
    ///
    /// Returns an error when `law` does not produce `kind`.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a law that does not produce this axis kind is refused"]
    pub fn try_from_law(
        kind: JointAxisKind,
        law: &ProducingLaw,
    ) -> Result<Self, JointAlignmentError> {
        if !law.produces(kind) {
            let law_name = law.as_str();
            return Err(invalid(format!("{kind} is not produced by a {law_name} law")));
        }
        Ok(Self { kind, id: Arc::<str>::from(law.identity().to_hex()) })
    }

    /// Kind of axis.
    #[must_use]
    pub const fn kind(&self) -> JointAxisKind {
        self.kind
    }

    /// Identity within that kind.
    ///
    /// For a replicate stream, posterior, or world, this is the hex digest from
    /// [`ProducingLaw::identity`]. A display name is not this identity. For an
    /// outcome, trajectory, or history coordinate, this is the coordinate token.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for JointAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} `{}`", self.kind.as_str(), self.id)
    }
}

/// Whether one mass stays in the joint law.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum MassFate {
    /// The mass remains in the law.
    Retained,
    /// The mass was removed. [`JointAlignmentRequest::align`] refuses this.
    Dropped,
}

impl MassFate {
    /// Both fates, in declaration order.
    pub const ALL: [Self; 2] = [Self::Retained, Self::Dropped];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retained => "retained",
            Self::Dropped => "dropped",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a mass fate"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|fate| fate.as_str() == name)
    }
}

/// Failed estimation mass and unidentified structural mass.
///
/// [`Self::retained`] is the account a joint law carries. [`MassFate::Dropped`]
/// may be declared, and alignment then refuses it. Removing either mass is a
/// different law.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MassAccount {
    failed: MassFate,
    unidentified: MassFate,
}

impl MassAccount {
    /// Failed and unidentified mass both remain.
    #[must_use]
    pub const fn retained() -> Self {
        Self { failed: MassFate::Retained, unidentified: MassFate::Retained }
    }

    /// Account with the two fates the caller declares.
    #[must_use]
    pub const fn new(failed: MassFate, unidentified: MassFate) -> Self {
        Self { failed, unidentified }
    }

    /// Fate of failed estimation mass.
    #[must_use]
    pub const fn failed(self) -> MassFate {
        self.failed
    }

    /// Fate of unidentified structural mass.
    #[must_use]
    pub const fn unidentified(self) -> MassFate {
        self.unidentified
    }

    const fn drops(self) -> bool {
        matches!(self.failed, MassFate::Dropped) || matches!(self.unidentified, MassFate::Dropped)
    }
}

/// Which scientific object an array is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum JointConsumer {
    /// A graph-atom, outcome, or trajectory coordinate.
    Coordinate,
    /// A contrast of intervention worlds.
    Contrast,
    /// A response surface.
    ResponseSurface,
}

impl JointConsumer {
    /// Every consumer, in canonical order.
    pub const ALL: [Self; 3] = [Self::Coordinate, Self::Contrast, Self::ResponseSurface];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Coordinate => "coordinate",
            Self::Contrast => "contrast",
            Self::ResponseSurface => "response_surface",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a consumer"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|consumer| consumer.as_str() == name)
    }
}

/// How the caller says the arrays form a law.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum LawCoupling {
    /// One measured joint on the shared axis identity.
    MeasuredJoint,
    /// A product of the arrays, declared as an assumption.
    DeclaredProduct,
    /// A product used without declaring it.
    UndeclaredProduct,
    /// Marginal output arrays offered as a joint procedure.
    Marginal,
}

impl LawCoupling {
    /// Every coupling, in declaration order.
    pub const ALL: [Self; 4] =
        [Self::MeasuredJoint, Self::DeclaredProduct, Self::UndeclaredProduct, Self::Marginal];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredJoint => "measured_joint",
            Self::DeclaredProduct => "declared_product",
            Self::UndeclaredProduct => "undeclared_product",
            Self::Marginal => "marginal",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a coupling"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|coupling| coupling.as_str() == name)
    }
}

/// One array that participates in a joint-law declaration.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LawArray {
    label: Arc<str>,
    len: usize,
    axes: Vec<JointAxis>,
    mass: MassAccount,
    consumer: JointConsumer,
}

impl LawArray {
    /// Array `label` of length `len`, indexed by `axes`.
    ///
    /// `len` is the number of positions along the draw or replicate axis. It is not an
    /// axis identity, but arrays that share a parameter-draw or bootstrap-replicate axis
    /// must agree on it to form a measured joint. Each kind appears at most
    /// once. The axes are stored in canonical kind order.
    ///
    /// # Errors
    ///
    /// Returns an error when `label` is blank, `axes` is empty, or two axes
    /// share a kind.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a blank label, an empty axis list, or a repeated kind is refused"]
    pub fn try_new(
        label: &str,
        len: usize,
        axes: &[JointAxis],
        mass: MassAccount,
        consumer: JointConsumer,
    ) -> Result<Self, JointAlignmentError> {
        if is_blank(label) {
            return Err(invalid("an array label is blank"));
        }
        let axes = canonical_axes(label, axes)?;
        Ok(Self { label: Arc::from(label), len, axes, mass, consumer })
    }

    /// Caller label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Recorded length. Alignment compares this only for arrays that share a
    /// parameter-draw or bootstrap-replicate axis.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the recorded length is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Axis identity, in canonical kind order.
    #[must_use]
    pub fn axes(&self) -> &[JointAxis] {
        &self.axes
    }

    /// Failed and unidentified mass declared for this array.
    #[must_use]
    pub const fn mass(&self) -> MassAccount {
        self.mass
    }

    /// Scientific object this array is.
    #[must_use]
    pub const fn consumer(&self) -> JointConsumer {
        self.consumer
    }
}

/// Arrays and the coupling the caller claims for them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointAlignmentRequest {
    arrays: Vec<LawArray>,
    coupling: LawCoupling,
}

impl JointAlignmentRequest {
    /// Request alignment of `arrays` under `coupling`.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two arrays are named, or two arrays
    /// share a label.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "fewer than two arrays, or a repeated label, is refused"]
    pub fn try_new(
        arrays: Vec<LawArray>,
        coupling: LawCoupling,
    ) -> Result<Self, JointAlignmentError> {
        if arrays.len() < 2 {
            return Err(invalid("a joint alignment names at least two arrays"));
        }
        let mut labels = BTreeSet::new();
        for array in &arrays {
            if !labels.insert(array.label()) {
                return Err(invalid(format!("array label `{}` is repeated", array.label())));
            }
        }
        Ok(Self { arrays, coupling })
    }

    /// Arrays in declaration order.
    #[must_use]
    pub fn arrays(&self) -> &[LawArray] {
        &self.arrays
    }

    /// Coupling the caller claimed.
    #[must_use]
    pub const fn coupling(&self) -> LawCoupling {
        self.coupling
    }

    /// Align the arrays by axis identity.
    ///
    /// A measured joint requires one shared axis identity and records
    /// covariance cross terms. A declared product is returned as a product even
    /// when the axis identities coincide. Equal lengths do not align arrays
    /// whose axis identities differ. A shared display name does not either:
    /// the identity of a replicate stream, posterior, or world is that law's
    /// digest.
    ///
    /// # Errors
    ///
    /// Returns an error when failed or unidentified mass was dropped, the
    /// coupling is an undeclared product, the coupling is marginal, or a
    /// measured joint's arrays do not share one axis identity. The error's
    /// [`JointAlignmentError::next_attempt`] is the legal declaration.
    ///
    /// # Panics
    ///
    /// Does not panic.
    #[must_use = "a refused alignment names its next attempt"]
    pub fn align(&self) -> Result<JointAlignment, JointAlignmentError> {
        if let Some(error) = dropped_mass(&self.arrays) {
            return Err(error);
        }
        match self.coupling {
            LawCoupling::UndeclaredProduct => Err(undeclared_product()),
            LawCoupling::Marginal => Err(marginal_array()),
            LawCoupling::DeclaredProduct => Ok(record(&self.arrays, AlignedLaw::DeclaredProduct)),
            LawCoupling::MeasuredJoint => measured_joint(&self.arrays),
        }
    }
}

/// Which law an alignment recorded.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum AlignedLaw {
    /// One measured joint on the shared axis identity.
    MeasuredJoint,
    /// A declared product. Not a measured joint.
    DeclaredProduct,
}

impl AlignedLaw {
    /// Both recorded laws, in declaration order.
    pub const ALL: [Self; 2] = [Self::MeasuredJoint, Self::DeclaredProduct];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredJoint => "measured_joint",
            Self::DeclaredProduct => "declared_product",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not an aligned law"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|law| law.as_str() == name)
    }
}

/// Whether a measured joint's covariance includes cross terms.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum JointCovariance {
    /// Cross terms of a measured joint, including a contrast of intervention worlds.
    CrossTerms,
    /// A declared product does not measure joint covariance.
    NotMeasured,
}

impl JointCovariance {
    /// Both records, in declaration order.
    pub const ALL: [Self; 2] = [Self::CrossTerms, Self::NotMeasured];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CrossTerms => "cross_terms",
            Self::NotMeasured => "not_measured",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a covariance record"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|record| record.as_str() == name)
    }
}

/// The law recorded for one alignment request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JointAlignment {
    shared_axes: Vec<JointAxis>,
    law: AlignedLaw,
    covariance: JointCovariance,
    mass: MassAccount,
    consumers: Vec<JointConsumer>,
}

impl JointAlignment {
    /// Axis identity shared by every array, in canonical order.
    ///
    /// Empty when a declared product's arrays do not share one identity.
    /// A non-empty slice on [`AlignedLaw::DeclaredProduct`] is the declared
    /// factor identity, not a measured joint.
    #[must_use]
    pub fn shared_axes(&self) -> &[JointAxis] {
        &self.shared_axes
    }

    /// Identity of `kind` in [`Self::shared_axes`], when that kind is present.
    #[must_use = "a missing axis is not part of the shared identity"]
    pub fn axis_id(&self, kind: JointAxisKind) -> Option<&str> {
        self.shared_axes.iter().find(|axis| axis.kind() == kind).map(JointAxis::id)
    }

    /// Recorded law.
    #[must_use]
    pub const fn law(&self) -> AlignedLaw {
        self.law
    }

    /// Whether the record is a measured joint.
    #[must_use]
    pub const fn is_measured_joint(&self) -> bool {
        matches!(self.law, AlignedLaw::MeasuredJoint)
    }

    /// Whether the record is a declared product.
    #[must_use]
    pub const fn is_product(&self) -> bool {
        matches!(self.law, AlignedLaw::DeclaredProduct)
    }

    /// Covariance record. Cross terms are present only on a measured joint.
    #[must_use]
    pub const fn covariance(&self) -> JointCovariance {
        self.covariance
    }

    /// Whether [`Self::covariance`] is [`JointCovariance::CrossTerms`].
    #[must_use]
    pub const fn has_covariance_cross_terms(&self) -> bool {
        matches!(self.covariance, JointCovariance::CrossTerms)
    }

    /// Mass account of the recorded law. Alignment retains both parts.
    #[must_use]
    pub const fn mass(&self) -> MassAccount {
        self.mass
    }

    /// Consumers present on the arrays, in canonical order, without duplicates.
    #[must_use]
    pub fn consumers(&self) -> &[JointConsumer] {
        &self.consumers
    }
}

/// The declaration that would make a refused alignment legal.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum NextAttempt {
    /// Declare the shared axes. A joint contrast includes covariance cross terms.
    DeclareSharedAxes,
    /// Put failed and unidentified mass back into the law.
    RetainDroppedMass,
    /// Name the product. It is an assumption, not a measured joint.
    DeclareProductLaw,
    /// Replace a blank, empty, or repeated declaration.
    RepairDeclaration,
}

impl NextAttempt {
    /// Every attempt, in the order refusals are checked.
    pub const ALL: [Self; 4] = [
        Self::DeclareSharedAxes,
        Self::RetainDroppedMass,
        Self::DeclareProductLaw,
        Self::RepairDeclaration,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeclareSharedAxes => "declare_shared_axes",
            Self::RetainDroppedMass => "retain_dropped_mass",
            Self::DeclareProductLaw => "declare_product_law",
            Self::RepairDeclaration => "repair_declaration",
        }
    }

    /// Prose printed after `next attempt:`.
    #[must_use]
    pub const fn sentence(self) -> &'static str {
        match self {
            Self::DeclareSharedAxes => DECLARE_SHARED_AXES,
            Self::RetainDroppedMass => RETAIN_DROPPED_MASS,
            Self::DeclareProductLaw => DECLARE_PRODUCT_LAW,
            Self::RepairDeclaration => REPAIR_DECLARATION,
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a next attempt"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|attempt| attempt.as_str() == name)
    }
}

impl fmt::Display for NextAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.sentence())
    }
}

/// A refused joint-law alignment.
///
/// [`Self::next_attempt`] is the legal declaration. [`Self::message`] states
/// the defect. [`Display`] prints both.
///
/// [`Display`]: fmt::Display
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "the next attempt is the legal declaration"]
pub struct JointAlignmentError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: NextAttempt,
}

impl JointAlignmentError {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `joint_alignment.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, without the next-attempt sentence.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Legal declaration after this refusal.
    #[must_use = "the next attempt names the legal declaration"]
    pub const fn next_attempt(&self) -> NextAttempt {
        self.next
    }
}

impl fmt::Display for JointAlignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = self.detail;
        let sentence = self.next.sentence();
        write!(formatter, "{detail}: {message}; next attempt: {sentence}", message = self.message)
    }
}

impl Error for JointAlignmentError {}

fn law_text(value: &str, what: &str) -> Result<Arc<str>, JointAlignmentError> {
    if is_blank(value) {
        return Err(invalid(format!("a {what} is blank")));
    }
    Ok(Arc::from(value))
}

fn invalid(message: impl Into<String>) -> JointAlignmentError {
    JointAlignmentError {
        code: INVALID_CODE,
        detail: INVALID_DETAIL,
        message: message.into(),
        next: NextAttempt::RepairDeclaration,
    }
}

fn canonical_axes(label: &str, axes: &[JointAxis]) -> Result<Vec<JointAxis>, JointAlignmentError> {
    if axes.is_empty() {
        return Err(invalid(format!("array `{label}` names no axis identity")));
    }
    let mut kinds = BTreeSet::new();
    let mut ordered = BTreeSet::new();
    for axis in axes {
        if !kinds.insert(axis.kind()) {
            let kind = axis.kind();
            return Err(invalid(format!("array `{label}` declares {kind} more than once")));
        }
        ordered.insert(axis.clone());
    }
    Ok(ordered.into_iter().collect())
}

fn record(arrays: &[LawArray], law: AlignedLaw) -> JointAlignment {
    let covariance = match law {
        AlignedLaw::MeasuredJoint => JointCovariance::CrossTerms,
        AlignedLaw::DeclaredProduct => JointCovariance::NotMeasured,
    };
    JointAlignment {
        shared_axes: shared_axes(arrays),
        law,
        covariance,
        mass: MassAccount::retained(),
        consumers: consumers_of(arrays),
    }
}

fn measured_joint(arrays: &[LawArray]) -> Result<JointAlignment, JointAlignmentError> {
    if !same_axes(arrays) {
        return Err(unrelated(arrays));
    }
    if let Some(error) = draw_count_mismatch(arrays) {
        return Err(error);
    }
    Ok(record(arrays, AlignedLaw::MeasuredJoint))
}

/// Arrays on one parameter-draw or bootstrap-replicate axis index the same draws,
/// so their lengths must be equal.
fn draw_count_mismatch(arrays: &[LawArray]) -> Option<JointAlignmentError> {
    let first = arrays.first()?;
    let on_draws = first.axes.iter().any(|axis| {
        matches!(axis.kind(), JointAxisKind::ParameterDraw | JointAxisKind::BootstrapReplicate)
    });
    if !on_draws {
        return None;
    }
    let other = arrays.iter().skip(1).find(|array| array.len() != first.len())?;
    Some(JointAlignmentError {
        code: JOINT_CODE,
        detail: DRAW_COUNT_DETAIL,
        message: format!(
            "arrays on the same draw or replicate axis differ in length: `{}` has {} and `{}` has {}",
            first.label(),
            first.len(),
            other.label(),
            other.len(),
        ),
        next: NextAttempt::RepairDeclaration,
    })
}

fn same_axes(arrays: &[LawArray]) -> bool {
    let Some(first) = arrays.first() else {
        return false;
    };
    arrays.iter().all(|array| array.axes == first.axes)
}

fn shared_axes(arrays: &[LawArray]) -> Vec<JointAxis> {
    let Some(first) = arrays.first() else {
        return Vec::new();
    };
    if same_axes(arrays) { first.axes.clone() } else { Vec::new() }
}

fn consumers_of(arrays: &[LawArray]) -> Vec<JointConsumer> {
    let mut consumers = BTreeSet::new();
    for array in arrays {
        consumers.insert(array.consumer());
    }
    consumers.into_iter().collect()
}

fn dropped_mass(arrays: &[LawArray]) -> Option<JointAlignmentError> {
    let mut failed = false;
    let mut unidentified = false;
    let mut labels = String::new();
    for array in arrays {
        if !array.mass().drops() {
            continue;
        }
        if !labels.is_empty() {
            labels.push_str(", ");
        }
        labels.push('`');
        labels.push_str(array.label());
        labels.push('`');
        failed = failed || matches!(array.mass().failed(), MassFate::Dropped);
        unidentified = unidentified || matches!(array.mass().unidentified(), MassFate::Dropped);
    }
    if labels.is_empty() {
        return None;
    }
    let phrase = match (failed, unidentified) {
        (true, true) => "failed and unidentified mass",
        (true, false) => "failed mass",
        (false, true) => "unidentified mass",
        (false, false) => "mass",
    };
    Some(JointAlignmentError {
        code: INVALID_CODE,
        detail: DROPPED_MASS_DETAIL,
        message: format!("dropped {phrase} from {labels}"),
        next: NextAttempt::RetainDroppedMass,
    })
}

fn undeclared_product() -> JointAlignmentError {
    JointAlignmentError {
        code: JOINT_CODE,
        detail: UNDECLARED_PRODUCT_DETAIL,
        message: String::from("an undeclared product law is not a measured joint"),
        next: NextAttempt::DeclareProductLaw,
    }
}

fn marginal_array() -> JointAlignmentError {
    JointAlignmentError {
        code: JOINT_CODE,
        detail: MARGINAL_DETAIL,
        message: String::from("a marginal array is not a joint uncertainty procedure"),
        next: NextAttempt::DeclareSharedAxes,
    }
}

struct Mismatch<'a> {
    left: &'a LawArray,
    right: &'a LawArray,
    axis: JointAxisKind,
}

fn unrelated(arrays: &[LawArray]) -> JointAlignmentError {
    match first_mismatch(arrays) {
        Some(mismatch) => mismatch_error(arrays, &mismatch),
        None => invalid("axis identities differ"),
    }
}

fn first_mismatch(arrays: &[LawArray]) -> Option<Mismatch<'_>> {
    let first = arrays.first()?;
    for array in arrays.iter().skip(1) {
        if let Some(axis) = differing_kind(&first.axes, &array.axes) {
            return Some(Mismatch { left: first, right: array, axis });
        }
    }
    None
}

fn differing_kind(left: &[JointAxis], right: &[JointAxis]) -> Option<JointAxisKind> {
    JointAxisKind::ALL.into_iter().find(|kind| id_of(left, *kind) != id_of(right, *kind))
}

fn id_of(axes: &[JointAxis], kind: JointAxisKind) -> Option<&str> {
    axes.iter().find(|axis| axis.kind() == kind).map(JointAxis::id)
}

fn mismatch_error(arrays: &[LawArray], mismatch: &Mismatch<'_>) -> JointAlignmentError {
    let mut message = String::new();
    match common_length(arrays) {
        Some(len) => {
            message.push_str("equal length ");
            message.push_str(&len.to_string());
            message.push_str(" does not align");
        }
        None => message.push_str("axis identities do not align"),
    }
    message.push_str(" `");
    message.push_str(mismatch.left.label());
    message.push_str("` and `");
    message.push_str(mismatch.right.label());
    message.push_str("` on ");
    message.push_str(mismatch.axis.as_str());
    message.push_str(" (");
    push_id(&mut message, id_of(mismatch.left.axes(), mismatch.axis));
    message.push_str(" versus ");
    push_id(&mut message, id_of(mismatch.right.axes(), mismatch.axis));
    message.push(')');
    JointAlignmentError {
        code: JOINT_CODE,
        detail: UNRELATED_DETAIL,
        message,
        next: NextAttempt::DeclareSharedAxes,
    }
}

fn push_id(message: &mut String, id: Option<&str>) {
    match id {
        Some(text) => {
            message.push('`');
            message.push_str(text);
            message.push('`');
        }
        None => message.push_str("absent"),
    }
}

fn common_length(arrays: &[LawArray]) -> Option<usize> {
    let first = arrays.first()?;
    let len = first.len();
    arrays.iter().all(|array| array.len() == len).then_some(len)
}

#[cfg(test)]
mod tests {
    use super::{
        AlignedLaw, JointAlignmentRequest, JointAxis, JointAxisKind, JointConsumer,
        JointCovariance, LawArray, LawCoupling, MassAccount, MassFate, NextAttempt, ProducingLaw,
    };

    fn axis(kind: JointAxisKind, id: &str) -> JointAxis {
        let law = match kind {
            JointAxisKind::BootstrapReplicate => {
                ProducingLaw::replicate_stream(id, id).expect("stream")
            }
            JointAxisKind::ParameterDraw | JointAxisKind::GraphAtom => {
                ProducingLaw::posterior(id, id).expect("posterior")
            }
            JointAxisKind::CounterfactualWorld => ProducingLaw::world(id, id).expect("world"),
            JointAxisKind::Outcome | JointAxisKind::Trajectory | JointAxisKind::History => {
                return JointAxis::try_new(kind, id).expect("axis");
            }
        };
        JointAxis::try_from_law(kind, &law).expect("axis")
    }

    fn axes(world: &str) -> Vec<JointAxis> {
        vec![
            axis(JointAxisKind::CounterfactualWorld, world),
            axis(JointAxisKind::ParameterDraw, "theta"),
            axis(JointAxisKind::BootstrapReplicate, "boot-1"),
            axis(JointAxisKind::History, "h0"),
        ]
    }

    fn array(
        label: &str,
        len: usize,
        world: &str,
        mass: MassAccount,
        consumer: JointConsumer,
    ) -> LawArray {
        LawArray::try_new(label, len, &axes(world), mass, consumer).expect("array")
    }

    fn contrast_array(len: usize, world: &str, mass: MassAccount) -> LawArray {
        array("contrast", len, world, mass, JointConsumer::Contrast)
    }

    fn surface_array(len: usize, world: &str, mass: MassAccount) -> LawArray {
        array("surface", len, world, mass, JointConsumer::ResponseSurface)
    }

    fn request(arrays: Vec<LawArray>, coupling: LawCoupling) -> JointAlignmentRequest {
        JointAlignmentRequest::try_new(arrays, coupling).expect("request")
    }

    #[test]
    fn equal_length_with_different_axes_is_refused() {
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(4, "do(a,0)", MassAccount::retained());
        assert_eq!(contrast.len(), surface.len());
        let err = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect_err("equal length is not alignment");
        assert_eq!(err.code(), "joint_law_required");
        assert_eq!(err.detail(), "joint_alignment.unrelated_axes");
        let left = ProducingLaw::world("do(a,b)", "do(a,b)").expect("world").identity().to_hex();
        let right = ProducingLaw::world("do(a,0)", "do(a,0)").expect("world").identity().to_hex();
        assert!(err.message().contains("equal length 4"));
        assert!(err.message().contains("counterfactual_world"));
        assert!(err.message().contains(&left));
        assert!(err.message().contains(&right));
        assert!(!err.message().contains("do(a,b)"));
        assert!(!err.message().contains("do(a,0)"));
        assert_eq!(err.next_attempt(), NextAttempt::DeclareSharedAxes);
        assert_eq!(
            err.next_attempt().sentence(),
            "declare the shared axes, including covariance cross terms for a joint contrast",
        );
        assert!(err.to_string().contains(err.next_attempt().sentence()));
    }

    #[test]
    fn same_caller_name_from_different_laws_is_refused() {
        let left_law = ProducingLaw::replicate_stream("stream-left", "shared").expect("left law");
        let right_law =
            ProducingLaw::replicate_stream("stream-right", "shared").expect("right law");
        assert_eq!(left_law.display_name(), right_law.display_name());
        assert_ne!(left_law.identity(), right_law.identity());
        let left_digest = left_law.identity().to_hex();
        let right_digest = right_law.identity().to_hex();
        let left_axis =
            JointAxis::try_from_law(JointAxisKind::BootstrapReplicate, &left_law).expect("left");
        let right_axis =
            JointAxis::try_from_law(JointAxisKind::BootstrapReplicate, &right_law).expect("right");
        assert_eq!(left_axis.id(), left_digest.as_str());
        assert_ne!(left_axis.id(), left_law.display_name());
        assert_ne!(left_axis, right_axis);
        let contrast = LawArray::try_new(
            "contrast",
            4,
            &[left_axis],
            MassAccount::retained(),
            JointConsumer::Contrast,
        )
        .expect("contrast");
        let surface = LawArray::try_new(
            "surface",
            4,
            &[right_axis],
            MassAccount::retained(),
            JointConsumer::ResponseSurface,
        )
        .expect("surface");
        assert_eq!(contrast.len(), surface.len());
        let err = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect_err("a shared display name is not one law");
        assert_eq!(err.code(), "joint_law_required");
        assert_eq!(err.detail(), "joint_alignment.unrelated_axes");
        assert!(err.message().contains("equal length 4"));
        assert!(err.message().contains(&left_digest));
        assert!(err.message().contains(&right_digest));
        assert!(!err.message().contains("shared"));
        assert_eq!(err.next_attempt(), NextAttempt::DeclareSharedAxes);
        assert!(JointAxis::try_new(JointAxisKind::BootstrapReplicate, "shared").is_err());
    }

    #[test]
    fn shared_producing_law_digest_is_admitted() {
        let left_law = ProducingLaw::posterior("posterior-1", "draws").expect("left law");
        let right_law = ProducingLaw::posterior("posterior-1", "samples").expect("right law");
        assert_eq!(left_law.identity(), right_law.identity());
        assert_ne!(left_law.display_name(), right_law.display_name());
        let digest = left_law.identity().to_hex();
        let contrast = LawArray::try_new(
            "contrast",
            4,
            &[JointAxis::try_from_law(JointAxisKind::ParameterDraw, &left_law).expect("left")],
            MassAccount::retained(),
            JointConsumer::Contrast,
        )
        .expect("contrast");
        let surface = LawArray::try_new(
            "surface",
            4,
            &[JointAxis::try_from_law(JointAxisKind::ParameterDraw, &right_law).expect("right")],
            MassAccount::retained(),
            JointConsumer::ResponseSurface,
        )
        .expect("surface");
        let aligned = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect("shared producing-law digest");
        assert!(aligned.is_measured_joint());
        assert_eq!(aligned.covariance(), JointCovariance::CrossTerms);
        assert_eq!(aligned.axis_id(JointAxisKind::ParameterDraw), Some(digest.as_str()));
        assert_ne!(aligned.axis_id(JointAxisKind::ParameterDraw), Some("draws"));
        assert_ne!(aligned.axis_id(JointAxisKind::ParameterDraw), Some("samples"));
    }

    #[test]
    fn shared_world_parameter_replicate_history_axes_align() {
        let contrast = contrast_array(9, "do(a,b)", MassAccount::retained());
        let surface = surface_array(9, "do(a,b)", MassAccount::retained());
        assert_eq!(contrast.len(), surface.len());
        let aligned = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect("shared axes");
        assert!(aligned.is_measured_joint());
        assert!(!aligned.is_product());
        assert_eq!(aligned.law(), AlignedLaw::MeasuredJoint);
        assert_eq!(aligned.covariance(), JointCovariance::CrossTerms);
        assert!(aligned.has_covariance_cross_terms());
        assert_eq!(aligned.mass(), MassAccount::retained());
        assert_eq!(aligned.consumers(), &[JointConsumer::Contrast, JointConsumer::ResponseSurface],);
        let kinds: Vec<_> = aligned.shared_axes().iter().map(JointAxis::kind).collect();
        assert_eq!(
            kinds,
            vec![
                JointAxisKind::BootstrapReplicate,
                JointAxisKind::ParameterDraw,
                JointAxisKind::CounterfactualWorld,
                JointAxisKind::History,
            ],
        );
        let world = ProducingLaw::world("do(a,b)", "do(a,b)").expect("world").identity().to_hex();
        let draw =
            ProducingLaw::posterior("theta", "theta").expect("posterior").identity().to_hex();
        let replicate =
            ProducingLaw::replicate_stream("boot-1", "boot-1").expect("stream").identity().to_hex();
        assert_eq!(aligned.axis_id(JointAxisKind::CounterfactualWorld), Some(world.as_str()));
        assert_eq!(aligned.axis_id(JointAxisKind::ParameterDraw), Some(draw.as_str()));
        assert_eq!(aligned.axis_id(JointAxisKind::BootstrapReplicate), Some(replicate.as_str()));
        assert_eq!(aligned.axis_id(JointAxisKind::History), Some("h0"));
        assert_eq!(aligned.axis_id(JointAxisKind::GraphAtom), None);
        assert_eq!(aligned.axis_id(JointAxisKind::Outcome), None);
        assert_eq!(aligned.axis_id(JointAxisKind::Trajectory), None);
    }

    #[test]
    fn unequal_draw_counts_on_a_shared_draw_axis_are_refused() {
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(9, "do(a,b)", MassAccount::retained());
        let err = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect_err("different numbers of draws on one axis");
        assert_eq!(err.code(), "joint_law_required");
        assert_eq!(err.detail(), "joint_alignment.draw_count_mismatch");
        assert!(err.message().contains("`contrast` has 4"));
        assert!(err.message().contains("`surface` has 9"));
        assert_eq!(err.next_attempt(), NextAttempt::RepairDeclaration);
        // Declared products do not need the draws to correspond.
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(9, "do(a,b)", MassAccount::retained());
        assert!(request(vec![contrast, surface], LawCoupling::DeclaredProduct).align().is_ok());
    }

    #[test]
    fn dropped_failed_mass_is_refused() {
        let mass = MassAccount::new(MassFate::Dropped, MassFate::Retained);
        let contrast = contrast_array(4, "do(a,b)", mass);
        let surface = surface_array(4, "do(a,b)", MassAccount::retained());
        let err = request(vec![contrast, surface], LawCoupling::MeasuredJoint)
            .align()
            .expect_err("dropped failed mass");
        assert_eq!(err.code(), "invalid_argument");
        assert_eq!(err.detail(), "joint_alignment.dropped_mass");
        assert!(err.message().contains("failed mass"));
        assert!(err.message().contains("contrast"));
        assert_eq!(err.next_attempt(), NextAttempt::RetainDroppedMass);
        assert!(err.next_attempt().sentence().contains("failed"));
        assert!(err.next_attempt().sentence().contains("unidentified"));
    }

    #[test]
    fn dropped_unidentified_mass_is_refused() {
        let mass = MassAccount::new(MassFate::Retained, MassFate::Dropped);
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(4, "do(a,b)", mass);
        let err = request(vec![contrast, surface], LawCoupling::DeclaredProduct)
            .align()
            .expect_err("dropped unidentified mass");
        assert!(err.message().contains("unidentified mass"));
        assert!(!err.message().contains("failed mass"));
        assert_eq!(err.next_attempt(), NextAttempt::RetainDroppedMass);
    }

    #[test]
    fn undeclared_product_law_is_refused() {
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(4, "do(a,b)", MassAccount::retained());
        let err = request(vec![contrast, surface], LawCoupling::UndeclaredProduct)
            .align()
            .expect_err("undeclared product");
        assert_eq!(err.code(), "joint_law_required");
        assert_eq!(err.detail(), "joint_alignment.undeclared_product");
        assert!(err.message().contains("undeclared product"));
        assert_eq!(err.next_attempt(), NextAttempt::DeclareProductLaw);
    }

    #[test]
    fn declared_product_law_is_a_product_not_a_measured_joint() {
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(4, "do(a,b)", MassAccount::retained());
        let aligned = request(vec![contrast, surface], LawCoupling::DeclaredProduct)
            .align()
            .expect("declared product");
        assert!(aligned.is_product());
        assert!(!aligned.is_measured_joint());
        assert_eq!(aligned.law(), AlignedLaw::DeclaredProduct);
        assert_eq!(aligned.covariance(), JointCovariance::NotMeasured);
        assert!(!aligned.has_covariance_cross_terms());
        assert_eq!(aligned.shared_axes().len(), 4);
        assert_eq!(aligned.mass(), MassAccount::retained());
    }

    #[test]
    fn marginal_array_is_not_a_joint() {
        let contrast = contrast_array(4, "do(a,b)", MassAccount::retained());
        let surface = surface_array(4, "do(a,b)", MassAccount::retained());
        let err = request(vec![contrast, surface], LawCoupling::Marginal)
            .align()
            .expect_err("marginal arrays");
        assert_eq!(err.code(), "joint_law_required");
        assert_eq!(err.detail(), "joint_alignment.marginal_array");
        assert!(err.message().contains("marginal array is not a joint"));
        assert_eq!(err.next_attempt(), NextAttempt::DeclareSharedAxes);
    }

    #[test]
    fn axis_kinds_use_the_joint_vocabulary() {
        let names = [
            "graph_atom",
            "bootstrap_replicate",
            "parameter_draw",
            "outcome",
            "trajectory",
            "counterfactual_world",
            "history",
        ];
        assert_eq!(JointAxisKind::ALL.len(), names.len());
        for (kind, name) in JointAxisKind::ALL.into_iter().zip(names) {
            assert_eq!(kind.as_str(), name);
            assert_eq!(JointAxisKind::from_name(name), Some(kind));
        }
        assert_eq!(JointAxisKind::from_name("draw"), None);
    }
}
