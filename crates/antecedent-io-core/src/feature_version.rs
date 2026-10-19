//! Feature and family-payload versions, separate from package `SemVer`.
//!
//! A new artifact feature or family payload carries a [`FeatureVersion`]. That
//! integer is not a [`SemanticVersion`] and not `CARGO_PKG_VERSION`. The
//! durable container stays [`STABLE_FORMAT`] (format `0.5`). Sources `0.1`
//! through `0.4` ([`MIGRATION_SOURCE_MINORS`]) are what
//! [`migrate_artifact`](crate::migrate::migrate_artifact) already migrates.
//! This module does not add a container migrator and does not decode bytes.
//! Each family keeps its own consumer; nothing here is a universal loader.
//!
//! [`classify_feature_read`] is the decision:
//!
//! * format `0.5` and an accepted feature version is [`FeatureRead::Compatible`]
//! * format `0.1` through `0.4` and an accepted feature version is
//!   [`FeatureRead::RequiresMigration`]
//! * any other feature version is [`FeatureRefusal::UnknownFeature`]
//! * a package [`SemanticVersion`] is [`FeatureRefusal::PackageSemVer`]
//! * any other container format is [`FeatureRefusal::UnsupportedContainer`]
//!
//! [`CONTAINER_VERSION`](crate::CONTAINER_VERSION) is the framing word (`1`),
//! not a feature version and not format `0.5`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use crate::migrate::{STABLE_FORMAT, is_supported_source};
use crate::wire::{FormatVersion, SemanticVersion};

/// Minors of major `0` that must be migrated before a feature read.
///
/// `1` through `4` are the migrate-from formats. Minor `5` is [`STABLE_FORMAT`],
/// a compatible read, and is not in this range.
pub const MIGRATION_SOURCE_MINORS: std::ops::RangeInclusive<u16> = 1..=4;

/// Integer version of one artifact feature or family payload.
///
/// `version` is not a package major, minor, or patch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureVersion {
    /// Feature or family name the owning consumer recognizes.
    pub name: &'static str,
    /// Version that consumer writes and accepts.
    pub version: u32,
}

/// A version presented to [`classify_feature_read`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestedVersion<'a> {
    /// A feature or family-payload version.
    Feature {
        /// Feature or family name.
        name: &'a str,
        /// Feature version.
        version: u32,
    },
    /// A package release. This is never a feature version.
    Package(&'a SemanticVersion),
}

/// What a reader may do with a container for one accepted feature.
///
/// Neither variant decodes a payload. After a compatible read, or after
/// [`migrate_artifact`](crate::migrate::migrate_artifact) for a required
/// migration, the caller hands the container to the consumer for
/// [`FeatureVersion::name`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FeatureRead {
    /// Container format is [`STABLE_FORMAT`]. The feature version is accepted.
    Compatible {
        /// Accepted feature.
        feature: FeatureVersion,
    },
    /// Container format is a migrate-from source `0.1` through `0.4`.
    ///
    /// Run [`migrate_artifact`](crate::migrate::migrate_artifact), then the
    /// family's consumer. Migration does not invent a family payload.
    RequiresMigration {
        /// Source format still on the container.
        from: FormatVersion,
        /// Accepted feature, read only after migration.
        feature: FeatureVersion,
    },
}

/// Why a feature read was refused.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum FeatureRefusal {
    /// `name` at `version` is not an accepted feature or family-payload version.
    #[error("unsupported feature `{name}` version {version}")]
    UnknownFeature {
        /// Requested feature or family name.
        name: String,
        /// Requested feature version.
        version: u32,
    },
    /// Container format is not `0.5` and not a migrate-from source `0.1` through `0.4`.
    #[error("unsupported container format {major}.{minor}")]
    UnsupportedContainer {
        /// Observed major.
        major: u16,
        /// Observed minor.
        minor: u16,
    },
    /// `major.minor.patch` is a package release, not a feature version.
    #[error("package version {major}.{minor}.{patch} is not a feature version")]
    PackageSemVer {
        /// Package major.
        major: u32,
        /// Package minor.
        minor: u32,
        /// Package patch.
        patch: u32,
    },
}

/// True when `container` is a migrate-from format `0.1` through `0.4`.
///
/// Format `0.5` is [`STABLE_FORMAT`] and returns false. A format the migration
/// registry does not list returns false as well.
#[must_use]
pub fn is_migration_source(container: FormatVersion) -> bool {
    container.major == 0
        && MIGRATION_SOURCE_MINORS.contains(&container.minor)
        && container != STABLE_FORMAT
        && is_supported_source(container)
}

/// Decide how to read `container` for `requested`.
///
/// `accepted` is the closed set of feature versions the caller implements.
/// An unknown feature is refused before a migration is offered, so an
/// unsupported feature is not migrated into a family consumer. A package
/// [`SemanticVersion`] is refused even when one of its components equals an
/// accepted feature version.
///
/// # Errors
///
/// Unknown feature versions, package `SemVer` presented as a feature version,
/// and container formats outside `0.5` and `0.1` through `0.4`.
pub fn classify_feature_read(
    container: FormatVersion,
    requested: RequestedVersion<'_>,
    accepted: &[FeatureVersion],
) -> Result<FeatureRead, FeatureRefusal> {
    let (name, version) = match requested {
        RequestedVersion::Package(package) => {
            return Err(FeatureRefusal::PackageSemVer {
                major: package.major,
                minor: package.minor,
                patch: package.patch,
            });
        }
        RequestedVersion::Feature { name, version } => (name, version),
    };
    let Some(feature) =
        accepted.iter().copied().find(|feature| feature.name == name && feature.version == version)
    else {
        return Err(FeatureRefusal::UnknownFeature { name: name.to_owned(), version });
    };
    if container == STABLE_FORMAT {
        return Ok(FeatureRead::Compatible { feature });
    }
    if is_migration_source(container) {
        return Ok(FeatureRead::RequiresMigration { from: container, feature });
    }
    Err(FeatureRefusal::UnsupportedContainer { major: container.major, minor: container.minor })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family() -> FeatureVersion {
        FeatureVersion { name: "family_payload", version: 1 }
    }

    fn request(feature: FeatureVersion) -> RequestedVersion<'static> {
        RequestedVersion::Feature { name: feature.name, version: feature.version }
    }

    #[test]
    fn unknown_feature_version_is_refused() {
        let accepted = [family()];
        let cases = [
            ("family_payload", 2, STABLE_FORMAT),
            ("other_family", 1, STABLE_FORMAT),
            ("family_payload", 2, FormatVersion { major: 0, minor: 4 }),
        ];
        for (name, version, container) in cases {
            let got = classify_feature_read(
                container,
                RequestedVersion::Feature { name, version },
                &accepted,
            );
            assert_eq!(got, Err(FeatureRefusal::UnknownFeature { name: name.to_owned(), version }));
        }
    }

    #[test]
    fn compatible_read_of_container_0_5() {
        assert_eq!(STABLE_FORMAT, FormatVersion { major: 0, minor: 5 });
        assert!(!is_migration_source(STABLE_FORMAT));
        let feature = family();
        assert_eq!(
            classify_feature_read(STABLE_FORMAT, request(feature), &[feature]),
            Ok(FeatureRead::Compatible { feature }),
        );
    }

    #[test]
    fn migration_sources_0_1_through_0_4_require_migration() {
        let feature = family();
        for minor in MIGRATION_SOURCE_MINORS {
            let from = FormatVersion { major: 0, minor };
            assert!(is_supported_source(from));
            assert!(is_migration_source(from));
            assert_eq!(
                classify_feature_read(from, request(feature), &[feature]),
                Ok(FeatureRead::RequiresMigration { from, feature }),
            );
        }
    }

    #[test]
    fn unsupported_container_is_refused() {
        let feature = family();
        for container in [
            FormatVersion { major: 0, minor: 0 },
            FormatVersion { major: 0, minor: 6 },
            FormatVersion { major: 1, minor: 0 },
        ] {
            assert!(!is_migration_source(container));
            assert_eq!(
                classify_feature_read(container, request(feature), &[feature]),
                Err(FeatureRefusal::UnsupportedContainer {
                    major: container.major,
                    minor: container.minor,
                }),
            );
        }
    }

    #[test]
    fn package_semver_is_not_a_feature_version() {
        let package = SemanticVersion::from_crate_version(antecedent_core::VERSION).unwrap();
        let accepted = [FeatureVersion { name: "family_payload", version: package.major }];
        assert_eq!(
            classify_feature_read(STABLE_FORMAT, RequestedVersion::Package(&package), &accepted),
            Err(FeatureRefusal::PackageSemVer {
                major: package.major,
                minor: package.minor,
                patch: package.patch,
            }),
        );
    }

    #[test]
    fn framing_word_is_not_a_feature_or_format_version() {
        assert_eq!(crate::CONTAINER_VERSION, 1);
        let feature = FeatureVersion { name: "family_payload", version: crate::CONTAINER_VERSION };
        assert_eq!(
            classify_feature_read(
                FormatVersion { major: 1, minor: 0 },
                request(feature),
                &[feature],
            ),
            Err(FeatureRefusal::UnsupportedContainer { major: 1, minor: 0 }),
        );
        assert_eq!(
            classify_feature_read(STABLE_FORMAT, request(feature), &[feature]),
            Ok(FeatureRead::Compatible { feature }),
        );
    }
}
