//! One shape for a typed refusal, for the attempt it names, and for the
//! name tables of fieldless enums.
//!
//! Evidence, dependence, summary and ledger errors each carry a registered
//! reason code, a namespaced detail, and a message. [`Refused`] is that shape.
//! [`Refusal`] borrows the three parts. [`NextAttempt`] is the one value a
//! refusal uses to name what to try instead: a stable name and a sentence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// The three parts of a refusal, borrowed from the error that carries them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Refusal<'a> {
    code: &'static str,
    detail: &'static str,
    message: &'a str,
}

impl<'a> Refusal<'a> {
    /// A refusal from its parts.
    #[must_use]
    pub const fn new(code: &'static str, detail: &'static str, message: &'a str) -> Self {
        Self { code, detail, message }
    }

    /// Registered reason code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        self.code
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        self.detail
    }

    /// What was refused.
    #[must_use]
    pub const fn message(self) -> &'a str {
        self.message
    }

    /// The parts as `(code, detail, explanation)`.
    #[must_use]
    pub const fn into_parts(self) -> (&'static str, &'static str, &'a str) {
        (self.code, self.detail, self.message)
    }
}

impl fmt::Display for Refusal<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message)
    }
}

/// An error that carries a registered code, a namespaced detail, and a message.
pub trait Refused {
    /// Registered reason code.
    fn refusal_code(&self) -> &'static str;

    /// Namespaced detail.
    fn refusal_detail(&self) -> &'static str;

    /// What was refused.
    fn refusal_message(&self) -> &str;

    /// The refusal as one value.
    fn as_refusal(&self) -> Refusal<'_> {
        Refusal::new(self.refusal_code(), self.refusal_detail(), self.refusal_message())
    }
}

/// An owned refusal: `(code, detail, explanation)`.
///
/// Artifact errors build one of these when their message was stored as text.
pub type RefusalParts = (&'static str, String, String);

/// The explanation of a `detail: explanation` message, or the whole text when
/// it has no detail prefix.
#[must_use]
pub fn explanation_of(text: &str) -> String {
    text.split_once(": ").map_or(text, |(_, rest)| rest).to_owned()
}

/// A refusal of an artifact whose stored identity field changed.
///
/// The code is `route_not_supported` and the explanation is `<field> changed`.
#[must_use]
pub fn changed_identity(detail: &str, field: impl fmt::Display) -> RefusalParts {
    (crate::reason_code!("route_not_supported"), detail.to_owned(), format!("{field} changed"))
}

/// A refusal of an artifact whose stored value does not replay.
///
/// The code is `invalid_argument`.
#[must_use]
pub fn replay_mismatch(detail: &str, explanation: String) -> RefusalParts {
    (crate::reason_code!("invalid_argument"), detail.to_owned(), explanation)
}

/// What to try instead of a refused request: a stable name and the sentence
/// printed after `next attempt:`.
///
/// Each module that refuses names its attempts as constants of this type.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NextAttempt {
    name: &'static str,
    sentence: &'static str,
}

impl NextAttempt {
    /// An attempt from its stable name and its sentence.
    #[must_use]
    pub const fn new(name: &'static str, sentence: &'static str) -> Self {
        Self { name, sentence }
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.name
    }

    /// Prose printed after `next attempt:`.
    #[must_use]
    pub const fn sentence(self) -> &'static str {
        self.sentence
    }

    /// The attempt in `attempts` with this spelling.
    #[must_use = "an unknown name is not a next attempt"]
    pub fn from_name(attempts: &[Self], name: &str) -> Option<Self> {
        attempts.iter().copied().find(|attempt| attempt.name == name)
    }

    /// `message; next attempt: sentence`.
    #[must_use]
    pub fn append_to(self, message: &str) -> String {
        format!("{message}; next attempt: {}", self.sentence)
    }
}

impl fmt::Display for NextAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.sentence)
    }
}

/// Implement [`Refused`] for an error with `code`, `detail`, and `message`
/// fields of the usual types.
///
/// `impl_refused!(Type, static_message)` is the form for an error whose
/// `message` is a `&'static str`.
#[macro_export]
macro_rules! impl_refused {
    ($ty:ty, static_message) => {
        impl $crate::refusal::Refused for $ty {
            fn refusal_code(&self) -> &'static str {
                self.code
            }

            fn refusal_detail(&self) -> &'static str {
                self.detail
            }

            fn refusal_message(&self) -> &str {
                self.message
            }
        }
    };
    ($ty:ty) => {
        impl $crate::refusal::Refused for $ty {
            fn refusal_code(&self) -> &'static str {
                self.code
            }

            fn refusal_detail(&self) -> &'static str {
                self.detail
            }

            fn refusal_message(&self) -> &str {
                self.message.as_str()
            }
        }
    };
}

/// Implement `ALL`, `as_str`, and `from_name` for a fieldless `Copy` enum.
#[macro_export]
macro_rules! named_variants {
    (@count) => { 0usize };
    (@count $head:ident $($tail:ident)*) => { 1usize + $crate::named_variants!(@count $($tail)*) };
    ($ty:ident { $($variant:ident => $name:literal),+ $(,)? }) => {
        impl $ty {
            /// Every variant, in declaration order.
            pub const ALL: [Self; $crate::named_variants!(@count $($variant)+)] =
                [$(Self::$variant),+];

            /// Stable `snake_case` name.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),+
                }
            }

            /// Parse the spelling of [`Self::as_str`].
            #[must_use = "an unknown name is not a variant"]
            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL.into_iter().find(|value| value.as_str() == name)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{NextAttempt, Refusal, changed_identity, explanation_of, replay_mismatch};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Colour {
        Red,
        DeepBlue,
    }

    crate::named_variants! {
        Colour {
            Red => "red",
            DeepBlue => "deep_blue",
        }
    }

    #[test]
    fn named_variants_round_trip() {
        assert_eq!(Colour::ALL, [Colour::Red, Colour::DeepBlue]);
        for colour in Colour::ALL {
            assert_eq!(Colour::from_name(colour.as_str()), Some(colour));
        }
        assert_eq!(Colour::from_name("blue"), None);
    }

    #[test]
    fn next_attempt_names_and_prints_its_sentence() {
        let wait = NextAttempt::new("wait", "wait for the data");
        let all = [wait, NextAttempt::new("stop", "stop")];
        assert_eq!(wait.name(), "wait");
        assert_eq!(wait.as_str(), "wait");
        assert_eq!(wait.to_string(), "wait for the data");
        assert_eq!(wait.append_to("refused"), "refused; next attempt: wait for the data");
        assert_eq!(NextAttempt::from_name(&all, "stop"), Some(all[1]));
        assert_eq!(NextAttempt::from_name(&all, "go"), None);
    }

    #[test]
    fn refusal_parts_and_artifact_helpers() {
        let refusal = Refusal::new("invalid_argument", "x.rule", "bad");
        assert_eq!(refusal.into_parts(), ("invalid_argument", "x.rule", "bad"));
        assert_eq!(refusal.to_string(), "x.rule: bad");
        assert_eq!(explanation_of("x.rule: bad"), "bad");
        assert_eq!(explanation_of("bare"), "bare");
        assert_eq!(
            changed_identity("x.identity_mismatch", "fit"),
            ("route_not_supported", "x.identity_mismatch".to_owned(), "fit changed".to_owned())
        );
        assert_eq!(
            replay_mismatch("x.replay", "no replay".to_owned()),
            ("invalid_argument", "x.replay".to_owned(), "no replay".to_owned())
        );
    }
}
