//! Keyword defaults the native signatures repeat as literals.
//!
//! PyO3 renders a non-literal default as `...` in `__text_signature__`, so the
//! `#[pyo3(signature = ...)]` attributes keep the numbers. This module is the
//! table those literals must match.
//!
//! `seed` is two constants on purpose. Most entry points default to [`SEED`]
//! (1). A smaller set defaults to [`SEED_ZERO`] (0). Unifying them would change
//! results, so each function keeps the default it has.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Discovery and estimation significance level (`alpha=0.05`).
pub(crate) const ALPHA: f64 = 0.05;

/// Seed of the discovery-style entry points (`seed=1`).
pub(crate) const SEED: u64 = 1;

/// Seed of the entry points that already default to zero (`seed=0`).
pub(crate) const SEED_ZERO: u64 = 0;

#[cfg(test)]
mod tests {
    use super::{ALPHA, SEED, SEED_ZERO};

    #[test]
    fn signature_literals_match_the_defaults_table() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut alpha = 0;
        let mut seed_one = 0;
        let mut seed_zero = 0;
        for entry in walk(&root) {
            let text = std::fs::read_to_string(&entry).unwrap();
            for line in text.lines() {
                if !line.contains("signature")
                    && !line.contains("seed=")
                    && !line.contains("alpha=")
                {
                    continue;
                }
                alpha += line.matches("alpha=0.05").count();
                if line.contains("alpha=")
                    && !line.contains("alpha=0.05")
                    && !line.contains("alpha=None")
                    && !line.contains("alphas")
                {
                    panic!(
                        "{} has an alpha default other than {}:\n{line}",
                        entry.display(),
                        ALPHA
                    );
                }
                seed_one += line.matches("seed=1").count();
                seed_zero += line.matches("seed=0").count();
            }
        }
        assert!(alpha > 0, "expected alpha={ALPHA} literals");
        assert!(seed_one > 0, "expected seed={SEED} literals");
        assert!(seed_zero > 0, "expected seed={SEED_ZERO} literals");
        assert_ne!(SEED, SEED_ZERO);
    }

    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap().path();
            if entry.is_dir() {
                out.extend(walk(&entry));
            } else if entry.extension().is_some_and(|ext| ext == "rs")
                && entry.file_name().is_none_or(|name| name != "defaults.rs")
            {
                out.push(entry);
            }
        }
        out
    }
}
