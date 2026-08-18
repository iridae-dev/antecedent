//! Semantic coalition evaluation cache.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::CachePolicy;

use crate::error::AttributionError;
use crate::result::CacheStats;

/// Exact-mode dense mask index is enabled at `k ≤` this (2^16 slots).
/// Larger exact games stay on the HashMap; 2^22 as a cutoff would be gigabytes.
pub(crate) const DENSE_MASK_MAX_PLAYERS: usize = 16;

/// Full coalition bitmask for up to 64 players.
pub(crate) fn full_coalition_mask(n_players: usize) -> Result<u64, AttributionError> {
    match n_players {
        0 => Ok(0),
        1..=63 => Ok((1u64 << n_players) - 1),
        64 => Ok(u64::MAX),
        requested => Err(AttributionError::SizeLimit { kind: "components", requested, max: 64 }),
    }
}

/// Key for a coalition / substitution evaluation.
///
/// `mask` bits select which components use the comparison (new) mechanism;
/// `tag` distinguishes baseline-vs-comparison model pairings or path contexts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CoalitionKey {
    /// Bitmask of active (comparison) components; supports up to 64 players.
    pub mask: u64,
    /// Caller-defined semantic tag (e.g. outcome dense id, measure id).
    pub tag: u64,
}

/// Cached scalar payoff for a coalition.
#[derive(Clone, Debug)]
struct CacheEntry {
    value: f64,
    bytes: u64,
}

/// Semantic cache keyed by intervention / substitution state.
#[derive(Clone, Debug, Default)]
pub struct CoalitionCache {
    enabled: bool,
    max_bytes: Option<u64>,
    used_bytes: u64,
    map: HashMap<CoalitionKey, CacheEntry>,
    /// Mask-indexed slot for exact games with `tag == 0` (`None` = uncached).
    dense: Option<Vec<Option<CacheEntry>>>,
    hits: u64,
    misses: u64,
    saturated: bool,
}

impl CoalitionCache {
    /// Construct from execution cache policy.
    #[must_use]
    pub fn from_policy(policy: CachePolicy) -> Self {
        Self {
            enabled: policy.enabled,
            max_bytes: policy.max_bytes,
            used_bytes: 0,
            map: HashMap::new(),
            dense: None,
            hits: 0,
            misses: 0,
            saturated: false,
        }
    }

    /// Enable a mask-indexed table for exact enumeration (`k ≤ 16`, `tag == 0`).
    ///
    /// Callers with a larger player count, or Monte Carlo that only visits a
    /// sparse subset of masks, leave this unset and stay on the HashMap.
    pub fn enable_dense_index(&mut self, n_players: usize) {
        if !self.enabled || n_players == 0 || n_players > DENSE_MASK_MAX_PLAYERS {
            return;
        }
        let slots = 1usize << n_players;
        self.dense = Some(vec![None; slots]);
    }

    /// Disabled cache (always miss).
    #[must_use]
    pub fn disabled() -> Self {
        Self::from_policy(CachePolicy::disabled())
    }

    fn dense_slot(&self, key: CoalitionKey) -> Option<usize> {
        if key.tag != 0 {
            return None;
        }
        let dense = self.dense.as_ref()?;
        let idx = usize::try_from(key.mask).ok()?;
        (idx < dense.len()).then_some(idx)
    }

    /// Lookup a cached payoff.
    pub fn get(&mut self, key: CoalitionKey) -> Option<f64> {
        if !self.enabled {
            self.misses += 1;
            return None;
        }
        if let Some(idx) = self.dense_slot(key) {
            if let Some(e) = self.dense.as_ref().and_then(|d| d[idx].as_ref()) {
                self.hits += 1;
                return Some(e.value);
            }
            self.misses += 1;
            return None;
        }
        if let Some(e) = self.map.get(&key) {
            self.hits += 1;
            Some(e.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a payoff, respecting the byte budget (LRU-free: refuse when full).
    ///
    /// Saturation is sticky: once a unique insert is refused, [`CacheStats::saturated`]
    /// stays true for the rest of the run.
    pub fn insert(&mut self, key: CoalitionKey, value: f64) {
        if !self.enabled {
            return;
        }
        let bytes = 32u64; // key + f64 + overhead estimate
        let replacing = if let Some(idx) = self.dense_slot(key) {
            self.dense.as_ref().is_some_and(|d| d[idx].is_some())
        } else {
            self.map.contains_key(&key)
        };
        if let Some(max) = self.max_bytes {
            if self.used_bytes + bytes > max && !replacing {
                self.saturated = true;
                return;
            }
        }
        let entry = CacheEntry { value, bytes };
        if let Some(idx) = self.dense_slot(key) {
            if let Some(dense) = self.dense.as_mut() {
                if let Some(old) = dense[idx].replace(entry) {
                    self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
                }
                self.used_bytes = self.used_bytes.saturating_add(bytes);
            }
            return;
        }
        if let Some(old) = self.map.insert(key, entry) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        self.used_bytes = self.used_bytes.saturating_add(bytes);
    }

    /// Snapshot statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let dense_entries =
            self.dense.as_ref().map(|d| d.iter().filter(|e| e.is_some()).count()).unwrap_or(0);
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: (self.map.len() + dense_entries) as u64,
            bytes: self.used_bytes,
            saturated: self.saturated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_miss_and_disable() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(10_000)));
        let k = CoalitionKey { mask: 0b101, tag: 1 };
        assert!(c.get(k).is_none());
        c.insert(k, 1.5);
        assert_eq!(c.get(k), Some(1.5));
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!(!s.saturated);

        let mut d = CoalitionCache::disabled();
        d.insert(k, 2.0);
        assert!(d.get(k).is_none());
    }

    #[test]
    fn full_mask_handles_u64_boundary_without_shifting_by_width() {
        assert_eq!(full_coalition_mask(0).unwrap(), 0);
        assert_eq!(full_coalition_mask(3).unwrap(), 0b111);
        assert_eq!(full_coalition_mask(64).unwrap(), u64::MAX);
        assert!(matches!(
            full_coalition_mask(65),
            Err(AttributionError::SizeLimit { requested: 65, max: 64, .. })
        ));
    }

    #[test]
    fn dense_index_skips_hash_for_tag_zero() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(10_000)));
        c.enable_dense_index(3);
        let k = CoalitionKey { mask: 0b101, tag: 0 };
        c.insert(k, 1.5);
        assert_eq!(c.get(k), Some(1.5));
        assert_eq!(c.stats().entries, 1);
        assert!(c.map.is_empty(), "tag-0 exact inserts must not land in the HashMap");
    }

    #[test]
    fn refuse_when_full_sets_saturated() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(32)));
        let k1 = CoalitionKey { mask: 1, tag: 1 };
        let k2 = CoalitionKey { mask: 2, tag: 1 };
        c.insert(k1, 1.0);
        c.insert(k2, 2.0);
        assert!(c.stats().saturated);
        assert_eq!(c.stats().entries, 1);
        assert!(c.get(k2).is_none());
    }
}
