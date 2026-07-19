//! Semantic coalition evaluation cache.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use causal_core::CachePolicy;

use crate::result::CacheStats;

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
    hits: u64,
    misses: u64,
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
            hits: 0,
            misses: 0,
        }
    }

    /// Disabled cache (always miss).
    #[must_use]
    pub fn disabled() -> Self {
        Self::from_policy(CachePolicy::disabled())
    }

    /// Lookup a cached payoff.
    pub fn get(&mut self, key: CoalitionKey) -> Option<f64> {
        if !self.enabled {
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
    pub fn insert(&mut self, key: CoalitionKey, value: f64) {
        if !self.enabled {
            return;
        }
        let bytes = 32u64; // key + f64 + overhead estimate
        if let Some(max) = self.max_bytes {
            if self.used_bytes + bytes > max && !self.map.contains_key(&key) {
                return;
            }
        }
        if let Some(old) = self.map.insert(key, CacheEntry { value, bytes }) {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        self.used_bytes = self.used_bytes.saturating_add(bytes);
    }

    /// Snapshot statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.map.len() as u64,
            bytes: self.used_bytes,
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

        let mut d = CoalitionCache::disabled();
        d.insert(k, 2.0);
        assert!(d.get(k).is_none());
    }
}
