//! Semantic coalition evaluation cache.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::CachePolicy;

use crate::error::AttributionError;
use crate::result::CacheStats;

/// Exact-mode dense mask index is enabled at `k ≤` this (2^16 slots).
/// Larger exact games stay on the `HashMap`; 2^22 as a cutoff would be gigabytes.
pub(crate) const DENSE_MASK_MAX_PLAYERS: usize = 16;

/// Most players a game may have. A structural sanity bound on a declaration: the work of a
/// sampled game is bounded by its sample budget, and of an exact game by
/// [`MAX_EXACT_COALITION_PLAYERS`] and the caller's exact-size limit.
pub(crate) const MAX_COALITION_PLAYERS: usize = 1 << 20;

/// Most players an exact game may have: its `2^k` coalitions are enumerated by `u64` index.
pub(crate) const MAX_EXACT_COALITION_PLAYERS: usize = 63;

/// Refuse a player count above [`MAX_COALITION_PLAYERS`].
///
/// # Errors
///
/// [`AttributionError::SizeLimit`] when `requested` exceeds [`MAX_COALITION_PLAYERS`].
pub(crate) fn ensure_player_bitset(requested: usize) -> Result<(), AttributionError> {
    if requested > MAX_COALITION_PLAYERS {
        Err(AttributionError::SizeLimit {
            kind: "components",
            requested,
            max: MAX_COALITION_PLAYERS,
        })
    } else {
        Ok(())
    }
}

/// Full coalition of `n_players` players.
pub(crate) fn full_coalition_mask(n_players: usize) -> Result<Coalition, AttributionError> {
    ensure_player_bitset(n_players)?;
    Ok(Coalition::full(n_players))
}

/// A set of players, as a bitset over player indices of any width.
///
/// Players `0..64` live in one inline word, so a game of at most 64 players never allocates;
/// further players spill into `high`, whose trailing zero words are trimmed so that equal sets
/// are equal values (and hash alike) however they were built.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct Coalition {
    low: u64,
    high: Vec<u64>,
}

impl Coalition {
    /// The empty coalition.
    #[must_use]
    pub const fn empty() -> Self {
        Self { low: 0, high: Vec::new() }
    }

    /// The coalition whose members are the set bits of `bits` (bit `i` is player `i`).
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self { low: bits, high: Vec::new() }
    }

    /// The coalition of players `0..n_players`.
    #[must_use]
    pub fn full(n_players: usize) -> Self {
        let words = n_players.div_ceil(64);
        let remainder = n_players % 64;
        let word_at = |w: usize| {
            if w + 1 == words && remainder != 0 { (1u64 << remainder) - 1 } else { u64::MAX }
        };
        Self {
            low: if words == 0 { 0 } else { word_at(0) },
            high: (1..words).map(word_at).collect(),
        }
    }

    /// Whether `player` is a member.
    #[must_use]
    pub fn contains(&self, player: usize) -> bool {
        let bit = 1u64 << (player % 64);
        if player < 64 {
            self.low & bit != 0
        } else {
            self.high.get(player / 64 - 1).is_some_and(|word| word & bit != 0)
        }
    }

    /// Add `player`.
    pub fn insert(&mut self, player: usize) {
        let bit = 1u64 << (player % 64);
        if player < 64 {
            self.low |= bit;
        } else {
            let word = player / 64 - 1;
            if self.high.len() <= word {
                self.high.resize(word + 1, 0);
            }
            self.high[word] |= bit;
        }
    }

    /// Remove `player`.
    pub fn remove(&mut self, player: usize) {
        let bit = 1u64 << (player % 64);
        if player < 64 {
            self.low &= !bit;
        } else if let Some(word) = self.high.get_mut(player / 64 - 1) {
            *word &= !bit;
            while self.high.last() == Some(&0) {
                self.high.pop();
            }
        }
    }

    /// This coalition with `player` added.
    #[must_use]
    pub fn with(mut self, player: usize) -> Self {
        self.insert(player);
        self
    }

    /// This coalition with `player` removed.
    #[must_use]
    pub fn without(mut self, player: usize) -> Self {
        self.remove(player);
        self
    }

    /// Number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.low.count_ones() as usize
            + self.high.iter().map(|word| word.count_ones() as usize).sum::<usize>()
    }

    /// Whether no player is a member.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.low == 0 && self.high.is_empty()
    }

    /// The membership of players `0..64` as bits.
    #[must_use]
    pub const fn low_bits(&self) -> u64 {
        self.low
    }

    /// The whole coalition as bits, or `None` when a player of index 64 or above is a member.
    #[must_use]
    pub fn to_bits(&self) -> Option<u64> {
        self.high.is_empty().then_some(self.low)
    }

    /// Members in increasing order.
    pub fn players(&self) -> impl Iterator<Item = usize> + '_ {
        std::iter::once((0usize, self.low))
            .chain(self.high.iter().copied().enumerate().map(|(w, word)| (w + 1, word)))
            .flat_map(|(w, mut word)| {
                std::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(w * 64 + bit)
                })
            })
    }

    /// Players that are in exactly one of `self` and `other`, in increasing order.
    #[must_use]
    pub fn symmetric_difference(&self, other: &Self) -> Vec<usize> {
        let words = self.high.len().max(other.high.len());
        let word = |c: &Self, w: usize| {
            if w == 0 { c.low } else { c.high.get(w - 1).copied().unwrap_or(0) }
        };
        let mut out = Vec::new();
        for w in 0..=words {
            let mut diff = word(self, w) ^ word(other, w);
            while diff != 0 {
                out.push(w * 64 + diff.trailing_zeros() as usize);
                diff &= diff - 1;
            }
        }
        out
    }
}

/// Key for a coalition / substitution evaluation.
///
/// `mask` selects which components use the comparison (new) mechanism;
/// `tag` distinguishes baseline-vs-comparison model pairings or path contexts.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct CoalitionKey {
    /// Members: the active (comparison) components.
    pub mask: Coalition,
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
pub(crate) struct CoalitionCache {
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
    /// sparse subset of masks, leave this unset and stay on the `HashMap`.
    pub fn enable_dense_index(&mut self, n_players: usize) {
        if !self.enabled || n_players == 0 || n_players > DENSE_MASK_MAX_PLAYERS {
            return;
        }
        let slots = 1usize << n_players;
        // The dense index allocates every slot eagerly, even for a cache whose
        // budget admits only one payoff. Keep small budgets on the sparse map;
        // choosing an index must not itself exceed the cache's byte allowance.
        let index_bytes = (slots * std::mem::size_of::<Option<CacheEntry>>()) as u64;
        if self.max_bytes.is_some_and(|max| index_bytes > max) {
            return;
        }
        self.dense = Some(vec![None; slots]);
    }

    /// Disabled cache (always miss).
    #[must_use]
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::from_policy(CachePolicy::disabled())
    }

    fn dense_slot(&self, key: &CoalitionKey) -> Option<usize> {
        if key.tag != 0 {
            return None;
        }
        let dense = self.dense.as_ref()?;
        let idx = usize::try_from(key.mask.to_bits()?).ok()?;
        (idx < dense.len()).then_some(idx)
    }

    /// Lookup a cached payoff.
    pub fn get(&mut self, key: &CoalitionKey) -> Option<f64> {
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
        if let Some(e) = self.map.get(key) {
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
        let replacing = if let Some(idx) = self.dense_slot(&key) {
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
        if let Some(idx) = self.dense_slot(&key) {
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
            self.dense.as_ref().map_or(0, |d| d.iter().filter(|e| e.is_some()).count());
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
        let k = CoalitionKey { mask: Coalition::from_bits(0b101), tag: 1 };
        assert!(c.get(&k).is_none());
        c.insert(k.clone(), 1.5);
        assert_eq!(c.get(&k), Some(1.5));
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!(!s.saturated);

        let mut d = CoalitionCache::disabled();
        d.insert(k.clone(), 2.0);
        assert!(d.get(&k).is_none());
    }

    #[test]
    fn full_coalition_covers_the_word_boundaries() {
        assert!(full_coalition_mask(0).unwrap().is_empty());
        assert_eq!(full_coalition_mask(3).unwrap().to_bits(), Some(0b111));
        assert_eq!(full_coalition_mask(64).unwrap().to_bits(), Some(u64::MAX));
        let wide = full_coalition_mask(130).unwrap();
        assert_eq!(wide.len(), 130);
        assert!(wide.contains(129) && !wide.contains(130));
        assert_eq!(wide.to_bits(), None);
        assert!(matches!(
            full_coalition_mask(MAX_COALITION_PLAYERS + 1),
            Err(AttributionError::SizeLimit { .. })
        ));
    }

    #[test]
    fn wide_coalitions_are_canonical_sets() {
        let mut a = Coalition::empty().with(3).with(64).with(200);
        let b = Coalition::from_bits(1 << 3).with(200).with(64);
        assert_eq!(a, b);
        assert_eq!(a.players().collect::<Vec<_>>(), vec![3, 64, 200]);
        a.remove(200);
        a.remove(64);
        assert_eq!(a, Coalition::from_bits(1 << 3), "trailing zero words are trimmed");
        assert_eq!(a.to_bits(), Some(8));
        let c = Coalition::empty().with(70);
        assert_eq!(a.symmetric_difference(&c), vec![3, 70]);
        assert_eq!(Coalition::full(70).without(69).len(), 69);
    }

    #[test]
    fn dense_index_skips_hash_for_tag_zero() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(10_000)));
        c.enable_dense_index(3);
        let k = CoalitionKey { mask: Coalition::from_bits(0b101), tag: 0 };
        c.insert(k.clone(), 1.5);
        assert_eq!(c.get(&k), Some(1.5));
        assert_eq!(c.stats().entries, 1);
        assert!(c.map.is_empty(), "tag-0 exact inserts must not land in the HashMap");
    }

    #[test]
    fn dense_index_respects_small_cache_budgets_without_changing_payoffs() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(32)));
        c.enable_dense_index(DENSE_MASK_MAX_PLAYERS);
        assert!(c.dense.is_none());
        let first = CoalitionKey { mask: Coalition::empty(), tag: 0 };
        let second = CoalitionKey { mask: Coalition::from_bits(1), tag: 0 };
        c.insert(first.clone(), 1.5);
        c.insert(second.clone(), 2.5);
        assert_eq!(c.get(&first), Some(1.5));
        assert!(c.get(&second).is_none());
        assert_eq!(c.stats().bytes, 32);
        assert!(c.stats().saturated);
    }

    #[test]
    fn refuse_when_full_sets_saturated() {
        let mut c = CoalitionCache::from_policy(CachePolicy::enabled(Some(32)));
        let k1 = CoalitionKey { mask: Coalition::from_bits(1), tag: 1 };
        let k2 = CoalitionKey { mask: Coalition::from_bits(2), tag: 1 };
        c.insert(k1, 1.0);
        c.insert(k2.clone(), 2.0);
        assert!(c.stats().saturated);
        assert_eq!(c.stats().entries, 1);
        assert!(c.get(&k2).is_none());
    }
}
