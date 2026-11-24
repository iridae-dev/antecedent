//! Kani proofs about production Antecedent code, through its public API.
#![cfg_attr(not(kani), allow(dead_code))]

#[cfg(kani)]
mod bound_identity;

#[cfg(kani)]
mod retained_artifact;
