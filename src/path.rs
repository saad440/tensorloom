//! A [`Path`] fixes the order in which a [`TensorNetwork`]'s nodes
//! merge: a sequence of node-handle pairs. It is plain data -
//! inspectable, cacheable, serializable later - so planning
//! ([`sequential_path`], [`greedy_path`]) is separable from execution
//! ([`TensorNetwork::contract`]).
//!
//! Two conventions tie paths to execution and make plans exact:
//!
//! - Steps reference the handles handed out at construction time. When
//!   a step contracts `x` and `y`, the intermediate is stored in `x`.
//!   Later steps naming `x` contract the intermediate, while `y`'s
//!   handle is consumed and must never be referenced again.
//! - Contracting `n` nodes takes `n - 1` steps.

use std::collections::BTreeMap;

use crate::network::{NetworkError, NodeId, TensorNetwork};
use crate::scalar::Scalar;

/// A contraction order: the sequence of node-pair merges.
///
/// Paths are scalar-agnostic but instance-bound: the handles refer to
/// one specific network, and [`TensorNetwork::contract`] validates a
/// path against the network it is applied to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    steps: Vec<[NodeId; 2]>,
}

impl Path {
    /// A path from raw steps, in execution order.
    pub fn new(steps: Vec<[NodeId; 2]>) -> Self {
        Path { steps }
    }

    /// The steps, in execution order.
    pub fn steps(&self) -> &[[NodeId; 2]] {
        &self.steps
    }
}

/// The trivial order: fold nodes in insertion order - `(n0, n1)`, then
/// `(n0, n2)`, and so on, with the first handle carrying the running
/// result.
/// Fine for chains, but its cost explodes on wide networks.
pub fn sequential_path<S: Scalar>(network: &TensorNetwork<S>) -> Result<Path, NetworkError> {
    if network.is_empty() {
        return Err(NetworkError::EmptyNetwork);
    }
    // Fold in insertion order: the first handle carries the running
    // result, every following node merges into it. The intermediate
    // keeps living at the first handle, so one constant name folds
    // the whole chain.
    let mut handles = network.iter().map(|(handle, _)| handle);
    let first = handles
        .next()
        .expect("internal bug: network checked non-empty above");
    let steps: Vec<[NodeId; 2]> = handles.map(|h| [first, h]).collect();
    Ok(Path::new(steps))
}

/// A greedy order: repeatedly contract the cheapest live pair.
///
/// Cost model per candidate pair: an estimate of the flop count,
/// `|result| * |shared|` - the result's element count times the
/// contraction depth - computed from leg ids and dimensions only.
/// Ties break on the smaller result size, then on the handles in
/// insertion order, so the same network always plans identically.
pub fn greedy_path<S: Scalar>(network: &TensorNetwork<S>) -> Result<Path, NetworkError> {
    // Fail fast on malformed structure.
    network.validate_structure()?;
    if network.is_empty() {
        return Err(NetworkError::EmptyNetwork);
    }
    // Planning is number-free: per-node leg summaries (id, dim) carry
    // everything the cost model needs, no tensor values involved.
    // BTreeMap iterates in slot order (= insertion order here), which
    // is what the tie-break leans on.
    // live: handle -> [(leg id, dim), ...] for that node.
    // Entries change as merges happen (merged = union minus shared).
    let mut live: BTreeMap<NodeId, Vec<(u64, usize)>> = BTreeMap::new();
    for (handle, tensor) in network.iter() {
        live.insert(
            handle,
            tensor
                .indices()
                .iter()
                .map(|leg| (leg.id(), leg.dim()))
                .collect(),
        );
    }
    let mut steps: Vec<[NodeId; 2]> = Vec::with_capacity(network.len() - 1);
    while live.len() > 1 {
        // Scan all pairs (x, y), x < y in insertion order. The cost of
        // contracting a pair is the same product one pairwise `contract`
        // multiplies through: result element count times contraction
        // depth. `best` compares (cost, result size, x, y)
        // lexicographically, so the same network always plans the same
        // way. u128 keeps the products overflow-free.
        let mut best: Option<(u128, u128, NodeId, NodeId)> = None;
        let keys: Vec<NodeId> = live.keys().copied().collect();
        for i in 0..keys.len() {
            for j in i + 1..keys.len() {
                let (x, y) = (keys[i], keys[j]);
                let legs_x = &live[&x];
                let legs_y = &live[&y];
                let mut shared_dim = 1u128; // product of shared legs' dims
                let mut result_dim = 1u128; // product of surviving legs' dims
                for &(id, dim) in legs_x {
                    if legs_y.iter().any(|&(other, _)| other == id) {
                        shared_dim *= dim as u128;
                    } else {
                        result_dim *= dim as u128;
                    }
                }
                for &(id, dim) in legs_y {
                    if !legs_x.iter().any(|&(other, _)| other == id) {
                        result_dim *= dim as u128;
                    }
                }
                let candidate = (result_dim * shared_dim, result_dim, x, y);
                if best.is_none_or(|prior| candidate < prior) {
                    best = Some(candidate);
                }
            }
        }
        // Merge the pair: the result's legs are the union minus the
        // shared ids, and the intermediate is stored under x, so
        // the emitted path is exactly executable.
        let (_, _, x, y) =
            best.expect("internal bug: no pair exists while more than one node is live");
        let mut merged = live
            .remove(&x)
            .expect("internal bug: x is not in the live set");
        for (id, dim) in live
            .remove(&y)
            .expect("internal bug: y is not in the live set")
        {
            if !merged.iter().any(|&(kept, _)| kept == id) {
                merged.push((id, dim));
            }
        }
        live.insert(x, merged);
        steps.push([x, y]);
    }
    Ok(Path::new(steps))
}
