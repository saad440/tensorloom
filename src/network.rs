//! Tensor networks: tensors joined by shared leg identity.
//!
//! A [`TensorNetwork`] holds tensors as graph nodes ([`NodeId`]). The
//! joins are derived, never stored. An [`Index`] whose id occurs on
//! two nodes connects them (an internal leg, summed on contraction),
//! and an id occurring on one node only is an open leg and survives
//! contraction. Legs on three or more nodes are rejected up front
//! (see [`NetworkError::HyperedgeNotSupported`]).
//!
//! Planning is separate from execution for contraction.
//! A [`Path`](crate::path::Path) produced by
//! [`sequential_path`](crate::path::sequential_path) or
//! [`greedy_path`](crate::path::greedy_path) sets the order in which
//! node pairs merge, and [`TensorNetwork::contract`] runs it as
//! pairwise [`crate::contraction::contract`] calls.

use std::collections::{BTreeMap, HashMap, HashSet};

use slotmap::SlotMap;
use thiserror::Error;

use crate::contraction::contract;
use crate::index::Index;
use crate::path::Path;
use crate::scalar::Scalar;
use crate::tensor::{Dense, Tensor};

slotmap::new_key_type! {
    /// Handle to one tensor inside a [`TensorNetwork`].
    /// A handle stays valid until its node is contracted away as the
    /// second input of a step (see [`Path`](crate::path::Path)).
    pub struct NodeId;
}

/// Errors raised when building or contracting a [`TensorNetwork`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NetworkError {
    /// The network holds no tensors, so there is nothing to plan for
    /// or to contract.
    #[error("the network holds no tensors")]
    EmptyNetwork,

    /// A tensor carries the same leg id twice. Self-contractions (a
    /// tensor joined to itself) are not supported yet.
    #[error("a tensor carries leg {id} more than once")]
    DuplicateLegInTensor {
        /// The repeated leg id.
        id: u64,
    },

    /// A leg id occurs on more than two nodes. Pairwise contraction
    /// would sum it before all carriers have contributed, so such
    /// networks are rejected before any arithmetic runs.
    #[error("leg {id} occurs on {multiplicity} nodes; only pairwise legs are supported")]
    HyperedgeNotSupported {
        /// The offending leg id.
        id: u64,
        /// How many nodes carry the leg.
        multiplicity: usize,
    },

    /// The path references a node handle the network does not contain.
    #[error("path references node {node:?}, which is not in the network")]
    UnknownNode {
        /// The unknown node handle.
        node: NodeId,
    },

    /// The path references a node that an earlier step contracted away.
    /// Every step's second input loses its identity. Only the first
    /// input's handle lives on as the intermediate's name.
    #[error("path references node {node:?} after it was contracted away")]
    NodeConsumed {
        /// The consumed node handle.
        node: NodeId,
    },

    /// The path has the wrong number of steps: contracting a network
    /// of `n` nodes takes `n - 1` pairwise merges.
    #[error("path has {actual} steps, but {expected} are needed for this network")]
    PathLengthMismatch {
        /// Number of steps required (node_count - 1).
        expected: usize,
        /// Number of steps the path carries.
        actual: usize,
    },

    /// The path contracts a node with itself in one step. Such a step
    /// has no meaningful merge. A node cannot pair its own legs.
    #[error("path contracts node {node:?} with itself")]
    ContractsWithItself {
        /// The node named on both sides of one step.
        node: NodeId,
    },
}

/// A collection of tensors joined by shared leg identity.
///
/// The graph structure is derived from the nodes' leg lists, so
/// that construction cannot desynchronize structure and data.
#[derive(Debug, Clone)]
pub struct TensorNetwork<S: Scalar> {
    nodes: SlotMap<NodeId, Tensor<S>>,
}

impl<S: Scalar> Default for TensorNetwork<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> TensorNetwork<S> {
    /// An empty network.
    pub fn new() -> Self {
        TensorNetwork {
            // slotmap's `new()` exists only for `DefaultKey`.
            // Custom key types construct with `with_key()`.
            nodes: SlotMap::<NodeId, Tensor<S>>::with_key(),
        }
    }

    /// Add `tensor` as a node and return its handle.
    ///
    /// To share a leg with an existing node, pass a clone of one of
    /// its indices. That is what joins the nodes. Errors with
    /// [`NetworkError::DuplicateLegInTensor`] if the tensor repeats one
    /// of its own leg ids.
    pub fn add_tensor(&mut self, tensor: Tensor<S>) -> Result<NodeId, NetworkError> {
        // Check before inserting so that a rejected tensor is not left
        // inside the network. A repeated id would join the tensor to
        // itself - a self-contraction, which `contract` cannot pair.
        // Ids are unique per leg, so `seen` fails exactly on the
        // second copy.
        let mut seen = HashSet::new();
        for leg in tensor.indices() {
            if !seen.insert(leg.id()) {
                return Err(NetworkError::DuplicateLegInTensor { id: leg.id() });
            }
        }
        Ok(self.nodes.insert(tensor))
    }

    /// The tensor stored at `node`, if present.
    pub fn get(&self, node: NodeId) -> Option<&Tensor<S>> {
        self.nodes.get(node)
    }

    /// The number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the network holds no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all nodes in insertion order, alongside their
    /// handles. Insertion order because currently we do not go
    /// beyond the first generation of the slotmap.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Tensor<S>)> {
        self.nodes.iter()
    }

    /// The open legs: leg ids occurring on exactly one node, in
    /// canonical order - first appearance while scanning nodes in
    /// insertion order and each node's legs in storage order.
    ///
    /// Contraction returns its result with same legs in same order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorloom::index::{Arrow, Index};
    /// use tensorloom::network::TensorNetwork;
    /// use tensorloom::tensor::Tensor;
    ///
    /// let i = Index::new(2, Arrow::Out);
    /// let j = Index::new(3, Arrow::In);
    /// let k = Index::new(4, Arrow::Out);
    /// let mut net = TensorNetwork::new();
    /// net.add_tensor(Tensor::from_fn(&[i.clone(), j.clone()], |_| 0.0)).unwrap();
    /// net.add_tensor(Tensor::from_fn(&[j.clone(), k.clone()], |_| 0.0)).unwrap();
    ///
    /// // j occurs on both tensors (internal); i and k occur once.
    /// // Open legs come back in first-appearance order.
    /// let open = net.open_indices();
    /// assert_eq!(open.len(), 2);
    /// assert_eq!(open[0].id(), i.id());
    /// assert_eq!(open[1].id(), k.id());
    /// ```
    pub fn open_indices(&self) -> Vec<Index> {
        // 1. Multiplicity per id across the network.
        let multiplicity = self.leg_multiplicities();
        // 2. Keep the multiplicity-1 indices. Each occurs exactly once,
        //    so the scan order itself (nodes by insertion, legs by
        //    storage position) already is the canonical order.
        let mut open = Vec::new();
        for node in self.nodes.values() {
            for leg in node.indices() {
                if multiplicity[&leg.id()] == 1 {
                    open.push(leg.clone());
                }
            }
        }
        open
    }

    /// Multiplicity of each leg id across the whole network: how many
    /// legs carry it, over all nodes.
    ///
    /// Derived per call from the leg lists - the single source of truth.
    /// If a cache is ever needed, it would be added inside this function,
    /// behind the same signature, so that callers stay unchanged.
    /// BTreeMap so iteration order is deterministic.
    pub(crate) fn leg_multiplicities(&self) -> BTreeMap<u64, usize> {
        let mut multiplicity: BTreeMap<u64, usize> = BTreeMap::new();
        for node in self.nodes.values() {
            for leg in node.indices() {
                *multiplicity.entry(leg.id()).or_insert(0) += 1;
            }
        }
        multiplicity
    }

    /// Reject malformed structure up front:
    /// - ids repeated inside a node.
    /// - ids carried by more than two nodes.
    ///
    /// `add_tensor` already rejects within-tensor duplicates at build time.
    /// This re-checks them so validity holds for a network however it was assembled.
    pub(crate) fn validate_structure(&self) -> Result<(), NetworkError> {
        // 1. Within-tensor duplicates (self-contractions). Checked per
        //    node - a different question than the network-wide
        //    multiplicity below, so it keeps its own pass.
        //    `add_tensor` already rejects them at build time. This
        //    re-checks so validity holds however the network was
        //    assembled.
        for node in self.nodes.values() {
            let mut seen = HashSet::new();
            for leg in node.indices() {
                if !seen.insert(leg.id()) {
                    return Err(NetworkError::DuplicateLegInTensor { id: leg.id() });
                }
            }
        }
        // 2. Legs on more than two nodes: a leg shared by exactly two
        //    nodes is the normal internal leg, summed when its two
        //    carriers meet in one pairwise contraction. Three or more
        //    carriers would be summed before all of them contributed,
        //    so the network is rejected here instead of computing a
        //    wrong value. BTreeMap iteration makes the error that
        //    surfaces first deterministic (the smallest offending id).
        for (id, count) in self.leg_multiplicities() {
            if count > 2 {
                return Err(NetworkError::HyperedgeNotSupported {
                    id,
                    multiplicity: count,
                });
            }
        }
        Ok(())
    }

    /// Contract the whole network down to one tensor, following `path`.
    ///
    /// The path must take the network from `n` nodes to one in `n - 1`
    /// steps, referencing only live nodes (see
    /// [`crate::path::Path`]). The result carries the open legs in
    /// canonical order (see [`TensorNetwork::open_indices`]).
    ///
    /// # Errors
    ///
    /// See [`NetworkError`]. All validation - empty network, malformed
    /// structure, invalid path - happens before any arithmetic runs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorloom::index::{Arrow, Index};
    /// use tensorloom::network::TensorNetwork;
    /// use tensorloom::path::Path;
    /// use tensorloom::tensor::Tensor;
    ///
    /// let i = Index::new(2, Arrow::Out);
    /// let j = Index::new(3, Arrow::In);
    /// let k = Index::new(4, Arrow::Out);
    /// let mut net = TensorNetwork::new();
    /// let a = net
    ///     .add_tensor(Tensor::from_fn(&[i.clone(), j.clone()], |c| {
    ///         (c[0] + 2 * c[1]) as f64
    ///     }))
    ///     .unwrap();
    /// let b = net
    ///     .add_tensor(Tensor::from_fn(&[j.clone(), k.clone()], |c| {
    ///         c[0] as f64 - c[1] as f64
    ///     }))
    ///     .unwrap();
    ///
    /// // One step: merge a and b. The result is exactly
    /// // contraction::contract of the two tensors, with legs in
    /// // canonical order [i, k].
    /// let result = net.contract(&Path::new(vec![[a, b]])).unwrap();
    /// assert_eq!(result.rank(), 2);
    /// assert_eq!(result.indices()[0].id(), i.id());
    /// assert_eq!(result.indices()[1].id(), k.id());
    /// assert_eq!(result.storage().as_array()[[0, 0]], 10.0);
    /// ```
    pub fn contract(self, path: &Path) -> Result<Tensor<S>, NetworkError> {
        // 1. Up-front validation, all of it before any arithmetic:
        //    empty network, malformed structure, then the path dry run.
        //    Paths are data that may arrive from elsewhere (a cache, a
        //    file), so they are checked before any tensor moves.
        if self.nodes.is_empty() {
            return Err(NetworkError::EmptyNetwork);
        }
        self.validate_structure()?;
        // canonical: the result's required leg order (open legs,
        // first appearance).
        let canonical = self.open_indices();
        self.validate_path(path)?;
        // 2. Execute with a live map keyed by the original handles:
        //    each step pops both inputs, contracts them, and stores the
        //    result under the step's first handle. Structure validation
        //    bounded every leg at two carriers, and merges only remove
        //    ids in complete pairs - so whenever a pair shares a leg,
        //    the pair carries it exactly twice and the pairwise sum is
        //    exact. No per-step leg checks needed.
        let mut live: HashMap<NodeId, Tensor<S>> = self.nodes.into_iter().collect();
        for &[x, y] in path.steps() {
            let a = live
                .remove(&x)
                .expect("internal bug: the path dry run missed an invalid step");
            let b = live
                .remove(&y)
                .expect("internal bug: the path dry run missed an invalid step");
            let merged = contract(&a, &b);
            live.insert(x, merged);
        }
        // 3. One node remains: its legs are the open legs, in whatever
        //    order this path happened to produce. Reordering into the
        //    canonical order makes results line up with `open_indices`
        //    (and with reference data) for every valid path.
        let survivor = live
            .into_values()
            .next()
            .expect("internal bug: n - 1 merges did not leave one node");
        // order: order[k] = the survivor's axis that currently carries
        // canonical[k]; fed to permuted_axes below.
        let mut order = Vec::with_capacity(canonical.len());
        for leg in &canonical {
            let axis = survivor
                .indices()
                .iter()
                .position(|l| l.id() == leg.id())
                .expect("internal bug: an open leg did not survive contraction");
            order.push(axis);
        }
        let data = survivor
            .storage()
            .as_array()
            .view()
            .permuted_axes(order)
            .into_owned();
        // The `.expect` is sound: the survivor's legs are exactly the
        // open legs - same ids, hence same dims - so the product rule
        // holds by construction.
        Ok(Tensor::new(canonical, Dense::from_array(data))
            .expect("internal bug: survivor legs do not satisfy the product rule"))
    }

    /// Dry-run the path over handles only: the right number of steps,
    /// every handle known, none referenced after its node was consumed
    /// as a step's second input. Catches a step contracting a node
    /// with itself, too.
    fn validate_path(&self, path: &Path) -> Result<(), NetworkError> {
        let expected = self.nodes.len() - 1;
        let actual = path.steps().len();
        if actual != expected {
            return Err(NetworkError::PathLengthMismatch { expected, actual });
        }
        // Extinction replay: a handle dies when it is the second input
        // of a step; the first input lives on as the intermediate's
        // name. References to dead handles are what the executor would
        // otherwise trip over.
        let mut consumed: HashSet<NodeId> = HashSet::new();
        for &[x, y] in path.steps() {
            if !self.nodes.contains_key(x) {
                return Err(NetworkError::UnknownNode { node: x });
            }
            if !self.nodes.contains_key(y) {
                return Err(NetworkError::UnknownNode { node: y });
            }
            if x == y {
                return Err(NetworkError::ContractsWithItself { node: x });
            }
            if consumed.contains(&x) {
                return Err(NetworkError::NodeConsumed { node: x });
            }
            if consumed.contains(&y) {
                return Err(NetworkError::NodeConsumed { node: y });
            }
            consumed.insert(y);
        }
        Ok(())
    }

    /// Find an order with [`crate::path::greedy_path`] and run it.
    pub fn contract_greedy(self) -> Result<Tensor<S>, NetworkError> {
        let path = crate::path::greedy_path(&self)?;
        self.contract(&path)
    }
}
