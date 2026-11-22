//! Indices: the diagrammatic identity of tensor legs.
//!
//! An [`Index`] bundles a unique id, a dimension, a direction ([`Arrow`])
//! and optional metadata ([`IndexMeta`]).
//! Two indices contract iff their ids are equal.
//!
//! The id plays the role of the repeated letter in Einstein-summation
//! notation. Sharing a cloned `Index` between tensors is writing the
//! same letter on two legs, and [`crate::contraction::contract`] sums
//! over such shared legs.

use std::sync::atomic::{AtomicU64, Ordering};

/// Direction tag of a leg. In bra-ket terms: [`Arrow::Out`] marks a
/// ket-side leg (pointing out of the tensor), [`Arrow::In`] a bra-side
/// leg (pointing in). For now, contraction matches on id alone.
/// Pairing opposite arrows arrives in a future version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arrow {
    /// Arrow pointing into a tensor.
    In,
    /// Arrow pointing out of a tensor.
    Out,
}

impl Arrow {
    /// Flip Arrow direction.
    pub fn flip(self) -> Self {
        match self {
            Arrow::In => Arrow::Out,
            Arrow::Out => Arrow::In,
        }
    }
}

/// Optional per-index metadata.
/// default: [`IndexMeta::Trivial`], meaning "no structure".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexMeta {
    /// No structure. The default for now.
    Trivial,
}

/// Diagrammatic identity of a leg: equality is id equality.
///
/// Cloning an index copies the id. A clone contracts with the original.
/// Construct only via [`Index::new`] so ids stay unique.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index {
    id: u64,
    dim: usize,
    arrow: Arrow,
    meta: IndexMeta,
}

/// Process-wide monotonic id source. `static` because one fixed global
/// storage cell (not fixed value): `fetch_add` mutates it at runtime,
/// so every [`Index::new`] call hands out a fresh id. The atomic makes
/// that thread-safe without unsafe code. `Relaxed` ordering suffices
/// because we need uniqueness only, never ordering.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Index {
    /// Allocate a fresh index with dimension `dim > 0` and
    /// [`IndexMeta::Trivial`] metadata.
    ///
    /// Panics if `dim == 0`. A zero-dimensional leg cannot occur in a
    /// tensor network. Rank-0 tensors carry no index.
    pub fn new(dim: usize, arrow: Arrow) -> Self {
        assert!(dim > 0, "index dimension must be nonzero");
        Index {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            dim,
            arrow,
            meta: IndexMeta::Trivial,
        }
    }

    /// The unique identity. Two indices are equal iff these are equal.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The dimension (number of possible values along this leg).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The arrow direction tag.
    pub fn arrow(&self) -> Arrow {
        self.arrow
    }

    /// The metadata variant.
    pub fn meta(&self) -> IndexMeta {
        self.meta
    }

    /// Builder: replace the metadata.
    pub fn with_meta(mut self, meta: IndexMeta) -> Self {
        self.meta = meta;
        self
    }

    /// Builder: flip the arrow.
    pub fn with_flipped_arrow(mut self) -> Self {
        self.arrow = self.arrow.flip();
        self
    }
}
