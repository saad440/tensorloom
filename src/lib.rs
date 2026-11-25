//! TensorLoom — a tensor network library (early development).
//!
//! Layers so far: [`index`] (diagrammatic leg identity), [`scalar`]
//! (supported element types), [`tensor`] (dense tensors),
//! [`contraction`] (id-based pairwise contraction), and [`network`]
//! with [`path`] (tensors joined into networks, plus their contraction
//! orders).

#![warn(missing_docs)]

pub(crate) mod gemm;
pub mod contraction;
pub mod index;
pub mod network;
pub mod path;
pub mod scalar;
pub mod tensor;

pub use index::{Arrow, Index, IndexMeta};
pub use network::{NetworkError, NodeId, TensorNetwork};
pub use path::Path;
pub use scalar::Scalar;
pub use tensor::{Dense, Tensor, TensorError};
