//! TensorLoom — a tensor network library (early development).
//!
//! Layers so far: [`index`] (diagrammatic leg identity), [`scalar`]
//! (supported element types), [`tensor`] (dense tensors), and
//! [`contraction`] (id-based contraction).

#![warn(missing_docs)]

pub mod index;
pub mod scalar;
pub mod tensor;

pub use index::{Arrow, Index, IndexMeta};
pub use scalar::Scalar;
pub use tensor::{Dense, Tensor, TensorError};
