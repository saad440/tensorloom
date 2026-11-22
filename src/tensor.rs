//! Dense tensors: an ordered index list paired with dense storage.
//!
//! Axis `k` of the storage corresponds to the `k`-th entry of the index
//! list. Index identity ([`crate::index::Index`]) is what contraction
//! matches on. Storage order is fixed at construction and is the tensor's
//! own business until [`crate::contraction::contract`] needs to permute.
//!
//! Number-free descriptions live one abstraction up: the leg list alone
//! (a `Vec<Index>`) already describes a tensor's structure, and a
//! dedicated structure-only type with shape algebra is added alongside
//! the network layer, when contraction planning needs it. `Tensor` keeps
//! requiring data. [`Dense`] is its only storage backend for now.

use ndarray::{ArrayD, IxDyn};
use thiserror::Error;

use crate::index::Index;
use crate::scalar::Scalar;

/// Errors raised when constructing tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TensorError {
    /// The storage length does not match the product of index dimensions.
    #[error("storage holds {actual} elements, but the indices describe {expected}")]
    DimensionMismatch {
        /// Product of the index dimensions.
        expected: usize,
        /// Number of elements in the storage.
        actual: usize,
    },
}

/// Dense storage: the raw numeric payload of a [`Tensor`].
///
/// The only storage backend for now. A storage trait abstracting over
/// dense and block-sparse layouts arrives later. `Tensor` will become
/// generic over it at that point, without changing the index layer.
#[derive(Debug, Clone)]
pub struct Dense<S> {
    /// The wrapped array. Axis `k` carries the `k`-th tensor index.
    data: ArrayD<S>,
}

impl<S> Dense<S> {
    /// Wrap an array as dense storage.
    pub fn from_array(data: ArrayD<S>) -> Self {
        Dense { data }
    }

    /// View the storage as an array (axis `k` = index `k` of the tensor).
    pub fn as_array(&self) -> &ArrayD<S> {
        &self.data
    }

    /// Mutable view of the storage. Shape must stay consistent with the
    /// tensor's index list.
    pub fn as_array_mut(&mut self) -> &mut ArrayD<S> {
        &mut self.data
    }

    /// Unwrap into the underlying array.
    pub fn into_array(self) -> ArrayD<S> {
        self.data
    }
}

/// A tensor: an ordered list of [`Index`]es plus dense storage whose axis
/// `k` corresponds to the `k`-th index.
///
/// Indices are stored by value; sharing an index (clone) across two
/// tensors is exactly what makes them contractible.
#[derive(Debug, Clone)]
pub struct Tensor<S: Scalar> {
    indices: Vec<Index>,
    storage: Dense<S>,
}

impl<S: Scalar> Tensor<S> {
    /// Build a tensor over `indices` from `storage`.
    ///
    /// Errors if the storage does not hold exactly one element per
    /// multi-index (the product of the index dimensions).
    pub fn new(indices: Vec<Index>, storage: Dense<S>) -> Result<Self, TensorError> {
        let expected: usize = indices.iter().map(Index::dim).product();
        let actual = storage.as_array().len();
        if expected != actual {
            return Err(TensorError::DimensionMismatch { expected, actual });
        }
        Ok(Tensor { indices, storage })
    }

    /// A rank-0 tensor holding `value` (a scalar viewed as a tensor).
    pub fn scalar(value: S) -> Self {
        Tensor {
            indices: Vec::new(),
            storage: Dense {
                data: ArrayD::from_elem(IxDyn(&[]), value),
            },
        }
    }

    /// A tensor with every entry equal to `S::zero()`.
    pub fn zeros(indices: &[Index]) -> Self {
        let shape: Vec<usize> = indices.iter().map(Index::dim).collect();
        let storage = Dense {
            data: ArrayD::from_elem(IxDyn(&shape), S::zero()),
        };
        Self::new(indices.to_vec(), storage).expect("shape derived from indices")
    }

    /// A tensor with entries `f(coord)`, where `coord` is the row-major
    /// multi-index: one coordinate per index, last axis fastest, matching
    /// ndarray's default layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorloom::index::{Arrow, Index};
    /// use tensorloom::tensor::Tensor;
    ///
    /// let i = Index::new(2, Arrow::Out);
    /// let j = Index::new(3, Arrow::In);
    /// let t = Tensor::from_fn(&[i.clone(), j.clone()], |c| (c[0] + c[1]) as f64);
    /// assert_eq!(t.rank(), 2);
    /// assert_eq!(t.storage().as_array().len(), 6);
    /// ```
    pub fn from_fn(indices: &[Index], mut f: impl FnMut(&[usize]) -> S) -> Self {
        let shape: Vec<usize> = indices.iter().map(Index::dim).collect();
        let total: usize = shape.iter().product();
        let mut data = Vec::with_capacity(total);
        let mut coord = vec![0usize; shape.len()];
        for _ in 0..total {
            data.push(f(&coord));
            // Odometer increment in C order (last axis fastest).
            for axis in (0..coord.len()).rev() {
                coord[axis] += 1;
                if coord[axis] < shape[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
        let storage = Dense {
            data: ArrayD::from_shape_vec(IxDyn(&shape), data).expect("shape matches data"),
        };
        Self::new(indices.to_vec(), storage).expect("shape derived from indices")
    }

    /// The index list, in storage axis order.
    pub fn indices(&self) -> &[Index] {
        &self.indices
    }

    /// The storage (numbers only; see [`Dense`]).
    pub fn storage(&self) -> &Dense<S> {
        &self.storage
    }

    /// Mutable access to the storage. The index list is unchanged, so the
    /// array shape must remain as constructed.
    pub fn storage_mut(&mut self) -> &mut Dense<S> {
        &mut self.storage
    }

    /// The number of indices (tensor rank).
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Split into the index list and the storage.
    pub fn into_parts(self) -> (Vec<Index>, Dense<S>) {
        (self.indices, self.storage)
    }
}
