//! Dense tensors: an ordered index list paired with dense storage.
//!
//! Axis `k` of the storage corresponds to the `k`-th entry of the index
//! list. Index identity ([`crate::index::Index`]) is what contraction
//! matches on. Storage order is fixed at construction and is the tensor's
//! own business until [`crate::contraction::contract`] needs to permute.
//!
//! Number-free descriptions live one abstraction up: the leg list alone
//! (a `Vec<Index>`) already describes a tensor's structure, and a
//! dedicated structure-only type with shape algebra joins the public
//! API when a public consumer needs one. Contraction planning works on
//! the leg list directly for now. `Tensor` keeps requiring data.
//! [`Dense`] is its only storage backend for now.

use std::fmt;

use ndarray::{ArrayD, IxDyn};
use thiserror::Error;

use crate::index::{Arrow, Index};
use crate::scalar::Scalar;

/// Errors raised when constructing tensors.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TensorError {
    /// The storage shape does not match the index list: the rank or
    /// some axis length disagrees.
    #[error("indices describe shape {expected:?}, but storage has shape {actual:?}")]
    ShapeMismatch {
        /// The shape the index list describes.
        expected: Vec<usize>,
        /// The shape of the storage array.
        actual: Vec<usize>,
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
    /// Errors if the storage shape does not match the index list -
    /// rank and every axis length must agree, so that axis `k` of the
    /// array carries the `k`-th index.
    pub fn new(indices: Vec<Index>, storage: Dense<S>) -> Result<Self, TensorError> {
        let expected: Vec<usize> = indices.iter().map(Index::dim).collect();
        let actual: Vec<usize> = storage.as_array().shape().to_vec();
        if expected != actual {
            return Err(TensorError::ShapeMismatch { expected, actual });
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

impl<S: Scalar + fmt::Display> fmt::Display for Tensor<S> {
    /// Human readable summary: the leg list, then the data - as a
    /// column-aligned grid for ranks ≤ 2, in flat storage order for
    /// higher ranks.
    ///
    /// Any axis longer than `2 * DISPLAY_EDGE` items is truncated: the
    /// first and last `DISPLAY_EDGE` items are shown with an ellipsis
    /// for the skipped interior.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensorloom::index::{Arrow, Index};
    /// use tensorloom::tensor::Tensor;
    ///
    /// let i = Index::new(2, Arrow::Out);
    /// let j = Index::new(3, Arrow::In);
    /// let t = Tensor::from_fn(&[i, j], |c| (c[0] + c[1]) as f64);
    /// assert_eq!(
    ///     format!("{t}"),
    ///     "Tensor[2 out, 3 in]\n[[0, 1, 2]\n [1, 2, 3]]"
    /// );
    /// ```
    ///
    /// An 8×5 tensor is truncated per axis:
    ///
    /// ```
    /// use tensorloom::index::{Arrow, Index};
    /// use tensorloom::tensor::Tensor;
    ///
    /// let i = Index::new(8, Arrow::Out);
    /// let j = Index::new(5, Arrow::In);
    /// let t = Tensor::from_fn(&[i, j], |c| (10 * c[0] + c[1]) as f64);
    /// assert_eq!(
    ///     format!("{t}"),
    ///     "Tensor[8 out, 5 in]\n[[ 0,  1,  2,  3,  4]\n [10, 11, 12, 13, 14]\n [20, 21, 22, 23, 24]\n [ …,  …,  …,  …,  …]\n [50, 51, 52, 53, 54]\n [60, 61, 62, 63, 64]\n [70, 71, 72, 73, 74]]"
    /// );
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[")?;
        for (axis, leg) in self.indices.iter().enumerate() {
            if axis > 0 {
                write!(f, ", ")?;
            }
            let dir = match leg.arrow() {
                Arrow::In => "in",
                Arrow::Out => "out",
            };
            write!(f, "{} {dir}", leg.dim())?;
        }
        write!(f, "]")?;

        let data = self.storage.as_array();
        match self.rank() {
            0 => {
                let value = data
                    .iter()
                    .next()
                    .expect("rank-0 storage holds one element");
                write!(f, "\n{value}")?;
            }
            1 => {
                let (idx, gap) = visible_axis(data.len());
                let mut texts: Vec<String> = idx.iter().map(|&i| data[i].to_string()).collect();
                if gap {
                    texts.insert(DISPLAY_EDGE, String::from("…"));
                }
                write!(f, "\n[")?;
                for (k, text) in texts.iter().enumerate() {
                    if k > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{text}")?;
                }
                write!(f, "]")?;
            }
            2 => {
                let (row_idx, row_gap) = visible_axis(data.shape()[0]);
                let (col_idx, col_gap) = visible_axis(data.shape()[1]);

                // One text row per visible matrix row. A row gap is
                // rendered as one ellipsis row.
                let mut rows: Vec<Vec<String>> = Vec::new();
                for (pos, &r) in row_idx.iter().enumerate() {
                    if row_gap && pos == DISPLAY_EDGE {
                        rows.push(vec![
                            String::from("…");
                            col_idx.len() + usize::from(col_gap)
                        ]);
                    }
                    let mut cells: Vec<String> =
                        col_idx.iter().map(|&c| data[[r, c]].to_string()).collect();
                    if col_gap {
                        cells.push(String::from("…"));
                    }
                    rows.push(cells);
                }

                let width = rows
                    .iter()
                    .flatten()
                    .map(|s| s.chars().count())
                    .max()
                    .unwrap_or(0);
                for (r, row_cells) in rows.iter().enumerate() {
                    write!(f, "{}", if r == 0 { "\n[[" } else { "\n [" })?;
                    for (c, text) in row_cells.iter().enumerate() {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{text:>width$}", width = width)?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?;
            }
            _ => {
                let (idx, gap) = visible_axis(data.len());
                let mut texts: Vec<String> = idx.iter().map(|&i| data[i].to_string()).collect();
                if gap {
                    texts.insert(DISPLAY_EDGE, String::from("…"));
                }
                write!(f, "\n[")?;
                for (k, text) in texts.iter().enumerate() {
                    if k > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{text}")?;
                }
                write!(f, "]")?;
            }
        }
        Ok(())
    }
}

/// Per-axis print truncation: an axis longer than `2 * DISPLAY_EDGE`
/// items shows the first and last `DISPLAY_EDGE` items with an
/// ellipsis for the skipped interior.
const DISPLAY_EDGE: usize = 3;

/// Which positions of an axis of length `len` to print, and whether
/// the middle is skipped.
fn visible_axis(len: usize) -> (Vec<usize>, bool) {
    if len > 2 * DISPLAY_EDGE {
        (
            (0..DISPLAY_EDGE).chain(len - DISPLAY_EDGE..len).collect(),
            true,
        )
    } else {
        ((0..len).collect(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert that construction fails with `ShapeMismatch` and return
    /// the two disagreeing shapes.
    fn expect_mismatch(result: Result<Tensor<f64>, TensorError>) -> (Vec<usize>, Vec<usize>) {
        let Err(err) = result else {
            panic!("mis-shaped storage must be rejected");
        };
        match err {
            TensorError::ShapeMismatch { expected, actual } => (expected, actual),
        }
    }

    /// The construction boundary enforces axis-by-axis agreement.
    /// Wrong rank, transposed axes, and stray length-1 axes are rejected.
    #[test]
    fn new_requires_full_shape_agreement() {
        let i = Index::new(2, Arrow::Out);
        let j = Index::new(3, Arrow::In);
        let legs = || vec![i.clone(), j.clone()];
        let dense = |shape: &[usize]| Dense::from_array(ArrayD::from_elem(IxDyn(shape), 0.0_f64));

        // Baseline: the well-shaped tensor constructs.
        assert!(Tensor::new(legs(), dense(&[2, 3])).is_ok());

        // Same element count, wrong rank.
        assert_eq!(
            expect_mismatch(Tensor::new(legs(), dense(&[6]))),
            (vec![2, 3], vec![6])
        );

        // Same element count, transposed axes.
        assert_eq!(
            expect_mismatch(Tensor::new(legs(), dense(&[3, 2]))),
            (vec![2, 3], vec![3, 2])
        );

        // Same element count, stray length-1 axis.
        assert_eq!(
            expect_mismatch(Tensor::new(legs(), dense(&[6, 1]))),
            (vec![2, 3], vec![6, 1])
        );

        // Rank 0: the scalar shape `[]` constructs, `[1]` does not.
        assert!(Tensor::new(Vec::new(), dense(&[])).is_ok());
        assert_eq!(
            expect_mismatch(Tensor::new(Vec::new(), dense(&[1]))),
            (vec![], vec![1])
        );
    }
}
