//! The matrix multiplication kernel, implemented by pure Rust faer.
//! A future BLAS/GPU backend can attach here.
//!
//! Inputs are row-major with unit inner stride (ndarray's owned
//! layout). faer views them in place via its row-major slice
//! constructors. The faer product is column-major, so converting to
//! the row-major result costs one transpose copy.

use faer::MatRef;
use faer::traits::ComplexImpl;
use ndarray::Array2;
use num_complex::Complex;

use crate::scalar::Scalar;

/// `a·b` for row-major matrices, computed by faer.
///
/// Every matrix product in the crate funnels through this function.
pub(crate) fn matmul<S: Scalar>(a: &Array2<S>, b: &Array2<S>) -> Array2<S> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    debug_assert_eq!(k, b.nrows(), "inner dimensions must match");
    debug_assert!(m > 0 && n > 0 && k > 0, "empty GEMM dimensions");

    let a_view = MatRef::from_row_major_slice(
        S::faer_view(a.as_slice().expect("contiguous row-major storage")),
        m,
        k,
    );
    let b_view = MatRef::from_row_major_slice(
        S::faer_view(b.as_slice().expect("contiguous row-major storage")),
        k,
        n,
    );
    let prod = &a_view * &b_view;

    Array2::from_shape_fn((m, n), |(i, j)| S::faer_from(&prod[(i, j)]))
}

/// How a crate scalar maps onto faer's element layer. Sealed by
/// construction. Only the impls below exist, one per element of the
/// crate's fixed scalar set, and [`Scalar`] supertraits this hook, so
/// generic code with an `S: Scalar` bound reaches the kernel without
/// further bounds.
pub(crate) trait FaerScalar {
    /// The element type faer computes with.
    type Elem: faer::traits::ComplexField;

    /// View a contiguous slice of crate scalars as faer elements.
    fn faer_view(slice: &[Self]) -> &[Self::Elem] where Self: Sized;

    /// Materialize one faer element as a crate scalar.
    fn faer_from(elem: &Self::Elem) -> Self where Self: Sized;
}

impl FaerScalar for f32 {
    type Elem = f32;
    fn faer_view(slice: &[Self]) -> &[Self::Elem] {
        slice
    }
    fn faer_from(elem: &Self::Elem) -> Self {
        *elem
    }
}

impl FaerScalar for f64 {
    type Elem = f64;
    fn faer_view(slice: &[Self]) -> &[Self::Elem] {
        slice
    }
    fn faer_from(elem: &Self::Elem) -> Self {
        *elem
    }
}

impl<T> FaerScalar for Complex<T>
where
    Complex<T>: Copy,
    ComplexImpl<T>: faer::traits::ComplexField,
{
    type Elem = ComplexImpl<T>;

    fn faer_view(slice: &[Self]) -> &[Self::Elem] {
        // SAFETY: `ComplexImpl<T>` is `#[repr(transparent)]` over
        // `num_complex::Complex<T>` - identical element and slice
        // layouts for every `T` in the crate's scalar set.
        unsafe { &*(std::ptr::from_ref(slice) as *const [ComplexImpl<T>]) }
    }
    fn faer_from(elem: &Self::Elem) -> Self {
        // SAFETY: same `#[repr(transparent)]` layout identity, inverse
        // direction.
        unsafe { *(std::ptr::from_ref(elem) as *const Self) }
    }
}
