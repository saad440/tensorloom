//! The scalar field over which tensors are defined.

use ndarray::LinalgScalar;
use num_complex::Complex;

/// Numeric scalar types supported by the tensor core.
///
/// Sealed on purpose. The set of element types is a property of the whole
/// library, not something callers extend. The [`LinalgScalar`] supertrait
/// supplies the arithmetic ndarray needs for element operations and
/// matrix multiplication.
///
/// A Lapack-bounded companion trait (SVDs etc.) arrives with the MPS layer.
/// It will share this crate's sealed impl set.

// `FaerScalar` is deliberately crate-internal so the
// private-supertrait lint is suppressed on purpose.
#[allow(private_bounds)]
pub trait Scalar: LinalgScalar + private::Sealed + crate::gemm::FaerScalar {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex<f32> {}
impl Scalar for Complex<f64> {}

mod private {
    use num_complex::Complex;

    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for Complex<f32> {}
    impl Sealed for Complex<f64> {}
}
