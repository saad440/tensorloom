//! Cross-check of dense contraction against numpy reference data.
//!
//! The `.npy` files under `tests/fixtures/` are reference values:
//! seeded random operands and their contractions, computed once by
//! `tests/fixtures/generate_numpy_fixtures.py` with `numpy.einsum`.
//! This test rebuilds each case from the table below, runs
//! [`contract`], and compares
//! element-wise with a relative tolerance. Summation order differs
//! between independent implementations, so bit equality is not
//! expected and would be meaningless.

use std::path::Path;

use ndarray::ArrayD;
use tensorloom::contraction::contract;
use tensorloom::index::{Arrow, Index};
use tensorloom::tensor::{Dense, Tensor};

/// Relative comparison bound. Independent f64 matrix products on
/// these small shapes agree to ~1e-15; 1e-9 leaves orders of
/// magnitude of slack for differing summation orders.
const RELATIVE_TOLERANCE: f64 = 1e-9;
/// Floor for near-zero elements, where relative bounds get too strict.
const ABSOLUTE_TOLERANCE: f64 = 1e-12;

/// One contraction case. The shared-leg pairing must stay in sync with
/// the case table in `tests/fixtures/generate_numpy_fixtures.py`.
struct Case {
    /// Fixture file prefix (`<name>-a.npy`, `<name>-b.npy`,
    /// `<name>-out.npy`).
    name: &'static str,
    a_shape: &'static [usize],
    b_shape: &'static [usize],
    /// Which legs are shared, as (axis in `a`, axis in `b`) pairs.
    shared: &'static [(usize, usize)],
}

const CASES: &[Case] = &[
    // Plain matrix product: shared leg last-in-a, first-in-b; no
    // permutation needed on either side.
    Case {
        name: "gemm",
        a_shape: &[3, 4],
        b_shape: &[4, 5],
        shared: &[(1, 0)],
    },
    // Shared leg at the front of `a`: exercises the a-side permutation
    // and its copy into contiguous order.
    Case {
        name: "permuted",
        a_shape: &[4, 3],
        b_shape: &[4, 5],
        shared: &[(0, 0)],
    },
    // Shared leg in the middle of a rank-3 tensor; the result leg order
    // (i, j, l) checks that surviving legs keep storage order.
    Case {
        name: "rank3",
        a_shape: &[2, 4, 3],
        b_shape: &[4, 5],
        shared: &[(1, 0)],
    },
    // Two shared legs per tensor, in the same relative order on both
    // sides: exercises multi-leg flattening. (`scalar` covers the
    // b-side permutation.)
    Case {
        name: "two-legs",
        a_shape: &[2, 4, 3],
        b_shape: &[4, 3, 5],
        shared: &[(1, 0), (2, 1)],
    },
    // All legs shared: contraction to a rank-0 result.
    Case {
        name: "scalar",
        a_shape: &[3, 4],
        b_shape: &[4, 3],
        shared: &[(0, 1), (1, 0)],
    },
    // No shared legs: outer product through the GEMM path.
    Case {
        name: "outer",
        a_shape: &[3],
        b_shape: &[4],
        shared: &[],
    },
];

#[test]
fn contraction_matches_numpy_fixtures() {
    for case in CASES {
        let expected = load_array(case.name, "out");
        let (a, b) = build_operands(case);
        let result = contract(&a, &b);

        let got = result.storage().as_array();
        assert_eq!(
            got.shape(),
            expected.shape(),
            "case {:?}: result shape {:?} disagrees with numpy's {:?} \
             (surviving-leg order wrong?)",
            case.name,
            got.shape(),
            expected.shape(),
        );

        for (flat, (got_elem, expected_elem)) in
            got.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                approx::relative_eq!(
                    got_elem,
                    expected_elem,
                    epsilon = ABSOLUTE_TOLERANCE,
                    max_relative = RELATIVE_TOLERANCE,
                ),
                "case {:?}, element {flat}: got {got_elem}, \
                 expected {expected_elem}",
                case.name,
            );
        }
    }
}

/// Load both operand fixtures and attach indices. One fresh [`Index`]
/// is created per shared leg pair. Clones of it are placed on both
/// sides. Cloning an index is writing the same summation letter on
/// two legs, which is what [`contract`] sums over.
fn build_operands(case: &Case) -> (Tensor<f64>, Tensor<f64>) {
    let shared_legs: Vec<Index> = case
        .shared
        .iter()
        .map(|&(a_axis, _)| Index::new(case.a_shape[a_axis], Arrow::Out))
        .collect();

    let a = operand(case, case.a_shape, "a", |axis| {
        case.shared
            .iter()
            .position(|&(a_axis, _)| a_axis == axis)
            .map(|pos| shared_legs[pos].clone())
    });
    let b = operand(case, case.b_shape, "b", |axis| {
        case.shared
            .iter()
            .position(|&(_, b_axis)| b_axis == axis)
            .map(|pos| shared_legs[pos].clone())
    });
    (a, b)
}

/// Load one operand fixture and attach indices. Each axis the closure
/// marks shared gets a clone of that pool leg (same id on both
/// operands), each open axis a fresh [`Index`].
/// Arrows play no role yet - id alone for now.
fn operand(
    case: &Case,
    shape: &[usize],
    part: &str,
    shared_leg: impl Fn(usize) -> Option<Index>,
) -> Tensor<f64> {
    let data = load_array(case.name, part);
    assert_eq!(
        data.shape(),
        shape,
        "case {:?}: fixture {}-{} has shape {:?}, the test table says {:?}",
        case.name,
        case.name,
        part,
        data.shape(),
        shape,
    );

    let indices: Vec<Index> = (0..shape.len())
        .map(|axis| {
            shared_leg(axis)
                .unwrap_or_else(|| Index::new(shape[axis], Arrow::Out))
        })
        .collect();
    Tensor::new(indices, Dense::from_array(data))
        .expect("fixture shape matches the test table")
}

/// Read one reference array from `tests/fixtures/`.
fn load_array(case_name: &str, part: &str) -> ArrayD<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{case_name}-{part}.npy"));
    ndarray_npy::read_npy(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}
