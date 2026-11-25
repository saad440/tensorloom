//! Cross-check of network contraction against numpy reference data.
//!
//! The `network` fixture (`tests/fixtures/network-*.npy`) is a
//! four-tensor chain: seeded random operands and their full
//! contraction, computed once by
//! `tests/fixtures/generate_numpy_fixtures.py` with `numpy.einsum`.
//! Each subscript letter appears in at most two operands, so every leg
//! is pairwise. The reference output order `pqrs` is the canonical
//! first-appearance order of the open letters - the same rule
//! [`TensorNetwork::open_indices`] uses.
//!
//! The test rebuilds the network, contracts it along two different
//! plans (sequential and greedy), and compares both results with the
//! reference element-wise. Summation order differs between
//! implementations, so bit equality is not expected.

use std::path::Path;

use ndarray::ArrayD;
use tensorloom::index::{Arrow, Index};
use tensorloom::network::TensorNetwork;
use tensorloom::path::{greedy_path, sequential_path};
use tensorloom::tensor::{Dense, Tensor};

/// Relative comparison bound. Independent f64 contraction paths on
/// these small shapes agree to ~1e-15; 1e-9 leaves orders of magnitude
/// of slack for differing summation orders.
const RELATIVE_TOLERANCE: f64 = 1e-9;
/// Floor for near-zero elements, where relative bounds get too strict.
const ABSOLUTE_TOLERANCE: f64 = 1e-12;


#[test]
fn network_contraction_matches_numpy_fixtures() {
    let golden = load_array("out");

    // Shared legs (a, b, c; dim 8) are each one Index cloned into its
    // two carriers - cloning is writing the same summation letter on
    // two legs, which is what the network joins on. Open legs (p, q,
    // r, s; dim 2) are fresh indices.
    let p = Index::new(2, Arrow::Out);
    let q = Index::new(2, Arrow::Out);
    let r = Index::new(2, Arrow::Out);
    let s = Index::new(2, Arrow::Out);
    let a = Index::new(8, Arrow::Out);
    let b = Index::new(8, Arrow::Out);
    let c = Index::new(8, Arrow::Out);

    let operands = [
        operand(0, &[p.clone(), a.clone()]),
        operand(1, &[a.clone(), q.clone(), b.clone()]),
        operand(2, &[b.clone(), r.clone(), c.clone()]),
        operand(3, &[c.clone(), s.clone()]),
    ];
    let expected_ids: Vec<u64> = [&p, &q, &r, &s].iter().map(|l| l.id()).collect();

    let mut network = TensorNetwork::new();
    for tensor in operands {
        network
            .add_tensor(tensor)
            .expect("operands carry no repeated legs");
    }

    // The open legs must come out in canonical first-appearance order
    // already, before any contraction.
    let open_ids: Vec<u64> = network.open_indices().iter().map(|l| l.id()).collect();
    assert_eq!(
        open_ids, expected_ids,
        "open legs are not in canonical first-appearance order"
    );

    let sequential = sequential_path(&network).expect("chain network plans");
    let from_sequential = network
        .clone()
        .contract(&sequential)
        .expect("valid path executes");

    let greedy = greedy_path(&network).expect("chain network plans");
    let from_greedy = network.contract(&greedy).expect("valid path executes");

    for (name, result) in [("sequential", &from_sequential), ("greedy", &from_greedy)] {
        let got_ids: Vec<u64> = result.indices().iter().map(|l| l.id()).collect();
        assert_eq!(
            got_ids, expected_ids,
            "{name}: result leg order disagrees with the canonical order"
        );

        let got = result.storage().as_array();
        assert_eq!(
            got.shape(),
            golden.shape(),
            "{name}: result shape {:?} disagrees with numpy's {:?}",
            got.shape(),
            golden.shape(),
        );

        for (flat, (got_elem, expected_elem)) in got.iter().zip(golden.iter()).enumerate() {
            assert!(
                approx::relative_eq!(
                    got_elem,
                    expected_elem,
                    epsilon = ABSOLUTE_TOLERANCE,
                    max_relative = RELATIVE_TOLERANCE,
                ),
                "{name}, element {flat}: got {got_elem}, expected {expected_elem}",
            );
        }
    }
}

/// Load one operand fixture (subscript letter `pa`, `aqb`, `brc` or
/// `cs`) and attach its index list. The fixture's storage order must
/// match the order of `indices` (axis k = leg k), which is how the
/// generator writes it; the expected shape is the dims of `indices`.
fn operand(part: usize, indices: &[Index]) -> Tensor<f64> {
    let letter = (b'a' + part as u8) as char;
    let data = load_array(&letter.to_string());
    let expected: Vec<usize> = indices.iter().map(|l| l.dim()).collect();
    assert_eq!(
        data.shape(),
        &expected[..],
        "operand {letter}: fixture shape {:?} disagrees with the index list {:?}",
        data.shape(),
        expected,
    );
    Tensor::new(indices.to_vec(), Dense::from_array(data))
        .expect("fixture shape matches the index list")
}

/// Read one reference array from `tests/fixtures/`.
fn load_array(part: &str) -> ArrayD<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("network-{part}.npy"));
    ndarray_npy::read_npy(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()))
}
