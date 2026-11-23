//! Contraction of tensors by shared index identity.
//!
//! Contraction logic lives here. Matrix multiplication lives in
//! `crate::gemm`, backed by the faer crate. Alternative compute
//! backends can attach at either layer later.

use std::collections::HashSet;

use ndarray::{Array2, IxDyn};

use crate::gemm::matmul;
use crate::index::Index;
use crate::scalar::Scalar;
use crate::tensor::{Dense, Tensor};

/// Contract `a` and `b` over the indices whose ids appear in both.
///
/// Matching is by index id alone. [`Arrow`](crate::index::Arrow) is not
/// used yet. The result carries `a`'s remaining indices followed by `b`'s,
/// in storage order. Tensors with no shared indices yield the outer product.
/// Tensors whose indices are all shared yield a rank-0 tensor.
///
/// A leg may appear at most once in each operand. Self-contractions (a
/// tensor sharing two of its own legs) are not supported yet.
///
/// # Examples
///
/// ```
/// use tensorloom::contraction::contract;
/// use tensorloom::index::{Arrow, Index};
/// use tensorloom::tensor::Tensor;
///
/// let i = Index::new(2, Arrow::Out);
/// let j = Index::new(3, Arrow::In);
/// let k = Index::new(4, Arrow::Out);
/// let t = Tensor::from_fn(&[i.clone(), j.clone()], |c| (c[0] + 2 * c[1]) as f64);
/// let u = Tensor::from_fn(&[j.clone(), k.clone()], |c| c[0] as f64 - c[1] as f64);
/// let r = contract(&t, &u);
/// assert_eq!(r.rank(), 2);
/// assert_eq!(r.storage().as_array().shape(), &[2, 4]);
/// let expected = ndarray::ArrayD::from_shape_vec(
///     ndarray::IxDyn(&[2, 4]),
///     vec![10.0, 4.0, -2.0, -8.0, 13.0, 4.0, -5.0, -14.0],
/// )
/// .unwrap();
/// assert_eq!(r.storage().as_array(), &expected);
/// ```
///
/// Panics if a leg appears more than once in either operand.
pub fn contract<S: Scalar>(a: &Tensor<S>, b: &Tensor<S>) -> Tensor<S> {
    // 1. Classify legs: open (survive) vs shared (summed over). Ids are
    //    unique per leg, so set intersection is the matching rule.
    let a_ids: HashSet<u64> = a.indices().iter().map(Index::id).collect();
    let b_ids: HashSet<u64> = b.indices().iter().map(Index::id).collect();
    assert_eq!(
        a_ids.len(),
        a.rank(),
        "repeated legs in `a` are not supported yet"
    );
    assert_eq!(
        b_ids.len(),
        b.rank(),
        "repeated legs in `b` are not supported yet"
    );
    let shared: HashSet<u64> = a_ids.intersection(&b_ids).copied().collect();

    // 2. Permute axes for the matrix product: a' = (open..., shared...),
    //    b' = (shared..., open...), each group keeping that operand's
    //    storage order.
    let (a_shared_axes, a_open_axes) = partition_axes(a.indices(), &shared);
    let (b_shared_storage_order, b_open_axes) =
        partition_axes(b.indices(), &shared);
    // The GEMM pairs the two flattened shared groups position by
    // position, so both must enumerate identical coordinate tuples.
    // `a` sets the canonical order (its shared legs in storage order);
    // b's shared axes are re-ordered to match.
    let b_shared_axes: Vec<usize> = a_shared_axes
        .iter()
        .map(|&k| {
            let id = a.indices()[k].id();
            b_shared_storage_order
                .iter()
                .copied()
                .find(|&axis| b.indices()[axis].id() == id)
                .expect("every shared id occurs in `b` by construction")
        })
        .collect();
    let perm_a: Vec<usize> = a_open_axes.iter().chain(a_shared_axes.iter()).copied().collect();
    let perm_b: Vec<usize> = b_shared_axes.iter().chain(b_open_axes.iter()).copied().collect();

    // 3. Leg-group sizes. Empty groups contribute the empty product 1,
    //    which is why no-shared (outer product) and all-shared (scalar)
    //    need no special cases.
    let oa = combined_dim(&a_open_axes, a.indices());
    let s_dim = combined_dim(&a_shared_axes, a.indices());
    let ob = combined_dim(&b_open_axes, b.indices());
    debug_assert_eq!(
        s_dim,
        combined_dim(&b_shared_axes, b.indices()),
        "shared legs carry equal ids, hence equal dims"
    );

    // 4. Reshape the permuted views into matrices (copies when the
    //    permutation makes the view non-contiguous).
    // as_standard_layout yields a Cow: it borrows a view that is
    // already C-contiguous and copies otherwise, so the owned array is
    // guaranteed standard layout, which the GEMM's flat-slice access
    // relies on.
    let a_mat: Array2<S> = a
        .storage()
        .as_array()
        .view()
        .permuted_axes(perm_a)
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((oa, s_dim))
        .expect("contraction: reshaping `a` to a matrix");
    let b_mat: Array2<S> = b
        .storage()
        .as_array()
        .view()
        .permuted_axes(perm_b)
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order((s_dim, ob))
        .expect("contraction: reshaping `b` to a matrix");

    // 5. The GEMM: result[i, k] = sum_j a_mat[i, j] * b_mat[j, k],
    //    routed through faer (see `crate::gemm`).
    let prod = matmul(&a_mat, &b_mat);

    // 6. Reshape back and attach the surviving legs.
    let mut result_indices: Vec<Index> =
        Vec::with_capacity(a_open_axes.len() + b_open_axes.len());
    result_indices.extend(a_open_axes.iter().map(|k| a.indices()[*k].clone()));
    result_indices.extend(b_open_axes.iter().map(|k| b.indices()[*k].clone()));
    let shape: Vec<usize> = result_indices.iter().map(Index::dim).collect();
    let data = prod
        .to_shape(IxDyn(&shape))
        .expect("contraction: reshaping result to the surviving legs")
        .into_owned();
    Tensor::new(result_indices, Dense::from_array(data)).expect("invariants hold by construction")
}

/// Split storage positions into (shared, open), each group in storage
/// order.
fn partition_axes(indices: &[Index], shared: &HashSet<u64>) -> (Vec<usize>, Vec<usize>) {
    let mut shared_axes = Vec::new();
    let mut open_axes = Vec::new();
    for (axis, leg) in indices.iter().enumerate() {
        if shared.contains(&leg.id()) {
            shared_axes.push(axis);
        } else {
            open_axes.push(axis);
        }
    }
    (shared_axes, open_axes)
}

/// Combined dimension of a group of legs: how many distinct values their
/// joint coordinate takes — the length the group contributes when
/// flattened into a single matrix axis (same product rule as the
/// component count in `Tensor::new`). An empty group contributes 1 (the
/// empty product, the multiplicative identity over zero factors).
fn combined_dim(axes: &[usize], indices: &[Index]) -> usize {
    axes.iter().map(|k| indices[*k].dim()).product()
}

