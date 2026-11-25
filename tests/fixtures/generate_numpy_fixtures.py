#!/usr/bin/env python3
"""Generate the reference .npy fixtures for the contraction cross-check test.

This script runs once. Then the Rust test suite (tests/numpy_cross_check.rs)
can read the .npy files directly.

Each case saves three float64 arrays in row-major order:

    <name>-a.npy    first operand
    <name>-b.npy    second operand
    <name>-out.npy  expected contraction, computed with numpy.einsum

The einsum subscript is written so the output leg order matches what
`tensorloom::contraction::contract` produces: the surviving legs of `a`
(in storage order), followed by the surviving legs of `b`. The
(shared-a-axis, shared-b-axis) pairs below must stay in sync with the
`Case` table in the Rust test.

Multi-operand network cases (for tests/network_cross_check.rs) save one
`<name>-<letter>.npy` per operand letter plus `<name>-out.npy`. Each
subscript letter appears in at most two operands, and the output order is
the canonical first-appearance order of the open letters - the same rule
`TensorNetwork::open_indices` uses.

Regenerate only deliberately, if committing fixtures to the repo.
The random stream depends on the numpy version.

Usage:  python3 tests/fixtures/generate_numpy_fixtures.py
"""

from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent
SEED = 202611

# (name, a_shape, b_shape, [(a_axis, b_axis) shared pairs], einsum subscript)
CASES = [
    # Plain matrix product: shared leg already last-in-a / first-in-b,
    # so the contraction hot path needs no permutation.
    ("gemm", (3, 4), (4, 5), [(1, 0)], "ij,jk->ik"),
    # Shared leg at the FRONT of a: the contraction must permute a.
    ("permuted", (4, 3), (4, 5), [(0, 0)], "ki,kj->ij"),
    # Rank-3 by matrix, shared leg in the middle of a; also checks that
    # surviving legs keep storage order (result is (i, j, l), not (i, l, j)).
    ("rank3", (2, 4, 3), (4, 5), [(1, 0)], "ikj,kl->ijl"),
    # Two shared legs per tensor, in the same relative order on both
    # sides: the contraction must flatten both leg groups. The b-side
    # permutation case is `scalar`.
    ("two-legs", (2, 4, 3), (4, 3, 5), [(1, 0), (2, 1)], "ijk,jkl->il"),
    # All legs shared: the result is a rank-0 tensor (a scalar).
    ("scalar", (3, 4), (4, 3), [(0, 1), (1, 0)], "ij,ji->"),
    # No shared legs: outer product through the GEMM path (the empty
    # shared group contributes the empty product 1).
    ("outer", (3,), (4,), [], "i,j->ij"),
]

# (name, [(subscript, shape) per operand], output subscript). The output
# order must stay in sync with tests/network_cross_check.rs.
NETWORK_CASES = [
    (
        "network",
        [("pa", (2, 8)), ("aqb", (8, 2, 8)), ("brc", (8, 2, 8)), ("cs", (8, 2))],
        "pqrs",
    ),
]


def main() -> None:
    print(f"numpy {np.__version__}, seed {SEED}")
    rng = np.random.default_rng(SEED)
    for name, a_shape, b_shape, shared, subscript in CASES:
        a = rng.standard_normal(a_shape)
        b = rng.standard_normal(b_shape)
        out = np.einsum(subscript, a, b)
        np.save(FIXTURES / f"{name}-a.npy", a)
        np.save(FIXTURES / f"{name}-b.npy", b)
        np.save(FIXTURES / f"{name}-out.npy", out)
        legs = shared if shared else "no shared legs"
        print(f"{name:8s} {a_shape} x {b_shape} over {legs} -> "
              f"{out.shape}  [{subscript}]")
    for name, operands, output in NETWORK_CASES:
        arrays = [rng.standard_normal(shape) for _, shape in operands]
        subscript = ",".join(letters for letters, _ in operands)
        out = np.einsum(f"{subscript}->{output}", *arrays)
        for letter, array in zip("abcd", arrays):
            np.save(FIXTURES / f"{name}-{letter}.npy", array)
        np.save(FIXTURES / f"{name}-out.npy", out)
        print(f"{name:8s} network of {len(arrays)} tensors -> "
              f"{out.shape}  [{subscript}->{output}]")


if __name__ == "__main__":
    main()
