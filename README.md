# TensorLoom

[![crates.io](https://img.shields.io/crates/v/tensorloom)](https://crates.io/crates/tensorloom)
[![docs.rs](https://img.shields.io/docsrs/tensorloom)](https://docs.rs/tensorloom)

Rust Library for Tensor Network Calculations. Early development, API unstable.

**Currently:** Dense tensors with ITensor-style identity-based indices,
tensor networks built from them, and network contraction planning.

**Planned:** Diagram rendering, MPS, block-sparse symmetry, Python bindings.

## Installation

Add with `cargo add tensorloom`, or:

```toml
[dependencies]
tensorloom = "0.0.1"
```

API documentation: [docs.rs](https://docs.rs/tensorloom).

## Usage Example

Tensors contract pairwise over shared legs, and join into networks
wherever they share a leg. The network's contraction order is a plain
value: find it, inspect it, then run it.

```rust
use tensorloom::contraction::contract;
use tensorloom::index::{Arrow, Index};
use tensorloom::network::TensorNetwork;
use tensorloom::path::greedy_path;
use tensorloom::tensor::Tensor;

fn main() {
    // Define indices. Arrows are currently not fully implemented
    // and do not affect contraction.
    let i = Index::new(2, Arrow::Out);
    let j = Index::new(3, Arrow::In);
    let k = Index::new(2, Arrow::Out);
    let l = Index::new(2, Arrow::Out);

    // Build tensors from a function.
    let a = Tensor::from_fn(&[i.clone(), j.clone()], |c| (c[0] + c[1]) as f64);
    let b = Tensor::from_fn(&[j.clone(), k.clone()], |c| (c[0] * c[1]) as f64);
    let c = Tensor::from_fn(&[k.clone(), l.clone()], |c| c[0] as f64 - c[1] as f64);

    // Pairwise contraction: sums `a` and `b` over the shared leg `j`.
    let ab = contract(&a, &b);

    // Network contraction: shared legs join the tensors into a graph,
    // the greedy finder plans the cheapest order, and the network
    // collapses to its open legs [i, l].
    let mut net = TensorNetwork::new();
    net.add_tensor(a).unwrap();
    net.add_tensor(b).unwrap();
    net.add_tensor(c).unwrap();
    let path = greedy_path(&net).unwrap();
    let abc = net.contract(&path).unwrap();

    println!("{ab}");
    println!("{abc}");
}
```

## License

MIT or Apache-2.0 — See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
