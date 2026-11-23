# TensorLoom

Rust Library for Tensor Network Calculations. Early development, API unstable.

**Currently:** Dense tensors with ITensor-style identity-based indices and
pairwise contraction.

**Planned:** Network graphs with contraction planning, diagram rendering,
MPS, block-sparse symmetry, Python bindings.

## Usage Example

```rust
use tensorloom::contraction::contract;
use tensorloom::index::{Arrow, Index};
use tensorloom::tensor::Tensor;

fn main() {
    // Define indices
    let i = Index::new(2, Arrow::Out);
    let j = Index::new(3, Arrow::In);
    let k = Index::new(2, Arrow::Out);

    // Build tensors from a function
    let a = Tensor::from_fn(&[i.clone(), j.clone()], |c| (c[0] + c[1]) as f64);
    let b = Tensor::from_fn(&[j.clone(), k.clone()], |c| (c[0] * c[1]) as f64);

    // Contract `a` and `b`; sums over the shared leg `j`
    let ab = contract(&a, &b);

    println!("{ab}");
}
```

## License

MIT or Apache-2.0 — See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
