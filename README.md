# wgpu-native-bench

Same WGSL shaders as [gpubench.dev](https://gpubench.dev), running natively via `wgpu` — no browser, no framework, no overhead.

For direct comparison: wgpu-native (Vulkan/Metal) vs CUDA vs PyTorch on the same GPU.

## Results

### Apple M2 Pro — Metal backend

| Benchmark | wgpu-native | WebGPU Chrome | Speedup |
|---|---|---|---|
| Rastrigin (POP=4096, DIM=2000) | **357.9 gen/s** | 170.3 gen/s | **2.1×** |
| Rastrigin (POP=4096, DIM=100) | **718.1 gen/s** | — | — |
| N-Body (POP=512, N=128, 200 steps) | 1.0 gen/s | — | — |
| N-Body (POP=512, N=64, 200 steps) | 4.0 gen/s | — | — |

### Comparison targets (from paper, same workloads)

| System | Rastrigin gen/s | Hardware |
|---|---|---|
| **wgpu-native (this repo)** | **357.9** | M2 Pro, Metal |
| wgpu-native (paper) | 326.5 | M2 Pro, Metal |
| WebGPU in Chrome | 170.3 | M2 Pro |
| PyTorch MPS | 160.5 | M2 Pro |
| **NVIDIA comparison pending** | **???** | T4, Vulkan |
| Hand-fused CUDA (paper) | 439.0 | T4 |
| PyTorch CUDA per-step | 0.61 | T4 |

### TODO: NVIDIA T4 (Vulkan)

Run on vast.ai ($0.10/hr) to get the apples-to-apples CUDA comparison:

```bash
cargo run --release
```

## Run

```bash
cargo build --release
./target/release/wgpu-bench
```

Requires Rust 1.70+. Automatically selects the best available GPU backend (Metal on Mac, Vulkan on Linux/Windows).

## Benchmarks

| Benchmark | Type | What it measures |
|---|---|---|
| Rastrigin | Parallel | 4096-population ES on 2000-dim multimodal function |
| N-Body | Sequential (fused) | Gravitational sim, all timesteps in one dispatch |

Same WGSL shaders as the browser benchmarks. Same math, same results, no browser overhead.

## Related

- [gpubench.dev](https://gpubench.dev) — browser benchmarks (624 devices)
- [kernelfusion.dev](https://kernelfusion.dev) — research hub
- [Paper: Kernel Fusion for Sequential Fitness Evaluation](https://doi.org/10.5281/zenodo.19343570)

## License

MIT
