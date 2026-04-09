# wgpu-native-bench

Same WGSL shaders as [gpubench.dev](https://gpubench.dev), running natively via `wgpu` — no browser, no framework, no overhead.

For direct comparison: wgpu-native (Vulkan/Metal) vs CUDA vs PyTorch on the same GPU.

## Results

### NVIDIA RTX 3090 — Vulkan backend

| Benchmark | wgpu-native (Vulkan) | vs WebGPU Chrome (M2 Pro) | vs Hand-fused CUDA (T4) |
|---|---|---|---|
| **Rastrigin (POP=4096, DIM=2000)** | **5,028.7 gen/s** | **29.5×** faster | **11.5×** faster |
| **Rastrigin (POP=4096, DIM=100)** | **8,010.0 gen/s** | — | — |
| N-Body (POP=512, N=128, 200 steps) | 2.5 gen/s | — | — |
| N-Body (POP=512, N=64, 200 steps) | 15.6 gen/s | — | — |

### Apple M2 Pro — Metal backend

| Benchmark | wgpu-native (Metal) | vs WebGPU Chrome | Speedup |
|---|---|---|---|
| Rastrigin (POP=4096, DIM=2000) | **357.9 gen/s** | 170.3 gen/s | **2.1×** |
| Rastrigin (POP=4096, DIM=100) | **718.1 gen/s** | — | — |
| N-Body (POP=512, N=128, 200 steps) | 1.0 gen/s | — | — |
| N-Body (POP=512, N=64, 200 steps) | 4.0 gen/s | — | — |

### Full comparison (Rastrigin POP=4096, DIM=2000)

| System | gen/s | Hardware | vs PyTorch CUDA |
|---|---|---|---|
| **wgpu-native Vulkan** | **5,028.7** | **RTX 3090** | **8,243×** |
| Hand-fused CUDA (paper) | 439.0 | T4 | 720× |
| wgpu-native Metal | 357.9 | M2 Pro | 587× |
| wgpu-native Metal (paper) | 326.5 | M2 Pro | 535× |
| WebGPU in Chrome | 170.3 | M2 Pro | 279× |
| PyTorch MPS | 160.5 | M2 Pro | 263× |
| PyTorch CUDA per-step | 0.61 | T4 | 1× |

**Key takeaway:** wgpu-native on Vulkan (RTX 3090) achieves 5,029 gen/s — the same WGSL shader, no CUDA, no framework. One codebase runs on NVIDIA (Vulkan), Apple (Metal), AMD, Intel, and Qualcomm.

## Run

```bash
cargo build --release
./target/release/wgpu-bench
```

Requires Rust 1.70+. Automatically selects the best available GPU backend (Metal on Mac, Vulkan on Linux/Windows).

For NVIDIA on Linux (Docker/vast.ai), set the Vulkan ICD:
```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json ./target/release/wgpu-bench
```

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
