"""PyTorch CUDA baseline — same Rastrigin workload as wgpu-bench.
Run on the same GPU for apples-to-apples comparison."""

import torch
import time
import math

POP = 4096
DIM = 2000
SIGMA = 0.1
WARMUP = 5
RUNS = 30

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print()

base = torch.randn(DIM, device=device) * 5.12

def rastrigin(x):
    return (x**2 - 10 * torch.cos(2 * math.pi * x) + 10).sum(dim=-1)

def run_one_gen():
    eps = torch.randn(POP, DIM, device=device) * SIGMA
    x_plus = base.unsqueeze(0) + eps
    x_minus = base.unsqueeze(0) - eps
    f_plus = -rastrigin(x_plus)
    f_minus = -rastrigin(x_minus)
    best = torch.max(f_plus.max(), f_minus.max())
    torch.cuda.synchronize()
    return best.item()

# Warmup
for _ in range(WARMUP):
    run_one_gen()

# Benchmark
times = []
for _ in range(RUNS):
    torch.cuda.synchronize()
    t = time.perf_counter()
    run_one_gen()
    times.append((time.perf_counter() - t) * 1000)

mean = sum(times) / len(times)
std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
print(f"PyTorch CUDA — Rastrigin (POP={POP}, DIM={DIM})")
print(f"  {mean:.2f} ms/gen  (±{std:.2f})  {1000/mean:.1f} gen/s  (N={RUNS})")
