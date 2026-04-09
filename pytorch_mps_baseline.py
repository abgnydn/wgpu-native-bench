"""PyTorch MPS baseline — all benchmarks on Apple Silicon.
Run on the same M2 Pro as wgpu-native Metal for apples-to-apples."""

import torch
import time
import math

WARMUP = 5
RUNS = 30

device = torch.device("mps")
print(f"PyTorch: {torch.__version__}")
print(f"Backend: MPS (Apple Silicon)")
print()

# ─── Rastrigin ───

def rastrigin(x):
    return (x**2 - 10 * torch.cos(2 * math.pi * x) + 10).sum(dim=-1)

def bench_rastrigin(pop, dim, sigma=0.1):
    base = torch.randn(dim, device=device) * 5.12
    def run():
        eps = torch.randn(pop, dim, device=device) * sigma
        x_plus = base.unsqueeze(0) + eps
        x_minus = base.unsqueeze(0) - eps
        f_plus = -rastrigin(x_plus)
        f_minus = -rastrigin(x_minus)
        best = torch.max(f_plus.max(), f_minus.max())
        torch.mps.synchronize()
        return best.item()

    for _ in range(WARMUP): run()
    times = []
    for _ in range(RUNS):
        torch.mps.synchronize()
        t = time.perf_counter()
        run()
        times.append((time.perf_counter() - t) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
    print(f"  Rastrigin (POP={pop}, DIM={dim}): {mean:.2f} ms/gen (±{std:.2f})  {1000/mean:.1f} gen/s  (N={RUNS})")

# ─── N-Body ───

def bench_nbody(pop, n_bodies, steps, dt=0.01):
    softening = 0.1
    def run():
        px = (torch.rand(pop, n_bodies, device=device) - 0.5) * 20
        py = (torch.rand(pop, n_bodies, device=device) - 0.5) * 20
        vx = (torch.rand(pop, n_bodies, device=device) - 0.5) * 0.5
        vy = (torch.rand(pop, n_bodies, device=device) - 0.5) * 0.5
        mass = torch.ones(pop, n_bodies, device=device)

        for _ in range(steps):
            # O(N^2) pairwise forces — per-step dispatch (PyTorch style)
            dx = px.unsqueeze(2) - px.unsqueeze(1)
            dy = py.unsqueeze(2) - py.unsqueeze(1)
            dist_sq = dx**2 + dy**2 + softening**2
            inv_dist3 = dist_sq ** (-1.5)
            ax = (mass.unsqueeze(1) * dx * inv_dist3).sum(dim=2)
            ay = (mass.unsqueeze(1) * dy * inv_dist3).sum(dim=2)
            vx += ax * dt
            vy += ay * dt
            px += vx * dt
            py += vy * dt

        energy = 0.5 * mass * (vx**2 + vy**2)
        result = energy.sum(dim=1)
        torch.mps.synchronize()
        return result

    for _ in range(WARMUP): run()
    times = []
    for _ in range(RUNS):
        torch.mps.synchronize()
        t = time.perf_counter()
        run()
        times.append((time.perf_counter() - t) * 1000)

    mean = sum(times) / len(times)
    std = (sum((t - mean)**2 for t in times) / len(times)) ** 0.5
    print(f"  N-Body (POP={pop}, N={n_bodies}, STEPS={steps}): {mean:.2f} ms/gen (±{std:.2f})  {1000/mean:.1f} gen/s  (N={RUNS})")

print("─── Rastrigin ───")
bench_rastrigin(4096, 2000)
bench_rastrigin(4096, 100)

print()
print("─── N-Body ───")
bench_nbody(512, 128, 200)
bench_nbody(512, 64, 200)
