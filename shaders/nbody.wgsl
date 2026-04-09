// N-Body gravitational simulation — fused sequential evaluation
// Each thread runs one complete N-body simulation for STEPS timesteps
// Same as gpubench.dev benchmark

struct Params {
    pop: u32,
    n_bodies: u32,
    steps: u32,
    dt: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>; // [POP] — final energy

const G: f32 = 1.0;
const SOFTENING: f32 = 0.1;
const MAX_BODIES: u32 = 512u;

fn lcg(s: u32) -> u32 { return s * 1664525u + 1013904223u; }
fn u2f(v: u32) -> f32 { return f32(v >> 8u) / 16777216.0 - 0.5; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.pop) { return; }

    let N = min(params.n_bodies, MAX_BODIES);

    // Initialize random positions and velocities from seed
    var seed = seeds[idx];
    var px: array<f32, 512>;
    var py: array<f32, 512>;
    var vx: array<f32, 512>;
    var vy: array<f32, 512>;
    var mass: array<f32, 512>;

    for (var i: u32 = 0u; i < N; i++) {
        seed = lcg(seed); px[i] = u2f(seed) * 20.0;
        seed = lcg(seed); py[i] = u2f(seed) * 20.0;
        seed = lcg(seed); vx[i] = u2f(seed) * 0.5;
        seed = lcg(seed); vy[i] = u2f(seed) * 0.5;
        mass[i] = 1.0;
    }

    // Run simulation for STEPS timesteps — all fused in one thread
    let dt = params.dt;
    for (var step: u32 = 0u; step < params.steps; step++) {
        // Compute accelerations (O(N^2))
        for (var i: u32 = 0u; i < N; i++) {
            var ax: f32 = 0.0;
            var ay: f32 = 0.0;
            for (var j: u32 = 0u; j < N; j++) {
                if (i == j) { continue; }
                let dx = px[j] - px[i];
                let dy = py[j] - py[i];
                let dist_sq = dx * dx + dy * dy + SOFTENING * SOFTENING;
                let inv_dist = 1.0 / sqrt(dist_sq);
                let inv_dist3 = inv_dist * inv_dist * inv_dist;
                ax += G * mass[j] * dx * inv_dist3;
                ay += G * mass[j] * dy * inv_dist3;
            }
            vx[i] += ax * dt;
            vy[i] += ay * dt;
        }
        // Update positions
        for (var i: u32 = 0u; i < N; i++) {
            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;
        }
    }

    // Compute total kinetic energy as fitness
    var energy: f32 = 0.0;
    for (var i: u32 = 0u; i < N; i++) {
        energy += 0.5 * mass[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
    }
    results[idx] = energy;
}
