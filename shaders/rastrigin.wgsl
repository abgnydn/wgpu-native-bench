// Rastrigin benchmark — fused ES evaluation
// POP individuals, DIM dimensions, all in one dispatch
// Same shader as gpubench.dev and the paper

struct Params {
    pop: u32,
    dim: u32,
    sigma: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> base: array<f32>;       // [DIM]
@group(0) @binding(2) var<storage, read> seeds: array<u32>;      // [POP]
@group(0) @binding(3) var<storage, read_write> results: array<f32>; // [POP * 3] — r+, r-, best

const PI: f32 = 3.14159265358979;

fn lcg(s: u32) -> u32 { return s * 1664525u + 1013904223u; }
fn u2f(v: u32) -> f32 { return f32(v >> 8u) / 16777216.0; }
fn box_muller(u1: f32, u2: f32) -> f32 {
    return sqrt(-2.0 * log(max(u1, 1e-10))) * cos(6.2831853 * u2);
}

fn rastrigin(x: f32) -> f32 {
    return x * x - 10.0 * cos(2.0 * PI * x) + 10.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.pop) { return; }

    var seed = seeds[idx];
    var fit_plus: f32 = 0.0;
    var fit_minus: f32 = 0.0;

    for (var d: u32 = 0u; d < params.dim; d++) {
        seed = lcg(seed); let u1 = u2f(seed);
        seed = lcg(seed); let u2 = u2f(seed);
        let eps = box_muller(u1, u2) * params.sigma;

        let x_plus = base[d] + eps;
        let x_minus = base[d] - eps;
        fit_plus -= rastrigin(x_plus);
        fit_minus -= rastrigin(x_minus);
    }

    results[idx * 3u] = fit_plus;
    results[idx * 3u + 1u] = fit_minus;
    results[idx * 3u + 2u] = max(fit_plus, fit_minus);
}
