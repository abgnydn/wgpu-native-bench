use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::time::Instant;
use wgpu::util::DeviceExt;

// ─── Params structs (must match WGSL) ───

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RastriginParams {
    pop: u32,
    dim: u32,
    sigma: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NBodyParams {
    pop: u32,
    n_bodies: u32,
    steps: u32,
    dt: f32,
}

// ─── GPU setup ───

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    backend: String,
}

fn init_gpu() -> Gpu {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("No GPU adapter found");

        let info = adapter.get_info();
        let backend = format!("{:?}", info.backend);
        let adapter_name = info.name.clone();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bench"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create device");

        Gpu { device, queue, adapter_name, backend }
    })
}

// ─── Benchmark helpers ───

struct BenchResult {
    name: String,
    mean_ms: f64,
    std_ms: f64,
    gen_per_sec: f64,
    runs: usize,
}

fn stats(times: &[f64]) -> (f64, f64) {
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let var = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    (mean, var.sqrt())
}

// ─── Rastrigin benchmark ───

fn run_rastrigin(gpu: &Gpu, pop: u32, dim: u32, warmup: usize, runs: usize) -> BenchResult {
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rastrigin"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/rastrigin.wgsl").into()),
    });

    let params = RastriginParams { pop, dim, sigma: 0.1, _pad: 0.0 };
    let mut rng = rand::thread_rng();
    let base: Vec<f32> = (0..dim).map(|_| rng.gen_range(-5.12f32..5.12)).collect();
    let seeds: Vec<u32> = (0..pop).map(|_| rng.gen()).collect();

    let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let base_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&base), usage: wgpu::BufferUsages::STORAGE,
    });
    let seeds_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&seeds),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let result_size = (pop as u64) * 3 * 4;
    let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        })),
        module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: base_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: seeds_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: results_buf.as_entire_binding() },
        ],
    });

    let wg = (pop + 63) / 64;

    let run_one = || {
        let new_seeds: Vec<u32> = (0..pop).map(|_| rand::thread_rng().gen()).collect();
        gpu.queue.write_buffer(&seeds_buf, 0, bytemuck::cast_slice(&new_seeds));

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(wg, 1, 1); }
        enc.copy_buffer_to_buffer(&results_buf, 0, &readback_buf, 0, result_size);
        gpu.queue.submit(Some(enc.finish()));

        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::Maintain::Wait);
        drop(slice.get_mapped_range());
        readback_buf.unmap();
    };

    for _ in 0..warmup { run_one(); }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        run_one();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let (mean, std) = stats(&times);
    BenchResult { name: format!("Rastrigin (POP={pop}, DIM={dim})"), mean_ms: mean, std_ms: std, gen_per_sec: 1000.0 / mean, runs }
}

// ─── N-Body benchmark ───

fn run_nbody(gpu: &Gpu, pop: u32, n_bodies: u32, steps: u32, warmup: usize, runs: usize) -> BenchResult {
    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nbody"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/nbody.wgsl").into()),
    });

    let params = NBodyParams { pop, n_bodies, steps, dt: 0.01 };
    let mut rng = rand::thread_rng();
    let seeds: Vec<u32> = (0..pop).map(|_| rng.gen()).collect();

    let params_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM,
    });
    let seeds_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&seeds), usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (pop as u64) * 4;
    let results_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: result_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        })),
        module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: seeds_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: results_buf.as_entire_binding() },
        ],
    });

    let wg = (pop + 63) / 64;

    let run_one = || {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        { let mut p = enc.begin_compute_pass(&Default::default()); p.set_pipeline(&pipeline); p.set_bind_group(0, &bg, &[]); p.dispatch_workgroups(wg, 1, 1); }
        enc.copy_buffer_to_buffer(&results_buf, 0, &readback_buf, 0, result_size);
        gpu.queue.submit(Some(enc.finish()));

        let slice = readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::Maintain::Wait);
        drop(slice.get_mapped_range());
        readback_buf.unmap();
    };

    for _ in 0..warmup { run_one(); }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        run_one();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let (mean, std) = stats(&times);
    BenchResult { name: format!("N-Body (POP={pop}, N={n_bodies}, STEPS={steps})"), mean_ms: mean, std_ms: std, gen_per_sec: 1000.0 / mean, runs }
}

// ─── Main ───

fn print_result(r: &BenchResult) {
    println!("  {:44} {:8.2} ms/gen  (±{:.2})  {:>10.1} gen/s  (N={})",
        r.name, r.mean_ms, r.std_ms, r.gen_per_sec, r.runs);
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  wgpu-native Benchmark — No Browser, No Framework          ║");
    println!("║  Same WGSL shaders as gpubench.dev, running on bare metal  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpu = init_gpu();
    println!("  GPU:     {}", gpu.adapter_name);
    println!("  Backend: {}", gpu.backend);
    println!();

    let warmup = 5;
    let runs = 30;

    println!("─── Rastrigin (embarrassingly parallel) ───");
    print_result(&run_rastrigin(&gpu, 4096, 2000, warmup, runs));
    print_result(&run_rastrigin(&gpu, 4096, 100, warmup, runs));

    println!();
    println!("─── N-Body (sequential, fused) ───");
    print_result(&run_nbody(&gpu, 512, 128, 200, warmup, runs));
    print_result(&run_nbody(&gpu, 512, 64, 200, warmup, runs));

    println!();
    println!("─── Paper comparison targets ───");
    println!("  PyTorch CUDA per-step (T4, Acrobot):   0.61 gen/s");
    println!("  Hand-fused CUDA (T4, Acrobot):         439.0 gen/s");
    println!("  WebGPU Chrome (M2 Pro, Rastrigin):     170.3 gen/s");
    println!("  wgpu-native Metal (M2 Pro, Rastrigin): 326.5 gen/s (paper)");
    println!();
    println!("  Run on NVIDIA T4 to compare directly with paper CUDA numbers.");
    println!("  vast.ai: ~$0.10/hr for a T4");
    println!();
}
