#![feature(const_generics)]

use bresenham::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn random_point<RNG, const D: usize>(rng: &mut RNG, start: i32, end: i32) -> PointD<i32, D>
where
    RNG: rand::Rng,
{
    array::new(|_d| rng.gen_range(start, end))
}

fn random_line<RNG, const D: usize>(rng: &mut RNG, start: i32, end: i32) -> LineD<i32, D>
where
    RNG: rand::Rng,
{
    (random_point(rng, start, end), random_point(rng, start, end))
}

fn random_lines<RNG, const D: usize>(
    rng: &mut RNG,
    start: i32,
    end: i32,
    count: usize,
) -> Vec<LineD<i32, D>>
where
    RNG: rand::Rng,
{
    (0..count).map(|_| random_line(rng, start, end)).collect()
}

struct BenchmarkConfig {
    seed: u64,
    count: usize,
    start: i32,
    end: i32,
}

fn sample_benchmark<const D: usize>(config: &BenchmarkConfig) -> Vec<LineD<i32, D>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    random_lines::<_, D>(&mut rng, config.start, config.end, config.count)
}

fn rasterize_inner<const D: usize>(line: &LineD<i32, D>) {
    line.rasterize().for_each(|p| {
        black_box(p);
    });
}

fn rasterize_outer<const D: usize>(line: &LineD<i32, D>) {
    for p in line.rasterize() {
        black_box(p);
    }
}

fn rasterize_inner2<const D: usize>(line: &LineD<i32, D>) {
    line.rasterize_two(|p| {
        black_box(p);
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let config = BenchmarkConfig {
        seed: 314_159_265_358_979_323,
        count: 100,
        start: -100,
        end: 100,
    };
    let lines2d = sample_benchmark::<2>(&config);
    //c.bench_function("bresenham  inner 2d", |b| {
    //    b.iter(|| lines2d.iter().for_each(|line| rasterize_inner(line)))
    //});
    //c.bench_function("bresenham  outer 2d", |b| {
    //    b.iter(|| lines2d.iter().for_each(|line| rasterize_outer(line)))
    //});
    c.bench_function("bresenham2 inner 2d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_inner2(line)))
    });

    let lines2d = sample_benchmark::<3>(&config);
    c.bench_function("bresenham inner 3d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_inner(line)))
    });
    c.bench_function("bresenham outer 3d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_outer(line)))
    });
    c.bench_function("bresenham2 inner 3d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_inner2(line)))
    });

    let lines2d = sample_benchmark::<4>(&config);
    c.bench_function("bresenham inner 4d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_inner(line)))
    });
    c.bench_function("bresenham outer 4d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_outer(line)))
    });

    let lines2d = sample_benchmark::<5>(&config);
    c.bench_function("bresenham inner 5d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_inner(line)))
    });
    c.bench_function("bresenham outer 5d", |b| {
        b.iter(|| lines2d.iter().for_each(|line| rasterize_outer(line)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
