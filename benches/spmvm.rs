// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};
use pasta_curves::group::ff::PrimeField;

use pasta_msm::SparseMatrix;
use rand::Rng;

pub fn generate_random_csr<F: PrimeField>(n: usize, m: usize) -> SparseMatrix<F> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = Vec::new();
    row_ptr.push(0);

    for _ in 0..n {
        let num_elements = rng.gen_range(5..=10); // Random number of elements between 5 to 10
        for _ in 0..num_elements {
            data.push(F::random(&mut rng)); // Random data value
            col_idx.push(rng.gen_range(0..m)); // Random column index
        }
        row_ptr.push(data.len()); // Add the index of the next row start
    }

    SparseMatrix::new(data, col_idx, row_ptr, n, m)
}

pub fn generate_scalars<F: PrimeField>(
    len: usize,
) -> Vec<F> {
    let mut rng = rand::thread_rng();

    let scalars = (0..len)
        .map(|_| F::random(&mut rng))
        .collect::<Vec<_>>();

        scalars
}

// fn criterion_benchmark(c: &mut Criterion) {
//     let bench_npow: usize = std::env::var("BENCH_NPOW")
//         .unwrap_or("17".to_string())
//         .parse()
//         .unwrap();
//     let npoints: usize = 1 << bench_npow;

//     //println!("generating {} random points, just hang on...", npoints);
//     let mut points = crate::tests::gen_points(npoints);
//     let mut scalars = crate::tests::gen_scalars(npoints);

//     #[cfg(feature = "cuda")]
//     {
//         unsafe { pasta_msm::CUDA_OFF = true };
//     }

//     let mut group = c.benchmark_group("CPU");
//     group.sample_size(10);

//     group.bench_function(format!("2**{} points", bench_npow), |b| {
//         b.iter(|| {
//             let _ = pasta_msm::pallas(&points, &scalars);
//         })
//     });

//     group.finish();

//     #[cfg(feature = "cuda")]
//     if unsafe { cuda_available() } {
//         unsafe { pasta_msm::CUDA_OFF = false };

//         const EXTRA: usize = 5;
//         let bench_npow = bench_npow + EXTRA;
//         let npoints: usize = 1 << bench_npow;

//         while points.len() < npoints {
//             points.append(&mut points.clone());
//         }
//         scalars.append(&mut crate::tests::gen_scalars(npoints - scalars.len()));

//         let mut group = c.benchmark_group("GPU");
//         group.sample_size(20);

//         group.bench_function(format!("2**{} points", bench_npow), |b| {
//             b.iter(|| {
//                 let _ = pasta_msm::pallas(&points, &scalars);
//             })
//         });

//         group.finish();
//     }
// }

// criterion_group!(benches, criterion_benchmark);
// criterion_main!(benches);

fn main() {}