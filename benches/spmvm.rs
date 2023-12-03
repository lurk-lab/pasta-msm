// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(non_snake_case)]

use criterion::{criterion_group, criterion_main, Criterion};

use pasta_curves::{group::ff::{PrimeField, Field}, pallas};
use pasta_msm::{self, utils::SparseMatrix, spmvm::{CudaSparseMatrix, sparse_matrix_witness_pallas, CudaWitness}};
use rand::Rng;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

pub fn generate_csr<F: PrimeField>(
    n: usize,
    m: usize,
) -> SparseMatrix<F> {
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

    data.shrink_to_fit();
    col_idx.shrink_to_fit();
    row_ptr.shrink_to_fit();

    let csr = SparseMatrix {
        data,
        indices: col_idx,
        indptr: row_ptr,
        cols: m,
    };

    csr
}

include!("../src/tests.rs");

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("17".to_string())
        .parse()
        .unwrap();
    let n = 1usize << (bench_npow + 1);
    let m = 1usize << bench_npow;

    println!("generating random matrix and scalars, just hang on...");
    let csr = generate_csr(n, m);
    let cuda_csr =
        CudaSparseMatrix::new(&csr.data, &csr.indices, &csr.indptr, n, m);
    let W = crate::tests::gen_scalars(m - 10);
    let U = crate::tests::gen_scalars(9);
    let witness = CudaWitness::new(&W, &pallas::Scalar::ONE, &U);
    let scalars = [W.clone(), vec![pallas::Scalar::ONE], U.clone()].concat();

    #[cfg(feature = "cuda")]
    {
        unsafe { pasta_msm::CUDA_OFF = true };
    }

    let mut group = c.benchmark_group("CPU");
    group.sample_size(10);

    group.bench_function(format!("2**{} points", bench_npow), |b| {
        b.iter(|| {
            let _ = csr.multiply_vec(&scalars);
        })
    });

    group.finish();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        let mut cuda_res = vec![pallas::Scalar::ONE; cuda_csr.num_rows];
        for nthreads in [128, 256, 512, 1024] {
            group.bench_function(format!("2**{} points, nthreads={}", bench_npow, nthreads), |b| {
                b.iter(|| {
                    let _ = sparse_matrix_witness_pallas(&cuda_csr, &witness, &mut cuda_res, nthreads);
                })
            });
        }

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
