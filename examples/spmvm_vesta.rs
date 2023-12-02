// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(non_snake_case)]

use std::time::Instant;

use pasta_curves::{
    group::ff::{Field, PrimeField},
    vesta,
};
use pasta_msm::{
    spmvm::{
        sparse_matrix_mul_vesta, sparse_matrix_witness_vesta, CudaSparseMatrix,
        CudaWitness, sparse_matrix_witness_vesta_cpu,
    },
    utils::SparseMatrix,
};
use rand::Rng;

pub fn generate_csr<F: PrimeField>(n: usize, m: usize) -> SparseMatrix<F> {
    let mut rng = rand::thread_rng();

    let mut data = Vec::new();
    let mut col_idx = Vec::new();
    let mut row_ptr = Vec::new();
    row_ptr.push(0);

    for _ in 0..n {
        let num_elements = rng.gen_range(2..=3); // Random number of elements between 5 to 10
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

pub fn generate_scalars<F: PrimeField>(len: usize) -> Vec<F> {
    let mut rng = rand::thread_rng();
    let scalars = (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>();

    scalars
}

/// cargo run --release --example spmvm
fn main() {
    let npow: usize = std::env::var("NPOW")
        .unwrap_or("3".to_string())
        .parse()
        .unwrap();
    let n = 1usize << npow;
    let nthreads: usize = std::env::var("NTHREADS")
        .unwrap_or("1".to_string())
        .parse()
        .unwrap();

    let csr_A = generate_csr(n, n);
    let cuda_csr_A =
        CudaSparseMatrix::new(&csr_A.data, &csr_A.indices, &csr_A.indptr, n, n);
    let csr_B = generate_csr(n, n);
    let cuda_csr_B =
        CudaSparseMatrix::new(&csr_B.data, &csr_B.indices, &csr_B.indptr, n, n);

    // let W = generate_scalars(n - 10);
    // let U = generate_scalars(9);
    // let scalars = [W.clone(), vec![vesta::Scalar::ONE], U.clone()].concat();
    let W = vec![vesta::Scalar::ZERO; n - 3];
    let U = vec![vesta::Scalar::ZERO; 2];
    let scalars = vec![vesta::Scalar::ZERO; n];

    let start = Instant::now();
    let res_A = csr_A.multiply_vec(&scalars);
    let res_B = csr_B.multiply_vec(&scalars);
    println!("native took: {:?}", start.elapsed());

    let witness = CudaWitness::new(&W, &vesta::Scalar::ZERO, &U);
    let mut cuda_res_A = vec![vesta::Scalar::ZERO; cuda_csr_A.num_rows];
    let mut cuda_res_B = vec![vesta::Scalar::ZERO; cuda_csr_B.num_rows];
    let start = Instant::now();
    sparse_matrix_witness_vesta(&cuda_csr_A, &witness, &mut cuda_res_A, nthreads);
    sparse_matrix_witness_vesta(&cuda_csr_B, &witness, &mut cuda_res_B, nthreads);
    println!("ffi took: {:?}", start.elapsed());

    assert_eq!(res_A, cuda_res_A);
    assert!(res_B == cuda_res_B);
    println!("success!");
}
