// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#![allow(non_snake_case)]

use std::time::Instant;

use pasta_curves::{group::ff::{PrimeField, Field}, pallas};
use pasta_msm::{
    spmvm::{CudaSparseMatrix, CudaWitness, pallas::{sparse_matrix_witness_with_pallas, sparse_matrix_witness_init_pallas}},
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

pub fn generate_scalars<F: PrimeField>(len: usize) -> Vec<F> {
    let mut rng = rand::thread_rng();
    let scalars = (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>();

    scalars
}

/// cargo run --release --example spmvm
fn main() {
    let npow: usize = std::env::var("NPOW")
        .unwrap_or("20".to_string())
        .parse()
        .unwrap();
    let n = 1usize << npow;
    let nthreads: usize = std::env::var("NTHREADS")
        .unwrap_or("256".to_string())
        .parse()
        .unwrap();

    let csr = generate_csr(n, n);
    let cuda_csr =
        CudaSparseMatrix::new(&csr.data, &csr.indices, &csr.indptr, n, n);
    let W = generate_scalars(n - 10);
    let U = generate_scalars(9);
    let scalars = [W.clone(), vec![pallas::Scalar::ONE], U.clone()].concat();

    let start = Instant::now();
    let res = csr.multiply_vec(&scalars);
    println!("cpu took: {:?}", start.elapsed());

    let spmvm_context = sparse_matrix_witness_init_pallas(&cuda_csr);
    let witness = CudaWitness::new(&W, &pallas::Scalar::ONE, &U);
    let mut cuda_res = vec![pallas::Scalar::ONE; cuda_csr.num_rows];
    let start = Instant::now();
    sparse_matrix_witness_with_pallas(&spmvm_context, &witness, &mut cuda_res, nthreads);
    println!("gpu took: {:?}", start.elapsed());

    assert_eq!(res, cuda_res);
    println!("success!");
}
