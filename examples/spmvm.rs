// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use pasta_curves::group::ff::PrimeField;
use pasta_msm::spmvm::double_pallas;

pub fn generate_scalars<F: PrimeField>(
    len: usize,
) -> Vec<F> {
    let mut rng = rand::thread_rng();
    let scalars = (0..len)
        .map(|_| F::random(&mut rng))
        .collect::<Vec<_>>();

    scalars
}

/// cargo run --release --example spmvm
fn main() {
    let npow: usize = std::env::var("NPOW")
        .unwrap_or("23".to_string())
        .parse()
        .unwrap();
    let n = 1usize << npow;

    let mut scalars = generate_scalars(n);

    let start = Instant::now();
    let double_scalars = scalars.iter().map(|x| x + x).collect::<Vec<_>>();
    println!("cpu took: {:?}", start.elapsed());

    let start = Instant::now();
    double_pallas(&mut scalars);
    println!("gpu took: {:?}", start.elapsed());

    assert_eq!(double_scalars, scalars);
    println!("success!");
}
