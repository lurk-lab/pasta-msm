// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use std::io::Read;

use abomonation::Abomonation;
use criterion::{criterion_group, criterion_main, Criterion};

use pasta_curves::{pallas, group::ff::PrimeField};
use pasta_msm::{self, utils::CommitmentKey};

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

include!("../src/tests.rs");

fn read_abomonated<T: Abomonation + Clone>(name: String) -> std::io::Result<T> {
    use std::fs::OpenOptions;
    use std::io::BufReader;

    let arecibo = home::home_dir().unwrap().join(".arecibo");

    let data = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(arecibo.join(name))?;
    let mut reader = BufReader::new(data);
    let mut bytes = vec![];
    reader.read_to_end(&mut bytes)?;

    let (data, _) = unsafe { abomonation::decode::<T>(&mut bytes).unwrap() };

    Ok(data.clone())
}

fn criterion_benchmark(c: &mut Criterion) {
    let witness_primary = read_abomonated::<
        Vec<<pallas::Scalar as PrimeField>::Repr>,
    >("witness_primary".into())
    .unwrap();
    let witness_primary = unsafe {
        std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_primary)
    };
    let npoints = witness_primary.len();

    let scalars = crate::tests::gen_scalars(npoints);
    let points = crate::tests::gen_points(npoints);

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        group.bench_function(format!("lurkrs {} scalars", npoints), |b| {
            b.iter(|| {
                let _ = pasta_msm::pallas(&points, &witness_primary);
            })
        });

        group.bench_function(format!("random {} scalars", npoints), |b| {
            b.iter(|| {
                let _ = pasta_msm::pallas(&points, &scalars);
            })
        });

        group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
