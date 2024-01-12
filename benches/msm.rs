// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use std::io::Read;

use abomonation::Abomonation;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use pasta_curves::{group::ff::{PrimeField, Field}, pallas};
use pasta_msm::{self, utils::CommitmentKey};
use rand::thread_rng;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

fn read_abomonated<T: Abomonation + Clone>(name: String) -> std::io::Result<T> {
    use std::fs::OpenOptions;
    use std::io::BufReader;

    let arecibo = home::home_dir().unwrap().join(".arecibo_witness");

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
    let npoints: usize = 10_000_000;
    let scalars = pasta_msm::utils::gen_scalars(npoints);
    // let nonuniform_scalars = pasta_msm::utils::generate_nonuniform_scalars(npoints);
    let points = pasta_msm::utils::gen_points(npoints);

    let mut rng = thread_rng();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        for i in 0..32 {
            let witness_i = read_abomonated::<
                Vec<<pallas::Scalar as PrimeField>::Repr>,
            >(i.to_string())
            .unwrap();
            let mut witness_i = unsafe {
                std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_i)
            };

            let witness_n = witness_i.len();

            if witness_n < 1_000_000 {
                continue;
            }

            let name = format!("#{i}, {} scalars", witness_n);

            // group.bench_function(BenchmarkId::new("random", &name), |b| {
            //     b.iter(|| {
            //         let _ = pasta_msm::pallas(
            //             &points[..witness_n],
            //             &scalars[..witness_n],
            //             witness_n,
            //         );
            //     })
            // });

            group.bench_function(
                BenchmarkId::new("lurkrs", &name),
                |b| {
                    b.iter(|| {
                        let _ = pasta_msm::pallas(
                            &points[..witness_n],
                            &witness_i,
                            witness_n,
                        );
                    })
                },
            );

            for w_i in witness_i.iter_mut() {
                if w_i.is_zero_vartime() {
                    *w_i = pallas::Scalar::random(&mut rng);
                }
            }
            group.bench_function(
                BenchmarkId::new("lurkrs no zeros", &name),
                |b| {
                    b.iter(|| {
                        let _ = pasta_msm::pallas(
                            &points[..witness_n],
                            &witness_i,
                            witness_n,
                        );
                    })
                },
            );
        }

        // group.bench_function(format!("biased {} scalars", npoints), |b| {
        //     b.iter(|| {
        //         let _ =
        //             pasta_msm::pallas(&points, &nonuniform_scalars, npoints);
        //     })
        // });

        // for npoints in [7941351, 9699051] {
        //     group.bench_function(format!("random {} scalars", npoints), |b| {
        //         b.iter(|| {
        //             let _ = pasta_msm::pallas(&points, &scalars, npoints);
        //         })
        //     });
        // }

        // let context = pasta_msm::pallas_init(&points, npoints);

        // group.bench_function(format!("preallocated lurkrs {} scalars", npoints), |b| {
        //     b.iter(|| {
        //         let _ = pasta_msm::pallas_with(&context, npoints, npoints, &witness_primary);
        //     })
        // });

        // group.bench_function(
        //     format!("preallocated biased {} scalars", npoints),
        //     |b| {
        //         b.iter(|| {
        //             let _ = pasta_msm::pallas_with(
        //                 &context,
        //                 npoints,
        //                 npoints,
        //                 &nonuniform_scalars,
        //             );
        //         })
        //     },
        // );

        // group.bench_function(
        //     format!("preallocated random {} scalars", npoints),
        //     |b| {
        //         b.iter(|| {
        //             let _ = pasta_msm::pallas_with(
        //                 &context, npoints, npoints, &scalars,
        //             );
        //         })
        //     },
        // );

        // group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
