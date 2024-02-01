// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_mut)]

use std::io::Read;

use abomonation::Abomonation;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::{self, utils::{collect, compress, new_compress, CommitmentKey}};
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
    let mut scalars = pasta_msm::utils::gen_scalars(npoints);
    // let nonuniform_scalars = pasta_msm::utils::generate_nonuniform_scalars(npoints);
    let points = pasta_msm::utils::gen_points(npoints);

    let mut _rng = thread_rng();

    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };

        let mut group = c.benchmark_group("GPU");
        group.sample_size(20);

        // let context = pasta_msm::pallas_init(&points, npoints);

        // let ones = vec![pallas::Scalar::ONE; 10_000_000];
        // let name = format!("preallocated random {} scalars", npoints);
        // group.bench_function(
        //     BenchmarkId::new("ones", name),
        //     |b| {
        //         b.iter(|| {
        //             let _ = pasta_msm::pallas_with(&context, npoints, npoints, &scalars);
        //         })
        //     },
        // );

        // let zeros = vec![pallas::Scalar::ZERO; 10_000_000];
        // let name = format!("preallocated random {} scalars", npoints);
        // group.bench_function(
        //     BenchmarkId::new("zeros", name),
        //     |b| {
        //         b.iter(|| {
        //             let _ = pasta_msm::pallas_with(&context, npoints, npoints, &zeros);
        //         })
        //     },
        // );
        // for pones in [0.00, 0.25, 0.50, 0.75, 0.90, 0.99] {
        //     let ones = (pones * npoints as f64) as usize;

        //     for i in 0..ones {
        //         scalars[i] = pallas::Scalar::ONE;
        //     }

        //     let name = format!("preallocated random {} scalars", npoints);
        //     group.bench_function(
        //         BenchmarkId::new(pones.to_string(), name),
        //         |b| {
        //             b.iter(|| {
        //                 let _ = pasta_msm::pallas_with(&context, npoints, npoints, &scalars);
        //             })
        //         },
        //     );
        // }

        for i in 0..32 {
            let witness_i = read_abomonated::<Vec<<pallas::Scalar as PrimeField>::Repr>>(i.to_string()).unwrap();
            let mut witness_i = unsafe { std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_i) };

            let witness_n = witness_i.len();

            if witness_n < 1_000_000 {
                continue;
            }

            let name = format!("#{i}, {} scalars", witness_n);

            group.bench_function(BenchmarkId::new("random", &name), |b| {
                b.iter(|| {
                    let _ = pasta_msm::pallas(
                        &points[..witness_n],
                        &scalars[..witness_n],
                        witness_n,
                    );
                })
            });

            // group.bench_function(BenchmarkId::new("lurkrs", &name), |b| {
            //     b.iter(|| {
            //         let _ = pasta_msm::pallas(
            //             &points[..witness_n],
            //             &witness_i,
            //             witness_n,
            //         );
            //     })
            // });

            // for w_i in witness_i.iter_mut() {
            //     if w_i.is_zero_vartime() || *w_i == pallas::Scalar::ONE {
            //         *w_i = pallas::Scalar::random(&mut rng);
            //     }
            // }
            // println!("com_n {i}: {com_n}");

            group.bench_function(
                BenchmarkId::new("lurkrs collect", &name),
                |b| {
                    b.iter(|| {
                        let _ = collect(&witness_i);
                    })
                },
            );

            group.bench_function(
                BenchmarkId::new("lurkrs compress", &name),
                |b| {
                    b.iter(|| {
                        let map = collect(&witness_i);
                        let _ = compress(&points[..witness_n], map);
                    })
                },
            );

            group.bench_function(
                BenchmarkId::new("lurkrs with compressed", &name),
                |b| {
                    b.iter(|| {
                        let map = collect(&witness_i);
                        let (com_i, com_points) = compress(&points[..witness_n], map);
                        let com_n = com_i.len();
                        let _ = pasta_msm::pallas(
                            &com_points,
                            &com_i,
                            com_n,
                        );
                    })
                },
            );

            group.bench_function(
                BenchmarkId::new("lurkrs new_compress", &name),
                |b| {
                    b.iter(|| {
                        let n = witness_i.len();
                        let _ = new_compress(&points[..n], &witness_i);
                    })
                },
            );

            group.bench_function(
                BenchmarkId::new("lurkrs with new_compressed", &name),
                |b| {
                    b.iter(|| {
                        let n = witness_i.len();
                        let (com_i, com_points) = new_compress(&points[..n], &witness_i);
                        let _ = pasta_msm::pallas(
                            &com_points,
                            &com_i,
                            com_i.len(),
                        );
                    })
                },
            );
        }

        //     // group.bench_function(format!("biased {} scalars", npoints), |b| {
        //     //     b.iter(|| {
        //     //         let _ =
        //     //             pasta_msm::pallas(&points, &nonuniform_scalars, npoints);
        //     //     })
        //     // });

        //     // for npoints in [7941351, 9699051] {
        //     //     group.bench_function(format!("random {} scalars", npoints), |b| {
        //     //         b.iter(|| {
        //     //             let _ = pasta_msm::pallas(&points, &scalars, npoints);
        //     //         })
        //     //     });
        //     // }

        //     // let context = pasta_msm::pallas_init(&points, npoints);

        //     // group.bench_function(format!("preallocated lurkrs {} scalars", npoints), |b| {
        //     //     b.iter(|| {
        //     //         let _ = pasta_msm::pallas_with(&context, npoints, npoints, &witness_primary);
        //     //     })
        //     // });

        //     // group.bench_function(
        //     //     format!("preallocated biased {} scalars", npoints),
        //     //     |b| {
        //     //         b.iter(|| {
        //     //             let _ = pasta_msm::pallas_with(
        //     //                 &context,
        //     //                 npoints,
        //     //                 npoints,
        //     //                 &nonuniform_scalars,
        //     //             );
        //     //         })
        //     //     },
        //     // );

        //     // group.bench_function(
        //     //     format!("preallocated random {} scalars", npoints),
        //     //     |b| {
        //     //         b.iter(|| {
        //     //             let _ = pasta_msm::pallas_with(
        //     //                 &context, npoints, npoints, &scalars,
        //     //             );
        //     //         })
        //     //     },
        //     // );

        //     // group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
