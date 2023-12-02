use std::{
    mem::transmute,
    sync::atomic::{AtomicUsize, Ordering}, cell::UnsafeCell,
};

use pasta_curves::{arithmetic::CurveExt, group::{Curve, ff::PrimeField}, pallas};
use rand::RngCore;


#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

pub fn generate_points(npoints: usize) -> Vec<pallas::Affine> {
    let mut ret: Vec<pallas::Affine> = Vec::with_capacity(npoints);
    unsafe { ret.set_len(npoints) };

    let mut rnd: Vec<u8> = Vec::with_capacity(32 * npoints);
    unsafe { rnd.set_len(32 * npoints) };
    rand::thread_rng().fill_bytes(&mut rnd);

    let n_workers = rayon::current_num_threads();
    let work = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let hash = pallas::Point::hash_to_curve("foobar");

                let mut stride = 1024;
                let mut tmp: Vec<pallas::Point> = Vec::with_capacity(stride);
                unsafe { tmp.set_len(stride) };

                loop {
                    let work = work.fetch_add(stride, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    if work + stride > npoints {
                        stride = npoints - work;
                        unsafe { tmp.set_len(stride) };
                    }
                    for i in 0..stride {
                        let off = (work + i) * 32;
                        tmp[i] = hash(&rnd[off..off + 32]);
                    }
                    #[allow(mutable_transmutes)]
                    pallas::Point::batch_normalize(&tmp, unsafe {
                        transmute::<&[pallas::Affine], &mut [pallas::Affine]>(
                            &ret[work..work + stride],
                        )
                    });
                }
            })
        }
    });

    ret
}

pub fn generate_scalars<F: PrimeField>(len: usize) -> Vec<F> {
    let mut rng = rand::thread_rng();

    let scalars = (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>();

    scalars
}

fn as_mut<T>(x: &T) -> &mut T {
    unsafe { &mut *UnsafeCell::raw_get(x as *const _ as *const _) }
}

pub fn naive_multiscalar_mul(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Affine {
    let n_workers = rayon::current_num_threads();

    let mut rets: Vec<pallas::Point> = Vec::with_capacity(n_workers);
    unsafe { rets.set_len(n_workers) };

    let npoints = points.len();
    let work = AtomicUsize::new(0);
    let tid = AtomicUsize::new(0);
    rayon::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|_| {
                let mut ret = pallas::Point::default();

                loop {
                    let work = work.fetch_add(1, Ordering::Relaxed);
                    if work >= npoints {
                        break;
                    }
                    ret += points[work] * scalars[work];
                }

                *as_mut(&rets[tid.fetch_add(1, Ordering::Relaxed)]) = ret;
            })
        }
    });

    let mut ret = pallas::Point::default();
    for i in 0..n_workers {
        ret += rets[i];
    }

    ret.to_affine()
}

/// cargo run --release --example msm
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };
    }

    let npow: usize = std::env::var("BENCH_NPOW")
        .unwrap_or("5".to_string())
        .parse()
        .unwrap();
    let npoints = 1usize << npow;

    let points = generate_points(npoints);
    let scalars = generate_scalars(npoints);

    let double_scalars = scalars.iter().map(|x| x + x).collect::<Vec<_>>();

    let naive = naive_multiscalar_mul(&points, &double_scalars);
    println!("{:?}", naive);

    let ret = pasta_msm::pallas(&points, &scalars).to_affine();
    println!("{:?}", ret);

    assert_eq!(ret, naive);
}
