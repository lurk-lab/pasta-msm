// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

pub mod spmvm;
pub mod utils;

extern crate semolina;

#[cfg(feature = "cuda")]
sppark::cuda_error!();
#[cfg(feature = "cuda")]
extern "C" {
    pub fn cuda_available() -> bool;
}
#[cfg(feature = "cuda")]
pub static mut CUDA_OFF: bool = false;

macro_rules! multi_scalar_mult {
    (
        $pasta:ident,
        $mult:ident,
        $cuda_mult:ident,
        $msm_context:ident
    ) => {
        use pasta_curves::$pasta;

        extern "C" {
            fn $mult(
                out: *mut $pasta::Point,
                points: *const $pasta::Affine,
                npoints: usize,
                scalars: *const $pasta::Scalar,
                is_mont: bool,
            );
        }

        paste::paste! {
            #[cfg(feature = "cuda")]
            #[repr(C)]
            #[derive(Debug, Clone)]
            pub struct $msm_context {
                context: *const std::ffi::c_void,
            }

            #[cfg(feature = "cuda")]
            unsafe impl Send for $msm_context {}

            #[cfg(feature = "cuda")]
            unsafe impl Sync for $msm_context {}

            #[cfg(feature = "cuda")]
            impl Default for $msm_context {
                fn default() -> Self {
                    Self { context: std::ptr::null() }
                }
            }

            #[cfg(feature = "cuda")]
            // TODO: check for device-side memory leaks
            impl Drop for $msm_context {
                fn drop(&mut self) {
                    extern "C" {
                        fn [<drop_msm_context_ $pasta>](by_ref: &$msm_context);
                    }
                    unsafe { [<drop_msm_context_ $pasta>](std::mem::transmute::<&_, &_>(self)) };
                    self.context = core::ptr::null();
                }
            }

            #[cfg(feature = "cuda")]
            pub fn [<$pasta _init>](
                points: &[$pasta::Affine],
                npoints: usize,
            ) -> $msm_context {
                unsafe { assert!(!CUDA_OFF && cuda_available(), "feature = \"cuda\" must be enabled") };
                if npoints != points.len() && npoints < 1 << 16 {
                    panic!("length mismatch or less than 10**16")
                }
                extern "C" {
                    fn [<cuda_pippenger_ $pasta _init>](
                        points: *const pallas::Affine,
                        npoints: usize,
                        d_points: &mut $msm_context,
                        is_mont: bool,
                    ) -> cuda::Error;

                }

                let mut ret = $msm_context::default();
                let err = unsafe {
                    [<cuda_pippenger_ $pasta _init>](
                        points.as_ptr() as *const _,
                        npoints,
                        &mut ret,
                        true,
                    )
                };
                if err.code != 0 {
                    panic!("{}", String::from(err));
                }
                ret
            }

            #[cfg(feature = "cuda")]
            pub fn [<$pasta _with>](
                context: &$msm_context,
                npoints: usize,
                nnz: usize,
                scalars: &[$pasta::Scalar],
            ) -> $pasta::Point {
                unsafe { assert!(!CUDA_OFF && cuda_available(), "feature = \"cuda\" must be enabled") };
                if npoints != scalars.len() && npoints < 1 << 16 {
                    panic!("length mismatch or less than 10**16")
                }
                extern "C" {
                    fn [<cuda_pippenger_ $pasta _with>](
                        out: *mut $pasta::Point,
                        context: &$msm_context,
                        npoints: usize,
                        nnz: usize,
                        scalars: *const $pasta::Scalar,
                        is_mont: bool,
                    ) -> cuda::Error;

                }

                let mut ret = $pasta::Point::default();
                let err = unsafe {
                    [<cuda_pippenger_ $pasta _with>](
                        &mut ret,
                        context,
                        npoints,
                        nnz,
                        &scalars[0],
                        true,
                    )
                };
                if err.code != 0 {
                    panic!("{}", String::from(err));
                }
                ret
            }
        }

        pub fn $pasta(
            points: &[$pasta::Affine],
            scalars: &[$pasta::Scalar],
            nnz: usize,
        ) -> $pasta::Point {
            let npoints = points.len();
            if npoints != scalars.len() {
                panic!("length mismatch")
            }

            #[cfg(feature = "cuda")]
            if npoints >= 1 << 16 && unsafe { !CUDA_OFF && cuda_available() } {
                extern "C" {
                    fn $cuda_mult(
                        out: *mut $pasta::Point,
                        points: *const $pasta::Affine,
                        npoints: usize,
                        nnz: usize,
                        scalars: *const $pasta::Scalar,
                        is_mont: bool,
                    ) -> cuda::Error;
                }
                let mut ret = $pasta::Point::default();
                let err = unsafe {
                    $cuda_mult(&mut ret, &points[0], npoints, nnz, &scalars[0], true)
                };
                if err.code != 0 {
                    panic!("{}", String::from(err));
                }
                return ret;
            }
            let mut ret = $pasta::Point::default();
            unsafe { $mult(&mut ret, &points[0], npoints, &scalars[0], true) };
            ret
        }
    };
}

multi_scalar_mult!(pallas, mult_pippenger_pallas, cuda_pippenger_pallas, MSMContextPallas);
multi_scalar_mult!(vesta, mult_pippenger_vesta, cuda_pippenger_vesta, MSMContextVesta);

#[cfg(test)]
mod tests {
    use pasta_curves::group::Curve;

    use crate::utils::{gen_points, gen_scalars, naive_multiscalar_mul};

    #[test]
    fn it_works() {
        #[cfg(not(debug_assertions))]
        const NPOINTS: usize = 128 * 1024;
        #[cfg(debug_assertions)]
        const NPOINTS: usize = 8 * 1024;

        let points = gen_points(NPOINTS);
        let scalars = gen_scalars(NPOINTS);

        let naive = naive_multiscalar_mul(&points, &scalars);
        println!("{:?}", naive);

        let ret = crate::pallas(&points, &scalars, NPOINTS).to_affine();
        println!("{:?}", ret);

        assert_eq!(ret, naive);
    }
}
