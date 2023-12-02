// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

pub mod utils;

use std::{ffi::c_void, marker::PhantomData};

use pasta_curves::group::ff::Field;

extern crate semolina;

#[cfg(feature = "cuda")]
sppark::cuda_error!();
#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}
#[cfg(feature = "cuda")]
pub static mut CUDA_OFF: bool = false;

use pasta_curves::pallas;

extern "C" {
    fn mult_pippenger_pallas(
        out: *mut pallas::Point,
        points: *const pallas::Affine,
        npoints: usize,
        scalars: *const pallas::Scalar,
        is_mont: bool,
    );

}
pub fn pallas(
    points: &[pallas::Affine],
    scalars: &[pallas::Scalar],
) -> pallas::Point {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
    #[cfg(feature = "cuda")]
    if unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_pallas(
                out: *mut pallas::Point,
                points: *const pallas::Affine,
                npoints: usize,
                scalars: *const pallas::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = pallas::Point::default();
        let err = unsafe {
            cuda_pippenger_pallas(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = pallas::Point::default();
    unsafe {
        mult_pippenger_pallas(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

use pasta_curves::vesta;

extern "C" {
    fn mult_pippenger_vesta(
        out: *mut vesta::Point,
        points: *const vesta::Affine,
        npoints: usize,
        scalars: *const vesta::Scalar,
        is_mont: bool,
    );

}
pub fn vesta(
    points: &[vesta::Affine],
    scalars: &[vesta::Scalar],
) -> vesta::Point {
    let npoints = points.len();
    if npoints != scalars.len() {
        panic!("length mismatch")
    }
    #[cfg(feature = "cuda")]
    if unsafe { !CUDA_OFF && cuda_available() } {
        extern "C" {
            fn cuda_pippenger_vesta(
                out: *mut vesta::Point,
                points: *const vesta::Affine,
                npoints: usize,
                scalars: *const vesta::Scalar,
                is_mont: bool,
            ) -> cuda::Error;

        }
        let mut ret = vesta::Point::default();
        let err = unsafe {
            cuda_pippenger_vesta(
                &mut ret,
                &points[0],
                npoints,
                &scalars[0],
                true,
            )
        };
        if err.code != 0 {
            panic!("{}", String::from(err));
        }
        return ret;
    }
    let mut ret = vesta::Point::default();
    unsafe {
        mult_pippenger_vesta(&mut ret, &points[0], npoints, &scalars[0], true)
    };
    ret
}

#[repr(C)]
pub struct CudaSparseMatrix<F> {
    data: *const c_void,
    col_idx: *const c_void,
    row_ptr: *const c_void,

    row_size: usize,
    col_size: usize,
    nnz: usize,
    _p: PhantomData<F>,
}

impl<F> CudaSparseMatrix<F> {
    pub fn new(
        data: Vec<F>,
        col_idx: Vec<usize>,
        row_ptr: Vec<usize>,
        row_size: usize,
        col_size: usize,
    ) -> Self {
        assert!(
            data.len() == data.capacity()
                && col_idx.len() == col_idx.capacity()
                && row_ptr.len() == row_ptr.capacity(),
            "must ensure length == capacity for FFI"
        );
        assert_eq!(
            data.len(),
            col_idx.len(),
            "data and col_idx length mismatch"
        );
        assert_eq!(
            row_ptr.len(),
            row_size + 1,
            "row_ptr length and row_size mismatch"
        );

        let data_ptr = data.as_ptr();
        let col_idx_ptr = col_idx.as_ptr();
        let row_ptr_ptr = row_ptr.as_ptr();
        let nnz = data.len();

        std::mem::forget(data);
        std::mem::forget(col_idx);
        std::mem::forget(row_ptr);

        CudaSparseMatrix {
            data: data_ptr as *const c_void,
            col_idx: col_idx_ptr as *const c_void,
            row_ptr: row_ptr_ptr as *const c_void,
            row_size,
            col_size,
            nnz,
            _p: PhantomData,
        }
    }
}

// Ensure to add proper cleanup for CudaSparseMatrix to avoid memory leaks.
impl<F> Drop for CudaSparseMatrix<F> {
    fn drop(&mut self) {
        unsafe {
            let _ =
                Vec::from_raw_parts(self.data as *mut F, self.nnz, self.nnz);
            let _ = Vec::from_raw_parts(
                self.col_idx as *mut usize,
                self.nnz,
                self.nnz,
            );
            let _ = Vec::from_raw_parts(
                self.row_ptr as *mut usize,
                self.row_size + 1,
                self.row_size + 1,
            );
        }
    }
}

pub fn sparse_matrix_vector_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    scalars: &[pallas::Scalar],
) -> Vec<pallas::Scalar> {
    extern "C" {
        fn spmvm_pallas(
            out: *mut pallas::Scalar,
            csr: &CudaSparseMatrix<pallas::Scalar>,
            scalars: *const pallas::Scalar,
        ) -> sppark::Error;
    }

    let ret = vec![pallas::Scalar::ZERO; csr.row_size];
    let err = unsafe {
        spmvm_pallas(ret.as_ptr() as *mut _, csr, scalars.as_ptr() as *const _)
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn sparse_matrix_vector_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    scalars: &[vesta::Scalar],
) -> Vec<vesta::Scalar> {
    extern "C" {
        fn spmvm_vesta(
            out: *mut vesta::Scalar,
            csr: &CudaSparseMatrix<vesta::Scalar>,
            scalars: *const vesta::Scalar,
        ) -> sppark::Error;
    }

    let ret = Vec::with_capacity(csr.row_size);
    let err = unsafe {
        spmvm_vesta(ret.as_ptr() as *mut _, csr, scalars.as_ptr() as *const _)
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn sparse_matrix_vector_cpu_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    scalars: &[pallas::Scalar],
) -> Vec<pallas::Scalar> {
    extern "C" {
        fn spmvm_cpu_pallas(
            out: *mut pallas::Scalar,
            csr: &CudaSparseMatrix<pallas::Scalar>,
            scalars: *const pallas::Scalar,
        ) -> sppark::Error;
    }

    let ret = vec![pallas::Scalar::ZERO; csr.row_size];
    let err = unsafe {
        spmvm_cpu_pallas(
            ret.as_ptr() as *mut _,
            csr,
            scalars.as_ptr() as *const _,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn sparse_matrix_vector_cpu_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    scalars: &[vesta::Scalar],
) -> Vec<vesta::Scalar> {
    extern "C" {
        fn spmvm_cpu_vesta(
            out: *mut vesta::Scalar,
            csr: &CudaSparseMatrix<vesta::Scalar>,
            scalars: *const vesta::Scalar,
        ) -> sppark::Error;
    }

    let ret = Vec::with_capacity(csr.row_size);
    let err = unsafe {
        spmvm_cpu_vesta(
            ret.as_ptr() as *mut _,
            csr,
            scalars.as_ptr() as *const _,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn add_test_pallas() {
    extern "C" {
        fn add_test_pallas();
    }

    unsafe { add_test_pallas() };
}

// pub fn scalar_add_test_pallas(
//     x: Vec<pallas::Scalar>,
//     y: Vec<pallas::Scalar>,
// ) -> Vec<pallas::Scalar> {
//     extern "C" {
//         fn scalar_add_test_pallas(
//             n: usize,
//             x: *const pallas::Scalar,
//             y: *const pallas::Scalar,
//             out: *mut pallas::Scalar,
//         ) -> sppark::Error;
//     }

//     let n = x.len();
//     let mut out = vec![pallas::Scalar::ZERO; n];
//     let err = unsafe {
//         scalar_add_test_pallas(n, x.as_ptr(), y.as_ptr(), out.as_mut_ptr())
//     };
//     if err.code != 0 {
//         panic!("{}", String::from(err));
//     }

//     out
// }

include!("tests.rs");
