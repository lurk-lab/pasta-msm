#![allow(non_snake_case)]

use std::marker::PhantomData;
use pasta_curves::{group::ff::Field, pallas, vesta};

#[repr(C)]
pub struct CudaSparseMatrix<'a, F> {
    pub data: *const F,
    pub col_idx: *const usize,
    pub row_ptr: *const usize,

    pub num_rows: usize,
    pub num_cols: usize,
    pub nnz: usize,

    _p: PhantomData<&'a F>,
}

impl<'a, F> CudaSparseMatrix<'a, F> {
    pub fn new(
        data: &[F],
        col_idx: &[usize],
        row_ptr: &[usize],
        num_rows: usize,
        num_cols: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            col_idx.len(),
            "data and col_idx length mismatch"
        );
        assert_eq!(
            row_ptr.len(),
            num_rows + 1,
            "row_ptr length and num_rows mismatch"
        );

        let nnz = data.len();
        CudaSparseMatrix {
            data: data.as_ptr(),
            col_idx: col_idx.as_ptr(),
            row_ptr: row_ptr.as_ptr(),
            num_rows,
            num_cols,
            nnz,
            _p: PhantomData,
        }
    }
}

#[repr(C)]
pub struct CudaWitness<'a, F> {
    pub W: *const F,
    pub u: *const F,
    pub U: *const F,
    pub nW: usize,
    pub nU: usize,
    _p: PhantomData<&'a F>,
}

impl<'a, F> CudaWitness<'a, F> {
    pub fn new(
        W: &[F],
        u: &F,
        U: &[F],
    ) -> Self {
        let nW = W.len();
        let nU = U.len();
        CudaWitness {
            W: W.as_ptr(),
            u: u as *const _,
            U: U.as_ptr(),
            nW,
            nU,
            _p: PhantomData,
        }
    }
}

pub fn sparse_matrix_mul_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    scalars: &[pallas::Scalar],
    nthreads: usize,
) -> Vec<pallas::Scalar> {
    extern "C" {
        fn cuda_sparse_matrix_mul_pallas(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            scalars: *const pallas::Scalar,
            out: *mut pallas::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let mut out = vec![pallas::Scalar::ZERO; csr.num_rows];
    let err = unsafe {
        cuda_sparse_matrix_mul_pallas(
            csr as *const _,
            scalars.as_ptr(),
            out.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    out
}

pub fn sparse_matrix_witness_pallas(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    witness: &CudaWitness<pallas::Scalar>,
    buffer: &mut [pallas::Scalar],
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_pallas(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            witness: *const CudaWitness<pallas::Scalar>,
            out: *mut pallas::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    assert_eq!(witness.nW + witness.nU + 1, csr.num_cols, "invalid witness size");

    let err = unsafe {
        cuda_sparse_matrix_witness_pallas(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn sparse_matrix_witness_pallas_cpu(
    csr: &CudaSparseMatrix<pallas::Scalar>,
    witness: &CudaWitness<pallas::Scalar>,
    buffer: &mut [pallas::Scalar],
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_pallas_cpu(
            csr: *const CudaSparseMatrix<pallas::Scalar>,
            witness: *const CudaWitness<pallas::Scalar>,
            out: *mut pallas::Scalar,
        ) -> sppark::Error;
    }

    assert_eq!(witness.nW + witness.nU + 1, csr.num_cols, "invalid witness size");

    let err = unsafe {
        cuda_sparse_matrix_witness_pallas_cpu(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn sparse_matrix_mul_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    scalars: &[vesta::Scalar],
    nthreads: usize,
) -> Vec<vesta::Scalar> {
    extern "C" {
        fn cuda_sparse_matrix_mul_vesta(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            scalars: *const vesta::Scalar,
            out: *mut vesta::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    let mut out = vec![vesta::Scalar::ZERO; csr.num_rows];
    let err = unsafe {
        cuda_sparse_matrix_mul_vesta(
            csr as *const _,
            scalars.as_ptr(),
            out.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    out
}

pub fn sparse_matrix_witness_vesta(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    witness: &CudaWitness<vesta::Scalar>,
    buffer: &mut [vesta::Scalar],
    nthreads: usize,
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_vesta(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            witness: *const CudaWitness<vesta::Scalar>,
            out: *mut vesta::Scalar,
            nthreads: usize,
        ) -> sppark::Error;
    }

    assert_eq!(witness.nW + witness.nU + 1, csr.num_cols, "invalid witness size");

    let err = unsafe {
        cuda_sparse_matrix_witness_vesta(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
            nthreads,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn sparse_matrix_witness_vesta_cpu(
    csr: &CudaSparseMatrix<vesta::Scalar>,
    witness: &CudaWitness<vesta::Scalar>,
    buffer: &mut [vesta::Scalar],
) {
    extern "C" {
        fn cuda_sparse_matrix_witness_vesta_cpu(
            csr: *const CudaSparseMatrix<vesta::Scalar>,
            witness: *const CudaWitness<vesta::Scalar>,
            out: *mut vesta::Scalar,
        ) -> sppark::Error;
    }

    assert_eq!(witness.nW + witness.nU + 1, csr.num_cols, "invalid witness size");

    let err = unsafe {
        cuda_sparse_matrix_witness_vesta_cpu(
            csr as *const _,
            witness as *const _,
            buffer.as_mut_ptr(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}
