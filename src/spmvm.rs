use pasta_curves::pallas;

pub fn sparse_matrix_mul_pallas(scalars: &mut [pallas::Scalar]) {
    extern "C" {
        fn spmvm_pallas(
            scalars: *mut pallas::Scalar,
            nscalars: usize,
        ) -> sppark::Error;
    }

    let nscalars = scalars.len();
    let err = unsafe {
        spmvm_pallas(
            scalars.as_mut_ptr(),
            nscalars,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}