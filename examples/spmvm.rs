use pasta_curves::{group::ff::PrimeField, pallas, arithmetic::CurveAffine};
use pasta_curves::group::Curve;

use pasta_msm::{utils::SparseMatrix, CudaSparseMatrix};
use rand::Rng;

pub fn generate_random_csr<F: PrimeField>(
    n: usize,
    m: usize,
) -> (SparseMatrix<F>, CudaSparseMatrix<F>) {
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
        data: data.clone(),
        indices: col_idx.clone(),
        indptr: row_ptr.clone(),
        cols: m,
    };
    let cuda_csr = CudaSparseMatrix::new(data, col_idx, row_ptr, n, m);

    (csr, cuda_csr)
}

pub fn generate_scalars<F: PrimeField>(len: usize) -> Vec<F> {
    let mut rng = rand::thread_rng();

    let scalars = (0..len).map(|_| F::random(&mut rng)).collect::<Vec<_>>();

    scalars
}

fn main() {
    pasta_msm::add_test_pallas();

    // let npow: usize = std::env::var("BENCH_NPOW")
    //     .unwrap_or("5".to_string())
    //     .parse()
    //     .unwrap();
    // let n = 1usize << npow;
    // let x = generate_scalars::<pallas::Scalar>(n);
    // let y = generate_scalars::<pallas::Scalar>(n);

    // let cuda_out = pasta_msm::scalar_add_test_pallas(x, y);
    // println!("{}", cuda_out.len());
    // let out = x
    //     .iter()
    //     .zip(y.iter())
    //     .map(|(x, y)| x + y)
    //     .collect::<Vec<_>>();
    // assert_eq!(out, cuda_out);

    // let (csr, cuda_csr) = generate_random_csr::<pallas::Scalar>(n, n);
    // let scalars = generate_scalars::<pallas::Scalar>(n);

    // let res = csr.multiply_vec(&scalars);
    // // let c_cpu_res = pasta_msm::sparse_matrix_vector_cpu_pallas(&cuda_csr, &scalars);
    // let cuda_res = pasta_msm::sparse_matrix_vector_pallas(&cuda_csr, &scalars);
    // assert_eq!(res, cuda_res);
    
    println!("success!");
}
