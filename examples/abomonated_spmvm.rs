#![allow(non_snake_case)]

use std::{io::Read, time::Instant, collections::HashMap};

use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use num::{BigInt, Num, Zero};
use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::{utils::{gen_scalars, CommitmentKey, SparseMatrix}, spmvm::{pallas::sparse_matrix_mul_pallas, CudaSparseMatrix}};

use plotters::prelude::*;
use rand::RngCore;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

#[derive(Clone, Debug, PartialEq, Eq, Abomonation)]
#[abomonation_bounds(where <F as PrimeField>::Repr: Abomonation)]
struct R1CSShape<F: PrimeField> {
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: SparseMatrix<F>,
    B: SparseMatrix<F>,
    C: SparseMatrix<F>,
    #[abomonate_with(F::Repr)]
    digest: F,
}

fn plot_scalars(
    scalars: &[pallas::Scalar],
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(out_file, (1280 * 2, 960 * 2)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption("Histogram Test", ("sans-serif", 50.0))
        .build_cartesian_2d((0u32..1000u32).into_segmented(), 0u32..50000u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("Bucket")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    let field_size =
        BigInt::from_str_radix(&pallas::Scalar::MODULUS[2..], 16).unwrap();
    chart.draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(scalars.iter().map(|x| {
                let x = format!("{:?}", x);
                let x = BigInt::from_str_radix(&x[2..], 16).unwrap();
                let res: BigInt = (1000 * x) / &field_size;
                (*res.to_u32_digits().1.get(0).unwrap_or(&0), 1)
            })),
    )?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", out_file);

    Ok(())
}

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

/// cargo run --release --example abomonated_spmvm
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };
    }

    let r1cs_primary =
        read_abomonated::<R1CSShape<pallas::Scalar>>("r1cs_primary".into())
            .unwrap();
    let witness_primary = read_abomonated::<
        Vec<<pallas::Scalar as PrimeField>::Repr>,
    >("witness_primary".into())
    .unwrap();
    let mut witness_primary = unsafe {
        std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_primary)
    };
    witness_primary.push(pallas::Scalar::ZERO);
    witness_primary.push(pallas::Scalar::from(37));
    witness_primary.push(pallas::Scalar::from(42));

    let npoints = witness_primary.len();
    println!("npoints: {}", npoints);

    // let scalars = gen_scalars(npoints);

    let csr_a = CudaSparseMatrix::from(&r1cs_primary.A);
    let start = Instant::now();
    let res = sparse_matrix_mul_pallas(&csr_a, &witness_primary, 128);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res.len());

    let start = Instant::now();
    let res = r1cs_primary.A.multiply_vec(&witness_primary);
    println!("time: {:?}", start.elapsed());
    println!("res: {:?}", res.len());
}
