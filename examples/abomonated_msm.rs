use std::{io::Read, time::Instant, collections::HashMap};

use abomonation::Abomonation;
use num::{BigInt, Num, Zero};
use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::utils::{gen_scalars, CommitmentKey};

use plotters::prelude::*;
use rand::RngCore;

#[cfg(feature = "cuda")]
extern "C" {
    fn cuda_available() -> bool;
}

fn plot_scalars(
    scalars: &[pallas::Scalar],
    out_file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(out_file, (640, 480)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(35)
        .y_label_area_size(40)
        .margin(5)
        .caption(out_file, ("sans-serif", 50.0))
        .build_cartesian_2d((0u32..1000u32).into_segmented(), 0u32..50000u32)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc("Count")
        .x_desc("1000 * scalar / MODULUS")
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

/// cargo run --release --example abomonated
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };
    }

    for i in 0..32 {
        
        let witness_i = read_abomonated::<
            Vec<<pallas::Scalar as PrimeField>::Repr>,
        >(i.to_string())
        .unwrap();
        let witness_i = unsafe {
            std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_i)
        };

        let witness_n = witness_i.len();

        if witness_n < 1_000_000 {
            continue;
        }

        // println!("witness_n: {}", witness_n);

        // let mut total = BigInt::zero();
        // let mut dist: HashMap<BigInt, usize> = HashMap::new();
        // // let mut small_count = vec![0; 10];
        // for j in 0..witness_n {
        //     let wj = format!("{:?}", witness_i[j]);
        //     let num = BigInt::from_str_radix(&wj[2..], 16).unwrap();
        //     total += &num;
        //     *dist.entry(num).or_insert(0) += 1;
        // }
    
        // let average = total / witness_n;
        // println!("avg: {:?}", average);
        // println!("dist len: {:?}", dist.len());
        // // println!("smalls: {:?}", small_count);
        
        let out_file = format!("plots/lurkrs_{i}.png");
        plot_scalars(&witness_i, &out_file).unwrap();

        println!("done {i}\n");
    }
}
