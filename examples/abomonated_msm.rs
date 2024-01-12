use std::{io::{Read, BufWriter, Write}, time::Instant, collections::HashMap};

use abomonation::Abomonation;
use num::{BigInt, Num, Zero};
use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::utils::{gen_scalars, CommitmentKey};

use plotters::prelude::*;
use rayon::{iter::{IntoParallelRefIterator, ParallelIterator}, slice::ParallelSliceMut};    
use std::fs::OpenOptions;
use std::io::BufReader;

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
    eprintln!("Result has been saved to {}", out_file);

    Ok(())
}

fn read_abomonated<T: Abomonation + Clone>(name: String) -> std::io::Result<T> {

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

struct Stats {
    count: usize,
    total: usize,
}

fn write_stats(scalars: &[pallas::Scalar], out_file: &str) -> std::io::Result<()> {
    let stats = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(out_file)?;

    let mut writer = BufWriter::new(stats);
    let mut hist: HashMap<BigInt, usize> = HashMap::new();

    scalars.iter().for_each(|x| {
        let x = format!("{:?}", x);
        let num = BigInt::from_str_radix(&x[2..], 16).unwrap();
        *hist.entry(num).or_insert(0) += 1;
    });

    let mut entries: Vec<_> = hist.par_iter().collect();
    entries.par_sort_by(|a, b| b.1.cmp(a.1));

    let mut stats_100 = Stats { count: 0, total: 0};
    let mut stats_500 = Stats { count: 0, total: 0};
    let mut stats_1000 = Stats { count: 0, total: 0};
    for (_, value) in entries.iter() {
        if **value > 1000 {
            stats_1000.count += 1;
            stats_1000.total += *value;
        } 
        if **value > 500 {
            stats_500.count += 1;
            stats_500.total += *value;
        } 
        if **value > 100 {
            stats_100.count += 1;
            stats_100.total += *value;
        } 
    }

    writeln!(&mut writer, "scalars length: {}", scalars.len())?;
    let p1000 = 100.0 * (stats_1000.total as f64 / scalars.len() as f64);
    let p500 = 100.0 * (stats_500.total as f64 / scalars.len() as f64);
    let p100 = 100.0 * (stats_100.total as f64 / scalars.len() as f64);
    writeln!(&mut writer, "freq>1000 (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_1000.count, stats_1000.total, p1000)?;
    writeln!(&mut writer, "freq>500  (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_500.count, stats_500.total, p500)?;
    writeln!(&mut writer, "freq>100  (count, total, %): {:>10}, {:>10}, {:>10.2}%", stats_100.count, stats_100.total, p100)?;
    writeln!(&mut writer, "")?;
    writeln!(&mut writer, "top 100 values:")?;

    let mut count = 0;
    // Print sorted (key, value) pairs
    for (key, value) in entries {
        if count < 100 {
            writeln!(&mut writer, "{:>2}, {:>64}, {:?}", count, key.to_str_radix(16), value)?;
        }
        count += 1;
    }
    writer.flush()?;
    Ok(())
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
        
        // let out_file = format!("plots/lurkrs_{i}.png");
        // plot_scalars(&witness_i, &out_file).unwrap();
        let out_stats = format!("stats/{i}.txt");
        write_stats(&witness_i, &out_stats).unwrap();

        println!("done {i}");
    }
}
