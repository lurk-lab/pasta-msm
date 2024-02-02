#![allow(unused)]
use std::{
    collections::HashMap,
    io::{BufWriter, Read, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::Instant,
};

use abomonation::Abomonation;
use dashmap::DashMap;
use num::{BigInt, Num, Zero};
use pasta_curves::{
    group::ff::{Field, PrimeField},
    pallas,
};
use pasta_msm::utils::collect;

use plotters::prelude::*;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
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
        .build_cartesian_2d((0u32..1000u32).into_segmented(), 0u32..50u32)?;

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

fn get_witness(i: usize) -> Vec<pallas::Scalar> {
    let witness_i =
        read_abomonated::<Vec<<pallas::Scalar as PrimeField>::Repr>>(
            i.to_string(),
        )
        .unwrap();
    unsafe { std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_i) }
}

struct Stats {
    count: usize,
    total: usize,
}

fn write_stats(
    scalars: &[pallas::Scalar],
    out_file: &str,
) -> std::io::Result<()> {
    let stats = OpenOptions::new()
        .read(true)
        .write(true)
        .truncate(true)
        .create(true)
        .open(out_file)?;

    let mut writer = BufWriter::new(stats);
    let map = collect(&scalars);

    let mut entries: Vec<_> = map.into_par_iter().collect();
    entries.par_sort_by(|a, b| {
        let len_cmp = b.1.len().cmp(&a.1.len());
        if len_cmp == std::cmp::Ordering::Equal {
            a.1[0].cmp(&b.1[0])
        } else {
            len_cmp
        }
    });

    let mut stats_20 = Stats { count: 0, total: 0 };
    let mut stats_100 = Stats { count: 0, total: 0 };
    let mut stats_500 = Stats { count: 0, total: 0 };
    let mut stats_1000 = Stats { count: 0, total: 0 };
    for (_, indices) in entries.iter() {
        let len = indices.len();
        if len > 1000 {
            stats_1000.count += 1;
            stats_1000.total += len;
        }
        if len > 500 {
            stats_500.count += 1;
            stats_500.total += len;
        }
        if len > 100 {
            stats_100.count += 1;
            stats_100.total += len;
        }
        if len > 20 {
            stats_20.count += 1;
            stats_20.total += len;
        }
    }

    writeln!(&mut writer, "scalars length: {}", scalars.len())?;
    let p1000 = 100.0 * (stats_1000.total as f64 / scalars.len() as f64);
    let p500 = 100.0 * (stats_500.total as f64 / scalars.len() as f64);
    let p100 = 100.0 * (stats_100.total as f64 / scalars.len() as f64);
    let p20 = 100.0 * (stats_20.total as f64 / scalars.len() as f64);
    writeln!(
        &mut writer,
        "freq>1000 (count, total, %): {:>10}, {:>10}, {:>10.2}%",
        stats_1000.count, stats_1000.total, p1000
    )?;
    writeln!(
        &mut writer,
        "freq>500  (count, total, %): {:>10}, {:>10}, {:>10.2}%",
        stats_500.count, stats_500.total, p500
    )?;
    writeln!(
        &mut writer,
        "freq>100  (count, total, %): {:>10}, {:>10}, {:>10.2}%",
        stats_100.count, stats_100.total, p100
    )?;
    writeln!(
        &mut writer,
        "freq>20   (count, total, %): {:>10}, {:>10}, {:>10.2}%",
        stats_20.count, stats_20.total, p20
    )?;
    writeln!(&mut writer, "")?;
    writeln!(&mut writer, "top values:")?;

    let mut count = 0;
    let mut prev: Vec<usize> = vec![];
    // Print sorted (key, value) pairs
    for (key, mut indices) in entries {
        let len = indices.len();
        if len > 1 {
            writeln!(&mut writer, "{:>4}, {:>64?}, {:?}", count, key, len)?;
            writeln!(
                &mut writer,
                "      {:?}",
                &indices[..std::cmp::min(len, 20)]
            )?;
            if indices.len() == prev.len() {
                let offsets = indices
                    .iter()
                    .zip(prev.iter())
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let equal = offsets.iter().all(|x| x == &offsets[0]);
                if equal {
                    writeln!(&mut writer, "      offset: {}", offsets[0] as isize);
                }
            }
        }
        prev = indices;
        count += 1;
    }
    writer.flush()?;
    Ok(())
}

fn generate_stats_and_plots() {
    for i in 0..32 {
        let witness_i = read_abomonated::<
            Vec<<pallas::Scalar as PrimeField>::Repr>,
        >(i.to_string())
        .unwrap();
        let witness_i = unsafe {
            std::mem::transmute::<Vec<_>, Vec<pallas::Scalar>>(witness_i)
        };

        // let witness_n = witness_i.len();

        // if witness_n < 1_000_000 {
        //     continue;
        // }

        let out_file = format!("plots/rc=1/{i}.png");
        plot_scalars(&witness_i, &out_file).unwrap();
        let out_stats = format!("stats/rc=1/{i}.txt");
        write_stats(&witness_i, &out_stats).unwrap();

        println!("done {i}");
    }
}

fn generate_subset_analysis() {
    for i in (3..32).step_by(4) {
        let out_file = format!("subset_stats/rc=1/{}", i);
        let subsets = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(out_file)
            .unwrap();
        let mut writer = Mutex::new(BufWriter::new(subsets));

        let curr = i;
        let next = i + 4;

        let witness_curr = get_witness(curr);
        let witness_next = get_witness(next);

        let map_curr = collect(&witness_curr);
        let map_next = collect(&witness_next);

        let mut total = AtomicUsize::new(0);

        map_curr.into_par_iter().for_each(|(x, indices_curr)| {
            for data in map_next.iter() {
                let y = data.key();
                let indices_next = data.value();
                let n = indices_curr.len();
                if indices_curr[0] == indices_next[0] && n > 1 {
                    let mut inter = vec![];
                    let mut curr = vec![];
                    let mut next = vec![];
                    for (a, b) in indices_curr.iter().zip(indices_next) {
                        if a == b {
                            inter.push(*a);
                        } else {
                            curr.push(*a);
                            next.push(*b);
                        }
                    }
                    let mut writer = writer.lock().unwrap();
                    writeln!(&mut writer, ">>> fuzzy head equality:").unwrap();
                    writeln!(&mut writer, "    x:       {:?}", x).unwrap();
                    writeln!(&mut writer, "    y:       {:?}", y).unwrap();
                    writeln!(
                        &mut writer,
                        "    inter: {:?}",
                        &inter[..std::cmp::min(20, inter.len())]
                    )
                    .unwrap();
                    writeln!(
                        &mut writer,
                        "    - curr: {:?}",
                        &curr[..std::cmp::min(20, curr.len())]
                    )
                    .unwrap();
                    writeln!(
                        &mut writer,
                        "    + next: {:?}",
                        &next[..std::cmp::min(20, next.len())]
                    )
                    .unwrap();
                    writeln!(&mut writer, "    length:  {:?}", n).unwrap();
                    total.fetch_add(n, Ordering::Relaxed);
                    writeln!(&mut writer,);
                }
            }
        });

        let mut writer = writer.get_mut().unwrap();
        writeln!(&mut writer, "total: {:?}", total).unwrap();
        println!("done {i}");
    }
}

/// cargo run --release --example abomonated
fn main() {
    #[cfg(feature = "cuda")]
    if unsafe { cuda_available() } {
        unsafe { pasta_msm::CUDA_OFF = false };
    }

    generate_stats_and_plots()
    // generate_subset_analysis()
}
