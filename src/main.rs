mod layer;
mod network;
mod utils;
mod tests;

use nalgebra::DVector;
use mnist::*;
use crate::layer::{Conv2D, Dense, LeakyReLU, MaxPooling2D, SoftMax};
use crate::network::Network;

/*
fn loss_moving_average(loss: Vec<f32>) -> Vec<f32> {
    let n = loss.len() / 100;
    let mut smooth_loss = vec![loss[0..(2 * n + 1)].iter().sum::<f32>() / (2 * n + 1) as f32];
    for i in (n + 1)..(loss.len() - n) {
        let x = smooth_loss.last().unwrap() + (loss[i + n] - loss[i - n - 1]) / (2 * n + 1) as f32;
        smooth_loss.push(x);
    }
    smooth_loss
}

fn draw_digit(digit: &DVector<f32>) {
    println!("  — — — — — — — — — — — — — — — — — — — — — — — — — — — — ");
    for i in 0..28 {
        print!("| ");
        for j in 0..28 {
            if digit[i * 28 + j] > 0.0 {
                print!("#");
            }
            else {
                print!(" ");
            }
            print!(" ");
        }
        println!("|");
    }
    println!("  — — — — — — — — — — — — — — — — — — — — — — — — — — — — ");
}

*/

fn prepare_inputs(inputs: Vec<u8>) -> Vec<DVector<f32>> {
    inputs.chunks_exact(784)
        .map(|x| DVector::<f32>::from_iterator(784, x.iter().map(|z| (*z as f32) / 255.0)))
        .collect::<Vec<_>>()
}

fn prepare_labels(labels: Vec<u8>) -> Vec<DVector<f32>> {
    labels.iter()
        .map(|x| DVector::<f32>::from_fn(10, |z, _| f32::from(z == (*x as usize))))
        .collect::<Vec<_>>()
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let train_inputs = prepare_inputs(trn_img);
    let train_labels = prepare_labels(trn_lbl);
    let valid_inputs = prepare_inputs(val_img);
    let valid_labels = prepare_labels(val_lbl);
    let test_inputs = prepare_inputs(tst_img);
    let test_labels = prepare_labels(tst_lbl);

    let mut network = Network::new(
        vec![
            Box::new(Conv2D::new((1, 28, 28), (3, 3), 8)),
            Box::new(LeakyReLU::new()),
            Box::new(MaxPooling2D::new((8, 26, 26), (2, 2))),
            Box::new(Conv2D::new((8, 13, 13), (3, 3), 16)),
            Box::new(LeakyReLU::new()),
            Box::new(MaxPooling2D::new((16, 11, 11), (2, 2))),
            Box::new(Dense::new(16 * 6 * 6, 80)),
            Box::new(LeakyReLU::new()),
            Box::new(Dense::new(80, 10)),
            Box::new(SoftMax::new())
        ]
    );

    network.train(&train_inputs, &train_labels, &valid_inputs, &valid_labels, 32, 32);
    let accuracy = network.test_accuracy(&test_inputs, &test_labels);
    println!("Results : {:.2}% success rate", accuracy * 100.0);
}
