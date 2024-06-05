mod layer;
mod network;
mod tensor;

use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;
use std::fmt::Write;
use std::process::exit;
use mnist::*;
use plotters::prelude::*;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use plotters::style::full_palette::ORANGE;
use rayon::prelude::*;
use rand_distr::Normal;
use crate::layer::{Dense, LeakyReLU, SoftMax};
use crate::network::Network;

struct NeuralNetwork {
    layers: usize,
    batch_size: usize,
    weights: Vec<DMatrix<f32>>,
    bias: Vec<DMatrix<f32>>,

    weighted_sums: Vec<DMatrix<f32>>,
    activations: Vec<DMatrix<f32>>
}

impl NeuralNetwork {

    fn init_params(layers: Vec<usize>, batch_size: usize) -> NeuralNetwork {
        let rng = &mut rand::thread_rng();
        let mut weights = vec![];
        let mut bias = vec![];

        let n_layers = layers.len();
        let mut n_cols = 784;
        for n_rows in layers {
            weights.push(DMatrix::<f32>::from_distribution(n_rows, n_cols, &Normal::new(0.0, f32::sqrt(2.0 / n_cols as f32)).unwrap(), rng));
            bias.push(DMatrix::<f32>::zeros(n_rows, batch_size));
            n_cols = n_rows;
        }

        NeuralNetwork {
            layers: n_layers,
            batch_size,
            weights,
            bias,

            weighted_sums: vec![DMatrix::<f32>::zeros(1, 1); n_layers],
            activations: vec![DMatrix::<f32>::zeros(1, 1); n_layers]
        }
    }

    fn train(&mut self, training_inputs: &Vec<DVector<f32>>, expected_outputs: &Vec<DVector<f32>>, validation_inputs: &Vec<DVector<f32>>, validation_outputs: &Vec<DVector<f32>>, epochs: usize) -> (Vec<f32>, Vec<f32>) {
        let (inputs, outputs) = self.create_batches(training_inputs, expected_outputs);
        let n_batches = inputs.len();

        let pb = ProgressBar::new((epochs * n_batches) as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] Status : [{bar:50.green/blue}] {msg} ({eta})")
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap())
            .progress_chars("=> "));

        let mut loss_values = vec![];
        let mut accuracies = vec![];
        for epoch in 0..epochs {
            for i in 0..n_batches {
                // Forward propagation
                let prediction = self.forward_propagation(&inputs[i]);
                let loss = self.loss(&prediction, &outputs[i]);
                loss_values.push(loss);

                // Backward propagation
                let (dw, db) = self.backward_propagation(&inputs[i], &outputs[i]);
                self.gradient_descent(dw, db);

                if loss.is_nan() {
                    pb.abandon();
                    println!("Training failed ! NaN Loss detected...");
                    exit(1);
                }
                pb.inc(1);
                pb.set_message(format!(
                    "Progress : {:.1}%, {}/{} batches, epoch {}/{}",
                    100.0 * (epoch * n_batches + i + 1) as f32 / (epochs * n_batches) as f32,
                    i + 1, n_batches, epoch + 1, epochs
                ));
            }

            // Validation
            let accuracy = self.validation(validation_inputs, validation_outputs);
            accuracies.push(accuracy);
        }

        pb.finish();
        println!("Model finished training ! ({:.1}s)", pb.elapsed().as_secs_f32());
        (loss_values, accuracies)
    }

    fn create_batches(&self, inputs: &Vec<DVector<f32>>, outputs: &Vec<DVector<f32>>) -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
        let n_batches = inputs.len() / self.batch_size;
        let mut input_matrices = vec![];
        let mut output_matrices = vec![];
        for i in 0..n_batches {
            let index = i * self.batch_size;
            let input_matrix = DMatrix::<f32>::from_columns(&inputs[index..(index + self.batch_size)]);
            let output_matrix = DMatrix::<f32>::from_columns(&outputs[index..(index + self.batch_size)]);
            input_matrices.push(input_matrix);
            output_matrices.push(output_matrix);
        }
        (input_matrices, output_matrices)
    }

    fn forward_propagation(&mut self, input: &DMatrix<f32>) -> DMatrix<f32> {
        let mut output = input.clone();
        for k in 0..self.layers {
            self.weighted_sums[k] = self.weights[k].clone() * output + self.bias[k].clone();
            self.activations[k] = relu(&self.weighted_sums[k]);
            output = self.activations[k].clone();
        }
        output
    }

    fn loss(&self, prediction: &DMatrix<f32>, expected: &DMatrix<f32>) -> f32 {
        let loss_sum = (prediction - expected).norm_squared();
        loss_sum / self.batch_size as f32
    }

    fn backward_propagation(&mut self, input: &DMatrix<f32>, expected: &DMatrix<f32>) -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
        let (weight_gradients, bias_gradients): (Vec<_>, Vec<_>) = (0..self.batch_size).into_par_iter().map(|k| {
            let mut batch_dw = VecDeque::new();
            let mut batch_db = VecDeque::new();
            let mut factor = 2.0 * (self.activations[self.layers - 1].column(k) - expected.column(k)).transpose() * grad_relu(&self.weighted_sums[self.layers - 1].column(k).into());
            for l in (0..self.layers).rev() {
                if l < self.layers - 1 {
                    let grad_relu = grad_relu(&self.weighted_sums[l].column(k).into());
                    let mut product = self.weights[l + 1].clone();
                    for (i, mut column) in product.column_iter_mut().enumerate() {
                        column *= grad_relu[(i, i)];
                    }
                    factor = factor * product;
                }
                let (n, m) = self.weights[l].shape();
                batch_dw.push_front(DMatrix::<f32>::from_fn(n, m, |i, j| {
                    let a_j = if l > 0 { self.activations[l - 1].column(k)[j] } else { input.column(k)[j] };
                    factor[i] * a_j
                }));
                batch_db.push_front(factor.transpose());
            }
            (batch_dw, batch_db)
        }).unzip();

        let mut dw = weight_gradients[0].iter().cloned().collect::<Vec<_>>();
        let mut db = bias_gradients[0].iter().cloned().collect::<Vec<_>>();
        for i in 1..self.batch_size {
            for l in 0..self.layers {
                dw[l] += weight_gradients[i][l].clone();
                db[l] += bias_gradients[i][l].clone();
            }
        }

        (
            dw.into_iter().map(|x| x / self.batch_size as f32).collect(),
            db.into_iter().map(|x| DMatrix::<f32>::from_columns(&vec![x / self.batch_size as f32; self.batch_size])).collect()
        )
    }

    fn gradient_descent(&mut self, weight_gradients: Vec<DMatrix<f32>>, bias_gradients: Vec<DMatrix<f32>>) {
        let learning_rate = 0.01;
        for k in 0..self.layers {
            self.weights[k] -= learning_rate * weight_gradients[k].clone();
            self.bias[k] -= learning_rate * bias_gradients[k].clone();
        }
    }

    pub fn validation(&self, validation_inputs: &Vec<DVector<f32>>, validation_outputs: &Vec<DVector<f32>>) -> f32 {
        let accuracy = (0..validation_inputs.len()).into_iter().map(|i| {
            let prediction = self.predict(&validation_inputs[i]);
            if prediction.argmax().0 == validation_outputs[i].argmax().0 { 1.0 } else { 0.0 }
        }).sum::<f32>() / validation_outputs.len() as f32;

        accuracy
    }

    pub fn predict(&self, input: &DVector<f32>) -> DVector<f32> {
        let mut output = input.clone();
        for k in 0..self.layers {
            let z = self.weights[k].clone() * output + self.bias[k].column(0);
            output = z.map(|zi| f32::max(zi, 0.1 * zi))
        }
        output
    }
}

fn relu(x: &DMatrix<f32>) -> DMatrix<f32> {
    x.map(|z| f32::max(z, 0.1 * z))
}

fn grad_relu(x: &DVector<f32>) -> DMatrix<f32> {
    DMatrix::<f32>::from_diagonal(
        &x.map(|z| if z > 0.0 { 1.0 } else { 0.1 })
    )
}

fn plot_loss(loss: Vec<f32>) {
    let n = loss.len();
    let root_drawing_area = BitMapBackend::new("images/loss_curve.png", (1024, 768))
        .into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Loss Curve", ("sans-serif", 40))
        .build_cartesian_2d(0.0..(n as f64), 0.0..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        (0..n).map(|x| (x as f64, loss[x] as f64)),
        &BLUE
    )).unwrap();

    root_drawing_area.present().unwrap();
}

fn plot_accuracy(accuracy: Vec<f32>) {
    let n = accuracy.len();
    let root_drawing_area = BitMapBackend::new("images/accuracy_curve.png", (1024, 768))
        .into_drawing_area();
    root_drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Accuracy Curve", ("sans-serif", 40))
        .build_cartesian_2d(0.0..(n as f64), 0.8..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        (0..n).map(|x| (x as f64, accuracy[x] as f64)),
        &ORANGE
    )).unwrap();

    root_drawing_area.present().unwrap();
}

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

/*
    // Best results so far with leaky reLu, alpha=0.01, batch_size=50, epochs=500
    let mut network = NeuralNetwork::init_params(vec![100, 10], 1);
    let (loss, accuracies) = network.train(&train_inputs, &train_labels, &valid_inputs, &valid_labels, 1);
    let smooth_loss = loss_moving_average(loss);
    plot_loss(smooth_loss);
    plot_accuracy(accuracies);

    let accuracy = network.validation(&test_inputs, &test_labels);
    println!("Results : {}% success rate", accuracy * 100.0);

    let example_index = 3222;
    draw_digit(&train_inputs[example_index]);
    let prediction = network.predict(&train_inputs[example_index]);
    println!("Prediction : {}, Label : {}", prediction.argmax().0, train_labels[example_index].argmax().0);
*/


    let mut network = Network::new(
        vec![
            Box::new(Dense::new(784, 100)),
            Box::new(LeakyReLU::new(100)),
            Box::new(Dense::new(100, 10)),
            Box::new(SoftMax::new(10))
        ]
    );

    network.train(&train_inputs, &train_labels, &valid_inputs, &valid_labels, 1);
    let accuracy = network.test_accuracy(&test_inputs, &test_labels);
    println!("Results : {}% success rate", accuracy * 100.0);
}
