use std::fmt::Write;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::{DVector};
use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BLUE, IntoDrawingArea, LineSeries, WHITE};
use plotters::prelude::full_palette::ORANGE;
use rayon::prelude::*;
use crate::layer::Layer;

const LEARNING_RATE: f32 = 0.001;

pub struct Network {
    layers: Vec<Box<dyn Layer + Send + Sync>>
}

impl Network {

    pub fn new(layers: Vec<Box<dyn Layer + Send + Sync>>) -> Self {
        Network {
            layers
        }
    }

    fn forward_propagation(&mut self, input_batch: &[DVector<f32>]) -> Vec<DVector<f32>> {
        let mut output = input_batch.to_vec();
        for layer in self.layers.iter_mut() {
            output = layer.forward_propagation(output);
        }
        output
    }

    fn backward_propagation(&mut self, cost_derivative_batch: Vec<DVector<f32>>, iteration: i32) {
        let mut factor_batch = cost_derivative_batch;
        for layer in self.layers.iter_mut().rev() {
            factor_batch = layer.backward_propagation(factor_batch, LEARNING_RATE, iteration);
        }
    }

    fn cost(output_batch: &Vec<DVector<f32>>, target_batch: &[DVector<f32>]) -> Vec<f32> {
        output_batch.par_iter().enumerate().map(|(k, output)| {
            let n = target_batch[k].len();
            -DVector::<f32>::from_fn(n, |i, _| target_batch[k][i] * output[i].ln().max(f32::MIN)).sum()
        }).collect()
    }

    fn grad_cost(output_batch: &Vec<DVector<f32>>, target_batch: &[DVector<f32>]) -> Vec<DVector<f32>> {
        output_batch.par_iter().enumerate().map(|(k, output)| {
            let n = target_batch[k].len();
            -DVector::<f32>::from_fn(n, |i, _| if output[i] == 0.0 { 0.0 } else { target_batch[k][i] / output[i] })
        }).collect()
    }

    fn validate(&mut self, validation_inputs: &[DVector<f32>], validation_outputs: &[DVector<f32>]) -> f32 {
        let prediction_batch = self.predict(validation_inputs);
        let accuracy = prediction_batch.par_iter().enumerate().map(|(i, prediction)| {
            if prediction.argmax().0 == validation_outputs[i].argmax().0 { 1.0 } else { 0.0 }
        }).sum::<f32>() / validation_outputs.len() as f32;
        accuracy
    }

    pub fn train(&mut self, train_inputs: &Vec<DVector<f32>>, train_outputs: &Vec<DVector<f32>>, valid_inputs: &Vec<DVector<f32>>, valid_outputs: &Vec<DVector<f32>>, batch_size: usize, epochs: usize) {
        let n = train_inputs.len();
        let n_batches = n.div_ceil(batch_size);
        let train_input_batches = prepare_batches(train_inputs, batch_size);
        let train_output_batches = prepare_batches(train_outputs, batch_size);

        let mut loss_curve = vec![];
        let mut accuracy_curve = vec![];

        let pb = ProgressBar::new((epochs * n_batches) as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] Status : [{bar:50.green/blue}] {msg} ({eta})")
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap())
            .progress_chars("=> "));

        let mut iteration = 0;
        for epoch in 0..epochs {
            for i in 0..n_batches {
                iteration += 1;
                let output_batch = self.forward_propagation(train_input_batches[i]);
                self.backward_propagation(Self::grad_cost(&output_batch, train_output_batches[i]), iteration);
                let mean_cost = Network::cost(&output_batch, train_output_batches[i]).iter().sum::<f32>()
                    / output_batch.len() as f32;

                if mean_cost.is_nan() {
                    pb.abandon();
                    plot_loss(loss_curve);
                    plot_accuracy(accuracy_curve);
                    println!("Training failed ! NaN Loss detected... ({:.1}s)", pb.elapsed().as_secs_f32());
                    return
                }

                loss_curve.push(mean_cost);
                pb.inc(1);
                pb.set_message(format!(
                    "Progress : {:.1}%, {}/{} batches, epoch {}/{}",
                    100.0 * (epoch * n_batches + i + 1) as f32 / (epochs * n_batches) as f32,
                    i + 1, n_batches, epoch + 1, epochs
                ));
            }
            accuracy_curve.push(self.validate(valid_inputs, valid_outputs));
        }

        pb.finish();
        plot_loss(loss_curve);
        plot_accuracy(accuracy_curve);
        println!("Model finished training ! ({:.1}s)", pb.elapsed().as_secs_f32());
    }

    pub fn test_accuracy(&mut self, test_inputs: &Vec<DVector<f32>>, test_outputs: &Vec<DVector<f32>>) -> f32 {
        self.validate(test_inputs, test_outputs)
    }

    pub fn predict(&mut self, input_batch: &[DVector<f32>]) -> Vec<DVector<f32>> {
        self.forward_propagation(input_batch)
    }
}

fn prepare_batches(data: &Vec<DVector<f32>>, batch_size: usize) -> Vec<&[DVector<f32>]> {
    data.chunks(batch_size).collect()
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
        .build_cartesian_2d(0.0..(n as f64), 0.0..0.5)
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
        .build_cartesian_2d(0.0..(n as f64), 0.92..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        (0..n).map(|x| (x as f64, accuracy[x] as f64)),
        &ORANGE
    )).unwrap();

    root_drawing_area.present().unwrap();
}