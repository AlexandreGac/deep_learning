use std::fmt::Write;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::{DMatrix, DVector};
use plotters::backend::BitMapBackend;
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::prelude::{BLUE, IntoDrawingArea, LineSeries, WHITE};
use plotters::prelude::full_palette::ORANGE;
use rayon::prelude::*;
use crate::layer::Layer;

pub struct Network {
    layers: Vec<Box<dyn Layer + Send + Sync>>
}

impl Network {

    pub fn new(layers: Vec<Box<dyn Layer + Send + Sync>>) -> Self {
        Network {
            layers
        }
    }

    fn forward_propagation(&mut self, input: DVector<f32>) -> DVector<f32> {
        let mut output = input;
        for layer in self.layers.iter_mut() {
            output = layer.forward_propagation(output);
        }
        output
    }

    fn backward_propagation(&mut self, cost_derivative: DVector<f32>) {
        let learning_rate = 0.002;
        let mut factor = cost_derivative;
        for layer in self.layers.iter_mut().rev() {
            factor = layer.backward_propagation(factor, learning_rate);
        }
    }

    fn cost(output: &DVector<f32>, target: &DVector<f32>) -> f32 {
        let n = target.len();
        -DVector::<f32>::from_fn(n, |i, _| target[i] * output[i].ln()).sum()
    }

    fn grad_cost(output: &DVector<f32>, target: &DVector<f32>) -> DVector<f32> {
        let n = target.len();
        -DVector::<f32>::from_fn(n, |i, _| target[i] / output[i])
    }

    fn validate(&mut self, validation_inputs: &Vec<DVector<f32>>, validation_outputs: &Vec<DVector<f32>>) -> f32 {
        let accuracy = (0..validation_inputs.len()).into_iter().map(|i| {
            let prediction = self.predict(&validation_inputs[i]);
            if prediction.argmax().0 == validation_outputs[i].argmax().0 { 1.0 } else { 0.0 }
        }).sum::<f32>() / validation_outputs.len() as f32;

        accuracy
    }

    pub fn train(&mut self, train_inputs: &Vec<DVector<f32>>, train_outputs: &Vec<DVector<f32>>, valid_inputs: &Vec<DVector<f32>>, valid_outputs: &Vec<DVector<f32>>, epochs: usize) {
        let n = train_inputs.len();
        let n_batches = n;
        let mut loss_curve = vec![];
        let mut accuracy_curve = vec![];

        let pb = ProgressBar::new((epochs * n_batches) as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.blue} [{elapsed_precise}] Status : [{bar:50.green/blue}] {msg} ({eta})")
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap())
            .progress_chars("=> "));

        for epoch in 0..epochs {
            for i in 0..n {
                let output = self.forward_propagation(train_inputs[i].clone());
                self.backward_propagation(Self::grad_cost(&output, &train_outputs[i]));
                let cost = Network::cost(&output, &train_outputs[i]);
                loss_curve.push(cost);

                if cost.is_nan() {
                    pb.abandon();
                    plot_loss(loss_curve);
                    plot_accuracy(accuracy_curve);
                    println!("Training failed ! NaN Loss detected... ({:.1}s)", pb.elapsed().as_secs_f32());
                    return
                }

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

    pub fn predict(&mut self, input: &DVector<f32>) -> DVector<f32> {
        self.forward_propagation(input.clone())
    }
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
        .build_cartesian_2d(0.0..(n as f64), 0.9..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        (0..n).map(|x| (x as f64, accuracy[x] as f64)),
        &ORANGE
    )).unwrap();

    root_drawing_area.present().unwrap();
}