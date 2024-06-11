use nalgebra::{DMatrix, DVector};
use rand_distr::Normal;
use rayon::prelude::*;

const BETA_1: f32 = 0.9;
const BETA_2: f32 = 0.999;
const EPSILON: f32 = 10e-8;

pub trait Layer {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>>;
    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, learning_rate: f32, iteration: i32) -> Vec<DVector<f32>>;
}

pub struct Dense {
    weights: DMatrix<f32>,
    bias: DVector<f32>,

    last_input_batch: Vec<DVector<f32>>,
    weight_first_moment: DMatrix<f32>,
    weight_second_moment: DMatrix<f32>,
    bias_first_moment: DVector<f32>,
    bias_second_moment: DVector<f32>
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Dense {
            weights: Self::he_init(input_size, output_size),
            bias: DVector::<f32>::zeros(output_size),

            last_input_batch: vec![],
            weight_first_moment: DMatrix::<f32>::zeros(output_size, input_size),
            weight_second_moment: DMatrix::<f32>::zeros(output_size, input_size),
            bias_first_moment: DVector::<f32>::zeros(output_size),
            bias_second_moment: DVector::<f32>::zeros(output_size),
        }
    }

    fn he_init(input_size: usize, output_size: usize) -> DMatrix<f32> {
        let rng = &mut rand::thread_rng();
        DMatrix::<f32>::from_distribution(
            output_size, input_size,
            &Normal::new(0.0, f32::sqrt(2.0 / input_size as f32)).unwrap(),
            rng
        )
    }

    fn adam_optimizer(&mut self, weights_gradient: DMatrix<f32>, bias_gradient: DVector<f32>, learning_rate: f32, iteration: i32) {
        let weights_gradient_squared = weights_gradient.component_mul(&weights_gradient);
        self.weight_first_moment = BETA_1 * self.weight_first_moment.clone() + (1.0 - BETA_1) * weights_gradient;
        self.weight_second_moment = BETA_2 * self.weight_second_moment.clone() + (1.0 - BETA_2) * weights_gradient_squared;
        let weight_corrected_first_moment = self.weight_first_moment.clone() / (1.0 - BETA_1.powi(iteration));
        let weight_corrected_second_moment = self.weight_second_moment.clone() / (1.0 - BETA_2.powi(iteration));
        let weight_denominator = weight_corrected_second_moment.map(|w| w.sqrt() + EPSILON);
        self.weights -= learning_rate * weight_corrected_first_moment.component_div(&weight_denominator);

        let bias_gradient_squared = bias_gradient.component_mul(&bias_gradient);
        self.bias_first_moment = BETA_1 * self.bias_first_moment.clone() + (1.0 - BETA_1) * bias_gradient;
        self.bias_second_moment = BETA_2 * self.bias_second_moment.clone() + (1.0 - BETA_2) * bias_gradient_squared;
        let bias_corrected_first_moment = self.bias_first_moment.clone() / (1.0 - BETA_1.powi(iteration));
        let bias_corrected_second_moment = self.bias_second_moment.clone() / (1.0 - BETA_2.powi(iteration));
        let bias_denominator = bias_corrected_second_moment.map(|b| b.sqrt() + EPSILON);
        self.bias -= learning_rate * bias_corrected_first_moment.component_div(&bias_denominator);
    }
}

impl Layer for Dense {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
        let result_batch = input_batch.par_iter().map(|input| {
            let mut result = self.bias.clone();
            self.weights.mul_to(input, &mut result);
            result += self.bias.clone();
            result
        }).collect();
        self.last_input_batch = input_batch;
        result_batch
    }

    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, learning_rate: f32, iteration: i32) -> Vec<DVector<f32>> {
        let m = factor_batch.len() as f32;
        let back_factor_batch = factor_batch.par_iter().map(|factor| {
            self.weights.transpose() * factor.clone()
        }).collect();
        let weights_gradient = factor_batch.par_iter().enumerate().map(|(i, factor)| {
            factor.clone() * self.last_input_batch[i].transpose()
        }).collect::<Vec<_>>().iter().sum::<DMatrix<f32>>() / m;
        let bias_gradient = factor_batch.iter().sum::<DVector<f32>>() / m;

        self.adam_optimizer(weights_gradient, bias_gradient, learning_rate, iteration);
        back_factor_batch
    }
}

pub struct LeakyReLU {
    last_input_batch: Vec<DVector<f32>>
}

impl LeakyReLU {
    pub fn new() -> Self {
        LeakyReLU {
            last_input_batch: vec![]
        }
    }
}

impl Layer for LeakyReLU {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
        let result_batch = input_batch.par_iter()
            .map(|input| input.map(|z| f32::max(z, 0.02 * z)))
            .collect();
        self.last_input_batch = input_batch;
        result_batch
    }

    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, _learning_rate: f32, _iteration: i32) -> Vec<DVector<f32>> {
        let grad_relu_batch = self.last_input_batch.par_iter().map(|last_input| {
            last_input.map(|z| if z > 0.0 { 1.0 } else { 0.02 })
        }).collect::<Vec<DVector<f32>>>();
        let back_factor_batch = grad_relu_batch.par_iter().enumerate().map(|(i, grad_relu)| {
            grad_relu.component_mul(&factor_batch[i])
        }).collect();
        back_factor_batch
    }
}

pub struct SoftMax {
    last_output_batch: Vec<DVector<f32>>
}

impl SoftMax {
    pub fn new() -> Self {
        SoftMax {
            last_output_batch: vec![]
        }
    }
}

impl Layer for SoftMax {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
        self.last_output_batch = input_batch.par_iter().map(|input| {
            let sum_exp = input.map(|z| z.exp()).sum();
            input.map(|z| z.exp() / sum_exp)
        }).collect();
        self.last_output_batch.clone()
    }

    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, _learning_rate: f32, _iteration: i32) -> Vec<DVector<f32>> {
        let grad_softmax_batch = self.last_output_batch.par_iter().map(|last_output| {
            let n = last_output.nrows();
            DMatrix::<f32>::from_fn(n, n, |i, j| {
                if i == j {
                    last_output[i] * (1.0 - last_output[j])
                }
                else {
                    -last_output[i] * last_output[j]
                }
            })
        }).collect::<Vec<_>>();
        let back_factor_batch = grad_softmax_batch.into_par_iter().enumerate().map(|(i, grad_softmax)| {
            grad_softmax * factor_batch[i].clone()
        }).collect();
        back_factor_batch
    }
}