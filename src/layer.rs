use nalgebra::{DMatrix, DVector};
use rand_distr::Normal;

pub trait Layer {
    fn forward_propagation(&mut self, input: DVector<f32>) -> DVector<f32>;
    fn backward_propagation(&mut self, factor: DVector<f32>, learning_rate: f32) -> DVector<f32>;
}

pub struct Dense {
    weights: DMatrix<f32>,
    bias: DVector<f32>,
    last_input: DVector<f32>
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Dense {
            weights: Self::he_init(input_size, output_size),
            bias: DVector::<f32>::zeros(output_size),
            last_input: DVector::<f32>::zeros(input_size)
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
}

impl Layer for Dense {
    fn forward_propagation(&mut self, input: DVector<f32>) -> DVector<f32> {
        let mut result = self.bias.clone();
        self.weights.mul_to(&input, &mut result);
        result += self.bias.clone();
        self.last_input = input;
        result
    }

    fn backward_propagation(&mut self, factor: DVector<f32>, learning_rate: f32) -> DVector<f32> {
        let back_factor = self.weights.transpose() * factor.clone();
        let weights_gradient = factor.clone() * self.last_input.transpose();
        let bias_gradient = factor;
        self.weights -= learning_rate * weights_gradient;
        self.bias -= learning_rate * bias_gradient;
        back_factor
    }
}

pub struct LeakyReLU {
    last_input: DVector<f32>
}

impl LeakyReLU {
    pub fn new(size: usize) -> Self {
        LeakyReLU {
            last_input: DVector::<f32>::zeros(size)
        }
    }
}

impl Layer for LeakyReLU {
    fn forward_propagation(&mut self, input: DVector<f32>) -> DVector<f32> {
        let result = input.map(|z| f32::max(z, 0.1 * z));
        self.last_input = input;
        result
    }

    fn backward_propagation(&mut self, factor: DVector<f32>, _learning_rate: f32) -> DVector<f32> {
        let grad_relu = self.last_input.map(|z| if z > 0.0 { 1.0 } else { 0.1 });
        let back_factor = grad_relu.component_mul(&factor);
        back_factor
    }
}

pub struct SoftMax {
    last_output: DVector<f32>
}

impl SoftMax {
    pub fn new(size: usize) -> Self {
        SoftMax {
            last_output: DVector::<f32>::zeros(size)
        }
    }
}

impl Layer for SoftMax {
    fn forward_propagation(&mut self, input: DVector<f32>) -> DVector<f32> {
        let sum_exp = input.map(|z| z.exp()).sum();
        self.last_output = input.map(|z| z.exp() / sum_exp);
        self.last_output.clone()
    }

    fn backward_propagation(&mut self, factor: DVector<f32>, _learning_rate: f32) -> DVector<f32> {
        let n = self.last_output.nrows();
        let grad_softmax = DMatrix::<f32>::from_fn(n, n, |i, j| {
            if i == j {
                self.last_output[i] * (1.0 - self.last_output[j])
            }
            else {
                - self.last_output[i] * self.last_output[j]
            }
        });
        let back_factor = grad_softmax * factor;
        back_factor
    }
}