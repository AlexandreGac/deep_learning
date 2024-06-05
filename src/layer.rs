use rand_distr::Normal;
use crate::tensor::Tensor;

pub trait Layer {
    fn forward_propagation(&mut self, input: Tensor) -> Tensor;
    fn backward_propagation(&mut self, factor: Tensor, learning_rate: f32) -> Tensor;
}

pub struct Dense {
    weights: Tensor,
    bias: Tensor,
    last_input: Tensor
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Dense {
            weights: Self::he_init(input_size, output_size),
            bias: Tensor::zeros(vec![output_size]),
            last_input:Tensor::zeros(vec![input_size])
        }
    }

    fn he_init(input_size: usize, output_size: usize) -> Tensor {
        let rng = &mut rand::thread_rng();
        Tensor::from_distribution(
            vec![output_size, input_size],
            &Normal::new(0.0, f32::sqrt(2.0 / input_size as f32)).unwrap(),
            rng
        )
    }
}

impl Layer for Dense {
    fn forward_propagation(&mut self, input: Tensor) -> Tensor {
        let result = self.weights.clone() * input.clone() + self.bias.clone();
        self.last_input = input;
        result
    }

    fn backward_propagation(&mut self, factor: Tensor, learning_rate: f32) -> Tensor {
        let back_factor = factor.contraction(&self.weights);
        let weights_gradient = factor.tensor_product(&self.last_input);
        let bias_gradient = factor;
        self.weights -= learning_rate * weights_gradient;
        self.bias -= learning_rate * bias_gradient;
        back_factor
    }
}

pub struct LeakyReLU {
    last_input: Tensor
}

impl LeakyReLU {
    pub fn new(size: usize) -> Self {
        LeakyReLU {
            last_input: Tensor::zeros(vec![size])
        }
    }
}

impl Layer for LeakyReLU {
    fn forward_propagation(&mut self, input: Tensor) -> Tensor {
        let result = input.map(|z| f32::max(z, 0.1 * z));
        self.last_input = input;
        result
    }

    fn backward_propagation(&mut self, factor: Tensor, _learning_rate: f32) -> Tensor {
        let grad_relu = self.last_input.map(|z| if z > 0.0 { 1.0 } else { 0.1 });
        let back_factor = grad_relu.component_mul(&factor);
        back_factor
    }
}

pub struct SoftMax {
    last_output: Tensor
}

impl SoftMax {
    pub fn new(size: usize) -> Self {
        SoftMax {
            last_output: Tensor::zeros(vec![size])
        }
    }
}

impl Layer for SoftMax {
    fn forward_propagation(&mut self, input: Tensor) -> Tensor {
        let sum_exp = input.map(|z| z.exp()).sum();
        self.last_output = input.map(|z| z.exp() / sum_exp);
        self.last_output.clone()
    }

    fn backward_propagation(&mut self, factor: Tensor, _learning_rate: f32) -> Tensor {
        let n = self.last_output.shape()[0];
        let grad_softmax = Tensor::from_fn(vec![n, n], |indices| {
            let (i, j) = (indices[0], indices[1]);
            if i == j {
                self.last_output[vec![i]] * (1.0 - self.last_output[vec![j]])
            }
            else {
                - self.last_output[vec![i]] * self.last_output[vec![j]]
            }
        });
        let back_factor = grad_softmax * factor;
        back_factor
    }
}