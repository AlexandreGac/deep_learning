use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use crate::utils::*;

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
            weights: he_init_dense(input_size, output_size),
            bias: DVector::<f32>::zeros(output_size),

            last_input_batch: vec![],
            weight_first_moment: DMatrix::<f32>::zeros(output_size, input_size),
            weight_second_moment: DMatrix::<f32>::zeros(output_size, input_size),
            bias_first_moment: DVector::<f32>::zeros(output_size),
            bias_second_moment: DVector::<f32>::zeros(output_size),
        }
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

        adam_optimizer_matrix(
            &mut self.weights, weights_gradient, &mut self.weight_first_moment,
            &mut self.weight_second_moment, learning_rate, iteration
        );
        adam_optimizer_vector(
            &mut self.bias, bias_gradient, &mut self.bias_first_moment,
            &mut self.bias_second_moment, learning_rate, iteration
        );
        back_factor_batch
    }
}

pub struct Conv2D {
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    kernels: Vec<Vec<DMatrix<f32>>>,
    bias: Vec<DMatrix<f32>>,

    last_input_batch: Vec<Vec<DMatrix<f32>>>,
    kernels_first_moment: Vec<Vec<DMatrix<f32>>>,
    kernels_second_moment: Vec<Vec<DMatrix<f32>>>,
    bias_first_moment: Vec<DMatrix<f32>>,
    bias_second_moment: Vec<DMatrix<f32>>
}

impl Conv2D {
    pub fn new(input_shape: (usize, usize, usize), kernels_shape: (usize, usize), output_features: usize) -> Self {
        let mut kernels = vec![];
        for _ in 0..output_features {
            let mut kernels_weights = vec![];
            for _ in 0..input_shape.0 {
                kernels_weights.push(he_init_conv2d(kernels_shape.0, kernels_shape.1, input_shape.0));
            }
            kernels.push(kernels_weights);
        }

        let (n_rows, n_cols) = (input_shape.1 - kernels_shape.0 + 1, input_shape.2 - kernels_shape.1 + 1);
        let bias = vec![DMatrix::<f32>::zeros(n_rows, n_cols); output_features];

        Self {
            input_shape,
            output_shape: (output_features, n_rows, n_cols),
            kernels,
            bias,

            last_input_batch: vec![],
            kernels_first_moment: vec![vec![DMatrix::<f32>::zeros(kernels_shape.0, kernels_shape.1); input_shape.0]; output_features],
            kernels_second_moment: vec![vec![DMatrix::<f32>::zeros(kernels_shape.0, kernels_shape.1); input_shape.0]; output_features],
            bias_first_moment: vec![DMatrix::<f32>::zeros(n_rows, n_cols); output_features],
            bias_second_moment: vec![DMatrix::<f32>::zeros(n_rows, n_cols); output_features]
        }
    }
}

impl Layer for Conv2D {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
        self.last_input_batch = input_batch.into_par_iter().map(|input| {
            let length = self.input_shape.1 * self.input_shape.2;
            (0..self.input_shape.0).map(|depth|
                DMatrix::<f32>::from_column_slice(
                    self.input_shape.1, self.input_shape.2,
                    &input.as_slice()[(depth * length)..(depth * length + length)]
                )
            ).collect::<Vec<_>>()
        }).collect();

        let result_batch = self.last_input_batch.par_iter().map(|features| {
            let (output_depth, n_rows, n_cols) = self.output_shape;
            let mut result_data = vec![];
            for depth in 0..output_depth {
                let cross_correlation = features.iter().enumerate().map(|(k, feature)| {
                    valid_cross_correlation(feature, &self.kernels[depth][k], n_rows, n_cols)
                }).sum::<DMatrix<f32>>() + &self.bias[depth];
                result_data.extend_from_slice(cross_correlation.as_slice());
            }
            DVector::<f32>::from(result_data)
        }).collect();

        result_batch
    }

    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, learning_rate: f32, iteration: i32) -> Vec<DVector<f32>> {
        let m = factor_batch.len() as f32;
        let (output_depth, output_n_rows, output_n_cols) = self.output_shape;
        let length = output_n_rows * output_n_cols;

        let back_factor_batch = factor_batch.par_iter().map(|factor| {
            let mut back_factor_data = vec![];
            for j in 0..self.input_shape.0 {
                let convolution = (0..self.kernels.len()).map(|i| {
                    let factor_i = DMatrix::<f32>::from_column_slice(
                        output_n_rows, output_n_cols,
                        &factor.as_slice()[(i * length)..(i * length + length)]
                    );
                    full_convolution(&factor_i, &self.kernels[i][j], self.input_shape.1, self.input_shape.2)
                }).sum::<DMatrix<f32>>();
                back_factor_data.extend_from_slice(convolution.as_slice());
            }
            DVector::<f32>::from(back_factor_data)
        }).collect();

        for i in 0..output_depth {
            for j in 0..self.input_shape.0 {
                let kernel_gradient = factor_batch.par_iter().enumerate().map(|(k, factor)| {
                    let factor_i = DMatrix::<f32>::from_column_slice(
                        output_n_rows, output_n_cols,
                        &factor.as_slice()[(i * length)..(i * length + length)]
                    );
                    valid_cross_correlation(&self.last_input_batch[k][j], &factor_i, self.kernels[i][j].shape().0, self.kernels[i][j].shape().1)
                }).collect::<Vec<_>>().iter().sum::<DMatrix<f32>>() / m;
                adam_optimizer_matrix(
                    &mut self.kernels[i][j], kernel_gradient, &mut self.kernels_first_moment[i][j],
                    &mut self.kernels_second_moment[i][j], learning_rate, iteration
                );
            }
        }

        for i in 0..output_depth {
            let bias_gradient = factor_batch.iter().map(|factor| {
                DMatrix::<f32>::from_column_slice(
                    output_n_rows, output_n_cols,
                    &factor.as_slice()[(i * length)..(i * length + length)]
                )
            }).sum::<DMatrix<f32>>() / m;
            adam_optimizer_matrix(
                &mut self.bias[i], bias_gradient, &mut self.bias_first_moment[i],
                &mut self.bias_second_moment[i], learning_rate, iteration
            );
        }

        back_factor_batch
    }
}

pub struct MaxPooling2D {
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    pooling_window: (usize, usize),

    pool_indices: Vec<Vec<(usize, usize)>>
}

impl MaxPooling2D {
    pub fn new(input_shape: (usize, usize, usize), pooling_window: (usize, usize)) -> Self {
        let output_n_rows = input_shape.1.div_ceil(pooling_window.0);
        let output_n_cols = input_shape.2.div_ceil(pooling_window.1);
        Self {
            input_shape,
            output_shape: (input_shape.0, output_n_rows, output_n_cols),
            pooling_window,
            pool_indices: vec![],
        }
    }
}

impl Layer for MaxPooling2D {
    fn forward_propagation(&mut self, input_batch: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
        let output_length = self.output_shape.0 * self.output_shape.1 * self.output_shape.2;
        let (result_batch, pool_indices): (Vec<_>, Vec<_>) = input_batch.par_iter().map(|input| {
            let (result_data, result_indices): (Vec<f32>, Vec<(usize, usize)>) = (0..output_length).map(|n| {
                let (depth, i, j) = index_to_coords(self.output_shape, n);
                let window_size = (
                    usize::min(self.pooling_window.0, self.input_shape.1 - i * self.pooling_window.0),
                    usize::min(self.pooling_window.1, self.input_shape.2 - j * self.pooling_window.1)
                );
                let mut max = f32::NEG_INFINITY;
                let mut pool_indices = (0, 0);
                for k in 0..window_size.0 {
                    let i_input = i * self.pooling_window.0 + k;
                    for l in 0..window_size.1 {
                        let j_input = j * self.pooling_window.1 + l;
                        let index = coords_to_index(self.input_shape, depth, i_input, j_input);
                        let value = *input.index(index);
                        if value > max {
                            max = value;
                            pool_indices = (k, l);
                        }
                    }
                }
                (max, pool_indices)
            }).unzip();

            (DVector::<f32>::from(result_data), result_indices)
        }).unzip();

        self.pool_indices = pool_indices;
        result_batch
    }

    fn backward_propagation(&mut self, factor_batch: Vec<DVector<f32>>, _learning_rate: f32, _iteration: i32) -> Vec<DVector<f32>> {
        let input_length = self.input_shape.0 * self.input_shape.1 * self.input_shape.2;
        let back_factor_batch = factor_batch.par_iter().enumerate().map(|(k, factor)| {
            DVector::<f32>::from_fn(input_length, |n, _| {
                let (depth, i, j) = index_to_coords(self.input_shape, n);
                let i_output = i / self.pooling_window.0;
                let j_output = j / self.pooling_window.1;
                let index = coords_to_index(self.output_shape, depth, i_output, j_output);
                if (i - i_output * self.pooling_window.0, j - j_output * self.pooling_window.1) == self.pool_indices[k][index] {
                    factor[index]
                }
                else {
                    0.0
                }
            })
        }).collect();
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