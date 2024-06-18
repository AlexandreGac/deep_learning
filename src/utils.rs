use nalgebra::{DMatrix, DVector};
use rand_distr::Normal;

const BETA_1: f32 = 0.9;
const BETA_2: f32 = 0.999;
const EPSILON: f32 = 10e-8;

pub fn he_init_dense(input_size: usize, output_size: usize, ) -> DMatrix<f32> {
    let rng = &mut rand::thread_rng();
    DMatrix::<f32>::from_distribution(
        output_size, input_size,
        &Normal::new(0.0, f32::sqrt(2.0 / input_size as f32)).unwrap(),
        rng
    )
}

pub fn he_init_conv2d(kernel_height: usize, kernel_width: usize, input_depth: usize) -> DMatrix<f32> {
    let rng = &mut rand::thread_rng();
    let n = input_depth * kernel_height * kernel_width;
    DMatrix::<f32>::from_distribution(
        kernel_height, kernel_width,
        &Normal::new(0.0, f32::sqrt(2.0 / n as f32)).unwrap(),
        rng
    )
}

pub fn adam_optimizer_matrix(parameters: &mut DMatrix<f32>, gradient: DMatrix<f32>, first_moment: &mut DMatrix<f32>, second_moment: &mut DMatrix<f32>, learning_rate: f32, iteration: i32) {
    let gradient_squared = gradient.component_mul(&gradient);
    *first_moment *= BETA_1;
    *first_moment += (1.0 - BETA_1) * gradient;
    *second_moment *= BETA_2;
    *second_moment += (1.0 - BETA_2) * gradient_squared;
    let corrected_first_moment = &*first_moment / (1.0 - BETA_1.powi(iteration));
    let corrected_second_moment = &*second_moment / (1.0 - BETA_2.powi(iteration));
    let denominator = corrected_second_moment.map(|z| z.sqrt() + EPSILON);
    *parameters -= learning_rate * corrected_first_moment.component_div(&denominator);
}

pub fn adam_optimizer_vector(parameters: &mut DVector<f32>, gradient: DVector<f32>, first_moment: &mut DVector<f32>, second_moment: &mut DVector<f32>, learning_rate: f32, iteration: i32) {
    let gradient_squared = gradient.component_mul(&gradient);
    *first_moment *= BETA_1;
    *first_moment += (1.0 - BETA_1) * gradient;
    *second_moment *= BETA_2;
    *second_moment += (1.0 - BETA_2) * gradient_squared;
    let corrected_first_moment = &*first_moment / (1.0 - BETA_1.powi(iteration));
    let corrected_second_moment = &*second_moment / (1.0 - BETA_2.powi(iteration));
    let denominator = corrected_second_moment.map(|z| z.sqrt() + EPSILON);
    *parameters -= learning_rate * corrected_first_moment.component_div(&denominator);
}

pub fn valid_cross_correlation(input: &DMatrix<f32>, kernel: &DMatrix<f32>, n_rows: usize, n_cols: usize) -> DMatrix<f32> {
    DMatrix::<f32>::from_fn(n_rows, n_cols, |i, j| {
        input.view((i, j), kernel.shape()).component_mul(&kernel).sum()
    })
}

pub fn full_convolution(input: &DMatrix<f32>, kernel: &DMatrix<f32>, n_rows: usize, n_cols: usize) -> DMatrix<f32> {
    let rotated_kernel = DMatrix::<f32>::from_column_slice(kernel.nrows(), kernel.ncols(), kernel.iter().cloned().rev().collect::<Vec<_>>().as_slice());
    let padded_input = input.clone()
        .insert_rows(0, kernel.nrows() - 1, 0.0)
        .insert_columns(0, kernel.ncols() - 1, 0.0)
        .resize(input.nrows() + 2 * kernel.nrows() - 2, input.ncols() + 2 * kernel.ncols() - 2, 0.0);

    valid_cross_correlation(&padded_input, &rotated_kernel, n_rows, n_cols)
}

pub fn index_to_coords(shape: (usize, usize, usize), index: usize) -> (usize, usize, usize) {
    let length = shape.1 * shape.2;
    let depth = index / length;
    let j = (index - depth * length) / shape.1;
    let i = index - depth * length - j * shape.1;

    (depth, i, j)
}

pub fn coords_to_index(shape: (usize, usize, usize), depth: usize, i: usize, j: usize) -> usize {
    let index = depth * shape.1 * shape.2 + j * shape.1 + i;
    index
}