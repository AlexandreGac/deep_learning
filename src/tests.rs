#[cfg(test)]
mod test {
    use nalgebra::DMatrix;
    use crate::utils::valid_cross_correlation;
    use crate::utils::full_convolution;

    #[test]
    fn test_correlation() {
        let test_matrix = DMatrix::<f32>::from_row_slice(3, 5, &[
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ]);

        let test_kernel = DMatrix::<f32>::from_row_slice(2, 3, &[
            -3.0, -2.0, -1.0,
            1.0, 2.0, 3.0
        ]);

        let test_result = valid_cross_correlation(&test_matrix, &test_kernel, 3 - 2 + 1, 5 - 3 + 1);
        println!("{test_result}");
    }

    #[test]
    fn test_convolution() {
        let test_matrix = DMatrix::<f32>::from_row_slice(3, 5, &[
            1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0
        ]);

        let test_kernel = DMatrix::<f32>::from_row_slice(2, 3, &[
            -3.0, -2.0, -1.0,
            1.0, 2.0, 3.0
        ]);

        let test_result = full_convolution(&test_matrix, &test_kernel, 3 + 2 - 1, 5 + 3 - 1);
        println!("{test_result}");
    }
}