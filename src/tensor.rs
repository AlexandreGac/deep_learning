use itertools::Itertools;
use std::ops::{Index, IndexMut, Add, Sub, Mul, SubAssign, Neg};
use rand::distributions::Distribution;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Tensor {
    dims: Vec<usize>,
    data: Vec<f32>
}

impl Tensor {

    pub fn zeros(dims: Vec<usize>) -> Tensor {
        let n = dims.iter().product::<usize>();
        Tensor {
            dims, data: vec![0.0; n]
        }
    }

    pub fn from_fn<F: FnMut(Vec<usize>) -> f32>(dims: Vec<usize>, mut f: F) -> Tensor {
        let mut tensor = Self::zeros(dims);
        let indices_list = tensor.dims.iter()
            .map(|&k| 0..k)
            .multi_cartesian_product();

        for indices in indices_list {
            let x = indices.clone();
            tensor[indices] = f(x);
        }

        tensor
    }

    pub fn from_distribution<D: Distribution<f32>, R: Rng>(dims: Vec<usize>, distribution: &D, rng: &mut R) -> Tensor {
        let mut tensor = Self::zeros(dims);
        let n = tensor.data.len();
        for i in 0..n {
            tensor.data[i] = distribution.sample(rng);
        }
        tensor
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let dims = self.dims.clone();
        let mut tensor = Self::zeros(dims);
        let n = tensor.data.len();
        for i in 0..n {
            tensor.data[i] = f(self.data[i]);
        }
        tensor
    }

    pub fn shape(&self) -> Vec<usize> {
        self.dims.clone()
    }

    pub fn get(&self, indices: Vec<usize>) -> f32 {
        let i = self.index(indices);
        self.data[i]
    }

    pub fn set(&mut self, indices: Vec<usize>, value: f32) {
        let i = self.index(indices);
        self.data[i] = value;
    }

    fn index(&self, indices: Vec<usize>) -> usize {
        assert_eq!(self.dims.len(), indices.len());

        let index = indices.into_iter().enumerate().map(|(n, i)| {
            i * self.dims[0..n].iter().product::<usize>()
        }).sum::<usize>();

        index
    }

    pub fn addition(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.dims, rhs.dims);

        let n = self.data.len();
        let dims = self.dims.clone();
        let mut result = Self::zeros(dims);
        for i in 0..n {
            result.data[i] = self.data[i] + rhs.data[i];
        }

        result
    }

    pub fn subtract(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.dims, rhs.dims);

        let n = self.data.len();
        let dims = self.dims.clone();
        let mut result = Self::zeros(dims);
        for i in 0..n {
            result.data[i] = self.data[i] - rhs.data[i];
        }

        result
    }

    pub fn contraction(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.dims.last(), rhs.dims.first());

        let n = *self.dims.last().unwrap();
        let length = self.dims.len() - 1;
        let dims = [self.dims[..length].to_vec(), rhs.dims[1..].to_vec()]
            .concat();

        let mut result = Self::zeros(dims);
        let indices_list = result.dims.iter()
            .map(|&k| 0..k)
            .multi_cartesian_product();

        for indices in indices_list {
            let (left_indices, right_indices) = indices.split_at(length);
            result[indices] = (0..n).into_iter().map(|k| {
                let li = [left_indices, &[k]].concat();
                let ri = [&[k], right_indices].concat();
                self[li] * rhs[ri]
            }).sum::<f32>();
        }

        result
    }

    pub fn tensor_product(&self, rhs: &Tensor) -> Tensor {
        let length = self.dims.len();
        let dims = [self.dims.clone(), rhs.dims.clone()].concat();
        let mut result = Self::zeros(dims);
        let indices_list = result.dims.iter()
            .map(|&k| 0..k)
            .multi_cartesian_product();

        for indices in indices_list {
            let (li, ri) = indices.split_at(length);
            result[indices] = self[li.to_vec()] * rhs[ri.to_vec()];
        }

        result
    }

    pub fn component_mul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(self.dims, rhs.dims);

        let n = self.data.len();
        let dims = self.dims.clone();
        let mut result = Self::zeros(dims);
        for i in 0..n {
            result.data[i] = self.data[i] * rhs.data[i];
        }

        result
    }

    pub fn sum(&self) -> f32 {
        let n = self.data.len();
        let mut result = 0.0;
        for i in 0..n {
            result += self.data[i];
        }
        result
    }

    pub fn argmax(&self) -> (f32, Vec<usize>) {
        let mut max = f32::NEG_INFINITY;
        let mut max_indices = vec![0; self.dims.len()];
        let indices_list = self.dims.iter()
            .map(|&k| 0..k)
            .multi_cartesian_product();

        for indices in indices_list {
            let value = self[indices.clone()];
            if value > max {
                max = value;
                max_indices = indices;
            }
        }

        (max, max_indices)
    }
}

impl Index<Vec<usize>> for Tensor {
    type Output = f32;

    fn index(&self, index: Vec<usize>) -> &Self::Output {
        let i = self.index(index);
        &self.data.index(i)
    }
}

impl IndexMut<Vec<usize>> for Tensor {
    fn index_mut(&mut self, index: Vec<usize>) -> &mut Self::Output {
        let i = self.index(index);
        self.data.index_mut(i)
    }
}

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.addition(&rhs)
    }
}

impl Sub for Tensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.addition(&rhs)
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.contraction(&rhs)
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let n = rhs.data.len();
        let dims = rhs.dims.clone();
        let mut result = Tensor::zeros(dims);
        for i in 0..n {
            result.data[i] = self * rhs.data[i]
        }
        result
    }
}

impl Neg for Tensor {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let n = self.data.len();
        let dims = self.dims.clone();
        let mut result = Self::zeros(dims);
        for i in 0..n {
            result.data[i] = -self.data[i];
        }

        result
    }
}

impl SubAssign for Tensor {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.dims, rhs.dims);
        let n = self.data.len();
        for i in 0..n {
            self.data[i] -= rhs.data[i];
        }
    }
}