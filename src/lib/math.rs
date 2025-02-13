use rand::Rng;
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Option<Vec<f64>>) -> Matrix {
        let data = data.unwrap_or_else(|| vec![0.0; rows * cols]);
        Matrix { rows, cols, data }
    }

    pub fn fill_random(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.data.len() {
            self.data[i] = rng.gen();
        }
    }
    pub fn transpose(&mut self) -> Matrix {
        let mut result_data = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix::new(self.cols, self.rows, Some(result_data))
    }


    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data[index] = value;
        } else {
            panic!("Index out of bounds");
        }
    }


    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows == other.rows && self.cols == other.cols {
            // ...existing elementwise addition code...
            let mut result_data = vec![0.0; self.rows * self.cols];
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let index = i * self.cols + j;
                    result_data[index] = self.data[index] + other.data[index];
                }
            }
            Matrix::new(self.rows, self.cols, Some(result_data))
        } else if other.rows == 1 && other.cols == self.cols {
            // Broadcast addition: add the 1 row vector to each row of self
            let mut result_data = Vec::with_capacity(self.rows * self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    let index = i * self.cols + j;
                    result_data.push(self.data[index] + other.data[j]);
                }
            }
            Matrix::new(self.rows, self.cols, Some(result_data))
        } else {
            panic!("Matrix dimensions must match for addition: self: ({}, {}), other: ({}, {})", self.rows, self.cols, other.rows, other.cols);
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix{
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match for subtraction: self: ({}, {}), other: ({}, {})", self.rows, self.cols, other.rows, other.cols);
        }

        let mut result_data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols{
                let index = i * self.cols + j;
                result_data[index] = self.data[index] - other.data[index];
            }
        }
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions must match for dot product: self: ({}, {}), other: ({}, {})", self.rows, self.cols, other.rows, other.cols);
        }
    
        let mut result_data = vec![0.0; self.rows * other.cols];
        
        for i in 0..self.rows {
            for k in 0..self.cols {
                let scalar = self.data[i * self.cols + k];
                for j in 0..other.cols {
                    result_data[i * other.cols + j] += scalar * other.data[k * other.cols + j];
                }
            }
        }
    
        Matrix::new(self.rows, other.cols, Some(result_data))
    }

    pub fn dot_1d(&self, other: &Matrix) -> f64 {
        if self.rows != 1 || other.rows != 1 || self.cols != other.cols {
            panic!("Both matrices must be 1-dimensional and have the same length");
        }

        let mut result = 0.0;
        for i in 0..self.cols {
            result += self.data[i] * other.data[i];
        }

        result
    }
    pub fn sigmoid(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn relu(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn tanh(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| x.tanh()).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    pub fn sigmoid_derivative(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            sigmoid * (1.0 - sigmoid)
        }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn relu_derivative(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn tanh_derivative(&self) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| {
            let tanh = x.tanh();
            1.0 - tanh * tanh
        }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    pub fn mse(&self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Matrix dimensions must match for mse: self: ({}, {}), target: ({}, {})", self.rows, self.cols, target.rows, target.cols);
        }
    
        let mut sum = 0.0;
        for i in 0..self.data.len() {
            let diff = self.data[i] - target.data[i];
            sum += diff * diff;
        }
    
        let mse_value = sum / self.data.len() as f64;
        Matrix::new(1, 1, Some(vec![mse_value]))
    }
    pub fn mul(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match");
        }

        let result_data: Vec<f64> = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| a * b).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    pub fn mse_derivative(&self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Matrix dimensions must match for mse_derivative: self: ({}, {}), target: ({}, {})", self.rows, self.cols, target.rows, target.cols);
        }

        let data = self.data.iter().zip(&target.data).map(|(p, a)| 2.0 * (p - a) / self.data.len() as f64).collect();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    pub fn sum_axis_0(&self) -> Matrix {
        let mut result_data = vec![0.0; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[j] += self.data[i * self.cols + j];
            }
        }
        Matrix::new(1, self.cols, Some(result_data))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Matrix {
        let result_data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
}

// Add these functions
pub fn sigmoid(matrix: &Matrix) -> Matrix {
    matrix.sigmoid()
}

pub fn mse(output: &Matrix, target: &Matrix) -> Matrix {
    output.mse(target)
}
pub fn sigmoid_derivative(matrix: &Matrix) -> Matrix{
    matrix.sigmoid_derivative()
}
pub fn mse_derivative(output: &Matrix, target: &Matrix) -> Matrix{
    output.mse_derivative(target)
}
pub fn relu(matrix: &Matrix) -> Matrix{
    matrix.relu()
}
pub fn relu_derivative(matrix: &Matrix) -> Matrix{
    matrix.relu_derivative()
}

