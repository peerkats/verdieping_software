use rand::Rng;
use rayon::prelude::*;
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Option<Vec<f32>>) -> Matrix {
        let data = data.unwrap_or_else(|| vec![0.0; rows * cols]);
        Matrix { rows, cols, data }
    }

    pub fn fill_random(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.data.len() {
            self.data[i] = rng.gen();
        }
    }

    // NEW: Fill with random values in the range [-0.5, 0.5]
     pub fn fill_random_centered(&mut self) {

        
        self.data.par_iter_mut().for_each(|v| {
            let mut rng = rand::thread_rng(); // Each thread gets its own RNG
            *v = rng.gen_range(-1.5..1.2);
        });
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
    pub fn get(&self, row: usize, col: usize) -> f32 {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data[index]
        } else {
            panic!("Index out of bounds: ({}, {}) for matrix of size ({}, {})", 
                   row, col, self.rows, self.cols);
        }
    }


    pub fn set(&mut self, row: usize, col: usize, value: f32) {
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

    pub fn dot_1d(&self, other: &Matrix) -> f32 {
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
        let result_data: Vec<f32> = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn relu(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn tanh(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x.tanh()).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    pub fn sigmoid_derivative(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            sigmoid * (1.0 - sigmoid)
        }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn relu_derivative(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    pub fn tanh_derivative(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| {
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
    
        let mse_value = sum / self.data.len() as f32;
        Matrix::new(1, 1, Some(vec![mse_value]))
    }
    pub fn mul(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Matrix dimensions must match");
        }

        let result_data: Vec<f32> = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| a * b).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    pub fn mse_derivative(&self, target: &Matrix) -> Matrix {
        if self.rows != target.rows || self.cols != target.cols {
            panic!("Matrix dimensions must match for mse_derivative: self: ({}, {}), target: ({}, {})", self.rows, self.cols, target.rows, target.cols);
        }

        let data = self.data.iter().zip(&target.data).map(|(p, a)| 2.0 * (p - a) / self.data.len() as f32).collect();
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

    pub fn mul_scalar(&self, scalar: f32) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| x * scalar).collect();
        Matrix::new(self.rows, self.cols, Some(result_data))
    }

    // Layer normalization
    pub fn layer_norm(&self, scale: &Matrix, bias: &Matrix, epsilon: f32) -> Matrix {
        let mut result = self.clone();
        
        // Normalize each row independently
        for i in 0..self.rows {
            // Calculate mean
            let mut mean = 0.0;
            for j in 0..self.cols {
                mean += self.get(i, j);
            }
            mean /= self.cols as f32;
            
            // Calculate variance
            let mut var = 0.0;
            for j in 0..self.cols {
                let diff = self.get(i, j) - mean;
                var += diff * diff;
            }
            var /= self.cols as f32;
            
            // Normalize, scale and shift
            for j in 0..self.cols {
                let normalized = (self.get(i, j) - mean) / (var + epsilon).sqrt();
                result.set(i, j, normalized * scale.get(0, j) + bias.get(0, j));
            }
        }
        
        result
    }
    
    // GELU activation (used in many transformer models)
    pub fn gelu(&self) -> Matrix {
        let result_data: Vec<f32> = self.data.iter().map(|&x| {
            // Approximation of GELU
            let cube = x * x * x;
            x * 0.5 * (1.0 + (0.797885 * (x + 0.044715 * cube)).tanh())
        }).collect();
        
        Matrix::new(self.rows, self.cols, Some(result_data))
    }
    
    // Masked self-attention for decoder
    pub fn apply_mask(&self, mask: &Matrix) -> Matrix {
        let mut result = self.clone();
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                if mask.get(i, j) == 0.0 {
                    // Apply very negative number to create near-zero attention weight after softmax
                    result.set(i, j, -1e9);
                }
            }
        }
        
        result
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
pub fn tanh(matrix: &Matrix) -> Matrix{
    matrix.tanh()
}
pub fn tanh_derivative(matrix: &Matrix) -> Matrix{
    matrix.tanh_derivative()
}
