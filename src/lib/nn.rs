use crate::lib::math::*;
use crate::lib::math::{sigmoid, mse, relu};

pub struct Nn{
    pub input: Matrix,
    pub layers_weights: Vec<Matrix>,
    pub layers_bias: Vec<Matrix>,
    pub activation: Vec<fn(&Matrix) -> Matrix>,
    pub loss: fn(&Matrix, &Matrix) -> Matrix,
}

impl Nn{
    pub fn new(input: Matrix, layers_weights: Vec<Matrix>, layers_bias: Vec<Matrix>, activation: Vec<fn(&Matrix) -> Matrix>, loss: fn(&Matrix, &Matrix) -> Matrix) -> Nn{
        Nn{
            input,
            layers_weights,
            layers_bias,
            activation,
            loss,
        }
    }
    pub fn forward(&self) -> Matrix {
        let mut current_output = self.input.clone();
        for (i, weight) in self.layers_weights.iter().enumerate() {
            let bias = &self.layers_bias[i];
            let activation_fn = self.activation[i];
            let preactivation = current_output.dot(&weight).add(&bias);
            current_output = activation_fn(&preactivation);
        }
        current_output
    }
    pub fn forward_cached(&self) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut current_output = self.input.clone();
        let mut pre_activations = Vec::new();
        let mut post_activations = Vec::new();

        for (i, weight) in self.layers_weights.iter().enumerate() {
            let bias = &self.layers_bias[i];
            let activation_fn = self.activation[i];
            let pre_activation = current_output.dot(&weight).add(&bias);
            let post_activation = activation_fn(&pre_activation);

            pre_activations.push(pre_activation);
            post_activations.push(post_activation.clone());

            current_output = post_activation;
        }

        (pre_activations, post_activations)
    }
    pub fn train(&mut self, epoch: u64, lr: f64, activation_derivatives: Vec<fn(&Matrix) -> Matrix>, target: &Matrix) -> i32 {
        for _ in 0..epoch {
            let (pre_activations, post_activations) = self.forward_cached();
            let output = self.forward(); // final output
            
            // Compute delta for the output layer
            let last_index = self.layers_weights.len() - 1;
            let delta_output = mse_derivative(&output, target)
                .mul(&activation_derivatives[last_index](&pre_activations[last_index]));
            
            // Update last layer weights and biases.
            let mut prev_activation = if self.layers_weights.len() == 1 {
                self.input.clone()
            } else {
                post_activations[last_index - 1].clone()
            };
            self.layers_weights[last_index] = self.layers_weights[last_index]
                .sub(&prev_activation.transpose().dot(&delta_output).mul_scalar(lr));
            self.layers_bias[last_index] = self.layers_bias[last_index]
                .sub(&delta_output.sum_axis_0().mul_scalar(lr));
            
            // Propagate delta backwards through hidden layers.
            let mut delta = delta_output;
            for l in (0..last_index).rev() {
                let pre_activation = &pre_activations[l];
                delta = delta.dot(&self.layers_weights[l+1].transpose())
                    .mul(&activation_derivatives[l](pre_activation));
                let mut layer_input = if l == 0 {
                    self.input.clone()
                } else {
                    post_activations[l - 1].clone()
                };
                self.layers_weights[l] = self.layers_weights[l]
                    .sub(&layer_input.transpose().dot(&delta).mul_scalar(lr));
                self.layers_bias[l] = self.layers_bias[l]
                    .sub(&delta.sum_axis_0().mul_scalar(lr));
            }
        }
        0
    }
}