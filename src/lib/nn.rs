use crate::lib::math::*;
use crate::lib::gpu::*;
use crate::lib::math::{sigmoid, mse, relu};
use metal::*;

pub struct Nn{
    pub layers_weights: Vec<Matrix>,
    pub layers_bias: Vec<Matrix>,
    pub activation: Vec<fn(&Matrix) -> Matrix>,
    pub loss: fn(&Matrix, &Matrix) -> Matrix,
}

impl Nn{
    // Notice input is removed from the constructor.
    pub fn new(layers_weights: Vec<Matrix>, layers_bias: Vec<Matrix>, activation: Vec<fn(&Matrix) -> Matrix>, loss: fn(&Matrix, &Matrix) -> Matrix) -> Nn{
        Nn{
            layers_weights,
            layers_bias,
            activation,
            loss,
        }
    }
    
    // forward now takes an input parameter dynamically.
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let mut current_output = input.clone();
        for (i, weight) in self.layers_weights.iter().enumerate() {
            let bias = &self.layers_bias[i];
            let activation_fn = self.activation[i];
            let preactivation = current_output.dot(&weight).add(&bias);
            current_output = activation_fn(&preactivation);
        }
        current_output
    }
    
    // forward_cached now also takes an input parameter.
    pub fn forward_cached(&self, input: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
        let mut current_output = input.clone();
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
    
    // train now takes the input dynamically.
    pub fn train(&mut self, input: &Matrix, epoch: u64, lr: f32, activation_derivatives: Vec<fn(&Matrix) -> Matrix>, target: &Matrix) -> i32 {
        for _ in 0..epoch {
            let (pre_activations, post_activations) = self.forward_cached(input);
            let output = post_activations.last().unwrap().clone(); // final output
            // Compute delta for the output layer.
            let last_index = self.layers_weights.len() - 1;
            let delta_output = mse_derivative(&output, target)
                .mul(&activation_derivatives[last_index](&pre_activations[last_index]));
            
            // Update last layer weights and biases.
            let mut prev_activation = if self.layers_weights.len() == 1 {
                input.clone()
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
                    input.clone()
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
    

    pub fn backward(&mut self, input: &Matrix, target: &Matrix, activation_derivatives: Vec<fn(&Matrix) -> Matrix>) -> (Vec<Matrix>, Vec<Matrix>) {
        // Forward pass with cached activations.
        let (pre_activations, post_activations) = self.forward_cached(input);
        let num_layers = self.layers_weights.len();
        let output = post_activations.last().unwrap().clone();
        
        // Compute delta for the output layer.
        let delta_output = mse_derivative(&output, target)
            .mul(&activation_derivatives[num_layers - 1](&pre_activations[num_layers - 1]));
        
        // Initialize gradients for weights and biases.
        let mut weight_gradients: Vec<Matrix> = Vec::with_capacity(num_layers);
        let mut bias_gradients: Vec<Matrix> = Vec::with_capacity(num_layers);
        
        // Compute gradients for the output layer.
        let mut prev_activation = if num_layers == 1 {
            input.clone()
        } else {
            post_activations[num_layers - 2].clone()
        };
        let grad_w = prev_activation.transpose().dot(&delta_output);
        let grad_b = delta_output.sum_axis_0();
        
        // Start by adding output layer gradients.
        weight_gradients.push(grad_w);
        bias_gradients.push(grad_b);
        
        // Propagate delta backwards through hidden layers.
        let mut delta = delta_output;
        // Loop from the second-last layer down to layer 0.
        for l in (0..(num_layers - 1)).rev() {
            let pre_activation = &pre_activations[l];
            delta = delta.dot(&self.layers_weights[l + 1].transpose())
                .mul(&activation_derivatives[l](pre_activation));
            let mut layer_input = if l == 0 {
                input.clone()
            } else {
                post_activations[l - 1].clone()
            };
            let mut grad_w = layer_input.transpose().dot(&delta);
            let mut grad_b = delta.sum_axis_0();
            // Insert gradients at the beginning so that index 0 remains the first layer.
            weight_gradients.insert(0, grad_w);
            bias_gradients.insert(0, grad_b);
        }
        
        (weight_gradients, bias_gradients)
    }
    pub fn update_weights(
        &mut self,
        weight_gradients: &Vec<Matrix>,
        bias_gradients: &Vec<Matrix>,
        lr: f32
    ) {
        for i in 0..self.layers_weights.len() {
            self.layers_weights[i] = self.layers_weights[i]
                .sub(&weight_gradients[i].mul_scalar(lr));
            self.layers_bias[i] = self.layers_bias[i]
                .sub(&bias_gradients[i].mul_scalar(lr));
        }
    }

    pub fn forward_gpu(
        &self, 
        input: &Matrix, 
        gpu: &Gpu,
        library: &metal::Library,
        command_queue: &metal::CommandQueue
    ) -> Matrix {
        // Create pipeline states only once
        let dot_pipeline = {
            let kernel = library.get_function("dot_product1", None)
                .expect("Could not find dot_product function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create dot product pipeline state")
        };
        
        let add_pipeline = {
            let kernel = library.get_function("matrix_add", None)
                .expect("Could not find matrix_add function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create addition pipeline state")
        };

        let mut current_output = input.clone();
        
        for (i, weight) in self.layers_weights.iter().enumerate() {
            let bias = &self.layers_bias[i];
            let activation_fn = self.activation[i];
            
            let preactivation_dot = Gpu::dot_hybrid(
                gpu,
                &current_output,
                weight,
                &dot_pipeline,
                command_queue
            );
            
            let preactivation = Gpu::add(
                gpu,
                &preactivation_dot,
                bias,
                &add_pipeline,
                command_queue
            );
            
            current_output = activation_fn(&preactivation);
        }
        
        current_output
    }

    pub fn forward_gpu_cached(
        &self, 
        input: &Matrix, 
        gpu: &Gpu,
        pipelines: &[&ComputePipelineState], // Accepts a slice of pipeline references
        command_queue: &metal::CommandQueue
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        // Use the provided pipelines instead of creating new ones
        // pipelines[0] = dot_product_exact pipeline
        // pipelines[1] = matrix_add pipeline
        
        let dot_pipeline = pipelines[0];
        let add_pipeline = pipelines[1];
    
        let mut current_output = input.clone();
        let mut pre_activations = Vec::new();
        let mut post_activations = Vec::new();
        
        for (i, weight) in self.layers_weights.iter().enumerate() {
            let bias = &self.layers_bias[i];
            let activation_fn = self.activation[i];
            
            let preactivation_dot = Gpu::dot_hybrid(
                gpu,
                &current_output,
                weight,
                dot_pipeline,
                command_queue
            );
            
            let preactivation = Gpu::add(
                gpu,
                &preactivation_dot,
                bias,
                add_pipeline,
                command_queue
            );
            
            // Store pre-activation value
            pre_activations.push(preactivation.clone());
            
            // Apply activation function and store post-activation
            let post_activation = activation_fn(&preactivation);
            post_activations.push(post_activation.clone());
            
            current_output = post_activation;
        }
        
        (pre_activations, post_activations)
    }
    pub fn train_gpu(
        &mut self, 
        input: &Matrix, 
        epoch: u64, 
        lr: f32,
        activation_derivatives: Vec<fn(&Matrix) -> Matrix>, 
        target: &Matrix, 
        gpu: &Gpu,
        library: &metal::Library,
        command_queue: &metal::CommandQueue
    ) -> i32 {
        // Create GPU pipelines for matrix operations
        let dot_pipeline = {
            let kernel = library.get_function("dot_product1", None)
                .expect("Could not find dot_product1 function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create dot product pipeline state")
        };
        
        let add_pipeline = {
            let kernel = library.get_function("matrix_add", None)
                .expect("Could not find matrix_add function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create addition pipeline state")
        };
        
        let sub_pipeline = {
            let kernel = library.get_function("matrix_sub", None)
                .expect("Could not find matrix_sub function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create subtraction pipeline state")
        };
        
        let mul_pipeline = {
            let kernel = library.get_function("matrix_multiply", None)
                .expect("Could not find matrix_multiply function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create multiplication pipeline state")
        };
        
        let transpose_pipeline = {
            let kernel = library.get_function("transpose", None)
                .expect("Could not find transpose function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create transpose pipeline state")
        };
        
        let sum_axis_pipeline = {
            let kernel = library.get_function("sum_axis_0", None)
                .expect("Could not find sum_axis_0 function");
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&kernel));
            gpu.device.new_compute_pipeline_state(&desc)
                .expect("Failed to create sum_axis pipeline state")
        };
        
        // Perform training for specified number of epochs
        for _ in 0..epoch {
            // Forward pass with CPU activation functions
            let mut current_output = input.clone();
            let mut pre_activations = Vec::new();
            let mut post_activations = Vec::new();
            
            // Forward pass
            for (i, weight) in self.layers_weights.iter().enumerate() {
                let bias = &self.layers_bias[i];
                let activation_fn = self.activation[i];
                
                // GPU matrix multiplication
                let preactivation_dot = Gpu::dot_hybrid(
                    gpu,
                    &current_output,
                    weight,
                    &dot_pipeline,
                    command_queue
                );
                
                // GPU bias addition
                let preactivation = Gpu::add(
                    gpu,
                    &preactivation_dot,
                    bias,
                    &add_pipeline,
                    command_queue
                );
                
                // Store pre-activation
                pre_activations.push(preactivation.clone());
                
                // CPU activation function (could be GPU in the future)
                let post_activation = activation_fn(&preactivation);
                post_activations.push(post_activation.clone());
                
                current_output = post_activation;
            }
            
            // Backpropagation
            let last_index = self.layers_weights.len() - 1;
            let output = post_activations.last().unwrap();
            
            // Calculate error (output - target)
            let error = Gpu::sub(gpu, output, target, &sub_pipeline, command_queue);
            
            // Apply activation derivative (CPU)
            let act_deriv = activation_derivatives[last_index](&pre_activations[last_index]);
            
            // Multiply error by activation derivative
            let delta_output = Gpu::mul(gpu, &error, &act_deriv, &mul_pipeline, command_queue);
            
            // Get previous activation
            let prev_activation = if last_index == 0 {
                input.clone()
            } else {
                post_activations[last_index - 1].clone()
            };
            
            // Calculate gradients for output layer
            let prev_act_t = Gpu::transpose(gpu, &prev_activation, &transpose_pipeline, command_queue);
            let weight_grad = Gpu::dot_hybrid(gpu, &prev_act_t, &delta_output, &dot_pipeline, command_queue);
            
            // Scale gradients by learning rate (CPU)
            let weight_update = weight_grad.mul_scalar(lr);
            
            // Update weights
            self.layers_weights[last_index] = Gpu::sub(
                gpu, 
                &self.layers_weights[last_index], 
                &weight_update, 
                &sub_pipeline, 
                command_queue
            );
            
            // Update biases
            let bias_grad = Gpu::sum_axis_0(gpu, &delta_output, &sum_axis_pipeline, command_queue);
            let bias_update = bias_grad.mul_scalar(lr);
            self.layers_bias[last_index] = Gpu::sub(
                gpu, 
                &self.layers_bias[last_index], 
                &bias_update, 
                &sub_pipeline, 
                command_queue
            );
            
            // Backpropagate through hidden layers
            let mut delta = delta_output;
            
            for l in (0..last_index).rev() {
                // Backpropagate delta
                let weights_t = Gpu::transpose(
                    gpu, 
                    &self.layers_weights[l+1], 
                    &transpose_pipeline, 
                    command_queue
                );
                
                let delta_dot = Gpu::dot_hybrid(gpu, &delta, &weights_t, &dot_pipeline, command_queue);
                
                // Apply activation derivative
                let act_deriv = activation_derivatives[l](&pre_activations[l]);
                delta = Gpu::mul(gpu, &delta_dot, &act_deriv, &mul_pipeline, command_queue);
                
                // Get input for this layer
                let layer_input = if l == 0 {
                    input.clone()
                } else {
                    post_activations[l - 1].clone()
                };
                
                // Calculate gradients
                let layer_input_t = Gpu::transpose(gpu, &layer_input, &transpose_pipeline, command_queue);
                let weight_grad = Gpu::dot_hybrid(gpu, &layer_input_t, &delta, &dot_pipeline, command_queue);
                
                // Update weights
                let weight_update = weight_grad.mul_scalar(lr);
                self.layers_weights[l] = Gpu::sub(
                    gpu, 
                    &self.layers_weights[l], 
                    &weight_update, 
                    &sub_pipeline, 
                    command_queue
                );
                
                // Update biases
                let bias_grad = Gpu::sum_axis_0(gpu, &delta, &sum_axis_pipeline, command_queue);
                let bias_update = bias_grad.mul_scalar(lr);
                self.layers_bias[l] = Gpu::sub(
                    gpu, 
                    &self.layers_bias[l], 
                    &bias_update, 
                    &sub_pipeline, 
                    command_queue
                );
            }
        }
        
        0
    }
    pub fn backward_gpu(     
        &self, 
        input: &Matrix, 
        activation_derivatives: Vec<fn(&Matrix) -> Matrix>, 
        target: &Matrix, 
        gpu: &Gpu,
        command_queue: &metal::CommandQueue,
        pipelines: &[&ComputePipelineState],
    ) -> (Vec<Matrix>, Vec<Matrix>) {
        // Use existing pipelines for matrix operations
        let dot_pipeline = pipelines[0];
        let add_pipeline = pipelines[1];
        let sub_pipeline = pipelines[2];
        let mul_pipeline = pipelines[3];
        let transpose_pipeline = pipelines[4];
        let sum_axis_pipeline = pipelines[5];
    
        // Forward pass with GPU operations
        
        let (pre_activations, post_activations) = self.forward_gpu_cached(
            &input, &gpu, &pipelines, &command_queue
        );
        
        // Initialize arrays for gradients
        let num_layers = self.layers_weights.len();
        let mut weight_gradients: Vec<Matrix> = Vec::with_capacity(num_layers);
        let mut bias_gradients: Vec<Matrix> = Vec::with_capacity(num_layers);
        
        // Calculate output layer error
        let output = post_activations.last().unwrap();
        let error = Gpu::sub(gpu, output, target, sub_pipeline, command_queue);
        
        // Apply activation derivative to error
        let act_deriv = activation_derivatives[num_layers - 1](&pre_activations[num_layers - 1]);
        let delta_output = Gpu::mul(gpu, &error, &act_deriv, mul_pipeline, command_queue);
        
        // Get previous activation (output of previous layer)
        let prev_activation = if num_layers == 1 {
            input.clone()
        } else {
            post_activations[num_layers - 2].clone()
        };
        
        // Calculate output layer gradients
        let prev_act_t = Gpu::transpose(gpu, &prev_activation, transpose_pipeline, command_queue);
        let weight_grad = Gpu::dot_hybrid(gpu, &prev_act_t, &delta_output, dot_pipeline, command_queue);
        let bias_grad = Gpu::sum_axis_0(gpu, &delta_output, sum_axis_pipeline, command_queue);
        
        // Add gradients for output layer
        weight_gradients.push(weight_grad);
        bias_gradients.push(bias_grad);
        
        // Backpropagate through hidden layers
        let mut delta = delta_output;
        
        for l in (0..num_layers - 1).rev() {
            // Backpropagate delta
            let weights_t = Gpu::transpose(
                gpu, 
                &self.layers_weights[l + 1], 
                transpose_pipeline, 
                command_queue
            );
            
            let delta_dot = Gpu::dot_hybrid(gpu, &delta, &weights_t, dot_pipeline, command_queue);
            
            // Apply activation derivative
            let act_deriv = activation_derivatives[l](&pre_activations[l]);
            delta = Gpu::mul(gpu, &delta_dot, &act_deriv, mul_pipeline, command_queue);
            
            // Get input for this layer
            let layer_input = if l == 0 {
                input.clone()
            } else {
                post_activations[l - 1].clone()
            };
            
            // Calculate gradients
            let layer_input_t = Gpu::transpose(gpu, &layer_input, transpose_pipeline, command_queue);
            let weight_grad = Gpu::dot_hybrid(gpu, &layer_input_t, &delta, dot_pipeline, command_queue);
            let bias_grad = Gpu::sum_axis_0(gpu, &delta, sum_axis_pipeline, command_queue);
            
            // Insert at beginning to maintain layer order
            weight_gradients.insert(0, weight_grad);
            bias_gradients.insert(0, bias_grad);
        }
        
        (weight_gradients, bias_gradients)
    }
        pub fn update_weights_gpu(
        &mut self,
        weight_gradients: &Vec<Matrix>,
        bias_gradients: &Vec<Matrix>,
        lr: f32,
        gpu: &Gpu,
        command_queue: &metal::CommandQueue,
        pipelines: &[&ComputePipelineState],
    ) {
        // Use existing pipelines
        let sub_pipeline = pipelines[2]; // Assuming sub_pipeline is at index 2
        
        // Update weights and biases for each layer
        for i in 0..self.layers_weights.len() {
            // Scale gradients by learning rate (currently CPU operation)
            let scaled_weight_grad = weight_gradients[i].mul_scalar(lr);
            let scaled_bias_grad = bias_gradients[i].mul_scalar(lr);
            
            // Subtract gradients from weights
            self.layers_weights[i] = Gpu::sub(
                gpu,
                &self.layers_weights[i],
                &scaled_weight_grad,
                sub_pipeline,
                command_queue
            );
            
            // Subtract gradients from biases
            self.layers_bias[i] = Gpu::sub(
                gpu,
                &self.layers_bias[i],
                &scaled_bias_grad,
                sub_pipeline,
                command_queue
            );
        }
    }

        // Save the model to a binary file
        pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
            use std::fs::File;
            use std::io::prelude::*;
            
            let mut file = File::create(filename)?;
            
            // Write number of layers
            let num_layers = self.layers_weights.len() as u32;
            file.write_all(&num_layers.to_le_bytes())?;
            
            // Write each layer's weights and biases
            for i in 0..num_layers as usize {
                // Write weight matrix dimensions
                let rows = self.layers_weights[i].rows as u32;
                let cols = self.layers_weights[i].cols as u32;
                file.write_all(&rows.to_le_bytes())?;
                file.write_all(&cols.to_le_bytes())?;
                
                // Write weight data
                for val in &self.layers_weights[i].data {
                    file.write_all(&val.to_le_bytes())?;
                }
                
                // Write bias dimensions
                let bias_rows = self.layers_bias[i].rows as u32;
                let bias_cols = self.layers_bias[i].cols as u32;
                file.write_all(&bias_rows.to_le_bytes())?;
                file.write_all(&bias_cols.to_le_bytes())?;
                
                // Write bias data
                for val in &self.layers_bias[i].data {
                    file.write_all(&val.to_le_bytes())?;
                }
            }
            
            Ok(())
        }
        
        // Load the model from a binary file
        pub fn load_from_file(
            filename: &str, 
            activations: Vec<fn(&Matrix) -> Matrix>, 
            loss: fn(&Matrix, &Matrix) -> Matrix
        ) -> std::io::Result<Self> {
            use std::fs::File;
            use std::io::prelude::*;
            
            let mut file = File::open(filename)?;
            
            // Read number of layers
            let mut buffer = [0u8; 4];
            file.read_exact(&mut buffer)?;
            let num_layers = u32::from_le_bytes(buffer);
            
            let mut weights = Vec::with_capacity(num_layers as usize);
            let mut biases = Vec::with_capacity(num_layers as usize);
            
            // Read each layer
            for _ in 0..num_layers {
                // Read weight dimensions
                file.read_exact(&mut buffer)?;
                let rows = u32::from_le_bytes(buffer) as usize;
                
                file.read_exact(&mut buffer)?;
                let cols = u32::from_le_bytes(buffer) as usize;
                
                // Read weight data
                let mut weight_data = vec![0f32; rows * cols];
                for val in &mut weight_data {
                    file.read_exact(&mut buffer)?;
                    *val = f32::from_le_bytes(buffer);
                }
                
                let weight_matrix = Matrix::new(rows, cols, Some(weight_data));
                weights.push(weight_matrix);
                
                // Read bias dimensions
                file.read_exact(&mut buffer)?;
                let bias_rows = u32::from_le_bytes(buffer) as usize;
                
                file.read_exact(&mut buffer)?;
                let bias_cols = u32::from_le_bytes(buffer) as usize;
                
                // Read bias data
                let mut bias_data = vec![0f32; bias_rows * bias_cols];
                for val in &mut bias_data {
                    file.read_exact(&mut buffer)?;
                    *val = f32::from_le_bytes(buffer);
                }
                
                let bias_matrix = Matrix::new(bias_rows, bias_cols, Some(bias_data));
                biases.push(bias_matrix);
            }
            
            // Create and return the network
            Ok(Nn::new(weights, biases, activations, loss))
        }

    
}