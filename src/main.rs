use crate::lib::math::*;
use crate::lib::nn::*;
use metal::*;
pub mod lib;
use crate::lib::math::{sigmoid, mse, sigmoid_derivative};
use crate::lib::dataloader::*;
use std::path::Path;
use std::mem::size_of;
use crate::lib::gpu::*;
use std::cmp::min;
use rand::seq::SliceRandom;

// Leaving the old helper functions for reference
fn making_lable(matrix: &Matrix) -> Vec<Matrix> {
    let mut result: Vec<Matrix> = Vec::new();
    for i in 0..matrix.cols {
        for j in 0..matrix.rows {
            let k = matrix.get(j, i);
            let m = Matrix::new(1, 1, Some(vec![k]));
            result.push(m);
        }
    }
    result
}

fn making_inputs(matrix: &Matrix) -> Vec<Matrix> {
    let mut result: Vec<Matrix> = Vec::new();
    
    for i in 0..matrix.rows {  // Iterate through rows (each row is an image)
        let mut image_data = Vec::new();
        for j in 0..matrix.cols {  // Collect all pixels for this image
            let pixel = matrix.get(i, j);
            image_data.push(pixel);
        }
        // Create a row vector (1×784) instead of a column vector (784×1)
        let image_matrix = Matrix::new(1, matrix.cols, Some(image_data));
        result.push(image_matrix);
    }
    result
}

fn create_mini_batches(input_matrix: &Matrix, target_matrix: &Matrix, batch_size: usize) 
    -> Vec<(Matrix, Matrix)> {
    
    let num_examples = input_matrix.rows;
    let num_batches = (num_examples + batch_size - 1) / batch_size;
    let mut batches = Vec::with_capacity(num_batches);
    
    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, num_examples);
        let actual_batch_size = end_idx - start_idx;
        
        // Create matrices for this batch
        let mut batch_input = Matrix::new(actual_batch_size, input_matrix.cols, None);
        let mut batch_target = Matrix::new(actual_batch_size, 10, None); // 10 for MNIST
        
        // Fill with data from multiple examples
        for (i, example_idx) in (start_idx..end_idx).enumerate() {
            // Copy input features
            for j in 0..input_matrix.cols {
                batch_input.set(i, j, input_matrix.get(example_idx, j));
            }
            
            // Set one-hot target
            let label = target_matrix.get(example_idx, 0) as usize;
            batch_target.set(i, label, 1.0);
        }
        
        batches.push((batch_input, batch_target));
    }
    
    batches
}

fn main() {
    // Load the MNIST dataset
    let activation_functions: Vec<fn(&Matrix) -> Matrix> = vec![sigmoid, sigmoid, sigmoid, sigmoid];
    let loaded_nn = Nn::load_from_file("test", activation_functions, mse).unwrap();

    let input = load_training_data("/Users/peerkats/Desktop/coding/verdieping/train-images.idx3-ubyte", 60000, 784)
        .expect("Failed to load training images");
    println!("Loaded training images: {}x{}", input.rows, input.cols);
    
    let label_matrix = load_labels("/Users/peerkats/Desktop/coding/verdieping/MNIST Train Labels", 60000)
        .expect("Failed to load training labels");
    println!("Loaded training labels: {}x{}", label_matrix.rows, label_matrix.cols);

    // Create mini-batches from the data
    let batch_size = 32; // Typical mini-batch size
    println!("Creating {} mini-batches with batch size {}", 
             (input.rows + batch_size - 1) / batch_size, batch_size);
    let mini_batches = create_mini_batches(&input, &label_matrix, batch_size);
    println!("Created {} mini-batches", mini_batches.len());
    
    // Initialize weights and biases for a network with 3 hidden layers and an output layer
    // Architecture: 784 -> 512 -> 256 -> 128 -> 10
    let mut w1 = Matrix::new(784, 512, None); w1.fill_random_centered();
    let mut b1 = Matrix::new(1, 512, None); b1.fill_random_centered();
    let mut w2 = Matrix::new(512, 256, None); w2.fill_random_centered();
    let mut b2 = Matrix::new(1, 256, None); b2.fill_random_centered();
    let mut w3 = Matrix::new(256, 128, None); w3.fill_random_centered();
    let mut b3 = Matrix::new(1, 128, None); b3.fill_random_centered();
    let mut w4 = Matrix::new(128, 10, None); w4.fill_random_centered();
    let mut b4 = Matrix::new(1, 10, None); b4.fill_random_centered();
    
    // Create the neural network with 4 layers
    let mut nn = Nn::new(
        vec![w1, w2, w3, w4],
        vec![b1, b2, b3, b4],
        vec![sigmoid, sigmoid, sigmoid, sigmoid],
        mse
    );
    
    // Training parameters
    let epochs = 5; // Reduced for testing
    let lr = 0.5;   // Adjusted learning rate for batch training
    
    // Activation derivatives
    let activation_derivatives: Vec<fn(&Matrix) -> Matrix> = vec![
        sigmoid_derivative,
        sigmoid_derivative,
        sigmoid_derivative,
        sigmoid_derivative
    ];
    
    // Initialize GPU context
    let gpu = Gpu::new(String::from("./metal/shader.metallib"));
    let lib_path = std::path::Path::new(&gpu.lib);
    let library = gpu.device.new_library_with_file(lib_path).expect("Failed to load Metal library");
    
    let command_queue = gpu.device.new_command_queue();
    let inputss =  making_inputs(&input);
    let output = loaded_nn.forward_gpu(&inputss[0], &gpu, &library, &command_queue);
    println!("{:?}", output);
    

    println!("Starting training with {} epochs", epochs);
    

    let num_test = 100;
    let mut test_examples = Vec::new();
    for i in 0..num_test {
        test_examples.push(i % input.rows);
    }
        // Create pipeline states once outside of any loops
    // This drastically reduces overhead
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
    
    // Store all pipelines in a single vector
    let pipelines = vec![
        &dot_pipeline, 
        &add_pipeline, 
        &sub_pipeline, 
        &mul_pipeline, 
        &transpose_pipeline, 
        &sum_axis_pipeline
    ];
    
    // Train the network using mini-batches
    for epoch in 0..epochs {
        println!("Epoch {}/{}", epoch+1, epochs);
        
        // Optional: shuffle mini-batches for each epoch
        let mut batch_indices: Vec<usize> = (0..mini_batches.len()).collect();
        // batch_indices.shuffle(&mut rand::thread_rng());  // Uncomment if you have rand crate
        
        let mut total_batches = 0;
        for &batch_idx in batch_indices.iter().take(1875) { // Limit to 100 batches for testing
            let (batch_input, batch_target) = &mini_batches[batch_idx];
            
            // Process the entire batch at once
            // 1. Calculate gradients with backward_gpu
            let (weight_gradients, bias_gradients) = nn.backward_gpu(
                batch_input,
                activation_derivatives.clone(),
                batch_target,
                &gpu,
                &command_queue,
                &pipelines
            );
            
            // 2. Update weights with update_weights_gpu
            nn.update_weights_gpu(
                &weight_gradients,
                &bias_gradients,
                lr,
                &gpu,
                &command_queue,
                &pipelines
            );
            
            total_batches += 1;
            if total_batches % 5 == 0 {
                println!("  Processed {}/{} batches", total_batches, min(1875, mini_batches.len()));
            }
        }
    }
        
        // Evaluation code remains the same...
    // Train the network using mini-batches
    println!("Training complete!");
    println!("\n=== Model Evaluation ===");

// Test a few examples
let eval_samples = 99; // Number of samples to evaluate
println!("Testing {} random samples from test set:", eval_samples);

// Use first few test examples for consistency
for i in 0..eval_samples {
    let test_idx = test_examples[i];
    
    // Extract test example
    let mut test_input = Matrix::new(1, input.cols, None);
    for j in 0..input.cols {
        test_input.set(0, j, input.get(test_idx, j));
    }
    
    // Forward pass
    let output = nn.forward_gpu(&test_input, &gpu, &library, &command_queue);
    
    // Get true label
    let true_label = label_matrix.get(test_idx, 0) as usize;
    
    // Print true label and network output
    print!("Sample #{}: True label: {}, Network outputs: [", test_idx, true_label);
    for j in 0..output.cols {
        print!("{:.4}", output.get(0, j));
        if j < output.cols - 1 {
            print!(", ");
        }
    }
    println!("]");
}
    nn.save_to_file("test");
}

// Helper to convert label to one-hot target
fn target(matrix: &Matrix, rows: usize, cols: usize) -> Matrix {
    let mut out = Matrix::new(rows, cols, None);
    let index = matrix.data[0] as usize;
    if index < out.data.len() {
        out.data[index] = 1.0;
    }
    out
}