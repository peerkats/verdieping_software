use crate::lib::math::*;
use crate::lib::nn::*;
pub mod lib;
use crate::lib::math::{sigmoid, mse, sigmoid_derivative};
use crate::lib::gpu::*;
use std::path::PathBuf;
use metal::MTLSize;
use crate::lib::loader::*;

// Demo helper: Split a Matrix into mini-batches.
fn create_batches(input: &Matrix, targets: &Matrix, batch_size: usize) -> Vec<(Matrix, Matrix)> {
    let total = input.rows;
    let mut batches = Vec::new();
    for start in (0..total).step_by(batch_size) {
        let end = (start + batch_size).min(total);
        let mut batch_data = Vec::new();
        for r in start..end {
            let row_start = r * input.cols;
            let row_end = row_start + input.cols;
            batch_data.extend_from_slice(&input.data[row_start..row_end]);
        }
        let mut batch_targets = Vec::new();
        for r in start..end {
            let row_start = r * targets.cols;
            let row_end = row_start + targets.cols;
            batch_targets.extend_from_slice(&targets.data[row_start..row_end]);
        }
        let batch_input = Matrix::new(end - start, input.cols, Some(batch_data));
        let batch_target = Matrix::new(end - start, targets.cols, Some(batch_targets));
        batches.push((batch_input, batch_target));
    }
    batches
}

fn main() {
    // let path = "/Users/peerkats/Desktop/coding/verdieping/train-images.idx3-ubyte";
    //     let result = load_first_training_image(path);
    // println!("{:?}", result);
    // XOR training data: 4 samples with 2 inputs and 4 targets with 1 output
    let xor_inputs = Matrix::new(4, 2, Some(vec![
        0.0, 0.0,  // sample 1
        0.0, 1.0,  // sample 2
        1.0, 0.0,  // sample 3
        1.0, 1.0,  // sample 4
    ]));
    let xor_targets = Matrix::new(4, 1, Some(vec![
        0.0, // sample 1 => 0 XOR 0 = 0
        1.0, // sample 2 => 0 XOR 1 = 1
        1.0, // sample 3 => 1 XOR 0 = 1
        0.0, // sample 4 => 1 XOR 1 = 0
    ]));

    // Create mini-batches with a batch size of 2.
    let batches = create_batches(&xor_inputs, &xor_targets, 2);
    println!("Created {} mini-batches", batches.len());
    for (i, (batch_in, batch_target)) in batches.iter().enumerate() {
        println!("Batch {}: input {:?}", i, batch_in.data);
        println!("Batch {}: target {:?}", i, batch_target.data);
    }

    // Define a more complex network:
    // Layer 1: 2 -> 4
    let mut weight1 = Matrix::new(2, 4, None);
    weight1.fill_random();
    let mut bias1 = Matrix::new(1, 4, None);
    bias1.fill_random();

    // Layer 2: 4 -> 3
    let mut weight2 = Matrix::new(4, 3, None);
    weight2.fill_random();
    let mut bias2 = Matrix::new(1, 3, None);
    bias2.fill_random();

    // Layer 3 (output): 3 -> 1
    let mut weight3 = Matrix::new(3, 1, None);
    weight3.fill_random();
    let mut bias3 = Matrix::new(1, 1, None);
    bias3.fill_random();

    let layers_weights = vec![weight1, weight2, weight3];
    let layers_bias = vec![bias1, bias2, bias3];
    let activations: Vec<fn(&Matrix) -> Matrix> = vec![sigmoid, sigmoid, sigmoid];
    let loss_fn: fn(&Matrix, &Matrix) -> Matrix = mse;

    let mut nn = Nn::new(xor_inputs, layers_weights, layers_bias, activations, loss_fn);
        
        
    let output1 = nn.forward();
    println!("Complex NN XOR predictions: {:?}", output1.data);
    // Train the network with increased epochs and an appropriate learning rate.
    // Using sigmoid_derivative for all layers.
    let activation_derivatives: Vec<fn(&Matrix) -> Matrix> = vec![sigmoid_derivative, sigmoid_derivative, sigmoid_derivative];
    nn.train(500000, 0.1, activation_derivatives, &xor_targets);

    // Print predicted outputs after training.
    let output = nn.forward();
    println!("Complex NN XOR predictions: {:?}", output.data);
}




