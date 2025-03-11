use crate::lib::math::Matrix;
use std::error::Error;
use std::fs;

/// Loads training data from an IDX file (e.g. MNIST images).
/// Assumes a 16-byte header and that images are stored sequentially.
/// Normalizes pixel values to [0, 1] and returns a Matrix of shape (num_samples, sample_size).
pub fn load_training_data(path: &str, num_samples: usize, sample_size: usize) -> Result<Matrix, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let header_size = 16;
    if bytes.len() < header_size + num_samples * sample_size {
        return Err("File too short for the specified dimensions".into());
    }
    let image_bytes = &bytes[header_size..header_size + num_samples * sample_size];
    let data: Vec<f32> = image_bytes.iter().map(|&b| b as f32 / 255.0).collect();
    Ok(Matrix::new(num_samples, sample_size, Some(data)))
}

/// Loads labels from an IDX file (e.g. MNIST labels).
/// Assumes an 8-byte header and that labels are stored sequentially.
/// Returns a Matrix of shape (num_samples, 1).
pub fn load_labels(path: &str, num_samples: usize) -> Result<Matrix, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let header_size = 8;
    if bytes.len() < header_size + num_samples {
        return Err("File too short for the specified number of labels".into());
    }
    let label_bytes = &bytes[header_size..header_size + num_samples];
    let data: Vec<f32> = label_bytes.iter().map(|&b| b as f32).collect();
    Ok(Matrix::new(num_samples, 1, Some(data)))
}
pub fn load_training_data_as_columns(
    path: &str,
    num_samples: usize,
    sample_size: usize, // for MNIST: 784
) -> Result<Matrix, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let header_size = 16;
    if bytes.len() < header_size + num_samples * sample_size {
        return Err("File too short for the specified dimensions".into());
    }
    
    // Read the raw image bytes and normalize them.
    let col_data: Vec<f32> = bytes[header_size..header_size + num_samples * sample_size]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();

    // Rearrange col_data so that each column (in the Matrix) is one image.
    // Our desired matrix dimensions are (sample_size, num_samples)
    // In row-major order, the element at row i, col j is placed at: i * num_samples + j.
    let mut data = vec![0.0; num_samples * sample_size];
    for col in 0..num_samples {
        for row in 0..sample_size {
            data[row * num_samples + col] = col_data[col * sample_size + row];
        }
    }
    
    Ok(Matrix::new(sample_size, num_samples, Some(data)))
}
