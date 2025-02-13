use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::error::Error;

/// Reads only the first MNIST training image from the IDX formatted file.
/// Returns a tuple containing:
/// - number of images (as indicated in the header)
/// - number of rows
/// - number of columns
/// - a vector representing the first image as u8 pixels.
pub fn load_first_training_image<P: AsRef<Path>>(path: P) -> Result<(u32, u32, u32, Vec<u8>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 16];
    reader.read_exact(&mut header)?;

    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]);

    if magic != 2051 {
        return Err("Invalid magic number for training images".into());
    }

    let image_size = (rows * cols) as usize;
    // Read only the first image.
    let mut first_image = vec![0u8; image_size];
    reader.read_exact(&mut first_image)?;

    Ok((num_images, rows, cols, first_image))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_first_training_image() {
        // Update the path below to point to your local train-images-idx3-ubyte file.
        let path = "/Users/peerkats/Desktop/coding/verdieping/train-images.idx3-ubyte";
        let result = load_first_training_image(path);
        assert!(result.is_ok());
        let (num_images, rows, cols, image) = result.unwrap();
        println!("File indicates {} images; first image dimensions: {}x{}", num_images, rows, cols);
        println!("First 10 pixels of the first image: {:?}", &image[..10.min(image.len())]);
    }
}