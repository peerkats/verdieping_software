use metal::*;
use crate::lib::math::*;
use std::path::Path;

#[derive(Debug)]
pub struct Gpu{
    pub device: metal::Device,
    pub lib: String,
}


impl Gpu{
    pub fn new(lib: String) -> Self{
        Gpu{
            device: metal::Device::system_default().expect("did not find a system default try doing manual testin"),
            lib,
        }
        
    }
    fn create_pipeline(&self, library: &metal::Library, function_name: &str) -> metal::ComputePipelineState {
        let kernel = library.get_function(function_name, None)
            .expect(&format!("Could not find function {}", function_name));
        let desc = metal::ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&kernel));
        self.device.new_compute_pipeline_state(&desc)
            .expect(&format!("Failed to create pipeline state for {}", function_name))
    }

    pub fn dot_hybrid(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Matrix {
        let mut result = Matrix::new(matrix.rows, other.cols, None);
    
        // Create buffers for the two input matrices with optimal alignment
        let buffer_0 = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged, // Use managed for better performance
        );
        let buffer_1 = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
    
        // Allocate a buffer for the output without any initial data
        let buffer_2 = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
    
        // Create buffers for the dimension parameters
        let m_val: u32 = matrix.rows as u32;
        let buffer_3 = gpu.device.new_buffer_with_data(
            &m_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let n_val: u32 = matrix.cols as u32;
        let buffer_4 = gpu.device.new_buffer_with_data(
            &n_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let p_val: u32 = other.cols as u32;
        let buffer_5 = gpu.device.new_buffer_with_data(
            &p_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeManaged,
        );
    
        // Create a command buffer
        let command_buffer = command_queue.new_command_buffer();
    
        // Synchronize managed buffers before access
        if cfg!(target_os = "macos") {
            buffer_0.did_modify_range(metal::NSRange::new(0, buffer_0.length()));
            buffer_1.did_modify_range(metal::NSRange::new(0, buffer_1.length()));
        }
    
        // Create the compute command encoder
        let encoder = command_buffer.new_compute_command_encoder();
    
        // Set the pipeline state and buffers on the encoder
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_0), 0);
        encoder.set_buffer(1, Some(&buffer_1), 0);
        encoder.set_buffer(2, Some(&buffer_2), 0);
        encoder.set_buffer(3, Some(&buffer_3), 0);
        encoder.set_buffer(4, Some(&buffer_4), 0);
        encoder.set_buffer(5, Some(&buffer_5), 0);
    
        // Optimize threadgroup sizes based on device capabilities
        let max_threads_per_group = pipeline_state.max_total_threads_per_threadgroup();
        let ts = ((max_threads_per_group as f32).sqrt() as u64).min(32);
        
        let total_threads = metal::MTLSize {
            width: other.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: ts,
            height: ts,
            depth: 1,
        };
        // Dispatch the threads for the compute shader
        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
    
        // Commit the command buffer and wait
        command_buffer.commit();
        command_buffer.wait_until_completed();
    
        // Copy the result from the GPU output buffer to the result matrix
        unsafe {
            let result_ptr = buffer_2.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }
    
        result
    }

    /// Subtracts matrix B from matrix A elementwise using the GPU kernel "matrix_subtract".
    /// Both matrices must have identical dimensions.
    pub fn sub(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        // Ensure dimensions match.
        assert_eq!(matrix.rows, other.rows);
        assert_eq!(matrix.cols, other.cols);
        
        let elems = matrix.rows * matrix.cols;
        let mut result = Matrix::new(matrix.rows, matrix.cols, None);

        // Create buffers.
        let buffer_a = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_result = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Create a buffer for the number of elements.
        let num_elems: u32 = elems as u32;
        let buffer_const = gpu.device.new_buffer_with_data(
            &num_elems as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Create a command buffer and encoder.
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
    
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_const), 0);
    
        // Dispatch threads in 1D.
        let ts: u64 = 32;
        let total_threads = metal::MTLSize {
            width: other.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: ts,
            height: ts,
            depth: 1,
        };
    
        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
    
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            // Copy back results.
            let result_ptr = buffer_result.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    /// Adds two matrices elementwise using the GPU kernel "matrix_add".
    /// Both matrices must have identical dimensions.
    pub fn add(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        assert_eq!(matrix.cols, other.cols);
        assert!(other.rows == matrix.rows || other.rows == 1);
        
        let mut result = Matrix::new(matrix.rows, matrix.cols, None);

        // Create buffers
        let buffer_a = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_result = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Add dimension information
        let rows: u32 = matrix.rows as u32;
        let cols: u32 = matrix.cols as u32;
        let elements: u32 = (matrix.rows * matrix.cols) as u32;
        
        let buffer_elements = gpu.device.new_buffer_with_data(
            &elements as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_rows = gpu.device.new_buffer_with_data(
            &rows as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_cols = gpu.device.new_buffer_with_data(
            &cols as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_elements), 0);
        encoder.set_buffer(4, Some(&buffer_rows), 0);
        encoder.set_buffer(5, Some(&buffer_cols), 0);

        let ts: u64 = 32;
        let total_threads = MTLSize {
            width: matrix.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: ts,
            height: ts,
            depth: 1,
        };

        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_result.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    /// Elementwise multiplication of two matrices on the GPU using the "matrix_multiply" kernel.
    /// Both matrices must have the same dimensions.
    pub fn mul(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        // Ensure dimensions match
        assert_eq!(matrix.rows, other.rows);
        assert_eq!(matrix.cols, other.cols);
        
        let elems = matrix.rows * matrix.cols;
        let mut result = Matrix::new(matrix.rows, matrix.cols, None);

        // Create buffers
        let buffer_a = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_result = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // Create a constant buffer containing the number of elements.
        let num_elems: u32 = elems as u32;
        let buffer_const = gpu.device.new_buffer_with_data(
            &num_elems as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder.
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_const), 0);

        // Update thread dispatching to match the shader
        let ts: u64 = 32;
        let total_threads = metal::MTLSize {
            width: matrix.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: ts,
            height: ts,
            depth: 1,
        };

        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_result.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    pub fn transpose(
        gpu: &Gpu,
        matrix: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        let mut result = Matrix::new(matrix.cols, matrix.rows, None);
        
        // Create input and output buffers
        let buffer_input = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_output = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create dimension buffers
        let rows: u32 = matrix.rows as u32;
        let cols: u32 = matrix.cols as u32;
        
        let buffer_rows = gpu.device.new_buffer_with_data(
            &rows as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_cols = gpu.device.new_buffer_with_data(
            &cols as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_input), 0);
        encoder.set_buffer(1, Some(&buffer_output), 0);
        encoder.set_buffer(2, Some(&buffer_rows), 0);
        encoder.set_buffer(3, Some(&buffer_cols), 0);

        let ts: u64 = 32;
        let total_threads = MTLSize {
            width: matrix.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: ts,
            height: ts,
            depth: 1,
        };

        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_output.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    pub fn sum_axis_0(
        gpu: &Gpu,
        matrix: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        let mut result = Matrix::new(1, matrix.cols, None);

        // Create input and output buffers
        let buffer_input = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_output = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create dimension buffers
        let rows: u32 = matrix.rows as u32;
        let cols: u32 = matrix.cols as u32;
        
        let buffer_rows = gpu.device.new_buffer_with_data(
            &rows as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_cols = gpu.device.new_buffer_with_data(
            &cols as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_input), 0);
        encoder.set_buffer(1, Some(&buffer_output), 0);
        encoder.set_buffer(2, Some(&buffer_rows), 0);
        encoder.set_buffer(3, Some(&buffer_cols), 0);

        let total_threads = MTLSize {
            width: matrix.cols as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_output.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    pub fn sigmoid(
        gpu: &Gpu,
        input: &Matrix,
        pipeline_state: &ComputePipelineState,
        command_queue: &CommandQueue,
    ) -> Matrix {
        let mut result = Matrix::new(input.rows, input.cols, None);
        let elements = input.rows * input.cols;

        let buffer_input = gpu.device.new_buffer_with_data(
            input.data.as_ptr() as *const _,
            (input.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_output = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let elements_u32: u32 = elements as u32;
        let buffer_elements = gpu.device.new_buffer_with_data(
            &elements_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_input), 0);
        encoder.set_buffer(1, Some(&buffer_output), 0);
        encoder.set_buffer(2, Some(&buffer_elements), 0);

        let threads = MTLSize {
            width: elements as u64,
            height: 1,
            depth: 1
        };
        let threadgroup_size = MTLSize {
            width: 32,
            height: 1,
            depth: 1
        };

        encoder.dispatch_threads(threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_output.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    pub fn activation_function(
        gpu: &Gpu,
        input: &Matrix,
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Matrix {
        let mut result = Matrix::new(input.rows, input.cols, None);
        let elements = input.rows * input.cols;

        let buffer_input = gpu.device.new_buffer_with_data(
            input.data.as_ptr() as *const _,
            (input.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_output = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let elements_u32: u32 = elements as u32;
        let buffer_elements = gpu.device.new_buffer_with_data(
            &elements_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_input), 0);
        encoder.set_buffer(1, Some(&buffer_output), 0);
        encoder.set_buffer(2, Some(&buffer_elements), 0);

        // Use 1D dispatch for activation functions
        let threads = metal::MTLSize {
            width: elements as u64,
            height: 1,
            depth: 1
        };
        
        let threadgroup_size = metal::MTLSize {
            width: 256, // Larger groups for better efficiency
            height: 1,
            depth: 1
        };

        encoder.dispatch_threads(threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_output.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }
    
    pub fn mse_derivative(
        gpu: &Gpu,
        output: &Matrix,
        target: &Matrix,
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Matrix {
        assert_eq!(output.rows, target.rows);
        assert_eq!(output.cols, target.cols);
        
        let mut result = Matrix::new(output.rows, output.cols, None);
        let elements = output.rows * output.cols;

        let buffer_output = gpu.device.new_buffer_with_data(
            output.data.as_ptr() as *const _,
            (output.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_target = gpu.device.new_buffer_with_data(
            target.data.as_ptr() as *const _,
            (target.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_result = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let elements_u32: u32 = elements as u32;
        let buffer_elements = gpu.device.new_buffer_with_data(
            &elements_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_output), 0);
        encoder.set_buffer(1, Some(&buffer_target), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_elements), 0);

        // Use 1D dispatch
        let threads = metal::MTLSize {
            width: elements as u64,
            height: 1,
            depth: 1
        };
        
        let threadgroup_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1
        };

        encoder.dispatch_threads(threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe {
            let result_ptr = buffer_result.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }

        result
    }

    pub fn dot_exact(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Matrix {
        // Check dimensions
        if matrix.cols != other.rows {
            panic!("Matrix dimensions don't match for dot product: {}x{} and {}x{}", 
                matrix.rows, matrix.cols, other.rows, other.cols);
        }

        let mut result = Matrix::new(matrix.rows, other.cols, None);

        // Create buffers with shared mode for CPU compatibility
        let buffer_0 = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_1 = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_2 = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Dimension parameters
        let m_val: u32 = matrix.rows as u32;
        let n_val: u32 = matrix.cols as u32;
        let p_val: u32 = other.cols as u32;
        
        let buffer_3 = gpu.device.new_buffer_with_data(
            &m_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_4 = gpu.device.new_buffer_with_data(
            &n_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_5 = gpu.device.new_buffer_with_data(
            &p_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Execute kernel
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_0), 0);
        encoder.set_buffer(1, Some(&buffer_1), 0);
        encoder.set_buffer(2, Some(&buffer_2), 0);
        encoder.set_buffer(3, Some(&buffer_3), 0);
        encoder.set_buffer(4, Some(&buffer_4), 0);
        encoder.set_buffer(5, Some(&buffer_5), 0);
        
        // Configure threads for exact kernel
        let total_threads = metal::MTLSize {
            width: other.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        
        // Use smaller thread groups for better work distribution
        let threadgroup_size = metal::MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        
        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Copy results back
        unsafe {
            let result_ptr = buffer_2.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            result.data.copy_from_slice(result_slice);
        }
        
        result
    }

    // Additionally, we should implement a batch version for the recently added batch_dot_product kernel
    pub fn dot_batch(
        gpu: &Gpu,
        matrices: &[Matrix],  // Array of input matrices A
        weights: &[Matrix],   // Array of weight matrices B
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Vec<Matrix> {
        // Check if all matrices have compatible dimensions
        let batch_size = matrices.len();
        if batch_size != weights.len() {
            panic!("Number of input matrices must match number of weight matrices");
        }
        
        // Check dimensions and calculate total buffer sizes
        let m = matrices[0].rows;
        let n = matrices[0].cols;
        let p = weights[0].cols;
        
        // Verify all matrices have compatible dimensions
        for i in 0..batch_size {
            if matrices[i].rows != m || matrices[i].cols != n || 
            weights[i].rows != n || weights[i].cols != p {
                panic!("All matrices in batch must have compatible dimensions");
            }
        }
        
        // Create result matrices
        let mut results: Vec<Matrix> = (0..batch_size).map(|_| Matrix::new(m, p, None)).collect();
        
        // Create flattened buffers for all matrices
        let a_size = m * n * batch_size;
        let b_size = n * p * batch_size;
        let c_size = m * p * batch_size;
        
        let mut a_data = vec![0.0; a_size];
        let mut b_data = vec![0.0; b_size];
        
        // Copy all matrices to the flattened buffers
        for i in 0..batch_size {
            let a_offset = i * m * n;
            let b_offset = i * n * p;
            a_data[a_offset..a_offset + m * n].copy_from_slice(&matrices[i].data);
            b_data[b_offset..b_offset + n * p].copy_from_slice(&weights[i].data);
        }
        
        // Create GPU buffers
        let buffer_a = gpu.device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            (a_data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_b = gpu.device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            (b_data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_c = gpu.device.new_buffer(
            (c_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Dimension parameters
        let batch_u32: u32 = batch_size as u32;
        let m_val: u32 = m as u32;
        let n_val: u32 = n as u32;
        let p_val: u32 = p as u32;
        
        let buffer_batch = gpu.device.new_buffer_with_data(
            &batch_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_m = gpu.device.new_buffer_with_data(
            &m_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_n = gpu.device.new_buffer_with_data(
            &n_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_p = gpu.device.new_buffer_with_data(
            &p_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Run the kernel
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_c), 0);
        encoder.set_buffer(3, Some(&buffer_batch), 0);
        encoder.set_buffer(4, Some(&buffer_m), 0);
        encoder.set_buffer(5, Some(&buffer_n), 0);
        encoder.set_buffer(6, Some(&buffer_p), 0);
        
        // Configure threads for batch processing with 3D grid
        let total_threads = metal::MTLSize {
            width: p as u64,
            height: m as u64,
            depth: batch_size as u64,
        };
        
        // Use appropriate thread group size
        let threadgroup_size = metal::MTLSize {
            width: 8,
            height: 8,
            depth: 4,
        };
        
        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Copy results back to individual matrices
        unsafe {
            let result_ptr = buffer_c.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, c_size);
            
            for i in 0..batch_size {
                let c_offset = i * m * p;
                results[i].data.copy_from_slice(&result_slice[c_offset..c_offset + m * p]);
            }
        }
        
        results
    }

    pub fn dot_ultra_precise(
        gpu: &Gpu,
        matrix: &Matrix,
        other: &Matrix,
        pipeline_state: &metal::ComputePipelineState,
        command_queue: &metal::CommandQueue,
    ) -> Matrix {
        // Check dimensions
        if matrix.cols != other.rows {
            panic!("Matrix dimensions don't match for dot product: {}x{} and {}x{}", 
                   matrix.rows, matrix.cols, other.rows, other.cols);
        }

        let mut result = Matrix::new(matrix.rows, other.cols, None);

        // Always use StorageModeShared for most predictable behavior
        let buffer_0 = gpu.device.new_buffer_with_data(
            matrix.data.as_ptr() as *const _,
            (matrix.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_1 = gpu.device.new_buffer_with_data(
            other.data.as_ptr() as *const _,
            (other.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_2 = gpu.device.new_buffer(
            (result.data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Dimension parameters
        let m_val: u32 = matrix.rows as u32;
        let n_val: u32 = matrix.cols as u32;
        let p_val: u32 = other.cols as u32;
        
        let buffer_3 = gpu.device.new_buffer_with_data(
            &m_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_4 = gpu.device.new_buffer_with_data(
            &n_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        let buffer_5 = gpu.device.new_buffer_with_data(
            &p_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        // Execute kernel
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer_0), 0);
        encoder.set_buffer(1, Some(&buffer_1), 0);
        encoder.set_buffer(2, Some(&buffer_2), 0);
        encoder.set_buffer(3, Some(&buffer_3), 0);
        encoder.set_buffer(4, Some(&buffer_4), 0);
        encoder.set_buffer(5, Some(&buffer_5), 0);
        
        // Use smaller thread groups for more predictable execution
        let total_threads = metal::MTLSize {
            width: other.cols as u64,
            height: matrix.rows as u64,
            depth: 1,
        };
        
        let threadgroup_size = metal::MTLSize {
            width: 8,  // Smaller threadgroups for more predictable execution
            height: 8,
            depth: 1, 
        };
        
        encoder.dispatch_threads(total_threads, threadgroup_size);
        encoder.end_encoding();
        
        // Wait for complete execution
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Copy results back with verification
        unsafe {
            let result_ptr = buffer_2.contents() as *const f32;
            let result_slice = std::slice::from_raw_parts(result_ptr, result.data.len());
            
            // Manually copy to ensure no optimization interferes
            for i in 0..result.data.len() {
                result.data[i] = result_slice[i];
            }
        }
        
        result
    }

    // Add a validation function to compare CPU and GPU results
    pub fn validate_gpu_precision(matrix: &Matrix, other: &Matrix, gpu: &Gpu, pipeline_state: &metal::ComputePipelineState, command_queue: &metal::CommandQueue) -> (f32, f32, f32) {
        // CPU calculation
        let cpu_result = matrix.dot(other);
        
        // GPU calculation with ultra-precise kernel
        let gpu_result = Gpu::dot_ultra_precise(gpu, matrix, other, pipeline_state, command_queue);
        
        // Compute statistics about the difference
        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut avg_diff = 0.0f32;
        
        for i in 0..cpu_result.data.len() {
            let cpu_val = cpu_result.data[i];
            let gpu_val = gpu_result.data[i];
            
            let abs_diff = (cpu_val - gpu_val).abs();
            let rel_diff = if cpu_val != 0.0 { abs_diff / cpu_val.abs() } else { 0.0 };
            
            max_abs_diff = max_abs_diff.max(abs_diff);
            max_rel_diff = max_rel_diff.max(rel_diff);
            avg_diff += abs_diff;
        }
        
        avg_diff /= cpu_result.data.len() as f32;
        
        (max_abs_diff, max_rel_diff, avg_diff)
    }
}