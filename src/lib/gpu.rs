use metal::*;
use crate::lib::math::*;
use std::path::PathBuf;
use std::mem;

#[derive(Debug)]
pub struct Gpu {
    pub device: metal::Device,
    pub buffer: Matrix,
    pub shader: PathBuf,
    pub function: String,
}

impl Gpu {
    /// Initializes a new Gpu instance with the given buffer, shader file path, and function name.
    pub fn new(buffer: Matrix, shader: PathBuf, function: String) -> Self {
        let device = Device::system_default().expect("No Metal device found!");
        Self { device, buffer, shader, function }
    }

    /// Loads the Metal shader library from the specified shader path.
    pub fn load_shader_library(&self) -> Library {
        self.device
            .new_library_with_file(&self.shader)
            .expect("Failed to load Metal shader library")
    }
    
    /// Prepares the compute pipeline by loading the shader function and creating a pipeline state.
    pub fn prepare_pipeline(&self) -> ComputePipelineState {
        let library = self.load_shader_library();
        let function = library.get_function(&self.function, None)
            .expect("Failed to get function from shader");
        self.device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline state")
    }

    /// Dispatches the compute command using the given pipeline state and optional threadgroup configurations.
    ///
    /// # Parameters
    ///
    /// - `pipeline_state`: A reference to the compute pipeline state to be used.
    /// - `threadgroup_size`: Optional threadgroup size. Defaults to (1, 1, 1) if `None`.
    /// - `threadgroup_count`: Optional threadgroup count. Defaults to (data.len(), 1, 1) if `None`.
    pub fn dispatch_compute(
        &self, 
        pipeline_state: &ComputePipelineState, 
        threadgroup_size: Option<MTLSize>, 
        threadgroup_count: Option<MTLSize>,
    ) {
        // Convert the Matrix data to a Vec<f32> (assumed ML-friendly format)
        let data: Vec<f32> = self.buffer.data.iter().map(|&x| x as f32).collect();
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let command_queue = self.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(pipeline_state);
        encoder.set_buffer(0, Some(&buffer), 0);

        // Use provided threadgroup configurations or default values.
        let tg_size = threadgroup_size.unwrap_or_else(|| MTLSize::new(1, 1, 1));
        let tg_count = threadgroup_count.unwrap_or_else(|| MTLSize::new(data.len() as u64, 1, 1));
        encoder.dispatch_threads(tg_count, tg_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Retrieve and print the results from the GPU.
        let result_ptr = buffer.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(result_ptr, data.len()) };
        println!("Computed results: {:?}", result);
    }

    /// Runs the complete GPU compute process by preparing the pipeline and dispatching the compute commands.
    ///
    /// This method accepts optional threadgroup configuration parameters to allow dynamic tuning.
    pub fn run(
        &self, 
        threadgroup_size: Option<MTLSize>, 
        threadgroup_count: Option<MTLSize>,
    ) {
        let pipeline_state = self.prepare_pipeline();
        self.dispatch_compute(&pipeline_state, threadgroup_size, threadgroup_count);
    }

    /// Runs a dot product kernel on two input vectors.
    ///
    /// Assumes that the dot_product kernel accepts:
    /// - Buffers 0 and 1 as the two input vectors,
    /// - Buffer 2 as the output scalar result,
    /// - Buffer 3 as a constant containing the vector length.
    pub fn run_matrix_dot(&self, a: &Matrix, b: &Matrix) -> Matrix {
        if a.cols != b.rows {
            panic!(
                "Matrix dimensions must match for dot product: A: ({}x{}), B: ({}x{})",
                a.rows, a.cols, b.rows, b.cols
            );
        }
        let m = a.rows as u32;
        let n = a.cols as u32;
        let p = b.cols as u32;

        // Convert data from f64 to f32.
        let a_data: Vec<f32> = a.data.iter().map(|&x| x as f32).collect();
        let b_data: Vec<f32> = b.data.iter().map(|&x| x as f32).collect();
        let result_size = (m * p) as usize;
        let mut result_data = vec![0.0f32; result_size];

        // Create Metal buffers.
        let buffer_a = self.device.new_buffer_with_data(
            a_data.as_ptr() as *const _,
            (a_data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = self.device.new_buffer_with_data(
            b_data.as_ptr() as *const _,
            (b_data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_result = self.device.new_buffer_with_data(
            result_data.as_ptr() as *const _,
            (result_data.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_m = self.device.new_buffer_with_data(
            (&m) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_n = self.device.new_buffer_with_data(
            (&n) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_p = self.device.new_buffer_with_data(
            (&p) as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Load the optimized kernel.
        let library = self.load_shader_library();
        // Ensure self.function is set to "matrixMultiplyOptimized"
        let function = library.get_function(&self.function, None)
            .expect("Failed to get kernel function from shader");
        let pipeline_state = self.device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline state");

        let command_queue = self.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&buffer_a), 0);
        encoder.set_buffer(1, Some(&buffer_b), 0);
        encoder.set_buffer(2, Some(&buffer_result), 0);
        encoder.set_buffer(3, Some(&buffer_m), 0);
        encoder.set_buffer(4, Some(&buffer_n), 0);
        encoder.set_buffer(5, Some(&buffer_p), 0);

        // TS is defined in the shader as 16.
        let ts = 16u64;
        // Calculate grid dimensions (rounding up to cover entire matrix).
        let grid_width = ((p as u64) + ts - 1) / ts * ts;
        let grid_height = ((m as u64) + ts - 1) / ts * ts;
        let grid_size = MTLSize::new(grid_width, grid_height, 1);
        let threadgroup_size = MTLSize::new(ts, ts, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Retrieve and convert results.
        let result_ptr = buffer_result.contents() as *const f32;
        let result_slice = unsafe { std::slice::from_raw_parts(result_ptr, result_size) };
        let result_f64: Vec<f64> = result_slice.iter().map(|&x| x as f64).collect();

        Matrix::new(a.rows, b.cols, Some(result_f64))
    }
}