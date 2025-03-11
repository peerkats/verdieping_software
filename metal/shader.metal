#include <metal_stdlib>
using namespace metal;

#define TS 32



kernel void simple(constant int &a        [[ buffer(0) ]],
                   constant int &b        [[ buffer(1) ]],
                   device int *result     [[ buffer(2) ]],
                   uint tid               [[ thread_position_in_grid ]]) {
    if (tid == 0) { // only one thread performs the addition
        result[0] = a * b;
    }
}


kernel void dot_product(const device float *A         [[ buffer(0) ]],
                                    const device float *B         [[ buffer(1) ]],
                                    device float *C               [[ buffer(2) ]],
                                    constant uint &m              [[ buffer(3) ]],
                                    constant uint &n              [[ buffer(4) ]],
                                    constant uint &p              [[ buffer(5) ]],
                                    uint2 tid                     [[ thread_position_in_threadgroup ]],
                                    uint2 gid                     [[ thread_position_in_grid ]])
{
    threadgroup float Atile[TS][TS];
    threadgroup float Btile[TS][TS];
    
    float sum = 0.0;
    
    for (uint t = 0; t < (n + TS - 1) / TS; t++) {
        // Clear shared memory
        Atile[tid.y][tid.x] = 0.0;
        Btile[tid.y][tid.x] = 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        uint tiledACol = t * TS + tid.x;
        uint tiledBRow = t * TS + tid.y;
        
        if (gid.y < m && tiledACol < n) {
            Atile[tid.y][tid.x] = A[gid.y * n + tiledACol];
        }
        
        if (tiledBRow < n && gid.x < p) {
            Btile[tid.y][tid.x] = B[tiledBRow * p + gid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (gid.y < m && gid.x < p) {
            for (uint k = 0; k < TS && (t * TS + k) < n; k++) {
                sum = fma(Atile[tid.y][k], Btile[k][tid.x], sum);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (gid.y < m && gid.x < p) {
        C[gid.y * p + gid.x] = sum;
    }
}

kernel void dot_product1(
    const device float *A         [[ buffer(0) ]],
    const device float *B         [[ buffer(1) ]],
    device float *C               [[ buffer(2) ]],
    constant uint &m              [[ buffer(3) ]],
    constant uint &n              [[ buffer(4) ]],
    constant uint &p              [[ buffer(5) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    // Precisely match CPU calculation order
    if (gid.y < m && gid.x < p) {
        float sum = 0.0;
        for (uint k = 0; k < n; k++) {
            // Use separate multiply and add for consistency with CPU
            float product = A[gid.y * n + k] * B[k * p + gid.x];
            sum += product;
        }
        C[gid.y * p + gid.x] = sum;
    }
}

// Updated version for exact CPU compatibility
kernel void dot_product_exact(
    const device float *A         [[ buffer(0) ]],
    const device float *B         [[ buffer(1) ]],
    device float *C               [[ buffer(2) ]],
    constant uint &m              [[ buffer(3) ]],
    constant uint &n              [[ buffer(4) ]],
    constant uint &p              [[ buffer(5) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    // Each thread handles exactly one output element
    if (gid.y < m && gid.x < p) {
        // Initialize the result element to 0
        float sum = 0.0;
        
        // Match the exact triple loop pattern from math.rs
        for (uint k = 0; k < n; k++) {
            float a_val = A[gid.y * n + k];
            float b_val = B[k * p + gid.x];
            
            // Separate operations to match CPU behavior
            float product = a_val * b_val;
            sum += product;
        }
        
        C[gid.y * p + gid.x] = sum;
    }
}

kernel void dot_exact(
    const device float *A         [[ buffer(0) ]],
    const device float *B         [[ buffer(1) ]],
    device float *C               [[ buffer(2) ]],
    constant uint &m              [[ buffer(3) ]],
    constant uint &n              [[ buffer(4) ]],
    constant uint &p              [[ buffer(5) ]],
    uint2 gid                     [[ thread_position_in_grid ]])
{
    // Each thread handles exactly one output element to match CPU behavior
    if (gid.y < m && gid.x < p) {
        // Initialize the result element to 0
        C[gid.y * p + gid.x] = 0.0;
        
        // Match the exact triple loop pattern from math.rs
        for (uint k = 0; k < n; k++) {
            float scalar = A[gid.y * n + k];
            float value = B[k * p + gid.x];
            
            // Accumulate product - separated to avoid FMA optimizations
            // that could change the numerical results
            float product = scalar * value;
            C[gid.y * p + gid.x] += product;
        }
    }
}

kernel void matrix_add(
    const device float *A    [[ buffer(0) ]],
    const device float *B     [[ buffer(1) ]],
    device float *C           [[ buffer(2) ]],
    constant uint &elements   [[ buffer(3) ]],
    constant uint &rows       [[ buffer(4) ]],
    constant uint &cols       [[ buffer(5) ]],
    uint2 gid                [[ thread_position_in_grid ]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < rows && col < cols) {
        uint idx_a = row * cols + col;
        // Support both full matrix and row vector (bias) addition
        uint idx_b = (B + elements == A + rows * cols) ? col : idx_a;
        C[idx_a] = A[idx_a] + B[idx_b];
    }
}

kernel void matrix_sub(const device float *A    [[ buffer(0) ]],
                          const device float *B      [[ buffer(1) ]],
                          device float *C            [[ buffer(2) ]],
                          constant uint &elements    [[ buffer(3) ]],
                          uint2 gid                  [[ thread_position_in_grid ]],
                          uint2 size                 [[threads_per_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    uint idx = row * size.x + col;
    
    if (idx < elements) {
        C[idx] = A[idx] - B[idx];
    }
}

kernel void matrix_multiply(const device float *A   [[ buffer(0) ]],
                          const device float *B     [[ buffer(1) ]],
                          device float *C           [[ buffer(2) ]],
                          constant uint &elements   [[ buffer(3) ]],
                          uint2 gid                [[ thread_position_in_grid ]],
                          uint2 size               [[threads_per_grid]]) 
{
    uint row = gid.y;
    uint col = gid.x;
    uint idx = row * size.x + col;
    
    if (idx < elements) {
        C[idx] = A[idx] * B[idx];
    }
}

kernel void transpose(const device float *input  [[ buffer(0) ]],
                           device float *output         [[ buffer(1) ]],
                           constant uint &rows          [[ buffer(2) ]],
                           constant uint &cols          [[ buffer(3) ]],
                           uint2 gid                    [[ thread_position_in_grid ]])
{
    if (gid.x < cols && gid.y < rows) {
        output[gid.x * rows + gid.y] = input[gid.y * cols + gid.x];
    }
}

kernel void sum_axis_0(const device float *input    [[ buffer(0) ]],
                            device float *output           [[ buffer(1) ]],
                            constant uint &rows            [[ buffer(2) ]],
                            constant uint &cols            [[ buffer(3) ]],
                            uint id                        [[ thread_position_in_grid ]])
{
    if (id < cols) {
        float sum = 0.0;
        for (uint i = 0; i < rows; i++) {
            sum += input[i * cols + id];
        }
        output[id] = sum;
    }
}

