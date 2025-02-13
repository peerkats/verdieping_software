#include <metal_stdlib>
#include <metal_stdlib>
using namespace metal;

#define TS 16

kernel void dot_product(const device float *A         [[ buffer(0) ]],
                                    const device float *B         [[ buffer(1) ]],
                                    device float *C               [[ buffer(2) ]],
                                    constant uint &m              [[ buffer(3) ]],
                                    constant uint &n              [[ buffer(4) ]],
                                    constant uint &p              [[ buffer(5) ]],
                                    uint2 tid                     [[ thread_position_in_threadgroup ]],
                                    uint2 gid                     [[ thread_position_in_grid ]])
{
    // allocate threadgroup shared memory for tiles of A and B
    threadgroup float Atile[TS][TS];
    threadgroup float Btile[TS][TS];
    
    float sum = 0.0;
    
    // Loop over tiles.
    for (uint t = 0; t < (n + TS - 1) / TS; t++) {
        // Calculate indices within A and B for this tile.
        uint tiledACol = t * TS + tid.x;
        uint tiledBRow = t * TS + tid.y;
        
        // Load A tile element (or 0 if out-of-bound)
        if (gid.y < m && tiledACol < n) {
            Atile[tid.y][tid.x] = A[gid.y * n + tiledACol];
        } else {
            Atile[tid.y][tid.x] = 0.0;
        }
        // Load B tile element (or 0 if out-of-bound)
        if (tiledBRow < n && gid.x < p) {
            Btile[tid.y][tid.x] = B[tiledBRow * p + gid.x];
        } else {
            Btile[tid.y][tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Multiply loaded tiles.
        for (uint k = 0; k < TS; k++) {
            sum += Atile[tid.y][k] * Btile[k][tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result if within matrix bounds.
    if (gid.y < m && gid.x < p) {
        C[gid.y * p + gid.x] = sum;
    }
}