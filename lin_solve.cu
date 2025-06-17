#include <stddef.h>
#include <cuda_runtime.h> // Core CUDA runtime APIs
#include <device_launch_parameters.h> // For thread/block ins
#include <stdio.h>
extern "C" {
#include "solver.h"
}

__global__ void set_bnd_kernel(unsigned int n, boundary b, float* x)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    // Handle the main boundary loops (i from 1 to n)
    if (i <= n) {
        // Left and right boundaries
        x[IX(0, i)] = (b == VERTICAL) ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = (b == VERTICAL) ? -x[IX(n, i)] : x[IX(n, i)];
        
        // Top and bottom boundaries
        x[IX(i, 0)] = (b == HORIZONTAL) ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = (b == HORIZONTAL) ? -x[IX(i, n)] : x[IX(i, n)];
    }
    
    // Handle corners with a single thread (thread 0 of block 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
        x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
        x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
    }
}

__device__ void lin_solve_rb_step_kernel(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float * __restrict__ same0,
                              const float * __restrict__ neigh,
                              float * __restrict__ same)
{
    unsigned int width = (n + 2) / 2;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    int shift = ((y % 2 == 0) ^ (color == RED)) ? 1 : -1;
    int start = (y % 2 == 0) == (color == RED) ? 0 : 1;
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x + start;

    if (y > n || x > width - (1 - start)) return;
    
    // for (unsigned int x = threadIdx.x + start; x < width - (1 - start); x += blockDim.x) {
    int index = y * width + x;
    
    same[index] = (same0[index] + a * (
        neigh[index - width] +
        neigh[index] +
        neigh[index + shift] +
        neigh[index + width])) / c;
}


void lin_solve(unsigned int n, boundary b,
                   float * d_x, const float * d_x0,
                   float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float * red = d_x;
    float * blk = d_x + color_size;
    const float * red0 = d_x0;
    const float * blk0 = d_x0 + color_size;

    dim3 blockDim(128);  // tunable
    dim3 gridDim((color_size + blockDim.x - 1) / blockDim.x);

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(RED, n, a, c, red0, blk, red);

        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(BLACK, n, a, c, blk0, red, blk);
        
        set_bnd_kernel<<<gridDim, blockDim>>>(n, b, d_x);
    }
}