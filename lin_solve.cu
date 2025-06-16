#include <stddef.h>
#include <cuda_runtime.h> // Core CUDA runtime APIs
#include <device_launch_parameters.h> // For thread/block ins
#include <stdio.h>
extern "C" {
#include "solver.h"
}

__global__ void lin_solve_rb_step_kernel(grid_color color,
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
                   float * x, const float * x0,
                   float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float *d_x0;
    float *d_x;
    size_t total_size = 2 * color_size * sizeof(float);

    cudaError_t err1 = cudaMalloc((void**)&d_x, total_size);
    cudaError_t err2 = cudaMalloc((void**)&d_x0, total_size);

    if(err1 != cudaSuccess || err2 != cudaSuccess)
    {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err1));
	fprintf(stderr, "CUDA malloc failed %s\n", cudaGetErrorString(err2));
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_x, x, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_x0, x0, total_size, cudaMemcpyHostToDevice) != cudaSuccess )
    {
        fprintf(stderr, "CUDA Memcpy failed \n");
        exit(EXIT_FAILURE);
    }
    
    float * red = d_x;
    float * blk = d_x + color_size;
    const float * red0 = d_x0;
    const float * blk0 = d_x0 + color_size;

    dim3 blockDim(32, 4);  // tunable
    dim3 gridDim((color_size + blockDim.x - 1) / blockDim.x, (color_size + blockDim.y - 1) / blockDim.y);

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(RED, n, a, c, red0, blk, red);

        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(BLACK, n, a, c, blk0, red, blk);
        
        set_bnd(n, b, x);
    }

    cudaMemcpy(x, d_x, total_size, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_x0);
}
