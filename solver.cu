#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <cuda_runtime.h> // Core CUDA runtime APIs
#include <device_launch_parameters.h> // For thread/block intrinsics
extern "C" {
#include "solver.h"
}

#include "solver.h"
#include "indices.h"

#define IX(i, j) ((i) + (n + 2) * (j))
#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
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

    if (y > n || x > width) return;
    
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

    dim3 blockDim(32, 8);  // tunable
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(RED, n, a, c, red0, blk, red);

        lin_solve_rb_step_kernel<<<gridDim, blockDim>>>(BLACK, n, a, c, blk0, red, blk);
        
        set_bnd_kernel<<<gridDim, blockDim>>>(n, b, d_x);
    }
}

// CUDA kernel for add_source operation
__global__ void add_source_kernel(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int N = n + 2;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if(idy < N || idx < N)
    {
        unsigned int index = idx + idy * N;
        x[index] += dt * s[index];
    }
}

__global__ static void kernel_advect(unsigned int n, boundary b, float* d_d, const float* d_d0, const float* d_u, const float* d_v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;
    float dt0 = dt * n;

    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= n && j <= n && i > 0 && j > 0)
    {
        x = i - dt0 * d_u[IX(i, j)];
        y = j - dt0 * d_v[IX(i, j)];
        if (x < 0.5f) {
            x = 0.5f;
        } else if (x > n + 0.5f) {
            x = n + 0.5f;
        }
        i0 = (int)x;
        i1 = i0 + 1;
        if (y < 0.5f) {
            y = 0.5f;
        } else if (y > n + 0.5f) {
            y = n + 0.5f;
        }
        j0 = (int)y;
        j1 = j0 + 1;
        s1 = x - i0;
        s0 = 1 - s1;
        t1 = y - j0;
        t0 = 1 - t1;
        d_d[IX(i, j)] = s0 * (t0 * d_d0[IX(i0, j0)] + t1 * d_d0[IX(i0, j1)]) + s1 * (t0 * d_d0[IX(i1, j0)] + t1 * d_d0[IX(i1, j1)]);
    }
}

static void diffuse (unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

__global__ static void kernel_project_p1(unsigned int n, float* u, float* v, float* p, float* div)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= n && j <= n && i > 0 && j > 0)
    {
        div[IX(i, j)] = -0.5f * (u[IX(i+1, j)] - u[IX(i-1, j)] +
                                v[IX(i, j+1)] - v[IX(i, j-1)]) / n;
        p[IX(i, j)] = 0;
    }
    
}

__global__ static void kernel_project_p2(unsigned int n, float* u, float* v, float* p, float* div)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= n && j <= n && i > 0 && j > 0)
    {
        u[IX(i, j)] -= 0.5f*n * (p[IX(i+1, j)] - p[IX(i-1, j)]);
        v[IX(i, j)] -= 0.5f*n * (p[IX(i, j+1)] - p[IX(i, j-1)]);
    }
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    float *d_x;
    float *d_x0;
    float *d_u;
    float *d_v;
    size_t total_size = (n + 2) * (n + 2) * sizeof(float);

    cudaError_t err1 = cudaMalloc((void**)&d_x, total_size);
    cudaError_t err2 = cudaMalloc((void**)&d_x0, total_size);
    cudaError_t err3 = cudaMalloc((void**)&d_u, total_size);
    cudaError_t err4 = cudaMalloc((void**)&d_v, total_size);

    if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess)
    {
        fprintf(stderr, "CUDA Malloc failed \n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_x, x, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_x0, x0, total_size, cudaMemcpyHostToDevice) != cudaSuccess || 
        cudaMemcpy(d_u, u, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v, total_size, cudaMemcpyHostToDevice) != cudaSuccess )
    {
        fprintf(stderr, "CUDA Memcpy failed \n");
        exit(EXIT_FAILURE);
    }

    //define block dimensions
    int N = n+2;
    dim3 blockDim(32, 4);  // 128 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    //ADD_SOURCE
    add_source_kernel<<<gridDim, blockDim>>>(n, d_x, d_x0, dt);
    SWAP(d_x0, d_x);

    //DIFUSE
    diffuse(n, NONE, d_x, d_x0, diff, dt);
    SWAP(d_x0, d_x);
    
    //ADVECT
    kernel_advect<<<gridDim, blockDim>>>(n, NONE, d_x, d_x0, d_u, d_v, dt);

    set_bnd_kernel<<<gridDim, blockDim>>>(n, NONE, d_x);

    //bring back results, u & v constants
    cudaMemcpy(x, d_x, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(x0, d_x0, total_size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_u);
    cudaFree(d_v);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    float *d_u;
    float *d_v;
    float *d_u0;
    float *d_v0;
    size_t total_size = (n + 2) * (n + 2) * sizeof(float);

    cudaError_t err1 = cudaMalloc((void**)&d_u, total_size);
    cudaError_t err2 = cudaMalloc((void**)&d_v, total_size);
    cudaError_t err3 = cudaMalloc((void**)&d_u0, total_size);
    cudaError_t err4 = cudaMalloc((void**)&d_v0, total_size);

        if(err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess)
    {
        fprintf(stderr, "CUDA Malloc failed \n");
        exit(EXIT_FAILURE);
    }

    if (cudaMemcpy(d_u, u, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v, total_size, cudaMemcpyHostToDevice) != cudaSuccess || 
        cudaMemcpy(d_u0, u0, total_size, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v0, v0, total_size, cudaMemcpyHostToDevice) != cudaSuccess )
    {
        fprintf(stderr, "CUDA Memcpy failed \n");
        exit(EXIT_FAILURE);
    }

    //define block dimensions
    int N = n+2;
    dim3 blockDim(32, 4);  // 128 threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    //ADD_SOURCE
    add_source_kernel<<<gridDim, blockDim>>>(n, d_u, d_u0, dt);
    add_source_kernel<<<gridDim, blockDim>>>(n, d_v, d_v0, dt);
    SWAP(d_u0, d_u);
    
    //DIFFUSE
    diffuse(n, VERTICAL, d_u, d_u0, visc, dt);
    SWAP(d_v0, d_v);
    diffuse(n, HORIZONTAL, d_v, d_v0, visc, dt);

    //PROJECT
    kernel_project_p1<<<gridDim, blockDim>>>(n, d_u, d_v, d_u0, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, NONE, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, NONE, d_u0);
    lin_solve(n, NONE, d_u0, d_v0, 1, 4);
    kernel_project_p2<<<gridDim, blockDim>>>(n, d_u, d_v, d_u0, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, VERTICAL, d_u);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, HORIZONTAL, d_v);

    SWAP(d_u0, d_u);
    SWAP(d_v0, d_v);

    //ADVECT
    kernel_advect<<<gridDim, blockDim>>>(n, VERTICAL, d_u, d_u0, d_u0, d_v0, dt);
    kernel_advect<<<gridDim, blockDim>>>(n, HORIZONTAL, d_v, d_v0, d_u0, d_v0, dt);
    
    //PROJECT
    kernel_project_p1<<<gridDim, blockDim>>>(n, d_u, d_v, d_u0, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, NONE, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, NONE, d_u0);
    lin_solve(n, NONE, d_u0, d_v0, 1, 4);
    kernel_project_p2<<<gridDim, blockDim>>>(n, d_u, d_v, d_u0, d_v0);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, VERTICAL, d_u);
    set_bnd_kernel<<<gridDim, blockDim>>>(n, HORIZONTAL, d_v);

    // check which pointer's data changed: d_u d_v changed
    cudaMemcpy(u, d_u, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u0, d_u0, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d_v, total_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(v0, d_v0, total_size, cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u0);
    cudaFree(d_v0);
}

// TODO: solver.h 
// lin_solve.h
