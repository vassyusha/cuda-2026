#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>


template <unsigned BLOCK_SIZE>
__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int n) {
    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;


    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;

    extern __shared__ float s[];
    float* As = s;
    float* Bs = s + BLOCK_SIZE * BLOCK_SIZE;
    float* Cs = s + 2 * BLOCK_SIZE * BLOCK_SIZE;

    // Initialise the shared accumulator for this element
    Cs[ty * BLOCK_SIZE + tx] = 0.0f;

    // Loop over tiles of A and B along the K dimension
    for (int k = 0; k < n; k += BLOCK_SIZE) {
        // Coalesced load of a tile from A and B into shared memory
        As[ty * BLOCK_SIZE + tx] = A[row * n + (k + tx)];
        Bs[ty * BLOCK_SIZE + tx] = B[(k + ty) * n + col];

        __syncthreads();  // ensure all loads are visible

        // Compute partial product and accumulate into shared C tile
        float c_val = Cs[ty * BLOCK_SIZE + tx];
        #pragma unroll
        for (unsigned kk = 0; kk < BLOCK_SIZE; ++kk) {
            c_val += As[ty * BLOCK_SIZE + kk] * Bs[kk * BLOCK_SIZE + tx];
        }
        Cs[ty * BLOCK_SIZE + tx] = c_val;

        __syncthreads();  // ensure As/Bs are not overwritten before all threads finished
    }

    // Write final result to global memory
    C[row * n + col] = Cs[ty * BLOCK_SIZE + tx];
}

static void launch_gemm(const float* d_A, const float* d_B, float* d_C,
                        int n, unsigned block_size) {
    dim3 block(block_size, block_size);
    dim3 grid(n / block_size, n / block_size);      // n is a multiple of block_size
    size_t shmem = 3 * block_size * block_size * sizeof(float);

    switch (block_size) {
        case 32:
            gemm_kernel<32><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        case 16:
            gemm_kernel<16><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        case 8:
            gemm_kernel<8><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        case 4:
            gemm_kernel<4><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        case 2:
            gemm_kernel<2><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        case 1:
            gemm_kernel<1><<<grid, block, shmem>>>(d_A, d_B, d_C, n); break;
        default:
            // We never expect this – n is a power of two ≤ 32 after clamping
            assert(false);
    }

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // n is always a power of two (given), so block_size divides n.
    const unsigned block_size = (n >= 32) ? 32 : n;

    const size_t size = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy inputs to device
    cudaMemcpy(d_A, a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), size, cudaMemcpyHostToDevice);

    // Run the tiled matrix multiplication
    launch_gemm(d_A, d_B, d_C, n, block_size);

    // Copy result back and synchronise
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}